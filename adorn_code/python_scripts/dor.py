# Copyright (c) 2025 Purdue University
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met: redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer;
# redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution;
# neither the name of the copyright holders nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Author(s): Conor Green

# std
import argparse
import collections
import os

# pipd
import networkit as nk

# locals
from tpuv4_symmetry import TPUv4_Symmetry


global ASSERT_BINARY_MAP
ASSERT_BINARY_MAP = True
global VERBOSE
VERBOSE = False
global SLOW_RUN
SLOW_RUN = False

ALLPATHS_DIR = 'topologies_and_routing/allpath_lists'


def build_graph_nk(n, edges):
    """
    Build an undirected NetworKit graph with n nodes and given edges (u,v).
    No parallel edges, no self-loops assumed.
    """
    G = nk.graph.Graph(n, weighted=False, directed=False)
    for u, v in edges:
        if u != v:
            G.addEdge(u, v)
    return G

def precompute_geometry_nk(G, xyzc_dims):
    """
    Precompute:
      - coords[u] = (x,y,z)
      - nbrX[u], nbrY[u], nbrZ[u]: lists of neighbors of u whose edge is along X/Y/Z
        (i.e., neighbor differs in exactly one coordinate)
    Edges that aren't single-axis steps are ignored for DOR.
    """
    n = G.numberOfNodes()
    rel_coords = [None]*n
    coords = [None]*n
    for u in range(n):
        rel_coords[u] = r_to_rel_xyz(u,xyzc_dims)
        coords[u] = r_to_xyz(u,xyzc_dims)

    nbrX = [[] for _ in range(n)]
    nbrY = [[] for _ in range(n)]
    nbrZ = [[] for _ in range(n)]

    it = G.iterEdges()  # yields (u,v)
    for u, v in it:
        xu, yu, zu = rel_coords[u]
        xv, yv, zv = rel_coords[v]
        dx = (xu != xv); dy = (yu != yv); dz = (zu != zv)
        changed = dx + dy + dz
        if changed != 1:
            # Not a pure axis step: ignore for DOR
            input(f'ERROR :: precompute_geometry_nk :: {u}<->{v} is PURE???')
            continue
        if dx:
            nbrX[u].append(v); nbrX[v].append(u)
            # print(f'{u}<->{v} is X')
        elif dy:
            nbrY[u].append(v); nbrY[v].append(u)
            # print(f'{u}<->{v} is Y')
        else:
            nbrZ[u].append(v); nbrZ[v].append(u)
            # print(f'{u}<->{v} is Z')

        # input(f'cont?')

    # Precompute:
    #   - coords[u] = (x,y,z)
    #   - nbrX[u], nbrY[u], nbrZ[u]: lists of neighbors of u whose edge is along X/Y/Z
    #     (i.e., neighbor differs in exactly one coordinate)
    # Edges that aren't single-axis steps are ignored for DOR.
    return coords, nbrX, nbrY, nbrZ

def all_shortest_dor_paths_nk(G, xyzc_dims, tpuv4_symmetry, symmetric, pair_iter=None, unordered=False, max_paths_per_pair=None):
    """
    Enumerate ALL shortest X->Y->Z DOR-valid paths between node pairs on a NetworKit graph.

    DOR rules:
      - Phase 0 (X): move only along X-edges until x == x_t, then transition to Phase 1 at zero cost.
      - Phase 1 (Y): move only along Y-edges until y == y_t, then transition to Phase 2 at zero cost.
      - Phase 2 (Z): move only along Z-edges until z == z_t; destination when (x,y,z)==(x_t,y_t,z_t).
      - Movement can be ± along each dimension; edges must change exactly one coordinate.

    Args:
      G: nk.graph.Graph (undirected, unweighted), nodes 0..n-1
      xyzc_dims: X, Y, Z, and cube dimensions
      pair_iter: optional iterable of (s,t). If None, iterates s!=t (ordered unless unordered=True).
      unordered: if True, iterate only s < t
      max_paths_per_pair: optional int cap on number of paths emitted per pair

    Yields:
      (s, t, path_nodes_list)
    """

    coords, nbrX, nbrY, nbrZ = precompute_geometry_nk(G, xyzc_dims)
    n = G.numberOfNodes()

    nodes = range(n)

    def xeq(u, t): return coords[u][0] == coords[t][0]
    def yeq(u, t): return coords[u][1] == coords[t][1]
    def zeq(u, t): return coords[u][2] == coords[t][2]

    def get_nbrs(phase, u):
        if phase == 0: return nbrX[u]
        if phase == 1: return nbrY[u]
        return nbrZ[u]

    # ---- 0?1 BFS over (node, phase) with predecessor sets ----
    def dor_shortest_paths_pair(s, t):
        # distance and predecessors in state space
        INF = 10**18
        dist = {}
        pred = collections.defaultdict(set)

        dq = collections.deque()
        start = (s, 0)
        dist[start] = 0
        dq.appendleft(start)

        best_goal_dist = None
        goal_states = set()

        def at_boundary(phase, u):
            return (phase == 0 and xeq(u, t)) or (phase == 1 and yeq(u, t)) or (phase == 2 and zeq(u, t))

        while dq:
            u, ph = dq.popleft()
            d = dist[(u, ph)]

            # Optional early exit: if we already found goals at distance D*, skip worse
            if best_goal_dist is not None and d > best_goal_dist:
                continue

            # Zero-cost phase transition if at boundary and not in last phase
            if ph < 2 and at_boundary(ph, u):
                ns = (u, ph + 1)
                nd = d  # zero cost
                od = dist.get(ns, INF)
                if nd < od:
                    dist[ns] = nd
                    pred[ns] = {(u, ph)}
                    dq.appendleft(ns)  # 0-cost => front
                elif nd == od:
                    pred[ns].add((u, ph))

            # Unit-cost moves along current dimension
            for v in get_nbrs(ph, u):
                ns = (v, ph)
                nd = d + 1
                od = dist.get(ns, INF)
                if nd < od:
                    dist[ns] = nd
                    pred[ns] = {(u, ph)}
                    dq.append(ns)  # cost 1 => back
                elif nd == od:
                    pred[ns].add((u, ph))

            # Goal states are (u,2) with full coordinate match
            if ph == 2 and coords[u] == coords[t]:
                if best_goal_dist is None:
                    best_goal_dist = d
                if d == best_goal_dist:
                    goal_states.add((u, ph))

        if not goal_states:
            return  # no DOR-valid path

        # ---- Backtrack all shortest state-paths; translate to node paths ----
        # We need to collapse zero-cost phase transitions (no node move).
        uniq = set()
        out_count = 0

        # Stack for iterative DFS backtracking: (state, iterator over predecessors, current node-path reversed list, last_node)
        for g in goal_states:
            stack = [(g, iter(pred[g] if pred[g] else []), [g[0]], g[0])]
            # Special case: start could be directly the goal via zero-cost transitions; handle if pred[g] is empty
            if not pred[g]:
                if g == (s, 2) and coords[s] == coords[t]:
                    key = tuple([s])
                    if key not in uniq:
                        uniq.add(key)
                        yield list(key)
                        out_count += 1
                        if max_paths_per_pair and out_count >= max_paths_per_pair:
                            return
                continue

            while stack:
                state, itpred, rev_nodes, last_node = stack[-1]
                try:
                    pstate = next(itpred)
                except StopIteration:
                    stack.pop()
                    continue

                pu, pph = pstate
                cu, cph = state

                # If predecessor changes node, append it to node-path
                if pu != last_node:
                    new_rev_nodes = rev_nodes + [pu]
                    new_last = pu
                else:
                    # phase-only transition, no node added
                    new_rev_nodes = rev_nodes
                    new_last = last_node

                if pstate == (s, 0):
                    # reached start state; emit path (reverse nodes)
                    path = list(reversed(new_rev_nodes))
                    key = tuple(path)
                    if key not in uniq:
                        uniq.add(key)
                        yield path
                        out_count += 1
                        if max_paths_per_pair and out_count >= max_paths_per_pair:
                            return
                else:
                    # continue backtracking
                    preds = pred[pstate]
                    stack.append((pstate, iter(preds), new_rev_nodes, new_last))

    src_nodes = range(n)
    if symmetric:
        src_nodes = tpuv4_symmetry.get_canonical_nodes()

    # ---- Pair iteration driver ----
    if pair_iter is None:
        if unordered:
            pair_iter = ((s, t) for s in src_nodes for t in range(s + 1, n))
        else:
            pair_iter = ((s, t) for s in src_nodes for t in range(n))

    for s, t in pair_iter:
        if s == t:
            yield (s,t,[s])
            continue
        if s < 0 or t < 0 or s >= n or t >= n:
            continue
        emitted = 0
        for path in dor_shortest_paths_pair(s, t):
            yield (s, t, path)
            emitted += 1
            if max_paths_per_pair and emitted >= max_paths_per_pair:
                break

# File stuff
###############################################################################

def ingest_map(path_name):
    global ASSERT_BINARY_MAP

    file_name = path_name.split('/')[-1]

    if True:
        print(f'Ingesting filename = {file_name} ({path_name})')

    this_map = []
    edge_list = []

    cur_src = 0
    with open(path_name, 'r') as inf:
        for row in inf:
            r_conns = row.split(' ')
            if '\n' in r_conns:
                r_conns.remove('\n')

            # deal with approximate values (from MIP)
            try:
                r_conns = [int(elem) for elem in r_conns]

            except:
                r_conns = [int(float(elem)) for elem in r_conns]

            this_map.append(r_conns)

            these_dests = [i for i, val in enumerate(r_conns) if val > 0]
            for dest in these_dests:
                edge_list.append((cur_src, dest))

            cur_src += 1


    # quick sanitization
    n_routers = len(this_map)
    for i in range(n_routers):
        this_map[i][i] = 0

    # assert binary?
    if ASSERT_BINARY_MAP:
        for src_map in this_map:
            for conn in src_map:
                # instead, make binary
                if conn > 0.1:
                    conn = 1
                # assert(conn == 1 or conn == 0)

    if VERBOSE:
        print(f'read {this_map}')

    return this_map, n_routers, edge_list

# TPU v4/5
###############################################################################


def r_to_xyz(r,xyzc_dims):
    (x_dim, y_dim, z_dim, cube_dim) = xyzc_dims
    xy_slice_size = x_dim*y_dim

    temp_r = r

    z = temp_r // xy_slice_size
    temp_r = temp_r % xy_slice_size
    y = temp_r // x_dim
    x = temp_r % x_dim

    return x,y,z

def r_to_rel_xyz(r, xyzc_dims):
    (x_dim, y_dim, z_dim, cube_dim) = xyzc_dims

    r_x,r_y,r_z = r_to_xyz(r, xyzc_dims)

    rel_r_x = r_x % cube_dim
    rel_r_y = r_y % cube_dim
    rel_r_z = r_z % cube_dim

    return rel_r_x, rel_r_y, rel_r_z


def calc_conn_type(s, d, xyzc_dims):
    (x_dim, y_dim, z_dim, cube_dim) = xyzc_dims

    rel_s_x, rel_s_y, rel_s_z = r_to_rel_xyz(s, xyzc_dims)
    rel_d_x, rel_d_y, rel_d_z = r_to_rel_xyz(d, xyzc_dims)

    # x+
    if rel_s_x == cube_dim - 1 and rel_d_x == 0:
        return 'x+'
    # x-
    if rel_s_x == 0 and rel_d_x == cube_dim - 1:
        return 'x-'

    # y+
    if rel_s_y == cube_dim - 1 and rel_d_y == 0:
        return 'y+'
    # y-
    if rel_s_y == 0 and rel_d_y == cube_dim - 1:
        return 'y-'

    # z+
    if rel_s_z == cube_dim - 1 and rel_d_z == 0:
        return 'z+'
    # z-
    if rel_s_z == 0 and rel_d_z == cube_dim - 1:
        return 'z-'

    if rel_s_x < rel_d_x:
        return 'x+'
    elif rel_s_x > rel_d_x:
        return 'x-'
    
    if rel_s_y < rel_d_y:
        return 'y+'
    elif rel_s_y > rel_d_y:
        return 'y-'

    if rel_s_z < rel_d_z:
        return 'z+'
    elif rel_s_z > rel_d_z:
        return 'z-'

# CLAs
###############################################################################


def define_and_parse_args():

    global VERBOSE
    global SLOW_RUN

    parser = argparse.ArgumentParser(description='Dimension ordered routing (DOR)')

    parser.add_argument('--topology',type=str,help='input graph', required=True)
    parser.add_argument('--xyzc_dims',nargs='+',type=int,help='type without parenthesis and use spaces, no commas', required=True)

    parser.add_argument('--symmetric',action='store_true',help='graph is vertex symmetric. route canonical flows')
    parser.add_argument('--sym_type',type=str,help="graph symmetry type. default 'trans'",choices=["trans","refl-trans"], default="trans")
    parser.add_argument('--mc_dims',nargs='+',type=int,help='Minimum "cube" of symmetry\'s x, y, and z dimensions. If unspecified then assuming single (zeroth) cube (i.e., x,y,z==cube). Type without parenthesis and use spaces, no commas')

    # script stuff
    parser.add_argument('--verbose','-v',action='store_true',help='extensive prints')
    parser.add_argument('--slow_run',action='store_true',help='ask user input on every iteration')


    args = parser.parse_args()

    if args.verbose:
        VERBOSE = True
    if args.slow_run:
        SLOW_RUN = True
    
    graph_filepath = args.topology
    xyzc_dims = tuple(args.xyzc_dims)

    assert(len(xyzc_dims) == 4)


    symmetric = args.symmetric
    sym_type = None
    mc_dims = None
    if symmetric:
        sym_type = args.sym_type
        mc_dims = tuple(args.mc_dims)
        assert(len(mc_dims) == 3)

    return graph_filepath, xyzc_dims, symmetric, sym_type, mc_dims

def main():
    global VERBOSE
    global SLOW_RUN

    graph_filepath, xyzc_dims, symmetric, sym_type, mc_dims = define_and_parse_args()

    print(f"Working on graph {graph_filepath} w/ dims {xyzc_dims}")

    adj_mat, n_routers, edge_list = ingest_map(graph_filepath)

    G = build_graph_nk(n_routers, edge_list)

    graph_basename = os.path.splitext(os.path.basename(graph_filepath))[0]

    if symmetric:
        out_filename = f'{graph_basename}_dor_{sym_type}sym_{mc_dims[0]}x{mc_dims[1]}x{mc_dims[2]}mc.rallpaths'
    else:
        out_filename = f'{graph_basename}_dor.rallpaths'

    out_filepath = os.path.join(ALLPATHS_DIR,out_filename)
    print(f"Streaming out to {out_filepath}")
    tot_hops = 0

    tpuv4_symmetry = None
    if symmetric:
        tpuv4_symmetry = TPUv4_Symmetry(xyzc_dims, mc_dims=mc_dims, sym_type=sym_type)
        canonical_nodes = tpuv4_symmetry.get_canonical_nodes()
    else:
        canonical_nodes = range(n_routers)

    longest_path_len = 0
    longest_path = None
    longest_flow = None
    seen_flows = set()

    with open(out_filepath,'w+') as of:
        for s, t, path in all_shortest_dor_paths_nk(G, xyzc_dims, tpuv4_symmetry, symmetric):
            if VERBOSE:
                print(f"{s} -> {t}: {path}")

            plen = len(path) - 1
            if (s,t) not in seen_flows:
                tot_hops += (len(path) - 1)
                seen_flows.add((s,t))
            
            if plen > longest_path_len:
                longest_path_len = plen
                longest_path = path
                longest_flow = (s,t)

            path_as_str = ' '.join(map(str, path))
            of.write(f'{path_as_str}\n')

            if SLOW_RUN:
                input(f'cont?')
            
            if s % 100 == 0 and t == 0:
                print(f'Starting src {s}')

    for i in canonical_nodes:
        for j in range(i+1,n_routers):
            if (i,j) not in seen_flows:
                input(f"ERROR: no paths for {i}->{j} @ {r_to_xyz(i,xyzc_dims)}->{r_to_xyz(j,xyzc_dims)}")

    print(f"avg hops = {tot_hops / ((n_routers**2) - n_routers)}")
    print(f"longest {longest_path_len} {longest_path} for {longest_flow}")
    print(f'Completed')

    print(f"Wrote to {out_filepath}")



if __name__ == "__main__":


    main()
