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

# pip
import networkit as nk

# locals
from symmetry_2d import Symmetry2D


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


def precompute_geometry_nk(G, xy_dims):
    """
    Precompute for 2D:
      - coords[u] = (x, y)
      - nbrX[u], nbrY[u]: lists of neighbors of u whose edge is along X or Y
        (neighbor differs in exactly one coordinate).
    Edges that aren't single-axis steps are ignored for DOR.
    """
    n = G.numberOfNodes()
    x_dim, y_dim = xy_dims
    coords = [None] * n
    for u in range(n):
        coords[u] = r_to_xy(u, xy_dims)

    nbrX = [[] for _ in range(n)]
    nbrY = [[] for _ in range(n)]

    it = G.iterEdges()
    for u, v in it:
        xu, yu = coords[u]
        xv, yv = coords[v]
        dx = (xu != xv)
        dy = (yu != yv)
        changed = dx + dy
        if changed != 1:
            continue
        if dx:
            nbrX[u].append(v)
            nbrX[v].append(u)
        else:
            nbrY[u].append(v)
            nbrY[v].append(u)

    return coords, nbrX, nbrY


def all_shortest_dor_paths_nk(G, xy_dims, symmetry_2d, symmetric, pair_iter=None,
                               unordered=False, max_paths_per_pair=None):
    """
    Enumerate ALL shortest X->Y DOR-valid paths between node pairs on a 2D NetworKit graph.

    DOR rules:
      - Phase 0 (X): move only along X-edges until x == x_t, then transition to Phase 1 at zero cost.
      - Phase 1 (Y): move only along Y-edges until y == y_t; destination when (x,y)==(x_t,y_t).
      - Movement can be ± along each dimension; edges must change exactly one coordinate.

    Args:
      G: nk.graph.Graph (undirected, unweighted), nodes 0..n-1
      xy_dims: (x_dim, y_dim)
      symmetry_2d: Symmetry2D instance (used only if symmetric=True)
      pair_iter: optional iterable of (s,t). If None, iterates s!=t (ordered unless unordered=True).
      unordered: if True, iterate only s < t
      max_paths_per_pair: optional int cap on number of paths emitted per pair

    Yields:
      (s, t, path_nodes_list)
    """
    coords, nbrX, nbrY = precompute_geometry_nk(G, xy_dims)
    n = G.numberOfNodes()

    def xeq(u, t):
        return coords[u][0] == coords[t][0]

    def yeq(u, t):
        return coords[u][1] == coords[t][1]

    def get_nbrs(phase, u):
        if phase == 0:
            return nbrX[u]
        return nbrY[u]

    def dor_shortest_paths_pair(s, t):
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
            return (phase == 0 and xeq(u, t)) or (phase == 1 and yeq(u, t))

        while dq:
            u, ph = dq.popleft()
            d = dist[(u, ph)]

            if best_goal_dist is not None and d > best_goal_dist:
                continue

            # Zero-cost phase transition if at boundary and not in last phase
            if ph < 1 and at_boundary(ph, u):
                ns = (u, ph + 1)
                nd = d
                od = dist.get(ns, INF)
                if nd < od:
                    dist[ns] = nd
                    pred[ns] = {(u, ph)}
                    dq.appendleft(ns)
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
                    dq.append(ns)
                elif nd == od:
                    pred[ns].add((u, ph))

            # Goal: phase 1 and full coordinate match
            if ph == 1 and coords[u] == coords[t]:
                if best_goal_dist is None:
                    best_goal_dist = d
                if d == best_goal_dist:
                    goal_states.add((u, ph))

        if not goal_states:
            return

        uniq = set()
        out_count = 0

        for g in goal_states:
            stack = [(g, iter(pred[g] if pred[g] else []), [g[0]], g[0])]
            if not pred[g]:
                if g == (s, 1) and coords[s] == coords[t]:
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

                if pu != last_node:
                    new_rev_nodes = rev_nodes + [pu]
                    new_last = pu
                else:
                    new_rev_nodes = rev_nodes
                    new_last = last_node

                if pstate == (s, 0):
                    path = list(reversed(new_rev_nodes))
                    key = tuple(path)
                    if key not in uniq:
                        uniq.add(key)
                        yield path
                        out_count += 1
                        if max_paths_per_pair and out_count >= max_paths_per_pair:
                            return
                else:
                    preds = pred[pstate]
                    stack.append((pstate, iter(preds), new_rev_nodes, new_last))

    src_nodes = range(n)
    if symmetric:
        src_nodes = symmetry_2d.get_canonical_nodes()

    if pair_iter is None:
        if unordered:
            pair_iter = ((s, t) for s in src_nodes for t in range(s + 1, n))
        else:
            pair_iter = ((s, t) for s in src_nodes for t in range(n))

    for s, t in pair_iter:
        if s == t:
            yield (s, t, [s])
            continue
        if s < 0 or t < 0 or s >= n or t >= n:
            continue
        emitted = 0
        for path in dor_shortest_paths_pair(s, t):
            yield (s, t, path)
            emitted += 1
            if max_paths_per_pair and emitted >= max_paths_per_pair:
                break


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

            try:
                r_conns = [int(elem) for elem in r_conns]
            except Exception:
                r_conns = [int(float(elem)) for elem in r_conns]

            this_map.append(r_conns)

            these_dests = [i for i, val in enumerate(r_conns) if val > 0]
            for dest in these_dests:
                edge_list.append((cur_src, dest))

            cur_src += 1

    n_routers = len(this_map)
    for i in range(n_routers):
        this_map[i][i] = 0

    if ASSERT_BINARY_MAP:
        for src_map in this_map:
            for conn in src_map:
                if conn > 0.1:
                    conn = 1

    if VERBOSE:
        print(f'read {this_map}')

    return this_map, n_routers, edge_list


# 2D geometry
###############################################################################


def r_to_xy(r, xy_dims):
    x_dim, y_dim = xy_dims
    x = r % x_dim
    y = r // x_dim
    return (x, y)


def calc_conn_type(s, d, xy_dims):
    """Return 'x+', 'x-', 'y+', or 'y-' for the 2D step from s to d."""
    xs, ys = r_to_xy(s, xy_dims)
    xd, yd = r_to_xy(d, xy_dims)

    if xs < xd:
        return 'x+'
    elif xs > xd:
        return 'x-'
    if ys < yd:
        return 'y+'
    elif ys > yd:
        return 'y-'
    return None  # s == d


# CLAs
###############################################################################


def define_and_parse_args():
    global VERBOSE
    global SLOW_RUN

    parser = argparse.ArgumentParser(description='Dimension-ordered routing (DOR) for 2D topologies')

    parser.add_argument('--topology', type=str, help='input graph', required=True)
    parser.add_argument('--xy_dims', nargs=2, type=int, metavar=('X_DIM', 'Y_DIM'),
                        help='Grid dimensions (row-major: r = x + y*X_DIM)', required=True)

    parser.add_argument('--symmetric', action='store_true',
                        help='Graph is vertex symmetric; route canonical flows only')
    parser.add_argument('--mc_dims', nargs=2, type=int, metavar=('MC_X', 'MC_Y'),
                        help='Canonical tile dimensions for symmetry (required if --symmetric)')

    parser.add_argument('--verbose', '-v', action='store_true', help='extensive prints')
    parser.add_argument('--slow_run', action='store_true', help='ask user input on every iteration')

    args = parser.parse_args()

    if args.verbose:
        VERBOSE = True
    if args.slow_run:
        SLOW_RUN = True

    graph_filepath = args.topology
    xy_dims = tuple(args.xy_dims)
    assert len(xy_dims) == 2

    symmetric = args.symmetric
    mc_dims = None
    if symmetric:
        assert args.mc_dims is not None, '--mc_dims required when --symmetric'
        mc_dims = tuple(args.mc_dims)
        assert len(mc_dims) == 2

    return graph_filepath, xy_dims, symmetric, mc_dims


def main():
    global VERBOSE
    global SLOW_RUN

    graph_filepath, xy_dims, symmetric, mc_dims = define_and_parse_args()

    print(f"Working on graph {graph_filepath} w/ xy_dims {xy_dims}")

    adj_mat, n_routers, edge_list = ingest_map(graph_filepath)
    assert n_routers == xy_dims[0] * xy_dims[1], \
        f"Node count {n_routers} != x_dim*y_dim {xy_dims[0]*xy_dims[1]}"

    G = build_graph_nk(n_routers, edge_list)

    graph_basename = os.path.splitext(os.path.basename(graph_filepath))[0]

    if symmetric:
        out_filename = f'{graph_basename}_dor_2d_sym_{mc_dims[0]}x{mc_dims[1]}mc.rallpaths'
    else:
        out_filename = f'{graph_basename}_dor_2d.rallpaths'

    out_filepath = os.path.join(ALLPATHS_DIR, out_filename)
    print(f"Streaming out to {out_filepath}")

    symmetry_2d = None
    if symmetric:
        symmetry_2d = Symmetry2D(xy_dims, mc_dims)
        canonical_nodes = symmetry_2d.get_canonical_nodes()
    else:
        canonical_nodes = range(n_routers)

    tot_hops = 0
    longest_path_len = 0
    longest_path = None
    longest_flow = None
    seen_flows = set()

    with open(out_filepath, 'w+') as of:
        for s, t, path in all_shortest_dor_paths_nk(G, xy_dims, symmetry_2d, symmetric):
            if VERBOSE:
                print(f"{s} -> {t}: {path}")

            plen = len(path) - 1
            if (s, t) not in seen_flows:
                tot_hops += plen
                seen_flows.add((s, t))

            if plen > longest_path_len:
                longest_path_len = plen
                longest_path = path
                longest_flow = (s, t)

            path_as_str = ' '.join(map(str, path))
            of.write(f'{path_as_str}\n')

            if SLOW_RUN:
                input('cont?')

            if s % 100 == 0 and t == 0:
                print(f'Starting src {s}')

    n_pairs = (n_routers ** 2) - n_routers
    print(f"avg hops = {tot_hops / n_pairs}")
    print(f"longest {longest_path_len} {longest_path} for {longest_flow}")
    print('Completed')
    print(f"Wrote to {out_filepath}")


if __name__ == "__main__":
    main()
