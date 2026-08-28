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
import threading
import ast
import os
import random
from collections import deque, defaultdict

# pipd
import networkit as nk

# locals
from tpuv4_symmetry import TPUv4_Symmetry


global VERBOSE
VERBOSE = False

global MAX_PROCS
MAX_PROCS = 128

global sem
sem = threading.Semaphore(MAX_PROCS)

class ATPathFinder():

    # class vars
    ############

    verbose = False
    slow  = False

    supported_graph_libraries = ['networkit']

    def __init__(self, topo_filepath, allowed_turns_filepath, all_allowed=False, graph_library='networkit', verbose=False):

        # basic
        self.verbose = verbose

        assert(graph_library in self.supported_graph_libraries)
        self.graph_library = graph_library

        # TODO parse from allowed turns
        self.n_vcs = 2

        # needs topo_filepath
        self.topo_filepath = topo_filepath
        # defines topo_adjmat, topo_adjlist, n_routers
        self.ingest_topo()

        # needs atv_filepath
        self.atv_filepath = allowed_turns_filepath
        # defines allowed_turns_list
        self.ingest_allowed_turns(all_allowed=all_allowed)

        # needs allowed_turns_list
        # defines edge_to_label_dict, label_to_edge_dict, n_labels
        self.create_edge_translations()

        # needs allowed_turns_list
        # defines allowed_cdg_G
        self.create_allowed_cdg()

    def ingest_topo(self):
        assert(self.topo_filepath is not None)
        self.topo_adjmat, self.topo_adjlist, self.n_routers = self.ingest_a_map_(self.topo_filepath)

    # _ implies it returns instead of setting self vars
    # these will be class methods
    @classmethod
    def ingest_a_map_(cls, path_name):

        if True:
            print(f'Ingesting r map ({path_name})')

        this_map = []
        this_adj_list = []

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

                adjacents = []
                for dest, is_conn in enumerate(r_conns):
                    if is_conn > 0:
                        adjacents.append(dest)
                this_adj_list.append(adjacents)

        n_routers = len(this_map)

        return this_map, this_adj_list, n_routers

    def ingest_allowed_turns(self, all_allowed=False):
        assert(self.atv_filepath)
        _, self.allowed_turns_list = self.ingest_an_allowed_turns_(self.atv_filepath, all_allowed=all_allowed)
        print(f"# allowed turns : {len(self.allowed_turns_list)}")

    @classmethod
    def ingest_an_allowed_turns_(cls, path_name, all_allowed=False):
        atvcs_dict = {}
        atvcs_list = []

        print(f'Ingesting allowed turns ({path_name})')

        with open(path_name, 'r') as inf:
            for line in inf.readlines():
                line_no_newline = line.strip('\n')
                line_w_curly = f'{{ {line_no_newline} }}'

                as_dict = ast.literal_eval(line_w_curly)

                k, v = as_dict.popitem()
                if all_allowed:
                    v = True

                atvcs_dict.update({k:v})

                # if allowed
                if v:
                    atvcs_list.append( k )
        return atvcs_dict, atvcs_list

    def create_edge_translations(self):
        assert(self.allowed_turns_list)

        self.edge_to_label_dict, self.label_to_edge_dict, self.n_labels = self.create_an_edge_translations_(self.allowed_turns_list)

    @classmethod
    def create_an_edge_translations_(cls, at_list):
        edge_to_label_dict = {}
        label_to_edge_dict = {}

        cur_label = 0
        translated_edges = set()
        for turn in at_list:
            for edge in turn:
                if edge not in translated_edges:
                    translated_edges.add(edge)
                    edge_to_label_dict[edge] = cur_label
                    label_to_edge_dict[cur_label] = edge
                    cur_label += 1

        n_labels = cur_label
        return edge_to_label_dict, label_to_edge_dict, n_labels

    def create_allowed_cdg(self):
        assert(self.allowed_turns_list)
        assert(self.edge_to_label_dict)
        assert(self.n_labels)
        assert(self.graph_library)

        allowed_turn_labels_list = [(self.edge_to_label_dict[e0], self.edge_to_label_dict[e1]) for (e0, e1) in self.allowed_turns_list]

        if self.graph_library == 'networkit':
            self.allowed_cdg_G = self.create_an_allowed_cdg_networkit_(allowed_turn_labels_list, self.n_labels)
        else:
            print(f'UNIMPLEMENTED :: create_allowed_cdg() :: graph_library {graph_library}')
            quit()

    def create_an_allowed_cdg_networkit_(self, edges, n):

        n = max(max(u, v) for u, v in edges) + 1
        G = nk.Graph(n, weighted=False, directed=True)
        for u, v in edges:
            G.addEdge(u, v)

        print("NetworKit:", G.numberOfNodes(), "nodes,", G.numberOfEdges(), "edges")

        return G


    ################################################################################

    @classmethod
    def bfs_distances_upper_(cls, G, source):
        """
        IMPORTANT: must use upperNodeIdBound() because you add/remove nodes
        (holes in node IDs). numberOfNodes() is NOT safe here.
        """
        n = G.upperNodeIdBound()
        dist = [-1] * n
        q = deque([source])
        dist[source] = 0

        while q:
            u = q.popleft()
            du = dist[u]
            for v in G.iterNeighbors(u):  # out-neighbors for directed graph
                if v >= n:
                    continue
                if dist[v] != -1:
                    continue
                dist[v] = du + 1
                q.append(v)

        return dist


    @classmethod
    def build_shortest_predecessors_(cls, G, dist):
        """
        preds[v] = list of u such that (u->v) is an edge AND dist[u] + 1 == dist[v].
        This defines the shortest-path DAG rooted at the BFS source.
        """
        n = len(dist)
        preds = [[] for _ in range(n)]

        # NetworKit graph edge iterator
        for (u, v) in G.iterEdges():
            if u >= n or v >= n:
                continue
            du = dist[u]
            dv = dist[v]
            if du != -1 and dv == du + 1:
                preds[v].append(u)

        return preds

    @classmethod
    def sample_one_shortest_path_(cls, preds, dist, src_node, dst_node, rng):
        """
        Samples ONE shortest path from src_node -> dst_node in the shortest-path DAG.
        Returns list of CDG nodes [src_node, ..., dst_node] or None if unreachable.
        """
        if dst_node >= len(dist) or dist[dst_node] == -1:
            return None

        cur = dst_node
        path = [cur]

        while cur != src_node:
            ps = preds[cur]
            if not ps:
                return None
            cur = rng.choice(ps)
            path.append(cur)

        path.reverse()
        return path

    @classmethod
    def stream_bounded_simple_paths_(cls, G, src, dst, max_len_edges):
        """
        Enumerate SIMPLE paths in the CDG from src->dst with <= max_len_edges edges.
        This is a bounded DFS generator. It yields node-id paths like [src, ..., dst].
        """
        if src == dst:
            yield [src]
            return

        path = [src]
        visited = {src}

        # stack holds (current_node, iterator_over_neighbors)
        stack = [(src, iter(G.iterNeighbors(src)))]

        while stack:
            u, it = stack[-1]

            # If we're already at depth limit, backtrack
            if (len(path) - 1) >= max_len_edges:
                stack.pop()
                visited.remove(u)
                path.pop()
                continue

            try:
                v = next(it)
            except StopIteration:
                stack.pop()
                visited.remove(u)
                path.pop()
                continue

            if v in visited:
                continue

            path.append(v)
            visited.add(v)

            if v == dst:
                yield list(path)
                visited.remove(v)
                path.pop()
            else:
                stack.append((v, iter(G.iterNeighbors(v))))

    @classmethod
    def stream_paths_(cls, bfs, t):
        for path in bfs.getPaths(t):  # returns all shortest s→t paths
            yield path

    def calculate_paths_single_source(self, src, max_paths=None, nonmin=0, attempts_factor=200, sample_minpaths=0,sample_seed=1,sample_attempts_factor=200):
        assert self.allowed_cdg_G
        assert self.topo_adjlist
        assert self.n_vcs
        assert self.n_routers

        allowed_cdg_G = self.allowed_cdg_G
        topo_adjlist = self.topo_adjlist
        n_vcs = self.n_vcs
        n_routers = self.n_routers
        edge_to_label_dict = self.edge_to_label_dict
        label_to_edge_dict = self.label_to_edge_dict

        nodes_to_remove = []

        # Build super source
        src_adjacents = topo_adjlist[src]
        src_edges = [(src, a, v) for a in src_adjacents for v in range(n_vcs)]
        src_labels = [edge_to_label_dict[e] for e in src_edges]

        super_src_label = allowed_cdg_G.addNode()
        nodes_to_remove.append(super_src_label)

        for src_label in src_labels:
            allowed_cdg_G.addEdge(super_src_label, src_label)

        # Build super dests
        dests_to_super_dests_dict = {}
        for dest in range(n_routers):
            if src == dest:
                continue

            super_dest_label = allowed_cdg_G.addNode()
            dests_to_super_dests_dict[dest] = super_dest_label
            nodes_to_remove.append(super_dest_label)

            dest_adjacents = topo_adjlist[dest]
            dest_edges = [(a, dest, v) for a in dest_adjacents for v in range(n_vcs)]
            dest_labels = [edge_to_label_dict[e] for e in dest_edges]

            for dest_label in dest_labels:
                allowed_cdg_G.addEdge(dest_label, super_dest_label)

        src_paths_tuples = []

        # ----------------------------------------------------------------------
        # NEW: sampled minimal paths mode (avoids enumerating all shortest paths)
        # ----------------------------------------------------------------------
        if sample_minpaths and sample_minpaths > 0:

            rng = random.Random(sample_seed + src)

            # 1) Compute BFS distances once (cheap)
            dist_from_super_src = self.bfs_distances_upper_(allowed_cdg_G, super_src_label)

            # 2) Build predecessor lists for shortest-path DAG
            preds = self.build_shortest_predecessors_(allowed_cdg_G, dist_from_super_src)

            src_paths_tuples = []

            for dest in range(n_routers):
                if src == dest:
                    continue

                dest_paths_tuples = set()
                super_dest_label = dests_to_super_dests_dict[dest]

                # attempt cap to avoid infinite duplicate sampling on high-multiplicity graphs
                max_attempts = sample_minpaths * sample_attempts_factor
                attempts = 0

                while len(dest_paths_tuples) < sample_minpaths and attempts < max_attempts:
                    attempts += 1

                    full_path = self.sample_one_shortest_path_(
                        preds, dist_from_super_src, super_src_label, super_dest_label, rng
                    )
                    if full_path is None:
                        break

                    # convert CDG path -> topology path (same as your existing logic)
                    path_as_labels = full_path[1:-1]
                    if not path_as_labels:
                        continue

                    path_as_edges = [label_to_edge_dict[l] for l in path_as_labels]
                    path_as_list = [e[0] for e in path_as_edges] + [path_as_edges[-1][1]]
                    path = tuple(path_as_list)

                    dest_paths_tuples.add(path)

                    if max_paths and len(dest_paths_tuples) >= max_paths:
                        break

                src_paths_tuples.append(list(dest_paths_tuples))

            # cleanup and return (same cleanup you already do at end)
            for node in nodes_to_remove:
                allowed_cdg_G.removeNode(node)

            return src_paths_tuples


        # ---------- CASE 2: shortest-only (your current behavior) ----------
        elif nonmin <= 0:
            bfs = nk.distance.BFS(allowed_cdg_G, source=super_src_label, storePaths=True)
            bfs.run()

            for dest in range(n_routers):
                if src == dest:
                    continue

                if dest % 100 == 0:
                    print(f"\tWorking on dest {dest}")

                dest_paths_tuples = set()
                super_dest_label = dests_to_super_dests_dict[dest]

                attempts = 0
                max_attempts = None
                if max_paths:
                    max_attempts = max_paths * attempts_factor

                for full_path in self.stream_paths_(bfs, super_dest_label):
                    attempts += 1
                    if max_attempts and attempts >= max_attempts:
                        break

                    path_as_labels = full_path[1:-1]
                    if not path_as_labels:
                        continue

                    path_as_edges = [label_to_edge_dict[l] for l in path_as_labels]
                    path_as_list = [e[0] for e in path_as_edges] + [path_as_edges[-1][1]]
                    path = tuple(path_as_list)

                    dest_paths_tuples.add(path)

                    if max_paths and len(dest_paths_tuples) >= max_paths:
                        break

                src_paths_tuples.append(list(dest_paths_tuples))

        # ---------- CASE 3: bounded non-minimal ----------
        else:
            # Get shortest CDG distances once (cheap) so we know min length to each super_dest
            dist_from_super_src = self.bfs_distances_upper_(allowed_cdg_G, super_src_label)

            for dest in range(n_routers):
                if src == dest:
                    continue

                dest_paths_tuples = set()
                super_dest_label = dests_to_super_dests_dict[dest]

                min_cdg_len = dist_from_super_src[super_dest_label]  # CDG edges
                if min_cdg_len < 0:
                    src_paths_tuples.append([])
                    continue

                # Allow CDG length <= min + nonmin
                max_cdg_len = min_cdg_len + nonmin

                attempts = 0
                max_attempts = None
                if max_paths:
                    max_attempts = max_paths * attempts_factor

                for full_path in self.stream_bounded_simple_paths_(
                    allowed_cdg_G,
                    super_src_label,
                    super_dest_label,
                    max_cdg_len
                ):
                    attempts += 1
                    if max_attempts and attempts >= max_attempts:
                        break

                    path_as_labels = full_path[1:-1]
                    if not path_as_labels:
                        continue

                    path_as_edges = [label_to_edge_dict[l] for l in path_as_labels]
                    path_as_list = [e[0] for e in path_as_edges] + [path_as_edges[-1][1]]
                    path = tuple(path_as_list)

                    dest_paths_tuples.add(path)

                    if max_paths and len(dest_paths_tuples) >= max_paths:
                        break

                src_paths_tuples.append(list(dest_paths_tuples))

        # Cleanup: remove all temporary nodes (removes their edges too)
        for node in nodes_to_remove:
            allowed_cdg_G.removeNode(node)

        return src_paths_tuples


    # def calculate_paths_single_source(self, src, max_paths=None): # targets = None
    #     assert(self.allowed_cdg_G)
    #     assert(self.topo_adjlist)
    #     assert(self.n_vcs)
    #     assert(self.n_routers)

    #     allowed_cdg_G = self.allowed_cdg_G
    #     topo_adjlist = self.topo_adjlist
    #     n_vcs = self.n_vcs
    #     n_routers = self.n_routers
    #     edge_to_label_dict = self.edge_to_label_dict
    #     label_to_edge_dict = self.label_to_edge_dict

    #     # for later cleanup
    #     nodes_to_remove = []
    #     edges_to_remove = []

    #     src_adjacents = topo_adjlist[src]
    #     src_edges = [(src, a, v) for a in src_adjacents for v in range(n_vcs)]
    #     src_labels = [edge_to_label_dict[e] for e in src_edges]

    #     # TODO figure out same graph object and threading?
    #     super_src_label = allowed_cdg_G.addNode()
    #     # print(f'Added node {super_src_label}')
    #     nodes_to_remove.append(super_src_label)

    #     for src_label in src_labels:
    #         allowed_cdg_G.addEdge(super_src_label, src_label)
    #         # print(f'Added edge {super_src_label}->{src_label} aka ss->{label_to_edge_dict[src_label]}')

    #     # print(f'Completed sources')

    #     # super_dests_to_dests_dict = {}
    #     dests_to_super_dests_dict = {}
    #     for dest in range(n_routers):
    #         if src==dest:
    #             continue
    #         super_dest_label = allowed_cdg_G.addNode()
    #         # print(f'Added node {super_dest_label}')
    #         dests_to_super_dests_dict[dest] = super_dest_label
    #         # super_dests_to_dests_dict[super_dest_label] = dest
    #         nodes_to_remove.append(super_dest_label)

    #         dest_adjacents = topo_adjlist[dest]
    #         dest_edges = [(a, dest, v) for a in dest_adjacents for v in range(n_vcs)]
    #         dest_labels = [self.edge_to_label_dict[e] for e in dest_edges]

    #         for dest_label in dest_labels:
    #             allowed_cdg_G.addEdge(dest_label, super_dest_label)
    #             # print(f'Added edge {dest_label}->{super_dest_label} aka {label_to_edge_dict[dest_label]}->sd')

    #     # print(f'Completed destinations')

    #     # networkit alg
    #     bfs = nk.distance.BFS(allowed_cdg_G, source=super_src_label, storePaths=True)  # BFS since unit weights
    #     bfs.run()

    #     src_paths_tuples = []

    #     for dest in range(n_routers):
    #         if src==dest:
    #             src_paths_tuples.append(src)
    #             continue
            
    #         if self.verbose:
    #             print(f'Working on dest {dest}')
    #         # avoid redundancy
    #         dest_paths_tuples = set()
    #         super_dest_label = dests_to_super_dests_dict[dest]
    #         for full_path in self.stream_paths_(bfs, super_dest_label):

    #             path_as_labels = full_path[1:-1]
    #             path_as_edges = [self.label_to_edge_dict[l] for l in path_as_labels]

    #             path_as_list = [e[0] for e in path_as_edges] + [path_as_edges[-1][1]]
    #             # path = (e[0] for e in path_as_edges) + (path_as_edges[-1][1])
    #             path = tuple(path_as_list)

    #             # input(f'full_path = {full_path}, path_as_labels = {path_as_labels}, path_as_edges = {path_as_edges}, path = {path}')

    #             dest_paths_tuples.add(path)

    #             if max_paths and len(dest_paths_tuples) >= max_paths:
    #                 # input(f'stopping early for max_paths')
    #                 break

    #         if self.verbose:
    #             print(f'Completed {src}->{dest} : paths {dest_paths_tuples}')

    #         n_paths = len(dest_paths_tuples)
    #         # print(f"# paths for {src}->{dest} : {n_paths}")

    #         if len(dest_paths_tuples) == 0:
    #             print(f"ERROR: No path from {src}->{dest}")
    #             quit()

    #         src_paths_tuples.append(list(dest_paths_tuples))

    #     # cleanup
    #     # removing node removes all its edges too
    #     for node in nodes_to_remove:
    #         allowed_cdg_G.removeNode(node)

    #     # for dest in range(n_routers):
    #     #     print(f'{src}->{dest} : {src_paths_tuples[dest]}')

    #     # all_src_path_tuples = [src_paths_tuples]
    #     # out_name = topo_name.split('/')[-1].replace('.map','')  + f'_{src}src' + '.rallpaths'
    #     # allpaths_output_path_prefix = '/scratch/negishi/green456/topologies_and_routing/allpath_lists'
    #     # allpaths_output_path = os.path.join(allpaths_output_path_prefix, out_name)
    #     # output_rallpaths(allpaths_output_path, all_src_path_tuples)


    #     # # JUST FOR DEBUGGING
    #     # out_name = topo_name.split('/')[-1].replace('.map','') + f'_{src}src' + '.paths'
    #     # chosenpaths_output_path_prefix = '/scratch/negishi/green456/topologies_and_routing/routepath_lists'
    #     # chosenpaths_output_path = os.path.join(chosenpaths_output_path_prefix, out_name)
    #     # output_chosenpaths(chosenpaths_output_path, all_src_path_tuples)


    #     return src_paths_tuples

def output_rallpaths(out_path, allpaths_tuples):

    with open(out_path, 'w+') as of:
        for list_of_dest_paths in allpaths_tuples:

            for dest_paths in list_of_dest_paths:

                if isinstance(dest_paths,int):
                    out_line = str(dest_paths)
                    # print(f'out_line = {out_line}')

                    of.write(out_line + '\n')
                    continue

                for dest_path in dest_paths:

                    # print(f'dest_path_tuple ({type(dest_path_tuple)}) = {dest_path_tuple}')

                    # out_line = ' '.join(list(dest_path_tuple))
                    out_line = ' '.join(map(str, dest_path))

                    # print(f'out_line = {out_line}')

                    of.write(out_line + '\n')

    print(f'Wrote out to {out_path}')


def output_rallpaths_from_path_dict(out_path, path_dict):
    """Write .rallpaths from (s,d) -> iterable of path tuples/lists."""
    with open(out_path, 'w+') as of:
        for (s, d) in sorted(path_dict.keys()):
            for path in path_dict[(s, d)]:
                of.write(' '.join(map(str, path)) + '\n')
    print(f'Wrote out to {out_path}')


def assert_path_uses_allowed_turns(path, allowed_turn_set):
    """
    Assert every consecutive router-edge pair (turn) on path is allowed.
    allowed_turn_set holds ((u,v),(v,w)) router-edge pairs (VC-agnostic).
    """
    assert path_uses_allowed_turns(path, allowed_turn_set), (
        f"ERROR: path {list(path)} uses disallowed turn"
    )


def path_uses_allowed_turns(path, allowed_turn_set):
    """
    Return True when every consecutive router-edge pair (turn) on path is allowed.
    """
    if len(path) < 3:
        return True
    for i in range(len(path) - 2):
        e0 = (path[i], path[i + 1])
        e1 = (path[i + 1], path[i + 2])
        turn = (e0, e1)
        if turn not in allowed_turn_set:
            return False
    return True


def path_is_simple(path):
    return len(set(path)) == len(path)


def router_allowed_turn_set(allowed_turns_list):
    """
    Collapse VC-annotated turns ((u,v,vc0),(v,w,vc1)) to router-edge pairs
    ((u,v),(v,w)). A router turn is allowed if any VC assignment allows it.
    """
    turn_set = set()
    for turn in allowed_turns_list:
        e0, e1 = turn
        turn_set.add(((e0[0], e0[1]), (e1[0], e1[1])))
    return turn_set


def flatten_all_src_path_tuples(all_src_path_tuples):
    """Flatten nested per-src path lists into path_dict[(s,d)] -> set of path tuples."""
    path_dict = defaultdict(set)
    for src_paths in all_src_path_tuples:
        for dest_paths in src_paths:
            if isinstance(dest_paths, int):
                continue
            for path in dest_paths:
                path_t = tuple(path)
                if len(path_t) < 2:
                    continue
                s, d = path_t[0], path_t[-1]
                if s == d:
                    continue
                path_dict[(s, d)].add(path_t)
    return path_dict


def close_subpaths_destination_based(path_dict, allowed_turn_set):
    """
    Fixed-point subpath closure: for every path P=[s,...,d], every suffix P[i:]
    is added as a candidate for (P[i], d), even if non-shortest.
    Each newly added subpath is checked against allowed turns.
    """
    n_added = 0
    changed = True
    while changed:
        changed = False
        snapshot = [(flow, list(paths)) for flow, paths in path_dict.items()]
        for (_s, d), paths in snapshot:
            for path in paths:
                path_hop_len = len(path) - 1
                for sub_idx in range(1, path_hop_len):
                    sub_path = tuple(path[sub_idx:])
                    assert_path_uses_allowed_turns(sub_path, allowed_turn_set)
                    key = (sub_path[0], d)
                    if sub_path not in path_dict[key]:
                        path_dict[key].add(sub_path)
                        changed = True
                        n_added += 1
    return path_dict, n_added


def close_composed_destination_based_paths(path_dict, allowed_turn_set):
    """
    Fixed-point composition closure for destination-based routing.

    Suffix closure guarantees that if [s, ..., u, ..., d] is present, then
    [u, ..., d] is present. Destination-based routing also needs the converse
    combinations: if s can legally route first to v for destination d, then any
    known legal path from v to d is a candidate path for s to d when the joined
    path remains simple and uses allowed turns.
    """
    total_added = 0
    iteration = 0

    while True:
        additions = defaultdict(set)
        next_hops_by_flow = {
            flow: {path[1] for path in paths if len(path) >= 2}
            for flow, paths in path_dict.items()
        }

        for (s, d), next_hops in next_hops_by_flow.items():
            flow = (s, d)
            existing_paths = path_dict[flow]

            for next_hop in next_hops:
                if next_hop == d:
                    candidate = (s, d)
                    if candidate not in existing_paths:
                        additions[flow].add(candidate)
                    continue

                suffix_paths = path_dict.get((next_hop, d))
                if not suffix_paths:
                    continue

                for suffix_path in suffix_paths:
                    candidate = (s,) + tuple(suffix_path)
                    if candidate in existing_paths:
                        continue
                    if candidate in additions[flow]:
                        continue
                    if not path_is_simple(candidate):
                        continue
                    if not path_uses_allowed_turns(candidate, allowed_turn_set):
                        continue
                    additions[flow].add(candidate)

        n_added = 0
        for flow, paths_to_add in additions.items():
            before = len(path_dict[flow])
            path_dict[flow].update(paths_to_add)
            n_added += len(path_dict[flow]) - before

        print(f"\tcomposition closure iter {iteration}: added = {n_added}")
        total_added += n_added
        iteration += 1

        if n_added == 0:
            break

    return path_dict, total_added


def assert_subpath_membership(path_dict):
    """Every intermediate suffix of every path must be present for that sub-flow."""
    for (_s, d), paths in path_dict.items():
        for path in paths:
            path_hop_len = len(path) - 1
            for sub_idx in range(1, path_hop_len):
                sub_path = tuple(path[sub_idx:])
                key = (sub_path[0], d)
                assert key in path_dict and sub_path in path_dict[key], (
                    f"ERROR: subpath {list(sub_path)} of {list(path)} "
                    f"not found in candidates for {key}"
                )


def apply_destination_based_closure(all_src_path_tuples, allowed_turn_set):
    """
    Flatten, close under subpaths (with allowed-turn checks), verify membership.
    Returns path_dict suitable for output_rallpaths_from_path_dict.
    """
    print("Applying destination-based subpath closure")
    path_dict = flatten_all_src_path_tuples(all_src_path_tuples)
    n_before = sum(len(v) for v in path_dict.values())
    path_dict, n_added_subpaths = close_subpaths_destination_based(path_dict, allowed_turn_set)
    path_dict, n_added_composed = close_composed_destination_based_paths(path_dict, allowed_turn_set)
    n_after = sum(len(v) for v in path_dict.values())
    print(
        f"\tpaths before closure = {n_before}, "
        f"subpaths added = {n_added_subpaths}, "
        f"composed added = {n_added_composed}, after = {n_after}"
    )

    # Final scan: every path (original + closed) uses allowed turns
    for paths in path_dict.values():
        for path in paths:
            assert_path_uses_allowed_turns(path, allowed_turn_set)

    assert_subpath_membership(path_dict)
    print("\tSubpath membership and allowed-turn checks passed")
    return path_dict

def output_chosenpaths(out_path, allpaths_tuples):

    with open(out_path, 'w+') as of:
        for list_of_dest_paths in allpaths_tuples:

            for dest_paths in list_of_dest_paths:

                if isinstance(dest_paths,int):
                    out_line = str([dest_paths])
                    # print(f'out_line = {out_line}')

                    of.write(out_line + '\n')
                    continue

                for dest_path in dest_paths:
                    # print(f'dest_path_tuple ({type(dest_path_tuple)}) = {dest_path_tuple}')
                    # out_line = ' '.join(list(dest_path_tuple))
                    # out_line = ' '.join(map(str, dest_path))
                    out_line = str(list(dest_path))

                    # print(f'out_line = {out_line}')

                    of.write(out_line + '\n')

    print(f'Wrote out to {out_path}')

def assure_flow_is_robust(paths):

    print(f'UNIMPLEMENTED :: assure_flow_is_robust. Exiting...')
    quit()

def drive_route_given_allowed_turns_w_vc(input_dict):

    atv_name = input_dict['atv_name']
    topo_name = input_dict['topo_name']
    partition_size = input_dict['partition_size']
    partition_start = input_dict['partition_start']
    max_paths = input_dict['max_paths_per_flow']

    robust = input_dict['robust']
    # robust_type = 'optical'
    robust_type = 'ocs'
    if input_dict['any_link_failure']:
        robust_type = 'all'

    symmetric = input_dict["symmetric"]
    xyzc_dims = input_dict["xyzc_dims"]
    mc_dims = input_dict["mc_dims"]
    sym_type = input_dict["sym_type"]

    just_canons = input_dict["just_canons"]

    all_allowed = input_dict["all_allowed"]
    destination_based = input_dict.get("destination_based", False)

    nonmin = input_dict["nonmin"]
    attempts_factor = input_dict["attempts_factor"]

    sample_minpaths = input_dict["sample_minpaths"]
    sample_seed = input_dict["sample_seed"]
    sample_attempts_factor = input_dict["sample_attempts_factor"]


    if all_allowed:
        base_name = topo_name.split('/')[-1].replace('.map','')
        base_name = f"{base_name}_allallowed"
    else:
        base_name = atv_name.split('/')[-1].replace('.allowvcturns','')


    if symmetric:
        base_name = f"{base_name}_symrouting"

    if destination_based:
        base_name = f"{base_name}_destbased"
        if partition_size != -1:
            print("WARNING: --destination_based with partition_size != -1 may yield "
                  "incomplete subpath closure across partitions; prefer partition_size=-1")

    atpf = ATPathFinder(topo_name, atv_name, all_allowed=all_allowed)

    n_routers = atpf.n_routers
    allowed_turn_set = router_allowed_turn_set(atpf.allowed_turns_list) if destination_based else None

    def maybe_close_and_write(out_path, all_src_path_tuples):
        if destination_based:
            path_dict = apply_destination_based_closure(all_src_path_tuples, allowed_turn_set)
            output_rallpaths_from_path_dict(out_path, path_dict)
        else:
            output_rallpaths(out_path, all_src_path_tuples)


    # symmetric

    if symmetric:# and just_canons:

        my_tpuv4_symmetry = TPUv4_Symmetry(xyzc_dims, mc_dims=mc_dims, sym_type=sym_type)
        canonical_nodes = my_tpuv4_symmetry.get_canonical_nodes()

        n_canons = len(canonical_nodes)

        new_name = f"{base_name}_canonsrcs"

        all_src_path_tuples = []
        for iter_num, src in enumerate(canonical_nodes):
            if partition_size != -1 and (src < partition_start or src >= partition_start + partition_size):
                continue

            print(f'Starting canon src {src}')
            src_paths_tuples = atpf.calculate_paths_single_source(src, max_paths=max_paths, nonmin=nonmin, attempts_factor=attempts_factor, sample_minpaths=sample_minpaths, sample_seed=sample_seed,sample_attempts_factor=sample_attempts_factor)

            # print(f'src_paths_tuples = {src_paths_tuples}')

            print(f'Completed canon src {src} ({iter_num+1}/{n_canons})')


            if robust:
                for dest in range(n_routers):
                    src_paths_tuples[dest] = assure_flow_is_robust(src_paths_tuples[dest], robust_type=robust_type)

            all_src_path_tuples.append(src_paths_tuples)

        if partition_size != -1:
            part_start = src - partition_size + 1
            out_name = new_name + f'_{part_start}pst' + f'_{partition_size}psz' + '.rallpaths'
            allpaths_output_path_prefix = '/scratch/negishi/green456/topologies_and_routing/allpath_lists'
            allpaths_output_path = os.path.join(allpaths_output_path_prefix, out_name)
            maybe_close_and_write(allpaths_output_path, all_src_path_tuples)



            # save memory
            all_src_path_tuples = []


        out_name = new_name + '.rallpaths'
        allpaths_output_path_prefix = '/scratch/negishi/green456/topologies_and_routing/allpath_lists'
        allpaths_output_path = os.path.join(allpaths_output_path_prefix, out_name)
        maybe_close_and_write(allpaths_output_path, all_src_path_tuples)

    if just_canons:
        return

    # all


    all_src_path_tuples = []
    for src in list(range(n_routers)):
        if partition_size != -1 and (src < partition_start or src >= partition_start + partition_size):
            continue


        print(f'Starting src {src}')
        src_paths_tuples = atpf.calculate_paths_single_source(src, max_paths=max_paths, nonmin=nonmin, attempts_factor=attempts_factor, sample_minpaths=sample_minpaths, sample_seed=sample_seed,sample_attempts_factor=sample_attempts_factor)

        # print(f'src_paths_tuples = {src_paths_tuples}')

        print(f'Completed src {src}')

        if robust:
            for dest in range(n_routers):
                src_paths_tuples[dest] = assure_flow_is_robust(src_paths_tuples[dest], robust_type=robust_type)

        all_src_path_tuples.append(src_paths_tuples)

        if len(all_src_path_tuples) >= partition_size and partition_size != -1:
            part_start = src - partition_size + 1
            out_name = base_name + f'_{part_start}pst' + f'_{partition_size}psz' + '.rallpaths'
            allpaths_output_path_prefix = '/scratch/negishi/green456/topologies_and_routing/allpath_lists'
            allpaths_output_path = os.path.join(allpaths_output_path_prefix, out_name)
            maybe_close_and_write(allpaths_output_path, all_src_path_tuples)



            # save memory
            all_src_path_tuples = []
            # input('cont?')


    if partition_size != -1:
        return


    out_name = base_name + '.rallpaths'
    allpaths_output_path_prefix = '/scratch/negishi/green456/topologies_and_routing/allpath_lists'
    allpaths_output_path = os.path.join(allpaths_output_path_prefix, out_name)
    maybe_close_and_write(allpaths_output_path, all_src_path_tuples)



def main():

    parser = argparse.ArgumentParser(description='Route given map and allowed turns')

    parser.add_argument('--topology',type=str,help='.map file to evaluate',required=True)
    parser.add_argument('--allowed_turns_w_vcs','-atv',type=str,help='.allowvcturns file to evaluate',required=True)

    parser.add_argument('--robust',action='store_true',help='assure at least two edge disjoin paths for each flow')
    parser.add_argument('--any_link_failure',action='store_true',help='to only consider failure for all links, not just OCS links')

    parser.add_argument('--out_name','-o',type=str,help='output name (without extension)')

    parser.add_argument('--unthreaded',action='store_true',help='do not use threading')

    parser.add_argument('--partition_start',type=int,default=-1,help='starting for routing')
    parser.add_argument('--partition_size',type=int,default=-1,help='# sources for routing')
    parser.add_argument('--max_paths_per_flow',type=int,help='# paths per (src,dest)')

    parser.add_argument('--symmetric',action='store_true',help='graph is vertex symmetric')
    parser.add_argument('--sym_type',type=str,help="graph symmetry type. default 'trans'",choices=["trans","refl-trans"], default="trans")
    parser.add_argument('--mc_dims',nargs='+',type=int,help='Minimum "cube" of symmetry\'s x, y, and z dimensions. If unspecified then assuming single (zeroth) cube (i.e., x,y,z==cube). Type without parenthesis and use spaces, no commas')
    parser.add_argument('--xyzc_dims',nargs='+',type=int,help='type without parenthesis and use spaces, no commas')

    parser.add_argument('--just_canons',action='store_true',help='route canonical flows')

    parser.add_argument('--all_allowed',action='store_true',help='all turns allowed')

    parser.add_argument("--nonmin",type=int,default=0,help="Allow paths up to min_hops + nonmin (default 0 = shortest only).")
    parser.add_argument("--attempts_factor",type=int,default=200,help="Max candidate paths tried per flow = max_paths_per_flow * attempts_factor (prevents blowups on many-equivalent-path graphs).")

    parser.add_argument("--sample_minpaths",type=int,default=0,help="Sample up to this many *minimal-hop* paths per flow instead of enumerating all shortest paths. 0 = disable (use existing enumeration).")
    parser.add_argument("--sample_seed",type=int,default=1,help="RNG seed for --sample_minpaths.")
    parser.add_argument("--sample_attempts_factor",type=int,default=200,help="Max sampling attempts per flow = sample_minpaths * sample_attempts_factor.")

    parser.add_argument('--destination_based',action='store_true',
                        help='Close under subpaths (add non-shortest suffixes) so MCLB can enforce destination-based routing. Prefer partition_size=-1.')

    parser.add_argument('--verbose','-v',action='store_true',help='debug prints')

    args = parser.parse_args()


    topo_name = args.topology
    out_name = args.out_name
    atv_name = args.allowed_turns_w_vcs
    threaded = not args.unthreaded
    partition_start = args.partition_start
    partition_size = args.partition_size
    max_paths_per_flow = args.max_paths_per_flow
    verbose = args.verbose
    robust = args.robust
    any_link_failure = args.any_link_failure

    symmetric = args.symmetric
    sym_type = None
    xyzc_dims = None
    mc_dims = None
    if symmetric:
        sym_type = args.sym_type
        xyzc_dims = tuple(args.xyzc_dims)
        assert(len(xyzc_dims) == 4)
        mc_dims = tuple(args.mc_dims)
        assert(len(mc_dims) == 3)
    just_canons = args.just_canons
    all_allowed = args.all_allowed
    destination_based = args.destination_based

    nonmin = args.nonmin
    attempts_factor = args.attempts_factor
    sample_minpaths = args.sample_minpaths
    sample_seed = args.sample_seed
    sample_attempts_factor = args.sample_attempts_factor

    if robust:
        print(f'UNIMPLEMENTED :: robust. Exiting...')
        quit()

    input_dict = {'topo_name':topo_name,
                    'atv_name':atv_name,

                    'robust':robust,
                    'any_link_failure':any_link_failure,

                    'out_name':out_name,
                    'threaded':threaded,

                    'partition_start':partition_start,
                    'partition_size':partition_size,
                    'max_paths_per_flow':max_paths_per_flow,

                    "symmetric":symmetric,
                    "sym_type":sym_type,
                    "mc_dims":mc_dims,
                    "xyzc_dims":xyzc_dims,

                    "just_canons":just_canons,
                    "all_allowed":all_allowed,
                    "destination_based":destination_based,

                    "nonmin":nonmin,
                    "attempts_factor":attempts_factor,
                    "sample_minpaths":sample_minpaths,
                    "sample_seed":sample_seed,
                    "sample_attempts_factor":sample_attempts_factor,

                    'verbose':args.verbose,
                    }

    drive_route_given_allowed_turns_w_vc(input_dict)

if __name__ == '__main__':
    main()
