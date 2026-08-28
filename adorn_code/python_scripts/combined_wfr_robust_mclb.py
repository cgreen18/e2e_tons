# std
import argparse
import os
import orjson
import ast
import sys
import time
from collections import defaultdict, deque
from copy import deepcopy

# pipd
import networkit as nk

# Add path to import from new_mclb
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "..", "src"))
from new_mclb import find_mclb, ingest_map as mclb_ingest_map, stream_raw_paths as mclb_stream_raw_paths

class DOR():
    """DOR path finding class from simple_wfr.py"""
    
    INF = 10**9

    def __init__(self, adj_list, xyzc_dims):
        self.xyzc_dims = xyzc_dims
        (x_dim, y_dim, z_dim, cube_dim) = self.xyzc_dims
        n_routers = x_dim*y_dim*z_dim
        self.n_routers = n_routers

        self.G = self.build_graph_nk(n_routers, adj_list)

    @classmethod
    def build_graph_nk(cls, n, adj_list):
        G = nk.graph.Graph(n, weighted=False, directed=False)
        for u, conns in enumerate(adj_list):
            for v in conns:
                G.addEdge(u, v)
        return G

    def r_to_xyz(self, r):
        assert(self.xyzc_dims)
        (x_dim, y_dim, z_dim, cube_dim) = self.xyzc_dims

        xy_slice_size = x_dim*y_dim

        temp_r = r

        z = temp_r // xy_slice_size
        temp_r = temp_r % xy_slice_size
        y = temp_r // x_dim
        x = temp_r % x_dim

        return x,y,z

    def r_to_rel_xyz(self, r):
        xyzc_dims = self.xyzc_dims
        (x_dim, y_dim, z_dim, cube_dim) = xyzc_dims

        r_x,r_y,r_z = self.r_to_xyz(r)

        rel_r_x = r_x % cube_dim
        rel_r_y = r_y % cube_dim
        rel_r_z = r_z % cube_dim

        return rel_r_x, rel_r_y, rel_r_z

    def precompute_geometry_nk(self):
        """Precompute geometry for DOR routing"""
        G = self.G
        xyzc_dims = self.xyzc_dims

        n = G.numberOfNodes()
        rel_coords = [None]*n
        coords = [None]*n
        for u in range(n):
            rel_coords[u] = self.r_to_rel_xyz(u)
            coords[u] = self.r_to_xyz(u)

        nbrX = [[] for _ in range(n)]
        nbrY = [[] for _ in range(n)]
        nbrZ = [[] for _ in range(n)]

        it = G.iterEdges()
        for u, v in it:
            xu, yu, zu = rel_coords[u]
            xv, yv, zv = rel_coords[v]
            dx = (xu != xv); dy = (yu != yv); dz = (zu != zv)
            changed = dx + dy + dz
            if changed != 1:
                continue
            if dx:
                nbrX[u].append(v); nbrX[v].append(u)
            elif dy:
                nbrY[u].append(v); nbrY[v].append(u)
            else:
                nbrZ[u].append(v); nbrZ[v].append(u)

        return coords, nbrX, nbrY, nbrZ

    def dor_shortest_paths_pair(self, s, t, max_paths_per_pair=None):
        """Find DOR shortest paths from s to t"""
        INF = self.INF
        n_routers = self.n_routers
        G = self.G

        coords, nbrX, nbrY, nbrZ = self.precompute_geometry_nk()
        n = n_routers

        def xeq(u, t): return coords[u][0] == coords[t][0]
        def yeq(u, t): return coords[u][1] == coords[t][1]
        def zeq(u, t): return coords[u][2] == coords[t][2]

        def get_nbrs(phase, u):
            if phase == 0: return nbrX[u]
            if phase == 1: return nbrY[u]
            return nbrZ[u]

        def at_boundary(phase, u):
            return (phase == 0 and xeq(u, t)) or (phase == 1 and yeq(u, t)) or (phase == 2 and zeq(u, t))

        nodes = range(n)

        INF = 10**18
        dist = {}
        pred = defaultdict(set)

        dq = deque()
        start = (s, 0)
        dist[start] = 0
        dq.appendleft(start)

        best_goal_dist = None
        goal_states = set()

        while dq:
            u, ph = dq.popleft()
            d = dist[(u, ph)]

            if best_goal_dist is not None and d > best_goal_dist:
                continue

            if ph < 2 and at_boundary(ph, u):
                ns = (u, ph + 1)
                nd = d
                od = dist.get(ns, INF)
                if nd < od:
                    dist[ns] = nd
                    pred[ns] = {(u, ph)}
                    dq.appendleft(ns)
                elif nd == od:
                    pred[ns].add((u, ph))

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

            if ph == 2 and coords[u] == coords[t]:
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


class ATPathFinder():
    """Allowed turns path finder from simple_robust.py"""
    
    verbose = False
    slow = False
    supported_graph_libraries = ['networkit']

    def __init__(self, topo_filepath, allowed_turns_filepath, disallowed_edges=[], graph_library='networkit', verbose=False):
        self.verbose = verbose
        assert(graph_library in self.supported_graph_libraries)
        self.graph_library = graph_library
        self.n_vcs = 2

        self.topo_filepath = topo_filepath
        self.ingest_topo(disallowed_edges=disallowed_edges)

        self.atv_filepath = allowed_turns_filepath
        self.ingest_allowed_turns(disallowed_edges=disallowed_edges)

        self.create_edge_translations()
        self.create_allowed_cdg()

    def ingest_topo(self, disallowed_edges=[]):
        assert(self.topo_filepath is not None)
        self.topo_adjmat, self.topo_adjlist, self.n_routers = self.ingest_a_map_(self.topo_filepath, disallowed_edges=disallowed_edges)

    @classmethod
    def ingest_a_map_(cls, path_name, disallowed_edges=[]):
        this_map = []
        this_adj_list = []

        with open(path_name, 'r') as inf:
            for row in inf:
                r_conns = row.split(' ')
                if '\n' in r_conns:
                    r_conns.remove('\n')

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

        for (i,j) in disallowed_edges:
            this_map[i][j] = 0
            this_map[j][i] = 0
            try:
                this_adj_list[i].remove(j)
            except:
                pass
            try:
                this_adj_list[j].remove(i)
            except:
                pass

        return this_map, this_adj_list, n_routers

    def ingest_allowed_turns(self, disallowed_edges=[]):
        assert(self.atv_filepath)
        _, self.allowed_turns_list = self.ingest_an_allowed_turns_(self.atv_filepath, disallowed_edges=disallowed_edges)

    @classmethod
    def ingest_an_allowed_turns_(cls, path_name, disallowed_edges=[]):
        atvcs_dict = {}
        atvcs_list = []

        with open(path_name, 'r') as inf:
            for line in inf.readlines():
                line_no_newline = line.strip('\n')
                line_w_curly = f'{{ {line_no_newline} }}'

                as_dict = ast.literal_eval(line_w_curly)

                # IMPROVEMENT: Use popitem() for cleaner code
                k, v = as_dict.popitem()

                e0 = (k[0][0], k[0][1])
                e1 = (k[1][0], k[1][1])

                if e0 in disallowed_edges or e1 in disallowed_edges:
                    v = False

                atvcs_dict.update({k: v})

                # if allowed
                if v:
                    atvcs_list.append(k)
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
                    edge_to_label_dict[edge] = cur_label
                    label_to_edge_dict[cur_label] = edge
                    cur_label += 1

        n_labels = cur_label + 1
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
            print(f'UNIMPLEMENTED :: create_allowed_cdg() :: graph_library {self.graph_library}')
            quit()

    def create_an_allowed_cdg_networkit_(self, edges, n):
        n = max(max(u, v) for u, v in edges) + 1
        G = nk.Graph(n, weighted=False, directed=True)
        for u, v in edges:
            G.addEdge(u, v)
        return G

    @classmethod
    def stream_paths_(cls, bfs, t):
        for path in bfs.getPaths(t):
            yield path

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

    def calculate_paths_single_source(self, src, max_paths=None, single_dest=None, attempts_factor=200):
        """
        Improved version using BFS distances computed once and better path enumeration.
        """
        assert(self.allowed_cdg_G)
        assert(self.topo_adjlist)
        assert(self.n_vcs)
        assert(self.n_routers)

        allowed_cdg_G = self.allowed_cdg_G
        topo_adjlist = self.topo_adjlist
        n_vcs = self.n_vcs
        n_routers = self.n_routers
        edge_to_label_dict = self.edge_to_label_dict
        label_to_edge_dict = self.label_to_edge_dict

        nodes_to_remove = []

        src_adjacents = topo_adjlist[src]
        src_edges = [(src, a, v) for a in src_adjacents for v in range(n_vcs)]
        src_labels = [edge_to_label_dict[e] for e in src_edges]

        super_src_label = allowed_cdg_G.addNode()
        nodes_to_remove.append(super_src_label)

        for src_label in src_labels:
            allowed_cdg_G.addEdge(super_src_label, src_label)

        dests = list(range(n_routers))
        if single_dest:
            dests = [single_dest]

        dests_to_super_dests_dict = {}
        for dest in dests:
            if src==dest:
                continue
            super_dest_label = allowed_cdg_G.addNode()
            dests_to_super_dests_dict[dest] = super_dest_label
            nodes_to_remove.append(super_dest_label)

            dest_adjacents = topo_adjlist[dest]
            dest_edges = [(a, dest, v) for a in dest_adjacents for v in range(n_vcs)]
            dest_labels = [self.edge_to_label_dict[e] for e in dest_edges]

            for dest_label in dest_labels:
                allowed_cdg_G.addEdge(dest_label, super_dest_label)

        # IMPROVEMENT: Use BFS with storePaths for shortest paths, but with attempts_factor limit
        bfs = nk.distance.BFS(allowed_cdg_G, source=super_src_label, storePaths=True)
        bfs.run()

        src_paths_tuples = []

        for dest in dests:
            if src==dest:
                src_paths_tuples.append(src)
                continue

            dest_paths_tuples = set()
            super_dest_label = dests_to_super_dests_dict[dest]

            # IMPROVEMENT: Add attempts_factor to prevent blowups on graphs with many equivalent paths
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

                path_as_edges = [self.label_to_edge_dict[l] for l in path_as_labels]
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


class CombinedWFRRobustMCLB():
    """Combined class that handles WFR, Robust, and MCLB"""
    
    verbose = False
    paths_output_dir = "topologies_and_routing/routepath_lists"
    nrl_output_dir = "topologies_and_routing/nr_lists"
    vc_output_dir = "topologies_and_routing/vc_mats"

    def __init__(self, r_map_filepath, allpathlist_filepath, 
                 allowedturns_filepath, xyzc_dims, algorithm='wfr', depth=2):
        """
        algorithm: 'wfr' or 'robust'
        depth: number of neighbor hops (WFR has depth=2)
        """
        self.algorithm = algorithm
        self.depth = depth
        
        self.allpathlist_filepath = allpathlist_filepath
        self.allpath_dict = self.ingest_allpath_list(allpathlist_filepath)

        # Extract base name from allpathlist filename
        base_filename = allpathlist_filepath.split("/")[-1]
        # Remove common extensions
        for ext in ['.rallpaths', '.allpaths', '.paths']:
            if base_filename.endswith(ext):
                base_filename = base_filename[:-len(ext)]
                break
        self.base_name = base_filename

        self.r_map_filepath = r_map_filepath
        self.adj_mat, self.adj_list = self.ingest_a_map(r_map_filepath)

        if allowedturns_filepath:
            self.allowedturns_filepath = allowedturns_filepath
            self.allowed_turns_dict, self.allowed_turns_list = self.ingest_an_allowed_turns(allowedturns_filepath)

        self.xyzc_dims = xyzc_dims
        (x_dim, y_dim, z_dim, _cube_dim) = xyzc_dims
        self.n_routers = x_dim*y_dim*z_dim
        assert(self.n_routers == len(self.adj_mat))

        self.init_ocs_to_edge_set_dict_()

    def init_ocs_to_edge_set_dict_(self):
        n_routers = self.n_routers
        adj_list = self.adj_list

        ocs_to_edge_set_dict = defaultdict(set)
        for i in range(n_routers):
            for j in adj_list[i]:
                ocs = self.calc_ocs_id(i,j)
                ocs_to_edge_set_dict[ocs].add( (i,j) )

        self.ocs_to_edge_set_dict = ocs_to_edge_set_dict

    @classmethod
    def stream_raw_paths(cls, filepath):
        with open(filepath, "rb", buffering=1024*1024) as inf:
            for bline in inf:
                bline = bline.strip()
                if not bline:
                    continue
                row = [int(b) for b in bline.split()]
                yield row

    @classmethod
    def output_pathlist(cls, flat_path_list, out_name):
        out_name = f'{out_name}.paths'
        out_dir = cls.paths_output_dir
        out_path = os.path.join(out_dir, out_name)

        with open(out_path, 'w+') as of:
            of.write(out_name + '\n')
            for path in flat_path_list:
                of.write(f'{path}\n')

        print(f'Wrote pathlist (.paths) to {out_path}')

    @classmethod
    def ingest_a_map(cls, path_name):
        if True:
            print(f'Ingesting r map ({path_name})')

        adj_list = []
        adj_mat = []

        with open(path_name, 'r') as inf:
            for row in inf:
                r_conns = row.split(' ')
                if '\n' in r_conns:
                    r_conns.remove('\n')

                try:
                    r_conns = [int(elem) for elem in r_conns]
                except:
                    r_conns = [int(float(elem)) for elem in r_conns]

                adj_mat.append(r_conns)
                adj_list.append( [r for r, c in enumerate(r_conns) if c > 0] )

        return adj_mat, adj_list

    @classmethod
    def ingest_allpath_list(cls, path_name):
        if True:
            print(f'Ingesting allpath list {path_name}')

        allpath_dict = defaultdict(list)
        for path in cls.stream_raw_paths(path_name):
            s = path[0]
            d = path[-1]
            allpath_dict[(s,d)].append(path)

        return allpath_dict

    @classmethod
    def ingest_an_allowed_turns(cls, path_name):
        atvcs_dict = {}
        atvcs_list = []

        print(f'Ingesting allowed turns ({path_name})')

        with open(path_name, 'r') as inf:
            for line in inf.readlines():
                line_no_newline = line.strip('\n')
                line_w_curly = f'{{ {line_no_newline} }}'

                as_dict = ast.literal_eval(line_w_curly)

                atvcs_dict.update(as_dict)

                if list(as_dict.values())[0]:
                    atvcs_list.append( list(as_dict.keys())[0] )

        return atvcs_dict, atvcs_list

    def r_to_xyz(self, r):
        assert(self.xyzc_dims)
        (x_dim, y_dim, z_dim, cube_dim) = self.xyzc_dims

        xy_slice_size = x_dim*y_dim

        temp_r = r

        z = temp_r // xy_slice_size
        temp_r = temp_r % xy_slice_size
        y = temp_r // x_dim
        x = temp_r % x_dim

        return x,y,z

    def r_to_rel_xyz(self, r):
        xyzc_dims = self.xyzc_dims
        (x_dim, y_dim, z_dim, cube_dim) = xyzc_dims

        r_x,r_y,r_z = self.r_to_xyz(r)

        rel_r_x = r_x % cube_dim
        rel_r_y = r_y % cube_dim
        rel_r_z = r_z % cube_dim

        return rel_r_x, rel_r_y, rel_r_z

    def rel_xyz_is_on_face(self, rel_x, rel_y, rel_z):
        xyzc_dims = self.xyzc_dims
        (x_dim, y_dim, z_dim, cube_dim) = xyzc_dims

        if rel_x == 0 or rel_x == cube_dim - 1:
            return True
        if rel_y == 0 or rel_y == cube_dim - 1:
            return True
        if rel_z == 0 or rel_z == cube_dim - 1:
            return True
        return False

    def calc_opt_conn_type(self, s, d):
        assert(self.xyzc_dims)
        (x_dim, y_dim, z_dim, cube_dim) = self.xyzc_dims

        rel_s_x, rel_s_y, rel_s_z = self.r_to_rel_xyz(s)
        rel_d_x, rel_d_y, rel_d_z = self.r_to_rel_xyz(d)
        
        if (not self.rel_xyz_is_on_face(rel_s_x, rel_s_y, rel_s_z)) or (not self.rel_xyz_is_on_face(rel_d_x, rel_d_y, rel_d_z)):
            return None

        if rel_s_x == cube_dim - 1 and rel_d_x == 0:
            return 'x+'
        if rel_s_x == 0 and rel_d_x == cube_dim - 1:
            return 'x-'

        if rel_s_y == cube_dim - 1 and rel_d_y == 0:
            return 'y+'
        if rel_s_y == 0 and rel_d_y == cube_dim - 1:
            return 'y-'

        if rel_s_z == cube_dim - 1 and rel_d_z == 0:
            return 'z+'
        if rel_s_z == 0 and rel_d_z == cube_dim - 1:
            return 'z-'

    def calc_conn_type(self, s, d):
        assert(self.xyzc_dims)
        (x_dim, y_dim, z_dim, cube_dim) = self.xyzc_dims

        rel_s_x, rel_s_y, rel_s_z = self.r_to_rel_xyz(s)
        rel_d_x, rel_d_y, rel_d_z = self.r_to_rel_xyz(d)

        if rel_s_x == cube_dim - 1 and rel_d_x == 0:
            return 'x','+'
        if rel_s_x == 0 and rel_d_x == cube_dim - 1:
            return 'x','-'

        if rel_s_y == cube_dim - 1 and rel_d_y == 0:
            return 'y','+'
        if rel_s_y == 0 and rel_d_y == cube_dim - 1:
            return 'y','-'

        if rel_s_z == cube_dim - 1 and rel_d_z == 0:
            return 'z','+'
        if rel_s_z == 0 and rel_d_z == cube_dim - 1:
            return 'z','-' 

        if rel_s_x < rel_d_x:
            return 'x', '+'
        if rel_s_x > rel_d_x:
            return 'x', '-'

        if rel_s_y < rel_d_y:
            return 'y', '+'
        if rel_s_y > rel_d_y:
            return 'y', '-'

        if rel_s_z < rel_d_z:
            return 'z', '+'
        if rel_s_z > rel_d_z:
            return 'z', '-'

        print(f"NONE?? {s}->{d} by {self.r_to_rel_xyz(s)}->{self.r_to_rel_xyz(d)}")
        return None, None

    def calc_conn_dim_basic(self, i,j):
        (x_dim, y_dim, z_dim, cube_dim) = self.xyzc_dims

        rel_i_x, rel_i_y, rel_i_z = self.r_to_rel_xyz(i)
        rel_j_x, rel_j_y, rel_j_z = self.r_to_rel_xyz(j)

        conn_type = None

        if(rel_i_y == rel_j_y and rel_i_z == rel_j_z):
            conn_type = 'x'

        if(rel_i_x == rel_j_x and rel_i_z == rel_j_z):
            conn_type = 'y'

        if(rel_i_x == rel_j_x and rel_i_y == rel_j_y):
            conn_type = 'z'
        
        if conn_type == None:
            print(f'{i} ({rel_i_x},{rel_i_y},{rel_i_z}) -> {j} ({rel_j_x},{rel_j_y},{rel_j_z}) => conn_type = {conn_type}')

        return conn_type

    def calc_ocs_id(self, i, j):
        xyzc_dims = self.xyzc_dims
        (x_dim, y_dim, z_dim, cube_dim) = self.xyzc_dims

        if not self.conn_is_optical(i,j):
            return -1

        ij_conn_type, ij_conn_dir = self.calc_conn_type(i,j)

        representative = i
        if '-' in ij_conn_dir:
            representative = j

        rel_x, rel_y, rel_z = self.r_to_rel_xyz(representative)

        if 'x' in ij_conn_type:
            base_val = 0
            return base_val + rel_y + cube_dim*rel_z

        elif 'y' in ij_conn_type:
            base_val = 16
            return base_val + rel_x + cube_dim*rel_z

        elif 'z' in ij_conn_type:
            base_val = 32
            return base_val + rel_x + cube_dim*rel_y

    def conn_is_optical(self,s,d):
        if self.calc_opt_conn_type(s,d) is None:
            return False
        return True

    def path_crosses_ocs(self, path, ocs):
        n_hops = len(path) - 1
        for h in range(n_hops):
            i = path[h]
            j = path[h+1]
            if self.calc_ocs_id(i,j) == ocs:
                return True
        return False

    def get_path_dimensions(self, path):
        """Get list of dimensions for a path"""
        dims = []
        for i in range(len(path) - 1):
            dim = self.calc_conn_dim_basic(path[i], path[i+1])
            dims.append(dim)
        return dims

    def is_wfr_valid_path(self, path_to_nn, path_from_nn):
        """
        Check if WFR path is valid.
        
        For original source to destination s->d that takes a DOR path along dimensions XYZ,
        we may consider neighbors n_s and possibly neighbors of n_s, n_n, such that:
        - path s->n_s->n_n takes dimensions d_0, d_1 in ZYX order (decreasing: Z=2, Y=1, X=0)
        - path n_n->d takes dimensions d_2, d_3, d_4 in XYZ order (increasing: X=0, Y=1, Z=2)
        - d_1 > d_2 where order is X=0, Y=1, Z=2
        
        Examples: 
        - zyXYZ is valid: zy (ZYX order, d_1=y=1) -> XYZ (d_2=X=0), and 1 > 0
        - yzXYZ is not: yz is not ZYX order (should be zy)
        - zYZ is valid: z (d_1=z=2) -> YZ (d_2=Y=1), and 2 > 1
        - xYZ is not: x (d_1=x=0) -> YZ (d_2=Y=1), but 0 is not > 1
        """
        xyz_order = {'x': 0, 'y': 1, 'z': 2}
        
        dims_to_nn = self.get_path_dimensions(path_to_nn)
        dims_from_nn = self.get_path_dimensions(path_from_nn)
        
        # Check if dims_to_nn are in ZYX order (decreasing: Z > Y > X)
        if len(dims_to_nn) >= 2:
            prev_order = 10  # Start high
            for dim in dims_to_nn:
                curr_order = xyz_order.get(dim, -1)
                # if curr_order >= prev_order:  # Should be strictly decreasing
                if curr_order > prev_order:  # allow equal order

                    return False
                prev_order = curr_order
        
        # Check if dims_from_nn are in XYZ order (increasing: X < Y < Z)
        if len(dims_from_nn) > 1:
            prev_order = -1
            for dim in dims_from_nn:
                curr_order = xyz_order.get(dim, -1)
                # if curr_order <= prev_order:  # Should be strictly increasing
                if curr_order < prev_order:  # allow equal order
                    return False
                prev_order = curr_order
        return True
        # Check d_1 > d_2
        if len(dims_to_nn) >= 1 and len(dims_from_nn) >= 1:
            d_1_order = xyz_order.get(dims_to_nn[-1], -1)  # Last dimension of path_to_nn
            d_2_order = xyz_order.get(dims_from_nn[0], -1)  # First dimension of path_from_nn
            # if d_1_order > d_2_order:
            if d_1_order >= d_2_order:  # allow equal order

                return True
        
        return False

    def find_all_wfr_alternatives(self, flow, ocs):
        """Find ALL WFR valid alternatives for all depths from 1 to self.depth"""
        adj_list = self.adj_list
        (s, d) = flow

        if self.verbose:
            print(f"\nFinding ALL alternatives for flow {flow} to avoid crossing ocs {ocs} (depth up to {self.depth})")

        viable_paths = []
        
        # Explore all depths from 1 to self.depth
        # Use BFS to explore neighbor paths
        queue = deque([(s, [s], 0)])  # (current_node, path_so_far, depth)
        visited = set([(s, tuple([s]))])
        
        while queue:
            current_node, current_path, current_depth = queue.popleft()
            
            # If we've made at least one hop, try to find paths to destination
            if current_depth >= 1 and current_depth <= self.depth:
                if current_node != d:
                    # Try paths from current_node to d
                    if (current_node, d) in self.allpath_dict:
                        for nd_path in self.allpath_dict[(current_node, d)]:
                            if len(nd_path) < 2:
                                continue
                            
                            # For depth=1, use simple WFR check
                            if current_depth == 1:
                                sn_conn_type, _sn_conn_dir = self.calc_conn_type(s, current_node)
                                if sn_conn_type is None:
                                    continue
                                j = nd_path[1]
                                first_conn_type, _first_conn_dir = self.calc_conn_type(current_node, j)
                                if first_conn_type is None:
                                    continue
                                
                                xyz_order = {'x': 0, 'y': 1, 'z': 2}
                                if xyz_order.get(sn_conn_type, -1) >= xyz_order.get(first_conn_type, -1):
                                # if True: # or (xyz_order.get(sn_conn_type, -1) >= xyz_order.get(first_conn_type, -1)):
                                    candidate_path = current_path + nd_path[1:]
                                    # print(f"Adding candidate path {candidate_path}")
                                    if not self.path_crosses_ocs(candidate_path, ocs):
                                        viable_paths.append(candidate_path)
                                    # else:
                                    #     print(f"Path {candidate_path} crosses ocs {ocs}")
                            else:
                                # For depth>=2, use extended WFR validity check
                                # The WFR rule applies to s->n_s->n_n (first 3 nodes) and n_n->d
                                # For depth>2, we still check s->n_s->n_n (the first neighbor hop sequence)
                                if len(current_path) >= 3:
                                    # Use first 3 nodes: [s, n_s, n_n] where n_n is the second neighbor
                                    path_to_nn = current_path[:3]
                                    # if True or (self.is_wfr_valid_path(path_to_nn, nd_path)):
                                    if (self.is_wfr_valid_path(path_to_nn, nd_path)):

                                        candidate_path = current_path + nd_path[1:]
                                        # print(f"Adding candidate path {candidate_path}")
                                        if not self.path_crosses_ocs(candidate_path, ocs):
                                            viable_paths.append(candidate_path)
                                        # else:
                                        #     print(f"Path {candidate_path} crosses ocs {ocs}")
                                elif len(current_path) == 2:
                                    # This is depth=1 case, already handled above
                                    pass
            
            # Explore neighbors if we haven't reached depth limit
            if current_depth < self.depth:
                for neighbor in adj_list[current_node]:
                    if neighbor in current_path:  # Avoid cycles
                        continue
                    if neighbor == d and current_depth == 0:
                        continue  # Skip direct connection at depth 0
                    
                    new_path = current_path + [neighbor]
                    path_key = (neighbor, tuple(new_path))
                    
                    if path_key not in visited:
                        visited.add(path_key)
                        queue.append((neighbor, new_path, current_depth + 1))

        return viable_paths

    def create_new_dor_path_alternatives(self, flow, ocs):
        """
        Fallback method: Use DOR solver to find new paths when allpaths dict fails.
        Creates a modified adjacency list with broken OCS removed, then uses DOR to find paths.
        """
        n_routers = self.n_routers
        adj_list = deepcopy(self.adj_list)
        adj_mat = self.adj_mat
        xyzc_dims = self.xyzc_dims

        xyz_order = {'x': 0, 'y': 1, 'z': 2}

        if self.verbose:
            print(f"Finding new DOR paths for flow {flow} to avoid OCS {ocs}")

        # Remove edges that cross the broken OCS
        for i in range(n_routers):
            for j in range(i+1, n_routers):
                if adj_mat[i][j] == 0:
                    continue
                if self.calc_ocs_id(i, j) == ocs:
                    if j in adj_list[i]:
                        adj_list[i].remove(j)
                    if i in adj_list[j]:
                        adj_list[j].remove(i)

        (s, d) = flow

        # Create DOR solver with modified adjacency list
        my_dor = DOR(adj_list, xyzc_dims)

        viable_paths = []

        neighbors = adj_list[s]
        for n in neighbors:
            if n == d:
                continue

            sn_conn_type, _sn_conn_dir = self.calc_conn_type(s, n)
            if sn_conn_type is None:
                continue

            # Get DOR paths from neighbor to destination
            nd_paths = list(my_dor.dor_shortest_paths_pair(n, d))

            for nd_path in nd_paths:
                if len(nd_path) < 2:
                    continue

                candidate_path = [s] + nd_path

                j = nd_path[1] if len(nd_path) > 1 else nd_path[0]
                if s == j:
                    continue

                first_conn_type, _first_conn_dir = self.calc_conn_type(n, j)
                if first_conn_type is None:
                    continue

                if self.verbose:
                    print(f"Candidate DOR path {candidate_path} w/ opt conns {[self.calc_conn_type(candidate_path[h],candidate_path[h+1]) for h in range( len(candidate_path) - 1 )]}")

                # WFR valid if sn (lowercase) > first (uppercase)
                # Note: Using > instead of >= as in simple_wfr.py line 1170
                if not (xyz_order.get(sn_conn_type, -1) > xyz_order.get(first_conn_type, -1)):
                    continue

                # Double-check it doesn't cross the broken OCS (shouldn't happen with modified adj_list, but be safe)
                if self.path_crosses_ocs(candidate_path, ocs):
                    continue

                viable_paths.append(candidate_path)

        if self.verbose:
            print(f"Created {len(viable_paths)} DOR alternatives for flow {flow}")

        return viable_paths

    def is_valid_allowed_turns_path(self, path):
        """Check if path is valid according to allowed turns"""
        max_n_vcs = 2
        allowed_turns_dict = self.allowed_turns_dict

        path_len = len(path) - 1
        if path_len <= 1:
            return True

        dest = path[-1]
        queue = deque()

        for vc_1 in range(max_n_vcs):
            for vc_2 in range(max_n_vcs):
                channel_a_w_vc = (path[0],path[1], vc_1)
                channel_b_w_vc = (path[1],path[2], vc_2)
                turn_w_vc = (channel_a_w_vc,channel_b_w_vc)

                if allowed_turns_dict.get(turn_w_vc, False):
                    new_q_obj = (path[1],path[2], vc_2, 2)
                    queue.append(new_q_obj)

        at_least_one_path = False
        while queue:
            node_a, node_b, last_vc, path_location = queue.popleft()

            if node_b == dest:
                at_least_one_path = True
                break

            if path_location + 1 >= len(path):
                continue

            next_node = path[path_location + 1]
            for next_vc in range(max_n_vcs):
                channel_a_w_vc = ( node_a,node_b, last_vc)
                channel_b_w_vc = (node_b, next_node, next_vc)        
                turn_w_vc = (channel_a_w_vc,channel_b_w_vc)

                if allowed_turns_dict.get(turn_w_vc, False):
                    new_q_obj = (node_b, next_node, next_vc, path_location + 1)
                    queue.append(new_q_obj)

        return at_least_one_path

    def find_all_robust_alternatives(self, flow, ocs, depth):
        """Find ALL Robust valid alternatives for all depths from 0 to depth"""
        adj_list = self.adj_list
        (s, d) = flow

        if self.verbose:
            print(f"\nFinding ALL alternatives for flow {flow} to avoid crossing ocs {ocs} with depth up to {depth}")

        ocs_edges = self.ocs_to_edge_set_dict[ocs]
        atpf = ATPathFinder(self.r_map_filepath, self.allowedturns_filepath, 
                           disallowed_edges=ocs_edges, verbose=self.verbose)

        viable_paths = []
        
        # Use BFS to explore neighbors from depth 0 to depth
        queue = deque([(s, [s], 0)])  # (current_node, path_so_far, depth)
        visited = set([(s, tuple([s]))])
        
        while queue:
            current_node, current_path, current_depth = queue.popleft()
            
            # Try to find paths from current_node to d at any depth (0 to depth)
            if current_depth <= depth and current_node != d:
                # print(f"Finding paths from {current_node} to {d} at depth {current_depth}")
                try:
                    nd_paths_result = atpf.calculate_paths_single_source(current_node, single_dest=d)
                    if nd_paths_result and len(nd_paths_result) > 0:
                        nd_paths = nd_paths_result[0]
                        
                        for nd_path in nd_paths:
                            nd_path = list(nd_path)
                            if len(nd_path) < 1:
                                continue
                            
                            # Construct full path
                            candidate_path = current_path + nd_path[1:]  # Skip first node of nd_path
                            
                            # Check if valid according to allowed turns
                            if self.is_valid_allowed_turns_path(candidate_path):
                                if not self.path_crosses_ocs(candidate_path, ocs):
                                    viable_paths.append(candidate_path)
                except Exception as e:
                    if self.verbose:
                        print(f"Error finding paths from {current_node} to {d}: {e}")
            
            # Explore neighbors if we haven't reached depth limit
            if current_depth < depth:
                for neighbor in adj_list[current_node]:
                    if neighbor in current_path:  # Avoid cycles
                        continue
                    
                    new_path = current_path + [neighbor]
                    path_key = (neighbor, tuple(new_path))
                    
                    if path_key not in visited:
                        visited.add(path_key)
                        queue.append((neighbor, new_path, current_depth + 1))

        return viable_paths

    def collect_all_valid_paths(self, flow, ocs):
        """Collect all valid paths for a flow (original + alternatives)"""
        (s, d) = flow
        depth = self.depth

        all_paths = []
        
        t0 = time.time()
        # Add original paths that don't cross the broken OCS
        if (s, d) in self.allpath_dict:
            for path in self.allpath_dict[(s, d)]:
                if not self.path_crosses_ocs(path, ocs):
                    all_paths.append(path)
        # print(f"PERF : time to find all that don't cross {time.time()- t0}")

        # Add alternatives based on algorithm
        if depth == 0:
            alternatives = []
            if self.verbose:
                print(f"\tAvoiding re-calculating paths")
        elif self.algorithm == 'wfr':
            alternatives = self.find_all_wfr_alternatives(flow, ocs)
            
            # FALLBACK: If no alternatives found using allpaths dict, try DOR solver
            # This is specifically for WFR when all neighbor paths cross the broken OCS
            if len(alternatives) == 0:
                if self.verbose:
                    print(f"No alternatives found for flow {flow} using allpaths, trying DOR solver...")
                dor_alternatives = self.create_new_dor_path_alternatives(flow, ocs)
                alternatives.extend(dor_alternatives)
                
        elif self.algorithm == 'robust':
            alternatives = self.find_all_robust_alternatives(flow, ocs, depth)
        else:
            alternatives = []
        
        all_paths.extend(alternatives)
        
        # FALLBACK: If still no paths found, try direct path finding using ATPathFinder
        # This handles cases where all neighbor-based paths cross the broken OCS
        # Works for both WFR and Robust algorithms when allowed_turns_filepath is available
        if len(all_paths) == 0:
            if self.verbose:
                print(f"No neighbor-based paths found for {flow}. Exiting...")
                quit()
            # fallback_paths = self.find_direct_paths_avoiding_ocs(flow, ocs)
            # if fallback_paths:
            #     all_paths.extend(fallback_paths)
            #     if self.verbose:
            #         print(f"Found {len(fallback_paths)} fallback paths for {flow}")

        if self.verbose:
            print(f"Found {len(all_paths)} total paths for flow {flow}")

        t0 = time.time()        
        # Remove duplicates
        unique_paths = []
        seen = set()
        for path in all_paths:
            path_tuple = tuple(path)
            if path_tuple not in seen:
                seen.add(path_tuple)
                unique_paths.append(path)
        # print(f"PERF : time to find uniques {time.time()- t0}")

        return unique_paths

    def find_direct_paths_avoiding_ocs(self, flow, ocs):
        """
        Fallback method: Use ATPathFinder directly from source to destination
        with broken OCS edges disallowed. This finds paths through any number
        of intermediate hops, not just neighbor-based paths.
        
        This is used when neighbor-based path finding fails to find any valid paths.
        """
        (s, d) = flow
        
        # Only works if we have allowed_turns_filepath (required for ATPathFinder)
        if not hasattr(self, 'allowedturns_filepath') or not self.allowedturns_filepath:
            if self.verbose:
                print(f"Cannot use ATPathFinder fallback for {flow}: no allowed_turns_filepath")
            return []
        
        try:
            ocs_edges = self.ocs_to_edge_set_dict[ocs]
            atpf = ATPathFinder(self.r_map_filepath, self.allowedturns_filepath, 
                               disallowed_edges=ocs_edges, verbose=self.verbose)
            
            # Find paths directly from source to destination
            # Use a reasonable max_paths limit to avoid excessive computation
            paths_result = atpf.calculate_paths_single_source(s, single_dest=d, max_paths=50)
            
            if paths_result and len(paths_result) > 0:
                paths = paths_result[0]
                viable_paths = []
                
                for path in paths:
                    path = list(path)
                    if len(path) < 2:
                        continue
                    
                    # For Robust algorithm, check if path is valid according to allowed turns
                    if self.algorithm == 'robust':
                        if not self.is_valid_allowed_turns_path(path):
                            continue
                    
                    # Double-check it doesn't cross the broken OCS (shouldn't happen with disallowed_edges, but verify)
                    if not self.path_crosses_ocs(path, ocs):
                        viable_paths.append(path)
                
                if self.verbose:
                    print(f"Found {len(viable_paths)} direct paths for {flow} avoiding OCS {ocs}")
                
                return viable_paths
            else:
                if self.verbose:
                    print(f"No paths found by ATPathFinder for {flow} avoiding OCS {ocs}")
        except Exception as e:
            if self.verbose:
                print(f"Error in find_direct_paths_avoiding_ocs for {flow}: {e}")
            import traceback
            if self.verbose:
                traceback.print_exc()
        
        return []

    def handle_ocs_failure(self, ocs, solver_params={}):
        """Handle OCS failure by finding all alternatives and running MCLB"""
        n_routers = self.n_routers
        
        # Collect all valid paths for each flow
        path_dict_for_mclb = defaultdict(list)
        single_path_flows = {}  # Store flows with only one path (MCLB will skip these)
        
        t0 = time.time()
        # Iterate over all flows from allpath_dict
        all_flows = set(self.allpath_dict.keys())
        for s in range(n_routers):
            for d in range(n_routers):
                flow = (s, d)
                if s == d:
                    single_path_flows[flow] = [s]
                    continue
                
                # If flow not in allpath_dict, skip it (shouldn't happen normally)
                if flow not in all_flows:
                    continue

                if self.verbose:
                    print(f"Working on flow {flow}")
                
                # print(f"PERF : time all else : {time.time() - t0}")

                t0 = time.time()
                valid_paths = self.collect_all_valid_paths(flow, ocs)
                # print(f"PERF : time to find all valid : {time.time() - t0}")
                t0 = time.time()

                if len(valid_paths) == 0:
                    print(f"ERROR: No valid paths for flow {flow} after OCS {ocs} failure")
                    print(f"  This flow cannot be routed when OCS {ocs} is broken.")
                    print(f"  Consider:")
                    print(f"    1. Increasing --depth parameter")
                    print(f"    2. Using --algorithm robust (if not already)")
                    print(f"    3. Checking if the topology has sufficient connectivity")
                    # Don't quit - use fallback path
                    if flow in self.allpath_dict and len(self.allpath_dict[flow]) > 0:
                        # Use first original path as last resort (even if it crosses OCS)
                        print(f"  WARNING: Using original path that may cross broken OCS {ocs}")
                        single_path_flows[flow] = self.allpath_dict[flow][0]
                    else:
                        # Minimal fallback: direct connection (will likely fail, but prevents crash)
                        print(f"  WARNING: Using minimal fallback path [s, d]")
                        single_path_flows[flow] = [s, d]
                    continue
                elif len(valid_paths) == 1:
                    # Only one path, no need for MCLB
                    single_path_flows[flow] = valid_paths[0]
                else:
                    # Multiple paths, let MCLB choose
                    path_dict_for_mclb[flow] = valid_paths
                
                if self.verbose:
                    print(f"\tFound # valid paths : {len(valid_paths)}")

            print(f"Completed source {s}")
        
        if self.verbose or True:
            print(f"Found {len(path_dict_for_mclb)} flows with multiple paths for MCLB")
        
        # Run MCLB to select best paths (only for flows with multiple options)
        chosen_paths = {}
        if len(path_dict_for_mclb) > 0:
            current_edge_state = defaultdict(float)
            print(f"Running MCLB for OCS failure {ocs}")
            chosen_paths = find_mclb(path_dict_for_mclb, current_edge_state, self.adj_list,
                                     solver_params=solver_params, robust=False)
        # input("Press Enter to continue...")

        # Convert to flat path list
        flat_path_list = []
        for s in range(n_routers):
            for d in range(n_routers):
                flow = (s, d)
                if s == d:
                    flat_path_list.append([s])
                elif flow in chosen_paths:
                    flat_path_list.append(chosen_paths[flow])
                elif flow in single_path_flows:
                    flat_path_list.append(single_path_flows[flow])
                else:
                    print(f"No paths found for flow {flow}")
                    quit()
                    # # Fallback to first path from allpath_dict
                    # if flow in self.allpath_dict and len(self.allpath_dict[flow]) > 0:
                    #     flat_path_list.append(self.allpath_dict[flow][0])
                    # else:
                    #     flat_path_list.append([s, d])  # Minimal fallback
        
        return flat_path_list

    def handle_all_failures(self, solver_params={}, single_ocs=None):
        """Handle all OCS failures, or a single OCS if specified"""
        base_name = self.base_name
        cube_dim = self.xyzc_dims[3]
        max_ocs = 3*(cube_dim**2)
        depth = self.depth

        if single_ocs is not None:
            # Handle single OCS failure
            if single_ocs < 0 or single_ocs >= max_ocs:
                print(f"ERROR: OCS {single_ocs} is out of range [0, {max_ocs-1}]")
                return
            ocs_list = [single_ocs]
        else:
            # Handle all OCS failures
            ocs_list = range(max_ocs)

        for ocs in ocs_list:
            print(f"\nWorking on OCS {ocs}")
            modified_path_list = self.handle_ocs_failure(ocs, solver_params=solver_params)

            new_name = f"{base_name}_{depth}d_failocs{ocs}"
            self.output_pathlist(modified_path_list, new_name)

            print(f"Completed backup for OCS {ocs} failure")


def define_and_parse_args():
    parser = argparse.ArgumentParser(description='Combined WFR/Robust with MCLB')

    parser.add_argument('--topology', type=str, help='.map file to evaluate', required=True)
    parser.add_argument('--allpathlist', type=str, help='allpath list', required=True)
    parser.add_argument('--allowed_turns_w_vcs', '-atv', type=str, help='.allowvcturns file (required for robust)', default=None)
    parser.add_argument('--xyzc_dims', nargs='+', type=int, help='type without parenthesis and use spaces, no commas', required=True)
    parser.add_argument('--algorithm', type=str, choices=['wfr', 'robust'], default='wfr', help='Algorithm to use: wfr or robust')
    parser.add_argument('--depth', type=int, default=2, help='Number of neighbor hops (WFR has depth=2)')
    parser.add_argument('--ocs_failure', type=int, help='Single OCS failure to handle (if not specified, handles all OCS failures)', default=None)
    parser.add_argument('--verbose', '-v', action='store_true', help='debug prints')

    # MCLB solver params (basic ones)
    parser.add_argument('--time_limit', type=int, help='time limit in minutes')
    parser.add_argument('--threads', type=int, help='# threads total')

    args = parser.parse_args()

    topology = args.topology
    allpathlist = args.allpathlist
    allowed_turns_w_vcs = args.allowed_turns_w_vcs
    algorithm = args.algorithm
    depth = args.depth
    ocs_failure = args.ocs_failure

    xyzc_dims = tuple(args.xyzc_dims)
    assert(len(xyzc_dims) == 4)

    if algorithm == 'robust' and not allowed_turns_w_vcs:
        print("ERROR: --allowed_turns_w_vcs required for robust algorithm")
        parser.print_help()
        sys.exit(1)

    verbose = args.verbose
    
    solver_params = {}
    if args.time_limit:
        solver_params['TimeLimit'] = args.time_limit * 60
    if args.threads:
        solver_params['Threads'] = args.threads

    return topology, allpathlist, allowed_turns_w_vcs, xyzc_dims, algorithm, depth, ocs_failure, verbose, solver_params


def main():
    topology, allpathlist, allowed_turns_w_vcs, xyzc_dims, algorithm, depth, ocs_failure, verbose, solver_params = define_and_parse_args()

    combined = CombinedWFRRobustMCLB(topology, allpathlist, allowed_turns_w_vcs, 
                                     xyzc_dims, algorithm=algorithm, depth=depth)
    combined.verbose = verbose

    combined.handle_all_failures(solver_params=solver_params, single_ocs=ocs_failure)


if __name__ == "__main__":
    main()
