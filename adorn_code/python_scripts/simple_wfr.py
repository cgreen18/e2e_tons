
# std
import argparse
import random
import os
import orjson

from collections import defaultdict, deque
from copy import deepcopy

# pipd
import networkit as nk

class DOR():

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
        """
        Precompute:
        - coords[u] = (x,y,z)
        - nbrX[u], nbrY[u], nbrZ[u]: lists of neighbors of u whose edge is along X/Y/Z
            (i.e., neighbor differs in exactly one coordinate)
        Edges that aren't single-axis steps are ignored for DOR.
        """
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

    def dor_shortest_paths_pair(self, s, t, max_paths_per_pair=None):
        INF = self.INF
        n_routers = self.n_routers
        G = self.G

        # setup
        coords, nbrX, nbrY, nbrZ = self.precompute_geometry_nk()
        n = n_routers


        # helpers
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


        # distance and predecessors in state space
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


class WFR():

    verbose = False

    paths_output_dir = "topologies_and_routing/routepath_lists"
    nrl_output_dir = "topologies_and_routing/nr_lists"
    vc_output_dir = "topologies_and_routing/vc_mats"

    def __init__(self, r_map_filepath, pathlist_filepath, allpathlist_filepath, xyzc_dims):

        if pathlist_filepath:
            self.pathlist_filepath = pathlist_filepath
            self.path_dict = self.ingest_path_list(pathlist_filepath)

        if allpathlist_filepath:
            self.allpathlist_filepath = allpathlist_filepath
            self.allpath_dict = self.ingest_allpath_list(allpathlist_filepath)

        self.base_name = pathlist_filepath.split("/")[-1].replace(".paths","")

        self.r_map_filepath = r_map_filepath
        self.adj_mat, self.adj_list = self.ingest_a_map(r_map_filepath)

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

    # class methods
    ################################################################################

    @classmethod
    def stream_pathlist(cls, filepath):
        with open(filepath, "r", buffering=1024*1024) as inf:
            next(inf, None)  # skip header
            for line in inf:
                line = line.strip()
                if line:
                    yield orjson.loads(line)  # produce one row at a time

    @classmethod
    def stream_raw_paths(cls, filepath):
        with open(filepath, "rb", buffering=1024*1024) as inf:
            for bline in inf:
                bline = bline.strip()
                if not bline:
                    # skip empty
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

                # deal with approximate values (from MIP)
                try:
                    r_conns = [int(elem) for elem in r_conns]
                except:
                    r_conns = [int(float(elem)) for elem in r_conns]

                adj_mat.append(r_conns)
                adj_list.append( [r for r, c in enumerate(r_conns) if c > 0] )


        return adj_mat, adj_list

    @classmethod
    def ingest_path_list(cls, path_name):
        if True:
            print(f'Ingesting path list {path_name}')

        path_dict = {}
        for path in cls.stream_pathlist(path_name):
            flow = ( path[0], path[-1])
            path_dict[flow] = path
        return path_dict

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
    def flatten_path_dict(cls, path_dict, n_routers):

        flat_path_list = []
        for s in range(n_routers):
            for d in range(n_routers):
                if s==d:
                    flat_path_list.append( [s] )
                    continue

                path = path_dict[ (s,d) ]
                flat_path_list.append( path )
        return flat_path_list

    @classmethod
    def convert_pathlist_to_nrl_and_output(cls, flat_path_list, out_name):

        out_name = f'{out_name}.nrl2'
        out_dir = cls.nrl_output_dir
        out_path = os.path.join(out_dir, out_name)

        with open(out_path,'a') as of:
            for path in flat_path_list:
                path_src = path[0]
                path_dest = path[-1]

                n_hops = len(path) - 1


                for i in range(n_hops):
                    hop_src = path[i]
                    hop_dest = path[i+1]

                    of.write(str((path_src, path_dest, hop_src, hop_dest)) + '\n')

        print(f'Wrote NRL (.nrl2) to {out_path}')

    @classmethod
    def output_vc_list(cls, flat_vc2_list, out_name):
        out_name = f'{out_name}.vcmat2'
        out_dir = cls.vc_output_dir
        out_path = os.path.join(out_dir, out_name)

        with open(out_path,'a') as of:
            for vc_tuple in flat_vc2_list:
                of.write(str(vc_tuple) + '\n')
        
        print(f'Wrote VCs (.vcmat2) to {out_path}')

    # geometry
    ################################################################################

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

    def xyz_to_r(self, x,y,z):
        assert(self.xyzc_dims)
        (x_dim, y_dim, z_dim, cube_dim) = self.xyzc_dims

        return x + y*x_dim + z*x_dim*y_dim

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

    def r_to_rel_xyz(self, r):
        xyzc_dims = self.xyzc_dims
        (x_dim, y_dim, z_dim, cube_dim) = xyzc_dims

        r_x,r_y,r_z = self.r_to_xyz(r)

        rel_r_x = r_x % cube_dim
        rel_r_y = r_y % cube_dim
        rel_r_z = r_z % cube_dim

        return rel_r_x, rel_r_y, rel_r_z

    def conn_is_optical(self,s,d):
        if self.calc_opt_conn_type(s,d) is None:
            return False
        return True

    def calc_opt_conn_type(self, s, d):
        assert(self.xyzc_dims)
        (x_dim, y_dim, z_dim, cube_dim) = self.xyzc_dims

        rel_s_x, rel_s_y, rel_s_z = self.r_to_rel_xyz(s)
        rel_d_x, rel_d_y, rel_d_z = self.r_to_rel_xyz(d)
        
        
        if (not self.rel_xyz_is_on_face(rel_s_x, rel_s_y, rel_s_z)) or (not self.rel_xyz_is_on_face(rel_d_x, rel_d_y, rel_d_z)):
            return None

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

    def calc_conn_type(self, s, d):
        assert(self.xyzc_dims)
        (x_dim, y_dim, z_dim, cube_dim) = self.xyzc_dims

        rel_s_x, rel_s_y, rel_s_z = self.r_to_rel_xyz(s)
        rel_d_x, rel_d_y, rel_d_z = self.r_to_rel_xyz(d)

        # x+
        if rel_s_x == cube_dim - 1 and rel_d_x == 0:
            return 'x','+'
        # x-
        if rel_s_x == 0 and rel_d_x == cube_dim - 1:
            return 'x','-'

        # y+
        if rel_s_y == cube_dim - 1 and rel_d_y == 0:
            return 'y','+'
        # y-
        if rel_s_y == 0 and rel_d_y == cube_dim - 1:
            return 'y','-'

        # z+
        if rel_s_z == cube_dim - 1 and rel_d_z == 0:
            return 'z','+'
        # z-
        if rel_s_z == 0 and rel_d_z == cube_dim - 1:
            return 'z','-' 

        # x+
        if rel_s_x < rel_d_x:
            return 'x', '+'
        # x-
        if rel_s_x > rel_d_x:
            return 'x', '-'

        # y+
        if rel_s_y < rel_d_y:
            return 'y', '+'
        # y-
        if rel_s_y > rel_d_y:
            return 'y', '-'

        # z+
        if rel_s_z < rel_d_z:
            return 'z', '+'
        # z-
        if rel_s_z > rel_d_z:
            return 'z', '-'

        input(f"NONE?? {s}->{d} by {self.r_to_rel_xyz(s)}->{self.r_to_rel_xyz(d)}")

    def calc_conn_dim_basic(self, i,j):
        (x_dim, y_dim, z_dim, cube_dim) = self.xyzc_dims

        rel_i_x, rel_i_y, rel_i_z = self.r_to_rel_xyz(i)
        rel_j_x, rel_j_y, rel_j_z = self.r_to_rel_xyz(j)

        # should be just one type!
        conn_type = None

        if(rel_i_y == rel_j_y and rel_i_z == rel_j_z):
            conn_type = 'x'

        if(rel_i_x == rel_j_x and rel_i_z == rel_j_z):
            conn_type = 'y'

        if(rel_i_x == rel_j_x and rel_i_y == rel_j_y):
            conn_type = 'z'
        
        if conn_type == None:
            input(f'{i} ({rel_i_x},{rel_i_y},{rel_i_z}) -> {j} ({rel_j_x},{rel_j_y},{rel_j_z}) => conn_type = {conn_type}')

        return conn_type

    def calc_ocs_id(self, i, j):
        xyzc_dims = self.xyzc_dims
        (x_dim, y_dim, z_dim, cube_dim) = self.xyzc_dims


        if not self.conn_is_optical(i,j):
            return -1

        ij_conn_type, ij_conn_dir = self.calc_conn_type(i,j)

        # x in [0,16)
        # y in [16, 32)
        # z in [32, 48)

        # + => i is representative
        # - => j is representative

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

    def path_crosses_ocs(self, path, ocs):
        n_hops = len(path) - 1
        for h in range(n_hops):
            i = path[h]
            j = path[h+1]
            if self.calc_ocs_id(i,j) == ocs:
                return True
        return False

    def c_to_xyz_cubes(self,c):
        assert(self.xyzc_dims)
        (x_dim, y_dim, z_dim, cube_dim) = self.xyzc_dims

        n_x_cube = x_dim // cube_dim
        n_y_cube = y_dim // cube_dim
        n_z_cube = z_dim // cube_dim

        xy_cube_slice = n_x_cube*n_y_cube

        temp_c = c
        z_cube = temp_c // xy_cube_slice
        temp_c = temp_c % xy_cube_slice
        y_cube = temp_c // n_x_cube
        x_cube = temp_c % n_x_cube

        return x_cube,y_cube,z_cube

    def rel_xyz_and_c_to_r(self,rel_x,rel_y,rel_z, c):
        x,y,z = self.rel_xyz_and_c_to_abs_xyz(rel_x,rel_y,rel_z, c)
        r = self.xyz_to_r(x,y,z)

        return r

    def rel_xyz_and_c_to_abs_xyz(self,rel_x,rel_y,rel_z, c):
        assert(self.xyzc_dims)
        (x_dim, y_dim, z_dim, cube_dim) = self.xyzc_dims

        x_cube,y_cube,z_cube = self.c_to_xyz_cubes(c)

        x = rel_x + cube_dim*x_cube
        y = rel_y + cube_dim*y_cube
        z = rel_z + cube_dim*z_cube

        return x,y,z

    # VC alloc
    ################################################################################

    def determine_datelines_multicube(self):
        (x_dim, y_dim, z_dim, cube_dim) = self.xyzc_dims
        r_map = self.adj_mat

        n_x_cube = x_dim // cube_dim
        n_y_cube = y_dim // cube_dim
        n_z_cube = z_dim // cube_dim

        n_cubes = n_x_cube*n_y_cube*n_z_cube

        # print(f'Working on total problem of {(x_dim,y_dim,z_dim)} w/ # cubes {n_cubes}\n')

        datelines_by_dim_dict = {'x':[],'y':[],'z':[]}

        # x pos
        # -----
        src_rel_x = cube_dim - 1
        dest_rel_x = 0
        # src_rel_x = 0
        # dest_rel_x = cube_dim - 1
        for rel_y in range(cube_dim):
            for rel_z in range(cube_dim):

                conns_list = []
                conns_dict = {}

                if self.verbose:
                    print(f'\nrel_x/y/z {(src_rel_x, rel_y, rel_z)} -> {(dest_rel_x, rel_y, rel_z)}')

                for src_cube in range(n_cubes):

                    src_r = self.rel_xyz_and_c_to_r(src_rel_x, rel_y, rel_z, src_cube)
                    if self.verbose:
                        print(f'\tsrc {src_r:02} @ cube {src_cube}')
                    

                    for dest_cube in range(n_cubes):
                        dest_r = self.rel_xyz_and_c_to_r(dest_rel_x, rel_y, rel_z, dest_cube)

                        if r_map[src_r][dest_r] > 0:
                            if self.verbose:
                                print(f'\t\t-> dest {dest_r:02} @ cube {dest_cube}')
                            conns_list.append((src_cube,dest_cube))
                            conns_dict.update({src_cube : dest_cube})
                
                # input(f'conns_dict={conns_dict}')
                
                queue = list(range(n_cubes))
                loops_list = []

                while queue:
                    cur_cube = queue.pop()
                    next_cube = conns_dict[cur_cube]

                    # new loop
                    loops_list.append([cur_cube])
                    # print(f'new q iter. cur={cur_cube}, next={next_cube}, and loops_list={loops_list}, queue={queue}')

                    while next_cube != cur_cube:
                        loops_list[-1].append(next_cube)
                        queue.remove(next_cube)
                        next_cube = conns_dict[next_cube]

                        # print(f'\tcur={cur_cube}, next={next_cube}, and loops_list={loops_list}, queue={queue}')
                if self.verbose:
                    input(f'loops_list = {loops_list}')

                # list of abs coord tuples
                datelines = []
                dateline_cubes = []
                for loop in loops_list:
                    # select enter/exit zeroth cube as dateline
                    src_dateline_cube = loop[0]
                    try:
                        dest_dateline_cube = loop[1]
                    except:
                        dest_dateline_cube = loop[0]
                    # # select enter/exit min numbered cube as dateline
                    # dateline_cube = min(loop)
                    src_dateline_coord = self.rel_xyz_and_c_to_abs_xyz(src_rel_x, rel_y, rel_z, src_dateline_cube)
                    dest_dateline_coord = self.rel_xyz_and_c_to_abs_xyz(dest_rel_x, rel_y, rel_z, dest_dateline_cube)
                    dateline_coords = (src_dateline_coord, dest_dateline_coord)
                    datelines.append(dateline_coords)
                    dateline_cubes.append((src_dateline_cube,dest_dateline_cube))

                if self.verbose:
                    input(f'dateline(s) : {datelines}  ({dateline_cubes})')
                datelines_by_dim_dict['x'] += datelines
        
        # input(f"datelines_by_dim_dict['x']={datelines_by_dim_dict['x']}")

        # y pos
        # -----
        src_rel_y = cube_dim - 1
        dest_rel_y = 0
        for rel_x in range(cube_dim):
            for rel_z in range(cube_dim):


                conns_list = []
                conns_dict = {}

                if self.verbose:
                    print(f'\nrel_x/y/z {(rel_x, src_rel_y, rel_z)} -> {(rel_x, dest_rel_y, rel_z)}')


                for src_cube in range(n_cubes):

                    src_r = self.rel_xyz_and_c_to_r(rel_x, src_rel_y, rel_z, src_cube)                    
                    if self.verbose:
                        print(f'\tsrc {src_r:02} @ cube {src_cube}')
                    for dest_cube in range(n_cubes):
                        dest_r = self.rel_xyz_and_c_to_r(rel_x, dest_rel_y, rel_z, dest_cube)

                        if r_map[src_r][dest_r] > 0:
                            if self.verbose:
                                print(f'\t\t-> dest {dest_r:02} @ cube {dest_cube}')
                            conns_list.append((src_cube,dest_cube))
                            conns_dict.update({src_cube : dest_cube})
                
                queue = list(range(n_cubes))
                loops_list = []

                while queue:
                    cur_cube = queue.pop()
                    next_cube = conns_dict[cur_cube]

                    # new loop
                    loops_list.append([cur_cube])

                    while next_cube != cur_cube:
                        loops_list[-1].append(next_cube)
                        queue.remove(next_cube)
                        next_cube = conns_dict[next_cube]
                
                if self.verbose:
                    input(f'loops_list = {loops_list}')

                # list of abs coord tuples
                datelines = []
                dateline_cubes= []
                for loop in loops_list:
                    # select enter/exit zeroth cube as dateline
                    src_dateline_cube = loop[0]
                    try:
                        dest_dateline_cube = loop[1]
                    except:
                        dest_dateline_cube = loop[0]
                    # # select enter/exit min numbered cube as dateline
                    # dateline_cube = min(loop)
                    src_dateline_coord = self.rel_xyz_and_c_to_abs_xyz(rel_x, src_rel_y, rel_z, src_dateline_cube)
                    dest_dateline_coord = self.rel_xyz_and_c_to_abs_xyz(rel_x, dest_rel_y, rel_z, dest_dateline_cube)
                    dateline_coords = (src_dateline_coord, dest_dateline_coord)
                    datelines.append(dateline_coords)
                    dateline_cubes.append((src_dateline_cube,dest_dateline_cube))

                if self.verbose:
                    input(f'dateline(s) : {datelines}  ({dateline_cubes})')
                datelines_by_dim_dict['y'] += datelines
    
        # input(f"datelines_by_dim_dict['y']={datelines_by_dim_dict['y']}")

        # z pos
        # -----
        src_rel_z = cube_dim - 1
        dest_rel_z = 0
        for rel_x in range(cube_dim):
            for rel_y in range(cube_dim):


                conns_list = []
                conns_dict = {}

                if self.verbose:
                    print(f'\nrel_x/y/z {(rel_x, rel_y, src_rel_z)} -> {(rel_x, rel_y, dest_rel_z)}')

                for src_cube in range(n_cubes):

                    src_r = self.rel_xyz_and_c_to_r(rel_x, rel_y, src_rel_z, src_cube)
                    if self.verbose:
                        print(f'\tsrc {src_r:02} @ cube {src_cube}')

                    for dest_cube in range(n_cubes):
                        dest_r = self.rel_xyz_and_c_to_r(rel_x, rel_y, dest_rel_z, dest_cube)

                        if r_map[src_r][dest_r] > 0:
                            if self.verbose:
                                print(f'\t\t-> dest {dest_r:02} @ cube {dest_cube}')
                            conns_list.append((src_cube,dest_cube))
                            conns_dict.update({src_cube : dest_cube})

                queue = list(range(n_cubes))
                loops_list = []

                while queue:
                    cur_cube = queue.pop()
                    next_cube = conns_dict[cur_cube]

                    # new loop
                    loops_list.append([cur_cube])

                    while next_cube != cur_cube:
                        loops_list[-1].append(next_cube)
                        queue.remove(next_cube)
                        # cur_cube = next_cube
                        next_cube = conns_dict[next_cube]
                
                if self.verbose:
                    input(f'loops_list = {loops_list}')

                # list of abs coord tuples
                datelines = []
                dateline_cubes= []
                for loop in loops_list:
                    # select enter/exit zeroth cube as dateline
                    src_dateline_cube = loop[0]
                    try:
                        dest_dateline_cube = loop[1]
                    except:
                        dest_dateline_cube = loop[0]
                    # # select enter/exit min numbered cube as dateline
                    # dateline_cube = min(loop)
                    src_dateline_coord = self.rel_xyz_and_c_to_abs_xyz(rel_x, rel_y, src_rel_z, src_dateline_cube)
                    dest_dateline_coord = self.rel_xyz_and_c_to_abs_xyz(rel_x, rel_y, dest_rel_z, dest_dateline_cube)
                    dateline_coords = (src_dateline_coord, dest_dateline_coord)
                    datelines.append(dateline_coords)
                    dateline_cubes.append((src_dateline_cube,dest_dateline_cube))

                if self.verbose:
                    input(f'dateline(s) : {datelines}  ({dateline_cubes})')
                datelines_by_dim_dict['z'] += datelines
        # input(f"datelines_by_dim_dict['z']={datelines_by_dim_dict['z']}")

        # quit()

        return datelines_by_dim_dict

    def path_crosses_dateline_arbitrary(self, path, datelines_by_dim_dict, dateline_dim=None):

        # datelines_by_dim_dict[dateline_dim] =
        #       list of absolute coords pairs (src,dest) that imply
        #       if src, dest OR dest,src coords are equal
        #       and the conn is on the dateline_dim
        #       then it crosses

        relevant_datelines = datelines_by_dim_dict[dateline_dim]
        # print(f'datelines for {dateline_dim} are {relevant_datelines}')

        crosses = False
        cross_idx = -1

        plen = len(path) - 1
        for i in range(plen):

            src = path[i]
            dest = path[i+1]

            src_coord = self.r_to_xyz(src)
            dest_coord = self.r_to_xyz(dest)

            conn_type = self.calc_conn_dim_basic(src, dest)

            # print(f'{src} ({src_coord}) -> {dest} ({dest_coord}) of type {conn_type}')

            if dateline_dim != conn_type:
                # input(f'ignoring as wrong dim')
                continue
            
            # datelines are calculated where sources are of that dateline
            if (src_coord,dest_coord) in relevant_datelines:
                # input(f'src,dest matches a dateline')
                cross_idx = i
                crosses = True

            # datelines are calculated where sources are of that dateline
            if (dest_coord,src_coord) in relevant_datelines:
                # input(f'dest,src matches a dateline')
                cross_idx = i
                crosses = True

        return crosses, cross_idx

    def reassign_vcs(self, flat_path_list):

        datelines_by_dim_dict = self.determine_datelines_multicube()

        flat_vc2_list = []

        for path in flat_path_list:


            path_src = path[0]
            path_dest = path[-1]

            plen = len(path) - 1
            if plen <= 1:
                flat_vc2_list.append( (path_src, path_dest, path_src, 0) )
                continue


            dim_subpaths = {'x':[],'y':[],'z':[]}

            path_dims = [ self.calc_conn_dim_basic(path[i],path[i+1])  for i in range(plen) ]

            cur_dim = path_dims[0]
            cur_idx = 0
            
            for i in range(1,plen):

                next_dim = path_dims[i]

                if cur_dim == next_dim:
                    continue
                else:
                    subpath = path[cur_idx:i+1]

                    # SUBPATHS
                    dim_subpaths[cur_dim].append( subpath)

                    # PATHS
                    # combine into megapath
                    # dim_subpaths[cur_dim] += subpath

                    cur_dim = next_dim
                    cur_idx = i
            
            # print(f'after, cur_dim={cur_dim} and cur_idx={cur_idx}')

            # after
            if cur_idx != plen:
                subpath = path[cur_idx:]

                # SUBPATHS
                dim_subpaths[cur_dim].append( subpath)

                # FULL PATHS
                # combine into megapath
                # dim_subpaths[cur_dim] += subpath

            if self.verbose:
                print(f'dim_subpaths {dim_subpaths}')

            # SUBPATHS
            for dim, subpaths in dim_subpaths.items():
                for subpath in subpaths:
                    # print(f'subpath={subpath}')
                    # does_cross = self.path_crosses_dateline(subpath, dateline_dim=dim)
                    # does_cross, cross_idx = self.path_crosses_dateline(subpath, dateline_dim=dim)
                    does_cross, cross_idx = self.path_crosses_dateline_arbitrary(subpath, datelines_by_dim_dict, dateline_dim=dim)

                    # does_cross, cross_idx = self.path_crosses_dateline_considering_twist(subpath, dateline_dim=dim)


                    if self.verbose:
                        print(f'subpath {subpath} on dim {dim} crosses? {does_cross}')

                    vc = 0
                    # if does_cross:
                    #     vc = 1

                    subplen = len(subpath) - 1
                    for i in range(subplen):
                        cur_node = subpath[i]

                        if does_cross and i >= cross_idx:
                            vc = 1


                        flat_vc2_list.append( (path_src, path_dest, cur_node, vc) )

        return flat_vc2_list

    def valid_wfr_path(self, sn_conn_type, first_conn_type):
        # valid if sn (lowercase) >= first (uppercase) => continue/skip if not

        xyz_order = {'x':0, 'y':1, 'z':2}
        if not (xyz_order[sn_conn_type] >= xyz_order[first_conn_type]):
        # if not (xyz_order[sn_conn_type] > xyz_order[first_conn_type]): # this fails
            return False

        return True

    # big workers
    ################################################################################

    def find_path_alternatives(self, flow, ocs, use_allpaths=False):
        adj_list = self.adj_list
        path_dict = self.path_dict
        allpath_dict = self.allpath_dict

        xyz_order = {'x':0, 'y':1, 'z':2}

        (s, d) = flow

        if self.verbose:
            print(f"\nFinding alternatives for flow {flow} to avoid crossing ocs {ocs}")
            print(f"Original path : {path_dict[flow]}")


        neighbors = adj_list[s]

        viable_paths = []
        for n in neighbors:
            if n==d: continue
            sn_conn_type, _sn_conn_dir = self.calc_conn_type(s,n)

            if self.verbose:
                print(f"Considering neighbor {n} where {s}->{n} of type {sn_conn_type}")

            if use_allpaths:
                nd_paths = allpath_dict[ (n,d) ]
            else:
                nd_paths = [ path_dict[ (n,d) ] ]

            for nd_path in nd_paths:

                j = nd_path[1]
                first_conn_type, _first_conn_dir = self.calc_conn_type(n,j)

                candidate_path = [s] + nd_path

                if self.verbose:
                    print(f"Candidate path {candidate_path} w/ opt conns {[self.calc_conn_type(candidate_path[h],candidate_path[h+1]) for h in range( len(candidate_path) - 1 )]}")

                # viable iff WFR valid AND not crosses bad OCS
                if not self.valid_wfr_path(sn_conn_type, first_conn_type):
                    continue

                if self.path_crosses_ocs(candidate_path, ocs):
                    continue

                viable_paths.append( candidate_path )

        return viable_paths

    def create_new_path_alternatives(self, flow, ocs):
        n_routers = self.n_routers
        adj_list = deepcopy(self.adj_list)
        adj_mat = self.adj_mat
        xyzc_dims = self.xyzc_dims

        xyz_order = {'x':0, 'y':1, 'z':2}

        if self.verbose:
            print(f"Finding new DOR paths for flow {flow} to avoid OCS {ocs}")


        for i in range(n_routers):
            for j in range(i+1,n_routers):
                if adj_mat[i][j] == 0: continue
                if self.calc_ocs_id(i,j) == ocs:
                    adj_list[i].remove(j)
                    adj_list[j].remove(i)

        (s,d) = flow

        my_dor = DOR( adj_list, xyzc_dims )

        viable_paths = []

        neighbors = adj_list[s]
        for n in neighbors:
            sn_conn_type, _sn_conn_dir = self.calc_conn_type(s,n)

            nd_paths = list(my_dor.dor_shortest_paths_pair(n,d))

            for nd_path in nd_paths:

                # print(f"nd_path = {nd_path}")
                # for p in nd_path:
                #     print(f"p = {p}")

                candidate_path = [s] + nd_path

                j = nd_path[1]
                if s==j: continue

                first_conn_type, _first_conn_dir = self.calc_conn_type(n,j)


                if self.verbose:
                    print(f"Candidate path {candidate_path} w/ opt conns {[self.calc_conn_type(candidate_path[h],candidate_path[h+1]) for h in range( len(candidate_path) - 1 )]}")

                # viable iff WFR valid AND not crosses bad OCS

                # # WFR valid if...
                # # yXYZ or zXYZ or xXYZ
                # # zYZ or yYZ
                # # zZ
                # # invalid if...
                # # -
                # # xYZ?
                # # xZ or yZ?
                # # valid if sn (lowercase) >= first (uppercase) => continue/skip if not
                # if not (xyz_order[sn_conn_type] >= xyz_order[first_conn_type]):
                if not (xyz_order[sn_conn_type] > xyz_order[first_conn_type]): # this fails
                    continue

                if self.path_crosses_ocs(candidate_path, ocs):
                    continue

                viable_paths.append( candidate_path )

        # input(f"Created alternatives {viable_paths}")

        return viable_paths


    def handle_ocs_failure(self, ocs):
        n_routers = self.n_routers
        path_dict = deepcopy(self.path_dict)

        replacements = {}

        for flow, path in path_dict.items():
            if len(path) == 1: continue

            flow = ( path[0], path[-1])

            # print(f"Considering path {path}")

            if self.path_crosses_ocs(path,ocs):
                # alternatives = self.find_path_alternatives(flow,ocs)

                # if len(alternatives) == 0:
                #     print(f"Trying allpaths")
                alternatives = self.find_path_alternatives(flow,ocs, use_allpaths=True)

                # if len(alternatives) == 0:
                #     alternatives = self.create_new_path_alternatives(flow,ocs)

                # print(f"=> alternatives {alternatives}")

                # TODO better selection alg
                # for now, random
                alternative_choice = random.choice(alternatives)
                replacements[flow] = alternative_choice


        for flow, path in replacements.items():
            path_dict[flow] = path

        flat_path_list = self.flatten_path_dict(path_dict, n_routers)
        
        return flat_path_list

    def handle_all_failures(self):

        base_name = self.base_name

        cube_dim = self.xyzc_dims[3]

        for ocs in range(3*(cube_dim**2)):
            print(f"\nWorking on OCS {ocs}")
            modified_path_list = self.handle_ocs_failure(ocs)

            new_name = f"{base_name}_failocs{ocs}"
            self.output_pathlist(modified_path_list, new_name)
            self.convert_pathlist_to_nrl_and_output(modified_path_list, new_name)

            flat_vc2_list = self.reassign_vcs(modified_path_list)
            self.output_vc_list(flat_vc2_list, new_name)

            print(f"Completed backup for OCS {ocs} failure")

def define_and_parse_args():

    parser = argparse.ArgumentParser(description='...')

    parser.add_argument('--topology',type=str,help='.map file to evaluate',required=True)
    parser.add_argument('--pathlist',type=str,help='path list',required=True)
    parser.add_argument('--allpathlist',type=str,help='allpath list',required=True)
    parser.add_argument('--xyzc_dims',nargs='+',type=int,help='type without parenthesis and use spaces, no commas', required=True)

    parser.add_argument('--verbose','-v',action='store_true',help='debug prints')


    args = parser.parse_args()

    topology = args.topology
    pathlist = args.pathlist
    allpathlist = args.allpathlist

    xyzc_dims = tuple(args.xyzc_dims)
    assert(len(xyzc_dims) == 4)

    verbose = args.verbose
    # class var
    Robust.verbose = verbose
    return topology, pathlist, allpathlist, xyzc_dims

def main():

    topology, pathlist, allpathlist, xyzc_dims = define_and_parse_args()

    my_wfr = WFR(topology, pathlist, allpathlist, xyzc_dims)

    my_wfr.handle_all_failures()    


if __name__ == "__main__":
    main()