
# std
import argparse
import random
import os
import orjson
import ast
from collections import defaultdict, deque
from copy import deepcopy

# pipd
import networkit as nk


class ATPathFinder():

    # class vars
    ############

    verbose = False
    slow  = False

    supported_graph_libraries = ['networkit']

    def __init__(self, topo_filepath, allowed_turns_filepath, disallowed_edges=[], graph_library='networkit', verbose=False):

        # basic
        self.verbose = verbose

        assert(graph_library in self.supported_graph_libraries)
        self.graph_library = graph_library

        # TODO parse from allowed turns
        self.n_vcs = 2

        # needs topo_filepath
        self.topo_filepath = topo_filepath
        # defines topo_adjmat, topo_adjlist, n_routers
        self.ingest_topo(disallowed_edges=disallowed_edges)

        # needs atv_filepath
        self.atv_filepath = allowed_turns_filepath
        # defines allowed_turns_list
        self.ingest_allowed_turns(disallowed_edges=disallowed_edges)

        # needs allowed_turns_list
        # defines edge_to_label_dict, label_to_edge_dict, n_labels
        self.create_edge_translations()

        # needs allowed_turns_list
        # defines allowed_cdg_G
        self.create_allowed_cdg()

    def ingest_topo(self, disallowed_edges=[]):
        assert(self.topo_filepath is not None)
        self.topo_adjmat, self.topo_adjlist, self.n_routers = self.ingest_a_map_(self.topo_filepath, disallowed_edges=disallowed_edges)

    # _ implies it returns instead of setting self vars
    # these will be class methods
    @classmethod
    def ingest_a_map_(cls, path_name, disallowed_edges=[]):

        # if True:
        #     print(f'Ingesting r map ({path_name})')

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

        # print(f'Ingesting allowed turns ({path_name})')
        # print(f"disallowed_edges = {disallowed_edges}")

        with open(path_name, 'r') as inf:
            for line in inf.readlines():
                line_no_newline = line.strip('\n')
                line_w_curly = f'{{ {line_no_newline} }}'

                as_dict = ast.literal_eval(line_w_curly)

                # print(f"as_dict = {as_dict}")


                e0 = (list(as_dict.keys())[0][0][0], list(as_dict.keys())[0][0][1])
                e1 = (list(as_dict.keys())[0][1][0], list(as_dict.keys())[0][1][1])


                if e0 in disallowed_edges or e1 in disallowed_edges:
                    k = list(as_dict.keys())[0]
                    as_dict[k] = False

                atvcs_dict.update(as_dict)

                # print(f"as_dict = {as_dict}")

                if list(as_dict.values())[0]:
                    atvcs_list.append( list(as_dict.keys())[0] )
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
            print(f'UNIMPLEMENTED :: create_allowed_cdg() :: graph_library {graph_library}')
            quit()

    def create_an_allowed_cdg_networkit_(self, edges, n):

        n = max(max(u, v) for u, v in edges) + 1
        G = nk.Graph(n, weighted=False, directed=True)
        for u, v in edges:
            G.addEdge(u, v)

        # print("NetworKit:", G.numberOfNodes(), "nodes,", G.numberOfEdges(), "edges")

        return G


    ################################################################################

    @classmethod
    def stream_paths_(cls, bfs, t):
        for path in bfs.getPaths(t):  # returns all shortest s→t paths
            yield path

    def calculate_paths_single_source(self, src, max_paths=None, single_dest=None): # targets = None
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

        # for later cleanup
        nodes_to_remove = []
        edges_to_remove = []

        src_adjacents = topo_adjlist[src]
        src_edges = [(src, a, v) for a in src_adjacents for v in range(n_vcs)]
        src_labels = [edge_to_label_dict[e] for e in src_edges]

        # TODO figure out same graph object and threading?
        super_src_label = allowed_cdg_G.addNode()
        # print(f'Added node {super_src_label}')
        nodes_to_remove.append(super_src_label)

        for src_label in src_labels:
            allowed_cdg_G.addEdge(super_src_label, src_label)
            # print(f'Added edge {super_src_label}->{src_label} aka ss->{label_to_edge_dict[src_label]}')

        # print(f'Completed sources')

        dests = list(range(n_routers))
        if single_dest:
            dests = [single_dest]

        # super_dests_to_dests_dict = {}
        dests_to_super_dests_dict = {}
        for dest in dests:
            if src==dest:
                continue
            super_dest_label = allowed_cdg_G.addNode()
            # print(f'Added node {super_dest_label}')
            dests_to_super_dests_dict[dest] = super_dest_label
            # super_dests_to_dests_dict[super_dest_label] = dest
            nodes_to_remove.append(super_dest_label)

            dest_adjacents = topo_adjlist[dest]
            dest_edges = [(a, dest, v) for a in dest_adjacents for v in range(n_vcs)]
            dest_labels = [self.edge_to_label_dict[e] for e in dest_edges]

            for dest_label in dest_labels:
                allowed_cdg_G.addEdge(dest_label, super_dest_label)
                # print(f'Added edge {dest_label}->{super_dest_label} aka {label_to_edge_dict[dest_label]}->sd')

        # print(f'Completed destinations')

        # networkit alg
        bfs = nk.distance.BFS(allowed_cdg_G, source=super_src_label, storePaths=True)  # BFS since unit weights
        bfs.run()

        src_paths_tuples = []

        for dest in dests:
            if src==dest:
                src_paths_tuples.append(src)
                continue
            # print(f'Working on dest {dest}')
            # avoid redundancy
            dest_paths_tuples = set()
            super_dest_label = dests_to_super_dests_dict[dest]
            for full_path in self.stream_paths_(bfs, super_dest_label):

                path_as_labels = full_path[1:-1]
                path_as_edges = [self.label_to_edge_dict[l] for l in path_as_labels]

                path_as_list = [e[0] for e in path_as_edges] + [path_as_edges[-1][1]]
                # path = (e[0] for e in path_as_edges) + (path_as_edges[-1][1])
                path = tuple(path_as_list)

                # input(f'full_path = {full_path}, path_as_labels = {path_as_labels}, path_as_edges = {path_as_edges}, path = {path}')

                dest_paths_tuples.add(path)

                if max_paths and len(dest_paths_tuples) >= max_paths:
                    # input(f'stopping early for max_paths')
                    break

            # input(f'Completed {src}->{dest} : paths {dest_paths_tuples}')
            src_paths_tuples.append(list(dest_paths_tuples))

        # cleanup
        # removing node removes all its edges too
        for node in nodes_to_remove:
            allowed_cdg_G.removeNode(node)

        return src_paths_tuples

class Robust():

    verbose = False

    paths_output_dir = "topologies_and_routing/routepath_lists"
    nrl_output_dir = "topologies_and_routing/nr_lists"
    vc_output_dir = "topologies_and_routing/vc_mats"

    def __init__(self, r_map_filepath, pathlist_filepath, allpathlist_filepath, allowedturns_filepath, xyzc_dims):

        self.pathlist_filepath = pathlist_filepath
        self.path_dict = self.ingest_path_list(pathlist_filepath)

        self.allpathlist_filepath = allpathlist_filepath
        self.allpath_dict = self.ingest_allpath_list(allpathlist_filepath)

        self.base_name = pathlist_filepath.split("/")[-1].replace(".paths","")

        self.r_map_filepath = r_map_filepath
        self.adj_mat, self.adj_list = self.ingest_a_map(r_map_filepath)

        self.allowedturns_filepath = allowedturns_filepath
        self.allowed_turns_dict, self.allowed_turns_list = self.ingest_an_allowed_turns(allowedturns_filepath)

        self.xyzc_dims = xyzc_dims
        (x_dim, y_dim, z_dim, _cube_dim) = xyzc_dims
        self.n_routers = x_dim*y_dim*z_dim
        assert(self.n_routers == len(self.adj_mat))

        self.init_ocs_to_edge_set_dict_()

        # TODO as CLA
        self.max_n_vcs = 2

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
        path_dict = {}
        for path in cls.stream_pathlist(path_name):
            flow = ( path[0], path[-1])
            path_dict[flow] = path
        return path_dict

    @classmethod
    def ingest_allpath_list(cls, path_name):
        if True:
            print(f'Ingesting path list {path_name}')

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


    def calc_vcs_of_path(self, path):
        assert(self.max_n_vcs is not None)
        assert(self.allowed_turns_dict is not None)
        max_n_vcs = self.max_n_vcs
        allowed_turns_dict = self.allowed_turns_dict

        path_len = len(path) - 1
        # any path of length 1 (e.g. [n0,n1]) or less valid
        if path_len <= 1:
            return [0]

        if self.verbose:
            print(f'-'*20)
            print(f'path={path}')

        dest = path[-1]
        queue = deque()
        # list of paths
        # list of list of turns
        # list of list (paths) of tuples (turns)
        vcs_of_paths_list = []
        # get first turns of path (if valid)
        for vc_1 in range(max_n_vcs):
            for vc_2 in range(max_n_vcs):
                channel_a_w_vc = (path[0],path[1], vc_1)
                channel_b_w_vc = (path[1],path[2], vc_2)
                turn_w_vc = (channel_a_w_vc,channel_b_w_vc)

                if allowed_turns_dict[turn_w_vc]:
                    # queue holds paths of turns
                    new_q_obj = [turn_w_vc]
                    queue.append(new_q_obj)
                if self.verbose:
                    print(f'turn {turn_w_vc} allowed? {allowed_turns_dict[turn_w_vc]}')

        a_path_of_turns = None
        while queue:
            list_of_turns = queue.popleft()

            most_recent_turn = list_of_turns[-1]
            most_recent_node = most_recent_turn[1][1]
            before_most_recent_node = most_recent_turn[0][1]

            if self.verbose:
                print(f'list_of_turns={list_of_turns}, most_recent_turn={most_recent_turn}, most_recent_node={most_recent_node}')

            # arrived
            if most_recent_node == dest:
                a_path_of_turns = list_of_turns
                break

            # else continue searching
            cur_loc = path.index(most_recent_node)
            next_node = path[cur_loc + 1]
            cur_vc =  most_recent_turn[1][2]
            for next_vc in range(max_n_vcs):
                channel_a_w_vc = ( before_most_recent_node,most_recent_node, cur_vc)
                channel_b_w_vc = (most_recent_node, next_node, next_vc)        
                turn_w_vc = (channel_a_w_vc,channel_b_w_vc)

                if allowed_turns_dict[turn_w_vc]:
                    new_list_of_turns = list_of_turns + [turn_w_vc]
                    queue.append(new_list_of_turns)
                
                if self.verbose:
                    print(f'turn {turn_w_vc} allowed? {allowed_turns_dict[turn_w_vc]}')


        # deconstruct
        path_vcs = []
        # manually set first
        first_vc = a_path_of_turns[0][0][2]
        path_vcs.append(first_vc)
        for turn in a_path_of_turns:
            turn_vc = turn[1][2]
            path_vcs.append(turn_vc)

        if self.verbose:
            print(f'a_path_of_turns = {a_path_of_turns} w/ vcs {path_vcs}')

        return a_path_of_turns, path_vcs

    def reassign_vcs(self, flat_path_list):
        assert(self.allowed_turns_dict is not None)
        assert(self.max_n_vcs != -1)

        flat_vc2_list = []

        for path in flat_path_list:

            path_src = path[0]
            path_dest = path[-1]

            plen = len(path) - 1
            if plen <= 1:
                # vc_matrix[path_src][path_dest][path_src] = 0
                flat_vc2_list.append( (path_src, path_dest, path_src, 0) )
                continue

            # need to handle as bfs as no clear way to move along channels and vcs
            _a_path_of_turns, vcs_along_path = self.calc_vcs_of_path(path)

            # plen iters
            for loc, vc in enumerate(vcs_along_path):
                cur_node = path[loc]
                
                flat_vc2_list.append( (path_src, path_dest, cur_node, vc) )

                if self.verbose:
                    print(f'Node {cur_node} will use downstream vc {vc}')

        return flat_vc2_list

    # big workers
    ################################################################################


    def is_valid_allowed_turns_path(self, path):
        assert(self.max_n_vcs is not None)
        assert(self.allowed_turns_dict is not None)
        max_n_vcs = self.max_n_vcs
        allowed_turns_dict = self.allowed_turns_dict
        verbose = self.verbose

        path_len = len(path) - 1
        # any path of length 1 (e.g. [n0,n1]) or less valid
        if path_len <= 1:
            return True

        if verbose:
            print(f'-'*20)
            print(f'path={path}')

        dest = path[-1]
        queue = deque()
        # get first turns of path (if valid)
        for vc_1 in range(max_n_vcs):
            for vc_2 in range(max_n_vcs):
                channel_a_w_vc = (path[0],path[1], vc_1)
                channel_b_w_vc = (path[1],path[2], vc_2)
                turn_w_vc = (channel_a_w_vc,channel_b_w_vc)

                if allowed_turns_dict[turn_w_vc]:
                    # a -> b -> c
                    # c is 2nd node of path
                    # add (b, c, last_vc, 2)
                    new_q_obj = (path[1],path[2], vc_2, 2)
                    queue.append(new_q_obj)
                if verbose:
                    print(f'turn {turn_w_vc} allowed? {allowed_turns_dict[turn_w_vc]}')

        at_least_one_path = False
        while queue:
            node_a, node_b, last_vc, path_location = queue.popleft()

            if verbose:
                print(f'node_b={node_b} & iter {path_location}')

            # arrived
            if node_b == dest:
                at_least_one_path = True
                break

            # else continue searching
            next_node = path[path_location + 1]
            for next_vc in range(max_n_vcs):
                channel_a_w_vc = ( node_a,node_b, last_vc)
                channel_b_w_vc = (node_b, next_node, next_vc)        
                turn_w_vc = (channel_a_w_vc,channel_b_w_vc)

                if allowed_turns_dict[turn_w_vc]:
                    new_q_obj = (node_b, next_node, next_vc, path_location + 1)
                    queue.append(new_q_obj)
                
                if verbose:
                    print(f'turn {turn_w_vc} allowed? {allowed_turns_dict[turn_w_vc]}')

        if verbose:
            print(f'at_least_one_path = {at_least_one_path}')

        return at_least_one_path

    def create_new_path_alternatives(self, flow, ocs):
        adj_list = self.adj_list
        r_map_filepath = self.r_map_filepath
        allowedturns_filepath = self.allowedturns_filepath

        (s,d) = flow

        if self.verbose:
            print(f"\nFinding alternatives for flow {flow} to avoid crossing ocs {ocs}")
            # print(f"Original path : {path_dict[flow]}")

        ocs_edges = self.ocs_to_edge_set_dict[ocs]

        atpf = ATPathFinder(r_map_filepath, allowedturns_filepath, disallowed_edges=ocs_edges)

        neighbors = adj_list[s]

        viable_paths = []
        for n in neighbors:
            if n==d: continue
    

            if self.verbose:
                print(f"Considering neighbor {n}")
            
            nd_paths = atpf.calculate_paths_single_source(n, single_dest=d)[0]

            if self.verbose:
                print(f"nd_paths = {nd_paths}")

            for nd_path in nd_paths:
                nd_path = list(nd_path)
                j = nd_path[1]
                if s==j: continue

                candidate_path = [s] + nd_path

                if self.verbose:
                    print(f"Candidate path {candidate_path}")
                    print(f"\tw/ opt conns {[self.calc_conn_type(candidate_path[h],candidate_path[h+1]) for h in range( len(candidate_path) - 1 )]}")

                # viable iff AT valid AND not crosses bad OCS

                if not self.is_valid_allowed_turns_path(candidate_path):
                    continue

                if self.path_crosses_ocs(candidate_path, ocs):
                    continue

                viable_paths.append( candidate_path )

        return viable_paths

    def find_path_alternatives(self, flow, ocs, use_allpaths=False):
        adj_list = self.adj_list
        path_dict = self.path_dict
        allpath_dict = self.allpath_dict

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

                # silly to return to self?
                if s==j: continue

                first_conn_type, _first_conn_dir = self.calc_conn_type(n,j)

                candidate_path = [s] + nd_path

                if self.verbose:
                    print(f"Candidate path {candidate_path} w/ opt conns {[self.calc_conn_type(candidate_path[h],candidate_path[h+1]) for h in range( len(candidate_path) - 1 )]}")

                # viable iff AT valid AND not crosses bad OCS

                if not self.is_valid_allowed_turns_path(candidate_path):
                    continue

                if self.path_crosses_ocs(candidate_path, ocs):
                    continue

                viable_paths.append( candidate_path )

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
                #     print(f"Trying to find new path")
                #     alternatives = self.create_new_path_alternatives(flow, ocs)

                if len(alternatives) == 0:
                    print(f"Trying allpaths")
                    alternatives = self.find_path_alternatives(flow, ocs, use_allpaths=True)

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

        print(f"\nWorking on OCS {-2}")
        modified_path_list = self.handle_ocs_failure(-2)

        new_name = f"{base_name}_failocs{ocs}"
        self.output_pathlist(modified_path_list, new_name)
        self.convert_pathlist_to_nrl_and_output(modified_path_list, new_name)

        flat_vc2_list = self.reassign_vcs(modified_path_list)
        self.output_vc_list(flat_vc2_list, new_name)

        print(f"Completed backup for OCS {ocs} failure")
        # input(f"Completed backup for OCS {ocs} failure")    

        for ocs in range(13, 3*(cube_dim**2)):
            print(f"\nWorking on OCS {ocs}")
            modified_path_list = self.handle_ocs_failure(ocs)

            new_name = f"{base_name}_failocs{ocs}"
            self.output_pathlist(modified_path_list, new_name)
            self.convert_pathlist_to_nrl_and_output(modified_path_list, new_name)

            flat_vc2_list = self.reassign_vcs(modified_path_list)
            self.output_vc_list(flat_vc2_list, new_name)

            print(f"Completed backup for OCS {ocs} failure")
            # input(f"Completed backup for OCS {ocs} failure")

def define_and_parse_args():

    parser = argparse.ArgumentParser(description='...')

    parser.add_argument('--topology',type=str,help='.map file to evaluate',required=True)
    parser.add_argument('--pathlist',type=str,help='path list',required=True)
    parser.add_argument('--allpathlist',type=str,help='allpath list',required=True)
    parser.add_argument('--allowed_turns_w_vcs','-atv',type=str,help='.allowvcturns file to evaluate', required=True)
    parser.add_argument('--xyzc_dims',nargs='+',type=int,help='type without parenthesis and use spaces, no commas', required=True)

    parser.add_argument('--verbose','-v',action='store_true',help='debug prints')


    args = parser.parse_args()

    topology = args.topology
    pathlist = args.pathlist
    allpathlist = args.allpathlist
    allowed_turns_w_vcs = args.allowed_turns_w_vcs

    xyzc_dims = tuple(args.xyzc_dims)
    assert(len(xyzc_dims) == 4)

    verbose = args.verbose
    # class var
    Robust.verbose = verbose

    return topology, pathlist, allpathlist, allowed_turns_w_vcs, xyzc_dims

def main():

    topology, pathlist, allpathlist, allowed_turns_w_vcs, xyzc_dims = define_and_parse_args()

    my_robust = Robust(topology, pathlist, allpathlist, allowed_turns_w_vcs, xyzc_dims)

    my_robust.handle_all_failures()    


if __name__ == "__main__":
    main()