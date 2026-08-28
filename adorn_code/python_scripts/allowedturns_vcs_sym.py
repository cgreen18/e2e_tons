# Copyright (c) 2024 Purdue University
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

# Authors: Conor Green

"""


"""

# std
import argparse
import copy
import ast
import os
import multiprocessing
import time
from collections import deque
from copy import deepcopy
import random

# pipd
import networkx as nx
from graph_tool.all import Graph, shortest_distance

# locals
from omnicdg import OmniCDG
from tpuv4_symmetry import TPUv4_Symmetry


RAND_SEED = 1
random.seed(RAND_SEED)

class SymmetricCDG:
    """Symmetry-reduced CDG.

    We maintain a CDG over **canonical channel orbits** rather than over every
    (u->v, VC) channel instance in the full topology.

    A CDG cycle in the full graph maps to a CDG cycle in the quotient (orbit)
    graph. Therefore, keeping this reduced CDG acyclic is a sufficient
    condition for deadlock-freedom in the full CDG.

    Implementation notes:
      * We use tuple-valued nodes directly: (u_canon, v_canon, vc).
      * "Canonical" means we map the channel source u to its canonical
        representative, and apply the same symmetry transform to the channel
        destination v.
      * VCs are treated as labels and are *not* transformed.
    """

    def __init__(self, symmetry: TPUv4_Symmetry):
        self.symmetry = symmetry
        self.cdg_as_nwx_G = nx.DiGraph()

    # --- canonicalization helpers -------------------------------------------------
    def canonicalize_turn(self, turn):
        (c_a, c_b) = turn
        (i, k0, vc0) = c_a
        (k1, j, vc1) = c_b
        assert k0 == k1

        i_canon, t = self.symmetry.get_canonical_equivalent(i)
        k_canon = self.symmetry.apply_transformation(k0, t)
        j_canon = self.symmetry.apply_transformation(j,  t)

        return ((i_canon, k_canon, vc0), (k_canon, j_canon, vc1))


    # --- API-compatible with OmniCDG ---------------------------------------------
    def probe_turn_for_deadlock(self, turn):
        """Return True iff adding this turn would create a cycle."""
        c_a, c_b = self.canonicalize_turn(turn)

        # exit early if edge already present
        if self.cdg_as_nwx_G.has_edge(c_a, c_b):
            return False

        # ensure nodes exist for has_path
        if c_a not in self.cdg_as_nwx_G:
            self.cdg_as_nwx_G.add_node(c_a)
        if c_b not in self.cdg_as_nwx_G:
            self.cdg_as_nwx_G.add_node(c_b)

        return nx.has_path(self.cdg_as_nwx_G, c_b, c_a)

    def add_turn(self, turn):
        c_a, c_b = self.canonicalize_turn(turn)
        if self.cdg_as_nwx_G.has_edge(c_a, c_b):
            return
        self.cdg_as_nwx_G.add_edge(c_a, c_b)


class DisjointSet:
    """Union-Find data structure for tracking connected components with fixed roots."""
    def __init__(self, n, root):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.root = root  # Fixed root for this disjoint set

    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]

    def union(self, u, v):
        root_u = self.find(u)
        root_v = self.find(v)
        
        if root_u == root_v:
            return False  # Already connected
        
        # Ensure the tree remains rooted at the fixed root
        if root_v == self.root:
            self.parent[root_u] = root_v
        else:
            self.parent[root_v] = root_u
        return True


class AllPaths_VNAlloc:

    # class vars
    ############

    verbose = False
    slow  = False

    INF = 999999

    all_paths_output_path_prefix = '/scratch/negishi/green456/topologies_and_routing/allpath_lists'
    allowed_turns_vcs_path_prefix = './topologies_and_routing/allowed_turns_vcs'

    def __init__(self):

        # object vars
        #############

        # # topology vars
        # -------------
        self.rmap_file_path = None
        self.rmap_name = None
        self.binary_r_map = None
        self.r_map = None
        self.n_routers = -1
        self.n_links = -1
        self.r_map_as_nwx_G = None
        self.r_adj_list = None

        # path list vars
        # --------------
        self.apl_file_path = None
        self.apl_name = None
        self.apl = None
        self.flat_apl = None

        # chosen path vars
        # ----------------
        self.cpl_file_path = None
        self.cpl_name = None
        self.flat_cpl = None
        self.cpl = None

        # VC vars
        # -------
        self.max_n_vcs = -1
        self.allowed_turns_dict = None

        # algorithm
        # ---------
        self.safe = None
        self.no_transitions = None
        self.symmetric = False
        self.my_tpuv4_symmetry = None
        self.mc_dims = None
        self.sym_type = None

        # prints
        # ------
        # base turns
        self.n_allowed_base_turns, self.n_removed_base_turns = (0,0)
        # turns w vcs
        self.n_dateline_crosses, self.n_higher_vcs, self.n_turn_w_vc_allowed, self.n_turn_w_vc_unallowed = (0,0,0,0)

    def setup(self, apl_file_path, safe=False, no_transitions=False):

        print('\n' + '-'*100)

        print('Setup')

        # path list vars
        # --------------
        self.apl_file_path = apl_file_path
        self.apl_name = apl_file_path.split('/')[-1].split('.')[0]
        # sets self.flat_apl
        self.ingest_flat_apl()

        # topology vars
        # -------------
        # sets self.n_links, self.n_routers
        self.set_nlinks_nrouters_from_flat_apl()

        # sets self.apl
        self.create_apl_from_flat_apl()

        # algorithm vars
        # --------------
        self.safe = safe
        self.no_transitions = no_transitions

        print(f'Completed setup apl {self.apl_name}')
        print(f'\t# routers : {self.n_routers}')
        print(f'\t# links   : {self.n_links}')

        print('\n' + '-'*100)

    def setup_w_r_map(self, rmap_file_path, binary_r_map=False, safe=False, no_transitions=False):

        print('\n' + '-'*100)
        print('Setup' )

        # topology vars
        # -------------
        self.rmap_file_path = rmap_file_path
        self.rmap_name = rmap_file_path.split('/')[-1].split('.')[0]
        self.binary_r_map = binary_r_map
        # sets self.r_map, self.n_routers
        self.ingest_rmap()
        # (potentially) modifies self.r_map
        self.sanitize_rmap()
        # sets self.n_links, self.links
        self.set_n_links_from_rmap()
        # sets self.r_map_as_nwx_G
        self.create_nwx_G_from_r_map()
        # sets self.r_adj_list
        self.create_adj_list_from_r_map()

        # algorithm vars
        # --------------
        self.safe = safe
        self.no_transitions = no_transitions

        print(f'Completed setup of rmap {self.rmap_name} and apl {self.apl_name}')
        print(f'\t# routers : {self.n_routers}')
        print(f'\t# links   : {self.n_links}')

        print('\n' + '-'*100)


    def setup_w_r_map_and_apl(self, rmap_file_path, apl_file_path, binary_r_map=False, safe=False, no_transitions=False):

        print('\n' + '-'*100)
        print('Setup' )

        # topology vars
        # -------------
        self.rmap_file_path = rmap_file_path
        self.rmap_name = rmap_file_path.split('/')[-1].split('.')[0]
        self.binary_r_map = binary_r_map
        # sets self.r_map, self.n_routers
        self.ingest_rmap()
        # (potentially) modifies self.r_map
        self.sanitize_rmap()
        # sets self.n_links
        self.set_n_links_from_rmap()
        # sets self.r_map_as_nwx_G
        self.create_nwx_G_from_r_map()
        # sets self.r_adj_list
        self.create_adj_list_from_r_map()

        # path list vars
        # --------------
        self.apl_file_path = apl_file_path
        self.apl_name = apl_file_path.split('/')[-1].split('.')[0]
        # sets self.apl, self.flat_apl
        self.ingest_apls()

        # algorithm vars
        # --------------
        self.safe = safe
        self.no_transitions = no_transitions


        print(f'Completed setup of rmap {self.rmap_name} and apl {self.apl_name}')
        print(f'\t# routers : {self.n_routers}')
        print(f'\t# links   : {self.n_links}')

        print('\n' + '-'*100)

    def setup_w_chosen_paths(self, rmap_file_path,cpl_file_path, binary_r_map=False):

        print('\n' + '-'*100)
        print('Setup' )

        # topology vars
        # -------------
        self.rmap_file_path = rmap_file_path
        self.rmap_name = rmap_file_path.split('/')[-1].split('.')[0]
        self.binary_r_map = binary_r_map
        # sets self.r_map, self.n_routers
        self.ingest_rmap()
        # (potentially) modifies self.r_map
        self.sanitize_rmap()
        # sets self.n_links
        self.set_n_links_from_rmap()
        # sets self.r_map_as_nwx_G
        self.create_nwx_G_from_r_map()
        # sets self.r_adj_list
        self.create_adj_list_from_r_map()

        # path list vars
        # --------------

        self.cpl_file_path = cpl_file_path
        self.cpl_name = cpl_file_path.split('/')[-1].split('.')[0]
        # sets self.cpl, self.flat_cpl
        self.ingest_flat_pl()

        print(f'Completed setup of rmap {self.rmap_name} and apl {self.apl_name} and cpl {self.cpl_name}')
        print(f'\t# routers : {self.n_routers}')
        print(f'\t# links   : {self.n_links}')

        print('\n' + '-'*100)

    # init funcs
    ####################################################################################################

    def ingest_rmap(self):
        assert(self.rmap_file_path is not None)
        self.r_map, self.n_routers = self.ingest_a_map_(self.rmap_file_path)

    # _ implies it returns instead of setting self vars
    # these will be class methods
    @classmethod
    def ingest_a_map_(cls, path_name):

        if True:
            print(f'Ingesting r map ({path_name})')

        this_map = []

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

        n_routers = len(this_map)

        return this_map, n_routers

    def sanitize_rmap(self):
        assert(self.r_map is not None)
        assert(self.binary_r_map is not None)
        self.r_map = self.sanitize_a_map_(self.r_map, binary_r_map=self.binary_r_map)

    @classmethod
    def sanitize_a_map_(cls, this_map, binary_r_map=False):
        # quick sanitization
        n_routers = len(this_map)
        for i in range(n_routers):
            this_map[i][i] = 0

        if binary_r_map:
            # assert binary
            for i in range(n_routers):
                for j in range(n_routers):
                    if i == j:
                        continue
                    if this_map[i][j] >= 1:
                        this_map[i][j] = 1

        return this_map

    def set_n_links_from_rmap(self):
        assert(self.r_map is not None)
        assert(self.n_routers != -1)
        n_routers = self.n_routers
        r_map = self.r_map

        links = []

        n_links = 0
        for i in range(n_routers):
            for j in range(n_routers):
                n_links += r_map[i][j]

                if r_map[i][j] > 0:
                    links.append((i,j))

        self.n_links = n_links
        self.links = links

    def set_nlinks_nrouters_from_flat_apl(self):
        assert(self.flat_apl is not None)

        n_routers = 0
        links = []
        for path in self.flat_apl:

            plen = len(path) - 1
            for i in range(plen):
                link = (path[i],path[i+1])
                if link not in links:
                    links.append(link)

            # set n_routers
            for n in path:
                n_routers = max(n, n_routers)

        self.n_links = len(links)
        # account for zero indexing
        self.n_routers = n_routers + 1

    def ingest_apls(self):
        assert(self.apl_file_path is not None)
        self.apl, self.flat_apl = self.ingest_an_apl_(self.apl_file_path)

    def ingest_flat_apl(self):
        assert(self.apl_file_path is not None)
        self.flat_apl = self.ingest_a_flat_apl_(self.apl_file_path)

    def create_apl_from_flat_apl(self):
        assert(self.flat_apl is not None)
        assert(self.n_routers != -1)
        self.apl = self.create_an_apl_from_a_flat_apl_(self.flat_apl, self.n_routers)

    @classmethod
    def ingest_a_flat_apl_(cls, path_name):

        if True:
            print(f'Ingesting path list {path_name}')

        flat_path_list = []
        n_routers = -1

        with open(path_name, 'r') as inf:

            line_num = 0
            for line in inf.readlines():

                line_no_newline = line.strip('\n')
                line_w_commas =  ','.join(line_no_newline.split(' '))
                line_w_square_brackets = f'[ {line_w_commas} ]'

                as_list = ast.literal_eval(line_w_square_brackets)
                clean_as_list = [e for e in as_list]
                flat_path_list.append(clean_as_list)

                # find n_routers
                for n in clean_as_list:
                    n_routers = max(n, n_routers)

                line_num += 1

        return flat_path_list

    @classmethod
    def create_an_apl_from_a_flat_apl_(cls, flat_path_list, n_routers):

        allpath_list = [[ [] for _ in range(n_routers) ] for __ in range(n_routers)]

        for path in flat_path_list:
            s = path[0]
            d = path[-1]
            allpath_list[s][d].append(path)

        return allpath_list

    @classmethod
    def ingest_an_apl_(cls, path_name):

        if True:
            print(f'Ingesting path list {path_name}')

        flat_path_list = []
        n_routers = -1

        with open(path_name, 'r') as inf:

            line_num = 0
            for line in inf.readlines():

                line_no_newline = line.strip('\n')
                line_w_commas =  ','.join(line_no_newline.split(' '))
                line_w_square_brackets = f'[ {line_w_commas} ]'

                as_list = ast.literal_eval(line_w_square_brackets)
                clean_as_list = [e for e in as_list]
                flat_path_list.append(clean_as_list)

                # find n_routers
                for n in clean_as_list:
                    n_routers = max(n, n_routers)

                line_num += 1

        # adjust for zero indexing
        n_routers += 1

        allpath_list = [[ [] for _ in range(n_routers) ] for __ in range(n_routers)]

        for path in flat_path_list:
            s = path[0]
            d = path[-1]
            allpath_list[s][d].append(path)

        return allpath_list, flat_path_list

    def ingest_flat_pl(self):
        assert(self.cpl_file_path is not None)
        self.cpl, self.flat_cpl = self.ingest_a_pl_(self.cpl_file_path)

    @classmethod
    def ingest_a_pl_(self, path_name):

        flat_path_list = []
        n_routers = -1

        with open(path_name, 'r') as inf:
            name = inf.readline()

            for line in inf.readlines():

                # print(f'line (type {type(line)}) = {line}')

                as_list = ast.literal_eval(line)
                clean_as_list = [e for e in as_list]

                flat_path_list.append(clean_as_list)

                # find n_routers
                for n in clean_as_list:
                    n_routers = max(n, n_routers)

        n_routers += 1

        twod_path_list = [[ None for _ in range(n_routers) ] for __ in range(n_routers)]

        for path in flat_path_list:
            s = path[0]
            d = path[-1]
            if s == d:
                twod_path_list[s][d] = [s]
                continue
            twod_path_list[s][d] = path

        return twod_path_list, flat_path_list

    def create_nwx_G_from_r_map(self):
        assert(self.r_map)

        if self.r_map_as_nwx_G is not None:
            return

        self.r_map_as_nwx_G = self.create_an_nwx_G_from_a_map_(self.r_map)

    def create_adj_list_from_r_map(self):
        assert(self.r_map)
        assert(self.n_routers)
        r_map = self.r_map
        n_routers = self.n_routers


        adj_list = []
        for i in range(n_routers):
            adj_list.append([])
            for j in range(n_routers):
                if r_map[i][j] > 0:
                    adj_list[i].append(j)

        self.r_adj_list = adj_list

    @classmethod
    def create_an_nwx_G_from_a_map_(cls, this_map):#, directed=True):

        n_routers = len(this_map)
        G = nx.DiGraph()

        for src in range(n_routers):
            for dest in range(n_routers):

                if(src == dest):
                    continue

                # if not directed and src > dest:
                #     continue

                if(this_map[src][dest] < 1):
                    continue

                G.add_edge(src,dest)

        return G

    def ingest_allowed_turns(self, atv_file_path):
        print(f'Ingesting allowed turns')
        start_time = time.time()
        self.allowed_turns_dict, self.allowed_turns_list = self.ingest_an_allowed_turns_(atv_file_path)

        # input(f'ingestion of allpaths took {time.time() - start_time}')

    @classmethod
    def ingest_an_allowed_turns_(cls, path_name):
        atvcs_dict = {}
        atvcs_list = []

        with open(path_name, 'r') as inf:
            for line in inf.readlines():
                line_no_newline = line.strip('\n')
                line_w_curly = f'{{ {line_no_newline} }}'

                as_dict = ast.literal_eval(line_w_curly)

                atvcs_dict.update(as_dict)

                if list(as_dict.values())[0]:
                    atvcs_list.append( list(as_dict.keys())[0] )
        return atvcs_dict, atvcs_list

    # def create_allowed_turns_graph(self):

    #     print(f'Creating allowed turns graph')

    #     assert(self.r_map_as_nwx_G)
    #     r_map_as_nwx_G = self.r_map_as_nwx_G

    #     # self.create_allowed_turns_no_vcs()

    #     assert(self.allowed_turns_list)
    #     allowed_turns_list = self.allowed_turns_list

    #     at_graph = nx.DiGraph()

    #     at_graph.add_nodes_from(r_map_as_nwx_G.edges())
    #     FLAG
    #     at_graph.add_edges_from(((u, v), (v, w)) for (u, v, w) in allowed_turns_list if r_map_as_nwx_G.has_edge(u, v) and r_map_as_nwx_G.has_edge(v, w))

    #     self.at_graph = at_graph

    # def create_allowed_turns_no_vcs(self):

    #     assert(self.allowed_turns_dict)
    #     allowed_turns_dict = self.allowed_turns_dict

    #     # allowed_turns_dict_no_vcs = {}
    #     allowed_turns_no_vcs_list = []
    #     for turn, allowed in allowed_turns_dict.items():
    #         ((a,b,v0),(b,c,v1)) = turn
    #         turn_no_vcs = ((a,b,c))

    #         if allowed and turn_no_vcs not in allowed_turns_no_vcs_list:
    #             allowed_turns_no_vcs_list.append(turn_no_vcs)
                
    #     self.allowed_turns_no_vcs_list = allowed_turns_no_vcs_list

    # outputs
    ####################################################################################################

    @classmethod
    def output_allpathslist_raw_(cls, allpath_list, base_file_name):
        full_name = f'{base_file_name}.rallpaths'

        full_out_path = os.path.join(cls.all_paths_output_path_prefix, \
                full_name)

        with open(full_out_path, 'w+') as of:

            for src, src_paths in enumerate(allpath_list):
                for dest, paths in enumerate(src_paths):
                    for path in paths:
                        line = ''
                        for n in path[:-1]:
                            line += f'{n} '
                        line += f'{path[-1]}'
                        of.write(f'{line}\n')

        print(f'Wrote to {full_out_path}')

    @classmethod
    def output_flat_pathlist_(cls, flat_list, name):
        with open(name, 'w+') as of:


            for path in flat_list:
                line = ''
                for n in path[:-1]:
                    line += f'{n} '
                line += f'{path[-1]}'
                of.write(f'{line}\n')

        print(f'Wrote to {name}')

    @classmethod
    def output_vc_partitioned_allpathslist_raw_(cls, vcpart_allpath_list, base_file_name):
        full_name = f'{base_file_name}.rallpaths'

        full_out_path = os.path.join(cls.all_paths_output_path_prefix, \
                full_name)

        with open(full_out_path, 'w+') as of:

            for vc, allpath_list in enumerate(vcpart_allpath_list):
                for src, src_paths in enumerate(allpath_list):
                    for dest, paths in enumerate(src_paths):
                        for path in paths:
                            line = ''
                            for n in path[:-1]:
                                line += f'{n} '
                            line += f'{path[-1]}'
                            of.write(f'{line}\n')

        print(f'Wrote to {full_out_path}')

    @classmethod
    def output_vc_partitioned_flat_allpathslist_raw_(cls, vcpart_flat_allpath_list, base_file_name):

        full_name = f'{base_file_name}.rallpaths'

        full_out_path = os.path.join(cls.all_paths_output_path_prefix, \
                full_name)

        with open(full_out_path, 'w+') as of:

            for allpath_list in vcpart_flat_allpath_list:

                    for path in allpath_list:
                        line = ''
                        for n in path[:-1]:
                            line += f'{n} '
                        line += f'{path[-1]}'
                        of.write(f'{line}\n')

        print(f'Wrote to {full_out_path}')

    @classmethod
    def output_allowed_turns_vcs_(cls, sorted_allowed_turns_dict, base_file_name):
        full_name = f'{base_file_name}.allowvcturns'
        full_out_path = os.path.join(cls.allowed_turns_vcs_path_prefix, \
                full_name)

        with open(full_out_path, 'w+') as of:
            # expect sorted_allowed_turns_dict to be complete (all paths) and sorted (numerically increasing)
            for turn, allowed in sorted_allowed_turns_dict.items():
                    of.write(f'{turn} : {allowed}\n')
        print(f'Wrote to {full_out_path}')

    # getters and setters
    ####################################################################################################

    def set_xyzc_dims(self, dims_tuple):
        self.xyzc_dims = dims_tuple
        self.x_dim, self.y_dim, self.z_dim, self.cube_dim = dims_tuple
        dim_str_dict = {'x':self.x_dim, 'y':self.y_dim, 'z':self.z_dim}

        min_ordered_dim_list = []
        min_ordered_dim_list.append( min([k for k,v in dim_str_dict.items()]) )
        min_ordered_dim_list.append( min([k for k,v in dim_str_dict.items() if k not in min_ordered_dim_list]) )
        min_ordered_dim_list.append( min([k for k,v in dim_str_dict.items() if k not in min_ordered_dim_list]) )

        max_ordered_dim_list = min_ordered_dim_list.copy()
        max_ordered_dim_list.reverse()

        self.dim_str_dict = dim_str_dict
        self.min_ordered_dim_list = min_ordered_dim_list
        self.max_ordered_dim_list = max_ordered_dim_list

    # TPUv4 geometry
    # ################################################################################

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

    def r_to_rel_xyz(self, r):
        assert(self.xyzc_dims)
        (x_dim, y_dim, z_dim, cube_dim) = self.xyzc_dims

        r_x,r_y,r_z = self.r_to_xyz(r)

        rel_r_x = r_x % cube_dim
        rel_r_y = r_y % cube_dim
        rel_r_z = r_z % cube_dim

        return rel_r_x, rel_r_y, rel_r_z

    def rel_xyz_is_on_face(self, rel_x, rel_y, rel_z):
        assert(self.xyzc_dims)
        (x_dim, y_dim, z_dim, cube_dim) = self.xyzc_dims

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

    def ocs_id(self, i, j):
        assert(self.xyzc_dims)
        (x_dim, y_dim, z_dim, cube_dim) = self.xyzc_dims

        ij_conn_type = self.calc_opt_conn_type(i,j)

        if ij_conn_type is None:
            return -1

        # x in [0,16)
        # y in [16, 32)
        # z in [32, 48)

        # + => i is representative
        # - => j is representative

        representative = i
        if '-' in ij_conn_type:
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

    # escape paths
    ####################################################################################################

    def create_convex_subgraph(self, relevant_nodes):
        """
            Calculate all shortest paths between relevant_nodes -> relevant_nodes
            All nodes included in these paths are nodes in the convex_subgraph
            All edges included in these paths are edges in the convex_subgraph

            Return convext_subgraph_adj_matrix
                and for convenience, list of convex_subgraph nodes
        """

        assert(self.n_routers != -1)
        n_routers = self.n_routers

        # n_relevant_nodes = len(relevant_nodes)

        convex_subgraph_nodes = []
        convex_subgraph_adj_mat = [ [ 0 for _ in range(n_routers) ] for __ in range(n_routers) ]

        nodeset_shortest_paths = self.calc_all_shortest_paths(relevant_nodes)

        for src in relevant_nodes:
            for dest in relevant_nodes:
                if src == dest:
                    continue

                src_dest_paths = nodeset_shortest_paths[src][dest]
                for path in src_dest_paths:
                    for node in path:
                        if node not in convex_subgraph_nodes:
                            convex_subgraph_nodes.append(node)

                    pathlen = len(path) - 1
                    for i in range(pathlen):
                        path_src = path[i]
                        path_dest = path[i+1]
                        convex_subgraph_adj_mat[path_src][path_dest] = 1

        return convex_subgraph_adj_mat, convex_subgraph_nodes

    def calc_most_central_node_old(self, this_map, relevant_nodes):

        nodeset_shortest_paths = self.calc_all_shortest_paths(relevant_nodes)

        max_centrality = -1
        most_central_node = -1

        for n in relevant_nodes:

            centrality_for_n = 0

            for s in relevant_nodes:
                for d in relevant_nodes:
                    if s == d or n==s or n==d:
                        continue
                    sd_paths = nodeset_shortest_paths[s][d]
                    base_n_paths = len(sd_paths)

                    n_paths_with_n = 0
                    for path in sd_paths:
                        if n in path:
                            n_paths_with_n += 1

                    inner_ratio = n_paths_with_n / base_n_paths
                    centrality_for_n += inner_ratio

            if centrality_for_n > max_centrality:
                max_centrality = centrality_for_n
                most_central_node = n

        return most_central_node

    def calc_most_central_node(self, exact_threshold=1024, seed=0):
        G = self.r_map_as_nwx_G
        n = G.number_of_nodes()
        if n <= exact_threshold:
            bc = nx.betweenness_centrality(G, normalized=False)  # exact
        else:
            # approximate: sample k_approx sources
            k = min(exact_threshold, n)
            bc = nx.betweenness_centrality(G, k=k, normalized=False, seed=seed)

        most_central = max(bc, key=bc.get)

        if self.symmetric:
            # TODO is this the best way? Or better to augment G?
            canonical_most_central, _tform = self.my_tpuv4_symmetry.get_canonical_equivalent(most_central)
        else:
            canonical_most_central = most_central

        return canonical_most_central

    def create_spanning_tree_given_root(self, root_node, relevant_nodes=None):
        assert(self.r_map is not None)
        assert(self.n_routers != -1)
        r_map = self.r_map
        n_routers = self.n_routers

        if relevant_nodes is None:
            relevant_nodes = list(range(n_routers))

        visited = []

        queue = deque()

        # input(f'TODO: spanning tree expects bidirectionality')

        # print(f'root_node={root_node}')

        # init root
        queue.append(root_node)
        visited.append(root_node)

        # print(f'Q={queue}')

        span_tree_adj_list = {}
        # spanning_paths_from_root = []

        q_iter = 0
        while queue:

            cur_node = queue.popleft()
            # visited.append(cur_node)

            # print(f'q_iter {q_iter} : cur_node={cur_node} and Q={queue}')
            q_iter += 1

            for next_node in relevant_nodes:
                if next_node in visited:
                    continue
                if r_map[cur_node][next_node] > 0:
                    queue.append(next_node)
                    visited.append(next_node)

                    try:
                        span_tree_adj_list[cur_node].append(next_node)
                    except:
                        span_tree_adj_list.update({cur_node : [next_node]})

        if self.verbose:
            print(f'relevant_nodes={relevant_nodes}')
            input(f'=> spanning tree {span_tree_adj_list}')

        return span_tree_adj_list

    def nx_bfs_tree_adj(self, root, relevant_nodes=None):
        assert(self.r_map is not None)
        assert(self.n_routers != -1)
        n_routers = self.n_routers

        G = self.r_map_as_nwx_G

        T = nx.bfs_tree(G, source=root)  # directed arborescence (rooted)
        # Convert to undirected adjacency list for the tree edges
        U = T.to_undirected()

        if relevant_nodes is None:
            relevant_nodes = list(range(n_routers))


        # Induce/prune to root->target paths in the BFS tree
        keep = {root}
        for t in set(relevant_nodes):
            if t not in T:
                continue
            path = nx.shortest_path(T, source=root, target=t)  # unique in a tree
            keep.update(path)

        U_sub = U.subgraph(keep)
        return {u: list(U_sub.neighbors(u)) for u in U_sub.nodes()}



    @classmethod
    def _tree_adj_to_nx_graph(cls, tree_adj, validate_tree=True):

        G = nx.Graph()
        added = set()

        for u, nbrs in tree_adj.items():
            G.add_node(u)
            for v in nbrs:
                if u == v:
                    continue
                # order-independent edge key (works for any hashable node labels)
                eid = frozenset((u, v))
                if eid in added:
                    continue
                G.add_edge(u, v)
                added.add(eid)

        if validate_tree:
            # For a spanning tree, this should be True; disable for max speed.
            if not nx.is_tree(G):
                raise ValueError("Input adjacency list does not describe a single connected tree.")

        return G




    def all_pairs_tree_paths_by_pair(self, tree_as_nwx_G, include_self=False):
        """
        Stream all-pairs shortest paths on a tree, yielding per (src, dst) pair.

        Yields:
            (src, dst, path_list)
        """

        def all_pairs_tree_paths_by_source(tree_as_nwx_G):

            # Generator: yields (src, {dst: path_list})
            yield from nx.all_pairs_shortest_path(tree_as_nwx_G)

        for src, paths in all_pairs_tree_paths_by_source(tree_as_nwx_G):
            for dest, path in paths.items():
                if not include_self and src == dest:
                    continue
                yield (src, dest), path

    def calc_escape_paths_from_nodeset_and_root(self, relevant_nodes, root_node):
        assert(self.n_routers != -1)
        n_routers = self.n_routers

        # span_tree_adj_lis t = self.create_spanning_tree_given_root( root_node, relevant_nodes=relevant_nodes)

        if self.symmetric:
            canonical_nodes = self.my_tpuv4_symmetry.get_canonical_nodes()
        else:
            canonical_nodes = list(range(n_routers))

        assert(root_node in canonical_nodes)

        # big fix: must create tree to all destinations not just relevant
        span_tree_adj_list = self.nx_bfs_tree_adj(root_node) #, relevant_nodes=relevant_nodes)

        print(f"Created spanning tree")

        # print(f'span_tree_adj_list={span_tree_adj_list}')

        # create span_tree_nwx_G from span_tree_adj_list
        span_tree_nwx_G = self._tree_adj_to_nx_graph(span_tree_adj_list)

        # DO NOT USE SETS. FOR SOME REASON IT BREAKS EVERYTHING >:(
        # escape_channels = set() #[]
        # escape_turns = set() #[]
        escape_channels = []
        escape_turns = []
        escape_paths = {}

        n_iters = 0
        expected_iters = len(relevant_nodes)*n_routers - n_routers
        for (src, dest), path in self.all_pairs_tree_paths_by_pair(span_tree_nwx_G):

            if src not in canonical_nodes:
                continue

            if n_iters % 1_000_000 == 0:
                print(f"Finding escape paths for iter {n_iters} / {expected_iters}")

            n_iters += 1

            # if dest not in relevant_nodes:
            #     continue

            if self.verbose:
                print(f'\t{src}->{dest} : {path}')

            plen = len(path) - 1
            if plen == 1:
                continue

            for n in range(plen):
                this_channel = (path[n],path[n+1])
                if this_channel not in escape_channels:
                    # escape_channels.add(this_channel)
                    escape_channels.append(this_channel)

            for n in range(plen-1):
                this_turn = ( (path[n],path[n+1]), (path[n+1],path[n+2]) )
                if this_turn not in escape_turns:
                    # escape_turns.add( this_turn )
                    escape_turns.append( this_turn )

            # print(f'\t{src}->{dest} : {path}')
            # print(f'escape_channels={escape_channels}')
            # input(f'escape_turns={escape_turns}')


        # return list(escape_channels), list(escape_turns)
        return escape_channels, escape_turns


    def calc_channels_turns_from_paths(self, path_list_twod):

        escape_channels = []
        escape_turns = []
        for src, dest_paths in enumerate(path_list_twod):
            if self.verbose:
                print(f'{src}->')
            for dest, path in enumerate(dest_paths):

                if self.verbose:
                    print(f'\t->{dest} : {path}')

                plen = len(path) - 1
                if plen == 1:
                    continue

                for n in range(plen):
                    this_channel = (path[n],path[n+1])
                    if this_channel not in escape_channels:
                        escape_channels.append(this_channel)

                for n in range(plen-1):
                    this_turn = ( (path[n],path[n+1]), (path[n+1],path[n+2]) )
                    if this_turn not in escape_turns:
                        escape_turns.append( this_turn )

        return escape_channels, escape_turns

    def create_two_edge_disjoint_spanning_trees(self, root1, root2):

        # chatgpt
        def find_edge_disjoint_spanning_trees(graph, root1, root2):
            """
            Finds two edge-disjoint spanning trees in an undirected graph,
            rooted at specified nodes.
            """
            nodes = list(graph.nodes())
            edges = sorted(graph.edges(), key=lambda e: (e[0], e[1]))  # Sort edges for deterministic behavior
            node_index = {node: i for i, node in enumerate(nodes)}
            n = len(nodes)
            
            if len(edges) < 2 * (n - 1):
                return None  # Not enough edges to form two spanning trees
            
            tree1, tree2 = [], []
            ds1, ds2 = DisjointSet(n, node_index[root1]), DisjointSet(n, node_index[root2])
            
            for u, v in edges:
                u_idx, v_idx = node_index[u], node_index[v]
                
                # Try adding to the tree that maintains its required root structure
                if len(tree1) <= len(tree2) and ds1.union(u_idx, v_idx):
                    tree1.append((u, v))
                elif len(tree2) < len(tree1) and ds2.union(u_idx, v_idx):
                    tree2.append((u, v))
                elif ds1.union(u_idx, v_idx):
                    tree1.append((u, v))
                elif ds2.union(u_idx, v_idx):
                    tree2.append((u, v))
                
                if len(tree1) == n - 1 and len(tree2) == n - 1:
                    return tree1, tree2
            
            return None  # No two disjoint spanning trees found

        assert(self.r_map is not None)
        assert(self.n_routers != -1)
        r_map = self.r_map
        n_routers = self.n_routers

        graph = self.r_map_as_nwx_G

        tree_a_edges, tree_b_edges = find_edge_disjoint_spanning_trees(graph, root1, root2)

        return tree_a_edges, tree_b_edges

    def calc_edge_disjoint_escape_paths_from_nodeset_and_roots(self, relevant_nodes, root_nodes):
        assert(self.n_routers != -1)
        n_routers = self.n_routers


        both_span_edges = self.create_two_edge_disjoint_spanning_trees(root_nodes[0],root_nodes[1])

        if self.verbose:
            print(f'1st span tree (root {root_nodes[0]}) : {both_span_edges[0]}')
            input(f'2nd span tree (root {root_nodes[1]}) : {both_span_edges[1]}')

        both_trees_paths_twod = []

        for i, span_tree_edges in enumerate(both_span_edges):

            # create span_tree_nwx_G from span_tree_adj_list
            span_tree_nwx_G = nx.DiGraph()
            for (src,dest) in span_tree_edges:
                    span_tree_nwx_G.add_edge(src,dest)
                    span_tree_nwx_G.add_edge(dest,src)

            both_trees_paths_twod.append( [[ None for _ in range(n_routers)] for __ in range(n_routers)] )
            # find shortest paths in span_tree_adj_mat
            for src in range(n_routers):
                for dest in range(n_routers):
                    if src == dest:
                        both_trees_paths_twod[i][src][src] = [src]
                        continue
                    
                    # TODO better way to do generator
                    short_path_generator = nx.all_shortest_paths(span_tree_nwx_G,src,dest)
                    for p in short_path_generator:
                        both_trees_paths_twod[i][src][dest] = p
                        # always one path in tree
                        break

        both_escape_channels = []
        both_escape_turns = []
        for tree_paths in both_trees_paths_twod:
            # input(f'tree_paths ({len(tree_paths)}) ={tree_paths}')
            escape_channels, escape_turns = self.calc_channels_turns_from_paths(tree_paths)
            both_escape_channels.append( escape_channels)
            both_escape_turns.append(escape_turns)

        if self.verbose:
            print(f'both_escape_channels={both_escape_channels}')
            print(f'both_escape_turns={both_escape_turns}')
            input(f'both_trees_paths_twod={both_trees_paths_twod}')

        return both_escape_channels, both_escape_turns, both_trees_paths_twod


    def create_two_ocs_disjoint_spanning_trees(self, root1, root2):
        """
        Creates two OCS-disjoint spanning trees concurrently using interleaved BFS.
        Non-OCS edges (id == -1) can be shared by both trees.
        """
        assert(self.r_map is not None)
        n_routers = self.n_routers

        graph = self.r_map_as_nwx_G

        tree_a_edges = []
        visited_a = {root1}
        queue_a = deque([root1])
        ocs_used_by_a = set()

        tree_b_edges = []
        visited_b = {root2}
        queue_b = deque([root2])
        ocs_used_by_b = set()

        # Continue as long as either tree is still growing
        while queue_a or queue_b:
            
            # --- Expand Tree A by one node ---
            if queue_a:
                u = queue_a.popleft()
                for v in sorted(graph.neighbors(u)):
                    if v not in visited_a:
                        oid = self.ocs_id(u, v)
                        # input(f"edge {(u,v)} @ ( {self.r_to_rel_xyz(u)} , {self.r_to_rel_xyz(v)} ) is type {self.calc_opt_conn_type(u,v)} => ID {oid}")
                        # Tree A can use if oid is -1 OR not claimed by Tree B
                        if oid == -1 or oid not in ocs_used_by_b:
                            visited_a.add(v)
                            queue_a.append(v)
                            tree_a_edges.append((u, v))
                            if oid != -1:
                                ocs_used_by_a.add(oid)
            
            # --- Expand Tree B by one node ---
            if queue_b:
                u = queue_b.popleft()
                for v in sorted(graph.neighbors(u)):
                    if v not in visited_b:
                        oid = self.ocs_id(u, v)
                        # input(f"edge {(u,v)} @ ( {self.r_to_rel_xyz(u)} , {self.r_to_rel_xyz(v)} ) is type {self.calc_opt_conn_type(u,v)} => ID {oid}")
                        # Tree B can use if oid is -1 OR not claimed by Tree A
                        if oid == -1 or oid not in ocs_used_by_a:
                            visited_b.add(v)
                            queue_b.append(v)
                            tree_b_edges.append((u, v))
                            if oid != -1:
                                ocs_used_by_b.add(oid)

        # Verify both trees fully spanned the graph
        if len(visited_a) == n_routers and len(visited_b) == n_routers:
            return tree_a_edges, tree_b_edges
        
        return None, None

    def calc_ocs_disjoint_escape_paths_from_nodeset_and_roots(self, relevant_nodes, root_nodes):
        assert(self.n_routers != -1)
        n_routers = self.n_routers

        if self.symmetric:
            canonical_nodes = self.my_tpuv4_symmetry.get_canonical_nodes()
        else:
            canonical_nodes = list(range(n_routers))


        both_span_edges = self.create_two_ocs_disjoint_spanning_trees(root_nodes[0],root_nodes[1])

        if self.verbose:
            print(f'1st span tree (root {root_nodes[0]}) : {both_span_edges[0]}')
            input(f'2nd span tree (root {root_nodes[1]}) : {both_span_edges[1]}')

        both_trees_paths_twod = []

        for i, span_tree_edges in enumerate(both_span_edges):

            # create span_tree_nwx_G from span_tree_adj_list
            span_tree_nwx_G = nx.DiGraph()
            for (src,dest) in span_tree_edges:
                    span_tree_nwx_G.add_edge(src,dest)
                    span_tree_nwx_G.add_edge(dest,src)

            both_trees_paths_twod.append( [[ None for _ in range(n_routers)] for __ in range(n_routers)] )
            # find shortest paths in span_tree_adj_mat
            for src in canonical_nodes:
                for dest in range(n_routers):
                    if src == dest:
                        both_trees_paths_twod[i][src][src] = [src]
                        continue
                    
                    # TODO better way to do generator
                    short_path_generator = nx.all_shortest_paths(span_tree_nwx_G,src,dest)
                    for p in short_path_generator:
                        both_trees_paths_twod[i][src][dest] = p
                        # always one path in tree
                        break

        both_escape_channels = []
        both_escape_turns = []
        for tree_paths in both_trees_paths_twod:
            # input(f'tree_paths ({len(tree_paths)}) ={tree_paths}')
            escape_channels, escape_turns = self.calc_channels_turns_from_paths(tree_paths)
            both_escape_channels.append( escape_channels)
            both_escape_turns.append(escape_turns)

        if self.verbose:
            print(f'both_escape_channels={both_escape_channels}')
            print(f'both_escape_turns={both_escape_turns}')
            input(f'both_trees_paths_twod={both_trees_paths_twod}')

        return both_escape_channels, both_escape_turns, both_trees_paths_twod

    # general functions
    ####################################################################################################

    def improved_calc_turn_weight_dict_apl(self, twod_path_list):

        twd = {}

        for src, dest_path_list in enumerate(twod_path_list):
            for dest, paths_list in enumerate(dest_path_list):

                turns_this_flow = []

                for path in paths_list:
                    plen = len(path) - 1

                    # paths of length 1 do not have turns
                    if plen <= 1:
                        continue

                    n_turns = plen - 1

                    for i in range(n_turns):
                        s1 = path[i]
                        d1 = path[i+1]  # = s2

                        s2 = path[i+1] # = d1
                        d2 = path[i+2]

                        this_turn = ((s1,d1),(s2,d2))

                        # # paths that use this turn
                        try:
                            twd[this_turn] += 1
                        except:
                            twd.update({this_turn : 1})

                        # # flows that use this turn
                        if this_turn not in turns_this_flow:
                            turns_this_flow.append(this_turn)

                is_sole_path = False
                if len(paths_list) == 1:
                    is_sole_path = True

                for this_turn in turns_this_flow:
                    # # flows that use this turn
                    twd[this_turn] += 10

                    # # flows that need this turn
                    if is_sole_path:
                        twd[this_turn] += 100

        return twd

    def improved_calc_turn_weight_dict_cpl(self, twod_path_list):

        twd = {}


        for src, flat_path_list in enumerate(twod_path_list):
            for i, path in enumerate(flat_path_list):

                if path is None:
                    continue

                plen = len(path) - 1

                # paths of length 1 do not have turns
                if plen <= 1:
                    continue

                n_turns = plen - 1
                for i in range(n_turns):
                    s1 = path[i]
                    d1 = path[i+1]  # = s2

                    s2 = path[i+1] # = d1
                    d2 = path[i+2]

                    this_turn = ((s1,d1),(s2,d2))

                    # # paths that use this turn
                    try:
                        twd[this_turn] += 1
                    except:
                        twd.update({this_turn : 1})

        return twd

    def get_allpaths_to_given_dests(self, destinations):
        assert(self.apl is not None)
        assert(self.n_routers != -1)
        apl_twod = self.apl
        n_routers = self.n_routers

        related_paths_flat = []
        related_paths_twod = [ [ [] for _ in range(n_routers) ] for __ in range(n_routers) ]

        for dest in destinations:
            for src in range(n_routers):
                related_paths_flat += apl_twod[src][dest]
                related_paths_twod[src][dest] = apl_twod[src][dest].copy()

        return related_paths_flat, related_paths_twod

    def get_chosenpaths_to_given_dests(self, destinations):
        assert(self.cpl is not None)
        assert(self.n_routers != -1)
        cpl_twod = self.cpl
        n_routers = self.n_routers

        related_paths_flat = []
        related_paths_twod = [ [ [] for _ in range(n_routers) ] for __ in range(n_routers) ]

        for dest in destinations:
            for src in range(n_routers):
                related_paths_flat += cpl_twod[src][dest]
                related_paths_twod[src][dest] = cpl_twod[src][dest].copy()

        return related_paths_flat, related_paths_twod

    def calc_all_shortest_paths(self, nodeset):
        assert(self.n_routers != -1)
        n_routers = self.n_routers

        if self.apl is not None:
            nodeset_shortest_paths = [[ None for _ in range(n_routers) ] for __ in range(n_routers)]

            for src in nodeset:
                for dest in nodeset:
                    nodeset_shortest_paths[src][dest] = all_shortest_paths_twod[src][dest]

            return nodeset_shortest_paths

        # else
        return self.nwx_all_shortest_paths_given_nodeset(nodeset)


    def nwx_all_shortest_paths_given_nodeset(self, nodeset):
        assert(self.r_map)
        assert(self.n_routers != -1)
        n_routers = self.n_routers
        r_map = self.r_map

        G = self.create_an_nwx_G_from_a_map_(r_map)

        nodeset_shortest_paths = [[ None for _ in range(n_routers) ] for __ in range(n_routers)]

        for s in nodeset:
            for d in nodeset:
                if s==d:
                    nodeset_shortest_paths[s][d] = [ [s] ]
                    continue

                paths = list(nx.all_shortest_paths(G, s, d))

                nodeset_shortest_paths[s][d] = paths

        return nodeset_shortest_paths

    def create_all_possible_turns(self):
        assert(self.r_map is not None)
        assert(self.n_routers != -1)

        r_map = self.r_map
        n_routers = self.n_routers

        turns = []
        for i in range(n_routers):
            for j in range(n_routers):
                if r_map[i][j] == 0:
                    continue
                if i==j:
                    continue
                for k in range(n_routers):
                    if r_map[j][k] == 0:
                        continue

                    if k==i or k==j:
                        continue
                    
                    turns.append( ( (i,j), (j,k) ) )
        return turns

    def create_turns_w_vcs_ordered(self, base_turn, force_vc=None ):
        assert(self.max_n_vcs != -1)
        # assert(self.no_transitions is not None)
        max_n_vcs = self.max_n_vcs
        # no_transitions = self.no_transitions

        channel_a_no_vc , channel_b_no_vc = base_turn

        # early exit
        if force_vc is not None:
            channel_a = channel_a_no_vc + (force_vc,)
            channel_b = channel_b_no_vc + (force_vc,)
            new_turn = (channel_a, channel_b)
            turns_w_vcs = [new_turn]
            return turns_w_vcs

        turns_w_vcs = []
        # first, try same vc for a and b channels
        for vc_both in range(max_n_vcs):
            channel_a = channel_a_no_vc + (vc_both,)
            channel_b = channel_b_no_vc + (vc_both,)
            new_turn = (channel_a, channel_b)
            turns_w_vcs.append(new_turn)

        # second, try all combos
        for vc_a in range(max_n_vcs):
            for vc_b in range(max_n_vcs):
                # handled above
                if vc_a == vc_b:
                    continue
                channel_a = channel_a_no_vc + (vc_a,)
                channel_b = channel_b_no_vc + (vc_b,)
                new_turn = (channel_a, channel_b)
                turns_w_vcs.append(new_turn)

        return turns_w_vcs

    def print_turns_allowed_dict(self, allowed_turns_dict):
        assert(self.n_routers != -1)
        assert(self.max_n_vcs != -1)
        assert(self.r_map is not None)
        n_routers = self.n_routers
        max_n_vcs = self.max_n_vcs
        r_map = self.r_map

        for i in range(n_routers):
            for j in range(n_routers):
                if i == j:
                    continue
                for k in range(n_routers):
                    if i ==k or j == k:
                        continue
                    if r_map[i][j] == 0 or r_map[j][k] == 0:
                        continue
                    for vc_1 in range(max_n_vcs):
                        for vc_2 in range(max_n_vcs):
                            turn = ((i,j,vc_1),(j,k,vc_2))
                            print(f'turn {turn} allowed? {allowed_turns_dict[turn]}')

    def calc_dists_from_allowed_turns_dict(self,allowed_turns_dict):
        assert(self.n_routers != -1)
        n_routers = self.n_routers

        graph = [[item if item==1 else self.INF for item in row] for row in self.r_map]
        for i in range(0,n_routers):
            graph[i][i]=0
        real_dist = list(map(lambda p: list(map(lambda q: q, p)), graph))

        turn_dist = [[self.INF for _ in range(n_routers)] for __ in range(n_routers)]

        # add in known direct turns
        for turn_w_vc,v in allowed_turns_dict.items():
            if not v:
                continue
            src = turn_w_vc[0][0]
            dest = turn_w_vc[1][1]
            real_dist[src][dest] = 2
            vc_1 = turn_w_vc[0][0]

    def calc_all_pairs_hops(self):
        G = self.r_map_as_nwx_G

        hop_dists = dict(nx.all_pairs_shortest_path_length(G))

        return hop_dists

    def calc_hop_dists_from_node(self, source):

        G = self.r_map_as_nwx_G

        hop_dists = nx.single_source_shortest_path_length(G, source)

        return hop_dists

    def find_antipode(self, node):
        
        n_routers = self.n_routers

        dists = self.calc_hop_dists_from_node(node)

        max_dist = -1
        antipode = -1
        for j in range(n_routers):
            dist = dists[j]
            if dist > max_dist:
                max_dist = dist
                antipode = j
        
        return antipode

    def create_base_file_name(self):
        # copy as locals for clarity
        rmap_file_path = self.rmap_file_path
        rmap_name = self.rmap_name
        r_map = self.r_map
        apl_name = self.apl_name
        apl = self.apl
        cpl_name = self.cpl_name
        cpl = self.cpl
        n_routers = self.n_routers
        n_links = self.n_links
        links = self.links
        max_n_vcs = self.max_n_vcs
        verbose = self.verbose
        safe = self.safe
        no_transitions = self.no_transitions
        robust = self.robust
        ocs_disjoint_faults = self.ocs_disjoint_faults
        symmetric = self.symmetric

        # TODO as CLA
        self.single_ocs_fault_tolerant = False
        single_ocs_fault_tolerant = self.single_ocs_fault_tolerant



        base_file_name = f'{rmap_name}_turns_allowed'
        if apl_name is not None:
            base_file_name = f'{base_file_name}_apl'
        if cpl_name:
            base_file_name = f'{base_file_name}_cpl'
        if safe:
            base_file_name = f'{base_file_name}_safe'
        if no_transitions > 0:
            base_file_name = f'{base_file_name}_notrans'
        
        if symmetric:
            sym_type = self.sym_type
            (mcx, mcy, mcz) = self.mc_dims
            base_file_name = f"{base_file_name}_{sym_type}sym_{mcx}x{mcy}x{mcz}mc"

        if robust:
            if not ocs_disjoint_faults:
                base_file_name = f"{base_file_name}_alledges_robust"
            else:
                base_file_name = f"{base_file_name}_ocsrobust"

        self.base_file_name = base_file_name
        return base_file_name

    def create_equivalent_turns(self, base_turn):
        my_tpuv4_symmetry = self.my_tpuv4_symmetry

        ((i,k0),(k1,j)) = base_turn
        i_canon, i_to_i_canon_tform = my_tpuv4_symmetry.get_canonical_equivalent(i)
        k_canon = my_tpuv4_symmetry.apply_transformation(k0,i_to_i_canon_tform)
        j_canon = my_tpuv4_symmetry.apply_transformation(j,i_to_i_canon_tform)
        canonical_turn = ( (i_canon, k_canon), (k_canon, j_canon))

        all_equivalent_turns = []
        all_i_primes = my_tpuv4_symmetry.get_all_noncanonical_equivalents(i_canon)
        for i_prime in all_i_primes:
            i_to_i_prime_tform = my_tpuv4_symmetry.calc_transform_delta(i_canon,i_prime)
            k_prime = my_tpuv4_symmetry.apply_transformation(k_canon,i_to_i_prime_tform)
            j_prime = my_tpuv4_symmetry.apply_transformation(j_canon,i_to_i_prime_tform)

            all_equivalent_turns.append( ((i_prime, k_prime), (k_prime, j_prime)) )

        return all_equivalent_turns

    def canonicalize_base_turn(self, base_turn):
        """Canonicalize a base (no-VC) turn using a *single* symmetry transform.

        This matches how we define equivalence classes/orbits of turns:
            ((i,k),(k,j))  -> map i to i_canon; apply same transform to k and j.

        Returns a base turn in the original node namespace.
        """
        assert(self.symmetric)
        my_tpuv4_symmetry = self.my_tpuv4_symmetry

        ((i, k0), (k1, j)) = base_turn
        assert(k0 == k1)

        i_canon, i_to_i_canon_tform = my_tpuv4_symmetry.get_canonical_equivalent(i)
        k_canon = my_tpuv4_symmetry.apply_transformation(k0, i_to_i_canon_tform)
        j_canon = my_tpuv4_symmetry.apply_transformation(j, i_to_i_canon_tform)

        return ((i_canon, k_canon), (k_canon, j_canon))

    def canonicalize_turn_w_vc(self, turn_w_vc):
        assert self.symmetric
        my_tpuv4_symmetry = self.my_tpuv4_symmetry

        (c_a, c_b) = turn_w_vc
        (i, k0, vc0) = c_a
        (k1, j, vc1) = c_b
        assert k0 == k1

        i_canon, t = my_tpuv4_symmetry.get_canonical_equivalent(i)
        k_canon = my_tpuv4_symmetry.apply_transformation(k0, t)
        j_canon = my_tpuv4_symmetry.apply_transformation(j,  t)

        return ((i_canon, k_canon, vc0), (k_canon, j_canon, vc1))


    def get_allowed_turn(self, turn_w_vc):
        """Return allowed/disallowed for a full-graph turn.

        In symmetric mode, allowed_turns_dict is keyed by canonical turn-orbit keys.
        In non-symmetric mode, it is keyed directly by the full turn.
        """
        if not hasattr(self, 'allowed_turns_dict') or self.allowed_turns_dict is None:
            raise RuntimeError('allowed_turns_dict is not initialized yet.')

        if not self.symmetric:
            return self.allowed_turns_dict[turn_w_vc]

        key = self.canonicalize_turn_w_vc(turn_w_vc)
        return self.allowed_turns_dict[key]

    # big worker
    ####################################################################################################

    # in: running_CDG, ordered_base_turns, allowed_turns_dict
    # out: (modified) running_CDG, allowed_turns_dict
    def iteratively_add_turns(self, running_CDG, ordered_base_turns, allowed_turns_dict, add_one_turn_vc_per_base_turn=False, no_transitions=False, force_vc=None):

        symmetric = self.symmetric
        my_tpuv4_symmetry = self.my_tpuv4_symmetry
        n_routers = self.n_routers

        verbose = self.verbose

        # iters
        loop_iter = 0
        n_turns_tot = len(ordered_base_turns)

        if symmetric:
            canonical_nodes = my_tpuv4_symmetry.get_canonical_nodes()
        else:
            canonical_nodes = list(range(n_routers))

        start_time = time.time()

        # symmetry: process each base-turn orbit once
        processed_base_turns = set()

        # look through priority turns and find at least one turn_w_vc
        for base_turn in ordered_base_turns:

            if symmetric:
                base_turn = self.canonicalize_base_turn(base_turn)
                if base_turn in processed_base_turns:
                    continue
                processed_base_turns.add(base_turn)


            if loop_iter % 100 == 0:
                print(f'completed turns {loop_iter} / {n_turns_tot}')
                print(f'\tBase turns : # used {self.n_allowed_base_turns}. # removed {self.n_removed_base_turns}.')
                print(f'\tTurns w/ VCs : # turns w/ vc allowed {self.n_turn_w_vc_allowed}. # turns w/ vc not allowed {self.n_turn_w_vc_unallowed}')
                print(f'\tSpecifics : # dateline crosses {self.n_dateline_crosses}. # wholly in higher vc(s) {self.n_higher_vcs}')
                # input(f"Cont?")
            loop_iter += 1

            turns_w_vcs = self.create_turns_w_vcs_ordered(base_turn, force_vc=force_vc)

            found_at_least_one = False
            for turn_w_vc in turns_w_vcs:

                (vc_a, vc_b) = (turn_w_vc[0][2], turn_w_vc[1][2])

                if verbose:
                    print(f'Considering turn w vc {turn_w_vc}')

                # skip previously processed (symmetry reduces key-space)
                turn_key = turn_w_vc
                if symmetric:
                    turn_key = self.canonicalize_turn_w_vc(turn_w_vc)

                if turn_key in allowed_turns_dict:
                    found_at_least_one = True
                    continue

                # skip early
                if no_transitions and (turn_w_vc[0][2] != turn_w_vc[1][2]):
                    allowed_turns_dict.update({turn_key: False})
                    self.n_turn_w_vc_unallowed += 1
                    continue

                # probe only the (possibly symmetry-canonicalized) turn once
                causes_dl = running_CDG.probe_turn_for_deadlock(turn_w_vc)

                # found a spot
                if not causes_dl:
                    allowed_turns_dict.update({turn_key: True})
                    running_CDG.add_turn(turn_w_vc)
                    self.n_turn_w_vc_allowed += 1

                    found_at_least_one = True

                    # input(f'turn_w_vc={turn_w_vc} added')

                    # just for printing/info
                    if vc_a != vc_b:
                        self.n_dateline_crosses += 1
                    elif vc_a > 0 and vc_b > 0:
                        self.n_higher_vcs += 1

                # causes deadlock
                else:
                    allowed_turns_dict.update({turn_key: False})
                    self.n_turn_w_vc_unallowed += 1

                # break out of checking vcs if only allowing one turn_w_vc per base_turn
                if add_one_turn_vc_per_base_turn and found_at_least_one:
                    break

            if found_at_least_one:
                self.n_allowed_base_turns += 1
            else:
                self.n_removed_base_turns += 1

            # avoid interactive halts in batch runs
            if verbose and (loop_iter % 2500 == 0):
                print("...progress checkpoint")

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f'Completed turn allocation in {elapsed_time}s')
        print(f'completed turns {loop_iter} / {n_turns_tot}')
        print(f'\tBase turns : # used {self.n_allowed_base_turns}. # removed {self.n_removed_base_turns}.')
        print(f'\tTurns w/ VCs : # turns w/ vc allowed {self.n_turn_w_vc_allowed}. # turns w/ vc not allowed {self.n_turn_w_vc_unallowed}')
        print(f'\tSpecifics : # dateline crosses {self.n_dateline_crosses}. # wholly in higher vc(s) {self.n_higher_vcs}')
        print('\n' + '-'*100)

        return running_CDG, allowed_turns_dict

    def apl_turn_addition_omnicdg(self, use_chosen_paths=False):
        assert(self.r_map is not None)
        assert(self.n_routers != -1)
        assert(self.n_links != -1)
        assert(self.rmap_file_path is not None)
        # assert(self.apl_name is not None)
        # assert(self.apl is not None)
        if use_chosen_paths:
            assert(self.cpl_name is not None)
            assert(self.cpl is not None)
        assert(self.max_n_vcs != -1)


        # copy as locals for clarity
        rmap_file_path = self.rmap_file_path
        rmap_name = self.rmap_name
        r_map = self.r_map
        apl_name = self.apl_name
        apl = self.apl
        cpl_name = self.cpl_name
        cpl = self.cpl
        n_routers = self.n_routers
        n_links = self.n_links
        links = self.links
        max_n_vcs = self.max_n_vcs
        verbose = self.verbose
        safe = self.safe
        no_transitions = self.no_transitions
        robust = self.robust
        ocs_disjoint_faults = self.ocs_disjoint_faults

        # TODO as CLA
        self.single_ocs_fault_tolerant = False
        single_ocs_fault_tolerant = self.single_ocs_fault_tolerant

        symmetric = self.symmetric
        my_tpuv4_symmetry = self.my_tpuv4_symmetry



        base_file_name = self.create_base_file_name()
        # keep for downstream output helpers
        self.base_file_name = base_file_name

        print(f'Beginning apl_turn_addition_omnicdg')

        # turns form all paths
        if use_chosen_paths:
            paths_turn_usage = self.improved_calc_turn_weight_dict_cpl(cpl)
        elif apl is not None:
            paths_turn_usage = self.improved_calc_turn_weight_dict_apl(apl)
            # input(f'paths_turn_usage={paths_turn_usage}')
        else:
            shuffled_links = deepcopy(links)
            random.shuffle(shuffled_links)
            paths_turn_usage = {((c_a,c_b1),(c_b2,c_c)) : 0 for (c_a,c_b1) in shuffled_links for (c_b2, c_c) in shuffled_links if c_b1 == c_b2}

        turns_by_priority = sorted(paths_turn_usage, key=paths_turn_usage.get, reverse=True)

        print(f'Determinend priority of turns')

        # if not symmetric then this is all routers
        if symmetric:
            N_i = my_tpuv4_symmetry.get_canonical_nodes()
        else:
            N_i = list(range(n_routers))
        
        if self.verbose:
            print(f"Relevant nodes {N_i}")

        escape_turns = []

        if single_ocs_fault_tolerant:

            print(f"UNIMPLEMENTED. Exiting...")
            quit()

        elif robust:

            root_node = self.calc_most_central_node()

            print(f"root_node = {root_node}")

            second_root_node = self.find_antipode(root_node)
            root_nodes = [root_node, second_root_node]

            if ocs_disjoint_faults:
                both_escape_channels, both_escape_turns, both_escape_paths = self.calc_ocs_disjoint_escape_paths_from_nodeset_and_roots(N_i, root_nodes)

            else:
                both_escape_channels, both_escape_turns, both_escape_paths = self.calc_edge_disjoint_escape_paths_from_nodeset_and_roots(N_i, root_nodes)

            all_escape_paths = [[ [] for _ in range(n_routers)] for __ in range(n_routers)]
            # output these to special .rallpaths file
            # escape paths (will be long, nonminimal)
            for escape_paths in both_escape_paths:
                for src in range(n_routers):
                    for dest in range(n_routers):

                        all_escape_paths[src][dest].append(escape_paths[src][dest])

                        if self.verbose:
                            input(f'added {escape_paths[src][dest]} to {all_escape_paths[src][dest]} ')

            base_escape_file_name = f"{base_file_name}_escapes"
            self.output_allpathslist_raw_(all_escape_paths, base_escape_file_name)

        elif safe:
            # H_i_adj_mat, H_i_nodes = self.create_convex_subgraph(N_i)
            # root_node = self.calc_most_central_node(H_i_adj_mat, N_i)
            root_node = self.calc_most_central_node()

            print(f"root_node = {root_node}")
            escape_channels, escape_turns = self.calc_escape_paths_from_nodeset_and_root(N_i, root_node)


        print(f'Determined escape turns')

        # Build a CDG. If symmetric, use the reduced CDG; otherwise, build the full OmniCDG.
        if symmetric:
            running_CDG = SymmetricCDG(my_tpuv4_symmetry)
        else:
            running_CDG = OmniCDG()
            # running_CDG.set_verbose()
            running_CDG.init_w_n_nodes(n_routers, n_vcs=max_n_vcs)

        print(f'Created (empty) running CDG')

        used_turns = []
        used_turns_w_vcs = []
        removed_turns = []

        allowed_turns = []
        allowed_turns_dict = {}

        reachable_dict = {}

        # completed setup
        # --------------------

        start_time = time.time()

        # 1)
        # maybe provide spannign tree safety

        if robust:
            # span tree 1. all on VC 0
            running_CDG, allowed_turns_dict = self.iteratively_add_turns(running_CDG, both_escape_turns[0], allowed_turns_dict, add_one_turn_vc_per_base_turn=True,no_transitions=no_transitions,force_vc=0)
            # span tree 2. all on VC 1
            running_CDG, allowed_turns_dict = self.iteratively_add_turns(running_CDG, both_escape_turns[1], allowed_turns_dict, add_one_turn_vc_per_base_turn=True,no_transitions=no_transitions,force_vc=1)

            print(f'completed 2x{len(both_escape_turns[0])} (escape) turns')

        elif safe:
            running_CDG, allowed_turns_dict = self.iteratively_add_turns(running_CDG, escape_turns, allowed_turns_dict, add_one_turn_vc_per_base_turn=True,no_transitions=no_transitions)

            print(f'completed {len(escape_turns)} (escape) turns')

            if verbose:
                input('cont?')

        

        # 2)
        # do priority
        print(f"one turn / VC")
        running_CDG, allowed_turns_dict = self.iteratively_add_turns(running_CDG, turns_by_priority, allowed_turns_dict, add_one_turn_vc_per_base_turn=True,no_transitions=no_transitions)

        print(f'completed {len(turns_by_priority)} (priority) turns')

        # 3)
        # come back for all vcs of all turns
        all_possible_turns_of_r_map = self.create_all_possible_turns()
        # reintroduce order?
        for t in all_possible_turns_of_r_map:
            if t not in paths_turn_usage:
                paths_turn_usage.update({t : 0})
                turns_by_priority.append(t)

        print(f"all turns / VC")
        running_CDG, allowed_turns_dict = self.iteratively_add_turns(running_CDG, turns_by_priority, allowed_turns_dict, add_one_turn_vc_per_base_turn=False,no_transitions=no_transitions)

        # # no order?
        # running_CDG, allowed_turns_dict = self.iteratively_add_turns(running_CDG, all_possible_turns_of_r_map, allowed_turns_dict, add_one_turn_vc_per_base_turn=False)

        print(f'completed {len(turns_by_priority)} (priority + all) turns')



        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f'All allocations in {elapsed_time}s')
        print('\n' + '-'*100)

        # completed allocation
        # --------------------

        # for i,turn in enumerate(used_turns_w_vcs):
        #     print(f'{i:02} : {turn}')
        if verbose:
            self.print_turns_allowed_dict(allowed_turns_dict)


        # running_CDG.visualize_cdg(viz_type='circular')

        # Done
        # ----
        # set important var
        self.allowed_turns_dict = allowed_turns_dict

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f'Completed allowed turns calculation in {elapsed_time}s')
        print('\n' + '-'*100)

# setup for big worker
####################################################################################################

def drive_prune_apl_against_cycles_w_map(input_dict ):


    fname = input_dict['fname']
    all_paths_list = input_dict['all_paths_name']
    chosen_paths_list = input_dict['chosen_paths_name']

    verbose = input_dict['verbose']


    my_AP_VN = AllPaths_VNAlloc()

    use_chosen_paths = False
    if chosen_paths_list is not None:
        my_AP_VN.setup_w_chosen_paths(fname, chosen_paths_list)
        use_chosen_paths = True
    elif all_paths_list is not None:
        my_AP_VN.setup_w_r_map_and_apl(fname, all_paths_list)
    else:
        my_AP_VN.setup_w_r_map(fname)

    print(f'Completed setup with r_map, apl, and/or cpl')

    my_AP_VN.max_n_vcs = input_dict["max_n_evns"]
    is_safe = input_dict['safe']
    my_AP_VN.safe = is_safe
    no_transitions = input_dict['no_transitions']
    my_AP_VN.no_transitions = no_transitions
    is_robust = input_dict['robust']
    my_AP_VN.robust = is_robust
    # is_destbased = input_dict['destbased']
    # my_AP_VN.destbased = is_destbased
    is_ocs_disjoint_faults = input_dict['ocs_disjoint_faults']
    my_AP_VN.ocs_disjoint_faults = is_ocs_disjoint_faults

    symmetric = input_dict["symmetric"]
    xyzc_dims = input_dict["xyzc_dims"]
    if xyzc_dims is not None:
        my_AP_VN.set_xyzc_dims(xyzc_dims)
    mc_dims = input_dict["mc_dims"]
    sym_type = input_dict["sym_type"]

    my_AP_VN.symmetric = symmetric
    if symmetric:
        my_tpuv4_symmetry = TPUv4_Symmetry(xyzc_dims, mc_dims=mc_dims, sym_type=sym_type)
        my_tpuv4_symmetry.verify_symmetry_for_topology(my_AP_VN.r_map)
        my_AP_VN.my_tpuv4_symmetry = my_tpuv4_symmetry
        my_AP_VN.mc_dims = mc_dims
        my_AP_VN.sym_type = sym_type

    my_AP_VN.verbose = verbose

    start_time = time.time()

    my_AP_VN.apl_turn_addition_omnicdg(use_chosen_paths=use_chosen_paths)

    # sort and output (expand symmetry-reduced dict back to the full graph on the fly)
    n_routers = my_AP_VN.n_routers
    r_map = my_AP_VN.r_map
    max_n_vcs = my_AP_VN.max_n_vcs
    base_file_name = getattr(my_AP_VN, 'base_file_name', my_AP_VN.create_base_file_name())

    sorted_allowed_turns_dict = {}
    for i in range(n_routers):
        for j in range(n_routers):
            if i == j:
                continue
            if r_map[i][j] == 0:
                continue

            for k in range(n_routers):
                if i == k or j == k:
                    continue
                if r_map[j][k] == 0:
                    continue
                for vc_1 in range(max_n_vcs):
                    for vc_2 in range(max_n_vcs):
                        turn = ((i, j, vc_1), (j, k, vc_2))
                        sorted_allowed_turns_dict[turn] = my_AP_VN.get_allowed_turn(turn)

    my_AP_VN.sorted_allowed_turns_dict = sorted_allowed_turns_dict
    my_AP_VN.output_allowed_turns_vcs_(sorted_allowed_turns_dict, base_file_name)

    end_time = time.time()
    print(f'Entire algorithm in {end_time-start_time}s')

def main():

    parser = argparse.ArgumentParser(description='Verify topology values')

    parser.add_argument('--topology',type=str,help='.map file to evaluate',required=True)
    parser.add_argument('--all_path_list','-apl',type=str,help='apl file to evaluate')
    parser.add_argument('--chosen_paths_list','-cpl',type=str,help='.paths file to evaluate')

    parser.add_argument('--out_name','-o',type=str,help='output name (without extension)')

    parser.add_argument('--max_n_evns','-mevns',type=int,default=2,help='max allowed escape virtual networks')
    parser.add_argument('--safe',action='store_true',help='first add spanning tree')
    parser.add_argument('--no_transitions',action='store_true',help='ie layered routing')

    parser.add_argument('--robust',action='store_true',help="robust to OCS faults")
    parser.add_argument('--ocs_disjoint_faults',action='store_true',help="ocs_disjoint_faults to OCS faults")
    parser.add_argument('--destbased',action='store_true',help="destination-based routing")

    parser.add_argument('--symmetric',action='store_true',help='graph is vertex symmetric. route canonical flows')
    parser.add_argument('--sym_type',type=str,help="graph symmetry type. default 'trans'",choices=["trans","refl-trans"], default="trans")
    parser.add_argument('--mc_dims',nargs='+',type=int,help='Minimum "cube" of symmetry\'s x, y, and z dimensions. If unspecified then assuming single (zeroth) cube (i.e., x,y,z==cube). Type without parenthesis and use spaces, no commas')
    parser.add_argument('--xyzc_dims',nargs='+',type=int,help='type without parenthesis and use spaces, no commas')

    parser.add_argument('--verbose','-v',action='store_true',help='debug prints')
    args = parser.parse_args()

    global MAX_PROCS
    global sem

    fname = args.topology
    out_name = args.out_name
    apl_name = args.all_path_list
    cpl_name = args.chosen_paths_list
    safe = args.safe
    no_transitions = args.no_transitions
    max_n_evns = args.max_n_evns
    robust = args.robust
    ocs_disjoint_faults = args.ocs_disjoint_faults
    destbased = args.destbased

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
    elif args.xyzc_dims is not None:
        xyzc_dims = tuple(args.xyzc_dims)
        assert(len(xyzc_dims) == 4)


    assert(fname is not None)

    verbose = args.verbose

    input_dict = {'fname':fname,
                    'all_paths_name':apl_name,
                    'chosen_paths_name':cpl_name,
                    'out_name':out_name,
                    'safe':safe,
                    'no_transitions':no_transitions,
                    "max_n_evns":max_n_evns,

                    "robust":robust,
                    "ocs_disjoint_faults":ocs_disjoint_faults,
                    "destbased":destbased,

                    "symmetric":symmetric,
                    "sym_type":sym_type,
                    "mc_dims":mc_dims,
                    'xyzc_dims':xyzc_dims,

                    'verbose':args.verbose,
                    }

    drive_prune_apl_against_cycles_w_map(input_dict)

if __name__ == '__main__':
    main()