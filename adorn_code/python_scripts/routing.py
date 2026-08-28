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

'''

Description:
    Gives naive (zero and random-indexed) routing function(s) given a topology.

'''
# default python libraries
import argparse
import random
import os
import ast

# installed
import networkx as nx


class Routing:


    # class vars
    verbose = False
    light_verbose = True

    paths_output_path_prefix = './topologies_and_routing/routepath_lists'
    all_paths_output_path_prefix = './topologies_and_routing/allpath_lists'
    nr_list_output_path_prefix = './topologies_and_routing/nr_lists'

    # arbitrary definition of infinity
    INF = 999

    def __init__(self):

        # object vars
        #############

        # topology vars
        # -------------
        self.rmap_file_path = None
        self.rmap_name = None

        self.binary_r_map = None
        self.r_map = None
        self.n_routers = -1
        self.n_links = -1
        self.n_flows = -1
        self.G = None

        # tpuv4
        self.xyzc_dims = None

        # all path vars
        # -------------
        self.apl_file_path = None
        self.apl_name = None

        self.all_paths_twod = None
        self.all_paths_flat = None
        self.n_total_paths_twod = None
        self.avg_n_total_paths_per_flow = None

        # hop vars
        # --------
        self.hop_dists = None
        self.avg_hops = None
        self.total_hops = None

        # chosen path vars
        # ----------------
        self.pl_file_path = None
        self.pl_name = None
        self.chosen_paths_flat = None
        self.chosen_paths_twod = None
    
        # avoid multiple setups
        self.setup_rmap = False
        self.setup_apl = False
        self.setup_pl = False

    def setup_given_r_map(self, rmap_file_path, binary_r_map=False, xyzc_dims=None):

        print('\n' + '='*100)
        print('Setup given r_map' )
        print('-----------------')

        if self.setup_rmap:
            print(f'Already run. Returning...')
            return

        self.setup_rmap = True

        # topology vars
        # -------------
        self.rmap_file_path = rmap_file_path
        self.rmap_name = rmap_file_path.split('/')[-1].split('.')[0]
        self.binary_r_map = binary_r_map
        # sets self.r_map, self.n_routers, self.n_flows
        self.ingest_rmap()
        # (potentially) modifies self.r_map 
        self.sanitize_rmap()
        # sets self.n_links
        self.set_n_links_from_rmap()
        # sets self.G
        self.create_nwx_G_from_r_map()

        self.xyzc_dims = xyzc_dims
        if xyzc_dims is not None:
            self.is_tpuv4 = True
        else:
            self.is_tpuv4 = False

            # TODOD remove
        return

        # path and hop vars
        # -----------------
        # self.apl_file_path = None
        # self.apl_name = None

        self.set_dists_given_map()
        self.set_allpaths_given_map()
        self.set_npaths_given_allpaths()

        # # chosen path vars
        # # ----------------
        # self.pl_file_path = None
        # self.pl_name = None
        # self.chosen_paths_flat = None
        # self.chosen_paths_twod = None

        print(f'Completed setup of rmap {self.rmap_name}')
        print(f'\t# routers                 : {self.n_routers}')
        print(f'\t# links                   : {self.n_links}')

        print(f'Shortest/all paths computed')
        print(f'\ttotal_hops                : {self.total_hops}')
        print(f'\t# flows                   : {self.n_flows}')
        print(f'\tavg_hops                  : {self.avg_hops}')
        print(f'\t# total paths total       : {self.n_total_paths}')
        print(f'\tavg # total paths/flow    : {self.avg_n_total_paths_per_flow}')

    def setup_given_all_path_list(self, apl_file_path):

        print('\n' + '='*100)
        print('Setup given all paths list' )
        print('--------------------------')

        if self.setup_apl:
            print(f'Already run. Returning...')
            return

        self.setup_apl = True


        # path and hop vars
        # -----------------
        self.apl_file_path = apl_file_path
        self.apl_name = apl_file_path.split('/')[-1].split('.')[0]

        # sets self.all_paths_twod, self.all_paths_flat
        self.ingest_flat_apl()

        # sets self.n_links, self.n_routers, self.n_flows 
        self.set_nlinks_nrouters_from_flat_apl()

        # sets self.n_shortest_paths_twod, self.n_total_paths, self.avg_n_total_paths_per_flow 
        self.set_npaths_given_allpaths()

        # sets self.hop_dists
        self.set_dists_given_allpaths()

        # topology vars
        # -------------
        # self.rmap_file_path = None
        # self.rmap_name = None

        # # will assume binary when building r_map next
        # self.binary_r_map = True

        # sets self.r_map, self.n_routers, self.n_flows
        self.set_rmap_from_allpaths()

        # sets self.n_links
        self.set_n_links_from_rmap()
        # sets self.G
        self.create_nwx_G_from_r_map()

        # # chosen path vars
        # # ----------------
        # self.pl_file_path = None
        # self.pl_name = None
        # self.chosen_paths_flat = None
        # self.chosen_paths_twod = None

        print(f'Completed setup of all paths list {self.apl_name}')
        print(f'\t# routers                 : {self.n_routers}')
        print(f'\t# links                   : {self.n_links}')

        print(f'Shortest/all paths computed')
        print(f'\ttotal_hops                : {self.total_hops}')
        print(f'\t# flows                   : {self.n_flows}')
        print(f'\tavg_hops                  : {self.avg_hops}')
        print(f'\t# total paths total       : {self.n_total_paths}')
        print(f'\tavg # total paths/flow    : {self.avg_n_total_paths_per_flow}')

    def setup_given_path_list(self, pl_file_path):

        print('\n' + '='*100)
        print('Setup given chosen paths list' )
        print('-----------------------------')

        if self.setup_pl:
            print(f'Already run. Returning...')
            return

        self.setup_pl = True


        # chosen path vars
        # ----------------
        self.pl_file_path = pl_file_path
        self.pl_name = pl_file_path.split('/')[-1].split('.')[0]

        # sets self.chosen_paths_flat, self.chosen_paths_twod
        self.ingest_flat_pl()

        # sets self.n_links, self.n_routers, self.n_flows 
        self.set_nlinks_nrouters_from_flat_pl()

        # sets self.r_map, self.n_routers, self.n_flows
        self.set_rmap_from_chosenpaths()



        # sets self.n_shortest_paths_twod, self.n_total_paths, self.avg_n_total_paths_per_flow 
        self.set_npaths_given_chosenpaths()

        # sets self.hop_dists
        self.set_dists_given_chosenpaths()

        # topology vars
        # -------------
        # self.rmap_file_path = None
        # self.rmap_name = None

        # # will assume binary when building r_map next
        # self.binary_r_map = True

        # sets self.n_links
        self.set_n_links_from_rmap()
        # sets self.G
        self.create_nwx_G_from_r_map()

        print(f'Completed setup of chosen paths list {self.pl_name}')
        print(f'\t# routers                 : {self.n_routers}')
        print(f'\t# links                   : {self.n_links}')

        print(f'Shortest/all paths computed')
        print(f'\ttotal_hops                : {self.total_hops}')
        print(f'\t# flows                   : {self.n_flows}')
        print(f'\tavg_hops                  : {self.avg_hops}')
        print(f'\t# total paths total       : {self.n_total_paths}')
        print(f'\tavg # total paths/flow    : {self.avg_n_total_paths_per_flow}')

    # init funcs
    ####################################################################################################

    def ingest_rmap(self):
        assert(self.rmap_file_path is not None)
        self.r_map, self.n_routers = self.ingest_a_map_(self.rmap_file_path)

        n_routers = self.n_routers
        self.n_flows = (n_routers**2) - n_routers

    def sanitize_rmap(self):
        assert(self.r_map is not None)
        assert(self.binary_r_map is not None)

        self.r_map = self.sanitize_a_map_(self.r_map, binary_r_map=self.binary_r_map)

    def set_n_links_from_rmap(self):
        assert(self.r_map is not None)
        assert(self.n_routers != -1)

        if self.n_links != -1:
            return

        r_map = self.r_map
        n_routers = self.n_routers

        n_links = 0
        for i in range(n_routers):
            for j in range(n_routers):
                n_links += r_map[i][j]

        self.n_links = n_links

    def create_nwx_G_from_r_map(self):
        assert(self.r_map is not None)

        if self.G is not None:
            return

        self.G = self.create_an_nwx_G_from_a_map_(self.r_map)

    def set_dists_given_map(self):
        assert(self.r_map is not None)
        assert(self.n_routers != -1)
        assert(self.n_flows != -1)
    
        self.hop_dists = self.floyd_warshall_(self.r_map)
        self.total_hops, self.avg_hops = self.calc_avg_total_hops_(self.hop_dists, self.n_routers, self.n_flows)

    def set_allpaths_given_map(self):
        assert(self.r_map is not None)
        assert(self.n_routers != -1)
        assert(self.G is not None)
    
        # sets self.all_paths_flat, self.all_paths_twod, self.n_total_paths_twod, self.avg_n_total_paths_per_flow_twod
        self.all_paths_twod = self.calc_shortest_paths_given_nwx_graph_(self.G, self.n_routers)
        self.all_paths_flat = self.flatten_twod_all_paths_(self.all_paths_twod, self.n_routers)

    def set_npaths_given_allpaths(self):
        assert(self.n_routers != -1)
        assert(self.n_flows != -1)
        assert(self.all_paths_flat is not None)
    
        # sets self.n_shortest_paths_twod, self.avg_n_shortest_paths_per_flow, self.n_total_paths
        self.n_shortest_paths_twod, self.n_total_paths, self.avg_n_total_paths_per_flow = \
            self.calc_n_shortest_paths_per_flow_(self.all_paths_flat, self.n_routers, self.n_flows)

    def set_npaths_given_chosenpaths(self):
        assert(self.n_routers != -1)
        assert(self.n_flows != -1)
        assert(self.chosen_paths_flat is not None)
    
        # sets self.n_shortest_paths_twod, self.avg_n_shortest_paths_per_flow, self.n_total_paths
        self.n_shortest_paths_twod, self.n_total_paths, self.avg_n_total_paths_per_flow = \
            self.calc_n_shortest_paths_per_flow_(self.chosen_paths_flat, self.n_routers, self.n_flows)

    def ingest_flat_apl(self):
        assert(self.apl_file_path is not None)
        self.all_paths_twod, self.all_paths_flat = self.ingest_an_apl_(self.apl_file_path)

    def ingest_flat_pl(self):
        assert(self.pl_file_path is not None)
        self.chosen_paths_twod, self.chosen_paths_flat = self.ingest_a_pl_(self.pl_file_path)

    def set_nlinks_nrouters_from_flat_apl(self):
        assert(self.all_paths_flat is not None)

        if self.n_links != -1 and self.n_routers != -1 and  self.n_flows != -1:
            return

        self.n_links, self.n_routers, self.n_flows = self.calc_nlinks_nrouters_from_flat_paths_(self.all_paths_flat)

    def set_nlinks_nrouters_from_flat_pl(self):
        assert(self.chosen_paths_flat is not None)

        if self.n_links != -1 and self.n_routers != -1 and  self.n_flows != -1:
            return

        self.n_links, self.n_routers, self.n_flows = self.calc_nlinks_nrouters_from_flat_paths_(self.chosen_paths_flat)

    def set_dists_given_allpaths(self):
        assert(self.all_paths_flat is not None)
        assert(self.n_routers != -1)
        assert(self.n_flows != -1)

        self.hop_dists = self.calc_hop_dists_given_paths_(self.all_paths_flat , self.n_routers)
        self.total_hops, self.avg_hops = self.calc_avg_total_hops_(self.hop_dists, self.n_routers, self.n_flows)

    def set_dists_given_chosenpaths(self):
        assert(self.chosen_paths_flat is not None)
        assert(self.n_routers != -1)
        assert(self.n_flows != -1)

        self.hop_dists = self.calc_hop_dists_given_paths_(self.chosen_paths_flat , self.n_routers)
        self.total_hops, self.avg_hops = self.calc_avg_total_hops_(self.hop_dists, self.n_routers, self.n_flows)

    def set_rmap_from_allpaths(self):
        assert(self.all_paths_flat is not None)
        assert(self.n_routers != -1)

        if self.r_map is not None:
            return

        self.r_map = self.create_rmap_from_paths_(self.all_paths_flat, self.n_routers)

    def set_rmap_from_chosenpaths(self):
        assert(self.chosen_paths_flat is not None)
        assert(self.n_routers != -1)

        if self.r_map is not None:
            return

        self.r_map = self.create_rmap_from_paths_(self.chosen_paths_flat, self.n_routers)

    # getters
    ####################################################################################################
    def get_chosen_paths(self):
        assert(self.chosen_paths_flat is not None)
        assert(self.chosen_paths_twod is not None)

        return self.chosen_paths_flat, self.chosen_paths_twod

    def get_n_routers(self):
        assert(self.n_routers != -1)
        return self.n_routers

    def get_base_name(self):
        if self.pl_name is not None:
            return self.pl_name
        if self.apl_name is not None:
            return self.apl_name
        if self.rmap_name is not None:
            return self.rmap_name

    # class methods
    ####################################################################################################

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

                # print(f'connecting {src} -> {dest}')

                G.add_edge(src,dest)

        return G

    @classmethod
    def calc_avg_total_hops_(cls, hop_dists, n_routers, n_flows):

        total_hops = 0
        for i in range(n_routers):
            for j in range(n_routers):
                total_hops += hop_dists[i][j]

        avg_hops = (total_hops / n_flows)

        return total_hops, avg_hops

    @classmethod
    def calc_n_shortest_paths_per_flow_(cls, paths_flat, n_routers, n_flows):

        n_shortest_paths_twod = [ [ 0 for _ in range(n_routers)] for __ in range(n_routers)]
        tot_n_shortest_paths = 0

        all_shortest_paths_twod = [ [ [] for _ in range(n_routers)] for __ in range(n_routers)]

        for path in paths_flat:
            s = path[0]
            d = path[-1]
            all_shortest_paths_twod[s][d].append(path)

        for i in range(n_routers):
            for j in range(n_routers):
                if i==j:
                    continue
                n_paths = len(all_shortest_paths_twod[i][j])
                n_shortest_paths_twod[i][j] = n_paths
                tot_n_shortest_paths += n_paths
        
        avg_n_shortest_paths_per_flow = tot_n_shortest_paths / n_flows

        return n_shortest_paths_twod, tot_n_shortest_paths, avg_n_shortest_paths_per_flow

    @classmethod
    def flatten_twod_all_paths_(cls, all_paths_twod, n_routers):

        all_paths_flat = []
        for i in range(n_routers):
            for j in range(n_routers):
                for p in all_paths_twod[i][j]:
                    all_paths_flat.append(p)
        return all_paths_flat

    @classmethod
    def calc_shortest_paths_given_nwx_graph_(cls, G, n_routers):
        all_paths = [ [ [] for _ in range(n_routers)] for __ in range(n_routers)]

        for src in range(n_routers):
            for dest in range(n_routers):

                if(src == dest):
                    all_paths[src][dest].append([src])
                    continue
                
                short_path_generator = nx.all_shortest_paths(G,src,dest)
                short_path_list = list()
                short_path_list += short_path_generator

                all_paths[src][dest] = short_path_list

        return all_paths

    @classmethod
    def floyd_warshall_(cls, this_map):
        INF = cls.INF

        n_routers = len(this_map)

        graph = [[item if item==1 else INF for item in row] for row in this_map]

        for i in range(0,n_routers):
            graph[i][i]=0

        dist = list(map(lambda p: list(map(lambda q: q, p)), graph))

        for r in range(n_routers):
            for p in range(n_routers):
                for q in range(n_routers):
                    # shorter path through r
                    if (dist[p][r]+ dist[r][q]) < dist[p][q]:
                        dist[p][q] = dist[p][r] + dist[r][q]

        return dist

    @classmethod
    def calc_hop_dists_given_paths_(cls, paths_flat, n_routers):
        INF = cls.INF

        hop_dists = [ [ INF for _ in range(n_routers) ] for __ in range(n_routers) ]


        for path in paths_flat:
            s = path[0]
            d = path[-1]
            if s==d:
                hop_dists[s][d] = 0
                continue
            
            plen = len(path) - 1

            hop_dists[s][d] = min(hop_dists[s][d], plen)

        return hop_dists

    @classmethod
    def ingest_an_apl_(cls, path_name):

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
        
        n_routers += 1

        allpath_list = [[ [] for _ in range(n_routers) ] for __ in range(n_routers)]

        for path in flat_path_list:
            s = path[0]
            d = path[-1]
            allpath_list[s][d].append(path)

        return allpath_list, flat_path_list

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

    @classmethod
    def calc_nlinks_nrouters_from_flat_paths_(cls, paths_flat):
        n_routers = 0
        links = []
        for path in paths_flat:

            plen = len(path) - 1
            for i in range(plen):
                link = (path[i],path[i+1])
                if link not in links:
                    links.append(link)

            # set n_routers
            for n in path:
                n_routers = max(n, n_routers)

        # account for zero indexing
        n_routers += 1

        n_links = len(links)
        n_routers = n_routers
        n_flows = (n_routers**2) - n_routers

        return n_links, n_routers, n_flows

    @classmethod
    def create_rmap_from_paths_(cls, paths_flat, n_routers):

        r_map = [ [ 0 for _ in range(n_routers) ] for __ in range(n_routers) ]

        for path in paths_flat:
            plen = len(path) - 1
            for i in range(plen):
                s = path[i]
                d = path[i+1]
                r_map[s][d] = 1
        
        return r_map

    ###################################################################
    # prints

    def print_twodmat(self, mat):
        for i, row in enumerate(mat):
            print(f'{i} : {row}')

    def print_path(self, p):

        print(f'path {p[0]} to {p[-1]} (len {len(p)-1}): ',end='')

        l = len(p)
        for i in range(l-1):
            e = p[i]
            print(f'{e}->',end='')
        print(f'{p[-1]}')

    def print_paths_2dmat(self, pmat):

        for src_paths in pmat:
            for p in src_paths:
                if len(p) == 0:
                    continue
                self.print_path(p)

    def print_path_list(self, pl):
        for p in pl:
            self.print_path(p)

    # output
    ####################################################################################################
    def output_pathlist(self, path_list, base_file_name):

        full_name = f'{base_file_name}.paths'

        full_out_path = os.path.join(self.paths_output_path_prefix, \
                full_name)

        with open(full_out_path, 'w+') as of:
            of.write(base_file_name + '\n')
            for path in path_list:
                of.write(f'{path}\n')

        print(f'Wrote to {full_out_path}')

    # naive
    ####################################################################################################

    def naive(self, all_path_list, first_index=False):
        assert(self.n_routers != -1)
        n_routers = self.n_routers
        
        naive_path_list = []

        for src in range(n_routers):
            for dest in range(n_routers):
                if src==dest:
                    chosen_path = [src]
                    naive_path_list.append(chosen_path.copy())
                    continue
                
                if first_index:
                    chosen_path = all_path_list[src][dest][0]
                else:
                    chosen_path = random.choice(all_path_list[src][dest])

                naive_path_list.append(chosen_path.copy())

        return naive_path_list

    # tpuv4 funcs
    ####################################################################################################

    def xyz_to_r(self, x,y,z):
        assert(self.xyzc_dims is not None)
        (x_dim, y_dim,z_dim,cube_dim) = self.xyzc_dims

        return x + y*x_dim + z*x_dim*y_dim

    def r_to_xyz(self, r):
        assert(self.xyzc_dims is not None)

        (x_dim, y_dim,z_dim,cube_dim) = self.xyzc_dims

        xy_slice_size = x_dim*y_dim

        temp_r = r

        z = temp_r // xy_slice_size
        temp_r = temp_r % xy_slice_size
        y = temp_r // x_dim
        x = temp_r % x_dim

        return x,y,z

    def r_to_xyz_as_dict(self, r):
        x, y, z = self.r_to_xyz(r)
        as_dict = {'x':x,'y':y,'z':z}

        return as_dict

    def calc_dists_direct_wraparound(self, src_coord, dest_coord, cube_dim):
        
        min_coord = min(dest_coord, src_coord)
        max_coord = max(dest_coord, src_coord)

        return abs(dest_coord - src_coord), abs(cube_dim - max_coord + min_coord)

    # dor
    ####################################################################################################

    def dor_pt(self, dim_tiebreak=True):
        assert(self.n_routers != -1)
        assert(self.xyzc_dims is not None)

        n_routers = self.n_routers
        (x_dim, y_dim,z_dim,cube_dim) = self.xyzc_dims

        dim_dict = {'x':x_dim, 'y':y_dim, 'z':z_dim}

        ordered_dim_list = []

        ordered_dim_list.append( min([k for k,v in dim_dict.items()]) )
        ordered_dim_list.append( min([k for k,v in dim_dict.items() if k not in ordered_dim_list]) )
        ordered_dim_list.append( min([k for k,v in dim_dict.items() if k not in ordered_dim_list]) )


        flat_dor_chosen_paths = []

        for src in range(n_routers):
            print(f'src={src}')
            for dest in range(n_routers):
                if src==dest:
                    flat_dor_chosen_paths.append([src])
                    continue

                
                src_coord_dict = self.r_to_xyz_as_dict(src)
                dest_coord_dict = self.r_to_xyz_as_dict(dest)

                if self.verbose:
                    print('-'*100)
                    print(f'{src} ({src_coord_dict}) -> {dest} ({dest_coord_dict})')

                # deep copy
                cur_coord_dict = { k : v for k,v in src_coord_dict.items() }
                cur_node = src

                this_path = []

                for dim in ordered_dim_list:
                    src_coord = src_coord_dict[dim]
                    dest_coord = dest_coord_dict[dim]

                    this_dim_length = dim_dict[dim]
                    direct_dist, wraparound_dist = self.calc_dists_direct_wraparound(src_coord,dest_coord, this_dim_length)

                    if self.verbose:
                        print(f'\tin dim {dim}, the distances are direct,wrap = {direct_dist}, {wraparound_dist}')

                    if src_coord == dest_coord:
                        continue


                    delta = None

                    # tie. break by dim
                    if direct_dist == wraparound_dist and dim_tiebreak:
                        if src_coord % 2 == 0:
                            delta = 1
                        else:
                            delta = -1
                    # tie always positive
                    elif direct_dist == wraparound_dist:
                        delta = 1
                    # direct better
                    elif direct_dist < wraparound_dist:
                        delta = (dest_coord - src_coord) // abs(dest_coord - src_coord)
                    else:
                        delta = (src_coord - dest_coord) // abs(dest_coord - src_coord)

                    if self.verbose:
                        print(f'\talong {dim} will use delta {delta}')
                        print(f'\tcurrent node is {cur_node} and coords are {cur_coord_dict}')
                    
                    # cur_coord = cur_coord_dict[dim]
                    while cur_coord_dict[dim] != dest_coord:

                        x = cur_coord_dict['x']
                        y = cur_coord_dict['y']
                        z = cur_coord_dict['z']
                        cur_node = self.xyz_to_r(x,y,z)

                        this_path.append(cur_node)
                        cur_coord_dict[dim] += delta

                        # wrap
                        if cur_coord_dict[dim] < 0:
                            cur_coord_dict[dim] = this_dim_length + cur_coord_dict[dim] 
                        if cur_coord_dict[dim] > this_dim_length - 1:
                            cur_coord_dict[dim] = cur_coord_dict[dim] % this_dim_length

                        if self.verbose:
                            print(f'\tnow, current node is {cur_node} and coords are {cur_coord_dict}')

                        if abs(cur_node) > n_routers:
                            quit()

                    if self.verbose:
                        print(f'\tcompleted dim {dim}. current path = {this_path}')

                # cherry on top
                this_path.append(dest)

                if self.verbose:
                    input(f'completed {src}->{dest} w/ path {this_path}')
                
                flat_dor_chosen_paths.append(this_path)

        return flat_dor_chosen_paths


    # big worker
    ####################################################################################################


    def route(self, topo_type='pt', alg_type='naive_random', output_all_paths_list=False):

        if output_all_paths_list:
            pass

        print('\n' + '='*100)
        print('Routing')
        print('-------')


        base_map_name = self.rmap_name
        base_paths_name = f'{base_map_name}_{alg_type}'

        # route and output
        if 'naive_random' == alg_type.lower():
            apl = self.all_paths_twod
            chosen_paths_flat = self.naive(apl)
        elif 'naive_first_index' == alg_type.lower():
            apl = self.all_paths_twod
            chosen_paths_flat = self.naive(apl, first_index=True)
        elif 'dor_dim_tiebreak' == alg_type.lower(): # and topo_type=='pt':
            # do not start with an apl. Use algorithmic path creation
            # ssert(self.xyzc_dims is not None)
            chosen_paths_flat = self.dor_pt(dim_tiebreak=True)
        else:
            print(f'Alg type unknown. Returning early...')
            return
        
        # output
        self.output_pathlist(chosen_paths_flat, base_paths_name)

        print(f'done')




# running as script
####################################################################################################

def route_file(input_dict):

    fname = input_dict['fname']
    alg = input_dict['alg']
    all_paths_list = input_dict['all_paths_name']

    topo_type = ''
    if 'pt_' in fname:
        topo_type = 'pt'

    is_tpuv4 = input_dict['is_tpuv4']

    r = Routing()

    if all_paths_list is not None:
        r.setup_given_all_path_list(all_paths_list)
    elif not is_tpuv4:
        r.setup_given_r_map(fname)
    else:
        r.setup_given_r_map(fname, xyzc_dims=input_dict['xyzc_dims'])

    r.verbose = input_dict['verbose']

    r.route( alg_type=alg,topo_type=topo_type)

    del(r)

def main():
    parser = argparse.ArgumentParser(description='Verify topology values')
    parser.add_argument('--topology',type=str,help='.map file to evaluate')
    parser.add_argument('--all_path_list',type=str,help='apl file to evaluate')
    parser.add_argument('--alg',type=str,help='alg (naive_first_index, naive_random)')
    parser.add_argument('--out_name','-o',type=str,help='output name (without extension)')

    parser.add_argument('--xyzc_dims',nargs='+',type=int,help='type without parenthesis and use spaces, no commas')

    parser.add_argument('--verbose','-v',action='store_true',help='debug prints')

    args = parser.parse_args()

    fname = args.topology
    out_name = args.out_name
    alg = args.alg
    apl_name = args.all_path_list

    try:
        xyzc_dims = tuple(args.xyzc_dims)
    except:
        xyzc_dims = None
    is_tpuv4 = False
    if xyzc_dims is not None:
        is_tpuv4 = True

    verbose = args.verbose

    input_dict = {'fname':fname,
                    'alg':alg,
                    'all_paths_name':apl_name,
                    'out_name':out_name,
                    'is_tpuv4':is_tpuv4,
                    'xyzc_dims':xyzc_dims,

                    'verbose':args.verbose,
                    }


    if fname is not None or apl_name is not None:
        route_file(input_dict)

    else:
        print('No file list provided. Exiting...')
        quit(-1)

if __name__ == '__main__':
    main()


