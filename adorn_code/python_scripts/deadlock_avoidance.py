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
    Allocates virtual networks to a given topology and routing scheme

    LASH 

    DFSSSP ( https://htor.inf.ethz.ch/publications/img/domke_deadlock_free_routing.pdf )
        Uses SSSP ( https://htor.inf.ethz.ch/publications/img/hoefler-ib-routing.pdf )

    Nue : 
        Uses Multilevel K-Way Partitioning Algorithm ( https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=d545a4e5a5e00935e59141f26041531ba1aa97a0 )

    Up/Down ( https://dl.acm.org/doi/10.1109/49.105178 )

'''

# std
import argparse
import random
import math
import os
import sys
import ast
import json
import multiprocessing
from copy import deepcopy
from collections import deque

# venv
import networkx as nx

# local
from omnicdg import OmniCDG

ASYNC_JOB_RESULT_FILE = 'job_files/running_results.txt'
ASYNC_JOB_RESULTS_JSON_DIR = 'job_files/json_results'


class VNAllocator:

    verbose = False
    slow  = False

    vn_map_output_path_prefix = './topologies_and_routing/vn_maps'
    paths_output_path_prefix = './topologies_and_routing/routepath_lists'
    all_paths_output_path_prefix = './topologies_and_routing/allpath_lists'

    # arbitrary definition of infinity
    # make huge to be greater than all n_routers^2
    INF = sys.maxsize#999

    MAX_PROCS = 32

    def __init__(self):

        # topology vars
        # -------------
        self.filename = None
        self.r_map = None
        self.edge_list = None
        self.n_routers = -1
        self.n_links = -1
        self.avg_hops = -1.0

        # VN vars
        # -------
        self.n_vns = -1
        self.vn_map = None
        self.allocated_path_list = None
        self.max_n_vcs = -1

        # path vars
        # ---------
        self.cpl_file_path = None
        self.cpl_name = None
        self.twod_path_list = None
        self.flat_path_list = None

        # file I/O
        # --------
        self.out_vn_map_path_prefix = None

    def setup_for_r_map_based(self, rmap_file_path, max_n_vcs, binary_r_map=True):

        print('\n' + '-'*100)
        print('Setup' )


        # topology vars
        # -------------
        self.rmap_file_path = rmap_file_path
        self.rmap_name = rmap_file_path.split('/')[-1].split('.')[0]
        self.binary_r_map = binary_r_map
        # sets self.r_map, self.n_routers, self.n_flows
        self.ingest_rmap()
        # (potentially) modifies self.r_map 
        self.sanitize_rmap()
        # sets self.edge_list
        self.set_edge_list_from_rmap()
        # sets self.n_links
        self.set_n_links_from_rmap()
        # sets self.G
        self.create_nwx_G_from_r_map()
        # sets self.all_shortest_paths_twod
        self.calc_all_shortest_paths()


        # deadlock vars
        # -------------
        self.max_n_vcs = max_n_vcs

        # for threading
        self.sem = multiprocessing.Semaphore(self.MAX_PROCS)

        print(f'Completed setup of rmap {self.rmap_file_path}')
        print(f'\t# routers : {self.n_routers}')
        print(f'\t# links   : {self.n_links}')
        print(f'\t# VCs     : {self.max_n_vcs}')

        print('\n' + '-'*100)

    def setup(self, cpl_file_path):

        print('\n' + '-'*100)
        print('Setup' )


        # path list vars
        # --------------
        self.cpl_file_path = cpl_file_path
        self.cpl_name = cpl_file_path.split('/')[-1].split('.')[0]
        # sets self.cpl, self.flat_cpl
        self.ingest_flat_cpl()

        # topology vars
        # -------------
        # sets self.n_links, self.n_routers
        self.set_topo_vars_from_cpl()

        print(f'Completed setup of cpl {self.cpl_name}')
        print(f'\t# routers : {self.n_routers}')
        print(f'\t# links   : {self.n_links}')
        print(f'\tavg hops  : {self.avg_hops}')

        print('\n' + '-'*100)

    def reset(self):
        self.vn_map = None
        self.allocated_path_list = None
        self.n_vns = -1

        print(f'Reset vn_map, allocated_path_list, and n_vns')

    def ingest_flat_cpl(self):
        assert(self.cpl_file_path is not None)
        self.twod_path_list, self.flat_path_list = \
                self.ingest_a_cpl_(self.cpl_file_path)

    @classmethod
    def ingest_a_cpl_(self, path_name):

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

    def set_topo_vars_from_cpl(self):

        self.n_routers, self.n_links, self.avg_hops = \
                self.set_topo_vars_from_a_cpl_(self.flat_path_list)

    @classmethod
    def set_topo_vars_from_a_cpl_(cls,flat_path_list):
        n_routers = -1
        links_dict = {}
        tot_hops = 0

        for path in flat_path_list:
            plen = len(path) - 1

            tot_hops += plen

            for i in range(plen):
                link = (path[i],path[i+1])
                try:
                    _ = links_dict[link]
                except:
                    links_dict.update({link : None})
                
            temp_list = [n_routers] + path
            n_routers = max(temp_list)
        
        # account for zero indexing
        n_routers += 1

        n_links = len(links_dict.keys())

        n_flows = (n_routers**2) - n_routers
        avg_hops = float(tot_hops) / float(n_flows)

        return n_routers, n_links, avg_hops

    def ingest_rmap(self):
        assert(self.rmap_file_path is not None)
        self.r_map, self.n_routers = self.ingest_a_map_(self.rmap_file_path)

        n_routers = self.n_routers
        self.n_flows = (n_routers**2) - n_routers

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

        r_map = self.r_map
        n_routers = self.n_routers

        n_links = 0
        for i in range(n_routers):
            for j in range(n_routers):
                n_links += r_map[i][j]

        self.n_links = n_links

    def set_edge_list_from_rmap(self):
        assert(self.r_map is not None)
        r_map = self.r_map

        edge_list = []
        for src, dest_conns in enumerate(r_map):
            for dest, conned in enumerate(dest_conns):
                if conned > 0:
                    edge_list.append( (src,dest) )
        self.edge_list = edge_list


    def create_nwx_G_from_r_map(self):
        assert(self.r_map is not None)
        self.G = self.create_an_nwx_G_from_a_map_(self.r_map)

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

    def calc_all_shortest_paths(self):
        assert(self.G is not None)
        assert(self.n_routers != -1)

        self.all_shortest_paths_twod = self.calc_shortest_paths_given_nwx_graph_(self.G, self.n_routers)

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

    # output
    ####################################################################################################

    def output_vn_map_to_file(self, base_file_name, vn_map=None):

        if vn_map is None:
            assert(self.vn_map is not None)
            vn_map = self.vn_map

        full_name = f'{base_file_name}.vn'

        out_dir = self.vn_map_output_path_prefix
        out_path = os.path.join(out_dir, full_name)

        self.output_a_vn_map_to_file_(out_path, vn_map)

    @classmethod
    def output_a_vn_map_to_file_(cls, out_path, vn_map):
        with open(out_path,'w+') as of:
            for row in vn_map:
                for e in row:
                    l = f'{e} '
                    of.write(l)
                of.write('\n')
        print(f'Wrote to {out_path}')

    def output_pathlist(self, base_file_name, path_list=None):

        if path_list is None:
            path_list = self.chosen_paths_flat

        full_name = f'{base_file_name}.paths'

        full_out_path = os.path.join(self.paths_output_path_prefix, \
                full_name)

        with open(full_out_path, 'w+') as of:
            of.write(base_file_name + '\n')
            for path in path_list:
                of.write(f'{path}\n')

        print(f'Wrote to {full_out_path}')

    def output_allpathslist(self, base_file_name, all_paths_list=None):

        if all_paths_list is None:
            all_paths_list = self.all_paths_list

        full_name = f'{base_file_name}.rallpaths'

        full_out_path = os.path.join(self.all_paths_output_path_prefix, \
                full_name)

        with open(full_out_path, 'w+') as of:

            for path in all_paths_list:
                line = ''
                for n in path[:-1]:
                    line += f'{n} '
                line += f'{path[-1]}'
                of.write(f'{line}\n')

        print(f'Wrote to {full_out_path}')

    # LASH
    ####################################################################################################

    def lash(self, alg):

        first_index = False
        if 'first_index' in alg:
            first_index = True
            alg.replace('_first_index','')

        if alg == 'lash_dag':
            self.lash_dag(first_index=first_index)
        elif alg == 'lash_ssd':
            self.lash_ssd(first_index=first_index)
        else:
            print(f'LASH variant unknown. Exiting...')
            quit()

    def lash_ssd(self, first_index=False):
        assert(self.flat_path_list is not None)
        # MUST DEEPCOPY OR IT WILL GROW
        flat_path_list = deepcopy(self.flat_path_list)

        gu_list = []

        for path in flat_path_list:
            small_path_list = [ path ]
            gu_list.append(small_path_list)
        
        if not first_index:
            random.shuffle(gu_list)

        return self.lash_given_gus(gu_list)

    def lash_dag(self, first_index=False):
        assert(self.twod_path_list is not None)
        # MUST DEEPCOPY OR IT WILL GROW
        twod_path_list = deepcopy(self.twod_path_list)

        gu_list = []

        for dest_paths in twod_path_list:
            if not first_index:
                random.shuffle(dest_paths)
            gu_list.append(dest_paths)

        if not first_index:
            random.shuffle(gu_list)

        return self.lash_given_gus(gu_list)

    def lash_given_gus(self, gu_list):
        assert(self.n_routers != -1)
        n_routers = self.n_routers

        first_cdg = OmniCDG(n_nodes = n_routers)#OmniCDG()
        # first_cdg.init_w_n_nodes(n_routers)
        cdg_list = [first_cdg ]

        allocated_path_list = [[]]

        gus_processed = 0

        while gu_list:
            this_gu = gu_list.pop(0)

            # probe adding to current cdg
            cur_cdg = cdg_list[-1]

            turn_group_no_vcs = self.turns_no_vcs_from_paths(this_gu)

            causes_dl = cur_cdg.probe_turns_for_deadlock_common_vc(turn_group_no_vcs)

            if self.verbose:
                cdg_num = len(cdg_list)
                print(f'CDG # {cdg_num} and gu # {gus_processed}')
                print(f'\tcauses dl? {causes_dl}')
            
            if not causes_dl:
                cdg_list[-1].add_node_turns_common_vc(turn_group_no_vcs)
                allocated_path_list[-1] += this_gu
            else:
                new_cdg = OmniCDG(n_nodes = n_routers) #OmniCDG()
                # new_cdg.init_w_n_nodes(n_routers)
                cdg_list.append(new_cdg )
                cdg_list[-1].add_node_turns_common_vc(turn_group_no_vcs)

                allocated_path_list.append(this_gu)

            gus_processed += 1

        if self.verbose:
            print(f'allocated_path_list=')
            for vn, paths in enumerate(allocated_path_list):
                print(f'{vn:02} : {len(paths)}\n\t{paths}')

        # Done
        n_vns = len(cdg_list)
        self.n_vns = n_vns
        self.allocated_path_list = deepcopy(allocated_path_list)

        # also set as member var
        vn_map = self.set_vn_map_from_path_list()
        self.vn_map = vn_map
        return vn_map, n_vns


    # DF of DFSSSP
    ####################################################################################################

    def dfsssp(self, alg):

        back_edge_heuristic = None
        options = ['weakest','strongest','random','first_index']
        for option in options:
            if option in alg:
                back_edge_heuristic = option
                alg.replace(f'_{option}','')

        if back_edge_heuristic is None:
            print(f'DFSSSP back edge removal not specified. Defaulting to "weakest"')
            back_edge_heuristic = 'weakest'

        self.dfsssp_alg_two(back_edge_heuristic=back_edge_heuristic)

        # return as member vars

    def dfsssp_alg_two(self, back_edge_heuristic='weakest'):
        assert(self.flat_path_list is not None)
        assert(self.n_routers != -1)
        n_routers = self.n_routers
        # MUST DEEPCOPY OR IT WILL GROW
        flat_path_list = deepcopy(self.flat_path_list)
    
        all_turns = self.turns_no_vcs_from_paths(flat_path_list)

        deadlock_free_paths = []
        deadlock_free_paths.append(flat_path_list)
        n_deadlock_free_cdgs = 0

        first_cdg = OmniCDG(n_nodes = n_routers) # OmniCDG()
        # first_cdg.init_w_n_nodes(n_routers)
        first_cdg.add_node_turns_common_vc(all_turns)
        cdg_list = [first_cdg ]

        all_turn_usage = self.turn_usage_from_paths(flat_path_list)
        turn_usage_list = [all_turn_usage]

        loop_iter = 0
        while (n_deadlock_free_cdgs < len(cdg_list)):
            # 1) find a cycle from current cdg
            # 2) get back edge from cycle
            # 3) find all paths of this cdg that utilize that back edge/turn
            # 4) remove these paths from this path_list and add to next path list
            # 5) regenerate this cdg and next cdg from this and next path list
            # 6) repeat

            if loop_iter % 100 == 0:
                print(f'\tLoop iter {loop_iter} : n_deadlock_free_cdgs = {n_deadlock_free_cdgs}')

            # 1) get cycle from current cdg
            cdg_cycle = cdg_list[n_deadlock_free_cdgs].networkx_get_cycle_as_nodes_no_vcs()

            if self.verbose:
                print(f'\tFound cycle {cdg_cycle}')

            if len(cdg_cycle) == 0:
                n_deadlock_free_cdgs += 1
                continue

            # 2) get back edge (aka turn)
            turn_usage = turn_usage_list[n_deadlock_free_cdgs]
            back_edge = self.get_back_edge(cdg_cycle, turn_usage, back_edge_heuristic=back_edge_heuristic )

            # 3) get paths OF THIS CDG that use back edge (turn)
            these_paths = deadlock_free_paths[n_deadlock_free_cdgs]
            bad_paths = self.get_paths_using_be(these_paths, back_edge)

            # 4) move bad_paths
            # init
            try:
                _ = deadlock_free_paths[n_deadlock_free_cdgs + 1]
            except:
                deadlock_free_paths.append([])

            for bp in bad_paths:
                if self.verbose:
                    print(f'\tMoving path {bp} from VN {n_deadlock_free_cdgs} to {n_deadlock_free_cdgs+1}')
                deadlock_free_paths[n_deadlock_free_cdgs + 1].append(bp)
                deadlock_free_paths[n_deadlock_free_cdgs].remove(bp)

            # 5) update CDGs and usage
            if self.verbose:
                print(f'recalculating cdgs {n_deadlock_free_cdgs} and {n_deadlock_free_cdgs + 1}')
            
            # cant just remove turns from bad_paths as some might be unnecessary casualties
            try:
                _ = turn_usage_list[n_deadlock_free_cdgs + 1]
            except:
                turn_usage_list.append(None)
            try:
                _ = cdg_list[n_deadlock_free_cdgs + 1]
            except:
                cdg_list.append(None)

            # so must regen from path list?
            donor = n_deadlock_free_cdgs
            donor_paths = deadlock_free_paths[donor]
            donor_turns = self.turns_no_vcs_from_paths(donor_paths)
            donor_turn_usage = self.turn_usage_from_paths(donor_paths)
            turn_usage_list[donor] = donor_turn_usage

            donor_cdg = OmniCDG(n_nodes = n_routers) #OmniCDG()
            # donor_cdg.init_w_n_nodes(n_routers)
            donor_cdg.add_node_turns_common_vc(donor_turns)
            cdg_list[donor] = donor_cdg

            receiver = n_deadlock_free_cdgs + 1
            receiver_paths = deadlock_free_paths[receiver]
            receiver_turns = self.turns_no_vcs_from_paths(receiver_paths)
            receiver_turn_usage = self.turn_usage_from_paths(receiver_paths)
            turn_usage_list[receiver] = receiver_turn_usage

            receiver_cdg = OmniCDG(n_nodes = n_routers) #OmniCDG()
            # receiver_cdg.init_w_n_nodes(n_routers)
            receiver_cdg.add_node_turns_common_vc(receiver_turns)
            cdg_list[receiver] = receiver_cdg

            loop_iter += 1
        
        print(f'Completed DFSSSP. # VNs = {n_deadlock_free_cdgs}')

        # Done
        n_vns = len(cdg_list)
        self.n_vns = n_vns
        self.allocated_path_list = deepcopy(deadlock_free_paths)

        # also set as member var
        vn_map = self.set_vn_map_from_path_list()
        self.vn_map = vn_map
        return vn_map, n_vns

    def turn_usage_from_paths(self, path_list):

        turn_usage_dict = {}
        for path in path_list:
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

                # paths that use this turn
                try:
                    turn_usage_dict[this_turn] += 1
                except:
                    turn_usage_dict.update({this_turn : 1})
        return turn_usage_dict

    def get_back_edge(self, cycle_as_nodes, turn_usage, back_edge_heuristic='random'):

        if back_edge_heuristic == 'random':
            one_turn = random.choice(cycle_as_nodes)
        elif back_edge_heuristic == 'weakest':
            one_turn, _ = self.get_weakest_strongest_edges(cycle_as_nodes, turn_usage)
        elif back_edge_heuristic == 'strongest':
            _ , one_turn = self.get_weakest_strongest_edges(cycle_as_nodes, turn_usage)
        elif back_edge_heuristic == 'first_index':
            one_turn = cycle_as_nodes[0]
        else:
            print(f'Unknown back edge heuristic. Exiting...')
            quit()

        return one_turn

    def get_weakest_strongest_edges(self, cycle_as_nodes, turn_usage):

        min_turn = None
        min_turn_val = self.INF
        max_turn = None
        max_turn_val = -1

        for turn in cycle_as_nodes:
            
            turn_val = turn_usage[turn]

            if turn_val < min_turn_val or min_turn is None:
                min_turn_val = turn_val
                min_turn = turn
            if turn_val > max_turn_val or max_turn is None:
                max_turn_val = turn_val
                max_turn = turn
        if self.verbose:
            print(f'Found min turn {min_turn} w/ val {min_turn_val}')
            print(f'Found max turn {max_turn} w/ val {max_turn_val}')

        return min_turn, max_turn

    def get_paths_using_be(self, potential_paths, back_edge):

        bad_paths = []

        for path in potential_paths:

            plen = len(path) - 1
            n_turns = plen - 1

            if n_turns <= 0:
                continue

            # determine if path contains the bad back edge
            related_path = False

            for i in range(n_turns):
                n_a = path[i]
                n_b = path[i+1]
                n_c = path[i+2]
                this_edge = ( (n_a, n_b), (n_b, n_c))

                if this_edge == back_edge:
                    if self.verbose:
                        print(f'match: this_edge={this_edge}, this_be={back_edge}')
                    bad_paths.append(path)
                    break

        if self.verbose:
            print(f'Found bad paths: {bad_paths}')

        return bad_paths

    # Up/Down
    ####################################################################################################

    def up_down(self, alg):

        first_index = False
        if 'first_index' in alg:
            first_index = True
            alg.replace('_first_index','')

        self.up_down_alg(first_index=first_index)

        # return as member vars

    def up_down_alg(self, first_index=False, multiprocessed=False):
        assert(self.n_routers != -1)
        n_routers = self.n_routers

        all_nodes = list(range(n_routers))

        H_adj_mat, H_nodes = self.create_convex_subgraph(all_nodes)

        root_node = self.calc_most_central_node(H_adj_mat, all_nodes)

        if True:#self.verbose:
            print(f'found root node {root_node}')

        span_tree_adj_list = self.create_spanning_tree_given_root(all_nodes, root_node)
        if True:#self.verbose:
            print(f'found span tree {span_tree_adj_list}')

        # traverse tree
        node_ordering = self.order_nodes_in_tree(span_tree_adj_list, root_node)
        if self.verbose:
            # print(f'Node ordering : {node_ordering}')
            for i in range(n_routers):
                print(f'{i:02} : {node_ordering[i]}')

        # use escape paths
        # escape_channels, escape_turns, escape_paths = \
        #         self.calc_escape_paths_from_nodeset_and_root(all_nodes, root_node, span_tree_adj_list)
        # up_down_paths_flat = escape_paths

        if multiprocessed:
            all_up_down_paths_twod = self.calc_up_down_paths_multiprocessed(node_ordering, radix_load_balancing=True)
        else:
            all_up_down_paths_twod = self.calc_up_down_paths(node_ordering, radix_load_balancing=True)
        

        all_up_down_paths_flat = []
        up_down_paths_flat = []
        for src, dest_paths in enumerate(all_up_down_paths_twod):
            for dest, paths in enumerate(dest_paths):
                all_up_down_paths_flat += paths

                if first_index:
                    selected_path = paths[0]
                else:
                    selected_path = random.choice(paths)
                up_down_paths_flat.append(selected_path)

        self.all_path_list = all_up_down_paths_flat

        vn_map, allocated_path_list = self.vn_sort_up_down(up_down_paths_flat, node_ordering)

        # Done
        self.n_vns = 2
        self.allocated_path_list = deepcopy(allocated_path_list)
        self.chosen_paths_flat = deepcopy(up_down_paths_flat)

        self.vn_map = vn_map
        return vn_map, 2, up_down_paths_flat

    def vn_sort_up_down(self, paths, order):
        assert(self.n_routers != -1)
        n_routers = self.n_routers

        n_vcs = 2
        vn_map = [[ -1 for _ in range(n_routers)] for __ in range(n_routers)]
        allocated_path_list = [[] for _ in range(n_vcs)]

        # vc 0 : up only or up then down
        # vc 1 : down only or down then up
        # only need to check first movement
        #       up => vc 0, else vc 1
        for path in paths:
            src = path[0]
            dest = path[-1]

            # short paths to vc 0
            if len(path) == 1:
                vc = 0
                vn_map[src][dest] = vc
                allocated_path_list[vc].append(path)
                continue

            # first_movement
            n_a = path[0]
            n_b = path[1]
            # up
            if self.calc_is_up_or_down( n_a, n_b, order) == 'up':
                vc = 0
            else:
                vc = 1

            vn_map[src][dest] = vc
            allocated_path_list[vc].append(path)
    
        return vn_map, allocated_path_list

    def order_nodes_in_tree(self, span_tree_adj_list, root_node ):

        # unused?

        node_ordering = {}

        queue = deque()
        queue.append(root_node)
        node_num = 0

        while queue:
            cur_node = queue.popleft()
            node_ordering.update({ cur_node : node_num })
            node_num += 1

            try:
                descendants = span_tree_adj_list[cur_node]
            except:
                continue

            for descendant in descendants:
                queue.append(descendant)
        
        return node_ordering

    def calc_up_down_paths_multiprocessed(self, order, radix_load_balancing=True):
        assert(self.n_routers != -1)
        n_routers = self.n_routers

        # MUST USE THIS FOR MULTIPLE WRITERS
        manager = multiprocessing.Manager()
        return_queue = manager.Queue()


        processes = [multiprocessing.Process(target=self.calc_up_down_paths_get_lock, args=( return_queue, order, src, True) ) for src in range(n_routers)]

        for process in processes:
            process.start()

        for process in processes:
            process.join()

        print(f'All threads complete')

        all_up_down_paths_twod = [ [ ] for _ in range(n_routers)]
        results_dict = {}
        # collect
        while not return_queue.empty():
            this_result_dict = return_queue.get()
            results_dict.update(this_result_dict)
        # apply
        for src in range(n_routers):
            dest_paths = results_dict[src]
            all_up_down_paths_twod[src] = deepcopy(dest_paths)

        print(f'done with up_down paths')

        return all_up_down_paths_twod

    def calc_up_down_paths_get_lock(self, return_queue, order, this_src, radix_load_balancing):

        sem = self.sem
        sem.acquire()

        this_src_paths = self.calc_up_down_paths(order,one_src=this_src, radix_load_balancing=radix_load_balancing)

        # allpaths_lock.acquire()
        return_queue.put({this_src : this_src_paths})
        # allpaths_lock.release()

        sem.release()

        print(f'completed src {this_src}')

    def calc_up_down_paths(self, order, one_src=None, radix_load_balancing=True):
        assert(self.r_map is not None)
        assert(self.n_routers != -1)

        n_routers = self.n_routers
        r_map = self.r_map

        # all_paths[src][dest] = list of candidate shortest paths (each is a list of nodes)
        all_paths = [[[] for _ in range(n_routers)] for _ in range(n_routers)]

        # Which sources to compute for
        sources = range(n_routers) if one_src is None else [one_src]

        if self.verbose:
            print("Enumerating shortest valid Up*/Down* paths")

        for src in sources:
            if self.verbose:
                print(f'  Source {src}')

            # per_dest_paths[dest] = list of shortest paths src -> dest
            per_dest_paths = [[] for _ in range(n_routers)]
            # shortest_dist[node] = hop count of shortest valid path found so far from src
            shortest_dist = [self.INF] * n_routers

            q = deque()
            q.append([src])

            while q:
                path = q.popleft()
                last = path[-1]
                plen = len(path) - 1

                if plen > shortest_dist[last]:
                    continue

                # reached 'last' with a shortest (or equal-shortest) path
                if plen < shortest_dist[last]:
                    shortest_dist[last] = plen
                    per_dest_paths[last] = []
                per_dest_paths[last].append(path)

                for next in range(n_routers):
                    # only neighbors
                    if r_map[last][next] == 0:
                        continue

                    # avoid cycles (simple paths only)
                    if next in path:
                        continue

                    new_path = path + [next]

                    # must respect Up*/Down* rule
                    if not self.is_valid_up_down_path(new_path, order):
                        continue
 
                    q.append(new_path)

            for dest in range(n_routers):
                if src == dest:
                    # trivial self path
                    all_paths[src][dest] = [[src]]
                else:
                    all_paths[src][dest] = per_dest_paths[dest][:]

        if not radix_load_balancing:
            return all_paths

        if self.verbose:
            print("Selecting load-balanced paths (greedy, channel-load aware)")

        # channel_load[(u, v)] = current load on directed channel u -> v
        channel_load = {}

        # chosen[src][dest] = [one chosen path] or [] if unreachable
        chosen = [[[] for _ in range(n_routers)] for _ in range(n_routers)]

        # list of flows (src, dest) to route
        # route those with fewer candidate paths first (harder)
        # and among them, those with longer paths first (more edges to balance)
        flows = []
        for src in sources:
            for dest in range(n_routers):
                if src == dest:
                    chosen[src][dest] = [[src]]
                    continue

                cand = all_paths[src][dest]
                if not cand:
                    # no valid Up*/Down* path
                    continue

                num_cand = len(cand)
                max_len = max(len(p) for p in cand)
                flows.append((num_cand, -max_len, src, dest))

        flows.sort()

        for _, _, src, dest in flows:
            candidates = all_paths[src][dest]
            best_path = None
            best_score = None

            for p in candidates:
                local_max = 0
                total_load = 0

                for u, v in zip(p, p[1:]):
                    k = (u, v)
                    new_load = channel_load.get(k, 0) + 1
                    total_load += new_load
                    if new_load > local_max:
                        local_max = new_load

                new_global_max = local_max  # global max is monotonically non-decreasing
                score = (new_global_max, total_load)

                if best_score is None or score < best_score:
                    best_score = score
                    best_path = p

            # commit the chosen path
            if best_path is None:
                continue

            for u, v in zip(best_path, best_path[1:]):
                k = (u, v)
                channel_load[k] = channel_load.get(k, 0) + 1

            chosen[src][dest] = [best_path]

            if self.verbose:
                print(f'  Chose path {best_path} for {src}->{dest}, '
                    f'max local edge load now = '
                    f'{max(channel_load.values()) if channel_load else 0}')

        if one_src is not None:
            return chosen[one_src]

        return chosen

    def calc_is_up_or_down(self, a, b, order):

        if order[b] < order[a]:
            return 'up'
        else:
            return 'down'


    def is_valid_up_down_path(self, path, order):

        plen = len(path) - 1
        if plen <= 1:
            return True

        path_dirs = [self.calc_is_up_or_down(path[i], path[i+1], order)
                    for i in range(plen)]

        seen_down = False
        for d in path_dirs:
            if d == 'down':
                seen_down = True
            elif d == 'up' and seen_down:
                # illegal down -> up
                return False

        return True


    # Nue
    ####################################################################################################

    def nue(self, alg):

        matching_type = 'random'
        first_index = False
        if 'first_index' in alg:
            matching_type = 'first_index'
            alg.replace('_first_index','')
            first_index = True

        self.algorithm_two(matching_type=matching_type, first_index=first_index)

        # return as member vars

    def algorithm_two(self, matching_type='random', first_index=False):

        assert(self.r_map is not None)
        assert(self.n_routers != -1)
        assert(self.n_links != -1)
        assert(self.max_n_vcs != -1)

        n_routers = self.n_routers
        max_n_vcs = self.max_n_vcs

        # 1) partition N |-> {N_1, N_2, ..., N_k} into k subgroups
        #       use multilevel k-way partitioning algorithm*
        # 2) for each layer i = 1, 2, ..., k
        # 3) select nodes of partition i, N_i
        # 4) create convex subgraph, H_i
        # 5) identify central/root node, n_{r,i} of N^H_i
        # 6) create complete CDG, Dbar_i
        # 7) create escape paths Dbar^s_i given n_{r,i}
        # 8) for each node n in N_i
        # 9) create deadlock free paths P_{*,n} for all * in N (ie algorithm_one)
        # 10) update channel weights in Dbar_i given P_{*,n}
        # 8') loop to 8)
        # 2') loop to 2)

        # input(f'max_n_vcs = {max_n_vcs}')

        # 1)
        partitions = self.partition(max_n_vcs, matching_type=matching_type)

        if not first_index:
            for i in range(max_n_vcs):
                random.shuffle(partitions[i])

        if True:#self.verbose:
            print(f'found partition = {partitions}')

        # input('cont?')

        init_channel_weight = 1
        init_channel_weight = n_routers**2
        channel_weights = { (s,d) : init_channel_weight for s in range(n_routers) for d in range(n_routers)}
        chosen_paths_twod = [ [ None for _ in range(n_routers)] for __ in range(n_routers)]

        # will already know allocation
        vn_map = [ [ None for _ in range(n_routers)] for __ in range(n_routers)]
        # for part, dest_nodes in enumerate(partitions):
        #     for dest in dest_nodes:
        #         for src in range(n_routers):
        #             vn_mat[src][dest] = part

        # 2)
        for i in range(max_n_vcs):

            if True:#self.verbose:
                print(f'partition {i}')
            # 3)
            N_i = partitions[i]

            # 4)
            H_i_adj_mat, H_i_nodes = self.create_convex_subgraph(N_i)

            # 5)
            root_node = self.calc_most_central_node(H_i_adj_mat, N_i)

            if True:#self.verbose:
                print(f'found root node {root_node}')

            # 6)
            Dbar_i = OmniCDG(n_nodes = n_routers)
            # Dbar_i = Dbar_i.init_w_n_nodes(n_routers)

            # 7)
            escape_channels, escape_turns, escape_paths = \
                    self.calc_escape_paths_from_nodeset_and_root(N_i, root_node)

            escape_paths_dict = {}
            for path in escape_paths:
                src = path[0]
                dest = path[-1]
                escape_paths_dict.update({ (src,dest) : path })

            # instead of "used"
            # assume all added turns are "used"
            # Dbar_i.set_channels_state(escape_channels,'used')
            # Dbar_i.set_turns_state(escape_turns, 'used')
            Dbar_i.add_node_turns_common_vc(escape_turns)

            if self.verbose:
                print(f'found escape channels : {escape_channels}')
                print(f'found escape turns : {escape_turns}')
                print(f'found escape paths : {escape_paths}')

            if True:#self.verbose:
                print(f'found spanning tree / escape routes')

            blocked_turns_dict = {}

            for n in N_i:

                dl_free_paths_for_n, blocked_turns_dict = self.algorithm_one(n, Dbar_i, channel_weights, blocked_turns_dict, first_index=first_index)

                # fix with escapes
                for src in range(n_routers):
                    chosen_path = dl_free_paths_for_n[src]
                    if n not in chosen_path:
                        chosen_path = escape_paths_dict[(src,n)]
                        dl_free_paths_for_n[src] = chosen_path

                channel_weights = self.update_channel_weights(channel_weights, dl_free_paths_for_n)

                dl_free_paths_turns = self.turns_no_vcs_from_paths(dl_free_paths_for_n)
                Dbar_i.add_node_turns_common_vc(dl_free_paths_turns)
                # Dbar_i.update_channels_turns_from_used_paths( dl_free_paths_for_n)
    
                now_is_deadlocky = Dbar_i.cdg_has_cycle()

                # ERROR
                if now_is_deadlocky:
                    cdg_cycle = Dbar_i.networkx_get_cycle_as_nodes_no_vcs()
                    print(f'cdg_cycle={cdg_cycle}')
                    # input('deadlocky! ERROR')
                    print('deadlocky! ERROR')
                
                    quit()


                # input('good?')
                if self.verbose:
                    print(f'dl_free_paths_for_n (n={n}) = {dl_free_paths_for_n}')

                # wait is this dest,src or src,dest????
                for src in range(n_routers):
                    chosen_path = dl_free_paths_for_n[src]
                    chosen_paths_twod[src][n] = chosen_path
                    vn_map[src][n] = i
                
                max_weight = max([ v - init_channel_weight for v in channel_weights.values() ])
                print(f'Completed partition {i}, dest node {n}. max weight = {max_weight} ')
                
                if self.verbose:
                    print(f'channel_weights={[(k,v) for k,v in channel_weights.items() if v>0]}')
                    for path in dl_free_paths_for_n:
                        src = path[0]
                        dest = path[-1]
                        print(f'{src} -> {dest} : {path} on VC {i}')
                # input('cont?')

        print_at_end =  self.verbose

        chosen_paths_flat = []
        deadlock_free_paths = [ [] for vc in range(max_n_vcs)]
        for src, dest_paths in enumerate(chosen_paths_twod):
            # if print_at_end:#self.verbose:
                # print(f'{src}->')
            for dest, path in enumerate(dest_paths):
                vc = vn_map[src][dest]
                if print_at_end:#self.verbose:
                    print(f'\t{src}->{dest} : {path} on VC {vc}')
                chosen_paths_flat.append(path)
                deadlock_free_paths[vc].append(path)

        # input('good?')

        # Done
        self.n_vns = max_n_vcs
        self.allocated_path_list = deepcopy(deadlock_free_paths)
        self.chosen_paths_flat = deepcopy(chosen_paths_flat)

        # input(f'deadlock_free_paths={deadlock_free_paths}')

        # also set as member var
        # vn_map = self.set_vn_map_from_path_list(allocated_path_list=deadlock_free_paths)
        self.vn_map = vn_map
        return vn_map, max_n_vcs, chosen_paths_flat


    def algorithm_one(self, dest_node, Dbar, channel_weights, blocked_turns_dict, first_index=False):
        assert(self.r_map is not None)
        assert(self.n_routers != -1)
        n_routers = self.n_routers
        r_map = self.r_map

        # input('also broken')
        # quit()

        # 1)
        node_dists = [self.INF for _ in range(n_routers)]
        node_paths = [[] for _ in range(n_routers)]
        channel_dists = { (s,d) : self.INF for s in range(n_routers) for d in range(n_routers)}

        # 2)
        src_channels = []
        for src in range(n_routers):
            if r_map[src][dest_node] > 0:
                src_channels.append( (src,dest_node) )

        # init node dists
        node_dists[dest_node] = 0
        for src_channel in src_channels:
            src_tail = src_channel[0]
            node_dists[src_tail] = channel_weights[src_channel]
            channel_dists[src_channel] = channel_weights[src_channel]
            node_paths[src_tail] = [src_channel]

        # manually maintain heap
        queue = []
        for src_channel in src_channels:
            queue.append( src_channel )

        q_iter = 0
        while len(queue) > 0:
            cur_channel = self.queue_min_dist(queue, channel_dists)
            queue.remove(cur_channel)

            head_node = cur_channel[1]
            mid_node = cur_channel[0]

            if self.verbose:
                print('='*100)
                print(f'cur_channel={cur_channel}, mid_node={mid_node} new heap={queue}')

                if q_iter % 10 == 0:
                    print(f'q_iter={q_iter}. q_size={len(queue)}')

            # want to modify Dbar within loop so must use iterator
            possible_tails = list(range(n_routers))
            if not first_index:
                random.shuffle(possible_tails)
            for tail_node in possible_tails:
                predecessor_channel = (tail_node, mid_node )
                this_turn = (predecessor_channel, cur_channel)

                if tail_node == mid_node or tail_node==head_node:
                    continue
                if r_map[mid_node][tail_node] == 0:
                    continue

                # ignore blocked
                if this_turn in blocked_turns_dict:
                    if self.verbose:
                        print(f'this_turn {this_turn} is blocked')
                    continue
                


                if self.verbose:
                    print(f'this_turn = (predecessor_channel, cur_channel)={this_turn}')
                    print('-'*100)
                    print(f'\tturn={this_turn}, tail_node={tail_node}, mid_node={mid_node}. predecessor_channel={predecessor_channel}')
                    # print(f'\tchannel_dists[{cur_channel}] = {channel_dists[cur_channel]} ')
                    print(f'\tchannel_weights[{predecessor_channel}] = {channel_weights[predecessor_channel]} ')
                    print(f'\tnode_dists[{tail_node}] = {node_dists[tail_node]}')

                # input('cont?')

                # if it is NOT faster to go distance to cur_channel and weight of predecessor than directly to tail_node
                if channel_dists[cur_channel] + channel_weights[predecessor_channel] >= node_dists[tail_node]:
                    if self.verbose:
                        print(f'not interested in turn {this_turn}')
                    continue
                
                if self.verbose:
                    print(f'\twant to use this turn (if it doesnt cause dl)')
                

                # Dbar.set_turn_state(this_turn, 'used')

                # now_is_deadlocky = self.is_deadlocky_including_state(Dbar)
                this_turn_is_deadlocky = Dbar.probe_turn_for_deadlock_common_vc(this_turn)

                # possible turn if taking parents
                possible_turn_is_deadlocky = False
                has_possible_turn = False
                if len(node_paths[mid_node]) > 0:
                    # has a possible turn
                    last_turn_of_parent = node_paths[mid_node][0]
                    possible_turn = (predecessor_channel, last_turn_of_parent)
                    possible_turn_is_deadlocky = Dbar.probe_turn_for_deadlock_common_vc(possible_turn)
                    has_possible_turn = True


                # actually dont use that turn
                if this_turn_is_deadlocky:
                    # Dbar.set_turn_state(this_turn, 'blocked')
                    blocked_turns_dict.update({this_turn : True})

                    if self.verbose:
                        print(f'causese dl!')
                        print(f'prohibitng {this_turn}')

                    # input('deadlocky')

                # or this one
                if possible_turn_is_deadlocky:
                    # Dbar.set_turn_state(this_turn, 'blocked')
                    blocked_turns_dict.update({possible_turn : True})

                    if self.verbose:
                        print(f'causese dl!')
                        print(f'prohibitng {possible_turn}')

                    # input('deadlocky')


                # good idea. update values
                # else:
                if not this_turn_is_deadlocky and not possible_turn_is_deadlocky :

                    if self.verbose:
                        print(f'adding turn {this_turn} to Dbar')
                    Dbar.add_node_turn_common_vc(this_turn)

                    if has_possible_turn:
                        Dbar.add_node_turn_common_vc(possible_turn)

                    queue.append(predecessor_channel)

                    # replace previous parent(s)
                    old_path = node_paths[tail_node]
                    new_path = [predecessor_channel] + node_paths[mid_node]

                    # old_path_real = [old_path[0][0]] + [chan[1] for chan in old_path]
                    # new_path_real = [new_path[0][0]] + [chan[1] for chan in new_path]

                    if self.verbose:
                        print(f'new_path={new_path}')

                    # old dist calculation
                    # updated_dist = channel_dists[cur_channel] + channel_weights[predecessor_channel]
                    # new dist calculation
                    
                    max_weight = max( [channel_weights[c] for c in new_path] )
                    updated_dist = max_weight
                    updated_dist = max_weight + channel_dists[cur_channel]

                    channel_dists[predecessor_channel] = updated_dist
                    node_dists[tail_node] = updated_dist


                    # input(f'FLAG: replacing node_paths[{tail_node}] from {node_paths[tail_node]} to {new_path}')
                    node_paths[tail_node] = new_path

                    if self.verbose:
                        print(f'not deadlocky so updated all vars')

                    # update all the time
                    # for chan in old_path:
                    #     if channel_weights[chan] > 0:
                    #         channel_weights[chan] -= 1
                    # for chan in new_path:
                    #     channel_weights[chan] += 1


                if self.verbose:
                    print(f'')
                    print('after using turn, status:')
                    print(f"node dists    {[f'{src} : {dist}' for src,dist in enumerate(node_dists)]}")
                    print(f"node partens  {[f'{src} : {parents}' for src,parents in enumerate(node_paths)]}")
                    # print(f"channel dists {[f'{(s,d)} : {dist}' for (s,d),dist in channel_dists.items()]}")
                    print(f'queue = {queue}')

                # input('cont?')

            if self.verbose:
                print(f'done w/ Q iter {q_iter}')
            
            q_iter += 1

        # src_node never a switch
        if self.verbose:
            print(f'node_paths')
            for n, channels in enumerate(node_paths):
                print(f'\t{n} : {channels}')
        
        # input('cont?')

        # build paths
        calculated_paths = [ [] for _ in range(n_routers) ]
        for i in range(n_routers):
            if i == dest_node:
                calculated_paths[i].append(i)
                continue
            
            my_path = [i]
            if self.verbose:
                print('-'*100 + f'\ncalculating path for {i}. node_paths[{i}]={node_paths[i]}')

            for pair in node_paths[i]:
                my_path.append(pair[1])
            calculated_paths[i]=my_path
        
        # for i in range(n_routers):
        #     calculated_paths[i].reverse()

        if self.verbose:
            print(f'calculated_paths={calculated_paths}')

        # input('done')
        # quit()

        return calculated_paths, blocked_turns_dict

    def algorithm_one_broken(self, src_ie_dest_node, Dbar, channel_weights, blocked_channels_dict):
        assert(self.r_map is not None)
        assert(self.n_routers != -1)
        n_routers = self.n_routers
        r_map = self.r_map

        # 1)
        node_dists = [self.INF for _ in range(n_routers)]
        node_paths = [[] for _ in range(n_routers)]
        node_used_channels = [ None for _ in range(n_routers)]
        channel_dists = { (s,d) : self.INF for s in range(n_routers) for d in range(n_routers)}

        # src_ie_dest_node is always terminal
        init_src_ie_dest_channel = None
        for i in range(n_routers):
            if r_map[src_ie_dest_node][i] > 0:
                init_src_ie_dest_channel = (src_ie_dest_node, i)
                break

        # src_channels = []
        # for src in range(n_routers):
        #     if r_map[src][dest_node] > 0:
        #         src_channels.append( (src,dest_node) )


        node_dists[src_ie_dest_node] = 0
        channel_dists[init_src_ie_dest_channel] = 0


        queue = [init_src_ie_dest_channel]
        # for src_channel in src_channels:
        #     queue.append( src_channel )

        channel_Dbar_adj_dict = {}
        for i in range(n_routers):
            for j in range(n_routers):
                for k in range(n_routers):
                    # why loop to self?
                    if i == k:
                        continue
                    if r_map[i][j] > 0 and r_map[j][k] > 0:
                        c_a = (i,j)
                        c_b = (j,k)
                        try:
                            channel_Dbar_adj_dict[c_a].append(c_b)
                        except:
                            channel_Dbar_adj_dict.update({ c_a : [c_b] })
        if self.verbose:
            print(f'channel_Dbar_adj_dict={channel_Dbar_adj_dict}')

        q_iter = 0
        while len(queue) > 0:
            cur_channel = self.queue_min_dist(queue, channel_dists)
            queue.remove(cur_channel)

            head_node = cur_channel[1]
            mid_node = cur_channel[0]

            if self.verbose:
                print('='*100)
                print(f'cur_channel={cur_channel}, mid_node={mid_node} new heap={queue}')
            if q_iter % 10 == 0:
                print(f'q_iter={q_iter}. q_size={len(queue)}')

            neighbor_channels = channel_Dbar_adj_dict[cur_channel]
            if self.verbose:
                print(f'Possible neighbors: {neighbor_channels}')

            for next_channel in neighbor_channels:

                # ignore blocked
                if next_channel in blocked_channels_dict:
                    if self.verbose:
                        print(f'next_channel {next_channel} is blocked')
                    continue

                tail_node = next_channel[0]

                cur_channel_dist = channel_dists[cur_channel]
                next_channel_weight = channel_weights[next_channel]
                new_dist = cur_channel_dist + next_channel_weight
                old_dist = node_dists[tail_node]

                if self.verbose:
                    print(f'next_channel {next_channel} w/ tail {tail_node}')
                    print(f'old_dist={old_dist}')
                    print(f'new_dist={new_dist} = {cur_channel_dist} + {next_channel_weight}')

                if new_dist < old_dist:
                    if self.verbose:
                        print(f'TEST ADDING IT')
                    new_turn = (cur_channel, next_channel)
                    causes_dl = Dbar.probe_turn_for_deadlock_common_vc(new_turn)

                    if self.verbose:
                        print(f'causes_dl? {causes_dl}')

                    # yay
                    if not causes_dl:
                        if node_used_channels[tail_node] is not None:
                            try:
                                queue.remove(node_used_channels[tail_node])
                            except:
                                pass
                        queue.append(next_channel)
                        channel_dists[next_channel] = new_dist
                        node_dists[tail_node] = new_dist
                        node_used_channels[tail_node] = next_channel
                        Dbar.add_node_turn_common_vc(new_turn)
                        if self.verbose:
                            print(f'added {next_channel} to Q')
                            print(f'channel_dists[{next_channel}] = {new_dist}')
                            print(f'node_dists[{tail_node}] = {new_dist}')
                            print(f'node_used_channels[{tail_node}] = {next_channel}')
                    # nay
                    else:
                        blocked_channels_dict.update({ next_channel : True })

        return calculated_paths, blocked_channels_dict

    def update_channel_weights(self,channel_weights, dl_free_paths_for_n):

        for dest, path in enumerate(dl_free_paths_for_n):
            plen = len(path) - 1

            for n in range(plen):
                channel = (path[n],path[n+1])
                channel_weights[channel] += 1
        return channel_weights

    def queue_min_dist(self, queue, distances_from_src_node ):

        min_n = None
        min_dist = self.INF

        for e in queue:
            e_dist = distances_from_src_node[e]
            if e_dist < min_dist or min_n is None:
                min_n = e
                min_dist = e_dist
        return min_n

    def partition(self, kway, matching_type='random'):
        assert(self.n_routers != -1)
        assert(self.r_map is not None)
        assert(self.edge_list is not None)
        n_routers = self.n_routers
        r_map = self.r_map
        edge_list = self.edge_list


        # idea: partition nodes into k sets s.t. the edges between all sets is minimized

        # algorithm
        # 1) coarsen
        #   iteratively move group of nodes, Gi, into a single node, Gi+1,
        #       s.t. weight of Gi+1 = sum(n_Gi for all nodes n_Gi in Gi)
        #           and edges(Gi+1) = UNION(edges(n_Gi) for all n_Gi in Gi)
        #   accomplish by finding maximally matching sets 
        # 2) initial partition
        #   compute initial k-way partition on coarsened graph, Gm
        #       s.t. each partition is roughly |V0|/k (ie (1/k)th the original) vertex weight
        # 3) refine


        # 0) redefine r_map (adj_mat) as G=(V,E)
        # identity
        base_to_cur_node_map = {n:n for n in range(n_routers)}
        edge_weights = {e : 1 for e in edge_list}
        node_weights = {n : 1 for n in range(n_routers)}
        nodes = list(range(n_routers))
        edges = deepcopy(edge_list)

        # 1)
        # a) stop when coarsened is 0.8 size of original
        ratio_thresh = 0.8
        # b) coarsen into kway sets (removes need to find initial partition)
        # a & b) safety valve: exit if no difference between two iterations
        coarsen_technique = 'a'

        n_no_difference_iters = 0
        done_coarsening = False
        while not done_coarsening:
        

            matching = self.find_matching(nodes, edges, matching_type=matching_type)

            # matching = [(5, 4), (1, 0)]
            if self.verbose:
                print(f'Got matching {matching}')
            # do here cuz its quick. v becomes u
            for (u,v) in matching:
                base_to_cur_node_map[v] = base_to_cur_node_map[u]
                # update parents
                for key,val in base_to_cur_node_map.items():
                    if val == v:
                        base_to_cur_node_map[key] = base_to_cur_node_map[u]

            nodes, edges, node_weights, edge_weights = \
                    self.collapse_matches(nodes, edges, matching, node_weights, edge_weights)
            if self.verbose:
                print(f'After collapsing:')
                print(f'\tnodes = {nodes}, edges = {edges}')
                print(f'\tnode_weights = {node_weights}, edge_weights = {edge_weights}')

            new_size = len(nodes)
            size_difference = n_routers - new_size


            if coarsen_technique == 'a':
                size_ratio = new_size / n_routers
                if size_ratio <= ratio_thresh:
                    done_coarsening = True
                    if self.verbose:
                        print(f'Done because ratio <= 0.8')
            elif coarsen_technique == 'b':
                if new_size <= kway:
                    done_coarsening = True
                    if self.verbose:
                        print(f'Done because kway # nodes')
            
            # safety
            if size_difference == 0:
                n_no_difference_iters += 1
                if n_no_difference_iters >= 2:
                    done_coarsening = True
                    if self.verbose:
                        print(f'Done because no difference')
            else:
                n_no_difference_iters = 0


        if self.verbose:
            print(f'Done coarsening :')
            print(f'\tnodes ({len(nodes)}) = {nodes}, edges = {edges}')
            print(f'\tnode_weights = {node_weights}, edge_weights = {edge_weights}')
            print(f'\ttransformation')
            for n in range(n_routers):
                print(f'\t\t{n} -> {base_to_cur_node_map[n]}')

        # 2)
        # TODO implement as in paper
        # deviation : just greedily apply to buckets
        # imho this is sufficient

        goal_weight = n_routers / kway
        # in terms of coarse nodes
        initial_partition = [[] for part in range(kway)]
        initial_partition_node_weights = [0 for part in range(kway)]
        coarse_node_to_partition_map = {}
        for coarse_node in nodes:
            coarse_node_weight = node_weights[coarse_node]

            found_match = False
            for dest_part in range(kway):
                potential_weight = initial_partition_node_weights[dest_part] + coarse_node_weight
                if potential_weight <= goal_weight:
                    initial_partition[dest_part].append(coarse_node)
                    coarse_node_to_partition_map.update({ coarse_node : dest_part })
                    initial_partition_node_weights[dest_part] = potential_weight
                    found_match = True
                    break

            # add to end
            if not found_match:
                dest_part = kway - 1
                # lightest
                min_weight = self.INF
                for part in range(kway):
                    potential_weight = initial_partition_node_weights[part] + coarse_node_weight
                    if potential_weight < min_weight:
                        dest_part = part
                        min_weight = potential_weight
                initial_partition[dest_part].append(coarse_node)
                coarse_node_to_partition_map.update({ coarse_node : dest_part })
                initial_partition_node_weights[dest_part] = potential_weight

        if self.verbose:
            print(f'coarse_node_to_partition_map={coarse_node_to_partition_map}')

        base_node_to_initial_partition_map = {}
        reverse_base_node_to_initial_partition_map = {p : [] for p in range(kway)}

        for base_node in range(n_routers):
            for coarse_node in nodes:
                # found match
                if base_to_cur_node_map[base_node] == coarse_node:
                    part = coarse_node_to_partition_map[coarse_node]
                    base_node_to_initial_partition_map.update({ base_node : part })
                    reverse_base_node_to_initial_partition_map[part].append( base_node )
        
        if self.verbose:
            print(f'base_node_to_initial_partition_map={base_node_to_initial_partition_map}')
            print(f'reverse_base_node_to_initial_partition_map={reverse_base_node_to_initial_partition_map}')

        # recalc edge weights
        initial_partition_edge_weights = {}
        for i in range(n_routers):
            for j in range(n_routers):
                if r_map[i][j] == 0:
                    continue
                orig_edge = (i,j)
                # new_edge = (base_node_to_initial_partition_map[i],base_node_to_initial_partition_map[j])
                i_part = base_node_to_initial_partition_map[i]
                j_part = base_node_to_initial_partition_map[j]
                new_edge = (i_part, j_part)
                try:
                    initial_partition_edge_weights[new_edge] += r_map[i][j]
                except:
                    initial_partition_edge_weights.update({ new_edge : r_map[i][j] })

        if self.verbose:
            print(f'Done inital partition :')
            print(f'\tinitial_partition ({len(initial_partition)}) = {initial_partition}')
            print(f'\tinitial_partition_node_weights = {initial_partition_node_weights}')
            print(f'\tinitial_partition_edge_eights = {initial_partition_edge_weights}')


        # 3) rebalance
        #   looking to reduce edge weight between partitions
        #   while maintaining node weight balance amongst partitions
        # a) greedy

        # from this point, work in original nodes
        partition = [[] for p in range(kway)]
        partition_node_weights = {p : 0 for p in range(kway)}
        base_node_to_partition_map = {}
        for part, base_nodes in reverse_base_node_to_initial_partition_map.items():
            partition[part] += base_nodes
            partition_node_weights[part] = initial_partition_node_weights[part]
            for n in base_nodes:
                base_node_to_partition_map.update({ n : part })

        if self.verbose:
            print(f'Beginning refinement :')
            print(f'\tpartition ({len(partition)}) = {partition}')
            print(f'\tpartition_node_weights = {partition_node_weights}')
            print(f'\tbase_node_to_partition_map = {base_node_to_partition_map}')


        # goal_weight set above
        C = 1.1
        D = 0.9

        no_changes = False
        while not no_changes:
            vertices = list(range(n_routers))

            random.shuffle(vertices)

            no_changes = True
            for v in vertices:
                v_part = base_node_to_partition_map[v]

                in_deg = 0
                for dest in range(n_routers):
                    if r_map[v][dest] == 0:
                        continue
                    if base_node_to_partition_map[dest] == v_part:
                        in_deg += 1

                max_ext_deg = -1
                max_ext_deg_part = None

                for other_part in range(kway):
                    if v_part == other_part:
                        continue
                    ext_deg = 0
                    for dest in range(n_routers):
                        if r_map[v][dest] == 0:
                            continue
                        # other
                        if base_node_to_partition_map[dest] == other_part:
                            ext_deg += 1
                    if ext_deg > max_ext_deg or max_ext_deg_part is None:
                        max_ext_deg = ext_deg
                        max_ext_deg_part = other_part

                # whether to move v
                # weight[v] = 1
                # from v_part to max_ext_deg_part
                # 0) must satisfy balancing condition :
                #       weight[max_ext_deg_part] + weight[v] <= C*goal_weight
                #       weight[v_part] - weight[v] >= D*goal_weight
                # 1) move if max_ext_deg > in_deg
                # or 2) max_ext_deg == in_deg and weight[v_part] - weight[max_ext_deg_part] > weight[v]

                if self.verbose:
                    print(f'node {v} in part {v_part} has in_deg {in_deg} and ext_deg {max_ext_deg} w/ part {max_ext_deg_part}')
                    print(f'\tpartition_node_weights={partition_node_weights}')

                to_move = False
                if max_ext_deg > in_deg:
                    to_move = True
                    if self.verbose:
                        print(f'improves degree')
                elif max_ext_deg == in_deg and \
                    (partition_node_weights[v_part] - partition_node_weights[max_ext_deg_part] > 1):
                    to_move = True
                    if self.verbose:
                        print(f'improves weight balance')
                
                if (partition_node_weights[max_ext_deg_part] + 1 > C*goal_weight) or \
                    (partition_node_weights[v_part] - 1 < D*goal_weight):
                    to_move = False
                    if self.verbose:
                        print(f'violates weight balance')

                if to_move:
                    base_node_to_partition_map[v] = max_ext_deg_part
                    partition[v_part].remove(v)
                    partition[max_ext_deg_part].append(v)
                    # orig_node so weight of one
                    partition_node_weights[v_part] -= 1
                    partition_node_weights[max_ext_deg_part] += 1
                    no_changes = False

                    if self.verbose:
                        print(f'Moving {v} from {v_part} to {max_ext_deg_part}')

            if self.verbose:
                print(f'done w an iter')
                print(f'partition={partition}')
                print(f'base_node_to_partition_map={base_node_to_partition_map}')
                print(f'\tpartition_node_weights={partition_node_weights}')
                print('')

        # recalc edge weights
        partition_edge_weights = {}
        for i in range(n_routers):
            for j in range(n_routers):
                if r_map[i][j] == 0:
                    continue
                orig_edge = (i,j)
                # new_edge = (base_node_to_partition_map[i],base_node_to_partition_map[j])
                i_part = base_node_to_partition_map[i]
                j_part = base_node_to_partition_map[j]
                new_edge = (i_part, j_part)
                try:
                    partition_edge_weights[new_edge] += 1
                except:
                    partition_edge_weights.update({ new_edge : 1 })

        if self.verbose:
            print(f'Done partition refinement :')
            print(f'\tpartition ({len(partition)}) = {partition}')
            print(f'\tpartition_node_weights = {partition_node_weights}')
            print(f'\tpartition_edge_eights = {partition_edge_weights}')
            for k,v in base_node_to_partition_map.items():
                print(f'{k} -> {v}')

        return partition

    def find_matching(self, nodes, edges, matching_type='random'):

        matching = None

        if matching_type == 'random':
            matching = self.find_matching_random_or_not(nodes, edges)
        elif matching_type == 'first_index':
            matching = self.find_matching_random_or_not(nodes, edges, use_shuffle=False)
        else:
            print(f'Matching {matching_type} unknown. Exiting...')
            quit()

        return matching

    def find_matching_random_or_not(self, nodes, edges, use_shuffle=True):
        node_status_dict = {n : 'unmatched' for n in nodes}

        matching_edges = []
        edges_of_vertex = {}
        for edge in edges:
            u, v = edge
            try:
                edges_of_vertex[u].append(v)
            except:
                edges_of_vertex.update({ u : [v] })

        # visit in random order
        random_nodes = deepcopy(nodes)
        if use_shuffle:
            random.shuffle(random_nodes)

        for u in random_nodes:
            if node_status_dict[u] == 'matched':
                continue
            # unmatched u
            if self.verbose:
                print(f'unmatched u = {u}')
            random_neighbors = edges_of_vertex[u]
            if use_shuffle:
                random.shuffle(random_neighbors)
            for v in random_neighbors:
                if node_status_dict[v] == 'unmatched':
                    matching_edges.append( (u,v) )
                    node_status_dict[u] = 'matched'
                    node_status_dict[v] = 'matched'

                    if self.verbose:
                        print(f'Matching {u} to {v}')
                    break

        return matching_edges

    def collapse_matches(self, nodes, edges, matching, node_weights, edge_weights):

        if self.verbose:
            print(f'Collapse matches\n' + '-'*50 )
            print(f'nodes={nodes}')

        # simple
        new_nodes = []
        excluded_nodes = []
        new_node_weights = {}
        # in to/from loops
        new_edges = []
        new_edge_weights = {}

        new_node_weights = {n : node_weights[n] for n in nodes}

        edges_from_vertex = {}
        edges_to_vertex = {}
        for edge in edges:
            u, v = edge
            try:
                edges_from_vertex[u].append(v)
            except:
                edges_from_vertex.update({ u : [v] })
            try:
                edges_to_vertex[v].append(u)
            except:
                edges_to_vertex.update({ v : [u] })

        matching_dict = {n : n for n in nodes}
        matching_dict.update( {v : u for (u,v) in matching})
        reversed_matching_dict = {u : v for (u,v) in matching}

        if self.verbose:
            print(f'matching_dict={matching_dict}')
            print(f'reversed_matching_dict={reversed_matching_dict}')

        # collapse u and v into u' (ie keep u label)
        for (u,v) in matching:
            # new_nodes.append(u)
            excluded_nodes.append(v)
            total_weight = node_weights[u] + node_weights[v]
            new_node_weights.update({ u : total_weight })
            del(new_node_weights[v])

            # do in one shot
            # ie think about future matches in for loop

            # from
            # u (all new values)
            for u_dest in edges_from_vertex[u]:
                if u_dest == v:
                    continue

                new_u_dest = matching_dict[u_dest]
                new_edge = (u,new_u_dest)
                old_edge = (u, u_dest)
                if new_edge not in new_edges:
                    new_edges.append(new_edge)

            # v (SOME new values)
            for v_dest in edges_from_vertex[v]:
                if v_dest == u:
                    continue

                new_v_dest = matching_dict[v_dest]

                new_edge = (u,new_v_dest)
                old_edge = (v,v_dest)
                if new_edge not in new_edges:
                    new_edges.append(new_edge)

            # to
            # u (all new values)
            for u_src in edges_to_vertex[u]:
                if u_src == v:
                    continue

                new_u_src = matching_dict[u_src]
                new_edge = (new_u_src,u)
                old_edge = (u_src, u)
                if new_edge not in new_edges:
                    new_edges.append(new_edge)

            # v (SOME new values)
            for v_src in edges_to_vertex[v]:
                if v_src == u:
                    continue

                new_v_src = matching_dict[v_src]

                new_edge = (new_v_src,u)
                old_edge = (v_src,v)
                if new_edge not in new_edges:
                    new_edges.append(new_edge)

        for new_edge in new_edges:
            i,j = new_edge
            # u -> u
            try:
                old_edge = new_edge
                prev_weight = edge_weights[old_edge]
            except:
                prev_weight = 0
                old_edge = None
            
            # print(f'Adding weight {prev_weight} to {new_edge} because of {old_edge}')
            new_edge_weights.update({new_edge : prev_weight })

            # u -> v
            try:
                old_edge = (i, reversed_matching_dict[j])
                prev_weight = edge_weights[old_edge]
            except:
                prev_weight = 0
                old_edge = None
            
            # print(f'Adding weight {prev_weight} to {new_edge} because of {old_edge}')
            new_edge_weights[new_edge] += prev_weight

            # v -> v
            try:
                old_edge = (reversed_matching_dict[i], reversed_matching_dict[j])
                prev_weight = edge_weights[old_edge]
            except:
                prev_weight = 0
                old_edge = None

            # print(f'Adding weight {prev_weight} to {new_edge} because of {old_edge}')
            new_edge_weights[new_edge] += prev_weight

            # v -> u
            try:
                old_edge = (reversed_matching_dict[i], j)
                prev_weight = edge_weights[old_edge]
            except:
                prev_weight = 0
                old_edge = None

            # print(f'Adding weight {prev_weight} to {new_edge} because of {old_edge}')
            new_edge_weights[new_edge] += prev_weight
        
        new_nodes = [n for n in nodes if n not in excluded_nodes]

        return new_nodes, new_edges, new_node_weights, new_edge_weights

    # All
    ####################################################################################################

    def create_spanning_tree_given_root(self, relevant_nodes, root_node):
        """
            Simple BFS
        """

        assert(self.r_map is not None)
        assert(self.n_routers != -1)
        r_map = self.r_map
        n_routers = self.n_routers

        all_nodes = list(range(n_routers))

        visited = []

        queue = deque()

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

            if self.verbose:
                print(f'q_iter {q_iter} : cur_node={cur_node} and Q={queue}')
            q_iter += 1

            # for next_node in relevant_nodes:
            for next_node in all_nodes:
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
                        print(f'cur_node {cur_node} visits {next_node}')

        if self.verbose:
            print(f'relevant_nodes={relevant_nodes}')
            print(f'=> spanning tree {span_tree_adj_list}')

        return span_tree_adj_list

    def calc_escape_paths_from_nodeset_and_root(self, relevant_nodes, root_node, span_tree_adj_list=None):
        assert(self.n_routers != -1)
        n_routers = self.n_routers

        if span_tree_adj_list is None:
            span_tree_adj_list = self.create_spanning_tree_given_root(relevant_nodes, root_node)

        # create span_tree_nwx_G from span_tree_adj_list
        span_tree_nwx_G = nx.DiGraph()
        for src, dest_list in span_tree_adj_list.items():
            for dest in dest_list:
                span_tree_nwx_G.add_edge(src,dest)
                span_tree_nwx_G.add_edge(dest,src)

        shortest_paths_twod = [[ None for _ in range(n_routers)] for __ in range(n_routers)]
        # find shortest paths in span_tree_adj_mat
        for src in range(n_routers):
            for dest in range(n_routers):
                if src == dest:
                    shortest_paths_twod[src][src] = [src]
                    continue
                short_path_generator = nx.all_shortest_paths(span_tree_nwx_G,src,dest)
                short_path_list = list()
                short_path_list += short_path_generator
                # always one path in tree
                shortest_paths_twod[src][dest] = short_path_list[0]

        escape_channels = []
        escape_turns = []
        escape_paths = []
        for src, dest_paths in enumerate(shortest_paths_twod):
            if self.verbose:
                print(f'{src}->')
            for dest, path in enumerate(dest_paths):
                if dest not in relevant_nodes:
                    continue

                if self.verbose:
                    print(f'\t->{dest} : {path}')

                escape_paths.append(path)

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

        return escape_channels, escape_turns, escape_paths

    def nwx_all_shortest_paths_given_nodeset(self, nodeset):
        assert(self.all_shortest_paths_twod is not None)
        assert(self.n_routers != -1)

        all_shortest_paths_twod = self.all_shortest_paths_twod
        n_routers = self.n_routers

        # n_nodes = len(nodeset)

        nodeset_shortest_paths = [[ None for _ in range(n_routers) ] for __ in range(n_routers)]

        for src in nodeset:
            for dest in nodeset:
                nodeset_shortest_paths[src][dest] = all_shortest_paths_twod[src][dest]
        
        return nodeset_shortest_paths

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

        nodeset_shortest_paths = self.nwx_all_shortest_paths_given_nodeset(relevant_nodes)

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

    def calc_most_central_node(self, this_map, relevant_nodes):

        nodeset_shortest_paths = self.nwx_all_shortest_paths_given_nodeset(relevant_nodes)

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

    def turns_no_vcs_from_paths(self, path_list):
        added_turns_dict = {}
        turn_group_no_vcs = []
        for path in path_list:
            plen = len(path) - 1
            if plen <= 1:
                continue

            n_turns = plen - 1
            for i in range(n_turns):
                c_a = (path[i],path[i+1])
                c_b = (path[i+1],path[i+2])
                turn = (c_a,c_b)
                if turn in added_turns_dict:
                    continue

                turn_group_no_vcs.append(turn)
                # val doesnt matter
                added_turns_dict.update({turn : None})
        return turn_group_no_vcs

    def set_vn_map_from_path_list(self, allocated_path_list=None):
        assert(self.n_routers != -1)

        if allocated_path_list is None:
            assert(self.allocated_path_list is not None)
            allocated_path_list = self.allocated_path_list

        nr = self.n_routers
        vm = [[-1 for _ in range(nr)]for __ in range(nr)]

        for vn, this_path_list in enumerate(allocated_path_list):
            # print(f'vn {vn}')
            for path in this_path_list:
                src = path[0]
                dest = path[-1]
                # dont RE-assign
                # print(f'\t{src}->{dest} via {path} vm[{src}][{dest}]={vm[src][dest]}')
                assert(vm[src][dest] == -1)
                vm[src][dest] = vn

        # force basic set
        for i in range(nr):
            vm[i][i] = 0

        self.vn_map = deepcopy(vm)

        return vm

    def calc_path_set_path_weight(self, sp_set):
        return len(sp_set)

    def calc_path_set_hop_weight(self,sp_set):
        w = 0
        for path in sp_set:
            w += self.calc_path_hop_weight(path)
        return w

    def calc_path_hop_weight(self,path):
        w = len(path) - 1
        return w

    # use this
    def calc_path_set_weight(self, sp_set, metric):

        if metric == 'hops':
            return self.calc_path_set_hop_weight(sp_set)
        # elif metric == 'paths':
        return self.calc_path_set_path_weight(sp_set)

    def load_balance(self, metric, min_n_vns=None):
        assert(self.n_vns != -1)
        assert(self.vn_map is not None)
        assert(self.n_routers != -1)
        assert(self.allocated_path_list is not None)
        allocated_path_list = self.allocated_path_list
        n_routers = self.n_routers
        n_vns = self.n_vns

        if min_n_vns is None:
            min_n_vns = n_vns

        # exit early
        if metric == 'none':
            return

        old_path_list = deepcopy(allocated_path_list)

        # completely local var
        new_allocated_path_list = [[] for _ in range(min_n_vns)]
        for vn, sp_set in enumerate(old_path_list):
            for p in sp_set:
                new_allocated_path_list[vn].append(p)

        paths_per_vn = [len(s) for s in old_path_list]
        all_weights = [self.calc_path_set_weight(pset, metric)\
                        for pset in old_path_list]

        while len(paths_per_vn) < min_n_vns:
            paths_per_vn.append(0)
            all_weights.append(0)

        avg_weight = 0
        for w in all_weights:
            avg_weight += w
        avg_weight = avg_weight / min_n_vns

        if True:#self.verbose:
            print('-'*72)
            print(f'Before load balancing of metric {metric}')
            print(f'\tcurrent #VCs {n_vns}, goal # VCs {min_n_vns} ')
            print(f'\tweights={all_weights}, avg = {avg_weight} ({metric}/vn)')
            print(f'\tfor {paths_per_vn} paths per vn')
        
        # donate from 0, 1, 2, 3...
        for donor_vn in range(min_n_vns):

            donor_path_list = []
            # for all paths in current path list
            for path in new_allocated_path_list[donor_vn]:
                donor_path_list.append(path.copy())

            # increment to not retry known immobile paths
            donor_path_idx = 0

            cur_weight = self.calc_path_set_weight(
                            new_allocated_path_list[donor_vn],
                            metric)


            # while this path set is heavy
            while(cur_weight > avg_weight \
                    and donor_path_idx < len(donor_path_list) ):

                if True:#self.verbose:
                    if donor_path_idx % 100 == 0:
                        print(f'{donor_path_idx}th iteration and ' +\
                                f'donor ({donor_vn}) ' + \
                                f'weight={cur_weight}' + \
                                f' (all = {[self.calc_path_set_weight(pset, metric) for pset in new_allocated_path_list]})')

                donor_path = donor_path_list[donor_path_idx]


                # target of this path. default to self
                target_vn = donor_vn

                # search others
                # look at end of list
                #   since it is most empty and thus least likely to have collision
                #   this works as this alg is exhaustive so might as well take free lunch when we can
                for other_vn in reversed(range(min_n_vns)):
                    # not self
                    if donor_vn == other_vn:
                        continue

                    # skip if potential target is heavy (do not overburden)
                    target_weight = self.calc_path_set_weight(new_allocated_path_list[other_vn], metric) 
                    if target_weight > avg_weight:
                        if self.verbose:
                            print(f'now recipient {other_vn} is heavy ({target_weight})')
                        continue


                    # if we add that donor path, is this new set DL free?

                    # completely new local obj
                    new_cdg = OmniCDG(n_nodes=n_routers)
                    new_cdg.add_paths_common_vc(new_allocated_path_list[other_vn])

                    # add new path
                    causes_dl = new_cdg.probe_path_for_deadlock_common_vc(donor_path)

                    if not causes_dl:
                        # print(f'moving path {donor_path} from {donor_vc}->{other_vc}')

                        target_vn = other_vn

                        # found. might as well exit early?
                        break

                        # else, loop to next possible vn

                # recalc weight
                cur_weight = self.calc_path_set_weight(
                            new_allocated_path_list[donor_vn],
                            metric)

                # found alternative
                if target_vn != donor_vn:
                    # donor_path_list.remove(donor_path)
                    new_allocated_path_list[donor_vn].remove(donor_path)
                    new_allocated_path_list[target_vn].append(donor_path)
                    if self.verbose:
                        print(f'after removal/addition,')
                        print(f'this_path_list[donor_vc]={this_path_list[donor_vc]}')
                        print(f'this_path_list[target_vc]={this_path_list[target_vc]}')
                else:
                    if self.verbose:
                        print(f'cannot move {donor_path} from {donor_vn}')
                    pass

                donor_path_idx += 1


            # done with a vn
            print(f'load balanced vn {donor_vn}')
            # # last check
            # my_cdg_list = []
            # for pl in new_allocated_path_list:
            #     my_cdg_list.append(self.create_a_cdg_from_paths(pl))
            # self.assert_deadlock_free(my_cdg_list)


        paths_per_vn = [len(s) for s in new_allocated_path_list]

        all_weights = [self.calc_path_set_weight(pset, metric) for pset in new_allocated_path_list]

        avg_weight = 0
        for w in all_weights:
            avg_weight += w
        avg_weight = avg_weight / min_n_vns

        if True:#self.verbose:
            print('-'*72)
            print(f'After load balancing')
            print(f'\tweights={all_weights} avg={avg_weight}')
            print(f'\tfor {paths_per_vn} paths and per node per vn')

        # input('continue?')

        # set obj vars
        # self.allocated_path_list = new_allocated_path_list.copy(deep=True)
        self.allocated_path_list = []
        for new_path_list in new_allocated_path_list:
            self.allocated_path_list.append(new_path_list.copy())


        self.n_vns = len(self.allocated_path_list)
        self.set_vn_map_from_path_list()

        return

    def final_check_on_vn_map(self):
        assert(self.allocated_path_list is not None)
        assert(self.vn_map is not None)
        assert(self.n_routers != -1)
        assert(self.n_vns != -1)
        allocated_path_list = self.allocated_path_list
        vn_map = self.vn_map
        n_routers = self.n_routers
        n_vns = self.n_vns

        print('-'*80)
        print(f'Final check :')
        print(f'\tvn_map ({len(vn_map)}x{len(vn_map[0])})')
        print(f'\tallocated_path_list ({len(allocated_path_list)}) :')
        total_n_paths = 0
        for vn, path_list in enumerate(allocated_path_list):
            n_paths = len(path_list)
            print(f'\t\tVN {vn} : # paths = {n_paths}')
            total_n_paths += n_paths
        print(f'\t=> total # paths = {total_n_paths}')

        omnicdg = OmniCDG(n_nodes=n_routers, n_vcs=n_vns)
        for vn, path_list in enumerate(allocated_path_list):
            # check agreement between allocation and vn_map
            for path in path_list:
                src = path[0]
                dest = path[-1]
                # forall src==dest, vn_map[src][dest]=1
                # but allocation undefined/specified 
                if src == dest:
                    continue

                assert(vn_map[src][dest] == vn)
            
            print(f'Validated agreement between vn_map and path allocation for vn {vn}')
            
            # check deadlock freedom
            omnicdg.add_paths_common_vc(path_list, common_vc=vn)
        
        has_dl = omnicdg.cdg_has_cycle()

        assert(has_dl == False)

        print(f'Validated deadlock freedom across all VCs')

    # Driver
    ####################################################################################################

    def assign_static_vcs(self, alg):

        if 'lash' in alg:
            self.lash(alg)
        elif 'dfsssp' in alg:
            self.dfsssp(alg)
        elif 'nue' in alg:
            self.nue(alg)
        elif 'up_down' in alg:
            self.up_down(alg)
        else:
            print(f'Alg {alg} unknown. Exiting...')
            quit()        

        # return as member vars


# Running as main
####################################################################################################

def calc_cload_from_cpl(flat_cpl):

    load_dict = {}
    n_routers = 0

    for path in flat_cpl:
        plen = len(path) - 1
        for i in range(plen):
            link = (path[i], path[i+1])
            try:
                load_dict[link] += 1
            except:
                load_dict.update({link : 1})

            n_routers = max(n_routers,path[i])
            n_routers = max(n_routers,path[i+1])

    n_routers += 1
    load_mat = [[0 for _ in range(n_routers)] for __ in range(n_routers)]
    for path in flat_cpl:
        plen = len(path) - 1
        for i in range(plen):
            load_mat[path[i]][path[i+1]] += 1

    return load_dict, load_mat

def calc_max_cload_from_cpl(flat_cpl):

    load_dict = {}

    for path in flat_cpl:
        plen = len(path) - 1
        for i in range(plen):
            link = (path[i], path[i+1])
            try:
                load_dict[link] += 1
            except:
                load_dict.update({link : 1})

    max_cload_link = max(load_dict, key=load_dict.get)
    max_cload = load_dict[max_cload_link]
    return max_cload

def route_and_alloc_vcs_map_file(rmap_file_path,
                        alg_type,
                        max_n_vcs=None,
                        given_out_name=None,
                        verbose=False,
                        async_job_name=None):
    
    # manual reset
    if max_n_vcs is None:
        max_n_vcs = 2
    my_VNAllocator = VNAllocator()
    my_VNAllocator.verbose = verbose
    my_VNAllocator.setup_for_r_map_based( rmap_file_path, max_n_vcs)

    # non-iterative. Will always complete

    # vn_map, n_vns set as member vars
    my_VNAllocator.assign_static_vcs(alg_type)


    # also set as member var
    # vn_map = my_VNAllocator.load_balance(load_balancing_type, min_n_vns=min_n_vns)
    # no load balance in Nue
    vn_map = my_VNAllocator.vn_map
    n_vns = my_VNAllocator.max_n_vcs

    # final check
    my_VNAllocator.final_check_on_vn_map()

    # for Nue, also must output path list
    out_name = given_out_name
    if given_out_name is None:
        rmap_name = my_VNAllocator.rmap_name
        assert(rmap_name is not None)
        out_name = f'{rmap_name}_{alg_type}_{n_vns}vns'

    if 'up_down' in alg_type:

        all_paths_flat = my_VNAllocator.all_path_list
        # uses member var. but pass as arg anyway
        my_VNAllocator.output_allpathslist(out_name, all_paths_list=all_paths_flat)

    chosen_paths_flat = my_VNAllocator.chosen_paths_flat
    # uses member var. but pass as arg anyway
    my_VNAllocator.output_pathlist(out_name, path_list=chosen_paths_flat)

    # uses member var. but pass vn_map as arg anyway
    my_VNAllocator.output_vn_map_to_file(out_name, vn_map=vn_map)

    # v.output_vn_to_file(outpath)

    # vn_file_path = outpath

    del(my_VNAllocator)

    if async_job_name is None:
        return


    # calc max cload
    max_cload = calc_max_cload_from_cpl(chosen_paths_flat)
    max_thru = 1.0 / float(max_cload)
    _heatdict, heatmap = calc_cload_from_cpl(chosen_paths_flat)

    print(f'{chosen_paths_flat}')
    print(f'{max_thru}')

    # write out as json
    out_dict = {
        'async_job_name':async_job_name,
        'n_vcs' : n_vns,
        'vc_mat' : vn_map,
        'max_thru' : max_thru,
        'routing_paths' : chosen_paths_flat,
        'channel_heatmat' : heatmap
    }
    out_json_path = os.path.join(ASYNC_JOB_RESULTS_JSON_DIR, f'{async_job_name}.json')
    with open(out_json_path, "w+") as of:
        json.dump(out_dict, of)

    # keep legacy running text result log
    out_line = f'{async_job_name} | {n_vns} | {vn_map} | {max_thru} | {chosen_paths_flat} | {heatmap}'
    out_line += '\n'
    with open(ASYNC_JOB_RESULT_FILE,'a') as of:
        of.write(out_line)

def alloc_vns_path_list(route_file_path,
                        alg_type,
                        load_balancing_type,
                        given_out_name=None,
                        min_n_vns=None,
                        max_n_vns=None,
                        max_retries=None,
                        verbose=False,
                        async_job_name=None):

    my_VNAllocator = VNAllocator()
    my_VNAllocator.verbose = verbose
    my_VNAllocator.setup( route_file_path)
    print(f'completed setup')

    successful = False
    iters = 0

    while(not successful):

        print('-'*50)

        # wont hurt zeroth iter
        my_VNAllocator.reset()

        # vn_map, n_vns set as member vars
        my_VNAllocator.assign_static_vcs(alg_type)

        n_vns = my_VNAllocator.n_vns

        print(f'Completed iter {iters} : # vns = {n_vns}')

        # if stringent on # vns
        successful = True

        if max_n_vns is not None:
            if n_vns > max_n_vns:
                successful = False
                print(f'Failure to reach under {max_n_vns} VNs. Got {n_vns}')

        if max_retries is not None:
            if iters > max_retries:
                successful = True
                print(f'Reached max iterations {iters}')

        iters += 1

    # set as member var
    my_VNAllocator.load_balance(load_balancing_type, min_n_vns=min_n_vns)

    vn_map = my_VNAllocator.vn_map
    n_vns = my_VNAllocator.n_vns


    # final check
    my_VNAllocator.final_check_on_vn_map()

    out_name = given_out_name
    if given_out_name is None:
        cpl_name = my_VNAllocator.cpl_name
        assert(cpl_name is not None)
        out_name = f'{cpl_name}_{alg_type}_{load_balancing_type}_{n_vns}vns'

    # uses member var. but pass vn_map as arg anyway
    my_VNAllocator.output_vn_map_to_file(out_name, vn_map=vn_map)

    # v.output_vn_to_file(outpath)

    # vn_file_path = outpath

    print(f'iter {iters} successful? {successful}')

    del(my_VNAllocator)

    if async_job_name is None:
        return

    # write out as json
    out_dict = {
        'async_job_name':async_job_name,
        'n_vcs' : n_vns,
        'vc_mat' : vn_map
    }
    out_json_path = os.path.join(ASYNC_JOB_RESULTS_JSON_DIR, f'{async_job_name}.json')
    with open(out_json_path, "w+") as of:
        json.dump(out_dict, of)

    # keep legacy running text result log
    out_line = f'{async_job_name} | {n_vns} | {vn_map}'
    out_line += '\n'
    with open(ASYNC_JOB_RESULT_FILE,'a') as of:
        of.write(out_line)


def main():

    parser = argparse.ArgumentParser(description='Verify topology values')
    parser.add_argument('--map_file','-g',type=str,help='Topology as adjacency matrix (.map)')
    parser.add_argument('--path_list','-cpl',type=str,help='Routing function (.paths format) to make deadlock free.')


    parser.add_argument('--min_n_vns',type=int,help='Minimum # of virtual networks/channels')
    parser.add_argument('--max_n_vns',type=int,help='Minimum # of virtual networks/channels')
    parser.add_argument('--max_retries',type=int,default=15)

    parser.add_argument('--alg',type=str,help='Deadlock avoidance (and/or routing) algorithm.',\
            choices=['lash','dfsssp','nue','up_down'],default='lash')
    parser.add_argument('--lb_type',type=str,default='none',choices=['none','paths','hops'],help='Metric for load balancing between VCs.')

    parser.add_argument('--out_vn_name',type=str)
    parser.add_argument('--async_job',type=str,help='Being run as async job of name given. Should output result to running file and JSON.')

    parser.add_argument('--verbose','-v',action='store_true',help='Debug prints')


    args = parser.parse_args()

    path_list_path = args.path_list
    r_map_path = args.map_file
    alg = args.alg
    lb_type = args.lb_type
    max_n_vns = args.max_n_vns

    verbose = args.verbose
    async_job = args.async_job

    route_plus_deadlock = False
    if 'nue' in alg.lower() or 'up_down' in alg.lower():
        route_plus_deadlock = True
        if max_n_vns is None:
            max_n_vns = 2

    if r_map_path is not None and route_plus_deadlock:
        route_and_alloc_vcs_map_file(r_map_path,
                            alg,
                            max_n_vcs=max_n_vns,
                            given_out_name=args.out_vn_name,
                            verbose=verbose,
                            async_job_name=async_job)
    elif path_list_path is not None and not route_plus_deadlock:
        alloc_vns_path_list(path_list_path,
                            alg,
                            lb_type,
                            given_out_name=args.out_vn_name,
                            min_n_vns=args.min_n_vns,
                            max_n_vns=max_n_vns,
                            max_retries=args.max_retries,
                            verbose=verbose,
                            async_job_name=async_job)

    else:
        print('No path list or router map provided. Exiting...')
        quit(-1)

if __name__ == '__main__':
    main()
