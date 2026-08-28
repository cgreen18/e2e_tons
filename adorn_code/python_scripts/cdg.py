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

Description:
    Modular, class version of channel dependency graph(s)

"""


import matplotlib.pyplot as plt
import networkx as nx

# class CDGTurnStatus(Enum):
    
#     UNUSED = 'unused'
#     USED = 'used'
#     BLOCKED = 'blocked'

#     def __str__(self):
#         return f'{self.value}'

class CDG:

    # basic
    verbose = False
    INF = 999

    # for nue
    valid_statuses = ['unused','blocked','routed']
    
    # general
    valid_types = ['complete','routed']

    def __init__(self):
        
        # basic scalars
        self.n_topo_nodes = -1
        self.n_channels = -1
        self.n_turns = -1
        self.n_possible_channels = -1
        self.n_possible_turns = -1

        # basic forms
        # nodes as labeled in given map/turns
        self.routers = None
        # list of channels as tuples
        self.channels = None
        # list of turns as tuples of tuples
        self.turns = None

        # channel as tuple |-> channel ID
        self.channel_to_node_map = None
        # channel ID |-> channel as tuple
        self.node_to_channel_map = None

        # for convienence
        # adj list. channel as tuple : dependent channels as tuples in list
        #   c1 : [(nb,nc),(nb,nd),...]
        self.cdg_adj_dict = None
        # adj mat. channel ID rows and columns
        #   turn = ((na,nb),(nb,nc)) = (c1,c2) <=> mat[c1][c2] = 1
        self.cdg_adj_mat = None
        # as networkx digraph
        self.cdg_as_nwx_G = None

        # for nue. (na,nb) : status as str
        self.channel_status_dict = None
        # for nue. ((na,nb),(nb,nc)) : status as str
        self.turn_status_dict = None

        # for nue. 'complete' 'routed'
        self.type = None

        # for vc allocation
        self.related_vc = -1

    # inits
    ####################################################################################################

    def init_complete_cdg_from_map(self, this_map, related_vc=-1, init_status=False):

        # define possiblities
        self.n_topo_nodes, self.n_possible_channels, self.n_possible_turns = \
            self.init_topo_nodes_and_possibilities_from_map_(this_map)

        # define actualalities
        self.routers, self.channels, self.turns = self.init_channels_and_turns_from_map_(this_map)

        # define translation to IDs and back
        self.channel_to_node_map, self.node_to_channel_map = \
            self.init_translation_dicts_(self.channels, self.n_topo_nodes)

        # create cdg as adj_dict and adj_mat from channels and turns
        self.cdg_adj_dict, self.cdg_adj_mat = \
            self.create_cdgs_from_channels_turns_(self.turns, self.channel_to_node_map, self.n_possible_channels)

        self.cdg_as_nwx_G = self.create_nwx_G_from_cdg_adj_mat_(self.cdg_adj_mat)

        # for complete (ie from map)
        self.type = 'complete'

        # optionals
        if init_status:
            self.channel_status_dict, self.turn_status_dict = \
                self.init_statuses_(self.channels, self.turns)

        self.related_vc = related_vc

    def init_cdg_from_turns(self, these_turns, related_vc=-1, init_status=False, n_routers=-1):

        # define possiblities
        self.n_topo_nodes, self.n_possible_channels, self.n_possible_turns = \
            self.init_topo_nodes_and_possibilities_from_turns_(these_turns)
        
        if n_routers != -1:
            self.n_topo_nodes = n_routers
            self.n_possible_channels = n_routers**2
            self.n_possible_turns = n_routers**3

        # define actualalities
        self.routers, self.channels, self.turns = self.init_channels_and_turns_from_turns_(these_turns)

        # define translation to IDs and back
        self.channel_to_node_map, self.node_to_channel_map = \
            self.init_translation_dicts_(self.channels, self.n_topo_nodes)

        # create cdg as adj_dict and adj_mat from channels and turns
        self.cdg_adj_dict, self.cdg_adj_mat = \
            self.create_cdgs_from_channels_turns_(self.turns, self.channel_to_node_map, self.n_possible_channels)

        self.cdg_as_nwx_G = self.create_nwx_G_from_cdg_adj_mat_(self.cdg_adj_mat)

        # for routed (ie from turns)
        self.type = 'routed'

        # optionals
        if init_status:
            self.channel_status_dict, self.turn_status_dict = \
                self.init_statuses_(self.channels, self.turns)

        self.related_vc = related_vc

    # class/local-based methods    
    ####################################################################################################

    @classmethod
    def init_topo_nodes_and_possibilities_from_map_(cls, this_map):
        n_routers = len(this_map)
        n_possible_channels = n_routers**2
        n_possible_turns = n_routers**3

        return n_routers, n_possible_channels, n_possible_turns

    @classmethod
    def init_topo_nodes_and_possibilities_from_turns_(cls, these_turns):


        max_router_val = -1
        min_router_val = cls.INF

        for turn in these_turns:

            channel_a = turn[0]
            channel_b = turn[1]

            node_a_1 = channel_a[0]
            node_a_2 = channel_a[1]
            node_b_2 = channel_b[1]

            min_router_val = min(min_router_val, node_a_1)
            min_router_val = min(min_router_val, node_a_2)
            min_router_val = min(min_router_val, node_b_2)

            max_router_val = max(max_router_val, node_a_1)
            max_router_val = max(max_router_val, node_a_2)
            max_router_val = max(max_router_val, node_b_2)

        # n_routers = max_router_val - min_router_val + 1
        n_routers = max_router_val + 1

        n_possible_channels = n_routers**2
        n_possible_turns = n_routers**3

        return n_routers, n_possible_channels, n_possible_turns

    @classmethod
    def init_channels_and_turns_from_map_(cls, this_map):

        n_routers = len(this_map)

        routers = []
        channels = []
        turns = []



        for i in range(n_routers):
            routers.append(i)
            for j in range(n_routers):
                if this_map[i][j] == 0:
                    continue
                if i==j:
                    continue
                #else valid channel
                channels.append((i,j))

                # if i==0 and j==1:
                #     cls.verbose = True
                # else:
                #     cls.verbose = False

                for k in range(n_routers):

                    # if cls.verbose:
                    #     print(f'considering k={k} w/ r_map[{j}][{k}]={this_map[j][k]}')

                    if this_map[j][k] == 0:
                        continue

                    if k==i or k==j:
                        continue
                    # else valid turn
                    turns.append( ( (i,j), (j,k) ) )

                    # if cls.verbose:
                    #     print(f'appending {( (i,j), (j,k) )}')

        
        return routers, channels, turns

    @classmethod
    def init_channels_and_turns_from_turns_(cls, these_turns):

        routers = []
        channels = []
        turns = []

        for turn in these_turns:

            channel_a = turn[0]
            channel_b = turn[1]

            node_a_1, node_a_2 = channel_a
            node_b_1, node_b_2 = channel_b
            if node_a_1 not in routers:
                routers.append(node_a_1)
            if node_a_2 not in routers:
                routers.append(node_a_2)
            if node_b_2 not in routers:
                routers.append(node_b_2)

            if channel_a not in channels:
                channels.append(channel_a)
            if channel_b not in channels:
                channels.append(channel_b)

            if turn not in turns:
                turns.append(turn)
        
        return routers, channels, turns

    @classmethod
    def init_translation_dicts_(cls, channels, n_routers):

        channel_to_node_map = {}
        node_to_channel_map = {}
        for i in range(n_routers):
            for j in range(n_routers):
                
                ij_channel_id = n_routers*i + j
                ij_channel = (i,j)

                if ij_channel in channels:

                    channel_to_node_map.update({ ij_channel : ij_channel_id })
                    node_to_channel_map.update({ ij_channel_id : ij_channel })
        
        return channel_to_node_map, node_to_channel_map        

    @classmethod
    def init_statuses_(cls, channels, turns):

        c_status = {}
        for c in channels:
            c_status.update({ c: 'unused'})

        t_status = {}
        for t in turns:
            t_status.update({ t: 'unused'})
        
        return c_status, t_status

    @classmethod
    def create_cdgs_from_channels_turns_(cls, turns, channel_to_node_map, n_possible_channels):

        # edge:[dependent edges] = (i,j):[(j,k1),(j,k2), ...]
        depends_adj_dict = {}
        depends_adj_mat = [ [ 0 for _ in range(n_possible_channels) ] for __ in range(n_possible_channels)]

        for channel_a, channel_b in turns:

            # update adj_dict
            try:
                depends_adj_dict[channel_a].append(channel_b)
            except:
                depends_adj_dict.update( {channel_a : [channel_b]} )

            # update adj_mat
            channel_a_id = channel_to_node_map[channel_a]
            channel_b_id = channel_to_node_map[channel_b]
            depends_adj_mat[channel_a_id][channel_b_id] = 1

        return depends_adj_dict, depends_adj_mat
    
    @classmethod
    def create_nwx_G_from_cdg_adj_mat_(cls, cdg_adj_mat):
        G = nx.DiGraph()

        for src, depens in enumerate(cdg_adj_mat):
            for dest, value in enumerate(depens):
                if value > 0:
                    G.add_edge(src, dest)

        return G

    def update_translation_and_nwx_G_adding_turn(self, turn):
        assert(self.n_topo_nodes != -1)
        n_routers = self.n_topo_nodes

        channel_a, channel_b = turn

        # update translatiosn
        try:
            _ = self.channel_to_node_map[channel_a]
        except:
            i,j = channel_a
            channel_id = n_routers*i + j
            self.channel_to_node_map.update({channel_a : channel_id})
            self.node_to_channel_map.update({channel_id : channel_a})
        try:
            _ = self.channel_to_node_map[channel_b]
        except:
            i,j = channel_b
            channel_id = n_routers*i + j
            self.channel_to_node_map.update({channel_b : channel_id})
            self.node_to_channel_map.update({channel_id : channel_b})
        
        # update cdg adj list and mat
        try:
            adj_channels = self.cdg_adj_dict[channel_a]
            if channel_b not in adj_channels:
                self.cdg_adj_dict[channel_a].append(channel_b)
        except:
            self.cdg_adj_dict.update({ channel_a : [channel_b] })

        channel_a_id = self.channel_to_node_map[channel_a]
        channel_b_id = self.channel_to_node_map[channel_b]
        self.cdg_adj_mat[channel_a_id][channel_b_id] = 1

        if not self.cdg_as_nwx_G.has_edge(channel_a_id,channel_b_id):
            self.cdg_as_nwx_G.add_edge(channel_a_id,channel_b_id)

    def update_translation_and_nwx_G_removing_turn(self, turn):
        assert(self.n_topo_nodes != -1)
        n_routers = self.n_topo_nodes

        channel_a, channel_b = turn
        
        # update cdg adj list and mat
        self.cdg_adj_dict[channel_a].remove(channel_b)

        channel_a_id = self.channel_to_node_map[channel_a]
        channel_b_id = self.channel_to_node_map[channel_b]
        self.cdg_adj_mat[channel_a_id][channel_b_id] = 0

        if self.cdg_as_nwx_G.has_edge(channel_a_id,channel_b_id):
            self.cdg_as_nwx_G.remove_edge(channel_a_id,channel_b_id)


    # getters and setters
    ####################################################################################################

    def add_turn(self, turn):
        if turn not in self.turns:
            self.turns.append(turn)
        c1, c2 = turn
        if c1 not in self.channels:
            self.channels.append(c1)
        if c2 not in self.channels:
            self.channels.append(c2)
        self.update_translation_and_nwx_G_adding_turn(turn)

    def remove_turn(self, turn):
        if turn in self.turns:
            self.turns.remove(turn)
        self.update_translation_and_nwx_G_removing_turn(turn)

    def get_n_turns(self):
        assert(self.turns is not None)
        return len(self.turns)

    def get_turn(self, turn_num):
        assert(self.turns is not None)
        assert(self.get_n_turns() > turn_num)
        return self.turns[turn_num]

    def get_turns(self):
        assert(self.turns is not None)
        return self.turns

    def get_turn_status(self, turn):
        assert(self.turn_status_dict is not None)
        assert(self.turn_status_dict[turn] is not None)
        return self.turn_status_dict[turn]

    def get_nodes_of_channel(self, channel):
        assert(self.node_to_channel_map is not None)
        assert(channel in self.channels)

        print(f'self.node_to_channel_map={self.node_to_channel_map}')

        # return (self.node_to_channel_map[channel[0]] , self.node_to_channel_map[channel[1]])
        return self.node_to_channel_map[channel]

    def set_turn_state(self, turn, new_state):
        assert(self.turn_status_dict is not None)

        # assert(turn in self.turns)
        if turn not in self.turns:
            self.turns.append(turn)
        
        self.turn_status_dict[turn] = new_state
    
    def set_turns_state(self, turns, new_state):
        for turn in turns:
            self.set_turn_state(turn, new_state)

    def set_channel_state(self, channel, new_state):
        assert(self.channel_status_dict is not None)
        assert(channel in self.channels)
        self.channel_status_dict[channel] = new_state

    def set_channels_state(self, channels, new_state):
        for channel in channels:
            self.set_channel_state(channel, new_state)

    def get_used_channels(self):
        assert(self.channel_status_dict is not None)
        assert(self.channels is not None)
        channel_status_dict = self.channel_status_dict
        channels = self.channels

        used_channels = []
        for channel in channels:
            if channel_status_dict[channel] == 'used':
                used_channels.append(channel)
        
        return used_channels
    
    def update_channels_turns_from_used_paths(self, used_paths):

        for dest, path in enumerate(used_paths):
            plen = len(path) - 1
            if plen == 1:
                continue
            for n in range(plen):
                channel = (path[n],path[n+1])
                self.set_channel_state(channel,'used')
            
            for n in range(plen-1):
                turn = ((path[n],path[n+1]), (path[n+1],path[n+2]))
                self.set_turn_state(turn,'used')

    # cycles
    ####################################################################################################

    def networkx_get_cycle(self):
        assert(self.cdg_as_nwx_G is not None)
        cdg_as_nwx_G = self.cdg_as_nwx_G

        cycle = []
        try:
            cycle = list(nx.find_cycle(cdg_as_nwx_G))

            if self.verbose:
                print(f'networkx found cycle: {cycle}')
            return cycle

        except:
            return []

    def networkx_get_all_cycles_as_generator(self):
        assert(self.cdg_as_nwx_G is not None)
        cdg_as_nwx_G = self.cdg_as_nwx_G

        print(f'Finding all cycles')

        cycle = []
        try:
            cycles_generator = nx.recursive_simple_cycles(cdg_as_nwx_G)
            # cycles_generator = nx.simple_cycles(cdg_as_nwx_G)

            # cycles = list(cycles_generator)

            # input(f'networkx found cycles: {cycles}')

            print(f'\tcomplete')

            # if self.verbose:
            #     print(f'networkx found cycles: {cycles}')
            return cycles_generator

        except:
            return []


    # def is_deadlocky_including_state(self):
    #     assert(self.turns is not None)
    #     assert(self.turn_status_dict is not None)
    #     turns = self.turns
    #     turn_status_dict = self.turn_status_dict

    #     for turn in turns:
    #         turn_status = turn_status_dict[turn]
    #         if turn_status != 'used':
    #             continue

    # visualization
    ####################################################################################################

    def visualize_cdg(self, viz_type=None):
        assert(self.cdg_as_nwx_G is not None)
        cdg_as_nwx_G = self.cdg_as_nwx_G

        options = {
            "font_size": 12,
            "node_size": 300,
            "with_labels":True,
            "node_color": "blue",
            "edgecolors": "black",
            "linewidths": 1,
            "width": 1
        }



        # nx.draw(cdg_as_nwx_G)
        if viz_type == 'circular':
            nx.draw_circular(cdg_as_nwx_G, **options)
        else:
            nx.draw(cdg_as_nwx_G, **options)
        # nx.draw_networkx_labels(cdg_as_nwx_G)

        # plt.savefig("cdg.png")

        ax = plt.gca()
        ax.margins(0.04)
        plt.axis("off")

        plt.show()

    def print_cycle_as_map_nodes(self, cycle):

        start_turn = cycle[0]
        start_channel = start_turn[0]
        node_1, node_2 = self.node_to_channel_map[start_channel]
        print(f'{node_1}->{node_2} ')
        for turn in cycle:
            channel_b = turn[1]
            node_b1, node_b2 = self.node_to_channel_map[channel_b]
            print(f'{node_b1}->{node_b2} ')

        print('')
    
    def translate_cycle_as_turns_of_nodes(self,cycle):

        turns_as_nodes = []
        for turn in cycle:
            channel_a = turn[0]
            channel_b = turn[1]

            node_a1, node_a2 = self.node_to_channel_map[channel_a]
            node_b1, node_b2 = self.node_to_channel_map[channel_b]

            turn_as_nodes = ((node_a1,node_a2),(node_b1,node_b2))
            turns_as_nodes.append(turn_as_nodes)
        return turns_as_nodes

    def translate_cycle_as_turns_of_nodes_v2(self,cycle):

        n_channels = len(cycle)
        turns_as_nodes = []
        for i in range(n_channels - 1):
            channel_a = cycle[i]
            channel_b = cycle[i+1]

            node_a1, node_a2 = self.node_to_channel_map[channel_a]
            node_b1, node_b2 = self.node_to_channel_map[channel_b]

            turn_as_nodes = ((node_a1,node_a2),(node_b1,node_b2))
            turns_as_nodes.append(turn_as_nodes)


        return turns_as_nodes

# main (for testing)
####################################################################################################


def ingest_map(path_name):

    assert_binary = True

    file_name = path_name.split('/')[-1]

    print(f'ingesting map filename = {file_name}')

    r_map = []

    with open(path_name, 'r') as in_file:

        for row in in_file:
            r_conns = row.split(" ")
            if '\n' in r_conns:
                r_conns.remove('\n')

            try:
                r_conns = [int(elem) for elem in r_conns]
            except:
                r_conns = [int(float(elem)) for elem in r_conns]
            r_map.append(r_conns)

    n_routers = len(r_map)

    for i in range(n_routers):

        for j in range(n_routers):

            if assert_binary:
                if r_map[i][j] > 0:
                    r_map[i][j] = 1

        r_map[i][i] = 0

    return r_map


def main():

    map_file_name = 'files/map_files/kite_large.map'

    r_map = ingest_map(map_file_name)

    my_CDG = CDG()
    # my_CDG.init_complete_cdg_from_map(r_map)
    temp_turns = [ ((0,1),(1,3)) , ( (1,3),(3,4)) , ((3,4),(4,2)) , ( (4,2),(2,0)) , ( (2,0),(0,1))  ]
    my_CDG.init_cdg_from_turns(temp_turns)

    print(f'channel_to_node_map = {my_CDG.channel_to_node_map}')
    print(f'node_to_channel_map = {my_CDG.node_to_channel_map}')

    cdg_cycles = my_CDG.networkx_get_cycle()
    print(f'cycle = {cdg_cycles}')
    my_CDG.print_cycle_as_map_nodes(cdg_cycles)

    turns_as_nodes = my_CDG.translate_cycle_as_turns_of_nodes(cdg_cycles)
    print(f'turns_as_nodes={turns_as_nodes}')


    # my_CDG.visualize_cdg()

if __name__ == '__main__':
    main()