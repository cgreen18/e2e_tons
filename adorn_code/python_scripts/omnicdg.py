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

class OmniCDG:

    # basic
    verbose = False
    INF = 999

    # init functions
    ####################################################################################################

    # mainly just to write out obj vars and defaults to detect uninitialization
    def _empty_init(self):

        # basic scalars
        # -------------
        # in r_map
        self.n_topo_nodes = -1
        # represented in cdg
        self.n_nodes = -1
        self.n_channels = -1
        self.n_turns = -1
        self.n_possible_channels = -1
        self.n_possible_turns = -1

        # basic forms
        # -----------
        # nodes as labeled in given map/turns
        self.nodes = None
        # dict of nodes for quick "in" operator
        self.node_dict = None
        # list of channels as tuples of r_map nodes and VC
        self.channels = None
        # list of channels as channel IDs
        self.channel_ids = None
        # dict of channel IDs for quick "in" operator
        self.channel_id_dict = None
        # list of turns as tuples of tuples of r_map nodes and VC
        self.turns = None
        # list of turns as tuples of channel IDs
        self.turn_ids = None
        # dict of turn IDs for quick "in" operator
        self.turn_id_dict = None

        # translations
        # ------------
        # channel as tuple of r_map nodes and VC |-> channel ID
        self._channel_to_node_map = None
        # channel ID |-> channel as tuple of r_map nodes and VC
        self._node_to_channel_map = None

        # graphs
        # ------
        # as networkx digraph
        self.cdg_as_nwx_G = None

        # extras
        # ------
        self.viz_options = None

    def __init__(self, n_nodes=None, n_vcs=1):

        self._empty_init()

        if n_nodes is not None:
            self.init_w_n_nodes(n_nodes, n_vcs=n_vcs)

    def init_w_n_nodes(self, n_nodes, n_vcs=1):

        # required for now
        assert(n_nodes is not None)


        # basic scalars
        # -------------
        # in r_map
        self.n_topo_nodes = n_nodes
        # leave empty
        # represented in cdg
        self.n_nodes = 0
        self.n_channels = 0
        self.n_turns = 0
        # init here for simplicity
        self.n_possible_channels = (n_nodes**2)*n_vcs
        self.n_possible_turns = self.n_possible_channels*self.n_possible_channels


        # basic forms
        # -----------
        # nodes as labeled in given map/turns
        self.nodes = []
        # dict of nodes for quick "in" operator
        self.node_dict = {}
        # list of channels as tuples of r_map nodes and VC
        self.channels = []
        # list of channels as channel IDs
        self.channel_ids = []
        # dict of channel IDs for quick "in" operator
        self.channel_id_dict = {}
        # list of turns as tuples of tuples of r_map nodes and VC
        self.turns = []
        # list of turns as tuples of channel IDs
        self.turn_ids = []
        # dict of turn IDs for quick "in" operator
        self.turn_id_dict = {}

        # translations
        # ------------
        print(f'OmniCDG : Before creating translation dics')
        # DO NOT MODIFY AFTER SETUP
        # define translation to IDs and back
        self._channel_to_node_map, self._node_to_channel_map = \
            self.init_translation_dicts_(n_nodes, n_vcs)

        print(f'OmniCDG : After creating translation dics')
        print(f'OmniCDG : Before creating adj mat')

        # graphs
        # ------

        self.cdg_as_nwx_G = nx.DiGraph()

    @classmethod
    def init_translation_dicts_(cls, n_nodes, n_vcs):

        channel_to_node_map = {}
        node_to_channel_map = {}
        for vc_k in range(n_vcs):
            for n_i in range(n_nodes):
                for n_j in range(n_nodes):
                    channel_id = n_i*n_nodes + n_j + vc_k*(n_nodes**2)
                    channel = (n_i,n_j,vc_k)


                    channel_to_node_map.update({ channel : channel_id })
                    node_to_channel_map.update({ channel_id : channel })

        return channel_to_node_map, node_to_channel_map
    
    # general class methods
    ####################################################################################################

    @classmethod
    def networkx_get_cycle_(cls,cdg_as_nwx_G):

        cycle = []
        try:
            cycle = nx.find_cycle(cdg_as_nwx_G)
            if cls.verbose:
                print(f'networkx found cycle: {cycle}')

        # expected when no cycle
        except nx.exception.NetworkXNoCycle:
            pass

        except Exception as e:
            input('ERROR: Unexpected exception')
        
        return cycle

    @classmethod
    def networkx_get_cycle_w_srcs_(cls,cdg_as_nwx_G, src_nodes):

        cycle = []
        try:
            cycle = nx.find_cycle(cdg_as_nwx_G, source=src_nodes)
            if cls.verbose:
                print(f'networkx found cycle: {cycle}')

        # expected when no cycle
        except nx.exception.NetworkXNoCycle:
            pass

        except Exception as e:
            input('ERROR: Unexpected exception')
        
        return cycle

    @classmethod
    def visualize_cdg_(cls, cdg_as_nwx_G, viz_type='regular', viz_options={}, node_labels=None, save_to_file=None, no_show=False):

        try:
            # nx.draw(cdg_as_nwx_G)
            if viz_type == 'circular':
                nx.draw_circular(cdg_as_nwx_G, labels=node_labels, **viz_options)
            elif viz_type == 'planar':
                nx.draw_planar(cdg_as_nwx_G, labels=node_labels, **viz_options)
            # regular
            else:
                nx.draw(cdg_as_nwx_G, labels=node_labels, **viz_options)
        except:
            nx.draw(cdg_as_nwx_G, labels=node_labels, **viz_options)


        ax = plt.gca()
        ax.margins(0.04)
        plt.axis("off")

        if save_to_file is not None:
            plt.savefig(save_to_file)

        if not no_show:
            plt.show()

    # getters and setters
    ####################################################################################################

    def set_verbose(self):
        self.verbose = True

    # translation(s)
    ####################################################################################################

    def translate_turn(self, turn):
        assert(isinstance(turn,tuple))
        assert(len(turn) == 2)

        # pull apart
        c_a, c_b = turn
        c_a_id = self.translate_channel(c_a)
        c_b_id = self.translate_channel(c_b)
        turn_ids = (c_a_id, c_b_id)

        return turn_ids

    def translate_channel(self, channel):
        assert(isinstance(channel,tuple))
        assert(self._channel_to_node_map is not None)

        channel_id = -1
        try:
            channel_id = self._channel_to_node_map[channel]
        except:
            input(f'ERROR: channel {channel} does not have translation')

        return channel_id

    def translate_channel_id(self, channel_id):
        assert(isinstance(channel_id,int))
        assert(self._node_to_channel_map is not None)

        channel = (-1,-1,-1)
        try:
            channel = self._node_to_channel_map[channel_id]
        except:
            input(f'ERROR: channel ID {channel_id} does not have translation')

        return channel

    # add turns/channels/nodes
    ####################################################################################################

    def add_edge_to_adj_mat_and_nx_cdg(self,edge):
        assert(isinstance(edge,tuple))
        assert(len(edge) == 2)
        # edge is tuple of (two) channel IDs
        assert(isinstance(edge[0],int))
        assert(isinstance(edge[1],int))

        c_a_id, c_b_id = edge

        # removed adj_mat for space
        # self.add_edge_to_adj_mat(c_a_id, c_b_id)
        self.add_edge_to_cdg(c_a_id, c_b_id)

        if self.verbose:
            print(f'Added edge {edge} to nx CDG and adj mat')

    def add_edge_to_cdg(self, c_a_id, c_b_id):
        assert(self.cdg_as_nwx_G is not None)

        self.cdg_as_nwx_G.add_edge(c_a_id, c_b_id)

    def add_node_turns_common_vc(self, node_turns, vc=0):

        for nt in node_turns:
            # pull apart and add vc to channels
            c_a, c_b = nt
            n_a_1, n_a_2 = c_a
            n_b_1, n_b_2 = c_b
            c_a_w_vc = (n_a_1, n_a_2, vc)
            c_b_w_vc = (n_b_1, n_b_2, vc)
            turn_w_vcs = (c_a_w_vc, c_b_w_vc)
            self.add_turn(turn_w_vcs)
    
        if self.verbose:
            # assume all
            print(f'Added {len(node_turns)} turns')

    def add_turns(self, turns):

        for turn in turns:
            self.add_turn(turn)

    def add_turn(self,turn):
        assert(isinstance(turn,tuple))
        assert(len(turn) == 2)
        # channels are channels
        assert(isinstance(turn[0],tuple))
        assert(isinstance(turn[1],tuple))
        assert(self.turn_id_dict is not None)
        turn_id_dict = self.turn_id_dict

        turn_id = self.translate_turn(turn)

        turn_in_cdg = False
        try:
            _ = turn_id_dict[turn_id]
            turn_in_cdg = True
        except:
            pass

        # return early
        if turn_in_cdg:
            return

        # else actually add
        self.turns.append(turn)
        self.turn_ids.append(turn_id)
        # val doesnt matter
        self.turn_id_dict.update({turn_id : True})
        self.n_turns += 1

        # MOST IMPORTANT
        # in terms of channel IDs
        new_edge = turn_id
        self.add_edge_to_adj_mat_and_nx_cdg(new_edge)

        # add constituent channels
        # these handle adding nodes within
        # pull apart
        c_a, c_b = turn
        self.add_node_vc_tuple_channel(c_a)
        self.add_node_vc_tuple_channel(c_b)

    # alternative call. preferred is as tuple
    def add_node_vc_channel(self,src,dest,vc):
        # vars are tuples
        self.add_node_vc_tuple_channel((src,dest,vc))

    def add_node_vc_tuple_channel(self,channel):
        assert(isinstance(channel,tuple))
        assert(self.channel_id_dict is not None)
        channel_id_dict = self.channel_id_dict

        # all should have translation
        channel_id = self.translate_channel(channel)
        
        channel_in_cdg = False
        try:
            _ = channel_id_dict[channel_id]
            channel_in_cdg = True
        except:
            pass

        # return early
        if channel_in_cdg:
            return
        
        # else actually add
        self.channels.append(channel)
        self.channel_ids.append(channel_id)
        # val doesnt matter
        self.channel_id_dict.update({channel : True})
        self.n_channels += 1

        # add constituent nodes
        n_a = channel[0]
        self.add_node(n_a)
        n_b = channel[1]
        self.add_node(n_b)

    def add_node(self, node):
        assert(self.node_dict is not None)
        node_dict = self.node_dict
        
        node_in_cdg = False
        try:
            _ = node_dict[node]
            node_in_cdg = True
        except:
            pass

        # return early
        if node_in_cdg:
            return
        
        # else actually add
        self.nodes.append(node)
        # val doesnt matter
        self.node_dict.update({node : True})
        self.n_nodes += 1

    # remove turns/channels/nodes
    ####################################################################################################

    def remove_turn(self, turn):
        input('NOT YET IMPLEMENTED')
        quit()
    
    # probe turns/channels/nodes
    ####################################################################################################

    def probe_turn_for_deadlock(self, turn):
        assert(isinstance(turn,tuple))
        assert(len(turn) == 2)
        # channels are channel as nodes and vcs
        assert(isinstance(turn[0],tuple))
        assert(isinstance(turn[1],tuple))

        turn_id = self.translate_turn(turn)
        c_a_id, c_b_id = turn_id

        # exit early if edge already in CDG
        # if already present it wont create deadlock
        if self.cdg_as_nwx_G.has_edge(c_a_id, c_b_id):
            if self.verbose:
                print(f'No need to probe. CDG already has turn {turn}')
            return False


        # # TODO move to function
        assert(self.cdg_as_nwx_G is not None)
        cdg_as_nwx_G = self.cdg_as_nwx_G
        # input(f'c_a_id = {c_a_id}, c_b_id = {c_b_id}')
        # input(f'cdg_as_nwx_G nodes = {self.cdg_as_nwx_G.nodes()}')

        if c_a_id not in cdg_as_nwx_G:
            cdg_as_nwx_G.add_node(c_a_id)
        if c_b_id not in cdg_as_nwx_G:
            cdg_as_nwx_G.add_node(c_b_id)

        if nx.has_path(cdg_as_nwx_G, c_b_id, c_a_id):
            return True
        return False

    def probe_turn_for_deadlock_old(self, turn):
        assert(isinstance(turn,tuple))
        assert(len(turn) == 2)
        # channels are channel as nodes and vcs
        assert(isinstance(turn[0],tuple))
        assert(isinstance(turn[1],tuple))

        turn_id = self.translate_turn(turn)
        c_a_id, c_b_id = turn_id

        # exit early if edge already in CDG
        # if already present it wont create deadlock
        if self.cdg_as_nwx_G.has_edge(c_a_id, c_b_id):
            if self.verbose:
                print(f'No need to probe. CDG already has turn {turn}')
            return False
        
        # lightly add to CDG (no point copying a local. too much time)
        # without modifying obj vars for turns, channels, nodes etc.
        self.cdg_as_nwx_G.add_edge(c_a_id, c_b_id)

        # smart! all cycles will defnitely include node c_a_id (and c_b_id) if/they it exists
        # creates_deadlock = self.cdg_has_cycle_from_new_edge((c_a_id, c_b_id))
        creates_deadlock = self.cdg_has_cycle_from_new_node(c_a_id)
        # creates_deadlock_3 = self.cdg_has_cycle()

        # if creates_deadlock != creates_deadlock_2:
        #     input('ERROR on 1 v 2')

        # if creates_deadlock_2 != creates_deadlock_3:
        #     input('ERROR on 2 v 3')

        # clean up no matter what!!
        self.cdg_as_nwx_G.remove_edge(c_a_id, c_b_id)

        return creates_deadlock

    # check cycles
    ####################################################################################################

    def cdg_has_cycle(self):
        cdg_cycle = self.networkx_get_cycle()
        if len(cdg_cycle) > 0:
            return True
        return False

    def cdg_has_cycle_from_new_node(self, new_node):
        # either possible
        src_nodes = [new_node]
        cdg_cycle = self.networkx_get_cycle(src_nodes=src_nodes)
        if len(cdg_cycle) > 0:
            return True
        return False

    def cdg_has_cycle_from_new_edge(self, new_edge):
        # either possible
        src_nodes = [new_edge[0], new_edge[1]]
        cdg_cycle = self.networkx_get_cycle(src_nodes=src_nodes)
        if len(cdg_cycle) > 0:
            return True
        return False

    def networkx_get_cycle(self, src_nodes=None):
        assert(self.cdg_as_nwx_G is not None)
        cdg_as_nwx_G = self.cdg_as_nwx_G

        cycle_as_channel_ids = None
        if src_nodes is None:
            cycle_as_channel_ids = self.networkx_get_cycle_(cdg_as_nwx_G)
        else:
            cycle_as_channel_ids = self.networkx_get_cycle_w_srcs_(cdg_as_nwx_G, src_nodes)

        return cycle_as_channel_ids

    # visualization
    ####################################################################################################

    def visualize_cdg(self, viz_type=None):
        assert(self.cdg_as_nwx_G is not None)
        cdg_as_nwx_G = self.cdg_as_nwx_G

        viz_options = {
            "font_size": 12,
            "node_size": 300,
            "with_labels":True,
            "node_color": "blue",
            "edgecolors": "black",
            "linewidths": 1,
            "width": 1
        }

        if self.viz_options is not None:
            viz_options = self.viz_options

        color_map = []
        # for understanding
        node_labels_dict = {  }
        for node in cdg_as_nwx_G:
            node_as_channel = self.translate_channel_id(node)
            node_vc = node_as_channel[2]
            if node_vc > 0:
                color_map.append('blue')
            else: 
                color_map.append('green')
            
            node_labels_dict.update( { node : node_as_channel })

        viz_options.update({'node_color':color_map})



        self.visualize_cdg_(cdg_as_nwx_G, node_labels=node_labels_dict, viz_type=viz_type,viz_options=viz_options)

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

    my_CDG = OmniCDG()

    # map_file_name = 'files/map_files/kite_large.map'
    # r_map = ingest_map(map_file_name)
    # my_CDG.init_complete_cdg_from_map(r_map)

    n_nodes = 5
    my_CDG.init_w_n_nodes(n_nodes, n_vcs=2)
    # temp_turns = [ ((0,1),(1,3)) , ( (1,3),(3,4)) , ((3,4),(4,2)) , ( (4,2),(2,0)) , ( (2,0),(0,1))  ]
    temp_turns = [ ((0,1),(1,3)) , ( (1,3),(3,4)) , ((3,4),(4,2)) , ( (4,2),(2,0))  ]

    my_CDG.add_node_turns_common_vc(temp_turns)

    print(f'channel_to_node_map = {my_CDG._channel_to_node_map}')
    print(f'node_to_channel_map = {my_CDG._node_to_channel_map}')

    cdg_cycles = my_CDG.networkx_get_cycle()
    print(f'cycle = {cdg_cycles}')

    my_CDG.set_verbose()

    vc = 0
    probe_turn = ( (2,0,vc),(0,1,vc))
    causes_dl = my_CDG.probe_turn_for_deadlock(probe_turn)
    print(f'{probe_turn} causes_dl = {causes_dl}')

    probe_turn = ( (2,0,vc),(0,3,vc))
    causes_dl = my_CDG.probe_turn_for_deadlock(probe_turn)
    print(f'{probe_turn} causes_dl = {causes_dl}')



    probe_turn = ( (0,3,vc),(3,4,vc))
    causes_dl = my_CDG.probe_turn_for_deadlock(probe_turn)
    print(f'{probe_turn} causes_dl = {causes_dl}')

    new_turn = ( (2,0,vc),(0,3,vc))
    my_CDG.add_turn(new_turn)

    vc = 0
    probe_turn = ( (0,3,vc),(3,4,vc))
    causes_dl = my_CDG.probe_turn_for_deadlock(probe_turn)
    print(f'{probe_turn} causes_dl = {causes_dl}')

    vc = 1
    probe_turn = ( (0,3,vc),(3,4,vc))
    causes_dl = my_CDG.probe_turn_for_deadlock(probe_turn)
    print(f'{probe_turn} causes_dl = {causes_dl}')

    # my_CDG.print_cycle_as_map_nodes(cdg_cycles)

    # turns_as_nodes = my_CDG.translate_cycle_as_turns_of_nodes(cdg_cycles)
    # print(f'turns_as_nodes={turns_as_nodes}')

    # my_CDG.visualize_cdg()

if __name__ == '__main__':
    main()