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
Verify constraints and calculate metrics of topologies from files
'''

import argparse
import networkx as nx


INF = 10**9


global VERBOSE
VERBOSE = False

global ASSERT_BINARY_MAP
ASSERT_BINARY_MAP = True


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

            # deal with approximate values (from MIP)
            try:
                r_conns = [int(elem) for elem in r_conns]

            except:
                r_conns = [int(float(elem)) for elem in r_conns]

            this_map.append(r_conns)

            these_dests = [i for i, val in enumerate(r_conns) if val > 0]
            for dest in these_dests:
                edge_list.append((cur_src, dest))

            cur_src += 1


    # quick sanitization
    n_routers = len(this_map)
    for i in range(n_routers):
        this_map[i][i] = 0

    # assert binary?
    if ASSERT_BINARY_MAP:
        for src_map in this_map:
            for conn in src_map:
                # instead, make binary
                if conn > 0.1:
                    conn = 1
                # assert(conn == 1 or conn == 0)

    if VERBOSE:
        print(f'read {this_map}')

    return this_map, n_routers, edge_list

def create_an_nwx_G_from_a_map(this_map):

    n_routers = len(this_map)

    G = nx.DiGraph()

    for src in range(n_routers):
        for dest in range(n_routers):

            if(src == dest):
                continue

            if(this_map[src][dest] < 1):
                continue

            # print(f'connecting {src} -> {dest}')

            G.add_edge(src,dest)

    return G

def calc_avg_hops(adj_mat):

    G = create_an_nwx_G_from_a_map(adj_mat)

    try:
        avg_hops = nx.average_shortest_path_length(G)
    except:
        avg_hops = INF

    d = nx.diameter(G)

    return avg_hops, d

def main():

    parser = argparse.ArgumentParser(description='Verify topology values')
    parser.add_argument('--topology',type=str,help='.map file to evaluate')
    parser.add_argument('--verbose',action='store_true',help='debug prints')

    args = parser.parse_args()


    global VERBOSE
    VERBOSE = args.verbose

    topology_filepath = args.topology

    adj_mat, n_routers, edge_list = ingest_map(topology_filepath)

    avg_hops, diam = calc_avg_hops(adj_mat)

    print(f'Average hops = {avg_hops}')
    print(f"diameter = {diam}")


if __name__ == '__main__':
    main()
