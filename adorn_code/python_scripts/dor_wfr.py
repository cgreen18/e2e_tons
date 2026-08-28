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
    Implementation of Nue (https://dl.acm.org/doi/10.1145/2907294.2907313)

    Uses "multilevel k-way partitioning algorithm"
        (https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=d545a4e5a5e00935e59141f26041531ba1aa97a0)

"""

# std
import argparse
import ast
import os
import random
from collections import defaultdict
import time

# pipd
import orjson



PATHS_OUT_DIR = './topologies_and_routing/backup_routepath_lists'


global VERBOSE
VERBOSE = False
global SLOW_RUN
SLOW_RUN = False
global ASSERT_BINARY_MAP
ASSERT_BINARY_MAP = True


# File stuff
###############################################################################

def ingest_map(path_name):
    global ASSERT_BINARY_MAP

    file_name = path_name.split('/')[-1]

    if True:
        print(f'Ingesting filename = {file_name} ({path_name})')

    adj_mat = []
    adj_list = []
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

            adj_mat.append(r_conns)

            these_dests = [i for i, val in enumerate(r_conns) if val > 0]
            for dest in these_dests:
                edge_list.append((cur_src, dest))

            adj_list.append(these_dests)

            cur_src += 1


    # quick sanitization
    n_routers = len(adj_mat)
    for i in range(n_routers):
        adj_mat[i][i] = 0

    # assert binary?
    if ASSERT_BINARY_MAP:
        for src_map in adj_mat:
            for conn in src_map:
                # instead, make binary
                if conn > 0.1:
                    conn = 1
                # assert(conn == 1 or conn == 0)

    if VERBOSE:
        print(f'read {adj_mat}')

    return adj_mat, adj_list, n_routers, edge_list

def stream_pathlist( path):

    with open(path, "r", buffering=1024*1024) as inf:
        next(inf, None)  # skip header
        for line in inf:
            line = line.strip()
            if line:
                yield orjson.loads(line)  # produce one row at a time


# alg
###############################################################################

def decide_wild_hops(adj_list):

    radix = 6
    n_routers = len(adj_list)

    wild_hops_twod = [[None for _ in range(n_routers)] for __ in range(n_routers)]

    for src, neighbors in enumerate(adj_list):
        for dest in range(n_routers):
            if src == dest:
                continue

            sel_idx = random.randint(0, radix-1)
            sel_hop = neighbors[sel_idx]

            # cant just hop to dest
            if sel_hop == dest:
                sel_idx = (sel_idx + 1 ) % radix
                sel_hop = neighbors[sel_idx]

            wild_hops_twod[src][dest] = sel_hop

    return wild_hops_twod

def convert_pathlist_to_nrl(map_filepath, pathlist_filepath):

    print(f'WFR for topology {map_filepath} w/ routing {pathlist_filepath}')

    out_name_base = os.path.splitext(os.path.basename(pathlist_filepath))[0]
    wfr_paths_filepath = os.path.join(PATHS_OUT_DIR, f'{out_name_base}_wfr.paths')

    adj_mat, adj_list, n_routers, edge_list = ingest_map(map_filepath)


    # pre-decide the wild hops for each src, dest pair
    wild_hops_twod = decide_wild_hops(adj_list)

    # reverse lookup
    reverse_hops_dict = defaultdict(list)
    for src, dest_hops in enumerate(wild_hops_twod):
        for dest, hop in enumerate(dest_hops):
            reverse_hops_dict[(hop,dest)].append( src )


    wfr_routes_twod = [[None for _ in range(n_routers)] for __ in range(n_routers)]
    for i in range(n_routers):
        wfr_routes_twod[i][i] = [i]

    n_completed = 0

    prev_time = time.time()


    for path in stream_pathlist(pathlist_filepath):
        path_src = path[0]
        path_dest = path[-1]

        n_hops = len(path) - 1

        
        # find srcs that use this hop and path
        for other_src in reverse_hops_dict[ (path_src, path_dest) ]:
            # path_src is the wild hop
            wfr_routes_twod[other_src][path_dest] = [other_src] + path



        cur_time = time.time()

        n_completed += 1
        if n_completed % 1_000_000 == 0:
            print(f'n_completed = {n_completed} in {round(cur_time - prev_time,2)}s')
            # print(f'path ({len(path)}) = {path}')
            prev_time = cur_time


    # cant stream out since paths are created out of order
    with open(wfr_paths_filepath, 'w+') as of:
        of.write(out_name_base + '\n')
        for dest_paths in wfr_routes_twod:
            for path in dest_paths:
                of.write(f'{path}\n')


    print(f'Completed write to {wfr_paths_filepath}')
# main
####################################################################################################


def main():
    parser = argparse.ArgumentParser(description='Choose an extra random hop')
    parser.add_argument('--filename',type=str,help='graph',required=True)
    parser.add_argument('--pathlist',type=str,help='pathlist to add wfr',required=True)


    args = parser.parse_args()
    
    convert_pathlist_to_nrl(args.filename,args.pathlist)

if __name__ == '__main__':
    main()