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
from cmath import sqrt
from itertools import combinations
import os
import math
import ast
import statistics
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import connected_components
import time
import operator as op
from functools import reduce
import networkx as nx

global n_routers
global n_ports
n_ports = 6
global longest_link
global r_map
r_map = []
global graph
graph = []
global dist
dist = []

# useless
global chiplet_dist
chiplet_dist = []

global n_links
global mem_diameter
mem_diameter = 0
global coh_diameter
coh_diameter = 0
global avg_hops
global avg_mem_hops
global bisection_bw # set default within function, when n_routers known
global bisection_bw_combo

global unidir_bibw
global vertical_bi_bw
global horizontal_bi_bw

global worst_cut
global worst_cut_combo

global worst_mem_cut
global worst_mem_cut_combo

global worst_mem_bi_cut
global worst_mem_bi_cut_combo

global worst_bottleneck
global worst_bottleneck_combo


global memcoh
memcoh = 0.5



# for floyd warshall
INF = 999


# CLA glboals
global weighted_dist
weighted_dist = False

global use_vll
use_vll = False

global no_bi_bw
no_bi_bw = False

global no_sc_bw
no_sc_bw = False

# for debugging
global verbose
verbose = False

def print_by_src(mat, n,m):

    i = 0
    for src in mat:
        print(f"r{i}")
        print(src)
        print_nxm(src, n,m)
        i+=1

def print_nxm(conns, n, m):
    for i in reversed(range(0,n)):
        for j in range(0,m):
            # print(f"{i*m+j}={conns[i*m+j]}",end = " ")
            print(f"{conns[i*m+j]}",end = " ")
        print("")
    print("")

def clear_previous():

    global r_map

    global dist
    global graph

    r_map = []
    dist = []
    graph = []

def ingest(file_name):

    global r_map
    global n_routers
    r_map = []
    n_routers = 0

    if '.sol' in file_name:
        ingest_sol(file_name)
    elif '.map' in file_name:
        ingest_map(file_name)
    else:
        print(f'error on input file type')
        quit()


    # sanitize

    # shallow/value copy
    old_map = [[e for e in row] for row in r_map]
    for i, row in enumerate(old_map):
        for j, elem in enumerate(row):
            if elem > 0.5:
                r_map[i][j] = 1
            else:
                r_map[i][j] = 0


    if len(r_map) > 0:
        return True

def ingest_map(path_name):
    global n_routers
    global r_map

    file_name = path_name.split('/')[-1]

    print(f'filename = {file_name}')

    with open(path_name, 'r') as in_file:

        # for _router in range(0,n_routers):
        #     row = in_file.readline()
        for row in in_file:
            r_conns = row.split(" ")
            try:
                r_conns.remove('\n')
            except:
                pass

            try:
                r_conns = [int(elem) for elem in r_conns]
            except:
                r_conns = [int(float(elem)) for elem in r_conns]
            r_map.append(r_conns)

        # print(f'r_map({len(r_map)})={r_map}')
        # input('cont?')

    n_routers = len(r_map)
    print(f'r_map={r_map}')

def ingest_sol(file_name):
    global n_routers
    global n_ports
    global r_map
    # global memcoh


    n_lines = 0
    with open(file_name, 'r') as in_file:
        for n_lines, line in enumerate(in_file):
            pass
    n_lines += 1


    with open(file_name, 'r') as in_file:


        n_routers = in_file.readline()
        n_ports = in_file.readline()


        n_routers = int(n_routers)
        n_ports = int(n_ports)

        if n_lines == n_routers + 3:
            memcoh = in_file.readline()
            memcoh = float(memcoh)
        else:
            memcoh = 1.0

        print(f'n_routers={n_routers}, n_ports={n_ports}, memcoh={memcoh}')

        for _router in range(0,n_routers):
            row = in_file.readline()
            r_conns = row.split(" ")
            try:
                r_conns = [int(elem) for elem in r_conns]
            except:
                r_conns = [int(float(elem)) for elem in r_conns]
            r_map.append(r_conns)

        # print(f'r_map({len(r_map)})={r_map}')
        # input('cont?')

def out_to_csv(config, out_file_name):

    # config ex) 20r_15ll_opt_whop_8bw_4diam_memcoh0_0
    # write config, n_links, diameter, avg hops, bi bw, vbi bw, hbi bw

    # quick reduction
    diameter = min([mem_diameter, coh_diameter])


    try:
        with open(out_file_name, 'a+') as of:
            out_line = f'{config},{n_links},{diameter},{bisection_bw},{avg_hops},{vertical_bi_bw},{horizontal_bi_bw},{worst_bottleneck},{len(worst_bottleneck_combo)},{half_bw(worst_bottleneck_combo)[1]}\n'
            of.writelines([out_line])
    except:

        pass

    print(f'Wrote {config} out to {out_file_name}')

def verify(map, radix):
    print('Verifying: radix, symmetry, reflexivity')

    r = 0
    success = True
    for s, row in enumerate(map):
        sum = 0
        for d, item in enumerate(row):
            if item == 1:
                sum += 1
                print(f'{s}->{d}')
        print(f'radix of r{r}={sum}')
        if (sum > radix):
            success = False
            input(f'Radix violation for router {r}')
        r +=1

    if success:
        print('Verified radix')
    else:
        print('Radix failure')

    input('verfication')



    reflex_success = True
    sym_success = True
    for i in range(n_routers):

        if r_map[i][i] != 0:
            # print(f'Map reflexivity violation for router {i}')
            reflex_success = False
        if dist[i][i] != 0:
            # print(f'Distance reflexivity violation for router {i}')
            reflex_success = False

        for j in range(i+1,n_routers):

            if r_map[i][j] != r_map[j][i]:
                # print(f'Map symmetry violation for routers {i},{j}')
                sym_success = False
            if dist[i][j] != dist[j][i]:
                # print(f'Distance symmetry violation for routers {i},{j}')
                sym_success = False

    if reflex_success:
        print('Verified reflexivity ')
    else:
        print('Reflexivity failure')

    if sym_success:
        print('Verified symmetry')
    else:
        print('Topology uses unidirectional links (or symmetry failure)')

def floyd(G):
    # print(G)
    global dist
    dist = list(map(lambda p: list(map(lambda q: q, p)), G))

    for r in range(n_routers):
        for p in range(n_routers):
            for q in range(n_routers):
                dist[p][q] = min(dist[p][q], dist[p][r] + dist[r][q])

def floyd_ret_val(G):
    dist = list(map(lambda p: list(map(lambda q: q, p)), G))

    for r in range(n_routers):
        for p in range(n_routers):
            for q in range(n_routers):
                dist[p][q] = min(dist[p][q], dist[p][r] + dist[r][q])

    return dist

def vert_bi_bw(per_row):
    global vertical_bi_bw

    # quick assumption?
    # per_row = round(sqrt(n_routers))

    # eg per_row = 5 => half_row = 2
    half_row = int(per_row / 2)

    lefts = [x for x in range(n_routers) if (x % per_row) <= half_row ]

    rights = [x for x in range(n_routers) if x not in lefts ]

    n_crossings = 0
    for l in lefts:
        for r in rights:
            n_crossings += r_map[l][r]

    vertical_bi_bw = n_crossings

def horiz_bi_bw():

    global horizontal_bi_bw

    n_half = n_routers / 2

    uppers = [x for x in range(n_routers) if x / n_half >= 1 ]

    lowers = [x for x in range(n_routers) if x not in uppers ]

    n_crossings = 0
    for up in uppers:
        for low in lowers:
            n_crossings += r_map[up][low]

    horizontal_bi_bw = n_crossings

def is_connected_two(in_set):
    global r_map
    global n_routers

    new_size = len(in_set)

    graph = [[r_map[i][j] for j in in_set] for i in in_set]
    # graph = [[0 for item in row] for row in r_map]

    # for n in in_set:
    #     for m in in_set:
    #         if r_map[n][m] >= 1:
    #             graph[n][m] = 1

    # for i in range(0,n_routers):
    #     graph[i][i]=0

    npgraph  = np.array(graph)

    # print(f'in_set={in_set}')
    # print(f'npgraph={npgraph}')

    n_partitions, indices = connected_components(npgraph)

    # print(f'{in_set} => ({n_partitions}) {indices}')

    if n_partitions != 1:
        return False
    return True

def is_connected(in_set):
    global r_map
    global n_routers

    graph = [[INF for item in row] for row in r_map]

    for n in in_set:
        for m in in_set:
            if r_map[n][m] >= 1:
                graph[n][m] = 1

    for i in range(0,n_routers):
        graph[i][i]=0

    #print(f'graph={graph}')
    #print(f'in_set={in_set}')
    #input('continue?')

    dists = floyd_ret_val(graph)


    for n in in_set:
        for m in in_set:
            if dists[n][m] >= INF:
                #print(f'unconected {n} -> {m} for in_set {in_set}')
                return False

    return True

def calc_worst_cut():
    global worst_cut
    global worst_cut_combo
    # global unidir_bibw

    global worst_bottleneck
    global worst_bottleneck_combo

    global n_routers

    print(f'Will calculate SC')

    # verbose = True

    start_time = time.time()

    routers = range(0,n_routers)
    all_combos = list()
    n_in_set = int(n_routers / 2)
    all_combos += combinations(routers, n_in_set)


    least_combo = all_combos[0]
    least_left_combo = all_combos[0]
    (least_bw, least_left_bw) = half_bw(least_combo, vb=verbose)

    n_pkts_crossing = n_in_set*float(n_routers-n_in_set)
    # worst_bottleneck = n_pkts_crossing / least_left_bw
    worst_bottleneck =  least_left_bw / n_pkts_crossing

    worst_bottleneck_combo = all_combos[0]




    if verbose:
        print(f'worst cut: first combo has bisection bw={least_bw} from combo {least_combo}')
        print(f'worst cut: first combo has worst_bottleneck={worst_bottleneck} from least left {least_left_bw} w/ n_kts_crossing {n_pkts_crossing}')


    print(f'about to consider sets of all sizes')



    for n_in_set in range(1,n_routers):

        n_pkts_crossing = n_in_set*float(n_routers-n_in_set)


        all_combos = list()
        all_combos += combinations(routers, n_in_set)

        # s_time = time.time()

        # all_connected_combos = [a for a in all_combos if is_connected_two(a)]
        # t_diff = time.time() - s_time

        # print(f'set size {n_in_set} => # connected = {len(all_connected_combos)} / # total {len(all_combos)}\t{t_diff}s')

        # quit()

        if verbose:
            print(f'working on cut of size {n_in_set}. # combos={len(all_combos)}')
            print(f'n_pkts_crossing = {n_pkts_crossing}')

        for combo in all_combos:
            (this_bw, this_left_bw) = half_bw(combo)

            # this_bottleneck = n_pkts_crossing / this_left_bw
            this_bottleneck = this_left_bw / n_pkts_crossing

            # add is_connected criteria
            if this_left_bw < least_left_bw and is_connected(combo):
                least_left_bw = this_left_bw
                least_left_combo = combo
                if verbose:
                    print(f'worst cut: lesser left bisection bw found: {this_left_bw} < {least_left_bw}')
                    (_,__) = half_bw(combo, vb=True)
                    print(f'worst cut: combo has worst_bottleneck={worst_bottleneck} from {worst_bottleneck_combo}')


            if this_bottleneck < worst_bottleneck and is_connected(combo):
                worst_bottleneck = this_bottleneck
                worst_bottleneck_combo = combo
                if verbose:
                    print(f'worst cut: worse bottleneck found: {this_bottleneck} < {worst_bottleneck} from left_bw {this_left_bw}')
                    (_,__) = half_bw(combo, vb=True)
                    print(f'worst cut: combo has worst_bottleneck={worst_bottleneck} from {worst_bottleneck_combo}')


    # print(f'worst cut:  bisection bandwidth={least_bw} from combo {least_combo}')
    # bisection_bw = least_bw
    # bisection_bw = least_bw

    worst_cut = worst_bottleneck
    worst_cut_combo = worst_bottleneck_combo
    print(f'worst cut: left bisection bandwidth={least_left_bw}, worst_cut={worst_cut}, from combo {least_left_combo}')
    print(f'worst cut: {round(time.time() - start_time,2)} seconds elapsed')

def calc_worst_mem_cut():
    global worst_mem_cut
    global worst_mem_cut_combo
    # global unidir_bibw


    global n_routers

    # verbose = True

    start_time = time.time()

    routers = range(0,n_routers)
    all_combos = list()
    n_in_set = 8 # int(n_routers / 2)
    all_combos += combinations(routers, n_in_set)

    # all_connected_combos = [a for a in all_combos if is_connected_two(a)]

    # quit()

    # print(f'total # combos {len(all_combos)}\ntotal # connected {len(all_connected_combos)}')

    # quit()

    # worst_mem_cut_combo = all_combos[0]
    worst_mem_cut_combo = [0,4,5,9,10,14,15,19]
    (_least_bw, least_left_bw) = half_bw(worst_mem_cut_combo, vb=verbose)

    n_pkts_crossing = n_in_set*float(n_routers-n_in_set)
    # worst_bottleneck = n_pkts_crossing / least_left_bw
    worst_mem_cut =  least_left_bw / n_pkts_crossing

    if verbose:
        pass
    print(f'worst mem cut: first combo has bisection bw={least_left_bw} from combo {worst_mem_cut_combo}')
    print(f'worst mem cut: first combo has worst_bottleneck={worst_mem_cut} from least left {least_left_bw} w/ n_kts_crossing {n_pkts_crossing}')
    # quit()

    print(f'about to consider sets of all sizes')

    lefts = [0,5,10,15]
    rights = [4,9,14,19]
    exteriors = lefts + rights

    for n_in_set in range(1,n_routers):

        n_pkts_crossing = n_in_set*float(n_routers-n_in_set)


        all_combos = list()
        all_combos += combinations(routers, n_in_set)

        all_valid_combos = []
        for combo in all_combos:
            valid = True
            for node in combo:
                if node not in exteriors:
                    valid = False
            if valid:
                all_valid_combos.append(combo)

        # print(f'combos {all_combos}\n=>\n{all_valid_combos}')
        # input('cont?')

        all_combos = all_valid_combos


        # s_time = time.time()

        # all_connected_combos = [a for a in all_combos if is_connected_two(a)]
        # t_diff = time.time() - s_time

        # print(f'set size {n_in_set} => # connected = {len(all_connected_combos)} / # total {len(all_combos)}\t{t_diff}s')

        # quit()

        if verbose:
            print(f'working on cut of size {n_in_set}. # combos={len(all_combos)}')
            print(f'n_pkts_crossing = {n_pkts_crossing}')

        for combo in all_combos:
            (this_bw, this_left_bw) = half_bw(combo)

            # this_bottleneck = n_pkts_crossing / this_left_bw
            this_bottleneck = this_left_bw / n_pkts_crossing

            if this_bottleneck < worst_mem_cut and is_connected(combo):
                worst_mem_cut = this_bottleneck
                worst_mem_cut_combo = combo
                least_left_bw = this_left_bw
                if verbose:
                    print(f'worst cut: worse bottleneck found: {this_bottleneck} < {worst_mem_cut} from left_bw {this_left_bw}')
                    (_,__) = half_bw(combo, vb=True)
                    print(f'worst cut: combo has worst_mem_cut={worst_mem_cut} from {worst_mem_cut_combo}')


    # print(f'worst cut:  bisection bandwidth={least_bw} from combo {least_combo}')
    # bisection_bw = least_bw
    # bisection_bw = least_bw

    print(f'worst mem cut: left bisection bandwidth={least_left_bw}, worst_cut={worst_mem_cut}, from combo {worst_mem_cut_combo}')
    print(f'worst mem cut: {round(time.time() - start_time,2)} seconds elapsed')

def calc_worst_mem_bi_cut():
    global worst_mem_bi_cut
    global worst_mem_bi_cut_combo
    # global unidir_bibw


    global n_routers

    # verbose = True

    start_time = time.time()

    routers = range(0,n_routers)
    all_combos = list()
    n_in_set = 4 # int(n_routers / 2)
    all_combos += combinations(routers, n_in_set)

    # all_connected_combos = [a for a in all_combos if is_connected_two(a)]

    # quit()

    # print(f'total # combos {len(all_combos)}\ntotal # connected {len(all_connected_combos)}')

    # quit()

    # worst_mem_cut_combo = all_combos[0]
    worst_mem_bi_cut_combo = [0,4,5,9]
    (_least_bw, least_left_bw) = half_bw(worst_mem_bi_cut_combo, vb=verbose)

    n_pkts_crossing = n_in_set*float(n_routers-n_in_set)
    # worst_bottleneck = n_pkts_crossing / least_left_bw
    worst_mem_bi_cut =  least_left_bw / n_pkts_crossing

    if verbose:
        pass
    print(f'worst mem bi cut: first combo has bisection bw={least_left_bw} from combo {worst_mem_bi_cut_combo}')
    print(f'worst mem bi cut: first combo has worst_bottleneck={worst_mem_bi_cut} from least left {least_left_bw} w/ n_kts_crossing {n_pkts_crossing}')
    # quit()

    print(f'about to consider sets of all sizes')

    lefts = [0,5,10,15]
    rights = [4,9,14,19]
    exteriors = lefts + rights




    all_combos = list()
    all_combos += combinations(routers, n_in_set)

    all_valid_combos = []
    for combo in all_combos:
        valid = True
        for node in combo:
            if node not in exteriors:
                valid = False
        if valid:
            all_valid_combos.append(combo)

    # print(f'combos {all_combos}\n=>\n{all_valid_combos}')
    # input('cont?')

    all_combos = all_valid_combos


    # s_time = time.time()

    # all_connected_combos = [a for a in all_combos if is_connected_two(a)]
    # t_diff = time.time() - s_time

    # print(f'set size {n_in_set} => # connected = {len(all_connected_combos)} / # total {len(all_combos)}\t{t_diff}s')

    # quit()

    if verbose:
        print(f'working on cut of size {n_in_set}. # combos={len(all_combos)}')
        print(f'n_pkts_crossing = {n_pkts_crossing}')

    for combo in all_combos:
        (this_bw, this_left_bw) = half_bw(combo)

        # this_bottleneck = n_pkts_crossing / this_left_bw
        this_bottleneck = this_left_bw / n_pkts_crossing

        if this_bottleneck < worst_mem_bi_cut and is_connected(combo):
            worst_mem_bi_cut = this_bottleneck
            worst_mem_bi_cut_combo = combo
            least_left_bw = this_left_bw
            if verbose:
                print(f'worst cut: worse bottleneck found: {this_bottleneck} < {worst_mem_bi_cut} from left_bw {this_left_bw}')
                (_,__) = half_bw(combo, vb=True)
                print(f'worst cut: combo has worst_mem_bi_cut={worst_mem_bi_cut} from {worst_mem_bi_cut_combo}')


    # print(f'worst cut:  bisection bandwidth={least_bw} from combo {least_combo}')
    # bisection_bw = least_bw
    # bisection_bw = least_bw

    print(f'worst bisected mem cut: left bisection bandwidth={least_left_bw}, worst_cut={worst_mem_bi_cut}, from combo {worst_mem_bi_cut_combo}')
    print(f'worst bisected mem cut: {round(time.time() - start_time,2)} seconds elapsed')

def calc_bi_combos(bi_bw):
    global n_routers

    routers = range(0,n_routers)
    all_combos = list()
    half = int(n_routers / 2)
    all_combos += combinations(routers, half)

    bi_combos = []

    for combo in all_combos:
        (this_bw,this_left_bw) = half_bw(combo)

        # # inequality
        # if this_bw - this_left_bw < this_left_bw:
        #     # take complement
        #     this_left_bw = this_bw - this_left_bw
        if this_left_bw == bi_bw:
            if verbose:
                print(f'bi combo found: {combo}')
            bi_combos.append(combo)

    return bi_combos

def calc_sc_combos(sc, n_in_set):
    global n_routers

    routers = range(0,n_routers)

    n_pkts_crossing = n_in_set*float(n_routers-n_in_set)

    all_combos = list()
    all_combos += combinations(routers, n_in_set)

    sc_combos = []

    for combo in all_combos:
        (this_bw,this_left_bw) = half_bw(combo)

        this_bottleneck = this_left_bw / n_pkts_crossing

        # # inequality
        # if this_bw - this_left_bw < this_left_bw:
        #     # take complement
        #     this_left_bw = this_bw - this_left_bw
        if this_bottleneck <= sc and is_connected(combo):
            if verbose:
                print(f'sc combo found: {combo}')
            sc_combos.append(combo)

    return sc_combos

def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom  # or / in Python 2

def calc_cube_crossing_bw():
    global min_cube_bw
    global min_cube_thru
    global x_dim
    global y_dim
    global z_dim
    global cube_dim

    global r_map
    global n_routers

    cube_dict = {}

    for i in range(n_routers):
        i_cube = which_cube(i, x_dim, y_dim, z_dim, cube_dim)

        try:
            cube_dict[i_cube].append(i)
        except:
            cube_dict.update({i_cube:[i]})
    
    min_cube_bw = INF
    min_cube_thru = INF
    per_cube = cube_dim**3
    demand = per_cube*(n_routers-per_cube)

    for cube_a, a_r_list in cube_dict.items():
        b_r_list = [b for b in range(n_routers) if b not in a_r_list]
            
        n_crossings = 0

        for a in a_r_list:
            for b in b_r_list:
                n_crossings += r_map[a][b]

        if n_crossings < min_cube_bw:
            min_cube_bw = n_crossings
            min_cube_thru = n_crossings / demand
            # print(f'\timplies thruput of {min_cube_thru}')

        print(f'cube {cube_a} w/ bw {n_crossings} and thruput of {n_crossings/demand}')


def calc_multi_cube_crossing_bw():
    global min_cube_bw
    global min_cube_thru
    global x_dim
    global y_dim
    global z_dim
    global cube_dim

    global r_map
    global n_routers

    cube_dict = {}

    for i in range(n_routers):
        i_cube = which_cube(i, x_dim, y_dim, z_dim, cube_dim)

        try:
            cube_dict[i_cube].append(i)
        except:
            cube_dict.update({i_cube:[i]})

    n_cubes = len(cube_dict.keys())

    all_combos = list()
    for i in range(n_cubes):
        all_combos += combinations(range(n_cubes), i)
    
    min_cube_thru = INF


    for cube_combo in all_combos:

        if len(cube_combo) == 0:
            continue

        a_r_list = []

        for cube in cube_combo:
            a_r_list += cube_dict[cube]


        b_r_list = [b for b in range(n_routers) if b not in a_r_list]
            
        n_crossings = 0

        for a in a_r_list:
            for b in b_r_list:
                n_crossings += r_map[a][b]

        demand = len(a_r_list)*len(b_r_list)

        cut_thru = n_crossings / demand

        if cut_thru < min_cube_thru:
            min_cube_thru = n_crossings / demand
            # print(f'\timplies thruput of {min_cube_thru}')

        print(f'cubes {cube_combo} w/ bw {n_crossings} and thruput of {cut_thru}')

    input('done')

def calc_unused_links():

    global min_cube_bw
    global min_cube_thru
    global x_dim
    global y_dim
    global z_dim
    global cube_dim

    global r_map
    global n_routers
    global n_ports

    tot_conns = 0

    new_conns = []

    for i, row in enumerate(r_map):

        i_x,i_y,i_z = r_to_xyz(i,x_dim,y_dim,z_dim)
        rel_i_x = i_x % cube_dim
        rel_i_y = i_y % cube_dim
        rel_i_z = i_z % cube_dim

        n_conns = sum(row)

        if n_conns < n_ports:
            print(f'row {i}=({i_x},{i_y},{i_z}) underutilized at {n_conns} conns')
            print(f'\t\t\trel pos=({rel_i_x},{rel_i_y},{rel_i_z})')

            # for j in range(n_routers):
            #     j_x,j_y,j_z = r_to_xyz(j,x_dim,y_dim,z_dim)
            #     rel_j_x = j_x % cube_dim
            #     rel_j_y = j_y % cube_dim
            #     rel_j_z = j_z % cube_dim

            #     if (rel_i_y,rel_i_z) == (rel_j_y,rel_j_z) and (rel_i_x + cube_dim - 1 ) % cube_dim  == rel_j_x and (rel_i_x == 0 or rel_i_x == cube_dim - 1) :
            #         print(f'\t\tpossible x conn to {j} ({j_x},{j_y},{j_z}) ({rel_j_x},{rel_j_y},{rel_j_z})')
            #         print(f'\t\tr_map[{i}][{j}]={r_map[i][j]}')
            #     if (rel_i_x,rel_i_z) == (rel_j_x,rel_j_z) and (rel_i_y + cube_dim - 1 ) % cube_dim  == rel_j_y and (rel_i_y == 0 or rel_i_y == cube_dim - 1):
            #         print(f'\t\tpossible y conn to {j} ({j_x},{j_y},{j_z}) ({rel_j_x},{rel_j_y},{rel_j_z})')
            #         print(f'\t\tr_map[{i}][{j}]={r_map[i][j]}')
            #     if (rel_i_x,rel_i_y) == (rel_j_x,rel_j_y) and (rel_i_z + cube_dim - 1 ) % cube_dim  == rel_j_z and (rel_i_z == 0 or rel_i_z == cube_dim - 1):
            #         print(f'\t\tpossible z conn to {j} ({j_x},{j_y},{j_z}) ({rel_j_x},{rel_j_y},{rel_j_z})')
            #         print(f'\t\tr_map[{i}][{j}]={r_map[i][j]}')
            to_make_change = input('make conn?')

            try:
                new_dest = int(to_make_change)
                new_conns.append((i,new_dest))
            except:
                pass
            
        
        tot_conns += n_conns
    
    print(f'tot_conns={tot_conns}')

    if len(new_conns) > 0:

        for src, dest in new_conns:
            r_map[src][dest] = 1

        new_name = input('new name: ')

        with open(new_name,'w+') as out_file:
            # out_file.write(str(n_routers) + '\n')
            # out_file.write(str(n_ports) + '\n')
            for row in r_map:
                line = ''
                for elem in row:
                    line += str(elem) + ' '
                line = line[:-1] + '\n'
                out_file.write(line)



    # quit()

def which_cube(i, x_dim, y_dim, z_dim, cube_dim):

    n_x = (x_dim // cube_dim)
    n_y = (y_dim // cube_dim)
    n_z = (z_dim // cube_dim)

    # print(f'n_x,n_y,n_z = {n_x},{n_y},{n_z}')

    i_x,i_y,i_z = r_to_xyz(i,x_dim,y_dim,z_dim)

    # rel_i_x = i_x % cube_dim
    # rel_i_y = i_y % cube_dim
    # rel_i_z = i_z % cube_dim

    n_i_x = i_x // cube_dim
    n_i_y = i_y // cube_dim
    n_i_z = i_z // cube_dim


    n_xy = n_x*n_y

    n_cube = (n_i_z)*n_xy + (n_i_y)*n_x + (n_i_x)

    # print(f'{i} @ ({i_x},{i_y},{i_z}) is cube # {n_cube}')

    return n_cube

def r_to_xyz(r,xd,yd,zd):
    xy_slice_size = xd*yd

    temp_r = r

    z = temp_r // xy_slice_size
    temp_r = temp_r % xy_slice_size
    y = temp_r // xd
    x = temp_r % xd

    return x,y,z

def calc_bi_bw():
    global bisection_bw
    global unidir_bibw
    global bisection_bw_combo

    global n_routers

    print(f'Will calculate bi bw')




    routers = range(0,n_routers)
    all_combos = list()
    half = int(n_routers / 2)
    # all_combos += combinations(routers, half)
    all_combos_generator = combinations(routers, half)

    bibw_dict = {}
    bibws_list = []

    print(f'created combos')
    # quit()

    all_combos = []
    for a_combo in all_combos_generator:
        print(f'a_combo={a_combo}')
        all_combos.append(a_combo)
        break


    least_combo = all_combos[0]
    least_left_combo = all_combos[0]
    # least_combo = [16,17,18,19,20,24,25,26,27,28,32,33,34,35,36,40,41,42,43,44,8,9,10,11]

    cross_link_combo = list()
    for i in range(n_routers):
        if i % 6 < 3:
            cross_link_combo.append(i)

    
        # print(f'i % 3 = {i % 3}')

    least_combo = cross_link_combo
    least_left_combo = cross_link_combo


    (least_bw, least_left_bw) = half_bw(least_combo, vb=verbose)
    if verbose:
        print(f'first combo has bisection bw={least_left_bw} from combo {least_combo}')

    print(f'first combo has bisection bw={least_left_bw} from combo {least_combo}')
    quit()
    n_combos = ncr(n_routers, half)
    print(f'n_combos={n_combos}')
    for i, combo in enumerate(all_combos_generator):

        # new_combo = list()
        # for c in combo:
        #     nc = (c + half) % n_routers
        #     new_combo.append(nc)

        # new_combo = []
        # for c in combo:
        #     nc = c
        #     col = c % 8
        #     row = c // 8
        #     if col >= 4 and row < 3:
        #         col -= 4
        #         row += 3
        #         nc = col + 8*row
        #         # print(f'c->nc = {c}-{nc}')
        #     new_combo.append(nc)

        # combo = new_combo

        (this_bw,this_left_bw) = half_bw(combo)

        bibws_list.append(this_left_bw)


        if i % 10000 == 0:
            print(f'Working on combo # {i:03}/{n_combos:03}')
            print(f'\tthis combo={combo} and left_bw={this_left_bw}')
            print(f'\tlesat left is still {least_left_bw}')
            # input('cont?')

        # # inequality
        # if this_bw - this_left_bw < this_left_bw:
        #     # take complement
        #     this_left_bw = this_bw - this_left_bw
        if this_bw < least_bw:
            if verbose:
                print(f'lesser bisection bw found: {this_bw} < {least_bw}')
                (_,__) = half_bw(combo, vb=True)
            least_bw = this_bw
            least_combo = combo

        if this_left_bw < least_left_bw:
            if verbose:
                print(f'lesser left bisection bw found: {this_left_bw} < {least_left_bw}')
                (_,__) = half_bw(combo, vb=True)

            least_left_bw = this_left_bw
            least_left_combo = combo




    print(f'bisection bandwidth={least_bw} from combo {least_combo}')
    bisection_bw = least_bw
    print(f'left bisection bandwidth={least_left_bw} from combo {least_left_combo}')
    bisection_bw = least_bw
    unidir_bibw = least_left_bw
    bisection_bw_combo = least_left_combo


    plot_bibws_histo(bibws_list)

def plot_bibws_histo(bws):

    bws.sort()

    min_val = min(bws)
    max_val = max(bws)

    n_bins = max_val - min_val + 1
    binwidth = 2

    bibw_dict = {}
    for b in bws:
        try:
            bibw_dict[b] += 1
        except:
            bibw_dict.update({b : 1})

    print(f'bws =...')
    for b,f in bibw_dict.items():
        print(f'{b:02} : {f:03}')


    bin_range = range(min_val, max_val + binwidth + binwidth, binwidth)
    bin_range = range(8, 42, binwidth)

    plt.hist(bws, bins=bin_range, color='skyblue', edgecolor='black')

    # plt.xticks(range(20))



    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Basic Histogram')
    
    plt.show()

def half_bw(routers, vb = False):

    global n_routers

    global r_map

    half = len(routers)
    others = [ x for x in range(0,n_routers) if x not in routers  ]



    if vb:
        print(f'routers: {routers}')
        print(f'others: {others}')

    cross_links = 0

    left_links = 0

    for router in routers:
        for other in others:
            if r_map[router][other] >= 1:
                cross_links += 1
                left_links += 1

                # if vb:
                #     print(f'r_map[{router}][{other}]==1')
    for router in others:
        for other in routers:
            if r_map[router][other] >= 1:
                cross_links += 1

                # if vb:
                #     print(f'r_map[{router}][{other}]==1')

    if vb:
        print(f'routers->others: left_links={left_links}. cross_links={cross_links}')

    return (cross_links,left_links)

def calc_num_links():
    global n_links
    sum = 0

    for p in range(n_routers):

        for q in range(n_routers):

            if p==q:
                continue

            sum += r_map[p][q]

    n_links = sum / 2
    # input(f"n_links={n_links}")


    link_type_count = [0 for _ in range(4)]

    # true for both 20 and 30 router configs
    per_row = 5

    for i in range(n_routers):
        for j in range(n_routers):

            if r_map[i][j] == 0:
                continue

            if r_map[i][j] != 1:
                continue

            i_row = i // per_row
            i_col = i % per_row

            j_row = j // per_row
            j_col = j % per_row

            delta_x = abs(i_col - j_col)
            delta_y = abs(i_row - j_row)

            if delta_x == 0:

                # (0,1)
                if delta_y == 1:
                    link_type_count[0] += 1

                # (0,2)
                if delta_y == 2:
                    link_type_count[2] += 1

            elif delta_x == 1:

                # (1,0)
                if delta_y == 0:
                    link_type_count[0] += 1

                # (1,1)
                if delta_y == 1:
                    link_type_count[1] += 1

                # (1,2)
                if delta_y == 2:
                    link_type_count[3] += 1

            elif delta_x == 2:

                # (2,0)
                if delta_y == 0:
                    link_type_count[2] += 1

                if delta_y == 1:
                    link_type_count[3] += 1

    print(f'link types [(0,1), (1,1), (2,0), (2,1)] = {link_type_count}')
    # quit()

def print_mat(mat):

    for p in range(n_routers):
        if p%5 == 0:
            print('----'*n_routers)

        for q in range(n_routers):
            if q%5 == 0:
                print('|',end='')

            if(mat[p][q] == INF):
                print("INF", end=" ")
            else:
                print(mat[p][q], end="  ")
        print(" ")

    print('----'*n_routers)

# old
def sol_chiplet_dist():

    sum = 0
    n_elems = 0
    for p in range(4):
        if p%2 == 0:
            print('---'*20+'-'*5)

        for q in range(8):
            if q%4 == 0:
                print('|',end='')

            if(chiplet_dist[p][q] == INF):
                print("INF", end=" ")
            else:
                print(chiplet_dist[p][q], end="  ")
                sum += chiplet_dist[p][q]
                n_elems += 1
        print(" ")
    print('---'*20+'-'*5)
    avg_hops = sum / n_elems
    print(f"Sum={sum} and average={avg_hops}")
    print(f"n_elems={n_elems}")
    print('\n')

# old
def calc_chiplet_dists_misaligned():
    global chiplet_dist

    chiplet_to_router_map = {
        0:[0,1,2,5,6,7],
        1:[2,3,4,7,8,9],
        2:[10,11,12,15,16,17],
        3:[12,13,14,17,18,19]
    }
    num_routers_per_chiplet = 6

    mem_to_router_map = {
        0:0,
        1:5,
        2:10,
        3:15,
        4:4,
        5:9,
        6:14,
        7:19
    }

    # 4 chiplets x 8 mem routers
    for chiplet in range(0,4):
        chiplet_dist.append([])

        # default max
        for mem in range(0,8):
            min_path = n_routers
            for chiplet_router in chiplet_to_router_map[chiplet]:
                mem_router = mem_to_router_map[mem]
                if dist[chiplet_router][mem_router] < min_path:
                    min_path = dist[chiplet_router][mem_router]

            chiplet_dist[chiplet].append(1 + min_path)

def calc_avghops_diam_20r_misaligned():

    lefts = [0,5,10,15]
    rights = [4,9,14,19]
    exteriors = lefts + rights
    interiors = [x for x in range(0,n_routers) if x not in exteriors]
    print(f'exterios: {exteriors}')
    print(f'interiors: {interiors}')

    # mems/cpus per interior/exterior node
    mems_ext = 2
    cpus_int = 4
    cpus_ext = 2

    calc_avghops_diam_from_ints_exts(exteriors, interiors, mems_ext, cpus_int, cpus_ext)

def calc_avghops_diam_24r_aligned():

    lefts = [0,6,12,18]
    rights = [5,11,17,23]
    exteriors = lefts + rights
    interiors = [x for x in range(0,n_routers) if x not in exteriors]
    print(f'exterios: {exteriors}')
    print(f'interiors: {interiors}')

    # mems/cpus per interior/exterior node
    mems_ext = 2
    cpus_int = 4
    cpus_ext = 0

    calc_avghops_diam_from_ints_exts(exteriors, interiors, mems_ext, cpus_int, cpus_ext)

def calc_avghops_diam_12r_aligned():

    lefts = [0,4,8]
    rights = [3,7,11]
    exteriors = lefts + rights
    interiors = [x for x in range(0,n_routers) if x not in exteriors]
    print(f'exterios: {exteriors}')
    print(f'interiors: {interiors}')

    # mems/cpus per interior/exterior node
    mems_ext = 2
    cpus_int = 4
    cpus_ext = 0

    calc_avghops_diam_from_ints_exts(exteriors, interiors, mems_ext, cpus_int, cpus_ext)

def calc_avghops_diam_64r_aligned():

    lefts = [x for x in range(n_routers) if x % 8 == 0]
    rights = [x for x in range(n_routers) if x % 8 ==7]
    exteriors = lefts + rights
    interiors = range(0,n_routers)

    # mems/cpus per interior/exterior node
    mems_ext = 1
    cpus_int = 1
    cpus_ext = 0

    calc_avghops_diam_from_ints_exts(exteriors, interiors, mems_ext, cpus_int, cpus_ext)

def calc_avghops_diam_from_ints_exts(exteriors, interiors, mems_ext, cpus_int, cpus_ext):

    global mem_diameter
    global coh_diameter
    global avg_hops

    print(f'memcoh={memcoh}')
    print(f'exteriors: {exteriors}')
    print(f'interiors: {interiors}')

    # memory
    # for all exteriors
    #   for all exteriors
    #       mems_ext*cpus_ext*memcoh*dist[i][j] // e.g. 2*2=4 cpu->dram pairs

    #   for all interiors
    #       mems_ext*cpus_int*memcoh*dist[i][j] // e.g. 2*4=8 cpu->dram pairs

    # coherence
    # for all exteriors
    #   for all exteriors
    #       cpus_ext*cpus_ext*(1-memcoh)*dist[i][j] // e.g. 2*2=4 cpu->cpu pairs

    #   for all interiors
    #       cpus_ext*cpus_int*(1-memcoh)*dist[i][j] // e.g. 2*4=8 cpu->cpu pairs
    # for all interiors
    #   for all exteriors
    #       cpus_int*cpus_ext*(1-memcoh)*dist[i][j] // e.g. 2*4=8 cpu->cpu pairs

    #   for all interiors
    #       cpus_int*cpus_int*(1-memcoh)*dist[i][j] // e.g. 4*4=16 cpu->cpu pairs

    sum = 0
    weighted_elems = 0
    n_paths = 0
    n_router_pairs = 0

    for src in exteriors:
        for dest in exteriors:
            num_pairs = mems_ext*cpus_ext
            weight = memcoh*num_pairs
            weighted_dist = weight*dist[src][dest]
            sum += weighted_dist

            weighted_elems += weight
            n_paths += num_pairs
            n_router_pairs += 1

            if verbose:
                print(f'exterior -> exterior: src={src} -> dest={dest}. num_pairs={num_pairs}. weight={weight}. dist={dist[src][dest]}. sum+={weighted_dist}')

            if dist[src][dest] > mem_diameter:
                mem_diameter = dist[src][dest]

        for dest in interiors:
            num_pairs = mems_ext*cpus_int
            weight = memcoh*num_pairs
            weighted_dist = weight*dist[src][dest]
            sum += weighted_dist
            weighted_elems += weight

            n_paths += num_pairs
            n_router_pairs += 1

            if verbose:
                print(f'exterior -> interior: src={src} -> dest={dest}. num_pairs={num_pairs}. weight={weight}. dist={dist[src][dest]}. sum+={weighted_dist}')

            if dist[src][dest] > mem_diameter:
                mem_diameter = dist[src][dest]

    for src in exteriors:
        for dest in exteriors:
            num_pairs = cpus_ext*cpus_ext
            weight = (1.0-memcoh)*num_pairs
            weighted_dist = weight*dist[src][dest]
            sum += weighted_dist
            weighted_elems += weight

            n_paths += num_pairs
            n_router_pairs += 1

            if verbose:
                print(f'exterior -> exterior: src={src} -> dest={dest}. num_pairs={num_pairs}. weight={weight}. dist={dist[src][dest]}. sum+={weighted_dist}')

            if dist[src][dest] > coh_diameter:
                coh_diameter = dist[src][dest]

        for dest in interiors:
            num_pairs = cpus_ext*cpus_int
            weight = (1.0-memcoh)*num_pairs
            weighted_dist = weight*dist[src][dest]
            sum += weighted_dist
            weighted_elems += weight

            n_paths += num_pairs
            n_router_pairs += 1

            if verbose:
                print(f'exterior -> interior: src={src} -> dest={dest}. num_pairs={num_pairs}. weight={weight}. dist={dist[src][dest]}. sum+={weighted_dist}')

            if dist[src][dest] > coh_diameter:
                coh_diameter = dist[src][dest]

    for src in interiors:
        for dest in exteriors:
            num_pairs = cpus_int*cpus_ext
            weight = (1.0-memcoh)*num_pairs
            weighted_dist = weight*dist[src][dest]
            sum += weighted_dist
            weighted_elems += weight

            n_paths += num_pairs
            n_router_pairs += 1

            if verbose:
                print(f'interior -> exterior: src={src} -> dest={dest}. num_pairs={num_pairs}. weight={weight}. dist={dist[src][dest]}. sum+={weighted_dist}')

            if dist[src][dest] > coh_diameter:
                coh_diameter = dist[src][dest]

        for dest in interiors:
            num_pairs = cpus_int*cpus_int
            weight = (1.0-memcoh)*num_pairs
            weighted_dist = weight*dist[src][dest]
            sum += weighted_dist
            weighted_elems += weight

            n_paths += num_pairs
            n_router_pairs += 1

            if verbose:
                print(f'interior -> interior: src={src} -> dest={dest}. num_pairs={num_pairs}. weight={weight}. dist={dist[src][dest]}. sum+={weighted_dist}')

            if dist[src][dest] > coh_diameter:
                coh_diameter = dist[src][dest]

    avg_hops = sum / weighted_elems
    print(f"Sum={sum} and average={avg_hops}")
    print(f"weighted_elems={weighted_elems}")
    print(f'n_paths={n_paths} and n_router_pairs={n_router_pairs}')
    print(f"mem_diameter={mem_diameter} and coh_diameter={coh_diameter}")


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

    return avg_hops


def calc_nx_avg_hops_diam():

    global avg_hops
    global avg_mem_hops
    global mem_diameter
    global coh_diameter

    avg_hops = calc_avg_hops(adj_mat)

    INF = 999

def calc_simple_avg_hops_diam():

    global avg_hops
    global avg_mem_hops
    global mem_diameter
    global coh_diameter

    diameter = 0

    runsum = 0
    mem_runsum = 0
    n_elems = 0
    n_mem_elems = 0


    lefts = [0,5,10,15]
    rights = [4,9,14,19]
    exteriors = lefts + rights

    for src in range(0,n_routers):
        for dest in range(0,n_routers):
            if(src == dest):
                continue

            this_dist = dist[src][dest]

            # lol
            if verbose:
                print(f'src={src} -> dest={dest}. dist={this_dist}.')



            runsum += this_dist
            n_elems += 1

            if this_dist > diameter:
                diameter = this_dist

            if dest in exteriors:
                mem_runsum += this_dist
                n_mem_elems += 1

                if this_dist > mem_diameter:
                    mem_diameter = this_dist





    avg_hops = runsum / n_elems
    avg_mem_hops = mem_runsum / n_mem_elems
    # mem_diameter = diameter
    coh_diameter = diameter
    print(f"Sum={runsum} and average={avg_hops}")
    print(f"Mem Sum={mem_runsum} and average={avg_mem_hops}")
    print(f"n_elems={n_elems}")
    print(f"diameter={diameter}")
    print(f"mem diameter={mem_diameter}")

def verify_all(dir, out_name,  mask=None):

    print(f'Verifying all in {dir}')

    for root, dirs, files, in os.walk(dir):

        if verbose:
            print(f'root={root}, dirs={dirs}, files={files}')


        # quit()
        for file in files:

            try:
                if mask not in file:
                    # print('mask not in file')
                    continue

            # none
            except Exception as e:
                print(e)
                # continue


            this_path = root + file
            # print(f'within verify_all, this_path={this_path}')
            verify_file(this_path, out_name)

def calc_routes():
    global route_weights
    route_weights = []

    for src in range(n_routers):

        options_weight = [0]*n_routers

        for dest in range(n_routers):
            if src == dest:
                # options[dest] = 0
                continue

            # direct
            if r_map[src][dest] == 1:
                options_weight[dest] = dist[option][dest]
                # options_weight[dest] = 1
            else:
                options_weight[dest] = 0


            # lowest_dist = n_routers
            # best_options = []
            # for option in range(n_routers):
            #     options_weight.append(dist[option][dest])
            #     if r_map[src][option] == 1:
            #         if dist[option][dest] < lowest_dist:
            #             lowest_dist = dist[option][dest]
            #             best_options = [option]
            #         elif dist[option][dest] == lowest_dist:
            #             best_options.append(option)

            # for

        print(options_weight)
        quit()

def calc_vll_mat():

    global n_routers


    weight_mat = []

    # true for both 20 and 30 router configs
    per_row = 5

    for i in range(n_routers):
        weight_mat.append([])
        for j in range(n_routers):

            i_row = i // per_row
            i_col = i % per_row

            j_row = j // per_row
            j_col = j % per_row

            delta_x = abs(i_col - j_col)
            delta_y = abs(i_row - j_row)

            dist = math.sqrt( delta_x*delta_x + delta_y*delta_y)

            # base dist is (1,1)
            one_cycle_dist = math.sqrt(1 + 1)

            # get double
            num_cycles = dist / one_cycle_dist

            # # take upper val
            weight = math.ceil(num_cycles)

            # init to 1
            weight_mat[i].append(weight)
            # weight_mat[i].append(num_cycles)

            # print(f'({i_col},{i_row}) -> ({j_col},{j_row}) = {weight}')

    # print(f'weight_mat init to {weight_mat}')
    # quit()
    return weight_mat

def make_graph_len_depen():
    global graph

    w = calc_vll_mat()

    for i in range(n_routers):
        for j in range(n_routers):
            if graph[i][j] == 1:
                graph[i][j] = w[i][j]

def ingest_path_list(path_name):

    if verbose:
        print(f'Ingesting path list {path_name}')

    path_list = []


    with open(path_name, 'r') as inf:


        for line in inf.readlines():
            if '[' not in line:
                name = inf.readline()
                continue
            # print(f'line (type {type(line)}) = {line}')

            as_list = ast.literal_eval(line)
            clean_as_list = [e for e in as_list]


            path_list.append(clean_as_list)

    n_routers = int(math.sqrt(len(path_list))) #+ 1

    if n_routers % 2 != 0:
        n_routers += 1

    print(f'n_routers={n_routers}')

    return path_list.copy(), n_routers

def hops_of_pl(pl, traf_type='coh'):
    t = 0
    i = 0

    lefts = [0,5,10,15]
    rights = [4,9,14,19]
    exteriors = lefts + rights

    for p in pl:

        # for traf
        d = p[-1]
        if traf_type == 'mem' and d not in exteriors:
            continue

        s = p[0]
        if s == d:
            continue

        i += 1
        for n in p:
            t += 1
        t -= 1

    return t, i

def cload_of_pl(pl, nrs, traf_type='coh'):

    cload = [[0 for _ in range(nrs)] for __ in range(nrs)]
    loaded_paths = [[[] for _ in range(nrs)] for __ in range(nrs)]

    lefts = [0,5,10,15]
    rights = [4,9,14,19]
    exteriors = lefts + rights

    for p in pl:
        p_len = len(p)

        # for traf
        d = p[-1]
        if traf_type == 'mem' and d not in exteriors:
            continue

        for i in range(1,p_len):
            a = p[i-1]
            b = p[i]
            cload[a][b] += 1

            if p not in loaded_paths[a][b]:
                loaded_paths[a][b].append(p)

    return cload.copy(), loaded_paths.copy()

def verify_path_list(pathlist_path, out_name):
    this_pl, n_routers = ingest_path_list(pathlist_path)

    # n_routers = 20

    tot_hops, n_elems = hops_of_pl(this_pl)

    avg = float(tot_hops) / n_elems


    print(f'Path list : {pathlist_path} w/ {n_routers} routers')
    print(f'\ttotal hops : {tot_hops}')
    print(f'\tn_elems : {n_elems}')
    print(f'\t\t=>')
    print(f'\tavg : {avg}')

    print(f'-'*72)

    print(f'Coherence')

    # this_coh_pl = [p for p in this_pl if ]

    cload_mat, associated_paths = cload_of_pl(this_pl, n_routers)
    # cload_mat, associated_paths = cload_of_pl(this_pl, n_routers)

    row_avgs = []
    cmax = 0

    max_pair = (0,0)

    col_sums = [0 for i in range(len(cload_mat))]
    max_col = 0
    max_col_r = -1

    for i,row in enumerate(cload_mat):
        row_avgs.append(statistics.mean(row))
        # print(f'{i:02} : {row}')

        for j,e in enumerate(row):

            col_sums[j] += e

            if cmax < e:
                cmax = e
                max_pair = (i,j)


    # print(f'cload_mat')
    # for i, row in enumerate(cload_mat):
    #     print(f'{i:02} : {row[:10]} - {row[10:]}')

    print(f'col sums={col_sums[:10]} - {col_sums[10:]}')
    print(f'col max={max(col_sums)}')

    print(f'related paths')

    lefts = [0,5,10,15]
    rights = [4,9,14,19]
    exteriors = lefts + rights
    n_mem_paths = 0
    for i, row in enumerate(cload_mat):
        for j, elem in enumerate(row):
            if elem == cmax:
                print(f'heaviest loaded link {i}->{j} traversed by {len(associated_paths[i][j])} paths...')
                for p in associated_paths[i][j]:
                #     print(f'{p}')
                    p_d = p[-1]
                    if p_d in exteriors:
                        n_mem_paths += 1

    print(f'# heavy loaded coh paths = {n_mem_paths}')
    print('')

    print(f'cload avg = {statistics.mean(row_avgs)}')
    print(f'cload maax = {cmax} from {max_pair}')

    print(f"=> throughput = {1/cmax}")



    print(f'avg = {statistics.mean(row_avgs)}')
    # print(f'double check = {cl/n_elems}')


    hops_n_paths_dict = {}

    for i in range(1,5):
        hops_n_paths_dict.update({i:0})

    for p in this_pl:
        # print(f'p={p}')
        p_len = len(p) - 1

        try:
            hops_n_paths_dict[p_len] += 1
        except:
            hops_n_paths_dict.update({p_len : 1})

    print(f'topo {pathlist_path}\n\thops and num paths')
    for h, n in hops_n_paths_dict.items():
        print(f'{h} : {n}')

    print(f'='*72)

def ingest_vn_local(vn_path):
    vn_map = []

    with open(vn_path,'r') as inf:
        for line in inf:
            line.strip('\n')
            line_split = line.split(' ')
            line_split = [l for l in line_split if '\n' not in l]

            vn_map.append([])
            for e in line_split:
                vn_map[-1].append(int(e))
    return vn_map

def prune_pathlist(oned_pl, vn_map, vn_interest):
    new_pl = []
    for p in oned_pl:
        s = p[0]
        d = p[-1]

        if vn_map[s][d] == vn_interest:
            new_pl.append(p)
    return new_pl

def print_pathlist(pl):
    for p in pl:
        s = p[0]
        d = p[-1]
        print(f'{s}->{d} = {p}')

def verify_path_list_w_vn(pathlist_path, vn_path, out_name):

    this_pl, n_routers = ingest_path_list(pathlist_path)

    # n_routers = 20

    tot_hops, n_elems = hops_of_pl(this_pl, traf_type='mem')

    avg = float(tot_hops) / n_elems


    print(f'Path list : {pathlist_path} w/ {n_routers} routers')
    print(f'\ttotal hops : {tot_hops}')
    print(f'\tn_elems : {n_elems}')
    print(f'\t\t=>')
    print(f'\tavg : {avg}')

    cload_mat, associated_paths = cload_of_pl(this_pl, n_routers, traf_type='mem')

    row_avgs = []
    cmax = 0

    max_pair = (0,0)

    for i,row in enumerate(cload_mat):
        row_avgs.append(statistics.mean(row))
        # print(f'{i:02} : {row}')

        for j,e in enumerate(row):

            if cmax < e:
                cmax = e
                max_pair = (i,j)

    print(f'cload_mat')
    for i, row in enumerate(cload_mat):
        print(f'{i:02} : {row[:10]} - {row[10:]}')

    print(f'related paths')

    lefts = [0,5,10,15]
    rights = [4,9,14,19]
    exteriors = lefts + rights
    n_mem_paths = 0
    for i, row in enumerate(cload_mat):
        for j, elem in enumerate(row):
            if elem == cmax:
                print(f'heaviest loaded link {i}->{j} traversed by paths...')
                for p in associated_paths[i][j]:
                    print(f'{p}')
                    p_d = p[-1]
                    if p_d in exteriors:
                        n_mem_paths += 1

    print(f'# heavy loaded mem paths = {n_mem_paths}')
    print('')

    print(f'cload avg = {statistics.mean(row_avgs)}')
    print(f'cload maax = {cmax} from {max_pair}')



    print(f'avg = {statistics.mean(row_avgs)}')
    # print(f'double check = {cl/n_elems}')

    print(f'='*72)

    vn_map = ingest_vn_local(vn_path)

    max_vn = 0
    for row in vn_map:
        max_vn = max([max_vn, max(row)])
    max_vn += 1
    print(f'vn_map {vn_path} has {max_vn} vns')

    for vn in range(max_vn):
        print('-'*72)
        print(f'VN {vn}')

        print(f'before pruning. size={len(this_pl)}')
        # print_pathlist(this_pl)

        vnet_pl = prune_pathlist(this_pl, vn_map, vn)

        print(f'after pruning. size={len(vnet_pl)}')
        # print_pathlist(vnet_pl)


        tot_hops, n_elems = hops_of_pl(vnet_pl, traf_type='mem')

        avg = float(tot_hops) / n_elems

        print(f'\ttotal hops : {tot_hops}')
        print(f'\tn_elems : {n_elems}')
        print(f'\t\t=>')
        print(f'\tavg : {avg}')



        print(f'cload avg = {statistics.mean(row_avgs)}')
        print(f'cload maax = {cmax}')

        # quit()

def verify_file(path, out_name):

    global use_vll
    global no_bi_bw
    global no_sc_bw

    non_empty = ingest(path)

    if not non_empty:
        return

    # #print('map')
    # if n_routers == 20:
    #     print_by_src(r_map, 4,5)


    global graph
    graph = [[item if item==1 else INF for item in row] for row in r_map]

    for i in range(0,n_routers):
        graph[i][i]=0
        r_map[i][i]=0


    if use_vll:
        make_graph_len_depen()


    # calc_nx_avg_hops_diam()


    global avg_hops
    avg_hops = -1
    global n_links
    n_links = -1
    global avg_mem_hops
    avg_mem_hops = -1
    global bisection_bw
    bisection_bw = -1
    global bisection_bw_combo
    bisection_bw_combo = list()
    global vertical_bi_bw
    vertical_bi_bw = -1
    global horizontal_bi_bw
    horizontal_bi_bw = -1


    global unidir_bibw
    unidir_bibw = -1

    global worst_cut
    worst_cut = -1

    global worst_cut_combo
    worst_cut_combo = []

    global worst_mem_cut
    worst_mem_cut = -1

    global worst_mem_cut_combo
    worst_mem_cut_combo = []

    global worst_mem_bi_cut
    worst_mem_bi_cut = -1

    global worst_mem_bi_cut_combo
    worst_mem_bi_cut_combo = []

    global min_cube_bw
    global min_cube_thru
    global x_dim
    global y_dim
    global z_dim
    global cube_dim


    name = path.split('/')[-1]
    name = name.split('.')[0]

    print(name)

    print(f'Will calculate n links')
    calc_num_links()
    input(f'n_links = {n_links}')

    verify(r_map,n_ports)
    input('Verified')

    floyd(graph)
    # global var simple_dist

    if weighted_dist:
        calc_avghops_diam_20r_misaligned()
    else:
        calc_simple_avg_hops_diam()
        # calc_nx_avg_hops_diam()





    cload_thru = (2*n_links)/(avg_hops*n_routers*(n_routers-1))


    if not no_bi_bw:


        # # TODO uncomment
        calc_bi_bw()

    if not no_sc_bw:

        calc_worst_cut()
        # calc_worst_mem_cut()
        # calc_worst_mem_bi_cut()
    # calc_chiplet_dists_misaligned()

    if calc_cube_crossings:
        calc_cube_crossing_bw()
        calc_multi_cube_crossing_bw()

    # if True:
        calc_unused_links()

    print_bw_metrics = True
    if no_sc_bw and no_bi_bw and not calc_cube_crossings:
        print_bw_metrics = False


    # print('chiplet_dist')
    # sol_chiplet_dist()

    print('dist')
    print_mat(dist)

    # calc_routes()


    # print('\nmap')
    # print_mat(r_map)

    # print(f'name\t\t\t\tn_routers\tn_links\tdiameter\tbi_bw\tavg_hops')
    # print(f'{name}\t{n_routers}\t\t{n_links}\t{diameter}\t{bisection_bw}\t{avg_hops}')
    print('='*72)
    print('| Summary Statistics')
    print('='*72)
    # print(f'memcoh          : {memcoh}')
    print(f'name            : {name}')
    print(f'n_routers       : {n_routers}')
    print(f'n_links         : {n_links}')
    # print(f'mem_diameter    : {mem_diameter}')
    print(f'coh_diameter    : {coh_diameter}')
    print(f'avg_hops        : {avg_hops}')
    print(f'cload thru      : {cload_thru}')
    # print(f'avg_mem_hops    : {avg_mem_hops}')

    if(print_bw_metrics):
        print(f'bisection_bw    : {bisection_bw}')
        print(f'left bibw       : {unidir_bibw}')
        print(f'left bibw       : {bisection_bw_combo}')
        # print(f'vert_bi_bw      : {vertical_bi_bw}')
        # print(f'horiz_bi_bw     : {horizontal_bi_bw}')
        print(f'worst/min cut   : {worst_cut} (bw {worst_cut*len(worst_cut_combo)*(n_routers - len(worst_cut_combo))})')
        print(f'  "  " combo    : {worst_cut_combo} (length {len(worst_cut_combo)})')
        # print(f'worst/min mem bi cut   : {worst_mem_bi_cut} (bw {worst_mem_bi_cut*len(worst_mem_bi_cut_combo)*(n_routers - len(worst_mem_bi_cut_combo))})')
        # print(f'  "  " " combo    : {worst_mem_bi_cut_combo} (length {len(worst_mem_bi_cut_combo)})')
        # print(f'worst/min mem cut   : {worst_mem_cut} (bw {worst_mem_cut*len(worst_mem_cut_combo)*(n_routers - len(worst_mem_cut_combo))})')
        # print(f'  "  " " combo    : {worst_mem_cut_combo} (length {len(worst_mem_cut_combo)})')
        print(f'bibw cut        : {unidir_bibw/((n_routers/2)**2)}')

        print(f'cube bw         : {min_cube_bw}')
        print(f'cube thru       : {min_cube_thru}')

    if not no_bi_bw:
        all_bi_combos = calc_bi_combos(unidir_bibw)
        all_sc_combos = calc_sc_combos(worst_cut , len(worst_cut_combo))

        print(f'all bisection combos ({len(all_bi_combos)}) ')
        print(f'all sparsest cut combos ({len(all_sc_combos)}) {all_sc_combos}')

    # out_file_name = 'files/metrics/' + name + '.csv'
    # # hardcode for now
    # out_path = 'files/metrics/' + out_name
    # out_to_csv(name, out_path)

def main():

    parser = argparse.ArgumentParser(description='Verify topology values')
    parser.add_argument('--filename',type=str,help='.map file to evaluate')
    parser.add_argument('--path_list',type=str,help='path list to evaluate')
    parser.add_argument('--vn_map',type=str,help='vn map to analyze. default is none')
    parser.add_argument('--memcoh',type=float,help='ratio of mem to coherence traffic. 1.0=> all mem')
    parser.add_argument('--longest_link',type=str,help='for now do 15ll, 2ll, or 25ll. Convenient')
    parser.add_argument('--verbose',action='store_true',help='debug prints')
    parser.add_argument('--all_in_dir', type=str, help='directory to run visualizer for all files')
    parser.add_argument('--all_mask', type=str, help='only do for files with this str in filename')
    parser.add_argument('--out_file_name', type=str, help='only do for files with this str in filename')
    parser.add_argument('--use_bi_bw',action='store_true',help='')
    parser.add_argument('--use_sc_bw',action='store_true',help='')
    parser.add_argument('--weighted_dist',action='store_true',help='')
    parser.add_argument('--use_vll',action='store_true')
    parser.add_argument('--cube_dim',type=int)
    parser.add_argument('--x_dim',type=int)
    parser.add_argument('--y_dim',type=int)
    parser.add_argument('--z_dim',type=int)
    parser.add_argument('--calc_cube_thru',action='store_true',help='')



    args = parser.parse_args()

    global no_bi_bw
    no_bi_bw = not args.use_bi_bw

    global no_sc_bw
    no_sc_bw = not args.use_sc_bw

    global calc_cube_crossings
    calc_cube_crossings = args.calc_cube_thru

    global cube_dim
    global x_dim
    global y_dim
    global z_dim
    if calc_cube_crossings:
        cube_dim = args.cube_dim
        x_dim = args.x_dim
        y_dim = args.y_dim
        z_dim = args.z_dim


    global verbose
    verbose = args.verbose

    global weighted_dist
    weighted_dist = args.weighted_dist

    global use_vll
    use_vll = args.use_vll

    # override file read
    global memcoh
    memcoh = 0.5
    if args.memcoh is not None:
        memcoh = args.memcoh

    if args.path_list is not None and args.vn_map is not None:
        out_name = 'temp_pl.csv'
        if args.out_file_name is not None:
            out_name = args.out_file_name
        verify_path_list_w_vn(args.path_list,args.vn_map ,out_name)
    elif args.path_list is not None:
        out_name = 'temp_pl.csv'
        if args.out_file_name is not None:
            out_name = args.out_file_name
        verify_path_list(args.path_list, out_name)
    elif args.all_in_dir is not None:
        out_name = 'temp.csv'
        if args.out_file_name is not None:
            out_name = args.out_file_name
        verify_all(args.all_in_dir, out_name, mask=args.all_mask)

    elif args.filename is not None:
        verify_file(args.filename, 'temp.csv')

if __name__ == '__main__':
    main()
