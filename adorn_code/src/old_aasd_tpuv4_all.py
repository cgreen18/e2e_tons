# Copyright (c) 2025 Purdue University
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

# Author(s): Conor Green


# HiGHS libraries
# ---------------
import highspy

# Gurobi libs
# -----------
import gurobipy as gp
from gurobipy import GRB
from gurobipy import Model, LinExpr

# OR-tools libraries
# ------------------
from ortools.linear_solver.python import model_builder
from ortools.math_opt.python import mathopt
from ortools.math_opt.python import solve as mathopt_solve

# imports
# -------
import numpy as np
import networkx as nx

# regular libs
# ------------
import argparse
import os
from copy import deepcopy
import time
import random
import re
import psutil



# constants
# ---------
global VERBOSE
VERBOSE = False

global SLOW_RUN
SLOW_RUN = False

global ASSERT_BINARY_MAP
ASSERT_BINARY_MAP = True

global MODEL_DIR
MODEL_DIR = "/scratch/negishi/green456/models"

global TIMELINE_DIR
TIMELINE_DIR = "./files/timeline_logs"


INF=999999


# Regular Python Functions
###############################################################################

def console_log(msg,msg_type,n_indents=0,min_msg_type_len=20):
    print('+ '+f'{msg_type}'.ljust(min_msg_type_len)+': '+'\t'*n_indents+f'{msg}')

def timeline_log(n_opt_links, avg_hops, cur_asc_obj, model_base_name, first_write=False, delimiter=','):
    
    global TIMELINE_DIR

    log_name = f'{model_base_name}_timeline.csv'
    log_path = os.path.join(TIMELINE_DIR,log_name)

    if first_write:
        with open(log_path,'w+') as of:
            out_line = 'time,n_opt_links,avg_hops,cur_asc_obj\n'
            of.write(out_line)
        return

    with open(log_path,'a') as of:
        out_line = f'{time.time()},{n_opt_links},{avg_hops},{cur_asc_obj}\n'
        of.write(out_line)

    console_log(f'Wrote to timeline {log_path}','STATUS')

def print_rss(label=""):
    rss = psutil.Process(os.getpid()).memory_info().rss  # bytes
    return f"{label}RSS = {rss/2**20:.1f} MiB"

def output_r_map(out_map, out_path, assert_binary=False):

    n_routers = len(out_map)

    out_lines = []
    for sr in range(n_routers):
        this_line = []
        for dr in range(n_routers):
            val = out_map[sr][dr]
            if assert_binary:
                val = min(1, round(val))
            this_line.append(f'{val} ')

        this_line.append('\n')
        out_lines.append(this_line)

    with open(out_path, 'w+') as of:
        for line_list in out_lines:
            of.write(''.join(line_list))

    print(f'Wrote out map to {out_path}')

def ingest_map(path_name):
    global VERBOSE

    file_name = path_name.split('/')[-1]

    if VERBOSE:
        print(f'Ingesting filename = {file_name} ({path_name})')

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


    # quick sanitization
    n_routers = len(this_map)
    for i in range(n_routers):
        this_map[i][i] = 0

    # assert binary?
    if ASSERT_BINARY_MAP:
        for src_map in this_map:
            for conn in src_map:
                # instead, make binary
                if conn > 0:
                    conn = 1
                # assert(conn == 1 or conn == 0)

    if VERBOSE:
        print(f'read {this_map}')

    return this_map

def create_an_nwx_G_from_adj_mat(this_map):

    n_routers = len(this_map)
    G = nx.DiGraph()

    for src in range(n_routers):
        for dest in range(src+1,n_routers):

            if(this_map[src][dest] < 1):
                continue

            G.add_edge(src,dest)
            G.add_edge(dest,src)

    return G

def calc_avg_hops(adj_mat):
    G = create_an_nwx_G_from_adj_mat(adj_mat)

    try:
        avg_hops = nx.average_shortest_path_length(G)
    except:
        avg_hops = INF
    
    return avg_hops

def calc_diameter(adj_mat):

    G = create_an_nwx_G_from_adj_mat(adj_mat)

    return nx.diameter(G)

def calc_all_pairs_hops(adj_mat):

    G = create_an_nwx_G_from_adj_mat(adj_mat)

    return nx.all_pairs_bellman_ford_path_length(G)

class UpperTriMatrix:
    def __init__(self, n):
        """
        Store only the upper-triangular (i < j) entries of an n x n matrix.
        Each entry can hold an arbitrary Python object.
        """
        self.n = n
        self.size = n * (n - 1) // 2
        self.data = [None] * self.size  # Python objects

    def _row_start(self, i):
        return i * (self.n - 1) - (i * (i - 1)) // 2

    def _index(self, i, j):
        if i == j:
            raise IndexError("Diagonal elements are not stored")
        if i > j:
            i, j = j, i
        return self._row_start(i) + (j - i - 1)

    def _decode_index(self, idx):
        n = self.n
        # find i such that row_start(i) <= idx < row_start(i+1)
        lo, hi = 0, n - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self._row_start(mid + 1) <= idx:
                lo = mid + 1
            else:
                hi = mid
        i = lo
        j = i + 1 + (idx - self._row_start(i))
        return (i, j)


    def __getitem__(self, key):
        i, j = key
        return self.data[self._index(i, j)]

    def __setitem__(self, key, value):
        i, j = key
        self.data[self._index(i, j)] = value

class SymTriDict(dict):
    @staticmethod
    def _n(key):
        i, j, k = key
        return (i, j, k) if i <= j else (j, i, k)
    def __getitem__(self, key): return super().__getitem__(self._n(key))
    def __setitem__(self, key, val): super().__setitem__(self._n(key), val)
    def __delitem__(self, key): return super().__delitem__(self._n(key))
    def __contains__(self, key): return super().__contains__(self._n(key))

# TPUv4 Functions
###############################################################################


# Basic
# --------------------------------------------------------------------------------

def r_to_xyz(r,xd,yd,zd):
    xy_slice_size = xd*yd

    temp_r = r

    z = temp_r // xy_slice_size
    temp_r = temp_r % xy_slice_size
    y = temp_r // xd
    x = temp_r % xd

    return x,y,z

def xyz_to_r(x,y,z,xd,yd,zd):
    return x + y*xd + z*xd*yd

def rel_xyz_is_on_face(rel_x, rel_y, rel_z, cube_dim):

    if rel_x == 0 or rel_x == cube_dim - 1:
        return True
    if rel_y == 0 or rel_y == cube_dim - 1:
        return True
    if rel_z == 0 or rel_z == cube_dim - 1:
        return True
    return False

def r_to_rel_xyz(r, cube_dim, x_dim, y_dim, z_dim):

    r_x,r_y,r_z = r_to_xyz(r, x_dim, y_dim, z_dim)

    rel_r_x = r_x % cube_dim
    rel_r_y = r_y % cube_dim
    rel_r_z = r_z % cube_dim

    return rel_r_x, rel_r_y, rel_r_z

def iter_rel_xyz_across_cubes(rel_x,rel_y,rel_z,cube_dim, x_dim, y_dim, z_dim):

    n_x_cubes = x_dim // cube_dim
    n_y_cubes = y_dim // cube_dim
    n_z_cubes = z_dim // cube_dim

    targs = []

    for xc in range(n_x_cubes):
        for yc in range(n_y_cubes):
            for zc in range(n_z_cubes):

                xprime = rel_x + cube_dim*xc
                yprime = rel_y + cube_dim*yc
                zprime = rel_z + cube_dim*zc
                targ = xyz_to_r(xprime, yprime, zprime, x_dim, y_dim, z_dim)
                targs.append(targ)
    
    return targs

def optical_to_r_map(xyzc_dims,optical_conns):

    (x_dim, y_dim, z_dim, cube_dim) = xyzc_dims
    n_routers = x_dim*y_dim*z_dim

    electrical_conns_adj_list = init_electrical_conns(xyzc_dims)

    r_map = [[0 for _ in range(n_routers)] for __ in range(n_routers)]

    for i, conns in enumerate(electrical_conns_adj_list):
        for j in conns:
            r_map[i][j] = 1
            r_map[j][i] = 1

    for (i,j) in optical_conns:
        r_map[i][j] = 1
        r_map[j][i] = 1
    
    return r_map

def r_map_to_optical(xyzc_dims, r_map):

    optical_conns = []

    (x_dim, y_dim, z_dim, cube_dim) = xyzc_dims
    n_routers = x_dim*y_dim*z_dim

    electrical_conns_adj_list = init_electrical_conns(xyzc_dims)

    for i in range(n_routers):
        for j in range(i+1,n_routers):
            if r_map[i][j] > 0 and j not in electrical_conns_adj_list[i]:
                optical_conns.append((i,j))
                optical_conns.append((j,i))
    return optical_conns

# More Complicated
# --------------------------------------------------------------------------------

def poss_optical_conns_for_r(r, xyzc_dims):
    global VERBOSE

    (x_dim, y_dim, z_dim, cube_dim) = xyzc_dims

    rel_r_x, rel_r_y, rel_r_z = r_to_rel_xyz(r, cube_dim,x_dim, y_dim, z_dim)

    n_x_cubes = x_dim // cube_dim
    n_y_cubes = y_dim // cube_dim
    n_z_cubes = z_dim // cube_dim

    poss_conns = {'x+':[],'x-':[],'y+':[],'y-':[],'z+':[],'z-':[]}

    # x+
    if rel_r_x == cube_dim - 1:
        poss_conns['x+'] += iter_rel_xyz_across_cubes(0, rel_r_y, rel_r_z,cube_dim, x_dim, y_dim, z_dim)
    # x-
    if rel_r_x == 0:
        poss_conns['x-'] += iter_rel_xyz_across_cubes(cube_dim - 1, rel_r_y, rel_r_z,cube_dim, x_dim, y_dim, z_dim)

    # y+
    if rel_r_y == cube_dim - 1:
        poss_conns['y+'] += iter_rel_xyz_across_cubes(rel_r_x, 0, rel_r_z,cube_dim, x_dim, y_dim, z_dim)
    # y-
    if rel_r_y == 0:
        poss_conns['y-'] += iter_rel_xyz_across_cubes(rel_r_x, cube_dim - 1, rel_r_z,cube_dim, x_dim, y_dim, z_dim)

    # z+
    if rel_r_z == cube_dim - 1:
        poss_conns['z+'] += iter_rel_xyz_across_cubes(rel_r_x, rel_r_y, 0,cube_dim, x_dim, y_dim, z_dim)
    # z-
    if rel_r_z == 0:
        poss_conns['z-'] += iter_rel_xyz_across_cubes(rel_r_x, rel_r_y, cube_dim - 1,cube_dim, x_dim, y_dim, z_dim)
    

    if VERBOSE:
        print(f'{r} may conn to')
        for k,v in poss_conns.items():
            print(f'\t{k} : {v}')
    
        # input('good?')

    return poss_conns

def init_electrical_conns(xyzc_dims):

    (x_dim, y_dim, z_dim, cube_dim) = xyzc_dims
    n_routers = x_dim*y_dim*z_dim

    # electrical_conns = [[0 for _ in range(n_routers)] for __ in range(n_routers)]
    # electrical_conns = {}
    electrical_conns_adj_list = [[] for _ in range(n_routers)]

    for src in range(n_routers):

        src_x,src_y,src_z = r_to_xyz(src, x_dim, y_dim, z_dim)

        # xpos
        # if not on edge then conn
        if(src_x % cube_dim != cube_dim - 1):
            targ_x = src_x + 1
            targ_y = src_y
            targ_z = src_z
            targ = xyz_to_r(targ_x, targ_y, targ_z, x_dim, y_dim, z_dim)
            # electrical_conns[(src,targ)] = 1
            # electrical_conns.append( (src,targ) )
            electrical_conns_adj_list[src].append(targ)

        # xneg
        # if not on edge then conn
        if(src_x % cube_dim != 0):
            targ_x = src_x - 1
            targ_y = src_y
            targ_z = src_z
            targ = xyz_to_r(targ_x, targ_y, targ_z, x_dim, y_dim, z_dim)
            # electrical_conns[(src,targ)] = 1
            # electrical_conns.append( (src,targ) )
            electrical_conns_adj_list[src].append(targ)


        # ypos
        # if not on edge then conn
        if(src_y % cube_dim != cube_dim - 1):
            targ_x = src_x
            targ_y = src_y + 1
            targ_z = src_z
            targ = xyz_to_r(targ_x, targ_y, targ_z, x_dim, y_dim, z_dim)
            # electrical_conns[(src,targ)] = 1
            # electrical_conns.append( (src,targ) )
            electrical_conns_adj_list[src].append(targ)

        # yneg
        # if not on edge then conn
        if(src_y % cube_dim != 0):
            targ_x = src_x
            targ_y = src_y - 1
            targ_z = src_z
            targ = xyz_to_r(targ_x, targ_y, targ_z, x_dim, y_dim, z_dim)
            # electrical_conns[(src,targ)] = 1
            # electrical_conns.append( (src,targ) )
            electrical_conns_adj_list[src].append(targ)


        # zpos
        # if not on edge then conn
        if(src_z % cube_dim != cube_dim - 1):
            targ_x = src_x
            targ_y = src_y
            targ_z = src_z + 1
            targ = xyz_to_r(targ_x, targ_y, targ_z, x_dim, y_dim, z_dim)
            # electrical_conns[(src,targ)] = 1
            # electrical_conns.append( (src,targ) )
            electrical_conns_adj_list[src].append(targ)

        # zneg
        # if not on edge then conn
        if(src_z % cube_dim != 0):
            targ_x = src_x
            targ_y = src_y
            targ_z = src_z - 1
            targ = xyz_to_r(targ_x, targ_y, targ_z, x_dim, y_dim, z_dim)
            # electrical_conns[(src,targ)] = 1
            # electrical_conns.append( (src,targ) )
            electrical_conns_adj_list[src].append(targ)

    return electrical_conns_adj_list

def calc_valid_conns(xyzc_dims, known_optical_conns):

    global VERBOSE

    # VERBOSE = True

    (x_dim, y_dim, z_dim, cube_dim) = xyzc_dims
    n_routers = x_dim*y_dim*z_dim
    electrical_conns_adj_list = init_electrical_conns(xyzc_dims)

    # 1) assume all invalid
    # 2) all electrical valid
    # 3) assume all optical valid
    # 4) remove all forms of known optical

    # 1) assume invalid
    valid_conns = [[0 for _ in range(n_routers)] for __ in range(n_routers)]

    # 2) update the electrical conns
    for i, conns in enumerate(electrical_conns_adj_list):
        for j in conns:
            valid_conns[i][j] = 1
            # print(f'added electrical {((i,j))}')

    # 3)
    for i in range(n_routers):
        i_x,i_y,i_z = r_to_xyz(i,x_dim,y_dim,z_dim)
        rel_i_x = i_x % cube_dim
        rel_i_y = i_y % cube_dim
        rel_i_z = i_z % cube_dim

        # early exit for inner routers
        if not rel_xyz_is_on_face(rel_i_x, rel_i_y, rel_i_z, cube_dim):
            continue

        poss_conns = poss_optical_conns_for_r(i, xyzc_dims)

        for direction, conns in poss_conns.items():
            for conn in conns:
                valid_conns[i][conn] = 1
                valid_conns[conn][i] = 1

                if VERBOSE:
                    print(f'\t\t{i}<->{conn} (potentially) allowed')

    # 4)

    if VERBOSE:
        print(f'init valid conns. known opticals {known_optical_conns}')

    for (src, dest) in known_optical_conns:

        src_dest_direction = calc_conn_type(src, dest, xyzc_dims)
        dest_src_direction = calc_conn_type(dest, src, xyzc_dims)

        # 4a)
        # src <-\-> possible[src]
        src_poss_conns = poss_optical_conns_for_r(src, xyzc_dims)
        for conn in src_poss_conns[src_dest_direction]:
            if conn == dest:
                continue
            valid_conns[src][conn] = 0
            valid_conns[conn][src] = 0
        
        # 4b)
        # dest <-\-> possible[dest]
        dest_poss_conns = poss_optical_conns_for_r(dest, xyzc_dims)
        for conn in dest_poss_conns[dest_src_direction]:
            if conn == src:
                continue
            valid_conns[dest][conn] = 0
            valid_conns[conn][dest] = 0
    
    return valid_conns

def calc_conn_type(s, d, xyzc_dims):
    (x_dim, y_dim, z_dim, cube_dim) = xyzc_dims

    rel_s_x, rel_s_y, rel_s_z = r_to_rel_xyz(s, cube_dim, x_dim, y_dim, z_dim)
    rel_d_x, rel_d_y, rel_d_z = r_to_rel_xyz(d, cube_dim, x_dim, y_dim, z_dim)

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

def calc_node_types(s, xyzc_dims):

    (x_dim, y_dim, z_dim, cube_dim) = xyzc_dims

    rel_s_x, rel_s_y, rel_s_z = r_to_rel_xyz(s, cube_dim, x_dim, y_dim, z_dim)

    node_types = []

    # x+
    if rel_s_x == cube_dim - 1:
        node_types.append( 'x+')
    # x-
    if rel_s_x == 0:
        node_types.append( 'x-')

    # y+
    if rel_s_y == cube_dim - 1:
        node_types.append( 'y+')
    # y-
    if rel_s_y == 0:
        node_types.append( 'y-')

    # z+
    if rel_s_z == cube_dim - 1:
        node_types.append( 'z+')
    # z-
    if rel_s_z == 0:
        node_types.append( 'z-')

    return node_types

def conn_set_for_rel_xyz(rel_xyz, dimension, cube_dim, x_dim, y_dim, z_dim):

    rel_x, rel_y, rel_z = rel_xyz
    assert((rel_x == 0 or rel_x == cube_dim -1) or (rel_y == 0 or rel_y == cube_dim -1) or (rel_z == 0 or rel_z == cube_dim -1))

    conn_set = []
    if 'x' in dimension:
        conn_set += iter_rel_xyz_across_cubes(0,            rel_y,          rel_z,      cube_dim, x_dim, y_dim, z_dim)
        conn_set += iter_rel_xyz_across_cubes(cube_dim-1,   rel_y,          rel_z,      cube_dim, x_dim, y_dim, z_dim)
    elif 'y' in dimension:
        conn_set += iter_rel_xyz_across_cubes(rel_x,        0,              rel_z,      cube_dim, x_dim, y_dim, z_dim)
        conn_set += iter_rel_xyz_across_cubes(rel_x,        cube_dim-1,     rel_z,      cube_dim, x_dim, y_dim, z_dim)
    elif 'z' in dimension:
        conn_set += iter_rel_xyz_across_cubes(rel_x,        rel_y,          0,          cube_dim, x_dim, y_dim, z_dim)
        conn_set += iter_rel_xyz_across_cubes(rel_x,        rel_y,          cube_dim-1, cube_dim, x_dim, y_dim, z_dim)

    assert(len(conn_set) > 0)
    return conn_set

def priority_removal(chosen_optical_conns, n_to_remove, xyzc_dims):

    current_r_map = optical_to_r_map(xyzc_dims, chosen_optical_conns )

    (x_dim, y_dim, z_dim, cube_dim) = xyzc_dims
    n_routers = x_dim*y_dim*z_dim

    diameter = calc_diameter(current_r_map)

    rel_xyzs_to_remove = set()

    conns_to_remove = []

    n_removed = 0
    for less_than_diameter in range(diameter - 1):
        goal = diameter - less_than_diameter

        dist_iterator = calc_all_pairs_hops(current_r_map)

        for src, hops_dict in dist_iterator:
            for dest in range(n_routers):

                # cant do more than 16?
                if len(rel_xyzs_to_remove) >= n_to_remove:
                    return conns_to_remove, current_r_map

                src_rel_xyz = r_to_rel_xyz(src, cube_dim, x_dim, y_dim, z_dim)
                dest_rel_xyz = r_to_rel_xyz(dest, cube_dim, x_dim, y_dim, z_dim)

                # skip electricals as sources or destinations
                if not rel_xyz_is_on_face(src_rel_xyz[0], src_rel_xyz[1], src_rel_xyz[2], cube_dim) or \
                    not rel_xyz_is_on_face(dest_rel_xyz[0], dest_rel_xyz[1], dest_rel_xyz[2], cube_dim):
                    continue

                # # skip conns not aligned by direction?
                # if opt_dimension != calc_conn_type_no_direction(src, dest, xyzc_dims):
                #     continue

                # previously removed
                if src_rel_xyz in rel_xyzs_to_remove and dest_rel_xyz in rel_xyzs_to_remove:
                    continue

                try:
                    hop_dist = hops_dict[dest]
                except:
                    # key error => infinite/unconnected
                    hop_dist = INF

                if hop_dist == goal:
                    console_log(f'Removing all conns from {src}->{dest}, xyzs {src_rel_xyz}->{dest_rel_xyz}','MODEL UPDATE')
                    s_opt_dimensions = calc_node_types(src, xyzc_dims)
                    d_opt_dimensions = calc_node_types(dest, xyzc_dims)


                    # remove this
                    rel_xyzs_to_remove.add(src_rel_xyz)

                    for opt_dimension in s_opt_dimensions:
                        src_conn_set = conn_set_for_rel_xyz(src_rel_xyz, opt_dimension, cube_dim, x_dim, y_dim, z_dim)
                        for a in src_conn_set:
                            for b in src_conn_set:
                                if a==b:
                                    continue
                                current_r_map[a][b] = 0
                                if (a,b) not in conns_to_remove:
                                    conns_to_remove.append((a,b))
                                    conns_to_remove.append((b,a))

                    # exit earlys
                    if len(rel_xyzs_to_remove) >= n_to_remove:
                        return conns_to_remove, current_r_map

                    rel_xyzs_to_remove.add(dest_rel_xyz)
                    for opt_dimension in d_opt_dimensions:

                        dest_conn_set = conn_set_for_rel_xyz(dest_rel_xyz, opt_dimension, cube_dim, x_dim, y_dim, z_dim)

                        for a in dest_conn_set:
                            for b in dest_conn_set:
                                if a==b:
                                    continue
                                current_r_map[a][b] = 0
                                if (a,b) not in conns_to_remove:
                                    conns_to_remove.append((a,b))
                                    conns_to_remove.append((b,a))

    return conns_to_remove, current_r_map

def which_cube(i, xyzc_dims):

    (x_dim, y_dim, z_dim, cube_dim) = xyzc_dims


    n_x = (x_dim // cube_dim)
    n_y = (y_dim // cube_dim)
    n_z = (z_dim // cube_dim)

    # print(f'n_x,n_y,n_z = {n_x},{n_y},{n_z}')

    i_x,i_y,i_z = r_to_xyz(i,x_dim,y_dim,z_dim)

    n_i_x = i_x // cube_dim
    n_i_y = i_y // cube_dim
    n_i_z = i_z // cube_dim


    n_xy = n_x*n_y

    n_cube = (n_i_z)*n_xy + (n_i_y)*n_x + (n_i_x)

    # print(f'{i} @ ({i_x},{i_y},{i_z}) is cube # {n_cube}')

    return n_cube

def chosen_optical_to_cubes(chosen_optical_conns, xyzc_dims):

    chosen_cube_conns = set()
    for (i,j) in chosen_optical_conns:
        i_cube = which_cube(i, xyzc_dims)
        j_cube = which_cube(j, xyzc_dims)
        chosen_cube_conns.add( (i_cube, j_cube) )
        chosen_cube_conns.add( (j_cube, i_cube) )

    return chosen_cube_conns

def get_max_valued_r_map_val(r_map_vals, valid_conns, chosen_optical_conns, xyzc_dims, chosen_cube_conns, spreading_cubes):

    (x_dim, y_dim, z_dim, cube_dim) = xyzc_dims
    n_routers = x_dim*y_dim*z_dim

    electrical_conns_adj_list = init_electrical_conns(xyzc_dims)

    max_val = -1
    max_conn = None

    for i in range(n_routers):
        for j in range(i+1,n_routers):

            if valid_conns[i][j] == 0:
                continue
            if j in electrical_conns_adj_list[i]:
                continue
            if (i,j) in chosen_optical_conns:
                continue

            i_cube = which_cube(i, xyzc_dims)
            j_cube = which_cube(j, xyzc_dims)
            if spreading_cubes and ((i_cube, j_cube) in chosen_cube_conns or i_cube == j_cube):
                continue

            val = r_map_vals[i,j]
            # try:
            #     val = r_map_vals[i,j]
            # except:
            #     val = -1

            if val > max_val:
                max_val = val
                max_conn = (i,j)

    return max_conn, max_val

# Solver Functions
###############################################################################

def calc_model_type(model):

    if isinstance(model, highspy.Highs):
        return 'highs'
    elif isinstance(model, gp.Model):
        return 'gurobi'
    elif isinstance(model, model_builder.ModelBuilder):
        return 'ortools_model_builder'
    elif isinstance(model, mathopt.Model):
        return 'ortools'

    # else
    print(f'ERROR: Unrecognized model type for {model} of type {type(model)}. Exiting...')
    quit()

def add_constr_sum_equality(model, lhs_vars, rhs_val, name):

    # errors on unrecognized type
    if calc_model_type(model) == 'highs':
        model.addConstr(model.qsum(lhs_vars) == rhs_val, name=name)
    elif calc_model_type(model) == 'gurobi':
        model.addConstr(gp.quicksum(lhs_vars) == rhs_val, name=name)
    elif calc_model_type(model) == 'ortools_model_builder':
        model.add(sum(lhs_vars) == rhs_val, name=name)
    elif calc_model_type(model) == 'ortools':
        model.add_linear_constraint(sum(lhs_vars) == rhs_val, name=name)
    return model

def add_constr_pos_neg_sum_lte(model, pos_lhs_vars, neg_lhs_vars, rhs_val, name):
    # errors on unrecognized type
    if calc_model_type(model) == 'highs':
        model.addConstr(model.qsum(pos_lhs_vars) - model.qsum(neg_lhs_vars) <= rhs_val, name=name)
    elif calc_model_type(model) == 'gurobi':
        model.addConstr(gp.quicksum(pos_lhs_vars) - gp.quicksum(neg_lhs_vars) <= rhs_val, name=name)
    elif calc_model_type(model) == 'ortools_model_builder':
        model.add(sum(pos_lhs_vars) - sum(neg_lhs_vars) <= rhs_val, name=name)
    elif calc_model_type(model) == 'ortools':
        model.add_linear_constraint(sum(pos_lhs_vars) - sum(neg_lhs_vars) <= rhs_val, name=name)
    return model

# different enough use whole separate function
def _translate_r_map_into_values_twod_highs(model, r_map_vars, n_routers,var_name_to_index_map=None):
    r_map = UpperTriMatrix(n_routers)

    solution = model.getSolution()
    solution_vals = solution.col_value

    if False: #var_name_to_index_map is None:
        lp = model.getLp()
        name_to_val = {name: solution.col_value[i] for i, name in enumerate(lp.col_names_) if 'r_map' in name}
        for r_map_name, val in name_to_val.items():
            # print(f'r_map_name = {r_map_name} = {val}')
            match = re.fullmatch(r'var_r_map_(-?\d+)i_(-?\d+)j', r_map_name).groups()
            i,j = map(int, match)

            r_map[i,j] = val
    else:
        for i in range(n_routers):
            for j in range(i+1, n_routers):

                if type(r_map_vars[i,j]) == int or type(r_map_vars[i,j]) == float:
                    continue

                # r_map[i,j] = solution[r_map_vars[i,j].index]
                # print(f'myvarname = {myvarname}, idx = {idx}, val = {val}')

                # myvarname = f'var_r_map_{i}i_{j}j'
                # print(f'{myvarname}')
                try:
                    # idx = var_name_to_index_map[myvarname]
                    # val = solution[idx]
                    # r_map[i,j] = val
                    idx = r_map_vars[i,j].index
                    r_map[i,j] = solution_vals[idx]
                    # print(f'myvarname = {myvarname}, idx = {idx}, val = {val}')
                except Exception as e:
                    # print(f'myvarname = {myvarname}. Exception ')
                    pass


    return r_map

def _translate_r_map_into_values_twod_gurobi(model,r_map_vars, n_routers):

    r_map = UpperTriMatrix(n_routers)

    for i in range(n_routers):
        for j in range(i+1, n_routers):

            # myvarname = f'var_r_map_{i}i_{j}j'

            try:
                # v = model.getVarByName(myvarname)
                # val = v.X
                # r_map[i,j] = val
                r_map[i,j] = r_map_vars[i,j].X
            except Exception as e:
                pass

    return r_map

def _translate_r_map_into_values_twod_ortools_model_builder(model, solver, r_map_vars, n_routers):

    r_map = UpperTriMatrix(n_routers)

    for i in range(n_routers):
        for j in range(i+1, n_routers):

            # myvarname = f'var_r_map_{i}i_{j}j'

            if type(r_map_vars[i,j]) == float or type(r_map_vars[i,j])==int:
                continue

            try:
                # v = model.getVarByName(myvarname)
                # val = v.X
                # r_map[i,j] = val
                r_map[i,j] = solver.value( r_map_vars[i,j])
            except Exception as e:
                input(f'r_map[{i},{j}] exception {e}')
                pass

    return r_map

def _translate_r_map_into_values_twod_ortools(model, solver_or_result, r_map_vars, n_routers):

    r_map = UpperTriMatrix(n_routers)

    for i in range(n_routers):
        for j in range(i+1, n_routers):

            # myvarname = f'var_r_map_{i}i_{j}j'

            if type(r_map_vars[i,j]) == float or type(r_map_vars[i,j])==int:
                continue

            try:
                # v = model.getVarByName(myvarname)
                # val = v.X
                # r_map[i,j] = val
                r_map[i,j] = solver_or_result.variable_values()[ r_map_vars[i,j]]
            except Exception as e:
                input(f'r_map[{i},{j}] exception {e}')
                pass

    return r_map

def translate_r_map_into_values_twod(model, solver_or_result, r_map_vars, n_routers,var_name_to_index_map=None):

    # errors on unrecognized type
    if calc_model_type(model) == 'highs':
        return _translate_r_map_into_values_twod_highs(model,r_map_vars, n_routers,var_name_to_index_map=var_name_to_index_map)
    elif calc_model_type(model) == 'gurobi':
        return _translate_r_map_into_values_twod_gurobi(model,r_map_vars, n_routers)
    elif calc_model_type(model) == 'ortools_model_builder':
        return _translate_r_map_into_values_twod_ortools_model_builder(model, solver_or_result, r_map_vars, n_routers)
    elif calc_model_type(model) == 'ortools':
        return _translate_r_map_into_values_twod_ortools(model, solver_or_result, r_map_vars, n_routers)
    else:
        print(f'ERROR :: translate_r_map_into_values_twod :: Unimplemetedn. Exiting...')
        quit()

def _translate_tri_ineq_wyes_into_values_threed_gurobi(model, tri_ineq_wye_vars, n_routers):

    tri_ineq_wyes = SymTriDict()

    for i in range(n_routers):
        for j in range(i+1,n_routers):
            for k in range(n_routers):
                if i==k or j==k:
                    continue

            myvarname = tri_ineq_wye_name(i,j,k)

            try:
                # v = model.getVarByName(myvarname)
                # val = v.X
                tri_ineq_wyes[i,j,k] = tri_ineq_wye_vars[i,j,k].X#val
            except Exception as e:
                pass

    return tri_ineq_wyes

def _translate_tri_ineq_wyes_into_values_threed_highs(model, tri_ineq_wye_vars, n_routers,var_name_to_index_map=None):

    tri_ineq_wyes = SymTriDict()

    solution = model.getSolution()
    solution_vals = solution.col_value

    # if var_name_to_index_map is None:
    #     lp = model.getLp()
    #     name_to_val = {name: solution.col_value[i] for i, name in enumerate(lp.col_names_) if 'tri_ineq_wye' in name}
    #     for tri_ineq_wye_name, val in name_to_val.items():
    #         # print(f'r_map_name = {r_map_name} = {val}')
    #         match = re.fullmatch(r'var_tri_ineq_wye_(-?\d+)i_(-?\d+)j_(-?\d+)k', tri_ineq_wye_name).groups()
    #         i,j,k = map(int, match)

    #         tri_ineq_wyes[i,j,k] = val

    if True:
    #     for i in range(n_routers):
    #         for j in range(i+1,n_routers):
    #             for k in range(n_routers):
    #                 if i==k or j==k:
    #                     continue
        for (i,j,k), var in tri_ineq_wye_vars.items():

                # if type(tri_ineq_wye_vars[i,j,k]) == int or type(tri_ineq_wye_vars[i,j,k]) == float:
                #     continue

                # myvarname = f'var_tri_ineq_wye_{i}i_{j}j_{k}k'

            try:
                # idx = var_name_to_index_map[myvarname]
                # val = solution[idx]
                # idx = tri_ineq_wye_vars[i,j,k].index
                idx = var.index
                tri_ineq_wyes[i,j,k] = solution_vals[idx]
            except Exception as e:
                pass

    else:
        for i in range(n_routers):
            for j in range(i+1,n_routers):
                for k in range(n_routers):
                    if i==k or j==k:
                        continue

                myvarname = tri_ineq_wye_name(i,j,k)

                try:
                    idx = var_name_to_index_map[myvarname]
                    val = solution[idx]
                    tri_ineq_wyes[i,j,k] = val
                except Exception as e:
                    pass

        
    return tri_ineq_wyes

def translate_tri_ineq_wyes_into_values_threed(model,tri_ineq_wye_vars, n_routers,var_name_to_index_map=None):

    # errors on unrecognized type
    if calc_model_type(model) == 'highs':
        return _translate_tri_ineq_wyes_into_values_threed_highs(model, tri_ineq_wye_vars, n_routers,var_name_to_index_map=var_name_to_index_map)
    elif calc_model_type(model) == 'gurobi':
        return _translate_tri_ineq_wyes_into_values_threed_gurobi(model,tri_ineq_wye_vars, n_routers)

def _get_r_map_vars_gurobi(model, n_routers):
    r_map = UpperTriMatrix(n_routers)

    for i in range(n_routers):
        for j in range(i+1, n_routers):
            myvarname = r_map_name(i,j)
            
            var = model.getVarByName(myvarname)
            r_map[i,j] = var
    return r_map

def _get_r_map_vars_ortools(model, n_routers):
    r_map = UpperTriMatrix(n_routers)

    all_vars = model.get_variables()

    for var in all_vars:

        var_name = var.name

        if 'tri_ineq_wye' in var_name:
            break
        if 'r_map' not in var_name:
            continue

        match = re.fullmatch(r'var_r_map_(-?\d+)i_(-?\d+)j', var_name).groups()
        i,j = map(int, match)
        r_map[i,j] = var

    return r_map

def get_r_map_vars(model, xyzc_dims, scale_factor=1.0):

    (x_dim, y_dim, z_dim, cube_dim) = xyzc_dims
    n_routers = x_dim*y_dim*z_dim

    if calc_model_type(model) == 'gurobi':
        r_map = _get_r_map_vars_gurobi(model, n_routers)
    elif calc_model_type(model) == 'ortools':
        r_map = _get_r_map_vars_ortools(model, n_routers)
    else:
        # all_names
        print('ERROR: Unimplemented get_r_map_vars. Exiting...')
        quit()

    electrical_conns_adj_list = init_electrical_conns(xyzc_dims)
    for i, conns in enumerate(electrical_conns_adj_list):
        for j in conns:
            r_map[i,j] = scale_factor
    for i in range(n_routers):
        for j in range(i+1, n_routers):
            if r_map[i,j] is None:
                r_map[i,j] = 0

    return r_map

def update_lbub_r_map(model, r_map_vars, conn, lb_val, ub_val):
    (i,j) = conn

    # errors on unrecognized type
    if calc_model_type(model) == 'highs':
        conn_idx = r_map_vars[i,j]
        model.changeColBounds(conn_idx,lb_val,ub_val)
    elif calc_model_type(model) == 'gurobi':
        r_map_vars[i,j].LB = lb_val
        r_map_vars[i,j].UB = ub_val
    elif calc_model_type(model) == 'ortools' or calc_model_type(model) == 'ortools_model_builder':
        r_map_vars[i,j].lower_bound = lb_val
        r_map_vars[i,j].upper_bound = ub_val
    else:
        print('ERROR: Unimplemented update_lbub_r_map. Exiting...')
        quit()    

    console_log(f'Setting r_map[{conn}] = [{lb_val},{ub_val}]','MODEL UPDATE',n_indents=1)

    return model

def update_multi_lbub_r_map(model, r_map_vars, conns, lb_val, ub_val):

    # errors on unrecognized type
    if calc_model_type(model) == 'highs':
        conn_idxs = []
        for (i,j) in conns:
            conn_idxs.append(r_map_vars[i,j])

        n_to_change = len(conn_idxs)
        # one Python->C++ call
        model.changeColsBounds(n_to_change,conn_idxs,[lb_val]*n_to_change,[ub_val]*n_to_change)
    elif calc_model_type(model) == 'gurobi':
        # Gurobi does lazy update so no need to call once
        for (i,j) in conns:
            r_map_vars[i,j].LB = lb_val
            r_map_vars[i,j].UB = ub_val
    elif calc_model_type(model) == 'ortools' or calc_model_type(model) == 'ortools_model_builder':
        for (i,j) in conns:
            r_map_vars[i,j].lower_bound = lb_val
            r_map_vars[i,j].upper_bound = ub_val
    else:
        print('ERROR: Unimplemented update_lbub_r_map. Exiting...')
        quit()  

    console_log(f'Setting r_map[{conns}] = [{lb_val},{ub_val}]','MODEL UPDATE',n_indents=1)

    return model

def get_obj_val(model, solver_or_result):

    if calc_model_type(model) == 'gurobi':
        return model.objVal
    elif calc_model_type(model) == 'highs':
        return model.getObjectiveValue()
    elif calc_model_type(model) == 'ortools':
        return solver_or_result.objective_value()
    elif calc_model_type(model) == 'ortools_model_builder':
        return solver_or_result.objective_value

    print(f'ERROR: Unimplemented get_obj_val: Exiting...')
    quit()

def load_model(model_path, solver_type):

    model = None
    if solver_type == 'gurobi':
        model = gp.read(model_path)
    elif solver_type == 'ortools':
        model = model_builder.ModelBuilder()
        model.import_from_mps_file(model_path)
    else:
        print(f'ERROR: Unimplemented load_model. Exiting...')
        quit()

    console_log(f'Read model from {model_path}', 'STATUS')

    return model

def save_model(model, model_path, solver_type):

    if solver_type == 'gurobi':
        model.write(model_path)
    elif solver_type == 'highs':
        model.writeModel(model_path)
    elif solver_type == 'ortools_model_builder' or solver_type == 'ortools':
        with open(model_path, "w+") as of:
            of.write(model.export_to_mps_string(obfuscate=False))
    else:
        print(f'ERROR: Unimplemented save_model. Exiting...')
        quit()
    
    console_log(f'Saved model to {model_path}', 'STATUS')

def create_model_savepath( model_base_name, solver_type, file_type='mps'):

    global MODEL_DIR

    model_name = f'{model_base_name}.{file_type}' # f'_{solver_type}',?
    model_path = os.path.join(MODEL_DIR, model_name)

    return model_path


# Formulation Functions
###############################################################################

# Naming
# --------------------------------------------------------------------------------

def unity_wye_name():
    return f'y0'

def r_map_name(i,j):
    return f'm_{i}_{j}'

def tri_ineq_wye_name(i,j,k):
    return f'y_{i}_{j}_{k}'

def limit_xyz_conns_name(i,d):
    return f'l{d}_{i}'

def constr_A_transpose_name(a,b):
    return f'At_{a}_{b}'

# Constraints
# --------------------------------------------------------------------------------

def constr_limit_xyz_conns(model, r_map, xyzc_dims,scale_factor=1.0):

    (x_dim, y_dim, z_dim, cube_dim) = xyzc_dims
    n_routers = x_dim*y_dim*z_dim

    max_conns = scale_factor

    for i in range(n_routers):

        i_x,i_y,i_z = r_to_xyz(i,x_dim,y_dim,z_dim)

        rel_i_x = i_x % cube_dim
        rel_i_y = i_y % cube_dim
        rel_i_z = i_z % cube_dim

        # save time on interiors
        if not rel_xyz_is_on_face(rel_i_x, rel_i_y, rel_i_z, cube_dim):
            continue
        
        poss_conns = poss_optical_conns_for_r(i, xyzc_dims)

        for direction, conns in poss_conns.items():

            if len(conns) == 0:
                continue
            
            var_list = [r_map[i,c] for c in conns]
            # input(f'var_list = {var_list} (={[model.variableName(x) for x in var_list]}) from conns {conns}')
            # model.addConstr(model.qsum(var_list) == max_conns)
            # myconstrname = f'constr_limitxyz_{direction}_{i}i'
            myconstrname = limit_xyz_conns_name(i,direction)
            model = add_constr_sum_equality(model, var_list, max_conns, myconstrname)

def constr_A_transpose_neighbors_only(model, unity_wye, tri_ineq_wyes, r_map, valid_conns, xyzc_dims, n_routers):

    electrical_conns_adj_list = init_electrical_conns(xyzc_dims)

    for a in range(n_routers):
        for b in range(n_routers):
            if a==b:
                continue
            if a > b:
                continue

            #if valid_conns[a][b] == 0:
            #    continue

            pos_lhs_vars = [unity_wye]
            pos_lhs_vars += [tri_ineq_wyes[a,j,b ] for j in range(n_routers) if (valid_conns[a][b]==1 or valid_conns[j][b] == 1) and j!=a and j!=b]
            pos_lhs_vars += [tri_ineq_wyes[i,b,a ] for i in range(n_routers) if (valid_conns[i][a]==1 or valid_conns[b][a] == 1) and i!=a and i!=b]
            # pos_lhs_vars += [tri_ineq_wyes[a,j,b ] for j in range(n_routers) if (valid_conns[a][b]==1) and j!=a and j!=b]
            # pos_lhs_vars += [tri_ineq_wyes[i,b,a ] for i in range(n_routers) if (valid_conns[i][a]==1) and i!=a and i!=b]
    
            neg_lhs_vars = [r_map[a,b]]
            neg_lhs_vars += [tri_ineq_wyes[a,b,k] for k in range(n_routers) if (valid_conns[a][k]==1 or valid_conns[b][k] == 1) and k!=a and k!=b]
            # neg_lhs_vars += [tri_ineq_wyes[a,b,k] for k in range(n_routers) if (valid_conns[a][k]==1) and k!=a and k!=b]

            # myconstrname = f'constr_Atranspose_{a}a_{b}b'
            myconstrname = constr_A_transpose_name(a,b)
            model = add_constr_pos_neg_sum_lte(model, pos_lhs_vars, neg_lhs_vars, 0, myconstrname)

    # for a in range(n_routers):
    #     for b in range(n_routers):
    #         continue
    #         if a==b:
    #             continue

    #         if valid_conns[a][b] == 1:
    #             continue

    #         pos_lhs_vars = [unity_wye]
    #         pos_lhs_vars += [tri_ineq_wyes[a,j,b ] for j in range(n_routers) if valid_conns[a][b]==1 and j!=a and j!=b]
    #         pos_lhs_vars += [tri_ineq_wyes[i,b,a ] for i in range(n_routers) if valid_conns[i][a]==1 and i!=a and i!=b]

    #         neg_lhs_vars = [r_map[a,b]]
    #         neg_lhs_vars += [tri_ineq_wyes[a,b,k] for k in range(n_routers) if valid_conns[a][k]==1 and k!=a and k!=b]

    #         model = add_constr_pos_neg_sum_lte(model, pos_lhs_vars, neg_lhs_vars, 0, f'constr_Atranspose_{a}a_{b}b')


# Modify/Interact
# --------------------------------------------------------------------------------

def calc_a_tri_cost(tri_ineq_wye_vals, a, b, n_routers):

    cost = 0
    # large loops to allow try-catch?
    for k in range(n_routers):
        try:
            cost -= tri_ineq_wye_vals[a,b,k]
        except:
            pass
    for j in range(n_routers):
        try:
            cost += tri_ineq_wye_vals[a,j,b ]
        except:
            pass
    for i in range(n_routers):
        try:
            cost += tri_ineq_wye_vals[i,b,a ]
        except:
            pass
    return cost

def calc_all_tri_costs(tri_ineq_wye_vals, n_routers):

    # TODO check if symmetric??

    return [[calc_a_tri_cost(tri_ineq_wye_vals, a, b, n_routers) for a in range(n_routers)] for b in range(n_routers)]
    # costs = [[0 for _ in range(n_routers)] for __ in range(n_routers)]

    # for a in range(n_routers):
    #     for b in range(n_routers):
    #         for c in range(n_routers):

    #             # lhs (neg)
    #             i = a
    #             j = b
    #             k = c
    #             costs[a][b] -= tri_ineq_wye_vals[i,j,k]

    #             # rhs 1 (pos)
    #             i = a
    #             j = c
    #             k = b
    #             costs[a][j] += tri_ineq_wye_vals[i,j,k]

    # return costs

def score_solution(model, solver, r_map_vars, tri_ineq_wye_vars, n_routers,var_name_to_index_map=None):


    r_map_vals = translate_r_map_into_values_twod(model, solver, r_map_vars, n_routers,var_name_to_index_map=var_name_to_index_map)
    print(f'Got r_map vals')

    tri_ineq_wye_vals = translate_tri_ineq_wyes_into_values_threed(model, tri_ineq_wye_vars, n_routers,var_name_to_index_map=var_name_to_index_map)
    print(f'Got tri_ineq_wye vals')

    # triangle_costs = calc_all_tri_costs(tri_ineq_wye_vals, n_routers)
    triangle_costs = [[0 for _ in range(n_routers)] for __ in range(n_routers)]
    print(f'Calc triangle costs')

    scores = UpperTriMatrix(n_routers)
    for i in range(n_routers):
        for j in range(i+1,n_routers):

            if r_map_vals[i,j] is None:
                continue

            scores[i,j] = r_map_vals[i,j]  # - triangle_costs[i][j]
            if VERBOSE:
                print(f'{i:02}->{j:02} : r_map = {r_map_vals[i,j]}, tri cost = {triangle_costs[i][j]}, score = {scores[i,j]}')

    return scores

def select_max_conn_and_update(model, solver_or_result, r_map_vars, tri_ineq_wye_vars, valid_conns, chosen_optical_conns, xyzc_dims, var_name_to_index_map=None, scale_factor=1.0, spreading_cubes=True):

    global VERBOSE

    (x_dim, y_dim, z_dim, cube_dim) = xyzc_dims
    n_routers = x_dim*y_dim*z_dim

    chosen_cube_conns = set()
    if spreading_cubes:
        chosen_cube_conns = chosen_optical_to_cubes(chosen_optical_conns, xyzc_dims)

    # read from model
    # r_map_vals = translate_r_map_into_values_twod(model, n_routers)
    conn_scores = translate_r_map_into_values_twod(model, solver_or_result,r_map_vars, n_routers,var_name_to_index_map=var_name_to_index_map)
    print(f'Got r_map vals')

    # conn_scores = score_solution(model, solver,r_map_vars, tri_ineq_wye_vars, n_routers,var_name_to_index_map=var_name_to_index_map)

    # max_conn, max_val = get_max_valued_r_map_val(r_map_vals, valid_conns, chosen_optical_conns, xyzc_dims, chosen_cube_conns,spreading_cubes)
    max_conn, max_val = get_max_valued_r_map_val(conn_scores, valid_conns, chosen_optical_conns, xyzc_dims, chosen_cube_conns,spreading_cubes)

    if max_conn == None:
        chosen_cube_conns = set()
        # max_conn, max_val = get_max_valued_r_map_val(r_map_vals, valid_conns, chosen_optical_conns, xyzc_dims, chosen_cube_conns,spreading_cubes)
        max_conn, max_val = get_max_valued_r_map_val(conn_scores, valid_conns, chosen_optical_conns, xyzc_dims, chosen_cube_conns,spreading_cubes)

    (i,j) = max_conn
    chosen_optical_conns.append((i,j))
    chosen_optical_conns.append((j,i))
    previously_valid_conns = deepcopy(valid_conns)
    valid_conns = calc_valid_conns(xyzc_dims, chosen_optical_conns)
    i_cube = which_cube(i, xyzc_dims)
    j_cube = which_cube(j, xyzc_dims)

    console_log(f'Selected max conn {max_conn} w/ score {max_val} to connect cubes {i_cube}->{j_cube}','STATUS')


    # udpate model
    # ------------

    # positive set
    model = update_lbub_r_map(model, r_map_vars, max_conn, scale_factor, scale_factor)

    rejected_conns = []
    for i in range(n_routers):
        for j in range(n_routers):
            if i >= j:
                continue
            if previously_valid_conns[i][j] != valid_conns[i][j]:
                rejected_conns.append((i,j))

    # negative unset
    model = update_multi_lbub_r_map(model, r_map_vars, rejected_conns, 0, 0)


    # TODO set the tri_ineq_wyes to 0 too

    # model = remove_conns_from_model(model, rejected_conns, r_map_vars,tri_ineq_wye_vars, n_routers)
    # rejected_r_map_idxs = [r_map_vars[a,b].index for (a,b) in rejected_conns]
    # n_to_change = len(rejected_r_map_idxs)
    # zeroes_arr = [0 for _ in range(n_to_change)]
    # model.changeColsBounds(n_to_change,rejected_r_map_idxs,zeroes_arr,zeroes_arr)

    # console_log(f'Setting r_map[{rejected_conns}] = 0','MODEL UPDATE',n_indents=1)

    # rejected_tri_ineq_wye_idxs = [tri_ineq_wye_vars[a,j,b].index for (a,b) in rejected_conns for j in range(n_routers) if not ( a==j or a==b or j==b) ]
    # n_to_change = len(rejected_tri_ineq_wye_idxs)
    # zeroes_arr = [0 for _ in range(n_to_change)]
    # model.changeColsBounds(n_to_change,rejected_tri_ineq_wye_idxs,zeroes_arr,zeroes_arr)

    return model, valid_conns, chosen_optical_conns

# Create and Solve
# --------------------------------------------------------------------------------

def create_model(xyzc_dims, valid_conns, chosen_optical_conns, script_params={}):

    console_log('Creating model','STATUS')

    # parameters and knowns
    # --------------------

    (x_dim, y_dim, z_dim, cube_dim) = xyzc_dims
    n_routers = x_dim*y_dim*z_dim
    n_ports = 6
    electrical_conns_adj_list = init_electrical_conns(xyzc_dims)

    binary_r_map = script_params['binary_r_map']
    solver_type = script_params['solver']
    scale_factor = script_params['scale_factor']
    # should be less than or equal to sparsest cut
    # cant be more than (# links) / (bisection demand)
    unity_wye_ub = scale_factor*((n_ports/2) / (n_routers))

    # same?
    tri_ineq_wye_ub = scale_factor *((n_ports/2) / (n_routers))

    r_map_ub = scale_factor


    # blank model
    # -----------

    if solver_type == 'highs':
        model = highspy.Highs()
    elif solver_type == 'gurobi':
        model = gp.Model()
    elif solver_type == 'ortools_model_builder':
        model = model_builder.ModelBuilder()
    elif solver_type == 'ortools':
        model = mathopt.Model()
    else:
        print(f'ERROR :: create_model :: Solver type {solver_type} unrecognized. Exiting...')
        quit()

    # unity_wye
    # ---------

    # notice obj handles objective weight
    # myvarname = f'var_unity_wye'
    myvarname = unity_wye_name()
    if calc_model_type(model) == 'highs':
        unity_wye = model.addVariable(lb=0, ub=unity_wye_ub, obj=1.0, name=myvarname)
    elif calc_model_type(model) == 'gurobi':
        # unity_wye = model.addVar(lb=0.0, ub=unity_wye_ub, vtype=GRB.CONTINUOUS, obj=1.0, name=myvarname)
        unity_wye = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, obj=1.0, name=myvarname)
    elif calc_model_type(model) == 'ortools_model_builder':
        # unity_wye = model.new_num_var(0.0, unity_wye_ub, "unity_wye")
        unity_wye = model.new_num_var(0.0,np.inf, "unity_wye")
    elif calc_model_type(model) == 'ortools':
        unity_wye = model.add_variable(lb=0.0,ub=unity_wye_ub, name="unity_wye")

    # r_map
    # -----

    r_map = UpperTriMatrix(n_routers)

    start_time = time.time()
    for i in range(n_routers):
        for j in range(i+1,n_routers):

            if j in electrical_conns_adj_list[i]:
                r_map[i,j] = scale_factor
                # print(f'Electrical {i}->{j}')
                continue
            if valid_conns[i][j] == 0:
                r_map[i,j] = 0
                # print(f'Invalid {i}->{j}')
                continue
            if (i,j) in chosen_optical_conns or (j,i) in chosen_optical_conns:
                r_map[i,j] = scale_factor
                # print(f'Known {i}->{j}')
                continue

            # myvarname = f'var_r_map_{i}i_{j}j'
            myvarname = r_map_name(i,j)
            if binary_r_map:
                # r_map[i,j] =  model.addBinary(name=myvarname)
                if calc_model_type(model) == 'highs':
                    r_map[i,j] = model.addBinary(name=myvarname)
                elif calc_model_type(model) == 'gurobi':
                    r_map[i,j] = model.addVar(vtype=GRB.BINARY, name=myvarname)
                elif solver_type == 'ortools_model_builder':
                    r_map[i,j] = model.new_bool_var(myvarname)
                elif solver_type == 'ortools':
                    print(f'ERROR :: create_model :: unimplemented')
                else:
                    print(f'ERROR :: create_model :: unimplemented')

            else:
                # r_map[i,j] =  model.addVariable(lb = 0, ub=r_map_ub,name=myvarname)
                if calc_model_type(model) == 'highs':
                    r_map[i,j] = model.addVariable(lb=0, ub=r_map_ub, name=myvarname)
                elif calc_model_type(model) == 'gurobi':
                    r_map[i,j] = model.addVar(lb=0.0, ub=r_map_ub, vtype=GRB.CONTINUOUS, name=myvarname)
                elif solver_type == 'ortools_model_builder':
                    r_map[i,j] = model.new_num_var(0.0, r_map_ub, myvarname)
                elif solver_type == 'ortools':
                    r_map[i,j] = model.add_variable(lb=0.0, ub=r_map_ub, name=myvarname)
                else:
                    print(f'ERROR :: create_model :: unimplemented')

    elapsed_time = time.time() - start_time
    console_log(f'{round(elapsed_time,1)}s for creating r_map vars','PERFORMANCE',n_indents=1)

    # tri_ineq_wyes
    # -------------

    # tri_ineq_wyes = SparseUpperTri3D(n_routers)
    tri_ineq_wyes = SymTriDict()
    # tri_ineq_wyes = {}

    start_time = time.time()
    # TODO check if only using i < j is valid for this???
    for i in range(n_routers):
        for j in range(i+1,n_routers):
        # for j in range(n_routers):
            if i==j:
                continue

            for k in range(n_routers):
                if i==k or j==k:
                    continue
                
                if valid_conns[i][k] == 0 and valid_conns[j][k] == 0:
                    continue

                # myvarname = f'var_tri_ineq_wye_{i}i_{j}j_{k}k'
                myvarname = tri_ineq_wye_name(i,j,k)
                # tri_ineq_wyes[i,j,k] = model.addVariable(name=myvarname)
                if calc_model_type(model) == 'highs':
                    # tri_ineq_wyes[i,j,k] = model.addVariable( name=myvarname)
                    tri_ineq_wyes[i,j,k] = model.addVariable(lb=0.0, name=myvarname)
                elif calc_model_type(model) == 'gurobi':
                    # tri_ineq_wyes[i,j,k] = model.addVar(vtype=GRB.CONTINUOUS, name=myvarname)
                    tri_ineq_wyes[i,j,k] = model.addVar(lb=0.0,vtype=GRB.CONTINUOUS, name=myvarname)
                elif solver_type == 'ortools_model_builder':
                    tri_ineq_wyes[i,j,k] = model.new_num_var(0.0,np.inf, name=myvarname)
                    # tri_ineq_wyes[i,j,k] = model.new_num_var(0.0,unity_wye_ub, name=myvarname)
                elif solver_type == 'ortools':
                    tri_ineq_wyes[i,j,k] = model.add_variable(lb=0.0, name=myvarname)
                else:
                    print(f'ERROR :: create_model :: unimplemented')

    elapsed_time = time.time() - start_time
    console_log(f'{round(elapsed_time,1)}s for creating tri_ineq_wyes vars','PERFORMANCE',n_indents=1)


    start_time = time.time()
    constr_A_transpose_neighbors_only(model, unity_wye, tri_ineq_wyes, r_map, valid_conns, xyzc_dims,n_routers)
    elapsed_time = time.time() - start_time
    console_log(f'{round(elapsed_time,1)}s for creating A_transpose_neighbors_only constr','PERFORMANCE',n_indents=1)

    start_time = time.time()
    constr_limit_xyz_conns(model, r_map, xyzc_dims, scale_factor=scale_factor)
    elapsed_time = time.time() - start_time
    console_log(f'{round(elapsed_time,1)}s for creating limit_xyz_conns constr','PERFORMANCE',n_indents=1)

    # if calc_model_type(model) == 'highs':
    # elif calc_model_type(model) == 'gurobi':

    if calc_model_type(model) == 'ortools' or calc_model_type(model) == 'ortools_model_builder':
        model.maximize(unity_wye)



    return model, r_map, tri_ineq_wyes

def solve_model(model, solver_params={"method":"pdlp"}, initial_solve=True):

    start_time = time.time()

    model_type = calc_model_type(model)

    solver = None
    if model_type == 'highs':
        model = solve_highs(model, solver_params)
    elif model_type == 'gurobi':
        model = solve_gurobi(model, solver_params)
    elif model_type == 'ortools':
        model, solver = solve_ortools(model, solver_params)
    elif model_type == 'ortools_model_builder':
        model, solver = solve_ortools_model_builder(model, solver_params)

    elapsed_time = time.time() - start_time
    console_log(f'{round(elapsed_time,1)}s for solving model','PERFORMANCE',n_indents=1)

    return model, solver

def solve_highs(model, solver_params):

    # CLAs
    for k,v in solver_params.items():
        try:
            model.setOptionValue(k,v)
            console_log(f'Set option {k} to value {v}','GIVENS',n_indents=1)
        except:
            pass

    # always
    model.setOptionValue("parallel", "on")
    model.setOptionValue("run_crossover", "off")


    # if initial_solve:
    #     model.setOptionValue("solver", "ipm")
    #     model.setOptionValue("run_crossover", "on")
    #     model.maximize()
    # else:
    #     model.setOptionValue("presolve", "off")
    #     model.setOptionValue("solver", "simplex")
    #     model.run()

    model.maximize()

    return model

def solve_gurobi(model, solver_params): 

    # CLAs
    for k,v in solver_params.items():
        try:
            model.setParam(k,v)
            console_log(f'Set option {k} to value {v}','GIVENS',n_indents=1)
        except:
            pass

    # always
    model.setParam("Crossover", 0)
    # faster
    model.setParam("BarOrder",0)

    model.ModelSense = gp.GRB.MAXIMIZE



    model.optimize()

    return model # presolved_model #model

def solve_ortools_model_builder(model, solver_params):

    solver = model_builder.ModelSolver("PDLP")
    solver.enable_output(True)

    # solver.setNumThreads(256)



    # # "Quick-to-good" pass (looser optimality gap + strict feasibility)
    # pdlp_quick = r"""
    # termination_criteria {
    # optimality_norm: OPTIMALITY_NORM_L_INF
    # detailed_optimality_criteria {
    #     eps_optimal_objective_gap_relative: 1e-3
    #     eps_optimal_primal_residual_relative: 1e-7
    #     eps_optimal_primal_residual_absolute: 1e-9
    # }
    # time_sec_limit: 400  # cap runtime for the first pass
    # }
    # presolve_options { use_glop: true }
    # restart_strategy: ADAPTIVE_HEURISTIC
    # # linesearch_rule: MALITSKY_POCK
    # l_inf_ruiz_iterations: 10
    # # num_threads: 64
    # """
    # solver.set_solver_specific_parameters(pdlp_quick)


    # solver.set_time_limit_in_seconds(100)
    # solver.set_solver_specific_parameters("num_workers:17, num_violation_ls:2, use_lb_relax_lns:true")
    result = solver.solve(model)

    # print(f'result = {result}')

    # Check termination
    # print("Termination reason:", result.termination.reason)
    # print("Objective:", result.objective_value)

    # Print variables
    # for var, val in result.variable_values.items():
    #     print(f"{var}: {val}")

    # # 1) All variables in the model (a Pandas Index of Variable objects)
    # vars_idx = model.get_variables()
    # # print("num vars:", len(vars_idx))

    # # 2) Solution values for all variables (Pandas Series, indexed by the variables)
    # vals = solver.values(vars_idx)
    # print(vals.head())

    # # 3) (Optional) get readable names for each variable (from the proto)
    # proto = model.export_to_proto()
    # name_map = {model.var_from_index(i): v_proto.name
    #             for i, v_proto in enumerate(proto.variable)}
    # named_vals = vals.rename(index=name_map)   # Series indexed by variable name
    # print(named_vals[:20])

    return model, solver

def solve_ortools(model, solver_params):

    solver = model_builder.ModelSolver("PDLP")
    # solver.enable_output(True)
    solver = mathopt.SolverType.PDLP

    params = mathopt.SolveParameters(
        enable_output=True
    )


    # solver.setNumThreads(256)



    # # "Quick-to-good" pass (looser optimality gap + strict feasibility)
    # pdlp_quick = r"""
    # termination_criteria {
    # optimality_norm: OPTIMALITY_NORM_L_INF
    # detailed_optimality_criteria {
    #     eps_optimal_objective_gap_relative: 1e-3
    #     eps_optimal_primal_residual_relative: 1e-7
    #     eps_optimal_primal_residual_absolute: 1e-9
    # }
    # time_sec_limit: 400  # cap runtime for the first pass
    # }
    # presolve_options { use_glop: true }
    # restart_strategy: ADAPTIVE_HEURISTIC
    # # linesearch_rule: MALITSKY_POCK
    # l_inf_ruiz_iterations: 10
    # # num_threads: 64
    # """
    # solver.set_solver_specific_parameters(pdlp_quick)


    # solver.set_time_limit_in_seconds(100)
    # solver.set_solver_specific_parameters("num_workers:17, num_violation_ls:2, use_lb_relax_lns:true")
    # result = solver.solve(model)
    result = mathopt_solve.solve(model, solver, params=params)

    # print(f'result = {result}')

    # Check termination
    # print("Termination reason:", result.termination.reason)
    # print("Objective:", result.objective_value)

    # Print variables
    # for var, val in result.variable_values.items():
    #     print(f"{var}: {val}")

    # # 1) All variables in the model (a Pandas Index of Variable objects)
    # vars_idx = model.get_variables()
    # # print("num vars:", len(vars_idx))

    # # 2) Solution values for all variables (Pandas Series, indexed by the variables)
    # vals = solver.values(vars_idx)
    # print(vals.head())

    # # 3) (Optional) get readable names for each variable (from the proto)
    # proto = model.export_to_proto()
    # name_map = {model.var_from_index(i): v_proto.name
    #             for i, v_proto in enumerate(proto.variable)}
    # named_vals = vals.rename(index=name_map)   # Series indexed by variable name
    # print(named_vals[:20])

    return model, result


# Main(s)
###############################################################################

def driver(xyzc_dims, script_params={}, solver_params={}):

    # constants
    # ---------
    (x_dim, y_dim, z_dim, cube_dim) = xyzc_dims
    n_routers = x_dim*y_dim*z_dim
    n_cubes = (x_dim // cube_dim)*(y_dim // cube_dim)*(z_dim // cube_dim)
    n_ports = 6
    n_tot_optical_links = n_cubes*(cube_dim**2)*6


    # parameters
    # ----------
    recalc_interval = script_params["recalc_interval"]
    solver = script_params["solver"]
    scale_factor = script_params['scale_factor']
    binary_r_map = script_params["binary_r_map"]

    if solver == 'gurobi':
        method = solver_params["Method"]
    elif solver == 'highs':
        method = solver_params["solver"]
    else:
        method = 'pdlp'

    if script_params['start_map'] is not None:
        rm_coords_interval = script_params["rm_coords_interval"]
        model_base_name = f"aasd_improve_{n_cubes}c_{n_routers}r_{n_ports}p_{x_dim}x{y_dim}x{z_dim}_{recalc_interval}interval_{rm_coords_interval}rminterval_{solver}_{method}"
        alg_type = 'improve'
    else:
        model_base_name = f"aasd_create_{n_cubes}c_{n_routers}r_{n_ports}p_{x_dim}x{y_dim}x{z_dim}_{recalc_interval}interval_{solver}_{method}"
        alg_type = 'create'

    if scale_factor > 1:
        model_base_name += f'_{int(scale_factor)}scalefactor'

    if binary_r_map:
        model_base_name += '_milp'

    if 'out_filename' in script_params.keys() and script_params['out_filename'] is not None:
        model_base_name = script_params['out_filename']

    # Quick print of givens
    # --------------------------------------------------------------------------------
    print('='*80)
    console_log(f"CLA/default/file values: n_cubes={n_cubes}, n_routers={n_routers}, n_ports={n_ports}","GIVENS")
    console_log(f"Dimensionally, problem is ({x_dim}x{y_dim}x{z_dim})","GIVENS",n_indents=1)
    console_log(f"Running a(n) {alg_type} type algorithm","GIVENS")
    console_log(f"Base output name is {model_base_name}","GIVENS",n_indents=1)
    print('='*80)

    if alg_type == 'improve':
        chosen_optical_conns = iteratively_improve(xyzc_dims,script_params=script_params,solver_params=solver_params,model_base_name=model_base_name)
    else:
        chosen_optical_conns = iteratively_create(xyzc_dims,script_params=script_params,solver_params=solver_params,model_base_name=model_base_name)

    print('='*80)
    console_log(f'Script complete','STATUS')
    avg_hops = calc_avg_hops(optical_to_r_map(xyzc_dims, chosen_optical_conns))
    console_log(f'# chosen / total optical = {len(chosen_optical_conns)} / {n_tot_optical_links}','STATUS',n_indents=1)
    console_log(f'avg hops {avg_hops}','STATUS',n_indents=1)

    # Output result
    out_name = f'./files/lp_iterative_solutions/{model_base_name}.map'
    output_r_map(optical_to_r_map(xyzc_dims, chosen_optical_conns),out_name)

def iteratively_create(xyzc_dims, script_params={}, solver_params={},model_base_name=None):

    global SLOW_RUN

    (x_dim, y_dim, z_dim, cube_dim) = xyzc_dims
    n_routers = x_dim*y_dim*z_dim
    n_cubes = (x_dim // cube_dim)*(y_dim // cube_dim)*(z_dim // cube_dim)
    n_ports = 6
    # unidirectional
    n_tot_optical_links = n_cubes*(cube_dim**2)*6

    recalc_interval = script_params["recalc_interval"]
    write_model = script_params["write_model"]
    solver_type = script_params['solver']
    if solver_type == 'gurobi':
        method = solver_params["Method"]
    elif solver_type == 'highs':
        method = solver_params["solver"]
    else:
        method = 'pdlp'
    
    binary_r_map = script_params["binary_r_map"]
    if binary_r_map:
        recalc_interval = 10**6

    console_log(f'Beginning iterative solve @ {time.ctime(time.time())}','STATUS')
    start_time = time.time()
    console_log(f'\tFinding {n_tot_optical_links} optical links. Recalc interval of {recalc_interval}','STATUS',1)

    chosen_optical_conns = []
    valid_conns = calc_valid_conns(xyzc_dims, chosen_optical_conns)

    load_model_successful = False
    model_path = create_model_savepath(model_base_name.replace(f'{solver_type}_{method}','gurobi_2'), solver_type)
    if model_base_name is not None and os.path.exists(model_path):
        # if script_params['solver'] != 'highs':
        if script_params['solver'] == 'gurobi':

            model = load_model(model_path, solver_type)
            r_map_vars = get_r_map_vars(model, xyzc_dims)
            tri_ineq_wye_vars = None
            load_model_successful = True

    if not load_model_successful:
        console_log(f'Existing model @ {model_path} DNE','STATUS')
        model, r_map_vars, tri_ineq_wye_vars = create_model(xyzc_dims, valid_conns, chosen_optical_conns, script_params=script_params)

        solver_type = script_params['solver']
        if write_model:
            save_model(model, model_path, solver_type)

    var_name_to_index_map = None
    # var_name_to_index_map = { f'var_r_map_{i}i_{j}j' : r_map_vars[i,j].index for i in range(n_routers) for j in range(i+1,n_routers) if (type(r_map_vars[i,j]) != int and type(r_map_vars[i,j]) != float)}
    # for k,v in tri_ineq_wye_vars.items():
    #     var_name_to_index_map.update({ v.name : v.index })


    console_log(f'{print_rss()} memory used after creating model','PERFORMANCE',n_indents=1)

    timeline_log(0, 0, 0,model_base_name, first_write=True)

    initial_solve = True
    done = False
    while not done:

        model, valid_conns, chosen_optical_conns = solve_and_select_maximums(model, valid_conns, chosen_optical_conns, xyzc_dims, recalc_interval, r_map_vars, tri_ineq_wye_vars, model_base_name,script_params=script_params, solver_params=solver_params, var_name_to_index_map=var_name_to_index_map)

        if len(chosen_optical_conns) == n_tot_optical_links:
            done = True
            break

        if SLOW_RUN:
            input('cont?')

    # all links chosen and binary
    model, solver_or_result = solve_model(model, solver_params=solver_params, initial_solve=initial_solve)

    obj_val = get_obj_val(model, solver_or_result)
    console_log(f'Objective value : {obj_val}','STATUS',n_indents=1)

    avg_hops = calc_avg_hops(optical_to_r_map(xyzc_dims, chosen_optical_conns))

    timeline_log(len(chosen_optical_conns), avg_hops, obj_val,  model_base_name)

    return chosen_optical_conns

def iteratively_improve(xyzc_dims, script_params={}, solver_params={}):

    (x_dim, y_dim, z_dim, cube_dim) = xyzc_dims
    n_routers = x_dim*y_dim*z_dim
    n_cubes = (x_dim // cube_dim)*(y_dim // cube_dim)*(z_dim // cube_dim)
    n_ports = 6
    # unidirectional
    n_tot_optical_links = n_cubes*(cube_dim**2)*6
    recalc_interval = script_params["recalc_interval"]
    rm_coords_interval = script_params["rm_coords_interval"]
    start_map_name = script_params['start_map']

    start_map_name = script_params['start_map']
    start_map = ingest_map(start_map_name)
    assert(len(start_map) == n_routers)
    chosen_optical_conns = r_map_to_optical(xyzc_dims, start_map)

    console_log(f'Beginning improvement solve @ {time.ctime(time.time())}','STATUS')
    start_time = time.time()
    console_log(f'\tFinding {n_tot_optical_links} optical links. Recalc interval of {recalc_interval}','STATUS',1)
    console_log(f'\tRemoving {rm_coords_interval} coordinates per iteration','STATUS',1)
    console_log(f'\tStarting map {start_map_name}','STATUS',1)
    console_log(f'\tStarting avg_hops {calc_avg_hops(optical_to_r_map(xyzc_dims, chosen_optical_conns))}','STATUS',1)

    # for now keep this as none...
    known_conns = [[0 for _ in range(n_routers)] for __ in range(n_routers)]
    valid_conns = calc_valid_conns(xyzc_dims, [])

    console_log('Creating model','STATUS')
    model, r_map_vars, tri_ineq_wye_vars = create_model(xyzc_dims, valid_conns, known_conns, script_params=script_params)
    r_map_var_idxs = [r_map_vars[i,j].index for i in range(n_routers) for j in range(i+1,n_routers) if type(r_map_vars[i,j]) != int]

    console_log(f'{print_rss()} memory used after creating model','PERFORMANCE',n_indents=1)

    initial_solve = True
    done = False
    while not done:

        conns_to_remove,current_r_map = priority_removal( chosen_optical_conns , rm_coords_interval, xyzc_dims)
        chosen_optical_conns = r_map_to_optical(xyzc_dims, current_r_map)
        model = force_conns_in_model(model,chosen_optical_conns,r_map_vars,tri_ineq_wye_vars, n_routers)

        illegals = []
        for conn in conns_to_remove:
            if conn in chosen_optical_conns:
                chosen_optical_conns.remove(conn)
            if valid_conns[conn[0]][conn[1]] == 0:
                illegals.append(conn)
        for ill in illegals:
            conns_to_remove.remove(ill)
        model = allow_conns_in_model(model, conns_to_remove,r_map_vars,tri_ineq_wye_vars, n_routers)

        while len(chosen_optical_conns) < n_tot_optical_links:
            model, valid_conns, chosen_optical_conns = solve_and_select_maximums(model, valid_conns, chosen_optical_conns, xyzc_dims, recalc_interval, r_map_vars, r_map_var_idxs, tri_ineq_wye_vars, script_params=script_params, solver_params=solver_params)

        if len(chosen_optical_conns) == n_tot_optical_links:
            done = True
            break

    return chosen_optical_conns

def solve_and_select_maximums(model, valid_conns, chosen_optical_conns, xyzc_dims, recalc_interval, r_map_vars, tri_ineq_wye_vars,model_base_name, script_params={}, solver_params={},initial_solve=False, var_name_to_index_map=None):
    # constants
    # ---------
    (x_dim, y_dim, z_dim, cube_dim) = xyzc_dims
    n_routers = x_dim*y_dim*z_dim
    n_cubes = (x_dim // cube_dim)*(y_dim // cube_dim)*(z_dim // cube_dim)
    n_ports = 6
    # unidirectional
    n_tot_optical_links = n_cubes*(cube_dim**2)*6

    scale_factor = script_params["scale_factor"]
    spreading_cubes = script_params["spreading_cubes"]
    connect_first = script_params["connect_first"]
    avg_hops = calc_avg_hops(optical_to_r_map(xyzc_dims, chosen_optical_conns))

    # stop spreading when connected
    if avg_hops < INF:
        spreading_cubes = False

    print('-'*80)
    console_log(f'Solving model','STATUS')
    console_log(f'# chosen / total optical = {len(chosen_optical_conns)} / {n_tot_optical_links}','STATUS',n_indents=1)
    console_log(f'avg hops {avg_hops}','STATUS',n_indents=1)

    model, solver_or_result = solve_model(model, solver_params=solver_params, initial_solve=initial_solve)

    obj_val = get_obj_val(model, solver_or_result)

    timeline_log(len(chosen_optical_conns), avg_hops, obj_val,  model_base_name)

    # if calc_model_type(model) == 'highs':
    #     sol = model.getSolution()
    #     row_dual = sol.row_dual        # shadow prices π_i for each row
    #     col_dual = sol.col_dual        # reduced costs for variables (r_j)

    #     print("Objective:", model.getInfo().objective_function_value)
    #     print("First 5 row duals:", row_dual[:5])
    #     print("First 5 reduced costs:", col_dual[:5])
    # else:
    #     for i,c in enumerate(model.getConstrs()):
    #         print(c.ConstrName, c.Pi)
    #         if i >= 5:
    #             break
    #     for i,v in enumerate(model.getVars()):
    #         print(f'{v.VarName} : val {v.X} & redcost {v.RC}')
    #         if i >= 5:
    #             break



    console_log(f'Selecting max conn(s)','STATUS')
    start_time = time.time()
    inner_iter = 0
    while inner_iter < recalc_interval or (connect_first and avg_hops == INF):
    # for _ in range(recalc_interval):
        model, valid_conns, chosen_optical_conns = select_max_conn_and_update(model, solver_or_result, r_map_vars, tri_ineq_wye_vars, valid_conns, chosen_optical_conns, xyzc_dims, scale_factor=scale_factor, spreading_cubes=spreading_cubes, var_name_to_index_map=var_name_to_index_map)

        if len(chosen_optical_conns) == n_tot_optical_links:
            break
        
        inner_iter += 1
        if connect_first:
            avg_hops = calc_avg_hops(optical_to_r_map(xyzc_dims, chosen_optical_conns))

    elapsed_time = time.time() - start_time
    console_log(f'{round(elapsed_time,1)}s for selecting all max_conns','PERFORMANCE',n_indents=1)

    return model, valid_conns, chosen_optical_conns

def main():

    xyzc_dims = (8, 4, 4, 4)

    args = define_and_parse_args()
    if args.xyzc_dims:
        xyzc_dims = tuple(args.xyzc_dims)
    solver_params = parse_and_package_solver_params_from_args(args)
    script_params = parse_and_package_script_params_from_args(args)


    console_log(f'solver_params : {solver_params}','GIVENS')
    console_log(f'script_params : {script_params}','GIVENS')

    # filter out unimplemented
    if args.binary_r_map:
        print(f'UNIMPLEMENTED :: main() :: binary_r_map is MILP and that usually needs callbacks. Use MILP specific script.')
        quit()

    driver(xyzc_dims, script_params=script_params, solver_params=solver_params)


# CLAs
###############################################################################


def define_and_parse_args(description='Generate (Direct) Topologies with Various Formulations'):

    global VERBOSE
    global SLOW_RUN

    parser = argparse.ArgumentParser(description=description)

    # out naming
    parser.add_argument('--out_filename','-of',type=str,help='')

    # constant/problem defs
    parser.add_argument('--xyzc_dims',nargs='+',type=int,help='type without parenthesis and use spaces, no commas')

    # direct Gurobi solver params
    parser.add_argument('--time_limit',type=int,help='time limit in minutes')
    parser.add_argument('--time_limit_secs',type=int,help='time limit in minutes')
    parser.add_argument('--threads',type=int,default=128,help='# threads total')
    parser.add_argument('--presolve',type=int,help='Presolve aggressiveness. -1=>auto. 0=>off. 1=>conservative. 2=>aggressive')
    parser.add_argument('--output_flag',type=bool,default=False,help='')
    parser.add_argument('--method',type=str,default='ipm',help='')
    parser.add_argument('--kkt_tolerance',type=float,help='single master tolerance. \
                        All feasibility/optimality tests (primal feasibility, dual feasibility, primal/dual residuals, and P-D objective error')
    parser.add_argument('--pdlp_optimality_tolerance',type=float,help='')
    parser.add_argument('--pdlp_scaling',type=bool,help='')
    parser.add_argument('--crossover',type=int,default=0,help='')

    # script stuff
    parser.add_argument('--verbose','-v',action='store_true',help='extensive prints')
    parser.add_argument('--slow_run',action='store_true',help='ask user input on every iteration')
    parser.add_argument('--write_model',action='store_true',help='write model out as multiple/all formats')
    parser.add_argument('--write_presolved',action='store_true',help='presolve and write (presolved) model out as multiple/all formats')
    parser.add_argument('--no_solve',action='store_true',help='quit after model creation but before solving')

    parser.add_argument('--solver',type=str,default='gurobi',help='')

    parser.add_argument('--binary_r_map',action='store_true',help='')
    parser.add_argument('--recalc_interval',type=int,default=1,help='')
    parser.add_argument('--rm_coords_interval',type=int,default=1,help='')
    parser.add_argument('--scale_factor',type=float,default=1.0,help='')
    parser.add_argument('--start_map',type=str,help='read a starting r_map')
    parser.add_argument('--spreading_cubes',type=bool,default=True,help='')
    parser.add_argument('--connect_first',type=bool,default=False,help='')

    parser.add_argument('--use_iter_checkpoints',action='store_true',help='write current_r_map')


    args = parser.parse_args()

    if args.verbose:
        VERBOSE = True

    if args.slow_run:
        SLOW_RUN = True

    print(f'Parsed args : {args}')

    return args

def parse_and_package_script_params_from_args(args):

    script_params = {
        'out_filename':args.out_filename,
        'write_model':args.write_model,
        'no_solve':args.no_solve,
        'recalc_interval':args.recalc_interval,
        'rm_coords_interval':args.rm_coords_interval,
        'scale_factor':args.scale_factor,
        'start_map':args.start_map,
        'spreading_cubes':args.spreading_cubes,
        'connect_first':args.connect_first,
        'solver':args.solver,
        'binary_r_map':args.binary_r_map
    }

    return script_params

def _parse_and_package_solver_params_from_args_ortools(args):

    # solver params
    solver_params = {}


    if args.time_limit_secs is not None:
        solver_params.update({'time_sec_limit':args.time_limit_secs})
    if args.output_flag is not None:
        solver_params.update({'output_flag':args.output_flag})

    return solver_params

def _parse_and_package_solver_params_from_args_highs(args):

    # solver params
    solver_params = {}

    if args.time_limit is not None:
        solver_params.update({'time_limit':args.time_limit*60})
    if args.time_limit_secs is not None:
        solver_params.update({'time_limit':args.time_limit_secs})
    if args.threads is not None:
        solver_params.update({'threads':args.threads})
    if args.output_flag is not None:
        solver_params.update({'output_flag':args.output_flag})
    if args.method is not None:
        # rename
        solver_params.update({'solver':args.method})
    if args.kkt_tolerance is not None:
        solver_params.update({'kkt_tolerance':args.kkt_tolerance})
    if args.pdlp_optimality_tolerance is not None:
        solver_params.update({'pdlp_optimality_tolerance':args.pdlp_optimality_tolerance})
    if args.pdlp_scaling is not None:
        solver_params.update({'pdlp_scaling':args.pdlp_scaling})
    if args.crossover is not None:
        _translation = {0:'off',1:'on'}
        solver_params.update({'run_crossover':_translation[args.crossover]})

    return solver_params

def _parse_and_package_solver_params_from_args_gurobi(args):

    # solver params
    solver_params = {}

    if args.time_limit is not None:
        solver_params.update({'TimeLimit':args.time_limit*60})
    if args.time_limit_secs is not None:
        solver_params.update({'TimeLimit':args.time_limit_secs})
    if args.threads is not None:
        solver_params.update({'Threads':args.threads})
    if args.output_flag is not None:
        _translation = {True:1,False:0}    
        solver_params.update({'OutputFlag':args.output_flag})
    if args.method is not None:
        # rename
        _translation = {'ipm':2,'simplex':0,'dual_simplex':1}    
        solver_params.update({'Method':_translation[args.method]})
    if args.crossover is not None:
        solver_params.update({'Crossover':args.crossover})

    return solver_params

def parse_and_package_solver_params_from_args(args):

    if args.solver == 'highs':
        return _parse_and_package_solver_params_from_args_highs(args)
    elif args.solver == 'ortools' or args.solver == 'ortools_model_builder':
        return _parse_and_package_solver_params_from_args_ortools(args)
    elif args.solver == 'gurobi':
        return _parse_and_package_solver_params_from_args_gurobi(args)

    print(f'ERROR: Unrecognized solver {solver}. Exiting...')
    quit()

# Script Stuff
###############################################################################

if __name__ == '__main__':

    main()

