#!/usr/bin/env python3
"""
Random topology statistical analysis:
- Samples random topologies via gen_topo()
- Computes average hops and approximate sparsest cut
- Reports descriptive statistics
- Computes 95% confidence intervals for the mean
- Tests normality of the sample distribution

You must provide:
    gen_topo()
    calc_avg_hops(topo)
    calc_approx_sc(topo)

Example:
    python topo_stats.py --n 200 --bootstrap 20000 --plots

Sequential stopping example:
    python topo_stats.py --max-n 5000 --stop \
        --tol-hops-abs 0.02 --tol-sc-rel 0.02
"""

import argparse
import math
import os
import sys
import time
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import networkx as nx
# SciPy is strongly recommended for robust stats tests/intervals.
from scipy import stats
SCIPY_OK = True

# Gurobi libs
import gurobipy as gp
from gurobipy import GRB


# Matplotlib only used if --plots is set.
try:
    import matplotlib.pyplot as plt
    MPL_OK = True
except Exception:
    MPL_OK = False

VERBOSE = False
INF = 10**10

# -----------------------------
# User-provided functions hook
# -----------------------------
# You should import from your project instead of defining stubs here, e.g.:
# from my_topo_module import gen_topo, calc_avg_hops, calc_approx_sc


class OffDiagMatrix:
    def __init__(self, n, init_val=None):
        """
        Store all off-diagonal (i != j) entries of an n x n matrix.
        (i, j) and (j, i) are distinct.
        """
        self.n = n
        self.size = n * (n - 1)
        self.data = [init_val] * self.size  # arbitrary Python objects

    def _index(self, i, j):
        n = self.n
        if not (0 <= i < n and 0 <= j < n):
            raise IndexError("Index out of bounds")
        # if i == j:
        #     raise IndexError("Diagonal elements are not stored")
        # pack row i, skipping column i
        return i * (n - 1) + (j if j < i else j - 1)

    def _decode_index(self, idx):
        n = self.n
        if not (0 <= idx < self.size):
            raise IndexError("Flat index out of bounds")
        i = idx // (n - 1)
        k = idx % (n - 1)
        j = k if k < i else k + 1
        return (i, j)

    def __getitem__(self, key):
        i, j = key
        if i==j:
            return 0
        return self.data[self._index(i, j)]

    def __setitem__(self, key, value):
        i, j = key
        if i==j:
            return
        self.data[self._index(i, j)] = value

def xyz_to_r(x,y,z,xd,yd,zd):
    return x + y*xd + z*xd*yd

def r_to_xyz(r,xd,yd,zd):
    xy_slice_size = xd*yd

    temp_r = r

    z = temp_r // xy_slice_size
    temp_r = temp_r % xy_slice_size
    y = temp_r // xd
    x = temp_r % xd

    return x,y,z

def init_known_conns(n_routers, cube_dim, x_dim, y_dim, z_dim):

    known_conns = [[0 for _ in range(n_routers)] for __ in range(n_routers)]


    for src in range(n_routers):

        src_x,src_y,src_z = r_to_xyz(src, x_dim, y_dim, z_dim)

        # xpos
        # if not on edge then conn
        if(src_x % cube_dim != cube_dim - 1):
            targ_x = src_x + 1
            targ_y = src_y
            targ_z = src_z
            targ = xyz_to_r(targ_x, targ_y, targ_z, x_dim, y_dim, z_dim)
            known_conns[src][targ] = 1

        # xneg
        # if not on edge then conn
        if(src_x % cube_dim != 0):
            targ_x = src_x - 1
            targ_y = src_y
            targ_z = src_z
            targ = xyz_to_r(targ_x, targ_y, targ_z, x_dim, y_dim, z_dim)
            known_conns[src][targ] = 1

        # ypos
        # if not on edge then conn
        if(src_y % cube_dim != cube_dim - 1):
            targ_x = src_x
            targ_y = src_y + 1
            targ_z = src_z
            targ = xyz_to_r(targ_x, targ_y, targ_z, x_dim, y_dim, z_dim)
            known_conns[src][targ] = 1

        # yneg
        # if not on edge then conn
        if(src_y % cube_dim != 0):
            targ_x = src_x
            targ_y = src_y - 1
            targ_z = src_z
            targ = xyz_to_r(targ_x, targ_y, targ_z, x_dim, y_dim, z_dim)
            known_conns[src][targ] = 1

        # zpos
        # if not on edge then conn
        if(src_z % cube_dim != cube_dim - 1):
            targ_x = src_x
            targ_y = src_y
            targ_z = src_z + 1
            targ = xyz_to_r(targ_x, targ_y, targ_z, x_dim, y_dim, z_dim)
            known_conns[src][targ] = 1

        # zneg
        # if not on edge then conn
        if(src_z % cube_dim != 0):
            targ_x = src_x
            targ_y = src_y
            targ_z = src_z - 1
            targ = xyz_to_r(targ_x, targ_y, targ_z, x_dim, y_dim, z_dim)
            known_conns[src][targ] = 1

    return known_conns

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

def init_valid_conns(n_routers, cube_dim, x_dim, y_dim, z_dim, known_conns=None):

    valid_conns = [[1 for _ in range(n_routers)] for __ in range(n_routers)]

    for i in range(n_routers):

        i_x,i_y,i_z = r_to_xyz(i,x_dim,y_dim,z_dim)

        rel_i_x = i_x % cube_dim
        rel_i_y = i_y % cube_dim
        rel_i_z = i_z % cube_dim

        for j in range(n_routers):
            if (i==j):
                valid_conns[i][j] = 0
                continue

            j_x,j_y,j_z = r_to_xyz(j,x_dim,y_dim,z_dim)

            rel_j_x = j_x % cube_dim
            rel_j_y = j_y % cube_dim
            rel_j_z = j_z % cube_dim

            # interiors
            if not rel_xyz_is_on_face(rel_i_x, rel_i_y, rel_i_z, cube_dim) or not rel_xyz_is_on_face(rel_j_x, rel_j_y, rel_j_z, cube_dim) :
                valid_conns[i][j] = 0
                continue


            # allowed to connect if statically known to connect
            # break early
            if known_conns is not None and known_conns[i][j] > 0:
                # print(f'{i} ({rel_i_x},{rel_i_y},{rel_i_z}) -> {j} ({rel_j_x},{rel_j_y},{rel_j_z}) allowed to conn')
                continue

            # along a dimension D allowed to conn
            #   if other rel dims match and rel dim D differ by cube_dim - 1

            # x pos
            if(rel_i_x == rel_j_x - (cube_dim - 1) and rel_i_y == rel_j_y and rel_i_z == rel_j_z):
                # match so continue
                # print(f'{i} ({rel_i_x},{rel_i_y},{rel_i_z}) -> {j} ({rel_j_x},{rel_j_y},{rel_j_z}) allowed to conn')
                continue
            # x neg
            if(rel_i_x == rel_j_x + (cube_dim - 1) and rel_i_y == rel_j_y and rel_i_z == rel_j_z):
                # match so continue
                # print(f'{i} ({rel_i_x},{rel_i_y},{rel_i_z}) -> {j} ({rel_j_x},{rel_j_y},{rel_j_z}) allowed to conn')
                continue

            # y pos
            if(rel_i_x == rel_j_x and rel_i_y == rel_j_y - (cube_dim - 1) and rel_i_z == rel_j_z):
                # match so continue
                # print(f'{i} ({rel_i_x},{rel_i_y},{rel_i_z}) -> {j} ({rel_j_x},{rel_j_y},{rel_j_z}) allowed to conn')
                continue
            # y neg
            if(rel_i_x == rel_j_x and rel_i_y == rel_j_y + (cube_dim - 1) and rel_i_z == rel_j_z):
                # match so continue
                # print(f'{i} ({rel_i_x},{rel_i_y},{rel_i_z}) -> {j} ({rel_j_x},{rel_j_y},{rel_j_z}) allowed to conn')
                continue

            # z pos
            if(rel_i_x == rel_j_x and rel_i_y == rel_j_y and rel_i_z == rel_j_z - (cube_dim - 1)):
                # match so continue
                # print(f'{i} ({rel_i_x},{rel_i_y},{rel_i_z}) -> {j} ({rel_j_x},{rel_j_y},{rel_j_z}) allowed to conn')
                continue
            # z neg
            if(rel_i_x == rel_j_x and rel_i_y == rel_j_y and rel_i_z == rel_j_z + (cube_dim - 1)):
                # match so continue
                # print(f'{i} ({rel_i_x},{rel_i_y},{rel_i_z}) -> {j} ({rel_j_x},{rel_j_y},{rel_j_z}) allowed to conn')
                continue

            valid_conns[i][j] = 0
    return valid_conns

def calc_conn_dim_w_pos_neg( i,j,x_dim,y_dim,z_dim,cube_dim):
    
    i_x,i_y,i_z = r_to_xyz(i,x_dim,y_dim,z_dim)

    rel_i_x = i_x % cube_dim
    rel_i_y = i_y % cube_dim
    rel_i_z = i_z % cube_dim


    j_x,j_y,j_z = r_to_xyz(j,x_dim,y_dim,z_dim)

    rel_j_x = j_x % cube_dim
    rel_j_y = j_y % cube_dim
    rel_j_z = j_z % cube_dim

    # should be just one type!
    # conn_types = []
    conn_type = None

    if(rel_i_y == rel_j_y and rel_i_z == rel_j_z):
        # conn_types.append('x')

        # pos face to neg face
        if (rel_i_x == cube_dim - 1) and (rel_j_x == 0):
            conn_type = 'x+'
        # neg face
        elif (rel_i_x == 0) and (rel_j_x == cube_dim - 1):
            conn_type = 'x-'
        # intracube conn
        elif j_x >= i_x:
            conn_type = 'x+'
        else:
            conn_type = 'x-'

    if(rel_i_x == rel_j_x and rel_i_z == rel_j_z):
        # conn_types.append('y')
        
        
        # pos face to neg face
        if (rel_i_y == cube_dim - 1) and (rel_j_y == 0):
            conn_type = 'y+'
        # neg face
        elif (rel_i_y == 0) and (rel_j_y == cube_dim - 1):
            conn_type = 'y-'
        # intracube conn
        elif j_y >= i_y:
            conn_type = 'y+'
        else:
            conn_type = 'y-'

    if(rel_i_x == rel_j_x and rel_i_y == rel_j_y):
        # conn_types.append('z')
        # pos face to neg face
        if (rel_i_z == cube_dim - 1) and (rel_j_z == 0):
            conn_type = 'z+'
        # neg face
        elif (rel_i_z == 0) and (rel_j_z == cube_dim - 1):
            conn_type = 'z-'
        # intracube conn
        elif j_z >= i_z:
            conn_type = 'z+'
        else:
            conn_type = 'z-'
    
    if conn_type == None:
        input(f'{i} ({rel_i_x},{rel_i_y},{rel_i_z}) -> {j} ({rel_j_x},{rel_j_y},{rel_j_z}) => conn_type = {conn_type}')

    # return conn_types[0]
    return conn_type

def poss_conns_for_r(r, cube_dim, x_dim, y_dim, z_dim):

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
    

    # print(f'{r} may conn to')
    # for k,v in poss_conns.items():
    #     print(f'\t{k} : {v}')
    
    # input('good?')

    return poss_conns

def conn_set_for_rel_xyz(rel_xyz, dimension, cube_dim, x_dim, y_dim, z_dim):

    rel_x, rel_y, rel_z = rel_xyz
    assert((rel_ix == 0 or rel_x == cube_dim -1) or (rel_y == 0 or rel_y == cube_dim -1) or (rel_z == 0 or rel_z == cube_dim -1))

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

def calc_diameter_flow(adj_mat, valid_conns, n_routers):

    dist_iterator = calc_all_pairs_hops(adj_mat)

    diameter_flows = []
    longest_dist = -1

    for src, hops_dict in dist_iterator:
        for dest in range(n_routers):

            if src >= dest:
                continue

            if valid_conns[src][dest] == 0:
                continue


            try:
                hop_dist = hops_dict[dest]
            except:
                # key error => infinite/unconnected
                return (src,dest)

            if hop_dist > longest_dist:
                longest_dist = hop_dist
                diameter_flows.append( (src,dest) )

    return random.choice(diameter_flows)

def calc_best_link(adj_mat,valid_conns,n_routers):


    lowest_hops = INF
    best_conn = None

    for i in range(n_routers):
        for j in range(i+1,n_routers):

            if valid_conns[i][j] == 0:
                continue
            
            # skip already connected
            if adj_mat[i][j] > 0:
                continue

            # set
            adj_mat[i][j] = 1
            adj_mat[j][i] = 1

            # print(f'Trying {(i,j)}')
            avg_hops = calc_avg_hops(adj_mat)
            # print(f'Avg hops {avg_hops} when best is {lowest_hops}')

            if avg_hops < lowest_hops:
                lowest_hops = avg_hops
                best_conn = (i,j)

            # reset
            adj_mat[i][j] = 0
            adj_mat[j][i] = 0

    # if unconnected no matter what then use a diameter
    if best_conn is None:
        return calc_diameter_flow(adj_mat,valid_conns,n_routers)

    return best_conn

def calc_all_pairs_hops(adj_mat):

    G = create_an_nwx_G_from_a_map(adj_mat)

    return nx.all_pairs_bellman_ford_path_length(G)

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

def calc_options(known_conns, valid_conns, n_routers):

    options = []

    for i in range(n_routers):
        for j in range(n_routers):
            if valid_conns[i][j] > 0 and known_conns[i][j] == 0:
                options.append((i,j))

    return options

def update_valid_conns(new_conn, valid_conns, xyzc_dims):

    (x_dim, y_dim, z_dim, cube_dim) = xyzc_dims

    (i,j) = new_conn

    # print(f'updating valid for {new_conn}')

    # get conn type and void all of those
    i_to_j_type = calc_conn_dim_w_pos_neg( i,j,x_dim,y_dim,z_dim,cube_dim)
    j_to_i_type = calc_conn_dim_w_pos_neg( j,i,x_dim,y_dim,z_dim,cube_dim)

    # print(f'i_to_j_type = {i_to_j_type}')
    # print(f'j_to_i_type = {j_to_i_type}')

    i_poss_conns = poss_conns_for_r(i, cube_dim, x_dim, y_dim, z_dim)
    j_poss_conns = poss_conns_for_r(j, cube_dim, x_dim, y_dim, z_dim)

    # print(f'i_poss_conns = {i_poss_conns}')
    # print(f'j_poss_conns = {j_poss_conns}')


    # void those that are along i_to_j
    for option in i_poss_conns[i_to_j_type]:
        valid_conns[i][option] = 0
        valid_conns[option][i] = 0

    # void those that are along j_to_i
    for option in j_poss_conns[j_to_i_type]:
        valid_conns[j][option] = 0
        valid_conns[option][j] = 0

    return valid_conns


# -----------------------------
# User-provided functions hook
# -----------------------------
# You should import from your project instead of defining stubs here, e.g.:
# from my_topo_module import gen_topo, calc_avg_hops, calc_approx_sc


def random_gen(xyzc_dims):

    (x_dim, y_dim, z_dim, cube_dim) = xyzc_dims
    n_routers = x_dim*y_dim*z_dim
    n_ports = 6

    known_conns = init_known_conns(n_routers, cube_dim, x_dim, y_dim, z_dim)
    valid_conns = init_valid_conns(n_routers, cube_dim, x_dim, y_dim, z_dim)

    
    n_known = sum([sum(row) for row in known_conns])
    n_remaining = n_routers*n_ports - n_known

    # print(f'# known = {n_known}')
    # print(f'# remaining = {n_remaining}')
    
    # print(f'avg hops {calc_avg_hops(known_conns)}')

    # dist_dict = nx.floyd_warshall(G)

    while n_remaining > 0:


        avail_conns = calc_options(known_conns, valid_conns, n_routers)


        (i,j) = random.choice(avail_conns)

        # symmetry
        second_conn = (j,i)


        known_conns[i][j] = 1
        known_conns[j][i] = 1

        valid_conns = update_valid_conns((i,j), valid_conns, xyzc_dims)
        # valid_conns = update_valid_conns((j,i), valid_conns, xyzc_dims)



        n_known = sum([sum(row) for row in known_conns])
        n_remaining = n_routers*n_ports - n_known

        # print(f'rand, # routers {n_routers}')
        # print(f'# known = {n_known}')
        # print(f'# remaining = {n_remaining}')

    # print('done')

    return known_conns

def calc_avg_hops(adj_mat):

    G = create_an_nwx_G_from_a_map(adj_mat)

    try:
        avg_hops = nx.average_shortest_path_length(G)
    except:
        avg_hops = INF

    return avg_hops


def calc_approx_sc(r_map, one_leg=True, dor_heur=False, xyzc_dims=None):


    # Constants
    # --------------------------------------------------------------------------------
    n_routers = len(r_map)


    demand = 1.0
    capacity = 1.0

    # convenience
    # list of edges (as tuples)
    edge_list = []
    for i in range(n_routers):
        for j in range(n_routers):
            if i >= j:
                continue
            if r_map[i][j] > 0:
                edge_list.append((i,j))



    try:
    # if True:
        # Create a new model
        model_base_name = "approxsc_lp"
        m = gp.Model(model_base_name)


        # Variables
        # --------------------------------------------------------------------------------

        # dist = UpperTriMatrix(n_routers)
        dist = OffDiagMatrix(n_routers)
        for i in range(n_routers):
            # for j in range(i+1,n_routers):
            for j in range(n_routers):
                if i==j:
                    continue

                myvarname = f'var_dist_{i}r_{j}r'

                # if using abs
                dist[i,j] = m.addVar(lb=0, ub=1.0, vtype=GRB.CONTINUOUS, name=myvarname)

        # print('Created variables')

        # Constraints
        # --------------------------------------------------------------------------------

        myconstrname = 'constr_dist_sum'
        # m.addConstr(gp.quicksum( [dist[i,j]  for i in range(n_routers) for j in range(i+1,n_routers,1) ] ) >= 1, name=myconstrname)
        # m.addConstr(gp.quicksum( [dist[i,j] for i in range(n_routers) for j in range(n_routers) if i!=j  ] ) >= 1, name=myconstrname)

        unity_expr = gp.LinExpr()
        for i in range(n_routers):
            for j in range(n_routers):
                if i==j:
                    continue
                unity_expr += dist[i,j]
        m.addConstr(unity_expr >= 1, name=myconstrname)

        # print('Completed unity constr')

        n_iters = 0

        # tri ineq
        for i in range(n_routers):
            # for j in range(i+1,n_routers):
            for j in range(n_routers):
                if i==j:
                    continue
                for k in range(n_routers):

                    # print(f"Tri ineq (i,j,k) = {(i,j,k)}")

                    # if k==i or k==j:
                    #     continue
                    if k==i:
                        continue

                    if r_map[i][j] == 0 and r_map[i][k] == 0 and r_map[k][j] == 0:
                        continue
                    
                    if one_leg and r_map[i][k] == 0:
                        continue

                    # if dor_heur and r_map[i][k] == 0:
                    #     continue

                    if dor_heur and not ijk_dor_relevant_v3(i,j,k,xyzc_dims):
                        continue

                    # if dor_heur and not ijk_dor_relevant(i,j,k,xyzc_dims):
                    #     continue

                        

                    myconstrname = f'constr_dist_tri_ineq_{i}i_{k}k_{j}j'
                    m.addConstr(dist[i,j] <= dist[i,k] + dist[k,j], name=myconstrname)
                    n_iters += 1

        # print(f'n_iters={n_iters}')

        # print('Completed tri ineq constrs')


        # Objective(s)
        # --------------------------------------------------------------------------------


        # define dist_sum
        dist_sum_expr = gp.LinExpr()
        for i in range(n_routers):
            # for j in range( i+1, n_routers):
            for j in range( n_routers):
                if r_map[i][j] == 0:
                    continue

                if VERBOSE:
                    print(f'adding edge ({i},{j}) to obj')

                dist_sum_expr += capacity*dist[(i,j)]

        m.setObjective(dist_sum_expr, GRB.MINIMIZE)

        # print('Completed objective')


        # Params and Model Output
        # --------------------------------------------------------------------------------

        # Params
        m.setParam('Method', 2)
        m.setParam('Crossover', 0)
        m.setParam('BarOrder', 0)
        m.setParam('OutputFlag', 0)

        # Model Output

        write_model = False #True #False
        model_types = ['lp','mps']
        if write_model:
            for mt in model_types:
                out_model_name = f'files/models/{model_base_name}.{mt}'
                m.write(out_model_name)
                print(f'Wrote to {out_model_name}')


            try:
                m.write("model.ilp")
            except:
                print(f'Model is feasible')

            try:
                m.write("model.dlp")
            except:
                print(f'Dual cannot be written')

        # Solve
        # --------------------------------------------------------------------------------


        solve_start_t = time.time()
        # Optimize model
        m.optimize()
        solve_end_t = time.time()

        # Output
        # --------------------------------------------------------------------------------

        # Check for infeasibility
        if m.status == GRB.INFEASIBLE:
            print("Model is infeasible.")

            # Attempt to find Irreducible Infeasible Set (IIS)
            m.computeIIS()
            m.write("model.ilp")  # Write model with IIS information
            print("IIS written to model.ilp")

            # Optionally, print IIS details
            m.write("iis.ilp")
            print("IIS details written to iis.ilp")

        obj_val = float(m.ObjVal)

        return obj_val

    # Error Handling
    # --------------------------------------------------------------------------------

    except AttributeError as e:
        print(f"Encountered an attribute error : {e}")

    except gp.GurobiError as e:
        print(f"Error code {e.errno}: {e}")

# -----------------------------
# Utility statistics
# -----------------------------

def describe(arr: np.ndarray) -> dict:
    arr = np.asarray(arr, dtype=float)
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size >= 2 else float("nan"),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
    }


def t_confidence_interval_mean(arr: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    """
    1-alpha CI for mean using Student-t:
        mean +/- t_{1-alpha/2, n-1} * s/sqrt(n)
    """
    arr = np.asarray(arr, dtype=float)
    n = arr.size
    if n < 2 or not SCIPY_OK:
        return (float("nan"), float("nan"))
    mean = float(np.mean(arr))
    s = float(np.std(arr, ddof=1))
    tcrit = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
    half = tcrit * s / math.sqrt(n)
    return (mean - half, mean + half)


def bootstrap_ci_mean(arr: np.ndarray, alpha: float = 0.05, n_boot: int = 20000,
                      rng: Optional[np.random.Generator] = None) -> Tuple[float, float]:
    """
    Percentile bootstrap CI for mean. Robust to non-normality.

    Returns:
        (lower, upper) for 1-alpha CI.
    """
    arr = np.asarray(arr, dtype=float)
    n = arr.size
    if n < 2:
        return (float("nan"), float("nan"))
    if rng is None:
        rng = np.random.default_rng(0)

    # Vectorized bootstrap
    idx = rng.integers(0, n, size=(n_boot, n))
    samples = arr[idx]
    means = np.mean(samples, axis=1)

    lo = float(np.percentile(means, 100 * (alpha / 2)))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return (lo, hi)


def normality_tests(arr: np.ndarray) -> dict:
    """
    Runs multiple normality tests.
    Returns p-values where applicable.

    Notes:
    - Shapiro-Wilk is common, but can be overly sensitive for large n.
    - For large n, tiny deviations from normal will be rejected; interpret practically.
    """
    arr = np.asarray(arr, dtype=float)
    out = {}
    n = arr.size

    if not SCIPY_OK or n < 3:
        out["note"] = "SciPy not available or sample too small; skipping tests."
        return out

    # Shapiro-Wilk: recommended for n <= ~5000
    if n <= 5000:
        w, p = stats.shapiro(arr)
        out["shapiro_W"] = float(w)
        out["shapiro_p"] = float(p)
    else:
        out["shapiro_note"] = "n > 5000: skipping Shapiro-Wilk (common practice)."

    # D’Agostino & Pearson K^2 test: requires n >= 8
    if n >= 8:
        k2, p = stats.normaltest(arr)
        out["dagostino_k2"] = float(k2)
        out["dagostino_p"] = float(p)

    # Anderson-Darling: returns statistic and critical values; no p-value
    ad = stats.anderson(arr, dist="norm")
    out["anderson_stat"] = float(ad.statistic)
    out["anderson_crit_5pct"] = float(ad.critical_values[list(ad.significance_level).index(5.0)])

    # Jarque–Bera: based on skewness/kurtosis
    jb, p = stats.jarque_bera(arr)
    out["jarque_bera"] = float(jb)
    out["jarque_p"] = float(p)

    return out


def ci_halfwidth(ci: Tuple[float, float]) -> float:
    if any(math.isnan(x) for x in ci):
        return float("nan")
    return 0.5 * (ci[1] - ci[0])


def ensure_dir(path: str) -> None:
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def save_plots(arr: np.ndarray, name: str, out_dir: str) -> None:
    """
    Saves histogram and QQ plot.
    """
    if not MPL_OK:
        return
    ensure_dir(out_dir)

    # Histogram
    plt.figure()
    plt.hist(arr, bins="auto")
    plt.title(f"{name} histogram")
    plt.xlabel(name)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}_hist.png"), dpi=200)
    plt.close()

    # QQ plot (requires SciPy)
    if SCIPY_OK:
        plt.figure()
        stats.probplot(arr, dist="norm", plot=plt)
        plt.title(f"{name} QQ plot vs Normal")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{name}_qq.png"), dpi=200)
        plt.close()


@dataclass
class RunConfig:
    one_leg: bool
    alpha: float
    n: int
    max_n: int
    stop_when_tight: bool
    tol_hops_abs: Optional[float]
    tol_sc_abs: Optional[float]
    tol_sc_rel: Optional[float]
    bootstrap_n: int
    seed: int
    plots: bool
    out_dir: str
    csv_path: str
    xyzc_dims: tuple


def parse_args() -> RunConfig:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.05, help="Significance level (alpha=0.05 -> 95% CI)")
    ap.add_argument("--n", type=int, default=200, help="Number of trials (ignored if --stop is enabled)")
    ap.add_argument("--max-n", type=int, default=5000, help="Maximum trials for sequential stopping")

    ap.add_argument("--stop", action="store_true",
                    help="Sequential stopping: run until CI half-width tolerances satisfied (or max-n).")

    ap.add_argument("--tol-hops-abs", type=float, default=None,
                    help="Stop condition: abs CI half-width for avg hops <= tol (e.g., 0.02).")
    ap.add_argument("--tol-sc-abs", type=float, default=None,
                    help="Stop condition: abs CI half-width for approx sc <= tol.")
    ap.add_argument("--tol-sc-rel", type=float, default=None,
                    help="Stop condition: rel CI half-width for approx sc <= tol * mean (e.g., 0.02).")

    ap.add_argument("--bootstrap", type=int, default=20000, help="Bootstrap resamples for CI")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for reproducibility")
    ap.add_argument("--plots", action="store_true", help="Save histograms and QQ plots")
    ap.add_argument("--out-dir", type=str, default="topo_stats_out", help="Output directory")
    ap.add_argument("--csv", type=str, default="topo_stats_out/samples.csv", help="CSV output path")

    ap.add_argument('--xyzc_dims',nargs='+',type=int,help='type without parenthesis and use spaces, no commas')
    ap.add_argument('--one_leg',action='store_true',help='whether to only consider tri inequalities where (i,k) in E')

    args = ap.parse_args()

    return RunConfig(
        alpha=args.alpha,
        n=args.n,
        max_n=args.max_n,
        stop_when_tight=args.stop,
        tol_hops_abs=args.tol_hops_abs,
        tol_sc_abs=args.tol_sc_abs,
        tol_sc_rel=args.tol_sc_rel,
        bootstrap_n=args.bootstrap,
        seed=args.seed,
        plots=args.plots,
        out_dir=args.out_dir,
        csv_path=args.csv,
        xyzc_dims=tuple(args.xyzc_dims)
    )


def summarize_metric(name: str, arr: np.ndarray, alpha: float, bootstrap_n: int, rng: np.random.Generator) -> dict:
    d = describe(arr)
    ci_t = t_confidence_interval_mean(arr, alpha=alpha)
    ci_b = bootstrap_ci_mean(arr, alpha=alpha, n_boot=bootstrap_n, rng=rng)
    nt = normality_tests(arr)

    d.update({
        "ci_t_lo": float(ci_t[0]),
        "ci_t_hi": float(ci_t[1]),
        "ci_t_half": float(ci_halfwidth(ci_t)),
        "ci_boot_lo": float(ci_b[0]),
        "ci_boot_hi": float(ci_b[1]),
        "ci_boot_half": float(ci_halfwidth(ci_b)),
    })
    d["normality"] = nt
    return d


def stop_criteria_met(hops_stats: dict, sc_stats: dict, cfg: RunConfig) -> bool:
    """
    Uses bootstrap CI half-widths as the stopping criterion (more robust).
    """
    hop_half = hops_stats.get("ci_boot_half", float("nan"))
    sc_half = sc_stats.get("ci_boot_half", float("nan"))

    ok = True

    if cfg.tol_hops_abs is not None:
        ok = ok and (hop_half <= cfg.tol_hops_abs)

    if cfg.tol_sc_abs is not None:
        ok = ok and (sc_half <= cfg.tol_sc_abs)

    if cfg.tol_sc_rel is not None:
        sc_mean = sc_stats.get("mean", float("nan"))
        if sc_mean == 0 or math.isnan(sc_mean):
            ok = False
        else:
            ok = ok and (sc_half <= cfg.tol_sc_rel * abs(sc_mean))

    # If no tolerances specified, can't stop.
    if (cfg.tol_hops_abs is None) and (cfg.tol_sc_abs is None) and (cfg.tol_sc_rel is None):
        return False

    return ok


def run_trials(cfg: RunConfig) -> Tuple[np.ndarray, np.ndarray]:

    xyzc_dims = cfg.xyzc_dims
    one_leg = cfg.one_leg


    rng = np.random.default_rng(cfg.seed)

    hops: List[float] = []
    sc: List[float] = []

    target_n = cfg.max_n if cfg.stop_when_tight else cfg.n
    ensure_dir(cfg.out_dir)
    ensure_dir(os.path.dirname(cfg.csv_path))

    # Write CSV header
    with open(cfg.csv_path, "w", encoding="utf-8") as f:
        f.write("trial,avg_hops,approx_sc,elapsed_sec\n")

    t0 = time.time()

    topo_gen_cumul_time = 0
    avg_hops_cumul_time = 0
    approx_sc_cumul_time = 0

    for i in range(target_n):
        print(f"Iteration {i}")

        t_start_topo_gen = time.time()
        topo = random_gen( xyzc_dims)
        t_end_topo_gen = time.time()
        print(f"Topo gen took {t_end_topo_gen - t_start_topo_gen} seconds")
        topo_gen_cumul_time += t_end_topo_gen - t_start_topo_gen

        t_start_avg_hops = time.time()
        h = float(calc_avg_hops(topo))
        t_end_avg_hops = time.time()
        avg_hops_cumul_time += t_end_avg_hops - t_start_avg_hops

        t_start_approx_sc = time.time()
        s = float(calc_approx_sc(topo, one_leg=one_leg))
        t_end_approx_sc = time.time()
        approx_sc_cumul_time += t_end_approx_sc - t_start_approx_sc
        # input(f"avg hops = {h}, approxsc = {s}")

        hops.append(h)
        sc.append(s)

        elapsed = time.time() - t0
        with open(cfg.csv_path, "a", encoding="utf-8") as f:
            f.write(f"{i},{h},{s},{elapsed}\n")

        # Sequential stopping check every few iterations to reduce bootstrap overhead
        if cfg.stop_when_tight and (i + 1) >= 20 and ((i + 1) % 10 == 0):
            hops_arr = np.array(hops, dtype=float)
            sc_arr = np.array(sc, dtype=float)

            hops_stats = summarize_metric("avg_hops", hops_arr, cfg.alpha, cfg.bootstrap_n, rng)
            sc_stats = summarize_metric("approx_sc", sc_arr, cfg.alpha, cfg.bootstrap_n, rng)

            print(f"hops_stats = {hops_stats}")
            print(f"sc_stats = {sc_stats}")

            if stop_criteria_met(hops_stats, sc_stats, cfg):
                break
            else:
                print(f"Stop criteria not met.")


    print(f"Topo gen cumul time: {topo_gen_cumul_time} seconds")
    print(f"Avg hops cumul time: {avg_hops_cumul_time} seconds")
    print(f"Approx sc cumul time: {approx_sc_cumul_time} seconds")

    return (np.array(hops, dtype=float), np.array(sc, dtype=float))


def print_report(hops_arr: np.ndarray, sc_arr: np.ndarray, cfg: RunConfig) -> None:
    rng = np.random.default_rng(cfg.seed + 12345)

    hops_stats = summarize_metric("avg_hops", hops_arr, cfg.alpha, cfg.bootstrap_n, rng)
    sc_stats = summarize_metric("approx_sc", sc_arr, cfg.alpha, cfg.bootstrap_n, rng)

    def fmt_ci(stats_dict: dict, which: str) -> str:
        lo = stats_dict[f"ci_{which}_lo"]
        hi = stats_dict[f"ci_{which}_hi"]
        half = stats_dict[f"ci_{which}_half"]
        return f"[{lo:.6g}, {hi:.6g}] (half-width={half:.6g})"

    print("\n=== Random Topology Statistical Report ===")
    print(f"Trials: {len(hops_arr)}")
    print(f"Confidence: {(1.0 - cfg.alpha)*100:.1f}%  (alpha={cfg.alpha})")
    print(f"Bootstrap resamples: {cfg.bootstrap_n}")
    print(f"CSV samples: {cfg.csv_path}")

    print("\n--- avg_hops ---")
    print(f"mean={hops_stats['mean']:.6g}  std={hops_stats['std']:.6g}  "
          f"min={hops_stats['min']:.6g}  max={hops_stats['max']:.6g}")
    print(f"median={hops_stats['median']:.6g}  p05={hops_stats['p05']:.6g}  p95={hops_stats['p95']:.6g}")
    if SCIPY_OK:
        print(f"95% CI (t):       {fmt_ci(hops_stats, 't')}")
    print(f"95% CI (bootstrap): {fmt_ci(hops_stats, 'boot')}")
    print("Normality tests:", hops_stats["normality"])

    print("\n--- approx_sc ---")
    print(f"mean={sc_stats['mean']:.6g}  std={sc_stats['std']:.6g}  "
          f"min={sc_stats['min']:.6g}  max={sc_stats['max']:.6g}")
    print(f"median={sc_stats['median']:.6g}  p05={sc_stats['p05']:.6g}  p95={sc_stats['p95']:.6g}")
    if SCIPY_OK:
        print(f"95% CI (t):       {fmt_ci(sc_stats, 't')}")
    print(f"95% CI (bootstrap): {fmt_ci(sc_stats, 'boot')}")
    print("Normality tests:", sc_stats["normality"])

    # Plots
    if cfg.plots:
        if not MPL_OK:
            print("\n[plots] Matplotlib not available; skipping plot generation.")
        else:
            save_plots(hops_arr, "avg_hops", cfg.out_dir)
            save_plots(sc_arr, "approx_sc", cfg.out_dir)
            print(f"\nSaved plots to: {cfg.out_dir}/")


def main() -> int:
    cfg = parse_args()

    if cfg.plots and not MPL_OK:
        print("Warning: --plots requested but matplotlib is not available.", file=sys.stderr)

    if not SCIPY_OK:
        print("Warning: SciPy not available. Normality tests and t-based CI will be limited.",
              file=sys.stderr)

    hops_arr, sc_arr = run_trials(cfg, )
    print_report(hops_arr, sc_arr, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
