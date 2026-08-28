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

'''

Description:
    Route to reduce the maximum channel load (balancing) => MCLB
    Modifications for destination based and robust (fault tolerance)

Loosely based on:
    MCF idea from https://dl.acm.org/doi/10.1145/77600.77620
    Symmetry from https://dl.acm.org/doi/10.1145/777412.777444

'''

# Gurobi libs
import gurobipy as gp
from gurobipy import GRB

# regular libs
import argparse
from collections import deque
import ast
import os
import time
import sys
from collections import defaultdict
import tracemalloc

# pipd
import matplotlib.pyplot as plt
import networkx as nx

# locals
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "..", "python_scripts"))

from tpuv4_symmetry import TPUv4_Symmetry

# globals
global VERBOSE
VERBOSE = False
global ASSERT_BINARY_MAP
ASSERT_BINARY_MAP = True

# constants
INF = 10**10
SCRATCH_DIR = '/scratch/negishi/green456'


# Regular Functions
################################################################################


def _read_rss_gb():
    # Linux: /proc/self/status is cheap and reliable
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    kb = int(line.split()[1])
                    return kb / (1024 * 1024)
    except FileNotFoundError:
        return None

def mem_checkpoint(tag, model=None):
    rss = _read_rss_gb()
    if rss is not None:
        print(f"[mem] {tag:28s} RSS={rss:8.2f} GB")
    else:
        print(f"[mem] {tag:28s} RSS=unknown")

    if model is not None:
        try:
            model.update()
            nz = model.getAttr("NumNZs")
            print(f"       model: vars={model.NumVars:,} constrs={model.NumConstrs:,} nz={nz:,}")
        except Exception:
            pass

    print(f"       time={time.strftime('%Y-%m-%d %H:%M:%S')}")

def tracemalloc_start():
    tracemalloc.start(25)

def tracemalloc_report(tag, limit=20):
    snap = tracemalloc.take_snapshot()
    top = snap.statistics("lineno")
    print(f"\n[tracemalloc] {tag}")
    for stat in top[:limit]:
        print(stat)
    print("")

def get_shape(nested_list):
    if isinstance(nested_list, list):
        return [len(nested_list)] + get_shape(nested_list[0])
    else:
        return []

def ingest_map(path_name):

    if True:
        print(f'Ingesting r map ({path_name})')

    adj_list = []
    adj_mat = []

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
            adj_list.append( [r for r, c in enumerate(r_conns) if c > 0] )

    return adj_mat, adj_list


def print_path(p):

    print(f'path {p[0]} to {p[-1]} (len {len(p)-1}): ',end='')

    l = len(p)
    for i in range(l-1):
        e = p[i]
        print(f'{e}->',end='')
    print(f'{p[-1]}')

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

def calc_all_pairs_hops(adj_mat):

    G = create_an_nwx_G_from_a_map(adj_mat)

    return nx.all_pairs_bellman_ford_path_length(G)

def calculate_min_hop_paths(r_map, hop_dists, allow_nonmin=0):

    n_routers = len(r_map)

    short_paths = []

    for src in range(n_routers):

        short_paths.append([])

        if VERBOSE:
            print(f'Min hop paths for src {src}')

        for dest in range(n_routers):
            short_paths[src].append([])

            this_path_list = []

            if src == dest:
                # path is nonexistent
                this_path_list.append(src)
                short_paths[src][dest].append(this_path_list)
                continue


            # perform psuedo-BFS

            shortest_dist = hop_dists[src][dest] + allow_nonmin

            if VERBOSE:
                print(f'Searching for path {src}->{dest} of dist {shortest_dist}')


            queue = deque()

            path = []
            path.append(src)
            queue.append(path.copy())

            while queue:
                path = queue.popleft()
                last = path[-1]

                # only consider the minimal paths
                if len(path) - 1 > shortest_dist:
                    if VERBOSE:
                        print(f'path {path} (len {len(path)}) > shortest {shortest_dist}')
                    continue

                if last == dest:
                    this_path_list.append(path)
                    if VERBOSE:
                        print_path(path)

                for i in range(n_routers):

                    # only consider neighbors
                    if r_map[last][i] == 0:
                        continue

                    # if self.is_not_visited(i, path):
                    if not i in path:
                        new_path = path.copy()
                        new_path.append(i)
                        queue.append(new_path)


            short_paths[src][dest] = this_path_list.copy()

            if VERBOSE:
                print(f'Found {src}->{dest}={this_path_list}')

            # end dest loop

    if VERBOSE:
        print(f'done with min hop paths')

    if VERBOSE:
        for i, src_paths in enumerate(short_paths):
            print(f'{i}->')
            for j, paths in enumerate(src_paths):
                print(f'\t{j} : {paths}')

    return short_paths

def stream_raw_paths(filepath):
    with open(filepath, "rb", buffering=1024*1024) as inf:
        for bline in inf:
            bline = bline.strip()
            if not bline:
                # skip empty
                continue
            # split bytes into byte tokens, convert each to int
            row = [int(tok) for tok in bline.split()]
            yield row

def ingest_path_list(path_name, n_routers, lb_flow=None, ub_flow=None, src_set=None, override_ingest_all=False):
    global VERBOSE
    if VERBOSE:
        print(f'Ingesting path list {path_name}')

    line_num = 0
    allpath_dict = defaultdict(list)
    for path in stream_raw_paths(path_name):

        s = path[0]
        d = path[-1]

        skip = False
        above_upper = False
        if lb_flow:
            (lb_s, lb_d) = lb_flow
            if s < lb_s:
                skip = True
                # print(f's<lb_s')
            elif s==lb_s and d < lb_d:
                skip = True
                # print(f's==lb_s and d < lb_d')
        if ub_flow:
            (ub_s, ub_d) = ub_flow
            if s > ub_s:
                skip = True
                above_upper = True
                # print(f's > ub_s')
            elif s==ub_s and d >= ub_d:
                skip = True
                above_upper = True
                # print(f's==ub_s and d > ub_d')
        
        if src_set:
            if s not in src_set:
                skip = True
        

        if above_upper:
            return allpath_dict

        if skip and not override_ingest_all:
            continue
        
        allpath_dict[(s,d)].append(path)

        line_num += 1
        if line_num % 1_000_000 == 0:
            print(f'read {line_num}')

    return allpath_dict

def create_nwx_G_from_map(r_map, n_routers):

    # directed =  False

    G = nx.DiGraph()

    for src in range(n_routers):
        for dest in range(n_routers):

            if(src == dest):
                continue

            # if not directed and src > dest:
            #     continue

            if(r_map[src][dest] < 1):
                continue

            # print(f'connecting {src} -> {dest}')

            G.add_edge(src,dest)

    return G

def nwx_all_shortest_paths(r_map):

    n_routers = len(r_map)

    G = create_nwx_G_from_map(r_map, n_routers)

    all_paths = [ [ [] for _ in range(n_routers)] for __ in range(n_routers)]

    for src in range(n_routers):
        for dest in range(n_routers):

            if(src == dest):
                all_paths[src][dest].append([src])
                continue

            short_path_generator = nx.all_shortest_paths(G,src,dest)
            short_path_list = list()
            short_path_list += short_path_generator

            # input(f'{src}->{dest} : {short_path_generator} = {short_path_list}')
            all_paths[src][dest] = short_path_list

        # print(f'completed src {src}')

    print(f'Completed all short path creation')

    return all_paths

def output_pathlist( path_list, base_file_name, paths_output_path_prefix):

    full_name = f'{base_file_name}.paths'

    full_out_path = os.path.join(paths_output_path_prefix, \
            full_name)

    with open(full_out_path, 'w+') as of:
        of.write(base_file_name + '\n')
        for path in path_list:
            of.write(f'{path}\n')

    print(f'Wrote to {full_out_path}')


# Basic
# --------------------------------------------------------------------------------

def r_to_xyz(r,xyzc_dims):
    (x_dim, y_dim, z_dim, cube_dim) = xyzc_dims

    xy_slice_size = x_dim*y_dim

    temp_r = r

    z = temp_r // xy_slice_size
    temp_r = temp_r % xy_slice_size
    y = temp_r // x_dim
    x = temp_r % x_dim

    return x,y,z

def xyz_to_r(x,y,z,xyzc_dims):
    (x_dim, y_dim, z_dim, cube_dim) = xyzc_dims

    return x + y*x_dim + z*x_dim*y_dim

def rel_xyz_is_on_face(rel_x, rel_y, rel_z, xyzc_dims):
    (x_dim, y_dim, z_dim, cube_dim) = xyzc_dims

    if rel_x == 0 or rel_x == cube_dim - 1:
        return True
    if rel_y == 0 or rel_y == cube_dim - 1:
        return True
    if rel_z == 0 or rel_z == cube_dim - 1:
        return True
    return False

def r_to_rel_xyz(r, xyzc_dims):
    (x_dim, y_dim, z_dim, cube_dim) = xyzc_dims

    r_x,r_y,r_z = r_to_xyz(r, xyzc_dims)

    rel_r_x = r_x % cube_dim
    rel_r_y = r_y % cube_dim
    rel_r_z = r_z % cube_dim

    return rel_r_x, rel_r_y, rel_r_z

def r_to_rel_r(r, xyzc_dims):
    (rel_x, rel_y, rel_z) = r_to_rel_xyz(r,xyzc_dims)
    return xyz_to_r(rel_x, rel_y, rel_z, xyzc_dims)


def calc_pos_delta(r,ref,xyzc_dims):
    (x_dim, y_dim, z_dim, cube_dim) = xyzc_dims

    (r_x, r_y, r_z) = r_to_xyz(r,xyzc_dims)
    (ref_x, ref_y, ref_z) = r_to_xyz(ref,xyzc_dims)

    d_x = (r_x - ref_x)
    d_y = (r_y - ref_y)
    d_z = (r_z - ref_z)

    return (d_x, d_y, d_z)


def calc_opt_conn_type(s, d, xyzc_dims):


    rel_s_x, rel_s_y, rel_s_z = r_to_rel_xyz(s, xyzc_dims)
    rel_d_x, rel_d_y, rel_d_z = r_to_rel_xyz(d, xyzc_dims)
    
    
    if (not rel_xyz_is_on_face(rel_s_x, rel_s_y, rel_s_z, xyzc_dims)) or (not rel_xyz_is_on_face(rel_d_x, rel_d_y, rel_d_z, xyzc_dims)):
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

def ocs_id(i, j, xyzc_dims):

    ij_conn_type = calc_opt_conn_type(i,j, xyzc_dims)

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

    rel_x, rel_y, rel_z = r_to_rel_xyz(representative, xyzc_dims)

    if 'x' in ij_conn_type:
        base_val = 0
        return base_val + rel_y + cube_dim*rel_z

    elif 'y' in ij_conn_type:
        base_val = 16
        return base_val + rel_x + cube_dim*rel_z

    elif 'z' in ij_conn_type:
        base_val = 32
        return base_val + rel_x + cube_dim*rel_y


def partition_path_list(path_list, sorted_keys_list, key_idx, partition_size, partition_metric='flows'):

    partitioned_path_list = {}

    if partition_metric == 'flows':
        lb = key_idx
        ub = min(key_idx+partition_size, len(sorted_keys_list))
        partitioned_path_list = { k : path_list[k] for k in sorted_keys_list[lb:ub]}

        key_idx = ub

    elif partition_metric == 'paths':

        runsum = 0
        while runsum < partition_size:

            key = sorted_keys_list[key_idx]
            this_pl = path_list[key]
            partitioned_path_list.update({ key : this_pl })

            runsum += len(this_pl)

            key_idx += 1

    else:
        print(f'partition_path_list() :: {partition_metric} :: UNIMPLEMENTED')
        quit()

    # print(f'partitioned_path_list = {partitioned_path_list}')
    # print(f'for partition of {loop_iter*partition_size}:{ub}')
    # input('cont?')

    return partitioned_path_list, key_idx

def sort_keys(path_list, sort_type='path_diversity', order='increasing'):

    sorted_keys_list = None

    # increasing order of number of paths
    if sort_type == 'path_diversity' and order == 'increasing':

        sorted_keys_list = sorted(path_list.keys(), key=lambda k: len(path_list[k]))
    
    #decreasing order of number of paths
    elif sort_type == 'path_diversity' and order == 'decreasing':

        sorted_keys_list = sorted(path_list.keys(), key=lambda k: len(path_list[k]), reverse=True)
    else:
        print('UNIMPLEMENTED')


    return sorted_keys_list

def setup_solver_params(args):

    # solver params
    solver_params = {}

    if args.time_limit:
        solver_params.update({'TimeLimit':args.time_limit*60})
    if args.threads:
        solver_params.update({'Threads':args.threads})
    if args.concurrent_mip:
        solver_params.update({'ConcurrentMIP':args.concurrent_mip})
    if args.mip_focus:
        solver_params.update({'MIPFocus':args.mip_focus})
    if args.heuristic_ratio:
        solver_params.update({'Heuristics':args.heuristic_ratio})
    if args.symmetry_detection:
        solver_params.update({'Symmetry':args.symmetry_detection})
    if args.barrier_iter_limit:
        solver_params.update({'BarIterLimit':args.barrier_iter_limit})
    if args.iter_limit:
        solver_params.update({'IterationLimit':args.iter_limit})
    if args.cut_passes:
        solver_params.update({'CutPasses':args.cut_passes})
    if args.method:
        solver_params.update({'Method':args.method})
    if args.node_method:
        solver_params.update({'NodeMethod':args.node_method})
    if args.crossover:
        solver_params.update({'Crossover':args.crossover})
    if args.crossover_basis:
        solver_params.update({'CrossoverBasis':args.crossover_basis})
    if args.no_rel_heur_time:
        solver_params.update({'NoRelHeurTime':args.no_rel_heur_time})
    if args.presolve:
        solver_params.update({'Presolve':args.presolve})
    if args.presparsify:
        solver_params.update({'PreSparsify':args.presparsify})
    if args.cuts:
        solver_params.update({'Cuts':args.cuts})
    if args.scale_flag:
        solver_params.update({'ScaleFlag':args.scale_flag})
    if args.feas_tol:
        solver_params.update({'FeasibilityTol':args.feas_tol})
    if args.predual:
        solver_params.update({'PreDual':args.predual})
    if args.degen_moves:
        solver_params.update({'DegenMoves':args.degen_moves})

    return solver_params

# Gurobi Functions
# --------------------------------------------------------------------------------


def directly_solve(m, path_list, solver_params):

    try:

        # Parameters
        # --------------------------------------------------------------------------------
        for param_key, param_val in solver_params.items():
            model.setParam(param_key, param_val)


        m.update()

        # Optimize model
        m.optimize()

        # Output
        # --------------------------------------------------------------------------------

        # for v in m.getVars():
        #     try:
        #         print(f"{v.VarName} {v.X:g}")
        #     except:
        #         pass

        max_throughput_varname = 'max_cload'
        max_cload_var = m.getVarByName(max_throughput_varname)
        print(f'{max_throughput_varname} : {max_cload_var.X}')

        print(f"\t(aka obj: {m.ObjVal:g} )")

        chosen_paths = [ [ None for _ in range(n_routers)] for __ in range(n_routers) ]

        for sr in range(n_routers):
            for dr in range(n_routers):

                if sr==dr:
                    chosen_paths[sr][dr] = [sr]
                    continue

                paths = path_list[sr][dr]

                for p, path in enumerate(paths):
                    path_chosen_varname = f'var_path_chosen_{sr}r_{dr}r_{p}p'
                    path_chosen_var = m.getVarByName(path_chosen_varname)
                    path_chosen_val = path_chosen_var.X

                    # print(f'\t{path_chosen_varname} : {path_chosen_val}')

                    if path_chosen_val > 0:
                        chosen_path = path_list[sr][dr][p]
                        # print(f'{sr}->{dr} : {chosen_path}')

                        if chosen_paths[sr][dr] is not None:
                            print(f'ERROR : Overwriting chosen_paths for {sr}->{dr}. Old {chosen_paths[sr][dr]}. New {chosen_path}. Exiting...')
                            quit()

                        chosen_paths[sr][dr] = chosen_path


        return chosen_paths

    # Error Handling
    # --------------------------------------------------------------------------------

    except AttributeError as e:
        print(f"Encountered an attribute error: {e}")

    except gp.GurobiError as e:
        print(f"Error code {e.errno}: {e}")

def find_mclb(path_dict, current_edge_state, adj_list,
                solver_params={}, write_presolved=False, foresight=False, my_tpuv4_symmetry=None, robust=False):#, destination_based=False):

    # try:
        n_routers = len(adj_list)

        # Create a new model
        model_base_name = "new_mclb"
        m = gp.Model(model_base_name)

        # Constants
        # --------------------------------------------------------------------------------
        # hardcode for tpuv4
        n_ports = 6
        print('='*80+'\n')
        print(f'Finding MCLB')
        print(f'\t# routers = {n_routers}')

        demand = 1.0
        capacity = 1.0

        flows = list(path_dict.keys())

        # construct path set given edge
        # edge_paths[i][j][n] is nth path signature that crosses edge (i,j)
        edge_paths = defaultdict(list) #  [[[] for _ in range(n_routers)] for __ in range(n_routers) ]

        # foresight_edge_paths[i][j][n] is nth canonical path signature that would induce load on edge (i,j)
        foresight_edge_paths =  defaultdict(list) #[[[] for _ in range(n_routers)] for __ in range(n_routers) ]
        for (sr,dr), paths in path_dict.items():

                # flow (sr,dr)
                for p, path in enumerate(paths):
                    path_signature = (sr,dr,p)

                    path_len = len(path)
                    for i in range(path_len-1):
                        edge_src = path[i]
                        edge_dest = path[i+1]

                        edge_paths[ (edge_src,edge_dest) ].append(path_signature)

                    if foresight:
                        # edge_paths[i][j] will only define the (sr,dr,p) in the canonical set
                        # instead need all non-canonical
                        for sr_outside in my_tpuv4_symmetry.get_all_noncanonical_equivalents(sr):
                            if sr_outside == sr: continue

                            # sr -> sr_outside
                            sr_to_sr_outside_tform = my_tpuv4_symmetry.calc_transform_delta(sr,sr_outside)

                            tform_path = []
                            for n in path:
                                n_outside = my_tpuv4_symmetry.apply_transformation(n, sr_to_sr_outside_tform)
                                tform_path.append(n_outside)
                            
                            # input(f"path {tform_path} is equivalent to path {path} for tform {sr_to_sr_outside_tform}")


                            tform_path_len = len(tform_path)
                            for i in range(tform_path_len-1):
                                edge_src = tform_path[i]
                                edge_dest = tform_path[i+1]

                                # print(f"edge src, dest = {edge_src}, {edge_dest}")

                                foresight_edge_paths[ (edge_src,edge_dest) ].append(path_signature)

                    # for i in range(n_routers):
                    #     for j in range(n_routers):
                    #         if (i==j):
                    #             continue
                    #         if len(edge_paths[i][j]) > 0:
                    #             print(f"edge_paths[{i}][{j}] = {edge_paths[i][j]}")
                    # for i in range(n_routers):
                    #     for j in range(n_routers):
                    #         if (i==j):
                    #             continue
                    #         if len(foresight_edge_paths[i][j]) > 0:
                    #             print(f"foresight_edge_paths[{i}][{j}] = {foresight_edge_paths[i][j]}")
                    # input("after one flow")

        mem_checkpoint("After setting up edge paths")

        # Variables
        # --------------------------------------------------------------------------------

        n_links = n_ports*n_routers
        n_total_flows = (n_routers**2) - n_routers
        min_cload = n_total_flows // n_links

        max_cload = m.addVar(lb=min_cload, ub=n_total_flows, vtype=GRB.INTEGER, name='max_cload')
        # max_cload = m.addVar(lb=min_cload, ub=n_flows, vtype=GRB.CONTINUOUS, name='max_cload')

        # this is a Gurobi var
        path_chosen = {}
        for (sr,dr), paths in path_dict.items():
            if len(paths) == 1:
                # should be pruned before this function
                input('unpruned?')
                continue

            path_chosen[(sr,dr)] = []
            for p,path in enumerate(paths):
                myvarname = f'var_path_chosen_{sr}r_{dr}r_{p}p'
                path_chosen[(sr,dr)].append( m.addVar(vtype=GRB.BINARY, name=myvarname) )

        print(f"\tCompleted variables")
        m.update()
        mem_checkpoint("After vars",model=m)

        # Constraints
        # --------------------------------------------------------------------------------

        # define cload
        # edge (i,j)
        n_possible = {}
        for i in range(n_routers):
            # for j in range(n_routers):
            #     if (i==j):
            #         continue
            for j in adj_list[i]:

                cload_expr = gp.LinExpr()

                for (sr,dr,p) in edge_paths[ (i, j) ]:                        
                    cload_expr += path_chosen[(sr,dr)][p]

                # known loads
                cload_expr += current_edge_state[(i,j)]

                # foresight
                if foresight:
                    for (sr,dr,p) in foresight_edge_paths[ (i, j) ]:
                        cload_expr += path_chosen[(sr,dr)][p]

                myconstrname = f'constr_cload_{i}r_{j}r'
                # m.addConstr(cload[i][j] >= cload_expr , myconstrname)
                m.addConstr(max_cload >= cload_expr , myconstrname)

        m.update()
        mem_checkpoint("After defining cload",model=m)

        # single path
        for (sr,dr), paths in path_dict.items():

            path_expr = gp.LinExpr()

            for path_chosen_var in path_chosen[(sr,dr)]:
                path_expr += path_chosen_var

            myconstrname = f'constr_path_chosen_{sr}r_{dr}r'
            m.addConstr( path_expr == 1, myconstrname)

        m.update()
        mem_checkpoint("After forcing single path",model=m)

        print(f"\tCompleted constraints")

        # Objectives
        # --------------------------------------------------------------------------------

        m.setObjective(max_cload, GRB.MINIMIZE)


        # Params and Model Output
        write_model = False
        if write_model:
            out_model_name = f'files/models/{model_base_name}.lp'
            m.write(out_model_name)

        m.update()
        mem_checkpoint("After objective",model=m)

        print(f"\tCompleted objective")

        m.update()


        # Parameters
        # --------------------------------------------------------------------------------

        # these are likely helpful
        
        # very important
        # m.setParam('CrossoverBasis',1) # is this helpful?
        m.setParam('DegenMoves',0)
        m.setParam('MIPFocus',1)
        m.setParam('Method',2)

        # likely important
        m.setParam('Cuts', 2)
        m.setParam('Threads', 32)
        m.setParam('Heuristics',0.5)
        m.setParam('Symmetry',2)

        # can be overriden
        for param_key, param_val in solver_params.items():
            # print(f'Setting {param_key} : {param_val}')
            m.setParam(param_key, param_val)
        
        # Solve
        # --------------------------------------------------------------------------------

        if write_presolved:
            m.presolve()


            out_presolved_model_name = f'{SCRATCH_DIR}/mclb_presolved_models/{model_base_name}_presolved.lp'

            try:
                m.write(out_presolved_model_name)
                print(f'Wrote to {out_presolved_model_name}')

            except Exception as e:
                print(f'{out_presolved_model_name} cannot be written')
                print(f'Error: {e}')



        # Optimize model
        m.optimize()

        # Output
        # --------------------------------------------------------------------------------


        # for v in m.getVars():
        #     try:
        #         print(f"{v.VarName} {v.X:g}")
        #     except:
        #         pass


        max_throughput_varname = 'max_cload'
        max_cload_var = m.getVarByName(max_throughput_varname)
        print(f'{max_throughput_varname} : {max_cload_var.X}')

        print(f"\t(aka obj: {m.ObjVal:g} )")

        chosen_paths = { f : None for f in flows}
        for (sr,dr) in flows:

            paths = path_dict[(sr,dr)]

            found_path = False

            for p, path in enumerate(paths):
                # path_chosen_varname = f'var_path_chosen_{sr}r_{dr}r_{p}p'
                # path_chosen_var = m.getVarByName(path_chosen_varname)
                path_chosen_var = path_chosen[(sr,dr)][p]
                path_chosen_val = path_chosen_var.X

                # print(f'\t{path_chosen_varname} : {path_chosen_val}')

                if path_chosen_val > 0:
                    chosen_path = path
                    # print(f'{sr}->{dr} : {chosen_path}')

                    if chosen_paths[(sr,dr)] is not None:
                        input('ERROR :: overwrite!')

                    chosen_paths[(sr,dr)] = chosen_path

                    found_path = True
            
            if not found_path:
                input(f"ERROR :: no path for {(sr,dr)}. Options : {paths} w/ values {[path_chosen[(sr,dr)][p].X for p in range(len(paths))  ]}")

        return chosen_paths

    # # Error Handling
    # # --------------------------------------------------------------------------------

    # except AttributeError:
    #     print("Encountered an attribute error")

    # except gp.GurobiError as e:
    #     print(f"Error code {e.errno}: {e}")

def verify_all_chosen(all_chosen_paths, n_routers):


    for s in range(n_routers):
        for d in range(s+1,n_routers):
            if (s,d) not in all_chosen_paths :
                print(f" {(s,d)} not in all_chosen_paths ")
                quit()
            if (d,s) not in all_chosen_paths :
                print(f" {(d,s)} not in all_chosen_paths ")
                quit()

            if len(all_chosen_paths[(s,d)]) == 0:
                print(f" {(s,d)} empty path")
                quit()
            if len(all_chosen_paths[(d,s)]) == 0:
                print(f" {(d,s)} empty path")
                quit()

# Main(s)
# --------------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(description='Route MCLB')

    # TODO make graph not required?
    parser.add_argument('--topology',type=str,help='.map file to evaluate',default='files/map_files/example_6r_25ll.map',required=True)
    parser.add_argument('--allpath_list','-apl',type=str,help='shortcut over path creation')

    parser.add_argument('--destination_based',action='store_true',help='all paths with same destination follow same next router')
    parser.add_argument('--robust',action='store_true',help='route backup paths for fault tolerance')
    parser.add_argument('--backup_paths_list',type=str,help='rallpaths for escapes')
    parser.add_argument('--any_link_failure',action='store_true',help='to only consider failure for all links, not just OCS links')

    parser.add_argument('--partition_size',type=int,help='...',default=INF)
    parser.add_argument('--partition_metric',type=str,choices=['flows','paths'],help='...',default='flows')

    parser.add_argument('--symmetric',action='store_true',help='graph is vertex symmetric. route canonical flows')
    parser.add_argument('--sym_type',type=str,help="graph symmetry type. default 'trans'",choices=["trans","refl-trans"], default="trans")
    parser.add_argument('--xyzc_dims',nargs='+',type=int,help='type without parenthesis and use spaces, no commas')
    parser.add_argument('--mc_dims',nargs='+',type=int,help='Minimum "cube" of symmetry\'s x, y, and z dimensions. If unspecified then assuming single (zeroth) cube (i.e., x,y,z==cube). Type without parenthesis and use spaces, no commas')

    parser.add_argument('--prob_load',action='store_true',help='assume 1/K load for K # paths per flow')
    parser.add_argument('--override_ingest_all',action='store_true',help='ingest all paths despite partitioning/symmetry scheme')

    parser.add_argument('--foresight',action='store_true',help='for symmetric, the choice of a canonical path applies load to all equivalent paths')


    parser.add_argument('--wfr',action='store_true',help='For PT/PDTT, iterate all possible OCS failures and generate chosen path set for each')
    parser.add_argument('--single_ocs_failure',action='store_true',help='For ASC, iterate all possible OCS failures and generate chosen path set for each')


    # direct Gurobi solver params
    parser.add_argument('--time_limit',type=int,help='time limit in minutes')
    parser.add_argument('--threads',type=int,help='# threads total')
    parser.add_argument('--concurrent_mip',type=int,help='# threads for concurrent')
    parser.add_argument('--heuristic_ratio',type=float,help='heuristic ratio [0,1]. 0=> none. 1=>all')
    parser.add_argument('--mip_focus',type=int,help='focus for MIP solver. 0=>balanced. 1=>feasible/first solution. 2=>optimality. 3=>bound')
    parser.add_argument('--symmetry_detection',type=int,help='control symmetry detection. -1 =>automatic. 0=>off. 1=>conservative. 2=>aggressive')
    parser.add_argument('--barrier_iter_limit',type=int,help='limit iterations of barrier algorithm')
    parser.add_argument('--iter_limit',type=int,help='limit iterations of something')
    parser.add_argument('--cut_passes',type=int,help='limit iterations of cut passes')
    parser.add_argument('--method',type=int,help='lp (root relax) method. -1=>auto. 0=>primal simplex. 1=>dual simplex. 2=>barrier. 3=>concurrent. 4=>deterministic concurrent. 5=>deterministic concurrent simplex')
    parser.add_argument('--node_method',type=int,help='-1=>auto. 0=>primal simplex. 1=>dual simplex. 2=>barrier')
    parser.add_argument('--crossover',type=int,help='')
    parser.add_argument('--crossover_basis',type=int,help='')
    parser.add_argument('--no_rel_heur_time',type=int,help='')
    parser.add_argument('--presolve',type=int,help='Presolve aggressiveness. -1=>auto. 0=>off. 1=>conservative. 2=>aggressive')
    parser.add_argument('--presparsify',type=int,help='')
    parser.add_argument('--cuts',type=int,help='')
    parser.add_argument('--scale_flag',type=int,help='')
    parser.add_argument('--feas_tol',type=float,help='')
    parser.add_argument('--degen_moves',type=int,help='')
    parser.add_argument('--write_presolved',action='store_true',help='presolve and write (presolved) model out as multiple/all formats')
    parser.add_argument('--read_presolved',type=str,help='read a presolved model of given name')
    parser.add_argument('--predual',type=int,help='')


    args = parser.parse_args()

    map_filename = args.topology

    write_presolved = args.write_presolved
    read_presolved = False
    presolved_model_name = None
    if args.read_presolved is not None:
        read_presolved = True
        presolved_model_name = args.read_presolved

    destination_based = args.destination_based
    if destination_based:
        print(f'UNIMPLEMENTED :: destination_based. Exiting...')
        quit()

    partition_size = args.partition_size
    if partition_size == -1:
        partition_size = INF
    partition_metric = args.partition_metric

    apl_name = args.allpath_list
    if apl_name is None:
        print(f'UNIMPLEMENTED :: calculating apl in this script. Exiting...')
        quit()

    robust = args.robust
    backup_paths_filepath = args.backup_paths_list
    if robust:
        assert(backup_paths_filepath)

    single_ocs_failure = args.wfr or args.single_ocs_failure

    # only if symmetric
    foresight = False
    symmetric = args.symmetric
    sym_type = args.sym_type
    if symmetric:
        xyzc_dims = tuple(args.xyzc_dims)
        assert(len(xyzc_dims) == 4)
        mc_dims = tuple(args.mc_dims)
        assert(len(mc_dims) == 3)

        # for now, force true
        foresight = True # args.foresight

        # cannot/shouldnt do both
        assert(partition_size == INF)

    prob_load = args.prob_load
    if prob_load:
        print(f"TODO: prob_load unimplemented")
        quit()
    override_ingest_all = args.override_ingest_all

    solver_params = setup_solver_params(args)

    base_file_name = apl_name.split('/')[-1].replace('.rallpaths','_new_mclb')

    if destination_based:
        base_file_name += '_destbased'

    if partition_size < INF:
        base_file_name += f'_{partition_size}ps_{partition_metric}pm'

    if symmetric:
        (mc_x, mc_y, mc_z) = mc_dims
        base_file_name += f"_{sym_type}sym_{mc_x}x{mc_y}x{mc_z}mc"

    if prob_load:
        base_file_name += f"_probld"
    if override_ingest_all:
        base_file_name += f"_allpathsinfo"

    # if source_start > 0 or source_end < INF or destination_start > 0 or destination_end < INF:
    #     base_file_name += f'_ss{source_start}'
    #     base_file_name += f'_se{source_end}'
    #     base_file_name += f'_ds{destination_start}'
    #     base_file_name += f'_de{destination_end}'

    # if source_partition_step < INF or destination_partition_step < INF:
    #     base_file_name += f'_sps{source_partition_step}_dps{destination_partition_step}'


    # ingest
    ################################################################################

    print(f'Running...')

    adj_mat, adj_list = ingest_map(map_filename)
    print(f'Ingested map')

    n_routers = len(adj_list)

    my_tpuv4_symmetry = None
    if symmetric:
        (x_dim, y_dim, z_dim, cube_dim) = xyzc_dims
        if (x_dim % 2 > 0 or y_dim % 2 > 0 or z_dim % 2 > 0) and sym_type != "trans":
            print(f"Dimensions of problem require 'trans' symmetry. Exiting...")
            quit()
        my_tpuv4_symmetry = TPUv4_Symmetry(xyzc_dims, mc_dims=mc_dims, sym_type=sym_type)
        my_tpuv4_symmetry.verify_symmetry_for_topology(adj_mat, verify_dist=True)

    adj_mat = None

    mem_checkpoint("Aftering ingesting map and setting up symmetry")


    # solve
    ################################################################################


    # prioritize
    # TODO
    # sorted_keys_list = sort_keys(path_list)

    sorted_keys_list = [(s,d) for s in range(n_routers) for d in range(n_routers) if s!=d]


    # loop over partitions

    if partition_metric != 'flows':
        print(f'UNIMPLEMENTED')

    src_set = set(range(n_routers))
    if symmetric:
        src_set = set(my_tpuv4_symmetry.get_canonical_nodes())

    # print(f"src_st = {src_set}")
    # quit()

    total_n_flows = len(sorted_keys_list)

    all_chosen_paths = {}

    if partition_size < INF:
        all_chosen_paths = route_partitioned(apl_name, adj_list, lb_flow, ub_flow, src_set, partition_size, total_n_flows, symmetric=symmetric, my_tpuv4_symmetry=my_tpuv4_symmetry, override_ingest_all=override_ingest_all, robust=robust, solver_params=solver_params, write_presolved=write_presolved, foresight=foresight)
    elif single_ocs_failure:
        print("TODO")


        # output from within above function
        quit()
    else:
        all_chosen_paths = route(apl_name, adj_list, src_set, symmetric=symmetric, my_tpuv4_symmetry=my_tpuv4_symmetry, override_ingest_all=override_ingest_all, robust=robust, solver_params=solver_params, write_presolved=write_presolved, foresight=foresight)




    # output
    ###########################################################################

    # sort and create final output
    flat_route_pathlist = create_flat_pathlist(all_chosen_paths, adj_list)

    output_pathlist( flat_route_pathlist, base_file_name, 'topologies_and_routing/routepath_lists')

    print_max_cload(flat_route_pathlist)

def find_wild_hop_alternatives(paths, all_path_dict, r_map, xyzc_dims):

    n_routers = len(r_map)

    alternatives = []

    for path in paths:
        s = path[0]
        d = path[-1]

        neighbors = [ n for n in range(n_routers) if r_map[s][n] > 0]
        for n in neighbors:
            neighbor_hop_type = calc_opt_conn_type(s, n, xyzc_dims)[0]
            
            alternative_options = all_path_dict

            FLAG

def route_wfr(apl_name, adj_list, src_set, xyzc_dims, symmetric=False, my_tpuv4_symmetry=None, override_ingest_all=False, robust=False, solver_params={}, write_presolved=False, foresight=False):

    pass
    # n_routers = len(adj_list)

    # input("TODO implement adj_list based approach")

    # ocs_id_to_r_ids = defaultdict(list)

    # for i in range(n_routers):
    #     for j in range(n_routers):
    #         if i==j:
    #             continue
    #         ocs = ocs_id(i,j, xyzc_dims)
    #         ocs_id_to_r_ids[ocs].append( (i,j) )

    # all_chosen_paths = {}

    # (lb_flow, ub_flow) = (None, None)

    # all_path_dict, current_edge_state, known_paths = ingest_and_setup_input_paths(apl_name, n_routers, lb_flow, ub_flow, src_set, symmetric=symmetric, my_tpuv4_symmetry=my_tpuv4_symmetry, override_ingest_all=override_ingest_all, robust=robust)

    # if symmetric:
    #     print(f"Working on canonical ({len(src_set)}) {src_set}")


    # # no failure
    # chosen_paths = find_mclb(all_path_dict, current_edge_state, adj_list, solver_params=solver_params,write_presolved=write_presolved, foresight=foresight, my_tpuv4_symmetry=my_tpuv4_symmetry, robust=robust)

    # print(f"Completed solve")

    # all_chosen_paths.update(chosen_paths)

    # for path in chosen_paths.values():
    #     n_hops = len(path) - 1
    #     for n in range(n_hops):
    #         i = path[n]
    #         j = path[n+1]
    #         current_edge_state[(i,j)] += 1

    # # finalize chosen paths with knowns and symmetries
    # ###########################################################################


    # # apply knowns
    # for (sr,dr), path in known_paths.items():
    #     all_chosen_paths[(sr,dr)] = path

    # if symmetric:
    #     all_chosen_paths = handle_symmetry(all_chosen_paths, adj_list, my_tpuv4_symmetry, n_routers)

    # # sort and create final output
    # flat_route_pathlist = create_flat_pathlist(all_chosen_paths, n_routers)

    # output_pathlist( flat_route_pathlist, base_file_name, 'topologies_and_routing/routepath_lists')

    # print(f"No failures")
    # print_max_cload(flat_route_pathlist)

    # # failure
    # for ocs in range(48):

    #     # all_path_dict has at least 2 paths per flow
    #     this_path_dict = deepcopy(all_path_dict)

    #     related_links = ocs_id_to_r_ids[ocs]
    #     for s in range(n_routers):
    #         for d in range(n_routers):
    #             if s==d:
    #                 continue
    #             paths = deepcopy(this_path_dict[(s,d)])

    #             path_idxs_to_remove = []
    #             paths_to_remove = []
    #             for p, path in enumerate(paths):
    #                 n_hops = len(path) - 1
    #                 for i in range(n_hops):
    #                     u = path[i]
    #                     v = path[i+1]
    #                     if (u,v) in related_links:
    #                         paths_to_remove.append(path)
    #             # path_idxs_to_remove.reverse()
    #             for path in paths_to_remove:
    #                 paths.remove(path)
                
    #             # must find a wild hop alternative
    #             if len(paths) == 0:
    #                 paths = find_wild_hop_alternatives(this_path_dict[(s,d)], all_path_dict, r_map, xyzc_dims)
                

    #     # known paths is 1 path per flow
    #     this_known_paths = deepcopy(known_paths)


def route(apl_name, adj_list, src_set, symmetric=False, my_tpuv4_symmetry=None, override_ingest_all=False, robust=False, solver_params={}, write_presolved=False, foresight=False):

    n_routers = len(adj_list)

    (lb_flow, ub_flow) = (None, None)

    this_path_dict, current_edge_state, known_paths = ingest_and_setup_input_paths(apl_name, n_routers, lb_flow, ub_flow, src_set, symmetric=symmetric, my_tpuv4_symmetry=my_tpuv4_symmetry, override_ingest_all=override_ingest_all, robust=robust)

    # for dr in range(n_routers):
    #     try:
    #         path = known_paths[(4,dr)]
    #         print(f"known {path}")
    #     except:
    #         print(f"no known for {(4,dr)}")

    # quit()

    mem_checkpoint("After ingesting paths")

    if symmetric:
        print(f"Working on canonical ({len(src_set)}) {src_set}")


    all_chosen_paths = find_mclb(this_path_dict, current_edge_state, adj_list, solver_params=solver_params,write_presolved=write_presolved, foresight=foresight, my_tpuv4_symmetry=my_tpuv4_symmetry, robust=robust)

    print(f"Completed solve")

    # all_chosen_paths.update(chosen_paths)
    # for path in chosen_paths.values():
    #     n_hops = len(path) - 1
    #     for n in range(n_hops):
    #         i = path[n]
    #         j = path[n+1]
    #         current_edge_state[(i,j)] += 1

    # finalize chosen paths with knowns and symmetries
    ###########################################################################


    # apply knowns
    print(f"Applying knowns")
    for (sr,dr), path in known_paths.items():
        if (sr,dr) in all_chosen_paths:
            input(f"overwriting chosen {all_chosen_paths[(sr,dr)]} w/ known {path}")
            quit()

        if len(path) == 0:
            input(f"Empty path for {(sr,dr)}")
            quit()
        all_chosen_paths[(sr,dr)] = path

    if symmetric:
        print(f"Applying symmetry")
        all_chosen_paths = handle_symmetry(all_chosen_paths, adj_list, my_tpuv4_symmetry, n_routers)

    verify_all_chosen(all_chosen_paths, n_routers)

    return all_chosen_paths


def route_partitioned(apl_name, adj_list, lb_flow, ub_flow, src_set, partition_size, total_n_flows, symmetric=False, my_tpuv4_symmetry=None, override_ingest_all=False, robust=False, solver_params={}, write_presolved=False, foresight=False):
    pass

    # n_routers = len(adj_list)


    # all_chosen_paths = {}

    # key_idx = 0
    # while key_idx < total_n_flows:

    #     lb = key_idx
    #     ub = min(key_idx+partition_size, n_routers**2)

    #     lb_flow = ((lb//n_routers), (lb%n_routers))
    #     ub_flow = ((ub//n_routers), (ub%n_routers))

    #     this_path_dict, current_edge_state, known_paths = ingest_and_setup_input_paths(apl_name, n_routers, lb_flow, ub_flow, src_set, symmetric=symmetric, my_tpuv4_symmetry=my_tpuv4_symmetry, override_ingest_all=override_ingest_all, robust=robust)

    #     print(f'Working on flow partition [{lb}:{ub}) out of {total_n_flows} (ie {lb//partition_size}/{total_n_flows//partition_size} iterations)')
    #     if symmetric:
    #         print(f"Working on canonical ({len(src_set)}) {src_set}")

    #     # input(f'this_path_dict ({len(this_path_dict)}) = {this_path_dict}')

    #     chosen_paths = find_mclb(this_path_dict, current_edge_state, adj_list, solver_params=solver_params,write_presolved=write_presolved, foresight=foresight, my_tpuv4_symmetry=my_tpuv4_symmetry, robust=robust)

    #     print(f"Completed solve")

    #     all_chosen_paths.update(chosen_paths)

    #     for path in chosen_paths.values():
    #         n_hops = len(path) - 1
    #         for n in range(n_hops):
    #             i = path[n]
    #             j = path[n+1]
    #             current_edge_state[(i,j)] += 1

    #     key_idx = ub

    # # finalize chosen paths with knowns and symmetries
    # ###########################################################################


    # # apply knowns
    # for (sr,dr), path in known_paths.items():
    #     all_chosen_paths[(sr,dr)] = path

    # if symmetric:
    #     all_chosen_paths = handle_symmetry(all_chosen_paths, adj_list, my_tpuv4_symmetry, n_routers)

    # return all_chosen_paths

def ingest_and_setup_input_paths(apl_name, n_routers, lb_flow, ub_flow, src_set, symmetric=False, my_tpuv4_symmetry=None, override_ingest_all=False, robust=False, prob_load=False, foresight=False):


    if symmetric:
        assert(my_tpuv4_symmetry)
        foresight = True

    seen_flows = set()
    known_paths = {}
    current_edge_state = defaultdict(float) #{(i,j) : 0 for i in range(n_routers) for j in range(n_routers)}


    this_path_dict = ingest_path_list(apl_name, n_routers, lb_flow=lb_flow, ub_flow=ub_flow, src_set=src_set,override_ingest_all=override_ingest_all )

    if robust:
        backup_path_dict = ingest_path_list(backup_paths_filepath, n_routers, lb_flow=lb_flow, ub_flow=ub_flow, src_set=src_set, override_ingest_all=override_ingest_all)
        for flow, paths in backup_path_dict.items():
            for path in paths:
                if path not in this_path_dict[flow]:
                    this_path_dict[flow].append(path)

    print(f"working on partition lb_flow : ub_flow = {lb_flow} : {ub_flow} &/ src_set {src_set}")
    print(f'Ingested {len(this_path_dict.keys())} flows')
    print(f"Ingested { sum( [ len(v) for v in this_path_dict.values() ] )} paths ")

    mem_checkpoint("Just after ingesting. Before seen and known flows")

    # assert all flows have paths
    seen_flows.update(this_path_dict.keys())
    for i in src_set:
        for j in range(n_routers):
            if i==j:
                continue
            if (lb_flow is not None) and i < lb_flow[0]:
                continue
            elif (lb_flow is not None) and i == lb_flow[0] and j < lb_flow[1]:
                continue
            if (ub_flow is not None) and i > ub_flow[0]:
                continue
            elif (ub_flow is not None) and i == ub_flow[0] and j >= ub_flow[1]:
                continue

            if (i,j) not in seen_flows:
                print(f"ERROR: {i}->{j} is NOT in seen_flows")
            assert( (i,j) in seen_flows )
    # clear memory
    seen_flows = None

    flows_to_remove = set()
    for flow, paths in this_path_dict.items():

        n_paths = len(paths)

        (s,d) = flow
        if s not in src_set and not override_ingest_all:
            continue

        # isolate easy knowns
        if n_paths == 1:
            path = paths[0]

            known_paths[flow] = path
            flows_to_remove.add(flow)

        elif prob_load:

            for path in paths:
                n_hops = len(path) - 1
                for n in range(n_hops):
                    i = path[n]
                    j = path[n+1]
                    current_edge_state[(i,j)] += (1/n_paths)


    # handle knowns
    for flow, path in known_paths.items():

        # basic
        n_hops = len(path) - 1

        if n_hops == 0:
            continue

        for n in range(n_hops):
            i = path[n]
            j = path[n+1]
            current_edge_state[(i,j)] += 1


        if symmetric and foresight:
            (sr,dr) = flow
            # edge_paths[i][j] will only define the (sr,dr,p) in the canonical set
            # instead need all non-canonical
            for sr_outside in my_tpuv4_symmetry.get_all_noncanonical_equivalents(sr):
                if sr_outside == sr: continue

                sr_to_sr_outside_tform = my_tpuv4_symmetry.calc_transform_delta(sr,sr_outside)

                tform_path = []
                for n in path:
                    n_outside = my_tpuv4_symmetry.apply_transformation(n, sr_to_sr_outside_tform)
                    tform_path.append(n_outside)
                
                # input(f"KNOWN path {tform_path} is equivalent to path {path} for tform {sr_to_sr_outside_tform}")

                tform_path_len = len(tform_path)
                for n in range(tform_path_len-1):
                    i = tform_path[n]
                    j = tform_path[n+1]

                    current_edge_state[(i,j)] += 1


    # remove all non canonical. Do here (after knowing single option paths) so that any info can be extracted
    if symmetric:
        canon_nodes = my_tpuv4_symmetry.get_canonical_nodes()
        src_set = canon_nodes
        for n in range(n_routers):
            if n not in canon_nodes:
                flows_to_remove.update([(n, d) for d in range(n_routers)])
                # flows_to_remove.update([(n, d) for d in range(n_routers)])


    for flow in flows_to_remove:
        try:
            del this_path_dict[flow]
        except:
            pass

    return this_path_dict, current_edge_state, known_paths

def create_flat_pathlist( all_chosen_paths, adj_list):

    n_routers = len(adj_list)

    print(f"Creating flat pathlist for # routers {n_routers}")

    flat_route_pathlist = []

    seen_flows = set()

    # for loop to get sr==dr too
    for sr in range(n_routers):
        for dr in range(n_routers):
            if sr==dr:
                flat_route_pathlist.append([sr])
                continue

            path = all_chosen_paths[(sr,dr)]
            n_hops = len(path)-1
            for n in range(n_hops):
                i = path[n]
                j = path[n+1]
                assert(j in adj_list[i])

            if len(path) == 0:
                input(f"Empty path {path}")

            flat_route_pathlist.append(path)

            seen_flows.add( (sr,dr) )

    for s in range(n_routers):
        for d in range(s+1, n_routers):
            if (s,d) not in seen_flows :
                print(f" {(s,d)} not in seen_flows ")
                quit()
            if (d,s) not in seen_flows :
                print(f" {(d,s)} not in seen_flows ")
                quit()

    print(f"Created flat pathlist")

    return flat_route_pathlist

def handle_symmetry(all_chosen_paths, adj_list, my_tpuv4_symmetry, n_routers):

    canon_nodes = my_tpuv4_symmetry.get_canonical_nodes()
    for sr_canon in set(canon_nodes):
        for dr_canon in range(n_routers):

            # canonical (sr_canon,dr_canon)
            base_path = all_chosen_paths[ (sr_canon,dr_canon) ]

            all_sr_noncanons = my_tpuv4_symmetry.get_all_noncanonical_equivalents(sr_canon)

            for sr_noncanon in all_sr_noncanons:
                if sr_noncanon == sr_canon:
                    continue

                # sr_canon -> sr_noncanon
                sr_canon_to_noncanon_tform = my_tpuv4_symmetry.calc_transform_delta(sr_canon, sr_noncanon)

                dr_noncanon = my_tpuv4_symmetry.apply_transformation(dr_canon, sr_canon_to_noncanon_tform)


                new_path = []
                for n_canon in base_path:

                    n_noncanon = my_tpuv4_symmetry.apply_transformation(n_canon, sr_canon_to_noncanon_tform)

                    new_path.append(n_noncanon)

                # validation
                assert new_path[0] == sr_noncanon
                assert new_path[-1] == dr_noncanon

                for i in range(len(new_path) - 1):
                    u = new_path[i]
                    v = new_path[i + 1]
                    assert v in adj_list[u], f"Invalid hop {u}->{v} for path {new_path} derived from base path {base_path}"


                if (sr_noncanon, dr_noncanon) in all_chosen_paths:
                    input(
                        f"Overwriting {(sr_noncanon, dr_noncanon)}\n"
                        f"  old={all_chosen_paths[(sr_noncanon, dr_noncanon)]}\n"
                        f"  new={new_path}\n"
                    )


                all_chosen_paths[ (sr_noncanon, dr_noncanon) ] = new_path


    print("Completed symmetry")
    return all_chosen_paths


def print_max_cload(flat_route_pathlist):

    print(f"Finding max channel load")

    max_cload = 0
    cload_dict = defaultdict(int)

    maximally_loaded_edges = []

    for path in flat_route_pathlist:
        n_hops = len(path)-1
        for i in range(n_hops):
            a = path[i]
            b = path [i+1]
            edge = (a,b)
    
            cload_dict[ edge ] += 1

            if cload_dict[ edge ] > max_cload:
                max_cload = cload_dict[ edge ]
                maximally_loaded_edges = [ edge ]
            elif cload_dict[ edge ] == max_cload:
                maximally_loaded_edges.append(edge)

    n_to_print = min(5,len(maximally_loaded_edges))
    print(f"After everything, throughput is {1/max_cload} from max cload {max_cload}")
    print(f"\tFrom edges {maximally_loaded_edges[:n_to_print]} (truncated)")


# Script Stuff
# --------------------------------------------------------------------------------

if __name__ == '__main__':

    main()
