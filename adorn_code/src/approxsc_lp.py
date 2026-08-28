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

For ...


Based on https://dl.acm.org/doi/pdf/10.1145/77600.77620

'''

# Gurobi libs
import gurobipy as gp
from gurobipy import GRB

# regular libs
import argparse
from collections import deque
import time
import os
import sys

# locals
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "..", "python_scripts"))

from tpuv4_symmetry import TPUv4_Symmetry


# constants
VERBOSE = False # for all
ASSERT_BINARY_MAP = True
INF = 999 # for FW

# Regular Functions
# --------------------------------------------------------------------------------

def get_shape(nested_list):
    if isinstance(nested_list, list):
        return [len(nested_list)] + get_shape(nested_list[0])
    else:
        return []

def ingest_map(path_name):
    file_name = path_name.split('/')[-1]

    if True:
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
                if conn > 0.1:
                    conn = 1
                # assert(conn == 1 or conn == 0)

    if VERBOSE:
        print(f'read {this_map}')

    return this_map

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

def r_to_rel_xyz(r, xyzc_dims):
    (x_dim, y_dim, z_dim, cube_dim) = xyzc_dims

    r_x,r_y,r_z = r_to_xyz(r, xyzc_dims)

    rel_r_x = r_x % cube_dim
    rel_r_y = r_y % cube_dim
    rel_r_z = r_z % cube_dim

    return rel_r_x, rel_r_y, rel_r_z

def r_to_rel_r(r,xyzc_dims):
    (rel_x, rel_y, rel_z) = r_to_rel_xyz(r,xyzc_dims)
    return xyz_to_r(rel_x, rel_y, rel_z, xyzc_dims)


def iter_rel_xyz_across_cubes(rel_x,rel_y,rel_z,xyzc_dims):
    (x_dim, y_dim, z_dim, cube_dim) = xyzc_dims

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
                targ = xyz_to_r(xprime, yprime, zprime, xyzc_dims)
                targs.append(targ)
    
    return targs


def calc_conn_type_no_direction(s, d, xyzc_dims):
    (x_dim, y_dim, z_dim, cube_dim) = xyzc_dims

    rel_s_x, rel_s_y, rel_s_z = r_to_rel_xyz(s, xyzc_dims)
    rel_d_x, rel_d_y, rel_d_z = r_to_rel_xyz(d, xyzc_dims)

    conn_type = ''

    # x
    if rel_s_x != rel_d_x:
        conn_type += 'x'

    # y
    if rel_s_y != rel_d_y:
        conn_type += 'y'

    # z
    if rel_s_z != rel_d_z:
        conn_type += 'z'

    return conn_type



def calc_coord_diff( s,d, xyzc_dims, relative=False):

    if not relative:
        s_x, s_y, s_z = r_to_xyz(s, xyzc_dims)
        d_x, d_y, d_z = r_to_xyz(d, xyzc_dims)
    else:
        s_x, s_y, s_z = r_to_rel_xyz(s, xyzc_dims)
        d_x, d_y, d_z = r_to_rel_xyz(d, xyzc_dims)

    coord_diffs = []

    # x
    if s_x != d_x:
        coord_diffs.append('x')

    # y
    if s_y != d_y:
        coord_diffs.append('y')

    # z
    if s_z != d_z:
        coord_diffs.append('z')

    return coord_diffs

def ijk_dor_relevant(i,j,k, xyzc_dims):

    is_electrical = True# if k in electrical_conns_adj_list[i] else False

    flow_types = calc_coord_diff(i,j,xyzc_dims)

    # this undersells the performance (obj < routing bound)
    if True: #is_electrical and False:
        if 'x' in flow_types:
            important_types = ['x']
        elif 'y' in flow_types:
            important_types = ['y']
        elif 'z' in flow_types:
            important_types = ['z']
    else:
        if 'x' in flow_types:
            important_types = ['x']
        elif 'y' in flow_types:
            important_types = ['x','y']
        elif 'z' in flow_types:
            important_types = ['x','y','z']

    allowed = False
    ik_dim = calc_conn_type_no_direction(i,k, xyzc_dims)
    if ik_dim in important_types:
        allowed = True

    if VERBOSE:
        print(f"Because coord diff {i}->{j} => {r_to_xyz(i,xyzc_dims)}->{r_to_xyz(j,xyzc_dims)} = {flow_types} then important {important_types} and so 'ik' {i}->{k} of type {ik_dim} allowed? {allowed}")
        input('cont?')

    return allowed

def ijk_dor_relevant_v1(i,j,k, xyzc_dims):

    flow_types = calc_coord_diff(i,j,xyzc_dims)

    if False: #is_electrical and False:
        if 'x' in flow_types:
            important_types = ['x']
        elif 'y' in flow_types:
            important_types = ['y']
        elif 'z' in flow_types:
            important_types = ['z']
    else:
        if 'x' in flow_types:
            important_types = ['x']
        elif 'y' in flow_types:
            important_types = ['x','y']
        elif 'z' in flow_types:
            important_types = ['x','y','z']

    allowed = False
    ik_dim = calc_conn_type_no_direction(i,k, xyzc_dims)
    if ik_dim in important_types:
        allowed = True

    if VERBOSE:
        print(f"Because coord diff {i}->{j} => {r_to_xyz(i,xyzc_dims)}->{r_to_xyz(j,xyzc_dims)} = {flow_types} then important {important_types} and so 'ik' {i}->{k} of type {ik_dim} allowed? {allowed}")
        input('cont?')

    return allowed

def ijk_dor_relevant_v2(i,j,k, xyzc_dims):

    dor_order = ['x','y','z']
    dor_idx = {v: i for i, v in enumerate(dor_order)}

    ik_flow_types = calc_coord_diff(i,k,xyzc_dims)
    ik_last = ik_flow_types[-1]
    kj_flow_types = calc_coord_diff(k,j,xyzc_dims)
    kj_first = kj_flow_types[0]

    allowed = True if dor_idx[ik_last] <= dor_idx[kj_first] else False 

    if VERBOSE:
        print(f"i->k : {i}->{k} by {ik_flow_types}")
        print(f"k->j : {k}->{j} by {kj_flow_types}")
        print(f'=> allowed? {allowed}')    

        # input('cont?')

    # TRY A) OPT CONNS HAVE DIFFERENT/MODIFIED/SPECIAL FLOW_TYPES AND/OR B) INCLUDE IJ_FLOW_TYPE IN CALCULATION

    return allowed

def ijk_dor_relevant_v3(i,j,k, xyzc_dims):

    # VERBOSE = True

    dor_order = ["free_first",'x','y','z',"free_last"]
    dor_idx = {v: i for i, v in enumerate(dor_order)}

    ik_flow_types = calc_coord_diff(i,k,xyzc_dims, relative=True)
    kj_flow_types = calc_coord_diff(k,j,xyzc_dims, relative=True)

    if VERBOSE:
        print(f"i->k : {i}->{k} @ {r_to_xyz(i,xyzc_dims)} -> {r_to_xyz(k,xyzc_dims)} by {ik_flow_types}")
        print(f"k->j : {k}->{j} @ {r_to_xyz(k,xyzc_dims)} -> {r_to_xyz(j,xyzc_dims)} by {kj_flow_types}")

    ik_last = "free_first" if len(ik_flow_types)==0 else ik_flow_types[-1]
    kj_first = "free_last" if len(kj_flow_types)==0 else kj_flow_types[0]

    allowed = True if dor_idx[ik_last] <= dor_idx[kj_first] else False 

    if VERBOSE:
        print(f"dor_idx[ik_last] = dor_idx[{ik_last}] = {dor_idx[ik_last]}")
        print(f"dor_idx[kj_first] = dor_idx[{kj_first}] = {dor_idx[kj_first]}")
        print(f'=> allowed? {allowed}')    

        input('cont?')

    # TRY A) OPT CONNS HAVE DIFFERENT/MODIFIED/SPECIAL FLOW_TYPES AND/OR B) INCLUDE IJ_FLOW_TYPE IN CALCULATION

    return allowed



class UpperTriMatrix:
    def __init__(self, n):
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



# Gurobi Functions
# --------------------------------------------------------------------------------

# Main(s)
# --------------------------------------------------------------------------------

def main():

    global VERBOSE

    parser = argparse.ArgumentParser(description='LR approximate sparsest cut')
    parser.add_argument('--topology',type=str,help='.map file to evaluate',required=True)
    parser.add_argument('--one_leg',action='store_true',help='whether to only consider tri inequalities where (i,k) self.verbosein E')
    parser.add_argument('--dor_heur',action='store_true',help='whether to only consider tri inequalities where (i,j,k) relevant to DOR')
    parser.add_argument("--xyzc_dims",nargs="+",type=int,help="type without parenthesis and use spaces, no commas")
    parser.add_argument("--verbose","-v",action="store_true",help="extensive prints")

    args = parser.parse_args()

    map_filename = args.topology
    one_leg = args.one_leg
    dor_heur = args.dor_heur

    xyzc_dims = None
    if dor_heur:
        xyzc_dims = tuple(args.xyzc_dims)

    if args.verbose:
        VERBOSE = True

    r_map = ingest_map(map_filename)

    find_sc(r_map, one_leg=one_leg, dor_heur=dor_heur, xyzc_dims=xyzc_dims)
 

def find_sc(r_map, one_leg=False, dor_heur=False, xyzc_dims=None):



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

        print('Created variables')

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

        print('Completed unity constr')

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

        print(f'n_iters={n_iters}')

        print('Completed tri ineq constrs')


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

        print('Completed objective')


        # Params and Model Output
        # --------------------------------------------------------------------------------

        # Params
        m.setParam('Method', 2)
        m.setParam('Crossover', 0)
        m.setParam('BarOrder', 0)

        # Model Output

        m.write("model.lp")

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

        # if True:#VERBOSE:
        #     for v in m.getVars():
        #             print(f"{v.VarName} {v.X:g}")



        # print("dists used")
        # for sr in range(n_routers):
        #     print(f'{sr:02} : [',end='')
        #     for dr in range(sr+1, n_routers):
        #         myvarname = f'var_dist_{sr}r_{dr}r'
        #         v = m.getVarByName(myvarname)
        #         val = v.X
        #         if val > 0:
        #             print(f'{round(val,4):05}',end=', ')
        #         else:
        #             print(f'  -  ',end=', ')
        #     print(']')

        in_set = []
        for i in range(1,n_routers):

            myvarname = f'var_dist_0r_{i}r'
            v = m.getVarByName(myvarname)
            val = v.X
            if val == 0:
                in_set.append(i)

        obj_val = float(m.ObjVal)

        print(f"obj: {obj_val:g} ")
        print(f'sparsest set ({len(in_set)}): {in_set}')

        print(f'solver time : {solve_end_t - solve_start_t}')


        return in_set, obj_val

    # Error Handling
    # --------------------------------------------------------------------------------

    except AttributeError as e:
        print(f"Encountered an attribute error : {e}")

    except gp.GurobiError as e:
        print(f"Error code {e.errno}: {e}")

# Script Stuff
# --------------------------------------------------------------------------------

if __name__ == '__main__':

    main()