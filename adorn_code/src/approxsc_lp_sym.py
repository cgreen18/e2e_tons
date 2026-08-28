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

TODO description



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



class UpperTriMatrix:
    def __init__(self, n):
        self.n = n
        self.size = n * (n - 1) // 2
        self.data = [None] * self.size  # Python objects

    def _row_start(self, i):
        return i * (self.n - 1) - (i * (i - 1)) // 2

    def _index(self, i, j):
        if i == j:
            pass # returns 0
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
        if i==j:
            return 0
        return self.data[self._index(i, j)]

    def __setitem__(self, key, value):
        i, j = key
        self.data[self._index(i, j)] = value



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


# Gurobi Functions
# --------------------------------------------------------------------------------

 
def find_sc(r_map, xyzc_dims, mc_dims, sym_type="refl-trans"):

    # Constants and/or non-Gurobi
    # --------------------------------------------------------------------------------
    model_base_name = "approxsc_lp"

    n_routers = len(r_map)
    (x_dim, y_dim, z_dim, cube_dim) = xyzc_dims
    n_cubes = (x_dim//cube_dim)*(y_dim//cube_dim)*(z_dim//cube_dim)

    demand = 1.0
    capacity = 1.0

    # convenience
    # list of edges (as tuples)
    edge_list = []
    edge_set = set()
    for i in range(n_routers):
        for j in range(n_routers):
            if i >= j:
                continue
            if r_map[i][j] > 0:
                edge_list.append((i,j))
                edge_set.add((i,j))


    if x_dim % 2 > 0 or y_dim % 2 > 0 or z_dim % 2 > 0:
        sym_type = "trans"
    my_tpuv4_symmetry = TPUv4_Symmetry(xyzc_dims, mc_dims=mc_dims, sym_type=sym_type)
    # precheck
    my_tpuv4_symmetry.verify_symmetry_for_topology(r_map, verify_dist=True)

    routers_of_canonical_cube = my_tpuv4_symmetry.get_canonical_nodes()
    global_to_canonical_ratio = n_routers // len(routers_of_canonical_cube)
    canonical_to_global_ratio = len(routers_of_canonical_cube) / n_routers

    print(f"Finding approximate sparsest cut and leveraging vertex/edge symmetry")
    print(f"    # routers               : {n_routers}")
    print(f"    Global dimensions       : {xyzc_dims}")
    print(f"    Mega cube dimensions    : {mc_dims}")
    print(f"    |mega cube| / |global|  : {1.0/canonical_to_global_ratio}")
    # print(f"routers_of_canonical_cube = {routers_of_canonical_cube}")

    # print(f"my_tpuv4_symmetry.get_canonical_equivalent_edge(176,130)={my_tpuv4_symmetry.get_canonical_equivalent_edge(176,130)}")
    # print(f"r_map[26][42] = {r_map[26][42]}")
    # quit()

    try:
    # if True:
        # Create a new model

        m = gp.Model(model_base_name)


        # Variables
        # --------------------------------------------------------------------------------

        dist = UpperTriMatrix(n_routers)

        for i in routers_of_canonical_cube:
        # for i in range(n_routers):
            for j in range(n_routers):

                if i==j:
                    continue
                if dist[i,j] is not None:
                    continue

                myvarname = f'var_dist_{i}r_{j}r'

                # print(f'Created dist[{i}][{j}]')

                # if using abs
                dist[i,j] = m.addVar(lb=0, ub=1.0, vtype=GRB.CONTINUOUS, name=myvarname)

        m.update()
        print(f'Created {m.numVars} variables')

        # Constraints
        # --------------------------------------------------------------------------------

        # unity_expr = gp.LinExpr()
        # for i in range(n_routers):
        #     for j in range(i+1,n_routers):
        #         i_prime = r_to_rel_r(i,xyzc_dims)
        #         j_prime = translation(i,j,xyzc_dims)

        #         unity_expr += dist[i_prime,j_prime]
        myconstrname = 'constr_dist_sum'
        # m.addConstr(unity_expr >= 1)

        # unity_expr = gp.LinExpr()
        # for i in routers_of_canonical_cube:
        #     for j in range(n_routers):
        #         if i==j:
        #             continue
        #         print(f"unity : dist[{i}][{j}]")
        #         unity_expr += global_to_canonical_ratio*dist[i,j]
        # m.addConstr(unity_expr >= 1)


        m.addConstr(gp.quicksum( [global_to_canonical_ratio*dist[i,j]  for i in routers_of_canonical_cube for j in range(n_routers)  if i!=j] ) >= 1, name=myconstrname)


        # m.addConstr(gp.quicksum( [dist[i,j]  for i in routers_of_canonical_cube for j in range(n_routers)  if i!=j] ) >= 1, name=myconstrname)
        # m.addConstr(gp.quicksum( [dist[i,j]  for i in range(n_routers) for j in range(i+1,n_routers,1) ] ) >= 1, name=myconstrname)

        print('Completed unity constr')

        n_iters = 0

        # for i in range(n_routers):
        #     for j in range(1+1,n_routers):
        for i in routers_of_canonical_cube:
            for j in range(n_routers):
                if i==j:
                    continue
                for k in range(n_routers):

                    if k==i or k==j:
                        continue

                    if r_map[i][k] == 0:
                        continue

                    i_prime, i_tform = my_tpuv4_symmetry.get_canonical_equivalent(i)
                    k_prime_a = my_tpuv4_symmetry.apply_transformation(k,i_tform)
                    j_prime_a = my_tpuv4_symmetry.apply_transformation(j,i_tform)
                    (k_prime_b, j_prime_b) = my_tpuv4_symmetry.get_canonical_equivalent_edge(k_prime_a,j_prime_a)


                    # if sym_type == 'refl-trans':
                    #     i_prime, i_tform = my_tpuv4_symmetry.get_canonical_equivalent(i)
                    #     k_prime_a = my_tpuv4_symmetry.apply_transformation(k,i_tform)
                    #     j_prime_a = my_tpuv4_symmetry.apply_transformation(j,i_tform)
                    #     (k_prime_b, j_prime_b) = my_tpuv4_symmetry.get_canonical_equivalent_edge(k_prime_a,j_prime_a)

                    # else:
                    #     i_prime, i_tlate = my_tpuv4_symmetry.translate_to_mc(i)
                    #     k_prime_a = my_tpuv4_symmetry.apply_translation(k,i_tlate)
                    #     j_prime_a = my_tpuv4_symmetry.apply_translation(j,i_tlate)

                    #     k_prime_b, kpa_tlate = my_tpuv4_symmetry.translate_to_mc(k_prime_a)

                    #     j_prime_b = my_tpuv4_symmetry.apply_translation(j_prime_a,kpa_tlate)

                    # if j_prime_a != j:
                    #     input('how??')
                    # print(f"i={i} => i'={i_prime}")
                    # print(f"k={k} => k'_a={k_prime_a}")
                    # print(f"j={j} => j'_a={j_prime_a}")
                    # print(f"r_map[{i}][{k}]={r_map[i][k]} and r_map[{i_prime}][{k_prime_a}]={r_map[i_prime][k_prime_a]}")
                    # print(f"({k_prime_a},{j_prime_a}) => {(k_prime_b, j_prime_b)}")
                    # input("cont?")

                    # (i_prime_a, j_prime_a) = my_tpuv4_symmetry.get_canonical_equivalent_edge(i,j)

                    # (i_prime_b, k_prime_a) = my_tpuv4_symmetry.get_canonical_equivalent_edge(i,k)

                    # (k_prime_b, j_prime_b) = my_tpuv4_symmetry.get_canonical_equivalent_edge(k,j)

                    myconstrname = f'constr_dist_tri_ineq_{i}r_{k}r_{j}r'
                    m.addConstr(dist[i_prime,j_prime_a] <= dist[i_prime,k_prime_a] + dist[k_prime_b,j_prime_b], name=myconstrname)
                    n_iters += 1


        print(f'tri ineq # iters = {n_iters}')

        print('Completed tri ineq constrs')


        # Objective(s)
        # --------------------------------------------------------------------------------



        dist_sum_expr = gp.LinExpr()
        for i in routers_of_canonical_cube:
            for j in range(n_routers):
                if i==j:
                    continue
                if i % 100 == 0 and j==0:
                    print(f"Working on objective i {i}")
                # if (i,j) not in edge_list:
                #     continue
                # if (i,j) not in edge_set:
                #     continue
                if r_map[i][j] == 0:
                    continue
                if VERBOSE:
                    print(f'adding edge ({i},{j}) to obj')

                dist_sum_expr += global_to_canonical_ratio*capacity*dist[(i,j)]

        m.setObjective(dist_sum_expr, GRB.MINIMIZE)

        print('Completed objective')


        # Params and Model Output
        # --------------------------------------------------------------------------------

        # Params
        m.setParam('Method', 2)
        m.setParam('Crossover', 0)
        m.setParam('BarOrder', 0)

        # Model Output

        write_model = False
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


        canonical_obj = 0
        for i in routers_of_canonical_cube:
            for j in range(n_routers):
                if r_map[i][j] == 0:
                    continue


                (ii,jj) = (min(i,j),max(i,j))
                (i_prime, j_prime) = my_tpuv4_symmetry.get_canonical_equivalent_edge(ii,jj)
                myvarname = f'var_dist_{i_prime}r_{j_prime}r'
                # print(f'Checking {myvarname}')
                v = m.getVarByName(myvarname)
                val = v.X
                canonical_obj += val




        obj_val = float(m.ObjVal)

        print(f"obj: {obj_val:g} ")
        print(f"canon obj: {canonical_obj:g} ")
        print(f"adjusted canon obj: {(global_to_canonical_ratio**2)*canonical_obj:g} ")
        print(f'sparsest set ({len(in_set)}): {in_set}')

        print(f'solver time : {solve_end_t - solve_start_t}')


        return in_set, obj_val

    # Error Handling
    # --------------------------------------------------------------------------------

    except AttributeError as e:
        print(f"Encountered an attribute error : {e}")

    except gp.GurobiError as e:
        print(f"Error code {e.errno}: {e}")


# Main(s)
# --------------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(description='Approximate SC for a graph with vertex symmetry to 0th cube')
    parser.add_argument('--topology',type=str,help='.map file to evaluate',required=True)
    parser.add_argument('--xyzc_dims',nargs='+',type=int,required=True,help='Global system x, y, z, and cube dimensions. Type without parenthesis and use spaces, no commas')
    parser.add_argument('--mc_dims',nargs='+',type=int,help='Minimum "cube" of symmetry\'s x, y, and z dimensions. If unspecified then assuming single (zeroth) cube (i.e., x,y,z==cube). Type without parenthesis and use spaces, no commas')
    parser.add_argument('--sym_type',type=str,choices=["trans","refl-trans"],default="trans",help='graph is translationally vertex symmetric')

    args = parser.parse_args()

    map_filename = args.topology
    xyzc_dims = tuple(args.xyzc_dims)
    (x_dim, y_dim, z_dim, cube_dim) = xyzc_dims
    mc_dims = (cube_dim,cube_dim,cube_dim)
    if args.mc_dims:
        mc_dims = tuple(args.mc_dims)

    assert(len(xyzc_dims) == 4)
    for d in xyzc_dims:
        assert(isinstance(d,int))
        assert( d % cube_dim == 0)
        assert( d > 0)
    assert(len(mc_dims) == 3)
    for d in mc_dims:
        # assert(d % cube_dim == 0)
        assert( d > 0)

    sym_type = args.sym_type

    r_map = ingest_map(map_filename)
    assert(len(r_map) == x_dim*y_dim*z_dim)


    find_sc(r_map, xyzc_dims, mc_dims, sym_type=sym_type)

if __name__ == '__main__':

    main()