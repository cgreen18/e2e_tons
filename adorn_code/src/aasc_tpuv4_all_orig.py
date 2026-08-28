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


"""
Title:
    aasc_tpuv4_all.py

Author(s):
    Conor Green

Description:
    Class version to generate approximate sparsest cut optimized TPU v4/5 topologies.

TODOs:
    Make the solver APIs generic and/or more modular

"""

# HiGHS
import highspy

# Gurobi
import gurobipy as gp
from gurobipy import GRB
from gurobipy import Model, LinExpr

# OR-tools
from ortools.linear_solver.python import model_builder
from ortools.math_opt.python import mathopt
from ortools.math_opt.python import solve as mathopt_solve

# pipd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib


# std
import argparse
import os
from copy import deepcopy
import time
import random
import re
import psutil
from math import gcd
import sys
import logging
import pathlib

# locals
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "..", "python_scripts"))

from tpuv4_symmetry import TPUv4_Symmetry

TREAT_ALL_DORABLE_LIKE_SYMMETRIC_DORABLE = True

# Logging
################################################################################

# TODO global?
PERFORMANCE_LEVEL = 25
logging.addLevelName(PERFORMANCE_LEVEL, "PERFORMANCE")
def performance(self, message, *args, **kwargs):
    if self.isEnabledFor(PERFORMANCE_LEVEL):
        self._log(PERFORMANCE_LEVEL, message, args, **kwargs)
logging.Logger.performance = performance

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


# Do this once early in your program (before any model.optimize()).
glog = logging.getLogger("gurobipy")
glog.setLevel(logging.WARNING)   # or logging.ERROR / logging.CRITICAL
glog.propagate = False           # critical: prevents INFO lines from going to root handlers
glog.handlers.clear()            # optional: remove any handlers attached to this logger

# Callbacks
################################################################################

class CallbackData:
    def __init__(self, modelvars):
        self.modelvars = modelvars
        self.lastiter = -GRB.INFINITY
        self.lastnode = -GRB.INFINITY


# def run_sol_callback(model, where, *, cbdata, logfile):
def run_sol_callback(model, where):

    """
    Callback function. 'model' and 'where' arguments are passed by gurobipy
    when the callback is invoked. The other arguments must be provided via
    functools.partial:
      1) 'cbdata' is an instance of CallbackData, which holds the model
         variables and tracks state information across calls to the callback.
      2) 'logfile' is a writeable file handle.
    """

    # unpack to local vars
    cbdata = model._rs_cbdata
    n_routers = model._rs_n_routers
    out_run_sol_name = model._rs_run_sol_name
    tpuv4_symmetry = model._rs_tpuv4_symmetry
    var_r_map = model._rs_var_r_map

    # sorted
    cur_r_map = [[0 for _ in range(n_routers)] for __ in range(n_routers)]

    if where == GRB.Callback.POLLING:
        # Ignore polling callback
        pass

    # new best solution found
    elif where == GRB.Callback.MIPSOL:
        # MIP solution callback
        nodecnt = model.cbGet(GRB.Callback.MIPSOL_NODCNT)
        obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        solcnt = model.cbGet(GRB.Callback.MIPSOL_SOLCNT)

        print(f"**** New solution at node {nodecnt:.0f}, obj {obj:g}, "
            f"sol {solcnt:.0f} ****")


        # # for very verbose printing of current solution
        # for cb_var in cbdata.modelvars:
        #     print(f'{cb_var.getAttr("VarName"):20} : {model.cbGetSolution(cb_var)}')


        # desired_var_names = []
        # for i in range(n_routers):
        #     for j in range(n_routers):
        #         desired_var_names.append(AASC_TPUv4._name_r_map(i,j))

        # input(f"desired_var_names = {desired_var_names}")

        # var_dict = {}
        # for cb_var in cbdata.modelvars:
        #     var_name = cb_var.getAttr('VarName')
        #     if 'm' in var_name:
        #         print(f"cb : var_name = {var_name}")
        #     if var_name in desired_var_names:
        #         cur_var_val = model.cbGetSolution(cb_var)

        #         # print(f'{var_name} = {cur_var_val}')
        #         var_dict.update({var_name : cur_var_val})

        #         cur_var_val_int = int(cur_var_val)

        #         var_name_split = var_name.split('_')

        #         src = int(var_name_split[-2].replace('r','') )
        #         dest = int(var_name_split[-1].replace('r','') )

        #         cur_r_map[src][dest] = cur_var_val_int
        #         cur_r_map[dest][src] = cur_var_val_int


        known_values = {}
        for i in tpuv4_symmetry.get_canonical_nodes():
            for j in range(n_routers):
                if i == j:
                    continue
                var = var_r_map[i,j]

                try:
                    val = model.cbGetSolution(var)
                except:
                    # float/int
                    val = var_r_map[i,j]

                known_values[(i,j)] = int(val)


        for (i,j), val in known_values.items():
            all_i_primes = tpuv4_symmetry.get_all_noncanonical_equivalents(i)

            for i_prime in all_i_primes:
                i_to_i_prime_tform = tpuv4_symmetry.calc_transform_delta(i,i_prime)
                j_prime = tpuv4_symmetry.apply_transformation(j,i_to_i_prime_tform)

                cur_r_map[i_prime][j_prime] = val
                cur_r_map[j_prime][i_prime] = val


        AASC_TPUv4.output_an_adj_mat(cur_r_map, out_run_sol_name)

        # input('cont?')


    # basic (unused) backup
    elif where == GRB.Callback.MESSAGE:
        pass


# Data structures
################################################################################

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
        if i == j:
            raise IndexError("Diagonal elements are not stored")
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
        return self.data[self._index(i, j)]

    def __setitem__(self, key, value):
        i, j = key
        self.data[self._index(i, j)] = value

class UpperTriMatrix:
    def __init__(self, n, init_val=None):
        """
        Store only the upper-triangular (i < j) entries of an n x n matrix.
        Each entry can hold an arbitrary Python object.
        """
        self.n = n
        self.size = n * (n - 1) // 2
        self.data = [init_val] * self.size  # Python objects

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
        return (i, j, k) # if i <= j else (j, i, k)
    def __getitem__(self, key): return super().__getitem__(self._n(key))
    def __setitem__(self, key, val): super().__setitem__(self._n(key), val)
    def __delitem__(self, key): return super().__delitem__(self._n(key))
    def __contains__(self, key): return super().__contains__(self._n(key))

class DSU:
    __slots__ = ("p", "sz")
    def __init__(self, nodes):
        self.p = {u: u for u in nodes}
        self.sz = {u: 1 for u in nodes}

    def find(self, a):
        p = self.p[a]
        if p != a:
            self.p[a] = self.find(p)
        return self.p[a]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        if self.sz[ra] < self.sz[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        self.sz[ra] += self.sz[rb]
        return ra


# Auto Approximate Sparsest Cut for TPUv4
################################################################################

class AASC_TPUv4():
    """

    Note:
        Function name with '_' suffix means it modifies object variables.
        Function name with '_' prefix means it is internal to the class.

    """

    basic_name = "aasc_create"

    verbose = False
    slow_run = False
    careful = True

    model_dir = "/scratch/negishi/green456/models/"
    timeline_dir = "./files/timeline_logs/"
    topology_dir = "./files/lp_iterative_solutions/"
    running_topology_dir = "./files/milp_running_solutions/"

    # nice to keep smaller than float("inf") in case working with ints or other libs
    INF = 2**31

    default_script_params = {"mc_dims":None,
                            "sym_type":"trans",
                            "dorable":False,
                            "no_same_cube":False,
                            "symmetric":False,
                            "spreading_cubes":False,
                            "write_model":False,
                            "non_integral_adjustment":0.5}

    # init/setup
    ################################################################################

    def __init__(self, xyzc_dims, script_params={}, solver_params={}):
        # hardcode
        self.n_ports = 6
        self._set_xyzc_dims_(xyzc_dims)
        self.n_routers = self.x_dim * self.y_dim * self.z_dim

        # includes sym_type, mc_dims, dorable, TODO
        self.script_params = self._set_default_script_params(script_params)
        self.solver_params = solver_params
        self.solver_library = self.script_params["solver"]

        # used often
        self.binary_r_map = self.script_params["binary_r_map"]
        self.symmetric = self.script_params["symmetric"]
        self.sym_type = self.script_params["sym_type"]
        self.mc_dims = script_params["mc_dims"]
        self.dorable = self.script_params["dorable"]
        self.no_same_cube = self.script_params["no_same_cube"]
        self.dor_heur = self.script_params["dor_heur"]
        self.scale_factor = self.script_params["scale_factor"]
        self.map_sym_breaker = self.script_params["map_sym_breaker"]
        self.uwye_sym_breaker = self.script_params["uwye_sym_breaker"]
        self.spreading_cubes = self.script_params["spreading_cubes"]
        self.write_model = self.script_params["write_model"]
        self.advanced_scoring = self.script_params["advanced_scoring"]
        self.connect_first = self.script_params["connect_first"]
        self.penalize_fractional_r_map = self.script_params["penalize_non_integral"]
        self.penalty_adjust = self.script_params["non_integral_adjustment"]
        self.climbing_penalty_adjust = False #True
        self.sos_for_limit_xyz = self.script_params["sos_for_limit_xyz"]

        self.live_plotting = True
        self.timeline_dict = {"time":[],"elapsed_time":[],"avg_hops":[],"obj_val":[],"n_unconnected":0}
        self.fig = None

        # TODO hardcode for now
        self.neighbors_only = True

        self._set_model_base_name_()

        self.tpuv4_symmetry = TPUv4_Symmetry(xyzc_dims, mc_dims=self.mc_dims, sym_type=self.sym_type)

        # stateful vars
        # -------------
        # sets self.electrical_conns_adj_list, self.electrical_conns_by_dim
        self.init_electrical_conns_()
        # sets self.adj_mat
        self.init_r_map_()
        self.chosen_optical_conns = []
        self.chosen_cube_conns = set()
        self.total_cube_conns = {(a, b) for a in range(self.n_cubes) for b in range(self.n_cubes)}
        self.chosen_optical_conns_by_dim = {'x':[],'y':[],'z':[]}
        # sets self.valid_conns
        self.init_valid_conns_()
        self.known_rejected_tri_ineqs = set()
        self.rejected_conns = set()
        self.iteration_number = 0

        self.r_map_vals = None
        self.tri_ineq_wye_vals = None
        self.opt_conn_options = None
        # self.prev_penalties = UpperTriMatrix(self.n_routers, init_val=0)

        # explicit for developers knowledge
        self.model = None
        self.solver_or_result = None
        self.var_unity_wye = None
        self.var_r_map = None
        self.var_tri_ineq_wyes = None
        self.recent_obj_val = None

        self._givens_print()

    def _set_default_script_params(self, script_params):
        assert(self.xyzc_dims)
        (x_dim, y_dim, z_dim, cube_dim) = self.xyzc_dims

        if (not script_params["mc_dims"]) or (not script_params["symmetric"]):
            script_params["mc_dims"] = (x_dim, y_dim, z_dim)
        elif (not script_params["mc_dims"]):
            script_params["mc_dims"] = (cube_dim, cube_dim, cube_dim)
        
        for def_k, def_v in self.default_script_params.items():
            if def_k not in script_params.keys():
                script_params[def_k] = def_v

        return script_params

    def _set_xyzc_dims_(self, xyzc_dims):
        assert(len(xyzc_dims) == 4)
        _cube_dim = xyzc_dims[-1]
        for dim in xyzc_dims:
            assert(isinstance(dim,int))
            assert(dim % _cube_dim == 0)

        self.xyzc_dims = xyzc_dims
        (self.x_dim, self.y_dim, self.z_dim, self.cube_dim) = xyzc_dims
        self.n_cubes = (self.x_dim // self.cube_dim)*(self.y_dim // self.cube_dim)*(self.z_dim // self.cube_dim)
        self.dim_dict = {'x':self.x_dim,'y':self.y_dim,'z':self.z_dim}

    def _set_model_base_name_(self):
        assert(self.xyzc_dims)
        assert(self.solver_params)
        assert(self.dorable is not None and self.symmetric is not None and self.dor_heur is not None)
        assert(self.solver_params)
        (x_dim, y_dim, z_dim, cube_dim) = self.xyzc_dims
        (n_routers, n_ports, n_cubes) = (self.n_routers, self.n_ports, self.n_cubes)

        model_base_name = f"{self.basic_name}_{n_cubes}c_{n_routers}r_{n_ports}p_{x_dim}x{y_dim}x{z_dim}"

        # script params
        # -------------
        dor_heur = self.dor_heur
        binary_r_map = self.binary_r_map
        dorable = self.dorable
        symmetric = self.symmetric
        no_same_cube = self.script_params["no_same_cube"]
        advanced_scoring = self.advanced_scoring
        spreading_cubes = self.spreading_cubes
        connect_first = self.connect_first
        penalize_fractional_r_map = self.penalize_fractional_r_map
        penalty_adjust = self.penalty_adjust
        climbing_penalty_adjust = self.climbing_penalty_adjust

        if symmetric:
            sym_type = self.sym_type
            (mc_x, mc_y, mc_z) = self.mc_dims

            model_base_name += f"_{sym_type}"
            model_base_name += f"_{mc_x}x{mc_y}x{mc_z}mc"

        if dor_heur:
            model_base_name += "_dor-heur"

        if binary_r_map:
            model_base_name += "_milp"

        if dorable:
            model_base_name += "_dorable"
        
        if no_same_cube:
            model_base_name += "_no-same-cube"

        if advanced_scoring:
            model_base_name += "_advscoring"

        if spreading_cubes:
            model_base_name += "_spreadc"

        if connect_first:
            model_base_name += "_connect1st"

        if penalize_fractional_r_map:
            penalty_adjust_str = str(round(penalty_adjust,2)).replace(".","p")
            model_base_name += f"_{penalty_adjust_str}non-int-penal"

        if climbing_penalty_adjust:
            model_base_name += "_climbpen"

        # solver params
        # -------------
        solver_library = self.script_params["solver"]
        scale_factor = self.script_params["scale_factor"]
        map_sym_breaker = self.script_params["map_sym_breaker"]
        uwye_sym_breaker = self.script_params["uwye_sym_breaker"]

        solver_method = self.solver_params["Method"]

        if scale_factor > 1:
            model_base_name += f"_{int(scale_factor)}sf"

        if map_sym_breaker > 0:
            dec_as_str = round(map_sym_breaker,3)
            dec_as_str = str(dec_as_str).replace(".","p")
            model_base_name += f"_{dec_as_str}rmapsbreak"

        if uwye_sym_breaker > 0:
            dec_as_str = round(uwye_sym_breaker,3)
            dec_as_str = str(dec_as_str).replace(".","p")
            model_base_name += f"_{dec_as_str}uwyesbreak"


        model_base_name += f"_{solver_library}_{solver_method}"
        
        self.model_base_name = model_base_name

    def _givens_print(self):
        assert(self.xyzc_dims)
        assert(self.model_base_name)
        (x_dim, y_dim, z_dim, cube_dim) = self.xyzc_dims
        (n_routers, n_ports, n_cubes) = (self.n_routers, self.n_ports, self.n_cubes)
        (dorable, symmetric, sym_type, mc_dims) = (self.dorable, self.symmetric, self.sym_type, self.mc_dims)
        model_base_name = self.model_base_name
        print('='*80)
        print(f"CLA/default/file values: n_cubes={n_cubes}, n_routers={n_routers}, n_ports={n_ports}")
        print(f"\tDimensionally, problem is ({x_dim}x{y_dim}x{z_dim})")
        print(f"Relevant script params: dorable={dorable}, symmetric={symmetric}, sym_type={sym_type}, mc_dims={mc_dims}")
        print(f"Base output name is {model_base_name}")
        print('='*80)

    # class methods
    ################################################################################

    # file I/O
    # --------

    @classmethod
    def timeline_log(cls, n_opt_links, avg_hops, cur_asc_obj, cumul_non_integrality, model_base_name, first_write=False, delimiter=','):
        timeline_dir = cls.timeline_dir

        log_name = f"{model_base_name}_timeline.csv"
        log_path = os.path.join(timeline_dir,log_name)

        if first_write:
            with open(log_path,'w+') as of:
                out_line = "time,n_opt_links,avg_hops,cur_asc_obj,cumul_non_integrality\n"
                of.write(out_line)
            return

        out_line = f"{time.time()},{n_opt_links},{avg_hops},{cur_asc_obj},{cumul_non_integrality}\n"
        with open(log_path,'a') as of:
            of.write(out_line)

        logger.info(f"Wrote to timeline {log_path}")

    @classmethod
    def print_rss(cls):
        rss = psutil.Process(os.getpid()).memory_info().rss  # bytes
        return f"{rss/2**20:.1f} MiB"

    @classmethod
    def output_an_adj_mat(cls, adj_mat, out_path, assert_binary=False):

        n_routers = len(adj_mat)

        out_lines = []
        for sr in range(n_routers):
            this_line = []
            for dr in range(n_routers):
                val = adj_mat[sr][dr]
                if assert_binary:
                    val = min(1, round(val))
                this_line.append(f"{val} ")

            this_line.append("\n")
            out_lines.append(this_line)

        with open(out_path, 'w+') as of:
            for line_list in out_lines:
                of.write("".join(line_list))

        logger.info(f"Wrote out adj mat to {out_path}")

    @classmethod
    def ingest_an_adj_mat(cls, in_path, assert_binary=False):

        this_map = []
        with open(in_path, 'r') as inf:
            for row in inf:
                r_conns = row.split(" ")
                if "\n" in r_conns:
                    r_conns.remove("\n")

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
        if assert_binary:
            for src_map in this_map:
                for conn in src_map:
                    # instead, make binary
                    if conn > 0:
                        conn = 1

        logger.info(f"Read in adj_mat from {in_path}")

        return this_map

    # graph functions
    # ---------------

    @classmethod
    def create_an_nwx_G_from_adj_mat(cls, adj_mat):

        n_routers = len(adj_mat)
        G = nx.Graph()

        for src in range(n_routers):
            for dest in range(src+1,n_routers):

                if(adj_mat[src][dest] < 1):
                    continue

                G.add_edge(src,dest)
                G.add_edge(dest,src)

        return G

    @classmethod
    def calc_avg_hops_from_nwx_G(cls, G):

        try:
            avg_hops = nx.average_shortest_path_length(G)
        except:
            avg_hops = cls.INF
        
        return avg_hops

    @classmethod
    def calc_diameter_from_nwx_G(cls, G):
        return nx.diameter(G)

    @classmethod
    def calc_all_pairs_hops_from_nwx_G(cls, G):
        return nx.all_pairs_bellman_ford_path_length(G)

    # utility
    # -------

    @classmethod
    def adj_mat_to_optical(self, adj_mat, electrical_conns_adj_list):
        n_routers = len(adj_mat)

        optical_conns = []
        for i in range(n_routers):
            for j in range(i+1,n_routers):
                if adj_mat[i][j] > 0 and j not in electrical_conns_adj_list[i]:
                    optical_conns.append((i,j))
                    optical_conns.append((j,i))
        return optical_conns

    # Graph functions
    # ################################################################################

    def calc_avg_hops(self):
        assert(self.adj_mat)
        adj_mat = self.adj_mat
        G = self.create_an_nwx_G_from_adj_mat(adj_mat)
        avg_hops = self.calc_avg_hops_from_nwx_G(G)
        return avg_hops

    # Utility
    # ################################################################################

    def plot_timeline(self, n_opt_links, avg_hops, cur_asc_obj, live_plotting=False):

        plt.ion()

        now_time = time.time()
        self.timeline_dict["time"].append(now_time)
        elapsed_time = now_time-self.timeline_dict["time"][0]
        self.timeline_dict["elapsed_time"].append(elapsed_time)
        self.timeline_dict["avg_hops"].append(avg_hops)
        if avg_hops >= self.INF:
            self.timeline_dict["n_unconnected"] += 1
        self.timeline_dict["obj_val"].append(cur_asc_obj)

        if not live_plotting:
            return

        if self.fig is None:
            self.fig, (self.ax0, self.ax1) = plt.subplots(2, 1, sharex=True)
            (self.line0,) = self.ax0.plot([], [], lw=2)
            (self.line1,) = self.ax1.plot([], [], lw=2)
            (self.plot_x, self.plot_y0, self.plot_y1) = ([],[],[])
            self.ax0.set_ylabel("Avg. Hops")
            self.ax1.set_ylabel("Obj. Value")
            self.ax1.set_xlabel("Time")
    
        (fig, ax0, ax1, line0, line1) = (self.fig, self.ax0, self.ax1, self.line0, self.line1)

        (x, y0, y1,n_unconnected) = (self.timeline_dict["elapsed_time"],self.timeline_dict["avg_hops"],self.timeline_dict["obj_val"],self.timeline_dict["n_unconnected"])


        # line0.set_data(x, y0)
        line0.set_data(x[1:], y0[1:])
        line1.set_data(x[n_unconnected+1:0],y1[n_unconnected+1:0])

        ax0.relim()
        ax0.autoscale_view()
        ax1.relim()
        ax1.autoscale_view()


        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(0.01)

        plt.show()



    # TPUv4 geometry
    # ################################################################################

    def r_to_xyz(self, r):
        assert(self.xyzc_dims)
        (x_dim, y_dim, z_dim, cube_dim) = self.xyzc_dims

        xy_slice_size = x_dim*y_dim

        temp_r = r

        z = temp_r // xy_slice_size
        temp_r = temp_r % xy_slice_size
        y = temp_r // x_dim
        x = temp_r % x_dim

        return x,y,z

    def xyz_to_r(self, x,y,z):
        assert(self.xyzc_dims)
        (x_dim, y_dim, z_dim, cube_dim) = self.xyzc_dims

        return x + y*x_dim + z*x_dim*y_dim

    def r_to_rel_xyz(self, r):
        assert(self.xyzc_dims)
        (x_dim, y_dim, z_dim, cube_dim) = self.xyzc_dims

        r_x,r_y,r_z = self.r_to_xyz(r)

        rel_r_x = r_x % cube_dim
        rel_r_y = r_y % cube_dim
        rel_r_z = r_z % cube_dim

        return rel_r_x, rel_r_y, rel_r_z

    def r_to_rel_r(self, r):
        (rel_x, rel_y, rel_z) = self.r_to_rel_xyz(r)
        return self.xyz_to_r(rel_x, rel_y, rel_z)

    def rel_xyz_is_on_face(self, rel_x, rel_y, rel_z):
        assert(self.xyzc_dims)
        (x_dim, y_dim, z_dim, cube_dim) = self.xyzc_dims

        if rel_x == 0 or rel_x == cube_dim - 1:
            return True
        if rel_y == 0 or rel_y == cube_dim - 1:
            return True
        if rel_z == 0 or rel_z == cube_dim - 1:
            return True
        return False

    def r_is_on_face(self, r):
        (rel_x, rel_y, rel_z) = self.r_to_rel_xyz(r)
        return self.rel_xyz_is_on_face(rel_x, rel_y, rel_z)

    def r_side_of_face(self, r, dim):
        assert(self.cube_dim)
        cube_dim = self.cube_dim
        (rel_x, rel_y, rel_z) = self.r_to_rel_xyz(r)
        rel_dict = {'x':rel_x, 'y':rel_y, 'z':rel_z}
        t = rel_dict[dim]
        if t == 0: return "low"
        if t == cube_dim - 1: return "high"
        return None

    # TPUv4 connections
    # ################################################################################

    # simple
    # ------

    def init_electrical_conns_(self):
        assert(self.xyzc_dims)
        assert(self.n_routers)
        (x_dim, y_dim, z_dim, cube_dim) = self.xyzc_dims
        n_routers = self.n_routers

        electrical_conns_adj_list = [[] for _ in range(n_routers)]
        electrical_conns_by_dim = {'x':[],'y':[],'z':[]}


        for src in range(n_routers):

            src_x,src_y,src_z = self.r_to_xyz(src)

            # xpos
            # if not on edge then conn
            if(src_x % cube_dim != cube_dim - 1):
                targ_x = src_x + 1
                targ_y = src_y
                targ_z = src_z
                targ = self.xyz_to_r(targ_x, targ_y, targ_z)
                # electrical_conns[(src,targ)] = 1
                # electrical_conns.append( (src,targ) )
                electrical_conns_adj_list[src].append(targ)
                electrical_conns_by_dim['x'].append((src,targ))

            # xneg
            # if not on edge then conn
            if(src_x % cube_dim != 0):
                targ_x = src_x - 1
                targ_y = src_y
                targ_z = src_z
                targ = self.xyz_to_r(targ_x, targ_y, targ_z)
                # electrical_conns[(src,targ)] = 1
                # electrical_conns.append( (src,targ) )
                electrical_conns_adj_list[src].append(targ)
                electrical_conns_by_dim['x'].append((src,targ))


            # ypos
            # if not on edge then conn
            if(src_y % cube_dim != cube_dim - 1):
                targ_x = src_x
                targ_y = src_y + 1
                targ_z = src_z
                targ = self.xyz_to_r(targ_x, targ_y, targ_z)
                # electrical_conns[(src,targ)] = 1
                # electrical_conns.append( (src,targ) )
                electrical_conns_adj_list[src].append(targ)
                electrical_conns_by_dim['y'].append((src,targ))

            # yneg
            # if not on edge then conn
            if(src_y % cube_dim != 0):
                targ_x = src_x
                targ_y = src_y - 1
                targ_z = src_z
                targ = self.xyz_to_r(targ_x, targ_y, targ_z)
                # electrical_conns[(src,targ)] = 1
                # electrical_conns.append( (src,targ) )
                electrical_conns_adj_list[src].append(targ)
                electrical_conns_by_dim['y'].append((src,targ))


            # zpos
            # if not on edge then conn
            if(src_z % cube_dim != cube_dim - 1):
                targ_x = src_x
                targ_y = src_y
                targ_z = src_z + 1
                targ = self.xyz_to_r(targ_x, targ_y, targ_z)
                # electrical_conns[(src,targ)] = 1
                # electrical_conns.append( (src,targ) )
                electrical_conns_adj_list[src].append(targ)
                electrical_conns_by_dim['z'].append((src,targ))

            # zneg
            # if not on edge then conn
            if(src_z % cube_dim != 0):
                targ_x = src_x
                targ_y = src_y
                targ_z = src_z - 1
                targ = self.xyz_to_r(targ_x, targ_y, targ_z)
                # electrical_conns[(src,targ)] = 1
                # electrical_conns.append( (src,targ) )
                electrical_conns_adj_list[src].append(targ)
                electrical_conns_by_dim['z'].append((src,targ))

        self.electrical_conns_adj_list = electrical_conns_adj_list
        self.electrical_conns_by_dim = electrical_conns_by_dim

    def init_r_map_(self):
        assert(self.n_routers)
        assert(self.electrical_conns_adj_list)
        n_routers = self.n_routers
        electrical_conns_adj_list = self.electrical_conns_adj_list

        adj_mat = [[0 for _ in range(n_routers)] for __ in range(n_routers)]

        for i, conns in enumerate(electrical_conns_adj_list):
            for j in conns:
                adj_mat[i][j] = 1
                adj_mat[j][i] = 1

        self.adj_mat = adj_mat

    def iter_rel_xyz_across_cubes(self, rel_x,rel_y,rel_z):
        assert(self.xyzc_dims)
        (x_dim, y_dim, z_dim, cube_dim) = self.xyzc_dims

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
                    targ = self.xyz_to_r(xprime, yprime, zprime)
                    targs.append(targ)
        
        return targs

    def poss_optical_conns_for_r(self, r):
        assert(self.xyzc_dims)
        (x_dim, y_dim, z_dim, cube_dim) = self.xyzc_dims

        rel_r_x, rel_r_y, rel_r_z = self.r_to_rel_xyz(r)

        n_x_cubes = x_dim // cube_dim
        n_y_cubes = y_dim // cube_dim
        n_z_cubes = z_dim // cube_dim

        poss_conns = {'x+':[],'x-':[],'y+':[],'y-':[],'z+':[],'z-':[]}

        # x+
        if rel_r_x == cube_dim - 1:
            poss_conns['x+'] += self.iter_rel_xyz_across_cubes(0, rel_r_y, rel_r_z)
        # x-
        if rel_r_x == 0:
            poss_conns['x-'] += self.iter_rel_xyz_across_cubes(cube_dim - 1, rel_r_y, rel_r_z)

        # y+
        if rel_r_y == cube_dim - 1:
            poss_conns['y+'] += self.iter_rel_xyz_across_cubes(rel_r_x, 0, rel_r_z)
        # y-
        if rel_r_y == 0:
            poss_conns['y-'] += self.iter_rel_xyz_across_cubes(rel_r_x, cube_dim - 1, rel_r_z)

        # z+
        if rel_r_z == cube_dim - 1:
            poss_conns['z+'] += self.iter_rel_xyz_across_cubes(rel_r_x, rel_r_y, 0)
        # z-
        if rel_r_z == 0:
            poss_conns['z-'] += self.iter_rel_xyz_across_cubes(rel_r_x, rel_r_y, cube_dim - 1)

        if self.verbose:
            print(f'{r} possibly optical conns to')
            for k,v in poss_conns.items():
                print(f'\t{k} : {v}')

        return poss_conns

    def get_group_edges(self, r, direction):
        conn_dict = self.poss_optical_conns_for_r(r)
        conn_nodes = conn_dict[direction]
        conn_edges = []
        for c in conn_nodes:
            conn_edges.append((min(r,c), max(r,c)))
        return conn_edges

    def conn_is_optical(self,s,d):
        if self.calc_opt_conn_type(s,d) is None:
            return False
        return True

    def calc_opt_conn_type(self, s, d):
        assert(self.xyzc_dims)
        (x_dim, y_dim, z_dim, cube_dim) = self.xyzc_dims

        rel_s_x, rel_s_y, rel_s_z = self.r_to_rel_xyz(s)
        rel_d_x, rel_d_y, rel_d_z = self.r_to_rel_xyz(d)
        
        
        if (not self.rel_xyz_is_on_face(rel_s_x, rel_s_y, rel_s_z)) or (not self.rel_xyz_is_on_face(rel_d_x, rel_d_y, rel_d_z)):
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

    def calc_conn_type(self, s, d):
        assert(self.xyzc_dims)
        (x_dim, y_dim, z_dim, cube_dim) = self.xyzc_dims

        rel_s_x, rel_s_y, rel_s_z = self.r_to_rel_xyz(s)
        rel_d_x, rel_d_y, rel_d_z = self.r_to_rel_xyz(d)

        # x+
        if rel_s_x < rel_d_x:
            return 'x', '+'
        # x-
        if rel_s_x > rel_d_x:
            return 'x', '-'

        # y+
        if rel_s_y < rel_d_y:
            return 'y', '+'
        # y-
        if rel_s_y > rel_d_y:
            return 'y', '-'

        # z+
        if rel_s_z < rel_d_z:
            return 'z', '+'
        # z-
        if rel_s_z > rel_d_z:
            return 'z', '-'

    def calc_conn_type_cumulative(self, s, d):
        assert(self.xyzc_dims)
        (x_dim, y_dim, z_dim, cube_dim) = self.xyzc_dims

        rel_s_x, rel_s_y, rel_s_z = self.r_to_rel_xyz(s)
        rel_d_x, rel_d_y, rel_d_z = self.r_to_rel_xyz(d)

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

        return conn_type, ''

    def which_cube(self, i):
        assert(self.xyzc_dims)
        (x_dim, y_dim, z_dim, cube_dim) = self.xyzc_dims

        n_x = (x_dim // cube_dim)
        n_y = (y_dim // cube_dim)
        n_z = (z_dim // cube_dim)

        i_x,i_y,i_z = self.r_to_xyz(i)

        n_i_x = i_x // cube_dim
        n_i_y = i_y // cube_dim
        n_i_z = i_z // cube_dim

        n_xy = n_x*n_y
        n_cube = (n_i_z)*n_xy + (n_i_y)*n_x + (n_i_x)

        return n_cube

    # complicated
    # -----------
    def init_valid_conns_(self):
        assert(self.xyzc_dims)
        assert(self.n_routers)
        assert(self.electrical_conns_adj_list)
        assert(self.dorable is not None and self.no_same_cube is not None and self.symmetric is not None)
        (x_dim, y_dim, z_dim, cube_dim) = self.xyzc_dims
        n_routers = self.n_routers
        electrical_conns_adj_list = self.electrical_conns_adj_list
        no_same_cube = self.no_same_cube
        dorable = self.dorable
        symmetric =self.symmetric

        # 1) assume all invalid
        # 2) all electrical valid
        # 3) assume all optical valid
        # 4) handle restrictions: dorable, no_same_cube

        # 1) assume invalid
        valid_conns = [[0 for _ in range(n_routers)] for __ in range(n_routers)]

        # 2) update the electrical conns
        for i, conns in enumerate(electrical_conns_adj_list):
            for j in conns:
                valid_conns[i][j] = 1
                # print(f"added electrical {((i,j))}")

        # 3)
        for i in range(n_routers):
            (rel_i_x, rel_i_y, rel_i_z) = self.r_to_rel_xyz(i)

            # early exit for inner routers
            if not self.rel_xyz_is_on_face(rel_i_x, rel_i_y, rel_i_z):
                continue

            poss_conns = self.poss_optical_conns_for_r(i)

            for direction, conns in poss_conns.items():
                for conn in conns:
                    valid_conns[i][conn] = 1
                    valid_conns[conn][i] = 1

                    if self.verbose:
                        print(f"\t\t{i}<->{conn} in direction {direction} (potentially) allowed")

        # 4)

        if no_same_cube:

            for i in range(n_routers):
                poss_conns = self.poss_optical_conns_for_r(i)
                for conn_type_w_direction, conns in poss_conns.items():
                    for j in conns:
                        if valid_conns[i][j] == 0:
                            continue

                        (i_x, i_y, i_z) = self.r_to_xyz(i)
                        (j_x, j_y, j_z) = self.r_to_xyz(j)

                        conn_type = conn_type_w_direction[0]

                        if self.verbose:
                            print(f"no_same_cube {i} @ {(i_x, i_y, i_z)} -> {j} @ {(j_x, j_y, j_z)} type {conn_type}")

                        # cube dims
                        (i_x_c, i_y_c, i_z_c) = (i_x // cube_dim, i_y // cube_dim, i_z // cube_dim)
                        (j_x_c, j_y_c, j_z_c) = (j_x // cube_dim, j_y // cube_dim, j_z // cube_dim)
                        (delta_x, delta_y, delta_z) = (abs(i_x_c - j_x_c), abs(i_y_c-j_y_c), abs(i_z_c-j_z_c))
                        delta_dict = {'x':delta_x,'y':delta_y,'z':delta_z}

                        relevant_dim = self.dim_dict[conn_type]
                        relevant_delta = delta_dict[conn_type]

                        if (relevant_dim > cube_dim) and relevant_delta==0:
                            valid_conns[i][j] = 0
                            if self.verbose:
                                print(f"no_same_cube precluding {conn_type} hop : {i}->{j}")

        if dorable:

            # 2 possibilities
            #   1) asymmetric => no self cube conns for dims > cube dim
            #   2) symmetric => only prime cube deltas along dims > cube dim

            # # cubes in each dimension
            (x_c, y_c, z_c) = (x_dim // cube_dim, y_dim // cube_dim, z_dim // cube_dim)
            cubes_dim_dict = {'x':x_c,'y':y_c,'z':z_c}

            def is_prime_wrt_dim_cubes(x, y):
                return gcd(x, y) == 1

            for i in range(n_routers):
                poss_conns = self.poss_optical_conns_for_r(i)
                for conn_type_w_direction, conns in poss_conns.items():
                    for j in conns:
                        if valid_conns[i][j] == 0:
                            continue

                        (i_x, i_y, i_z) = self.r_to_xyz(i)
                        (j_x, j_y, j_z) = self.r_to_xyz(j)

                        conn_type = conn_type_w_direction[0]

                        if self.verbose:
                            print(f"dorable {i} @ {(i_x, i_y, i_z)} -> {j} @ {(j_x, j_y, j_z)} type {conn_type}")

                        (i_x_c, i_y_c, i_z_c) = (i_x // cube_dim, i_y // cube_dim, i_z // cube_dim)
                        (j_x_c, j_y_c, j_z_c) = (j_x // cube_dim, j_y // cube_dim, j_z // cube_dim)
                        (delta_x, delta_y, delta_z) = (abs(i_x_c - j_x_c), abs(i_y_c-j_y_c), abs(i_z_c-j_z_c))
                        delta_dict = {'x':delta_x,'y':delta_y,'z':delta_z}

                        relevant_dim = self.dim_dict[conn_type]
                        relevant_delta = delta_dict[conn_type]
                        relevant_n_cubes = cubes_dim_dict[conn_type]

                        # always acceptable
                        if (relevant_dim <= cube_dim):
                            continue

                        if not symmetric and (not TREAT_ALL_DORABLE_LIKE_SYMMETRIC_DORABLE):
                            # ie self conn
                            if relevant_delta==0:
                                valid_conns[i][j] = 0
                                if self.verbose:
                                    print(f"dorable asymmetric precluding {conn_type} hop : {i}->{j} because  (relevant_dim > cube_dim)=({relevant_dim} > {cube_dim})? {relevant_dim > cube_dim} and relevant_delta={relevant_delta}==0? {relevant_delta==0}")

                        else:
                            other_types = ['x','y','z']
                            other_types.remove(conn_type)
                            if not is_prime_wrt_dim_cubes(relevant_delta,relevant_n_cubes):
                                valid_conns[i][j] = 0
                                if self.verbose:
                                    print(f"dorable symmetric precluding {conn_type} hop : {i}->{j}")
                                continue
                            
                            for other_type in other_types:
                                other_dim = self.dim_dict[other_type]
                                other_delta = delta_dict[other_type]
                                other_n_cubes = cubes_dim_dict[other_type]
                                if (other_dim > cube_dim) and  other_delta > 0:#not is_prime_wrt_dim_cubes(other_delta,other_n_cubes):
                                    valid_conns[i][j] = 0
                                    if self.verbose:
                                        print(f"dorable {i} @ {(i_x, i_y, i_z)} -> {j} @ {(j_x, j_y, j_z)} type {conn_type}")
                                        print(f"dorable symmetric precluding {conn_type} hop : {i}->{j}")
                                        print(f"because other dim {other_type} w/ size {other_dim} has delta {other_delta} for # cubes {other_n_cubes}")
                                        input('cont?')

        if self.verbose:
            for i in range(n_routers):
                print(f"{i} has valid conns {[j for j in range(n_routers) if valid_conns[i][j] > 0]}")

        self.valid_conns = valid_conns

        if self.careful or True:
            for i in range(self.n_routers):
                assert(sum(self.valid_conns[i]) >= 6)

    def update_valid_conns_(self, opt_conn):
        assert(self.valid_conns)
        assert(self.dorable is not None)
        dorable = self.dorable

        if self.verbose:
            print(f"Updating valid conns for opt conn {opt_conn}")

        (i,j) = opt_conn
        assert(self.valid_conns[i][j] > 0 and self.valid_conns[j][i] > 0)

        ij_dir = self.calc_opt_conn_type(i,j)
        ji_dir = self.calc_opt_conn_type(j,i)

        # i <-> possible[i] along direction ij_dir
        i_poss_conns = self.poss_optical_conns_for_r(i)
        for conn in i_poss_conns[ij_dir]:
            if conn == j:
                continue
            if self.verbose:
                print(f'precluding opt conn for {i} in dir {ij_dir} : {i}<->{conn} => {self.r_to_xyz(i)}<->{self.r_to_xyz(conn)}. previously was {self.valid_conns[i][conn]}')
            self.valid_conns[i][conn] = 0
            self.valid_conns[conn][i] = 0

        # j <-> possible[j] along direction ij_dir
        j_poss_conns = self.poss_optical_conns_for_r(j)
        for conn in j_poss_conns[ji_dir]:
            if conn == i:
                continue
            if self.verbose:
                print(f'precluding opt conn for {j} in dir {ji_dir} : {j}<->{conn} => {self.r_to_xyz(j)}<->{self.r_to_xyz(conn)}. previously was {self.valid_conns[j][conn]}')
            self.valid_conns[j][conn] = 0
            self.valid_conns[conn][j] = 0

        if self.careful or True:
            for i in range(self.n_routers):
                if sum(self.valid_conns[i]) < 6:
                    logger.critical(f"valid conns for {i} does not have enough")
                assert(sum(self.valid_conns[i]) >= 6)
            

            for i in range(self.n_routers):
                poss_conns = self.poss_optical_conns_for_r(i)
                for conn_type_w_direction, conns in poss_conns.items():
                    if len(conns) == 0:
                        continue
                    at_least_one_valid = False
                    for j in conns:
                        if self.valid_conns[i][j] > 0 or self.adj_mat[i][j] > 0:
                            at_least_one_valid = True
                    
                    if not at_least_one_valid:
                        logger.critical(f"valid conns for {i} in direction {conn_type_w_direction} does not have enough")
                        print(f"adj_mat : {self.adj_mat[i]}")
                    assert(at_least_one_valid)


    # Symmetry (mainly indirect to TPUv4_Symmetry with safety checks)
    ################################################################################

    def get_canonical_equivalent(self, r):
        assert(self.tpuv4_symmetry)
        return self.tpuv4_symmetry.get_canonical_equivalent(r)[0]

    def get_canonical_nodes(self):
        assert(self.n_routers)
        assert(self.symmetric is not None)
        assert(self.tpuv4_symmetry)
        nodes = self.tpuv4_symmetry.get_canonical_nodes()

        if not self.careful:
            return nodes
        
        if not self.symmetric:
            node_set = set(nodes)
            for n in range(self.n_routers):
                assert(n in node_set)

        return nodes

    def get_all_noncanonical_equivalents(self, r):
        assert(self.n_routers)
        assert(self.symmetric is not None)
        assert(self.tpuv4_symmetry)
        nodes = self.tpuv4_symmetry.get_all_noncanonical_equivalents(r)

        if not self.careful:
            return nodes

        for n in nodes:
            assert(r == self.get_canonical_equivalent(n))

        return nodes

    def calc_transform_delta(self,r,r_prime):
        assert(self.tpuv4_symmetry)
        tform = self.tpuv4_symmetry.calc_transform_delta(r,r_prime)

        if not self.careful:
            return tform

        assert(r_prime == self.tpuv4_symmetry.apply_transformation(r,tform))
        return tform

    def apply_transformation(self, r, tform):
        assert(self.tpuv4_symmetry)
        r_prime = self.tpuv4_symmetry.apply_transformation(r,tform)

        if not self.careful:
            return r_prime

        # TODO careful
        return r_prime

    # DOR
    ################################################################################

    def calc_theoretical_loop_length(self, r, new_conn, loop_dim):
        
        pos_len, open_loop = self.calc_theoretical_directed_loop_length(r, new_conn, f"{loop_dim}+")
        if not open_loop:
            return pos_len, open_loop
        
        neg_len, open_loop = self.calc_theoretical_directed_loop_length(r, new_conn, f"{loop_dim}-")

        return pos_len + neg_len, open_loop

    def calc_line_nodes_through(self, r, dim):
        assert(self.xyzc_dims)
        (x_dim, y_dim, z_dim, cube_dim) = self.xyzc_dims

        x, y, z = self.r_to_xyz(r)
        d = dim[0].lower()
        if d == "x":
            return [self.xyz_to_r(xx, y, z) for xx in range(x_dim)]
        if d == "y":
            return [self.xyz_to_r(x, yy, z) for yy in range(y_dim)]
        if d == "z":
            return [self.xyz_to_r(x, y, zz) for zz in range(z_dim)]
        logging.critical(f"Unknown dim={dim}")
        quit()

    def prune_disallowed(self, edges, dim, require_opposite_faces = True):
        assert(self.valid_conns)
        valid_conns = self.valid_conns

        pruned = []
        for a, b in edges:
            fa = self.r_side_of_face(a, dim)
            fb = self.r_side_of_face(b, dim)
            if fa is None or fb is None:
                continue  # at least one endpoint is interior
            if require_opposite_faces and fa == fb:
                continue  # same face-to-same face is not a valid inter-cube optical hop

            if valid_conns[a][b] == 0:
                continue
            pruned.append((a, b))
        return pruned

    @classmethod
    def calc_disallowed_set_for_line(cls, line_nodes, dim, chosen_edges_set):
        """
        Disallowed (a,b) due to subtour constraint only, for this one line.
        """
        L = len(line_nodes)
        dsu = DSU(line_nodes)

        # union endpoints of already-chosen edges that lie on this line
        line_set = set(line_nodes)
        for (u, v) in chosen_edges_set:
            if u in line_set and v in line_set:
                dsu.union(u, v)

        # compute component sizes
        comp_size = {}
        for u in line_nodes:
            ru = dsu.find(u)
            comp_size[ru] = dsu.sz[ru]

        # any directed edge inside a component of size < L would close a subtour
        dis = set()
        for a in line_nodes:
            ra = dsu.find(a)
            if comp_size[ra] >= L:
                continue
            for b in line_nodes:
                if a == b:
                    continue
                if dsu.find(b) == ra:
                    e = (a, b)
                    if e not in chosen_edges_set:
                        dis.add(e)
        return dis

    def calc_dor_disallowed(self, new_conn):
        assert(self.chosen_optical_conns_by_dim)
        assert(self.electrical_conns_by_dim)
        assert(self.electrical_conns_adj_list)
        chosen_optical_conns_by_dim = self.chosen_optical_conns_by_dim
        electrical_conns_by_dim = self.electrical_conns_by_dim
        electrical_conns_adj_list = self.electrical_conns_adj_list

        (i, j) = new_conn
        d = self.calc_opt_conn_type(i,j)[0]

        line_nodes = self.calc_line_nodes_through(i, d)

        # current chosen edges along dim (before)
        before = set(electrical_conns_by_dim[d] + chosen_optical_conns_by_dim[d])

        dis_before = self.calc_disallowed_set_for_line(line_nodes, d, before)

        newly = [(i,j) for (i,j) in dis_before if self.valid_conns[i][j] > 0]
        newly = [(i,j) for (i,j) in newly if j not in electrical_conns_adj_list[i]]

        return newly

    def calc_coord_diff(self, s,d, relative=False):

        if not relative:
            s_x, s_y, s_z = self.r_to_xyz(s)
            d_x, d_y, d_z = self.r_to_xyz(d)
        else:
            s_x, s_y, s_z = self.r_to_rel_xyz(s)
            d_x, d_y, d_z = self.r_to_rel_xyz(d)

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


    def ijk_dor_relevant(self, i,j,k, dor_order = ["free_first",'x','y','z',"free_last"]):
        assert(self.valid_conns)
        valid_conns = self.valid_conns

        if valid_conns[i][k] == 0:
            return False

        dor_idx = {v: i for i, v in enumerate(dor_order)}

        ik_flow_types = self.calc_coord_diff(i,k, relative=True)
        kj_flow_types = self.calc_coord_diff(k,j, relative=True)

        if self.verbose:
            print(f"i->k : {i}->{k} @ {self.r_to_xyz(i)} -> {self.r_to_xyz(k)} by {ik_flow_types}")
            print(f"k->j : {k}->{j} @ {self.r_to_xyz(k)} -> {self.r_to_xyz(j)} by {kj_flow_types}")

        ik_last = "free_first" if len(ik_flow_types)==0 else ik_flow_types[-1]
        kj_first = "free_last" if len(kj_flow_types)==0 else kj_flow_types[0]

        allowed = True if dor_idx[ik_last] <= dor_idx[kj_first] else False 

        if self.verbose:
            print(f"dor_idx[ik_last] = dor_idx[{ik_last}] = {dor_idx[ik_last]}")
            print(f"dor_idx[kj_first] = dor_idx[{kj_first}] = {dor_idx[kj_first]}")
            print(f'=> allowed? {allowed}')    

            input('cont?')

        return allowed

    # def ijk_dor_relevant(self, i,j,k):
    #     assert(self.valid_conns)
    #     assert(self.electrical_conns_adj_list)
    #     valid_conns = self.valid_conns
    #     electrical_conns_adj_list = self.electrical_conns_adj_list

    #     if valid_conns[i][k] == 0:
    #         return False

    #     is_electrical = True if k in electrical_conns_adj_list[i] else False

    #     flow_types = self.calc_coord_diff(i,j)

    #     if is_electrical or True:
    #         if 'x' in flow_types:
    #             important_types = ['x']
    #         elif 'y' in flow_types:
    #             important_types = ['y']
    #         elif 'z' in flow_types:
    #             important_types = ['z']
    #         else:
    #             if self.verbose:
    #                 print(f"No coord diff {i}->{j} => {self.r_to_xyz(i)}->{self.r_to_xyz(j)}")
    #             important_types = ['x','y','z']
    #     else:
    #         if 'x' in flow_types:
    #             important_types = ['x']
    #         elif 'y' in flow_types:
    #             important_types = ['x','y']
    #         elif 'z' in flow_types:
    #             important_types = ['x','y','z']
    #         else:
    #             if self.verbose:
    #                 print(f"No coord diff {i}->{j} => {self.r_to_xyz(i)}->{self.r_to_xyz(j)}")
    #             important_types = ['x','y','z']

    #     allowed = False
    #     ik_dim, _ik_pos_neg = self.calc_conn_type(i,k)
    #     if ik_dim in important_types:
    #         allowed = True

    #     # ik_dim, _ik_pos_neg = self.calc_conn_type_cumulative(i,k)
    #     # for d in ik_dim:
    #     #     if d in important_types:
    #     #         allowed = True


    #     if self.verbose:
    #         print(f"Because coord diff {i}->{j} => {self.r_to_xyz(i)}->{self.r_to_xyz(j)} = {flow_types} then important {important_types} and so 'ik' {i}->{k} of type {ik_dim} allowed? {allowed}")
    #         # input('cont?')

    #     return allowed

    # Model creation/updating (not migrated to new GenericMILPAPI)
    ################################################################################

    # naming
    # ------
    @classmethod
    def _name_unity_wye(cls):
        return f"y0"

    @classmethod
    def _name_r_map(cls,i,j):
        return f"m_{i}_{j}"

    @classmethod
    def _name_tri_ineq_wye(cls, i,j,k):
        return f"y_{i}_{j}_{k}"

    @classmethod
    def _name_limit_xyz_conns(cls, i,d):
        return f"l{d}_{i}"

    @classmethod
    def _name_constr_A_transpose(cls, a,b):
        return f"At_{a}_{b}"

    # mux solver apis
    # ---------------
    # TODO make these class methods
    def create_model_(self):
        assert(self.solver_library)
        solver_library = self.solver_library

        if solver_library == 'highs':
            self.model = highspy.Highs()
        elif solver_library == 'gurobi':
            self.model = gp.Model()
        elif solver_library == 'ortools_model_builder':
            self.model = model_builder.ModelBuilder()
        elif solver_library == 'ortools':
            self.model = mathopt.Model()
        else:
            logger.error(f'Unrecognized model type {solver_library}. Exiting...')
            quit()

    def add_constr_sum_equality_(self, lhs_vars, rhs_val, name):
        assert(self.model)
        assert(self.solver_library)
        solver_library = self.solver_library

        # errors on unrecognized type
        if solver_library == 'highs':
            self.model.addConstr(model.qsum(lhs_vars) == rhs_val, name=name)
        elif solver_library == 'gurobi':
            self.model.addConstr(gp.quicksum(lhs_vars) == rhs_val, name=name)
        elif solver_library == 'ortools_model_builder':
            self.model.add(sum(lhs_vars) == rhs_val, name=name)
        elif solver_library == 'ortools':
            self.model.add_linear_constraint(sum(lhs_vars) == rhs_val, name=name)

    def add_constr_pos_neg_sum_lte_(self, pos_lhs_vars, neg_lhs_vars, rhs_val, name):
        assert(self.model)
        assert(self.solver_library)
        solver_library = self.solver_library

        if solver_library == 'highs':
            self.model.addConstr(model.qsum(pos_lhs_vars) - model.qsum(neg_lhs_vars) <= rhs_val, name=name)
        elif solver_library == 'gurobi':
            self.model.addConstr(gp.quicksum(pos_lhs_vars) - gp.quicksum(neg_lhs_vars) <= rhs_val, name=name)
        elif solver_library == 'ortools_model_builder':
            self.model.add(sum(pos_lhs_vars) - sum(neg_lhs_vars) <= rhs_val, name=name)
        elif solver_library == 'ortools':
            self.model.add_linear_constraint(sum(pos_lhs_vars) - sum(neg_lhs_vars) <= rhs_val, name=name)

    def add_var_continuous_(self, lb, ub, name):
        assert(self.model)
        assert(self.solver_library)
        solver_library = self.solver_library

        if solver_library == 'highs':
            var_ptr = self.model.addVariable(lb=lb, ub=ub, name=name)
        elif solver_library == 'gurobi':
            var_ptr = self.model.addVar(lb=lb, ub=ub, vtype=GRB.CONTINUOUS,  name=name)
        elif solver_library == 'ortools_model_builder':
            var_ptr = self.model.new_num_var(lb, ub, name)
        elif solver_library == 'ortools':
            var_ptr = self.model.add_variable(lb=lb, ub=ub, name=name)
        else:
            logger.error(f'Unrecognized model type {solver_library}. Exiting...')
            quit()
        
        return var_ptr

    def add_var_integer_(self, lb, ub, name):
        assert(self.model)
        assert(self.solver_library)
        solver_library = self.solver_library

        if solver_library == 'highs':
            # var_ptr = self.model.addVariable(lb=lb, ub=ub, name=name)
            logger.critical("unimplemented")
            quit()
        elif solver_library == 'gurobi':
            var_ptr = self.model.addVar(lb=lb, ub=ub, vtype=GRB.INTEGER,  name=name)
        elif solver_library == 'ortools_model_builder':
            # var_ptr = self.model.new_num_var(lb, ub, name)
            logger.critical("unimplemented")
            quit()
        elif solver_library == 'ortools':
            var_ptr = self.model.add_variable(lb=lb, ub=ub, name=name)
            logger.critical("unimplemented")
            quit()
        else:
            logger.error(f'Unrecognized model type {solver_library}. Exiting...')
            quit()
        
        return var_ptr

    def add_var_binary_(self, name):
        assert(self.model)
        assert(self.solver_library)
        solver_library = self.solver_library

        if solver_library == 'highs':
            var_ptr = self.model.addBinary(name=name)
        elif solver_library == 'gurobi':
            var_ptr = self.model.addVar(vtype=GRB.BINARY,  name=name)
        elif solver_library == 'ortools_model_builder':
            var_ptr = self.model.new_bool_var(name)
        elif solver_library == 'ortools':
            logger.critical("ORTools binary variables unimplemented")
            quit()
        else:
            logger.error(f'Unrecognized model type {solver_library}. Exiting...')
            quit()
        
        return var_ptr

    # AASD general
    # ------------

    def relevant_ijk(self,i,j,k):
        assert(self.dor_heur is not None)
        assert(self.neighbors_only is not None)
        assert(self.valid_conns)

        if self.dor_heur:
            allowed = self.ijk_dor_relevant(i,j,k)
            # if not allowed:
            #     allowed = self.ijk_dor_relevant(i,j,k, dor_order = ["free_first",'z','y','x',"free_last"])
            return allowed
        elif self.neighbors_only:
            # return self.adj_mat[i][k] == 1
            return self.valid_conns[i][k]==1
        else:
            return True

    # AASD constraints
    # ----------------
    def constr_limit_xyz_conns_(self):
        assert(self.var_r_map)
        assert(self.scale_factor)
        assert(self.sos_for_limit_xyz is not None)
        var_r_map = self.var_r_map
        scale_factor = self.scale_factor
        sos_for_limit_xyz = self.sos_for_limit_xyz

        routers_of_canonical_cube = self.get_canonical_nodes()

        self.model.update()

        for i in routers_of_canonical_cube:
            
            poss_conns = self.poss_optical_conns_for_r(i)
            for direction, conns in poss_conns.items():

                if len(conns) == 0:
                    continue
                
                conn_list = [var_r_map[i,c] for c in conns]

                myconstrname = self._name_limit_xyz_conns(i,direction)


                if self.sos_for_limit_xyz:

                    # detect knowns
                    known_list =  [v for v in conn_list if isinstance(v,int) ]
                    if 1 in known_list:
                        continue
                    var_list = [v for v in conn_list if not isinstance(v,int) ]
                    self.model.addSOS(GRB.SOS_TYPE1,var_list)

                # self.add_constr_sum_equality_(var_list, scale_factor, myconstrname)
                self.add_constr_sum_equality_(conn_list, 1, myconstrname)

    def constr_dor_conns_(self):
        assert(self.n_cubes)
        assert(self.xyzc_dims)
        assert(self.var_r_map)


        (x_dim, y_dim, z_dim, cube_dim) = self.xyzc_dims
        n_cubes = self.n_cubes
        var_r_map = self.var_r_map

        # x line
        for y in range(y_dim):
            for z in range(z_dim):
                # O(n^2)
                
                nc_x = x_dim // cube_dim
                if nc_x == 1:
                    continue

                nodes_neg_side = []
                nodes_pos_side = []
                for x in range(0,nc_x*cube_dim,cube_dim):
                    nodes_neg_side.append(self.xyz_to_r(x, y, z))
                    nodes_pos_side.append(self.xyz_to_r(x+(cube_dim-1), y, z))

                cube_flow_vars = {}
                
                for a in range(nc_x):
                    cube_flow_vars[ (a,a) ] = 0
                    for b in range(nc_x):
                        if a==b:
                            continue

                        # create flow var
                        myvarname = f"c_f_xline_{y}y{z}z_{a}a_{b}b"
                        flow_var = self.add_var_continuous_(0, nc_x-1, myvarname)
                        cube_flow_vars[ (a,b) ] = flow_var

                        # constr flow var

                        
                        # these are different and not symmetric
                        conn_0 = ( nodes_neg_side[a], nodes_pos_side[b] )
                        conn_1 = ( nodes_pos_side[a], nodes_neg_side[b] )

                        myconstrname = f"constr_f_xline_{y}y{z}z_{a}a_{b}b"
                        self.model.addConstr( flow_var <= (nc_x-1)*(var_r_map[conn_0] + var_r_map[conn_1]) , name=myconstrname )
                
                # root source
                # out - in = nc - 1

                myconstrname = f"constr_f_xline_{y}y{z}z_root"
                self.model.addConstr( gp.quicksum([cube_flow_vars[0, b] for b in range(nc_x)]) - gp.quicksum([cube_flow_vars[b, 0] for b in range(nc_x)]) == nc_x - 1, name=myconstrname )

                # others
                # in - out = 1
                for b in range(1,nc_x):
                    myconstrname = f"constr_f_xline_{y}y{z}z_cube{b}"
                    self.model.addConstr( gp.quicksum([cube_flow_vars[a, b] for a in range(nc_x)]) - gp.quicksum([cube_flow_vars[b, a] for a in range(nc_x)]) == 1, name=myconstrname )

        # y line
        for x in range(x_dim):
            for z in range(z_dim):
                
                nc_y = y_dim // cube_dim
                if nc_y == 1:
                    continue

                nodes_neg_side = []
                nodes_pos_side = []
                for y in range(0,nc_y*cube_dim,cube_dim):
                    nodes_neg_side.append(self.xyz_to_r(x, y, z))
                    nodes_pos_side.append(self.xyz_to_r(x, y+(cube_dim-1), z))

                cube_flow_vars = {}
                
                for a in range(nc_y):
                    cube_flow_vars[ (a,a) ] = 0
                    for b in range(nc_y):
                        if a==b:
                            continue

                        # create flow var
                        myvarname = f"c_f_yline_{x}x{z}z_{a}a_{b}b"
                        flow_var = self.add_var_continuous_(0, nc_y-1, myvarname)
                        cube_flow_vars[ (a,b) ] = flow_var

                        # constr flow var

                        
                        # these are different and not symmetric
                        conn_0 = ( nodes_neg_side[a], nodes_pos_side[b] )
                        conn_1 = ( nodes_pos_side[a], nodes_neg_side[b] )

                        myconstrname = f"constr_f_yline_{x}x{z}z_{a}a_{b}b"
                        self.model.addConstr( flow_var <= (nc_y-1)*(var_r_map[conn_0] + var_r_map[conn_1]) , name=myconstrname )
                
                # root source
                # out - in = nc - 1

                myconstrname = f"constr_f_yline_{x}x{z}z_root"
                self.model.addConstr( gp.quicksum([cube_flow_vars[0, b] for b in range(nc_y)]) - gp.quicksum([cube_flow_vars[b, 0] for b in range(nc_y)]) == nc_y - 1, name=myconstrname )

                # others
                # in - out = 1
                for b in range(1,nc_y):
                    myconstrname = f"constr_f_xline_{x}x{z}z_cube{b}"
                    self.model.addConstr( gp.quicksum([cube_flow_vars[a, b] for a in range(nc_y)]) - gp.quicksum([cube_flow_vars[b, a] for a in range(nc_y)]) == 1, name=myconstrname )

        # z line
        for x in range(x_dim):
            for y in range(y_dim):
                
                nc_z = z_dim // cube_dim
                if nc_z == 1:
                    continue

                nodes_neg_side = []
                nodes_pos_side = []
                for z in range(0,nc_z*cube_dim,cube_dim):
                    nodes_neg_side.append(self.xyz_to_r(x, y, z))
                    nodes_pos_side.append(self.xyz_to_r(x, y, z+(cube_dim-1)))

                # print(f"z line is nodes: {nodes_neg_side} + {nodes_pos_side}")


                cube_flow_vars = {}
                
                for a in range(nc_z):
                    cube_flow_vars[ (a,a) ] = 0
                    for b in range(nc_z):
                        if a==b:
                            continue

                        # create flow var
                        myvarname = f"c_f_zline_{x}x{y}y_{a}a_{b}b"
                        flow_var = self.add_var_continuous_(0, nc_z-1, myvarname)
                        cube_flow_vars[ (a,b) ] = flow_var

                        # constr flow var

                        
                        # these are different and not symmetric
                        conn_0 = ( nodes_neg_side[a], nodes_pos_side[b] )
                        conn_1 = ( nodes_pos_side[a], nodes_neg_side[b] )

                        myconstrname = f"constr_f_zline_{x}x{y}y_{a}a_{b}b"
                        self.model.addConstr( flow_var <= (nc_z-1)*(var_r_map[conn_0] + var_r_map[conn_1]) , name=myconstrname )
                
                        # print(f"flow var {myvarname} <= (nc_z-1)*(r_map[{conn_0}]+r_map[{conn_1}]) = {(nc_z-1)}*{var_r_map[conn_0]} + { var_r_map[conn_1]}")

                # root source
                # out - in = nc - 1

                myconstrname = f"constr_f_zline_{x}x{y}y_root"
                self.model.addConstr( gp.quicksum([cube_flow_vars[0, b] for b in range(nc_z)]) - gp.quicksum([cube_flow_vars[b, 0] for b in range(nc_z)]) == nc_z - 1, name=myconstrname )

                # others
                # in - out = 1
                for b in range(1,nc_z):
                    myconstrname = f"constr_f_zline_{x}x{y}y_cube{b}"
                    self.model.addConstr( gp.quicksum([cube_flow_vars[a, b] for a in range(nc_z)]) - gp.quicksum([cube_flow_vars[b, a] for a in range(nc_z)]) == 1, name=myconstrname )

    def constr_dor_conns_old_(self, integer_constraints=True):
        assert(self.n_cubes)
        assert(self.xyzc_dims)
        assert(self.var_r_map)

        # ui − uj ​ + N*m_i_j <= N−1
        # mij = 1 => ui - uj + N <= N-1   =>   ui - uj <= -1 so uj higher than ui
        # mij = 0 => ui - uj <= N - 1   =>   ui - uj <= N - 1  so unconstrainted
        # mij = 0.5 => ui - uj + 0.5N <= N-1   =>   ui - uj <= 0.5N - 1 so semi constained?


        (x_dim, y_dim, z_dim, cube_dim) = self.xyzc_dims
        n_cubes = self.n_cubes
        var_r_map = self.var_r_map

        # x line
        for y in range(y_dim):
            for z in range(z_dim):
                
                nc_x = x_dim // cube_dim
                if nc_x == 1:
                    continue

                nodes_neg_side = []
                nodes_pos_side = []
                for x in range(0,nc_x*cube_dim,cube_dim):
                    nodes_neg_side.append(self.xyz_to_r(x, y, z))
                    nodes_pos_side.append(self.xyz_to_r(x+(cube_dim-1), y, z))

                var_cube_order = {}
                for n_i in nodes_neg_side:
                    i_cube = self.which_cube(n_i)
                    myvarname = f"var_dor_xline_{y}y{z}z_{i_cube}cid"
                    if integer_constraints:
                        var = self.add_var_integer_(0,nc_x-1,myvarname)
                    else:
                        var = self.add_var_continuous_(0,nc_x-1,myvarname)
                    var_cube_order[i_cube] = var

                N = nc_x
                root_cube = nodes_neg_side[0]
                i_root = self.which_cube(root_cube)
                for n_i in nodes_neg_side:
                    for n_j in nodes_pos_side:


                        i_cube = self.which_cube(n_i)
                        j_cube = self.which_cube(n_j)

                        # if i_cube == i_root or j_cube == i_root:
                        #     continue
                        if j_cube == i_root:
                            continue

                        # TODO generic
                        myconstrname = f"constr_dor_xline_{y}y{z}z_{n_i}i_{n_j}j"
                        self.model.addConstr(var_cube_order[i_cube]-var_cube_order[j_cube] + N*var_r_map[n_i,n_j] <= N - 1, name=myconstrname)

        # y line
        for x in range(x_dim):
            for z in range(z_dim):
                
                nc_y = y_dim // cube_dim
                if nc_y == 1:
                    continue

                nodes_neg_side = []
                nodes_pos_side = []
                for y in range(0,nc_y*cube_dim,cube_dim):
                    nodes_neg_side.append(self.xyz_to_r(x, y, z))
                    nodes_pos_side.append(self.xyz_to_r(x, y+(cube_dim-1), z))

                var_cube_order = {}
                for n_i in nodes_neg_side:
                    i_cube = self.which_cube(n_i)
                    myvarname = f"var_dor_yline_{x}x{z}z_{i_cube}cid"
                    if integer_constraints:
                        var = self.add_var_integer_(0,nc_y-1,myvarname)
                    else:
                        var = self.add_var_continuous_(0,nc_y-1,myvarname)
                    var_cube_order[i_cube] = var

                N = nc_y
                root_cube = nodes_neg_side[0]
                i_root = self.which_cube(root_cube)
                for n_i in nodes_neg_side:
                    for n_j in nodes_pos_side:

                        i_cube = self.which_cube(n_i)
                        j_cube = self.which_cube(n_j)

                        # if i_cube == i_root or j_cube == i_root:
                        #     continue
                        if j_cube == i_root:
                            continue

                        # TODO generic
                        myconstrname = f"constr_dor_yline_{x}x{z}z_{n_i}i_{n_j}j"
                        self.model.addConstr(var_cube_order[i_cube]-var_cube_order[j_cube] + N*var_r_map[n_i,n_j] <= N - 1, name=myconstrname)

        # z line
        for x in range(x_dim):
            for y in range(y_dim):
                
                nc_z = z_dim // cube_dim
                if nc_z == 1:
                    continue

                nodes_neg_side = []
                nodes_pos_side = []
                for z in range(0,nc_z*cube_dim,cube_dim):
                    nodes_neg_side.append(self.xyz_to_r(x, y, z))
                    nodes_pos_side.append(self.xyz_to_r(x, y, z+(cube_dim-1)))

                var_cube_order = {}
                for n_i in nodes_neg_side:
                    i_cube = self.which_cube(n_i)
                    myvarname = f"var_dor_zline_{x}x{y}y_{i_cube}cid"
                    # print(f"creating {myvarname}")
                    if integer_constraints:
                        var = self.add_var_integer_(0,nc_z-1,myvarname)
                    else:
                        var = self.add_var_continuous_(0,nc_z-1,myvarname)
                    var_cube_order[i_cube] = var

                N = nc_z
                root_cube = nodes_neg_side[0]
                i_root = self.which_cube(root_cube)
                for n_i in nodes_neg_side:
                    for n_j in nodes_pos_side:

                        i_cube = self.which_cube(n_i)
                        j_cube = self.which_cube(n_j)

                        # if i_cube == i_root or j_cube == i_root:
                        #     continue
                        if j_cube == i_root:
                            continue

                        # TODO generic
                        myconstrname = f"constr_dor_zline_{x}x{y}y_{n_i}i_{n_j}j"
                        self.model.addConstr(var_cube_order[i_cube]-var_cube_order[j_cube] + N*var_r_map[n_i,n_j] <= N - 1, name=myconstrname)

    def constr_A_transpose_(self):
        assert(self.var_unity_wye)
        assert(self.var_tri_ineq_wyes)
        assert(self.var_r_map)
        assert(self.valid_conns)
        assert(self.n_routers)
        assert(self.scale_factor)
        assert(self.map_sym_breaker is not None)
        assert(self.uwye_sym_breaker is not None)

        self.model.update()

        var_unity_wye = self.var_unity_wye
        var_tri_ineq_wyes = self.var_tri_ineq_wyes
        var_r_map = self.var_r_map
        valid_conns = self.valid_conns
        n_routers = self.n_routers
        scale_factor = self.scale_factor
        map_sym_breaker = self.map_sym_breaker
        uwye_sym_breaker = self.uwye_sym_breaker

        if map_sym_breaker > 0:
            rand_mult = random.uniform(1-map_sym_breaker, 1+map_sym_breaker)
            scale_factor *= rand_mult

        uwye_mult = 1
        if uwye_sym_breaker > 0:
            uwye_mult = random.uniform(1-uwye_sym_breaker, 1+uwye_sym_breaker)

        routers_of_canonical_cube = self.get_canonical_nodes()

        considered = set()
        for a in routers_of_canonical_cube:
            for b in range(n_routers):
                if a==b:
                    continue

                considered.add((a,b))

                # y0 + sum_{j ...}(tri_ineq_wyes[a,j,b]) + sum_{i ...}(tri_ineq_wyes[i,a',b']) - sum_{k ...}(tri_ineq_wyes[a,b,k]) - r_map[a,b] <= 0

                pos_lhs_vars = [uwye_mult*var_unity_wye]
                neg_lhs_vars = [scale_factor*var_r_map[a,b]]

                # i->k
                # ----
                pos_lhs_vars += [var_tri_ineq_wyes[a,j,b ] for j in range(n_routers) if j!=a and j!=b and self.relevant_ijk(a,j,b)]

                # k->j
                # ----
                for a_outside in self.get_all_noncanonical_equivalents(a):

                    # a -> a_outside
                    a_to_a_outside_tform = self.calc_transform_delta(a,a_outside)
                    b_outside = self.apply_transformation(b,a_to_a_outside_tform)

                    for i in routers_of_canonical_cube:
                        if i==a_outside or i==b_outside:
                            continue
                        
                        if not self.relevant_ijk(i,b_outside,a_outside):
                            continue

                        pos_lhs_vars.append(var_tri_ineq_wyes[i,b_outside,a_outside ])

                # i->j
                # ----
                neg_lhs_vars += [var_tri_ineq_wyes[a,b,k] for k in range(n_routers) if k!=a and k!=b and self.relevant_ijk(a,b,k)]

                myconstrname = self._name_constr_A_transpose(a,b)
                self.add_constr_pos_neg_sum_lte_(pos_lhs_vars, neg_lhs_vars, 0, myconstrname)

                # if a==0 and b==1:
                #     input(f"pos_lhs_vars = {pos_lhs_vars}, neg_lhs_vars={neg_lhs_vars}")

        return 
        for a in range(n_routers):
            for b in routers_of_canonical_cube:
                if a==b:
                    continue

                if (a,b) in considered:
                    continue

                # y0 + sum_{j ...}(tri_ineq_wyes[a,j,b]) + sum_{i ...}(tri_ineq_wyes[i,a',b']) - sum_{k ...}(tri_ineq_wyes[a,b,k]) - r_map[a,b] <= 0

                pos_lhs_vars = [uwye_mult*var_unity_wye]
                neg_lhs_vars = [scale_factor*var_r_map[a,b]]

                # i->k
                # ----
                pos_lhs_vars += [var_tri_ineq_wyes[a,j,b ] for j in range(n_routers) if j!=a and j!=b and self.relevant_ijk(a,j,b)]

                # k->j
                # ----
                for a_outside in self.get_all_noncanonical_equivalents(a):

                    # a -> a_outside
                    a_to_a_outside_tform = self.calc_transform_delta(a,a_outside)
                    b_outside = self.apply_transformation(b,a_to_a_outside_tform)

                    for i in routers_of_canonical_cube:
                        if i==a_outside or i==b_outside:
                            continue
                        
                        if not self.relevant_ijk(i,b_outside,a_outside):
                            continue

                        pos_lhs_vars.append(var_tri_ineq_wyes[i,b_outside,a_outside ])

                # i->j
                # ----
                neg_lhs_vars += [var_tri_ineq_wyes[a,b,k] for k in range(n_routers) if k!=a and k!=b and self.relevant_ijk(a,b,k)]

                myconstrname = self._name_constr_A_transpose(a,b)
                self.add_constr_pos_neg_sum_lte_(pos_lhs_vars, neg_lhs_vars, 0, myconstrname)

                # if a==0 and b==1:
                #     input(f"pos_lhs_vars = {pos_lhs_vars}, neg_lhs_vars={neg_lhs_vars}")

    def constr_wye_tri_ineqs(self):
        n_routers = self.n_routers

        var_r_map = self.var_r_map
        var_tri_ineq_wyes = self.var_tri_ineq_wyes

        routers_of_canonical_cube = self.get_canonical_nodes()

        for a in routers_of_canonical_cube:
            for b in range(n_routers):
                if a==b:
                    continue
                for j in range(n_routers):
                    if a==j or b==j:
                        continue
                        
                    if not self.relevant_ijk(a,j,b):
                        continue
                    self.model.addConstr(var_tri_ineq_wyes[a,j,b] <= var_r_map[a,b])

    # AASD variables
    # --------------

    def update_lbub_r_map_(self, conn, lb, ub):
        if self.model is None:
            return
        assert(self.model)
        assert(self.var_r_map)
        assert(self.solver_library)
        solver_library = self.solver_library

        if isinstance(self.var_r_map[conn], (int, float)):
            return

        (i,j) = conn
        canonical_nodes = self.get_canonical_nodes()
        (min_idx, max_idx) = (min(i,j), max(i,j))
        if min_idx not in canonical_nodes:
            return

        logger.info(f'Setting r_map[{conn}] = [{lb},{ub}]')

        if solver_library== 'highs':
            conn_idx = self.var_r_map[i,j]
            self.model.changeColBounds(conn_idx,lb,ub)
        elif solver_library == 'gurobi':
            self.var_r_map[i,j].LB = lb
            self.var_r_map[i,j].UB = ub
        elif solver_library == 'ortools' or solver_library == 'ortools_model_builder':
            self.var_r_map[i,j].lower_bound = lb
            self.var_r_map[i,j].upper_bound = ub
        else:
            logger.critical('Unimplemented update_lbub_r_map. Exiting...')
            quit()    


    def update_multi_lbub_r_map_(self, conns, lb, ub):
        """
        For some libraries there are much faster APIs for batch LB/UB setting
        """
        if self.model is None:
            return
        assert(self.model)
        assert(self.var_r_map)
        assert(self.solver_library)
        solver_library = self.solver_library

        conns = [c for c in conns if not isinstance(self.var_r_map[c], (int, float))]

        if solver_library == 'highs':
            # batch API
            conn_idxs = []
            for (i,j) in conns:
                conn_idxs.append(self.var_r_map[i,j])
            n_to_change = len(conn_idxs)
            # one Python->C++ call
            self.model.changeColsBounds(n_to_change,conn_idxs,[lb]*n_to_change,[ub]*n_to_change)
        elif solver_library == 'gurobi':
            # Gurobi does lazy update so no need to call once
            for conn in conns:
                self.update_lbub_r_map_(conn, lb, ub)
        elif solver_library == 'ortools' or solver_library == 'ortools_model_builder':
            # no batch API
            for conn in conns:
                self.update_lbub_r_map_(conn, lb, ub)
        else:
            logger.critical('ERROR: Unimplemented update_lbub_r_map. Exiting...')
            quit()  

        logger.info(f'Setting r_map[{conns}] = [{lb},{ub}]')

    def update_lbub_tri_ineq_wyes_(self, ijk_tuple, lb, ub):
        if self.model is None:
            return
        assert(self.model)
        assert(self.var_tri_ineq_wyes)
        assert(self.solver_library)
        solver_library = self.solver_library

        if isinstance(self.var_tri_ineq_wyes[ijk_tuple], (int, float)):
            return

        # logger.info(f'Setting tri_ineq_wyes[{ijk_tuple}] = [{lb},{ub}]')

        if solver_library== 'highs':
            tri_ineq_idx = self.var_tri_ineq_wyes[ijk_tuple]
            self.model.changeColBounds(tri_ineq_idx,lb,ub)
        elif solver_library == 'gurobi':
            self.var_tri_ineq_wyes[ijk_tuple].LB = lb
            self.var_tri_ineq_wyes[ijk_tuple].UB = ub
        elif solver_library == 'ortools' or solver_library == 'ortools_model_builder':
            self.var_tri_ineq_wyes[ijk_tuple].lower_bound = lb
            self.var_tri_ineq_wyes[ijk_tuple].upper_bound = ub
        else:
            logger.critical('Unimplemented update_lbub_r_map. Exiting...')
            quit()    

    def update_multi_lbub_tri_ineq_wyes_(self, ijk_tuples, lb, ub):
        """
        For some libraries there are much faster APIs for batch LB/UB setting
        """
        if self.model is None:
            return
        assert(self.model)
        assert(self.var_tri_ineq_wyes)
        assert(self.solver_library)
        solver_library = self.solver_library

        ijk_tuples = [t for t in ijk_tuples if t in self.var_tri_ineq_wyes]


        if solver_library == 'highs':
            # batch API
            conn_idxs = []
            for ijk_tuple in ijk_tuples:
                conn_idxs.append(self.var_r_map[ijk_tuple])
            n_to_change = len(conn_idxs)
            # one Python->C++ call
            self.model.changeColsBounds(n_to_change,conn_idxs,[lb]*n_to_change,[ub]*n_to_change)
        elif solver_library == 'gurobi':
            # Gurobi does lazy update so no need to call once
            for ijk_tuple in ijk_tuples:
                self.update_lbub_tri_ineq_wyes_(ijk_tuple, lb, ub)
        elif solver_library == 'ortools' or solver_library == 'ortools_model_builder':
            # no batch API
            for ijk_tuple in ijk_tuples:
                self.update_lbub_tri_ineq_wyes_(conn, lb, ub)
        else:
            logger.critical('ERROR: Unimplemented update_lbub_r_map. Exiting...')
            quit()  

        # logger.info(f'Setting tri_ineq_wyes[{ijk_tuples}] = [{lb},{ub}]')

    def add_var_unity_wye_(self, unity_wye_ub):
        assert(self.var_unity_wye is None)
        assert(unity_wye_ub > 0)

        myvarname = self._name_unity_wye()
        self.var_unity_wye = self.add_var_continuous_(0, unity_wye_ub, myvarname)

    def add_var_r_map_(self):
        assert(self.binary_r_map is not None)
        assert(self.scale_factor is not None)
        assert(self.electrical_conns_adj_list)
        assert(self.valid_conns)
        assert(self.chosen_optical_conns is not None)
        assert(self.n_routers)
        assert(self.var_r_map is None)
        scale_factor = self.scale_factor
        binary_r_map = self.binary_r_map
        electrical_conns_adj_list = self.electrical_conns_adj_list
        valid_conns = self.valid_conns
        chosen_optical_conns = self.chosen_optical_conns
        n_routers = self.n_routers

        routers_of_canonical_cube = self.get_canonical_nodes()

        var_r_map = UpperTriMatrix(n_routers)

        # considered = set()
        for i in routers_of_canonical_cube:
            for j in range(n_routers):
                if i==j:
                    continue

                # considered.add( (i,j) )

                if var_r_map[i,j] is not None:
                    continue

                if j in electrical_conns_adj_list[i]:
                    var_r_map[i,j] = 1 #scale_factor
                    logger.debug(f'Electrical {i}->{j}')
                    continue
                if valid_conns[i][j] == 0:
                    var_r_map[i,j] = 0
                    logger.debug(f'Invalid {i}->{j}')
                    continue
                if (i,j) in chosen_optical_conns or (j,i) in chosen_optical_conns:
                    var_r_map[i,j] = 1 #scale_factor
                    logger.debug(f'Known {i}->{j}')
                    continue

                myvarname = self._name_r_map(i,j)
                if binary_r_map:# and scale_factor == 1:
                    var_r_map[i,j] = self.add_var_binary_(myvarname)
                # elif binary_r_map:
                #     var_r_map[i,j] = self.add_var_integer_(0.0,scale_factor,myvarname)
                else:
                    # var_r_map[i,j] = self.add_var_continuous_(0.0,scale_factor,myvarname)
                    var_r_map[i,j] = self.add_var_continuous_(0.0,1.0,myvarname)


        # for i in routers_of_canonical_cube:
        #     for j in range(n_routers):
        #         if i==j:
        #             continue

        #         if (i,j) in considered:
        #             continue

        #         if var_r_map[i,j] is not None:
        #             continue

        #         if j in electrical_conns_adj_list[i]:
        #             var_r_map[i,j] = 1 #scale_factor
        #             logger.debug(f'Electrical {i}->{j}')
        #             continue
        #         if valid_conns[i][j] == 0:
        #             var_r_map[i,j] = 0
        #             logger.debug(f'Invalid {i}->{j}')
        #             continue
        #         if (i,j) in chosen_optical_conns or (j,i) in chosen_optical_conns:
        #             var_r_map[i,j] = 1 #scale_factor
        #             logger.debug(f'Known {i}->{j}')
        #             continue

        #         myvarname = self._name_r_map(i,j)
        #         if binary_r_map:# and scale_factor == 1:
        #             var_r_map[i,j] = self.add_var_binary_(myvarname)
        #         # elif binary_r_map:
        #         #     var_r_map[i,j] = self.add_var_integer_(0.0,scale_factor,myvarname)
        #         else:
        #             # var_r_map[i,j] = self.add_var_continuous_(0.0,scale_factor,myvarname)
        #             var_r_map[i,j] = self.add_var_continuous_(0.0,1.0,myvarname)

        self.var_r_map = var_r_map

    def add_var_tri_ineq_wyes_(self, tri_ineq_wye_lb, tri_ineq_wye_ub):
        assert(self.n_routers)
        assert(self.var_tri_ineq_wyes is None)
        n_routers = self.n_routers

        routers_of_canonical_cube = self.get_canonical_nodes()

        var_tri_ineq_wyes = SymTriDict()
        for i in routers_of_canonical_cube:
                for j in range(n_routers):
                    if i==j:
                        continue
                    for k in range(n_routers):
                        if i==k or j==k:
                            continue

                        if not self.relevant_ijk(i,j,k):
                            continue

                        myvarname = self._name_tri_ineq_wye(i,j,k)
                        var_tri_ineq_wyes[i,j,k] = self.add_var_continuous_(tri_ineq_wye_lb,tri_ineq_wye_ub,myvarname)

        self.var_tri_ineq_wyes = var_tri_ineq_wyes

    # AASD objective(s)
    # -----------------

    def create_objective_(self):
        assert(self.model)
        assert(self.solver_library)
        assert(self.var_unity_wye)
        assert(self.n_routers)
        solver_library = self.solver_library
        var_unity_wye = self.var_unity_wye
        var_r_map = self.var_r_map
        n_routers = self.n_routers

        # can later make into function
        if solver_library == 'highs':
            logger.critical("HiGHS objective unimplemented")
            quit()
        elif solver_library == 'gurobi':
            wye_obj_expr = gp.LinExpr()
            wye_obj_expr += var_unity_wye

            self.model.setObjective(wye_obj_expr, GRB.MAXIMIZE)

        elif solver_library == 'ortools' or solver_library == 'ortools_model_builder':
            self.model.maximize(var_unity_wye)

    def update_objective_for_penalized_r_map_(self, r_map_penalty_coeffs):
        assert(self.model)
        assert(self.solver_library)
        assert(self.var_unity_wye)
        assert(self.n_routers)
        solver_library = self.solver_library
        var_unity_wye = self.var_unity_wye
        var_r_map = self.var_r_map
        n_routers = self.n_routers
        chosen_optical_conns = self.chosen_optical_conns
        valid_conns = self.valid_conns

        # can later make into function
        if solver_library == 'highs':
            logger.critical("HiGHS objective unimplemented")
            quit()
        elif solver_library == 'gurobi':
            wye_obj_expr = gp.LinExpr()
            wye_obj_expr += var_unity_wye

            # penalties
            for (i,j) in self.calc_opt_conn_options():
                wye_obj_expr += r_map_penalty_coeffs[i,j]*var_r_map[i,j]

            # for i in range(n_routers):
            #     for j in range(n_routers):
            #         wye_obj_expr += r_map_penalty_coeffs[i,j]*var_r_map[i,j]

            self.model.setObjective(wye_obj_expr, GRB.MAXIMIZE)

            # self.model.Params.Cutoff = float('-inf')

        elif solver_library == 'ortools' or solver_library == 'ortools_model_builder':
            logger.critical("HiGHS objective unimplemented")
            quit()

    def write_unsolved_model(self):
        assert(self.solver_library)
        assert(self.model)

        model = self.model

        name = f"{self.model_base_name}.lp"
        model_dir = self.model_dir

        out_model_path = os.path.join(model_dir,name)

        solver_library = self.solver_library
        if solver_library == 'gurobi':
            model.write(out_model_path)
        else:
            logger.critical("write_unsolved_model Unimplemented")
            quit()

        logger.info(f"Wrote unsolved model out to {out_model_path}")

    # create model
    # ------------
    def create_model(self):
        assert(self.scale_factor)
        assert(self.n_routers)
        scale_factor = self.scale_factor
        n_routers = self.n_routers

        logger.info("Creating model")

        start_time = time.time()

        self.create_model_()

        # is this best place?
        unity_wye_ub = scale_factor #   (2*6*scale_factor)/n_routers
        tri_ineq_wye_ub = unity_wye_ub
        tri_ineq_wye_lb = 0 #-1.0

        self.add_var_unity_wye_(unity_wye_ub)
        self.add_var_r_map_()
        self.add_var_tri_ineq_wyes_(tri_ineq_wye_lb, tri_ineq_wye_ub)

        self.constr_A_transpose_()
        self.constr_limit_xyz_conns_()

        if False :#True:
            self.constr_wye_tri_ineqs()

        # TODO this is turned off. GOOD? ACCEPTABLE??? IT SLOWS THINGS DOWN SO MUCH
        if self.dorable and not self.symmetric:
            # pass
            # self.constr_dor_conns_(integer_constraints=False)
            self.constr_dor_conns_()

        self.create_objective_()

        logger.info("Created model")
        elapsed_time = time.time() - start_time
        logger.performance(f'{round(elapsed_time,1)}s for creating model')

        if self.write_model:
            start_time = time.time()
            self.model.update()
            self.write_unsolved_model()
            elapsed_time = time.time() - start_time
            logger.performance(f'{round(elapsed_time,1)}s for writing model')


        logger.performance(f'{self.print_rss()} memory used after creating model')

        # setup callbacks for milp
        if self.binary_r_map:
            out_name = f"{self.model_base_name}_runsol.map"
            run_sol_filename = os.path.join(self.running_topology_dir,out_name)

            callback_data = CallbackData(self.model.getVars())
            # callback_func = partial(run_sol_callback, cbdata=callback_data,logfile=log_file)
            # print(f'callback_func ({type(callback_func)})={callback_func}')

            # user defined params
            # for the runsol callback
            self.model._rs_cbdata = callback_data
            self.model._rs_n_routers = self.n_routers
            self.model._rs_run_sol_name = run_sol_filename
            self.model._rs_tpuv4_symmetry = self.tpuv4_symmetry
            self.model._rs_var_r_map = self.var_r_map



    # Solving model
    ################################################################################

    def solve_model_(self, initial_solve=True):
        assert(self.model)
        assert(self.solver_library)
        assert(self.solver_params)
        model = self.model
        solver_library = self.solver_library
        solver_params = self.solver_params

        start_time = time.time()

        if solver_library == 'highs':
            self.model = self.solve_highs(model, solver_params)
        elif solver_library == 'gurobi':
            self.model = self.solve_gurobi(model, solver_params)
        elif solver_library == 'ortools':
            self.model, self.solver_result = solve_ortools(model, solver_params)
        elif solver_library == 'ortools_model_builder':
            self.model, self.solver_result = solve_ortools_model_builder(model, solver_params)

        elapsed_time = time.time() - start_time
        logger.performance(f'{round(elapsed_time,1)}s for solving model')

    def solve_highs(self, model, solver_params):

        # CLAs
        for k,v in solver_params.items():
            try:
                model.setOptionValue(k,v)
                logger.info(f'Set option {k} to value {v}','GIVENS',n_indents=1)
            except:
                pass

        # always
        model.setOptionValue("parallel", "on")
        model.setOptionValue("run_crossover", "off")

        model.maximize()

        return model

    def solve_gurobi(self, model, solver_params): 

        # TODO testing commenting all out?
        # defaults
        model.setParam("Crossover", 0)
        model.setParam("Method", 2)
        #model.setParam("OutputFlag", 1)
        model.setParam("LogFile", "gurobi.log")

        # faster
        model.setParam("BarOrder",0)

        # CLAs
        for k,v in solver_params.items():
            try:
                model.setParam(k,v)
                logger.info(f'Set option {k} to value {v}','GIVENS',n_indents=1)
            except:
                pass

        # this is ugly fix later
        try:
            _ = solver_params["iteration_number"]
        except:
            solver_params["iteration_number"] = 0

            try:
                model.setParam("Method",solver_params["MethodFirstRound"])
            except:
                pass

        model.ModelSense = gp.GRB.MAXIMIZE

        if self.binary_r_map:
            model.optimize(run_sol_callback)
        else:
            model.optimize()

        solver_params["iteration_number"] += 1

        if (model.status == GRB.INFEASIBLE or model.status == GRB.UNBOUNDED): 
            print(f"Gurobi model is {model.status}.") 

            model.computeIIS()
            model.write("model.ilp")  # Write model with IIS information
            model.write("iis.ilp")
            print("IIS details written to iis.ilp")

        return model # presolved_model #model


    def solve_ortools(self, model, solver_params):

        solver = model_builder.ModelSolver("PDLP")
        # solver.enable_output(True)
        solver = mathopt.SolverType.PDLP

        params = mathopt.SolveParameters(
            enable_output=True
        )

        result = mathopt_solve.solve(model, solver, params=params)

        return model, result


    def solve_ortools_model_builder(self, model, solver_params):

        solver = model_builder.ModelSolver("PDLP")
        solver.enable_output(True)

        result = solver.solve(model)

        return model, solver

    # Scoring
    ################################################################################

    # for debugging
    def get_A_transpose_row_vals(self, a, b, y0_val, r_map_vals, tri_ineq_wye_vals, print_too=False):

        # y0_val = self.var_unity_wye.X
        if print_too:
            print(f"y0 = {y0_val}")
        r_map_val = r_map_vals[a,b]
        # try:
        #     r_map_val = self.var_r_map[a,b].X
        # except:
        #     r_map_val = self.var_r_map[a,b]
        if print_too:
            print(f"r_map[{a,b}] = {r_map_val}")

        ij_tri_ineq_wye_vals = SymTriDict()
        i = a
        j = b
        for k in range(self.n_routers):
            if (i,j,k) in self.var_tri_ineq_wyes:
                # val = self.var_tri_ineq_wyes[(i,j,k)].X
                val = tri_ineq_wye_vals[(i,j,k)]
                ij_tri_ineq_wye_vals[(i,j,k)] = val
                if print_too:
                    print(f"ij tri ineq {(i,j,k)} = {val}")

        ik_tri_ineq_wye_vals = SymTriDict()
        i = a
        k = b
        for j in range(self.n_routers):
            if (i,j,k) in self.var_tri_ineq_wyes:
                # val = self.var_tri_ineq_wyes[(i,j,k)].X
                val = tri_ineq_wye_vals[(i,j,k)]
                ik_tri_ineq_wye_vals[(i,j,k)] = val
                if print_too:
                    print(f"ik tri ineq {(i,j,k)} = {val}")

        kj_tri_ineq_wye_vals = SymTriDict()
        k = a
        j = b
        for i in range(self.n_routers):
            if (i,j,k) in self.var_tri_ineq_wyes:
                # val = self.var_tri_ineq_wyes[(i,j,k)].X
                val = tri_ineq_wye_vals[(i,j,k)]
                kj_tri_ineq_wye_vals[(i,j,k)] = val
                if print_too:
                    print(f"kj tri ineq {(i,j,k)} = {val}")

        total_sum = y0_val - r_map_val
        ij_sum = 0
        for k,v in ij_tri_ineq_wye_vals.items():
            ij_sum -= v
        ik_sum = 0
        for k,v in ik_tri_ineq_wye_vals.items():
            ik_sum += v
        kj_sum = 0
        for k,v in kj_tri_ineq_wye_vals.items():
            kj_sum += v
        
        total_sum = total_sum + ij_sum + ik_sum + kj_sum
        cost_sum = r_map_val - (ij_sum + ik_sum + kj_sum)

        if print_too:
            print(f"in total...")
            print(f"{y0_val} - {r_map_val} - {ij_sum} + {ik_sum} + {kj_sum}")

        return (total_sum, cost_sum, y0_val, r_map_val, ij_tri_ineq_wye_vals, ik_tri_ineq_wye_vals, kj_tri_ineq_wye_vals)

    def get_obj_val(self):
        # actually gets y0 (differs if penalties in obj)
        assert(self.solver_library)
        assert(self.model)
        assert(self.var_unity_wye)
        solver_library = self.solver_library

        unity_wye_val = 0
        obj_val = 0

        if solver_library == 'gurobi':
            obj_val = self.model.objVal
            unity_wye_val = self.var_unity_wye.X
        # elif solver_library == 'highs':
        #     return self.model.getObjectiveValue()
        # elif solver_library == 'ortools':
        #     return self.solver_or_result.objective_value()
        # elif solver_library == 'ortools_model_builder':
        #     return self.solver_or_result.objective_value
        else:
            logger.critical(f'Unimplemented get_obj_val: Exiting...')
            quit()

        return (obj_val, unity_wye_val)

    def set_start_map(self, start_map_path):
        assert(self.var_r_map)
        assert(self.valid_conns)
        valid_conns = self.valid_conns

        start_map = self.ingest_an_adj_mat(start_map_path)

        for i, row in enumerate(start_map):
            for j, conn in enumerate(row):
                if valid_conns[i][j] == 0:
                    continue

                try:
                    self.var_r_map[i,j].Start = conn
                except:
                    pass

    def select_best_score_opt_conn(self, conn_scores):
        assert(self.n_routers)
        assert(self.valid_conns)
        assert(self.electrical_conns_adj_list)
        assert(self.chosen_optical_conns is not None)
        assert(self.spreading_cubes is not None)
        assert(self.chosen_cube_conns is not None)
        n_routers = self.n_routers
        valid_conns = self.valid_conns
        electrical_conns_adj_list = self.electrical_conns_adj_list
        chosen_optical_conns = self.chosen_optical_conns
        spreading_cubes = self.spreading_cubes
        connect_first = self.connect_first
        chosen_cube_conns = self.chosen_cube_conns

        # logger.info(f"Selecting max conn. Params : spreading_cubes={spreading_cubes}, connect_first={connect_first}, chosen_cube_conns={chosen_cube_conns}")

        max_val = -1*self.INF
        max_conns = []

        for i in self.get_canonical_nodes():
            for j in range(n_routers):

                if valid_conns[i][j] == 0:
                    continue
                if j in electrical_conns_adj_list[i]:
                    continue
                if (i,j) in chosen_optical_conns:
                    continue

                i_cube = self.which_cube(i)
                j_cube = self.which_cube(j)
                if (spreading_cubes or connect_first) and ((i_cube, j_cube) in chosen_cube_conns):
                    continue
                if connect_first and (i_cube == j_cube):
                    continue

                val = conn_scores[i,j]

                if self.verbose:
                    print(f'r_map sol [{i},{j}] = {val}')

                if val > max_val:
                    max_val = val
                    max_conns = [(i,j)]
                elif val == max_val:
                    max_conns.append((i,j))


        selection_method = "random"
        selection_method = "least_preclusive"
        if selection_method == "first_idx":
            max_conn = max_conns[0]
        elif selection_method == "random":
            max_conn = random.choice(max_conns)
        elif selection_method == "least_preclusive":
            least_num_precluded = self.INF
            for conn in max_conns:
                n_precluded = len(self.precluded_conns_given_conn(conn))
                if n_precluded < least_num_precluded:
                    least_num_precluded = n_precluded
                    max_conn = conn


        return max_conn, max_val

    def calculate_cost_matrix(self, y0_val, r_map_vals, tri_ineq_wye_vals):


        n_routers = self.n_routers

        cost_matrix = [[0 for _ in range(n_routers)] for __ in range(n_routers)]


        for a in range(n_routers):
            for b in range(n_routers):
                if a == b:
                    continue

                (_, cost_sum, __, ___, ____, _____, ______) = self.get_A_transpose_row_vals(a,b,y0_val, r_map_vals, tri_ineq_wye_vals)

                cost_matrix[a][b] = cost_sum

        return cost_matrix


    @classmethod
    def _other_endpoint(cls, edge, u):
        a, b = edge
        if u == a:
            return b
        if u == b:
            return a
        raise ValueError(f"edge {edge} does not touch node {u}")



    # def _solve_dim_matching_and_edge_scores(self, dim, weight_mode):
    #     """
    #     Solves max-weight perfect matching for one dimension dim in {"x","y","z"}.
    #     Returns:
    #     matched_edges: set of (min(u,v), max(u,v))
    #     edge_weight: dict (min(u,v), max(u,v)) -> weight used by matching
    #     """

    #     n_nodes = self.n_routers
    #     r_map_vals = self.translate_r_map_into_values_twod()

    #     left_tag = dim + "-"
    #     right_tag = dim + "+"

    #     deg_left = [len(self.get_group_edges(u, left_tag)) for u in range(n_nodes)]
    #     deg_right = [len(self.get_group_edges(v, right_tag)) for v in range(n_nodes)]

    #     G = nx.Graph()
    #     left_nodes = [("L", u) for u in range(n_nodes)]
    #     right_nodes = [("R", v) for v in range(n_nodes)]
    #     G.add_nodes_from(left_nodes, bipartite=0)
    #     G.add_nodes_from(right_nodes, bipartite=1)

    #     edge_weight = {}

    #     for u in range(n_nodes):
    #         du = deg_left[u] if deg_left[u] else 1
    #         for edge in self.get_group_edges(u, left_tag):
    #             v = self._other_endpoint(edge, u)
    #             dv = deg_right[v] if deg_right[v] else 1

    #             a, b = edge
    #             a2, b2 = (a, b) if a <= b else (b, a)
    #             val = r_map_vals[a2,b2]

    #             if weight_mode == "raw":
    #                 w = val
    #             elif weight_mode == "baseline":
    #                 # Corrects bias when group sizes differ (e.g., z has 2 options, x has 3)
    #                 w = val - (1.0 / du) - (1.0 / dv)
    #             else:
    #                 raise ValueError(f"Unknown weight_mode={weight_mode}")

    #             G.add_edge(("L", u), ("R", v), weight=w, edge=(a2, b2))
    #             edge_weight[(a2, b2)] = w

    #     mwm = nx.algorithms.matching.max_weight_matching(G, maxcardinality=True, weight="weight")

    #     matched_edges = set()
    #     covered_left = set()

    #     for n1, n2 in mwm:
    #         if n1[0] == "L":
    #             left, right = n1, n2
    #         else:
    #             left, right = n2, n1

    #         if left[0] != "L" or right[0] != "R":
    #             continue

    #         u = left[1]
    #         covered_left.add(u)

    #         data = G.get_edge_data(left, right)
    #         if data is None:
    #             continue

    #         matched_edges.add(data["edge"])

    #     if len(covered_left) != n_nodes:
    #         raise RuntimeError(f"Matching did not cover all left nodes for dim={dim}: {len(covered_left)}/{n_nodes}")

    #     return matched_edges, edge_weight


    def matching_score_matrix(
        self,
        dims=("x", "y", "z"),
        weight_mode="baseline",
        nonmatch_value=float("-inf"),
        require_perfect=True,
        verbose=True,
    ):
        """
        Single entry point.

        Returns:
        scores[i][j] = weight for edge (i,j) if selected by the matching in its dimension,
                        else nonmatch_value.

        Key fix vs prior version:
        - We match only over indices u that actually have non-empty groups for dim- / dim+.
        """
        n_nodes = self.n_routers
        r_map_vals = self.translate_r_map_into_values_twod()
        scores = UpperTriMatrix(n_nodes, init_val=nonmatch_value)


        for dim in dims:
            left_tag = dim + "-"
            right_tag = dim + "+"

            # Only include "group indices" that actually exist (non-empty candidate sets)
            left_idx = [u for u in range(n_nodes) if self.get_group_edges(u, left_tag)]
            right_idx = [v for v in range(n_nodes) if self.get_group_edges(v, right_tag)]

            if verbose:
                print(f"[{dim}] left groups: {len(left_idx)}, right groups: {len(right_idx)}")

            if not left_idx or not right_idx:
                if verbose:
                    print(f"[{dim}] skipping (no groups found)")
                continue

            if len(left_idx) != len(right_idx):
                # A perfect matching is impossible if partitions differ in size
                msg = f"[{dim}] partitions not equal: left={len(left_idx)} right={len(right_idx)}"
                if require_perfect:
                    raise RuntimeError(msg)
                if verbose:
                    print(msg + " (continuing with max-cardinality matching)")

            # Degrees for baseline normalization
            deg_left = {u: len(self.get_group_edges(u, left_tag)) for u in left_idx}
            deg_right = {v: len(self.get_group_edges(v, right_tag)) for v in right_idx}

            # Build bipartite graph
            G = nx.Graph()
            L_nodes = [("L", u) for u in left_idx]
            R_nodes = [("R", v) for v in right_idx]
            G.add_nodes_from(L_nodes, bipartite=0)
            G.add_nodes_from(R_nodes, bipartite=1)

            # Add candidate edges from left groups
            for u in left_idx:
                du = deg_left[u] if deg_left[u] else 1
                for edge in self.get_group_edges(u, left_tag):
                    v = self._other_endpoint(edge, u)
                    if v not in deg_right:
                        # Right endpoint doesn't have a right-group (inferred), skip
                        continue

                    dv = deg_right[v] if deg_right[v] else 1

                    a, b = edge
                    a2, b2 = (a, b) if a <= b else (b, a)
                    val = r_map_vals[a2, b2]

                    if weight_mode == "raw":
                        w = val
                    elif weight_mode == "baseline":
                        # removes bias from differing group sizes (e.g. 2-choice vs 3-choice groups)
                        w = val - (1.0 / du) - (1.0 / dv)
                    else:
                        raise ValueError(f"Unknown weight_mode={weight_mode}")

                    G.add_edge(("L", u), ("R", v), weight=w, edge=(a2, b2))

            # Compute maximum-weight matching (max cardinality helps when perfect isn't possible)
            mwm = nx.algorithms.matching.max_weight_matching(G, maxcardinality=True, weight="weight")

            matched_left = set()
            for n1, n2 in mwm:
                if n1[0] == "L":
                    left, right = n1, n2
                else:
                    left, right = n2, n1

                if left[0] != "L" or right[0] != "R":
                    continue

                u = left[1]
                matched_left.add(u)

                data = G.get_edge_data(left, right)
                if not data:
                    continue
                a, b = data["edge"]
                w = data["weight"]

                scores[a,b] = w
                scores[b,a] = w

            # Enforce “perfect on participating groups” if requested
            if require_perfect:
                # Only meaningful if partitions are equal size
                target = min(len(left_idx), len(right_idx))
                if len(matched_left) != target:
                    missing = [u for u in left_idx if u not in matched_left]
                    raise RuntimeError(
                        f"Matching did not cover all participating left groups for dim={dim}: "
                        f"{len(matched_left)}/{target}. Missing left indices (first 20): {missing[:20]}"
                    )

        return scores


    # def matching_score_matrix(
    #     self,
    #     dims=("x", "y", "z"),
    #     weight_mode="baseline",
    #     nonmatch_value=float("-inf"),
    # ):
    #     """
    #     Single entry point.

    #     What it does:
    #     1) For each dim in dims, solves a max-weight perfect matching between (dim-) and (dim+).
    #     2) Builds and returns a 2D matrix 'scores' such that:
    #         - scores[i][j] is the matching weight of edge (i,j) IF (i,j) was selected by the matching
    #         - otherwise scores[i][j] = nonmatch_value
    #         Higher scores are better, so your downstream "pick highest scored connection" works directly.

    #     Notes:
    #     - This returns scores only for edges selected by the matching(s). If you want *all candidate*
    #         edges scored (not only matched edges), say so and I’ll adjust (easy change).
    #     - Assumes every node has a group for dim- and dim+. If only a subset participates, pass that
    #         subset instead of range(n_nodes) (or I can modify to accept an index list).
    #     """
    #     n_routers = self.n_routers
    #     scores = UpperTriMatrix(n_routers, init_val=nonmatch_value)

    #     for dim in dims:
    #         matched_edges, edge_weight = self._solve_dim_matching_and_edge_scores(
    #             dim=dim,
    #             weight_mode=weight_mode,
    #         )

    #         for (a, b) in matched_edges:
    #             w = edge_weight[(a, b)]
    #             scores[a][b] = w
    #             scores[b][a] = w

    #     return scores





    def downstream_effects(self):

        # TODO refactor
        
        valid_conns = self.valid_conns
        n_routers = self.n_routers
        electrical_conns_adj_list = self.electrical_conns_adj_list
        chosen_optical_conns = self.chosen_optical_conns

        start_time = time.time()
        opt_conn_options = []
        for i in range(n_routers):
            for j in range(n_routers):
                if valid_conns[i][j] == 0:
                    continue
                if j in electrical_conns_adj_list[i]:
                    continue
                if (i,j) in chosen_optical_conns:
                    continue
                opt_conn_options.append((i,j))

        elapsed_time = round(time.time() - start_time,2)
        logger.performance(f"downstream_effects scoring : find opt conn options : completed in {elapsed_time}s")

        scores = OffDiagMatrix(n_routers, init_val=-1*self.INF)

        start_time = time.time()

        (obj_val, unity_wye_val) = self.get_obj_val()
        r_map_vals = self.translate_r_map_into_values_twod()
        tri_ineq_wye_vals = self.translate_tri_ineq_wyes_into_values()
        cost_matrix = self.calculate_cost_matrix(unity_wye_val, r_map_vals, tri_ineq_wye_vals)
        elapsed_time = round(time.time() - start_time,2)
        logger.performance(f"downstream_effects scoring : calc r map, tri ineq wye, and cost matrix : completed in {elapsed_time}s")

        start_time = time.time()


        for (a,b) in opt_conn_options:
            (min_i, min_j) = (0,0)
            min_cost_given_ab = self.INF
            # print(f"calculating cost for {(a,b)}")

            cost_matrix_given_ab = deepcopy(cost_matrix) #[[cost_matrix[a][b] for _ in range(n_routers)] for __ in range(n_routers)]

            precluded_conns_given_ab = self.precluded_conns_given_conn((a,b))
            # print(f"precluded_conns_given_ab for {(a,b)} = {precluded_conns_given_ab}")

            for (i,j) in precluded_conns_given_ab:
                cost_matrix_given_ab[i][j] -= r_map_vals[i,j]

                for p in range(n_routers):
                    if p == i or p ==j or i==j:
                        continue
                    if (i,j,p) in tri_ineq_wye_vals:
                        cost_matrix_given_ab[i][j] -= tri_ineq_wye_vals[(i,j,p)]

                for p in range(n_routers):
                    if p == i or p ==j:
                        continue
                    if (i,p,j) in tri_ineq_wye_vals:
                        cost_matrix_given_ab[i][p] += tri_ineq_wye_vals[(i,p,j)]

                for p in range(n_routers):
                    if p == i or p ==j:
                        continue
                    if (p,j,i) in tri_ineq_wye_vals:
                        cost_matrix_given_ab[p][j] += tri_ineq_wye_vals[(p,j,i)]



                if cost_matrix_given_ab[i][j] < min_cost_given_ab:
                    min_cost_given_ab = cost_matrix_given_ab[i][j]
                    (min_i, min_j) = (i,j)

            scores[a,b] = min_cost_given_ab
        
            # print(f"conn {(a,b)} causes min cost {min_cost_given_ab} from downstream effect {(min_i, min_j)}")

        elapsed_time = round(time.time() - start_time,2)
        logger.performance(f"downstream_effects scoring : find min cost abs : completed in {elapsed_time}s")


        return scores


    def downstream_effects_v2(self):

        # TODO refactor
        model = self.model
        valid_conns = self.valid_conns
        n_routers = self.n_routers
        electrical_conns_adj_list = self.electrical_conns_adj_list
        chosen_optical_conns = self.chosen_optical_conns


        opt_conn_options = self.calc_opt_conn_options()

        (obj_val, unity_wye_val) = self.get_obj_val()
        r_map_vals = self.translate_r_map_into_values_twod()
        tri_ineq_wye_vals = self.translate_tri_ineq_wyes_into_values()
        cost_matrix = self.calculate_cost_matrix(unity_wye_val, r_map_vals, tri_ineq_wye_vals)

        # heuristic
        only_relevant_constrs = False #True
        epsilon = 10**(-6)
        relevant_constrs = set()
        pi_matrix = OffDiagMatrix(n_routers, init_val=0)
        for i in range(n_routers):
            for j in range(n_routers):
                if i==j:
                    continue
                ij_At_constr_name = self._name_constr_A_transpose(i,j)
                ij_At_constr = model.getConstrByName(ij_At_constr_name)
                ij_dual_pi = ij_At_constr.Pi
                pi_matrix[i,j] = ij_dual_pi

                if ij_dual_pi > epsilon:
                    relevant_constrs.add((i,j))

        scores = OffDiagMatrix(n_routers, init_val=-1*self.INF)

        for (a,b) in opt_conn_options:
            # print(f"calculating cost for {(a,b)}")

            delta_cost_matrix_given_ab = OffDiagMatrix(n_routers, init_val=0)

            precluded_conns_given_ab = self.precluded_conns_given_conn((a,b))
            # print(f"precluded_conns_given_ab for {(a,b)} = {precluded_conns_given_ab}")

            # cost_{a',b'}=m_a'_b' + sum_{k}(y_a'_b'_k) - sum_{j}(y_a'_j_b') - sum_{i}(y_i_b'_a')
            # therefore cost'_{a',b'}(a,b)=(m_a'_b' if (a'b') excluded) + sum_{k}(y_a'_b'_k if (a',k) excluded) - sum_{j}(y_a'_j_b' if (a'b') excluded) - sum_{i}(y_i_b'_a' if (i,a') excluded)
            # define delta_cost_{a',b'}(a,b) = cost'_{a',b'}(a,b) - cost_{a',b'}=m_a'_b'

            
            # choosing (a,b) => m_a_b = 1
            # therefore the delta is 
            # 1 - m_a_b
            # TODO decide whether to keep this
            # delta_cost_matrix_given_ab[a,b] += (1-r_map_vals[a,b])

            # also iterate over excluded (i,j) and make those changes (notice negative sign in second term of delta expression)

            for (i,j) in precluded_conns_given_ab:

                if (i,j) not in relevant_constrs and only_relevant_constrs:
                    continue

                delta_cost_matrix_given_ab[i,j] += r_map_vals[i,j]

                for p in range(n_routers):
                    if p == i or p ==j or i==j:
                        continue
                    if (i,j,p) in tri_ineq_wye_vals:
                        delta_cost_matrix_given_ab[i,j] += tri_ineq_wye_vals[(i,j,p)]

                for p in range(n_routers):
                    if p == i or p ==j:
                        continue
                    if (i,p,j) in tri_ineq_wye_vals:
                        delta_cost_matrix_given_ab[i,p] -= tri_ineq_wye_vals[(i,p,j)]

                for p in range(n_routers):
                    if p == i or p ==j:
                        continue
                    if (p,j,i) in tri_ineq_wye_vals:
                        delta_cost_matrix_given_ab[p,j] -= tri_ineq_wye_vals[(p,j,i)]

            # at this point, the delta costs for all i,j known

            delta_y0 = 0
            for i in range(n_routers):
                for j in range(n_routers):
                    if i==j:
                        continue

                    if (i,j) not in relevant_constrs and only_relevant_constrs:
                        continue

                    ij_dual_pi = pi_matrix[i,j]
                    delta_y0_by_ij = ij_dual_pi*delta_cost_matrix_given_ab[i,j]
                    delta_y0 += delta_y0_by_ij

            scores[a,b] = delta_y0

        
            # print(f"conn {(a,b)} causes delta y0 {delta_y0}")

        return scores


    def slack_based(self):
        
        valid_conns = self.valid_conns
        n_routers = self.n_routers
        electrical_conns_adj_list = self.electrical_conns_adj_list
        chosen_optical_conns = self.chosen_optical_conns

        opt_conn_options = []
        for i in range(n_routers):
            for j in range(n_routers):
                if valid_conns[i][j] == 0:
                    continue
                if j in electrical_conns_adj_list[i]:
                    continue
                if (i,j) in chosen_optical_conns:
                    continue
                opt_conn_options.append((i,j))

        r_map_vals = self.translate_r_map_into_values_twod()
        tri_ineq_wye_vals = self.translate_tri_ineq_wyes_into_values()
        cost_matrix = self.calculate_cost_matrix(r_map_vals, tri_ineq_wye_vals)
        (obj_val, unity_wye_val) = self.get_obj_val()


        best_slack_val = self.INF
        least_slacks = []
        for (a,b) in opt_conn_options:

            # FLAG TODO determine if this should be swapped
            slack = unity_wye_val - cost_matrix[a][b]
            # slack = cost_matrix[a][b] - unity_wye_val

            # print(f"calculating cost for {(a,b)}: slack = {slack} = {unity_wye_val} - {cost_matrix[a][b]}")


            if slack > best_slack_val:
                best_slack_val = slack
                least_slacks = [(a,b)]
            elif slack == best_slack_val:
                least_slacks.append((a,b))

            # if slack < 0:
            #     least_slacks.append((a,b))

        # input(f"least slack val {best_slack_val} from slacks {least_slacks}")

        # scores = [[0 for _ in range(n_routers)] for __ in range(n_routers)]
        scores = OffDiagMatrix(n_routers, init_val=0)
        # for (a,b) in least_slacks:
        for (a,b) in opt_conn_options:

            # scores[a,b] = r_map_vals[a,b]
            scores[a,b] = r_map_vals[a,b] - cost_matrix[a][b]

        return scores


    def regret_based(self):
        # TODO asserts
        (x_dim, y_dim, z_dim, cube_dim) = self.xyzc_dims
        all_directions = ['x+','y+','z+','x-','y-','z-']
        n_routers = self.n_routers
        chosen_optical_conns = self.chosen_optical_conns

        all_groups = []
        for z in range(z_dim):
            for y in range(y_dim):
                for x in range(x_dim):
                    canon_r = self.xyz_to_r(x,y,z)
                    if not self.r_is_on_face(canon_r):
                        continue

                    groups = {d:set() for d in all_directions}
                    poss_conns = self.poss_optical_conns_for_r(canon_r)
                    for direction, conns in poss_conns.items():

                        if len(conns) == 0:
                            continue
                        for conn in conns:
                            groups[direction].add( (canon_r, conn) )

                    # input(f"groups for {canon_r} : {groups}")
                    all_groups.append(groups)
            #         break
            #     break
            # break
        

        r_map_vals = self.translate_r_map_into_values_twod()

        # order the groups
        best_group_regret = -1
        best_groups = []
        for groups in all_groups:
            for direction in all_directions:
                group = groups[direction]
                if len(group) == 0:
                    continue
                conns = list(group)
                vals = [r_map_vals[c] for c in group]
                # input(f"group {group} w/ vals {vals}")

                pairs = sorted(zip(conns, vals), key=lambda cv: (-cv[1], cv[0]))

                sorted_conns = [c for c, v in pairs]
                sorted_vals  = [v for c, v in pairs]

                # print(f"sorted_conns = {sorted_conns}")
                # print(f"sorted_vals = {sorted_vals}")

                m1 = sorted_vals[0]
                m2 = sorted_vals[1] if len(sorted_vals) > 1 else 0.0
                group_regret = m1 - m2

                print(f"group_regret = {group_regret} for {group}")

                group_has_been_chosen = False
                for conn in group:
                    if conn in chosen_optical_conns:
                        group_has_been_chosen = True
                if group_has_been_chosen:
                    continue

                if group_regret > best_group_regret:
                    best_group_regret = group_regret
                    best_groups = [group]
                elif group_regret == best_group_regret:
                    best_groups.append(group)

        if len(best_groups) == 0:
            input(f"no best groups?")

        scores = OffDiagMatrix(n_routers, init_val=0)
        for group in best_groups:
            for (a,b) in group:
                scores[a,b] = r_map_vals[a,b]
                print(f"{(a,b)} w/ score {r_map_vals[a,b]}")

        one_nonzero = False
        for i in range(n_routers):
            for j in range(n_routers):
                if i == j:
                    continue
                if scores[i,j] > 0:
                    one_nonzero = True

        if not one_nonzero:
            input("cont?")

        return scores

    def normalize_r_map_values(self, r_map_vals, weighting_type="v4"):
        (x_dim, y_dim, z_dim, cube_dim) = self.xyzc_dims
        all_directions = ['x+','y+','z+','x-','y-','z-']
        n_routers = self.n_routers
        valid_conns = self.valid_conns

        adjusted_r_map_vals = deepcopy(r_map_vals)


        for z in range(z_dim):
            for y in range(y_dim):
                for x in range(x_dim):
                    canon_r = self.xyz_to_r(x,y,z)
                    if not self.r_is_on_face(canon_r):
                        continue

                    groups = {d:set() for d in all_directions}
                    poss_conns = self.poss_optical_conns_for_r(canon_r)
                    for direction, conns in poss_conns.items():

                        if len(conns) == 0:
                            continue
                        for conn in conns:
                            groups[direction].add( (canon_r, conn) )

                    for direction in all_directions:
                        group = groups[direction]
                        if len(group) == 0:
                            continue


                        n_options = 0
                        for (i,j) in group:
                            if valid_conns[i][j] > 0:
                                n_options += 1
                        
                        if ( weighting_type == "v3" or weighting_type == "v4") and n_options == 1:
                            continue

                        for (a,b) in group:
                            # print(f"adjusting {(a,b)}")
                            if weighting_type == "v1":
                                adjusted_r_map_vals[a,b] *= n_options
                            elif weighting_type == "v2":
                                adjusted_r_map_vals[a,b] *= (1/n_options)
                            elif weighting_type == "v3":
                                adjusted_r_map_vals[a,b] = (r_map_vals[a,b] - (1/n_options)) / (1 - (1/n_options))
                            elif weighting_type == "v4":
                                adjusted_r_map_vals[a,b] = r_map_vals[a,b] - (1/n_options)

                            # print(f"previously was {r_map_vals[a,b]}, now is {adjusted_r_map_vals[a,b]}")
                        # input(f"cont?")

        return adjusted_r_map_vals

    def prune_to_longest_dims_v2(self, scores, dim_order):
        n_routers = self.n_routers
        valid_conns = self.valid_conns
        (x_dim, y_dim, z_dim, cube_dim) = self.xyzc_dims
        dim_dict = self.dim_dict
        electrical_conns_adj_list = self.electrical_conns_adj_list

        sorted_dim_dict = dict(sorted(dim_dict.items(), key=lambda kv: kv[1], reverse=True))

        # dim_order = ['z','y','x']

        unchosen_dim = None
        for dim in dim_order:

            dim_has_unchosen = False

            for z in range(z_dim):
                for y in range(y_dim):
                    for x in range(x_dim):
                        r = self.xyz_to_r(x,y,z)
                        if not self.r_is_on_face(r):
                            continue

                        poss_conns = self.poss_optical_conns_for_r(r)
                        # print(f"poss_conns = {poss_conns}")

                        dim_pos = f"{dim}+"
                        dim_neg = f"{dim}-"

                        for conn in poss_conns[dim_pos]:
                            if valid_conns[r][conn] == 0:
                                continue
                            # TODO
                            # if not self.conn_is_optical(r,conn):
                            #     continue
                            if (r,conn) not in self.chosen_optical_conns:
                                dim_has_unchosen = True

                        for conn in poss_conns[dim_neg]:
                            if valid_conns[r][conn] == 0:
                                continue
                            # TODO
                            # if not self.conn_is_optical(r,conn):
                            #     continue
                            if (r,conn) not in self.chosen_optical_conns:
                                dim_has_unchosen = True

            if dim_has_unchosen:
                unchosen_dim = dim
                break

        print(f"unchosen dim : {unchosen_dim}")
    
        new_scores = OffDiagMatrix(n_routers, init_val=0)

        for i in range(n_routers):
            for j in range(n_routers):
                if i == j:
                    continue
                if valid_conns[i][j] == 0:
                    continue

                if not self.conn_is_optical(i,j):
                    continue
                ij_conn_dim = self.calc_opt_conn_type(i,j)
                ij_conn_dim = ij_conn_dim[0]

                # allow all
                if unchosen_dim is None:
                    # print(f"allowing conn type {ij_conn_dim}")
                    new_scores[i,j] = scores[i,j]
                    continue

                # if z_dim_unchosen and ij_conn_dim != 'z':
                #     continue
                # elif ij_conn_dim == 'z':
                if ij_conn_dim != unchosen_dim:
                    continue
                
                new_scores[i,j] = scores[i,j]
        return new_scores


    def prune_to_longest_dims(self, scores):
        n_routers = self.n_routers
        valid_conns = self.valid_conns
        (x_dim, y_dim, z_dim, cube_dim) = self.xyzc_dims
        dim_dict = self.dim_dict
        electrical_conns_adj_list = self.electrical_conns_adj_list

        sorted_dim_dict = dict(sorted(dim_dict.items(), key=lambda kv: kv[1], reverse=True))

        unchosen_dim = 'z'
        z_dim_unchosen = False
        # for dim in sorted_dim_dict.keys():

            # dim_has_unchosen = False

        for z in range(z_dim):
            for y in range(y_dim):
                for x in range(x_dim):
                    r = self.xyz_to_r(x,y,z)
                    if not self.r_is_on_face(r):
                        continue

                    poss_conns = self.poss_optical_conns_for_r(r)
                    # print(f"poss_conns = {poss_conns}")

                    # dim_pos = f"{dim}+"
                    # dim_neg = f"{dim}-"
                    dim_pos = f"z+"
                    dim_neg = f"z-"
                    for conn in poss_conns[dim_pos]:
                        if valid_conns[r][conn] == 0:
                            continue

                        if (r,conn) not in self.chosen_optical_conns:
                            z_dim_unchosen = True

                    for conn in poss_conns[dim_neg]:
                        if valid_conns[r][conn] == 0:
                            continue
                        if (r,conn) not in self.chosen_optical_conns:
                            z_dim_unchosen = True

        if z_dim_unchosen :
            print(f"z has unchosen? {z_dim_unchosen}")

    
        new_scores = OffDiagMatrix(n_routers, init_val=0)

        for i in range(n_routers):
            for j in range(n_routers):
                if i == j:
                    continue
                if valid_conns[i][j] == 0:
                    continue
                
                ij_conn_dim = self.calc_opt_conn_type(i,j)
                if ij_conn_dim is None:
                    continue
                ij_conn_dim = ij_conn_dim[0]
                if z_dim_unchosen and ij_conn_dim != 'z':
                    continue
                elif ij_conn_dim == 'z':
                # if ij_conn_dim != unchosen_dim:
                    continue
                new_scores[i,j] = scores[i,j]
        return new_scores

    def preclusion_sum_based(self):

        
        valid_conns = self.valid_conns
        n_routers = self.n_routers
        electrical_conns_adj_list = self.electrical_conns_adj_list
        chosen_optical_conns = self.chosen_optical_conns


        opt_conn_options = self.calc_opt_conn_options()


        r_map_vals = self.translate_r_map_into_values_twod()
        scores = OffDiagMatrix(n_routers, init_val=-1*self.INF)

        for (a,b) in opt_conn_options:

            precluded_conns_given_ab = self.precluded_conns_given_conn((a,b))

            lost_sum = 0
            for conn in precluded_conns_given_ab:
                lost_sum -= r_map_vals[conn]
            
            scores[a,b] = lost_sum

            print(f"calculating cost for {(a,b)} = {lost_sum}")


        return scores


    def score_solution(self):

        start_time = time.time()

        # best/most reasonable:
        #   downstream_effects
        # not longer dims or normalization

        self.normalized_scoring = False #True #False
        self.longer_dims_first = False #True # False #True

        if self.advanced_scoring:
            scores = self.downstream_effects()
            # scores = self.downstream_effects_v2()

            # scores = self.matching_score_matrix()

            # scores = self.slack_based()
            # scores = self.regret_based()
            # return self.normalized_r_map_values()
            # return self.normalized_r_map_values_v2()
            # return self.normalized_r_map_values_v3()
            # scores = self.preclusion_sum_based()
        else:
            
            # scores = self.translate_r_map_into_values_twod()
            scores = self.translate_r_map_into_values_twod()

            # scores = self.get_At_dual_vals()

        if self.longer_dims_first:
            # # z first
            # scores = self.prune_to_longest_dims(scores)
            dim_order = ['z']
            scores = self.prune_to_longest_dims_v2(scores, dim_order)
            print(f"using z first")
            # dim_order = ['x']
            # scores = self.prune_to_longest_dims_v2(scores, dim_order)
            # print(f"using x first")
            # z -> y -> x
            # dim_order = ['z','y','x']
            # scores = self.prune_to_longest_dims_v2(scores, dim_order)
            # print(f"using dim order {dim_order}")
            # x -> y -> z
            # dim_order = ['x','y','z']
            # scores = self.prune_to_longest_dims_v2(scores, dim_order)

        if self.normalized_scoring:
            scores = self.normalize_r_map_values(scores, weighting_type="v4")

        elapsed_time = round(time.time() - start_time,2)
        logger.performance(f"Scoring completed in {elapsed_time}s")

        return scores

    def calc_non_integrality(self):
        assert(self.n_routers)
        n_routers = self.n_routers

        r_map_vals = self.translate_r_map_into_values_twod()

        cumul_non_integrality = 0
        for i in range(n_routers):
            for j in range(i+1,n_routers):
                cumul_non_integrality += 2*min(r_map_vals[i,j], 1.0 - r_map_vals[i,j])

        return cumul_non_integrality


    def calc_opt_conn_options(self):
        assert(self.n_routers)
        assert(self.valid_conns)
        assert(self.electrical_conns_adj_list)
        assert(self.chosen_optical_conns is not None)
        n_routers = self.n_routers
        valid_conns = self.valid_conns
        electrical_conns_adj_list = self.electrical_conns_adj_list
        chosen_optical_conns = self.chosen_optical_conns

        if self.opt_conn_options is not None:
            return self.opt_conn_options

        opt_conn_options = []
        for i in range(n_routers):
            for j in range(n_routers):
                if valid_conns[i][j] == 0:
                    continue
                if j in electrical_conns_adj_list[i]:
                    continue
                if (i,j) in chosen_optical_conns:
                    continue
                opt_conn_options.append((i,j))
        self.opt_conn_options = opt_conn_options
        return opt_conn_options

    def calc_groups(self):
        # TODO asserts
        (x_dim, y_dim, z_dim, cube_dim) = self.xyzc_dims
        all_directions = ['x+','y+','z+','x-','y-','z-']
        n_routers = self.n_routers
        chosen_optical_conns = self.chosen_optical_conns

        all_groups = []
        all_groups_by_direction = []
        for z in range(z_dim):
            for y in range(y_dim):
                for x in range(x_dim):
                    canon_r = self.xyz_to_r(x,y,z)
                    if not self.r_is_on_face(canon_r):
                        continue

                    groups = {d:set() for d in all_directions}
                    poss_conns = self.poss_optical_conns_for_r(canon_r)
                    for direction, conns in poss_conns.items():
                        if len(conns) == 0:
                            continue
                        group = set()
                        for conn in conns:
                            groups[direction].add( (canon_r, conn) )
                            group.add( (canon_r, conn) )
                        
                        all_groups.append(group)
    
                    # input(f"groups for {canon_r} : {groups}")
                    all_groups_by_direction.append(groups)

        return all_groups, all_groups_by_direction

    def calc_r_map_penalties(self):
        assert(self.n_routers)
        assert(self.penalty_adjust)
        n_routers = self.n_routers
        penalty_adjust = self.penalty_adjust
        climbing_penalty_adjust = self.climbing_penalty_adjust

        # TODO as CLA
        penalize_fractional_more = False #True
        penalize_within_group = False #True
        # TODO make CLA
        auto_scale_penalty = False #True

        (obj_val, unity_wye_val) = self.get_obj_val()
        r_map_vals = self.translate_r_map_into_values_twod()
        penalties = UpperTriMatrix(n_routers,init_val=0)
        opt_conn_options = self.calc_opt_conn_options()

        (all_groups, all_groups_by_direction) = self.calc_groups()

        # non_integrality_per_conn = UpperTriMatrix(n_routers,init_val=0)
        cumul_penal = 0
        if not penalize_within_group:

            for (i,j) in opt_conn_options:
                r_map_val = r_map_vals[i,j]
                
                if penalize_fractional_more:
                    # ie diff from 0 or 1
                    non_integrality = min(r_map_val,1-r_map_val)
                    sign = 1
                    if r_map_val < 0.5:
                        sign = -1                
                    penal_val = sign*non_integrality
                else:
                    penal_val = (2*r_map_val - 1.0)

                # input(f"r_map_val[{(i,j)}] = {r_map_val} => non-integrality = {non_integrality} => penal_val {penal_val}")

                penalties[i,j] = penal_val
                cumul_penal += abs(penal_val*r_map_val)
        else:
            for group in all_groups:
                n_elems = len(group)

                for (i,j) in group:
                    r_map_val = r_map_vals[i,j]
                    penal_val = r_map_val - (1/n_elems)
                    penalties[i,j] = penal_val
                    cumul_penal += abs(penal_val*r_map_val)

                    # input(f"r_map_val[{(i,j)}] = {r_map_val} in group of size {n_elems} => penal_val {penal_val}")



        eps = 0.00001

        if auto_scale_penalty:
            penal_weight = penalty_adjust*(unity_wye_val / (cumul_penal + eps))
        else:
            # complete integrality will weight at penalty_adjust rate to objective (e.g. penalty_adjust = 10% => integrality 1/10 importance)
            penal_weight = penalty_adjust*(unity_wye_val / (n_routers))


        if climbing_penalty_adjust:
            penal_weight *= max(1,len(self.chosen_optical_conns))


        if self.iteration_number % 5 == 0:
            penal_weight = 0


        print(f"penalty weight = {penal_weight}")

        for (i,j) in opt_conn_options:
            penalties[i,j] = penal_weight*penalties[i,j]

        return penalties

    @classmethod
    def _translate_r_map_into_values_twod_gurobi(cls, model,r_map_vars, n_routers):

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
                    if r_map[i,j] is None:
                        r_map[i,j] = 0
                    else:
                        r_map[i,j] = r_map_vars[i,j]
        return r_map

    @classmethod
    def _translate_tri_ineq_wyes_into_values_gurobi(cls, model, var_tri_ineq_wyes):

        tri_ineq_wyes = SymTriDict()

        for k, v in var_tri_ineq_wyes.items():
            tri_ineq_wyes[k] = v.X

        return tri_ineq_wyes


    def translate_r_map_into_values_twod(self):
        assert(self.solver_library)
        assert(self.model)
        assert(self.var_r_map)
        assert(self.n_routers)
        solver_library = self.solver_library
        model = self.model
        # may be None
        solver_or_result = self.solver_or_result
        var_r_map = self.var_r_map
        n_routers = self.n_routers

        # compute once
        if self.r_map_vals is not None:
            return self.r_map_vals

        if solver_library == 'gurobi':
            r_map_vals = self._translate_r_map_into_values_twod_gurobi(model,var_r_map, n_routers)
        # elif solver_library == 'highs':
        #     return _translate_r_map_into_values_twod_highs(model,var_r_map, n_routers,var_name_to_index_map=var_name_to_index_map)
        # elif solver_library == 'ortools_model_builder':
        #     return _translate_r_map_into_values_twod_ortools_model_builder(model, solver_or_result, var_r_map, n_routers)
        # elif solver_library == 'ortools':
        #     return _translate_r_map_into_values_twod_ortools(model, solver_or_result, var_r_map, n_routers)
        else:
            logger.critical(f'translate_r_map_into_values_twod :: Unimplemetedn. Exiting...')
            quit()
        
        self.r_map_vals = r_map_vals
        return r_map_vals

    def translate_tri_ineq_wyes_into_values(self):
        assert(self.solver_library)
        assert(self.model)
        assert(self.var_tri_ineq_wyes)
        assert(self.n_routers)
        solver_library = self.solver_library
        model = self.model
        # may be None
        solver_or_result = self.solver_or_result
        var_tri_ineq_wyes = self.var_tri_ineq_wyes
        n_routers = self.n_routers

        # compute once
        if self.tri_ineq_wye_vals is not None:
            return self.tri_ineq_wye_vals

        if solver_library == 'gurobi':
            tri_ineq_wye_vals = self._translate_tri_ineq_wyes_into_values_gurobi(model,var_tri_ineq_wyes)
        # elif solver_library == 'highs':
        #     return _translate_r_map_into_values_twod_highs(model,var_r_map, n_routers,var_name_to_index_map=var_name_to_index_map)
        # elif solver_library == 'ortools_model_builder':
        #     return _translate_r_map_into_values_twod_ortools_model_builder(model, solver_or_result, var_r_map, n_routers)
        # elif solver_library == 'ortools':
        #     return _translate_r_map_into_values_twod_ortools(model, solver_or_result, var_r_map, n_routers)
        else:
            logger.critical(f'translate_tri_ineq_wyes_into_values :: Unimplemetedn. Exiting...')
            quit()

        self.tri_ineq_wye_vals = tri_ineq_wye_vals
        return tri_ineq_wye_vals

    def get_At_dual_vals(self):
        assert(self.n_routers)
        assert(self.model)
        n_routers = self.n_routers
        model = self.model

        # pi_matrix = OffDiagMatrix(n_routers, init_val=0)
        pi_matrix = OffDiagMatrix(n_routers, init_val=-self.INF)
        for i in range(n_routers):
            for j in range(n_routers):
                if i==j:
                    continue
                ij_At_constr_name = self._name_constr_A_transpose(i,j)
                ij_At_constr = model.getConstrByName(ij_At_constr_name)
                ij_dual_pi = ij_At_constr.Pi
                # pi_matrix[i,j] = ij_dual_pi
                pi_matrix[i,j] = -1*ij_dual_pi

                # print(f"(i,j) {(i,j)} : w/ dual {ij_dual_pi}")
        return pi_matrix

    def reset_sol_vals_(self):

        self.r_map_vals = None
        self.tri_ineq_wye_vals = None
        self.opt_conn_options = None

    # Looping and completion
    ################################################################################

    def print_dorness(self):
        model = self.model
        (x_dim, y_dim, z_dim, cube_dim) = self.xyzc_dims
        r_map_vals = self.translate_r_map_into_values_twod()

        # z line
        for x in range(2):
            for y in range(2):
                
                nc_z = z_dim // cube_dim
                if nc_z == 1:
                    continue

                var_dors_names = [f"var_dor_zline_{x}x{y}y_{c_id}cid" for c_id in range(nc_z)]

                print(f"Vars dor")
                for i,vname in enumerate(var_dors_names):
                    print(f"vname = {vname}")
                    v = model.getVarByName(vname)
                    print(f"cube {i} : {vname} = {v.X}")


                nodes_neg_side = []
                nodes_pos_side = []
                # 0, 4, 8, ...
                for z in range(0,nc_z*cube_dim,cube_dim):
                    nodes_neg_side.append(self.xyz_to_r(x, y, z))
                # 3, 7, 11, ...
                for z in range(cube_dim-1,nc_z*cube_dim,cube_dim):
                    nodes_pos_side.append(self.xyz_to_r(x, y, z))

                N = nc_z
                for n_i in nodes_neg_side:
                    for n_j in nodes_pos_side:

                        i_cube = self.which_cube(n_i)
                        j_cube = self.which_cube(n_j)

                        print(f"{n_i}->{n_j} cubes {i_cube}->{j_cube} w/ conn {r_map_vals[n_i,n_j]}")

            input("cont?")


    def drive_generation(self):
        assert(self.script_params)
        assert(self.model_base_name)
        model_base_name = self.model_base_name
        script_params = self.script_params

        recalc_interval = script_params["recalc_interval"]

        binary_r_map = script_params["binary_r_map"]
        if binary_r_map:
            recalc_interval = self.INF

        

        n_tot_optical_links = self.n_cubes*(self.cube_dim**2)*6

        self.timeline_log(0, 0, 0, 0, model_base_name, first_write=True)
        # self.plot_timeline(0, 0, 0, live_plotting=self.live_plotting)

        self.connect_forced_moves()

        obj_val = 0
        avg_hops = self.INF
        initial_solve = True
        done = False
        while not done:

            print("")
            logger.info("Beginning outer loop")

            start_time = time.time()

            if not initial_solve and self.penalize_fractional_r_map:
                r_map_penalty_coeffs = self.calc_r_map_penalties()
                self.update_objective_for_penalized_r_map_(r_map_penalty_coeffs)

            self.reset_sol_vals_()

            # modifies self.model, self.solver_or_result
            self.solve_model_( initial_solve=initial_solve)
            initial_solve = False
            (obj_val, unity_wye_val) = self.get_obj_val()

            # input('cont?')
            logger.info(f"obj = {obj_val}, unity_wye_val = {unity_wye_val}")

            conn_scores = self.score_solution()

            cumul_non_integrality = self.calc_non_integrality()
            logger.info(f"Cumulative non-integrality = {cumul_non_integrality} for {len(self.calc_opt_conn_options())} options")

            # self.print_dorness()

            inner_iter = 0
            while inner_iter < recalc_interval and len(self.chosen_optical_conns) <= n_tot_optical_links:

                chosen_max_conn, max_val = self.select_best_score_opt_conn(conn_scores)

                logger.info(f"inner iter {inner_iter} / {recalc_interval}. Chosen max conn {chosen_max_conn} w/ value {max_val}")

                (i,j) = chosen_max_conn
                i_cube = self.which_cube(i)
                j_cube = self.which_cube(j)

                logger.info(f'Selected max conn {chosen_max_conn} w/ score {max_val} => (i,j) {(i,j)} conn type {self.calc_opt_conn_type(i,j)} to connect cubes {i_cube}->{j_cube}')
                self.add_opt_conn_and_equivalents_(chosen_max_conn)

                self.connect_forced_moves()

                n_opt_links = len(self.chosen_optical_conns)
                avg_hops = self.calc_avg_hops()
                self.timeline_log(n_opt_links, avg_hops, unity_wye_val, cumul_non_integrality, model_base_name)
                # self.plot_timeline(n_opt_links, avg_hops, obj_val, live_plotting=self.live_plotting)

                if avg_hops < self.INF:
                    self.connect_first = False

                inner_iter += 1

                if self.verbose:
                    for i in range(self.n_routers):
                        out_line = f"{i:02} : "
                        for j in range(self.n_routers):
                            if self.valid_conns[i][j] > 0:
                                out_line += f"({j:02} {self.calc_conn_type(i,j)}/{self.calc_opt_conn_type(i,j)} {self.adj_mat[i][j]}) "
                        print(out_line)


                self.iteration_number += 1

                # extra
                # -----

                if self.careful and self.symmetric:
                    self.tpuv4_symmetry.verify_symmetry_for_topology(self.adj_mat)


            cumul_non_integrality = self.calc_non_integrality()
            logger.info(f"Cumulative non-integrality = {cumul_non_integrality} for {len(self.calc_opt_conn_options())} options")

            elapsed_time = round(time.time() - start_time,2)
            logger.info(f"Completed outer iter")
            logger.info(f"Iteration #           : {self.iteration_number}")
            logger.info(f"# links chosen        : {n_opt_links} / {n_tot_optical_links}")
            logger.info(f"Avg hops              : {avg_hops}")
            logger.info(f"Obj val               : {obj_val}")
            logger.info(f"~SC                   : {unity_wye_val}")
            logger.info(f"Non integrality       : {cumul_non_integrality}")
            logger.performance(f"Outer loop completed in {elapsed_time}s")

            if len(self.chosen_optical_conns) >= n_tot_optical_links:
                done = True
                break

            if self.slow_run:
                input('cont?')

            if self.careful and self.symmetric:
                self.tpuv4_symmetry.verify_symmetry_for_topology(self.adj_mat)

        logger.info(f'Completed execution')


        cumul_non_integrality = self.calc_non_integrality()

        n_opt_links = len(self.chosen_optical_conns)
        self.solve_model_( )
        (obj_val, unity_wye_val) = self.get_obj_val()
        avg_hops = self.calc_avg_hops()
        logger.info(f"Average hops     : {avg_hops}")
        logger.info(f"Obj val          : {obj_val}")
        logger.info(f"~SC              : {unity_wye_val}")
        self.timeline_log(n_opt_links, avg_hops, unity_wye_val, 0, model_base_name)

        self.write_topology()

    def add_opt_conn_and_equivalents_(self, opt_conn):
        # basically handling symmetry

        tpuv4_symmetry = self.tpuv4_symmetry

        (i,j) = opt_conn

        all_i_primes = tpuv4_symmetry.get_all_noncanonical_equivalents(i)
        # print(f"all_i_primes = {all_i_primes}")
        for i_prime in all_i_primes:
            i_to_i_prime_tform = tpuv4_symmetry.calc_transform_delta(i,i_prime)
            j_prime = tpuv4_symmetry.apply_transformation(j,i_to_i_prime_tform)

            # remove later
            self.add_opt_conn_((i_prime,j_prime))
        
        # self.update_tri_ineq_wyes_()



    def update_tri_ineq_wyes_(self, rejected_conns):
        assert(self.n_routers)
        n_routers = self.n_routers

        rejected_wye_tri_ineqs = set()
        # for a in range(n_routers):
        #     for c in range(n_routers):
        #         if self.valid_conns[a][c] > 0:
        #             continue

        #         for b in range(n_routers):
        #             if (a,b,c) in self.known_rejected_tri_ineqs:
        #                 continue
        #             if self.relevant_ijk(a,b,c):
        #                 continue
        #             rejected_wye_tri_ineqs.add((a,b,c))
        #             rejected_wye_tri_ineqs.add((c,b,a))
        #             self.known_rejected_tri_ineqs.add((a,b,c))
        #             self.known_rejected_tri_ineqs.add((c,b,a))

        for (i,k) in rejected_conns:
            for j in range(n_routers):
                if (i,j,k) in self.known_rejected_tri_ineqs:
                        continue
                rejected_wye_tri_ineqs.add((i,j,k))
                rejected_wye_tri_ineqs.add((i,j,k))
                self.known_rejected_tri_ineqs.add((i,j,k))
                self.known_rejected_tri_ineqs.add((i,j,k))

        rejected_wye_tri_ineqs = list(rejected_wye_tri_ineqs)
        rejected_wye_tri_ineqs = [t for t in rejected_wye_tri_ineqs if t in self.var_tri_ineq_wyes]
        # logger.info(f"Rejecting {rejected_wye_tri_ineqs}")

        self.update_multi_lbub_tri_ineq_wyes_(rejected_wye_tri_ineqs, 0, 0)

    def connect_forced_moves(self):
        assert(self.n_routers)
        assert(self.valid_conns)
        assert(self.adj_mat)
        n_routers = self.n_routers
        valid_conns = self.valid_conns
        adj_mat = self.adj_mat

        routers_of_canonical_cube = self.get_canonical_nodes()


        forced_conns = set()
        for i in routers_of_canonical_cube:
            poss_conn_dict = self.poss_optical_conns_for_r(i)
            for direction, conns in poss_conn_dict.items():
                if len(conns) == 0:
                    continue
                n_possible = 0
                for conn in conns:
                    if valid_conns[i][conn] > 0:
                        n_possible += 1

                if self.verbose:
                    print(f"router {i} in dir {direction} w/ possible {conns} (valid {[valid_conns[i][c] for c in conns]}) has # valid {n_possible}")
                # if forced and not already present
                if n_possible == 1:
                    for conn in conns:
                        if valid_conns[i][conn] and adj_mat[i][conn] == 0:

                            canonical_force = (min(i,conn),max(i,conn))
                            forced_conns.add(canonical_force)
                            if self.verbose:
                                print(f"router {i} in dir {direction} has forced conn to {canonical_force}")

        logger.info(f"adding forced conns {forced_conns}")

        for conn in forced_conns:
            # input(f"adding {conn}")

            # previous connections may already connect it
            if conn in self.chosen_optical_conns:
                continue
            self.add_opt_conn_and_equivalents_(conn)

    def precluded_conns_given_conn(self, opt_conn):

        (i,j) = opt_conn
        ij_conn_dim = self.calc_opt_conn_type(i,j)
        ji_conn_dim = self.calc_opt_conn_type(j,i)

        # negatives
        rejected_conns = []
        # reject i<->possible along ij_conn_dim
        poss_conn_dict = self.poss_optical_conns_for_r(i)
        poss_conns = poss_conn_dict[ij_conn_dim]
        for conn in poss_conns:
            if conn != j and self.valid_conns[i][conn] > 0:
                rejected_conns.append((i,conn))
                rejected_conns.append((conn,i))
        # reject j<->possible along ji_conn_dim
        poss_conn_dict = self.poss_optical_conns_for_r(j)
        poss_conns = poss_conn_dict[ji_conn_dim]
        for conn in poss_conns:
            if conn != i and self.valid_conns[j][conn] > 0:
                rejected_conns.append((j,conn))
                rejected_conns.append((conn,j))

        # DOR
        # ---
        if self.dorable: # and not self.symmetric and (not TREAT_ALL_DORABLE_LIKE_SYMMETRIC_DORABLE):

            disallowed = self.calc_dor_disallowed(opt_conn)

            # logger.info(f"For DOR, disallowing {disallowed}")

            for (a,b) in disallowed:
                if (a,b) not in rejected_conns:
                    rejected_conns.append((a,b))
                    rejected_conns.append((a,b))

        return rejected_conns

    def add_opt_conn_(self, opt_conn):
        assert(self.chosen_optical_conns is not None)
        assert(self.chosen_optical_conns_by_dim)
        assert(self.adj_mat)
        assert(self.scale_factor)
        assert(self.n_routers)
        scale_factor = self.scale_factor
        n_routers = self.n_routers

        logger.info(f"Adding opt conn {opt_conn}")

        (i,j) = opt_conn

        assert(self.valid_conns[i][j] > 0 and self.valid_conns[j][i] > 0)
        assert((i,j) not in self.chosen_optical_conns and (j,i) not in self.chosen_optical_conns)
        assert(self.adj_mat[i][j] == 0 and self.adj_mat[i][j] == 0)

        # object tracking vars
        # --------------------
        self.chosen_optical_conns.append((i,j))
        self.chosen_optical_conns.append((j,i))
        conn_dim = self.calc_opt_conn_type(i,j)[0]
        ij_conn_dim = self.calc_opt_conn_type(i,j)
        ji_conn_dim = self.calc_opt_conn_type(j,i)
        self.chosen_optical_conns_by_dim[conn_dim].append((i,j))
        self.chosen_optical_conns_by_dim[conn_dim].append((j,i))
        self.adj_mat[i][j] = 1
        self.adj_mat[j][i] = 1
        i_cube = self.which_cube(i)
        j_cube = self.which_cube(j)
        self.chosen_cube_conns.add((i_cube,j_cube))
        self.chosen_cube_conns.add((j_cube,i_cube))
        if self.chosen_cube_conns == self.total_cube_conns:
            self.chosen_cube_conns = set()

        # valid conns
        # -----------
        self.update_valid_conns_(opt_conn)

        # update model
        # ------------
        # positive(s)
        self.update_lbub_r_map_(opt_conn, 1,1) # scale_factor, scale_factor)

        # negatives
        rejected_conns = self.precluded_conns_given_conn(opt_conn)

        logger.info(f"Rejecting {rejected_conns}")

        for (a,b) in rejected_conns:

            # print(f"Rejecting conn {(a,b)} @ {self.r_to_xyz(a)}, {self.r_to_xyz(b)} type {self.calc_opt_conn_type(a,b)} was valid? {self.valid_conns[a][b]}")
            self.valid_conns[a][b] = 0
            self.valid_conns[b][a] = 0

        # input('cont?')

        self.update_multi_lbub_r_map_(rejected_conns, 0, 0)

        self.rejected_conns.update(rejected_conns)

        self.update_tri_ineq_wyes_(rejected_conns)

        # rejected_wye_tri_ineqs = set()
        # for a in range(n_routers):
        #     for c in range(n_routers):
        #         if self.valid_conns[a][c] > 0:
        #             continue

        #         for b in range(n_routers):
        #             if (a,b,c) in self.known_rejected_tri_ineqs:
        #                 continue
        #             rejected_wye_tri_ineqs.add((a,b,c))
        #             rejected_wye_tri_ineqs.add((c,b,a))
        #             self.known_rejected_tri_ineqs.add((a,b,c))
        #             self.known_rejected_tri_ineqs.add((c,b,a))


        # # for (a,b) in rejected_conns:
        # #     for c in range(self.n_routers):
        # #         rejected_wye_tri_ineqs.add((a,c,b))
        # #         rejected_wye_tri_ineqs.add((b,c,a))

        # rejected_wye_tri_ineqs = list(rejected_wye_tri_ineqs)
        # rejected_wye_tri_ineqs = [t for t in rejected_wye_tri_ineqs if t in self.var_tri_ineq_wyes]
        # # logger.info(f"Rejecting {rejected_wye_tri_ineqs}")

        # self.update_multi_lbub_tri_ineq_wyes_(rejected_wye_tri_ineqs, 0, 0)

    def write_topology(self):
        assert(self.adj_mat)
        assert(self.model_base_name)
        adj_mat = self.adj_mat
        model_base_name = self.model_base_name
        # TODO safety checks

        out_name = f"{model_base_name}.map"
        out_path = os.path.join(self.topology_dir,out_name)

        self.output_an_adj_mat(adj_mat, out_path)

# CLAs
###############################################################################

def define_and_parse_args(description="Generate (Direct) Topologies with Various Formulations"):

    parser = argparse.ArgumentParser(description=description)

    # out naming
    parser.add_argument("--out_filename","-of",type=str,help="")
    parser.add_argument("--dev_mode",action="store_true",help="Lots of asserts")

    # constant/problem defs
    parser.add_argument("--xyzc_dims",nargs="+",type=int,help="type without parenthesis and use spaces, no commas")


    # script stuff
    parser.add_argument("--verbose","-v",action="store_true",help="extensive prints")
    parser.add_argument("--slow_run",action="store_true",help="ask user input on every iteration")

    parser.add_argument("--solver",type=str,default="gurobi",help="")

    parser.add_argument("--write_model",action="store_true",help="write model out as multiple/all formats")
    parser.add_argument("--write_presolved",action="store_true",help="presolve and write (presolved) model out as multiple/all formats")
    parser.add_argument("--no_solve",action="store_true",help="quit after model creation but before solving")

    parser.add_argument("--symmetric",action="store_true",help="")
    parser.add_argument("--mc_dims",nargs="+",type=int,help="type without parenthesis and use spaces, no commas")
    parser.add_argument("--sym_type",type=str,default="trans",choices=["trans","refl-trans"],help="")

    parser.add_argument("--binary_r_map",action="store_true",help="")
    parser.add_argument("--recalc_interval",type=int,default=1,help="")
    # parser.add_argument("--rm_coords_interval",type=int,default=1,help="")
    parser.add_argument("--scale_factor",type=float,default=1.0,help="")
    parser.add_argument("--start_map",type=str,help="read a starting r_map")
    parser.add_argument("--spreading_cubes",action="store_true",help="")
    parser.add_argument("--connect_first",action="store_true",help="")
    parser.add_argument("--dorable",action="store_true",help="")
    parser.add_argument("--no_same_cube",action="store_true",help="")
    parser.add_argument("--dor_heur",action="store_true",help="")
    parser.add_argument("--advanced_scoring",action="store_true",help="")
    parser.add_argument("--penalize_non_integral",action="store_true",help="")
    parser.add_argument("--non_integral_adjustment",type=float,default=0.1,help="")
    parser.add_argument("--sos_for_limit_xyz",action="store_true",help="")

    parser.add_argument("--map_sym_breaker",type=float,default=0.0,help="")
    parser.add_argument("--uwye_sym_breaker",type=float,default=0.0,help="")

    
    # parser.add_argument("--use_iter_checkpoints",action="store_true",help="write current_r_map")



    # direct Gurobi solver params
    parser.add_argument("--time_limit",type=int,help="time limit in minutes")
    parser.add_argument("--time_limit_secs",type=int,help="time limit in minutes")
    parser.add_argument("--threads",type=int,default=128,help="# threads total")
    parser.add_argument("--presolve",type=int,help="Presolve aggressiveness. -1=>auto. 0=>off. 1=>conservative. 2=>aggressive")
    parser.add_argument("--output_flag",type=int,default=1,help="")
    parser.add_argument("--method",type=str,default="ipm",choices=["ipm","simplex","pdlp","dual_simplex"],help="")
    parser.add_argument("--method_first_round",type=str,choices=["ipm","simplex","pdlp"],help="")
    parser.add_argument("--node_method",type=str,choices=["ipm","simplex","pdlp","dual_simplex"],help="")
    parser.add_argument("--kkt_tolerance",type=float,help="single master tolerance. \
                        All feasibility/optimality tests (primal feasibility, dual feasibility, primal/dual residuals, and P-D objective error")
    parser.add_argument("--pdlp_optimality_tolerance",type=float,help="")
    parser.add_argument("--pdlp_scaling",type=bool,help="")
    parser.add_argument("--crossover",type=int,default=0,help="")
    parser.add_argument("--mip_focus",type=int,help="1=>incumbent. 2=>optimality. 3=>bounds")
    parser.add_argument("--degen_moves",type=int,help="")
    parser.add_argument("--symmetry",type=int,help="solver symmetry focus")
    parser.add_argument("--cuts",type=int,help="")
    parser.add_argument("--cover_cuts",type=int,help="")
    parser.add_argument("--clique_cuts",type=int,help="")
    parser.add_argument("--var_branch",type=int,help="")
    parser.add_argument("--numeric_focus",type=int,help="")
    parser.add_argument("--bar_conv_tol",type=float,help="")


    args = parser.parse_args()

    print(f"Parsed args : {args}")

    return args

def parse_and_package_script_params_from_args(args):

    script_params = {
        "out_filename":args.out_filename,
        "write_model":args.write_model,
        "no_solve":args.no_solve,
        "recalc_interval":args.recalc_interval,
        # "rm_coords_interval":args.rm_coords_interval,
        "scale_factor":args.scale_factor,
        "map_sym_breaker":args.map_sym_breaker,
        "uwye_sym_breaker":args.uwye_sym_breaker,
        "start_map":args.start_map,
        "spreading_cubes":args.spreading_cubes,
        "connect_first":args.connect_first,
        "solver":args.solver,
        "binary_r_map":args.binary_r_map,
        "dorable":args.dorable,
        "dor_heur":args.dor_heur,
        "no_same_cube":args.no_same_cube,
        "symmetric":args.symmetric,
        "sym_type":args.sym_type,
        "mc_dims":args.mc_dims,
        "advanced_scoring":args.advanced_scoring,
        "penalize_non_integral":args.penalize_non_integral,
        "non_integral_adjustment":args.non_integral_adjustment,
        "sos_for_limit_xyz":args.sos_for_limit_xyz

    }

    return script_params

def _parse_and_package_solver_params_from_args_ortools(args):

    # solver params
    solver_params = {}


    if args.time_limit_secs is not None:
        solver_params.update({"time_sec_limit":args.time_limit_secs})
    if args.output_flag is not None:
        solver_params.update({"output_flag":args.output_flag})

    return solver_params

def _parse_and_package_solver_params_from_args_highs(args):

    # solver params
    solver_params = {}

    if args.time_limit is not None:
        solver_params.update({"time_limit":args.time_limit*60})
    if args.time_limit_secs is not None:
        solver_params.update({"time_limit":args.time_limit_secs})
    if args.threads is not None:
        solver_params.update({"threads":args.threads})
    if args.output_flag is not None:
        solver_params.update({"output_flag":args.output_flag})
    if args.method is not None:
        # rename
        solver_params.update({"solver":args.method})
    if args.method_first_round is not None:
        solver_params.update({"Method":args.method_first_round})
    if args.kkt_tolerance is not None:
        solver_params.update({"kkt_tolerance":args.kkt_tolerance})
    if args.pdlp_optimality_tolerance is not None:
        solver_params.update({"pdlp_optimality_tolerance":args.pdlp_optimality_tolerance})
    if args.pdlp_scaling is not None:
        solver_params.update({"pdlp_scaling":args.pdlp_scaling})
    if args.crossover is not None:
        _translation = {0:"off",1:"on"}
        solver_params.update({"run_crossover":_translation[args.crossover]})

    return solver_params

def _parse_and_package_solver_params_from_args_gurobi(args):

    # solver params
    solver_params = {}

    # rename
    _solver_translation = {"ipm":2,"simplex":0,"dual_simplex":1}    

    if args.time_limit is not None:
        solver_params.update({"TimeLimit":args.time_limit*60})
    if args.time_limit_secs is not None:
        solver_params.update({"TimeLimit":args.time_limit_secs})
    if args.threads is not None:
        solver_params.update({"Threads":args.threads})
    if args.output_flag is not None:
        _translation = {True:1,False:0}    
        solver_params.update({"OutputFlag":args.output_flag})
    if args.method is not None:
        solver_params.update({"Method":_solver_translation[args.method]})
    if args.node_method is not None:
        solver_params.update({"NodeMethod":_solver_translation[args.node_method]})
    if args.crossover is not None:
        solver_params.update({"Crossover":args.crossover})
    if args.method_first_round is not None:
        solver_params.update({"MethodFirstRound":_solver_translation[args.method_first_round]})
    if args.mip_focus is not None:
        solver_params.update({"MIPFocus":args.mip_focus})
    if args.degen_moves is not None:
        solver_params.update({"DegenMoves":args.degen_moves})
    if args.symmetry is not None:
        solver_params.update({"Symmetry":args.symmetry})
    if args.cuts is not None:
        solver_params.update({"Cuts":args.cuts})
    if args.cover_cuts is not None:
        solver_params.update({"CoverCuts":args.cover_cuts})
    if args.clique_cuts is not None:
        solver_params.update({"CliqueCuts":args.clique_cuts})
    if args.var_branch is not None:
        solver_params.update({"VarBranch":args.var_branch})
    if args.numeric_focus is not None:
        solver_params.update({"NumericFocus":args.numeric_focus})
    if args.bar_conv_tol is not None:
        solver_params.update({"BarConvTol":args.bar_conv_tol})

    return solver_params

def parse_and_package_solver_params_from_args(args):

    if args.solver == "highs":
        return _parse_and_package_solver_params_from_args_highs(args)
    elif args.solver == "ortools" or args.solver == "ortools_model_builder":
        return _parse_and_package_solver_params_from_args_ortools(args)
    elif args.solver == "gurobi":
        return _parse_and_package_solver_params_from_args_gurobi(args)

    print(f"ERROR: Unrecognized solver {solver}. Exiting...")
    quit()

# Script functions
################################################################################

def main():

    # default
    xyzc_dims = (6, 3, 3, 3)

    args = define_and_parse_args()
    if args.xyzc_dims:
        xyzc_dims = tuple(args.xyzc_dims)
    solver_params = parse_and_package_solver_params_from_args(args)
    script_params = parse_and_package_script_params_from_args(args)

    start_map = args.start_map


    logger.info(f"solver_params : {solver_params}")
    logger.info(f"script_params : {script_params}")

    # filter out unimplemented
    if args.binary_r_map:
        logger.warning(f"binary_r_map is MILP and that usually needs callbacks. Use MILP specific script.")


    if args.verbose:
        AASC_TPUv4.verbose = True
    if args.slow_run:
        AASC_TPUv4.slow_run = True
    if args.dev_mode:
        AASC_TPUv4.careful = True
    aasc_tpuv4 = AASC_TPUv4(xyzc_dims, script_params=script_params, solver_params=solver_params)

    # for conn in [(0,99),(18,27),(45,54),(72,81)]:
    #     aasc_tpuv4.add_opt_conn_(conn)
    
    # print(f"{aasc_tpuv4.chosen_optical_conns}")

    # quit()


    # aasc_tpuv4.regret_based()

    aasc_tpuv4.create_model()

    if start_map is not None:
        aasc_tpuv4.set_start_map(start_map)

    aasc_tpuv4.drive_generation()

    # also validates
    # aasc_tpuv4.write_topology()

if __name__ == "__main__":

    main()

