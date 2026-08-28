"""
Title:
    complete_cdg_generation.py

Description:
    Topology, routing, and deadlock freedom (VC allocation) synthesis using an complete channel dependency graph (CDG).
    Optimizes for Min-Max Channel Load (Congestion) with Exact Integer Demand Routing.

Reference(s):
    Basis for MCF formulation: The Maximum Concurrent Flow Problem by Shahrokhi and Matula (https://dl.acm.org/doi/abs/10.1145/77600.77620)
    Miller-Tucker-Zemlin (MTZ) acyclicality constraint: "Integer programming formulations and traveling salesman problems" by Miller, Tucker, and Zemlin (https://dl.acm.org/doi/10.1145/321043.321046)
    CDG for deadlock freedom: "Deadlock-Free Message Routing in Multiprocessor Interconnection Networks" by Dally and Seitz (https://ieeexplore.ieee.org/abstract/document/1676939)
    Complete CDG: "Routing on the Dependency Graph: A New Approach to Deadlock-Free High-Performance Routing" by Domke, Hoefler, and Matsuoka (https://dl.acm.org/doi/abs/10.1145/2907294.2907313)

Author(s):
    Conor Green (green456@purdue.edu | conor.green.2020@gmail.com)

"""

# std
import argparse
import json
import math
import os
import random
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from collections import defaultdict, deque

# pipd
import gurobipy as gp
from gurobipy import GRB

# local

# Helpers
####################################################################################################

def stripped_path_dict(s, d, cdg_nodes, phys_nodes, vcs, W_k):
    return {
        "s": s,
        "d": d,
        "cdg_nodes": tuple(cdg_nodes),
        "phys_nodes": tuple(phys_nodes),
        "vcs": tuple(vcs),
        "W_k": float(W_k),
    }


def _read_matrix_file(path, n):
    """
    Load an n x n numeric matrix (whitespace-separated rows).
    Row i has n values.
    """
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != n:
                raise ValueError(
                    f"{path}: expected {n} columns, got {len(parts)} in row {len(rows)}"
                )
            rows.append([float(x) for x in parts])
    if len(rows) != n:
        raise ValueError(f"{path}: expected {n} rows, got {len(rows)}")
    return rows

def _uniform_matrix(n, value):
    return [[float(value) for _ in range(n)] for _ in range(n)]

def _default_commodities(n):
    return defaultdict(list, {s: [d for d in range(n) if s != d] for s in range(n)})

def _parse_commodities_file(path, n):
    pairs = defaultdict(list)
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"commodities file: expected two ints per line, got {line!r}")
            s, d = int(parts[0]), int(parts[1])
            if not (0 <= s < n and 0 <= d < n):
                raise ValueError(f"commodities file: pair ({s},{d}) out of range for n={n}")
            if s == d:
                continue
            pairs[s].append(d)
    if not pairs:
        raise ValueError("commodities file: no valid pairs")
    return pairs

# Model Creation
####################################################################################################

class CCDGModel:

    # class variables
    VERBOSE = False

    def __init__(self, n_nodes, radix, n_vcs, capacity, demand, commodities, relax_topo_edges=False, relax_turn_edges=False, integral_flow=True, per_source_solves=False, k_paths=1, symmetric_links=False, symmetric_paths=False, allow_vc_trans=False, known_load_links=defaultdict(float), known_load_turns=defaultdict(float), known_lmax=0, hardened_topo_edges=None, hardened_turn_edges=None):

        # problem definition variables
        self.n_nodes = n_nodes
        self.radix = radix
        self.n_vcs = n_vcs
        self.capacity = capacity
        self.demand = demand
        self.commodities = commodities
        self.source_commodities = commodities.keys()
        self.known_lmax = known_lmax # empty list for non-per-source solves or 0th iteration of per-source solves
        self.relax_topo_edges = relax_topo_edges
        self.relax_turn_edges = relax_turn_edges
        self.integral_flow = integral_flow
        self.per_source_solves = per_source_solves
        self.symmetric_links = symmetric_links
        self.symmetric_paths = symmetric_paths
        self.k_paths = k_paths
        self.allow_vc_trans = allow_vc_trans

        # Calculate absolute worst-case big-M based on total demand matrix
        self.total_network_demand = sum(self.demand[s][d] for s in self.commodities for d in self.commodities[s])
        self.iteration_flow_scale = self._iteration_flow_scale_()

        # Flow counts from prior sequential iterations (scale by iteration_flow_scale for load).
        self.known_load_links = known_load_links
        self.known_load_turns = known_load_turns
        self.hardened_topo_edges = set(hardened_topo_edges or ())
        self.hardened_turn_edges = set(hardened_turn_edges or ())
        if self.symmetric_links:
            self.hardened_topo_edges |= {(j, i) for (i, j) in self.hardened_topo_edges}

        # tracking
        self.u_id = 0
        self.cdg_u_to_topo_ijv_map = None
        self.topo_ijv_to_cdg_u_map = None
        self.ss_to_u_conn_map = None
        self.sd_to_u_conn_map = None
        self.uv_turn_set = None
        self.u_to_v_turns = None
        self.v_to_u_turns = None

        # Gurobi/model (object) variables
        self.model = None
        self.vars_topo_adj_mat = None
        self.vars_turn_adj_mat = None
        self.vars_flow = None
        self.vars_mtz = None
        self.var_lmax = None

        print(f"Initialized CCDGModel (Min-Max Load)")
        print("-" * 80)
        print(f"Working on problem with {n_nodes} nodes, {radix} radix, {n_vcs} VCs, {len(commodities)} commodities")
        print(f"Expecting {n_nodes * (n_nodes - 1) * (n_nodes - 2) * n_vcs} turns")
        print(f"Integral flow: {integral_flow}")
        print(f"Total Network Demand (Big-M bound): {self.total_network_demand}")
        print("-" * 80)

    # Iterators
    ####################################################################################################

    def _iter_turns(self):
        for i in range(self.n_nodes):
            for k in range(self.n_nodes):
                if i == k: continue
                for j in range(self.n_nodes):
                    if i == j or j == k: continue
                    for l0 in range(self.n_vcs):
                        for l1 in range(self.n_vcs):
                            if not self.allow_vc_trans and l0 != l1: continue
                            yield ((i, k, l0), (k, j, l1))

    def _iter_topo_edges(self):
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i == j: continue
                yield (i, j)

    def _iter_ccdg_nodes(self):
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i == j: continue
                for l in range(self.n_vcs):
                    yield (i, j, l)

    # CDG to Topology Mapping
    ####################################################################################################

    def _create_cdg_u_to_topo_ijv_maps_(self):
        self.u_id = 0
        self.cdg_u_to_topo_ijv_map = {}
        self.topo_ijv_to_cdg_u_map = {}
        self.ss_to_u_conn_map = defaultdict(list)
        self.sd_to_u_conn_map = defaultdict(list)
        self.uv_turn_set = set()
        self.u_to_v_turns = defaultdict(list)
        self.v_to_u_turns = defaultdict(list)

        for (i, j, l) in self._iter_ccdg_nodes():
            u = self.u_id
            self.u_id += 1
            self.cdg_u_to_topo_ijv_map[u] = (i, j, l)
            self.topo_ijv_to_cdg_u_map[(i, j, l)] = u
            self.ss_to_u_conn_map[i].append(u)
            self.sd_to_u_conn_map[j].append(u)

        self.n_cdg = self.u_id

        for ((i, k, l0), (k, j, l1)) in self._iter_turns():
            u = self.topo_ijv_to_cdg_u_map[(i, k, l0)]
            v = self.topo_ijv_to_cdg_u_map[(k, j, l1)]
            self.uv_turn_set.add((u, v))
            self.u_to_v_turns[u].append(v)
            self.v_to_u_turns[v].append(u)

        if self.VERBOSE:
            for u, (i, j, l) in self.cdg_u_to_topo_ijv_map.items():
                print(f"CDG node {u} maps to topology edge {i}i_{j}j_{l}l")
            for (i, j, l), u in self.topo_ijv_to_cdg_u_map.items():
                print(f"Topology edge {i}i_{j}j_{l}l maps to CDG node {u}")
            for s, u_list in self.ss_to_u_conn_map.items():
                print(f"Super-source {s} has {len(u_list)} CDG nodes: {u_list}")
            for d, u_list in self.sd_to_u_conn_map.items():
                print(f"Destination {d} has {len(u_list)} CDG nodes: {u_list}")
            for u, v_list in self.u_to_v_turns.items():
                print(f"CDG node {u} has {len(v_list)} outgoing turns: {v_list}")

    # Variables
    ####################################################################################################

    def _vars_topo_adj_mat_(self):
        self.vars_topo_adj_mat = {}
        topo_edge_var_type = GRB.BINARY if not self.relax_topo_edges else GRB.CONTINUOUS
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i == j: continue
                if (i, j) in self.hardened_topo_edges:
                    lb, ub, vtype = 1.0, 1.0, GRB.BINARY
                else:
                    lb, ub, vtype = 0.0, 1.0, topo_edge_var_type
                self.vars_topo_adj_mat[(i, j)] = self.model.addVar(lb=lb, ub=ub, vtype=vtype, name=f"m_{i}i_{j}j")

    def _vars_turn_adj_mat_(self):
        self.vars_turn_adj_mat = {}
        turn_edge_var_type = GRB.BINARY if not self.relax_turn_edges else GRB.CONTINUOUS
        for (u, v) in self.uv_turn_set:
            if (u, v) in self.hardened_turn_edges:
                lb, ub, vtype = 1.0, 1.0, GRB.BINARY
            else:
                lb, ub, vtype = 0.0, 1.0, turn_edge_var_type
            self.vars_turn_adj_mat[(u, v)] = self.model.addVar(lb=lb, ub=ub, vtype=vtype, name=f"c_{u}u_{v}v")

    def _vars_flow_(self):
        self.vars_uv_flow = defaultdict(dict)
        self.vars_ss_flow = defaultdict(dict)
        flow_var_type = GRB.INTEGER if self.integral_flow else GRB.CONTINUOUS
        
        for s in self.source_commodities:
            for (u, v) in self.uv_turn_set:
                # Upper bound is the total network demand for robustness
                self.vars_uv_flow[s][(u, v)] = self.model.addVar(lb=0.0, ub=self.total_network_demand, vtype=flow_var_type, name=f"fuv_{s}s_{u}u_{v}v")

        for s, u_list in self.ss_to_u_conn_map.items():
            for u in u_list:
                self.vars_ss_flow[s][u] = self.model.addVar(lb=0.0, ub=self.total_network_demand, vtype=flow_var_type, name=f"fss_{s}s_{u}u")

    def _vars_mtz_(self):
        self.vars_o = {}
        for u in range(self.u_id):
            # self.vars_o[u] = self.model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"o_{u}u")
            self.vars_o[u] = self.model.addVar(lb=0.0, ub=self.n_cdg - 1, vtype=GRB.CONTINUOUS, name=f"o_{u}u")

    def _vars_sink_(self):
        self.vars_sink = defaultdict(dict)
        flow_var_type = GRB.INTEGER if self.integral_flow else GRB.CONTINUOUS
        for s in self.source_commodities:
            for u in range(self.n_cdg):
                self.vars_sink[s][u] = self.model.addVar(lb=0.0, ub=self.total_network_demand, vtype=flow_var_type, name=f"s_{s}s_{u}u")

    # Constraints
    ####################################################################################################

    def _constr_symmetric_links_(self):
        for (i, j) in self._iter_topo_edges():
            self.model.addConstr(self.vars_topo_adj_mat[(i, j)] == self.vars_topo_adj_mat[(j, i)], name=f"sym_link_{i}i_{j}j")

    def _constr_radix_(self):
        for i in range(self.n_nodes):
            self.model.addConstr(gp.quicksum(self.vars_topo_adj_mat[(i, j)] for j in range(self.n_nodes) if i != j) <= self.radix, name=f"r_{i}")

    def _constr_cdg_to_topo_mapping_(self):
        for ((i, k, l0), (k, j, l1)) in self._iter_turns():
            u = self.topo_ijv_to_cdg_u_map[(i, k, l0)]
            v = self.topo_ijv_to_cdg_u_map[(k, j, l1)]
            self.model.addConstr(self.vars_turn_adj_mat[(u, v)] <= self.vars_topo_adj_mat[(i, k)], name=f"c2t_{i}i_{k}k_{j}j_{l0}l0_{l1}l1_1")
            self.model.addConstr(self.vars_turn_adj_mat[(u, v)] <= self.vars_topo_adj_mat[(k, j)], name=f"c2t_{i}i_{k}k_{j}j_{l0}l0_{l1}l1_2")

    def _iteration_flow_scale_(self):
        """Load contributed by each committed flow unit in this subproblem (max concurrent flow)."""
        return max(
            (float(self.demand[s][d]) for s in self.commodities for d in self.commodities[s]),
            default=1.0,
        )

    def _scaled_known_link_load_(self, i, j):
        return self.known_load_links.get((i, j), 0.0) * self.iteration_flow_scale

    def _scaled_known_turn_load_(self, u, v):
        return self.known_load_turns.get((u, v), 0.0) * self.iteration_flow_scale

    def _constr_flow_to_cdg_mapping_(self):
        M_uv = self.total_network_demand  # Big-M bound for new flow in this iteration
        for (u, v) in self.uv_turn_set:
            new_flow = gp.quicksum(self.vars_uv_flow[s][(u, v)] for s in self.source_commodities)
            known_scaled = self._scaled_known_turn_load_(u, v)
            total_flow = new_flow + known_scaled
            self.model.addConstr(
                total_flow <= (known_scaled + M_uv) * self.vars_turn_adj_mat[(u, v)],
                name=f"f2c_{u}u_{v}v",
            )

    def _expr_new_physical_link_flow_(self, i, j):
        """Flow from current decision variables only (excludes known_load from prior iterations)."""
        expr = gp.LinExpr()
        for l in range(self.n_vcs):
            u = self.topo_ijv_to_cdg_u_map[(i, j, l)]
            for s in self.source_commodities:
                expr += self.vars_sink[s][u]
                for v in self.u_to_v_turns[u]:
                    expr += self.vars_uv_flow[s][(u, v)]
        return expr

    def _expr_physical_link_flow_(self, i, j):
        """Total load on directed physical link (i,j): known_flow_count*lambda + new flow."""
        return self._expr_new_physical_link_flow_(i, j) + self._scaled_known_link_load_(i, j)

    def total_link_load_expr(self):
        """Sum of flow over all directed physical links."""
        return gp.quicksum(
            self._expr_physical_link_flow_(i, j) for (i, j) in self._iter_topo_edges()
        )

    def _constr_physical_link_capacity_(self):
        M_ij = self.total_network_demand  # Big-M bound for new flow in this iteration
        for (i, j) in self._iter_topo_edges():
            new_flow = self._expr_new_physical_link_flow_(i, j)
            known_scaled = self._scaled_known_link_load_(i, j)
            total_edge_flow = new_flow + known_scaled

            # C8a: Min-Max Load Definition (cumulative load including prior iterations)
            self.model.addConstr(total_edge_flow <= self.var_lmax, name=f"lmax_{i}i_{j}j")
            
            # C8b: Topology Big-M — prior load plus up to M_ij new flow when edge is on
            self.model.addConstr(
                total_edge_flow <= (known_scaled + M_ij) * self.vars_topo_adj_mat[(i, j)],
                name=f"phys_cap_{i}i_{j}j",
            )

    def _constr_flow_conservation_(self):
        for s in self.source_commodities:
            for u in range(self.n_cdg):
                in_expr = gp.LinExpr()
                out_expr = gp.LinExpr()
                if u in self.ss_to_u_conn_map[s]:
                    in_expr += self.vars_ss_flow[s][u]
                in_expr += gp.quicksum(self.vars_uv_flow[s][(v, u)] for v in self.v_to_u_turns[u])
                out_expr += self.vars_sink[s][u]
                out_expr += gp.quicksum(self.vars_uv_flow[s][(u, v)] for v in self.u_to_v_turns[u])

                self.model.addConstr(in_expr == out_expr, name=f"flow_con_{s}s_{u}u")

    def _constr_super_source_production_(self):
        for s, dests in self.commodities.items():
            tot_demand = sum(self.demand[s][d] for d in dests)
            self.model.addConstr(gp.quicksum(self.vars_ss_flow[s][u] for u in self.ss_to_u_conn_map[s]) == tot_demand, name=f"ss_prod_{s}s")

    def _constr_ccdg_node_consumption_(self):
        for s, dests in self.commodities.items():
            for d in dests:
                if s == d: continue
                self.model.addConstr(gp.quicksum(self.vars_sink[s][u] for u in self.sd_to_u_conn_map[d]) == self.demand[s][d], name=f"sd_con_{s}s_{d}d")

    def _constr_mtz_acyclicality_(self):
        for (u, v) in self.uv_turn_set:
            self.model.addConstr(self.vars_o[u] - self.vars_o[v] + self.n_cdg * self.vars_turn_adj_mat[(u, v)] <= self.n_cdg - 1, name=f"mtz_{u}u_{v}v")

    def _constr_global_valid_inequalities_(self):
        """
        Valid inequalities (cuts) to drastically tighten the LP relaxation
        by bounding the volumetric flow against the Moore limit.
        """
        # 1. Calculate Total Hop Volume Expression
        # Every time flow enters a CDG node from a super-source, or traverses a turn, 
        # it consumes one physical edge hop.
        total_hop_volume = gp.LinExpr()
        for s in self.source_commodities:
            for u in self.ss_to_u_conn_map[s]:
                total_hop_volume += self.vars_ss_flow[s][u]
            for (u, v) in self.uv_turn_set:
                total_hop_volume += self.vars_uv_flow[s][(u, v)]

        # 2. Maximum Possible Network Capacity Bound
        # The sum of all flow cannot exceed L_max * (max possible physical edges)
        max_active_edges = self.n_nodes * self.radix
        self.model.addConstr(
            total_hop_volume <= self.var_lmax * max_active_edges, 
            name="global_lmax_capacity_bound"
        )

        # 3. Demand-Aware Moore Bound Lower Limit
        # Calculate the absolute minimum hop-volume for a perfect tree of degree `radix`
        min_total_hop_volume = 0
        for s in self.source_commodities:
            # Sort demands descending to optimistically place heaviest traffic 1-hop away
            demands = sorted([self.demand[s][d] for d in self.commodities[s]], reverse=True)
            
            current_dist = 1
            nodes_at_dist = self.radix
            d_idx = 0
            
            while d_idx < len(demands):
                take = min(len(demands) - d_idx, nodes_at_dist)
                for _ in range(take):
                    min_total_hop_volume += demands[d_idx] * current_dist
                    d_idx += 1
                current_dist += 1
                nodes_at_dist *= (self.radix - 1)

        self.model.addConstr(
            total_hop_volume >= min_total_hop_volume, 
            name="moore_bound_min_hops"
        )

    def _calculate_theoretical_min_(self):
        """
        Calculates the absolute minimum makespan for an all-to-all based on the 
        best possible theoretical graph diameter and volume (Moore Bound).
        """

        num_nodes = self.n_nodes
        radix = self.radix
        # take maximum to be conservative
        bandwidth = max( [ max(cap_row) for cap_row in self.capacity ] )

        # 1. Calculate minimum hop volume from the perspective of one node
        unreached_nodes = num_nodes - 1
        current_distance = 1
        nodes_at_current_distance = radix
        total_hops_for_one_node = 0
        
        while unreached_nodes > 0:
            # We can only reach as many nodes as are left, or the capacity of this "ring"
            nodes_to_reach = min(unreached_nodes, nodes_at_current_distance)
            total_hops_for_one_node += nodes_to_reach * current_distance
            
            unreached_nodes -= nodes_to_reach
            current_distance += 1
            nodes_at_current_distance *= (radix - 1)
            
        # Total volume of all chunks traveling their shortest possible paths
        n_sources = len(self.source_commodities)
        total_network_hop_volume = total_hops_for_one_node * (n_sources)

        # 2. Calculate maximum physical network capacity per time step
        # Max undirected edges = floor((N * R) / 2)
        max_undirected_edges = (num_nodes * radix) // 2
        
        # Because links are bidirectional, each undirected edge yields 2 directed links.
        max_directed_links = max_undirected_edges * 2
        max_capacity_per_step = max_directed_links * bandwidth
        
        # 3. The Bound: Volume / Capacity
        hop_bound = math.ceil(total_network_hop_volume / max_capacity_per_step)
        
        # We must also respect the standard injection bound (whichever is higher)
        injection_bound = math.ceil((num_nodes - 1) / radix)

        print(f"For source commodities {self.source_commodities}")
        print(f'hop_bound = {hop_bound}, injection_bound = {injection_bound}')
        print(f'self.known_lmax = {self.known_lmax}')
        # return hop_bound
        return max(hop_bound, injection_bound, self.known_lmax)

    def _constr_heuristic_lower_bound_(self):
        theoretical_min = self._calculate_theoretical_min_()
        self.model.addConstr(self.var_lmax >= theoretical_min, name="heuristic_lower_bound")


    # Build Model
    ####################################################################################################

    def build_model_(self):
        self._create_cdg_u_to_topo_ijv_maps_()
        print("-" * 80)
        print("Building model...")

        self.model = gp.Model("ccdg_minmax_model")

        print("Building variables...")
        # L_max replaces MCF lambda
        self.var_lmax = self.model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="L_max")
        self._vars_topo_adj_mat_()
        self._vars_turn_adj_mat_()
        self._vars_flow_()
        self._vars_mtz_()
        self._vars_sink_()
        print("Completed variable creation")

        print("Building constraints...")
        self._constr_radix_()
        if self.symmetric_links:
            self._constr_symmetric_links_()
        self._constr_cdg_to_topo_mapping_()
        self._constr_flow_to_cdg_mapping_()
        self._constr_physical_link_capacity_()
        self._constr_flow_conservation_()
        self._constr_super_source_production_()
        self._constr_ccdg_node_consumption_()
        self._constr_mtz_acyclicality_()

        # heuristic lower bound on L_max

        self._constr_heuristic_lower_bound_()

        # minimums for LP relaxation
        # self._constr_global_valid_inequalities_()
        print("Completed constraint building")

        print("Building objective...")
        self.model.setObjective(self.var_lmax, GRB.MINIMIZE)
        print("Completed objective building")
        print("-" * 80)

        self.model.write("model.lp")
        print(f"Wrote model to model.lp")

    def write_model_(self, path):
        self.model.write(path)
        print(f"Wrote model to {path}")

# Solving and Result Extraction
####################################################################################################

class CCDGOptimizer:

    VERBOSE = False
    epsilon = 1e-6

    def __init__(self, ccdg_model, model_params):
        self.ccdg_model = ccdg_model
        self.model = ccdg_model.model
        self.n_nodes = ccdg_model.n_nodes
        self.n_cdg = ccdg_model.n_cdg
        self.radix = ccdg_model.radix
        self.n_vcs = ccdg_model.n_vcs
        self.symmetric_links = ccdg_model.symmetric_links
        self.k_paths = ccdg_model.k_paths
        self.commodities = ccdg_model.commodities
        self.beta = model_params.get("beta", 0.5)
        self.weight_type = model_params.get("weight_type", "max")
        self.choice_type = model_params.get("choice_type", "strict")
        self.k_path_traf_policy = model_params.get("k_path_traf_policy", "equal")

        self.model_params = model_params
        self._rng = random.Random(model_params.get("random_seed"))

        self.capacity = ccdg_model.capacity
        self.demand = ccdg_model.demand

        self.cdg_u_to_topo_ijv_map = ccdg_model.cdg_u_to_topo_ijv_map
        self.ss_to_u_conn_map = ccdg_model.ss_to_u_conn_map
        self.sd_to_u_conn_map = ccdg_model.sd_to_u_conn_map
        self.v_to_u_turns = ccdg_model.v_to_u_turns
        self.integral_flow = ccdg_model.integral_flow

        if self.model_params.get("silent"):
            self.model.Params.OutputFlag = 0
        if self.model_params.get("threads") is not None:
            self.model.Params.Threads = int(self.model_params["threads"])
        if self.model_params.get("time_limit") is not None:
            self.model.Params.TimeLimit = float(self.model_params["time_limit"])
        obj_gap_pct = self.model_params.get("obj_gap_pct")
        mip_gap = self.model_params.get("mip_gap")
        if obj_gap_pct is not None:
            self.model.Params.MIPGap = float(obj_gap_pct) / 100.0
        elif mip_gap is not None:
            self.model.Params.MIPGap = float(mip_gap)

        self.topo_adj_mat_vals = None
        self.turn_adj_mat_vals = None
        self.flow_vals = None
        self.lmax = None
        self.total_link_load = None
        self.lmax_obj = None
        self.hierarchical_objectives = model_params.get("hierarchical_objectives", False)

        self.topology = None
        self.paths_w_vcs = None

    def _eval_total_link_load(self):
        total = 0.0
        for (i, j) in self.ccdg_model._iter_topo_edges():
            total += self.ccdg_model._expr_physical_link_flow_(i, j).getValue()
        return total

    def _finalize_solve_status(self, phase_label):
        if self.model.Status == GRB.INFEASIBLE:
            self.model.computeIIS()
            self.model.write("iis.ilp")
            print("IIS details written to iis.ilp")
            raise RuntimeError(f"Infeasible solution ({phase_label}). Status={self.model.Status}")
        if self.model.SolCount == 0:
            raise RuntimeError(f"No solution available ({phase_label}). Status={self.model.Status}")
        if self.model.Status != GRB.OPTIMAL:
            print(f"Non-optimal solution ({phase_label}). Status={self.model.Status}")

    def solve(self):
        print("Phase 1: minimize L_max (primary objective)")
        self.model.optimize()
        print(f"Gurobi ended with status {self.model.Status}")
        self._finalize_solve_status("phase 1")

        var_lmax = self.model.getVarByName("L_max")
        self.lmax_obj = var_lmax.X
        self.lmax = self.lmax_obj
        self.total_link_load = self._eval_total_link_load()
        print(f"Phase 1: L_max={self.lmax_obj:.6g}  total_link_load={self.total_link_load:.6g}")

        if self.hierarchical_objectives:
            lmax_opt = self.lmax_obj
            self.model.addConstr(
                var_lmax <= lmax_opt + self.epsilon,
                name="hier_fix_lmax",
            )
            total_load_expr = self.ccdg_model.total_link_load_expr()
            self.model.setObjective(total_load_expr, GRB.MINIMIZE)
            print(f"Phase 2: minimize total link load subject to L_max <= {lmax_opt:.6g}")
            self.model.optimize()
            print(f"Gurobi ended with status {self.model.Status}")
            self._finalize_solve_status("phase 2")
            self.lmax = var_lmax.X
            self.total_link_load = total_load_expr.getValue()
            print(
                f"Phase 2: L_max={self.lmax:.6g}  total_link_load={self.total_link_load:.6g} "
                f"(primary optimum preserved)"
            )

        self.objval = self.lmax
        return self.model

    def dump_var_vals_to_file_(self, out_file_path):
        with open(out_file_path, "w", encoding="utf-8") as f:
            for v in sorted(self.model.getVars(), key=lambda x: x.VarName):
                f.write(f"{v.VarName} = {v.X}\n")

    def _get_solved_values(self):
        topo_adj_mat_vals = {(i, j): var.X for (i, j), var in self.ccdg_model.vars_topo_adj_mat.items()}
        turn_adj_mat_vals = {(u, v): var.X for (u, v), var in self.ccdg_model.vars_turn_adj_mat.items() if not isinstance(var, int)}
        flow_uv_vals = {s: {(u, v): var.X for (u, v), var in self.ccdg_model.vars_uv_flow[s].items()} for s in self.ccdg_model.vars_uv_flow.keys()}
        flow_ss_vals = {s: {u: var.X for u, var in self.ccdg_model.vars_ss_flow[s].items()} for s in self.ccdg_model.vars_ss_flow.keys()}
        sink_vals = {s: {u: var.X for u, var in self.ccdg_model.vars_sink[s].items()} for s in self.ccdg_model.vars_sink.keys()}
        lmax = self.model.getVarByName("L_max").X

        return lmax, topo_adj_mat_vals, turn_adj_mat_vals, flow_uv_vals, flow_ss_vals, sink_vals

    def extract_resultant_values(self):
        self.lmax, self.topo_adj_mat_vals, self.turn_adj_mat_vals, self.flow_uv_vals, self.flow_ss_vals, self.sink_vals = self._get_solved_values()
        if self.total_link_load is None:
            self.total_link_load = self._eval_total_link_load()
        return {
            "L_max": self.lmax,
            "total_link_load": self.total_link_load,
            "topo_adj_mat_vals": self.topo_adj_mat_vals,
            "turn_adj_mat_vals": self.turn_adj_mat_vals,
            "flow_uv_vals": self.flow_uv_vals,
            "flow_ss_vals": self.flow_ss_vals,
            "sink_vals": self.sink_vals,
        }

    def harden_results(self):
        topology = self.harden_topology(self.topo_adj_mat_vals)
        paths_pkg = self.harden_paths(self.flow_uv_vals, self.flow_ss_vals, self.sink_vals)

        paths_by_sd = paths_pkg["paths_by_sd"]
        link_load = paths_pkg["current_load"]
        routing_table = paths_pkg["route_table"]
        vc_table = paths_pkg["vc_table"]

        return topology, paths_by_sd, link_load, routing_table, vc_table

    def harden_topology(self, topo_adj_mat_vals):
        n_nodes = self.n_nodes
        topo_adj_mat = [[0] * n_nodes for _ in range(n_nodes)]
        cur_radix = [0 for _ in range(n_nodes)]
        for (i, j), val in topo_adj_mat_vals.items():
            if val > self.epsilon and cur_radix[i] < self.radix:
                if topo_adj_mat[i][j] == 0:
                    cur_radix[i] += 1
                    topo_adj_mat[i][j] = 1
                if self.symmetric_links and topo_adj_mat[j][i] == 0:
                    cur_radix[j] += 1
                    topo_adj_mat[j][i] = 1
        return topo_adj_mat

    def _cdg_to_phys(self, cdg_nodes):
        ij_map = self.cdg_u_to_topo_ijv_map
        phys_nodes, vcs = [], []
        for t, u in enumerate(cdg_nodes):
            i, j, l = ij_map[u]
            if t == 0:
                phys_nodes = [i, j]
            else:
                phys_nodes.append(j)
            vcs.append(l)
        return phys_nodes, vcs

    def _decompose_paths_for_source(self, s, flow_uv_vals, flow_ss_vals, sink_vals):
        """Iteratively strip min-flow paths; tolerates split flow at CDG nodes."""
        eps = self.epsilon
        ij_map = self.cdg_u_to_topo_ijv_map
        ss_nodes = set(self.ss_to_u_conn_map[s])
        vpred = self.v_to_u_turns
        n_cdg = self.n_cdg

        def cdg_to_phys(cdg_nodes):
            phys_nodes, vcs = [], []
            for t, u in enumerate(cdg_nodes):
                i, j, l = ij_map[u]
                if t == 0:
                    phys_nodes = [i, j]
                else:
                    phys_nodes.append(j)
                vcs.append(l)
            return phys_nodes, vcs

        def backtrack(u_end, flow_uv, flow_ss):
            back, cur = [u_end], u_end
            for _ in range(n_cdg + 3):
                if cur in ss_nodes and flow_ss.get(cur, 0.0) > eps:
                    return list(reversed(back))
                preds = sorted(
                    v for v in vpred[cur]
                    if flow_uv.get((v, cur), 0.0) > eps
                )
                if not preds:
                    return None
                cur = preds[0]
                back.append(cur)
            return None

        def strip_flow(cdg_fwd, flow_uv, flow_ss, sink):
            u_start, u_end = cdg_fwd[0], cdg_fwd[-1]
            vals = [flow_ss.get(u_start, 0.0), sink.get(u_end, 0.0)]
            vals.extend(flow_uv.get((a, b), 0.0) for a, b in zip(cdg_fwd, cdg_fwd[1:]))
            W = min(vals)
            if W <= eps:
                return 0.0
            flow_ss[u_start] = flow_ss.get(u_start, 0.0) - W
            sink[u_end] = sink.get(u_end, 0.0) - W
            for a, b in zip(cdg_fwd, cdg_fwd[1:]):
                flow_uv[(a, b)] = flow_uv.get((a, b), 0.0) - W
            return W

        flow_uv = {k: float(v) for k, v in flow_uv_vals.get(s, {}).items()}
        flow_ss = {u: float(v) for u, v in flow_ss_vals.get(s, {}).items()}
        sink = {u: float(sink_vals.get(s, {}).get(u, 0.0)) for u in range(n_cdg)}

        out = []
        max_iter = max(1000, 50 * n_cdg * max(1, len(flow_uv)))
        for _ in range(max_iter):
            terminals = sorted(u for u in range(n_cdg) if sink.get(u, 0.0) > eps)
            if not terminals:
                break
            cdg_fwd = None
            for u_end in terminals:
                chain = backtrack(u_end, flow_uv, flow_ss)
                if chain is not None:
                    cdg_fwd = chain
                    break
            if cdg_fwd is None:
                break
            _, d, _ = ij_map[cdg_fwd[-1]]
            phys_nodes, vcs = cdg_to_phys(cdg_fwd)
            W = strip_flow(cdg_fwd, flow_uv, flow_ss, sink)
            if W > eps:
                out.append(stripped_path_dict(s, d, cdg_fwd, phys_nodes, vcs, W))
        return out

    @staticmethod
    def _plain_nested_dict(d):
        if isinstance(d, defaultdict):
            return {k: CCDGOptimizer._plain_nested_dict(v) for k, v in d.items()}
        return d

    def harden_paths(self, flow_uv_vals, flow_ss_vals, sink_vals):
        if not self.integral_flow:
            raise RuntimeError(
                "harden_paths requires integral flow; use integer flow (default) or the MCF optimizer"
            )

        eps = self.epsilon
        commodities = self.commodities
        current_load = defaultdict(float)
        paths_by_sd = {}
        route_table = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        vc_table = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        for s, dests in commodities.items():
            decomposed = self._decompose_paths_for_source(
                s, flow_uv_vals, flow_ss_vals, sink_vals,
            )
            by_sd = defaultdict(list)
            for rec in decomposed:
                d = rec["d"]
                if d in dests and d != s:
                    by_sd[(s, d)].append(rec)

            for d in dests:
                if s == d:
                    continue
                paths_for_sd = by_sd.get((s, d), [])
                if not paths_for_sd:
                    raise RuntimeError(f"no decomposed CDG path for commodity ({s},{d})")

                W_dem = float(self.demand[s][d])
                total_W = sum(p["W_k"] for p in paths_for_sd)
                if abs(total_W - W_dem) > eps:
                    raise RuntimeError(
                        f"decomposed flow {total_W} != demand {W_dem} for commodity ({s},{d})"
                    )

                hardened_list = []
                for path_idx, rec in enumerate(paths_for_sd):
                    phys_nodes = list(rec["phys_nodes"])
                    vcs = list(rec["vcs"])
                    alloc = rec["W_k"]
                    for h in range(len(phys_nodes) - 1):
                        i, j = phys_nodes[h], phys_nodes[h + 1]
                        current_load[(i, j)] += alloc
                        r, nxt, vc = i, j, vcs[h] if h < len(vcs) else 0
                        rl = route_table[r][s][d]
                        while len(rl) <= path_idx:
                            rl.append(None)
                        rl[path_idx] = nxt
                        vl = vc_table[r][s][d]
                        while len(vl) <= path_idx:
                            vl.append(None)
                        vl[path_idx] = vc
                    hardened_list.append({
                        "phys_nodes": phys_nodes,
                        "vcs": vcs,
                        "cdg_nodes": list(rec["cdg_nodes"]),
                        "allocation": alloc,
                        "W_k": alloc,
                        "S_k": alloc,
                    })
                paths_by_sd[(s, d)] = hardened_list

        link_utilization = {}
        overflow_links = []
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i == j:
                    continue
                cap_ij = self.capacity[i][j]
                if cap_ij <= 0:
                    continue
                ld = current_load.get((i, j), 0.0)
                u = ld / cap_ij
                link_utilization[(i, j)] = u
                if u > 1.0 + eps:
                    overflow_links.append((i, j, u))

        out = {
            "paths_by_sd": dict(paths_by_sd),
            "route_table": self._plain_nested_dict(route_table),
            "vc_table": self._plain_nested_dict(vc_table),
            "link_utilization": dict(link_utilization),
            "overflow_links": overflow_links,
            "current_load": dict(current_load),
        }
        self.paths_w_vcs = out
        return out

    @staticmethod
    def extract_hardened_topology_from_paths(paths_by_sd, symmetric_links=False):
        edges = set()
        for paths in paths_by_sd.values():
            for p in paths:
                phys = p["phys_nodes"]
                for h in range(len(phys) - 1):
                    i, j = phys[h], phys[h + 1]
                    edges.add((i, j))
                    if symmetric_links:
                        edges.add((j, i))
        return edges

    @staticmethod
    def extract_hardened_turns_from_paths(paths_by_sd):
        turns = set()
        for paths in paths_by_sd.values():
            for p in paths:
                cdg_fwd = p["cdg_nodes"]
                for a, b in zip(cdg_fwd, cdg_fwd[1:]):
                    turns.add((a, b))
        return turns

    @staticmethod
    def _endpoint_nodes_from_turns(hardened_turn_edges, cdg_u_to_topo_ijv_map, root):
        """Physical endpoints k of CDG turns (u,v) where u=(i,k,*), v=(k,*,*)."""
        nodes = set()
        for u, v in hardened_turn_edges:
            i, k, _ = cdg_u_to_topo_ijv_map[u]
            # _, _, _ = cdg_u_to_topo_ijv_map[v]

            if i == root:
                nodes.add(k)
        return nodes

    @staticmethod
    def _reversed_cdg_turn_edge(u, v, cdg_u_to_topo_ijv_map, topo_ijv_to_cdg_u_map, uv_turn_set):
        """
        Mirror a physical turn in the CDG.

        Forward u->v is channel arc (i,k,l0) then (k,j,l1). The spanning-tree
        reverse is v'->u' = (j,k,l1) then (k,i,l0), not the invalid CDG swap (v,u).
        """
        i, k, l0 = cdg_u_to_topo_ijv_map[u]
        k2, j, l1 = cdg_u_to_topo_ijv_map[v]
        if k != k2:
            return None
        v_prime = topo_ijv_to_cdg_u_map[(j, k, l1)]
        u_prime = topo_ijv_to_cdg_u_map[(k, i, l0)]
        rev = (v_prime, u_prime)
        if rev not in uv_turn_set:
            return None
        return rev

    def expand_spanning_tree_turns_at_root(self, root, path_turn_edges, topo_adj_mat):
        """
        After the first source solve: complete the implicit spanning-tree CDG by
        1) adding the physical reverse of each path turn, and
        2) hardening root-pivot turns ((i, root, l0), (root, j, l1)) for every
           pair of physical endpoints i,j used on path turns.
        """
        cm = self.ccdg_model
        ij_map = self.cdg_u_to_topo_ijv_map
        ijv_to_u = cm.topo_ijv_to_cdg_u_map
        uv_turn_set = cm.uv_turn_set

        expanded = set(path_turn_edges)
        for u, v in path_turn_edges:
            rev = self._reversed_cdg_turn_edge(u, v, ij_map, ijv_to_u, uv_turn_set)
            if rev is not None:
                expanded.add(rev)


        # endpoint_nodes = self._endpoint_nodes_from_turns(path_turn_edges, ij_map, root)
        # endpoint_nodes.remove(root)
        endpoint_nodes = [i for i in range(self.n_nodes) if (root, i) in topo_adj_mat]
        # endpoint_nodes = [i for i, v in enumerate(topo_adj_mat[root]) if v == 1]
        # endpoint_nodes.discard(root)

        for i in endpoint_nodes:
            for j in endpoint_nodes:
                if i == j:
                    continue
                for l0 in range(self.n_vcs):
                    for l1 in range(self.n_vcs):
                        if not cm.allow_vc_trans and l0 != l1:
                            continue
                        u = ijv_to_u[(i, root, l0)]
                        v = ijv_to_u[(root, j, l1)]
                        if (u, v) in uv_turn_set:
                            expanded.add((u, v))

                            # print(f'Adding root-pivot turn {u} -> {v} = {ij_map[u]} -> {ij_map[v]} ?')

        return expanded

    @staticmethod
    def accumulate_link_flow_counts(paths_by_sd, demand, known_load_links):
        """Increment per-link flow counts (each unit contributes lambda load in later iterations)."""
        for (s, d), paths in paths_by_sd.items():
            dem = float(demand[s][d])
            if dem <= 0:
                continue
            for p in paths:
                flow_units = float(p.get("W_k", p.get("allocation", 0.0))) / dem
                phys = p["phys_nodes"]
                for h in range(len(phys) - 1):
                    known_load_links[(phys[h], phys[h + 1])] += flow_units

    @staticmethod
    def accumulate_turn_flow_counts(paths_by_sd, demand, known_load_turns):
        """Increment per-turn flow counts (each unit contributes lambda load in later iterations)."""
        for (s, d), paths in paths_by_sd.items():
            dem = float(demand[s][d])
            if dem <= 0:
                continue
            for p in paths:
                flow_units = float(p.get("W_k", p.get("allocation", 0.0))) / dem
                cdg_fwd = p["cdg_nodes"]
                for a, b in zip(cdg_fwd, cdg_fwd[1:]):
                    known_load_turns[(a, b)] += flow_units

    @staticmethod
    def absolute_link_load_from_counts(known_load_links, flow_scale):
        return {k: v * flow_scale for k, v in known_load_links.items()}

    @staticmethod
    def absolute_link_load_from_paths(paths_by_sd):
        """Sum absolute per-link load over all hardened paths."""
        loads = defaultdict(float)
        for paths in paths_by_sd.values():
            for p in paths:
                alloc = float(p.get("W_k", p.get("allocation", 0.0)))
                phys = p["phys_nodes"]
                for h in range(len(phys) - 1):
                    loads[(phys[h], phys[h + 1])] += alloc
        return dict(loads)

    @staticmethod
    def merge_routing_tables(dst_route, dst_vc, src_route, src_vc):
        for r, by_s in src_route.items():
            for s, by_d in by_s.items():
                for d, paths in by_d.items():
                    dst_route[r][s][d] = list(paths)
        for r, by_s in src_vc.items():
            for s, by_d in by_s.items():
                for d, vcs in by_d.items():
                    dst_vc[r][s][d] = list(vcs)

    def topology_from_hardened_edges(self, hardened_topo_edges):
        topo_adj_mat_vals = {(i, j): 1.0 for (i, j) in hardened_topo_edges}
        return self.harden_topology(topo_adj_mat_vals)

# Per-source debug visualization
####################################################################################################

def _layout_node_positions(n_nodes):
    """Circular layout with radius scaled to node count for readable spacing."""
    if n_nodes <= 1:
        return {0: (0.0, 0.0)}
    radius = max(6.0, 0.45 * n_nodes)
    pos = {}
    for i in range(n_nodes):
        theta = 2.0 * math.pi * i / n_nodes - math.pi / 2.0
        pos[i] = (radius * math.cos(theta), radius * math.sin(theta))
    return pos


def _cdg_node_label(u, cdg_u_to_topo_ijv_map):
    i, j, l = cdg_u_to_topo_ijv_map[u]
    return f"{i}>{j},{l}"


def _decode_single_cdg_turn(u, v, cdg_u_to_topo_ijv_map):
    """
    Decode hardened CDG edge (u, v) to physical turn i -> k -> j with VCs l0, l1.

    Forward CDG edges use u=(i,k,l0), v=(k,j,l1). Reverse hardened edges (v,u)
    from spanning-tree expansion store the same physical turn with swapped CDG ids.
    """
    i, k, l0 = cdg_u_to_topo_ijv_map[u]
    k2, j, l1 = cdg_u_to_topo_ijv_map[v]
    if k == k2:
        return i, k, j, l0, l1, False

    i2, k3, l0b = cdg_u_to_topo_ijv_map[v]
    k4, j2, l1b = cdg_u_to_topo_ijv_map[u]
    if k3 == k4:
        return i2, k3, j2, l0b, l1b, True

    raise ValueError(
        f"CDG edge ({u},{v}) is not a valid turn: "
        f"forward pivots {k}!={k2}, reverse pivots {k3}!={k4}"
    )


def _plot_topology_debug(out_path, n_nodes, topo_edges, title):
    """Physical topology: routers as nodes, hardened directed links as edges."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    pos = _layout_node_positions(n_nodes)
    layout_radius = max(6.0, 0.45 * n_nodes)
    node_inset = min(0.06, 0.45 / layout_radius)

    fig_w = max(8.0, min(20.0, 0.55 * n_nodes))
    fig, ax = plt.subplots(figsize=(fig_w, fig_w))
    ax.set_title(title, fontsize=11)

    ax.scatter(
        [pos[i][0] for i in range(n_nodes)],
        [pos[i][1] for i in range(n_nodes)],
        s=360, c="#b8e0b8", edgecolors="black", zorder=2,
    )
    for i in range(n_nodes):
        ax.text(pos[i][0], pos[i][1], str(i), ha="center", va="center", fontsize=9, fontweight="bold", zorder=3)

    for i, j in sorted(topo_edges):
        pi, pj = pos[i], pos[j]
        p0 = (
            pi[0] + node_inset * (pj[0] - pi[0]),
            pi[1] + node_inset * (pj[1] - pi[1]),
        )
        p1 = (
            pj[0] + node_inset * (pi[0] - pj[0]),
            pj[1] + node_inset * (pi[1] - pj[1]),
        )
        ax.add_patch(
            FancyArrowPatch(
                p0, p1, arrowstyle="-|>", color="#2b5c8a",
                linewidth=1.8, mutation_scale=14, zorder=1,
            )
        )

    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _check_cdg_dag(turn_edges):
    """Return (is_dag, cycle_edges) for the hardened CDG turn subgraph."""
    import networkx as nx

    G = nx.DiGraph()
    for u, v in turn_edges:
        G.add_edge(u, v)
    if G.number_of_edges() == 0:
        return True, None
    if nx.is_directed_acyclic_graph(G):
        return True, None
    try:
        cycle = nx.find_cycle(G, orientation="original")
    except nx.NetworkXNoCycle:
        cycle = None
    return False, cycle


def _plot_cdg_debug(out_path, turn_edges, cdg_u_to_topo_ijv_map, title):
    """CDG subgraph: channel nodes as vertices, hardened turns as directed edges."""
    import matplotlib.pyplot as plt
    import networkx as nx

    G = nx.DiGraph()
    for u, v in turn_edges:
        G.add_edge(u, v)

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_title(title, fontsize=11)

    if G.number_of_edges() == 0:
        ax.text(0.5, 0.5, "no CDG turns", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    n = max(G.number_of_nodes(), 1)
    pos = nx.circular_layout(G, scale=max(3.0, 0.35 * n))

    node_labels = {u: _cdg_node_label(u, cdg_u_to_topo_ijv_map) for u in G.nodes}
    edge_labels = {}
    for u, v in G.edges:
        _, _, l0 = cdg_u_to_topo_ijv_map[u]
        _, _, l1 = cdg_u_to_topo_ijv_map[v]
        edge_labels[(u, v)] = f"{l0}/{l1}"

    nx.draw_networkx_nodes(G, pos, node_color="#f7d6d6", edgecolors="black", node_size=500, ax=ax)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=6, ax=ax)
    nx.draw_networkx_edges(
        G, pos, edge_color="#c0392b", arrows=True, arrowsize=12,
        arrowstyle="-|>", width=1.2, connectionstyle="arc3,rad=0.05", ax=ax,
    )
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=5, ax=ax)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_solve_values(f, results, cdg_u_to_topo_ijv_map, epsilon=1e-6):
    f.write("\n# solved topology edge values (i j val)\n")
    for (i, j), val in sorted(results["topo_adj_mat_vals"].items()):
        if val > epsilon:
            f.write(f"topo {i} {j} {val}\n")
    f.write("\n# solved CDG turn edge values (u v val)\n")
    for (u, v), val in sorted(results["turn_adj_mat_vals"].items()):
        if val > epsilon:
            f.write(f"turn {u} {v} {val}\n")
    f.write("\n# positive turn flows fuv (s u v val)\n")
    for s, flows in sorted(results["flow_uv_vals"].items()):
        for (u, v), val in sorted(flows.items()):
            if val > epsilon:
                u_lbl = _cdg_node_label(u, cdg_u_to_topo_ijv_map)
                v_lbl = _cdg_node_label(v, cdg_u_to_topo_ijv_map)
                f.write(f"fuv s={s} {u} {v} {val}  ({u_lbl} -> {v_lbl})\n")
    f.write("\n# positive source flows fss (s u val)\n")
    for s, flows in sorted(results["flow_ss_vals"].items()):
        for u, val in sorted(flows.items()):
            if val > epsilon:
                f.write(
                    f"fss s={s} {u} {val}  {_cdg_node_label(u, cdg_u_to_topo_ijv_map)}\n"
                )
    f.write("\n# positive sink flows (s u d val)\n")
    for s, sinks in sorted(results["sink_vals"].items()):
        for u, val in sorted(sinks.items()):
            if val > epsilon:
                _, d, _ = cdg_u_to_topo_ijv_map[u]
                f.write(
                    f"sink s={s} u={u} d={d} val={val}  "
                    f"{_cdg_node_label(u, cdg_u_to_topo_ijv_map)}\n"
                )


def _iter_debug_stem(base_name, idx, source, dest=None):
    stem = f"{base_name}_iter{idx:02d}_src{source}"
    if dest is not None:
        stem += f"_dst{dest}"
    return stem


def write_per_source_solve_debug(out_dir, base_name, idx, source, results, cdg_u_to_topo_ijv_map, dest=None):
    """After solve, before harden: dump solved variable values for debugging."""
    _ensure_parent_dir(os.path.join(out_dir, "placeholder"))
    stem = _iter_debug_stem(base_name, idx, source, dest)
    txt_path = os.path.join(out_dir, stem + "_after_solve.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"iteration={idx} source={source}\n")
        if dest is not None:
            f.write(f"destination={dest}\n")
        f.write(f"iter_L_max={results['L_max']}\n")
        if results.get("total_link_load") is not None:
            f.write(f"total_link_load={results['total_link_load']}\n")
        _write_solve_values(f, results, cdg_u_to_topo_ijv_map)
    print(f"Wrote {txt_path}")


def _write_turn_list(f, turn_edges, cdg_u_to_topo_ijv_map, header):
    f.write(header + "\n")
    for u, v in sorted(turn_edges):
        u_lbl = _cdg_node_label(u, cdg_u_to_topo_ijv_map)
        v_lbl = _cdg_node_label(v, cdg_u_to_topo_ijv_map)
        _, _, l0 = cdg_u_to_topo_ijv_map[u]
        _, _, l1 = cdg_u_to_topo_ijv_map[v]
        try:
            i, k, j, _, _, _ = _decode_single_cdg_turn(u, v, cdg_u_to_topo_ijv_map)
            phys = f"phys {i}>{k}>{j}"
        except ValueError:
            phys = "phys n/a"
        f.write(f"cdg {u} {v}  {phys}  vc {l0}/{l1}  nodes {u_lbl} -> {v_lbl}\n")


def write_per_source_harden_debug(
    out_dir,
    base_name,
    idx,
    source,
    n_nodes,
    iter_topo_edges,
    cum_topo_edges,
    iter_turn_edges,
    cum_turn_edges,
    cdg_u_to_topo_ijv_map,
    iter_lmax,
    cumulative_max_load,
    solve_results=None,
    dest=None,
):
    """After hardening: write topology + CDG plots for iteration-only and cumulative state."""
    _ensure_parent_dir(os.path.join(out_dir, "placeholder"))
    stem = _iter_debug_stem(base_name, idx, source, dest)
    txt_path = os.path.join(out_dir, stem + "_after_harden.txt")

    meta = f"iter {idx} source {source}  L_max={iter_lmax:.4g}  cum_load={cumulative_max_load:.4g}"
    cum_cdg_is_dag, cum_cdg_cycle = _check_cdg_dag(cum_turn_edges)
    dag_tag = "DAG" if cum_cdg_is_dag else "NOT A DAG"
    plots = (
        ("topo", "iteration", iter_topo_edges, f"Topology added ({meta})"),
        ("topo", "cumulative", cum_topo_edges, f"Topology cumulative ({meta})"),
        ("cdg", "iteration", iter_turn_edges, f"CDG turns added ({meta})"),
        ("cdg", "cumulative", cum_turn_edges, f"CDG turns cumulative ({meta})  [{dag_tag}]"),
    )

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"iteration={idx} source={source}\n")
        if dest is not None:
            f.write(f"destination={dest}\n")
        f.write(f"iter_L_max={iter_lmax}\n")
        f.write(f"cumulative_max_load={cumulative_max_load}\n")
        f.write(f"n_topo_added={len(iter_topo_edges)} n_topo_cumulative={len(cum_topo_edges)}\n")
        f.write(f"n_cdg_added={len(iter_turn_edges)} n_cdg_cumulative={len(cum_turn_edges)}\n")
        f.write(f"cdg_cumulative_is_dag={cum_cdg_is_dag}\n")
        if cum_cdg_cycle:
            f.write("cdg_cumulative_cycle:\n")
            for u, v, _key in cum_cdg_cycle:
                f.write(f"  cdg {u} {v}\n")
        f.write("\n# topology added this iteration (i j)\n")
        for i, j in sorted(iter_topo_edges):
            f.write(f"edge {i} {j}\n")
        f.write("\n# topology cumulative (i j)\n")
        for i, j in sorted(cum_topo_edges):
            f.write(f"edge {i} {j}\n")
        _write_turn_list(f, iter_turn_edges, cdg_u_to_topo_ijv_map, "\n# CDG turns added this iteration")
        _write_turn_list(f, cum_turn_edges, cdg_u_to_topo_ijv_map, "\n# CDG turns cumulative")
        if solve_results is not None:
            _write_solve_values(f, solve_results, cdg_u_to_topo_ijv_map)

    for kind, scope, edge_set, title in plots:
        png_path = os.path.join(out_dir, f"{stem}_{kind}_{scope}.png")
        if kind == "topo":
            _plot_topology_debug(png_path, n_nodes, edge_set, title)
        else:
            _plot_cdg_debug(png_path, edge_set, cdg_u_to_topo_ijv_map, title)
        print(f"Wrote {png_path}")

    if cum_cdg_is_dag:
        print(f"[sanity] cumulative CDG after iter {idx} source {source}: acyclic (DAG)")
    else:
        n_cycle = len(cum_cdg_cycle) if cum_cdg_cycle else 0
        print(
            f"[sanity] WARNING: cumulative CDG after iter {idx} source {source}: "
            f"contains cycle ({n_cycle} edges); see {txt_path}"
        )

    print(f"Wrote per-source debug summary to {txt_path}")


# Script Functions
####################################################################################################

_LOAD_STATS_EPS = 1e-9

def write_json(path, payload):
    parent = os.path.dirname(path)
    if parent: os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)

def print_results(final_lmax_val, final_topology, final_paths, final_link_load, final_routing_table, final_vc_table, capacity, demand, final_total_link_load=None):
    n = len(final_topology)
    print(f"L_max (Min-Max Load) = {final_lmax_val}")
    if final_total_link_load is not None:
        print(f"Total link load (secondary objective) = {final_total_link_load:.6g}")
    print("topology:")
    for i, row in enumerate(final_topology):
        print(f"  {i}: {row}")
    
    loads = []
    for i in range(n):
        for j in range(n):
            if i == j: continue
            ld = float(final_link_load.get((i, j), 0.0))
            if ld > 0: loads.append(ld)

    if loads:
        print(f"Physical link load stats: count={len(loads)} min={min(loads):.6g} mean={sum(loads)/len(loads):.6g} max={max(loads):.6g}")
    else:
        print("No loaded edges.")

def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent: os.makedirs(parent, exist_ok=True)

def write_results(file_args, problem_args, final_lmax_val, final_topology, final_paths, final_link_load, final_routing_table, final_vc_table):
    topo_path = file_args["topo_out_path"]
    paths_path = file_args["paths_out_path"]
    paths_jsonl_path = file_args["paths_jsonl_out_path"]
    nr_path = file_args["nr_out_path"]
    vc_path = file_args["vc_out_path"]
    base = file_args["base_out_name"]

    _ensure_parent_dir(topo_path)
    with open(topo_path, "w", encoding="utf-8") as f:
        for row in final_topology:
            f.write(" ".join(str(int(x)) for x in row) + "\n")

    _ensure_parent_dir(paths_path)
    with open(paths_path, "w", encoding="utf-8") as f:
        f.write(base + "\n")
        for (s, d) in sorted(final_paths.keys(), key=lambda x: (x[0], x[1])):
            for p in final_paths[(s, d)]:
                f.write(json.dumps(p["phys_nodes"]) + "\n")

    _ensure_parent_dir(paths_jsonl_path)
    with open(paths_jsonl_path, "w", encoding="utf-8") as f:
        for (s, d) in sorted(final_paths.keys(), key=lambda x: (x[0], x[1])):
            for path_idx, p in enumerate(final_paths[(s, d)]):
                row = {
                    "s": s, "d": d, "path_idx": path_idx,
                    "phys_nodes": p.get("phys_nodes"), "vcs": p.get("vcs"),
                    "allocation": p.get("allocation"), "W_k": p.get("W_k"), "S_k": p.get("S_k"),
                }
                f.write(json.dumps(row) + "\n")

    _ensure_parent_dir(nr_path)
    with open(nr_path, "w", encoding="utf-8") as f:
        for (s, d) in sorted(final_paths.keys(), key=lambda x: (x[0], x[1])):
            for p in final_paths[(s, d)]:
                phys = p["phys_nodes"]
                for h in range(len(phys) - 1):
                    f.write(f"({s}, {d}, {phys[h]}, {phys[h + 1]})\n")

    _ensure_parent_dir(vc_path)
    with open(vc_path, "w", encoding="utf-8") as f:
        for (s, d) in sorted(final_paths.keys(), key=lambda x: (x[0], x[1])):
            for p in final_paths[(s, d)]:
                phys = p["phys_nodes"]
                vcs = p["vcs"]
                for h in range(len(phys) - 1):
                    vc = vcs[h] if h < len(vcs) else 0
                    f.write(f"({s}, {d}, {phys[h]}, {vc})\n")


# Per-source throughput priority proxy (used when --prioritized-per-source=throughput).
# Options: "link" | "turn" | "moore" | "cdg_penalty" | "combined"
NEXT_SOURCE_THROUGHPUT_PROXY = "combined"
_CDG_UNREACHABLE_PENALTY_SCALE = 1e4


def _build_cdg_maps_for_priority(n_nodes, n_vcs, allow_vc_trans):
    """Lightweight CDG node/turn maps for per-source priority scoring (no Gurobi model)."""
    cdg_u_to_topo_ijv_map = {}
    topo_ijv_to_cdg_u_map = {}
    ss_to_u_conn_map = defaultdict(list)
    sd_to_u_conn_map = defaultdict(list)
    u_to_v_turns = defaultdict(list)
    u_id = 0
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j:
                continue
            for l in range(n_vcs):
                cdg_u_to_topo_ijv_map[u_id] = (i, j, l)
                topo_ijv_to_cdg_u_map[(i, j, l)] = u_id
                ss_to_u_conn_map[i].append(u_id)
                sd_to_u_conn_map[j].append(u_id)
                u_id += 1
    uv_turn_set = set()
    for i in range(n_nodes):
        for k in range(n_nodes):
            if i == k:
                continue
            for j in range(n_nodes):
                if i == j or j == k:
                    continue
                for l0 in range(n_vcs):
                    for l1 in range(n_vcs):
                        if not allow_vc_trans and l0 != l1:
                            continue
                        u = topo_ijv_to_cdg_u_map[(i, k, l0)]
                        v = topo_ijv_to_cdg_u_map[(k, j, l1)]
                        uv_turn_set.add((u, v))
                        u_to_v_turns[u].append(v)
    return {
        "topo_ijv_to_cdg_u_map": topo_ijv_to_cdg_u_map,
        "ss_to_u_conn_map": ss_to_u_conn_map,
        "sd_to_u_conn_map": sd_to_u_conn_map,
        "u_to_v_turns": u_to_v_turns,
        "uv_turn_set": uv_turn_set,
    }


def _adjacency_from_topo_edges(hardened_topo_edges, n_nodes):
    adj = [[] for _ in range(n_nodes)]
    for i, j in hardened_topo_edges:
        if 0 <= i < n_nodes and 0 <= j < n_nodes and i != j:
            adj[i].append(j)
    return adj


def _bfs_hop_distances(adj, source):
    n = len(adj)
    dist = [float("inf")] * n
    if not (0 <= source < n):
        return dist
    dist[source] = 0
    queue = deque([source])
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if dist[v] == float("inf"):
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist


def _avg_hop_distance_to_destinations(s, dests, hardened_topo_edges, n_nodes):
    """Mean shortest-path hop count from s to each destination on the current topology."""
    adj = _adjacency_from_topo_edges(hardened_topo_edges, n_nodes)
    dist = _bfs_hop_distances(adj, s)
    hops = []
    for d in dests:
        if d == s:
            continue
        d_dist = dist[d]
        hops.append(d_dist if d_dist != float("inf") else n_nodes)
    return sum(hops) / len(hops) if hops else 0.0


def _shortest_path_phys_nodes(adj, source, dest):
    """Return shortest physical node path [source, ..., dest] or None if unreachable."""
    n = len(adj)
    if not (0 <= source < n and 0 <= dest < n):
        return None
    dist = [float("inf")] * n
    parent = [-1] * n
    dist[source] = 0
    queue = deque([source])
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if dist[v] == float("inf"):
                dist[v] = dist[u] + 1
                parent[v] = u
                queue.append(v)
    if dist[dest] == float("inf"):
        return None
    path = []
    cur = dest
    while cur != -1:
        path.append(cur)
        cur = parent[cur]
    return list(reversed(path))


def _cdg_reachable_from_source_to_dest(s, d, cdg_maps, hardened_turn_edges):
    """True if some hardened CDG walk connects super-source s to a sink node for dest d."""
    ss_nodes = cdg_maps["ss_to_u_conn_map"].get(s, [])
    sd_nodes = set(cdg_maps["sd_to_u_conn_map"].get(d, []))
    if not ss_nodes or not sd_nodes:
        return False
    u_to_v = cdg_maps["u_to_v_turns"]
    seen = set()
    queue = deque()
    for u in ss_nodes:
        if u in sd_nodes:
            return True
        seen.add(u)
        queue.append(u)
    while queue:
        u = queue.popleft()
        for v in u_to_v.get(u, []):
            if (u, v) not in hardened_turn_edges or v in seen:
                continue
            if v in sd_nodes:
                return True
            seen.add(v)
            queue.append(v)
    return False


def _hardened_turns_on_phys_path(phys_nodes, hardened_turn_edges, topo_ijv_to_cdg_u_map, n_vcs, allow_vc_trans):
    """Map a physical node path to hardened CDG turn arcs, or None if a required turn is missing."""
    if len(phys_nodes) < 2:
        return []
    if len(phys_nodes) == 2:
        return []
    turns = []
    for h in range(len(phys_nodes) - 2):
        i, k, j = phys_nodes[h], phys_nodes[h + 1], phys_nodes[h + 2]
        matched = None
        for l0 in range(n_vcs):
            for l1 in range(n_vcs):
                if not allow_vc_trans and l0 != l1:
                    continue
                u = topo_ijv_to_cdg_u_map[(i, k, l0)]
                v = topo_ijv_to_cdg_u_map[(k, j, l1)]
                edge = (u, v)
                if edge in hardened_turn_edges:
                    matched = edge
                    break
            if matched is not None:
                break
        if matched is None:
            return None
        turns.append(matched)
    return turns


def _flow_scale_for_source(s, dests, demand):
    return max((float(demand[s][d]) for d in dests if d != s), default=1.0)


def _link_bottleneck_throughput_proxy(s, dests, demand, hardened_topo_edges, known_load_links, capacity, known_lmax, n_nodes):
    """Predicted min-max link load from greedy shortest-path routing on hardened topology."""
    flow_scale = _flow_scale_for_source(s, dests, demand)
    adj = _adjacency_from_topo_edges(hardened_topo_edges, n_nodes)
    delta_load = defaultdict(float)
    for d in dests:
        if d == s:
            continue
        w = float(demand[s][d])
        phys = _shortest_path_phys_nodes(adj, s, d)
        if phys is None:
            return float("inf")
        for h in range(len(phys) - 1):
            delta_load[(phys[h], phys[h + 1])] += w
    l_hat = float(known_lmax)
    for (i, j), extra in delta_load.items():
        cap_ij = float(capacity[i][j])
        if cap_ij <= 0:
            return float("inf")
        prior = known_load_links.get((i, j), 0.0) * flow_scale
        l_hat = max(l_hat, (prior + extra) / cap_ij)
    return l_hat


def _turn_bottleneck_throughput_proxy(
    s, dests, demand, hardened_topo_edges, hardened_turn_edges, known_load_turns,
    cdg_maps, n_nodes, n_vcs, allow_vc_trans, known_lmax,
):
    """Predicted min-max CDG turn load from greedy shortest-path turn mapping."""
    flow_scale = _flow_scale_for_source(s, dests, demand)
    adj = _adjacency_from_topo_edges(hardened_topo_edges, n_nodes)
    topo_ijv_to_cdg_u_map = cdg_maps["topo_ijv_to_cdg_u_map"]
    delta_turn = defaultdict(float)
    for d in dests:
        if d == s:
            continue
        w = float(demand[s][d])
        phys = _shortest_path_phys_nodes(adj, s, d)
        if phys is None:
            return float("inf")
        turns = _hardened_turns_on_phys_path(
            phys, hardened_turn_edges, topo_ijv_to_cdg_u_map, n_vcs, allow_vc_trans,
        )
        if turns is None:
            return float("inf")
        for turn in turns:
            delta_turn[turn] += w
    l_hat = float(known_lmax)
    for (u, v), extra in delta_turn.items():
        prior = known_load_turns.get((u, v), 0.0) * flow_scale
        l_hat = max(l_hat, prior + extra)
    return l_hat


def _moore_throughput_proxy(s, dests, demand, hardened_topo_edges, known_load_links, capacity, known_lmax, n_nodes):
    """Volume/capacity lower-bound style proxy on residual hardened link capacity."""
    flow_scale = _flow_scale_for_source(s, dests, demand)
    adj = _adjacency_from_topo_edges(hardened_topo_edges, n_nodes)
    dist = _bfs_hop_distances(adj, s)
    hop_volume = 0.0
    for d in dests:
        if d == s:
            continue
        w = float(demand[s][d])
        d_dist = dist[d]
        hop_volume += w * (d_dist if d_dist != float("inf") else n_nodes)
    residual_cap = 0.0
    for (i, j) in hardened_topo_edges:
        cap_ij = float(capacity[i][j])
        prior = known_load_links.get((i, j), 0.0) * flow_scale
        residual_cap += max(0.0, cap_ij - prior)
    if residual_cap <= 0:
        return float("inf")
    return max(float(known_lmax), hop_volume / residual_cap)


def _cdg_penalty_throughput_proxy(
    s, dests, demand, hardened_topo_edges, known_load_links, capacity, known_lmax,
    n_nodes, hardened_turn_edges, cdg_maps,
):
    """Link bottleneck plus large penalty when physical path exists but hardened CDG path does not."""
    base = _link_bottleneck_throughput_proxy(
        s, dests, demand, hardened_topo_edges, known_load_links, capacity, known_lmax, n_nodes,
    )
    if base == float("inf"):
        return base
    adj = _adjacency_from_topo_edges(hardened_topo_edges, n_nodes)
    penalty = 0.0
    max_w = max((float(demand[s][d]) for d in dests if d != s), default=1.0)
    for d in dests:
        if d == s:
            continue
        if _shortest_path_phys_nodes(adj, s, d) is None:
            penalty += _CDG_UNREACHABLE_PENALTY_SCALE * max_w
            continue
        if not _cdg_reachable_from_source_to_dest(s, d, cdg_maps, hardened_turn_edges):
            penalty += _CDG_UNREACHABLE_PENALTY_SCALE * float(demand[s][d])
    return base + penalty


def _combined_throughput_proxy(
    s, dests, demand, hardened_topo_edges, hardened_turn_edges, known_load_links,
    known_load_turns, capacity, known_lmax, n_nodes, n_vcs, allow_vc_trans, cdg_maps,
):
    """Max of link, turn, moore, and CDG-penalty proxies (most pessimistic)."""
    scores = [
        _link_bottleneck_throughput_proxy(
            s, dests, demand, hardened_topo_edges, known_load_links, capacity, known_lmax, n_nodes,
        ),
        _turn_bottleneck_throughput_proxy(
            s, dests, demand, hardened_topo_edges, hardened_turn_edges, known_load_turns,
            cdg_maps, n_nodes, n_vcs, allow_vc_trans, known_lmax,
        ),
        _moore_throughput_proxy(
            s, dests, demand, hardened_topo_edges, known_load_links, capacity, known_lmax, n_nodes,
        ),
        _cdg_penalty_throughput_proxy(
            s, dests, demand, hardened_topo_edges, known_load_links, capacity, known_lmax,
            n_nodes, hardened_turn_edges, cdg_maps,
        ),
    ]
    return max(scores)


def _source_throughput_proxy(s, dests, demand, hardened_topo_edges, hardened_turn_edges,
                             known_load_links, known_load_turns, capacity, known_lmax,
                             n_nodes, n_vcs, allow_vc_trans, cdg_maps, proxy=None):
    """Higher score = lower achievable throughput; pick largest among remaining sources."""
    proxy = proxy or NEXT_SOURCE_THROUGHPUT_PROXY
    if proxy == "link":
        return _link_bottleneck_throughput_proxy(
            s, dests, demand, hardened_topo_edges, known_load_links, capacity, known_lmax, n_nodes,
        )
    if proxy == "turn":
        return _turn_bottleneck_throughput_proxy(
            s, dests, demand, hardened_topo_edges, hardened_turn_edges, known_load_turns,
            cdg_maps, n_nodes, n_vcs, allow_vc_trans, known_lmax,
        )
    if proxy == "moore":
        return _moore_throughput_proxy(
            s, dests, demand, hardened_topo_edges, known_load_links, capacity, known_lmax, n_nodes,
        )
    if proxy == "cdg_penalty":
        return _cdg_penalty_throughput_proxy(
            s, dests, demand, hardened_topo_edges, known_load_links, capacity, known_lmax,
            n_nodes, hardened_turn_edges, cdg_maps,
        )
    if proxy == "combined":
        return _combined_throughput_proxy(
            s, dests, demand, hardened_topo_edges, hardened_turn_edges, known_load_links,
            known_load_turns, capacity, known_lmax, n_nodes, n_vcs, allow_vc_trans, cdg_maps,
        )
    raise ValueError(
        f"Unknown NEXT_SOURCE_THROUGHPUT_PROXY {proxy!r}; "
        "expected link, turn, moore, cdg_penalty, or combined."
    )


def _all_commodity_flows(commodities):
    flows = []
    for s in sorted(commodities.keys()):
        for d in sorted(commodities[s]):
            if s != d:
                flows.append((s, d))
    return flows


def _node_has_topo_incident_edge(n, hardened_topo_edges):
    return any(i == n or j == n for i, j in hardened_topo_edges)


def _flow_connectivity_tier(s, d, hardened_topo_edges):
    """0: both endpoints unconnected; 1: one unconnected; 2: both connected."""
    s_conn = _node_has_topo_incident_edge(s, hardened_topo_edges)
    d_conn = _node_has_topo_incident_edge(d, hardened_topo_edges)
    if not s_conn and not d_conn:
        return 0
    if not s_conn or not d_conn:
        return 1
    return 2


def _flow_hop_metric(s, d, hardened_topo_edges, n_nodes):
    adj = _adjacency_from_topo_edges(hardened_topo_edges, n_nodes)
    dist = _bfs_hop_distances(adj, s)
    d_dist = dist[d]
    return d_dist if d_dist != float("inf") else float("inf")


def _flow_throughput_metric(
    s, d, demand, hardened_topo_edges, hardened_turn_edges, known_load_links,
    known_load_turns, capacity, known_lmax, n_nodes, n_vcs, allow_vc_trans, cdg_maps,
):
    return _source_throughput_proxy(
        s, [d], demand, hardened_topo_edges, hardened_turn_edges,
        known_load_links, known_load_turns, capacity, known_lmax,
        n_nodes, n_vcs, allow_vc_trans, cdg_maps,
    )


def _flow_priority_sort_key(
    s, d, priority_policy, hardened_topo_edges, n_nodes, demand,
    hardened_turn_edges, known_load_links, known_load_turns, capacity,
    known_lmax, n_vcs, allow_vc_trans, cdg_maps,
):
    """Lower key = higher priority. Connectivity tier first, then worse metric."""
    tier = _flow_connectivity_tier(s, d, hardened_topo_edges)
    if priority_policy == "hops":
        metric = _flow_hop_metric(s, d, hardened_topo_edges, n_nodes)
    elif priority_policy == "throughput":
        metric = _flow_throughput_metric(
            s, d, demand, hardened_topo_edges, hardened_turn_edges, known_load_links,
            known_load_turns, capacity, known_lmax, n_nodes, n_vcs, allow_vc_trans, cdg_maps,
        )
    else:
        metric = 0.0
    metric_key = float("-inf") if metric == float("inf") else -metric
    return (tier, metric_key, s, d)


def _format_flow_metric(metric):
    return "inf" if metric == float("inf") else f"{metric:.4g}"


def _select_next_flow(
    remaining_flows, priority_policy, idx, hardened_topo_edges, problem_args,
    hardened_turn_edges, known_load_links, known_load_turns, known_lmax, cdg_maps,
):
    n_nodes = problem_args["n_nodes"]
    if not priority_policy or idx == 1:
        return min(remaining_flows)
    scored = []
    for s, d in remaining_flows:
        tier = _flow_connectivity_tier(s, d, hardened_topo_edges)
        if priority_policy == "hops":
            metric = _flow_hop_metric(s, d, hardened_topo_edges, n_nodes)
        elif priority_policy == "throughput":
            metric = _flow_throughput_metric(
                s, d, problem_args["dem_mat"], hardened_topo_edges, hardened_turn_edges,
                known_load_links, known_load_turns, problem_args["cap_mat"], known_lmax,
                n_nodes, problem_args["n_vcs"], problem_args["allow_vc_trans"], cdg_maps,
            )
        else:
            raise ValueError(f"Unknown priority policy: {priority_policy!r}")
        sort_key = _flow_priority_sort_key(
            s, d, priority_policy, hardened_topo_edges, n_nodes, problem_args["dem_mat"],
            hardened_turn_edges, known_load_links, known_load_turns, problem_args["cap_mat"],
            known_lmax, problem_args["n_vcs"], problem_args["allow_vc_trans"], cdg_maps,
        )
        scored.append((sort_key, s, d, tier, metric))
    scored.sort(key=lambda x: x[0])
    _, s, d, tier, metric = scored[0]
    rem = ", ".join(
        f"({fs},{fd}):t{tier}/{_format_flow_metric(fm)}"
        for _, fs, fd, t, fm in sorted(
            ((sk, fs, fd, t, fm) for sk, fs, fd, t, fm in scored),
            key=lambda x: x[0],
        )
    )
    tier_labels = {0: "both_unconnected", 1: "one_unconnected", 2: "both_connected"}
    print(
        f"selected ({s},{d}) tier={tier_labels[tier]} "
        f"metric={_format_flow_metric(metric)} ({priority_policy}); remaining: {rem}"
    )
    return (s, d)


def _run_single_source_optimization(problem_args, file_args, solver_params):
    ccdg_model = CCDGModel(
        n_nodes=problem_args['n_nodes'], radix=problem_args['radix'], n_vcs=problem_args['n_vcs'],
        capacity=problem_args['cap_mat'], demand=problem_args['dem_mat'], commodities=problem_args['commodities'],
        relax_topo_edges=problem_args['relax_topo_edges'], relax_turn_edges=problem_args['relax_turn_edges'],
        integral_flow=problem_args['integral_flow'], k_paths=problem_args['k_paths'],
        allow_vc_trans=problem_args['allow_vc_trans'], symmetric_links=problem_args['symmetric_links'],
    )
    ccdg_model.build_model_()
    if file_args['write_model']:
        ccdg_model.write_model_(os.path.join("files/models", file_args['base_out_name'] + ".lp"))

    ccdg_optimizer = CCDGOptimizer(ccdg_model=ccdg_model, model_params=solver_params)
    ccdg_optimizer.solve()
    results = ccdg_optimizer.extract_resultant_values()
    print(results)
    final_lmax_val = results['L_max']
    final_topology, final_paths, final_link_load, final_routing_table, final_vc_table = ccdg_optimizer.harden_results()
    return final_lmax_val, final_topology, final_paths, final_link_load, final_routing_table, final_vc_table, None


def run_per_source_optimization(problem_args, file_args, solver_params):
    commodities = problem_args['commodities']
    n_nodes = problem_args['n_nodes']
    safe = problem_args['safe']
    n_sources = len(commodities)
    remaining_sources = set(commodities.keys())

    hardened_topo_edges = set()
    hardened_turn_edges = set()
    known_load_links = defaultdict(float)
    known_load_turns = defaultdict(float)

    merged_paths = {}
    merged_route_table = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    merged_vc_table = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    lmax_per_source = {}

    debug_viz_dir = file_args.get("debug_viz_dir")

    priority_policy = problem_args.get("prioritized_per_source")
    cdg_maps = _build_cdg_maps_for_priority(
        n_nodes, problem_args["n_vcs"], problem_args["allow_vc_trans"],
    )

    print("-" * 80)
    print(f"Starting per-source sequential optimization ({n_sources} sources)")
    if priority_policy == "hops":
        print(
            "Source order: iteration 1 uses lowest source id; "
            "later iterations pick highest avg hop distance."
        )
    elif priority_policy == "throughput":
        print(
            f"Source order: iteration 1 uses lowest source id; "
            f"later iterations pick lowest throughput proxy ({NEXT_SOURCE_THROUGHPUT_PROXY})."
        )
    else:
        print("Source order: fixed ascending source id.")
    print("-" * 80)

    known_lmax = 0
    idx = 0
    while remaining_sources:
        idx += 1
        if priority_policy and idx > 1:
            if priority_policy == "hops":
                scored = [
                    (
                        s_cand,
                        _avg_hop_distance_to_destinations(
                            s_cand, commodities[s_cand], hardened_topo_edges, n_nodes,
                        ),
                    )
                    for s_cand in remaining_sources
                ]
                scored.sort(key=lambda x: (-x[1], x[0]))
                s, metric = scored[0]
                print(
                    f"[iter {idx}/{n_sources}] selected s={s} "
                    f"(avg_hop={metric:.4g}); remaining avg hops: "
                    + ", ".join(f"{sc[0]}:{sc[1]:.4g}" for sc in scored)
                )
            elif priority_policy == "throughput":
                scored = [
                    (
                        s_cand,
                        _source_throughput_proxy(
                            s_cand, commodities[s_cand], problem_args["dem_mat"],
                            hardened_topo_edges, hardened_turn_edges,
                            known_load_links, known_load_turns,
                            problem_args["cap_mat"], known_lmax,
                            n_nodes, problem_args["n_vcs"], problem_args["allow_vc_trans"],
                            cdg_maps,
                        ),
                    )
                    for s_cand in remaining_sources
                ]
                scored.sort(key=lambda x: (-x[1], x[0]))
                s, metric = scored[0]
                metric_str = "inf" if metric == float("inf") else f"{metric:.4g}"
                rem = ", ".join(
                    f"{sc[0]}:{'inf' if sc[1] == float('inf') else f'{sc[1]:.4g}'}"
                    for sc in scored
                )
                print(
                    f"[iter {idx}/{n_sources}] selected s={s} "
                    f"(throughput_proxy={metric_str}); remaining: {rem}"
                )
            else:
                raise ValueError(f"Unknown prioritized_per_source policy: {priority_policy!r}")
        else:
            s = min(remaining_sources)
            if priority_policy:
                print(f"[iter {idx}/{n_sources}] first source (unordered): s={s}")
            else:
                print(f"[iter {idx}/{n_sources}] source s={s}")
        remaining_sources.remove(s)
        sub_commodities = {s: commodities[s]}
        ccdg_model = CCDGModel(
            n_nodes=problem_args['n_nodes'], radix=problem_args['radix'], n_vcs=problem_args['n_vcs'],
            capacity=problem_args['cap_mat'], demand=problem_args['dem_mat'], commodities=sub_commodities,
            relax_topo_edges=problem_args['relax_topo_edges'], relax_turn_edges=problem_args['relax_turn_edges'],
            integral_flow=problem_args['integral_flow'], per_source_solves=True, k_paths=problem_args['k_paths'],
            allow_vc_trans=problem_args['allow_vc_trans'], symmetric_links=problem_args['symmetric_links'],
            known_load_links=known_load_links, known_load_turns=known_load_turns,
            known_lmax=known_lmax,
            hardened_topo_edges=hardened_topo_edges, hardened_turn_edges=hardened_turn_edges,
        )
        ccdg_model.build_model_()
        stem = f"{file_args['base_out_name']}_iter{idx:02d}_src{s}"
        if file_args['write_model']:
            model_path = os.path.join("files/models", f"{file_args['base_out_name']}_src{s}.lp")
            ccdg_model.write_model_(model_path)
        if debug_viz_dir:
            ccdg_model.write_model_(os.path.join(debug_viz_dir, stem + ".lp"))

        ccdg_optimizer = CCDGOptimizer(ccdg_model=ccdg_model, model_params=solver_params)
        prev_topo_edges = set(hardened_topo_edges)
        prev_turn_edges = set(hardened_turn_edges)

        ccdg_optimizer.solve()

        results = ccdg_optimizer.extract_resultant_values()
        iter_lmax = results['L_max']
        lmax_per_source[s] = iter_lmax

        if debug_viz_dir:
            write_per_source_solve_debug(
                debug_viz_dir, file_args["base_out_name"], idx, s,
                results, ccdg_optimizer.cdg_u_to_topo_ijv_map,
            )

        topology, paths, _link_load, route_table, vc_table = ccdg_optimizer.harden_results()

        merged_paths.update(paths)
        CCDGOptimizer.merge_routing_tables(merged_route_table, merged_vc_table, route_table, vc_table)

        CCDGOptimizer.accumulate_link_flow_counts(paths, problem_args["dem_mat"], known_load_links)
        CCDGOptimizer.accumulate_turn_flow_counts(paths, problem_args["dem_mat"], known_load_turns)

        hardened_topo_edges |= CCDGOptimizer.extract_hardened_topology_from_paths(
            paths, symmetric_links=problem_args['symmetric_links'],
        )
        path_turn_edges = CCDGOptimizer.extract_hardened_turns_from_paths(paths)
        if idx == 1 and safe:
            hardened_turn_edges |= ccdg_optimizer.expand_spanning_tree_turns_at_root(
                s, path_turn_edges,hardened_topo_edges)
        else:
            hardened_turn_edges |= path_turn_edges

        iter_flow_scale = _flow_scale_for_source(s, sub_commodities[s], problem_args["dem_mat"])
        cumulative_max_load = max(
            (v * iter_flow_scale for v in known_load_links.values()),
            default=0.0,
        )
        print(
            f"[source {idx}/{n_sources}] s={s} L_max={iter_lmax:.6g} "
            f"cumulative_max_load={cumulative_max_load:.6g}"
        )

        if debug_viz_dir:
            write_per_source_harden_debug(
                out_dir=debug_viz_dir,
                base_name=file_args["base_out_name"],
                idx=idx,
                source=s,
                n_nodes=problem_args["n_nodes"],
                iter_topo_edges=hardened_topo_edges - prev_topo_edges,
                cum_topo_edges=set(hardened_topo_edges),
                iter_turn_edges=hardened_turn_edges - prev_turn_edges,
                cum_turn_edges=set(hardened_turn_edges),
                cdg_u_to_topo_ijv_map=ccdg_optimizer.cdg_u_to_topo_ijv_map,
                iter_lmax=iter_lmax,
                cumulative_max_load=cumulative_max_load,
                solve_results=results,
            )

        # known_source_commodities.append(s)
        known_lmax = max(known_lmax, iter_lmax)

    final_link_load = CCDGOptimizer.absolute_link_load_from_paths(merged_paths)
    final_lmax_val = max(final_link_load.values()) if final_link_load else 0.0
    final_topology = ccdg_optimizer.topology_from_hardened_edges(hardened_topo_edges)
    final_routing_table = CCDGOptimizer._plain_nested_dict(merged_route_table)
    final_vc_table = CCDGOptimizer._plain_nested_dict(merged_vc_table)

    print("-" * 80)
    print(f"Per-source optimization complete. Global L_max (max edge load) = {final_lmax_val:.6g}")
    print("-" * 80)

    return (
        final_lmax_val, final_topology, merged_paths, final_link_load,
        final_routing_table, final_vc_table, lmax_per_source,
    )


def run_per_flow_optimization(problem_args, file_args, solver_params):
    commodities = problem_args['commodities']
    n_nodes = problem_args['n_nodes']
    safe = problem_args['safe']
    remaining_flows = set(_all_commodity_flows(commodities))
    n_flows = len(remaining_flows)

    hardened_topo_edges = set()
    hardened_turn_edges = set()
    known_load_links = defaultdict(float)
    known_load_turns = defaultdict(float)

    merged_paths = {}
    merged_route_table = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    merged_vc_table = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    lmax_per_flow = {}

    debug_viz_dir = file_args.get("debug_viz_dir")
    priority_policy = problem_args.get("prioritized_per_source")
    cdg_maps = _build_cdg_maps_for_priority(
        n_nodes, problem_args["n_vcs"], problem_args["allow_vc_trans"],
    )

    print("-" * 80)
    print(f"Starting per-flow sequential optimization ({n_flows} commodities)")
    if priority_policy == "hops":
        print(
            "Flow order: iteration 1 uses lowest (s,d); later iterations prioritize "
            "both-unconnected, then one-unconnected, then highest hop distance."
        )
    elif priority_policy == "throughput":
        print(
            f"Flow order: iteration 1 uses lowest (s,d); later iterations prioritize "
            f"both-unconnected, then one-unconnected, then worst throughput proxy "
            f"({NEXT_SOURCE_THROUGHPUT_PROXY})."
        )
    else:
        print("Flow order: fixed ascending (source, destination) lexicographic.")
    print("-" * 80)

    known_lmax = 0
    idx = 0
    ccdg_optimizer = None
    while remaining_flows:
        idx += 1
        if priority_policy and idx > 1:
            print(f"[iter {idx}/{n_flows}] ", end="")
            s, d = _select_next_flow(
                remaining_flows, priority_policy, idx, hardened_topo_edges, problem_args,
                hardened_turn_edges, known_load_links, known_load_turns, known_lmax, cdg_maps,
            )
        else:
            s, d = min(remaining_flows)
            if priority_policy:
                print(f"[iter {idx}/{n_flows}] first flow (unordered): ({s},{d})")
            else:
                print(f"[iter {idx}/{n_flows}] flow ({s},{d})")
        remaining_flows.remove((s, d))

        sub_commodities = {s: [d]}
        ccdg_model = CCDGModel(
            n_nodes=problem_args['n_nodes'], radix=problem_args['radix'], n_vcs=problem_args['n_vcs'],
            capacity=problem_args['cap_mat'], demand=problem_args['dem_mat'], commodities=sub_commodities,
            relax_topo_edges=problem_args['relax_topo_edges'], relax_turn_edges=problem_args['relax_turn_edges'],
            integral_flow=problem_args['integral_flow'], per_source_solves=True, k_paths=problem_args['k_paths'],
            allow_vc_trans=problem_args['allow_vc_trans'], symmetric_links=problem_args['symmetric_links'],
            known_load_links=known_load_links, known_load_turns=known_load_turns,
            known_lmax=known_lmax,
            hardened_topo_edges=hardened_topo_edges, hardened_turn_edges=hardened_turn_edges,
        )
        ccdg_model.build_model_()
        stem = _iter_debug_stem(file_args['base_out_name'], idx, s, d)
        if file_args['write_model']:
            model_path = os.path.join("files/models", f"{file_args['base_out_name']}_src{s}_dst{d}.lp")
            ccdg_model.write_model_(model_path)
        if debug_viz_dir:
            ccdg_model.write_model_(os.path.join(debug_viz_dir, stem + ".lp"))

        ccdg_optimizer = CCDGOptimizer(ccdg_model=ccdg_model, model_params=solver_params)
        prev_topo_edges = set(hardened_topo_edges)
        prev_turn_edges = set(hardened_turn_edges)

        ccdg_optimizer.solve()

        results = ccdg_optimizer.extract_resultant_values()
        iter_lmax = results['L_max']
        lmax_per_flow[(s, d)] = iter_lmax

        if debug_viz_dir:
            write_per_source_solve_debug(
                debug_viz_dir, file_args["base_out_name"], idx, s,
                results, ccdg_optimizer.cdg_u_to_topo_ijv_map, dest=d,
            )

        topology, paths, _link_load, route_table, vc_table = ccdg_optimizer.harden_results()

        merged_paths.update(paths)
        CCDGOptimizer.merge_routing_tables(merged_route_table, merged_vc_table, route_table, vc_table)

        CCDGOptimizer.accumulate_link_flow_counts(paths, problem_args["dem_mat"], known_load_links)
        CCDGOptimizer.accumulate_turn_flow_counts(paths, problem_args["dem_mat"], known_load_turns)

        hardened_topo_edges |= CCDGOptimizer.extract_hardened_topology_from_paths(
            paths, symmetric_links=problem_args['symmetric_links'],
        )
        path_turn_edges = CCDGOptimizer.extract_hardened_turns_from_paths(paths)
        if idx == 1 and safe:
            hardened_turn_edges |= ccdg_optimizer.expand_spanning_tree_turns_at_root(
                s, path_turn_edges, hardened_topo_edges)
        else:
            hardened_turn_edges |= path_turn_edges

        iter_flow_scale = float(problem_args["dem_mat"][s][d])
        cumulative_max_load = max(
            (v * iter_flow_scale for v in known_load_links.values()),
            default=0.0,
        )
        print(
            f"[flow {idx}/{n_flows}] ({s},{d}) L_max={iter_lmax:.6g} "
            f"cumulative_max_load={cumulative_max_load:.6g}"
        )

        if debug_viz_dir:
            write_per_source_harden_debug(
                out_dir=debug_viz_dir,
                base_name=file_args["base_out_name"],
                idx=idx,
                source=s,
                n_nodes=problem_args["n_nodes"],
                iter_topo_edges=hardened_topo_edges - prev_topo_edges,
                cum_topo_edges=set(hardened_topo_edges),
                iter_turn_edges=hardened_turn_edges - prev_turn_edges,
                cum_turn_edges=set(hardened_turn_edges),
                cdg_u_to_topo_ijv_map=ccdg_optimizer.cdg_u_to_topo_ijv_map,
                iter_lmax=iter_lmax,
                cumulative_max_load=cumulative_max_load,
                solve_results=results,
                dest=d,
            )

        known_lmax = max(known_lmax, iter_lmax)

    final_link_load = CCDGOptimizer.absolute_link_load_from_paths(merged_paths)
    final_lmax_val = max(final_link_load.values()) if final_link_load else 0.0
    final_topology = ccdg_optimizer.topology_from_hardened_edges(hardened_topo_edges)
    final_routing_table = CCDGOptimizer._plain_nested_dict(merged_route_table)
    final_vc_table = CCDGOptimizer._plain_nested_dict(merged_vc_table)

    print("-" * 80)
    print(f"Per-flow optimization complete. Global L_max (max edge load) = {final_lmax_val:.6g}")
    print("-" * 80)

    return (
        final_lmax_val, final_topology, merged_paths, final_link_load,
        final_routing_table, final_vc_table, lmax_per_flow,
    )


def run_optimization(problem_args, file_args, solver_params):
    if problem_args.get('per_flow_solves'):
        return run_per_flow_optimization(problem_args, file_args, solver_params)
    if problem_args.get('per_source_solves'):
        return run_per_source_optimization(problem_args, file_args, solver_params)
    return _run_single_source_optimization(problem_args, file_args, solver_params)


# Script Functions
####################################################################################################

def define_all_arguments():
    ap = argparse.ArgumentParser(description="One-shot topology, routing, and deadlock freedom synthesis via CDG. Primary objective: min L_max; optional secondary: min total link load.")
    ap.add_argument("--n_nodes", type=int, required=True, help="Number of nodes in the network.")
    ap.add_argument("--radix", type=int, required=True, help="Maximum out-degree (radix) per node.")
    ap.add_argument("--n_vcs", type=int, required=True, help="Number of virtual channels per physical link.")
    ap.add_argument("--out", type=str, help="Base output name for topology, routing, and VC files (default: ccdg_minmax_{n}n_{radix}r_{n_vcs}vcs).")
    ap.add_argument("--out-json", type=str, default=None, help="Optional JSON path for summary results and load statistics.")
    ap.add_argument("--write-model", action="store_true", help="Write Gurobi .lp model file(s) under files/models/.")
    ap.add_argument("--capacity", type=float, default=1.0, help="Uniform per-link capacity when --capacity-matrix is not set.")
    ap.add_argument("--demand", type=float, default=1.0, help="Uniform per-commodity demand when --demand-matrix is not set.")
    ap.add_argument("--capacity-matrix", type=str, default=None, help="Path to n x n whitespace-separated link capacity matrix file.")
    ap.add_argument("--demand-matrix", type=str, default=None, help="Path to n x n whitespace-separated demand matrix file.")
    ap.add_argument("--commodities-file", type=str, default=None, help="Path to (source, destination) pairs file; default is all-to-all except self.")
    ap.add_argument("--continuous-flow", action="store_true", help="Relax integer flow variables to continuous.")
    ap.add_argument("--continuous-topo-edges", action="store_true", help="Relax topology edge variables to continuous [0,1].")
    ap.add_argument("--continuous-turn-edges", action="store_true", help="Relax CDG turn edge variables to continuous [0,1].")
    ap.add_argument("--k-paths", type=int, default=1, help="Maximum number of paths per (source, destination) commodity.")
    ap.add_argument("--symmetric-links", action="store_true", help="Require undirected physical links: edge (i,j) implies (j,i).")
    ap.add_argument("--hierarchical-objectives", action="store_true", help="Lexicographic objectives: first minimize L_max, then minimize total link load subject to optimal L_max.")
    ap.add_argument("--per-source-solves", action="store_true", help="Solve one source at a time; harden topology, turns, and paths between iterations.")
    ap.add_argument("--per-flow-solves", action="store_true", help="Solve one (source, destination) commodity at a time; harden topology, turns, and paths between iterations.")
    ap.add_argument("--prioritized-per-source", choices=["hops", "throughput"], default=None, help="With --per-source-solves or --per-flow-solves: after iteration 1, reorder remaining items by hops or throughput (see NEXT_SOURCE_THROUGHPUT_PROXY); per-flow also tiers by endpoint connectivity.")
    ap.add_argument("--debug-viz-dir", type=str, default=None, help="Directory for per-iteration debug PNGs, LP models, and solve/harden text summaries.")
    ap.add_argument("--safe", action="store_true", help="With --per-source-solves or --per-flow-solves: on first iteration only, also harden spanning-tree reverse turns and root-pivot turns.")
    ap.add_argument("--allow-vc-trans", action="store_true", help="Allow CDG turns that change virtual channel between consecutive hops.")
    ap.add_argument("--time_limit", type=float, default=None, help="Gurobi solver time limit in seconds.")
    ap.add_argument("--mip_gap", type=float, default=None, help="Gurobi MIP relative gap tolerance (fraction, e.g. 0.01 for 1%%).")
    ap.add_argument("--obj-gap-pct", type=float, default=None, dest="obj_gap_pct", help="Early stop when (incumbent - bound) / incumbent * 100 is at most this value; sets Gurobi MIPGap to obj_gap_pct/100.")
    ap.add_argument("--threads", type=int, default=None, help="Gurobi thread count.")
    ap.add_argument("--silent", action="store_true", help="Suppress Gurobi solver console output.")
    return ap

def parse_problem_arguments(args):
    n = args.n_nodes
    radix = args.radix
    n_vcs = args.n_vcs
    cap_mat = _read_matrix_file(args.capacity_matrix, n) if args.capacity_matrix else _uniform_matrix(n, args.capacity)
    dem_mat = _read_matrix_file(args.demand_matrix, n) if args.demand_matrix else _uniform_matrix(n, args.demand)
    commodities = _parse_commodities_file(args.commodities_file, n) if args.commodities_file else _default_commodities(n)
    
    if args.out is None:
        args.out = f"ccdg_minmax_{n}n_{radix}r_{n_vcs}vcs"

    base_out_name = args.out
    problem_args = {
        "n_nodes": n, "radix": radix, "n_vcs": n_vcs,
        "cap_mat": cap_mat, "dem_mat": dem_mat, "commodities": commodities,
        "relax_topo_edges": args.continuous_topo_edges,
        "relax_turn_edges": args.continuous_turn_edges,
        "integral_flow": not args.continuous_flow,  # Default to integer flow per intent
        "k_paths": args.k_paths,
        "allow_vc_trans": args.allow_vc_trans,
        "symmetric_links": args.symmetric_links,
        "hierarchical_objectives": args.hierarchical_objectives,
        "per_source_solves": args.per_source_solves,
        "per_flow_solves": args.per_flow_solves,
        "prioritized_per_source": args.prioritized_per_source,  # None | "hops" | "throughput"
        "safe": args.safe,
    }

    file_args = {
        "base_out_name": base_out_name,
        "topo_out_path": os.path.join("topologies_and_routing/topo_maps", base_out_name + ".map"),
        "graphml_out_path": os.path.join("topologies_and_routing/topo_maps", base_out_name + ".graphml"),
        "paths_out_path": os.path.join("topologies_and_routing/routepath_lists", base_out_name + ".paths"),
        "paths_jsonl_out_path": os.path.join("topologies_and_routing/routepath_lists", base_out_name + ".paths.jsonl"),
        "nr_out_path": os.path.join("topologies_and_routing/nr_lists", base_out_name + ".nrl2"),
        "vc_out_path": os.path.join("topologies_and_routing/vc_mats", base_out_name + ".vcmat2"),
        "write_model": args.write_model,
        "debug_viz_dir": args.debug_viz_dir,
    }
    solver_params = {
        "silent": args.silent, "threads": args.threads,
        "time_limit": args.time_limit, "mip_gap": args.mip_gap,
        "obj_gap_pct": args.obj_gap_pct,
        "hierarchical_objectives": args.hierarchical_objectives,
    }
    return problem_args, file_args, solver_params

# Main Function
####################################################################################################

def main():
    ap = define_all_arguments()
    args = ap.parse_args()
    problem_args, file_args, solver_params = parse_problem_arguments(args)

    if solver_params.get("mip_gap") is not None and solver_params.get("obj_gap_pct") is not None:
        print("ERROR: --mip_gap and --obj-gap-pct are mutually exclusive.")
        sys.exit(1)
    if solver_params.get("obj_gap_pct") is not None and solver_params["obj_gap_pct"] < 0:
        print("ERROR: --obj-gap-pct must be non-negative.")
        sys.exit(1)

    if problem_args["per_source_solves"] and problem_args["per_flow_solves"]:
        print("ERROR: --per-source-solves and --per-flow-solves are mutually exclusive.")
        sys.exit(1)
    if problem_args["per_source_solves"] or problem_args["per_flow_solves"]:
        if problem_args["relax_topo_edges"] or problem_args["relax_turn_edges"]:
            print("UNIMPLEMENTED: sequential per-source/per-flow solves require binary topology and turn edges.")
            sys.exit(1)
        if not problem_args["integral_flow"]:
            print("UNIMPLEMENTED: sequential per-source/per-flow solves require integral flow.")
            sys.exit(1)

    (
        final_lmax_val, final_topology, final_paths, final_link_load,
        final_routing_table, final_vc_table, lmax_per_source,
    ) = run_optimization(problem_args, file_args, solver_params)

    final_total_link_load = sum(final_link_load.values()) if final_link_load else 0.0
    print_results(
        final_lmax_val, final_topology, final_paths, final_link_load,
        final_routing_table, final_vc_table, problem_args["cap_mat"], problem_args["dem_mat"],
        final_total_link_load=final_total_link_load,
    )
    write_results(file_args, problem_args, final_lmax_val, final_topology, final_paths, final_link_load, final_routing_table, final_vc_table)

    print(f"Wrote topology (map) to {file_args['topo_out_path']}")
    print(f"Wrote pathlist to {file_args['paths_out_path']}")
    print(f"L_max = {final_lmax_val}")
    print(f"Total link load = {final_total_link_load:.6g}")

    if args.out_json:
        out_payload = {
            "L_max": final_lmax_val,
            "total_link_load": final_total_link_load,
            "hierarchical_objectives": problem_args["hierarchical_objectives"],
            "n_nodes": problem_args["n_nodes"],
            "radix": problem_args["radix"],
            "n_vcs": problem_args["n_vcs"],
            "datetime_solved": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        if problem_args["per_source_solves"]:
            out_payload["per_source_solves"] = True
            out_payload["lmax_per_source"] = {str(k): v for k, v in (lmax_per_source or {}).items()}
        if problem_args["per_flow_solves"]:
            out_payload["per_flow_solves"] = True
            out_payload["lmax_per_flow"] = {f"{s},{d}": v for (s, d), v in (lmax_per_source or {}).items()}
        write_json(args.out_json, out_payload)

if __name__ == "__main__":
    main()