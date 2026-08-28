"""Gurobi MILP model for complete CDG topology and routing synthesis."""

from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB


class CCDGModel:
    VERBOSE = False

    def __init__(
        self,
        n_nodes,
        radix,
        n_vcs,
        capacity,
        demand,
        commodities,
        relax_topo_edges=False,
        relax_turn_edges=False,
        integral_flow=False,
        per_source_solves=False,
        k_paths=1,
        symmetric_links=False,
        symmetric_paths=False,
        allow_vc_trans=False,
        known_load=None,
        hardened_topo_edges=None,
        hardened_turn_edges=None,
    ):
        self.n_nodes = n_nodes
        self.radix = radix
        self.n_vcs = n_vcs
        self.capacity = capacity
        self.demand = demand
        self.commodities = commodities
        self.source_commodities = commodities.keys()
        self.relax_topo_edges = relax_topo_edges
        self.relax_turn_edges = relax_turn_edges
        self.integral_flow = integral_flow
        self.per_source_solves = per_source_solves
        self.symmetric_links = symmetric_links
        self.symmetric_paths = symmetric_paths
        self.k_paths = k_paths
        self.allow_vc_trans = allow_vc_trans
        self.known_load = (
            {(i, j): 0 for i in range(n_nodes) for j in range(n_nodes)}
            if known_load is None
            else known_load
        )
        self.hardened_topo_edges = set(hardened_topo_edges or ())
        self.hardened_turn_edges = set(hardened_turn_edges or ())
        if self.symmetric_links:
            self.hardened_topo_edges |= {
                (j, i) for (i, j) in self.hardened_topo_edges
            }

        self.u_id = 0
        self.n_cdg = 0
        self.cdg_u_to_topo_ijv_map = None
        self.topo_ijv_to_cdg_u_map = None
        self.ss_to_u_conn_map = None
        self.sd_to_u_conn_map = None
        self.uv_turn_set = None
        self.u_to_v_turns = None
        self.v_to_u_turns = None

        self.model = None
        self.vars_topo_adj_mat = None
        self.vars_turn_adj_mat = None
        self.vars_uv_flow = None
        self.vars_ss_flow = None
        self.vars_o = None
        self.vars_sink = None
        self.var_mcf = None

        n_comm = len(commodities)
        n_turns = n_nodes * (n_nodes - 1) * (n_nodes - 2) * n_vcs
        print(f"CCDGModel: {n_nodes} nodes, radix {radix}, {n_vcs} VCs, {n_comm} sources, ~{n_turns} turns")

    def _iter_turns(self):
        for i in range(self.n_nodes):
            for k in range(self.n_nodes):
                if i == k:
                    continue
                for j in range(self.n_nodes):
                    if i == j or j == k:
                        continue
                    for l0 in range(self.n_vcs):
                        for l1 in range(self.n_vcs):
                            if not self.allow_vc_trans and l0 != l1:
                                continue
                            yield ((i, k, l0), (k, j, l1))

    def _iter_topo_edges(self):
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i != j:
                    yield (i, j)

    def _iter_ccdg_nodes(self):
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i != j:
                    for l in range(self.n_vcs):
                        yield (i, j, l)

    def _create_cdg_maps(self):
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

    def _add_topo_vars(self):
        topo_edge_var_type = (
            GRB.CONTINUOUS if self.relax_topo_edges else GRB.BINARY
        )
        self.vars_topo_adj_mat = {}
        for (i, j) in self._iter_topo_edges():
            if (i, j) in self.hardened_topo_edges:
                lb, ub, vtype = 1.0, 1.0, GRB.BINARY
            elif self.hardened_topo_edges and not self.relax_topo_edges:
                lb, ub, vtype = 0.0, 0.0, GRB.BINARY
            else:
                lb, ub, vtype = 0.0, 1.0, topo_edge_var_type
            self.vars_topo_adj_mat[(i, j)] = self.model.addVar(
                lb=lb, ub=ub, vtype=vtype, name=f"m_{i}i_{j}j"
            )

    def _add_turn_vars(self):
        turn_edge_var_type = (
            GRB.CONTINUOUS if self.relax_turn_edges else GRB.BINARY
        )
        self.vars_turn_adj_mat = {}
        for (u, v) in self.uv_turn_set:
            if (u, v) in self.hardened_turn_edges:
                lb, ub, vtype = 1.0, 1.0, GRB.BINARY
            elif self.hardened_turn_edges and not self.relax_turn_edges:
                lb, ub, vtype = 0.0, 0.0, GRB.BINARY
            else:
                lb, ub, vtype = 0.0, 1.0, turn_edge_var_type
            self.vars_turn_adj_mat[(u, v)] = self.model.addVar(
                lb=lb, ub=ub, vtype=vtype, name=f"c_{u}u_{v}v"
            )

    def _add_flow_vars(self):
        vtype = GRB.INTEGER if self.integral_flow else GRB.CONTINUOUS
        self.vars_uv_flow = defaultdict(dict)
        self.vars_ss_flow = defaultdict(dict)
        for s in self.source_commodities:
            for (u, v) in self.uv_turn_set:
                self.vars_uv_flow[s][(u, v)] = self.model.addVar(
                    lb=0.0, ub=1.0, vtype=vtype, name=f"fuv_{s}s_{u}u_{v}v"
                )
            for u in self.ss_to_u_conn_map[s]:
                self.vars_ss_flow[s][u] = self.model.addVar(
                    lb=0.0, ub=1.0, vtype=vtype, name=f"fss_{s}s_{u}u"
                )

    def _add_mtz_vars(self):
        self.vars_o = {
            u: self.model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"o_{u}u")
            for u in range(self.u_id)
        }

    def _add_sink_vars(self):
        self.vars_sink = defaultdict(dict)
        for s in self.source_commodities:
            for u in range(self.n_cdg):
                self.vars_sink[s][u] = self.model.addVar(
                    lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"s_{s}s_{u}u"
                )

    def _constr_symmetric_links(self):
        for (i, j) in self._iter_topo_edges():
            self.model.addConstr(
                self.vars_topo_adj_mat[(i, j)] == self.vars_topo_adj_mat[(j, i)],
                name=f"sym_link_{i}i_{j}j",
            )

    def _constr_radix(self):
        for i in range(self.n_nodes):
            self.model.addConstr(
                gp.quicksum(
                    self.vars_topo_adj_mat[(i, j)]
                    for j in range(self.n_nodes)
                    if i != j
                )
                <= self.radix,
                name=f"r_{i}",
            )

    def _constr_cdg_to_topo_mapping(self):
        for ((i, k, l0), (k, j, l1)) in self._iter_turns():
            u = self.topo_ijv_to_cdg_u_map[(i, k, l0)]
            v = self.topo_ijv_to_cdg_u_map[(k, j, l1)]
            tag = f"c2t_{i}i_{k}k_{j}j_{l0}l0_{l1}l1"
            self.model.addConstr(
                self.vars_turn_adj_mat[(u, v)] <= self.vars_topo_adj_mat[(i, k)],
                name=f"{tag}_1",
            )
            self.model.addConstr(
                self.vars_turn_adj_mat[(u, v)] <= self.vars_topo_adj_mat[(k, j)],
                name=f"{tag}_2",
            )

    def _constr_flow_to_cdg_mapping(self):
        for (u, v) in self.uv_turn_set:
            i, k, _ = self.cdg_u_to_topo_ijv_map[u]
            k2, j, _ = self.cdg_u_to_topo_ijv_map[v]
            assert k == k2
            min_cap = min(self.capacity[i][k], self.capacity[k][j])
            self.model.addConstr(
                gp.quicksum(self.vars_uv_flow[s][(u, v)] for s in self.source_commodities)
                <= min_cap * self.vars_turn_adj_mat[(u, v)],
                name=f"f2c_{u}u_{v}v",
            )

    def _constr_physical_link_capacity(self):
        for (i, j) in self._iter_topo_edges():
            expr = gp.LinExpr()
            for l in range(self.n_vcs):
                u = self.topo_ijv_to_cdg_u_map[(i, j, l)]
                for s in self.source_commodities:
                    expr += self.vars_sink[s][u]
                    for v in self.u_to_v_turns[u]:
                        expr += self.vars_uv_flow[s][(u, v)]
                self.model.addConstr(
                    expr <= self.capacity[i][j] * self.vars_topo_adj_mat[(i, j)],
                    name=f"phys_cap_{i}i_{j}j",
                )

    def _constr_flow_conservation(self):
        for s in self.source_commodities:
            for u in range(self.n_cdg):
                in_expr = gp.quicksum(
                    self.vars_uv_flow[s][(v, u)] for v in self.v_to_u_turns[u]
                )
                if u in self.ss_to_u_conn_map[s]:
                    in_expr += self.vars_ss_flow[s][u]
                out_expr = self.vars_sink[s][u] + gp.quicksum(
                    self.vars_uv_flow[s][(u, v)] for v in self.u_to_v_turns[u]
                )
                self.model.addConstr(in_expr == out_expr, name=f"flow_con_{s}s_{u}u")

    def _constr_super_source_production(self):
        for s, dests in self.commodities.items():
            tot_demand = sum(self.demand[s][d] for d in dests)
            self.model.addConstr(
                gp.quicksum(self.vars_ss_flow[s][u] for u in self.ss_to_u_conn_map[s])
                >= self.var_mcf * tot_demand,
                name=f"ss_prod_{s}s",
            )

    def _constr_ccdg_node_consumption(self):
        for s, dests in self.commodities.items():
            for d in dests:
                if s == d:
                    continue
                self.model.addConstr(
                    gp.quicksum(self.vars_sink[s][u] for u in self.sd_to_u_conn_map[d])
                    >= self.var_mcf * self.demand[s][d],
                    name=f"sd_con_{s}s_{d}d",
                )

    def _constr_mtz_acyclicality(self):
        for (u, v) in self.uv_turn_set:
            self.model.addConstr(
                self.vars_o[u] - self.vars_o[v] + self.n_cdg * self.vars_turn_adj_mat[(u, v)]
                <= self.n_cdg - 1,
                name=f"mtz_{u}u_{v}v",
            )

    def build_model_(self):
        self._create_cdg_maps()
        print("Building Gurobi model...")
        self.model = gp.Model("ccdg_model")
        self.var_mcf = self.model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="lambda")

        self._add_topo_vars()
        self._add_turn_vars()
        self._add_flow_vars()
        self._add_mtz_vars()
        self._add_sink_vars()

        self._constr_radix()
        if self.symmetric_links:
            self._constr_symmetric_links()
        self._constr_cdg_to_topo_mapping()
        self._constr_flow_to_cdg_mapping()
        self._constr_physical_link_capacity()
        self._constr_flow_conservation()
        self._constr_super_source_production()
        self._constr_ccdg_node_consumption()
        self._constr_mtz_acyclicality()

        self.model.setObjective(self.var_mcf, GRB.MAXIMIZE)
        print("Model build complete.")

    def write_model_(self, path):
        self.model.write(path)
        print(f"Wrote model to {path}")
