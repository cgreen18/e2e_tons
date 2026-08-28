#!/usr/bin/env python3
"""
One-shot CDG topology + routing synthesis (Gurobi).

Implements the formulation in main_formulation.tex: maximize lambda (O1) subject
to C4-C12. C1-C3 are enforced by construction (no self edges, no U-turns, no
rebuffer turns in the turn list).
"""

import argparse
import json
import os
import sys
from collections import defaultdict, deque

import gurobipy as gp
from gurobipy import GRB


def _edge_key(u, v):
    """Canonical undirected edge key (min, max). Implements C4 by representation."""
    return (u, v) if u < v else (v, u)


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


def _build_candidate_undirected_edges_ring(n, cand_k, allow_self=False):
    """Undirected candidate links; C1 (no self-loop) by skipping u==v."""
    if cand_k is None:
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                edges.append((u, v))
        return edges

    r = max(1, cand_k // 2)
    eset = set()
    for u in range(n):
        for d in range(1, r + 1):
            v1 = (u + d) % n
            v2 = (u - d) % n
            if not allow_self and v1 == u:
                continue
            eset.add(_edge_key(u, v1))
            if not allow_self and v2 == u:
                continue
            eset.add(_edge_key(u, v2))
    return sorted(eset)


def _build_directed_arcs_from_undirected(undirected_edges, n_vcs):
    """CDG nodes are directed VC-aware physical arcs (i, j, l)."""
    arcs = []
    for u, v in undirected_edges:
        for vc in range(n_vcs):
            arcs.append((u, v, vc))
            arcs.append((v, u, vc))
    return arcs


def _build_turns_filtered(arcs):
    """
    Valid CDG turns (u, v) with u, v in V_C.

    Excludes C2 U-turns: (i,j,l0) followed by (j,i,l1).
    Excludes C3 rebuffer: same directed physical hop (i,j) with only VC change.
    """
    turns = []
    for arc_u in arcs:
        for arc_v in arcs:
            i, k, _l0 = arc_u
            k2, j, _l1 = arc_v
            if k != k2:
                continue
            if i == j:
                continue
            if arc_u[0] == arc_v[1] and arc_u[1] == arc_v[0]:
                continue
            if (
                arc_u[0] == arc_v[0]
                and arc_u[1] == arc_v[1]
                and arc_u[2] != arc_v[2]
            ):
                continue
            turns.append((arc_u, arc_v))
    return turns


class TopologyGraph:
    """
    Physical topology candidates, CDG nodes (directed VC arcs), and turns.

    C1-C3: reflexivity and forbidden turn types are enforced when building arcs
    and turns (no m_{i,i}, no c_{u,u}, no U-turn or rebuffer turns in E_C).
    """

    def __init__(
        self,
        n_nodes,
        n_vcs,
        cand_k,
        capacity_matrix,
        demand_matrix,
        commodities,
    ):
        self.n_nodes = int(n_nodes)
        self.n_vcs = int(n_vcs)
        self.cand_k = cand_k
        self.capacity_matrix = capacity_matrix
        self.demand_matrix = demand_matrix
        self.commodities = commodities

        self.undirected_edges = _build_candidate_undirected_edges_ring(
            self.n_nodes, self.cand_k, allow_self=False
        )
        self.undirected_set = set(self.undirected_edges)

        self.arcs = _build_directed_arcs_from_undirected(
            self.undirected_edges, self.n_vcs
        )
        self.arc_set = set(self.arcs)
        self.n_c = len(self.arcs)

        self.turns = _build_turns_filtered(self.arcs)
        self._build_adjacency_indexes()

    def _build_adjacency_indexes(self):
        """Sparse structures for CDGOptimizer (C7, C8, C9)."""
        self.in_turns = defaultdict(list)
        self.out_turns = defaultdict(list)
        for turn in self.turns:
            u, v = turn
            self.out_turns[u].append(turn)
            self.in_turns[v].append(turn)

        self.arcs_by_tail = defaultdict(list)
        for a in self.arcs:
            self.arcs_by_tail[a[0]].append(a)

        self.arcs_by_head = defaultdict(list)
        for a in self.arcs:
            self.arcs_by_head[a[1]].append(a)

        self.directed_physical_inflow_terms = defaultdict(list)
        for arc in self.arcs:
            i, k, _l0 = arc
            self.directed_physical_inflow_terms[(i, k)].append(arc)

        self.turn_inflow_to_arc = defaultdict(list)
        for turn in self.turns:
            pred, succ = turn
            self.turn_inflow_to_arc[succ].append(turn)

        self.B_turn = {}
        for turn in self.turns:
            pred, succ = turn
            i, k, _l0 = pred
            _k2, j, _l1 = succ
            c_ik = self.capacity_matrix[i][k]
            c_kj = self.capacity_matrix[k][j]
            self.B_turn[turn] = min(c_ik, c_kj)

    def ss_arc_exists(self, commodity_id, arc):
        """Super-source may only inject at arcs whose tail is the commodity source."""
        s, _d = self.commodities[commodity_id]
        return arc[0] == s and arc in self.arc_set

    def sd_arc_exists(self, commodity_id, arc):
        """Super-sink may only absorb at arcs whose head is the commodity sink."""
        _s, d = self.commodities[commodity_id]
        return arc[1] == d and arc in self.arc_set

    def demand_for(self, commodity_id):
        s, d = self.commodities[commodity_id]
        return self.demand_matrix[s][d]


class CDGOptimizer:
    """
    Gurobi MILP: O1 maximize lambda; constraints C4–C12.

    Indexing:
      m[e] for e in undirected_edges (C4 symmetric).
      c[t] for t in turns.
      f[(kid, u, v)] for flow on arc u->v in extended network (CDG nodes or ss/sd).
      o[u] for u in arcs (MTZ on CDG nodes only per C12 on E_C).
    """

    def __init__(
        self,
        graph,
        radix,
        binary_edges,
        model_params,
        fixed_topology_edges=None,
    ):
        self.graph = graph
        self.radix = int(radix)
        self.binary_edges = bool(binary_edges)
        self.model_params = dict(model_params) if model_params else {}
        self.fixed_topology_edges = (
            set(fixed_topology_edges) if fixed_topology_edges is not None else None
        )
        self.model = None
        self.m_vars = None
        self.c_vars = None
        self.f_vars = None
        self.o_vars = None
        self.lam = None

    def _ss_label(self, kid):
        return ("ss", kid)

    def _sd_label(self, kid):
        return ("sd", kid)

    def build_and_solve(self):
        """Construct model, optimize, return (model, var dict refs)."""
        g = self.graph
        self.model = gp.Model("cdg_one_shot_mcf")

        mp = self.model_params
        if mp.get("silent"):
            self.model.Params.OutputFlag = 0
        if mp.get("threads") is not None:
            self.model.Params.Threads = int(mp["threads"])
        if mp.get("time_limit") is not None:
            self.model.Params.TimeLimit = float(mp["time_limit"])
        if mp.get("mip_gap") is not None:
            self.model.Params.MIPGap = float(mp["mip_gap"])

        vtype = GRB.BINARY if self.binary_edges else GRB.CONTINUOUS

        self.m_vars = self.model.addVars(
            g.undirected_edges,
            lb=0.0,
            ub=1.0,
            vtype=vtype,
            name="m",
        )

        self.c_vars = self.model.addVars(
            g.turns,
            lb=0.0,
            ub=1.0,
            vtype=vtype,
            name="c",
        )

        n_c = g.n_c
        self.o_vars = self.model.addVars(
            g.arcs,
            lb=0.0,
            ub=float(max(0, n_c - 1)),
            vtype=GRB.CONTINUOUS,
            name="o",
        )

        self.lam = self.model.addVar(lb=0.0, name="lambda")

        f_keys = []
        commodities = range(len(g.commodities))
        for kid in commodities:
            ss = self._ss_label(kid)
            sd = self._sd_label(kid)
            for arc in g.arcs:
                if g.ss_arc_exists(kid, arc):
                    f_keys.append((kid, ss, arc))
                if g.sd_arc_exists(kid, arc):
                    f_keys.append((kid, arc, sd))
            for turn in g.turns:
                u, v = turn
                f_keys.append((kid, u, v))

        self.f_vars = self.model.addVars(f_keys, lb=0.0, name="f")

        self._add_fixed_topology_constraints()
        self._add_c5_radix()
        self._add_c6_cdg_topology_map()
        self._add_c7_turn_capacity()
        self._add_c8_physical_link_capacity()
        self._add_c9_flow_conservation()
        self._add_c10_c11_super_source_sink()
        self._add_c12_mtz()

        self.model.setObjective(self.lam, GRB.MAXIMIZE)
        self.model.optimize()
        return self.model

    def _add_fixed_topology_constraints(self):
        """
        Fix topology to a provided integral edge set.

        If fixed_topology_edges is provided, each candidate edge is fixed to
        1 when selected, else 0. This enforces routing/VC search on one chosen
        binary topology.
        """
        if self.fixed_topology_edges is None:
            return
        g = self.graph
        chosen = self.fixed_topology_edges
        for e in g.undirected_edges:
            rhs = 1.0 if e in chosen else 0.0
            self.model.addConstr(
                self.m_vars[e] == rhs,
                name=f"fix_topology_{e}",
            )

    def _add_c5_radix(self):
        """
        C5: per node i, sum of m over incident undirected links <= radix.

        Each undirected candidate edge appears once in the sum at i, so this is
        the physical neighbor count (and matches directed out-degree from i in
        a symmetric topology).
        """
        g = self.graph
        inc_by_node = defaultdict(list)
        for e in g.undirected_edges:
            u, v = e
            inc_by_node[u].append(e)
            inc_by_node[v].append(e)
        for i in range(g.n_nodes):
            edges_i = inc_by_node[i]
            self.model.addConstr(
                gp.quicksum(self.m_vars[e] for e in edges_i) <= self.radix,
                name=f"c5_radix_{i}",
            )

    def _add_c6_cdg_topology_map(self):
        """C6: turn selection implies both incident undirected links are present."""
        g = self.graph
        for turn in g.turns:
            pred, succ = turn
            i, k, _l0 = pred
            _k2, j, _l1 = succ
            e_ik = _edge_key(i, k)
            e_kj = _edge_key(k, j)
            self.model.addConstr(
                self.c_vars[turn] <= self.m_vars[e_ik],
                name=f"c6_turn_le_m_{turn}",
            )
            self.model.addConstr(
                self.c_vars[turn] <= self.m_vars[e_kj],
                name=f"c6_turn_le_m2_{turn}",
            )

    def _add_c7_turn_capacity(self):
        """C7: sum over commodities of turn flow <= B_{uv} * c_{uv}."""
        g = self.graph
        for turn in g.turns:
            b = g.B_turn[turn]
            flow_sum = gp.quicksum(self.f_vars[kid, turn[0], turn[1]] for kid in range(len(g.commodities)))
            self.model.addConstr(
                flow_sum <= b * self.c_vars[turn],
                name=f"c7_turn_cap_{turn}",
            )

    def _add_c8_physical_link_capacity(self):
        """
        C8: for each directed physical pair (i,k), aggregate inflow to all VC
        copies of arc (i,k) <= C_{i,k} * m_{i,k}.
        """
        g = self.graph
        for i in range(g.n_nodes):
            for k in range(g.n_nodes):
                if i == k:
                    continue
                e_ik = _edge_key(i, k)
                if e_ik not in g.undirected_set:
                    continue
                c_ik = g.capacity_matrix[i][k]
                terms = []
                for kid in range(len(g.commodities)):
                    ss = self._ss_label(kid)
                    for arc in g.directed_physical_inflow_terms.get((i, k), []):
                        if (kid, ss, arc) in self.f_vars:
                            terms.append(self.f_vars[kid, ss, arc])
                        for turn in g.turn_inflow_to_arc.get(arc, []):
                            if (kid, turn[0], turn[1]) in self.f_vars:
                                terms.append(self.f_vars[kid, turn[0], turn[1]])
                if not terms:
                    continue
                lhs = gp.quicksum(terms)
                self.model.addConstr(
                    lhs <= c_ik * self.m_vars[e_ik],
                    name=f"c8_phys_{i}_{k}",
                )

    def _add_c9_flow_conservation(self):
        """
        C9: for each commodity s and each CDG node u in V_C,
        f_{ss,u} + sum_v f_{v,u} = f_{u,sd} + sum_v f_{u,v}.
        """
        g = self.graph
        for kid in range(len(g.commodities)):
            ss = self._ss_label(kid)
            sd = self._sd_label(kid)
            for u in g.arcs:
                flow_in_ss = self.f_vars[kid, ss, u] if (kid, ss, u) in self.f_vars else 0
                flow_in_turn = gp.quicksum(
                    self.f_vars[kid, p, q]
                    for p, q in g.in_turns.get(u, [])
                    if (kid, p, q) in self.f_vars
                )
                flow_out_sd = self.f_vars[kid, u, sd] if (kid, u, sd) in self.f_vars else 0
                flow_out_turn = gp.quicksum(
                    self.f_vars[kid, p, q]
                    for p, q in g.out_turns.get(u, [])
                    if (kid, p, q) in self.f_vars
                )
                self.model.addConstr(
                    flow_in_ss + flow_in_turn == flow_out_sd + flow_out_turn,
                    name=f"c9_bal_k{kid}_u{u}",
                )

    def _add_c10_c11_super_source_sink(self):
        """C10/C11: production and consumption at least lambda * D_s."""
        g = self.graph
        for kid in range(len(g.commodities)):
            ss = self._ss_label(kid)
            sd = self._sd_label(kid)
            d_k = g.demand_for(kid)
            out_ss = gp.quicksum(
                self.f_vars[kid, ss, arc]
                for arc in g.arcs
                if (kid, ss, arc) in self.f_vars
            )
            in_sd = gp.quicksum(
                self.f_vars[kid, arc, sd]
                for arc in g.arcs
                if (kid, arc, sd) in self.f_vars
            )
            self.model.addConstr(
                out_ss >= self.lam * d_k,
                name=f"c10_ss_k{kid}",
            )
            self.model.addConstr(
                in_sd >= self.lam * d_k,
                name=f"c11_sd_k{kid}",
            )

    def _add_c12_mtz(self):
        """C12: o_u - o_v + n_c * c_{uv} <= n_c - 1 for each turn (u,v) in E_C."""
        g = self.graph
        n_c = g.n_c
        if n_c == 0:
            return
        for turn in g.turns:
            u, v = turn
            self.model.addConstr(
                self.o_vars[u] - self.o_vars[v] + float(n_c) * self.c_vars[turn]
                <= float(n_c - 1),
                name=f"c12_mtz_{turn}",
            )


class ResultExtractor:
    """Parse Gurobi solution into matrices, flow support, and routepath / VC outputs."""

    def __init__(self, graph, optimizer, tol=1e-5):
        self.graph = graph
        self.optimizer = optimizer
        self.tol = tol

    def _f_x(self, key):
        """Gurobi flow variable value for key (kid, u, v) or 0 if missing."""
        f = self.optimizer.f_vars
        if f is None or key not in f:
            return 0.0
        return float(f[key].X)

    def _adj_from_edge_set(self, edge_set):
        """Undirected adjacency from a set of canonical undirected edges (u,v), u<v."""
        n = self.graph.n_nodes
        adj = {i: [] for i in range(n)}
        if not edge_set:
            return adj
        for u, v in edge_set:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def _topology_edges_under_radix(self, m_vars, radix):
        """
        Choose a subset of undirected edges so every node has degree <= radix.

        With continuous m, thresholding m.X > tol can mark more than radix
        edges per node even when sum_e m_e <= radix holds. This greedy keeps
        the highest-m edges while respecting the degree cap (same cap as C5).
        """
        g = self.graph
        if m_vars is None or radix is None:
            return set()
        r = int(radix)
        scored = sorted(
            ((float(m_vars[e].X), e) for e in g.undirected_edges),
            key=lambda t: t[0],
            reverse=True,
        )
        deg = [0] * g.n_nodes
        chosen = set()
        for _score, e in scored:
            u, v = e
            if deg[u] < r and deg[v] < r:
                chosen.add(e)
                deg[u] += 1
                deg[v] += 1
        return chosen

    def extract_path_and_vcs_for_commodity(self, kid, m_vars, physical_edges):
        """
        Build one physical path and per-hop VCs from commodity kid's flow support.

        State is the current CDG arc (tail, head, vc). BFS follows turns with
        positive f; starts from ss injection arcs; ends at any arc with head d.
        If no flow path is found, falls back to shortest hop-count path on m
        with vc=0 (may be inconsistent with strict CDG; emits stderr warning).
        """
        g = self.graph
        ss = ("ss", kid)
        sd = ("sd", kid)
        s, d = g.commodities[kid]
        tol = self.tol

        start_arcs = []
        for arc in g.arcs:
            if arc[0] != s:
                continue
            if self._f_x((kid, ss, arc)) > tol:
                start_arcs.append(arc)

        visited = set()
        queue = deque()
        for arc0 in start_arcs:
            i0, j0, l0 = arc0
            path = [i0, j0]
            vcs = [l0]
            queue.append((arc0, path, vcs))
            visited.add(arc0)

        while queue:
            tail_arc, path, vcs = queue.popleft()
            _ti, tj, _tl = tail_arc
            if tj == d:
                return path, vcs
            for turn in g.out_turns.get(tail_arc, []):
                if self._f_x((kid, turn[0], turn[1])) <= tol:
                    continue
                pred, succ = turn
                _a, b, _lb = pred
                jb, nbr, lc = succ
                if jb != b:
                    continue
                if succ in visited:
                    continue
                visited.add(succ)
                queue.append((succ, path + [nbr], vcs + [lc]))

        adj = self._adj_from_edge_set(physical_edges)
        if d in adj[s]:
            for vc in range(g.n_vcs):
                arc = (s, d, vc)
                if arc in g.arc_set and self._f_x((kid, ss, arc)) > tol:
                    return [s, d], [vc]

        path_fb = self._shortest_physical_path(adj, s, d)
        if path_fb is not None and len(path_fb) >= 2:
            if not self._silent_extract_warnings():
                print(
                    f"Warning: commodity ({s},{d}) id={kid}: no positive-flow path; "
                    f"using shortest-path fallback with vc=0.",
                    file=sys.stderr,
                )
            return path_fb, [0] * (len(path_fb) - 1)

        if not self._silent_extract_warnings():
            print(
                f"Warning: commodity ({s},{d}) id={kid}: unreachable; path [ {s} ].",
                file=sys.stderr,
            )
        return [s], []

    def _silent_extract_warnings(self):
        mp = getattr(self.optimizer, "model_params", None) or {}
        return bool(mp.get("silent"))

    def _shortest_physical_path(self, adj, s, d):
        """Unweighted BFS on adj for path [s,...,d] or None."""
        if s == d:
            return [s]
        prev = {s: None}
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj.get(u, []):
                if v in prev:
                    continue
                prev[v] = u
                q.append(v)
                if v == d:
                    out = []
                    cur = d
                    while cur is not None:
                        out.append(cur)
                        cur = prev[cur]
                    out.reverse()
                    return out
        return None

    def build_routing_outputs(self, m_vars, physical_edges):
        """
        Per commodity (in graph.commodities order): path (list of routers) and
        per-hop VC list aligned to edges path[i]->path[i+1].
        """
        rows = []
        for kid in range(len(self.graph.commodities)):
            path, vcs = self.extract_path_and_vcs_for_commodity(
                kid, m_vars, physical_edges
            )
            s, d = self.graph.commodities[kid]
            rows.append({"s": s, "d": d, "path": path, "vcs": vcs})
        return rows

    def write_paths_file(self, path, routing_rows, header=None):
        """
        Routepath_lists .paths format (see pathlist_to_xml.py): first line header
        string, then one JSON array per line [s, ..., d] per commodity row order.
        """
        if header is None:
            header = os.path.splitext(os.path.basename(path))[0]
        with open(path, "w") as fp:
            fp.write(header + "\n")
            for row in routing_rows:
                p = row["path"]
                fp.write(json.dumps(p) + "\n")

    def write_vcmat2_file(self, path, routing_rows):
        """
        One line per hop: (path_src, path_dest, cur_node, vc) with spaces after
        commas, matching topologies_and_routing/vc_mats/*.vcmat2 style.
        cur_node is the tail of the hop; vc is used on hop cur_node -> next.
        """
        with open(path, "w") as fp:
            for row in routing_rows:
                s, d = row["s"], row["d"]
                p = row["path"]
                vcs = row["vcs"]
                for i in range(len(p) - 1):
                    vc = vcs[i] if i < len(vcs) else 0
                    fp.write(f"({s}, {d}, {p[i]}, {vc})\n")

    def extract(self):
        """Return dict with topology matrix, lambda, active edges/turns, flows."""
        m = self.optimizer.m_vars
        c = self.optimizer.c_vars
        f = self.optimizer.f_vars
        lam_var = self.optimizer.lam
        n = self.graph.n_nodes

        radix = int(getattr(self.optimizer, "radix", 0) or 0)
        physical_edges = self._topology_edges_under_radix(m, radix)

        topology = [[0] * n for _ in range(n)]
        active_edges = sorted(physical_edges)
        for e in physical_edges:
            u, v = e
            topology[u][v] = 1
            topology[v][u] = 1

        active_turns = []
        if c is not None:
            for t, var in c.items():
                if var.X > self.tol:
                    active_turns.append(t)

        flow_support = defaultdict(float)
        if f is not None:
            for key, var in f.items():
                if var.X > self.tol:
                    flow_support[key] = float(var.X)

        lam_val = float(lam_var.X) if lam_var is not None else 0.0

        routing_rows = self.build_routing_outputs(m, physical_edges)

        return {
            "topology_matrix": topology,
            "lambda": lam_val,
            "active_undirected_edges": active_edges,
            "active_turns": active_turns,
            "flow_support": dict(flow_support),
            "objective_value": lam_val,
            "routing_rows": routing_rows,
        }

    def write_topology_matrix(self, path, topology_matrix):
        with open(path, "w") as fp:
            for row in topology_matrix:
                fp.write(" ".join(str(x) for x in row) + "\n")

    def write_json(self, path, payload):
        with open(path, "w") as fp:
            json.dump(payload, fp, indent=2, default=str)


def _default_commodities(n):
    pairs = []
    for s in range(n):
        for d in range(n):
            if s != d:
                pairs.append((s, d))
    return pairs


def _parse_commodities_file(path, n):
    pairs = []
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
            pairs.append((s, d))
    if not pairs:
        raise ValueError("commodities file: no valid pairs")
    return pairs


def _uniform_matrix(n, value):
    return [[float(value) for _ in range(n)] for _ in range(n)]


def main():
    ap = argparse.ArgumentParser(
        description="One-shot CDG + MCF topology synthesis (main_formulation.tex)",
    )
    ap.add_argument("--n_nodes", type=int, required=True, help="Number of topology nodes n_T")
    ap.add_argument("--radix", type=int, required=True, help="Maximum degree r (C5)")
    ap.add_argument("--n_vcs", type=int, required=True, help="Number of virtual channels n_vc")
    ap.add_argument("--out", type=str, required=True, help="Output topology adjacency matrix path")
    ap.add_argument(
        "--out-paths",
        type=str,
        default=None,
        help="Routepath .paths file (header line + JSON path per commodity order); "
        "default: same basename as --out with .paths extension",
    )
    ap.add_argument(
        "--out-vcmat2",
        type=str,
        default=None,
        help="VC matrix .vcmat2 file: one (path_src, path_dest, cur_node, vc) per hop; "
        "default: same basename as --out with .vcmat2 extension",
    )
    ap.add_argument(
        "--paths-header",
        type=str,
        default=None,
        help="First line of .paths file (default: basename of --out-paths without ext)",
    )
    ap.add_argument(
        "--out-json",
        type=str,
        default=None,
        help="Optional JSON dump of solution summary",
    )
    ap.add_argument("--binary_edges", action="store_true", help="Binary m,c (full MILP)")
    ap.add_argument("--cand_k", type=int, default=None, help="Ring candidate neighborhood size")
    ap.add_argument("--capacity", type=float, default=1.0, help="Scalar C when no capacity matrix")
    ap.add_argument("--demand", type=float, default=1.0, help="Scalar D when no demand matrix")
    ap.add_argument(
        "--capacity-matrix",
        type=str,
        default=None,
        help="n x n file overriding per-pair capacities C_ij",
    )
    ap.add_argument(
        "--demand-matrix",
        type=str,
        default=None,
        help="n x n file overriding per-pair demands D_ij (commodities use D[s,d])",
    )
    ap.add_argument(
        "--commodities-file",
        type=str,
        default=None,
        help="Optional file of lines 's d' (default: all ordered pairs s!=d)",
    )
    ap.add_argument("--time_limit", type=float, default=None)
    ap.add_argument("--mip_gap", type=float, default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--silent", action="store_true")

    args = ap.parse_args()
    n = args.n_nodes

    cap_mat = _uniform_matrix(n, args.capacity)
    dem_mat = _uniform_matrix(n, args.demand)
    if args.capacity_matrix:
        cap_mat = _read_matrix_file(args.capacity_matrix, n)
    if args.demand_matrix:
        dem_mat = _read_matrix_file(args.demand_matrix, n)

    if args.commodities_file:
        commodities = _parse_commodities_file(args.commodities_file, n)
    else:
        commodities = _default_commodities(n)

    graph = TopologyGraph(
        n_nodes=n,
        n_vcs=args.n_vcs,
        cand_k=args.cand_k,
        capacity_matrix=cap_mat,
        demand_matrix=dem_mat,
        commodities=commodities,
    )

    model_params = {
        "silent": args.silent,
        "threads": args.threads,
        "time_limit": args.time_limit,
        "mip_gap": args.mip_gap,
    }
    optimizer = CDGOptimizer(
        graph=graph,
        radix=args.radix,
        binary_edges=args.binary_edges,
        model_params=model_params,
    )
    model = optimizer.build_and_solve()

    if model.SolCount == 0:
        print(f"No solution; status={model.Status}", file=sys.stderr)
        sys.exit(1)

    # Pass 1: topology optimization. Pick one integral topology that respects radix.
    topo_extractor = ResultExtractor(graph, optimizer)
    chosen_edges = topo_extractor._topology_edges_under_radix(
        optimizer.m_vars, args.radix
    )

    # Pass 2: routing + VC assignment on the fixed integral topology only.
    routing_optimizer = CDGOptimizer(
        graph=graph,
        radix=args.radix,
        binary_edges=args.binary_edges,
        model_params=model_params,
        fixed_topology_edges=chosen_edges,
    )
    routing_model = routing_optimizer.build_and_solve()
    if routing_model.SolCount == 0:
        print(
            "No routing solution on the selected integral topology; "
            f"status={routing_model.Status}",
            file=sys.stderr,
        )
        sys.exit(1)

    extractor = ResultExtractor(graph, routing_optimizer)
    result = extractor.extract()
    extractor.write_topology_matrix(args.out, result["topology_matrix"])

    out_paths = args.out_paths
    if out_paths is None:
        out_paths = os.path.splitext(args.out)[0] + ".paths"
    out_vcmat2 = args.out_vcmat2
    if out_vcmat2 is None:
        out_vcmat2 = os.path.splitext(args.out)[0] + ".vcmat2"

    paths_header = args.paths_header
    if paths_header is None:
        paths_header = os.path.splitext(os.path.basename(out_paths))[0]

    extractor.write_paths_file(
        out_paths, result["routing_rows"], header=paths_header
    )
    extractor.write_vcmat2_file(out_vcmat2, result["routing_rows"])

    if not args.silent:
        print(f"Wrote topology to {args.out}")
        print(f"Wrote pathlist to {out_paths}")
        print(f"Wrote VC allocation to {out_vcmat2}")
        print(f"lambda = {result['lambda']:.6g}")
        print(f"active undirected edges = {len(result['active_undirected_edges'])}")

    if args.out_json:
        out_payload = {
            "lambda": result["lambda"],
            "n_nodes": n,
            "n_commodities": len(commodities),
            "n_cdg_nodes": graph.n_c,
            "n_turns": len(graph.turns),
            "active_undirected_edges": [list(e) for e in result["active_undirected_edges"]],
            "out_paths": out_paths,
            "out_vcmat2": out_vcmat2,
        }
        extractor.write_json(args.out_json, out_payload)


if __name__ == "__main__":
    main()
