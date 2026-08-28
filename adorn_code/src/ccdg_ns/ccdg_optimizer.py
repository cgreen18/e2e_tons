"""Solve, extract, and harden complete CDG synthesis results."""

import math
import random
from collections import defaultdict

from gurobipy import GRB


def stripped_path_dict(s, d, cdg_nodes, phys_nodes, vcs, W_k):
    return {
        "s": s,
        "d": d,
        "cdg_nodes": tuple(cdg_nodes),
        "phys_nodes": tuple(phys_nodes),
        "vcs": tuple(vcs),
        "W_k": float(W_k),
    }


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
        self.relax_topo_edges = ccdg_model.relax_topo_edges
        self.relax_turn_edges = ccdg_model.relax_turn_edges
        self.k_paths = ccdg_model.k_paths
        self.commodities = ccdg_model.commodities
        self.capacity = ccdg_model.capacity
        self.demand = ccdg_model.demand
        self.cdg_u_to_topo_ijv_map = ccdg_model.cdg_u_to_topo_ijv_map
        self.ss_to_u_conn_map = ccdg_model.ss_to_u_conn_map
        self.v_to_u_turns = ccdg_model.v_to_u_turns

        self.model_params = model_params
        self.beta = model_params.get("beta", 0.5)
        self.weight_type = model_params.get("weight_type", "max")
        self.choice_type = model_params.get("choice_type", "strict")
        self.k_path_traf_policy = model_params.get("k_path_traf_policy", "equal")
        self._rng = random.Random(model_params.get("random_seed"))

        _allowed_pol = ("equal", "weight_proportional", "score_proportional")
        if self.k_path_traf_policy not in _allowed_pol:
            raise ValueError(
                f"k_path_traf_policy must be one of {_allowed_pol}, got {self.k_path_traf_policy!r}"
            )
        if self.weight_type not in ("max", "average", "avg"):
            raise ValueError(f"weight_type must be 'max' or 'average', got {self.weight_type!r}")
        if self.weight_type == "avg":
            self.weight_type = "average"
        if self.choice_type not in ("strict", "probabilistic"):
            raise ValueError(f"choice_type must be 'strict' or 'probabilistic', got {self.choice_type!r}")

        if model_params.get("silent"):
            self.model.Params.OutputFlag = 0
        if model_params.get("threads") is not None:
            self.model.Params.Threads = int(model_params["threads"])
        if model_params.get("time_limit") is not None:
            self.model.Params.TimeLimit = float(model_params["time_limit"])
        if model_params.get("mip_gap") is not None:
            self.model.Params.MIPGap = float(model_params["mip_gap"])

        self.lam = None
        self.objval = None
        self.topo_adj_mat_vals = None
        self.turn_adj_mat_vals = None
        self.flow_uv_vals = None
        self.flow_ss_vals = None
        self.sink_vals = None
        self.topology = None
        self.hardened_turn_edges = None
        self.paths_w_vcs = None

    def solve(self):
        self.model.optimize()
        print(f"Gurobi ended with status {self.model.Status}")
        if self.model.SolCount == 0:
            raise RuntimeError(f"No solution available. Status={self.model.Status}")
        if self.model.Status != GRB.OPTIMAL:
            print(f"Non-optimal solution. Status={self.model.Status}")
        self.objval = self.model.objVal
        print(f"Solve w/ obj {self.objval}")
        return self.model

    def dump_var_vals_to_file_(self, out_file_path):
        with open(out_file_path, "w", encoding="utf-8") as f:
            for v in sorted(self.model.getVars(), key=lambda x: x.VarName):
                f.write(f"{v.VarName} = {v.X}\n")

    def _get_solved_values(self):
        cm = self.ccdg_model
        lam = self.model.getVarByName("lambda").X
        topo = {(i, j): var.X for (i, j), var in cm.vars_topo_adj_mat.items()}
        turns = {
            (u, v): var.X
            for (u, v), var in cm.vars_turn_adj_mat.items()
            if not isinstance(var, int)
        }
        flow_uv = {
            s: {(u, v): var.X for (u, v), var in flows.items()}
            for s, flows in cm.vars_uv_flow.items()
        }
        flow_ss = {
            s: {u: var.X for u, var in flows.items()}
            for s, flows in cm.vars_ss_flow.items()
        }
        sink = {
            s: {u: var.X for u, var in flows.items()}
            for s, flows in cm.vars_sink.items()
        }
        return lam, topo, turns, flow_uv, flow_ss, sink

    def extract_resultant_values(self):
        (
            self.lam,
            self.topo_adj_mat_vals,
            self.turn_adj_mat_vals,
            self.flow_uv_vals,
            self.flow_ss_vals,
            self.sink_vals,
        ) = self._get_solved_values()
        return {
            "lambda": self.lam,
            "topo_adj_mat_vals": self.topo_adj_mat_vals,
            "turn_adj_mat_vals": self.turn_adj_mat_vals,
            "flow_uv_vals": self.flow_uv_vals,
            "flow_ss_vals": self.flow_ss_vals,
            "sink_vals": self.sink_vals,
        }

    def harden_results(self, constrain_paths=True):
        topology = self.harden_topology(self.topo_adj_mat_vals)
        self.topology = topology
        if self.relax_turn_edges:
            self.hardened_turn_edges = self.harden_turns(self.turn_adj_mat_vals)
        else:
            self.hardened_turn_edges = None
        paths_pkg = self.harden_paths(
            self.flow_uv_vals,
            self.flow_ss_vals,
            self.sink_vals,
            topology=topology if constrain_paths else None,
            hardened_turns=self.hardened_turn_edges if constrain_paths else None,
        )
        if self.VERBOSE:
            for i, row in enumerate(topology):
                print(f"topology {i}: {row}")
            for (s, d), paths_list in paths_pkg["paths_by_sd"].items():
                print(f"{s} -> {d}:")
                for path in paths_list:
                    print(
                        f"  phys={path['phys_nodes']} vcs={path['vcs']} "
                        f"W_k={path['W_k']} S_k={path['S_k']}"
                    )
        return (
            topology,
            paths_pkg["paths_by_sd"],
            paths_pkg["current_load"],
            paths_pkg["route_table"],
            paths_pkg["vc_table"],
        )

    @staticmethod
    def topology_to_edges(topology):
        n = len(topology)
        return {
            (i, j)
            for i in range(n)
            for j in range(n)
            if i != j and topology[i][j]
        }

    def harden_topology(self, topo_adj_mat_vals):
        n_nodes = self.n_nodes
        topo_adj_mat = [[0] * n_nodes for _ in range(n_nodes)]
        out_deg = [0] * n_nodes
        in_deg = [0] * n_nodes
        r = self.radix
        eps = self.epsilon

        def can_add_directed(i, j):
            return topo_adj_mat[i][j] == 0 and out_deg[i] < r and in_deg[j] < r

        def add_directed(i, j):
            topo_adj_mat[i][j] = 1
            out_deg[i] += 1
            in_deg[j] += 1

        def can_add_symmetric_pair(i, j):
            """Both (i,j) and (j,i) must fit out/in radix at i and j."""
            return can_add_directed(i, j) and can_add_directed(j, i)

        def add_symmetric_pair(i, j):
            add_directed(i, j)
            add_directed(j, i)

        if self.symmetric_links:
            scored = sorted(
                (
                    (
                        max(
                            topo_adj_mat_vals.get((i, j), 0.0),
                            topo_adj_mat_vals.get((j, i), 0.0),
                        ),
                        i,
                        j,
                    )
                    for i in range(n_nodes)
                    for j in range(i + 1, n_nodes)
                ),
                key=lambda t: t[0],
                reverse=True,
            )
            for val, i, j in scored:
                if val <= eps:
                    break
                if can_add_symmetric_pair(i, j):
                    add_symmetric_pair(i, j)
        else:
            scored = sorted(
                topo_adj_mat_vals.items(),
                key=lambda kv: kv[1],
                reverse=True,
            )
            for (i, j), val in scored:
                if val <= eps:
                    break
                if can_add_directed(i, j):
                    add_directed(i, j)

        return topo_adj_mat

    def harden_turns(self, turn_adj_mat_vals):
        return {
            (u, v)
            for (u, v), val in turn_adj_mat_vals.items()
            if val > self.epsilon
        }

    def _turn_respects_hardening(self, u, v, topology, hardened_turns):
        ij_map = self.cdg_u_to_topo_ijv_map
        i, k, _ = ij_map[u]
        k2, j, _ = ij_map[v]
        if k != k2:
            return False
        if not topology[i][k] or not topology[k][j]:
            return False
        if hardened_turns is not None and (u, v) not in hardened_turns:
            return False
        return True

    def _cdg_node_in_topology(self, u, topology):
        i, j, _ = self.cdg_u_to_topo_ijv_map[u]
        return bool(topology[i][j])

    def _decompose_paths_for_source(
        self,
        s,
        flow_uv_vals,
        flow_ss_vals,
        sink_vals,
        topology=None,
        hardened_turns=None,
    ):
        eps = self.epsilon
        ij_map = self.cdg_u_to_topo_ijv_map
        ss_map = self.ss_to_u_conn_map
        vpred = self.v_to_u_turns
        n_cdg = self.n_cdg
        constrain = topology is not None
        turn_ok = (
            (lambda u, v: self._turn_respects_hardening(u, v, topology, hardened_turns))
            if constrain
            else (lambda u, v: True)
        )

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
                at_source = cur in ss_map[s] and flow_ss.get(cur, 0.0) > eps
                if constrain:
                    at_source = at_source and self._cdg_node_in_topology(cur, topology)
                if at_source:
                    return list(reversed(back))
                preds = sorted(
                    v
                    for v in vpred[cur]
                    if flow_uv.get((v, cur), 0.0) > eps and turn_ok(v, cur)
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

    def _allocations_for_chosen(self, total, chosen_scored):
        n = len(chosen_scored)
        if n == 0 or total <= 0:
            return []
        chosen = [p for p, _ in chosen_scored]
        pol = self.k_path_traf_policy
        if pol == "equal":
            return [total / n] * n
        if pol == "weight_proportional":
            ws = [max(p["W_k"], 0.0) for p in chosen]
        else:
            ws = [max(sk, 0.0) for _, sk in chosen_scored]
        s = sum(ws)
        return [total * w / s for w in ws] if s > 0 else [total / n] * n

    @staticmethod
    def _plain_nested_dict(d):
        if isinstance(d, defaultdict):
            return {k: CCDGOptimizer._plain_nested_dict(v) for k, v in d.items()}
        return d

    def _score_and_select_paths(self, candidates, current_load):
        if current_load is None:
            current_load = {}
        scored = []
        for p in candidates:
            W_k = p["W_k"]
            hops = []
            phys = p["phys_nodes"]
            for h in range(len(phys) - 1):
                i, j = phys[h], phys[h + 1]
                cap_ij = self.capacity[i][j]
                hops.append(
                    float("inf") if cap_ij <= 0 else current_load.get((i, j), 0.0) / cap_ij
                )
            if not hops:
                L_k = 0.0
            elif self.weight_type == "max":
                L_k = max(hops)
            else:
                L_k = sum(hops) / len(hops)
            S_k = 0.0 if math.isinf(L_k) else float(W_k) * math.exp(-self.beta * L_k)
            scored.append((p, S_k))

        if not scored:
            return []
        k = min(self.k_paths, len(scored))
        if self.choice_type == "strict":
            return sorted(scored, key=lambda x: x[1], reverse=True)[:k]

        pool, chosen = list(scored), []
        for _ in range(k):
            if not pool:
                break
            weights = [max(self.epsilon, max(sk, 0.0)) for _, sk in pool]
            totw = sum(weights)
            if totw <= 0:
                pick = self._rng.choice(range(len(pool)))
            else:
                r = self._rng.uniform(0.0, totw)
                acc, pick = 0.0, 0
                for idx, w in enumerate(weights):
                    acc += w
                    if r <= acc:
                        pick = idx
                        break
                else:
                    pick = len(weights) - 1
            chosen.append(pool.pop(pick))
        return chosen

    def harden_paths(
        self,
        flow_uv_vals,
        flow_ss_vals,
        sink_vals,
        topology=None,
        hardened_turns=None,
    ):
        if self.lam is None:
            raise RuntimeError("harden_paths requires self.lam; call extract_resultant_values first")
        mcf = float(self.lam)
        eps = self.epsilon
        commodities = self.commodities

        candidates_by_sd = defaultdict(list)
        for s in commodities:
            for rec in self._decompose_paths_for_source(
                s,
                flow_uv_vals,
                flow_ss_vals,
                sink_vals,
                topology,
                hardened_turns=hardened_turns,
            ):
                if rec["d"] in commodities[s]:
                    candidates_by_sd[(s, rec["d"])].append(rec)

        sd_pairs = sorted(
            (
                -mcf * self.demand[s][d],
                len(candidates_by_sd[(s, d)]),
                s,
                d,
            )
            for s, dests in commodities.items()
            for d in dests
            if s != d
        )

        current_load = defaultdict(float)
        paths_by_sd = {}
        route_table = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        vc_table = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        for _neg_tot, _n_c, s, d in sd_pairs:
            total_need = mcf * self.demand[s][d]
            cands = list(candidates_by_sd[(s, d)])
            if not cands:
                paths_by_sd[(s, d)] = []
                continue

            chosen_scored = self._score_and_select_paths(cands, current_load)
            allocs = self._allocations_for_chosen(total_need, chosen_scored)
            hardened_list = []

            for path_idx, ((rec, S_k), alloc) in enumerate(zip(chosen_scored, allocs)):
                phys, vcs = list(rec["phys_nodes"]), list(rec["vcs"])
                for h in range(len(phys) - 1):
                    i, j = phys[h], phys[h + 1]
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
                    "phys_nodes": phys,
                    "vcs": vcs,
                    "cdg_nodes": list(rec["cdg_nodes"]),
                    "allocation": alloc,
                    "W_k": rec["W_k"],
                    "S_k": S_k,
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
