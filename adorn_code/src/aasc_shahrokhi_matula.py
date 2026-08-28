#!/usr/bin/env python3
"""
aasc_shahrokhi_matula.py

Topology synthesis using a Shahrokhi–Matula-style node–arc concurrent-flow primal:

- One "commodity" per source s (multi-sink): delivers Z*D(s,t) to every t != s.
- Edge/arc existence variables decide the topology (no path output).
- Output is ONLY the adjacency matrix (space-delimited 0/1).

IMPORTANT: This version uses ONLY scalar variables (addVar/addVars) — no MVar/addMVar.
It also fixes the tuple-indexing bug: f is indexed as f[s,u,v] (NOT f[s,(u,v)]).
"""

import argparse
import math
import sys
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB


# ----------------------------
# Candidate edge/arc generation
# ----------------------------

def build_candidate_undirected_edges_ring(n, cand_k, allow_self=False):
    """
    Build candidate undirected edges using a ring-distance neighborhood.
    cand_k: number of nearest neighbors *per node* (approx), must be >= radix.
    Returns list of undirected edges (u,v) with u < v.
    """
    if cand_k is None:
        # full complete graph
        E = []
        for u in range(n):
            for v in range( n):
                if u==v: continue
                E.append((u, v))
        return E

    if cand_k < 1:
        raise ValueError("cand_k must be >= 1")

    # Use symmetric neighborhood: offsets 1..r where r ~ cand_k/2
    r = max(1, cand_k // 2)
    Eset = set()
    for u in range(n):
        for d in range(1, r + 1):
            v1 = (u + d) % n
            v2 = (u - d) % n
            if not allow_self and v1 == u:
                continue
            a, b = (u, v1) if u < v1 else (v1, u)
            if a != b:
                Eset.add((a, b))
            if not allow_self and v2 == u:
                continue
            a, b = (u, v2) if u < v2 else (v2, u)
            if a != b:
                Eset.add((a, b))
    return sorted(Eset)


def build_candidate_directed_arcs_from_undirected(n, E_und):
    """
    From undirected edges, create directed arcs (u,v) and (v,u).
    """
    A = []
    for (u, v) in E_und:
        A.append((u, v))
        A.append((v, u))
    return A


def build_candidate_directed_arcs_ring(n, cand_k):
    """
    Build candidate directed arcs using ring-distance neighborhood.
    Returns list of directed arcs (u,v), u!=v.
    """
    if cand_k is None:
        A = []
        for u in range(n):
            for v in range(n):
                if u == v:
                    continue
                A.append((u, v))
        return A

    if cand_k < 1:
        raise ValueError("cand_k must be >= 1")

    r = max(1, cand_k // 2)
    Aset = set()
    for u in range(n):
        for d in range(1, r + 1):
            v1 = (u + d) % n
            v2 = (u - d) % n
            if v1 != u:
                Aset.add((u, v1))
            if v2 != u:
                Aset.add((u, v2))
    return sorted(Aset)


# ----------------------------
# Model building (Shahrokhi–Matula primal with topology vars)
# ----------------------------

def build_model(args):
    n = args.n_nodes
    radix = args.radix
    cap = args.capacity
    demand_val = args.demand

    if n < 2:
        raise ValueError("n_nodes must be >= 2")
    if radix < 1:
        raise ValueError("radix must be >= 1")
    if radix >= n:
        raise ValueError("radix must be < n (simple graph / no self loops)")
    if cap <= 0:
        raise ValueError("capacity must be > 0")
    if demand_val <= 0:
        raise ValueError("demand must be > 0")

    # Demand matrix is uniform for all s != t:
    # D(s,t) = demand_val, D(s,s) = 0
    # For source s, total out-demand = (n-1)*demand_val

    model = gp.Model("aasc_shahrokhi_matula_topology")

    if args.silent:
        model.Params.OutputFlag = 0

    if args.threads is not None:
        model.Params.Threads = int(args.threads)
    if args.time_limit is not None:
        model.Params.TimeLimit = float(args.time_limit)
    if args.mip_gap is not None:
        model.Params.MIPGap = float(args.mip_gap)

    # model.Params.MIPFocus = 1

    # model.Params.Method = 2
    # model.Params.NodeMethod = 2

    model.Params.NumericFocus = 3        # be conservative numerically
    model.Params.ScaleFlag = 2           # aggressive scaling (often helps)
    model.Params.FeasibilityTol = 1e-9   # default is looser
    model.Params.OptimalityTol = 1e-9
    model.Params.BarConvTol = 1e-12      # barrier convergence (only matters if Method=2)
    # model.Params.BarHomogeneous = 1      # helps on ill-conditioned LPs



    # Build candidate set
    if args.directed:
        A = build_candidate_directed_arcs_ring(n, args.cand_k)
        if len(A) == 0:
            raise RuntimeError("No candidate arcs generated.")
        # Topology vars x[u,v] for arcs
        vtype = GRB.BINARY if args.binary_edges else GRB.CONTINUOUS
        x = model.addVars(A, lb=0.0, ub=1.0, vtype=vtype, name="x")

        # Directed regularity: outdegree == radix and indegree == radix
        out_arcs = defaultdict(list)
        in_arcs = defaultdict(list)
        for (u, v) in A:
            out_arcs[u].append((u, v))
            in_arcs[v].append((u, v))

        for u in range(n):
            if len(out_arcs[u]) < radix:
                raise RuntimeError(f"Not enough candidate outgoing arcs for node {u}: have {len(out_arcs[u])}, need {radix}")
            if len(in_arcs[u]) < radix:
                raise RuntimeError(f"Not enough candidate incoming arcs for node {u}: have {len(in_arcs[u])}, need {radix}")

        model.addConstrs((gp.quicksum(x[a] for a in out_arcs[u]) == radix for u in range(n)), name="outdeg")
        model.addConstrs((gp.quicksum(x[a] for a in in_arcs[u]) == radix for u in range(n)), name="indeg")

        # Flow vars: f[s,u,v] for each source s and arc (u,v)
        S = list(range(n))
        f = model.addVars(S, A, lb=0.0, name="f")  # keys are (s,u,v) because A is tuple indices

        # Build arc adjacency lists for flow conservation
        out_by_node = defaultdict(list)
        in_by_node = defaultdict(list)
        for (u, v) in A:
            out_by_node[u].append((u, v))
            in_by_node[v].append((u, v))

        Z = model.addVar(lb=0.0, name="Z")

        # Flow conservation for each source s and node l
        total_out_demand = (n - 1) * demand_val
        for s in range(n):
            for l in range(n):
                if l == s:
                    rhs = Z * total_out_demand
                else:
                    rhs = -Z * demand_val
                model.addConstr(
                    gp.quicksum(f[s, u, v] for (u, v) in out_by_node[l]) -
                    gp.quicksum(f[s, u, v] for (u, v) in in_by_node[l]) == rhs,
                    name=f"flowbal_s{s}_l{l}"
                )

        # Capacity gating per arc: sum_s f[s,u,v] <= cap * x[u,v]
        for (u, v) in A:
            model.addConstr(
                gp.quicksum(f[s, u, v] for s in range(n)) <= cap * x[u, v],
                name=f"cap_{u}_{v}"
            )

    else:
        # Undirected
        E = build_candidate_undirected_edges_ring(n, args.cand_k)
        if len(E) == 0:
            raise RuntimeError("No candidate edges generated.")
        A = build_candidate_directed_arcs_from_undirected(n, E)

        vtype = GRB.BINARY if args.binary_edges else GRB.CONTINUOUS
        x = model.addVars(E, lb=0.0, ub=1.0, vtype=vtype, name="x")  # keys x[u,v] for u<v

        # Degree regularity: deg == radix
        inc_edges = defaultdict(list)
        for (u, v) in E:
            inc_edges[u].append((u, v))
            inc_edges[v].append((u, v))

        for u in range(n):
            if len(inc_edges[u]) < radix:
                raise RuntimeError(f"Not enough candidate incident edges for node {u}: have {len(inc_edges[u])}, need {radix}")

        model.addConstrs((gp.quicksum(x[e] for e in inc_edges[u]) == radix for u in range(n)), name="deg")

        # Flow vars on directed arcs (u,v) induced by candidate undirected edges
        S = list(range(n))
        f = model.addVars(S, A, lb=0.0, name="f")  # keys are (s,u,v)

        out_by_node = defaultdict(list)
        in_by_node = defaultdict(list)
        for (u, v) in A:
            out_by_node[u].append((u, v))
            in_by_node[v].append((u, v))

        Z = model.addVar(lb=0.0, name="Z")

        total_out_demand = (n - 1) * demand_val
        for s in range(n):
            for l in range(n):
                if l == s:
                    rhs = Z * total_out_demand
                else:
                    rhs = -Z * demand_val
                model.addConstr(
                    gp.quicksum(f[s, u, v] for (u, v) in out_by_node[l]) -
                    gp.quicksum(f[s, u, v] for (u, v) in in_by_node[l]) == rhs,
                    name=f"flowbal_s{s}_l{l}"
                )

        # Capacity gating per undirected edge: sum_s (f[s,u,v] + f[s,v,u]) <= cap * x[u,v]
        for (u, v) in E:
            model.addConstr(
                gp.quicksum(f[s, u, v] + f[s, v, u] for s in range(n)) <= cap * x[u, v],
                name=f"cap_{u}_{v}"
            )

    model.setObjective(Z, GRB.MAXIMIZE)

    info = {
        "n": n,
        "directed": args.directed,
        "E_und": None if args.directed else E,
        "A_dir": A,
        "x": x,
        "Z": Z,
    }
    return model, info


# ----------------------------
# Rounding (LP -> discrete topology)
# ----------------------------

def greedy_round_undirected(n, E, xvals, radix):
    """
    Greedy rounding for undirected regular graph:
    - Sort edges by xval desc
    - Add edge if both endpoints still need degree
    - Returns chosen edge set (u,v) with u<v
    """
    need = [radix] * n
    chosen = set()

    edges_sorted = sorted(E, key=lambda e: xvals.get(e, 0.0), reverse=True)
    for (u, v) in edges_sorted:
        if need[u] > 0 and need[v] > 0:
            chosen.add((u, v))
            need[u] -= 1
            need[v] -= 1
        if all(d == 0 for d in need):
            break

    if any(d != 0 for d in need):
        raise RuntimeError("Greedy rounding failed to satisfy all degrees. Increase cand_k or use --binary_edges.")

    return chosen


def greedy_round_directed(n, A, xvals, radix):
    """
    Greedy rounding for directed regular digraph:
    - Sort arcs by xval desc
    - Add arc if outdeg[u] < radix and indeg[v] < radix
    - Returns chosen arcs (u,v)
    """
    out_need = [radix] * n
    in_need = [radix] * n
    chosen = set()

    arcs_sorted = sorted(A, key=lambda a: xvals.get(a, 0.0), reverse=True)
    for (u, v) in arcs_sorted:
        if out_need[u] > 0 and in_need[v] > 0:
            chosen.add((u, v))
            out_need[u] -= 1
            in_need[v] -= 1
        if all(d == 0 for d in out_need) and all(d == 0 for d in in_need):
            break

    if any(d != 0 for d in out_need) or any(d != 0 for d in in_need):
        raise RuntimeError("Greedy rounding failed to satisfy all in/out degrees. Increase cand_k or use --binary_edges.")

    return chosen


# ----------------------------
# Output
# ----------------------------

def write_adj_matrix(path, n, directed, chosen_edges_or_arcs):
    """
    Writes space-delimited adjacency matrix with 0/1 entries.
    """
    mat = [[0] * n for _ in range(n)]
    if directed:
        for (u, v) in chosen_edges_or_arcs:
            mat[u][v] = 1
    else:
        for (u, v) in chosen_edges_or_arcs:
            mat[u][v] = 1
            mat[v][u] = 1

    with open(path, "w") as f:
        for i in range(n):
            f.write(" ".join(str(mat[i][j]) for j in range(n)) + "\n")


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Topology synthesis via Shahrokhi–Matula node–arc concurrent flow + topology vars.")
    ap.add_argument("--n_nodes", type=int, required=True, help="Number of routers/nodes (n)")
    ap.add_argument("--radix", type=int, required=True, help="Regular degree (undirected) or in/out degree (directed)")
    ap.add_argument("--out", type=str, required=True, help="Output adjacency matrix path")

    ap.add_argument("--directed", action="store_true", help="Generate directed regular digraph (in=out=radix)")
    ap.add_argument("--binary_edges", action="store_true", help="Solve full MILP with binary topology vars. If not set, solve LP and greedy-round.")
    ap.add_argument("--cand_k", type=int, default=None, help="Candidate neighborhood size (ring-based). If omitted, uses complete graph.")

    ap.add_argument("--capacity", type=float, default=1.0, help="Per-link capacity C")
    ap.add_argument("--demand", type=float, default=1.0, help="Uniform demand D(s,t) for s!=t")

    ap.add_argument("--time_limit", type=float, default=None, help="Gurobi time limit (seconds)")
    ap.add_argument("--mip_gap", type=float, default=None, help="Gurobi MIPGap (MILP only)")
    ap.add_argument("--threads", type=int, default=None, help="Gurobi Threads")
    ap.add_argument("--silent", action="store_true", help="Silence Gurobi output")

    args = ap.parse_args()

    model, info = build_model(args)
    model.optimize()

    if model.SolCount == 0:
        raise RuntimeError(f"No solution available. Status={model.Status}")

    n = info["n"]
    directed = info["directed"]
    x = info["x"]

    # Extract x values robustly (scalar vars)
    xvals = {}
    for key, var in x.items():
        xvals[key] = float(var.X)

    if args.binary_edges:
        # Read discrete x from solution
        if directed:
            chosen = {(u, v) for (u, v) in x.keys() if xvals[(u, v)] >= 0.5}
        else:
            chosen = {(u, v) for (u, v) in x.keys() if xvals[(u, v)] >= 0.5}
    else:
        # LP relax -> greedy rounding
        if directed:
            A = info["A_dir"]
            chosen = greedy_round_directed(n, A, xvals, args.radix)
        else:
            E = info["E_und"]
            chosen = greedy_round_undirected(n, E, xvals, args.radix)

    write_adj_matrix(args.out, n, directed, chosen)

    if not args.silent:
        Z = float(info["Z"].X)
        print(f"Wrote topology to {args.out}")
        print(f"Concurrent throughput Z = {Z:.6g}")
        print(f"Selected links = {len(chosen)} ({'arcs' if directed else 'undirected edges'})")


if __name__ == "__main__":
    main()
