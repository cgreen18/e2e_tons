#!/usr/bin/env python3
"""
aasc_shahrokhi_matula_loop.py

Topology synthesis via Shahrokhi–Matula node–arc concurrent flow formulation,
with an outer greedy loop:

  - Solve an LP relaxation (x_e in [0,1]) maximizing concurrent throughput Z
  - Fix the "best" edge (largest x_e) to 1
  - Repeat until all nodes meet degree/radix (or in/out degree for directed)

Outputs ONLY the adjacency matrix (space-delimited 0/1). No paths.

This version uses ONLY scalar Gurobi vars (addVar/addVars); NO MVar/addMVar.
It also correctly indexes flow vars as f[s,u,v] (not f[s,(u,v)]).

Notes:
- This is a heavy model: O(n^2) flow-balance constraints and O(n*|A|) flow vars.
  Use --cand_k to restrict candidate edges/arcs, and consider symmetry reduction
  if you have it (canonical sources) for large n.
"""

import argparse
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB


# ----------------------------
# Candidate set construction
# ----------------------------

def build_candidate_undirected_edges_ring(n, cand_k):
    """
    Candidate undirected edges (u,v) with u<v, using ring neighborhood.

    If cand_k is None: complete graph.
    Else: approx cand_k neighbors per node using offsets 1..r with r=cand_k//2.
    """
    if cand_k is None:
        return [(u, v) for u in range(n) for v in range(u + 1, n)]

    if cand_k < 1:
        raise ValueError("cand_k must be >= 1")

    r = max(1, cand_k // 2)
    Eset = set()
    for u in range(n):
        for d in range(1, r + 1):
            v1 = (u + d) % n
            v2 = (u - d) % n
            a, b = (u, v1) if u < v1 else (v1, u)
            if a != b:
                Eset.add((a, b))
            a, b = (u, v2) if u < v2 else (v2, u)
            if a != b:
                Eset.add((a, b))
    return sorted(Eset)


def build_candidate_directed_arcs_ring(n, cand_k):
    """
    Candidate directed arcs (u,v), u!=v, using ring neighborhood.

    If cand_k is None: complete digraph (excluding self).
    Else: approx cand_k outgoing per node using offsets 1..r with r=cand_k//2.
    """
    if cand_k is None:
        return [(u, v) for u in range(n) for v in range(n) if v != u]

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


def directed_arcs_from_undirected_edges(E_und):
    A = []
    for (u, v) in E_und:
        A.append((u, v))
        A.append((v, u))
    return A


# ----------------------------
# Model building (for one LP solve)
# ----------------------------

def build_shahrokhi_matula_lp(
    n,
    directed,
    radix,
    cap,
    demand,
    cand_edges_or_arcs,
    fixed1,
    fixed0,
    time_limit=None,
    threads=None,
    silent=False,
):
    """
    Build the LP relaxation with fixed edges/arcs.

    fixed1/fixed0:
      - undirected: sets of edges (u,v) with u<v
      - directed: sets of arcs (u,v)

    Returns (model, x, Z, aux_dict)
    """
    m = gp.Model("aasc_shahrokhi_matula_loop")
    if silent:
        m.Params.OutputFlag = 0
    if time_limit is not None:
        m.Params.TimeLimit = float(time_limit)
    if threads is not None:
        m.Params.Threads = int(threads)

    # m.Params.MIPFocus = 1

    # m.Params.Method = 2
    # m.Params.NodeMethod = 2

    m.Params.NumericFocus = 3        # be conservative numerically
    m.Params.ScaleFlag = 2           # aggressive scaling (often helps)
    m.Params.FeasibilityTol = 1e-9   # default is looser
    m.Params.OptimalityTol = 1e-9
    m.Params.BarConvTol = 1e-12      # barrier convergence (only matters if Method=2)
    # m.Params.BarHomogeneous = 1      # helps on ill-conditioned LPs



    if directed:
        A = cand_edges_or_arcs
        # Topology vars x[u,v] in triggered candidates
        x = m.addVars(A, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="x")

        # Apply fixings
        for a in fixed1:
            if a in x:
                x[a].LB = 1.0
                x[a].UB = 1.0
        for a in fixed0:
            if a in x:
                x[a].LB = 0.0
                x[a].UB = 0.0

        out_arcs = defaultdict(list)
        in_arcs = defaultdict(list)
        for (u, v) in A:
            out_arcs[u].append((u, v))
            in_arcs[v].append((u, v))

        # Degree constraints with fixed arcs accounted for:
        # sum_{out arcs} x = radix, sum_{in arcs} x = radix
        for u in range(n):
            m.addConstr(gp.quicksum(x[a] for a in out_arcs[u]) == radix, name=f"outdeg_{u}")
            m.addConstr(gp.quicksum(x[a] for a in in_arcs[u]) == radix, name=f"indeg_{u}")

        # Flow vars: f[s,u,v] for each s and arc (u,v)
        S = range(n)
        f = m.addVars(S, A, lb=0.0, name="f")  # keys are (s,u,v)

        out_by_node = defaultdict(list)
        in_by_node = defaultdict(list)
        for (u, v) in A:
            out_by_node[u].append((u, v))
            in_by_node[v].append((u, v))

        Z = m.addVar(lb=0.0, name="Z")

        total_out_demand = (n - 1) * demand
        for s in range(n):
            for l in range(n):
                rhs = Z * total_out_demand if l == s else -Z * demand
                m.addConstr(
                    gp.quicksum(f[s, u, v] for (u, v) in out_by_node[l]) -
                    gp.quicksum(f[s, u, v] for (u, v) in in_by_node[l]) == rhs,
                    name=f"flowbal_s{s}_l{l}",
                )

        # Capacity gating per arc: sum_s f[s,u,v] <= cap * x[u,v]
        for (u, v) in A:
            m.addConstr(gp.quicksum(f[s, u, v] for s in range(n)) <= cap * x[u, v], name=f"cap_{u}_{v}")

        m.setObjective(Z, GRB.MAXIMIZE)
        aux = {"A": A, "out_arcs": out_arcs, "in_arcs": in_arcs}
        return m, x, Z, aux

    else:
        E = cand_edges_or_arcs  # undirected edges u<v
        A = directed_arcs_from_undirected_edges(E)

        x = m.addVars(E, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="x")

        # Apply fixings
        for e in fixed1:
            if e in x:
                x[e].LB = 1.0
                x[e].UB = 1.0
        for e in fixed0:
            if e in x:
                x[e].LB = 0.0
                x[e].UB = 0.0

        inc_edges = defaultdict(list)
        for (u, v) in E:
            inc_edges[u].append((u, v))
            inc_edges[v].append((u, v))

        for u in range(n):
            m.addConstr(gp.quicksum(x[e] for e in inc_edges[u]) == radix, name=f"deg_{u}")

        # Flow vars on directed arcs induced by E
        S = range(n)
        f = m.addVars(S, A, lb=0.0, name="f")  # keys are (s,u,v)

        out_by_node = defaultdict(list)
        in_by_node = defaultdict(list)
        for (u, v) in A:
            out_by_node[u].append((u, v))
            in_by_node[v].append((u, v))

        Z = m.addVar(lb=0.0, name="Z")

        total_out_demand = (n - 1) * demand
        for s in range(n):
            for l in range(n):
                rhs = Z * total_out_demand if l == s else -Z * demand
                m.addConstr(
                    gp.quicksum(f[s, u, v] for (u, v) in out_by_node[l]) -
                    gp.quicksum(f[s, u, v] for (u, v) in in_by_node[l]) == rhs,
                    name=f"flowbal_s{s}_l{l}",
                )

        # Capacity gating per undirected edge: sum_s (f[s,u,v] + f[s,v,u]) <= cap * x[u,v]
        for (u, v) in E:
            m.addConstr(
                gp.quicksum(f[s, u, v] + f[s, v, u] for s in range(n)) <= cap * x[u, v],
                name=f"cap_{u}_{v}",
            )

        m.setObjective(Z, GRB.MAXIMIZE)
        aux = {"E": E, "A": A, "inc_edges": inc_edges}
        return m, x, Z, aux


# ----------------------------
# Greedy loop
# ----------------------------

def greedy_select_edges_undirected(n, E, radix, cap, demand, time_limit, threads, silent, max_iters=None):
    """
    Iteratively:
      - solve LP relaxation
      - pick best free edge (largest x) that doesn't exceed remaining degree
      - fix it to 1, repeat
    """
    inc_edges = defaultdict(list)
    for (u, v) in E:
        inc_edges[u].append((u, v))
        inc_edges[v].append((u, v))

    # quick feasibility check
    for u in range(n):
        if len(inc_edges[u]) < radix:
            raise RuntimeError(f"Node {u} has only {len(inc_edges[u])} candidate edges; need radix {radix}. Increase --cand_k.")

    fixed1 = set()
    fixed0 = set()
    deg1 = [0] * n
    remaining = [radix] * n

    it = 0
    while True:
        if all(r == 0 for r in remaining):
            break
        if max_iters is not None and it >= max_iters:
            raise RuntimeError(f"Reached max_iters={max_iters} without satisfying all degrees.")

        # Auto-prune: if a node has remaining=0, force all other incident free edges to 0
        for u in range(n):
            if remaining[u] == 0:
                for e in inc_edges[u]:
                    if e not in fixed1 and e not in fixed0:
                        fixed0.add(e)

        model, x, Z, _aux = build_shahrokhi_matula_lp(
            n=n, directed=False, radix=radix, cap=cap, demand=demand,
            cand_edges_or_arcs=E, fixed1=fixed1, fixed0=fixed0,
            time_limit=time_limit, threads=threads, silent=silent
        )
        model.optimize()
        if model.SolCount == 0:
            raise RuntimeError(f"LP became infeasible at iter {it}. This usually means greedy fixings boxed in the degree constraints. Try larger --cand_k.")

        # Extract x values on free edges
        best_e = None
        best_val = -1.0
        for (u, v), var in x.items():
            if (u, v) in fixed1 or (u, v) in fixed0:
                continue
            # only consider edges that can still be added
            if remaining[u] <= 0 or remaining[v] <= 0:
                continue
            val = float(var.X)
            if val > best_val + 1e-12:
                best_val = val
                best_e = (u, v)

        if best_e is None:
            # Fallback: pick any feasible free edge that doesn't violate remaining degrees
            for (u, v) in E:
                if (u, v) in fixed1 or (u, v) in fixed0:
                    continue
                if remaining[u] > 0 and remaining[v] > 0:
                    best_e = (u, v)
                    best_val = 0.0
                    break

        if best_e is None:
            raise RuntimeError("No feasible edge found to add. Increase --cand_k or revise candidate set constraints.")

        u, v = best_e
        fixed1.add(best_e)
        deg1[u] += 1
        deg1[v] += 1
        remaining[u] -= 1
        remaining[v] -= 1

        if not silent:
            print(f"[iter {it:4d}] Z={float(Z.X):.6g} picked edge {best_e} x={best_val:.6g} rem_deg(u)={remaining[u]} rem_deg(v)={remaining[v]}")

        it += 1

    # Final sanity
    for u in range(n):
        if deg1[u] != radix:
            raise RuntimeError(f"Internal error: node {u} has degree {deg1[u]} != radix {radix}")
    return fixed1


def greedy_select_arcs_directed(n, A, radix, cap, demand, time_limit, threads, silent, max_iters=None):
    """
    Directed version: enforce outdeg=radix, indeg=radix.
    Iteratively fix arcs to 1 based on LP x values.
    """
    out_arcs = defaultdict(list)
    in_arcs = defaultdict(list)
    for (u, v) in A:
        out_arcs[u].append((u, v))
        in_arcs[v].append((u, v))

    for u in range(n):
        if len(out_arcs[u]) < radix:
            raise RuntimeError(f"Node {u} has only {len(out_arcs[u])} candidate outgoing arcs; need radix {radix}. Increase --cand_k.")
        if len(in_arcs[u]) < radix:
            raise RuntimeError(f"Node {u} has only {len(in_arcs[u])} candidate incoming arcs; need radix {radix}. Increase --cand_k.")

    fixed1 = set()
    fixed0 = set()
    out_remaining = [radix] * n
    in_remaining = [radix] * n
    outdeg1 = [0] * n
    indeg1 = [0] * n

    it = 0
    while True:
        if all(r == 0 for r in out_remaining) and all(r == 0 for r in in_remaining):
            break
        if max_iters is not None and it >= max_iters:
            raise RuntimeError(f"Reached max_iters={max_iters} without satisfying all degrees.")

        # Auto-prune: if out_remaining[u]==0, force all other outgoing free arcs to 0
        for u in range(n):
            if out_remaining[u] == 0:
                for a in out_arcs[u]:
                    if a not in fixed1 and a not in fixed0:
                        fixed0.add(a)
            if in_remaining[u] == 0:
                for a in in_arcs[u]:
                    if a not in fixed1 and a not in fixed0:
                        fixed0.add(a)

        model, x, Z, _aux = build_shahrokhi_matula_lp(
            n=n, directed=True, radix=radix, cap=cap, demand=demand,
            cand_edges_or_arcs=A, fixed1=fixed1, fixed0=fixed0,
            time_limit=time_limit, threads=threads, silent=silent
        )
        model.optimize()
        if model.SolCount == 0:
            raise RuntimeError(f"LP became infeasible at iter {it}. Try larger --cand_k.")

        best_a = None
        best_val = -1.0
        for (u, v), var in x.items():
            if (u, v) in fixed1 or (u, v) in fixed0:
                continue
            if out_remaining[u] <= 0 or in_remaining[v] <= 0:
                continue
            val = float(var.X)
            if val > best_val + 1e-12:
                best_val = val
                best_a = (u, v)

        if best_a is None:
            for (u, v) in A:
                if (u, v) in fixed1 or (u, v) in fixed0:
                    continue
                if out_remaining[u] > 0 and in_remaining[v] > 0:
                    best_a = (u, v)
                    best_val = 0.0
                    break

        if best_a is None:
            raise RuntimeError("No feasible arc found to add. Increase --cand_k or revise candidate set.")

        u, v = best_a
        fixed1.add(best_a)
        outdeg1[u] += 1
        indeg1[v] += 1
        out_remaining[u] -= 1
        in_remaining[v] -= 1

        if True:
            print(f"[iter {it:4d}] Z={float(Z.X):.6g} picked arc {best_a} x={best_val:.6g} rem_out(u)={out_remaining[u]} rem_in(v)={in_remaining[v]}")

        it += 1

    for u in range(n):
        if outdeg1[u] != radix or indeg1[u] != radix:
            raise RuntimeError(f"Internal error: node {u} out={outdeg1[u]} in={indeg1[u]} expected {radix}")
    return fixed1


# ----------------------------
# Output
# ----------------------------

def write_adj_matrix(path, n, directed, chosen):
    mat = [[0] * n for _ in range(n)]
    if directed:
        for (u, v) in chosen:
            mat[u][v] = 1
    else:
        for (u, v) in chosen:
            mat[u][v] = 1
            mat[v][u] = 1

    with open(path, "w") as f:
        for i in range(n):
            f.write(" ".join(str(mat[i][j]) for j in range(n)) + "\n")


# ----------------------------
# CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Greedy LP topology synthesis using Shahrokhi–Matula node–arc MCFP primal.")
    ap.add_argument("--n_nodes", type=int, required=True)
    ap.add_argument("--radix", type=int, required=True)
    ap.add_argument("--out", type=str, required=True)

    ap.add_argument("--directed", action="store_true", help="Directed regular digraph (in=out=radix). Default undirected.")
    ap.add_argument("--cand_k", type=int, default=None, help="Candidate neighborhood size (ring-based). If omitted, uses complete graph.")
    ap.add_argument("--capacity", type=float, default=1.0, help="Per-link capacity.")
    ap.add_argument("--demand", type=float, default=1.0, help="Uniform demand D(s,t) for s!=t.")

    ap.add_argument("--time_limit", type=float, default=None, help="Per-iteration Gurobi time limit (seconds).")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--silent", action="store_true")
    ap.add_argument("--max_iters", type=int, default=None, help="Hard cap on greedy iterations (debug).")

    args = ap.parse_args()

    n = args.n_nodes
    radix = args.radix

    if n < 2:
        raise ValueError("n_nodes must be >= 2")
    if radix < 1:
        raise ValueError("radix must be >= 1")
    if radix >= n:
        raise ValueError("radix must be < n")

    if args.directed:
        A = build_candidate_directed_arcs_ring(n, args.cand_k)
        chosen = greedy_select_arcs_directed(
            n=n, A=A, radix=radix, cap=args.capacity, demand=args.demand,
            time_limit=args.time_limit, threads=args.threads, silent=args.silent,
            max_iters=args.max_iters
        )
        write_adj_matrix(args.out, n, True, chosen)
        if True:
            print(f"Wrote directed topology to {args.out} with {len(chosen)} arcs.")
    else:
        E = build_candidate_undirected_edges_ring(n, args.cand_k)
        chosen = greedy_select_edges_undirected(
            n=n, E=E, radix=radix, cap=args.capacity, demand=args.demand,
            time_limit=args.time_limit, threads=args.threads, silent=args.silent,
            max_iters=args.max_iters
        )
        write_adj_matrix(args.out, n, False, chosen)
        if not args.silent:
            print(f"Wrote undirected topology to {args.out} with {len(chosen)} edges (expected {n*radix//2}).")


if __name__ == "__main__":
    main()
