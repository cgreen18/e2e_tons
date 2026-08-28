#!/usr/bin/env python3
"""
complete_cdg_generation.py

Topology synthesis using an MTZ-constrained Channel Dependency Graph (CDG)
formulation to ensure deadlock-free maximum concurrent flow.

- Variables m[i,j] define the undirected topology.
- Variables c[u,v] define the active routing turns in the CDG.
- MTZ constraints on CDG nodes ensure the routing is acyclic.
- Multi-commodity flow routes from super-sources to super-destinations.
"""

import argparse
import math
import sys
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB


def edge_key(u, v):
    """Return a consistent undirected edge key."""
    return (u, v) if u < v else (v, u)


def build_candidate_undirected_edges_ring(n, cand_k, allow_self=False):
    if cand_k is None:
        E = []
        for u in range(n):
            for v in range(u + 1, n):
                E.append((u, v))
        return E

    r = max(1, cand_k // 2)
    Eset = set()
    for u in range(n):
        for d in range(1, r + 1):
            v1 = (u + d) % n
            v2 = (u - d) % n
            if not allow_self and v1 == u: continue
            Eset.add(edge_key(u, v1))
            if not allow_self and v2 == u: continue
            Eset.add(edge_key(u, v2))
    return sorted(Eset)


def build_candidate_directed_arcs_from_undirected(E_und):
    A = []
    for (u, v) in E_und:
        A.append((u, v))
        A.append((v, u))
    return A


def build_turns(A):
    """
    Build valid CDG turns from arc (i,k) to arc (k,j).
    Excludes U-turns (i == j) to conform with cycle-free networking norms.
    """
    turns = []
    for (i, k) in A:
        for (k2, j) in A:
            if k == k2 and i != j:
                turns.append(((i, k), (k, j)))
    return turns


def build_model(args):
    n = args.n_nodes
    radix = args.radix
    cap = args.capacity
    demand_val = args.demand

    model = gp.Model("cdg_mcf_synthesis")
    if args.silent:
        model.Params.OutputFlag = 0
    if args.threads:
        model.Params.Threads = int(args.threads)
    if args.time_limit:
        model.Params.TimeLimit = float(args.time_limit)
    if args.mip_gap:
        model.Params.MIPGap = float(args.mip_gap)

    # Be conservative numerically for complex MCF
    model.Params.NumericFocus = 3
    model.Params.ScaleFlag = 2
    model.Params.FeasibilityTol = 1e-9
    model.Params.OptimalityTol = 1e-9

    # 1. Graph Structures
    E = build_candidate_undirected_edges_ring(n, args.cand_k)
    A = build_candidate_directed_arcs_from_undirected(E)
    n_c = len(A)
    Turns = build_turns(A)

    in_turns = defaultdict(list)
    out_turns = defaultdict(list)
    for t in Turns:
        in_turns[t[1]].append(t)
        out_turns[t[0]].append(t)

    S_nodes = list(range(n))
    
    # 2. Variables
    vtype = GRB.BINARY if args.binary_edges else GRB.CONTINUOUS
    
    # Topology Edges
    m_vars = model.addVars(E, lb=0.0, ub=1.0, vtype=vtype, name="m")
    
    # CDG Turns
    c_vars = model.addVars(Turns, lb=0.0, ub=1.0, vtype=vtype, name="c")
    
    # MTZ Levels
    level = model.addVars(A, lb=1.0, ub=n_c, vtype=GRB.CONTINUOUS, name="level")
    
    # Throughput
    Z = model.addVar(lb=0.0, name="Z")

    # Flow Variables
    f_src = model.addVars([(s, (s, k)) for s in S_nodes for (i, k) in A if i == s], lb=0.0, name="f_src")
    f_turn = model.addVars([(s, t) for s in S_nodes for t in Turns], lb=0.0, name="f_turn")
    f_dst = model.addVars([(s, (i, d)) for s in S_nodes for (i, d) in A if d != s], lb=0.0, name="f_dst")

    # 3. Constraints

    # Topology Degree Bound
    for u in range(n):
        inc_edges = [edge_key(u, v) for v in range(n) if edge_key(u, v) in E]
        model.addConstr(gp.quicksum(m_vars[e] for e in inc_edges) <= radix, name=f"deg_{u}")

    # CDG to Topology Mapping
    for ((i, k), (k, j)) in Turns:
        model.addConstr(c_vars[((i, k), (k, j))] <= m_vars[edge_key(i, k)])
        model.addConstr(c_vars[((i, k), (k, j))] <= m_vars[edge_key(k, j)])

    # MTZ Acyclicity on CDG
    for u, v in Turns:
        model.addConstr(level[u] - level[v] + n_c * c_vars[(u, v)] <= n_c - 1, name=f"mtz_{u}_{v}")

    # Routing Activation & Turn Capacity
    for t in Turns:
        model.addConstr(gp.quicksum(f_turn[s, t] for s in S_nodes) <= cap * c_vars[t], name=f"act_{t}")

    # Physical Link Capacity
    for e in E:
        u, v = e
        # Flow entering CDG node (u,v)
        in_uv = gp.quicksum(f_src.get((s, (u, v)), 0) + 
                            gp.quicksum(f_turn[s, t] for t in in_turns[(u, v)]) for s in S_nodes)
        # Flow entering CDG node (v,u)
        in_vu = gp.quicksum(f_src.get((s, (v, u)), 0) + 
                            gp.quicksum(f_turn[s, t] for t in in_turns[(v, u)]) for s in S_nodes)
        
        model.addConstr(in_uv + in_vu <= cap * m_vars[e], name=f"cap_{u}_{v}")

    # Flow Conservation at CDG Nodes
    for s in S_nodes:
        for cdg_node in A:
            i, k = cdg_node
            flow_in = f_src.get((s, cdg_node), 0) + gp.quicksum(f_turn[s, t] for t in in_turns[cdg_node])
            flow_out = f_dst.get((s, cdg_node), 0) + gp.quicksum(f_turn[s, t] for t in out_turns[cdg_node])
            model.addConstr(flow_in == flow_out, name=f"bal_{s}_{i}_{k}")

    # Demand Production (Super-Source)
    total_out_demand = (n - 1) * demand_val
    for s in S_nodes:
        model.addConstr(gp.quicksum(f_src[s, (s, k)] for k in range(n) if (s, k) in A) == Z * total_out_demand)

    # Demand Consumption (Super-Destination)
    for s in S_nodes:
        for d in S_nodes:
            if s != d:
                model.addConstr(gp.quicksum(f_dst[s, (i, d)] for i in range(n) if (i, d) in A) == Z * demand_val)

    model.setObjective(Z, GRB.MAXIMIZE)
    return model, {"n": n, "E_und": E, "m_vars": m_vars, "Z": Z}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_nodes", type=int, required=True, help="Number of nodes (n)")
    ap.add_argument("--radix", type=int, required=True, help="Regular degree")
    ap.add_argument("--out", type=str, required=True, help="Output adjacency matrix path")
    ap.add_argument("--binary_edges", action="store_true", help="Solve full MILP with binary vars")
    ap.add_argument("--cand_k", type=int, default=None, help="Candidate neighborhood size")
    ap.add_argument("--capacity", type=float, default=1.0, help="Per-link capacity C")
    ap.add_argument("--demand", type=float, default=1.0, help="Uniform demand D(s,t)")
    ap.add_argument("--time_limit", type=float, default=None, help="Time limit (seconds)")
    ap.add_argument("--mip_gap", type=float, default=None, help="MIPGap")
    ap.add_argument("--threads", type=int, default=None, help="Threads")
    ap.add_argument("--silent", action="store_true", help="Silence Gurobi")

    args = ap.parse_args()

    model, info = build_model(args)
    model.optimize()

    model.write("model.lp")
    # if (model.status == GRB.INFEASIBLE or model.status == GRB.UNBOUNDED): 
    print(f"Gurobi model is {model.status}.") 

    # model.computeIIS()
    # model.write("model.ilp")  # Write model with IIS information
    # model.write("iis.ilp")
    # print("IIS details written to iis.ilp")

    if model.SolCount == 0:
        raise RuntimeError(f"No solution available. Status={model.Status}")

    n = info["n"]
    m_vars = info["m_vars"]

    mat = [[0] * n for _ in range(n)]
    chosen = []
    for e, var in m_vars.items():
        if var.X >= 0.5:
            u, v = e
            mat[u][v] = 1
            mat[v][u] = 1
            chosen.append(e)

    with open(args.out, "w") as f:
        for i in range(n):
            f.write(" ".join(str(mat[i][j]) for j in range(n)) + "\n")

    if not args.silent:
        print(f"Wrote topology to {args.out}")
        print(f"Concurrent throughput Z = {info['Z'].X:.6g}")
        print(f"Selected undirected edges = {len(chosen)}")

if __name__ == "__main__":
    main()