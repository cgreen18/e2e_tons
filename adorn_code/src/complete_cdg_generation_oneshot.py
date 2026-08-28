#!/usr/bin/env python3
"""
complete_cdg_generation.py

Topology synthesis using an MTZ-constrained Channel Dependency Graph (CDG)
formulation to ensure deadlock-free maximum concurrent flow.

- Variables m[i,j] define the undirected topology.
- Variables c[u,v] define the active routing turns in the CDG across VCs.
- MTZ constraints on CDG nodes ensure the routing is globally acyclic.
- Multi-commodity flow routes from super-sources to super-destinations.
"""

import argparse
import math
import sys
from collections import defaultdict, deque

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


def build_candidate_directed_arcs_from_undirected_vc(E_und, n_vcs):
    A_vc = []
    for (u, v) in E_und:
        for vc in range(n_vcs):
            A_vc.append((u, v, vc))
            A_vc.append((v, u, vc))
    return A_vc


def build_turns(A_vc):
    """
    Build valid CDG turns from arc (i,k,v1) to arc (k,j,v2).
    Excludes U-turns (i == j) to conform with cycle-free networking norms.
    Allows transitioning between any virtual channels.
    """
    turns = []
    for (i, k, v1) in A_vc:
        for (k2, j, v2) in A_vc:
            if k == k2 and i != j:
                turns.append(((i, k, v1), (k, j, v2)))
    return turns


def build_model(args):
    n = args.n_nodes
    radix = args.radix
    n_vcs = args.n_vcs
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
    A_vc = build_candidate_directed_arcs_from_undirected_vc(E, n_vcs)
    n_c = len(A_vc)
    Turns = build_turns(A_vc)

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
    level = model.addVars(A_vc, lb=1.0, ub=n_c, vtype=GRB.CONTINUOUS, name="level")
    
    # Throughput
    Z = model.addVar(lb=0.0, name="Z")

    # Flow Variables
    f_src = model.addVars([(s, arc) for s in S_nodes for arc in A_vc if arc[0] == s], lb=0.0, name="f_src")
    f_turn = model.addVars([(s, t) for s in S_nodes for t in Turns], lb=0.0, name="f_turn")
    f_dst = model.addVars([(s, arc) for s in S_nodes for arc in A_vc if arc[1] != s], lb=0.0, name="f_dst")

    # 3. Constraints

    # Topology Degree Bound
    for u in range(n):
        inc_edges = [edge_key(u, v) for v in range(n) if edge_key(u, v) in E]
        model.addConstr(gp.quicksum(m_vars[e] for e in inc_edges) <= radix, name=f"deg_{u}")

    # CDG to Topology Mapping
    for ((i, k, v1), (k2, j, v2)) in Turns:
        model.addConstr(c_vars[((i, k, v1), (k2, j, v2))] <= m_vars[edge_key(i, k)])
        model.addConstr(c_vars[((i, k, v1), (k2, j, v2))] <= m_vars[edge_key(k, j)])

    # MTZ Acyclicity on CDG
    for u, v in Turns:
        model.addConstr(level[u] - level[v] + n_c * c_vars[(u, v)] <= n_c - 1, name=f"mtz_{u}_{v}")

    # Routing Activation & Turn Capacity
    for t in Turns:
        model.addConstr(gp.quicksum(f_turn[s, t] for s in S_nodes) <= cap * c_vars[t], name=f"act_{t}")

    # Physical Link Capacity (Aggregated across all VCs)
    for e in E:
        u, v = e
        in_uv = gp.quicksum(f_src.get((s, (u, v, vc)), 0) + 
                            gp.quicksum(f_turn[s, t] for t in in_turns[(u, v, vc)]) 
                            for s in S_nodes for vc in range(n_vcs))
        
        in_vu = gp.quicksum(f_src.get((s, (v, u, vc)), 0) + 
                            gp.quicksum(f_turn[s, t] for t in in_turns[(v, u, vc)]) 
                            for s in S_nodes for vc in range(n_vcs))
        
        model.addConstr(in_uv + in_vu <= cap * m_vars[e], name=f"cap_{u}_{v}")

    # Flow Conservation at CDG Nodes
    for s in S_nodes:
        for cdg_node in A_vc:
            i, k, vc = cdg_node
            flow_in = f_src.get((s, cdg_node), 0) + gp.quicksum(f_turn[s, t] for t in in_turns[cdg_node])
            flow_out = f_dst.get((s, cdg_node), 0) + gp.quicksum(f_turn[s, t] for t in out_turns[cdg_node])
            model.addConstr(flow_in == flow_out, name=f"bal_{s}_{i}_{k}_{vc}")

    # Demand Production (Super-Source)
    total_out_demand = (n - 1) * demand_val
    for s in S_nodes:
        model.addConstr(gp.quicksum(f_src[s, (s, k, vc)] for k in range(n) for vc in range(n_vcs) if (s, k, vc) in A_vc) == Z * total_out_demand)

    # Demand Consumption (Super-Destination)
    for s in S_nodes:
        for d in S_nodes:
            if s != d:
                model.addConstr(gp.quicksum(f_dst[s, (i, d, vc)] for i in range(n) for vc in range(n_vcs) if (i, d, vc) in A_vc) == Z * demand_val)

    model.setObjective(Z, GRB.MAXIMIZE)
    return model, {"n": n, "E_und": E, "m_vars": m_vars, "c_vars": c_vars, "Z": Z}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_nodes", type=int, required=True, help="Number of nodes (n)")
    ap.add_argument("--radix", type=int, required=True, help="Regular degree")
    ap.add_argument("--n_vcs", type=int, required=True, help="Number of VCs")
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

    print(f"\n--- Gurobi Status: {model.status} ---")
    if model.SolCount == 0:
        raise RuntimeError(f"No solution available. Status={model.Status}")

    n = info["n"]
    m_vars = info["m_vars"]
    c_vars = info["c_vars"]
    n_vcs = args.n_vcs

    mat = [[0] * n for _ in range(n)]
    chosen = []
    adj = {i: [] for i in range(n)}
    
    for e, var in m_vars.items():
        if var.X > 0.001:  # Capture strictly non-zero even if continuous
            u, v = e
            mat[u][v] = 1
            mat[v][u] = 1
            chosen.append(e)
            adj[u].append(v)
            adj[v].append(u)

    with open(args.out, "w") as f:
        for i in range(n):
            f.write(" ".join(str(mat[i][j]) for j in range(n)) + "\n")

    if not args.silent:
        print(f"Wrote topology to {args.out}")
        print(f"Concurrent throughput Z = {info['Z'].X:.6g}")
        print(f"Selected undirected edges = {len(chosen)}\n")

    # --- TOPOLOGY STATISTICS ---
    def bfs_topo(start):
        dist = {start: 0}
        q = [start]
        for node in q:
            for neighbor in adj[node]:
                if neighbor not in dist:
                    dist[neighbor] = dist[node] + 1
                    q.append(neighbor)
        return dist

    topo_dists = {i: bfs_topo(i) for i in range(n)}
    topo_hops = [topo_dists[i][j] for i in range(n) for j in range(n) if i != j and j in topo_dists[i]]
    
    avg_radix = sum(len(adj[i]) for i in range(n)) / n
    avg_topo_hops = sum(topo_hops) / len(topo_hops) if topo_hops else 0
    topo_diameter = max(topo_hops) if topo_hops else 0

    print("--- Topology Statistics ---")
    print(f"Average Radix:  {avg_radix:.2f}")
    print(f"Average Hops:   {avg_topo_hops:.2f}")
    print(f"Diameter:       {topo_diameter}")
    if len(topo_hops) < n * (n - 1):
        print("WARNING: Topology is NOT fully connected.")
    print("---------------------------\n")

    # --- ROUTING STATISTICS ---
    # Extract valid turns (c_vars > 0)
    valid_turns = set()
    for t, var in c_vars.items():
        if var.X > 0.001: 
            valid_turns.add(t)

    valid_arcs = set()
    for u in range(n):
        for v in adj[u]:
            valid_arcs.add((u, v))

    routes = {}
    unroutable = 0
    
    for s in range(n):
        for d in range(n):
            if s == d: continue
            
            # 1-hop path check
            if (s, d) in valid_arcs:
                routes[(s, d)] = [s, d]
                continue
                
            queue = deque()
            for k in adj[s]:
                for vc in range(n_vcs):
                    queue.append((k, s, vc, [s, k]))
                    
            visited = set()
            found_path = None
            
            # BFS on the CDG
            while queue:
                curr, prev, vc_curr, path = queue.popleft()
                if curr == d:
                    found_path = path
                    break
                    
                state = (curr, prev, vc_curr)
                if state in visited: continue
                visited.add(state)
                
                for next_node in adj[curr]:
                    if next_node == prev: continue  # Exclude U-turns
                    for vc_next in range(n_vcs):
                        turn = ((prev, curr, vc_curr), (curr, next_node, vc_next))
                        if turn in valid_turns:
                            queue.append((next_node, curr, vc_next, path + [next_node]))
            
            if found_path:
                routes[(s, d)] = found_path
            else:
                routes[(s, d)] = []
                unroutable += 1

    route_hops = []
    edge_load = defaultdict(int)
    
    for (s, d), path in routes.items():
        if not path: continue
        route_hops.append(len(path) - 1)
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            edge_load[(u, v)] += 1

    avg_route_hops = sum(route_hops) / len(route_hops) if route_hops else 0
    route_diameter = max(route_hops) if route_hops else 0
    max_edge_load = max(edge_load.values()) if edge_load else 0

    print("--- Routing Statistics ---")
    print(f"Maximally Loaded Edge (# flows): {max_edge_load}")
    print(f"Average Route Hops:              {avg_route_hops:.2f}")
    print(f"Routing Diameter:                {route_diameter}")
    if unroutable > 0:
        print(f"WARNING: {unroutable} source-destination pairs were unroutable via valid CDG turns.")
    print("---------------------------\n")

if __name__ == "__main__":
    main()