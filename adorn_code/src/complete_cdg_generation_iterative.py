#!/usr/bin/env python3
"""
complete_cdg_generation.py

Iterative topology synthesis using an MTZ-constrained Channel Dependency Graph (CDG).
Routes flow-by-flow, hardening valid CDG turns and edges into the topology 
for subsequent iterations to ensure global deadlock-freedom. 
Includes dynamic residual capacity tracking based on the MCF bottleneck.
"""

import argparse
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
    Excludes U-turns (i == j).
    """
    turns = []
    for (i, k, v1) in A_vc:
        for (k2, j, v2) in A_vc:
            if k == k2 and i != j:
                turns.append(((i, k, v1), (k, j, v2)))
    return turns


def build_model_for_pair(args, s, d, E, A_vc, Turns, in_turns, out_turns, n_c, 
                         hardened_turns, hardened_edges, known_load, min_mcf_seen):
    """Builds a single-commodity model for routing from s to d."""
    model = gp.Model(f"cdg_route_{s}_{d}")
    if args.silent:
        model.Params.OutputFlag = 0
    if args.threads:
        model.Params.Threads = int(args.threads)
    if args.time_limit:
        model.Params.TimeLimit = float(args.time_limit)
    if args.mip_gap:
        model.Params.MIPGap = float(args.mip_gap)

    # Variables
    vtype = GRB.BINARY if args.binary_edges else GRB.CONTINUOUS
    m_vars = model.addVars(E, lb=0.0, ub=1.0, vtype=vtype, name="m")
    c_vars = model.addVars(Turns, lb=0.0, ub=1.0, vtype=vtype, name="c")
    level = model.addVars(A_vc, lb=1.0, ub=n_c, vtype=GRB.CONTINUOUS, name="level")
    Z = model.addVar(lb=0.0, ub=1.0, name="Z")  # Cap Z at 1.0 since we just need 1 path

    f_src = model.addVars([arc for arc in A_vc if arc[0] == s], lb=0.0, name="f_src")
    f_turn = model.addVars(Turns, lb=0.0, name="f_turn")
    f_dst = model.addVars([arc for arc in A_vc if arc[1] == d], lb=0.0, name="f_dst")

    # 1. Topology Degree Bound
    for u in range(args.n_nodes):
        inc_edges = [edge_key(u, v) for v in range(args.n_nodes) if edge_key(u, v) in E]
        model.addConstr(gp.quicksum(m_vars[e] for e in inc_edges) <= args.radix, name=f"deg_{u}")

    # 2. CDG to Topology Mapping
    for ((i, k, v1), (k2, j, v2)) in Turns:
        model.addConstr(c_vars[((i, k, v1), (k2, j, v2))] <= m_vars[edge_key(i, k)])
        model.addConstr(c_vars[((i, k, v1), (k2, j, v2))] <= m_vars[edge_key(k, j)])

    # 3. MTZ Acyclicity on CDG
    for u, v in Turns:
        model.addConstr(level[u] - level[v] + n_c * c_vars[(u, v)] <= n_c - 1, name=f"mtz_{u}_{v}")

    # 4. Turn Activation
    for t in Turns:
        model.addConstr(f_turn[t] <= args.capacity * c_vars[t], name=f"act_{t}")

    # 5. Flow Conservation at CDG Nodes
    for cdg_node in A_vc:
        i, k, vc = cdg_node
        
        flow_in = f_src.get(cdg_node, 0) + gp.quicksum(f_turn[t] for t in in_turns[cdg_node])
        flow_out = f_dst.get(cdg_node, 0) + gp.quicksum(f_turn[t] for t in out_turns[cdg_node])
        
        model.addConstr(flow_in == flow_out, name=f"bal_{i}_{k}_{vc}")

    # 6. Demand Production & Consumption (Single Flow)
    model.addConstr(gp.quicksum(f_src[arc] for arc in A_vc if arc[0] == s) == Z * args.demand)
    model.addConstr(gp.quicksum(f_dst[arc] for arc in A_vc if arc[1] == d) == Z * args.demand)

    # 7. ENFORCE PREVIOUSLY HARDENED STATE
    for t in hardened_turns:
        model.addConstr(c_vars[t] == 1.0, name=f"hardened_turn_{t}")
    for e in hardened_edges:
        model.addConstr(m_vars[e] == 1.0, name=f"hardened_edge_{e}")

    # 8. Physical Link Capacity (Dynamic Residual)
    for e in E:
        u, v = e
        in_uv = gp.quicksum(f_src.get((u, v, vc), 0) + 
                            gp.quicksum(f_turn[t] for t in in_turns[(u, v, vc)]) 
                            for vc in range(args.n_vcs))
        
        in_vu = gp.quicksum(f_src.get((v, u, vc), 0) + 
                            gp.quicksum(f_turn[t] for t in in_turns[(v, u, vc)]) 
                            for vc in range(args.n_vcs))
        
        avail_cap = args.capacity
        if min_mcf_seen != float('inf'):
            avail_cap = max(0.0, args.capacity - (known_load[e] * min_mcf_seen))
            
        model.addConstr(in_uv + in_vu <= avail_cap * m_vars[e], name=f"cap_{u}_{v}")

    model.setObjective(Z, GRB.MAXIMIZE)
    return model, {"m_vars": m_vars, "c_vars": c_vars, "Z": Z}


def get_route(s, d, valid_turns, valid_arcs, n_vcs, adj):
    """Extracts a single valid route from the active support graph."""
    # Check for direct 1-hop connection
    if (s, d) in valid_arcs:
        return [s, d], []
        
    queue = deque()
    for k in adj[s]:
        for vc in range(n_vcs):
            queue.append((k, s, vc, [s, k], []))
            
    visited = set()
    
    while queue:
        curr, prev, vc_curr, path_nodes, path_turns = queue.popleft()
        if curr == d:
            return path_nodes, path_turns
            
        state = (curr, prev, vc_curr)
        if state in visited: continue
        visited.add(state)
        
        for next_node in adj[curr]:
            if next_node == prev: continue
            for vc_next in range(n_vcs):
                turn = ((prev, curr, vc_curr), (curr, next_node, vc_next))
                if turn in valid_turns:
                    queue.append((next_node, curr, vc_next, 
                                  path_nodes + [next_node], 
                                  path_turns + [turn]))
    return None, None


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

    n = args.n_nodes
    E = build_candidate_undirected_edges_ring(n, args.cand_k)
    A_vc = build_candidate_directed_arcs_from_undirected_vc(E, args.n_vcs)
    n_c = len(A_vc)
    Turns = build_turns(A_vc)

    in_turns = defaultdict(list)
    out_turns = defaultdict(list)
    for t in Turns:
        in_turns[t[1]].append(t)
        out_turns[t[0]].append(t)

    hardened_turns = set()
    hardened_edges = set()
    routes = {}

    known_load = defaultdict(int)
    min_mcf_seen = float('inf')
    unroutable = 0

    print(f"--- Starting Iterative Routing ({n*(n-1)} flows) ---")

    for s in range(n):
        for d in range(n):
            if s == d: continue
            
            model, info = build_model_for_pair(args, s, d, E, A_vc, Turns, in_turns, out_turns, n_c, 
                                               hardened_turns, hardened_edges, known_load, min_mcf_seen)
            model.optimize()

            if model.SolCount == 0 or info["Z"].X < 0.001:
                print(f"[{s:02d} -> {d:02d}] Failed (Infeasible, Z=0, or Insufficient Capacity)")
                unroutable += 1
                continue

            # Extract active subgraph from this iteration's solution
            valid_turns = {t for t, var in info["c_vars"].items() if var.X > 0.001}
            valid_arcs = set()
            adj = {i: [] for i in range(n)}
            
            for e, var in info["m_vars"].items():
                if var.X > 0.001:
                    u, v = e
                    adj[u].append(v)
                    adj[v].append(u)
                    valid_arcs.add((u, v))
                    valid_arcs.add((v, u))

            # Extract integer route
            path_nodes, path_turns = get_route(s, d, valid_turns, valid_arcs, args.n_vcs, adj)
            
            if path_nodes:
                routes[(s, d)] = path_nodes
                
                # Update global bottleneck
                z_val = info["Z"].X
                min_mcf_seen = min(min_mcf_seen, z_val)
                
                # Harden edges & increment load
                for i in range(len(path_nodes) - 1):
                    e = edge_key(path_nodes[i], path_nodes[i+1])
                    hardened_edges.add(e)
                    known_load[e] += 1
                
                # Harden turns
                for t in path_turns:
                    hardened_turns.add(t)
                
                print(f"[{s:02d} -> {d:02d}] Routed (Z={z_val:.4f}, min_MCF={min_mcf_seen:.4f}): {' -> '.join(map(str, path_nodes))}")
            else:
                print(f"[{s:02d} -> {d:02d}] Failed to extract integer path from support")
                unroutable += 1

    # --- TOPOLOGY RECONSTRUCTION & OUTPUT ---
    mat = [[0] * n for _ in range(n)]
    final_adj = {i: [] for i in range(n)}
    
    for e in hardened_edges:
        u, v = e
        mat[u][v] = 1
        mat[v][u] = 1
        final_adj[u].append(v)
        final_adj[v].append(u)

    with open(args.out, "w") as f:
        for i in range(n):
            f.write(" ".join(str(mat[i][j]) for j in range(n)) + "\n")

    print(f"\n--- Topology Exported to {args.out} ---")

    # --- TOPOLOGY STATISTICS ---
    def bfs_topo(start):
        dist = {start: 0}
        q = [start]
        for node in q:
            for neighbor in final_adj[node]:
                if neighbor not in dist:
                    dist[neighbor] = dist[node] + 1
                    q.append(neighbor)
        return dist

    topo_dists = {i: bfs_topo(i) for i in range(n)}
    topo_hops = [topo_dists[i][j] for i in range(n) for j in range(n) if i != j and j in topo_dists[i]]
    
    avg_radix = sum(len(final_adj[i]) for i in range(n)) / n
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
    route_hops = []
    
    for (s, d), path in routes.items():
        if not path: continue
        route_hops.append(len(path) - 1)

    avg_route_hops = sum(route_hops) / len(route_hops) if route_hops else 0
    route_diameter = max(route_hops) if route_hops else 0
    max_edge_load = max(known_load.values()) if known_load else 0

    print("--- Routing Statistics ---")
    print(f"Maximally Loaded Edge (# flows): {max_edge_load}")
    print(f"Average Route Hops:              {avg_route_hops:.2f}")
    print(f"Routing Diameter:                {route_diameter}")
    print(f"Final MCF Bottleneck Estimate:   {min_mcf_seen:.6f}")
    if unroutable > 0:
        print(f"WARNING: {unroutable} source-destination pairs were unroutable.")
    print("---------------------------\n")


if __name__ == "__main__":
    main()