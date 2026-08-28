#!/usr/bin/env python3
import argparse
import math
import os
import sys
import heapq
from collections import defaultdict

import networkit as nk
import gurobipy as gp
from gurobipy import GRB


# ----------------------------
# IO: adjacency matrix + demands
# ----------------------------

def read_adjacency_matrix(path, directed=False, weighted=None):
    """
    Reads a space-delimited adjacency matrix from text.
    - If weighted is None: infer weighted if any entry not in {0,1}
    - If weighted is False: treat any nonzero as edge with weight=1
    - If weighted is True: use matrix entry as weight (must be >=0)
    Returns (G, weight_dict) where weight_dict[(u,v)] is weight for directed edges.
    """
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            rows.append([float(x) for x in parts])

    if not rows:
        raise ValueError(f"Empty adjacency matrix: {path}")

    n = len(rows)
    for r in rows:
        if len(r) != n:
            raise ValueError("Adjacency matrix must be square")

    if weighted is None:
        weighted = any((abs(x - 0.0) > 1e-12 and abs(x - 1.0) > 1e-12) for row in rows for x in row)

    G = nk.graph.Graph(n, weighted=weighted, directed=directed)
    w = {}

    for i in range(n):
        # For undirected, we only need to check the upper triangle (i, j>=i)
        # but we must check BOTH rows[i][j] AND rows[j][i].
        start_j = i + 1 if not directed else 0
        for j in range(start_j, n):
            if i == j: continue
            
            if not directed:
                val = max(rows[i][j], rows[j][i])
            else:
                val = rows[i][j]

            if abs(val) <= 1e-12: continue
            # ... proceed to add edge ...
            if not weighted:
                wt = 1.0
            else:
                if val < 0:
                    raise ValueError("Negative weights not supported")
                wt = float(val)

            # Add edge once for undirected by only i<j
            if not directed and j < i:
                continue

            if weighted:
                G.addEdge(i, j, wt)
            else:
                G.addEdge(i, j)

            # store directed weights for both directions if undirected
            w[(i, j)] = wt
            if not directed:
                w[(j, i)] = wt

    return G, w, weighted


def read_demands(path, n):
    """
    Optional demands file format (space-delimited):
        s t demand
    Lines starting with # are ignored.
    Returns dict demands[(s,t)] = demand for s!=t.
    """
    d = {}
    if path is None:
        return d
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            s, t, dem = line.split()
            s = int(s); t = int(t); dem = float(dem)
            if s < 0 or s >= n or t < 0 or t >= n or s == t:
                continue
            if dem < 0:
                raise ValueError("Demand must be nonnegative")
            if dem == 0:
                continue
            d[(s, t)] = dem
    return d


def build_default_commodities(n, demands=None, sources=None, sinks=None):
    """
    If demands is provided and nonempty: use those commodities.
    Else: default unit demand for all ordered pairs s!=t, optionally restricted by sources/sinks sets.
    sources: optional set/list of allowed sources
    sinks: optional set/list of allowed sinks
    Returns list of (s,t,demand).
    """
    comms = []
    if demands:
        for (s, t), dem in demands.items():
            comms.append((s, t, float(dem)))
        return comms

    src_set = set(range(n)) if sources is None else set(sources)
    dst_set = set(range(n)) if sinks is None else set(sinks)

    for s in range(n):
        if s not in src_set:
            continue
        for t in range(n):
            if t == s or t not in dst_set:
                continue
            comms.append((s, t, 1.0))
    return comms


# ----------------------------
# NetworKit shortest path helpers
# ----------------------------

def nk_shortest_path_nodes(G, wts, s, t, weighted):
    """
    Returns a list of nodes along a shortest path s->t using NetworKit.
    If no path exists, returns None.
    """
    if s == t:
        return [s]

    if weighted:
        dij = nk.distance.Dijkstra(G, s, storePaths=True)
        dij.run()
        dist = dij.distance(t)
        if math.isinf(dist):
            return None
        path = dij.getPath(t)
        if not path:
            return None
        return path
    else:
        bfs = nk.distance.BFS(G, s, storePaths=True)
        bfs.run()
        dist = bfs.distance(t)
        if dist < 0:
            return None
        path = bfs.getPath(t)
        if not path:
            return None
        return path


def path_length(wts, nodes):
    if nodes is None or len(nodes) < 2:
        return 0.0
    total = 0.0
    for i in range(len(nodes) - 1):
        total += wts[(nodes[i], nodes[i+1])]
    return total


def clone_graph(G, weighted, directed):
    """
    NetworKit-safe clone: rebuild graph by iterating edges.
    """
    H = nk.graph.Graph(G.numberOfNodes(), weighted=weighted, directed=directed)
    if weighted:
        for (u, v, w) in G.iterEdgesWeights():
            H.addEdge(u, v, w)
    else:
        for (u, v) in G.iterEdges():
            H.addEdge(u, v)
    return H
def remove_incident_edges(H, node, directed):
    """
    Remove all edges incident to 'node' in H.
    Works on NetworKit versions that expose iterNeighbors(u) but not forNeighborsOf(u, fn).
    """
    neigh = list(H.iterNeighbors(node))  # iterator form; no callback

    for v in neigh:
        if H.hasEdge(node, v):
            H.removeEdge(node, v)
        if directed and H.hasEdge(v, node):
            H.removeEdge(v, node)


# def remove_incident_edges(H, node, directed):
#     """
#     Remove all edges incident to 'node' in H.
#     NetworKit doesn't directly expose an 'incident edge iterator' that is stable under deletion,
#     so we collect neighbors first.
#     """
#     neigh = []
#     H.iterNeighbors(node, lambda v: neigh.append(v))
#     for v in neigh:
#         if H.hasEdge(node, v):
#             H.removeEdge(node, v)
#         if directed and H.hasEdge(v, node):
#             H.removeEdge(v, node)


# def remove_incident_edges(H, node, directed):
#     """
#     Remove all edges incident to 'node' in H.
#     Collect neighbors first because removing while iterating neighbors is unsafe.
#     """
#     neigh = []
#     H.forNeighborsOf(node, lambda v: neigh.append(v))

#     for v in neigh:
#         if H.hasEdge(node, v):
#             H.removeEdge(node, v)
#         if directed and H.hasEdge(v, node):
#             H.removeEdge(v, node)


def yen_k_shortest_paths(G, wts, s, t, K, weighted, directed):
    """
    Yen's K-shortest simple paths (by total weight / hop count),
    implemented by repeatedly calling NetworKit shortest path on modified graphs.
    Returns list of node-lists.
    """
    if K <= 0:
        return []

    p0 = nk_shortest_path_nodes(G, wts, s, t, weighted)
    if p0 is None:
        return []
    A = [p0]               # shortest paths found
    B = []                 # heap of candidates: (length, tie_id, path_nodes)
    tie = 0

    for k in range(1, K):
        prev = A[k-1]
        for i in range(len(prev) - 1):
            spur_node = prev[i]
            root_path = prev[:i+1]

            H = clone_graph(G, weighted, directed)

            # Remove edges that would recreate any previously found path with same root
            for p in A:
                if len(p) > i and p[:i+1] == root_path:
                    u = p[i]
                    v = p[i+1]
                    if H.hasEdge(u, v):
                        H.removeEdge(u, v)
                    if (not directed) and H.hasEdge(v, u):
                        H.removeEdge(v, u)

            # Enforce simplicity: remove all edges incident to root_path nodes except spur_node
            for root_node in root_path[:-1]:
                remove_incident_edges(H, root_node, directed)

            spur_path = nk_shortest_path_nodes(H, wts, spur_node, t, weighted)
            if spur_path is None:
                continue

            total_path = root_path[:-1] + spur_path
            L = path_length(wts, total_path)
            tie += 1
            heapq.heappush(B, (L, tie, total_path))

        if not B:
            break

        # Get next unique candidate
        next_path = None
        while B:
            _, _, cand = heapq.heappop(B)
            if cand not in A:
                next_path = cand
                break
        if next_path is None:
            break
        A.append(next_path)

    return A


def undirected_edge_key(u, v):
    return (u, v) if u <= v else (v, u)


# ----------------------------
# MCF: edge-based (no paths)
# ----------------------------

def solve_mcf_edge_based(G, commodities, cap, weighted, directed, time_limit=None, threads=None, verbose=True):
    """
    Edge-based multicommodity flow:
      maximize lambda
      flow conservation with supplies/demands scaled by lambda
      per-(undirected) edge capacity: sum_k (f_k(u,v)+f_k(v,u)) <= cap_e
    """
    n = G.numberOfNodes()

    # Build directed arc list from graph
    arcs = []
    if directed:
        for (u, v) in G.iterEdges():
            arcs.append((u, v))
    else:
        for (u, v) in G.iterEdges():
            arcs.append((u, v))
            arcs.append((v, u))

    # Map arcs out/in for flow conservation
    out_arcs = [[] for _ in range(n)]
    in_arcs = [[] for _ in range(n)]
    for (u, v) in arcs:
        out_arcs[u].append((u, v))
        in_arcs[v].append((u, v))

    m = gp.Model("MCF_edge_based")
    if not verbose:
        m.Params.OutputFlag = 0
    if time_limit is not None:
        m.Params.TimeLimit = float(time_limit)
    if threads is not None:
        m.Params.Threads = int(threads)

    # m.Params.Crossover = 0
    m.Params.Method = 2
    # Stronger numerical settings
    m.Params.NumericFocus = 3        # be conservative numerically
    m.Params.ScaleFlag = 2           # aggressive scaling (often helps)
    m.Params.FeasibilityTol = 1e-9   # default is looser
    m.Params.OptimalityTol = 1e-9
    m.Params.BarConvTol = 1e-12      # barrier convergence (only matters if Method=2)
    m.Params.BarHomogeneous = 1      # helps on ill-conditioned LPs



    lam = m.addVar(lb=0.0, name="lambda")

    # Flow variables f[k, u, v] for each commodity k and arc (u,v)
    f = {}
    for k, (s, t, dem) in enumerate(commodities):
        for (u, v) in arcs:
            f[(k, u, v)] = m.addVar(lb=0.0, name=f"f[{k},{u},{v}]")

    m.update()

    # Flow conservation
    for k, (s, t, dem) in enumerate(commodities):
        for v in range(n):
            expr_out = gp.quicksum(f[(k, u, w)] for (u, w) in out_arcs[v])
            expr_in = gp.quicksum(f[(k, u, w)] for (u, w) in in_arcs[v])

            if v == s:
                m.addConstr(expr_out - expr_in == lam * dem, name=f"flow_src[{k},{v}]")
            elif v == t:
                m.addConstr(expr_out - expr_in == -lam * dem, name=f"flow_dst[{k},{v}]")
            else:
                m.addConstr(expr_out - expr_in == 0.0, name=f"flow_mid[{k},{v}]")

    # Capacity constraints
    if directed:
        # Directed capacity per arc directly
        for (u, v) in G.iterEdges():
            ce = cap[(u, v)]
            m.addConstr(gp.quicksum(f[(k, u, v)] for k in range(len(commodities))) <= ce,
                        name=f"cap[{u},{v}]")
    else:
        # Undirected capacity: sum both directions across commodities
        for (u, v) in G.iterEdges():
            ce = cap[undirected_edge_key(u, v)]
            m.addConstr(
                gp.quicksum(f[(k, u, v)] + f[(k, v, u)] for k in range(len(commodities))) <= ce,
                name=f"cap[{u},{v}]"
            )

    m.setObjective(lam, GRB.MAXIMIZE)
    m.optimize()

    status = m.Status
    result = {
        "status": status,
        "lambda": None,
        "model": m,
        "flows": f,
        "lambda_var": lam
    }

    if status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        if m.SolCount > 0:
            result["lambda"] = lam.X

    return result


# ----------------------------
# MCF: path-based
# ----------------------------

def solve_mcf_path_based(G, wts, commodities, cap, K, weighted, directed,
                         time_limit=None, threads=None, verbose=True):
    """
    Path-based MCF:
      - Generate up to K candidate paths per commodity (K-shortest via Yen).
      - Variables x[k,p] >= 0 is amount of commodity k routed on path p.
      - For each commodity k: sum_p x[k,p] >= lambda * dem_k
      - Capacity: for each (undirected) edge e: sum_{k,p: e in path(k,p)} x[k,p] <= cap_e
    """
    m = gp.Model("MCF_path_based")
    if not verbose:
        m.Params.OutputFlag = 0
    if time_limit is not None:
        m.Params.TimeLimit = float(time_limit)
    if threads is not None:
        m.Params.Threads = int(threads)

    lam = m.addVar(lb=0.0, name="lambda")

    # Enumerate paths
    paths = {}          # paths[(k, pidx)] = node list
    x = {}

    # Edge incidence accumulation
    # inc[(edge_key)] = list of variables that use it (we'll build expressions)
    inc = defaultdict(list)

    for k, (s, t, dem) in enumerate(commodities):
        pk = yen_k_shortest_paths(G, wts, s, t, K, weighted, directed)
        if not pk:
            # If any commodity has no candidate path, lambda must be 0 in path model
            # We keep it feasible by adding constraint lam == 0 for this commodity.
            m.addConstr(lam == 0.0, name=f"no_path_force_lambda0[{k}]")
            continue

        for pidx, nodes in enumerate(pk):
            var = m.addVar(lb=0.0, name=f"x[{k},{pidx}]")
            x[(k, pidx)] = var
            paths[(k, pidx)] = nodes

            # Register edge usage for capacity constraints
            for i in range(len(nodes) - 1):
                u = nodes[i]
                v = nodes[i + 1]
                if directed:
                    ek = (u, v)
                else:
                    ek = undirected_edge_key(u, v)
                inc[ek].append(var)

        # Demand satisfaction
        m.addConstr(gp.quicksum(x[(k, pidx)] for pidx in range(len(pk))) >= lam * dem,
                    name=f"demand[{k}]")

    m.update()

    # Capacity constraints
    if directed:
        for (u, v) in G.iterEdges():
            ce = cap[(u, v)]
            expr = gp.quicksum(inc.get((u, v), []))
            m.addConstr(expr <= ce, name=f"cap[{u},{v}]")
    else:
        for (u, v) in G.iterEdges():
            ek = undirected_edge_key(u, v)
            ce = cap[ek]
            expr = gp.quicksum(inc.get(ek, []))
            m.addConstr(expr <= ce, name=f"cap[{u},{v}]")

    m.setObjective(lam, GRB.MAXIMIZE)
    m.optimize()

    status = m.Status
    result = {
        "status": status,
        "lambda": None,
        "model": m,
        "path_flows": x,
        "paths": paths,
        "lambda_var": lam
    }

    if status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        if m.SolCount > 0:
            result["lambda"] = lam.X

    return result


# ----------------------------
# Capacity model
# ----------------------------

def build_capacities(G, directed, base_capacity):
    """
    base_capacity: float
    For undirected graphs, cap keyed by (min(u,v), max(u,v)).
    For directed graphs, cap keyed by (u,v).
    """
    cap = {}
    if directed:
        for (u, v) in G.iterEdges():
            cap[(u, v)] = float(base_capacity)
    else:
        for (u, v) in G.iterEdges():
            cap[undirected_edge_key(u, v)] = float(base_capacity)
    return cap


# ----------------------------
# Reporting
# ----------------------------

def status_str(code):
    mp = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
    }
    return mp.get(code, str(code))


def main():
    ap = argparse.ArgumentParser(description="Maximum Concurrent Flow (MCF) in Gurobi: edge-based and path-based.")
    ap.add_argument("--matrix", required=True, help="Path to space-delimited adjacency matrix file.")
    ap.add_argument("--directed", action="store_true", help="Treat the matrix/graph as directed.")
    ap.add_argument("--weighted", action="store_true", help="Force weighted interpretation of matrix entries.")
    ap.add_argument("--unweighted", action="store_true", help="Force unweighted interpretation (any nonzero becomes edge).")
    ap.add_argument("--capacity", type=float, default=1.0, help="Uniform per-edge capacity.")
    ap.add_argument("--demands", default=None, help="Optional demands file: lines 's t demand'. If omitted, all s!=t have unit demand.")
    ap.add_argument("--sources", default=None, help="Optional comma-separated list of sources (restrict commodities).")
    ap.add_argument("--sinks", default=None, help="Optional comma-separated list of sinks (restrict commodities).")
    ap.add_argument("--formulation", choices=["edge", "path"], default="edge", help="Which formulation to solve.")
    ap.add_argument("--kpaths", type=int, default=8, help="For path formulation: number of candidate paths per commodity.")
    ap.add_argument("--time-limit", type=float, default=None, help="Gurobi time limit (seconds).")
    ap.add_argument("--threads", type=int, default=None, help="Gurobi Threads parameter.")
    ap.add_argument("--quiet", action="store_true", help="Suppress Gurobi output.")
    ap.add_argument("--mcf_out", type=str, default=None, help="Output file for MCF value.")
    args = ap.parse_args()

    if args.weighted and args.unweighted:
        raise ValueError("Choose at most one of --weighted / --unweighted")

    weighted = None
    if args.weighted:
        weighted = True
    if args.unweighted:
        weighted = False

    G, wts, inferred_weighted = read_adjacency_matrix(args.matrix, directed=args.directed, weighted=weighted)
    weighted = inferred_weighted if weighted is None else weighted

    n = G.numberOfNodes()
    if n == 0:
        raise ValueError("Graph has 0 nodes")

    cap = build_capacities(G, args.directed, args.capacity)

    demands = read_demands(args.demands, n)
    sources = None
    sinks = None
    if args.sources:
        sources = [int(x) for x in args.sources.split(",") if x.strip() != ""]
    if args.sinks:
        sinks = [int(x) for x in args.sinks.split(",") if x.strip() != ""]

    commodities = build_default_commodities(n, demands=demands, sources=sources, sinks=sinks)
    if not commodities:
        raise ValueError("No commodities to route (check demands/sources/sinks).")

    verbose = (not args.quiet)

    if args.formulation == "edge":
        res = solve_mcf_edge_based(
            G, commodities, cap, weighted, args.directed,
            time_limit=args.time_limit, threads=args.threads, verbose=verbose
        )
        print(f"[edge] status={status_str(res['status'])}  lambda={res['lambda']}")
    else:
        res = solve_mcf_path_based(
            G, wts, commodities, cap, args.kpaths, weighted, args.directed,
            time_limit=args.time_limit, threads=args.threads, verbose=verbose
        )
        print(f"[path] status={status_str(res['status'])}  lambda={res['lambda']}  kpaths={args.kpaths}")

    # Basic sanity note for TIME_LIMIT: report incumbent if exists
    if res["status"] == GRB.TIME_LIMIT and res["lambda"] is None:
        print("TIME_LIMIT reached but no feasible solution found (SolCount==0).")

    # Write MCF value to output file if provided
    if args.mcf_out:
        try:
            with open(args.mcf_out, "w") as f:
                if res["lambda"] is not None:
                    f.write(f"{res['lambda']}\n")
                else:
                    f.write("None\n")
        except Exception as e:
            print(f"Warning: Could not write MCF value to {args.mcf_out}: {e}")


if __name__ == "__main__":
    main()
