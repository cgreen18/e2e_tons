#!/usr/bin/env python3
"""
xpander_fixed.py

A paper-faithful Xpander topology generator:

Paper construction (Xpander: Towards Optimal-Performance Datacenters):
- Start with the complete d-regular graph on d+1 vertices (K_{d+1}).
- Repeatedly lift (2-lift / k-lift): for each base edge, insert a matching between copies.
- For arbitrary sizes, incrementally add a node by removing d/2 links and connecting the
  d incident endpoints to the new node; choose removed links via a spectral-gap heuristic.

This implementation:
- Builds a d-regular undirected graph via repeated lifts from K_{d+1}.
- If n is not reached exactly via lifts, grows by incremental rewiring (needs even d).
- Provides two growth modes:
    * --grow-method random   : remove a random matching of size d/2
    * --grow-method spectral : remove a matching chosen using an approximate spectral heuristic
                               (power iteration to approximate the top nontrivial eigenvector)
- Can generate multiple candidates and pick the best by approximate nontrivial eigenvalue.

Output:
- Adjacency matrix as a text file, space-delimited 0/1, one row per source.

Notes:
- Writing an adjacency matrix is O(n^2) output size.
- For large n, consider writing an edge list instead; but you explicitly requested matrix output.
"""

import argparse
import math
import random
import sys


# ----------------------------
# Graph utilities (undirected)
# ----------------------------

def complete_graph_adj(n):
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            adj[u].add(v)
            adj[v].add(u)
    return adj


def adj_to_edges(adj):
    edges = []
    for u in range(len(adj)):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    return edges


def check_simple_undirected(adj):
    n = len(adj)
    for u in range(n):
        if u in adj[u]:
            return False, f"self-loop at {u}"
        for v in adj[u]:
            if u not in adj[v]:
                return False, f"asymmetry: {u}-{v} missing reverse"
    return True, ""


def check_d_regular(adj, d):
    for u in range(len(adj)):
        if len(adj[u]) != d:
            return False, f"node {u} has degree {len(adj[u])}, expected {d}"
    return True, ""


# ----------------------------
# Lifts
# ----------------------------

def k_lift(adj, k, rng):
    """
    k-lift of an undirected simple graph:
      - create k copies per vertex
      - for each base edge {u,v}, choose a random permutation pi over [0..k-1]
        and add edges (u,i)--(v,pi(i)) for all i
    """
    if k < 2:
        raise ValueError("k must be >= 2")
    n0 = len(adj)
    edges0 = adj_to_edges(adj)

    n1 = n0 * k
    adj1 = [set() for _ in range(n1)]

    for u, v in edges0:
        perm = list(range(k))
        rng.shuffle(perm)
        for i in range(k):
            a = u * k + i
            b = v * k + perm[i]
            if a == b:
                raise RuntimeError("unexpected self-loop created in lift")
            adj1[a].add(b)
            adj1[b].add(a)

    ok, msg = check_simple_undirected(adj1)
    if not ok:
        raise RuntimeError(f"lift produced non-simple graph: {msg}")
    return adj1


# ---------------------------------------
# Spectral proxy: approx nontrivial lambda
# ---------------------------------------

def matvec_adj(adj, x):
    """y = A x for adjacency-list graph."""
    n = len(adj)
    y = [0.0] * n
    for i in range(n):
        s = 0.0
        for j in adj[i]:
            s += x[j]
        y[i] = s
    return y


def orthogonalize_to_ones(x):
    """Project x onto subspace orthogonal to all-ones vector."""
    n = len(x)
    mean = sum(x) / float(n)
    for i in range(n):
        x[i] -= mean
    return x


def normalize(x):
    norm = math.sqrt(sum(v * v for v in x))
    if norm == 0.0:
        return x, 0.0
    inv = 1.0 / norm
    for i in range(len(x)):
        x[i] *= inv
    return x, norm


def approx_nontrivial_eigenpair(adj, iters, rng):
    """
    Approximate the dominant eigenpair in the subspace orthogonal to all-ones
    (for d-regular graphs, this approximates the largest-magnitude nontrivial eigenvalue).
    Uses power iteration with centering each step.
    Returns (x, lambda_est).
    """
    n = len(adj)
    x = [rng.uniform(-1.0, 1.0) for _ in range(n)]
    orthogonalize_to_ones(x)
    x, _ = normalize(x)

    for _ in range(iters):
        y = matvec_adj(adj, x)
        orthogonalize_to_ones(y)
        y, yn = normalize(y)
        if yn == 0.0:
            break
        x = y

    # Rayleigh quotient: lambda ~ x^T A x / x^T x (x is normalized)
    Ax = matvec_adj(adj, x)
    lam = sum(x[i] * Ax[i] for i in range(n))
    return x, lam


# ----------------------------------------
# Incremental growth (add 1 node, keep d)
# ----------------------------------------

def sample_edges(edges, k, rng):
    if k is None or k <= 0 or k >= len(edges):
        return edges
    # Reservoir sampling would be better for huge lists; but |E|=n*d/2 is manageable.
    return rng.sample(edges, k)


def pick_matching_by_score(edges, score_fn, need_m, rng):
    """
    Pick a matching (no shared endpoints) of size need_m from edges,
    preferring smaller scores.
    """
    scored = [(score_fn(u, v), u, v) for (u, v) in edges]
    scored.sort(key=lambda t: t[0])

    used = set()
    chosen = []
    for _, u, v in scored:
        if u in used or v in used:
            continue
        used.add(u)
        used.add(v)
        chosen.append((u, v))
        if len(chosen) == need_m:
            return chosen

    # Fallback: randomized greedy if score ordering couldn't find enough disjoint edges
    for _ in range(50):
        used = set()
        chosen = []
        tmp = edges[:]
        rng.shuffle(tmp)
        for (u, v) in tmp:
            if u in used or v in used:
                continue
            used.add(u)
            used.add(v)
            chosen.append((u, v))
            if len(chosen) == need_m:
                return chosen

    raise RuntimeError(f"failed to find matching of size {need_m}")


def incremental_add_node(adj, d, rng, method, spectral_state):
    """
    Add one node to a d-regular undirected graph by:
      - removing m=d/2 edges (must be integer => even d)
      - connect the 2m = d freed endpoints to the new node

    method:
      - "random": remove a random matching of size d/2
      - "spectral": remove a matching chosen by spectral proxy

    spectral_state: dict for reusing eigenvector across steps.
    """
    if d % 2 != 0:
        raise ValueError("Incremental growth requires even d (removes d/2 edges).")

    n = len(adj)
    edges = adj_to_edges(adj)
    m = d // 2

    if method == "random":
        # random matching: shuffle edges then greedy pick disjoint
        rng.shuffle(edges)
        used = set()
        chosen = []
        for (u, v) in edges:
            if u in used or v in used:
                continue
            used.add(u)
            used.add(v)
            chosen.append((u, v))
            if len(chosen) == m:
                break
        if len(chosen) != m:
            raise RuntimeError("failed to find random matching for incremental add")

    elif method == "spectral":
        # Use (approx) second eigenvector x to score edges by |x_u * x_v|
        x = spectral_state.get("x")
        if x is None or len(x) != n:
            # Shouldn't happen; caller is expected to refresh x periodically
            x, _ = approx_nontrivial_eigenpair(adj, spectral_state["iters"], spectral_state["rng"])
            spectral_state["x"] = x

        # optionally sample edges to limit work
        cand = sample_edges(edges, spectral_state["candidates"], rng)

        def score_fn(u, v):
            return abs(x[u] * x[v])

        chosen = pick_matching_by_score(cand, score_fn, m, rng)

    else:
        raise ValueError(f"unknown method: {method}")

    # Build new graph
    new_adj = [set(neis) for neis in adj]
    new_adj.append(set())  # new node id = n

    endpoints = []
    for (u, v) in chosen:
        new_adj[u].remove(v)
        new_adj[v].remove(u)
        endpoints.append(u)
        endpoints.append(v)

    for u in endpoints:
        new_adj[n].add(u)
        new_adj[u].add(n)

    ok, msg = check_d_regular(new_adj, d)
    if not ok:
        raise RuntimeError(f"incremental add broke d-regularity: {msg}")

    return new_adj


# ----------------------------
# Xpander generator
# ----------------------------

def build_xpander(n, d, seed, lift_plan, grow_method, spectral_iters,
                 spectral_candidates, spectral_refresh, verbose):
    """
    Build a d-regular graph with exactly n nodes.

    lift_plan:
      - "2": repeated 2-lifts while size*2 <= n
      - "greedy-k": choose k each step as large as possible (<= max_k) s.t. size*k <= n
    """
    if n < d + 1:
        raise ValueError(f"need n >= d+1 for d-regular simple graph, got n={n}, d={d}")
    if d >= n:
        raise ValueError("need d <= n-1")
    if d < 1:
        raise ValueError("need d >= 1")

    rng = random.Random(seed)

    # Start with K_{d+1}
    adj = complete_graph_adj(d + 1)
    ok, msg = check_d_regular(adj, d)
    if not ok:
        raise RuntimeError(f"base K_(d+1) not d-regular? {msg}")

    # Lifting stage
    if lift_plan == "2":
        while len(adj) * 2 <= n:
            adj = k_lift(adj, 2, rng)
            if verbose:
                print(f"[lift 2] n={len(adj)}", file=sys.stderr)

    elif lift_plan == "greedy-k":
        # Greedily pick k to jump closer to n (still a true k-lift each time)
        # This is consistent with the paper's comment that different sequences of k-lifts are possible.
        max_k = 64
        while True:
            cur = len(adj)
            if cur >= n:
                break
            best_k = None
            for k in range(max_k, 1, -1):
                if cur * k <= n:
                    best_k = k
                    break
            if best_k is None:
                break
            adj = k_lift(adj, best_k, rng)
            if verbose:
                print(f"[lift {best_k}] n={len(adj)}", file=sys.stderr)
    else:
        raise ValueError(f"unknown lift plan: {lift_plan}")

    # Incremental stage (if needed)
    if len(adj) < n:
        if d % 2 != 0:
            raise ValueError(
                "n not reachable by lifts alone and incremental growth is needed, "
                "but incremental growth requires even d (removes d/2 edges). "
                "Either choose n reachable via lifts, or use even d."
            )

        spectral_state = {
            "iters": spectral_iters,
            "candidates": spectral_candidates,
            "refresh": spectral_refresh,
            "rng": rng,
            "x": None,
        }

        steps = 0
        while len(adj) < n:
            # refresh eigenvector periodically for spectral method
            if grow_method == "spectral":
                if steps % spectral_refresh == 0 or spectral_state["x"] is None:
                    x, lam = approx_nontrivial_eigenpair(adj, spectral_iters, rng)
                    spectral_state["x"] = x
                    spectral_state["lam"] = lam
                    if verbose:
                        print(f"[spectral refresh] n={len(adj)}  lambda~{lam:.4f}", file=sys.stderr)

                # extend x when adding a new node (cheap, then next refresh fixes it)
                # (we do this after the add by setting x=None if you prefer strictness)
            adj = incremental_add_node(adj, d, rng, grow_method, spectral_state)
            steps += 1
            if verbose and steps % 200 == 0:
                print(f"[grow] n={len(adj)}", file=sys.stderr)

            if grow_method == "spectral":
                # extend eigenvector with a 0 entry for the new node and re-center+normalize
                x = spectral_state["x"]
                x = x + [0.0]
                orthogonalize_to_ones(x)
                x, _ = normalize(x)
                spectral_state["x"] = x

    ok, msg = check_simple_undirected(adj)
    if not ok:
        raise RuntimeError(f"final graph not simple/undirected: {msg}")
    ok, msg = check_d_regular(adj, d)
    if not ok:
        raise RuntimeError(f"final graph not d-regular: {msg}")

    return adj


# ----------------------------
# Candidate selection
# ----------------------------

def score_graph_by_lambda(adj, iters, seed):
    rng = random.Random(seed)
    _, lam = approx_nontrivial_eigenpair(adj, iters, rng)
    return abs(lam)


# ----------------------------
# Output: adjacency matrix
# ----------------------------

def write_adjacency_matrix(adj, out_fh, directed):
    """
    Write space-delimited adjacency matrix (0/1).
    For directed output, we emit arcs; in this generator, directed mode means
    replacing each undirected edge with two arcs, so matrix remains symmetric.
    """
    n = len(adj)
    for u in range(n):
        row = []
        nbrs = adj[u]
        for v in range(n):
            if v in nbrs:
                row.append("1")
            else:
                row.append("0")
        out_fh.write(" ".join(row) + "\n")


# ----------------------------
# CLI
# ----------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Generate Xpander topology (paper-faithful) and output adjacency matrix.")
    p.add_argument("--n", type=int, required=True, help="Number of nodes (ToR switches).")
    p.add_argument("--d", type=int, required=True, help="Inter-switch degree (radix used for ToR-ToR links).")
    p.add_argument("--directed", action="store_true",
                   help="Output a directed adjacency matrix (here: bidirectional arcs for every undirected edge).")
    p.add_argument("--seed", type=int, default=1, help="RNG seed.")
    p.add_argument("--out", type=str, default="-", help="Output path ('-' for stdout).")

    p.add_argument("--lift-plan", choices=["2", "greedy-k"], default="2",
                   help="How to choose lift factors. '2' = repeated 2-lifts; 'greedy-k' = greedy k-lifts up to 64.")

    p.add_argument("--grow-method", choices=["spectral", "random"], default="spectral",
                   help="Incremental growth edge-removal method when lifts don't hit n exactly.")
    p.add_argument("--spectral-iters", type=int, default=30,
                   help="Power-iteration steps for spectral proxy (used for growth + optional candidate scoring).")
    p.add_argument("--spectral-candidates", type=int, default=50000,
                   help="Max edges sampled when choosing removal edges via spectral method (0 or negative = use all).")
    p.add_argument("--spectral-refresh", type=int, default=20,
                   help="Refresh approximate eigenvector every this many incremental steps.")

    p.add_argument("--candidates", type=int, default=1,
                   help="Generate this many independent Xpanders and select the best by approx |lambda_nontrivial|.")
    p.add_argument("--verbose", action="store_true", help="Progress to stderr.")
    return p.parse_args()


def main():
    args = parse_args()

    if args.candidates < 1:
        raise ValueError("--candidates must be >= 1")

    best_adj = None
    best_score = None

    for i in range(args.candidates):
        seed_i = args.seed + 1337 * i
        if args.verbose and args.candidates > 1:
            print(f"[candidate {i+1}/{args.candidates}] seed={seed_i}", file=sys.stderr)

        adj = build_xpander(
            n=args.n,
            d=args.d,
            seed=seed_i,
            lift_plan=args.lift_plan,
            grow_method=args.grow_method,
            spectral_iters=args.spectral_iters,
            spectral_candidates=args.spectral_candidates if args.spectral_candidates > 0 else None,
            spectral_refresh=max(1, args.spectral_refresh),
            verbose=args.verbose,
        )

        # score by nontrivial eigenvalue magnitude (smaller is better expander proxy)
        score = score_graph_by_lambda(adj, args.spectral_iters, seed_i + 7)

        if best_score is None or score < best_score:
            best_score = score
            best_adj = adj

    if args.verbose:
        print(f"[selected] approx |lambda_nontrivial|={best_score:.4f}", file=sys.stderr)

    if args.out == "-":
        write_adjacency_matrix(best_adj, sys.stdout, args.directed)
    else:
        with open(args.out, "w", encoding="utf-8") as f:
            write_adjacency_matrix(best_adj, f, args.directed)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
