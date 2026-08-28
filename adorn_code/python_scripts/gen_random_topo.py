#!/usr/bin/env python3
"""
Random r-regular topology generator.

Simple approach:
- Directed: For each source, randomly select r distinct destinations (exclude self).
  Result: r-out-regular digraph (each node has out-degree r).
- Undirected: For each source, repeatedly pick a random valid destination (not self,
  not already connected, and with remaining degree budget > 0); add edge and
  decrement both endpoints' budgets. Ensures radix is never exceeded.

No third-party dependencies (uses only random, argparse).
"""

import argparse
import random


def random_regular_directed(n, r):
    """
    For each source s, randomly select r distinct destinations (excluding s).
    Returns list of (u, v) directed edges. Each node has out-degree r.
    """
    if r >= n:
        raise ValueError("radix must be < n_nodes")
    edges = []
    for s in range(n):
        candidates = [t for t in range(n) if t != s]
        chosen = random.sample(candidates, r)
        for d in chosen:
            edges.append((s, d))
    return edges


def random_regular_undirected(n, r):
    """
    For each source s, add edges until s has degree r. For each choice of
    destination d: require d != s, edge (s,d) not already present, and
    d has remaining degree budget > 0. Add edge (s,d), decrement budget
    at s and d. n*r must be even.
    Returns list of (u, v) with u < v (edges).
    """
    if n * r % 2 != 0:
        raise ValueError(f"n*r must be even for undirected r-regular (n={n}, r={r})")
    if r >= n:
        raise ValueError("radix must be < n_nodes")
    budget = [r] * n  # remaining degree capacity per node
    edges = set()     # (min(u,v), max(u,v))

    for s in range(n):
        need = budget[s]
        for _ in range(need):
            valid = [
                t for t in range(n)
                if t != s and budget[t] > 0 and (min(s, t), max(s, t)) not in edges
            ]
            if not valid:
                raise RuntimeError(f"Cannot complete undirected r-regular (n={n}, r={r}) at source {s}")
            d = random.choice(valid)
            e = (min(s, d), max(s, d))
            edges.add(e)
            budget[s] -= 1
            budget[d] -= 1

    return list(edges)


def build_adjsets(n, edges, directed):
    """Build adjacency sets for writing matrix. directed: only out-neighbors; else symmetrize."""
    adj = [set() for _ in range(n)]
    if directed:
        for u, v in edges:
            if u != v:
                adj[u].add(v)
    else:
        for u, v in edges:
            if u != v:
                adj[u].add(v)
                adj[v].add(u)
    return adj


def write_adj_matrix(adj, out_path=None):
    """Write adjacency matrix as space-delimited 0/1, one row per node."""
    n = len(adj)
    f = open(out_path, "w") if out_path else __import__("sys").stdout
    try:
        for i in range(n):
            row = ["1" if j in adj[i] else "0" for j in range(n)]
            f.write(" ".join(row) + "\n")
    finally:
        if out_path:
            f.close()


def main():
    ap = argparse.ArgumentParser(description="Random r-regular topology generator")
    ap.add_argument("-n", "--n_nodes", type=int, required=True, help="Number of nodes")
    ap.add_argument("-d", "--radix", type=int, required=True, help="Degree (radix) per node")
    ap.add_argument("--directed", action="store_true", help="Directed graph (default: undirected)")
    ap.add_argument("-o", "--output", type=str, default=None, help="Output adjacency matrix path (default: stdout)")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = ap.parse_args()

    n = args.n_nodes
    r = args.radix
    if args.seed is not None:
        random.seed(args.seed)

    if args.directed:
        edges = random_regular_directed(n, r)
    else:
        edges = random_regular_undirected(n, r)

    adj = build_adjsets(n, edges, args.directed)
    write_adj_matrix(adj, args.output)
    if args.output:
        print(f"Random {'directed' if args.directed else 'undirected'} r-regular graph (n={n}, r={r}) saved to {args.output}")


if __name__ == "__main__":
    main()
