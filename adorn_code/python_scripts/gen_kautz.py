#!/usr/bin/env python3
"""
kautz_tools.py

Generates:
  1) Generalized Kautz graph (Imase-Itoh 1983 construction): nodes 0..n-1
     Directed edges: j ≡ -i*d - q (mod n), for q=1..d
     (Directed, d-in and d-out regular as per the paper.)

  2) Classic Kautz digraph K(d, m): nodes are strings of length m+1 over alphabet {0..d}
     with no equal adjacent symbols. Directed edges by shift+append (out-degree d).

Outputs:
  - Space-delimited adjacency matrix (rows=src, cols=dst, entry 0/1).

No third-party dependencies.
"""

import argparse
from itertools import product


def genkautz_edges(n, d):
    """
    Imase-Itoh construction:
      add edge i -> j where j = (-i*d - q) mod n, q=1..d
    Returns list of (u,v) directed edges with possible duplicates removed per u.
    """
    edges = []
    for i in range(n):
        nbrs = set()
        for q in range(1, d + 1):
            j = (-i * d - q) % n
            if j != i:  # avoid self-loop if it occurs (rare, but possible for small n)
                nbrs.add(j)
        for j in nbrs:
            edges.append((i, j))
    return edges


def kautz_nodes(d, m):
    """
    Vertices of Kautz K(d,m): all tuples length (m+1) over {0..d} with no equal adjacent.
    Returns list of node labels (tuples).
    """
    alphabet = list(range(d + 1))
    nodes = []
    for tup in product(alphabet, repeat=m + 1):
        ok = True
        for i in range(m):
            if tup[i] == tup[i + 1]:
                ok = False
                break
        if ok:
            nodes.append(tup)
    return nodes


def kautz_edges(d, m):
    """
    Directed edges of Kautz K(d,m):
      v=(x0..xm) -> w=(x1..xm, a) for any a != xm
    Returns (nodes, edges) where nodes are tuples and edges reference integer ids.
    """
    nodes = kautz_nodes(d, m)
    node_id = {v: idx for idx, v in enumerate(nodes)}

    alphabet = list(range(d + 1))
    edges = []
    for v in nodes:
        suffix = v[1:]
        last = v[-1]
        src = node_id[v]
        for a in alphabet:
            if a == last:
                continue
            w = suffix + (a,)
            dst = node_id[w]
            edges.append((src, dst))
    return nodes, edges


def build_adjsets(n, edges, undirected):
    """
    Build adjacency sets for fast row-wise writing of adjacency matrix.
    For directed: store outgoing neighbors.
    For undirected: symmetrize.
    """
    adj = [set() for _ in range(n)]
    if undirected:
        for u, v in edges:
            if u == v:
                continue
            adj[u].add(v)
            adj[v].add(u)
    else:
        for u, v in edges:
            if u == v:
                continue
            adj[u].add(v)
    return adj


def write_adj_matrix(adj, out_path=None):
    """
    Write adjacency matrix as space-delimited 0/1.
    Each row i: N entries for columns 0..N-1.
    Streamed row-by-row to avoid O(N^2) memory.
    """
    n = len(adj)
    out = open(out_path, "w") if out_path else None
    f = out if out is not None else __import__("sys").stdout

    # Row template: b"0 0 0 ... 0\n" with spaces; we patch positions to '1'
    # Layout: positions 0,2,4,... are digits; odd positions are spaces; last digit has '\n' after it.
    row_len = 2 * n - 1
    for i in range(n):
        row = bytearray(b"0 " * n)
        # row currently length 2n; fix last char to '0' (remove trailing space) and add newline
        row = row[:row_len]  # drop trailing space
        for j in adj[i]:
            row[2 * j] = ord("1")
        f.write(row.decode("ascii"))
        f.write("\n")

    if out is not None:
        out.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", choices=["genkautz", "kautz"], required=True,
                    help="Which graph family to generate.")
    ap.add_argument("--directed", choices=["0", "1"], default="1",
                    help="1 = directed (default), 0 = undirected (symmetrize edges).")

    # GenKautz params
    ap.add_argument("--n", type=int, help="Number of nodes (genkautz).")
    ap.add_argument("--d", type=int, required=True, help="Degree parameter (out-degree target).")

    # Kautz params
    ap.add_argument("--m", type=int, help="Kautz dimension m (nodes are length m+1 strings).")
    ap.add_argument("--mapping_out", type=str, default=None,
                    help="For kautz: write node-id mapping file (id -> label).")

    # Output
    ap.add_argument("--out_matrix", type=str, default=None,
                    help="Output path for adjacency matrix. If omitted, writes to stdout.")

    args = ap.parse_args()
    undirected = (args.directed == "0")

    if args.graph == "genkautz":
        if args.n is None:
            raise SystemExit("--n is required for --graph genkautz")
        n = args.n
        d = args.d
        edges = genkautz_edges(n, d)
        adj = build_adjsets(n, edges, undirected)
        write_adj_matrix(adj, args.out_matrix)

    elif args.graph == "kautz":
        if args.m is None:
            raise SystemExit("--m is required for --graph kautz")
        d = args.d
        m = args.m
        nodes, edges = kautz_edges(d, m)
        n = len(nodes)
        adj = build_adjsets(n, edges, undirected)
        if args.mapping_out:
            with open(args.mapping_out, "w") as f:
                for idx, label in enumerate(nodes):
                    f.write(f"{idx} {''.join(map(str, label))}\n")
        write_adj_matrix(adj, args.out_matrix)


if __name__ == "__main__":
    main()
