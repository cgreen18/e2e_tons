#!/usr/bin/env python3
"""
viz_graph_from_spaceints.py

Reads a space-delimited file of ints and visualizes the graph.

Auto-detect input format:
  A) Adjacency MATRIX (like your example):
       n lines, each with n ints (often 0/1). Row i, col j => edge i--j (or i->j).
  B) Adjacency LIST:
       each line: u v1 v2 v3 ...
       meaning edges u--v1, u--v2, ...

Layout:
  cols = ceil(sqrt(n))
  rows = ceil(n / cols)
  node i positioned at (i % cols, -(i // cols))

Examples:
  python viz_graph_from_spaceints.py --in radix_10_r2_lp.map --with-labels
  python viz_graph_from_spaceints.py --in topo.txt --directed --with-labels
  python viz_graph_from_spaceints.py --in topo.txt --save topo.png --no-show
"""

import argparse
import math

import matplotlib.pyplot as plt
import networkx as nx


def read_space_int_lines(path):
    lines = []
    with open(path, "r") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            parts = raw.split()
            lines.append([int(x) for x in parts])
    return lines


def is_square_matrix(lines):
    if not lines:
        return False
    n = len(lines)
    return all(len(row) == n for row in lines)


def looks_binary_matrix(lines):
    # Heuristic: all entries in {0,1} and square
    if not is_square_matrix(lines):
        return False
    for row in lines:
        for x in row:
            if x not in (0, 1):
                return False
    return True


def build_graph_from_matrix(mat, directed=False, treat_nonzero_as_edge=True):
    n = len(mat)
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(range(n))

    for i in range(n):
        row = mat[i]
        for j in range(n):
            if i == j:
                continue
            val = row[j]
            if treat_nonzero_as_edge:
                if val != 0:
                    G.add_edge(i, j) if directed else G.add_edge(i, j)
            else:
                # reserved if you ever want weighted edges, etc.
                pass

    if (not directed) and isinstance(G, nx.Graph):
        # If matrix is asymmetric but user wanted undirected, Graph() already merges.
        pass

    return G


def build_graph_from_adjlist(lines, directed=False):
    G = nx.DiGraph() if directed else nx.Graph()

    # determine node set robustly (include neighbors)
    nodes = set()
    edges = []
    for row in lines:
        if len(row) == 0:
            continue
        u = row[0]
        nodes.add(u)
        for v in row[1:]:
            nodes.add(v)
            edges.append((u, v))

    if nodes:
        # If nodes are 0..n-1, this keeps that. Otherwise, it still draws fine.
        G.add_nodes_from(sorted(nodes))
    G.add_edges_from(edges)
    return G


def grid_positions(nodes):
    # nodes may not be 0..n-1; position them in sorted order on the grid
    node_list = sorted(nodes)
    n = len(node_list)
    if n == 0:
        return {}, 0, 0

    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))

    pos = {}
    for idx, node in enumerate(node_list):
        r = idx // cols
        c = idx % cols
        pos[node] = (c, -r)
    return pos, rows, cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Space-delimited int file")
    ap.add_argument("--directed", action="store_true", help="Treat edges as directed")
    ap.add_argument("--force-matrix", action="store_true", help="Force interpret input as adjacency matrix")
    ap.add_argument("--force-adjlist", action="store_true", help="Force interpret input as adjacency list")
    ap.add_argument("--with-labels", action="store_true", help="Draw node labels")
    ap.add_argument("--node-size", type=float, default=180.0, help="Node marker size")
    ap.add_argument("--font-size", type=float, default=8.0, help="Label font size")
    ap.add_argument("--title", type=str, default=None, help="Plot title")
    ap.add_argument("--save", type=str, default=None, help="Save to image file")
    ap.add_argument("--no-show", action="store_true", help="Do not open an interactive window")
    args = ap.parse_args()

    if args.force_matrix and args.force_adjlist:
        raise SystemExit("Choose at most one of --force-matrix or --force-adjlist")

    lines = read_space_int_lines(args.inp)
    if not lines:
        raise SystemExit("Empty/invalid input file (no integer lines found).")

    # Detect format
    interpret_as_matrix = False
    if args.force_matrix:
        interpret_as_matrix = True
    elif args.force_adjlist:
        interpret_as_matrix = False
    else:
        # Prefer matrix if it's square (especially if binary like your example)
        interpret_as_matrix = is_square_matrix(lines)

    if interpret_as_matrix:
        mat = lines
        G = build_graph_from_matrix(mat, directed=args.directed)
        n = len(mat)
        fmt = f"matrix {n}x{n}"
    else:
        G = build_graph_from_adjlist(lines, directed=args.directed)
        fmt = "adjlist"

    pos, rows, cols = grid_positions(G.nodes())

    fig, ax = plt.subplots(figsize=(max(6, cols * 0.7), max(4, rows * 0.7)))
    ax.set_aspect("equal")

    nx.draw_networkx_edges(G, pos, ax=ax, width=1.0, alpha=0.8, arrows=args.directed)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=args.node_size)

    if args.with_labels:
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=args.font_size)

    if args.title is None:
        title = f"Graph ({fmt}) | n={G.number_of_nodes()} m={G.number_of_edges()} | cols={cols} rows={rows}"
    else:
        title = args.title
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=200, bbox_inches="tight")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
