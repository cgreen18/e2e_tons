"""I/O helpers for complete CDG topology/routing synthesis."""

import json
import math
import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime

_LOAD_STATS_EPS = 1e-9


def read_matrix_file(path, n):
    """Load an n x n numeric matrix (whitespace-separated rows)."""
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != n:
                raise ValueError(
                    f"{path}: expected {n} columns, got {len(parts)} in row {len(rows)}"
                )
            rows.append([float(x) for x in parts])
    if len(rows) != n:
        raise ValueError(f"{path}: expected {n} rows, got {len(rows)}")
    return rows


def uniform_matrix(n, value):
    return [[float(value) for _ in range(n)] for _ in range(n)]


def default_commodities(n):
    pairs = defaultdict(list)
    for s in range(n):
        for d in range(n):
            if s != d:
                pairs[s].append(d)
    return pairs


def parse_commodities_file(path, n):
    pairs = defaultdict(list)
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"commodities file: expected two ints per line, got {line!r}")
            s, d = int(parts[0]), int(parts[1])
            if not (0 <= s < n and 0 <= d < n):
                raise ValueError(f"commodities file: pair ({s},{d}) out of range for n={n}")
            if s != d:
                pairs[s].append(d)
    if not pairs:
        raise ValueError("commodities file: no valid pairs")
    return pairs


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def write_json(path, payload):
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


def write_topology_graphml(path, adj):
    """Write a directed graph from a binary adjacency matrix."""
    n = len(adj)
    root = ET.Element(
        "graphml",
        {
            "xmlns": "http://graphml.graphdrawing.org/xmlns",
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:schemaLocation": (
                "http://graphml.graphdrawing.org/xmlns "
                "http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd"
            ),
        },
    )
    graph = ET.SubElement(root, "graph", {"id": "G", "edgedefault": "directed"})
    for i in range(n):
        ET.SubElement(graph, "node", {"id": f"n{i}"})
    eid = 0
    for i in range(n):
        for j in range(n):
            if i != j and int(adj[i][j]) != 0:
                ET.SubElement(
                    graph,
                    "edge",
                    {"id": f"e{eid}", "source": f"n{i}", "target": f"n{j}"},
                )
                eid += 1
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    with open(path, "w", encoding="utf-8") as fp:
        tree.write(fp, encoding="unicode", xml_declaration=True)


def print_results(
    final_mcf_val,
    final_topology,
    final_paths,
    final_link_load,
    final_routing_table,
    final_vc_table,
    capacity,
    demand,
):
    n = len(final_topology)
    mcf = float(final_mcf_val)

    def summarize_physical_loads(title, edge_load):
        loads = []
        overflow_ct = 0
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                cap_ij = capacity[i][j]
                if cap_ij <= 0:
                    continue
                ld = float(edge_load.get((i, j), 0.0))
                loads.append(ld)
                if ld > cap_ij * (1.0 + _LOAD_STATS_EPS):
                    overflow_ct += 1
        print(title)
        if loads:
            print(
                "  physical link load (absolute): "
                f"count={len(loads)} min={min(loads):.6g} "
                f"mean={sum(loads)/len(loads):.6g} max={max(loads):.6g}"
            )
        else:
            print("  (no capacitated directed edges)")
        print(f"  capacity overflow edges (load > capacity): {overflow_ct}")

    print(f"lambda = {final_mcf_val}")
    print("topology:")
    for i, row in enumerate(final_topology):
        print(f"  {i}: {row}")
    print("paths (by (s,d)):")
    for (s, d) in sorted(final_paths.keys(), key=lambda x: (x[0], x[1])):
        print(f"  {s} -> {d}:")
        for path in final_paths[(s, d)]:
            print(
                f"    phys={path.get('phys_nodes')} vcs={path.get('vcs')} "
                f"alloc={path.get('allocation')} W_k={path.get('W_k')} S_k={path.get('S_k')}"
            )

    summarize_physical_loads(
        "Link load (hardened routing, MCF throughput: demand[s,d] scaled by lambda):",
        final_link_load,
    )

    unit_load = defaultdict(float)
    for (s, d), path_list in final_paths.items():
        denom = mcf * float(demand[s][d])
        if denom <= _LOAD_STATS_EPS:
            continue
        for p in path_list:
            alloc = float(p.get("allocation", 0.0))
            if alloc <= _LOAD_STATS_EPS:
                continue
            inc = alloc / denom
            phys = p.get("phys_nodes") or []
            for h in range(len(phys) - 1):
                unit_load[(phys[h], phys[h + 1])] += inc

    summarize_physical_loads(
        "Link load (same routing split; unit 1.0 flow per source-destination pair):",
        unit_load,
    )

    rt_entries = sum(
        1
        for by_s in final_routing_table.values()
        for by_d in by_s.values()
        for nxt_list in by_d.values()
        for x in nxt_list
        if x is not None
    )
    vc_entries = sum(
        1
        for by_s in final_vc_table.values()
        for by_d in by_s.values()
        for vc_list in by_d.values()
        for x in vc_list
        if x is not None
    )
    print(f"routing table next-hop entries (non-null): {rt_entries}")
    print(f"VC table entries (non-null): {vc_entries}")


def write_results(file_args, problem_args, final_mcf_val, final_topology, final_paths,
                  final_link_load, final_routing_table, final_vc_table):
    """Write .map, .graphml, .paths, .paths.jsonl, .nrl2, and .vcmat2 outputs."""
    del final_mcf_val, final_link_load  # kept for API compatibility
    assert problem_args["n_nodes"] == len(final_topology)
    base = file_args["base_out_name"]

    ensure_parent_dir(file_args["topo_out_path"])
    with open(file_args["topo_out_path"], "w", encoding="utf-8") as f:
        for row in final_topology:
            f.write(" ".join(str(int(x)) for x in row) + "\n")

    write_topology_graphml(file_args["graphml_out_path"], final_topology)

    ensure_parent_dir(file_args["paths_out_path"])
    with open(file_args["paths_out_path"], "w", encoding="utf-8") as f:
        f.write(base + "\n")
        for (s, d) in sorted(final_paths.keys(), key=lambda x: (x[0], x[1])):
            for p in final_paths[(s, d)]:
                f.write(json.dumps(p["phys_nodes"]) + "\n")

    ensure_parent_dir(file_args["paths_jsonl_out_path"])
    with open(file_args["paths_jsonl_out_path"], "w", encoding="utf-8") as f:
        for (s, d) in sorted(final_paths.keys(), key=lambda x: (x[0], x[1])):
            for path_idx, p in enumerate(final_paths[(s, d)]):
                f.write(json.dumps({
                    "s": s, "d": d, "path_idx": path_idx,
                    "phys_nodes": p.get("phys_nodes"), "vcs": p.get("vcs"),
                    "allocation": p.get("allocation"),
                    "W_k": p.get("W_k"), "S_k": p.get("S_k"),
                }) + "\n")

    ensure_parent_dir(file_args["nr_out_path"])
    with open(file_args["nr_out_path"], "w", encoding="utf-8") as f:
        for (s, d) in sorted(final_paths.keys(), key=lambda x: (x[0], x[1])):
            for p in final_paths[(s, d)]:
                phys = p["phys_nodes"]
                for h in range(len(phys) - 1):
                    f.write(f"({s}, {d}, {phys[h]}, {phys[h + 1]})\n")

    ensure_parent_dir(file_args["vc_out_path"])
    with open(file_args["vc_out_path"], "w", encoding="utf-8") as f:
        for (s, d) in sorted(final_paths.keys(), key=lambda x: (x[0], x[1])):
            for p in final_paths[(s, d)]:
                phys, vcs = p["phys_nodes"], p["vcs"]
                for h in range(len(phys) - 1):
                    vc = vcs[h] if h < len(vcs) else 0
                    f.write(f"({s}, {d}, {phys[h]}, {vc})\n")


def write_topology_image(path, topology, title=None, directed=True):
    """Render a binary adjacency matrix to a PNG (directed graph, grid layout)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    n = len(topology)
    g = nx.DiGraph()
    g.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if i != j and int(topology[i][j]) != 0:
                g.add_edge(i, j)

    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    pos = {node: (node % cols, -(node // cols)) for node in range(n)}

    fig_w = max(6.0, cols * 0.85)
    fig_h = max(4.0, rows * 0.85)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_aspect("equal")
    nx.draw_networkx_edges(
        g,
        pos,
        ax=ax,
        width=1.0,
        alpha=0.85,
        arrows=directed,
        arrowsize=14,
        connectionstyle="arc3,rad=0.08",
    )
    nx.draw_networkx_nodes(g, pos, ax=ax, node_size=max(120, 420 - 12 * n), node_color="lightblue")
    nx.draw_networkx_labels(g, pos, ax=ax, font_size=max(6, min(10, 160 // max(1, n))))

    out_deg = [sum(int(topology[i][j]) for j in range(n) if i != j) for i in range(n)]
    in_deg = [sum(int(topology[i][j]) for i in range(n) if i != j) for j in range(n)]
    if title is None:
        title = (
            f"Hardened topology: n={n}, edges={g.number_of_edges()}, "
            f"max_out_deg={max(out_deg) if out_deg else 0}"
        )
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    ensure_parent_dir(path)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"out_degrees": out_deg, "in_degrees": in_deg, "n_edges": g.number_of_edges()}


def build_output_paths(base_out_name):
    """Return file_args dict from a base output name."""
    return {
        "base_out_name": base_out_name,
        "topo_out_path": os.path.join("topologies_and_routing/topo_maps", base_out_name + ".map"),
        "graphml_out_path": os.path.join("topologies_and_routing/topo_maps", base_out_name + ".graphml"),
        "hardened_topo_png_path": os.path.join(
            "topologies_and_routing/topo_maps", base_out_name + "_hardened_topo.png"
        ),
        "paths_out_path": os.path.join("topologies_and_routing/routepath_lists", base_out_name + ".paths"),
        "paths_jsonl_out_path": os.path.join(
            "topologies_and_routing/routepath_lists", base_out_name + ".paths.jsonl"
        ),
        "nr_out_path": os.path.join("topologies_and_routing/nr_lists", base_out_name + ".nrl2"),
        "vc_out_path": os.path.join("topologies_and_routing/vc_mats", base_out_name + ".vcmat2"),
    }


def solution_summary_payload(final_mcf_val, n, radix, n_vcs, commodities, file_args):
    return {
        "lambda": final_mcf_val,
        "n_nodes": n,
        "radix": radix,
        "n_vcs": n_vcs,
        "n_commodities": sum(len(v) for v in commodities.values()),
        "out_topology_map": file_args["topo_out_path"],
        "out_topology_graphml": file_args["graphml_out_path"],
        "out_path_file": file_args["paths_out_path"],
        "out_path_jsonl": file_args["paths_jsonl_out_path"],
        "out_nr_file": file_args["nr_out_path"],
        "out_vc_file": file_args["vc_out_path"],
        "datetime_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "datetime_solved": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
