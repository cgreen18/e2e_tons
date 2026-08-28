#!/usr/bin/env python3
"""
Translate a chosen path set (pathlist) into an MSCCL-style XML plan.

Paths are decomposed into epochs via the same pMCF-style compiler as
gen_coll_comm_a2a_tpuv4_sym.py, with each flow over a path assigned bw/flow = 1.0.

Pathlist format: use .paths files (same as convert_pathlist.py):
  - First line is a header (skipped by default for .paths).
  - One path per (source, destination) pair per line, each line a JSON array of
    node IDs (e.g. [0, 1], [0, 2, 3]).
  - Self-paths (s == d) are ignored; pathlist must cover all (s,d) with s != d.

Example:
  topology: four_mesh.map
  pathlist: topologies_and_routing/routepath_lists/four_mesh_naive_random.paths

  python_scripts/pathlist_to_xml.py \\
    --pathlist topologies_and_routing/routepath_lists/four_mesh_naive_random.paths \\
    --topology four_mesh.map \\
    --xml four_mesh_naive_random.xml

Topology (adj_list) is inferred from path edges if --topology is not given.
"""

import argparse
import math
import os
import sys
from collections import defaultdict

# Optional: JSONL support (like convert_pathlist.py)
try:
    import orjson
except ImportError:
    orjson = None
import json


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Import tpuv4_symmetry for symmetric allocation
PYTHON_SCRIPTS_DIR = os.path.normpath(os.path.join(BASE_DIR))
if PYTHON_SCRIPTS_DIR not in sys.path:
    sys.path.append(PYTHON_SCRIPTS_DIR)
try:
    from tpuv4_symmetry import TPUv4_Symmetry
except ImportError:
    TPUv4_Symmetry = None

print(f'importing',flush=True)
# Import compiler and XML writer from gen_coll_comm
import gen_coll_comm_a2a_tpuv4_sym as gen_coll_comm_module
from gen_coll_comm_a2a_tpuv4_sym import (
    compile_pmcf_to_link_epochs,
    compile_pmcf_to_link_epochs_sym,
    write_msccl_xml_from_link_epochs,
    verify_epochs_demand_satisfaction,
    ingest_map,
)

print(f'imported gen_coll_comm_a2a_tpuv4_sym',flush=True)
# Larger buffer for faster sequential read (bytes)
_READ_BUFFER_SIZE = 4 * 1024 * 1024  # 4 MiB


def stream_pathlist(path, skip_header=False):
    """Yield one path (list of int node IDs) per line. Fast path for .paths (binary + orjson)."""
    is_paths_file = path.endswith(".paths")
    # Fast path: .paths files are JSON-only; binary mode + orjson is much faster
    if is_paths_file and orjson is not None:
        with open(path, "rb", buffering=_READ_BUFFER_SIZE) as inf:
            if skip_header:
                next(inf, None)
            for line in inf:
                line = line.strip()
                if not line:
                    continue
                yield orjson.loads(line)
        return

    # General path: text mode, JSON or space-separated fallback
    with open(path, "r", buffering=_READ_BUFFER_SIZE) as inf:
        if skip_header:
            next(inf, None)
        for line in inf:
            line = line.strip()
            if not line:
                continue
            try:
                if orjson is not None:
                    yield orjson.loads(line)
                else:
                    yield json.loads(line)
                continue
            except (ValueError, TypeError, json.JSONDecodeError):
                pass
            try:
                yield [int(x) for x in line.split()]
            except ValueError:
                raise ValueError(f"Path line is neither JSON array nor space-separated ints: {line[:80]!r}...")


def load_pathlist_to_path_dict(pathlist_filepath, skip_header=False):
    """
    Load pathlist into path_dict: (s, d) -> [path1, path2, ...].
    Each path is a list of node IDs from s to d.
    Skips self-paths (s == d) and single-node lines.
    """
    path_dict = defaultdict(list)
    for path in stream_pathlist(pathlist_filepath, skip_header=skip_header):
        if len(path) < 2:
            continue
        s, d = path[0], path[-1]
        if s == d:
            continue
        path_dict[(s, d)].append(path)
        if s%100 == 0 and d==0:
            print(f's={s}, d={d}')
    return dict(path_dict)


def adj_list_from_path_dict(path_dict, n_nodes=None):
    """
    Build adjacency list from edges that appear in path_dict.
    If n_nodes is None, use max(node_id over all paths) + 1.
    """
    edges = set()
    max_node = -1
    for paths in path_dict.values():
        for path in paths:
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                edges.add((u, v))
                max_node = max(max_node, u, v)
    if n_nodes is None:
        n_nodes = max_node + 1
    adj_list = [[] for _ in range(n_nodes)]
    for u, v in edges:
        if u < n_nodes:
            adj_list[u].append(v)
    for u in range(n_nodes):
        adj_list[u] = sorted(set(adj_list[u]))
    return adj_list


def flow_paths_to_frac_bw_one_per_path(path_dict: dict):
    """Assign 1.0 bw/flow to each path: flow_paths_to_frac_bw[(s,d)] = [1.0, 1.0, ...]."""
    return {(s, d): [1.0] * len(paths) for (s, d), paths in path_dict.items()}


def main():
    parser = argparse.ArgumentParser(
        description="Translate a chosen path set (pathlist) into an MSCCL XML plan (pMCF-style epochs, 1.0 bw/flow per path)."
    )
    parser.add_argument("--pathlist", type=str, required=True,
                        help="Pathlist file (.paths: JSON per line, one path per (s,d); same format as convert_pathlist.py)")
    parser.add_argument("--xml", type=str, required=True, help="Output XML file path")
    parser.add_argument("--topology", type=str, default=None,
                        help="Topology file (adjacency matrix, e.g. four_mesh.map). If not set, topology is inferred from path edges.")
    parser.add_argument("--skip_header", action="store_true", default=None,
                        help="Skip first line of pathlist (default: True for .paths files)")
    parser.add_argument("--no_skip_header", action="store_false", dest="skip_header",
                        help="Do not skip first line")
    parser.add_argument("--n_chunks", type=int, default=1, help="Chunks per (s,d) for all-to-all (default 1)")
    parser.add_argument("--n_channels", type=int, default=1, help="Channels in XML (default 1)")
    parser.add_argument("--max_epochs", type=int, default=None, help="Max epochs (default: no limit)")
    parser.add_argument("--verify", action="store_true", help="Run epoch demand verification after compile")
    parser.add_argument("--symmetric", action="store_true",
                        help="Use symmetry: canonical sources/commodities only, then expand")
    parser.add_argument("--mc_dims", nargs=3, type=int, metavar=("MCX", "MCY", "MCZ"),
                        help="Mega-cube dimensions for symmetry (required if --symmetric)")
    parser.add_argument("--xyzc_dims", nargs=4, type=int, metavar=("X", "Y", "Z", "C"),
                        help="Global x, y, z, cube dimensions (required if --symmetric)")
    parser.add_argument("--sym_type", type=str, choices=["trans", "refl-trans"], default="trans",
                        help="Symmetry type (default: trans)")
    args = parser.parse_args()

    print(f'args = {args}',flush=True)
    
    # Validate symmetric arguments
    symmetric = args.symmetric
    if symmetric:
        if not args.mc_dims or not args.xyzc_dims:
            print('--symmetric requires --mc_dims <mcx> <mcy> <mcz> and --xyzc_dims <x> <y> <z> <c>')
            sys.exit(1)
        if TPUv4_Symmetry is None:
            print('tpuv4_symmetry module not found; cannot use --symmetric')
            sys.exit(1)
    
    skip_header = args.skip_header if args.skip_header is not None else args.pathlist.endswith(".paths")
    
    print(f'Loading pathlist from {args.pathlist}')
    path_dict = load_pathlist_to_path_dict(args.pathlist, skip_header=skip_header)
    if not path_dict:
        print("No paths loaded; pathlist is empty or invalid.")
        sys.exit(1)

    if args.topology:
        adj_mat, adj_list = ingest_map(args.topology, assert_uniform_capacity=True)
        n_nodes = len(adj_list)
    else:
        n_nodes = max(max(p[0], p[-1]) for paths in path_dict.values() for p in paths) + 1
        adj_list = adj_list_from_path_dict(path_dict, n_nodes)
        adj_mat = None  # Not needed if topology not provided

    # Create symmetry object if symmetric mode is enabled
    my_tpuv4_symmetry = None
    canonical_sources = None
    if symmetric:
        xyzc_dims = tuple(args.xyzc_dims)
        mc_dims = tuple(args.mc_dims)
        sym_type = args.sym_type
        my_tpuv4_symmetry = TPUv4_Symmetry(xyzc_dims, mc_dims=mc_dims, sym_type=sym_type)
        if args.topology:
            # Verify symmetry matches topology
            my_tpuv4_symmetry.verify_symmetry_for_topology(adj_mat)
        canonical_sources = my_tpuv4_symmetry.get_canonical_nodes()
        print(f"Symmetry verified; canonical sources: {len(canonical_sources)}")

    print(f"Checking for all-to-all paths")
    # Require all-to-all: check based on symmetric mode
    missing = []
    if symmetric:
        # For symmetric: only check canonical sources have paths to all destinations
        canon_set = set(canonical_sources)
        for sc in canonical_sources:
            for d in range(n_nodes):
                if sc == d:
                    continue
                if (sc, d) not in path_dict or not path_dict[(sc, d)]:
                    missing.append((sc, d))
        if missing:
            print(f"Pathlist missing canonical (sc,d) pairs: {len(missing)} (e.g. {missing[:5]}).")
            sys.exit(1)
        # Note: path_dict is kept full (not filtered) because compile_pmcf_to_link_epochs_sym
        # expects full path_dict and filters internally
        canon_path_count = sum(1 for (s, d) in path_dict.keys() if s in canon_set)
        print(f"Canonical source paths in pathlist: {canon_path_count} (s,d) pairs")
    else:
        # Original all-to-all check for all (s,d)
        for s in range(n_nodes):
            for d in range(n_nodes):
                if s == d:
                    continue
                if (s, d) not in path_dict or not path_dict[(s, d)]:
                    missing.append((s, d))
        if missing:
            print(f"Pathlist is not complete all-to-all: missing (s,d) for {len(missing)} pairs (e.g. {missing[:5]}).")
            sys.exit(1)

    print(f"Ingested pathlist")

    print(f"Pathlist: {args.pathlist} -> n_nodes={n_nodes}, (s,d) pairs={len(path_dict)}")
    
    # Compile epochs: use symmetric compiler if symmetric mode is enabled
    if symmetric:
        # Filter to canonical sources only for flow dict creation
        canon_set = set(canonical_sources)
        path_dict_canonical = {k: v for k, v in path_dict.items() if k[0] in canon_set}
        # For canonical sources only, assign 1.0 per path
        flow_paths_to_frac_bw_canonical = flow_paths_to_frac_bw_one_per_path(path_dict_canonical)
        
        # Workaround: compile_pmcf_to_link_epochs_sym calls _quantize_pmcf_paths with path_dict_canonical
        # (only canonical sources), but _quantize_pmcf_paths iterates over ALL (s,d) pairs and expects
        # paths for all of them. We need to monkey-patch _quantize_pmcf_paths to only check canonical sources.
        original_quantize = gen_coll_comm_module._quantize_pmcf_paths
        
        def _quantize_pmcf_paths_canonical_only(path_dict, flow_paths_to_frac_bw, n_chunks, n_nodes, eps=1e-12):
            """Wrapper that only quantizes canonical sources."""
            chunk_assignments = []
            for sc in canonical_sources:
                for d in range(n_nodes):
                    if sc == d:
                        continue
                    paths = path_dict.get((sc, d), None)
                    if paths is None or len(paths) == 0:
                        raise RuntimeError(f"pMCF compiler: missing path(s) for commodity {sc}->{d}")
                    path_fracs = flow_paths_to_frac_bw.get((sc, d), [])
                    if not path_fracs:
                        chosen = [0] * n_chunks
                    else:
                        total = sum(max(0.0, float(bw)) for bw in path_fracs)
                        if total <= eps:
                            chosen = [0] * n_chunks
                        else:
                            raw = [(p, (max(0.0, float(bw)) / total) * n_chunks) for p, bw in enumerate(path_fracs)]
                            floors = [(p, int(math.floor(val + 1e-15))) for p, val in raw]
                            used = sum(f for _, f in floors)
                            rem = n_chunks - used
                            remainders = sorted([(p, val - math.floor(val + 1e-15)) for p, val in raw],
                                                key=lambda x: x[1], reverse=True)
                            alloc = {p: f for p, f in floors}
                            if remainders:
                                for i in range(rem):
                                    alloc[remainders[i % len(remainders)][0]] += 1
                            else:
                                alloc[0] = alloc.get(0, 0) + rem
                            chosen = []
                            for p in sorted(alloc.keys()):
                                cnt = alloc[p]
                                chosen += [p] * cnt
                            chosen = chosen[:n_chunks]
                            if len(chosen) < n_chunks:
                                chosen += [0] * (n_chunks - len(chosen))
                    for q, pidx in enumerate(chosen[:n_chunks]):
                        pidx = int(pidx)
                        pidx = max(0, min(pidx, len(paths) - 1))
                        path = paths[pidx]
                        chunk_assignments.append((sc, d, q, path))
            return chunk_assignments
        
        # Monkey-patch the function
        gen_coll_comm_module._quantize_pmcf_paths = _quantize_pmcf_paths_canonical_only
        
        print("Compiling path set to link epochs (pMCF-sym-style, canonical sources only, then expand)...")
        try:
            epochs, meta = compile_pmcf_to_link_epochs_sym(
                adj_list=adj_list,
                path_dict=path_dict,  # Should have all canonical (sc,d) pairs
                flow_paths_to_frac_bw_canonical=flow_paths_to_frac_bw_canonical,
                canonical_sources=canonical_sources,
                my_tpuv4_symmetry=my_tpuv4_symmetry,
                n_chunks=args.n_chunks,
                n_channels=args.n_channels,
                max_epochs=args.max_epochs,
            )
        finally:
            # Restore original function
            gen_coll_comm_module._quantize_pmcf_paths = original_quantize
    else:
        flow_paths_to_frac_bw = flow_paths_to_frac_bw_one_per_path(path_dict)
        print("Compiling path set to link epochs (pMCF-style, 1.0 bw/flow per path)...")
        epochs, meta = compile_pmcf_to_link_epochs(
            adj_list=adj_list,
            path_dict=path_dict,
            flow_paths_to_frac_bw=flow_paths_to_frac_bw,
            n_chunks=args.n_chunks,
            n_channels=args.n_channels,
            max_epochs=args.max_epochs,
        )
    print(f"Epochs: {len(epochs)} (meta: {meta})")

    write_msccl_xml_from_link_epochs(
        adj_list=adj_list,
        epochs=epochs,
        out_xml_path=args.xml,
        n_chunks=args.n_chunks,
        n_channels=args.n_channels,
    )
    demand_ok, capacity_ok = verify_epochs_demand_satisfaction(adj_list, epochs, args.n_chunks)
    if not demand_ok or not capacity_ok:
        raise RuntimeError("Epoch verification failed")

    print(f"Wrote MSCCL XML to: {args.xml}")

    if args.verify:
        verify_epochs_demand_satisfaction(adj_list, epochs, args.n_chunks)


if __name__ == "__main__":
    main()
