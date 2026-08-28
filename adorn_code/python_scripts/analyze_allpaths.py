#!/usr/bin/env python3
import argparse
from collections import defaultdict
from pathlib import Path

def parse_args():
    ap = argparse.ArgumentParser(
        description="Analyze path-list metrics: count number of paths per (src,dst) flow."
    )
    ap.add_argument(
        "pathlist",
        type=Path,
        help="Text file where each line is a space-delimited path (custom extension is fine).",
    )
    ap.add_argument(
        "--n-routers",
        type=int,
        default=None,
        help="Number of routers. If omitted, inferred as max_node_id+1 from file.",
    )
    ap.add_argument(
        "--print-flows",
        action="store_true",
        help="Print (src, dst, n_paths) for every non-trivial flow found.",
    )
    ap.add_argument(
        "--sort",
        choices=["srcdst", "count_desc", "count_asc"],
        default="srcdst",
        help="Sorting for printed flows.",
    )
    ap.add_argument(
        "--show-missing",
        action="store_true",
        help="Print missing (src,dst) flows (can be large). Requires known n_routers.",
    )
    return ap.parse_args()

def read_counts(pathlist_path):
    flow_to_count = defaultdict(int)
    max_node = -1
    n_lines = 0
    n_empty = 0
    n_bad = 0
    n_trivial = 0

    with pathlist_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            n_lines += 1
            s = line.strip()
            if not s:
                n_empty += 1
                continue

            parts = s.split()
            try:
                nodes = [int(x) for x in parts]
            except ValueError:
                n_bad += 1
                continue

            if not nodes:
                n_empty += 1
                continue

            # Track max node id for n_routers inference
            local_max = max(nodes)
            if local_max > max_node:
                max_node = local_max

            src = nodes[0]
            dst = nodes[-1]

            # Exclude trivial flows (src == dst), including single-node paths like "0"
            if src == dst:
                n_trivial += 1
                continue

            flow_to_count[(src, dst)] += 1

    return {
        "flow_to_count": flow_to_count,
        "max_node": max_node,
        "n_lines": n_lines,
        "n_empty": n_empty,
        "n_bad": n_bad,
        "n_trivial": n_trivial,
    }

def main():
    args = parse_args()
    if not args.pathlist.exists():
        raise SystemExit(f"ERROR: file not found: {args.pathlist}")

    info = read_counts(args.pathlist)
    flow_to_count = info["flow_to_count"]

    inferred_n_routers = (info["max_node"] + 1) if info["max_node"] >= 0 else 0
    n_routers = args.n_routers if args.n_routers is not None else inferred_n_routers

    expected_flows = n_routers * n_routers - n_routers  # excluding src==dst
    present_flows = len(flow_to_count)
    missing_flows = expected_flows - present_flows if expected_flows >= present_flows else 0

    counts = list(flow_to_count.values())
    total_nontrivial_paths = sum(counts)
    min_paths = min(counts) if counts else 0
    max_paths = max(counts) if counts else 0
    mean_paths = (total_nontrivial_paths / present_flows) if present_flows else 0.0

    print("==== Path-list metrics ====")
    print(f"File: {args.pathlist}")
    print(f"Total lines read: {info['n_lines']}")
    print(f"Empty lines: {info['n_empty']}")
    print(f"Unparseable (non-int) lines: {info['n_bad']}")
    print(f"Trivial paths excluded (src==dst): {info['n_trivial']}")
    print()
    print(f"Max node id seen: {info['max_node']}")
    print(f"n_routers: {n_routers} {'(user-specified)' if args.n_routers is not None else '(inferred)'}")
    print(f"Expected non-trivial flows (n^2 - n): {expected_flows}")
    print(f"Flows present in file (unique src!=dst): {present_flows}")
    print(f"Flows missing: {missing_flows}")
    print()
    print(f"Total non-trivial paths: {total_nontrivial_paths}")
    print(f"Path diversity ratio: {total_nontrivial_paths/expected_flows}")
    print(f"Paths per present-flow: min={min_paths} mean={mean_paths:.3f} max={max_paths}")

    if args.print_flows:
        items = list(flow_to_count.items())
        if args.sort == "srcdst":
            items.sort(key=lambda kv: (kv[0][0], kv[0][1]))
        elif args.sort == "count_desc":
            items.sort(key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))
        elif args.sort == "count_asc":
            items.sort(key=lambda kv: (kv[1], kv[0][0], kv[0][1]))

        print()
        print("src dst n_paths")
        for (src, dst), c in items:
            print(f"{src} {dst} {c}")

    if args.show_missing:
        if n_routers <= 0:
            print("Cannot show missing flows: n_routers is not known.")
            return
        present = set(flow_to_count.keys())
        print()
        print("Missing flows (src dst):")
        for src in range(n_routers):
            for dst in range(n_routers):
                if src == dst:
                    continue
                if (src, dst) not in present:
                    print(f"{src} {dst}")

if __name__ == "__main__":
    main()
