#!/usr/bin/env python3
"""
Verify destination-based routing for a chosen .paths file.

Destination-based means next hops are determined solely by the destination
router ID. Equivalently (see src/new_mclb.py destination_based subpath
constraints): for every path P = [s, ..., d] and every intermediate node
u = P[i], the subpath P[i:] must equal the chosen path for flow (u, d).
"""

import argparse
import json
import sys
from collections import defaultdict

try:
    import orjson
except ImportError:
    orjson = None


def stream_pathlist(filepath):
    """Yield paths from a .paths file (header line + one JSON array per line)."""
    open_mode = "rb" if orjson is not None else "r"
    with open(filepath, open_mode, buffering=1024 * 1024) as inf:
        next(inf, None)  # skip header
        for line in inf:
            if open_mode == "rb":
                line = line.strip()
            else:
                line = line.strip()
            if not line:
                continue
            if orjson is not None:
                yield orjson.loads(line)
            else:
                yield json.loads(line)


def load_paths(filepath):
    """
    Load .paths into dict (s, d) -> path (list of router IDs).
    Skips self-paths / single-node lines. Warns if a flow has multiple paths;
    the last path for that flow is kept.
    """
    path_dict = {}
    multi_path_flows = 0
    n_routers = 0

    for path in stream_pathlist(filepath):
        if len(path) < 2:
            continue
        s, d = path[0], path[-1]
        if s == d:
            continue
        n_routers = max(n_routers, s, d)
        key = (s, d)
        if key in path_dict:
            multi_path_flows += 1
        path_dict[key] = list(path)

    return path_dict, n_routers + 1, multi_path_flows


def verify_destbased(path_dict, max_examples=10):
    """
    Check subpath consistency and next-hop uniqueness by destination.

    Returns (ok, report_dict).
    """
    subpath_mismatches = []
    missing_subflows = []

    for (s, d), path in path_dict.items():
        # For each intermediate node along the path (exclude source and dest)
        for i in range(1, len(path) - 1):
            sub_path = path[i:]
            sub_src = sub_path[0]
            key = (sub_src, d)
            if key not in path_dict:
                missing_subflows.append({
                    "outer": path,
                    "sub": sub_path,
                    "flow": key,
                })
                continue
            chosen_sub = path_dict[key]
            if chosen_sub != sub_path:
                subpath_mismatches.append({
                    "outer": path,
                    "expected_sub": sub_path,
                    "chosen": chosen_sub,
                    "flow": key,
                })

    # Next hop at router u toward destination d must be unique
    next_hops = defaultdict(set)
    for (_s, d), path in path_dict.items():
        for i in range(len(path) - 1):
            u, nxt = path[i], path[i + 1]
            next_hops[(u, d)].add(nxt)

    nexthop_conflicts = [
        {"router": u, "dest": d, "next_hops": sorted(hops)}
        for (u, d), hops in next_hops.items()
        if len(hops) > 1
    ]

    ok = (
        len(subpath_mismatches) == 0
        and len(missing_subflows) == 0
        and len(nexthop_conflicts) == 0
    )

    report = {
        "n_flows": len(path_dict),
        "n_subpath_mismatches": len(subpath_mismatches),
        "n_missing_subflows": len(missing_subflows),
        "n_nexthop_conflicts": len(nexthop_conflicts),
        "subpath_mismatches": subpath_mismatches[:max_examples],
        "missing_subflows": missing_subflows[:max_examples],
        "nexthop_conflicts": nexthop_conflicts[:max_examples],
    }
    return ok, report


def print_report(filepath, n_routers, multi_path_flows, ok, report, max_examples):
    print(f"Verifying destination-based property for: {filepath}")
    print(f"  # routers (inferred) = {n_routers}")
    print(f"  # flows              = {report['n_flows']}")
    if multi_path_flows:
        print(f"  WARNING: {multi_path_flows} flows had multiple paths "
              f"(kept last path per flow)")

    print(f"  subpath mismatches   = {report['n_subpath_mismatches']}")
    print(f"  missing subflows     = {report['n_missing_subflows']}")
    print(f"  next-hop conflicts   = {report['n_nexthop_conflicts']}")

    for kind, key in (
        ("subpath mismatch", "subpath_mismatches"),
        ("missing subflow", "missing_subflows"),
        ("next-hop conflict", "nexthop_conflicts"),
    ):
        examples = report[key]
        if not examples:
            continue
        total = report[
            "n_subpath_mismatches" if key == "subpath_mismatches"
            else "n_missing_subflows" if key == "missing_subflows"
            else "n_nexthop_conflicts"
        ]
        shown = min(len(examples), max_examples, total)
        print(f"\n  Examples ({kind}, showing {shown}/{total}):")
        for ex in examples:
            if key == "nexthop_conflicts":
                print(f"    router {ex['router']} -> dest {ex['dest']}: "
                      f"next hops {ex['next_hops']}")
            elif key == "missing_subflows":
                print(f"    outer {ex['outer']}")
                print(f"      sub {ex['sub']} for flow {ex['flow']} not in pathlist")
            else:
                print(f"    outer {ex['outer']}")
                print(f"      flow {ex['flow']}: expected subpath {ex['expected_sub']}")
                print(f"      chosen path          {ex['chosen']}")

    print()
    if ok:
        print("PASS: pathlist is destination-based")
    else:
        print("FAIL: pathlist is NOT destination-based")


def main():
    parser = argparse.ArgumentParser(
        description="Verify that a .paths file is destination-based "
                    "(subpath consistency / unique next hop by destination)."
    )
    parser.add_argument(
        "--pathlist", "-p", type=str, required=True,
        help=".paths file to verify (header + JSON path arrays)",
    )
    parser.add_argument(
        "--max_examples", type=int, default=10,
        help="max violation examples to print (default: 10)",
    )
    args = parser.parse_args()

    path_dict, n_routers, multi_path_flows = load_paths(args.pathlist)
    if not path_dict:
        print(f"ERROR: no flows found in {args.pathlist}", file=sys.stderr)
        sys.exit(2)

    ok, report = verify_destbased(path_dict, max_examples=args.max_examples)
    print_report(
        args.pathlist, n_routers, multi_path_flows, ok, report, args.max_examples
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
