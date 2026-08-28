#!/usr/bin/env python3
"""
Analyze .nrl2, .vcmat2, and optionally .allowvcturns files.

Metrics:
  (1a) Allowed turns ignoring VC (only if --allowvcturns provided)
  (1b) Allowed turns considering VC (only if --allowvcturns provided)
  (2) Number of distinct channels used per VC
  (3) Maximally loaded channel per VC
  (4) Total number of hops per VC
  (5) Deadlock check: build Channel Dependency Graph (CDG) from turns implied by
      nrl2+vcmat2, add all turns, then check for a cycle once (cycle => deadlock risk).

Usage:
  python analyze_vcs.py \
      --nrl2 path/to/file.nrl2 \
      --vcmat2 path/to/file.vcmat2 \
      --nrouters 256 \
      [--allowvcturns path/to/file.allowvcturns]
"""

import argparse
import re
from collections import defaultdict

from omnicdg import OmniCDG


# -------------------------
# Fast parsing helpers
# -------------------------

def parse_4int_tuple_line(line):
    """
    Parse a line like "(0, 1, 2, 3)" into a 4-int tuple.
    Returns None on blank/comment/malformed.
    """
    s = line.strip()
    if not s or s.startswith("#"):
        return None

    if s[0] == "(" and s[-1] == ")":
        s = s[1:-1]

    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        return None

    try:
        return (int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]))
    except ValueError:
        return None


def parse_allowvcturns_line(line):
    """
    Parse a line like:
      "((0, 1, 0), (1, 2, 1)) : True"

    Returns:
      (a, b, vc0, c, vc1, is_allowed_bool) or None if malformed/blank/comment.

    The expected integers on LHS are:
      a, b, vc0, b, c, vc1
    """
    s = line.strip()
    if not s or s.startswith("#"):
        return None

    if ":" not in s:
        return None

    left, _, right = s.partition(":")
    truth = right.strip()
    if truth not in ("True", "False"):
        return None

    nums = re.findall(r"-?\d+", left)
    if len(nums) != 6:
        return None

    a, b, vc0, b2, c, vc1 = map(int, nums)
    if b != b2:
        return None

    return (a, b, vc0, c, vc1, truth == "True")


# -------------------------
# Loading + analysis
# -------------------------

def load_allowed_turns_metrics(allowvcturns_path, nrouters):
    """
    Computes BOTH:
      - allowed turns ignoring VC      : set of (a, b, c)
      - allowed turns considering VC   : set of (a,b,vc0,c,vc1)

    Only counts entries whose RHS is True.
    """
    allowed_turns_ignore_vc = set()   # (a,b,c)
    allowed_turns_with_vc = set()     # (a,b,vc0,c,vc1)

    total_lines = 0
    parsed_lines = 0
    true_lines = 0

    with open(allowvcturns_path, "r") as f:
        for line in f:
            total_lines += 1
            parsed = parse_allowvcturns_line(line)
            if parsed is None:
                continue
            parsed_lines += 1

            a, b, vc0, c, vc1, is_allowed = parsed
            if not is_allowed:
                continue

            true_lines += 1

            if not (0 <= a < nrouters and 0 <= b < nrouters and 0 <= c < nrouters):
                continue

            allowed_turns_ignore_vc.add((a, b, c))
            allowed_turns_with_vc.add((a, b, vc0, c, vc1))

    return {
        "allowed_turns_ignore_vc": allowed_turns_ignore_vc,
        "allowed_turns_with_vc": allowed_turns_with_vc,
        "total_lines": total_lines,
        "parsed_lines": parsed_lines,
        "true_lines": true_lines,
    }


def load_vcmat(vcmat2_path, nrouters):
    """
    Loads (path_src, path_dest, cur_node) -> vc mapping from .vcmat2
    """
    mapping = {}
    max_vc = -1

    total_lines = 0
    parsed_lines = 0

    with open(vcmat2_path, "r") as f:
        for line in f:
            total_lines += 1
            tup = parse_4int_tuple_line(line)
            if tup is None:
                continue
            parsed_lines += 1

            ps, pd, cur, vc = tup
            if not (0 <= ps < nrouters and 0 <= pd < nrouters and 0 <= cur < nrouters):
                continue

            mapping[(ps, pd, cur)] = vc
            if vc > max_vc:
                max_vc = vc

    return mapping, max_vc, total_lines, parsed_lines


def analyze_nrl2(nrl2_path, vcmat_map, nrouters):
    """
    Reads .nrl2 and combines with .vcmat2 to compute:
      - channels used per VC (set)
      - channel load per VC (count)
      - path_edges: (ps, pd) -> list of (hs, hd) in path order (for CDG turn extraction)
    """
    per_vc_channels = defaultdict(set)                 # vc -> set((u,v))
    per_vc_channel_count = defaultdict(lambda: defaultdict(int))  # vc -> (u,v)->count
    path_edges = defaultdict(list)                     # (ps, pd) -> [(hs, hd), ...] in path order

    missing_vc_entries = 0
    out_of_range = 0

    total_lines = 0
    parsed_lines = 0

    with open(nrl2_path, "r") as f:
        for line in f:
            total_lines += 1
            tup = parse_4int_tuple_line(line)
            if tup is None:
                continue
            parsed_lines += 1

            ps, pd, hs, hd = tup
            if not (0 <= ps < nrouters and 0 <= pd < nrouters and 0 <= hs < nrouters and 0 <= hd < nrouters):
                out_of_range += 1
                continue

            vc = vcmat_map.get((ps, pd, hs))
            if vc is None:
                missing_vc_entries += 1
                continue

            ch = (hs, hd)
            per_vc_channels[vc].add(ch)
            per_vc_channel_count[vc][ch] += 1
            path_edges[(ps, pd)].append((hs, hd))

    return {
        "per_vc_channels": per_vc_channels,
        "per_vc_channel_count": per_vc_channel_count,
        "path_edges": path_edges,
        "missing_vc_entries": missing_vc_entries,
        "out_of_range": out_of_range,
        "total_lines": total_lines,
        "parsed_lines": parsed_lines,
    }


def summarize(per_vc_channels, per_vc_channel_count, topk):
    """
    Returns:
      rows: list of (vc, n_channels, total_hops, max_load, max_channels_list)
      topk_detail: vc -> list of ((u,v), load)
    """
    vlist = sorted(per_vc_channels.keys())
    rows = []
    topk_detail = {}

    for vc in vlist:
        n_channels = len(per_vc_channels[vc])
        counts = per_vc_channel_count[vc]
        total_hops = sum(counts.values())

        if counts:
            max_load = max(counts.values())
            max_channels = [ch for ch, c in counts.items() if c == max_load]
        else:
            max_load = 0
            max_channels = []

        rows.append((vc, n_channels, total_hops, max_load, max_channels))

        items = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:topk]
        topk_detail[vc] = items

    return rows, topk_detail


def get_turns_from_path_edges(path_edges, vcmat_map):
    """
    Build the set of turns from path-ordered hops and VC mapping.

    path_edges: (ps, pd) -> list of (hs, hd) in path order
    vcmat_map: (ps, pd, cur) -> vc

    Returns set of turns: ((a, b, vc0), (b, c, vc1)) for consecutive hops (a,b),(b,c).
    """
    turns = set()
    for (ps, pd), edges in path_edges.items():
        for i in range(len(edges) - 1):
            a, b = edges[i]
            b2, c = edges[i + 1]
            if b != b2:
                continue
            vc0 = vcmat_map.get((ps, pd, a))
            vc1 = vcmat_map.get((ps, pd, b))
            if vc0 is not None and vc1 is not None:
                turn = ((a, b, vc0), (b, c, vc1))
                turns.add(turn)
    return turns


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nrl2", required=True, help="Path to .nrl2 routing file")
    ap.add_argument("--vcmat2", required=True, help="Path to .vcmat2 VC assignment file")
    ap.add_argument("--allowvcturns", default=None, help="(Optional) Path to .allowvcturns file")
    ap.add_argument("--nrouters", type=int, required=True, help="Number of routers (node IDs are assumed 0..nrouters-1)")
    ap.add_argument("--topk", type=int, default=5, help="Show top-K most loaded channels per VC (default: 5)")
    args = ap.parse_args()

    nrouters = args.nrouters

    # Optional allowvcturns metrics
    allow_stats = None
    allowed_turns_ignore_vc = None
    allowed_turns_with_vc = None

    if args.allowvcturns is not None:
        allow_stats = load_allowed_turns_metrics(args.allowvcturns, nrouters)
        allowed_turns_ignore_vc = allow_stats["allowed_turns_ignore_vc"]
        allowed_turns_with_vc = allow_stats["allowed_turns_with_vc"]

    # vcmat
    vcmat_map, max_vc, vcmat_total, vcmat_parsed = load_vcmat(args.vcmat2, nrouters)

    # nrl2
    nrl_stats = analyze_nrl2(args.nrl2, vcmat_map, nrouters)
    per_vc_channels = nrl_stats["per_vc_channels"]
    per_vc_channel_count = nrl_stats["per_vc_channel_count"]
    path_edges = nrl_stats["path_edges"]

    summary_rows, topk_detail = summarize(per_vc_channels, per_vc_channel_count, args.topk)

    # -------------------------
    # Print report
    # -------------------------
    print("=" * 80)
    print("Routing/VC Analytical Evaluation")
    print("=" * 80)
    print(f"nrouters: {nrouters}")
    print()

    print("Input parsing:")
    if args.allowvcturns is not None:
        print(f"  allowvcturns: {args.allowvcturns}")
        print(f"    lines total: {allow_stats['total_lines']:}   parsed: {allow_stats['parsed_lines']:}   True: {allow_stats['true_lines']:}")
    else:
        print("  allowvcturns: (not provided)")

    print(f"  vcmat2:       {args.vcmat2}")
    print(f"    lines total: {vcmat_total:}   parsed: {vcmat_parsed:}   entries kept: {len(vcmat_map):}")
    print(f"  nrl2:         {args.nrl2}")
    print(f"    lines total: {nrl_stats['total_lines']:}   parsed: {nrl_stats['parsed_lines']:}")
    print(f"    missing VC entries: {nrl_stats['missing_vc_entries']:}")
    print(f"    out-of-range tuples: {nrl_stats['out_of_range']:}")
    print()

    if args.allowvcturns is not None:
        print("Metric (1): Allowed turns")
        print(f"  (1a) ignoring VC  : {len(allowed_turns_ignore_vc):}   (unique a->b->c)")
        print(f"  (1b) with VC      : {len(allowed_turns_with_vc):}   (unique ((a,b,vc0)->(b,c,vc1)) that are True)")
        print()
    else:
        print("Metric (1): Allowed turns (skipped: no --allowvcturns provided)")
        print()

    print("Metric (2) + (3) + (4): Per-VC channel usage, maximal load, and total hops")
    if not summary_rows:
        print("  No VC/channel usage found (did not match vcmat2 entries).")
    else:
        print(f"{'VC':>6}  {'#ChannelsUsed':>14}  {'TotalHops':>12}  {'MaxLoad':>10}  {'#MaxChannels':>13}")
        print("-" * 75)
        for (vc, n_channels, total_hops, max_load, max_channels) in summary_rows:
            print(f"{vc:>6}  {n_channels:>14}  {total_hops:>12}  {max_load:>10}  {len(max_channels):>13}")

        print()
        print(f"Top-{args.topk} most loaded channels per VC (channel = hop_src -> hop_dest):")
        for (vc, _, _, _, _) in summary_rows:
            print("-" * 80)
            print(f"VC {vc}:")
            items = topk_detail.get(vc, [])
            if not items:
                print("  (no channels)")
                continue
            for (ch, cnt) in items:
                print(f"  {ch[0]} -> {ch[1]} : load {cnt:}")

    print("=" * 80)

    # -------------------------
    # CDG deadlock check (single cycle check after adding all turns)
    # -------------------------
    turns = get_turns_from_path_edges(path_edges, vcmat_map)
    n_vcs = max_vc + 1 if max_vc >= 0 else 1
    cdg = OmniCDG()
    cdg.init_w_n_nodes(nrouters, n_vcs=n_vcs)
    for turn in turns:
        cdg.add_turn(turn)
    cycle_edges = cdg.networkx_get_cycle()
    if cycle_edges:
        print("Deadlock risk: CDG contains a cycle (VC allocation may be deadlocky).")
        print("Cycle (channel IDs -> (u, v, vc)):")
        for (c_a_id, c_b_id) in cycle_edges:
            ch_a = cdg.translate_channel_id(c_a_id)
            ch_b = cdg.translate_channel_id(c_b_id)
            print(f"  {ch_a} -> {ch_b}")
    else:
        print("CDG acyclic: deadlock-free.")

    print("=" * 80)


if __name__ == "__main__":
    main()
