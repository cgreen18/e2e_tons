#!/usr/bin/env python3
"""
Fast, memory-efficient max channel-load / throughput for pathlists.

Bottlenecks in measure_pathlist_throughput.py (profiled on .rallpaths):
  - bytes.split() + list(int) per line (~80% of time)
  - full path parse on duplicate (src,dst) flows
  - set/dict hashing for seen flows and per-edge counts

This version uses a hand-rolled byte parser, numpy 2D arrays for dedup and
edge loads (O(1) indexing, dense memory), and skips hop parsing when a flow
was already counted.
"""

from __future__ import annotations

import argparse
import sys
from array import array

import numpy as np

try:
    import orjson
except ImportError:
    orjson = None  # type: ignore

_BUF = 4 * 1024 * 1024
_INIT_N = 256


def _grow2d(a: np.ndarray, need: int) -> np.ndarray:
    if need <= a.shape[0]:
        return a
    n = max(need, a.shape[0] * 2)
    out = np.zeros((n, n), dtype=a.dtype)
    old = a.shape[0]
    out[:old, :old] = a
    return out


def _first_last_ints(bline: bytes) -> tuple[int, int] | None:
    """Parse first and last decimal ints on a space-separated line."""
    n = len(bline)
    if n == 0:
        return None

    i = 0
    while bline[i] == 32:
        i += 1
        if i >= n:
            return None
    val = 0
    while i < n and 48 <= bline[i] <= 57:
        val = val * 10 + (bline[i] - 48)
        i += 1
    first = val

    j = n - 1
    while j >= 0 and bline[j] == 32:
        j -= 1
    if j < 0:
        return None
    end = j + 1
    val = 0
    while j >= 0 and 48 <= bline[j] <= 57:
        val = val * 10 + (bline[j] - 48)
        j -= 1
    last = val

    if first == last and i > end:
        # single-router path
        return first, last
    return first, last


def _parse_line_ints(bline: bytes, out: array) -> int:
    """Append all ints from bline into out; return count."""
    out_len = len(out)
    n = len(bline)
    i = 0
    while i < n:
        c = bline[i]
        if c == 32:
            i += 1
            continue
        val = 0
        while i < n and 48 <= bline[i] <= 57:
            val = val * 10 + (bline[i] - 48)
            i += 1
        out.append(val)
    return len(out) - out_len


def calc_cload_fast(
    pathlist_filepath: str,
    *,
    check_complete: bool = False,
) -> tuple[float, int, float, int]:
    """
    Returns (throughput, max_cload, avg_hops, n_routers).
    """
    print(f"Calculating max channel load from file {pathlist_filepath}")

    is_rallpaths = ".rallpaths" in pathlist_filepath
    is_jsonl = not is_rallpaths

    seen = np.zeros((_INIT_N, _INIT_N), dtype=np.bool_)
    cload = np.zeros((_INIT_N, _INIT_N), dtype=np.int32)
    max_r = 0
    total_hops = 0
    n_flows = 0
    scratch = array("I")

    if is_rallpaths:
        with open(pathlist_filepath, "rb", buffering=_BUF) as inf:
            for bline in inf:
                bline = bline.strip()
                if not bline:
                    continue

                ends = _first_last_ints(bline)
                if ends is None:
                    continue
                src, dst = ends
                max_r = max(max_r, src, dst)
                need = max_r + 1
                if need > seen.shape[0]:
                    seen = _grow2d(seen, need)
                    cload = _grow2d(cload, need)

                if seen[src, dst]:
                    continue
                seen[src, dst] = True
                n_flows += 1

                del scratch[:]
                nh = _parse_line_ints(bline, scratch)
                if nh < 2:
                    continue
                total_hops += nh - 1
                for k in range(nh - 1):
                    hop_src = scratch[k]
                    hop_dest = scratch[k + 1]
                    cload[hop_src, hop_dest] += 1
                    max_r = max(max_r, hop_dest)

    else:
        if orjson is None:
            raise RuntimeError("orjson is required for .paths JSONL files")
        with open(pathlist_filepath, "r", buffering=_BUF) as inf:
            next(inf, None)  # header
            for line in inf:
                line = line.strip()
                if not line:
                    continue
                path = orjson.loads(line)
                src = path[0]
                dst = path[-1]
                max_r = max(max_r, src, dst)
                need = max_r + 1
                if need > seen.shape[0]:
                    seen = _grow2d(seen, need)
                    cload = _grow2d(cload, need)

                if seen[src, dst]:
                    continue
                seen[src, dst] = True
                n_flows += 1

                nh = len(path) - 1
                total_hops += nh
                for k in range(nh):
                    cload[path[k], path[k + 1]] += 1

    n_routers = max_r + 1

    if check_complete and is_jsonl:
        for s in range(n_routers):
            for d in range(n_routers):
                if s != d and not seen[s, d]:
                    print(f"ERROR: ({s},{d}) not in pathlist", file=sys.stderr)
                    sys.exit(1)

    sub = cload[:n_routers, :n_routers]
    max_cload = int(sub.max())
    if max_cload == 0:
        print("ERROR: no channel load recorded", file=sys.stderr)
        sys.exit(1)

    ys, xs = np.where(sub == max_cload)
    maximally_loaded_edges = list(zip(ys.tolist(), xs.tolist()))

    throughput = 1.0 / max_cload
    denom = n_routers * (n_routers - 1)
    avg_hops = total_hops / denom if denom else 0.0

    n_show = min(5, len(maximally_loaded_edges))
    print(f"After everything, throughput is {throughput} from max cload {max_cload}")
    print(f"\tFrom edges {maximally_loaded_edges[:n_show]}")
    print(f"Average number of hops: {avg_hops}")

    return throughput, max_cload, avg_hops, n_routers


def _ingest_map(path_name: str) -> list[list[int]]:
    adj_mat: list[list[int]] = []
    with open(path_name, "r") as inf:
        for row in inf:
            r_conns = row.split()
            try:
                r_conns = [int(elem) for elem in r_conns]
            except ValueError:
                r_conns = [int(float(elem)) for elem in r_conns]
            adj_mat.append(r_conns)
    return adj_mat


def verify_pathlist_fast(pathlist_filepath: str, topo_filepath: str) -> None:
    print(f"Verifying pathlist {pathlist_filepath} w/ topology {topo_filepath}")
    topo_adjmat = _ingest_map(topo_filepath)
    scratch = array("I")

    if ".rallpaths" in pathlist_filepath:
        with open(pathlist_filepath, "rb", buffering=_BUF) as inf:
            for bline in inf:
                bline = bline.strip()
                if not bline:
                    continue
                del scratch[:]
                nh = _parse_line_ints(bline, scratch)
                for k in range(nh - 1):
                    i, j = scratch[k], scratch[k + 1]
                    if topo_adjmat[i][j] == 0:
                        print(f"ERROR: disconnected edge ({i},{j})")
                        sys.exit(1)
    else:
        if orjson is None:
            raise RuntimeError("orjson is required for .paths JSONL files")
        with open(pathlist_filepath, "r", buffering=_BUF) as inf:
            next(inf, None)
            for line in inf:
                line = line.strip()
                if not line:
                    continue
                path = orjson.loads(line)
                for k in range(len(path) - 1):
                    i, j = path[k], path[k + 1]
                    if topo_adjmat[i][j] == 0:
                        print(f"ERROR: disconnected edge ({i},{j}) in path {path}")
                        sys.exit(1)
    print("VALID")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fast max channel-load / throughput for pathlists",
    )
    parser.add_argument("--pathlist", type=str, help="pathlist (.paths or .rallpaths)")
    parser.add_argument("--topology", type=str, help=".map adjacency matrix")
    parser.add_argument(
        "--check-complete",
        action="store_true",
        help="for .paths only: verify every (s,d) pair exists (slow)",
    )
    args = parser.parse_args()

    if args.topology:
        verify_pathlist_fast(args.pathlist, args.topology)
    else:
        calc_cload_fast(args.pathlist, check_complete=args.check_complete)


if __name__ == "__main__":
    main()
