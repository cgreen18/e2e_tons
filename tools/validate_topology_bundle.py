#!/usr/bin/env python3
"""Validate a known or explicitly named TONS topology/routing bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tons_topology.validation import (  # noqa: E402
    ArtifactFormatError,
    BundlePaths,
    known_bundle,
    validate_bundle,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cross-check topology, route, next-hop, VC, and allowed-turn artifacts."
    )
    parser.add_argument(
        "--bundle", choices=("pt-128", "pt-dor-128", "pdtt-128", "tons-128")
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("topologies_and_routing"),
        help="artifact root used with --bundle (default: %(default)s)",
    )
    parser.add_argument("--map", dest="topology", type=Path)
    parser.add_argument("--routes", type=Path)
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--next-hops", type=Path)
    parser.add_argument("--vc-matrix", type=Path)
    parser.add_argument("--allowed-turns", type=Path)
    parser.add_argument("--expected-degree", type=int)
    parser.add_argument("--destination-based", action="store_true")
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.bundle:
        explicit = (
            args.topology,
            args.routes,
            args.candidates,
            args.next_hops,
            args.vc_matrix,
            args.allowed_turns,
        )
        if any(item is not None for item in explicit):
            parser.error("--bundle cannot be combined with explicit artifact paths")
        paths = known_bundle(args.root, args.bundle)
        expected_degree = 6 if args.expected_degree is None else args.expected_degree
        destination_based = True
    else:
        if args.topology is None or args.routes is None:
            parser.error("provide --bundle or both --map and --routes")
        paths = BundlePaths(
            topology=args.topology,
            routes=args.routes,
            candidates=args.candidates,
            next_hops=args.next_hops,
            vc_matrix=args.vc_matrix,
            allowed_turns=args.allowed_turns,
        )
        expected_degree = args.expected_degree
        destination_based = args.destination_based

    try:
        report = validate_bundle(
            paths,
            expected_degree=expected_degree,
            destination_based=destination_based,
        )
    except ArtifactFormatError as exc:
        if args.json:
            print(json.dumps({"ok": False, "format_error": str(exc)}, indent=2))
        else:
            print(f"FORMAT ERROR: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        status = "OK" if report.ok else "FAILED"
        print(f"[{status}] {report.name}: {report.routers} routers, "
              f"degree {report.min_degree}..{report.max_degree}")
        print(f"  selected flows/hops: {report.selected_flows}/{report.selected_hops}")
        print(f"  average hops: {report.average_hops:.6f}")
        print(f"  maximum directed channel load: {report.maximum_directed_channel_load}")
        print(f"  unit-flow throughput bound: {report.unit_flow_throughput_bound:.8f}")
        if report.candidate_paths is not None:
            print(f"  candidate paths: {report.candidate_paths}")
        if report.virtual_channels is not None:
            print(f"  virtual channels: {report.virtual_channels}")
        for error in report.errors:
            print(f"  ERROR: {error}", file=sys.stderr)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
