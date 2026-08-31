#!/usr/bin/env python3
"""Generate and verify TONS MSCCL schedules."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tons_collectives import (  # noqa: E402
    generate_direct_alltoall,
    generate_fixed_route_alltoall,
    generate_pmcf_alltoall,
    verify_schedule,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    direct = subparsers.add_parser("direct-a2a")
    direct.add_argument("--ranks", type=int, required=True)
    direct.add_argument("--subchunks", type=int, default=8)
    direct.add_argument("--output", type=Path, required=True)
    pipeline = subparsers.add_parser("pipeline-a2a")
    pipeline.add_argument("--routes", type=Path, required=True)
    pipeline.add_argument("--subchunks", type=int, default=8)
    pipeline.add_argument("--output", type=Path, required=True)
    pmcf = subparsers.add_parser("pmcf-a2a")
    pmcf.add_argument("--topology", type=Path, required=True)
    pmcf.add_argument("--candidates", type=Path, required=True)
    pmcf.add_argument("--subchunks", type=int, default=8)
    pmcf.add_argument("--threads", type=int, default=16)
    pmcf.add_argument("--solver", choices=("highs", "gurobi"), default="highs")
    pmcf.add_argument("--seed", type=int, default=1)
    pmcf.add_argument("--output", type=Path, required=True)
    pmcf.add_argument(
        "--report",
        type=Path,
        default=None,
        help="pMCF JSON report path; defaults to the schedule with a .pmcf.json suffix",
    )
    verify = subparsers.add_parser("verify")
    verify.add_argument("schedule", type=Path)
    args = parser.parse_args(argv)
    if args.command == "direct-a2a":
        output = generate_direct_alltoall(args.ranks, args.subchunks, args.output)
        report = verify_schedule(output)
    elif args.command == "pipeline-a2a":
        output = generate_fixed_route_alltoall(args.routes, args.subchunks, args.output)
        report = verify_schedule(output)
    elif args.command == "pmcf-a2a":
        result = generate_pmcf_alltoall(
            args.topology,
            args.candidates,
            args.subchunks,
            args.output,
            report_path=args.report,
            solver=args.solver,
            threads=args.threads,
            seed=args.seed,
        )
        print(json.dumps(asdict(result), indent=2, sort_keys=True, default=str))
        report = verify_schedule(result.schedule)
    else:
        report = verify_schedule(args.schedule)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
