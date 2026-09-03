#!/usr/bin/env python3
"""Emit the synthetic communicator-aware acceptance trace.

The default shape is the one specified for the initial metric of success:

    compute -> all-reduce over a 64-rank communicator
            -> compute -> all-to-all over the full 128-rank communicator
            -> compute

Process group 1 is the 64-rank subgroup and process group 2 covers every rank.
Group 0 is reserved by ASTRA-sim for the implicit default communicator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tons_collectives.synthetic_trace import (  # noqa: E402
    Collective,
    Communicator,
    Compute,
    generate_synthetic_trace,
)


MIB = 1 << 20


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--ranks", type=int, default=128)
    parser.add_argument(
        "--subgroup-ranks",
        type=int,
        default=64,
        help="size of the all-reduce communicator, taken as ranks [0, N)",
    )
    parser.add_argument("--allreduce-bytes", type=int, default=16 * MIB)
    parser.add_argument("--alltoall-bytes", type=int, default=16 * MIB)
    parser.add_argument(
        "--compute-micros",
        type=int,
        default=100,
        help="replayed duration of each compute stage",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="optional path for a JSON description of the emitted trace",
    )
    args = parser.parse_args(argv)

    if not 1 <= args.subgroup_ranks <= args.ranks:
        parser.error("--subgroup-ranks must be within [1, --ranks]")

    communicators = [
        Communicator("1", tuple(range(args.subgroup_ranks))),
        Communicator("2", tuple(range(args.ranks))),
    ]
    stages = [
        Compute("compute_0", args.compute_micros),
        Collective("allreduce_pg1", "allreduce", "1", args.allreduce_bytes),
        Compute("compute_1", args.compute_micros),
        Collective("alltoall_pg2", "alltoall", "2", args.alltoall_bytes),
        Compute("compute_2", args.compute_micros),
    ]
    paths = generate_synthetic_trace(
        args.output_prefix, args.ranks, stages, communicators
    )

    description = {
        "ranks": args.ranks,
        "workload_prefix": str(Path(args.output_prefix).resolve()),
        "trace_files": len(paths),
        "compute_micros": args.compute_micros,
        "communicators": [
            {"pg_name": c.pg_name, "size": len(c.members), "members": list(c.members)}
            for c in communicators
        ],
        "stages": [
            {"name": s.name, "kind": "compute", "micros": s.micros}
            if isinstance(s, Compute)
            else {
                "name": s.name,
                "kind": "collective",
                "collective": s.collective,
                "pg_name": s.pg_name,
                "size_bytes": s.size_bytes,
            }
            for s in stages
        ],
    }
    if args.manifest is not None:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(description, indent=2) + "\n")
    json.dump(description, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
