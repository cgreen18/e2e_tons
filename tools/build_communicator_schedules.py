#!/usr/bin/env python3
"""Pre-compute topology-aware collective schedules for a trace's communicators.

Reads a communicator plan from ``tools/chakra_comm_groups.py`` (or an explicit
``--group`` spec), synthesizes one schedule per (communicator, collective),
verifies it, lowers it to Chakra ETs, and writes the YAML that ASTRA-sim's
``per-communicator-custom-implementation`` key consumes.

Schedules are shared between communicators only when sharing is provably safe.
For the tree collectives that means an identical reduced adjacency matrix: the
matrix is the entire topology-aware input to ``l3ss_tree``, so two groups with
the same matrix necessarily receive the same algorithm.  All-to-all is never
shared, because its cost depends on physical link sharing rather than on
adjacency alone.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tons_collectives.chakra import lower_msccl_to_chakra  # noqa: E402
from tons_collectives.a2a import generate_direct_alltoall  # noqa: E402
from tons_collectives.pmcf import _read_map, generate_pmcf_alltoall  # noqa: E402
from tons_collectives.subgroup import (  # noqa: E402
    SUBMAP_MODES,
    generate_broadcast,
    generate_gather,
    member_submap,
    write_map,
)
from tons_collectives.verify import verify_schedule  # noqa: E402

# Chakra collective name -> the l3ss_tree sub-command that synthesizes it.
TREE_COLLECTIVES = {
    "ALL_GATHER": "ag",
    "ALL_REDUCE": "ar",
    "REDUCE_SCATTER": "rs",
}
STRUCTURAL_COLLECTIVES = {"BROADCAST", "REDUCE"}


def _run_l3ss(binary: Path, topology: Path, kind: str, chunks: int, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [str(binary), "--xml", str(output), "build", str(topology), kind, "rr", str(chunks)]
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode:
        raise RuntimeError(
            f"l3ss_tree failed for {kind} on {topology}:\n"
            f"{completed.stdout}\n{completed.stderr}"
        )


def _finalize(xml: Path, prefix: Path, reports: Path, label: str) -> Path:
    report = verify_schedule(xml)
    reports.mkdir(parents=True, exist_ok=True)
    (reports / f"{label}.schedule.json").write_text(
        json.dumps(report.to_dict(), indent=2) + "\n"
    )
    if not report.ok:
        raise RuntimeError(f"invalid schedule for {label}: {report.errors}")
    lower_msccl_to_chakra(xml, prefix)
    return prefix


def build(
    plan: dict,
    topology_map: Path,
    output_root: Path,
    *,
    l3ss_binary: Path,
    subchunks: int,
    submap_mode: str,
    candidates_path: Path | None,
    alltoall_mode: str,
    solver: str,
    seed: int,
) -> dict:
    adjacency = _read_map(topology_map)
    schedules_dir = output_root / "schedules"
    maps_dir = output_root / "submaps"
    reports_dir = output_root / "reports"

    # Reuse one generated schedule across every communicator whose reduced
    # topology is identical.
    tree_cache: dict[tuple[str, tuple], Path] = {}
    structural_cache: dict[tuple[str, int], Path] = {}
    entries: dict[str, list[dict]] = defaultdict(list)
    records: list[dict] = []

    for group in plan["groups"]:
        members = list(group["members"])
        signature = group["signature"]
        for collective in sorted(group["collectives"]):
            if collective in TREE_COLLECTIVES:
                matrix, radius = member_submap(adjacency, members, submap_mode)
                key = (collective, tuple(tuple(row) for row in matrix))
                prefix = tree_cache.get(key)
                shared = prefix is not None
                if prefix is None:
                    label = f"{collective.lower()}.{signature}"
                    submap = write_map(matrix, maps_dir / f"{signature}.map")
                    xml = schedules_dir / f"{label}.xml"
                    _run_l3ss(
                        l3ss_binary, submap, TREE_COLLECTIVES[collective],
                        subchunks, xml,
                    )
                    prefix = _finalize(
                        xml, schedules_dir / label, reports_dir, label
                    )
                    tree_cache[key] = prefix
                records.append({
                    "signature": signature, "collective": collective,
                    "members": members, "algorithm": str(prefix.resolve()),
                    "shared": shared, "submap_radius": radius,
                })
            elif collective in STRUCTURAL_COLLECTIVES:
                # Trivial send/receive equivalents: a broadcast is one-to-all
                # and a reduce is lowered to a gather with no reduction
                # compute, so both depend only on the member count.
                key = (collective, len(members))
                prefix = structural_cache.get(key)
                shared = prefix is not None
                if prefix is None:
                    label = f"{collective.lower()}.n{len(members)}"
                    xml = schedules_dir / f"{label}.xml"
                    generator = (
                        generate_broadcast
                        if collective == "BROADCAST"
                        else generate_gather
                    )
                    generator(len(members), xml)
                    prefix = _finalize(
                        xml, schedules_dir / label, reports_dir, label
                    )
                    structural_cache[key] = prefix
                records.append({
                    "signature": signature, "collective": collective,
                    "members": members, "algorithm": str(prefix.resolve()),
                    "shared": shared, "submap_radius": None,
                })
            elif collective == "ALL_TO_ALL":
                label = f"alltoall-{alltoall_mode}.{signature}"
                xml = schedules_dir / f"{label}.xml"
                if alltoall_mode == "direct":
                    generate_direct_alltoall(len(members), subchunks, xml)
                elif alltoall_mode == "pmcf":
                    if candidates_path is None:
                        raise ValueError("pMCF all-to-all requires --candidates")
                    generate_pmcf_alltoall(
                        topology_map, candidates_path, subchunks, xml,
                        solver=solver, seed=seed,
                        members=None if len(members) == len(adjacency) else members,
                    )
                else:
                    raise ValueError(f"unknown all-to-all mode {alltoall_mode!r}")
                prefix = _finalize(xml, schedules_dir / label, reports_dir, label)
                records.append({
                    "signature": signature, "collective": collective,
                    "members": members, "algorithm": str(prefix.resolve()),
                    "shared": False, "submap_radius": None,
                })
            else:
                raise ValueError(
                    f"no schedule generator for collective {collective!r}"
                )

    for record in records:
        entries[record["collective"]].append(
            {"ranks": record["members"], "algorithm": record["algorithm"]}
        )
    return {
        "topology_map": str(Path(topology_map).resolve()),
        "submap_mode": submap_mode,
        "subchunks": subchunks,
        "alltoall_mode": alltoall_mode,
        "records": records,
        "distinct_tree_schedules": len(tree_cache),
        "distinct_structural_schedules": len(structural_cache),
        "yaml_entries": {name: entries[name] for name in sorted(entries)},
    }


def write_yaml(result: dict, path: Path) -> Path:
    """Emit the per-communicator lookup table ASTRA-sim reads."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Generated by tools/build_communicator_schedules.py.",
        "# Maps an exact communicator membership to its pre-computed",
        "# topology-aware collective algorithm.",
    ]
    for collective, groups in result["yaml_entries"].items():
        lines.append(f"{collective}:")
        lines.append("  groups:")
        for group in groups:
            ranks = ", ".join(str(rank) for rank in group["ranks"])
            lines.append(f"    - ranks: [{ranks}]")
            lines.append(f'      algorithm: "{group["algorithm"]}"')
    path.write_text("\n".join(lines) + "\n")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path,
                        help="communicator plan from tools/chakra_comm_groups.py")
    parser.add_argument("--topology-map", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--l3ss-binary", type=Path,
                        default=Path("coll_synth/l3ss_tree/target/release/l3ss_tree"))
    parser.add_argument("--subchunks", type=int, default=8)
    parser.add_argument("--submap-mode", choices=SUBMAP_MODES, default="proximity")
    parser.add_argument("--candidates", type=Path, default=None)
    parser.add_argument("--alltoall-mode", choices=("direct", "pmcf"), default="direct")
    parser.add_argument("--solver", default="highs")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--yaml", type=Path, default=None,
                        help="where to write the ASTRA lookup table")
    args = parser.parse_args(argv)

    plan = json.loads(args.plan.read_text())
    result = build(
        plan, args.topology_map, args.output_root,
        l3ss_binary=args.l3ss_binary, subchunks=args.subchunks,
        submap_mode=args.submap_mode, candidates_path=args.candidates,
        alltoall_mode=args.alltoall_mode, solver=args.solver, seed=args.seed,
    )
    yaml_path = args.yaml or (args.output_root / "communicator_schedules.yml")
    write_yaml(result, yaml_path)
    result["yaml"] = str(yaml_path.resolve())
    (args.output_root / "communicator_schedules.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    shared = sum(1 for record in result["records"] if record["shared"])
    print(
        f"{len(result['records'])} (communicator, collective) pairs -> "
        f"{result['distinct_tree_schedules']} tree + "
        f"{result['distinct_structural_schedules']} structural distinct schedules "
        f"({shared} reused)"
    )
    print(f"lookup table: {yaml_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
