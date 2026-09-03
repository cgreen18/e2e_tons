#!/usr/bin/env python3
"""Pre-scan Chakra traces for the communicators that need custom collectives.

Topology-aware collectives are pre-computed per communicator, so the trace must
first be read to discover which communicators exist, which collective types run
on each, and over what byte range.  The scan resolves membership from the
PyTorch process-group registry that ASTRA-sim itself parses
(``Workload::issue_pytorch_pg_metadata``), rather than the ``record_param_comms``
correlation heuristic in :mod:`tools.chakra_collective_profile`.  The registry
is authoritative and complete; the heuristic resolves only a small fraction of
some models' collectives.

Membership is required to agree across every rank that declares a group.  A
disagreement is reported as a blocking error rather than silently reconciled,
because it would make the pre-computed schedule wrong on some ranks.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.chakra_et_rewrite import (  # noqa: E402
    PROCESS_GROUP_METADATA_NAME,
    TraceFormatError,
    discover_inputs,
    infer_model_rank_count,
    iter_nodes,
    load_protobuf_module,
    parse_process_group_registry,
)

# ASTRA-sim maps an absent pg_name and the literal "0" onto the implicit
# default communicator covering every rank.  See [default communication group]
# in Workload.cc.
DEFAULT_PG_NAME = "0"


def _parse_registry_members(values: str, model_rank_count: int) -> dict[str, tuple[int, ...]]:
    """Return pg_name -> member ranks, expanding an empty list to all ranks."""

    if len(values) < 4:
        raise ValueError("process-group metadata is too short for ASTRA trim")
    parsed = json.loads(values[2:-2])
    if not isinstance(parsed, list):
        raise ValueError("process-group metadata must contain a JSON list")
    members: dict[str, tuple[int, ...]] = {}
    for item in parsed:
        pg_id = item.get("pg_name")
        ranks = item.get("ranks")
        if not isinstance(pg_id, str) or not pg_id.isdigit():
            raise ValueError("process-group pg_name must be a decimal string")
        if not isinstance(ranks, list) or not all(
            isinstance(rank, int) and not isinstance(rank, bool) for rank in ranks
        ):
            raise ValueError("process-group ranks must be an integer list")
        if pg_id in members:
            raise ValueError(f"duplicate process-group id {pg_id}")
        members[pg_id] = tuple(sorted(ranks)) if ranks else tuple(range(model_rank_count))
    return members


@dataclass
class Usage:
    """Aggregated statistics for one (communicator, collective type) pair."""

    op_count: int = 0
    total_bytes: int = 0
    min_bytes: int | None = None
    max_bytes: int | None = None
    distinct_sizes: set[int] = field(default_factory=set)

    def observe(self, size: int) -> None:
        self.op_count += 1
        self.total_bytes += size
        self.min_bytes = size if self.min_bytes is None else min(self.min_bytes, size)
        self.max_bytes = size if self.max_bytes is None else max(self.max_bytes, size)
        # Bounded so a pathological trace cannot exhaust memory; the count and
        # min/max stay exact either way.
        if len(self.distinct_sizes) < 4096:
            self.distinct_sizes.add(size)

    def merge(self, other: "Usage") -> None:
        self.op_count += other.op_count
        self.total_bytes += other.total_bytes
        for value in (other.min_bytes,):
            if value is not None:
                self.min_bytes = value if self.min_bytes is None else min(self.min_bytes, value)
        for value in (other.max_bytes,):
            if value is not None:
                self.max_bytes = value if self.max_bytes is None else max(self.max_bytes, value)
        for size in other.distinct_sizes:
            if len(self.distinct_sizes) < 4096:
                self.distinct_sizes.add(size)


def scan_rank(
    rank: int, path: Path, model_rank_count: int
) -> tuple[int, dict[str, tuple[int, ...]], dict[tuple[str, str], Usage], list[str]]:
    """Scan one rank's trace.  Returns membership, usage, and any warnings."""

    pb = load_protobuf_module()
    members: dict[str, tuple[int, ...]] = {}
    usage: dict[tuple[str, str], Usage] = defaultdict(Usage)
    warnings: list[str] = []

    for record in iter_nodes(path, pb):
        node_type = getattr(record, "type", None)
        if node_type is None:
            continue  # GlobalMetadata
        if node_type == pb.METADATA_NODE and record.name == PROCESS_GROUP_METADATA_NAME:
            try:
                found = _parse_registry_members(record.inputs.values, model_rank_count)
            except (ValueError, json.JSONDecodeError) as error:
                warnings.append(f"rank {rank}: unparsable process-group metadata: {error}")
                continue
            for pg_name, ranks in found.items():
                if pg_name in members and members[pg_name] != ranks:
                    warnings.append(
                        f"rank {rank}: process group {pg_name} redeclared with "
                        "different membership within the same trace"
                    )
                members[pg_name] = ranks
            continue
        if node_type != pb.COMM_COLL_NODE:
            continue
        pg_name = DEFAULT_PG_NAME
        comm_type: int | None = None
        size = 0
        for attribute in record.attr:
            if attribute.name == "pg_name":
                pg_name = attribute.string_val or DEFAULT_PG_NAME
            elif attribute.name == "comm_type":
                comm_type = attribute.int64_val or attribute.int32_val
            elif attribute.name == "comm_size":
                size = attribute.int64_val or attribute.int32_val
        if comm_type is None:
            warnings.append(f"rank {rank}: collective node {record.id} has no comm_type")
            continue
        name = pb.CollectiveCommType.Name(comm_type)
        usage[(pg_name, name)].observe(size)
    return rank, members, dict(usage), warnings


def _scan_rank_star(item):
    return scan_rank(*item)


def scan(
    source: Path, model_rank_count: int, jobs: int = 1
) -> dict:
    """Scan every rank and return the communicator plan."""

    inputs = discover_inputs(source, None)
    work = [(rank, path, model_rank_count) for rank, path in inputs]

    if jobs > 1 and len(work) > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            results = list(pool.map(_scan_rank_star, work))
    else:
        results = [_scan_rank_star(item) for item in work]

    membership: dict[str, tuple[int, ...]] = {}
    declared_by: dict[str, list[int]] = defaultdict(list)
    conflicts: list[str] = []
    warnings: list[str] = []
    usage: dict[tuple[str, str], Usage] = defaultdict(Usage)
    used_by: dict[str, set[int]] = defaultdict(set)

    for rank, members, rank_usage, rank_warnings in results:
        warnings.extend(rank_warnings)
        for pg_name, ranks in members.items():
            declared_by[pg_name].append(rank)
            if pg_name in membership:
                if membership[pg_name] != ranks:
                    conflicts.append(
                        f"process group {pg_name}: rank {rank} declares "
                        f"{len(ranks)} members, an earlier rank declared "
                        f"{len(membership[pg_name])}"
                    )
            else:
                membership[pg_name] = ranks
        for key, value in rank_usage.items():
            usage[key].merge(value)
            used_by[key[0]].add(rank)

    membership.setdefault(DEFAULT_PG_NAME, tuple(range(model_rank_count)))

    groups = []
    unresolved = []
    for pg_name in sorted(set(pg for pg, _ in usage) | set(membership), key=int):
        collectives = {
            name: {
                "op_count": value.op_count,
                "total_bytes": value.total_bytes,
                "min_bytes": value.min_bytes,
                "max_bytes": value.max_bytes,
                "distinct_size_count": len(value.distinct_sizes),
                "distinct_sizes": sorted(value.distinct_sizes)
                if len(value.distinct_sizes) <= 32
                else None,
            }
            for (group, name), value in sorted(usage.items())
            if group == pg_name
        }
        if not collectives:
            # Declared but never used; nothing to pre-compute for it.
            continue
        if pg_name not in membership:
            unresolved.append(pg_name)
            continue
        members = membership[pg_name]
        entry = {
            "pg_name": pg_name,
            "size": len(members),
            "members": list(members),
            "signature": signature(members),
            "is_default": pg_name == DEFAULT_PG_NAME,
            "declared_by_rank_count": len(declared_by.get(pg_name, [])),
            "used_by_rank_count": len(used_by.get(pg_name, ())),
            "collectives": collectives,
        }
        groups.append(entry)

    required = sorted(
        {(entry["signature"], name) for entry in groups for name in entry["collectives"]}
    )
    return {
        "source": str(Path(source).resolve()),
        "model_rank_count": model_rank_count,
        "ranks_scanned": len(results),
        "groups": groups,
        "unresolved_pg_names": unresolved,
        "membership_conflicts": conflicts,
        "warnings": warnings,
        "required_schedules": [
            {"signature": sig, "collective": name} for sig, name in required
        ],
        "group_sizes": sorted({entry["size"] for entry in groups}),
    }


def signature(members) -> str:
    """Canonical, stable identity for a communicator's exact member set."""

    members = tuple(sorted(members))
    if not members:
        raise ValueError("communicator has no members")
    contiguous = members == tuple(range(members[0], members[0] + len(members)))
    if contiguous:
        return f"n{len(members)}-r{members[0]}_{members[-1]}"
    import hashlib

    digest = hashlib.sha256(
        ",".join(str(rank) for rank in members).encode("ascii")
    ).hexdigest()[:16]
    return f"n{len(members)}-h{digest}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path,
                        help="trace directory containing chakra.<rank>.et files")
    parser.add_argument("--model-ranks", type=int, default=None,
                        help="total ranks; inferred from the directory when omitted")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--allow-conflicts", action="store_true",
                        help="report membership conflicts without failing")
    args = parser.parse_args(argv)

    model_rank_count = args.model_ranks or infer_model_rank_count(args.source)
    try:
        plan = scan(args.source, model_rank_count, jobs=args.jobs)
    except TraceFormatError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    text = json.dumps(plan, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        sys.stdout.write(text)

    for warning in plan["warnings"][:20]:
        print(f"warning: {warning}", file=sys.stderr)
    if plan["unresolved_pg_names"]:
        print(
            "error: collectives reference process groups with no registry entry: "
            + ", ".join(plan["unresolved_pg_names"]),
            file=sys.stderr,
        )
        return 3
    if plan["membership_conflicts"]:
        for conflict in plan["membership_conflicts"][:20]:
            print(f"error: {conflict}", file=sys.stderr)
        if not args.allow_conflicts:
            return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
