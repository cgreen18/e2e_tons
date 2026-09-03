#!/usr/bin/env python3
"""Profile collective types, group sizes, and message sizes in Chakra traces.

Buffer bins are half-open except for the final unbounded bin:
``[0, 4 KiB)``, ``[4, 64 KiB)``, ``[64 KiB, 1 MiB)``,
``[1, 16 MiB)``, ``[16, 256 MiB)``, ``[256 MiB, 1 GiB)``, and
``[1 GiB, infinity)``.  Existing device-kernel collectives retain their trace
``comm_size``.  MoE8x70B has no device collectives, so its known CPU NCCL
launchers are profiled using the in-memory promotion and rounded-size logic
from :mod:`tools.chakra_et_rewrite`; no rewritten traces are written.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import tempfile
from typing import Iterable, Sequence

from tools.chakra_et_rewrite import (
    GROUP_SEARCH_WINDOW,
    TRACE_NAME_RE,
    TraceFormatError,
    inspect_promotable_comm_op,
    load_protobuf_module,
    resolve_collective_group_sizes,
    round_comm_size,
    summarize_group_size_resolutions,
)


DEFAULT_TRACE_ROOT = Path("/home/green456/e2e_tons/ai_traces")
DEFAULT_CSV = Path("generated/chakra_collective_profile.csv")
DEFAULT_JSON = Path("generated/chakra_collective_profile.json")
UNRESOLVED = "UNRESOLVED"

KIB = 1 << 10
MIB = 1 << 20
GIB = 1 << 30
BUFFER_BINS: tuple[tuple[int, int | None, str], ...] = (
    (0, 4 * KIB, "0-4KiB"),
    (4 * KIB, 64 * KIB, "4-64KiB"),
    (64 * KIB, MIB, "64KiB-1MiB"),
    (MIB, 16 * MIB, "1-16MiB"),
    (16 * MIB, 256 * MIB, "16-256MiB"),
    (256 * MIB, GIB, "256MiB-1GiB"),
    (GIB, None, "1GiB+"),
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    directory_name: str
    rank_count: int
    promote_cpu_launchers: bool = False


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec("Llama7B", "Llama7B_N32_GPU128_PP1_DP128_7B_BS128", 128),
    ModelSpec(
        "MoE8x13B", "MoE8x13B_N32_GPU128_TP4_PP4_DP8_EP4_13B_BS128", 128
    ),
    ModelSpec("Llama70B", "Llama70B_N64_GPU256_TP1_PP8_DP32_70B_BS32", 256),
    ModelSpec(
        "MoE8x70B",
        "MoE8x70B_N64_GPU256_TP4_PP8_DP8_EP8_70B_BS128",
        256,
        promote_cpu_launchers=True,
    ),
)


@dataclass(frozen=True)
class _RawCollective:
    node_id: int
    comm_type: str
    size: int
    promoted: bool


def buffer_size_range(size: int) -> str:
    """Return the documented half-open bin containing ``size`` bytes."""

    if size < 0:
        raise ValueError("communication size must be non-negative")
    for lower, upper, label in BUFFER_BINS:
        if size >= lower and (upper is None or size < upper):
            return label
    raise AssertionError(f"no buffer bin for {size}")


def _int64_attr(node, name: str) -> int:
    matches = [attr for attr in node.attr if attr.name == name]
    if len(matches) != 1 or matches[0].WhichOneof("value") != "int64_val":
        raise TraceFormatError(
            f"collective node {node.id} does not have exactly one int64 {name}"
        )
    return int(matches[0].int64_val)


def _comm_type_name(pb, value: int) -> str:
    try:
        return str(pb.CollectiveCommType.Name(value))
    except ValueError as error:
        raise TraceFormatError(f"unknown collective communication type {value}") from error


def _profile_rank(arguments) -> dict[str, object]:
    spec, trace_path, rank, search_window = arguments
    pb = load_protobuf_module()
    raw_collectives: list[_RawCollective] = []

    def target_selector(node) -> bool:
        if node.type == pb.COMM_COLL_NODE:
            raw_collectives.append(
                _RawCollective(
                    node_id=int(node.id),
                    comm_type=_comm_type_name(pb, _int64_attr(node, "comm_type")),
                    size=_int64_attr(node, "comm_size"),
                    promoted=False,
                )
            )
            return True
        if spec.promote_cpu_launchers:
            enum_name, raw_size, _ = inspect_promotable_comm_op(node, pb)
            if enum_name is not None:
                assert raw_size is not None
                raw_collectives.append(
                    _RawCollective(
                        node_id=int(node.id),
                        comm_type=enum_name,
                        size=raw_size,
                        promoted=True,
                    )
                )
                return True
        return False

    resolutions = resolve_collective_group_sizes(
        trace_path,
        spec.rank_count,
        target_selector,
        pb,
        search_window=search_window,
        fallback_to_all_ranks=False,
    )
    aggregates: dict[tuple[str, str, str], list[int]] = {}
    for collective in raw_collectives:
        resolution = resolutions[collective.node_id]
        group: int | str = (
            resolution.group_size if resolution.source == "direct" else UNRESOLVED
        )
        if collective.promoted:
            rounding_group_size = (
                resolution.group_size
                if resolution.group_size is not None
                else spec.rank_count
            )
            size = round_comm_size(collective.size, rounding_group_size)
        else:
            size = collective.size
        key = (collective.comm_type, str(group), buffer_size_range(size))
        stats = aggregates.setdefault(key, [0, 0, size, size])
        stats[0] += 1
        stats[1] += size
        stats[2] = min(stats[2], size)
        stats[3] = max(stats[3], size)

    summary = summarize_group_size_resolutions(resolutions.values())
    distances = Counter(
        resolution.distance
        for resolution in resolutions.values()
        if resolution.source == "direct" and resolution.distance is not None
    )
    groups = Counter(
        str(resolution.group_size)
        if resolution.source == "direct"
        else UNRESOLVED
        for resolution in resolutions.values()
    )
    return {
        "model": spec.name,
        "rank": rank,
        "input_file": str(trace_path.resolve()),
        "collectives": len(raw_collectives),
        "promoted_collectives": sum(item.promoted for item in raw_collectives),
        "resolution": summary,
        "resolved_group_sizes": dict(sorted(groups.items())),
        "direct_resolution_distances": {
            str(distance): count for distance, count in sorted(distances.items())
        },
        "aggregates": [
            {
                "comm_type": key[0],
                "group_size": key[1],
                "buffer_size_range": key[2],
                "op_count": values[0],
                "total_bytes": values[1],
                "min_bytes": values[2],
                "max_bytes": values[3],
            }
            for key, values in sorted(aggregates.items())
        ],
    }


def _validate_model_inputs(
    trace_root: Path, specifications: Sequence[ModelSpec]
) -> list[tuple[ModelSpec, Path, int]]:
    tasks: list[tuple[ModelSpec, Path, int]] = []
    for spec in specifications:
        directory = trace_root / spec.directory_name
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        available = {
            int(match.group(1)): path
            for path in directory.iterdir()
            if (match := TRACE_NAME_RE.fullmatch(path.name)) is not None
        }
        expected = set(range(spec.rank_count))
        if set(available) != expected:
            missing = sorted(expected - set(available))
            extra = sorted(set(available) - expected)
            raise ValueError(
                f"{spec.name}: expected ranks 0..{spec.rank_count - 1}; "
                f"missing={missing[:10]} extra={extra[:10]}"
            )
        tasks.extend((spec, available[rank], rank) for rank in range(spec.rank_count))
    return tasks


def _merge_resolution_summaries(reports: Iterable[dict[str, object]]) -> dict[str, object]:
    reports = list(reports)
    direct = sum(int(report["resolution"]["direct"]) for report in reports)
    fallback = sum(int(report["resolution"]["fallback"]) for report in reports)
    unresolved = sum(int(report["resolution"]["unresolved"]) for report in reports)
    reasons: Counter[str] = Counter()
    distances: Counter[str] = Counter()
    groups: Counter[str] = Counter()
    for report in reports:
        reasons.update(report["resolution"]["reasons"])
        distances.update(report["direct_resolution_distances"])
        groups.update(report["resolved_group_sizes"])
    total = direct + fallback + unresolved
    return {
        "collectives": sum(int(report["collectives"]) for report in reports),
        "promoted_collectives": sum(
            int(report["promoted_collectives"]) for report in reports
        ),
        "direct": direct,
        "fallback": fallback,
        "unresolved": unresolved,
        "hit_rate": direct / total if total else None,
        "reasons": dict(sorted(reasons.items())),
        "resolved_group_sizes": dict(sorted(groups.items())),
        "direct_resolution_distances": dict(sorted(distances.items(), key=lambda x: int(x[0]))),
    }


def profile_models(
    trace_root: Path,
    specifications: Sequence[ModelSpec],
    *,
    jobs: int,
    search_window: int = GROUP_SEARCH_WINDOW,
) -> dict[str, object]:
    """Profile every rank in ``specifications`` and return JSON-ready data."""

    if jobs < 1:
        raise ValueError("jobs must be at least one")
    if search_window < 1:
        raise ValueError("group search window must be at least one")
    trace_root = Path(trace_root)
    tasks = [
        (spec, path, rank, search_window)
        for spec, path, rank in _validate_model_inputs(trace_root, specifications)
    ]
    load_protobuf_module()  # Bootstrap once before workers import concurrently.
    if jobs == 1 or len(tasks) == 1:
        per_rank = [_profile_rank(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=min(jobs, len(tasks))) as executor:
            per_rank = list(executor.map(_profile_rank, tasks, chunksize=1))
    per_rank.sort(key=lambda item: (str(item["model"]), int(item["rank"])))

    merged: dict[tuple[str, str, str, str], list[int]] = {}
    for report in per_rank:
        for aggregate in report["aggregates"]:
            key = (
                str(report["model"]),
                str(aggregate["comm_type"]),
                str(aggregate["group_size"]),
                str(aggregate["buffer_size_range"]),
            )
            values = merged.setdefault(
                key,
                [
                    0,
                    0,
                    int(aggregate["min_bytes"]),
                    int(aggregate["max_bytes"]),
                ],
            )
            values[0] += int(aggregate["op_count"])
            values[1] += int(aggregate["total_bytes"])
            values[2] = min(values[2], int(aggregate["min_bytes"]))
            values[3] = max(values[3], int(aggregate["max_bytes"]))

    rows = [
        {
            "model": key[0],
            "comm_type": key[1],
            "communicator_group_size": (
                UNRESOLVED if key[2] == UNRESOLVED else int(key[2])
            ),
            "buffer_size_range": key[3],
            "op_count": values[0],
            "total_bytes": values[1],
            "min_bytes": values[2],
            "max_bytes": values[3],
            "mean_bytes": values[1] / values[0],
        }
        for key, values in sorted(
            merged.items(),
            key=lambda item: (
                item[0][0],
                item[0][1],
                item[0][2] == UNRESOLVED,
                int(item[0][2]) if item[0][2] != UNRESOLVED else 0,
                next(
                    index
                    for index, (_, _, label) in enumerate(BUFFER_BINS)
                    if label == item[0][3]
                ),
            ),
        )
    ]
    per_model = {
        spec.name: _merge_resolution_summaries(
            report for report in per_rank if report["model"] == spec.name
        )
        for spec in specifications
    }
    return {
        "caption": (
            "All ranks of four original AI-trace models. MoE8x70B rows use "
            "in-memory promote-comm-ops sizing; no trace files were written."
        ),
        "trace_root": str(trace_root.resolve()),
        "group_search_window_node_ids": search_window,
        "buffer_bins": [
            {"label": label, "lower_inclusive": lower, "upper_exclusive": upper}
            for lower, upper, label in BUFFER_BINS
        ],
        "models": [
            {
                "name": spec.name,
                "directory": spec.directory_name,
                "rank_count": spec.rank_count,
                "in_memory_promote_comm_ops": spec.promote_cpu_launchers,
            }
            for spec in specifications
        ],
        "per_model_resolution": per_model,
        "per_rank_resolution": {
            f"{report['model']}:{report['rank']}": {
                key: value
                for key, value in report.items()
                if key not in {"aggregates", "input_file", "model", "rank"}
            }
            for report in per_rank
        },
        "rows": rows,
    }


def markdown_table(profile: dict[str, object]) -> str:
    """Render every profile row as a readable Markdown table."""

    lines = [
        str(profile["caption"]),
        "",
        "| Model | Comm type | Group size | Buffer range | Ops | Total bytes | Min bytes | Max bytes | Mean bytes |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in profile["rows"]:
        lines.append(
            "| {model} | {comm_type} | {communicator_group_size} | "
            "{buffer_size_range} | {op_count:,} | {total_bytes:,} | "
            "{min_bytes:,} | {max_bytes:,} | {mean_bytes:,.2f} |".format(**row)
        )
    return "\n".join(lines)


def _atomic_json(path: Path, contents: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary_name = stream.name
            json.dump(contents, stream, indent=2, sort_keys=True)
            stream.write("\n")
        Path(temporary_name).replace(path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def _atomic_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "comm_type",
        "communicator_group_size",
        "buffer_size_range",
        "op_count",
        "total_bytes",
        "min_bytes",
        "max_bytes",
        "mean_bytes",
    ]
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary_name = stream.name
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        Path(temporary_name).replace(path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def write_profile_outputs(
    profile: dict[str, object], csv_path: Path, json_path: Path
) -> None:
    """Atomically write the complete CSV and JSON profile outputs."""

    _atomic_csv(Path(csv_path), profile["rows"])
    _atomic_json(Path(json_path), profile)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", type=Path, default=DEFAULT_TRACE_ROOT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument(
        "--group-search-window",
        type=int,
        default=GROUP_SEARCH_WINDOW,
        help=f"maximum node-id distance to record_param_comms (default: {GROUP_SEARCH_WINDOW})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    try:
        profile = profile_models(
            arguments.trace_root,
            MODEL_SPECS,
            jobs=arguments.jobs,
            search_window=arguments.group_search_window,
        )
        write_profile_outputs(profile, arguments.output_csv, arguments.output_json)
    except (FileNotFoundError, RuntimeError, TraceFormatError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(markdown_table(profile))
    return 0


if __name__ == "__main__":
    sys.exit(main())
