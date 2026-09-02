"""Prepare, execute, and analyze the 128-rank Chakra trace experiment."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import subprocess
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

from .pipeline import _dump, _load, _repo


EXPECTED_MODELS = ("llama7b", "moe8x13b")
EXPECTED_TOPOLOGIES = ("pt", "pdtt", "tons")
EXPECTED_BACKENDS = ("congestion_unaware", "congestion_aware")
EXPECTED_RANKS = 128
NATIVE_RING_KEYS = (
    "all-reduce-implementation",
    "all-gather-implementation",
    "reduce-scatter-implementation",
    "all-to-all-implementation",
)
TOPOLOGY_COLORS = {
    "pt": "#4c78a8",
    "pdtt": "#f58518",
    "tons": "#54a24b",
}


def _require_mapping(value: object, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def _resolve(repo: Path, value: object, description: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{description} must be a non-empty path string")
    return (repo / value).resolve()


def _read_json_object(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{description} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{description} is not valid JSON: {path}") from exc
    return _require_mapping(value, description)


def _validate_trace_prefix(prefix: Path, ranks: int, model: str) -> None:
    missing: list[int] = []
    unreadable: list[int] = []
    empty: list[int] = []
    for rank in range(ranks):
        trace = Path(f"{prefix}.{rank}.et")
        if not trace.is_file():
            missing.append(rank)
        elif not os.access(trace, os.R_OK):
            unreadable.append(rank)
        elif trace.stat().st_size == 0:
            empty.append(rank)
    problems: list[str] = []
    if missing:
        problems.append("missing ranks " + ",".join(map(str, missing)))
    if unreadable:
        problems.append("unreadable ranks " + ",".join(map(str, unreadable)))
    if empty:
        problems.append("empty ranks " + ",".join(map(str, empty)))
    if problems:
        raise ValueError(
            f"trace prefix for {model} is incomplete ({'; '.join(problems)}): {prefix}"
        )


def _validate_graph_network(path: Path, ranks: int, topology: str) -> None:
    if not path.is_file():
        raise ValueError(f"network configuration for {topology} does not exist: {path}")
    content = path.read_text(encoding="utf-8")
    if not re.search(r"(?m)^\s*topology:\s*\[\s*Graph\s*\]\s*$", content):
        raise ValueError(f"network configuration for {topology} is not a Graph config: {path}")
    count = re.search(r"(?m)^\s*npus_count:\s*\[\s*(\d+)\s*\]", content)
    if count is None or int(count.group(1)) != ranks:
        actual = count.group(1) if count else "missing"
        raise ValueError(
            f"network configuration for {topology} has npus_count={actual}, expected {ranks}: {path}"
        )


def _validate_manifest(manifest_path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Validate every immutable input before prepare writes an artifact."""

    if manifest.get("schema_version") != 1:
        raise ValueError("trace manifest schema_version must be 1")
    repo = _repo(manifest_path, manifest)
    ranks = manifest.get("ranks")
    if ranks != EXPECTED_RANKS:
        raise ValueError(f"ranks must be exactly {EXPECTED_RANKS}, got {ranks!r}")
    seed = manifest.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")

    output = _resolve(repo, manifest.get("output_directory"), "output_directory")
    trace_root = _resolve(repo, manifest.get("trace_root"), "trace_root")
    if not trace_root.is_dir():
        raise ValueError(f"trace_root does not exist: {trace_root}")
    if trace_root == (repo / "ai_traces").resolve():
        raise ValueError("trace_root must point at repaired traces, not ai_traces directly")

    model_defs = _require_mapping(manifest.get("models"), "models")
    if tuple(model_defs) != EXPECTED_MODELS:
        raise ValueError(
            f"models must be ordered exactly as {EXPECTED_MODELS}, got {tuple(model_defs)}"
        )
    models: dict[str, dict[str, Any]] = {}
    for model, raw_definition in model_defs.items():
        definition = _require_mapping(raw_definition, f"model {model}")
        directory = definition.get("directory")
        prefix_name = definition.get("workload_prefix", "chakra")
        if not isinstance(directory, str) or not directory or Path(directory).is_absolute():
            raise ValueError(f"model {model} directory must be a non-empty relative path")
        if prefix_name != "chakra":
            raise ValueError(f"model {model} workload_prefix must be 'chakra'")
        model_dir = (trace_root / directory).resolve()
        try:
            model_dir.relative_to(trace_root)
        except ValueError as exc:
            raise ValueError(f"model {model} directory escapes trace_root") from exc
        prefix = model_dir / prefix_name
        _validate_trace_prefix(prefix, ranks, model)
        models[model] = {
            "directory": directory,
            "resolved_directory": str(model_dir),
            "workload_prefix": str(prefix),
        }

    topology_defs = _require_mapping(manifest.get("topologies"), "topologies")
    if tuple(topology_defs) != EXPECTED_TOPOLOGIES:
        raise ValueError(
            "topologies must be ordered exactly as "
            f"{EXPECTED_TOPOLOGIES}, got {tuple(topology_defs)}"
        )
    topologies: dict[str, dict[str, Any]] = {}
    for topology, raw_definition in topology_defs.items():
        definition = _require_mapping(raw_definition, f"topology {topology}")
        network = _resolve(
            repo,
            definition.get("network_configuration"),
            f"network_configuration for {topology}",
        )
        _validate_graph_network(network, ranks, topology)
        label = definition.get("label")
        bundle = definition.get("bundle")
        if not isinstance(label, str) or not label:
            raise ValueError(f"topology {topology} label must be a non-empty string")
        if not isinstance(bundle, str) or not bundle:
            raise ValueError(f"topology {topology} bundle must be a non-empty string")
        topologies[topology] = {
            "label": label,
            "bundle": bundle,
            "network_configuration": str(network),
        }

    backends = manifest.get("backends")
    if backends != list(EXPECTED_BACKENDS):
        raise ValueError(
            f"backends must be {list(EXPECTED_BACKENDS)}, got {backends!r}"
        )
    binary_defs = _require_mapping(manifest.get("binaries"), "binaries")
    if tuple(binary_defs) != EXPECTED_BACKENDS:
        raise ValueError(
            f"binaries must define {EXPECTED_BACKENDS}, got {tuple(binary_defs)}"
        )
    binaries = {
        backend: str(_resolve(repo, binary_defs[backend], f"binary for {backend}"))
        for backend in EXPECTED_BACKENDS
    }

    remote_memory = _resolve(
        repo,
        manifest.get("remote_memory_configuration"),
        "remote_memory_configuration",
    )
    _read_json_object(remote_memory, "remote_memory_configuration")

    comm_group = manifest.get("comm_group_configuration", "empty")
    if comm_group != "empty":
        raise ValueError(
            "comm_group_configuration must be the literal 'empty'; trace metadata supplies groups"
        )
    system_policy = _require_mapping(
        manifest.get("system_configuration_policy"),
        "system_configuration_policy",
    )
    for key in NATIVE_RING_KEYS:
        if system_policy.get(key) != ["ring"]:
            raise ValueError(f"system_configuration_policy {key} must be ['ring']")
    if any(key.endswith("-implementation-custom") for key in system_policy):
        raise ValueError("system_configuration_policy must use native ring, not custom schedules")

    return {
        "repository_root": str(repo),
        "output_directory": str(output),
        "trace_root": str(trace_root),
        "models": models,
        "topologies": topologies,
        "backends": list(EXPECTED_BACKENDS),
        "binaries": binaries,
        "remote_memory_configuration": str(remote_memory),
        "comm_group_configuration": "empty",
        "system_configuration_policy": dict(system_policy),
        "ranks": ranks,
        "seed": seed,
    }


def prepare(manifest_file: Path | str) -> Path:
    """Validate trace/config inputs and write the fully expanded 12-job manifest."""

    manifest_path, manifest = _load(manifest_file)
    expanded = _validate_manifest(manifest_path, manifest)
    output = Path(expanded["output_directory"])
    prepared_dir = output / "prepared"
    system_path = prepared_dir / "system-native-ring.json"
    _dump(system_path, expanded["system_configuration_policy"])

    jobs: list[dict[str, Any]] = []
    for model, model_definition in expanded["models"].items():
        for topology, topology_definition in expanded["topologies"].items():
            for backend in expanded["backends"]:
                jobs.append(
                    {
                        "run_id": f"{model}__{topology}__{backend}",
                        "model": model,
                        "model_directory": model_definition["directory"],
                        "topology": topology,
                        "topology_label": topology_definition["label"],
                        "topology_bundle": topology_definition["bundle"],
                        "backend": backend,
                        "ranks": expanded["ranks"],
                        "seed": expanded["seed"],
                        "workload": model_definition["workload_prefix"],
                        "network": topology_definition["network_configuration"],
                        "system": str(system_path.resolve()),
                        "remote_memory": expanded["remote_memory_configuration"],
                        "comm_group_configuration": "empty (ASTRA CLI default; omitted)",
                        "collective_algorithms": {
                            key: ["ring"] for key in NATIVE_RING_KEYS
                        },
                    }
                )

    prepared = {
        "schema_version": 1,
        "experiment": "chakra-traces-128",
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        **expanded,
        "model_order": list(expanded["models"]),
        "topology_order": list(expanded["topologies"]),
        "backend_order": list(expanded["backends"]),
        "system_configuration": str(system_path.resolve()),
        "jobs": jobs,
    }
    prepared_path = prepared_dir / "prepared.json"
    _dump(prepared_path, prepared)
    return prepared_path


def _revision(directory: Path) -> str:
    """Return the owning Git revision without misreporting an empty submodule."""

    if not directory.is_dir() or not any(directory.iterdir()):
        return "unavailable"
    result = subprocess.run(
        ["git", "-C", str(directory), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _validate_statistics(stats: object, ranks: int, run_id: str) -> dict[str, Any]:
    value = _require_mapping(stats, f"statistics for {run_id}")
    rank_rows = value.get("ranks")
    if not isinstance(rank_rows, list):
        raise RuntimeError(f"{run_id} statistics.ranks is not a list")
    rank_ids: list[int] = []
    incomplete: list[int] = []
    for index, rank in enumerate(rank_rows):
        if not isinstance(rank, dict) or not isinstance(rank.get("rank"), int):
            raise RuntimeError(f"{run_id} has an invalid rank entry at index {index}")
        rank_id = rank["rank"]
        rank_ids.append(rank_id)
        if rank.get("complete") is not True:
            incomplete.append(rank_id)
    expected = set(range(ranks))
    actual = set(rank_ids)
    if len(rank_rows) != ranks or len(actual) != ranks or actual != expected:
        missing = sorted(expected - actual)
        duplicates = sorted({rank for rank in rank_ids if rank_ids.count(rank) > 1})
        unexpected = sorted(actual - expected)
        raise RuntimeError(
            f"{run_id} incomplete rank set: {len(actual)}/{ranks} unique; "
            f"missing={missing}, duplicates={duplicates}, unexpected={unexpected}"
        )
    if value.get("complete") is not True or incomplete:
        raise RuntimeError(f"{run_id} incomplete ranks: {incomplete or 'top-level complete=false'}")
    return value


def _preflight_run(prepared: dict[str, Any], jobs: list[dict[str, Any]], dry_run: bool) -> None:
    remote_memory = Path(prepared["remote_memory_configuration"])
    _read_json_object(remote_memory, "remote_memory_configuration")
    ranks = int(prepared["ranks"])
    for job in jobs:
        for description, field in (("system", "system"), ("network", "network")):
            path = Path(job[field])
            if not path.is_file():
                raise RuntimeError(f"{job['run_id']} {description} file does not exist: {path}")
        _validate_trace_prefix(Path(job["workload"]), ranks, str(job["model"]))
        binary = Path(prepared["binaries"][job["backend"]])
        if not dry_run and (not binary.is_file() or not os.access(binary, os.X_OK)):
            raise RuntimeError(f"simulator binary is missing or not executable: {binary}")


def run(
    prepared_file: Path | str,
    *,
    dry_run: bool = False,
    limit: int | None = None,
) -> Path:
    """Execute jobs in order, recording provenance and blocking on any rank failure."""

    _, prepared = _load(prepared_file)
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    jobs = prepared["jobs"][:limit] if limit is not None else prepared["jobs"]
    _preflight_run(prepared, jobs, dry_run)

    repo = Path(prepared["repository_root"])
    output = Path(prepared["output_directory"])
    run_root = output / "runs"
    revisions = {
        "repository": _revision(repo),
        "astra_sim": _revision(repo / "simul" / "astra-sim"),
        "analytical_backend": _revision(
            repo / "simul" / "astra-sim" / "extern" / "network_backend" / "analytical"
        ),
        "chakra": _revision(
            repo / "simul" / "astra-sim" / "extern" / "graph_frontend" / "chakra"
        ),
    }
    for job in jobs:
        run_dir = run_root / job["run_id"]
        stats_path = run_dir / "statistics.json"
        stdout_path = run_dir / "stdout.log"
        stderr_path = run_dir / "stderr.log"
        binary = Path(prepared["binaries"][job["backend"]])
        command = [
            str(binary),
            f"--workload-configuration={job['workload']}",
            f"--system-configuration={job['system']}",
            f"--remote-memory-configuration={job['remote_memory']}",
            f"--network-configuration={job['network']}",
            f"--statistics-output={stats_path}",
        ]
        record = {
            **job,
            "command": command,
            "revisions": revisions,
            "statistics": str(stats_path),
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "dry_run": dry_run,
            "returncode": None,
        }
        _dump(run_dir / "run.json", record)
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        if dry_run:
            continue
        completed = subprocess.run(command, cwd=repo, text=True, capture_output=True)
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        record["returncode"] = completed.returncode
        _dump(run_dir / "run.json", record)
        if completed.returncode != 0:
            raise RuntimeError(
                f"{job['run_id']} failed with exit code {completed.returncode}; see {stderr_path}"
            )
        if not stats_path.is_file():
            raise RuntimeError(f"{job['run_id']} did not emit structured statistics")
        try:
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"{job['run_id']} emitted invalid structured statistics") from exc
        _validate_statistics(stats, int(job["ranks"]), job["run_id"])
    return run_root


def _nonnegative_int(value: object, description: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RuntimeError(f"{description} must be a non-negative integer")
    return value


def _mean(values: list[int]) -> float:
    return float(sum(values)) / len(values)


def _normalize_run(
    job: dict[str, Any], stats: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, list[float | int]]]:
    rank_rows = sorted(stats["ranks"], key=lambda rank: rank["rank"])
    wall = [_nonnegative_int(rank.get("wall_time_ns"), "wall_time_ns") for rank in rank_rows]
    communication = [
        _nonnegative_int(rank.get("communication_time_ns"), "communication_time_ns")
        for rank in rank_rows
    ]
    exposed = [
        _nonnegative_int(
            rank.get("exposed_communication_time_ns"), "exposed_communication_time_ns"
        )
        for rank in rank_rows
    ]
    compute = [
        _nonnegative_int(rank.get("compute_time_ns"), "compute_time_ns")
        for rank in rank_rows
    ]
    overlap = [
        _nonnegative_int(rank.get("overlap_time_ns"), "overlap_time_ns")
        for rank in rank_rows
    ]
    simulation_end = _nonnegative_int(stats.get("simulation_end_ns"), "simulation_end_ns")

    collective_arrays_available = all("collectives" in rank for rank in rank_rows)
    collective_rows: list[dict[str, Any]] = []
    if collective_arrays_available:
        totals: dict[str, dict[str, int]] = defaultdict(
            lambda: {"count": 0, "bytes": 0, "time_ns": 0}
        )
        for rank in rank_rows:
            entries = rank["collectives"]
            if not isinstance(entries, list):
                raise RuntimeError(f"{job['run_id']} rank {rank['rank']} collectives is not a list")
            seen: set[str] = set()
            for entry in entries:
                if not isinstance(entry, dict) or not isinstance(entry.get("comm_type"), str):
                    raise RuntimeError(
                        f"{job['run_id']} rank {rank['rank']} has an invalid collective entry"
                    )
                comm_type = entry["comm_type"]
                if comm_type in seen:
                    raise RuntimeError(
                        f"{job['run_id']} rank {rank['rank']} repeats collective {comm_type}"
                    )
                seen.add(comm_type)
                totals[comm_type]["count"] += _nonnegative_int(
                    entry.get("count"), f"{comm_type}.count"
                )
                totals[comm_type]["bytes"] += _nonnegative_int(
                    entry.get("bytes"), f"{comm_type}.bytes"
                )
                totals[comm_type]["time_ns"] += _nonnegative_int(
                    entry.get("time_ns"), f"{comm_type}.time_ns"
                )
        for comm_type in sorted(totals):
            total = totals[comm_type]
            collective_rows.append(
                {
                    "run_id": job["run_id"],
                    "model": job["model"],
                    "backend": job["backend"],
                    "topology": job["topology"],
                    "topology_label": job["topology_label"],
                    "comm_type": comm_type,
                    "rank_reduction": "sum; mean divides by all 128 ranks",
                    "count_sum": total["count"],
                    "bytes_sum": total["bytes"],
                    "time_ns_sum": total["time_ns"],
                    "time_ns_mean_per_rank": total["time_ns"] / len(rank_rows),
                }
            )

    links = stats.get("links", [])
    if not isinstance(links, list):
        raise RuntimeError(f"{job['run_id']} statistics.links is not a list")
    utilization = [float(link.get("utilization", 0.0)) for link in links]
    queue_wait = [
        _nonnegative_int(link.get("total_queue_wait_ns", 0), "link total_queue_wait_ns")
        for link in links
    ]
    queued_chunks = [
        _nonnegative_int(link.get("queued_chunks", 0), "link queued_chunks")
        for link in links
    ]
    max_queue_wait = [
        _nonnegative_int(link.get("max_queue_wait_ns", 0), "link max_queue_wait_ns")
        for link in links
    ]
    maximum_queue_depth = [
        _nonnegative_int(link.get("maximum_queue_depth", 0), "link maximum_queue_depth")
        for link in links
    ]
    if any(not math.isfinite(value) or value < 0 for value in utilization):
        raise RuntimeError(f"{job['run_id']} has invalid link utilization")

    row = {
        "run_id": job["run_id"],
        "model": job["model"],
        "model_directory": job["model_directory"],
        "topology": job["topology"],
        "topology_label": job["topology_label"],
        "topology_bundle": job["topology_bundle"],
        "backend": job["backend"],
        "ranks": job["ranks"],
        "seed": job["seed"],
        "workload": job["workload"],
        "network": job["network"],
        "system": job["system"],
        "simulation_end_ns": simulation_end,
        "rank_wall_time_ns_min": min(wall),
        "rank_wall_time_ns_mean": _mean(wall),
        "rank_wall_time_ns_max": max(wall),
        "communication_time_ns_sum": sum(communication),
        "communication_time_ns_mean_per_rank": _mean(communication),
        "exposed_communication_time_ns_sum": sum(exposed),
        "exposed_communication_time_ns_mean_per_rank": _mean(exposed),
        "compute_time_ns_mean_per_rank": _mean(compute),
        "overlap_time_ns_mean_per_rank": _mean(overlap),
        "collectives_available": collective_arrays_available,
        "collectives_unavailable_reason": (
            "" if collective_arrays_available else "ranks[].collectives absent in older statistics"
        ),
        "directed_link_count": len(links),
        "link_utilization_mean": (
            sum(utilization) / len(utilization) if utilization else 0.0
        ),
        "link_utilization_max": max(utilization, default=0.0),
        "queued_chunks_sum": sum(queued_chunks),
        "total_queue_wait_ns_sum": sum(queue_wait),
        "max_queue_wait_ns": max(max_queue_wait, default=0),
        "maximum_queue_depth": max(maximum_queue_depth, default=0),
    }
    return row, collective_rows, {"utilization": utilization, "queue_wait_ns": queue_wait}


def _speedup_record(
    *,
    family: str,
    metric: str,
    collective_type: str | None,
    model: str,
    backend: str,
    topology: str,
    topology_label: str,
    reduction: str,
    baseline_value: float | int | None,
    topology_value: float | int | None,
    baseline_run_id: str,
    run_id: str,
    reason: str = "",
) -> dict[str, Any]:
    available = (
        baseline_value is not None
        and topology_value is not None
        and float(topology_value) > 0.0
    )
    return {
        "family": family,
        "metric": metric,
        "collective_type": collective_type,
        "model": model,
        "backend": backend,
        "topology": topology,
        "topology_label": topology_label,
        "reference_topology": "pt",
        "reduction": reduction,
        "baseline_value_ns": baseline_value,
        "topology_value_ns": topology_value,
        "unit": "x",
        "speedup": (
            float(baseline_value) / float(topology_value) if available else None
        ),
        "available": available,
        "reason": "" if available else reason or "metric is absent or zero",
        "baseline_run_id": baseline_run_id,
        "run_id": run_id,
    }


def _build_speedups(
    runs: list[dict[str, Any]],
    collectives: list[dict[str, Any]],
    models: list[str],
    backends: list[str],
    topologies: list[str],
) -> list[dict[str, Any]]:
    run_index = {(row["model"], row["backend"], row["topology"]): row for row in runs}
    collective_index = {
        (row["model"], row["backend"], row["topology"], row["comm_type"]): row
        for row in collectives
    }
    speedups: list[dict[str, Any]] = []
    for model in models:
        for backend in backends:
            baseline = run_index[(model, backend, "pt")]
            comm_types = sorted(
                {
                    row["comm_type"]
                    for row in collectives
                    if row["model"] == model and row["backend"] == backend
                }
            )
            if not comm_types:
                comm_types_or_placeholder: list[str | None] = [None]
            else:
                comm_types_or_placeholder = list(comm_types)
            for topology in topologies:
                candidate = run_index[(model, backend, topology)]
                for comm_type in comm_types_or_placeholder:
                    baseline_collective = (
                        collective_index.get((model, backend, "pt", comm_type))
                        if comm_type is not None
                        else None
                    )
                    candidate_collective = (
                        collective_index.get((model, backend, topology, comm_type))
                        if comm_type is not None
                        else None
                    )
                    reason = ""
                    if not baseline["collectives_available"] or not candidate["collectives_available"]:
                        reason = "ranks[].collectives absent in older statistics"
                    elif baseline_collective is None or candidate_collective is None:
                        reason = "collective type was not executed on both topologies"
                    speedups.append(
                        _speedup_record(
                            family="collective",
                            metric="collective_time_ns",
                            collective_type=comm_type,
                            model=model,
                            backend=backend,
                            topology=topology,
                            topology_label=candidate["topology_label"],
                            reduction="mean across all ranks of per-rank summed busy duration",
                            baseline_value=(
                                baseline_collective["time_ns_mean_per_rank"]
                                if baseline_collective else None
                            ),
                            topology_value=(
                                candidate_collective["time_ns_mean_per_rank"]
                                if candidate_collective else None
                            ),
                            baseline_run_id=baseline["run_id"],
                            run_id=candidate["run_id"],
                            reason=reason,
                        )
                    )

                for metric in (
                    "communication_time_ns_mean_per_rank",
                    "exposed_communication_time_ns_mean_per_rank",
                ):
                    speedups.append(
                        _speedup_record(
                            family="communication",
                            metric=metric,
                            collective_type=None,
                            model=model,
                            backend=backend,
                            topology=topology,
                            topology_label=candidate["topology_label"],
                            reduction="mean of rank-local interval-union time across all ranks",
                            baseline_value=baseline[metric],
                            topology_value=candidate[metric],
                            baseline_run_id=baseline["run_id"],
                            run_id=candidate["run_id"],
                        )
                    )
                speedups.append(
                    _speedup_record(
                        family="end_to_end",
                        metric="simulation_end_ns",
                        collective_type=None,
                        model=model,
                        backend=backend,
                        topology=topology,
                        topology_label=candidate["topology_label"],
                        reduction="global simulation makespan reported by ASTRA",
                        baseline_value=baseline["simulation_end_ns"],
                        topology_value=candidate["simulation_end_ns"],
                        baseline_run_id=baseline["run_id"],
                        run_id=candidate["run_id"],
                    )
                )
    return speedups


def _plot_bars(axis: Any, categories: list[str], rows: list[dict[str, Any]], topologies: list[str]) -> None:
    width = 0.24
    center = (len(topologies) - 1) / 2
    labels = {row["topology"]: row["topology_label"] for row in rows}
    for offset, topology in enumerate(topologies):
        indexed = {
            row["plot_category"]: row
            for row in rows
            if row["topology"] == topology and row["available"]
        }
        values = [indexed.get(category, {}).get("speedup", math.nan) for category in categories]
        positions = [index + (offset - center) * width for index in range(len(categories))]
        axis.bar(
            positions,
            values,
            width=width,
            label=labels.get(topology, topology.upper()),
            color=TOPOLOGY_COLORS.get(topology),
        )
    axis.set_xticks(range(len(categories)))
    axis.set_xticklabels(categories)
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
    axis.grid(True, axis="y", alpha=0.25)


def _plots(
    summary: dict[str, Any],
    link_samples: dict[str, dict[str, list[float | int]]],
    output: Path,
) -> list[str]:
    matplotlib_config = output / ".matplotlib"
    matplotlib_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config))
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for trace experiment analysis") from exc

    plot_dir = output / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    models = summary["models"]
    backends = summary["backends"]
    topologies = summary["topologies"]
    topology_labels = summary["topology_labels"]
    topology_handles = [
        Patch(color=TOPOLOGY_COLORS.get(topology), label=topology_labels[topology])
        for topology in topologies
    ]
    scope = f"{', '.join(models)}; " + ", ".join(
        topology_labels[topology] for topology in topologies
    )

    collective_rows = [row.copy() for row in summary["speedups"] if row["family"] == "collective"]
    comm_types = sorted(
        {row["collective_type"] for row in collective_rows if row["collective_type"] is not None}
    )
    categories = [f"{model}\n{comm_type}" for model in models for comm_type in comm_types]
    fig, axes = plt.subplots(
        len(backends),
        1,
        figsize=(max(9, len(categories) * 1.05), 4 * len(backends)),
        squeeze=False,
    )
    for row_index, backend in enumerate(backends):
        axis = axes[row_index][0]
        selected = [row for row in collective_rows if row["backend"] == backend]
        for row in selected:
            row["plot_category"] = f"{row['model']}\n{row['collective_type']}"
        if categories:
            _plot_bars(axis, categories, selected, topologies)
        else:
            axis.text(
                0.5,
                0.5,
                "Per-collective results unavailable:\nranks[].collectives is absent",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.set_xticks(range(len(models)))
            axis.set_xticklabels(models)
        axis.set_ylabel("Speedup (x)\nPT mean busy / topology mean busy")
        axis.set_xlabel("Model / collective type")
        axis.set_title(backend)
    fig.suptitle(f"Per-collective topology speedup ({scope})", y=0.995)
    fig.legend(
        handles=topology_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=len(topologies),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    path = plot_dir / "collective-speedups.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))

    communication_metrics = (
        (
            "communication_time_ns_mean_per_rank",
            "Raw communication busy time\n(includes overlapped time)",
        ),
        (
            "exposed_communication_time_ns_mean_per_rank",
            "Exposed communication time\n(unhidden wall-time cost)",
        ),
    )
    fig, axes = plt.subplots(len(backends), 2, figsize=(12, 4 * len(backends)), squeeze=False)
    for backend_index, backend in enumerate(backends):
        for metric_index, (metric, title) in enumerate(communication_metrics):
            axis = axes[backend_index][metric_index]
            selected = [
                row.copy()
                for row in summary["speedups"]
                if row["family"] == "communication"
                and row["backend"] == backend
                and row["metric"] == metric
            ]
            for row in selected:
                row["plot_category"] = row["model"]
            _plot_bars(axis, models, selected, topologies)
            axis.set_xlabel("Model")
            axis.set_ylabel("Speedup (x)\nPT mean rank time / topology mean rank time")
            axis.set_title(f"{backend}: {title}")
    fig.suptitle(f"Communication topology speedup ({scope})", y=0.995)
    fig.legend(
        handles=topology_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        ncol=len(topologies),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.89), h_pad=2.2, w_pad=2.2)
    path = plot_dir / "communication-speedups.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))

    run_index = {
        (row["model"], row["backend"], row["topology"]): row for row in summary["runs"]
    }
    fig, axes = plt.subplots(len(backends), 2, figsize=(13, 4.5 * len(backends)), squeeze=False)
    for backend_index, backend in enumerate(backends):
        speed_axis = axes[backend_index][0]
        selected = [
            row.copy()
            for row in summary["speedups"]
            if row["family"] == "end_to_end" and row["backend"] == backend
        ]
        for row in selected:
            row["plot_category"] = row["model"]
        _plot_bars(speed_axis, models, selected, topologies)
        speed_axis.set_xlabel("Model")
        speed_axis.set_ylabel("Speedup (x)\nPT simulation end / topology simulation end")
        speed_axis.set_title(f"{backend}: end-to-end speedup")

        spread_axis = axes[backend_index][1]
        positions: list[int] = []
        tick_labels: list[str] = []
        for model in models:
            for topology in topologies:
                row = run_index[(model, backend, topology)]
                position = len(positions)
                positions.append(position)
                tick_labels.append(f"{model}\n{topology_labels[topology]}")
                mean_ms = row["rank_wall_time_ns_mean"] / 1e6
                low_ms = (row["rank_wall_time_ns_mean"] - row["rank_wall_time_ns_min"]) / 1e6
                high_ms = (row["rank_wall_time_ns_max"] - row["rank_wall_time_ns_mean"]) / 1e6
                spread_axis.errorbar(
                    position,
                    mean_ms,
                    yerr=[[low_ms], [high_ms]],
                    fmt="o",
                    capsize=4,
                    color=TOPOLOGY_COLORS.get(topology),
                )
                spread_axis.scatter(
                    [position],
                    [row["simulation_end_ns"] / 1e6],
                    marker="D",
                    color=TOPOLOGY_COLORS.get(topology),
                    edgecolor="black",
                    linewidth=0.5,
                )
        spread_axis.set_xticks(positions)
        spread_axis.set_xticklabels(tick_labels)
        spread_axis.set_xlabel("Model / topology")
        spread_axis.set_ylabel("Time (ms)")
        spread_axis.set_title(
            f"{backend}: rank wall-time mean ± min/max\n"
            "diamond = global simulation_end_ns"
        )
        spread_axis.grid(True, axis="y", alpha=0.25)
    timing_handles = topology_handles + [
        Line2D([0], [0], marker="o", color="black", linestyle="None", label="rank mean ± min/max"),
        Line2D([0], [0], marker="D", color="black", linestyle="None", label="simulation_end_ns"),
    ]
    fig.suptitle(f"End-to-end speedup and rank spread ({scope})", y=0.995)
    fig.legend(
        handles=timing_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        ncol=len(timing_handles),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.89), h_pad=2.4, w_pad=2.0)
    path = plot_dir / "end-to-end-speedups-and-rank-spread.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))

    fig, axes = plt.subplots(len(models), 2, figsize=(11, 4 * len(models)), squeeze=False)
    for model_index, model in enumerate(models):
        for metric_index, (metric, ylabel) in enumerate(
            (
                ("utilization", "Directed-link utilization\nserialization busy / simulation end"),
                ("queue_wait_ns", "Total queue wait per directed link (ns)\nsymlog scale"),
            )
        ):
            axis = axes[model_index][metric_index]
            samples = [
                link_samples[run_index[(model, "congestion_aware", topology)]["run_id"]][metric]
                for topology in topologies
            ]
            if any(samples):
                boxes = axis.boxplot(samples, patch_artist=True, showfliers=False)
                for box, topology in zip(boxes["boxes"], topologies):
                    box.set_facecolor(TOPOLOGY_COLORS.get(topology))
                    box.set_alpha(0.75)
            else:
                axis.text(0.5, 0.5, "No directed-link samples", ha="center", transform=axis.transAxes)
            axis.set_xticks(range(1, len(topologies) + 1))
            axis.set_xticklabels([topology_labels[topology] for topology in topologies])
            axis.set_xlabel("Topology")
            axis.set_ylabel(ylabel)
            axis.set_title(f"{model}: congestion-aware {metric.replace('_', ' ')}")
            axis.grid(True, axis="y", alpha=0.25)
            if metric == "queue_wait_ns":
                axis.set_yscale("symlog", linthresh=1.0)
    fig.suptitle(f"Congestion-aware link utilization and queue wait ({scope})", y=0.995)
    fig.legend(
        handles=topology_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        ncol=len(topologies),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.89), h_pad=2.0, w_pad=2.0)
    path = plot_dir / "congestion-aware-link-distributions.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))
    return paths


def _write_report(output: Path, summary: dict[str, Any]) -> Path:
    unavailable = sum(
        1
        for row in summary["speedups"]
        if row["family"] == "collective" and not row["available"]
    )
    plot_descriptions = {
        "collective-speedups.png": (
            "PT mean collective busy time divided by topology mean, by collective, "
            "model, and backend"
        ),
        "communication-speedups.png": (
            "raw communication-busy and exposed/unhidden communication speedups"
        ),
        "end-to-end-speedups-and-rank-spread.png": (
            "simulation-end speedups plus rank wall-time mean and min/max spread"
        ),
        "congestion-aware-link-distributions.png": (
            "per-directed-link utilization and total queue-wait distributions"
        ),
    }
    plot_lines = "\n".join(
        f"- `{Path(path).name}` — {plot_descriptions[Path(path).name]}"
        for path in summary["plots"]
    )
    report = f"""# Chakra Trace Experiment Analysis

Date: {summary['analysis_date']}

All {summary['completed_job_count']} of {summary['job_count']} jobs have complete, unique rank sets of 128 ranks.

## Controlled system policy

All-reduce, all-gather, reduce-scatter, and all-to-all use ASTRA's native `ring` implementation. The system and workload prefix are identical within each model/backend triplet; only the manifest-provided Graph network configuration changes. `--comm-group-configuration` is omitted, retaining ASTRA's literal `empty` default so PyTorch process-group metadata is read from each Chakra trace.

## Reductions and speedups

- Per-collective time is the sum of `time_ns` across ranks divided by all 128 ranks. A topology speedup is PT mean busy time divided by that topology's mean busy time.
- Raw communication speedup uses the mean `communication_time_ns` across ranks. Exposed communication speedup uses the mean `exposed_communication_time_ns`; exposed time is the portion that costs wall time. Both are PT time divided by topology time.
- End-to-end speedup is PT `simulation_end_ns` divided by topology `simulation_end_ns`. Rank `wall_time_ns` is shown as mean with min/max spread and is not substituted for the global makespan.
- Link utilization and queue-wait distributions are per directed link and include congestion-aware runs only.

Per-collective unavailable comparison rows: {unavailable}. These rows remain explicit when older statistics omit `ranks[].collectives`.

## Plots

{plot_lines}
"""
    path = output / "REPORT.md"
    path.write_text(report, encoding="utf-8")
    return path


def analyze(prepared_file: Path | str) -> Path:
    """Require all 12 complete results, normalize them, compute speedups, and plot."""

    _, prepared = _load(prepared_file)
    jobs = prepared.get("jobs", [])
    if len(jobs) != 12:
        raise RuntimeError(f"prepared trace experiment has {len(jobs)} jobs, expected 12")
    run_root = Path(prepared["output_directory"]) / "runs"
    runs: list[dict[str, Any]] = []
    collectives: list[dict[str, Any]] = []
    link_samples: dict[str, dict[str, list[float | int]]] = {}
    for job in jobs:
        record_path = run_root / job["run_id"] / "run.json"
        if not record_path.is_file():
            raise RuntimeError(f"{job['run_id']} has no run record: {record_path}")
        record = _read_json_object(record_path, f"run record for {job['run_id']}")
        stats_path = Path(record["statistics"])
        if not stats_path.is_file():
            raise RuntimeError(f"{job['run_id']} has no structured statistics: {stats_path}")
        try:
            raw_stats = json.loads(stats_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"{job['run_id']} has invalid structured statistics") from exc
        stats = _validate_statistics(raw_stats, int(job["ranks"]), job["run_id"])
        row, collective_rows, samples = _normalize_run(job, stats)
        runs.append(row)
        collectives.extend(collective_rows)
        link_samples[job["run_id"]] = samples

    models = list(prepared.get("model_order", EXPECTED_MODELS))
    backends = list(prepared.get("backend_order", EXPECTED_BACKENDS))
    topologies = list(prepared.get("topology_order", EXPECTED_TOPOLOGIES))
    speedups = _build_speedups(runs, collectives, models, backends, topologies)
    output = Path(prepared["output_directory"]) / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "experiment": prepared["experiment"],
        "analysis_date": date.today().isoformat(),
        "job_count": len(jobs),
        "completed_job_count": len(runs),
        "models": models,
        "backends": backends,
        "topologies": topologies,
        "topology_labels": {
            topology: prepared["topologies"][topology]["label"] for topology in topologies
        },
        "system_configuration_policy": prepared["system_configuration_policy"],
        "comm_group_configuration": prepared["comm_group_configuration"],
        "reduction_policy": {
            "collective_time_ns": "sum over rank/type entries, divided by all 128 ranks",
            "communication_time_ns": "mean of rank-local interval-union busy time",
            "exposed_communication_time_ns": "mean rank-local unhidden communication time",
            "simulation_end_ns": "global ASTRA makespan",
            "rank_wall_time_ns": "mean with min/max spread",
            "speedup": "PT time divided by candidate topology time; larger is faster",
        },
        "runs": runs,
        "collectives": collectives,
        "speedups": speedups,
        "plots": [],
    }
    summary["plots"] = _plots(summary, link_samples, output)
    report_path = _write_report(output, summary)
    summary["report"] = str(report_path)

    csv_fields = (
        "family",
        "metric",
        "collective_type",
        "model",
        "backend",
        "topology",
        "topology_label",
        "reference_topology",
        "reduction",
        "baseline_value_ns",
        "topology_value_ns",
        "unit",
        "speedup",
        "available",
        "reason",
        "baseline_run_id",
        "run_id",
    )
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(speedups)
    summary_path = output / "summary.json"
    _dump(summary_path, summary)
    return summary_path
