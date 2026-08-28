"""Prepare, execute, and analyze the TONS isolated-collective experiment."""

from __future__ import annotations

import csv
import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

from tons_collectives import (
    generate_collective_workload,
    generate_direct_alltoall,
    generate_fixed_route_alltoall,
    generate_pmcf_alltoall,
    lower_msccl_to_chakra,
    verify_schedule,
)
from tons_topology import generate_graph_artifacts, known_bundle, validate_bundle


COLLECTIVE_KEY = {
    "allgather": "all-gather-implementation-custom",
    "allreduce": "all-reduce-implementation-custom",
    "reduce_scatter": "reduce-scatter-implementation-custom",
    "alltoall": "all-to-all-implementation-custom",
}


def _load(path: Path | str) -> tuple[Path, dict[str, Any]]:
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as stream:
        return path, json.load(stream)


def _dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _repo(manifest_path: Path, manifest: dict[str, Any]) -> Path:
    configured = manifest.get("repository_root", "../..")
    return (manifest_path.parent / configured).resolve()


def _make_system(path: Path, collective: str, schedule_prefix: Path) -> Path:
    _dump(
        path,
        {
            "scheduling-policy": "FIFO",
            "preferred-dataset-splits": 1,
            COLLECTIVE_KEY[collective]: [str(schedule_prefix.resolve())],
            "local-mem-bw": 50,
        },
    )
    return path


def _l3ss_schedule(
    binary: Path,
    topology: Path,
    collective: str,
    chunks: int,
    output: Path,
) -> None:
    kind = {"allgather": "ag", "allreduce": "ar", "reduce_scatter": "rs"}[collective]
    # l3ss_tree writes the XML directly and does not create its parent.  Keep
    # the manifest output directory self-contained rather than depending on
    # the legacy scratch directories used by older regeneration wrappers.
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(binary),
        "--xml",
        str(output),
        "build",
        str(topology),
        kind,
        "rr",
        str(chunks),
    ]
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode:
        raise RuntimeError(
            f"l3ss_tree failed for {collective}:\n{completed.stdout}\n{completed.stderr}"
        )


def prepare(manifest_file: Path | str) -> Path:
    """Validate inputs and generate Graph, XML, ET, system, and run records."""

    manifest_path, manifest = _load(manifest_file)
    repo = _repo(manifest_path, manifest)
    output = (repo / manifest["output_directory"]).resolve()
    generated = output / "prepared"
    reports_dir = generated / "reports"
    schedules_dir = generated / "schedules"
    workloads_dir = generated / "workloads"
    systems_dir = generated / "systems"
    topology_root = (repo / manifest["topology_root"]).resolve()
    ranks = int(manifest["ranks"])
    subchunks = int(manifest["subchunks"])

    bundles: dict[str, Any] = {}
    schedule_prefixes: dict[tuple[str, str, str], Path] = {}
    for topology_name, definition in manifest["topologies"].items():
        bundle = known_bundle(topology_root, definition["bundle"])
        report = validate_bundle(bundle, expected_degree=6, destination_based=True)
        _dump(reports_dir / f"{topology_name}.topology.json", report.to_dict())
        if not report.ok or report.routers != ranks:
            raise RuntimeError(f"topology {topology_name} failed validation: {report.errors}")
        graph = generate_graph_artifacts(
            bundle.topology,
            bundle.routes,
            generated / "networks" / topology_name,
            stem=topology_name,
        )
        bundles[topology_name] = {
            "bundle": definition["bundle"],
            "route_policy": definition["route_policy"],
            "topology": str(bundle.topology.resolve()),
            "routes": str(bundle.routes.resolve()),
            "network": str(graph.network_config.resolve()),
            "validation": str((reports_dir / f"{topology_name}.topology.json").resolve()),
            "maximum_directed_channel_load": report.maximum_directed_channel_load,
        }

        for collective in ("allgather", "allreduce", "reduce_scatter"):
            xml = schedules_dir / topology_name / f"{collective}.xml"
            _l3ss_schedule(
                (repo / manifest["l3ss_binary"]).resolve(),
                bundle.topology,
                collective,
                subchunks,
                xml,
            )
            verification = verify_schedule(xml)
            _dump(reports_dir / f"{topology_name}.{collective}.schedule.json", verification.to_dict())
            if not verification.ok:
                raise RuntimeError(f"invalid {topology_name} {collective} schedule: {verification.errors}")
            prefix = schedules_dir / topology_name / collective
            lower_msccl_to_chakra(xml, prefix)
            schedule_prefixes[(topology_name, collective, "topology-aware")] = prefix

        direct_xml = generate_direct_alltoall(
            ranks, subchunks, schedules_dir / topology_name / "alltoall-direct.xml"
        )
        pipeline_xml = generate_fixed_route_alltoall(
            bundle.routes,
            subchunks,
            schedules_dir / topology_name / "alltoall-fixed-route-pipeline.xml",
        )
        pmcf_candidates = bundle.pmcf_candidates or bundle.candidates
        if pmcf_candidates is None:
            raise RuntimeError(f"topology {topology_name} has no pMCF candidate paths")
        pmcf = generate_pmcf_alltoall(
            bundle.topology,
            pmcf_candidates,
            subchunks,
            schedules_dir / topology_name / "alltoall-pmcf.xml",
            report_path=reports_dir / f"{topology_name}.alltoall.pmcf.json",
            solver=str(manifest.get("pmcf", {}).get("solver", "highs")),
            threads=int(manifest.get("pmcf", {}).get("threads", 16)),
            seed=int(manifest.get("pmcf", {}).get("seed", manifest["random_seed"])),
            reuse=True,
        )
        bundles[topology_name]["pmcf"] = {
            "candidate_policy": manifest.get("pmcf", {}).get("candidate_policy", "cpl-safe"),
            "candidate_paths": pmcf.candidate_paths,
            "positive_paths": pmcf.positive_paths,
            "maximum_concurrent_flow": pmcf.maximum_concurrent_flow,
            "quantized_maximum_link_load": pmcf.quantized_maximum_link_load,
            "epochs": pmcf.epochs,
            "report": str(pmcf.report),
        }
        for mode, xml in (
            ("direct", direct_xml),
            ("fixed-route-pipeline", pipeline_xml),
            ("pmcf", pmcf.schedule),
        ):
            verification = verify_schedule(xml)
            _dump(reports_dir / f"{topology_name}.alltoall.{mode}.schedule.json", verification.to_dict())
            if not verification.ok:
                raise RuntimeError(f"invalid {topology_name} alltoall {mode}: {verification.errors}")
            prefix = schedules_dir / topology_name / f"alltoall-{mode}"
            lower_msccl_to_chakra(xml, prefix)
            schedule_prefixes[(topology_name, "alltoall", mode)] = prefix

    workload_prefixes: dict[tuple[str, int], Path] = {}
    for collective in COLLECTIVE_KEY:
        for size in manifest["sizes_bytes"]:
            prefix = workloads_dir / f"{collective}-{size}"
            generate_collective_workload(prefix, ranks, collective, int(size))
            workload_prefixes[(collective, int(size))] = prefix

    modes = {
        "allgather": ["topology-aware"],
        "allreduce": ["topology-aware"],
        "reduce_scatter": ["topology-aware"],
        "alltoall": ["direct", "fixed-route-pipeline", "pmcf"],
    }
    jobs: list[dict[str, Any]] = []
    for topology_name in manifest["topologies"]:
        for backend in manifest["backends"]:
            for collective, collective_modes in modes.items():
                for mode in collective_modes:
                    schedule_prefix = schedule_prefixes[(topology_name, collective, mode)]
                    system = _make_system(
                        systems_dir / topology_name / f"{collective}-{mode}.json",
                        collective,
                        schedule_prefix,
                    )
                    for size in manifest["sizes_bytes"]:
                        run_id = f"{topology_name}__{backend}__{collective}__{mode}__{size}"
                        jobs.append(
                            {
                                "run_id": run_id,
                                "stage": "primary",
                                "topology": topology_name,
                                "topology_bundle": bundles[topology_name]["bundle"],
                                "route_policy": bundles[topology_name]["route_policy"],
                                "network_profile": manifest["network_profile"],
                                "collective": collective,
                                "schedule_mode": mode,
                                "ranks": ranks,
                                "bytes_per_rank": int(size),
                                "subchunks": subchunks,
                                "backend": backend,
                                "network": bundles[topology_name]["network"],
                                "workload": str(workload_prefixes[(collective, int(size))].resolve()),
                                "schedule": str(schedule_prefix.resolve()),
                                "system": str(system.resolve()),
                                "seed": int(manifest["random_seed"]),
                                "route_load_bound": bundles[topology_name]["maximum_directed_channel_load"],
                                "pmcf_maximum_concurrent_flow": (
                                    bundles[topology_name]["pmcf"]["maximum_concurrent_flow"]
                                    if mode == "pmcf" else None
                                ),
                                "pmcf_quantized_maximum_link_load": (
                                    bundles[topology_name]["pmcf"]["quantized_maximum_link_load"]
                                    if mode == "pmcf" else None
                                ),
                            }
                        )

    # A minimal one-subchunk control is always executed before primary jobs.
    smoke_jobs: list[dict[str, Any]] = []
    for topology_name, definition in manifest["topologies"].items():
        bundle = known_bundle(topology_root, definition["bundle"])
        xml = generate_direct_alltoall(
            ranks, 1, schedules_dir / topology_name / "smoke-alltoall-direct.xml"
        )
        prefix = schedules_dir / topology_name / "smoke-alltoall-direct"
        lower_msccl_to_chakra(xml, prefix)
        system = _make_system(
            systems_dir / topology_name / "smoke-alltoall-direct.json", "alltoall", prefix
        )
        size = int(manifest["smoke"]["bytes_per_rank"])
        smoke_workload = workloads_dir / f"alltoall-smoke-{size}"
        if ("alltoall", size) not in workload_prefixes:
            generate_collective_workload(smoke_workload, ranks, "alltoall", size)
        else:
            smoke_workload = workload_prefixes[("alltoall", size)]
        smoke_jobs.append(
            {
                "run_id": f"smoke__{topology_name}__alltoall__direct__{size}",
                "stage": "smoke",
                "topology": topology_name,
                "topology_bundle": definition["bundle"],
                "route_policy": definition["route_policy"],
                "network_profile": manifest["network_profile"],
                "collective": "alltoall",
                "schedule_mode": "direct",
                "ranks": ranks,
                "bytes_per_rank": size,
                "subchunks": 1,
                "backend": "congestion_unaware",
                "network": bundles[topology_name]["network"],
                "workload": str(smoke_workload.resolve()),
                "schedule": str(prefix.resolve()),
                "system": str(system.resolve()),
                "seed": int(manifest["random_seed"]),
            }
        )

    prepared = {
        "schema_version": 1,
        "source_manifest": str(manifest_path),
        "repository_root": str(repo),
        "output_directory": str(output),
        "binaries": {
            key: str((repo / value).resolve()) for key, value in manifest["binaries"].items()
        },
        "remote_memory": str((repo / manifest["remote_memory_configuration"]).resolve()),
        "bundles": bundles,
        "jobs": smoke_jobs + jobs,
    }
    prepared_path = generated / "prepared.json"
    _dump(prepared_path, prepared)
    return prepared_path


def _revision(directory: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(directory), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def run(prepared_file: Path | str, *, dry_run: bool = False, limit: int | None = None) -> Path:
    """Execute prepared jobs in order and fail immediately on incomplete ranks."""

    prepared_path, prepared = _load(prepared_file)
    output = Path(prepared["output_directory"])
    run_root = output / "runs"
    repo = Path(prepared["repository_root"])
    revisions = {
        "repository": _revision(repo),
        "astra_sim": _revision(repo / "simul" / "astra-sim"),
        "analytical_backend": _revision(
            repo / "simul" / "astra-sim" / "extern" / "network_backend" / "analytical"
        ),
        "chakra": _revision(repo / "simul" / "astra-sim" / "extern" / "graph_frontend" / "chakra"),
    }
    jobs = prepared["jobs"][:limit] if limit is not None else prepared["jobs"]
    for job in jobs:
        binary = Path(prepared["binaries"][job["backend"]])
        run_dir = run_root / job["run_id"]
        stats = run_dir / "statistics.json"
        command = [
            str(binary),
            f"--workload-configuration={job['workload']}",
            f"--system-configuration={job['system']}",
            f"--remote-memory-configuration={prepared['remote_memory']}",
            f"--network-configuration={job['network']}",
            f"--statistics-output={stats}",
        ]
        record = {
            **job,
            "command": command,
            "revisions": revisions,
            "statistics": str(stats),
        }
        _dump(run_dir / "run.json", record)
        if dry_run:
            continue
        if not binary.is_file():
            raise RuntimeError(f"simulator binary does not exist: {binary}")
        completed = subprocess.run(command, cwd=repo, text=True, capture_output=True)
        (run_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
        (run_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(f"{job['run_id']} failed with exit code {completed.returncode}")
        if not stats.is_file():
            raise RuntimeError(f"{job['run_id']} did not emit structured statistics")
        result = json.loads(stats.read_text(encoding="utf-8"))
        complete_ranks = sum(bool(rank["complete"]) for rank in result.get("ranks", []))
        if not result.get("complete") or complete_ranks != int(job["ranks"]):
            raise RuntimeError(
                f"{job['run_id']} incomplete: {complete_ranks}/{job['ranks']} ranks"
            )
    return run_root


def _collect_results(run_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record_path in sorted(run_root.glob("*/run.json")):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        stats_path = Path(record["statistics"])
        if not stats_path.is_file():
            continue
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        end = int(stats["simulation_end_ns"])
        links = stats.get("links", [])
        rows.append(
            {
                **{key: record[key] for key in (
                    "run_id", "stage", "topology", "topology_bundle", "route_policy",
                    "collective", "schedule_mode", "ranks", "bytes_per_rank", "subchunks",
                    "backend", "seed"
                )},
                "simulation_end_ns": end,
                "throughput_GBps": float(record["bytes_per_rank"]) / end if end else 0.0,
                "complete": bool(stats.get("complete")),
                "rank_count": len(stats.get("ranks", [])),
                "total_link_bytes": sum(int(link["bytes"]) for link in links),
                "queued_chunks": sum(int(link["queued_chunks"]) for link in links),
                "max_link_utilization": max((float(link["utilization"]) for link in links), default=0.0),
                "total_queue_wait_ns": sum(int(link["total_queue_wait_ns"]) for link in links),
                "max_queue_wait_ns": max((int(link["max_queue_wait_ns"]) for link in links), default=0),
                "route_load_bound": record.get("route_load_bound"),
                "pmcf_maximum_concurrent_flow": record.get("pmcf_maximum_concurrent_flow"),
                "pmcf_quantized_maximum_link_load": record.get("pmcf_quantized_maximum_link_load"),
            }
        )
    return rows


def _acceptance(
    rows: list[dict[str, Any]], expected_run_ids: set[str] | None = None
) -> dict[str, Any]:
    failures: list[str] = []
    checks: list[dict[str, Any]] = []
    if expected_run_ids is not None:
        missing = sorted(expected_run_ids - {row["run_id"] for row in rows})
        if missing:
            failures.append(f"{len(missing)} prepared runs have no structured result")
    for row in rows:
        if not row["complete"] or row["rank_count"] != row["ranks"]:
            failures.append(f"{row['run_id']}: incomplete ranks")
    indexed = {
        (row["topology"], row["backend"], row["collective"], row["schedule_mode"], row["bytes_per_rank"]): row
        for row in rows if row["stage"] == "primary"
    }
    for row in rows:
        if (
            row["stage"] == "primary"
            and row["backend"] == "congestion_aware"
            and row["collective"] == "alltoall"
            and row["schedule_mode"] == "direct"
            and row["queued_chunks"] == 0
        ):
            failures.append(f"{row['run_id']}: shared-link queue accounting is empty")
        if (
            row["stage"] == "primary"
            and row["backend"] == "congestion_aware"
            and row["collective"] == "alltoall"
            and row["schedule_mode"] in {"fixed-route-pipeline", "pmcf"}
            and row["queued_chunks"] != 0
        ):
            failures.append(f"{row['run_id']}: scheduled A2A unexpectedly queued chunks")
    large_sizes = sorted({row["bytes_per_rank"] for row in rows})[-2:]
    comparison_topologies = sorted({row["topology"] for row in rows if row["topology"] != "pt"})
    for backend in {row["backend"] for row in rows}:
        for size in large_sizes:
            for topology in comparison_topologies:
                for mode in ("direct", "fixed-route-pipeline", "pmcf"):
                    pt = indexed.get(("pt", backend, "alltoall", mode, size))
                    candidate = indexed.get((topology, backend, "alltoall", mode, size))
                    if pt and candidate:
                        speedup = candidate["throughput_GBps"] / pt["throughput_GBps"]
                        route_reference = (
                            float(pt["route_load_bound"]) / float(candidate["route_load_bound"])
                            if pt.get("route_load_bound") and candidate.get("route_load_bound")
                            else None
                        )
                        checks.append({"check": "alltoall_comparison", "topology": topology,
                                       "backend": backend, "mode": mode,
                                       "bytes_per_rank": size, "speedup": speedup,
                                       "route_load_reference": route_reference, "passed": True})
                for collective in ("allgather", "allreduce", "reduce_scatter"):
                    pt = indexed.get(("pt", backend, collective, "topology-aware", size))
                    candidate = indexed.get((topology, backend, collective, "topology-aware", size))
                    if pt and candidate:
                        ratio = candidate["throughput_GBps"] / pt["throughput_GBps"]
                        deviation = abs(ratio - 1.0)
                        passed = deviation <= 0.05
                        checks.append({"check": "collective_equality", "topology": topology,
                                       "backend": backend, "collective": collective,
                                       "bytes_per_rank": size, "ratio": ratio,
                                       "deviation": deviation, "passed": passed,
                                       "hypothesis": collective == "reduce_scatter"})
                        if not passed:
                            failures.append(
                                f"{topology} {collective} differs by more than 5%: {backend} {size}"
                            )
    return {"passed": not failures, "failures": failures, "checks": checks}


def _plots(rows: list[dict[str, Any]], run_root: Path, output: Path) -> list[str]:
    matplotlib_config = output / ".matplotlib"
    matplotlib_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config))
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    plot_dir = output / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    primary = [row for row in rows if row["stage"] == "primary"]
    for collective in sorted({row["collective"] for row in primary}):
        for backend in sorted({row["backend"] for row in primary}):
            subset = [row for row in primary if row["collective"] == collective and row["backend"] == backend]
            if not subset:
                continue
            if collective == "alltoall":
                modes = sorted({row["schedule_mode"] for row in subset})
                fig, axes = plt.subplots(1, len(modes), figsize=(5.2 * len(modes), 4.5))
                if len(modes) == 1:
                    axes = [axes]
                for axis, mode in zip(axes, modes):
                    for topology in sorted({row["topology"] for row in subset}):
                        values = [
                            row for row in subset
                            if row["topology"] == topology and row["schedule_mode"] == mode
                        ]
                        values.sort(key=lambda value: value["bytes_per_rank"])
                        axis.plot(
                            [value["bytes_per_rank"] / 2**20 for value in values],
                            [value["throughput_GBps"] for value in values],
                            marker="o",
                            label=topology,
                        )
                    axis.set_xscale("log", base=2)
                    axis.set_xlabel("Bytes per rank (MiB)")
                    axis.set_ylabel("Throughput (GB/s)")
                    axis.set_title(mode)
                    axis.grid(True, alpha=0.25)
                    axis.legend()
                fig.suptitle(f"alltoall — {backend}")
                fig.tight_layout()
                path = plot_dir / f"throughput-{collective}-{backend}.png"
                fig.savefig(path, dpi=160)
                plt.close(fig)
                paths.append(str(path))
                continue
            fig, axis = plt.subplots(figsize=(7, 4.5))
            groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
            for row in subset:
                groups[(row["topology"], row["schedule_mode"])].append(row)
            for (topology, mode), values in sorted(groups.items()):
                values.sort(key=lambda value: value["bytes_per_rank"])
                axis.plot([value["bytes_per_rank"] / 2**20 for value in values],
                          [value["throughput_GBps"] for value in values], marker="o",
                          label=f"{topology} {mode}")
            axis.set_xscale("log", base=2)
            axis.set_xlabel("Bytes per rank (MiB)")
            axis.set_ylabel("Throughput (GB/s)")
            axis.set_title(f"{collective} — {backend}")
            axis.grid(True, alpha=0.25)
            axis.legend()
            fig.tight_layout()
            path = plot_dir / f"throughput-{collective}-{backend}.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            paths.append(str(path))

    # Topology/PT speedup with unity and fixed-route load references.
    by_pair: dict[tuple[str, str, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in primary:
        by_pair[(row["backend"], row["collective"], row["schedule_mode"], row["bytes_per_rank"])][row["topology"]] = row
    fig, axis = plt.subplots(figsize=(8, 5))
    series: dict[tuple[str, str, str, str], list[tuple[int, float]]] = defaultdict(list)
    for (backend, collective, mode, size), pair in by_pair.items():
        if "pt" not in pair:
            continue
        for topology, candidate in pair.items():
            if topology == "pt":
                continue
            speedup = candidate["throughput_GBps"] / pair["pt"]["throughput_GBps"]
            series[(topology, backend, collective, mode)].append((size, speedup))
    for key, values in sorted(series.items()):
        values.sort()
        axis.plot([size / 2**20 for size, _ in values], [value for _, value in values],
                  marker="o", label=" / ".join(key))
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1, label="unity")
    route_bounds = {
        row["topology"]: row["route_load_bound"]
        for row in primary
        if row.get("route_load_bound") and row["topology"] != "pt"
    }
    pt_bound = next(
        (row["route_load_bound"] for row in primary if row["topology"] == "pt" and row.get("route_load_bound")),
        None,
    )
    if pt_bound:
        for topology, bound in sorted(route_bounds.items()):
            axis.axhline(
                float(pt_bound) / float(bound),
                linestyle=":",
                linewidth=1.25,
                label=f"{topology}/PT route-load {pt_bound}/{bound}",
            )
    axis.set_xscale("log", base=2)
    axis.set_xlabel("Bytes per rank (MiB)")
    axis.set_ylabel("Topology / PT throughput")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=7)
    fig.tight_layout()
    speedup_path = plot_dir / "topology-pt-speedup.png"
    fig.savefig(speedup_path, dpi=160)
    plt.close(fig)
    paths.append(str(speedup_path))

    utilizations: list[float] = []
    waits: list[int] = []
    for record_path in run_root.glob("*/run.json"):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record["backend"] != "congestion_aware" or not Path(record["statistics"]).is_file():
            continue
        stats = json.loads(Path(record["statistics"]).read_text(encoding="utf-8"))
        utilizations.extend(float(link["utilization"]) for link in stats.get("links", []))
        waits.extend(int(link["total_queue_wait_ns"]) for link in stats.get("links", []))
    if utilizations:
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        axes[0].hist(utilizations, bins=30)
        axes[0].set_xlabel("Directed-link utilization")
        axes[1].hist(waits, bins=30)
        axes[1].set_xlabel("Total queue wait (ns)")
        for axis in axes:
            axis.set_ylabel("Directed links")
            axis.grid(True, alpha=0.2)
        fig.tight_layout()
        congestion_path = plot_dir / "congestion-distributions.png"
        fig.savefig(congestion_path, dpi=160)
        plt.close(fig)
        paths.append(str(congestion_path))
    return paths


def analyze(prepared_file: Path | str) -> Path:
    """Normalize structured output, evaluate acceptance, and render plots."""

    _, prepared = _load(prepared_file)
    output = Path(prepared["output_directory"]) / "analysis"
    run_root = Path(prepared["output_directory"]) / "runs"
    rows = _collect_results(run_root)
    if not rows:
        raise RuntimeError(f"no completed structured results under {run_root}")
    output.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    acceptance = _acceptance(
        rows, {job["run_id"] for job in prepared.get("jobs", [])}
    )
    plots = _plots(rows, run_root, output)
    summary = {"schema_version": 1, "rows": rows, "acceptance": acceptance, "plots": plots}
    summary_path = output / "summary.json"
    _dump(summary_path, summary)
    if not acceptance["passed"]:
        raise RuntimeError("acceptance failures: " + "; ".join(acceptance["failures"]))
    return summary_path
