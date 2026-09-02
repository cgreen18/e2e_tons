import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tons_sim.trace_pipeline import analyze, prepare, run


MODELS = {
    "llama7b": "Llama7B_N32_GPU128_PP1_DP128_7B_BS128",
    "moe8x13b": "MoE8x13B_N32_GPU128_TP4_PP4_DP8_EP4_13B_BS128",
}
TOPOLOGIES = {
    "pt": ("PT", "pt-dor-128"),
    "pdtt": ("PDTT", "pdtt-128"),
    "tons": ("TONS", "tons-128"),
}
BACKENDS = ("congestion_unaware", "congestion_aware")


def _system_policy() -> dict[str, object]:
    return {
        "scheduling-policy": "FIFO",
        "endpoint-delay": 10,
        "active-chunks-per-dimension": 4,
        "preferred-dataset-splits": 4,
        "all-reduce-implementation": ["ring"],
        "all-gather-implementation": ["ring"],
        "reduce-scatter-implementation": ["ring"],
        "all-to-all-implementation": ["ring"],
        "collective-optimization": "baseline",
        "local-mem-bw": 1600,
        "boost-mode": 0,
    }


def _synthetic_manifest(root: Path) -> Path:
    trace_root = root / "repaired"
    for directory in MODELS.values():
        model_dir = trace_root / directory
        model_dir.mkdir(parents=True)
        for rank in range(128):
            (model_dir / f"chakra.{rank}.et").write_bytes(b"et")

    network_dir = root / "networks"
    network_dir.mkdir()
    for topology in TOPOLOGIES:
        (network_dir / f"{topology}.yml").write_text(
            "topology: [ Graph ]\nnpus_count: [ 128 ]\n",
            encoding="utf-8",
        )
    (root / "remote.json").write_text("{}\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "repository_root": ".",
        "output_directory": "output",
        "trace_root": "repaired",
        "models": {
            model: {"directory": directory, "workload_prefix": "chakra"}
            for model, directory in MODELS.items()
        },
        "topologies": {
            topology: {
                "label": label,
                "bundle": bundle,
                "network_configuration": f"networks/{topology}.yml",
            }
            for topology, (label, bundle) in TOPOLOGIES.items()
        },
        "backends": list(BACKENDS),
        "binaries": {
            "congestion_unaware": "bin/unaware",
            "congestion_aware": "bin/aware",
        },
        "remote_memory_configuration": "remote.json",
        "comm_group_configuration": "empty",
        "system_configuration_policy": _system_policy(),
        "ranks": 128,
        "seed": 1,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_statistics(
    path: Path,
    *,
    backend: str,
    topology: str,
    with_collectives: bool = True,
    rank_count: int = 128,
) -> None:
    topology_time = {"pt": 100, "pdtt": 50, "tons": 25}[topology]
    ranks = []
    for rank in range(rank_count):
        row = {
            "rank": rank,
            "complete": True,
            "wall_time_ns": topology_time * 10 + rank % 3,
            "compute_time_ns": 800,
            "communication_time_ns": topology_time,
            "exposed_communication_time_ns": topology_time // 2,
            "overlap_time_ns": topology_time - topology_time // 2,
        }
        if with_collectives:
            row["collectives"] = [
                {
                    "comm_type": "ALL_REDUCE",
                    "count": 2,
                    "bytes": 4096,
                    "time_ns": topology_time,
                }
            ]
        ranks.append(row)
    links = []
    if backend == "congestion_aware":
        links = [
            {
                "src": 0,
                "dst": 1,
                "bytes": 4096,
                "chunks": 2,
                "serialization_busy_ns": topology_time,
                "queued_chunks": 1,
                "total_queue_wait_ns": topology_time,
                "max_queue_wait_ns": topology_time,
                "maximum_queue_depth": 1,
                "utilization": 0.5,
            },
            {
                "src": 1,
                "dst": 0,
                "bytes": 2048,
                "chunks": 1,
                "serialization_busy_ns": topology_time // 2,
                "queued_chunks": 0,
                "total_queue_wait_ns": 0,
                "max_queue_wait_ns": 0,
                "maximum_queue_depth": 0,
                "utilization": 0.25,
            },
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "backend": backend,
                "simulation_end_ns": topology_time * 10 + 2,
                "complete": rank_count == 128,
                "ranks": ranks,
                "links": links,
            }
        ),
        encoding="utf-8",
    )


def _prepared_with_statistics(
    root: Path, *, with_collectives: bool = True
) -> tuple[Path, dict[str, object]]:
    prepared_path = prepare(_synthetic_manifest(root))
    run(prepared_path, dry_run=True)
    prepared = _load(prepared_path)
    for job in prepared["jobs"]:
        record_path = root / "output" / "runs" / job["run_id"] / "run.json"
        record = _load(record_path)
        _write_statistics(
            Path(record["statistics"]),
            backend=job["backend"],
            topology=job["topology"],
            with_collectives=with_collectives,
        )
    return prepared_path, prepared


class TracePipelineTest(unittest.TestCase):
    def test_analyze_blocks_when_any_job_has_no_statistics(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared_path = prepare(_synthetic_manifest(root))
            run(prepared_path, dry_run=True)
            prepared = _load(prepared_path)
            for job in prepared["jobs"][:-1]:
                record = _load(root / "output" / "runs" / job["run_id"] / "run.json")
                _write_statistics(
                    Path(record["statistics"]),
                    backend=job["backend"],
                    topology=job["topology"],
                )
            with self.assertRaisesRegex(RuntimeError, "has no structured statistics"):
                analyze(prepared_path)

    def test_collective_speedup_uses_mean_rank_busy_time(self) -> None:
        with TemporaryDirectory() as temporary:
            summary_path = analyze(_prepared_with_statistics(Path(temporary))[0])
            summary = _load(summary_path)
            pdtt = next(
                row
                for row in summary["speedups"]
                if row["family"] == "collective"
                and row["model"] == "llama7b"
                and row["backend"] == "congestion_aware"
                and row["topology"] == "pdtt"
                and row["collective_type"] == "ALL_REDUCE"
            )
            tons = next(
                row
                for row in summary["speedups"]
                if row["family"] == "collective"
                and row["model"] == "llama7b"
                and row["backend"] == "congestion_aware"
                and row["topology"] == "tons"
                and row["collective_type"] == "ALL_REDUCE"
            )
            self.assertEqual(2.0, pdtt["speedup"])
            self.assertEqual(4.0, tons["speedup"])
            self.assertEqual(128 * 2, summary["collectives"][0]["count_sum"])

    def test_dry_run_records_exact_prefix_command_and_empty_logs(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared_path = prepare(_synthetic_manifest(root))
            run_root = run(prepared_path, dry_run=True, limit=1)
            record = _load(next(run_root.glob("*/run.json")))
            expected = [
                str(root / "bin" / "unaware"),
                "--workload-configuration="
                + str(root / "repaired" / MODELS["llama7b"] / "chakra"),
                "--system-configuration=" + str(root / "output" / "prepared" / "system-native-ring.json"),
                "--remote-memory-configuration=" + str(root / "remote.json"),
                "--network-configuration=" + str(root / "networks" / "pt.yml"),
                "--statistics-output="
                + str(run_root / "llama7b__pt__congestion_unaware" / "statistics.json"),
            ]
            self.assertEqual(expected, record["command"])
            self.assertFalse(any("comm-group" in argument for argument in record["command"]))
            self.assertEqual("", Path(record["stdout"]).read_text(encoding="utf-8"))
            self.assertEqual("", Path(record["stderr"]).read_text(encoding="utf-8"))

    def test_missing_collectives_are_reported_unavailable(self) -> None:
        with TemporaryDirectory() as temporary:
            summary_path = analyze(
                _prepared_with_statistics(Path(temporary), with_collectives=False)[0]
            )
            summary = _load(summary_path)
            collective_speedups = [
                row for row in summary["speedups"] if row["family"] == "collective"
            ]
            self.assertTrue(collective_speedups)
            self.assertTrue(all(not row["available"] for row in collective_speedups))
            self.assertTrue(
                all("older statistics" in row["reason"] for row in collective_speedups)
            )
            self.assertEqual([], summary["collectives"])

    def test_missing_trace_rank_is_detected_before_output(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = _synthetic_manifest(root)
            (root / "repaired" / MODELS["moe8x13b"] / "chakra.127.et").unlink()
            with self.assertRaisesRegex(ValueError, "missing ranks 127"):
                prepare(manifest_path)
            self.assertFalse((root / "output").exists())

    def test_plot_labels_name_both_models_and_all_topologies(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared_path, _ = _prepared_with_statistics(root)
            with patch.dict(os.environ, {"MPLCONFIGDIR": str(root / "matplotlib")}):
                import matplotlib

                matplotlib.use("Agg", force=True)
                import matplotlib.pyplot as plt

                with patch.object(plt, "close"):
                    analyze(prepared_path)
                    figures = [plt.figure(number) for number in plt.get_fignums()]
                    self.assertEqual(4, len(figures))
                    for figure in figures:
                        text = " ".join(item.get_text() for item in figure.texts)
                        legend_text = " ".join(
                            item.get_text()
                            for legend in figure.legends
                            for item in legend.get_texts()
                        )
                        self.assertIn("llama7b", text)
                        self.assertIn("moe8x13b", text)
                        for label in ("PT", "PDTT", "TONS"):
                            self.assertIn(label, text + " " + legend_text)
                plt.close("all")

    def test_prepare_expands_exactly_12_jobs(self) -> None:
        with TemporaryDirectory() as temporary:
            prepared = _load(prepare(_synthetic_manifest(Path(temporary))))
            self.assertEqual(12, len(prepared["jobs"]))
            self.assertEqual(
                {
                    (model, topology, backend)
                    for model in MODELS
                    for topology in TOPOLOGIES
                    for backend in BACKENDS
                },
                {
                    (job["model"], job["topology"], job["backend"])
                    for job in prepared["jobs"]
                },
            )
            systems = {job["system"] for job in prepared["jobs"]}
            self.assertEqual(1, len(systems))
            self.assertTrue(
                all(
                    value == ["ring"]
                    for key, value in prepared["system_configuration_policy"].items()
                    if key.endswith("-implementation")
                )
            )

    def test_prepare_rejects_missing_network_before_output(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = _synthetic_manifest(root)
            (root / "networks" / "pdtt.yml").unlink()
            with self.assertRaisesRegex(ValueError, "network configuration for pdtt"):
                prepare(manifest_path)
            self.assertFalse((root / "output").exists())

    def test_prepare_rejects_non_ring_algorithm_policy(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = _synthetic_manifest(root)
            manifest = _load(manifest_path)
            manifest["system_configuration_policy"]["all-to-all-implementation"] = [
                "direct"
            ]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "all-to-all-implementation"):
                prepare(manifest_path)
            self.assertFalse((root / "output").exists())

    def test_run_rejects_incomplete_structured_rank_set(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared_path = prepare(_synthetic_manifest(root))
            binary_dir = root / "bin"
            binary_dir.mkdir()
            for name in ("unaware", "aware"):
                binary = binary_dir / name
                binary.write_text("#!/bin/sh\n", encoding="utf-8")
                binary.chmod(0o755)

            def execute(command: list[str], **_: object) -> SimpleNamespace:
                if command[0] == "git":
                    return SimpleNamespace(returncode=0, stdout="test-revision\n", stderr="")
                stats_argument = next(
                    argument for argument in command if argument.startswith("--statistics-output=")
                )
                _write_statistics(
                    Path(stats_argument.split("=", 1)[1]),
                    backend="congestion_unaware",
                    topology="pt",
                    rank_count=127,
                )
                return SimpleNamespace(returncode=0, stdout="sim stdout", stderr="sim stderr")

            with patch("tons_sim.trace_pipeline.subprocess.run", side_effect=execute):
                with self.assertRaisesRegex(RuntimeError, "incomplete rank set"):
                    run(prepared_path, limit=1)


if __name__ == "__main__":
    unittest.main()
