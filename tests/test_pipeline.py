import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tons_sim.pipeline import _acceptance, _plots, prepare, run


REPOSITORY = Path(__file__).resolve().parents[1]
TOPOLOGY_BOUNDS = {"pt": 128, "pdtt": 74, "tons": 72}


def _result_row(
    topology: str,
    collective: str,
    mode: str,
    size: int,
    throughput: float,
    *,
    backend: str = "congestion_aware",
    queued_chunks: int = 0,
    total_queue_wait_ns: int = 0,
    max_queue_wait_ns: int = 0,
) -> dict[str, object]:
    return {
        "run_id": f"{topology}-{backend}-{collective}-{mode}-{size}",
        "stage": "primary",
        "topology": topology,
        "backend": backend,
        "collective": collective,
        "schedule_mode": mode,
        "bytes_per_rank": size,
        "throughput_GBps": throughput,
        "complete": True,
        "rank_count": 4,
        "ranks": 4,
        "queued_chunks": queued_chunks,
        "total_queue_wait_ns": total_queue_wait_ns,
        "max_queue_wait_ns": max_queue_wait_ns,
        "route_load_bound": TOPOLOGY_BOUNDS[topology],
    }


def _accepted_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for size in (16 * 2**20, 256 * 2**20):
        for topology, a2a_throughput in (("pt", 100.0), ("pdtt", 150.0), ("tons", 160.0)):
            for mode in ("direct", "fixed-route-pipeline", "pmcf"):
                queue_values = (4, 40, 10) if mode == "direct" else (0, 0, 0)
                rows.append(
                    _result_row(
                        topology,
                        "alltoall",
                        mode,
                        size,
                        a2a_throughput,
                        queued_chunks=queue_values[0],
                        total_queue_wait_ns=queue_values[1],
                        max_queue_wait_ns=queue_values[2],
                    )
                )
            for collective in ("allgather", "allreduce", "reduce_scatter"):
                throughput = 100.0 if topology == "pt" else 98.0
                rows.append(
                    _result_row(
                        topology, collective, "topology-aware", size, throughput
                    )
                )
    return rows


def _synthetic_manifest(
    root: Path, *, certification: str = "exact-monolithic-lp"
) -> tuple[Path, dict[str, SimpleNamespace], dict[str, Path]]:
    manifest = {
        "schema_version": 1,
        "repository_root": ".",
        "output_directory": "output",
        "topology_root": "topology",
        "l3ss_binary": "missing-l3ss",
        "binaries": {
            "congestion_unaware": "missing-unaware",
            "congestion_aware": "missing-aware",
        },
        "remote_memory_configuration": "missing-remote.json",
        "ranks": 4,
        "subchunks": 8,
        "pmcf": {"solver": "gurobi", "seed": 1},
        "sizes_bytes": [2**20, 16 * 2**20, 256 * 2**20],
        "backends": ["congestion_unaware", "congestion_aware"],
        "random_seed": 1,
        "smoke": {"bytes_per_rank": 2**20, "subchunks": 1},
        "network_profile": {"clock_GHz": 1.0},
        "topologies": {
            topology: {
                "bundle": f"{topology}-bundle",
                "route_policy": f"{topology}-route-policy",
            }
            for topology in TOPOLOGY_BOUNDS
        },
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    bundles: dict[str, SimpleNamespace] = {}
    reports: dict[str, Path] = {}
    for topology in TOPOLOGY_BOUNDS:
        bundle = SimpleNamespace(
            topology=root / "topology" / f"{topology}.map",
            routes=root / "topology" / f"{topology}.paths",
            pmcf_candidates=root / "topology" / f"{topology}.rallpaths",
            candidates=None,
        )
        bundles[f"{topology}-bundle"] = bundle
        schedule = root / "output" / "prepared" / "schedules" / topology / "alltoall-pmcf.xml"
        report = root / "output" / "prepared" / "reports" / f"{topology}.alltoall.pmcf.json"
        schedule.parent.mkdir(parents=True, exist_ok=True)
        report.parent.mkdir(parents=True, exist_ok=True)
        schedule.write_text("<algo />\n", encoding="utf-8")
        report.write_text(
            json.dumps(
                {
                    "schedule": str(schedule.resolve()),
                    "report": str(report.resolve()),
                    "ranks": 4,
                    "candidate_paths": 20,
                    "active_paths": 18,
                    "positive_paths": 12,
                    "maximum_concurrent_flow": 0.5,
                    "quantized_maximum_link_load": 16,
                    "epochs": 16,
                    "solver": "gurobi",
                    "solver_version": "test",
                    "iterations": 1,
                    "certification": certification,
                    "topology": str(bundle.topology.resolve()),
                    "candidates": str(bundle.pmcf_candidates.resolve()),
                    "subchunks": 8,
                    "seed": 1,
                }
            ),
            encoding="utf-8",
        )
        reports[topology] = report
    return manifest_path, bundles, reports


def _write_schedule(*args: object) -> Path:
    output = Path(args[-1])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("<algo />\n", encoding="utf-8")
    return output


class ExperimentPipelineTest(unittest.TestCase):
    def test_missing_prepared_result_is_blocking(self) -> None:
        result = _acceptance([], {"expected-run"})
        self.assertFalse(result["passed"])
        self.assertIn("no structured result", result["failures"][0])

    def test_dry_run_records_exact_command_without_binary(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = {
                "repository_root": str(REPOSITORY),
                "output_directory": str(root / "output"),
                "binaries": {"congestion_unaware": str(root / "missing-binary")},
                "remote_memory": str(root / "remote.json"),
                "jobs": [
                    {
                        "run_id": "smoke",
                        "backend": "congestion_unaware",
                        "workload": str(root / "workload"),
                        "system": str(root / "system.json"),
                        "network": str(root / "network.yml"),
                    }
                ],
            }
            prepared_path = root / "prepared.json"
            prepared_path.write_text(json.dumps(prepared), encoding="utf-8")
            run(prepared_path, dry_run=True)
            record = json.loads(
                (root / "output" / "runs" / "smoke" / "run.json").read_text(encoding="utf-8")
            )
            self.assertIn("--statistics-output=", record["command"][-1])
            self.assertEqual("smoke", record["run_id"])

    def test_prepare_and_dry_run_expand_exactly_111_jobs(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, bundles, _ = _synthetic_manifest(root)

            def validate(bundle: SimpleNamespace, **_: object) -> SimpleNamespace:
                topology = bundle.topology.stem
                return SimpleNamespace(
                    ok=True,
                    routers=4,
                    errors=[],
                    maximum_directed_channel_load=TOPOLOGY_BOUNDS[topology],
                    to_dict=lambda: {"ok": True},
                )

            def graph(*_: object, **kwargs: object) -> SimpleNamespace:
                return SimpleNamespace(
                    network_config=root / "networks" / f"{kwargs['stem']}.yml"
                )

            verified = SimpleNamespace(ok=True, errors=[], to_dict=lambda: {"ok": True})
            with (
                patch("tons_sim.pipeline.known_bundle", side_effect=lambda _, name: bundles[name]),
                patch("tons_sim.pipeline.validate_bundle", side_effect=validate),
                patch("tons_sim.pipeline.generate_graph_artifacts", side_effect=graph),
                patch("tons_sim.pipeline._l3ss_schedule", side_effect=_write_schedule),
                patch("tons_sim.pipeline.generate_direct_alltoall", side_effect=_write_schedule),
                patch("tons_sim.pipeline.generate_fixed_route_alltoall", side_effect=_write_schedule),
                patch("tons_sim.pipeline.verify_schedule", return_value=verified),
                patch("tons_sim.pipeline.lower_msccl_to_chakra", return_value=[]),
                patch("tons_sim.pipeline.generate_collective_workload", return_value=[]),
            ):
                prepared_path = prepare(manifest_path)

            prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
            jobs = prepared["jobs"]
            smoke = [job for job in jobs if job["stage"] == "smoke"]
            primary = [job for job in jobs if job["stage"] == "primary"]
            self.assertEqual(111, len(jobs))
            self.assertEqual(3, len(smoke))
            self.assertEqual(108, len(primary))
            self.assertEqual(set(TOPOLOGY_BOUNDS), {job["topology"] for job in smoke})
            self.assertTrue(all(job["subchunks"] == 1 for job in smoke))
            self.assertTrue(all(job["backend"] == "congestion_unaware" for job in smoke))

            combinations = {
                ("allgather", "topology-aware"),
                ("allreduce", "topology-aware"),
                ("reduce_scatter", "topology-aware"),
                ("alltoall", "direct"),
                ("alltoall", "fixed-route-pipeline"),
                ("alltoall", "pmcf"),
            }
            expected = {
                (topology, backend, collective, mode, size)
                for topology in TOPOLOGY_BOUNDS
                for backend in ("congestion_unaware", "congestion_aware")
                for collective, mode in combinations
                for size in (2**20, 16 * 2**20, 256 * 2**20)
            }
            actual = {
                (
                    job["topology"],
                    job["backend"],
                    job["collective"],
                    job["schedule_mode"],
                    job["bytes_per_rank"],
                )
                for job in primary
            }
            self.assertEqual(expected, actual)

            run_root = run(prepared_path, dry_run=True)
            self.assertEqual(111, len(list(run_root.glob("*/run.json"))))

    def test_prepare_blocks_missing_or_uncertified_pmcf_report(self) -> None:
        for failure in ("missing", "uncertified"):
            with self.subTest(failure=failure), TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifest_path, bundles, reports = _synthetic_manifest(root)
                if failure == "missing":
                    reports["pt"].unlink()
                    message = "missing certified pMCF report"
                else:
                    report = json.loads(reports["pt"].read_text(encoding="utf-8"))
                    report["certification"] = "not-certified"
                    reports["pt"].write_text(json.dumps(report), encoding="utf-8")
                    message = "uncertified pMCF report"
                with patch(
                    "tons_sim.pipeline.known_bundle",
                    side_effect=lambda _, name: bundles[name],
                ):
                    with self.assertRaisesRegex(RuntimeError, message):
                        prepare(manifest_path)

    def test_acceptance_compares_pdtt_and_tons_to_pt_and_blocks_reversal(self) -> None:
        rows = _accepted_rows()
        accepted = _acceptance(rows)
        self.assertTrue(accepted["passed"], accepted["failures"])
        comparisons = [
            check
            for check in accepted["checks"]
            if check["check"] == "alltoall_comparison"
        ]
        self.assertEqual({"pdtt", "tons"}, {check["topology"] for check in comparisons})
        references = {
            check["topology"]: check["route_load_reference"] for check in comparisons
        }
        self.assertAlmostEqual(128 / 74, references["pdtt"], places=12)
        self.assertAlmostEqual(128 / 72, references["tons"], places=12)
        equality_topologies = {
            check["topology"]
            for check in accepted["checks"]
            if check["check"] == "collective_equality"
        }
        self.assertEqual({"pdtt", "tons"}, equality_topologies)

        reversed_row = next(
            row
            for row in rows
            if row["topology"] == "pdtt"
            and row["collective"] == "alltoall"
            and row["schedule_mode"] == "pmcf"
            and row["bytes_per_rank"] == 256 * 2**20
        )
        reversed_row["throughput_GBps"] = 90.0
        rejected = _acceptance(rows)
        self.assertFalse(rejected["passed"])
        check = next(
            check
            for check in rejected["checks"]
            if check["check"] == "alltoall_comparison"
            and check["topology"] == "pdtt"
            and check["mode"] == "pmcf"
            and check["bytes_per_rank"] == 256 * 2**20
        )
        self.assertFalse(check["passed"])
        self.assertTrue(any("reversed versus PT" in failure for failure in rejected["failures"]))

    def test_acceptance_enforces_direct_and_scheduled_queue_policies(self) -> None:
        size = 16 * 2**20
        rows = [
            _result_row(
                "pt",
                "alltoall",
                "direct",
                size,
                100.0,
                queued_chunks=2,
                total_queue_wait_ns=20,
                max_queue_wait_ns=10,
            ),
            _result_row("pt", "alltoall", "fixed-route-pipeline", size, 100.0),
            _result_row("pt", "alltoall", "pmcf", size, 100.0),
        ]
        self.assertTrue(_acceptance(rows)["passed"])

        rows[0]["total_queue_wait_ns"] = 0
        self.assertTrue(
            any(
                "queue accounting is empty" in failure
                for failure in _acceptance(rows)["failures"]
            )
        )
        rows[0]["total_queue_wait_ns"] = 20
        rows[2]["max_queue_wait_ns"] = 1
        self.assertTrue(
            any(
                "scheduled A2A unexpectedly queued" in failure
                for failure in _acceptance(rows)["failures"]
            )
        )

    def test_acceptance_keeps_ag_ar_rs_within_five_percent(self) -> None:
        for collective in ("allgather", "allreduce", "reduce_scatter"):
            with self.subTest(collective=collective):
                size = 16 * 2**20
                rows = [
                    _result_row("pt", collective, "topology-aware", size, 100.0),
                    _result_row("pdtt", collective, "topology-aware", size, 95.0),
                ]
                self.assertTrue(_acceptance(rows)["passed"])
                rows[1]["throughput_GBps"] = 94.9
                result = _acceptance(rows)
                self.assertFalse(result["passed"])
                self.assertTrue(any("more than 5%" in failure for failure in result["failures"]))

    def test_alltoall_plot_has_three_schedule_panels_and_topology_labels(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.dict(os.environ, {"MPLCONFIGDIR": str(root / "matplotlib")}):
                import matplotlib

                matplotlib.use("Agg", force=True)
                import matplotlib.pyplot as plt

                rows = [
                    _result_row(
                        topology,
                        "alltoall",
                        mode,
                        16 * 2**20,
                        float(index + 1),
                        backend="congestion_unaware",
                    )
                    for index, (topology, mode) in enumerate(
                        (topology, mode)
                        for topology in TOPOLOGY_BOUNDS
                        for mode in ("direct", "fixed-route-pipeline", "pmcf")
                    )
                ]
                with patch.object(plt, "close"):
                    paths = _plots(rows, root / "runs", root / "analysis")
                    figures = [plt.figure(number) for number in plt.get_fignums()]
                    alltoall = next(figure for figure in figures if len(figure.axes) == 3)
                    self.assertEqual(
                        ["direct", "fixed-route-pipeline", "pmcf"],
                        [axis.get_title() for axis in alltoall.axes],
                    )
                    self.assertFalse(
                        alltoall.axes[0]
                        .get_shared_y_axes()
                        .joined(alltoall.axes[0], alltoall.axes[1])
                    )
                    for axis in alltoall.axes:
                        self.assertEqual(
                            set(TOPOLOGY_BOUNDS),
                            set(axis.get_legend_handles_labels()[1]),
                        )
                plt.close("all")
            self.assertTrue(
                any(path.endswith("throughput-alltoall-congestion_unaware.png") for path in paths)
            )


if __name__ == "__main__":
    unittest.main()
