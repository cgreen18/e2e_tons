from __future__ import annotations

import ast
from dataclasses import replace
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tons_topology.validation import BundlePaths, known_bundle, validate_bundle


REPOSITORY = Path(__file__).resolve().parents[1]
FIXTURE = REPOSITORY / "tests" / "fixtures" / "topology_bundle"
ARTIFACT_ROOT = REPOSITORY / "topology_fixtures" / "tons_128"


def fixture_bundle(root: Path = FIXTURE) -> BundlePaths:
    return BundlePaths(
        name="line3",
        topology=root / "line3.map",
        candidates=root / "line3.rallpaths",
        routes=root / "line3.paths",
        next_hops=root / "line3.nrl2",
        vc_matrix=root / "line3.vcmat2",
        allowed_turns=root / "line3.allowvcturns",
    )


class TopologyValidationTest(unittest.TestCase):
    def test_complete_small_bundle(self) -> None:
        report = validate_bundle(fixture_bundle(), destination_based=True)

        self.assertEqual([], report.errors)
        self.assertTrue(report.ok)
        self.assertEqual(3, report.routers)
        self.assertEqual(6, report.selected_flows)
        self.assertEqual(8, report.selected_hops)
        self.assertAlmostEqual(4 / 3, report.average_hops)
        self.assertEqual(2, report.maximum_directed_channel_load)
        self.assertEqual(0.5, report.unit_flow_throughput_bound)
        self.assertEqual(6, report.candidate_paths)
        self.assertEqual(1, report.virtual_channels)

    def test_non_physical_selected_hop_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copytree(FIXTURE, root, dirs_exist_ok=True)
            route_file = root / "line3.paths"
            route_file.write_text(
                route_file.read_text(encoding="utf-8").replace("[0, 1, 2]", "[0, 2]"),
                encoding="utf-8",
            )

            report = validate_bundle(fixture_bundle(root), destination_based=True)

        self.assertFalse(report.ok)
        self.assertTrue(any("uses non-edge 0->2" in error for error in report.errors))

    def test_disallowed_vc_turn_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copytree(FIXTURE, root, dirs_exist_ok=True)
            allowed_file = root / "line3.allowvcturns"
            allowed_file.write_text(
                allowed_file.read_text(encoding="utf-8").replace(
                    "((0, 1, 0), (1, 2, 0)) : True",
                    "((0, 1, 0), (1, 2, 0)) : False",
                ),
                encoding="utf-8",
            )

            report = validate_bundle(fixture_bundle(root), destination_based=True)

        self.assertFalse(report.ok)
        self.assertTrue(any("uses disallowed turn" in error for error in report.errors))

    def test_candidate_without_end_to_end_vc_assignment_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copytree(FIXTURE, root, dirs_exist_ok=True)
            map_file = root / "line3.map"
            map_file.write_text(
                "0 1 1\n1 0 1\n1 1 0\n",
                encoding="utf-8",
            )
            candidates_file = root / "line3.rallpaths"
            candidates_file.write_text(
                candidates_file.read_text(encoding="utf-8") + "0 2 1\n",
                encoding="utf-8",
            )

            report = validate_bundle(fixture_bundle(root), destination_based=True)

        self.assertFalse(report.ok)
        self.assertIn(
            "1 candidate routes have no allowed end-to-end VC assignment",
            report.errors,
        )

    def test_standalone_cli(self) -> None:
        command = [
            sys.executable,
            str(REPOSITORY / "tools" / "validate_topology_bundle.py"),
            "--map",
            str(FIXTURE / "line3.map"),
            "--routes",
            str(FIXTURE / "line3.paths"),
            "--candidates",
            str(FIXTURE / "line3.rallpaths"),
            "--next-hops",
            str(FIXTURE / "line3.nrl2"),
            "--vc-matrix",
            str(FIXTURE / "line3.vcmat2"),
            "--allowed-turns",
            str(FIXTURE / "line3.allowvcturns"),
            "--destination-based",
            "--json",
        ]
        completed = subprocess.run(command, check=False, capture_output=True, text=True)

        self.assertEqual(0, completed.returncode, completed.stderr)
        self.assertIn('"ok": true', completed.stdout)


class TargetBundleIntegrationTest(unittest.TestCase):
    def test_target_128_router_bundles(self) -> None:
        expectations = {
            "pt-dor-128": (65536, 128, 4.031496062992126, None, None),
            "pdtt-128": (56430, 74, 3.4713336614173227, 171960, 2),
            "tons-128": (54864, 72, 3.375, 152917, 2),
        }
        for name, expectation in expectations.items():
            with self.subTest(bundle=name):
                selected_hops, maximum_load, average_hops, candidates, vcs = expectation
                report = validate_bundle(
                    known_bundle(ARTIFACT_ROOT, name),
                    expected_degree=6,
                    destination_based=True,
                )
                self.assertEqual([], report.errors)
                self.assertEqual(128, report.routers)
                self.assertEqual(768, report.directed_links)
                self.assertEqual((6, 6), (report.min_degree, report.max_degree))
                self.assertEqual(16256, report.selected_flows)
                self.assertEqual(selected_hops, report.selected_hops)
                self.assertEqual(maximum_load, report.maximum_directed_channel_load)
                self.assertEqual(average_hops, report.average_hops)
                self.assertEqual(candidates, report.candidate_paths)
                self.assertEqual(vcs, report.virtual_channels)

    def test_pdtt_uses_tons_artifact_naming_pattern(self) -> None:
        pdtt = known_bundle(ARTIFACT_ROOT, "pdtt-128")
        tons = known_bundle(ARTIFACT_ROOT, "tons-128")
        self.assertEqual("pdtt_2c_128r_6p_4x4x8", pdtt.topology.stem)

        for role in (
            "topology",
            "candidates",
            "routes",
            "next_hops",
            "vc_matrix",
            "allowed_turns",
            "pmcf_candidates",
        ):
            with self.subTest(role=role):
                pdtt_path = getattr(pdtt, role)
                tons_path = getattr(tons, role)
                self.assertIsNotNone(pdtt_path)
                self.assertIsNotNone(tons_path)
                assert pdtt_path is not None and tons_path is not None
                self.assertEqual(pdtt_path.parent.name, tons_path.parent.name)
                self.assertEqual(
                    pdtt_path.name[len(pdtt.topology.stem) :],
                    tons_path.name[len(tons.topology.stem) :],
                )

        vc_ids = {
            ast.literal_eval(line)[3]
            for line in pdtt.vc_matrix.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        self.assertEqual({0, 1}, vc_ids)

    def test_pdtt_route_vc_alignment_and_selected_cdg_checks_are_active(self) -> None:
        pdtt = known_bundle(ARTIFACT_ROOT, "pdtt-128")
        focused = replace(pdtt, candidates=None, next_hops=None, allowed_turns=None)
        original_records = pdtt.vc_matrix.read_text(encoding="utf-8").splitlines()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            misaligned = root / "pdtt-misaligned.vcmat2"
            swapped_records = list(original_records)
            swapped_records[:2] = reversed(swapped_records[:2])
            misaligned.write_text("\n".join(swapped_records) + "\n", encoding="utf-8")
            alignment_report = validate_bundle(replace(focused, vc_matrix=misaligned))

            cyclic = root / "pdtt-cyclic.vcmat2"
            cyclic.write_text(
                "\n".join(
                    str((*ast.literal_eval(record)[:3], 0))
                    for record in original_records
                    if record.strip()
                )
                + "\n",
                encoding="utf-8",
            )
            cycle_report = validate_bundle(replace(focused, vc_matrix=cyclic))

        self.assertIn(
            "VC matrix differs from selected-route hop order at record 0 "
            "(56558 records versus 56558 expected)",
            alignment_report.errors,
        )
        self.assertIn(
            "selected-route channel-dependency graph contains a cycle",
            cycle_report.errors,
        )


if __name__ == "__main__":
    unittest.main()
