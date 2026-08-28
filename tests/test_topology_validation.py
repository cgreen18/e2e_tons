from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tons_topology.validation import BundlePaths, known_bundle, validate_bundle


REPOSITORY = Path(__file__).resolve().parents[1]
FIXTURE = REPOSITORY / "tests" / "fixtures" / "topology_bundle"
ARTIFACT_ROOT = REPOSITORY / "topologies_and_routing"


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


@unittest.skipUnless((ARTIFACT_ROOT / "topo_maps").is_dir(), "artifact symlink unavailable")
class TargetBundleIntegrationTest(unittest.TestCase):
    def test_target_128_router_bundles(self) -> None:
        expectations = {
            "pt-dor-128": (128, 4.031496062992126),
            "pdtt-128": (74, 3.4713336614173227),
            "tons-128": (72, 3.375),
        }
        for name, (maximum_load, average_hops) in expectations.items():
            with self.subTest(bundle=name):
                report = validate_bundle(
                    known_bundle(ARTIFACT_ROOT, name),
                    expected_degree=6,
                    destination_based=True,
                )
                self.assertEqual([], report.errors)
                self.assertEqual(128, report.routers)
                self.assertEqual(maximum_load, report.maximum_directed_channel_load)
                self.assertAlmostEqual(average_hops, report.average_hops, places=5)


if __name__ == "__main__":
    unittest.main()
