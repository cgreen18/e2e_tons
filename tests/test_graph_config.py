from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from tons_topology.graph_config import (
    PAPER_1_GHZ_PROFILE,
    TopologyDimensions,
    classify_directed_edge,
    generate_graph_artifacts,
)


REPOSITORY = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPOSITORY / "topology_fixtures" / "tons_128"


class EdgeClassificationTest(unittest.TestCase):
    def test_in_cube_neighbors_are_electrical(self) -> None:
        self.assertEqual("electrical", classify_directed_edge(0, 1))
        self.assertEqual("electrical", classify_directed_edge(0, 4))
        self.assertEqual("electrical", classify_directed_edge(0, 16))

    def test_cube_boundary_or_non_neighbor_is_optical(self) -> None:
        self.assertEqual("optical", classify_directed_edge(3, 0))
        self.assertEqual("optical", classify_directed_edge(0, 64))

    def test_profile_matches_one_ghz_contract(self) -> None:
        self.assertEqual(128.0, PAPER_1_GHZ_PROFILE.bandwidth_GBps)
        self.assertEqual(75.0, PAPER_1_GHZ_PROFILE.electrical_latency_ns)
        self.assertEqual(50.0, PAPER_1_GHZ_PROFILE.optical_latency_ns)
        self.assertEqual(25.0, PAPER_1_GHZ_PROFILE.injection_latency_ns)


class FullGraphConfigTest(unittest.TestCase):
    def test_target_topologies_have_expected_link_classes(self) -> None:
        for bundle in ("pt-dor-128", "pdtt-128", "tons-128"):
            with self.subTest(bundle=bundle), tempfile.TemporaryDirectory() as temporary:
                from tons_topology import known_bundle

                paths = known_bundle(ARTIFACT_ROOT, bundle)
                artifacts = generate_graph_artifacts(
                    paths.topology,
                    paths.routes,
                    temporary,
                    stem=bundle,
                )
                self.assertEqual(576, artifacts.electrical_directed_edges)
                self.assertEqual(192, artifacts.optical_directed_edges)
                with artifacts.edge_properties.open(newline="") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(768, len(rows))
                self.assertIn("topology: [ Graph ]", artifacts.network_config.read_text())
                self.assertIn("injection-latency: 25.0", artifacts.network_config.read_text())

    def test_pdtt_electrical_links_are_exactly_the_fixed_cube_mesh(self) -> None:
        from tons_topology import known_bundle

        dimensions = TopologyDimensions()
        paths = known_bundle(ARTIFACT_ROOT, "pdtt-128")
        rows = [
            [int(float(token)) for token in line.split()]
            for line in paths.topology.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        pdtt_edges = {
            (source, destination)
            for source, row in enumerate(rows)
            for destination, connected in enumerate(row)
            if connected
        }

        fixed_mesh_edges = set()
        for source in range(dimensions.routers):
            source_coordinates = (
                source % dimensions.x,
                (source // dimensions.x) % dimensions.y,
                source // (dimensions.x * dimensions.y),
            )
            for axis in range(3):
                for step in (-1, 1):
                    destination_coordinates = list(source_coordinates)
                    destination_coordinates[axis] += step
                    limits = (dimensions.x, dimensions.y, dimensions.z)
                    if not 0 <= destination_coordinates[axis] < limits[axis]:
                        continue
                    if (
                        destination_coordinates[axis] // dimensions.cube
                        != source_coordinates[axis] // dimensions.cube
                    ):
                        continue
                    x, y, z = destination_coordinates
                    destination = x + dimensions.x * y + dimensions.x * dimensions.y * z
                    fixed_mesh_edges.add((source, destination))

        self.assertEqual(576, len(fixed_mesh_edges))
        self.assertEqual(set(), fixed_mesh_edges - pdtt_edges)
        self.assertEqual(192, len(pdtt_edges - fixed_mesh_edges))
        self.assertTrue(
            all(classify_directed_edge(*edge) == "electrical" for edge in fixed_mesh_edges)
        )
        self.assertTrue(
            all(
                classify_directed_edge(*edge) == "optical"
                for edge in pdtt_edges - fixed_mesh_edges
            )
        )


if __name__ == "__main__":
    unittest.main()
