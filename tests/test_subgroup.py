"""Reduced sub-maps and the structural (broadcast/gather) schedules."""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from xml.etree import ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tons_collectives.chakra import lower_msccl_to_chakra
from tons_collectives.subgroup import (
    generate_broadcast,
    generate_gather,
    hop_distances,
    member_submap,
    write_map,
)
from tons_collectives.verify import verify_schedule


def _ring(size: int) -> list[list[int]]:
    adjacency = [[0] * size for _ in range(size)]
    for node in range(size):
        adjacency[node][(node + 1) % size] = 1
        adjacency[node][(node - 1) % size] = 1
    return adjacency


class SubmapTest(unittest.TestCase):
    def test_hop_distances_on_a_ring(self) -> None:
        self.assertEqual(hop_distances(_ring(6), 0), [0, 1, 2, 3, 2, 1])

    def test_contiguous_members_use_the_induced_subgraph(self) -> None:
        matrix, radius = member_submap(_ring(8), [0, 1, 2, 3], "proximity")
        self.assertEqual(radius, 1)
        self.assertEqual(
            matrix,
            [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]],
        )
        induced, _ = member_submap(_ring(8), [0, 1, 2, 3], "induced")
        self.assertEqual(induced, matrix)

    def test_scattered_members_need_a_wider_radius(self) -> None:
        # Every other node of an 8-ring: no two members are adjacent.
        members = [0, 2, 4, 6]
        with self.assertRaises(ValueError):
            member_submap(_ring(8), members, "induced")
        matrix, radius = member_submap(_ring(8), members, "proximity")
        self.assertEqual(radius, 2)
        # Radius 2 makes it a ring over the four members.
        self.assertEqual(
            matrix,
            [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]],
        )

    def test_complete_mode_is_topology_blind(self) -> None:
        matrix, radius = member_submap(_ring(8), [0, 2, 5], "complete")
        self.assertEqual(radius, -1)
        self.assertEqual(matrix, [[0, 1, 1], [1, 0, 1], [1, 1, 0]])

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            member_submap(_ring(4), [0, 0], "proximity")
        with self.assertRaises(ValueError):
            member_submap(_ring(4), [0, 9], "proximity")
        with self.assertRaises(ValueError):
            member_submap(_ring(4), [], "proximity")
        with self.assertRaises(ValueError):
            member_submap(_ring(4), [0, 1], "nonsense")

    def test_written_map_round_trips(self) -> None:
        with TemporaryDirectory() as directory:
            matrix, _ = member_submap(_ring(8), [0, 1, 2, 3], "proximity")
            path = write_map(matrix, Path(directory) / "sub.map")
            rows = [
                [int(value) for value in line.split()]
                for line in path.read_text().strip().splitlines()
            ]
            self.assertEqual(rows, matrix)


class StructuralScheduleTest(unittest.TestCase):
    def _peers(self, xml: Path) -> dict[int, tuple[list[int], list[int]]]:
        root = ET.parse(xml).getroot()
        peers: dict[int, tuple[list[int], list[int]]] = {}
        for gpu in root.findall("gpu"):
            sends, receives = [], []
            for tb in gpu.findall("tb"):
                if int(tb.attrib["send"]) >= 0:
                    sends.append(int(tb.attrib["send"]))
                if int(tb.attrib["recv"]) >= 0:
                    receives.append(int(tb.attrib["recv"]))
            peers[int(gpu.attrib["id"])] = (sorted(sends), sorted(receives))
        return peers

    def test_broadcast_is_one_to_all(self) -> None:
        with TemporaryDirectory() as directory:
            xml = generate_broadcast(4, Path(directory) / "bcast.xml")
            report = verify_schedule(xml)
            self.assertTrue(report.ok, report.errors)
            self.assertEqual(report.collective, "broadcast")
            self.assertEqual((report.sends, report.receives), (3, 3))
            # No reduction compute is emitted.
            self.assertEqual(report.reductions, 0)
            peers = self._peers(xml)
            self.assertEqual(peers[0], ([1, 2, 3], []))
            for rank in (1, 2, 3):
                self.assertEqual(peers[rank], ([], [0]))

    def test_gather_is_all_to_one(self) -> None:
        with TemporaryDirectory() as directory:
            xml = generate_gather(4, Path(directory) / "gather.xml")
            report = verify_schedule(xml)
            self.assertTrue(report.ok, report.errors)
            # A REDUCE lowered without its reduction is named a gather, so the
            # verifier's reduce-op requirement stays meaningful elsewhere.
            self.assertEqual(report.collective, "gather")
            self.assertEqual((report.sends, report.receives), (3, 3))
            self.assertEqual(report.reductions, 0)
            peers = self._peers(xml)
            self.assertEqual(peers[0], ([], [1, 2, 3]))
            for rank in (1, 2, 3):
                self.assertEqual(peers[rank], ([0], []))

    def test_structural_schedules_lower_to_one_et_per_rank(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for generator, name in ((generate_broadcast, "b"), (generate_gather, "g")):
                xml = generator(6, root / f"{name}.xml")
                lowered = lower_msccl_to_chakra(xml, root / name)
                self.assertEqual(len(lowered), 6)

    def test_whole_buffer_is_one_logical_chunk(self) -> None:
        # Each send must carry the invoking node's full comm_size, so the
        # schedule declares exactly one workload chunk.
        with TemporaryDirectory() as directory:
            xml = generate_broadcast(3, Path(directory) / "bcast.xml")
            root = ET.parse(xml).getroot()
            self.assertEqual(root.attrib["input_chunks"], "1")
            for step in root.iter("step"):
                self.assertEqual(step.attrib["cnt"], "1")
                self.assertEqual(step.attrib["logical_chunk"], "0")

    def test_degenerate_sizes_and_roots_are_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "x.xml"
            with self.assertRaises(ValueError):
                generate_broadcast(1, path)
            with self.assertRaises(ValueError):
                generate_gather(1, path)
            with self.assertRaises(ValueError):
                generate_broadcast(4, path, root_rank=4)


if __name__ == "__main__":
    unittest.main()
