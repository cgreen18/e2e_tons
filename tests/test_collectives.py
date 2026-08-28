from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from xml.etree import ElementTree as ET

from tons_collectives import (
    generate_collective_workload,
    generate_direct_alltoall,
    generate_fixed_route_alltoall,
    generate_pmcf_alltoall,
    lower_msccl_to_chakra,
    verify_schedule,
)


FIXTURES = Path(__file__).parent / "fixtures" / "topology_bundle"


class CollectiveScheduleTest(unittest.TestCase):
    def test_direct_alltoall_is_balanced_and_lowers(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            xml = generate_direct_alltoall(4, 2, root / "direct.xml")
            report = verify_schedule(xml)
            self.assertTrue(report.ok, report.errors)
            self.assertEqual(4 * 3 * 2, report.sends)
            self.assertEqual(report.sends, report.receives)
            schedules = lower_msccl_to_chakra(xml, root / "direct_schedule")
            self.assertEqual(4, len(schedules))
            self.assertTrue(all(path.stat().st_size > 0 for path in schedules))

    def test_fixed_route_pipeline_is_causal_and_balanced(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            xml = generate_fixed_route_alltoall(
                FIXTURES / "line3.paths", 2, root / "pipeline.xml"
            )
            report = verify_schedule(xml)
            self.assertTrue(report.ok, report.errors)
            self.assertEqual((4 * 1 + 2 * 2) * 2, report.sends)
            self.assertEqual(report.sends, report.receives)
            self.assertEqual(3, len(lower_msccl_to_chakra(xml, root / "pipeline_schedule")))

    def test_pmcf_line_schedule_solves_quantizes_and_lowers(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = generate_pmcf_alltoall(
                FIXTURES / "line3.map",
                FIXTURES / "line3.rallpaths",
                2,
                root / "pmcf.xml",
                threads=1,
            )
            self.assertAlmostEqual(0.5, result.maximum_concurrent_flow)
            self.assertEqual(4, result.quantized_maximum_link_load)
            report = verify_schedule(result.schedule)
            self.assertTrue(report.ok, report.errors)
            self.assertEqual(16, report.sends)
            self.assertEqual(3, len(lower_msccl_to_chakra(result.schedule, root / "pmcf_schedule")))

    def test_verifier_rejects_unmatched_endpoint(self) -> None:
        with TemporaryDirectory() as temporary:
            xml = generate_direct_alltoall(3, 1, Path(temporary) / "broken.xml")
            tree = ET.parse(xml)
            first_receive = tree.getroot().find("./gpu/tb/step[@type='r']")
            self.assertIsNotNone(first_receive)
            for tb in tree.getroot().findall("./gpu/tb"):
                if first_receive in list(tb):
                    tb.remove(first_receive)
                    break
            tree.write(xml, encoding="unicode")
            report = verify_schedule(xml)
            self.assertFalse(report.ok)
            self.assertTrue(any("matching receive" in error for error in report.errors))

    def test_workload_generator_writes_one_trace_per_rank(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for collective in ("allgather", "allreduce", "reduce_scatter", "alltoall"):
                paths = generate_collective_workload(root / collective, 4, collective, 1025)
                self.assertEqual(
                    [f"{collective}.{rank}.et" for rank in range(4)],
                    [path.name for path in paths],
                )
                self.assertTrue(all(path.stat().st_size > 0 for path in paths))


if __name__ == "__main__":
    unittest.main()
