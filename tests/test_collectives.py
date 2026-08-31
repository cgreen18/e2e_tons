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
from tons_collectives.pmcf import _read_candidates, _read_map, _solve_highs


FIXTURES = Path(__file__).parent / "fixtures" / "topology_bundle"


def _reference_concurrent_flow(
    candidates: list[list[int]], subset: list[int] | None = None
) -> tuple[float, dict[int, float]]:
    """Solve the pMCF LP directly, without column generation.

    This is written independently of ``tons_collectives.pmcf`` on purpose: it
    enumerates every candidate column up front, so it is a cross-check of the
    Dantzig-Wolfe implementation rather than a re-use of it.  Only viable for
    small fixtures.
    """

    import numpy as np
    from scipy.optimize import linprog

    columns = list(range(len(candidates))) if subset is None else sorted(subset)
    edges = sorted(
        {edge for index in columns for edge in zip(candidates[index], candidates[index][1:])}
    )
    commodities = sorted({(candidates[index][0], candidates[index][-1]) for index in columns})
    edge_row = {edge: row for row, edge in enumerate(edges)}
    commodity_row = {commodity: row for row, commodity in enumerate(commodities)}

    flow_count = len(columns)
    concurrent = flow_count
    a_ub = np.zeros((len(edges), flow_count + 1))
    a_eq = np.zeros((len(commodities), flow_count + 1))
    for column, index in enumerate(columns):
        route = candidates[index]
        for edge in zip(route, route[1:]):
            a_ub[edge_row[edge], column] = 1.0
        a_eq[commodity_row[(route[0], route[-1])], column] = 1.0
    for row in range(len(commodities)):
        a_eq[row, concurrent] = -1.0

    cost = np.zeros(flow_count + 1)
    cost[concurrent] = -1.0
    solved = linprog(
        cost,
        A_ub=a_ub,
        b_ub=np.ones(len(edges)),
        A_eq=a_eq,
        b_eq=np.zeros(len(commodities)),
        bounds=(0.0, None),
        method="highs-ds",
    )
    if not solved.success:
        raise AssertionError(f"reference pMCF LP failed: {solved.message}")
    values = {index: float(solved.x[column]) for column, index in enumerate(columns)}
    return float(solved.x[concurrent]), values


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

    def test_pmcf_diamond_column_generation_equals_direct_lp(self) -> None:
        """Column generation must reach the exact finite-candidate LP optimum.

        ``diamond6`` is K(2,4): four leaves, each attached to both hubs.  Every
        leaf-to-leaf commodity therefore has exactly two two-hop routes, one per
        hub, and the shortest-path tie-break that seeds the restricted master
        sends all of them through the lower-numbered hub.  Reaching the optimum
        requires correctly pricing in the second hub's columns, so this fixture
        fails if the reduced-cost sign convention is wrong -- which the
        single-path line fixture cannot detect.
        """

        topology = _read_map(FIXTURES / "diamond6.map")
        candidates, by_commodity, by_edge = _read_candidates(
            FIXTURES / "diamond6.rallpaths", topology
        )
        reference, _ = _reference_concurrent_flow(candidates)

        # Guard the guard: if the seeded shortest-path columns alone were
        # already optimal, this test could pass without exercising pricing.
        seed_columns = [
            min(indices, key=lambda index: (len(candidates[index]), index))
            for indices in by_commodity.values()
        ]
        seeded, _ = _reference_concurrent_flow(candidates, seed_columns)
        self.assertLess(seeded + 1e-9, reference)

        solution = _solve_highs(candidates, by_commodity, by_edge, len(topology))
        self.assertAlmostEqual(reference, solution.maximum_concurrent_flow, places=9)
        self.assertEqual(
            "exact-column-generation-no-improving-column", solution.certification
        )
        self.assertGreater(solution.active_paths, len(seed_columns))
        self.assertLessEqual(solution.active_paths, len(candidates))

        # The solution must be a feasible concurrent flow, not just the right
        # objective value: unit directed-link capacity and equal per-commodity
        # flow.
        load: dict[tuple[int, int], float] = {}
        for index, value in solution.values.items():
            if value <= 1e-12:
                continue
            route = candidates[index]
            for edge in zip(route, route[1:]):
                load[edge] = load.get(edge, 0.0) + value
        self.assertLessEqual(max(load.values()), 1.0 + 1e-7)
        for commodity, indices in by_commodity.items():
            carried = sum(solution.values[index] for index in indices)
            self.assertAlmostEqual(
                reference, carried, places=7, msg=f"commodity {commodity}"
            )

    def test_pmcf_diamond_schedule_is_certified_and_lowers(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = generate_pmcf_alltoall(
                FIXTURES / "diamond6.map",
                FIXTURES / "diamond6.rallpaths",
                8,
                root / "pmcf.xml",
                threads=1,
            )
            self.assertAlmostEqual(4 / 11, result.maximum_concurrent_flow, places=9)
            self.assertEqual(
                "exact-column-generation-no-improving-column", result.certification
            )
            self.assertEqual(144, result.candidate_paths)
            self.assertGreater(result.iterations, 1)
            self.assertEqual(1, result.seed)
            self.assertTrue(result.solver_version.startswith("scipy-"))
            report = verify_schedule(result.schedule)
            self.assertTrue(report.ok, report.errors)
            # 30 ordered commodities x 8 subchunks, each hop a distinct send.
            self.assertEqual(report.sends, report.receives)
            self.assertEqual(6, len(lower_msccl_to_chakra(result.schedule, root / "sched")))

    def test_pmcf_column_generation_is_deterministic(self) -> None:
        topology = _read_map(FIXTURES / "diamond6.map")
        candidates, by_commodity, by_edge = _read_candidates(
            FIXTURES / "diamond6.rallpaths", topology
        )
        first = _solve_highs(candidates, by_commodity, by_edge, len(topology))
        second = _solve_highs(candidates, by_commodity, by_edge, len(topology))
        self.assertEqual(first.active_paths, second.active_paths)
        self.assertEqual(first.iterations, second.iterations)
        self.assertEqual(first.values, second.values)

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
