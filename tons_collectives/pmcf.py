"""Path-based maximum-concurrent-flow all-to-all synthesis."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from .a2a import _compile_assigned_transfers, _generate_routed_alltoall


@dataclass(frozen=True)
class PmcfResult:
    schedule: Path
    report: Path
    ranks: int
    candidate_paths: int
    active_paths: int
    positive_paths: int
    maximum_concurrent_flow: float
    quantized_maximum_link_load: int
    epochs: int
    solver: str
    solver_version: str
    iterations: int
    certification: str
    topology: str
    candidates: str
    subchunks: int
    seed: int


@dataclass(frozen=True)
class PmcfSolution:
    """A solved pMCF instance plus the evidence needed to certify it."""

    values: dict[int, float]
    maximum_concurrent_flow: float
    active_paths: int
    iterations: int
    solver_version: str
    certification: str


def _certify(
    candidates: list[list[int]],
    by_commodity: dict[tuple[int, int], list[int]],
    values: dict[int, float],
    concurrent_value: float,
    *,
    tolerance: float = 1e-7,
) -> None:
    """Fail loudly unless the returned flow is feasible and concurrent.

    Column generation only certifies optimality over the finite candidate set;
    this independently re-checks the two primal properties a caller relies on,
    so a solver or bookkeeping regression cannot be published as a result.
    """

    load: dict[tuple[int, int], float] = defaultdict(float)
    for index, value in values.items():
        if value <= 0.0:
            continue
        route = candidates[index]
        for edge in zip(route, route[1:]):
            load[edge] += value
    if load:
        worst_edge, worst_load = max(load.items(), key=lambda item: item[1])
        if worst_load > 1.0 + tolerance:
            raise RuntimeError(
                f"pMCF solution violates unit capacity on {worst_edge[0]}->{worst_edge[1]}: "
                f"{worst_load:.12g}"
            )
    for commodity, indices in sorted(by_commodity.items()):
        carried = sum(max(0.0, values[index]) for index in indices)
        if abs(carried - concurrent_value) > tolerance:
            raise RuntimeError(
                f"pMCF commodity {commodity[0]}->{commodity[1]} carries {carried:.12g}, "
                f"expected the concurrent value {concurrent_value:.12g}"
            )


def _read_map(path: Path) -> list[list[int]]:
    rows = [
        [int(float(token)) for token in line.split()]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or any(len(row) != len(rows) for row in rows):
        raise ValueError(f"{path}: expected a non-empty square adjacency matrix")
    return rows


def _read_candidates(
    path: Path, topology: list[list[int]]
) -> tuple[list[list[int]], dict[tuple[int, int], list[int]], dict[tuple[int, int], list[int]]]:
    candidates: list[list[int]] = []
    by_commodity: dict[tuple[int, int], list[int]] = defaultdict(list)
    by_edge: dict[tuple[int, int], list[int]] = defaultdict(list)
    ranks = len(topology)
    with path.open(encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, start=1):
            if not raw.strip():
                continue
            try:
                route = [int(token) for token in raw.split()]
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: non-integer route") from exc
            if len(route) < 2 or route[0] == route[-1]:
                raise ValueError(f"{path}:{line_number}: invalid non-self route")
            if any(node < 0 or node >= ranks for node in route):
                raise ValueError(f"{path}:{line_number}: router outside [0, {ranks})")
            if len(set(route)) != len(route):
                raise ValueError(f"{path}:{line_number}: route is not simple")
            index = len(candidates)
            for source, destination in zip(route, route[1:]):
                if not topology[source][destination]:
                    raise ValueError(
                        f"{path}:{line_number}: non-physical edge {source}->{destination}"
                    )
                by_edge[(source, destination)].append(index)
            candidates.append(route)
            by_commodity[(route[0], route[-1])].append(index)
    expected = {(source, destination) for source in range(ranks) for destination in range(ranks) if source != destination}
    missing = expected - set(by_commodity)
    if missing:
        raise ValueError(f"{path}: missing {len(missing)} ordered commodities")
    return candidates, dict(by_commodity), dict(by_edge)


def _quantize(
    candidates: list[list[int]],
    by_commodity: dict[tuple[int, int], list[int]],
    values: dict[int, float],
    subchunks: int,
) -> list[tuple[int, int, int, list[int]]]:
    assignments: list[tuple[int, int, int, list[int]]] = []
    for (source, destination), indices in sorted(by_commodity.items()):
        total = sum(max(0.0, values[index]) for index in indices)
        if total <= 1e-12:
            raise RuntimeError(f"pMCF returned no flow for {source}->{destination}")
        raw = [(index, max(0.0, values[index]) * subchunks / total) for index in indices]
        allocation = {index: int(math.floor(value + 1e-15)) for index, value in raw}
        remaining = subchunks - sum(allocation.values())
        order = sorted(raw, key=lambda item: (-(item[1] - math.floor(item[1] + 1e-15)), item[0]))
        for index, _ in order[:remaining]:
            allocation[index] += 1
        q = 0
        for index in sorted(indices):
            for _ in range(allocation[index]):
                assignments.append((source, destination, q, candidates[index]))
                q += 1
        if q != subchunks:
            raise RuntimeError(f"pMCF quantization produced {q}/{subchunks} chunks for {source}->{destination}")
    return assignments


def generate_pmcf_alltoall(
    topology_path: Path | str,
    candidates_path: Path | str,
    subchunks: int,
    output: Path | str,
    *,
    report_path: Path | str | None = None,
    solver: str = "highs",
    threads: int = 16,
    seed: int = 1,
    reuse: bool = False,
) -> PmcfResult:
    """Solve pMCF, quantize it, and emit a causal hop-by-hop MSCCL schedule."""

    if subchunks < 1:
        raise ValueError("subchunks must be positive")
    topology_path = Path(topology_path)
    candidates_path = Path(candidates_path)
    output = Path(output)
    report_path = Path(report_path) if report_path is not None else output.with_suffix(".pmcf.json")
    if reuse and output.is_file() and report_path.is_file():
        cached = json.loads(report_path.read_text(encoding="utf-8"))
        expected = {
            "topology": str(topology_path.resolve()),
            "candidates": str(candidates_path.resolve()),
            "subchunks": subchunks,
            "solver": solver,
        }
        fields = set(PmcfResult.__dataclass_fields__)
        # A report written before the certification fields existed cannot be
        # trusted as a certified solve, so re-solve instead of adapting it.
        if fields <= set(cached) and all(
            cached.get(key) == value for key, value in expected.items()
        ):
            return PmcfResult(
                **{
                    key: (Path(cached[key]) if key in {"schedule", "report"} else cached[key])
                    for key in fields
                }
            )
    topology = _read_map(topology_path)
    ranks = len(topology)
    candidates, by_commodity, by_edge = _read_candidates(candidates_path, topology)

    if solver == "highs":
        solution = _solve_highs(candidates, by_commodity, by_edge, ranks)
    elif solver == "gurobi":
        solution = _solve_gurobi(
            candidates, by_commodity, by_edge, threads=threads, seed=seed
        )
    else:
        raise ValueError("pMCF solver must be 'highs' or 'gurobi'")
    values = solution.values
    concurrent_value = solution.maximum_concurrent_flow
    _certify(candidates, by_commodity, values, concurrent_value)
    assignments = _quantize(candidates, by_commodity, values, subchunks)
    quantized_load: Counter[tuple[int, int]] = Counter()
    for _, _, _, route in assignments:
        quantized_load.update(zip(route, route[1:]))
    transfers = _compile_assigned_transfers(assignments)
    schedule = _generate_routed_alltoall(
        transfers, ranks, subchunks, "tons_pmcf_alltoall", output
    )
    result = PmcfResult(
        schedule=schedule.resolve(),
        report=report_path.resolve(),
        ranks=ranks,
        candidate_paths=len(candidates),
        active_paths=solution.active_paths,
        positive_paths=sum(value > 1e-12 for value in values.values()),
        maximum_concurrent_flow=concurrent_value,
        quantized_maximum_link_load=max(quantized_load.values()),
        epochs=max((transfer.epoch for transfer in transfers), default=-1) + 1,
        solver=solver,
        solver_version=solution.solver_version,
        iterations=solution.iterations,
        certification=solution.certification,
        topology=str(topology_path.resolve()),
        candidates=str(candidates_path.resolve()),
        subchunks=subchunks,
        seed=seed,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps({**asdict(result), "schedule": str(result.schedule), "report": str(result.report)}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _solve_gurobi(
    candidates: list[list[int]],
    by_commodity: dict[tuple[int, int], list[int]],
    by_edge: dict[tuple[int, int], list[int]],
    *,
    threads: int,
    seed: int,
) -> PmcfSolution:
    try:
        import gurobipy as gp
    except ImportError as exc:
        raise RuntimeError("Gurobi pMCF synthesis requires gurobipy") from exc
    version = ".".join(str(part) for part in gp.gurobi.version())
    model = gp.Model("tons_pmcf")
    model.Params.OutputFlag = 0
    model.Params.Threads = max(1, threads)
    model.Params.Seed = seed
    flows = model.addVars(len(candidates), lb=0.0, name="flow")
    concurrent = model.addVar(lb=0.0, name="concurrent_flow")
    for indices in by_edge.values():
        model.addConstr(gp.quicksum(flows[index] for index in indices) <= 1.0)
    for indices in by_commodity.values():
        model.addConstr(gp.quicksum(flows[index] for index in indices) == concurrent)
    model.setObjective(concurrent, gp.GRB.MAXIMIZE)
    model.optimize()
    if model.Status != gp.GRB.OPTIMAL:
        raise RuntimeError(f"pMCF did not reach optimal status (status={model.Status})")
    return PmcfSolution(
        values={index: float(flows[index].X) for index in range(len(candidates))},
        maximum_concurrent_flow=float(concurrent.X),
        active_paths=len(candidates),
        iterations=1,
        solver_version=version,
        # Gurobi builds every finite candidate column at once, so an OPTIMAL
        # status is already an exact optimum over the candidate set.
        certification="exact-monolithic-lp",
    )


def _solve_highs(
    candidates: list[list[int]],
    by_commodity: dict[tuple[int, int], list[int]],
    by_edge: dict[tuple[int, int], list[int]],
    ranks: int,
) -> PmcfSolution:
    try:
        import numpy as np
        import scipy
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except ImportError as exc:
        raise RuntimeError("HiGHS pMCF synthesis requires scipy and numpy") from exc
    version = f"scipy-{scipy.__version__}-highs-ds"

    path_count = len(candidates)
    edge_rows = {edge: row for row, edge in enumerate(sorted(by_edge))}
    commodities = sorted(by_commodity)
    commodity_rows = {commodity: row for row, commodity in enumerate(commodities)}

    # Exact Dantzig-Wolfe column generation over the finite candidate set.
    # Start from one shortest legal path per commodity, then price every
    # inactive candidate with the restricted master's dual edge costs.
    active: set[int] = {
        min(indices, key=lambda index: (len(candidates[index]), index))
        for indices in by_commodity.values()
    }
    result = None
    active_list: list[int] = []
    for iteration in range(1, 257):
        active_list = sorted(active)
        column_of = {path_index: column for column, path_index in enumerate(active_list)}
        concurrent_index = len(active_list)
        ub_rows: list[int] = []
        ub_cols: list[int] = []
        eq_rows: list[int] = []
        eq_cols: list[int] = []
        eq_data: list[float] = []
        for path_index in active_list:
            column = column_of[path_index]
            route = candidates[path_index]
            for edge in zip(route, route[1:]):
                ub_rows.append(edge_rows[edge])
                ub_cols.append(column)
            eq_rows.append(commodity_rows[(route[0], route[-1])])
            eq_cols.append(column)
            eq_data.append(1.0)
        for row in range(len(commodities)):
            eq_rows.append(row)
            eq_cols.append(concurrent_index)
            eq_data.append(-1.0)
        a_ub = coo_matrix(
            (np.ones(len(ub_rows)), (ub_rows, ub_cols)),
            shape=(len(edge_rows), len(active_list) + 1),
        ).tocsr()
        a_eq = coo_matrix(
            (eq_data, (eq_rows, eq_cols)),
            shape=(len(commodities), len(active_list) + 1),
        ).tocsr()
        objective = np.zeros(len(active_list) + 1)
        objective[concurrent_index] = -1.0
        result = linprog(
            objective,
            A_ub=a_ub,
            b_ub=np.ones(len(edge_rows)),
            A_eq=a_eq,
            b_eq=np.zeros(len(commodities)),
            bounds=(0.0, None),
            method="highs-ds",
        )
        if not result.success:
            raise RuntimeError(f"HiGHS pMCF restricted master failed: {result.message}")

        # SciPy/HiGHS marginals are d(objective)/d(rhs) for the minimization
        # form, so capacity duals are non-positive and commodity duals are
        # non-negative.  The reduced cost of an inactive path column p of
        # commodity k is therefore
        #     rc(p) = 0 - (y_eq[k] + sum_{e in p} y_ub[e])
        #           = length(p) - price(k)
        # with the non-negative edge length length(e) = -y_ub[e] and the
        # non-negative commodity price price(k) = y_eq[k].  Pricing in a path
        # shorter than its commodity's price is what improves the master; using
        # the unnegated sum prices in the *longest* paths and terminates while
        # genuinely improving columns remain (see the diamond regression test).
        edge_length = {edge: -float(result.ineqlin.marginals[row]) for edge, row in edge_rows.items()}
        commodity_price = {
            commodity: float(result.eqlin.marginals[row])
            for commodity, row in commodity_rows.items()
        }
        improving: dict[tuple[int, int], list[tuple[float, int]]] = defaultdict(list)
        for path_index, route in enumerate(candidates):
            if path_index in active:
                continue
            commodity = (route[0], route[-1])
            reduced_cost = sum(
                edge_length[edge] for edge in zip(route, route[1:])
            ) - commodity_price[commodity]
            if reduced_cost < -1e-8:
                improving[commodity].append((reduced_cost, path_index))
        print(
            f"pMCF column generation iteration {iteration}: "
            f"active={len(active_list)} objective={result.x[concurrent_index]:.12g} "
            f"improving_commodities={len(improving)}",
            flush=True,
        )
        if not improving:
            values = {index: 0.0 for index in range(path_count)}
            for column, path_index in enumerate(active_list):
                values[path_index] = float(result.x[column])
            return PmcfSolution(
                values=values,
                maximum_concurrent_flow=float(result.x[concurrent_index]),
                active_paths=len(active_list),
                iterations=iteration,
                solver_version=version,
                # The loop only exits after pricing *every* finite candidate
                # path and finding no negative reduced cost, which is the
                # Dantzig-Wolfe optimality certificate for this candidate set.
                certification="exact-column-generation-no-improving-column",
            )
        # Eight paths match the experiment's quantization granularity while
        # substantially reducing the number of restricted-master rounds.  The
        # final no-negative-column scan still certifies the exact LP optimum.
        for candidates_for_commodity in improving.values():
            active.update(
                path_index
                for _, path_index in sorted(candidates_for_commodity)[:8]
            )
    raise RuntimeError("pMCF column generation did not converge within 256 iterations")
