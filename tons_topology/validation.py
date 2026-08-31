"""Cross-file validation for topology, routing, and VC artifact bundles.

The parsers intentionally use only the Python standard library.  This keeps the
validator usable before the simulation and solver environments are installed.
"""

from __future__ import annotations

import ast
import json
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable


Channel = tuple[int, int, int]
Turn = tuple[Channel, Channel]


@dataclass(frozen=True)
class BundlePaths:
    """Files that together describe one fixed-routing topology."""

    topology: Path
    routes: Path
    candidates: Path | None = None
    next_hops: Path | None = None
    vc_matrix: Path | None = None
    allowed_turns: Path | None = None
    pmcf_candidates: Path | None = None
    name: str | None = None


@dataclass
class ValidationReport:
    """Machine-readable result returned even when semantic checks fail."""

    name: str
    files: dict[str, str]
    errors: list[str] = field(default_factory=list)
    routers: int = 0
    directed_links: int = 0
    min_degree: int = 0
    max_degree: int = 0
    selected_flows: int = 0
    selected_hops: int = 0
    average_hops: float = 0.0
    maximum_directed_channel_load: int = 0
    unit_flow_throughput_bound: float = 0.0
    candidate_paths: int | None = None
    virtual_channels: int | None = None
    allowed_turn_entries: int | None = None

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["ok"] = self.ok
        return result


class ArtifactFormatError(ValueError):
    """Raised when an artifact cannot be parsed safely."""


_KNOWN_BUNDLES = {
    # Kept for compatibility with the original validator.  The initial TONS
    # experiments use ``pt-dor-128`` below.
    "pt-128": "pt_2c_128r_6p_4x4x8",
    "pt-dor-128": "pt_2c_128r_6p_4x4x8",
    "pdtt-128": "pdtt_2c_128r_6p_4x4x8",
    "tons-128": "asc_lp_sym_2c_128r_6p_4x4x8_4x4x4",
}


def known_bundle(root: Path | str, name: str) -> BundlePaths:
    """Resolve a canonical 128-router PT, PDTT, or TONS bundle below *root*."""

    try:
        topology_stem = _KNOWN_BUNDLES[name]
    except KeyError as exc:
        choices = ", ".join(sorted(_KNOWN_BUNDLES))
        raise ValueError(f"unknown bundle {name!r}; choose one of: {choices}") from exc

    root = Path(root)
    if name == "pt-dor-128":
        route_stem = f"{topology_stem}_dor_dim_tiebreak_destbased"
        return BundlePaths(
            name=name,
            topology=root / "topo_maps" / f"{topology_stem}.map",
            pmcf_candidates=(
                root
                / "allpath_lists"
                / f"{topology_stem}_turns_allowed_cpl_safe_destbased.rallpaths"
            ),
            routes=root / "routepath_lists" / f"{route_stem}.paths",
            next_hops=root / "nr_lists" / f"{route_stem}.nrl2",
            vc_matrix=root / "vc_mats" / f"{route_stem}.vcmat2",
        )

    candidate_stem = f"{topology_stem}_turns_allowed_cpl_safe_destbased"
    route_stem = f"{candidate_stem}_new_mclb_destbased"
    allowed_stem = f"{topology_stem}_turns_allowed_cpl_safe"
    return BundlePaths(
        name=name,
        topology=root / "topo_maps" / f"{topology_stem}.map",
        candidates=root / "allpath_lists" / f"{candidate_stem}.rallpaths",
        routes=root / "routepath_lists" / f"{route_stem}.paths",
        next_hops=root / "nr_lists" / f"{route_stem}.nrl2",
        vc_matrix=root / "vc_mats" / f"{route_stem}_olb.vcmat2",
        allowed_turns=root / "allowed_turns_vcs" / f"{allowed_stem}.allowvcturns",
        pmcf_candidates=root / "allpath_lists" / f"{candidate_stem}.rallpaths",
    )


def _nonempty_lines(path: Path) -> Iterable[tuple[int, str]]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, raw_line in enumerate(stream, start=1):
                line = raw_line.strip()
                if line:
                    yield line_number, line
    except OSError as exc:
        raise ArtifactFormatError(f"cannot read {path}: {exc}") from exc


def _read_topology(path: Path) -> list[list[int]]:
    rows: list[list[int]] = []
    for line_number, line in _nonempty_lines(path):
        row: list[int] = []
        for token in line.split():
            try:
                numeric = float(token)
            except ValueError as exc:
                raise ArtifactFormatError(
                    f"{path}:{line_number}: non-numeric map entry {token!r}"
                ) from exc
            integral = int(numeric)
            if numeric != integral or integral not in (0, 1):
                raise ArtifactFormatError(
                    f"{path}:{line_number}: map entries must be binary, got {token!r}"
                )
            row.append(integral)
        rows.append(row)
    if not rows:
        raise ArtifactFormatError(f"{path}: empty topology map")
    size = len(rows)
    for index, row in enumerate(rows):
        if len(row) != size:
            raise ArtifactFormatError(
                f"{path}:{index + 1}: expected {size} columns, found {len(row)}"
            )
    return rows


def _integer_sequence(value: object, path: Path, line_number: int) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ArtifactFormatError(f"{path}:{line_number}: expected a non-empty node sequence")
    if any(type(node) is not int for node in value):
        raise ArtifactFormatError(f"{path}:{line_number}: node identifiers must be integers")
    return tuple(value)


def _read_routes(path: Path) -> tuple[str, list[tuple[int, ...]]]:
    lines = iter(_nonempty_lines(path))
    try:
        _, header = next(lines)
    except StopIteration as exc:
        raise ArtifactFormatError(f"{path}: empty selected-route file") from exc
    routes: list[tuple[int, ...]] = []
    for line_number, line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            try:
                value = ast.literal_eval(line)
            except (SyntaxError, ValueError) as exc:
                raise ArtifactFormatError(
                    f"{path}:{line_number}: invalid selected path"
                ) from exc
        routes.append(_integer_sequence(value, path, line_number))
    return header, routes


def _read_candidates(path: Path) -> list[tuple[int, ...]]:
    candidates: list[tuple[int, ...]] = []
    for line_number, line in _nonempty_lines(path):
        try:
            value = tuple(int(token) for token in line.split())
        except ValueError as exc:
            raise ArtifactFormatError(
                f"{path}:{line_number}: candidate paths must be space-separated integers"
            ) from exc
        candidates.append(_integer_sequence(value, path, line_number))
    return candidates


def _literal_tuple(path: Path, line_number: int, line: str, length: int) -> tuple[int, ...]:
    try:
        value = ast.literal_eval(line)
    except (SyntaxError, ValueError) as exc:
        raise ArtifactFormatError(f"{path}:{line_number}: invalid tuple record") from exc
    if not isinstance(value, tuple) or len(value) != length:
        raise ArtifactFormatError(
            f"{path}:{line_number}: expected a {length}-integer tuple"
        )
    if any(type(item) is not int for item in value):
        raise ArtifactFormatError(f"{path}:{line_number}: tuple fields must be integers")
    return value


def _read_tuple_records(path: Path, length: int) -> list[tuple[int, ...]]:
    return [
        _literal_tuple(path, line_number, line, length)
        for line_number, line in _nonempty_lines(path)
    ]


def _read_allowed_turns(path: Path) -> dict[Turn, bool]:
    turns: dict[Turn, bool] = {}
    for line_number, line in _nonempty_lines(path):
        try:
            parsed = ast.literal_eval("{" + line + "}")
        except (SyntaxError, ValueError) as exc:
            raise ArtifactFormatError(
                f"{path}:{line_number}: invalid allowed-turn record"
            ) from exc
        if not isinstance(parsed, dict) or len(parsed) != 1:
            raise ArtifactFormatError(
                f"{path}:{line_number}: expected one turn-to-Boolean mapping"
            )
        key, value = next(iter(parsed.items()))
        valid_channels = (
            isinstance(key, tuple)
            and len(key) == 2
            and all(
                isinstance(channel, tuple)
                and len(channel) == 3
                and all(type(item) is int for item in channel)
                for channel in key
            )
        )
        if not valid_channels or type(value) is not bool:
            raise ArtifactFormatError(
                f"{path}:{line_number}: expected ((u,v,vc),(v,w,vc)): Boolean"
            )
        if key in turns:
            raise ArtifactFormatError(f"{path}:{line_number}: duplicate turn {key}")
        turns[key] = value
    if not turns:
        raise ArtifactFormatError(f"{path}: empty allowed-turn table")
    return turns


def _path_error(path: tuple[int, ...], topology: list[list[int]]) -> str | None:
    size = len(topology)
    if any(node < 0 or node >= size for node in path):
        return f"contains a node outside [0, {size})"
    if len(set(path)) != len(path):
        return "is not simple (a router is repeated)"
    for source, destination in zip(path, path[1:]):
        if not topology[source][destination]:
            return f"uses non-edge {source}->{destination}"
    return None


def _graph_is_acyclic(edges: Iterable[tuple[object, object]]) -> bool:
    adjacency: dict[object, set[object]] = defaultdict(set)
    indegree: Counter[object] = Counter()
    nodes: set[object] = set()
    for source, destination in edges:
        nodes.update((source, destination))
        if destination not in adjacency[source]:
            adjacency[source].add(destination)
            indegree[destination] += 1
            indegree[source] += 0
    ready = deque(node for node in nodes if indegree[node] == 0)
    visited = 0
    while ready:
        source = ready.popleft()
        visited += 1
        for destination in adjacency[source]:
            indegree[destination] -= 1
            if indegree[destination] == 0:
                ready.append(destination)
    return visited == len(nodes)


def _path_has_allowed_vc_assignment(
    path: tuple[int, ...],
    turns: dict[Turn, bool],
    vc_ids: set[int],
) -> bool:
    """Return whether one consistent VC sequence permits every turn in *path*."""

    possible_vcs = set(vc_ids)
    for first, middle, last in zip(path, path[1:], path[2:]):
        possible_vcs = {
            next_vc
            for prior_vc in possible_vcs
            for next_vc in vc_ids
            if turns.get(
                ((first, middle, prior_vc), (middle, last, next_vc)), False
            )
        }
        if not possible_vcs:
            return False
    return True


def _file_dict(paths: BundlePaths) -> dict[str, str]:
    return {
        key: str(value)
        for key, value in {
            "topology": paths.topology,
            "routes": paths.routes,
            "candidates": paths.candidates,
            "next_hops": paths.next_hops,
            "vc_matrix": paths.vc_matrix,
            "allowed_turns": paths.allowed_turns,
        }.items()
        if value is not None
    }


def validate_bundle(
    paths: BundlePaths,
    *,
    expected_degree: int | None = None,
    destination_based: bool = False,
) -> ValidationReport:
    """Validate each supplied artifact and all available cross-file contracts."""

    report = ValidationReport(
        name=paths.name or paths.topology.stem,
        files=_file_dict(paths),
    )
    topology = _read_topology(paths.topology)
    _, routes = _read_routes(paths.routes)
    size = len(topology)
    report.routers = size

    degrees = [sum(row) for row in topology]
    report.directed_links = sum(degrees)
    report.min_degree = min(degrees)
    report.max_degree = max(degrees)
    if any(topology[node][node] for node in range(size)):
        report.errors.append("topology contains a self-loop")
    if any(topology[u][v] != topology[v][u] for u in range(size) for v in range(size)):
        report.errors.append("topology adjacency matrix is not symmetric")
    if expected_degree is not None and any(degree != expected_degree for degree in degrees):
        report.errors.append(
            f"topology is not {expected_degree}-regular (degree range "
            f"{report.min_degree}..{report.max_degree})"
        )

    reachable = {0}
    queue = deque([0])
    while queue:
        source = queue.popleft()
        for destination, connected in enumerate(topology[source]):
            if connected and destination not in reachable:
                reachable.add(destination)
                queue.append(destination)
    if len(reachable) != size:
        report.errors.append(
            f"topology is disconnected: router 0 reaches {len(reachable)} of {size} routers"
        )

    expected_route_count = size * size
    if len(routes) != expected_route_count:
        report.errors.append(
            f"selected-route table has {len(routes)} paths; expected {expected_route_count}"
        )
    selected_set: set[tuple[int, ...]] = set()
    channel_load: Counter[tuple[int, int]] = Counter()
    destination_next_hop: dict[tuple[int, int], int] = {}
    expected_next_hops: list[tuple[int, int, int, int]] = []
    expected_vc_keys: list[tuple[int, int, int]] = []
    valid_routes: dict[tuple[int, int], tuple[int, ...]] = {}

    for index, route in enumerate(routes):
        expected_source, expected_destination = divmod(index, size)
        if route[0] != expected_source or route[-1] != expected_destination:
            report.errors.append(
                f"selected route index {index} should be {expected_source}->{expected_destination}, "
                f"found {route[0]}->{route[-1]}"
            )
            continue
        error = _path_error(route, topology)
        if error:
            report.errors.append(
                f"selected route {expected_source}->{expected_destination} {error}"
            )
            continue
        if expected_source == expected_destination and route != (expected_source,):
            report.errors.append(
                f"self route {expected_source}->{expected_destination} must contain one router"
            )
            continue
        valid_routes[(expected_source, expected_destination)] = route
        selected_set.add(route)
        if expected_source == expected_destination:
            expected_vc_keys.append(
                (expected_source, expected_destination, expected_source)
            )
            continue
        report.selected_flows += 1
        report.selected_hops += len(route) - 1
        for current, following in zip(route, route[1:]):
            channel_load[(current, following)] += 1
            expected_next_hops.append(
                (expected_source, expected_destination, current, following)
            )
            expected_vc_keys.append((expected_source, expected_destination, current))
            if destination_based:
                key = (current, expected_destination)
                previous = destination_next_hop.setdefault(key, following)
                if previous != following:
                    report.errors.append(
                        f"routes are not destination-based for current router {current}, "
                        f"destination {expected_destination}: next hops {previous} and {following}"
                    )

    if report.selected_flows:
        report.average_hops = report.selected_hops / report.selected_flows
    if channel_load:
        report.maximum_directed_channel_load = max(channel_load.values())
        report.unit_flow_throughput_bound = 1.0 / report.maximum_directed_channel_load

    candidates: list[tuple[int, ...]] | None = None
    if paths.candidates is not None:
        candidates = _read_candidates(paths.candidates)
        report.candidate_paths = len(candidates)
        candidate_set: set[tuple[int, ...]] = set()
        covered_pairs: set[tuple[int, int]] = set()
        for candidate in candidates:
            error = _path_error(candidate, topology)
            if error:
                report.errors.append(
                    f"candidate route {candidate[0]}->{candidate[-1]} {error}"
                )
                continue
            if candidate[0] == candidate[-1]:
                report.errors.append(f"candidate file contains self route {candidate[0]}")
                continue
            candidate_set.add(candidate)
            covered_pairs.add((candidate[0], candidate[-1]))
        expected_pairs = {(s, d) for s in range(size) for d in range(size) if s != d}
        missing_pairs = expected_pairs - covered_pairs
        if missing_pairs:
            report.errors.append(
                f"candidate routes omit {len(missing_pairs)} ordered source/destination pairs"
            )
        missing_selected = {
            route
            for (source, destination), route in valid_routes.items()
            if source != destination and route not in candidate_set
        }
        if missing_selected:
            report.errors.append(
                f"{len(missing_selected)} selected routes are absent from the candidate file"
            )

    if paths.next_hops is not None:
        next_hops = _read_tuple_records(paths.next_hops, 4)
        if next_hops != expected_next_hops:
            mismatch = next(
                (
                    index
                    for index, (actual, expected) in enumerate(
                        zip(next_hops, expected_next_hops)
                    )
                    if actual != expected
                ),
                min(len(next_hops), len(expected_next_hops)),
            )
            report.errors.append(
                f"next-hop expansion differs from selected routes at record {mismatch} "
                f"({len(next_hops)} records versus {len(expected_next_hops)} expected)"
            )

    turns: dict[Turn, bool] | None = None
    vc_ids: set[int] = set()
    if paths.allowed_turns is not None:
        turns = _read_allowed_turns(paths.allowed_turns)
        report.allowed_turn_entries = len(turns)
        for (first, second), _ in turns.items():
            vc_ids.update((first[2], second[2]))
            if first[1] != second[0]:
                report.errors.append(f"non-contiguous turn in allowed-turn table: {(first, second)}")
            if first[0] == second[1]:
                report.errors.append(f"U-turn appears in allowed-turn table: {(first, second)}")
            for source, destination, _ in (first, second):
                if not (0 <= source < size and 0 <= destination < size):
                    report.errors.append(f"out-of-range channel in allowed-turn table: {(first, second)}")
                elif not topology[source][destination]:
                    report.errors.append(f"non-physical channel in allowed-turn table: {(first, second)}")
        if not vc_ids or vc_ids != set(range(max(vc_ids) + 1)):
            report.errors.append(f"virtual-channel identifiers are not dense from zero: {sorted(vc_ids)}")
        else:
            report.virtual_channels = len(vc_ids)
            expected_turns = {
                ((u, v, first_vc), (v, w, second_vc))
                for u in range(size)
                for v, uv in enumerate(topology[u])
                if uv
                for w, vw in enumerate(topology[v])
                if vw and w != u
                for first_vc in vc_ids
                for second_vc in vc_ids
            }
            missing = expected_turns - turns.keys()
            extra = turns.keys() - expected_turns
            if missing or extra:
                report.errors.append(
                    f"allowed-turn table is incomplete: {len(missing)} missing, {len(extra)} extra"
                )
        allowed_edges = [turn for turn, allowed in turns.items() if allowed]
        if not _graph_is_acyclic(allowed_edges):
            report.errors.append("allowed channel-dependency graph contains a cycle")

        if candidates is not None and vc_ids:
            incompatible_candidate_count = sum(
                1
                for candidate in candidates
                if not _path_has_allowed_vc_assignment(candidate, turns, vc_ids)
            )
            if incompatible_candidate_count:
                report.errors.append(
                    f"{incompatible_candidate_count} candidate routes have no allowed "
                    "end-to-end VC assignment"
                )

    if paths.vc_matrix is not None:
        vc_records = _read_tuple_records(paths.vc_matrix, 4)
        actual_vc_keys = [record[:3] for record in vc_records]
        if actual_vc_keys != expected_vc_keys:
            mismatch = next(
                (
                    index
                    for index, (actual, expected) in enumerate(
                        zip(actual_vc_keys, expected_vc_keys)
                    )
                    if actual != expected
                ),
                min(len(actual_vc_keys), len(expected_vc_keys)),
            )
            report.errors.append(
                f"VC matrix differs from selected-route hop order at record {mismatch} "
                f"({len(actual_vc_keys)} records versus {len(expected_vc_keys)} expected)"
            )
        if len(set(actual_vc_keys)) != len(actual_vc_keys):
            report.errors.append("VC matrix contains duplicate (source,destination,current) keys")
        vc_by_key = {record[:3]: record[3] for record in vc_records}
        used_dependencies: list[Turn] = []
        for (source, destination), route in valid_routes.items():
            if len(route) < 3:
                continue
            for index in range(len(route) - 2):
                first_key = (source, destination, route[index])
                second_key = (source, destination, route[index + 1])
                if first_key not in vc_by_key or second_key not in vc_by_key:
                    continue
                turn = (
                    (route[index], route[index + 1], vc_by_key[first_key]),
                    (route[index + 1], route[index + 2], vc_by_key[second_key]),
                )
                used_dependencies.append(turn)
                if turns is not None and not turns.get(turn, False):
                    report.errors.append(
                        f"VC assignment for {source}->{destination} uses disallowed turn {turn}"
                    )
        if vc_ids:
            invalid_vcs = sorted({record[3] for record in vc_records} - vc_ids)
            if invalid_vcs:
                report.errors.append(f"VC matrix uses undefined VCs: {invalid_vcs}")
        if not _graph_is_acyclic(used_dependencies):
            report.errors.append("selected-route channel-dependency graph contains a cycle")

    return report
