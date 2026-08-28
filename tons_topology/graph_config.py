"""Generate deterministic ASTRA analytical Graph inputs.

The topology and route files remain the source of truth.  Generated YAML and
edge-property files are intentionally small adapters and may be regenerated at
any time.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TopologyDimensions:
    x: int = 4
    y: int = 4
    z: int = 8
    cube: int = 4

    @property
    def routers(self) -> int:
        return self.x * self.y * self.z


@dataclass(frozen=True)
class GraphNetworkProfile:
    bandwidth_GBps: float
    electrical_latency_ns: float
    optical_latency_ns: float
    injection_latency_ns: float
    clock_GHz: float


@dataclass(frozen=True)
class GraphArtifacts:
    edge_properties: Path
    network_config: Path
    electrical_directed_edges: int
    optical_directed_edges: int


PAPER_1_GHZ_PROFILE = GraphNetworkProfile(
    bandwidth_GBps=128.0,
    electrical_latency_ns=75.0,
    optical_latency_ns=50.0,
    injection_latency_ns=25.0,
    clock_GHz=1.0,
)


def _coordinates(router: int, dimensions: TopologyDimensions) -> tuple[int, int, int]:
    if not 0 <= router < dimensions.routers:
        raise ValueError(f"router {router} is outside [0, {dimensions.routers})")
    x = router % dimensions.x
    y = (router // dimensions.x) % dimensions.y
    z = router // (dimensions.x * dimensions.y)
    return x, y, z


def classify_directed_edge(
    source: int,
    destination: int,
    dimensions: TopologyDimensions = TopologyDimensions(),
) -> str:
    """Return ``electrical`` for fixed in-cube mesh edges, else ``optical``."""

    sx, sy, sz = _coordinates(source, dimensions)
    dx, dy, dz = _coordinates(destination, dimensions)
    deltas = (abs(sx - dx), abs(sy - dy), abs(sz - dz))
    adjacent = sum(delta == 1 for delta in deltas) == 1 and sum(deltas) == 1
    same_cube = (
        sx // dimensions.cube == dx // dimensions.cube
        and sy // dimensions.cube == dy // dimensions.cube
        and sz // dimensions.cube == dz // dimensions.cube
    )
    return "electrical" if adjacent and same_cube else "optical"


def _read_map(path: Path) -> list[list[int]]:
    rows = [[int(float(token)) for token in line.split()] for line in path.read_text().splitlines() if line.strip()]
    if not rows or any(len(row) != len(rows) for row in rows):
        raise ValueError(f"{path}: topology must be a non-empty square matrix")
    if any(value not in (0, 1) for row in rows for value in row):
        raise ValueError(f"{path}: topology entries must be binary")
    return rows


def write_edge_properties(
    topology: Path | str,
    output: Path | str,
    *,
    dimensions: TopologyDimensions = TopologyDimensions(),
    profile: GraphNetworkProfile = PAPER_1_GHZ_PROFILE,
) -> tuple[int, int]:
    topology = Path(topology)
    output = Path(output)
    rows = _read_map(topology)
    if len(rows) != dimensions.routers:
        raise ValueError(
            f"{topology}: contains {len(rows)} routers, expected {dimensions.routers}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    electrical = 0
    optical = 0
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("src", "dst", "bandwidth_GBps", "latency_ns", "link_type"))
        for source, row in enumerate(rows):
            for destination, connected in enumerate(row):
                if not connected:
                    continue
                kind = classify_directed_edge(source, destination, dimensions)
                latency = (
                    profile.electrical_latency_ns
                    if kind == "electrical"
                    else profile.optical_latency_ns
                )
                writer.writerow((source, destination, profile.bandwidth_GBps, latency, kind))
                if kind == "electrical":
                    electrical += 1
                else:
                    optical += 1
    return electrical, optical


def _yaml_path(path: Path, relative_to: Path) -> str:
    try:
        return str(path.resolve().relative_to(relative_to.resolve()))
    except ValueError:
        return str(path.resolve())


def write_graph_config(
    topology: Path | str,
    routes: Path | str,
    edge_properties: Path | str,
    output: Path | str,
    *,
    routers: int,
    profile: GraphNetworkProfile = PAPER_1_GHZ_PROFILE,
) -> None:
    topology = Path(topology)
    routes = Path(routes)
    edge_properties = Path(edge_properties)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    content = (
        "topology: [ Graph ]\n"
        f"npus_count: [ {routers} ]\n"
        f"bandwidth: [ {profile.bandwidth_GBps} ]  # GB/s\n"
        f"latency: [ {profile.optical_latency_ns} ]  # ns default\n"
        "graph:\n"
        f"  adjacency-matrix: {_yaml_path(topology, output.parent)}\n"
        f"  routing-paths: {_yaml_path(routes, output.parent)}\n"
        f"  edge-properties: {_yaml_path(edge_properties, output.parent)}\n"
        f"  injection-latency: {profile.injection_latency_ns}\n"
    )
    output.write_text(content, encoding="utf-8")


def generate_graph_artifacts(
    topology: Path | str,
    routes: Path | str,
    output_dir: Path | str,
    *,
    stem: str | None = None,
    dimensions: TopologyDimensions = TopologyDimensions(),
    profile: GraphNetworkProfile = PAPER_1_GHZ_PROFILE,
) -> GraphArtifacts:
    topology = Path(topology)
    routes = Path(routes)
    output_dir = Path(output_dir)
    stem = stem or topology.stem
    edge_path = output_dir / f"{stem}.edges.csv"
    config_path = output_dir / f"{stem}.yml"
    electrical, optical = write_edge_properties(
        topology, edge_path, dimensions=dimensions, profile=profile
    )
    write_graph_config(
        topology,
        routes,
        edge_path,
        config_path,
        routers=dimensions.routers,
        profile=profile,
    )
    return GraphArtifacts(edge_path, config_path, electrical, optical)
