#!/usr/bin/env python3
"""Generate ASTRA Graph YAML and per-directed-edge properties."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tons_topology import generate_graph_artifacts, known_bundle  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle", choices=("pt-dor-128", "pdtt-128", "tons-128"), required=True
    )
    parser.add_argument("--root", type=Path, default=Path("topologies_and_routing"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    bundle = known_bundle(args.root, args.bundle)
    artifacts = generate_graph_artifacts(
        bundle.topology, bundle.routes, args.output_dir, stem=args.bundle
    )
    print(artifacts.network_config)
    print(
        f"electrical={artifacts.electrical_directed_edges} "
        f"optical={artifacts.optical_directed_edges}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
