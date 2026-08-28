"""Generate one-collective Chakra workload traces."""

from __future__ import annotations

from pathlib import Path

from .chakra import COLLECTIVE_TYPES, COMM_COLL_NODE, ChakraNode, write_trace


def generate_collective_workload(
    output_prefix: Path | str,
    ranks: int,
    collective: str,
    size_bytes: int,
) -> list[Path]:
    """Write one workload ET per rank while keeping schedule ETs separate."""

    if ranks < 1:
        raise ValueError("ranks must be positive")
    if size_bytes < 1:
        raise ValueError("size_bytes must be positive")
    try:
        comm_type = COLLECTIVE_TYPES[collective]
    except KeyError as error:
        raise ValueError(f"unsupported collective {collective!r}") from error
    prefix = Path(output_prefix)
    paths: list[Path] = []
    for rank in range(ranks):
        node = ChakraNode(
            0,
            collective.upper(),
            COMM_COLL_NODE,
            attributes=[
                ("is_cpu_op", False, "bool"),
                ("comm_type", comm_type, "int64"),
                ("comm_size", size_bytes, "int64"),
            ],
        )
        path = prefix.parent / f"{prefix.name}.{rank}.et"
        paths.append(write_trace(path, [node]))
    return paths
