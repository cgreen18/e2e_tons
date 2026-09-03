"""Generate small synthetic Chakra workload traces with explicit communicators.

The trace-driven experiments need a workload whose communicator membership is
known exactly and whose replay cost is trivial, so that communicator-aware
custom collective selection can be validated without depending on the large
public AI traces.

The emitted traces use the same conventions ASTRA-sim expects from real
PyTorch-derived traces:

* one ``METADATA_NODE`` named ``## process_group:init ##`` whose
  ``inputs.values`` carries the process-group registry, wrapped in the two
  leading and two trailing characters that ``Workload::issue_pytorch_pg_metadata``
  strips before parsing;
* ``COMM_COLL_NODE`` nodes carrying ``comm_type``, ``comm_size``, and a
  ``pg_name`` string that selects the communicator; and
* ``COMP_NODE`` nodes replayed from ``duration_micros``.

Process group ``0`` is reserved: ASTRA treats it, and an absent ``pg_name``, as
the default communicator covering every rank.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from .chakra import COLLECTIVE_TYPES, COMM_COLL_NODE, ChakraNode, write_trace


COMP_NODE = 4
METADATA_NODE = 1
PROCESS_GROUP_METADATA_NAME = "## process_group:init ##"

# ``Workload::issue_pytorch_pg_metadata`` parses ``values[2:-2]``.  Real traces
# wrap the registry in a Python-style list of one string; any two characters
# work, so keep the real shape.
_VALUES_PREFIX = '["'
_VALUES_SUFFIX = '"]'


@dataclass(frozen=True)
class Communicator:
    """One process group.  ``pg_name`` must be a positive decimal string."""

    pg_name: str
    members: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.pg_name.isdigit():
            raise ValueError(f"pg_name {self.pg_name!r} must be a decimal string")
        if int(self.pg_name) == 0:
            raise ValueError("pg_name '0' is reserved for the default communicator")
        if not self.members:
            raise ValueError(f"communicator {self.pg_name} has no members")
        if list(self.members) != sorted(set(self.members)):
            raise ValueError(
                f"communicator {self.pg_name} members must be sorted and unique"
            )


@dataclass(frozen=True)
class Compute:
    name: str
    micros: int


@dataclass(frozen=True)
class Collective:
    name: str
    collective: str
    pg_name: str
    size_bytes: int


def _registry_values(communicators: list[Communicator]) -> str:
    registry = [
        {"pg_name": communicator.pg_name, "ranks": list(communicator.members)}
        for communicator in communicators
    ]
    return _VALUES_PREFIX + json.dumps(registry, separators=(", ", ": ")) + _VALUES_SUFFIX


def generate_synthetic_trace(
    output_prefix: Path | str,
    ranks: int,
    stages: list[Compute | Collective],
    communicators: list[Communicator],
) -> list[Path]:
    """Write one workload ET per rank.

    A rank only receives the collective nodes of the communicators it belongs
    to, mirroring a real trace.  Compute stages are emitted on every rank, and
    each stage depends on the previous stage, so the per-rank DAG is a chain.

    The process-group metadata node is deliberately left *outside* that chain,
    as it is in real PyTorch traces.  ``Workload::issue_metadata`` completes
    synchronously without registering a simulator event, while
    ``Workload::issue_dep_free_nodes`` iterates a snapshot of the ready set
    taken before issuing.  A metadata node that is the sole dependency root
    therefore frees its children but nothing ever re-scans for them, and the
    run ends at tick 0 reporting no completed ranks and no error.  Leaving it
    unattached keeps it dependency-free alongside the first stage; the ready
    set is ordered by node id, so metadata (node 0) is still issued first and
    the registry is in place well before the first collective.
    """

    if ranks < 1:
        raise ValueError("ranks must be positive")
    if not stages:
        raise ValueError("at least one stage is required")

    by_name = {communicator.pg_name: communicator for communicator in communicators}
    if len(by_name) != len(communicators):
        raise ValueError("duplicate pg_name in communicators")
    for stage in stages:
        if not isinstance(stage, Collective):
            continue
        if stage.collective not in COLLECTIVE_TYPES:
            raise ValueError(f"unsupported collective {stage.collective!r}")
        if stage.size_bytes < 1:
            raise ValueError(f"stage {stage.name} needs a positive size")
        if stage.pg_name not in by_name:
            raise ValueError(f"stage {stage.name} references unknown pg {stage.pg_name}")
    for communicator in communicators:
        if communicator.members[-1] >= ranks:
            raise ValueError(
                f"communicator {communicator.pg_name} references rank "
                f"{communicator.members[-1]} outside {ranks} ranks"
            )

    values = _registry_values(communicators)
    prefix = Path(output_prefix)
    paths: list[Path] = []
    for rank in range(ranks):
        nodes = [
            ChakraNode(
                0,
                PROCESS_GROUP_METADATA_NAME,
                METADATA_NODE,
                # HardwareResource::is_available reads is_cpu_op with no
                # default and throws when it is absent, so every node needs
                # it -- including metadata.  Process-group registration is a
                # host-side record, so it is a CPU op.
                attributes=[("is_cpu_op", True, "bool")],
                inputs_values=values,
            )
        ]
        for stage in stages:
            if isinstance(stage, Collective) and rank not in by_name[stage.pg_name].members:
                continue
            node_id = len(nodes)
            # The first stage has no predecessor: it must stay dependency-free
            # alongside the metadata node.  See the note above.
            previous = [] if node_id == 1 else [nodes[-1].node_id]
            if isinstance(stage, Compute):
                nodes.append(
                    ChakraNode(
                        node_id,
                        stage.name,
                        COMP_NODE,
                        dependencies=previous,
                        attributes=[("is_cpu_op", False, "bool")],
                        duration_micros=stage.micros,
                    )
                )
            else:
                nodes.append(
                    ChakraNode(
                        node_id,
                        stage.name,
                        COMM_COLL_NODE,
                        dependencies=previous,
                        attributes=[
                            ("is_cpu_op", False, "bool"),
                            ("comm_type", COLLECTIVE_TYPES[stage.collective], "int64"),
                            ("comm_size", stage.size_bytes, "int64"),
                            ("pg_name", stage.pg_name, "string"),
                        ],
                    )
                )
        path = prefix.parent / f"{prefix.name}.{rank}.et"
        paths.append(write_trace(path, nodes))
    return paths
