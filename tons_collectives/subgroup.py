"""Topology-aware collective schedules for an arbitrary communicator subset.

A trace's collectives run on communicators that are usually smaller than the
machine.  ASTRA-sim's ``CustomAlgorithm`` loads one schedule ET per *position*
in the communicator and maps algo rank ``i`` onto ``involved_NPUs[i]``, so a
schedule generated here is written in algo-rank space ``0..k-1`` while its
topology awareness comes from the *physical* placement of those members.

Two schedule families live here:

* structural collectives (``broadcast``, ``gather``) that reduce to a trivial
  one-to-all or all-to-one send/receive tree; and
* the member-restricted all-to-all used when only a subset exchanges data.

All-gather, all-reduce, and reduce-scatter are synthesized by ``l3ss_tree``
from the reduced adjacency matrix produced by :func:`member_submap`.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from xml.etree import ElementTree as ET

from .a2a import _step, _write


SUBMAP_MODES = ("induced", "proximity", "complete")


def hop_distances(adjacency: list[list[int]], source: int) -> list[int]:
    """Unweighted shortest-path hop counts from ``source`` over the full map."""

    size = len(adjacency)
    distance = [-1] * size
    distance[source] = 0
    queue = deque([source])
    while queue:
        node = queue.popleft()
        row = adjacency[node]
        for peer in range(size):
            if row[peer] and distance[peer] < 0:
                distance[peer] = distance[node] + 1
                queue.append(peer)
    return distance


def _connected(matrix: list[list[int]]) -> bool:
    size = len(matrix)
    if size <= 1:
        return True
    seen = {0}
    queue = deque([0])
    while queue:
        node = queue.popleft()
        for peer in range(size):
            if matrix[node][peer] and peer not in seen:
                seen.add(peer)
                queue.append(peer)
    return len(seen) == size


def member_submap(
    adjacency: list[list[int]],
    members: list[int],
    mode: str = "proximity",
) -> tuple[list[list[int]], int]:
    """Reduce a full topology to a ``k x k`` adjacency over ``members``.

    Returns the matrix and the hop radius it used.  The radius is ``1`` for an
    induced subgraph and larger when members had to be joined through
    non-member routers.

    ``induced``
        The physical induced subgraph.  Most faithful, but a scattered
        communicator is frequently disconnected under it.
    ``proximity``
        Members are adjacent when their physical hop distance is at most the
        smallest radius that connects every member.  Degrades to ``induced``
        when radius 1 already suffices.  Non-adjacent member pairs are carried
        by the network backend over the bundle's selected routes, exactly as a
        direct all-to-all already is.
    ``complete``
        All member pairs adjacent.  Topology-blind; kept as a control.
    """

    if mode not in SUBMAP_MODES:
        raise ValueError(f"unsupported submap mode {mode!r}")
    size = len(adjacency)
    members = list(members)
    if not members:
        raise ValueError("communicator has no members")
    if sorted(set(members)) != sorted(members):
        raise ValueError("communicator members must be unique")
    for member in members:
        if not 0 <= member < size:
            raise ValueError(f"member {member} is outside the {size}-router topology")

    count = len(members)
    if mode == "complete":
        matrix = [[1 if i != j else 0 for j in range(count)] for i in range(count)]
        return matrix, -1

    distances = {member: hop_distances(adjacency, member) for member in members}
    radii = sorted(
        {
            distances[u][v]
            for u in members
            for v in members
            if u != v and distances[u][v] > 0
        }
    )
    if not radii and count > 1:
        raise ValueError("communicator members are mutually unreachable")

    def build(radius: int) -> list[list[int]]:
        return [
            [
                1 if i != j and 0 < distances[members[i]][members[j]] <= radius else 0
                for j in range(count)
            ]
            for i in range(count)
        ]

    if mode == "induced":
        matrix = build(1)
        if not _connected(matrix):
            raise ValueError(
                f"induced subgraph over {count} members is disconnected; "
                "use the 'proximity' mode or supply an explicit sub-map"
            )
        return matrix, 1

    for radius in radii:
        matrix = build(radius)
        if _connected(matrix):
            return matrix, radius
    raise ValueError(f"no hop radius connects the {count} members")


def write_map(matrix: list[list[int]], output: Path | str) -> Path:
    """Write an adjacency matrix in the ``.map`` format l3ss_tree consumes."""

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(" ".join(str(int(value)) for value in row) for row in matrix) + "\n"
    )
    return output


def _tree_root(ranks: int, name: str, collective: str, chunks: int) -> ET.Element:
    return ET.Element(
        "algo",
        {
            "name": name,
            "proto": "Simple",
            "nchannels": "1",
            "nchunksperloop": str(chunks),
            "ngpus": str(ranks),
            "coll": collective,
            "input_chunks": str(chunks),
            "chunk_layout": "destination-major",
            "inplace": "0",
            "outofplace": "1",
            "minBytes": "1",
            "maxBytes": "1073741824",
        },
    )


def generate_broadcast(ranks: int, output: Path | str, root_rank: int = 0) -> Path:
    """One-to-all: the root sends the whole buffer to every other member.

    The buffer is a single logical chunk, so ASTRA scales each send to the
    invoking collective node's full ``comm_size``.
    """

    if ranks < 2:
        raise ValueError("broadcast needs at least two ranks")
    if not 0 <= root_rank < ranks:
        raise ValueError(f"root {root_rank} is outside {ranks} ranks")
    root = _tree_root(ranks, "tons_broadcast", "broadcast", 1)
    for rank in range(ranks):
        gpu = ET.SubElement(
            root, "gpu", {"id": str(rank), "i_chunks": "1", "o_chunks": "1", "s_chunks": "0"}
        )
        if rank == root_rank:
            for tb_id, peer in enumerate(p for p in range(ranks) if p != root_rank):
                send = ET.SubElement(
                    gpu, "tb", {"id": str(tb_id), "send": str(peer), "recv": "-1", "chan": "0"}
                )
                _step(
                    send, step=0, kind="s", srcbuf="i", srcoff=0, dstbuf="o", dstoff=0,
                    logical_chunk=0, flow_source=root_rank, flow_destination=peer, epoch=0,
                )
        else:
            recv = ET.SubElement(
                gpu, "tb", {"id": "0", "send": "-1", "recv": str(root_rank), "chan": "0"}
            )
            _step(
                recv, step=0, kind="r", srcbuf="i", srcoff=0, dstbuf="o", dstoff=0,
                logical_chunk=0, flow_source=root_rank, flow_destination=rank, epoch=0,
            )
    return _write(root, output)


def generate_gather(ranks: int, output: Path | str, root_rank: int = 0) -> Path:
    """All-to-one: every non-root member sends its whole buffer to the root.

    This is the send/receive equivalent of a REDUCE with the reduction compute
    omitted, so it is named ``gather`` rather than ``reduce``.
    """

    if ranks < 2:
        raise ValueError("gather needs at least two ranks")
    if not 0 <= root_rank < ranks:
        raise ValueError(f"root {root_rank} is outside {ranks} ranks")
    root = _tree_root(ranks, "tons_gather", "gather", 1)
    for rank in range(ranks):
        gpu = ET.SubElement(
            root, "gpu", {"id": str(rank), "i_chunks": "1", "o_chunks": "1", "s_chunks": "0"}
        )
        if rank == root_rank:
            for tb_id, peer in enumerate(p for p in range(ranks) if p != root_rank):
                recv = ET.SubElement(
                    gpu, "tb", {"id": str(tb_id), "send": "-1", "recv": str(peer), "chan": "0"}
                )
                _step(
                    recv, step=0, kind="r", srcbuf="i", srcoff=0, dstbuf="o", dstoff=0,
                    logical_chunk=0, flow_source=peer, flow_destination=root_rank, epoch=0,
                )
        else:
            send = ET.SubElement(
                gpu, "tb", {"id": "0", "send": str(root_rank), "recv": "-1", "chan": "0"}
            )
            _step(
                send, step=0, kind="s", srcbuf="i", srcoff=0, dstbuf="o", dstoff=0,
                logical_chunk=0, flow_source=rank, flow_destination=root_rank, epoch=0,
            )
    return _write(root, output)
