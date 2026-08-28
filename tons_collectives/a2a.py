"""Deterministic MSCCL all-to-all schedule generators."""

from __future__ import annotations

import ast
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET


@dataclass
class _Transfer:
    epoch: int
    source: int
    destination: int
    flow_source: int
    flow_destination: int
    subchunk: int
    hop: int
    last_hop: bool


def read_selected_routes(path: Path | str) -> list[list[int]]:
    lines = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if lines and not lines[0].startswith("["):
        lines.pop(0)
    routes: list[list[int]] = []
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            value = ast.literal_eval(line)
        if not isinstance(value, (list, tuple)) or not value:
            raise ValueError(f"invalid selected route: {line}")
        routes.append([int(node) for node in value])
    size = int(len(routes) ** 0.5)
    if size * size != len(routes):
        raise ValueError("selected route file must contain N^2 paths")
    for index, route in enumerate(routes):
        source, destination = divmod(index, size)
        if route[0] != source or route[-1] != destination:
            raise ValueError(f"route {index} is not source-major {source}->{destination}")
    return routes


def _root(ranks: int, subchunks: int, name: str) -> ET.Element:
    chunks = ranks * subchunks
    return ET.Element(
        "algo",
        {
            "name": name,
            "proto": "Simple",
            "nchannels": "1",
            "nchunksperloop": str(chunks),
            "ngpus": str(ranks),
            "coll": "alltoall",
            "input_chunks": str(chunks),
            "chunk_layout": "destination-major",
            "inplace": "0",
            "outofplace": "1",
            "minBytes": "1",
            "maxBytes": "1073741824",
        },
    )


def _step(
    parent: ET.Element,
    *,
    step: int,
    kind: str,
    srcbuf: str,
    srcoff: int,
    dstbuf: str,
    dstoff: int,
    logical_chunk: int,
    flow_source: int,
    flow_destination: int,
    epoch: int,
    depid: int = -1,
    deps: int = -1,
) -> None:
    ET.SubElement(
        parent,
        "step",
        {
            "s": str(step),
            "type": kind,
            "srcbuf": srcbuf,
            "srcoff": str(srcoff),
            "dstbuf": dstbuf,
            "dstoff": str(dstoff),
            "cnt": "1",
            "depid": str(depid),
            "deps": str(deps),
            "hasdep": "1" if depid >= 0 else "0",
            "logical_chunk": str(logical_chunk),
            "flow_src": str(flow_source),
            "flow_dst": str(flow_destination),
            "epoch": str(epoch),
        },
    )


def _write(root: ET.Element, output: Path | str) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(output, encoding="unicode", xml_declaration=False)
    with output.open("a", encoding="utf-8") as stream:
        stream.write("\n")
    return output


def generate_direct_alltoall(ranks: int, subchunks: int, output: Path | str) -> Path:
    """Emit the same direct all-pairs schedule for any physical topology."""

    if ranks < 2 or subchunks < 1:
        raise ValueError("ranks must be >=2 and subchunks must be positive")
    root = _root(ranks, subchunks, "tons_direct_alltoall")
    chunks = ranks * subchunks
    for rank in range(ranks):
        gpu = ET.SubElement(
            root,
            "gpu",
            {"id": str(rank), "i_chunks": str(chunks), "o_chunks": str(chunks), "s_chunks": "0"},
        )
        tb_id = 0
        for peer in range(ranks):
            if peer == rank:
                continue
            recv = ET.SubElement(gpu, "tb", {"id": str(tb_id), "send": "-1", "recv": str(peer), "chan": "0"})
            tb_id += 1
            for q in range(subchunks):
                _step(
                    recv,
                    step=q,
                    kind="r",
                    srcbuf="i",
                    srcoff=rank * subchunks + q,
                    dstbuf="o",
                    dstoff=peer * subchunks + q,
                    logical_chunk=rank * subchunks + q,
                    flow_source=peer,
                    flow_destination=rank,
                    epoch=q,
                )
        for peer in range(ranks):
            if peer == rank:
                continue
            send = ET.SubElement(gpu, "tb", {"id": str(tb_id), "send": str(peer), "recv": "-1", "chan": "0"})
            tb_id += 1
            for q in range(subchunks):
                _step(
                    send,
                    step=q,
                    kind="s",
                    srcbuf="i",
                    srcoff=peer * subchunks + q,
                    dstbuf="o",
                    dstoff=rank * subchunks + q,
                    logical_chunk=peer * subchunks + q,
                    flow_source=rank,
                    flow_destination=peer,
                    epoch=q,
                )
    return _write(root, output)


def _compile_transfers(routes: list[list[int]], subchunks: int) -> list[_Transfer]:
    ranks = int(len(routes) ** 0.5)
    assignments: list[tuple[int, int, int, list[int]]] = []
    for source in range(ranks):
        for destination in range(ranks):
            if source == destination:
                continue
            path = routes[source * ranks + destination]
            for q in range(subchunks):
                assignments.append((source, destination, q, path))
    return _compile_assigned_transfers(assignments)


def _compile_assigned_transfers(
    assignments: list[tuple[int, int, int, list[int]]],
) -> list[_Transfer]:
    """Pack explicitly routed flow subchunks into unit-capacity link epochs."""

    jobs: list[dict[str, object]] = [
        {"src": source, "dst": destination, "q": q, "path": path, "hop": 0, "ready": 0}
        for source, destination, q, path in assignments
    ]
    transfers: list[_Transfer] = []
    unfinished = len(jobs)
    epoch = 0
    while unfinished:
        used: set[tuple[int, int]] = set()
        progressed = False
        for job in jobs:
            path = job["path"]
            hop = int(job["hop"])
            if hop >= len(path) - 1 or int(job["ready"]) > epoch:
                continue
            edge = (path[hop], path[hop + 1])
            if edge in used:
                continue
            used.add(edge)
            progressed = True
            transfers.append(
                _Transfer(
                    epoch,
                    edge[0],
                    edge[1],
                    int(job["src"]),
                    int(job["dst"]),
                    int(job["q"]),
                    hop,
                    hop + 1 == len(path) - 1,
                )
            )
            job["hop"] = hop + 1
            job["ready"] = epoch + 1
            if hop + 1 == len(path) - 1:
                unfinished -= 1
        if not progressed:
            raise RuntimeError("routed all-to-all scheduler made no progress")
        epoch += 1
    return transfers


def generate_fixed_route_alltoall(
    routes_path: Path | str,
    subchunks: int,
    output: Path | str,
) -> Path:
    """Emit a causal hop-by-hop schedule constrained to selected routes."""

    if subchunks < 1:
        raise ValueError("subchunks must be positive")
    routes = read_selected_routes(routes_path)
    ranks = int(len(routes) ** 0.5)
    transfers = _compile_transfers(routes, subchunks)
    return _generate_routed_alltoall(
        transfers,
        ranks,
        subchunks,
        "tons_fixed_route_alltoall",
        output,
    )


def _generate_routed_alltoall(
    transfers: list[_Transfer],
    ranks: int,
    subchunks: int,
    name: str,
    output: Path | str,
) -> Path:
    """Emit causal MSCCL XML for an already packed routed transfer list."""

    root = _root(ranks, subchunks, name)
    chunks = ranks * subchunks

    incoming: list[set[int]] = [set() for _ in range(ranks)]
    outgoing: list[set[int]] = [set() for _ in range(ranks)]
    for transfer in transfers:
        outgoing[transfer.source].add(transfer.destination)
        incoming[transfer.destination].add(transfer.source)

    tb_elements: list[dict[tuple[str, int], ET.Element]] = []
    tb_ids: list[dict[tuple[str, int], int]] = []
    local_steps: list[defaultdict[tuple[str, int], int]] = []
    for rank in range(ranks):
        gpu = ET.SubElement(
            root,
            "gpu",
            {
                "id": str(rank),
                "i_chunks": str(chunks),
                "o_chunks": str(chunks),
                "s_chunks": str(ranks * ranks * subchunks),
            },
        )
        elements: dict[tuple[str, int], ET.Element] = {}
        ids: dict[tuple[str, int], int] = {}
        next_id = 0
        for peer in sorted(incoming[rank]):
            key = ("r", peer)
            ids[key] = next_id
            elements[key] = ET.SubElement(gpu, "tb", {"id": str(next_id), "send": "-1", "recv": str(peer), "chan": "0"})
            next_id += 1
        for peer in sorted(outgoing[rank]):
            key = ("s", peer)
            ids[key] = next_id
            elements[key] = ET.SubElement(gpu, "tb", {"id": str(next_id), "send": str(peer), "recv": "-1", "chan": "0"})
            next_id += 1
        tb_elements.append(elements)
        tb_ids.append(ids)
        local_steps.append(defaultdict(int))

    previous_receive: dict[tuple[int, int, int], tuple[int, int]] = {}
    for transfer in transfers:
        token = (transfer.flow_source, transfer.flow_destination, transfer.subchunk)
        logical = transfer.flow_destination * subchunks + transfer.subchunk
        scratch = (transfer.flow_source * ranks + transfer.flow_destination) * subchunks + transfer.subchunk
        if transfer.hop == 0:
            srcbuf, srcoff = "i", logical
            depid, deps = -1, -1
        else:
            srcbuf, srcoff = "s", scratch
            depid, deps = previous_receive[token]
        if transfer.last_hop:
            dstbuf, dstoff = "o", transfer.flow_source * subchunks + transfer.subchunk
        else:
            dstbuf, dstoff = "s", scratch

        send_key = ("s", transfer.destination)
        send_step = local_steps[transfer.source][send_key]
        _step(
            tb_elements[transfer.source][send_key],
            step=send_step,
            kind="s",
            srcbuf=srcbuf,
            srcoff=srcoff,
            dstbuf=dstbuf,
            dstoff=dstoff,
            logical_chunk=logical,
            flow_source=transfer.flow_source,
            flow_destination=transfer.flow_destination,
            epoch=transfer.epoch,
            depid=depid,
            deps=deps,
        )
        local_steps[transfer.source][send_key] += 1

        recv_key = ("r", transfer.source)
        recv_step = local_steps[transfer.destination][recv_key]
        _step(
            tb_elements[transfer.destination][recv_key],
            step=recv_step,
            kind="r",
            srcbuf=srcbuf,
            srcoff=srcoff,
            dstbuf=dstbuf,
            dstoff=dstoff,
            logical_chunk=logical,
            flow_source=transfer.flow_source,
            flow_destination=transfer.flow_destination,
            epoch=transfer.epoch,
        )
        local_steps[transfer.destination][recv_key] += 1
        previous_receive[token] = (tb_ids[transfer.destination][recv_key], recv_step)
    return _write(root, output)
