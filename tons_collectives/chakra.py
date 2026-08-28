"""Small, dependency-free Chakra ET writer and MSCCL lowering.

Chakra execution traces are length-delimited protobuf messages.  The project
normally generates Python protobuf bindings during installation, but keeping a
minimal writer here makes experiment preparation reproducible even when only
the checked-out submodules are available.  The wire fields below are taken
from Chakra's ``schema/protobuf/et_def.proto``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET


COMM_SEND_NODE = 5
COMM_RECV_NODE = 6
COMM_COLL_NODE = 7

COLLECTIVE_TYPES = {
    "allreduce": 0,
    "reduce": 1,
    "allgather": 2,
    "broadcast": 5,
    "alltoall": 6,
    "reduce_scatter": 7,
}


def _varint(value: int) -> bytes:
    if value < 0:
        value &= (1 << 64) - 1
    output = bytearray()
    while value > 0x7F:
        output.append((value & 0x7F) | 0x80)
        value >>= 7
    output.append(value)
    return bytes(output)


def _key(field: int, wire: int) -> bytes:
    return _varint((field << 3) | wire)


def _integer(field: int, value: int) -> bytes:
    return _key(field, 0) + _varint(value)


def _bytes(field: int, value: bytes) -> bytes:
    return _key(field, 2) + _varint(len(value)) + value


def _string(field: int, value: str) -> bytes:
    return _bytes(field, value.encode("utf-8"))


def _attribute(name: str, value: int | bool | str, kind: str) -> bytes:
    message = _string(1, name)
    fields = {"int32": 7, "int64": 9, "bool": 27, "string": 29}
    if kind not in fields:
        raise ValueError(f"unsupported Chakra attribute type {kind!r}")
    if kind == "string":
        return message + _string(fields[kind], str(value))
    return message + _integer(fields[kind], int(value))


def _metadata() -> bytes:
    return _string(1, "0.0.4")


@dataclass
class ChakraNode:
    node_id: int
    name: str
    node_type: int
    dependencies: list[int] = field(default_factory=list)
    attributes: list[tuple[str, int | bool | str, str]] = field(default_factory=list)

    def serialize(self) -> bytes:
        message = _integer(1, self.node_id) + _string(2, self.name) + _integer(3, self.node_type)
        for dependency in self.dependencies:
            message += _integer(5, dependency)
        for name, value, kind in self.attributes:
            message += _bytes(10, _attribute(name, value, kind))
        return message


def write_trace(path: Path | str, nodes: list[ChakraNode]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        for message in [_metadata(), *(node.serialize() for node in nodes)]:
            stream.write(_varint(len(message)))
            stream.write(message)
    return path


def _common_step_attributes(
    step: ET.Element,
    tb: ET.Element,
    *,
    workload_chunks: int,
    receive: bool,
) -> list[tuple[str, int, str]]:
    offset_name = "dstoff" if receive else "srcoff"
    logical_chunk = int(step.attrib.get("logical_chunk", step.attrib[offset_name]))
    return [
        ("msg_chunk_cnt", int(step.attrib["cnt"]), "int32"),
        ("msg_chunk_idx", logical_chunk, "int32"),
        ("workload_chunk_cnt", workload_chunks, "int32"),
        # Kept for ASTRA versions that predate workload_chunk_cnt.
        ("total_chunk_cnt", workload_chunks, "int32"),
        ("local_time_step", int(step.attrib["s"]), "int32"),
        ("chunk_offset", int(step.attrib[offset_name]), "int32"),
        ("hasdep", int(step.attrib.get("hasdep", "0")), "int32"),
        ("deps", int(step.attrib.get("deps", "-1")), "int32"),
        ("depid", int(step.attrib.get("depid", "-1")), "int32"),
        ("tb_id", int(tb.attrib["id"]), "int32"),
    ]


def lower_msccl_to_chakra(input_xml: Path | str, output_prefix: Path | str) -> list[Path]:
    """Lower one MSCCL XML schedule to one schedule ET per rank."""

    root = ET.parse(input_xml).getroot()
    output_prefix = Path(output_prefix)
    paths: list[Path] = []
    root_chunks = root.attrib.get("input_chunks")
    for gpu in root.findall("gpu"):
        rank = int(gpu.attrib["id"])
        workload_chunks = int(root_chunks or gpu.attrib.get("i_chunks") or gpu.attrib["o_chunks"])
        nodes: list[ChakraNode] = []
        emitted: dict[tuple[int, int], int] = {}
        terminal: dict[tuple[int, int], int] = {}
        next_id = 0
        # Create every node before resolving cross-threadblock dependencies;
        # l3ss emits outgoing threadblocks before the receives they depend on.
        for tb in gpu.findall("tb"):
            tb_id = int(tb.attrib["id"])
            for step in tb.findall("step"):
                step_id = int(step.attrib["s"])
                kind = step.attrib["type"]
                if kind == "nop":
                    continue
                receive = kind in {"r", "rrc"}
                if kind not in {"s", "r", "rrc"}:
                    raise ValueError(f"rank {rank}: unsupported MSCCL step type {kind!r}")
                peer_attr = "comm_src" if receive else "comm_dst"
                peer = int(tb.attrib["recv"] if receive else tb.attrib["send"])
                node_type = COMM_RECV_NODE if receive else COMM_SEND_NODE
                attrs: list[tuple[str, int | bool | str, str]] = [
                    ("comm_type", node_type, "int64"),
                    (peer_attr, peer, "int32"),
                    ("comm_tag", int(tb.attrib.get("chan", "0")), "int32"),
                    *_common_step_attributes(
                        step, tb, workload_chunks=workload_chunks, receive=receive
                    ),
                ]
                if kind == "rrc":
                    attrs.append(("is_rrc", True, "bool"))
                node = ChakraNode(
                    next_id,
                    f"COMM_{'RECV' if receive else 'SEND'}_NODE_tb{tb_id}_step{step_id}",
                    node_type,
                    [],
                    attrs,
                )
                nodes.append(node)
                emitted[(tb_id, step_id)] = next_id
                terminal[(tb_id, step_id)] = next_id
                next_id += 1
                if kind == "rrc":
                    nodes.append(
                        ChakraNode(
                            next_id,
                            f"COMP_NODE_tb{tb_id}_step{step_id}",
                            4,
                            [terminal[(tb_id, step_id)]],
                        )
                    )
                    terminal[(tb_id, step_id)] = next_id
                    next_id += 1
        for tb in gpu.findall("tb"):
            tb_id = int(tb.attrib["id"])
            previous: int | None = None
            for step in tb.findall("step"):
                if step.attrib["type"] == "nop":
                    continue
                key = (tb_id, int(step.attrib["s"]))
                target = nodes[emitted[key]]
                depid = int(step.attrib.get("depid", "-1"))
                deps = int(step.attrib.get("deps", "-1"))
                if depid >= 0:
                    dependency = terminal.get((depid, deps))
                    if dependency is None:
                        raise ValueError(f"rank {rank}: unresolved dependency {(depid, deps)}")
                    target.dependencies.append(dependency)
                if previous is not None and previous not in target.dependencies:
                    target.dependencies.append(previous)
                previous = terminal[key]
        path = output_prefix.parent / f"{output_prefix.name}.{rank}.et"
        paths.append(write_trace(path, nodes))
    return paths
