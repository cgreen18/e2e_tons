#!/usr/bin/env python3
"""Stream deterministic repair and communication rewrites over Chakra ET files.

The protobuf module is deliberately not vendored here.  Run this tool with a
modern generated ``et_def_pb2.py`` on ``PYTHONPATH`` and a compatible protobuf
runtime, for example::

    PYTHONPATH=/path/to/generated/chakra_pb2 \
      /path/to/venv_chakra/bin/python tools/chakra_et_rewrite.py ...
"""

from __future__ import annotations

import argparse
import ast
from bisect import bisect_left, bisect_right
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import BinaryIO, Callable, Iterable, Sequence


PASS_ORDER = ("repair-deps", "promote-comm-ops")
COMM_NAME_TO_ENUM_NAME = {
    "nccl:all_reduce": "ALL_REDUCE",
    "nccl:_all_gather_base": "ALL_GATHER",
    "nccl:all_gather": "ALL_GATHER",
    "nccl:all_gather_into_tensor_coalesced": "ALL_GATHER",
    "nccl:_reduce_scatter_base": "REDUCE_SCATTER",
    "nccl:reduce_scatter": "REDUCE_SCATTER",
    "nccl:reduce_scatter_tensor_coalesced": "REDUCE_SCATTER",
    "nccl:all_to_all": "ALL_TO_ALL",
    "nccl:broadcast": "BROADCAST",
    "nccl:_broadcast_oop": "BROADCAST",
    "nccl:_reduce_oop": "REDUCE",
}
TRACE_NAME_RE = re.compile(r"^chakra\.(\d+)\.et$")
MANIFEST_NAME = "chakra_et_rewrite_manifest.json"
PROCESS_GROUP_METADATA_NAME = "## process_group:init ##"
RECORD_PARAM_COMMS_NAME = "record_param_comms"
GROUP_SEARCH_WINDOW = 8
_RECORD_PARAM_PG_RE = re.compile(
    r"\[\s*['\"](?P<pg_id>\d+)['\"]\s*,\s*"
    r"['\"](?P<kind>[A-Z][A-Z0-9_]*)['\"]\s*\]"
)


class TraceFormatError(ValueError):
    """Raised when a Chakra file is not a complete delimited protobuf stream."""


@dataclass(frozen=True)
class GroupSizeResolution:
    """Communicator-group resolution for one collective node."""

    group_size: int | None
    source: str
    pg_id: str | None
    record_node_id: int | None
    distance: int | None
    reason: str | None


@dataclass(frozen=True)
class _CollectiveControlContext:
    node_id: int
    control_scope: frozenset[int]


@dataclass(frozen=True)
class _RecordParamContext:
    node_id: int
    control_scope: frozenset[int]
    pg_id: str


def parse_process_group_registry(values: str, model_rank_count: int) -> dict[str, int]:
    """Parse ASTRA-sim's process-group metadata representation.

    ``Workload::issue_pytorch_pg_metadata`` removes two characters from each
    end of ``inputs.values`` before parsing the remaining JSON.  Empty rank
    lists mean all model ranks.
    """

    if model_rank_count < 1:
        raise ValueError("model rank count must be at least one")
    if len(values) < 4:
        raise ValueError("process-group metadata is too short for ASTRA trim")
    try:
        parsed = json.loads(values[2:-2])
    except json.JSONDecodeError as error:
        raise ValueError("process-group metadata is not valid trimmed JSON") from error
    if not isinstance(parsed, list):
        raise ValueError("process-group metadata must contain a JSON list")

    registry: dict[str, int] = {}
    for item in parsed:
        if not isinstance(item, dict):
            raise ValueError("process-group metadata entries must be objects")
        pg_id = item.get("pg_name")
        ranks = item.get("ranks")
        if not isinstance(pg_id, str) or not pg_id.isdigit():
            raise ValueError("process-group pg_name must be a decimal string")
        if not isinstance(ranks, list) or not all(
            isinstance(rank, int) and not isinstance(rank, bool) for rank in ranks
        ):
            raise ValueError("process-group ranks must be an integer list")
        if pg_id in registry:
            raise ValueError(f"duplicate process-group id {pg_id}")
        registry[pg_id] = len(ranks) if ranks else model_rank_count
    return registry


def parse_record_param_pg_id(values: str) -> str | None:
    """Extract a process-group id from a record_param_comms input value."""

    match = _RECORD_PARAM_PG_RE.search(values)
    return match.group("pg_id") if match is not None else None


class CollectiveGroupResolver:
    """Correlate collective nodes with nearby process-group records.

    Nodes must be observed in increasing id order.  A target and a
    ``record_param_comms`` node match when they are within ``search_window``
    ids and their direct/one-parent control scopes intersect.  The latter
    handles the common layout where the collective depends on a c10d launcher
    and the record node depends directly on their shared ProfilerStep.
    """

    def __init__(
        self,
        model_rank_count: int,
        metadata_node_type: int,
        *,
        search_window: int = GROUP_SEARCH_WINDOW,
    ) -> None:
        if model_rank_count < 1:
            raise ValueError("model rank count must be at least one")
        if search_window < 1:
            raise ValueError("group search window must be at least one")
        self.model_rank_count = model_rank_count
        self.metadata_node_type = metadata_node_type
        self.search_window = search_window
        self.registry: dict[str, int] | None = None
        self._last_node_id: int | None = None
        self._recent_ids: deque[int] = deque()
        self._recent_ctrl_deps: dict[int, tuple[int, ...]] = {}
        self._targets: list[_CollectiveControlContext] = []
        self._records: list[_RecordParamContext] = []

    def _control_scope(self, node) -> frozenset[int]:
        scope = set(node.ctrl_deps)
        for dependency in node.ctrl_deps:
            scope.update(self._recent_ctrl_deps.get(dependency, ()))
        return frozenset(scope)

    def observe_node(self, node, *, target: bool = False) -> None:
        """Observe one node and optionally register it for later resolution."""

        node_id = int(node.id)
        if self._last_node_id is not None and node_id <= self._last_node_id:
            raise TraceFormatError("group-size resolution requires increasing node ids")
        self._last_node_id = node_id
        while self._recent_ids and self._recent_ids[0] < node_id - self.search_window:
            expired = self._recent_ids.popleft()
            self._recent_ctrl_deps.pop(expired, None)

        scope = self._control_scope(node)
        if node.type == self.metadata_node_type and node.name == PROCESS_GROUP_METADATA_NAME:
            if self.registry is not None:
                raise TraceFormatError("trace has multiple process-group metadata nodes")
            try:
                self.registry = parse_process_group_registry(
                    node.inputs.values, self.model_rank_count
                )
            except ValueError as error:
                raise TraceFormatError(f"invalid process-group metadata: {error}") from error

        if node.name == RECORD_PARAM_COMMS_NAME:
            pg_id = parse_record_param_pg_id(node.inputs.values)
            if pg_id is not None:
                self._records.append(_RecordParamContext(node_id, scope, pg_id))
        if target:
            self._targets.append(_CollectiveControlContext(node_id, scope))

        self._recent_ids.append(node_id)
        self._recent_ctrl_deps[node_id] = tuple(int(dep) for dep in node.ctrl_deps)

    def resolve(self, *, fallback_to_all_ranks: bool) -> dict[int, GroupSizeResolution]:
        """Resolve all registered targets after the file has been observed."""

        if self.registry is None:
            raise TraceFormatError("trace is missing process-group metadata node")
        record_ids = [record.node_id for record in self._records]
        resolutions: dict[int, GroupSizeResolution] = {}
        for target in self._targets:
            first = bisect_left(record_ids, target.node_id - self.search_window)
            last = bisect_right(record_ids, target.node_id + self.search_window)
            candidates = [
                record
                for record in self._records[first:last]
                if target.control_scope & record.control_scope
            ]
            candidates.sort(
                key=lambda record: (
                    abs(record.node_id - target.node_id),
                    record.node_id < target.node_id,
                    record.node_id,
                )
            )
            record = candidates[0] if candidates else None
            if record is not None and record.pg_id in self.registry:
                resolutions[target.node_id] = GroupSizeResolution(
                    group_size=self.registry[record.pg_id],
                    source="direct",
                    pg_id=record.pg_id,
                    record_node_id=record.node_id,
                    distance=abs(record.node_id - target.node_id),
                    reason=None,
                )
                continue

            reason = "record-param-not-found"
            pg_id = None
            record_node_id = None
            distance = None
            if record is not None:
                reason = "pg-id-not-in-metadata"
                pg_id = record.pg_id
                record_node_id = record.node_id
                distance = abs(record.node_id - target.node_id)
            resolutions[target.node_id] = GroupSizeResolution(
                group_size=self.model_rank_count if fallback_to_all_ranks else None,
                source="fallback" if fallback_to_all_ranks else "unresolved",
                pg_id=pg_id,
                record_node_id=record_node_id,
                distance=distance,
                reason=reason,
            )
        return resolutions


def resolve_collective_group_sizes(
    path: Path,
    model_rank_count: int,
    target_selector: Callable[[object], bool],
    protobuf_module=None,
    *,
    search_window: int = GROUP_SEARCH_WINDOW,
    fallback_to_all_ranks: bool = False,
) -> dict[int, GroupSizeResolution]:
    """Scan one trace once and resolve selected collective node ids."""

    pb = protobuf_module or load_protobuf_module()
    resolver = CollectiveGroupResolver(
        model_rank_count, pb.METADATA_NODE, search_window=search_window
    )
    for record in iter_nodes(Path(path), pb):
        if isinstance(record, pb.GlobalMetadata):
            continue
        resolver.observe_node(record, target=target_selector(record))
    return resolver.resolve(fallback_to_all_ranks=fallback_to_all_ranks)


def summarize_group_size_resolutions(
    resolutions: Iterable[GroupSizeResolution],
) -> dict[str, object]:
    """Return JSON-friendly resolution counters and hit rate."""

    items = list(resolutions)
    sources = Counter(item.source for item in items)
    reasons = Counter(item.reason for item in items if item.reason is not None)
    direct = sources["direct"]
    not_direct = len(items) - direct
    return {
        "direct": direct,
        "fallback": sources["fallback"],
        "unresolved": sources["unresolved"],
        "not_direct": not_direct,
        "hit_rate": direct / len(items) if items else None,
        "reasons": dict(sorted(reasons.items())),
    }


def load_protobuf_module():
    """Load explicit or bootstrapped bindings, never the ancient committed copy."""

    try:
        import et_def_pb2  # type: ignore[import-not-found]
        return et_def_pb2
    except Exception:
        sys.modules.pop("et_def_pb2", None)

    from tools.chakra_pb2_bootstrap import (
        GENERATED_RELATIVE_DIR,
        REPOSITORY_ROOT,
        bootstrap_bindings,
    )

    generated_dir = REPOSITORY_ROOT / GENERATED_RELATIVE_DIR
    generated_path = str(generated_dir)
    if generated_path not in sys.path:
        sys.path.insert(0, generated_path)
    importlib.invalidate_caches()
    try:
        return importlib.import_module("et_def_pb2")
    except Exception:
        sys.modules.pop("et_def_pb2", None)

    # An import failure means an existing generated file cannot be trusted,
    # even if its schema stamp says it is current.
    generated_dir = bootstrap_bindings(force=True)
    generated_path = str(generated_dir)
    if generated_path not in sys.path:
        sys.path.insert(0, generated_path)
    importlib.invalidate_caches()
    try:
        return importlib.import_module("et_def_pb2")
    except Exception as error:
        raise RuntimeError(
            f"cannot import regenerated et_def_pb2 from {generated_dir}"
        ) from error


def _read_varint32(stream: BinaryIO) -> int | None:
    value = 0
    for byte_index in range(5):
        raw = stream.read(1)
        if not raw:
            if byte_index == 0:
                return None
            raise TraceFormatError("truncated record-length varint")
        byte = raw[0]
        value |= (byte & 0x7F) << (7 * byte_index)
        if not byte & 0x80:
            if byte_index == 4 and byte > 0x0F:
                raise TraceFormatError("record length exceeds uint32")
            return value
    raise TraceFormatError("record-length varint exceeds five bytes")


def _encode_varint32(value: int) -> bytes:
    if not 0 <= value <= 0xFFFFFFFF:
        raise TraceFormatError(f"record length {value} is outside uint32")
    encoded = bytearray()
    while value > 0x7F:
        encoded.append((value & 0x7F) | 0x80)
        value >>= 7
    encoded.append(value)
    return bytes(encoded)


def read_delimited(stream: BinaryIO, message_type):
    """Read one varint32-delimited message, returning ``None`` only at EOF."""

    size = _read_varint32(stream)
    if size is None:
        return None
    payload = stream.read(size)
    if len(payload) != size:
        raise TraceFormatError(
            f"truncated protobuf record: expected {size} bytes, got {len(payload)}"
        )
    message = message_type()
    try:
        message.ParseFromString(payload)
    except Exception as error:
        raise TraceFormatError(f"invalid {message_type.__name__} record") from error
    return message, payload


def write_delimited(stream: BinaryIO, message, *, payload: bytes | None = None) -> None:
    """Write one record using ASTRA-sim's varint32 protobuf framing."""

    if payload is None:
        payload = message.SerializeToString(deterministic=True)
    stream.write(_encode_varint32(len(payload)))
    stream.write(payload)


def iter_nodes(path: Path, protobuf_module=None):
    """Yield metadata followed by nodes without retaining serialized records."""

    pb = protobuf_module or load_protobuf_module()
    with path.open("rb") as stream:
        metadata_record = read_delimited(stream, pb.GlobalMetadata)
        if metadata_record is None:
            raise TraceFormatError(f"{path}: missing GlobalMetadata record")
        metadata, _ = metadata_record
        yield metadata
        while True:
            record = read_delimited(stream, pb.Node)
            if record is None:
                break
            node, _ = record
            yield node


def _scan_node_ids(path: Path, pb) -> tuple[set[int], int, str]:
    node_ids: set[int] = set()
    nodes_read = 0
    with path.open("rb") as stream:
        metadata_record = read_delimited(stream, pb.GlobalMetadata)
        if metadata_record is None:
            raise TraceFormatError(f"{path}: missing GlobalMetadata record")
        metadata, _ = metadata_record
        while True:
            record = read_delimited(stream, pb.Node)
            if record is None:
                break
            node, _ = record
            node_ids.add(node.id)
            nodes_read += 1
    return node_ids, nodes_read, metadata.version


def _is_tensor_descriptor(value: object) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 6
        and all(isinstance(field, int) and not isinstance(field, bool) for field in value[:5])
        and isinstance(value[5], str)
        and all(field >= 0 for field in value[:5])
    )


def _sum_tensor_descriptors(value: object) -> tuple[int, int]:
    if _is_tensor_descriptor(value):
        descriptor = value  # type: ignore[assignment]
        return descriptor[3] * descriptor[4], 1
    if isinstance(value, (list, tuple)):
        size = 0
        count = 0
        for child in value:
            child_size, child_count = _sum_tensor_descriptors(child)
            size += child_size
            count += child_count
        return size, count
    return 0, 0


def derive_comm_size(values: str) -> int:
    """Safely sum ``num_elem * elem_bytes`` over nested tensor descriptors."""

    try:
        parsed = ast.literal_eval(values)
    except (SyntaxError, ValueError) as error:
        raise ValueError("inputs.values is not a Python literal") from error
    size, descriptors = _sum_tensor_descriptors(parsed)
    if descriptors == 0:
        raise ValueError("inputs.values contains no tensor descriptors")
    if size > (1 << 63) - 1:
        raise ValueError("derived communication size exceeds int64")
    return size


def round_comm_size(raw_size: int, communicator_group_size: int) -> int:
    """Round to the nearest ``256 * group_size`` bytes, with a one-unit floor.

    Exact half-unit ties round upward.  This intentionally avoids Python's
    ties-to-even ``round`` behavior.
    """

    if raw_size < 0:
        raise ValueError("raw communication size must be non-negative")
    if communicator_group_size < 1:
        raise ValueError("communicator group size must be at least one")
    unit = 256 * communicator_group_size
    rounded = max(unit, ((raw_size + unit // 2) // unit) * unit)
    if rounded > (1 << 63) - 1:
        raise ValueError("rounded communication size exceeds int64")
    return rounded


def _has_true_cpu_attr(node) -> bool:
    return any(
        attr.name == "is_cpu_op"
        and attr.WhichOneof("value") == "bool_val"
        and attr.bool_val
        for attr in node.attr
    )


def _set_int64_attr(node, name: str, value: int) -> None:
    matches = [index for index, attr in enumerate(node.attr) if attr.name == name]
    if matches:
        node.attr[matches[0]].int64_val = value
        for index in reversed(matches[1:]):
            del node.attr[index]
    else:
        attr = node.attr.add()
        attr.name = name
        attr.int64_val = value


def inspect_promotable_comm_op(
    node, pb
) -> tuple[str | None, int | None, str | None]:
    """Classify a possible launcher without mutating it."""

    if not node.name.startswith("nccl:"):
        return None, None, None
    enum_name = COMM_NAME_TO_ENUM_NAME.get(node.name)
    if enum_name is None:
        return None, None, "unknown-name"
    if node.type != pb.COMP_NODE or not _has_true_cpu_attr(node):
        return None, None, "not-cpu-comp-node"
    try:
        size = derive_comm_size(node.inputs.values)
    except ValueError:
        return None, None, "communication-size-unavailable"
    return enum_name, size, None


def promote_comm_op(
    node, pb, communicator_group_size: int
) -> tuple[str | None, int | None, str | None]:
    """Promote one known CPU NCCL launcher.

    Returns ``(enum_name, size, reason)``.  Non-NCCL nodes return three
    ``None`` values; an NCCL node left unchanged returns a reason.
    """

    enum_name, raw_size, reason = inspect_promotable_comm_op(node, pb)
    if enum_name is None:
        return None, None, reason
    assert raw_size is not None
    try:
        size = round_comm_size(raw_size, communicator_group_size)
    except ValueError:
        return None, None, "communication-size-unavailable"
    node.type = pb.COMM_COLL_NODE
    _set_int64_attr(node, "comm_type", getattr(pb, enum_name))
    _set_int64_attr(node, "comm_size", size)
    return enum_name, size, None


def _empty_stats(rank: int, source: Path, target: Path) -> dict[str, object]:
    return {
        "rank": rank,
        "input_file": str(source.resolve()),
        "output_file": str(target.resolve()),
        "schema_version": "",
        "nodes_read": 0,
        "ctrl_deps_dropped": 0,
        "data_deps_dropped": 0,
        "deps_dropped": 0,
        "ops_promoted": 0,
        "ops_left_unmapped": 0,
        "group_size_resolutions_direct": 0,
        "group_size_resolutions_fallback": 0,
        "group_size_resolution_hit_rate": None,
        "group_size_resolution_fallback_reasons": {},
        "unmapped_names": {},
        "unmapped_reasons": {},
        "bytes_attributed_per_collective_type": {},
    }


def rewrite_trace(
    source: Path,
    target: Path,
    passes: Sequence[str],
    rank: int,
    protobuf_module=None,
    model_rank_count: int | None = None,
) -> dict[str, object]:
    """Rewrite one trace atomically and return its provenance counters."""

    pb = protobuf_module or load_protobuf_module()
    selected = tuple(name for name in PASS_ORDER if name in passes)
    unknown_passes = sorted(set(passes) - set(PASS_ORDER))
    if unknown_passes:
        raise ValueError(f"unknown passes: {', '.join(unknown_passes)}")
    if not selected:
        raise ValueError("at least one rewrite pass is required")

    source = Path(source)
    target = Path(target)
    if source.resolve() == target.resolve():
        raise ValueError("input and output files must differ")
    if not source.is_file():
        raise FileNotFoundError(source)
    with source.open("rb") as stream:
        if stream.read(2) == b"\x1f\x8b":
            raise TraceFormatError(f"{source}: gzip input is not supported")

    node_ids: set[int] | None = None
    scanned_nodes: int | None = None
    scanned_version: str | None = None
    if "repair-deps" in selected:
        node_ids, scanned_nodes, scanned_version = _scan_node_ids(source, pb)

    group_resolutions: dict[int, GroupSizeResolution] = {}
    if "promote-comm-ops" in selected:
        if model_rank_count is None:
            model_rank_count = infer_model_rank_count(source)

        def promotion_target(node) -> bool:
            enum_name, _, _ = inspect_promotable_comm_op(node, pb)
            return enum_name is not None

        group_resolutions = resolve_collective_group_sizes(
            source,
            model_rank_count,
            promotion_target,
            pb,
            fallback_to_all_ranks=True,
        )
        resolution_summary = summarize_group_size_resolutions(
            group_resolutions.values()
        )
        stats_resolution_summary = resolution_summary
    else:
        stats_resolution_summary = None

    target.parent.mkdir(parents=True, exist_ok=True)
    stats = _empty_stats(rank, source, target)
    bytes_by_type: Counter[str] = Counter()
    unmapped_names: Counter[str] = Counter()
    unmapped_reasons: Counter[str] = Counter()
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as output:
            temporary_name = output.name
            with source.open("rb") as input_stream:
                metadata_record = read_delimited(input_stream, pb.GlobalMetadata)
                if metadata_record is None:
                    raise TraceFormatError(f"{source}: missing GlobalMetadata record")
                metadata, metadata_payload = metadata_record
                stats["schema_version"] = metadata.version
                write_delimited(output, metadata, payload=metadata_payload)

                while True:
                    record = read_delimited(input_stream, pb.Node)
                    if record is None:
                        break
                    node, _ = record
                    stats["nodes_read"] += 1

                    if node_ids is not None:
                        original_ctrl = len(node.ctrl_deps)
                        original_data = len(node.data_deps)
                        kept_ctrl = [dep for dep in node.ctrl_deps if dep in node_ids]
                        kept_data = [dep for dep in node.data_deps if dep in node_ids]
                        del node.ctrl_deps[:]
                        del node.data_deps[:]
                        node.ctrl_deps.extend(kept_ctrl)
                        node.data_deps.extend(kept_data)
                        stats["ctrl_deps_dropped"] += original_ctrl - len(kept_ctrl)
                        stats["data_deps_dropped"] += original_data - len(kept_data)

                    if "promote-comm-ops" in selected:
                        inspected_enum, _, _ = inspect_promotable_comm_op(node, pb)
                        resolution = (
                            group_resolutions.get(int(node.id))
                            if inspected_enum is not None
                            else None
                        )
                        if inspected_enum is not None and (
                            resolution is None or resolution.group_size is None
                        ):
                            raise TraceFormatError(
                                f"node {node.id}: missing fallback communicator group size"
                            )
                        group_size = (
                            resolution.group_size if resolution is not None else 1
                        )
                        enum_name, size, reason = promote_comm_op(
                            node, pb, int(group_size)
                        )
                        if enum_name is not None:
                            stats["ops_promoted"] += 1
                            bytes_by_type[enum_name] += size
                        elif reason is not None:
                            stats["ops_left_unmapped"] += 1
                            unmapped_names[node.name] += 1
                            unmapped_reasons[reason] += 1

                    write_delimited(output, node)

        if scanned_nodes is not None and stats["nodes_read"] != scanned_nodes:
            raise TraceFormatError(f"{source}: node count changed between streaming scans")
        if scanned_version is not None and stats["schema_version"] != scanned_version:
            raise TraceFormatError(f"{source}: metadata changed between streaming scans")
        stats["deps_dropped"] = (
            stats["ctrl_deps_dropped"] + stats["data_deps_dropped"]
        )
        stats["bytes_attributed_per_collective_type"] = dict(sorted(bytes_by_type.items()))
        stats["unmapped_names"] = dict(sorted(unmapped_names.items()))
        stats["unmapped_reasons"] = dict(sorted(unmapped_reasons.items()))
        if stats_resolution_summary is not None:
            stats["group_size_resolutions_direct"] = stats_resolution_summary["direct"]
            stats["group_size_resolutions_fallback"] = stats_resolution_summary["fallback"]
            stats["group_size_resolution_hit_rate"] = stats_resolution_summary["hit_rate"]
            stats["group_size_resolution_fallback_reasons"] = (
                stats_resolution_summary["reasons"]
            )
        os.replace(temporary_name, target)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass
    return stats


def _parse_rank(path: Path) -> int:
    match = TRACE_NAME_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"trace filename must match chakra.<rank>.et: {path.name}")
    return int(match.group(1))


def parse_rank_range(specification: str) -> tuple[int, int] | None:
    """Parse ``all``, one rank, inclusive ``A-B``, or half-open ``A:B``."""

    if specification == "all":
        return None
    if specification.isdigit():
        rank = int(specification)
        return rank, rank + 1
    separator = ":" if ":" in specification else "-" if "-" in specification else None
    if separator is None:
        raise ValueError("rank range must be all, RANK, START-END, or START:END")
    fields = specification.split(separator)
    if len(fields) != 2 or not all(field.isdigit() for field in fields):
        raise ValueError("rank range must contain two non-negative integers")
    start, end = map(int, fields)
    if separator == "-":
        end += 1
    if end <= start:
        raise ValueError("rank range must be non-empty and increasing")
    return start, end


def discover_inputs(
    source: Path, rank_range: tuple[int, int] | None
) -> list[tuple[int, Path]]:
    source = Path(source)
    if source.is_file():
        rank = _parse_rank(source)
        if rank_range is not None and not rank_range[0] <= rank < rank_range[1]:
            raise ValueError(f"rank {rank} is outside the requested range")
        return [(rank, source)]
    if not source.is_dir():
        raise FileNotFoundError(source)
    available = {
        _parse_rank(path): path
        for path in source.iterdir()
        if TRACE_NAME_RE.fullmatch(path.name)
    }
    if rank_range is None:
        ranks = sorted(available)
    else:
        ranks = list(range(*rank_range))
        missing = [rank for rank in ranks if rank not in available]
        if missing:
            preview = ", ".join(map(str, missing[:10]))
            suffix = " ..." if len(missing) > 10 else ""
            raise FileNotFoundError(f"missing requested trace ranks: {preview}{suffix}")
    if not ranks:
        raise FileNotFoundError(f"no chakra.<rank>.et files found in {source}")
    return [(rank, available[rank]) for rank in ranks]


def infer_model_rank_count(source: Path) -> int:
    """Infer a model's total ranks from its complete trace directory."""

    source = Path(source)
    directory = source if source.is_dir() else source.parent
    ranks = sorted(
        _parse_rank(path)
        for path in directory.iterdir()
        if TRACE_NAME_RE.fullmatch(path.name)
    )
    if not ranks:
        raise ValueError(f"cannot infer model rank count from {directory}")
    expected = list(range(ranks[-1] + 1))
    if ranks != expected:
        raise ValueError(
            "cannot infer model rank count from a non-contiguous trace directory; "
            "provide model_rank_count/--model-ranks"
        )
    return len(ranks)


def _rewrite_task(arguments):
    source, target, passes, rank, model_rank_count = arguments
    return rewrite_trace(
        source, target, passes, rank, model_rank_count=model_rank_count
    )


def _tool_git_commit() -> str:
    repository = Path(__file__).resolve().parents[1]
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _sum_stats(per_rank: Iterable[dict[str, object]]) -> dict[str, object]:
    scalar_fields = (
        "nodes_read",
        "ctrl_deps_dropped",
        "data_deps_dropped",
        "deps_dropped",
        "ops_promoted",
        "ops_left_unmapped",
        "group_size_resolutions_direct",
        "group_size_resolutions_fallback",
    )
    totals: dict[str, object] = {field: 0 for field in scalar_fields}
    bytes_by_type: Counter[str] = Counter()
    unmapped_names: Counter[str] = Counter()
    unmapped_reasons: Counter[str] = Counter()
    group_resolution_reasons: Counter[str] = Counter()
    for stats in per_rank:
        for field in scalar_fields:
            totals[field] += stats[field]
        bytes_by_type.update(stats["bytes_attributed_per_collective_type"])
        unmapped_names.update(stats["unmapped_names"])
        unmapped_reasons.update(stats["unmapped_reasons"])
        group_resolution_reasons.update(
            stats["group_size_resolution_fallback_reasons"]
        )
    totals["bytes_attributed_per_collective_type"] = dict(sorted(bytes_by_type.items()))
    totals["unmapped_names"] = dict(sorted(unmapped_names.items()))
    totals["unmapped_reasons"] = dict(sorted(unmapped_reasons.items()))
    total_resolutions = (
        totals["group_size_resolutions_direct"]
        + totals["group_size_resolutions_fallback"]
    )
    totals["group_size_resolution_hit_rate"] = (
        totals["group_size_resolutions_direct"] / total_resolutions
        if total_resolutions
        else None
    )
    totals["group_size_resolution_fallback_reasons"] = dict(
        sorted(group_resolution_reasons.items())
    )
    return totals


def rewrite_collection(
    source: Path,
    output_dir: Path,
    passes: Sequence[str],
    rank_range: tuple[int, int] | None,
    jobs: int,
    model_rank_count: int | None = None,
) -> tuple[Path, dict[str, object]]:
    """Rewrite a file/range and write one deterministic provenance manifest."""

    if jobs < 1:
        raise ValueError("jobs must be at least one")
    # Bootstrap once in the parent before rank workers start concurrently.
    load_protobuf_module()
    source = Path(source)
    output_dir = Path(output_dir)
    selected = tuple(name for name in PASS_ORDER if name in passes)
    if len(selected) != len(set(passes)) or set(selected) != set(passes):
        raise ValueError("passes must be unique supported pass names")
    if source.is_dir() and source.resolve() == output_dir.resolve():
        raise ValueError("input and output directories must differ")
    if model_rank_count is None:
        model_rank_count = infer_model_rank_count(source)
    elif model_rank_count < 1:
        raise ValueError("model rank count must be at least one")
    inputs = discover_inputs(source, rank_range)
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        (path, output_dir / path.name, selected, rank, model_rank_count)
        for rank, path in inputs
    ]
    for input_path, output_path, _, _, _ in tasks:
        if input_path.resolve() == output_path.resolve():
            raise ValueError("input and output files must differ")

    if jobs == 1 or len(tasks) == 1:
        reports = [_rewrite_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=min(jobs, len(tasks))) as executor:
            reports = list(executor.map(_rewrite_task, tasks))
    reports.sort(key=lambda item: item["rank"])

    versions = sorted({str(report["schema_version"]) for report in reports})
    if len(versions) != 1:
        raise ValueError(f"input ranks use inconsistent Chakra schema versions: {versions}")
    manifest: dict[str, object] = {
        "source_dir": str((source if source.is_dir() else source.parent).resolve()),
        "source_input": str(source.resolve()),
        "output_dir": str(output_dir.resolve()),
        "passes_applied": list(selected),
        "ranks": [report["rank"] for report in reports],
        "model_rank_count": model_rank_count,
        "per_rank": {str(report["rank"]): report for report in reports},
        "totals": _sum_stats(reports),
        "name_to_enum": COMM_NAME_TO_ENUM_NAME if "promote-comm-ops" in selected else {},
        "chakra_schema_version": versions[0],
        "tool_git_commit": _tool_git_commit(),
    }
    manifest_path = output_dir / MANIFEST_NAME
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{MANIFEST_NAME}.",
            suffix=".tmp",
            dir=output_dir,
            delete=False,
        ) as stream:
            temporary_name = stream.name
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary_name, manifest_path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass
    return manifest_path, manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="input directory or chakra.<rank>.et file")
    parser.add_argument("output_dir", type=Path, help="distinct output directory")
    parser.add_argument(
        "--passes",
        nargs="+",
        choices=PASS_ORDER,
        required=True,
        help="one or both independent rewrite passes",
    )
    parser.add_argument(
        "--rank-range",
        required=True,
        help="all, RANK, inclusive START-END, or half-open START:END",
    )
    parser.add_argument("--jobs", type=int, default=1, help="parallel rank workers (default: 1)")
    parser.add_argument(
        "--model-ranks",
        type=int,
        help="total model ranks (default: infer contiguous ranks from input directory)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    try:
        rank_range = parse_rank_range(arguments.rank_range)
        manifest_path, manifest = rewrite_collection(
            arguments.input,
            arguments.output_dir,
            arguments.passes,
            rank_range,
            arguments.jobs,
            arguments.model_ranks,
        )
    except (FileNotFoundError, RuntimeError, TraceFormatError, ValueError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {"manifest": str(manifest_path), "totals": manifest["totals"]},
            sort_keys=True,
        )
    )
    promoted = int(manifest["totals"]["ops_promoted"])
    fallback = int(manifest["totals"]["group_size_resolutions_fallback"])
    if promoted and fallback / promoted >= 0.10:
        print(
            "WARNING: communicator group size fell back to all model ranks for "
            f"{fallback}/{promoted} promoted ops ({fallback / promoted:.1%}); "
            "this run is not mostly directly resolved",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
