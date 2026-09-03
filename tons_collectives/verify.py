"""Static consistency checks for MSCCL XML schedules."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET


@dataclass
class ScheduleReport:
    path: str
    collective: str = ""
    ranks: int = 0
    sends: int = 0
    receives: int = 0
    reductions: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "collective": self.collective,
            "ranks": self.ranks,
            "sends": self.sends,
            "receives": self.receives,
            "reductions": self.reductions,
            "errors": self.errors,
            "ok": self.ok,
        }


def _message_key(rank: int, tb: ET.Element, step: ET.Element) -> tuple[object, ...]:
    kind = step.attrib.get("type")
    peer = int(tb.attrib["send"] if kind == "s" else tb.attrib["recv"])
    source, destination = (rank, peer) if kind == "s" else (peer, rank)
    identity = (
        step.attrib.get("flow_src"),
        step.attrib.get("flow_dst"),
        step.attrib.get("logical_chunk"),
        step.attrib.get("epoch", step.attrib.get("s")),
    )
    if all(value is None for value in identity[:3]):
        identity = (
            step.attrib.get("srcoff"),
            step.attrib.get("dstoff"),
            step.attrib.get("s"),
            None,
        )
    return (
        source,
        destination,
        int(tb.attrib.get("chan", "0")),
        step.attrib.get("cnt"),
        *identity,
    )


def verify_schedule(path: Path | str) -> ScheduleReport:
    path = Path(path)
    report = ScheduleReport(str(path))
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as error:
        report.errors.append(str(error))
        return report
    if root.tag != "algo":
        report.errors.append("root element must be <algo>")
        return report
    try:
        report.ranks = int(root.attrib["ngpus"])
    except (KeyError, ValueError):
        report.errors.append("algo.ngpus must be an integer")
        return report
    report.collective = root.attrib.get("coll", "")
    if report.collective not in {
        "allgather",
        "allreduce",
        "alltoall",
        "reduce_scatter",
        "reduce",
        "broadcast",
        # A gather carries no reduction, so it is named separately from
        # "reduce" to keep the reduce-op check below meaningful.
        "gather",
    }:
        report.errors.append(f"unsupported collective {report.collective!r}")

    gpus = root.findall("gpu")
    ids: list[int] = []
    sends: Counter[tuple[object, ...]] = Counter()
    receives: Counter[tuple[object, ...]] = Counter()
    reduction_chunks: Counter[int] = Counter()
    try:
        workload_chunks = int(root.attrib["input_chunks"])
    except (KeyError, ValueError):
        workload_chunks = 0
    for gpu in gpus:
        try:
            rank = int(gpu.attrib["id"])
        except (KeyError, ValueError):
            report.errors.append("gpu.id must be an integer")
            continue
        ids.append(rank)
        threadblocks: dict[int, ET.Element] = {}
        step_ids: dict[int, set[int]] = {}
        graph: dict[tuple[int, int], set[tuple[int, int]]] = defaultdict(set)
        indegree: Counter[tuple[int, int]] = Counter()
        for tb in gpu.findall("tb"):
            try:
                tb_id = int(tb.attrib["id"])
                send = int(tb.attrib["send"])
                recv = int(tb.attrib["recv"])
            except (KeyError, ValueError):
                report.errors.append(f"rank {rank}: malformed threadblock")
                continue
            if tb_id in threadblocks:
                report.errors.append(f"rank {rank}: duplicate threadblock {tb_id}")
            threadblocks[tb_id] = tb
            if (send >= 0) == (recv >= 0):
                report.errors.append(f"rank {rank} tb {tb_id}: exactly one of send/recv must be set")
            if send >= report.ranks or recv >= report.ranks:
                report.errors.append(f"rank {rank} tb {tb_id}: peer is out of range")
            seen: set[int] = set()
            previous: tuple[int, int] | None = None
            for step in tb.findall("step"):
                try:
                    step_id = int(step.attrib["s"])
                    count = int(step.attrib["cnt"])
                except (KeyError, ValueError):
                    report.errors.append(f"rank {rank} tb {tb_id}: malformed step")
                    continue
                node = (tb_id, step_id)
                if step_id in seen:
                    report.errors.append(f"rank {rank} tb {tb_id}: duplicate step {step_id}")
                seen.add(step_id)
                indegree[node] += 0
                if count <= 0:
                    report.errors.append(f"rank {rank} tb {tb_id} step {step_id}: cnt must be positive")
                if "logical_chunk" in step.attrib and workload_chunks:
                    logical = int(step.attrib["logical_chunk"])
                    if logical < 0 or logical + count > workload_chunks:
                        report.errors.append(
                            f"rank {rank} tb {tb_id} step {step_id}: logical chunk range "
                            f"[{logical}, {logical + count}) exceeds workload chunk count {workload_chunks}"
                        )
                if previous is not None:
                    graph[previous].add(node)
                    indegree[node] += 1
                previous = node
                kind = step.attrib.get("type")
                if kind == "s":
                    report.sends += 1
                    sends[_message_key(rank, tb, step)] += 1
                elif kind == "r":
                    report.receives += 1
                    receives[_message_key(rank, tb, step)] += 1
                elif kind == "rrc":
                    report.receives += 1
                    report.reductions += 1
                    receives[_message_key(rank, tb, step)] += 1
                    if "logical_chunk" in step.attrib:
                        reduction_chunks[int(step.attrib["logical_chunk"])] += count
                elif kind not in {"cpy", "nop"}:
                    report.errors.append(f"rank {rank} tb {tb_id} step {step_id}: unknown type {kind!r}")
                for attribute in ("logical_chunk", "cnt"):
                    if attribute in step.attrib and int(step.attrib[attribute]) < 0:
                        report.errors.append(
                            f"rank {rank} tb {tb_id} step {step_id}: {attribute} is negative"
                        )
            step_ids[tb_id] = seen

        for tb_id, tb in threadblocks.items():
            for step in tb.findall("step"):
                depid = int(step.attrib.get("depid", "-1"))
                deps = int(step.attrib.get("deps", "-1"))
                if depid < 0:
                    continue
                node = (tb_id, int(step.attrib["s"]))
                dependency = (depid, deps)
                if depid not in step_ids or deps not in step_ids[depid]:
                    report.errors.append(f"rank {rank}: missing dependency {dependency} for {node}")
                    continue
                if node not in graph[dependency]:
                    graph[dependency].add(node)
                    indegree[node] += 1
        ready = deque(node for node in indegree if indegree[node] == 0)
        visited = 0
        while ready:
            node = ready.popleft()
            visited += 1
            for child in graph[node]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
        if visited != len(indegree):
            report.errors.append(f"rank {rank}: dependency graph contains a cycle")

    if sorted(ids) != list(range(report.ranks)):
        report.errors.append(f"gpu ids must be dense 0..{report.ranks - 1}")
    missing_receives = sends - receives
    missing_sends = receives - sends
    if missing_receives:
        report.errors.append(f"{sum(missing_receives.values())} sends have no matching receive")
    if missing_sends:
        report.errors.append(f"{sum(missing_sends.values())} receives have no matching send")
    if report.collective in {"allreduce", "reduce_scatter", "reduce"} and report.reductions == 0:
        report.errors.append(f"{report.collective} schedule contains no receive-reduce operations")
    if (
        report.collective in {"allreduce", "reduce_scatter"}
        and workload_chunks
        and reduction_chunks
    ):
        incomplete = [
            chunk
            for chunk in range(workload_chunks)
            if reduction_chunks[chunk] != report.ranks - 1
        ]
        if incomplete:
            report.errors.append(
                f"{len(incomplete)} logical chunks do not contain exactly "
                f"{report.ranks - 1} reductions"
            )
    return report
