import csv
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

from tools.chakra_pb2_bootstrap import (
    ChakraVenvUnavailable,
    bootstrap_bindings,
    chakra_venv_fix_commands,
)
from tools.chakra_et_rewrite import write_delimited
from tools.chakra_collective_profile import (
    ModelSpec,
    buffer_size_range,
    markdown_table,
    profile_models,
    write_profile_outputs,
)


try:
    generated_bindings = bootstrap_bindings()
    sys.path.insert(0, str(generated_bindings))
    import et_def_pb2 as chakra_pb
except ChakraVenvUnavailable:
    chakra_pb = None


PROTOBUF_SKIP_REASON = (
    "venv_chakra is genuinely absent; fix with exactly: "
    f"{chakra_venv_fix_commands()}"
)


def _metadata(groups):
    node = chakra_pb.Node(
        id=1,
        name="## process_group:init ##",
        type=chakra_pb.METADATA_NODE,
    )
    node.inputs.values = "[[" + json.dumps(groups) + "]]"
    return node


def _record(node_id: int, pg_id: str, parent: int):
    node = chakra_pb.Node(
        id=node_id,
        name="record_param_comms",
        type=chakra_pb.COMP_NODE,
        ctrl_deps=[parent],
    )
    node.inputs.values = f"['{pg_id}', 'DATA_PARALLEL_GROUP']"
    return node


def _set_int64(node, name: str, value: int) -> None:
    attr = node.attr.add()
    attr.name = name
    attr.int64_val = value


def _write(path: Path, nodes) -> None:
    with path.open("wb") as stream:
        write_delimited(stream, chakra_pb.GlobalMetadata(version="profile-test-v1"))
        for node in nodes:
            write_delimited(stream, node)


@unittest.skipUnless(chakra_pb is not None, PROTOBUF_SKIP_REASON)
class ChakraCollectiveProfileTest(unittest.TestCase):
    def test_buffer_bins_have_explicit_half_open_boundaries(self) -> None:
        self.assertEqual("0-4KiB", buffer_size_range(4095))
        self.assertEqual("4-64KiB", buffer_size_range(4096))
        self.assertEqual("64KiB-1MiB", buffer_size_range(65536))
        self.assertEqual("1GiB+", buffer_size_range(1 << 30))

    def test_profiles_device_and_in_memory_promoted_collectives(self) -> None:
        groups = [
            {
                "pg_name": "3",
                "pg_desc": "test",
                "backend_config": "",
                "ranks": [0, 1, 2, 3],
            }
        ]
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            device_dir = root / "device"
            promoted_dir = root / "promoted"
            device_dir.mkdir()
            promoted_dir.mkdir()

            step = chakra_pb.Node(id=10, name="ProfilerStep#1", type=chakra_pb.COMP_NODE)
            c10d = chakra_pb.Node(
                id=11,
                name="c10d::allreduce_",
                type=chakra_pb.COMP_NODE,
                ctrl_deps=[10],
            )
            kernel = chakra_pb.Node(
                id=12,
                name="ncclDevKernel_AllReduce",
                type=chakra_pb.COMM_COLL_NODE,
                ctrl_deps=[11],
            )
            _set_int64(kernel, "comm_type", chakra_pb.ALL_REDUCE)
            _set_int64(kernel, "comm_size", 4096)
            other_step = chakra_pb.Node(
                id=18, name="ProfilerStep#2", type=chakra_pb.COMP_NODE
            )
            other_c10d = chakra_pb.Node(
                id=19,
                name="c10d::allgather_",
                type=chakra_pb.COMP_NODE,
                ctrl_deps=[18],
            )
            unresolved_kernel = chakra_pb.Node(
                id=20,
                name="ncclDevKernel_AllGather",
                type=chakra_pb.COMM_COLL_NODE,
                ctrl_deps=[19],
            )
            _set_int64(unresolved_kernel, "comm_type", chakra_pb.ALL_GATHER)
            _set_int64(unresolved_kernel, "comm_size", 65536)
            _write(
                device_dir / "chakra.0.et",
                [
                    _metadata(groups),
                    step,
                    c10d,
                    kernel,
                    _record(13, "3", 10),
                    other_step,
                    other_c10d,
                    unresolved_kernel,
                ],
            )

            step2 = chakra_pb.Node(id=10, name="ProfilerStep#2", type=chakra_pb.COMP_NODE)
            c10d2 = chakra_pb.Node(
                id=11,
                name="c10d::broadcast_",
                type=chakra_pb.COMP_NODE,
                ctrl_deps=[10],
            )
            launcher = chakra_pb.Node(
                id=12,
                name="nccl:broadcast",
                type=chakra_pb.COMP_NODE,
                ctrl_deps=[11],
            )
            launcher.inputs.values = "[[1, 2, 0, 375, 4, 'cuda:0']]"
            cpu = launcher.attr.add()
            cpu.name = "is_cpu_op"
            cpu.bool_val = True
            _write(
                promoted_dir / "chakra.0.et",
                [_metadata(groups), step2, c10d2, launcher, _record(13, "3", 10)],
            )

            profile = profile_models(
                root,
                [
                    ModelSpec("Device", "device", 1),
                    ModelSpec("Promoted", "promoted", 1, True),
                ],
                jobs=2,
            )
            rows = profile["rows"]
            self.assertEqual(3, len(rows))
            self.assertEqual(
                {
                    (
                        "Device",
                        "ALL_REDUCE",
                        4,
                        "4-64KiB",
                        1,
                        4096,
                    ),
                    (
                        "Promoted",
                        "BROADCAST",
                        4,
                        "0-4KiB",
                        1,
                        1024,
                    ),
                    (
                        "Device",
                        "ALL_GATHER",
                        "UNRESOLVED",
                        "64KiB-1MiB",
                        1,
                        65536,
                    ),
                },
                {
                    (
                        row["model"],
                        row["comm_type"],
                        row["communicator_group_size"],
                        row["buffer_size_range"],
                        row["op_count"],
                        row["total_bytes"],
                    )
                    for row in rows
                },
            )
            self.assertEqual(0.5, profile["per_model_resolution"]["Device"]["hit_rate"])
            self.assertEqual(1, profile["per_model_resolution"]["Device"]["unresolved"])
            self.assertEqual(
                1,
                profile["per_model_resolution"]["Promoted"]["promoted_collectives"],
            )

            csv_path = root / "profile.csv"
            json_path = root / "profile.json"
            write_profile_outputs(profile, csv_path, json_path)
            with csv_path.open() as stream:
                self.assertEqual(3, len(list(csv.DictReader(stream))))
            self.assertEqual(rows, json.loads(json_path.read_text())["rows"])
            rendered = markdown_table(profile)
            self.assertIn("| Device | ALL_REDUCE | 4 |", rendered)
            self.assertIn("in-memory promote-comm-ops", rendered)


if __name__ == "__main__":
    unittest.main()
