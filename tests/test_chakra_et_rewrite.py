from collections import Counter
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
from tools.chakra_et_rewrite import (
    COMM_NAME_TO_ENUM_NAME,
    derive_comm_size,
    iter_nodes,
    parse_rank_range,
    parse_process_group_registry,
    promote_comm_op,
    resolve_collective_group_sizes,
    round_comm_size,
    rewrite_collection,
    rewrite_trace,
    write_delimited,
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
REAL_MOE13_RANK0 = Path(
    "/home/green456/e2e_tons/ai_traces/"
    "MoE8x13B_N32_GPU128_TP4_PP4_DP8_EP4_13B_BS128/chakra.0.et"
)
REAL_MOE70_RANK0 = Path(
    "/home/green456/e2e_tons/ai_traces/"
    "MoE8x70B_N64_GPU256_TP4_PP8_DP8_EP8_70B_BS128/chakra.0.et"
)


def _cpu_node(node_id: int, name: str, values: str):
    node = chakra_pb.Node(
        id=node_id,
        name=name,
        type=chakra_pb.COMP_NODE,
    )
    node.inputs.values = values
    cpu_attr = node.attr.add()
    cpu_attr.name = "is_cpu_op"
    cpu_attr.bool_val = True
    return node


def _metadata_node(node_id: int, groups) -> object:
    node = chakra_pb.Node(
        id=node_id,
        name="## process_group:init ##",
        type=chakra_pb.METADATA_NODE,
    )
    node.inputs.values = "[[" + json.dumps(groups) + "]]"
    return node


def _record_param_node(node_id: int, pg_id: str, ctrl_dep: int) -> object:
    node = chakra_pb.Node(
        id=node_id,
        name="record_param_comms",
        type=chakra_pb.COMP_NODE,
        ctrl_deps=[ctrl_dep],
    )
    node.inputs.values = f"['{pg_id}', 'DATA_PARALLEL_GROUP']"
    return node


def _write_trace(path: Path, nodes, version: str = "chakra-test-v1") -> None:
    with path.open("wb") as stream:
        write_delimited(stream, chakra_pb.GlobalMetadata(version=version))
        for node in nodes:
            write_delimited(stream, node)


def _read_nodes(path: Path):
    records = list(iter_nodes(path, chakra_pb))
    return records[0], records[1:]


def _int64_attr(node, name: str) -> int:
    matches = [attr for attr in node.attr if attr.name == name]
    if len(matches) != 1 or matches[0].WhichOneof("value") != "int64_val":
        raise AssertionError(f"node {node.id} does not have one int64 {name} attribute")
    return matches[0].int64_val


@unittest.skipUnless(chakra_pb is not None, PROTOBUF_SKIP_REASON)
class ChakraEtRewriteTest(unittest.TestCase):
    def test_repair_deps_drops_only_ids_absent_from_same_file(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "chakra.0.et"
            target = root / "output" / source.name
            first = chakra_pb.Node(
                id=1,
                name="first",
                type=chakra_pb.COMP_NODE,
                ctrl_deps=[0],
                data_deps=[2, 99],
            )
            second = chakra_pb.Node(
                id=2,
                name="second",
                type=chakra_pb.COMP_NODE,
                ctrl_deps=[1],
                data_deps=[1],
            )
            _write_trace(source, [first, second])

            stats = rewrite_trace(source, target, ["repair-deps"], rank=0)
            metadata, nodes = _read_nodes(target)

            self.assertEqual("chakra-test-v1", metadata.version)
            self.assertEqual(2, stats["nodes_read"])
            self.assertEqual(1, stats["ctrl_deps_dropped"])
            self.assertEqual(1, stats["data_deps_dropped"])
            self.assertEqual(2, stats["deps_dropped"])
            self.assertEqual([], list(nodes[0].ctrl_deps))
            self.assertEqual([2], list(nodes[0].data_deps))
            self.assertEqual([1], list(nodes[1].ctrl_deps))
            self.assertEqual([1], list(nodes[1].data_deps))

    def test_promote_comm_ops_rounds_to_group_unit_and_leaves_unknowns(self) -> None:
        values = (
            "[[1, 2, 0, 3, 4, 'cuda:0'], "
            "[[[5, 6, 0, 7, 8, 'cuda:0']]]]"
        )
        all_reduce = _cpu_node(1, "nccl:all_reduce", values)
        coalesced = _cpu_node(2, "nccl:coalesced", "[]")
        malformed = _cpu_node(3, "nccl:broadcast", "not a literal")

        enum_name, size, reason = promote_comm_op(all_reduce, chakra_pb, 4)
        self.assertEqual(("ALL_REDUCE", 1024, None), (enum_name, size, reason))
        self.assertEqual(68, derive_comm_size(values))
        self.assertEqual(chakra_pb.COMM_COLL_NODE, all_reduce.type)
        self.assertEqual(chakra_pb.ALL_REDUCE, _int64_attr(all_reduce, "comm_type"))
        self.assertEqual(1024, _int64_attr(all_reduce, "comm_size"))

        self.assertEqual(1024, round_comm_size(1, 4))
        self.assertEqual(1024, round_comm_size(1535, 4))
        self.assertEqual(2048, round_comm_size(1536, 4))
        self.assertEqual(2048, round_comm_size(2048, 4))

        self.assertEqual(
            (None, None, "unknown-name"), promote_comm_op(coalesced, chakra_pb, 4)
        )
        self.assertEqual(chakra_pb.COMP_NODE, coalesced.type)
        self.assertEqual(
            (None, None, "communication-size-unavailable"),
            promote_comm_op(malformed, chakra_pb, 4),
        )
        self.assertEqual(chakra_pb.COMP_NODE, malformed.type)

    def test_group_size_resolution_uses_metadata_and_shared_control_parent(self) -> None:
        groups = [
            {"pg_name": "3", "pg_desc": "dp", "backend_config": "", "ranks": [0, 2, 4, 6]},
            {"pg_name": "9", "pg_desc": "world", "backend_config": "", "ranks": []},
        ]
        self.assertEqual(
            {"3": 4, "9": 8},
            parse_process_group_registry("[[" + json.dumps(groups) + "]]", 8),
        )
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "chakra.0.et"
            target = root / "output" / source.name
            profiler_step = chakra_pb.Node(
                id=10, name="ProfilerStep#1", type=chakra_pb.COMP_NODE
            )
            c10d = chakra_pb.Node(
                id=11,
                name="c10d::allreduce_",
                type=chakra_pb.COMP_NODE,
                ctrl_deps=[10],
            )
            launcher = _cpu_node(
                12,
                "nccl:all_reduce",
                "[[1, 2, 0, 375, 4, 'cuda:0']]",
            )
            launcher.ctrl_deps.append(11)
            _write_trace(
                source,
                [
                    _metadata_node(1, groups),
                    profiler_step,
                    c10d,
                    launcher,
                    _record_param_node(13, "3", 10),
                ],
            )

            stats = rewrite_trace(
                source,
                target,
                ["promote-comm-ops"],
                rank=0,
                model_rank_count=8,
            )
            _, nodes = _read_nodes(target)
            promoted = next(node for node in nodes if node.id == 12)
            self.assertEqual(1024, _int64_attr(promoted, "comm_size"))
            self.assertEqual(1, stats["group_size_resolutions_direct"])
            self.assertEqual(0, stats["group_size_resolutions_fallback"])
            self.assertEqual(1.0, stats["group_size_resolution_hit_rate"])

    def test_group_size_resolution_falls_back_to_all_model_ranks_and_counts(self) -> None:
        groups = [
            {"pg_name": "3", "pg_desc": "dp", "backend_config": "", "ranks": [0, 1]},
        ]
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "chakra.0.et"
            target = root / "output" / source.name
            launcher = _cpu_node(
                2, "nccl:broadcast", "[[1, 2, 0, 1, 1, 'cuda:0']]"
            )
            launcher.ctrl_deps.append(1)
            _write_trace(source, [_metadata_node(1, groups), launcher])

            stats = rewrite_trace(
                source,
                target,
                ["promote-comm-ops"],
                rank=0,
                model_rank_count=8,
            )
            _, nodes = _read_nodes(target)
            promoted = next(node for node in nodes if node.id == 2)
            self.assertEqual(2048, _int64_attr(promoted, "comm_size"))
            self.assertEqual(0, stats["group_size_resolutions_direct"])
            self.assertEqual(1, stats["group_size_resolutions_fallback"])
            self.assertEqual(0.0, stats["group_size_resolution_hit_rate"])
            self.assertEqual(
                {"record-param-not-found": 1},
                stats["group_size_resolution_fallback_reasons"],
            )

    def test_collection_manifest_parallelism_and_output_are_deterministic(self) -> None:
        binding = generated_bindings / "et_def_pb2.py"
        original_mtime = binding.stat().st_mtime_ns
        self.assertEqual(generated_bindings, bootstrap_bindings())
        self.assertEqual(original_mtime, binding.stat().st_mtime_ns)

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_dir = root / "input"
            output_dir = root / "output"
            source_dir.mkdir()
            for rank in range(2):
                groups = [
                    {
                        "pg_name": "0",
                        "pg_desc": "world",
                        "backend_config": "",
                        "ranks": [],
                    }
                ]
                parent = chakra_pb.Node(
                    id=2, name="c10d::allgather_", type=chakra_pb.COMP_NODE
                )
                mapped = _cpu_node(
                    3,
                    "nccl:_all_gather_base",
                    "[[1, 2, 0, 16, 2, 'cuda:0']]",
                )
                mapped.ctrl_deps.extend([2, 999])
                _write_trace(
                    source_dir / f"chakra.{rank}.et",
                    [
                        _metadata_node(1, groups),
                        parent,
                        mapped,
                        _record_param_node(4, "0", 2),
                    ],
                )

            manifest_path, manifest = rewrite_collection(
                source_dir,
                output_dir,
                ["promote-comm-ops", "repair-deps"],
                parse_rank_range("0:2"),
                jobs=2,
            )
            first_outputs = {
                path.name: path.read_bytes()
                for path in sorted(output_dir.glob("chakra.*.et"))
            }
            first_manifest = manifest_path.read_bytes()

            second_path, second_manifest = rewrite_collection(
                source_dir,
                output_dir,
                ["repair-deps", "promote-comm-ops"],
                parse_rank_range("0-1"),
                jobs=2,
            )

            self.assertEqual(first_outputs, {
                path.name: path.read_bytes()
                for path in sorted(output_dir.glob("chakra.*.et"))
            })
            self.assertEqual(first_manifest, second_path.read_bytes())
            self.assertEqual(manifest, second_manifest)
            loaded = json.loads(second_path.read_text())
            self.assertEqual([0, 1], loaded["ranks"])
            self.assertEqual(
                ["repair-deps", "promote-comm-ops"], loaded["passes_applied"]
            )
            self.assertEqual(8, loaded["totals"]["nodes_read"])
            self.assertEqual(2, loaded["totals"]["deps_dropped"])
            self.assertEqual(2, loaded["totals"]["ops_promoted"])
            self.assertEqual(2, loaded["totals"]["group_size_resolutions_direct"])
            self.assertEqual(0, loaded["totals"]["group_size_resolutions_fallback"])
            self.assertEqual(1.0, loaded["totals"]["group_size_resolution_hit_rate"])
            self.assertEqual(
                {"ALL_GATHER": 1024},
                loaded["totals"]["bytes_attributed_per_collective_type"],
            )
            self.assertEqual(COMM_NAME_TO_ENUM_NAME, loaded["name_to_enum"])

    def test_moe13_rank0_host_derivation_matches_4342_device_collectives(self) -> None:
        self.assertTrue(
            REAL_MOE13_RANK0.is_file(), f"required real trace unavailable: {REAL_MOE13_RANK0}"
        )

        host_collectives: Counter[tuple[int, int]] = Counter()
        device_collectives: Counter[tuple[int, int]] = Counter()
        for record in iter_nodes(REAL_MOE13_RANK0, chakra_pb):
            if isinstance(record, chakra_pb.GlobalMetadata):
                continue
            enum_name = COMM_NAME_TO_ENUM_NAME.get(record.name)
            if enum_name is not None and record.type == chakra_pb.COMP_NODE:
                if any(
                    attr.name == "is_cpu_op"
                    and attr.WhichOneof("value") == "bool_val"
                    and attr.bool_val
                    for attr in record.attr
                ):
                    host_collectives[
                        (getattr(chakra_pb, enum_name), derive_comm_size(record.inputs.values))
                    ] += 1
            if record.type == chakra_pb.COMM_COLL_NODE:
                device_collectives[
                    (_int64_attr(record, "comm_type"), _int64_attr(record, "comm_size"))
                ] += 1

        agreement = sum((host_collectives & device_collectives).values())
        self.assertEqual(4448, sum(device_collectives.values()))
        self.assertEqual(4342, agreement)
        self.assertEqual(106, sum(device_collectives.values()) - agreement)

    def test_moe70_rank0_promotes_every_mapped_name_and_only_coalesced_is_left(self) -> None:
        self.assertTrue(
            REAL_MOE70_RANK0.is_file(), f"required real trace unavailable: {REAL_MOE70_RANK0}"
        )

        promoted: Counter[str] = Counter()
        bytes_by_type: Counter[str] = Counter()
        unmapped: Counter[str] = Counter()

        def promotion_target(node) -> bool:
            return (
                COMM_NAME_TO_ENUM_NAME.get(node.name) is not None
                and node.type == chakra_pb.COMP_NODE
                and any(
                    attr.name == "is_cpu_op"
                    and attr.WhichOneof("value") == "bool_val"
                    and attr.bool_val
                    for attr in node.attr
                )
            )

        resolutions = resolve_collective_group_sizes(
            REAL_MOE70_RANK0,
            256,
            promotion_target,
            chakra_pb,
            fallback_to_all_ranks=True,
        )
        for record in iter_nodes(REAL_MOE70_RANK0, chakra_pb):
            if isinstance(record, chakra_pb.GlobalMetadata):
                continue
            resolution = resolutions.get(record.id)
            enum_name, size, reason = promote_comm_op(
                record,
                chakra_pb,
                resolution.group_size if resolution is not None else 1,
            )
            if enum_name is not None:
                promoted[enum_name] += 1
                bytes_by_type[enum_name] += size
            elif reason is not None:
                unmapped[record.name] += 1

        self.assertEqual(8632, sum(promoted.values()))
        self.assertEqual({"nccl:coalesced": 1226}, dict(unmapped))
        self.assertEqual(8632, len(resolutions))
        self.assertTrue(all(size > 0 for size in bytes_by_type.values()))


if __name__ == "__main__":
    unittest.main()
