"""Synthetic trace emission and communicator pre-scan."""

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tons_collectives.synthetic_trace import (
    Collective,
    Communicator,
    Compute,
    generate_synthetic_trace,
)
from tools.chakra_comm_groups import scan, signature
from tools.chakra_et_rewrite import (
    PROCESS_GROUP_METADATA_NAME,
    iter_nodes,
    load_protobuf_module,
    parse_process_group_registry,
)


def _default_stages(size: int = 1024) -> list:
    return [
        Compute("compute_0", 100),
        Collective("allreduce_pg1", "allreduce", "1", size),
        Compute("compute_1", 100),
        Collective("alltoall_pg2", "alltoall", "2", size),
        Compute("compute_2", 100),
    ]


def _default_communicators(ranks: int, subgroup: int) -> list[Communicator]:
    return [
        Communicator("1", tuple(range(subgroup))),
        Communicator("2", tuple(range(ranks))),
    ]


class SyntheticTraceTest(unittest.TestCase):
    def test_traces_parse_and_carry_expected_nodes(self) -> None:
        pb = load_protobuf_module()
        with TemporaryDirectory() as directory:
            prefix = Path(directory) / "chakra"
            paths = generate_synthetic_trace(
                prefix, 8, _default_stages(), _default_communicators(8, 4)
            )
            self.assertEqual(len(paths), 8)

            # A member of the subgroup runs both collectives.
            nodes = [n for n in iter_nodes(paths[0], pb) if hasattr(n, "type")]
            self.assertEqual(
                [pb.NodeType.Name(n.type) for n in nodes],
                [
                    "METADATA_NODE",
                    "COMP_NODE",
                    "COMM_COLL_NODE",
                    "COMP_NODE",
                    "COMM_COLL_NODE",
                    "COMP_NODE",
                ],
            )
            # A non-member skips the all-reduce entirely.
            outside = [n for n in iter_nodes(paths[5], pb) if hasattr(n, "type")]
            self.assertEqual(
                [pb.NodeType.Name(n.type) for n in outside],
                [
                    "METADATA_NODE",
                    "COMP_NODE",
                    "COMP_NODE",
                    "COMM_COLL_NODE",
                    "COMP_NODE",
                ],
            )
            # The metadata node is deliberately detached: it completes
            # synchronously without registering a simulator event, and
            # Workload::issue_dep_free_nodes iterates a snapshot of the ready
            # set, so a metadata node that is the sole dependency root frees
            # its children but nothing re-scans for them.  It stays
            # dependency-free alongside the first stage, and the ready set is
            # ordered by node id so metadata is still issued first.
            self.assertEqual(list(nodes[0].data_deps), [])
            self.assertEqual(nodes[0].id, 0)
            self.assertEqual(list(nodes[1].data_deps), [])
            # Every later stage chains onto the previous stage.
            for index, node in enumerate(nodes[2:], start=2):
                self.assertEqual(list(node.data_deps), [index - 1])

    def test_process_group_registry_matches_astra_trim(self) -> None:
        pb = load_protobuf_module()
        with TemporaryDirectory() as directory:
            prefix = Path(directory) / "chakra"
            paths = generate_synthetic_trace(
                prefix, 8, _default_stages(), _default_communicators(8, 4)
            )
            metadata = next(
                node
                for node in iter_nodes(paths[0], pb)
                if getattr(node, "type", None) == pb.METADATA_NODE
            )
            self.assertEqual(metadata.name, PROCESS_GROUP_METADATA_NAME)
            # Parsed by the same helper that models ASTRA's two-character trim.
            self.assertEqual(
                parse_process_group_registry(metadata.inputs.values, 8),
                {"1": 4, "2": 8},
            )

    def test_comm_coll_nodes_carry_type_size_and_group(self) -> None:
        pb = load_protobuf_module()
        with TemporaryDirectory() as directory:
            prefix = Path(directory) / "chakra"
            paths = generate_synthetic_trace(
                prefix, 4, _default_stages(4096), _default_communicators(4, 2)
            )
            collectives = [
                node
                for node in iter_nodes(paths[0], pb)
                if getattr(node, "type", None) == pb.COMM_COLL_NODE
            ]
            attributes = [
                {a.name: (a.string_val if a.name == "pg_name" else a.int64_val)
                 for a in node.attr if a.name in {"comm_type", "comm_size", "pg_name"}}
                for node in collectives
            ]
            self.assertEqual(
                attributes,
                [
                    {"comm_type": pb.ALL_REDUCE, "comm_size": 4096, "pg_name": "1"},
                    {"comm_type": pb.ALL_TO_ALL, "comm_size": 4096, "pg_name": "2"},
                ],
            )

    def test_reserved_and_malformed_groups_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Communicator("0", (0, 1))
        with self.assertRaises(ValueError):
            Communicator("1", (1, 0))
        with self.assertRaises(ValueError):
            Communicator("1", ())
        with TemporaryDirectory() as directory:
            prefix = Path(directory) / "chakra"
            with self.assertRaises(ValueError):
                generate_synthetic_trace(
                    prefix, 4, _default_stages(), [Communicator("1", (0, 1))]
                )
            with self.assertRaises(ValueError):
                # A member outside the rank count.
                generate_synthetic_trace(
                    prefix, 2, _default_stages(), _default_communicators(4, 2)
                )


class CommGroupScanTest(unittest.TestCase):
    def test_scan_recovers_exact_membership_and_usage(self) -> None:
        with TemporaryDirectory() as directory:
            generate_synthetic_trace(
                Path(directory) / "chakra",
                8,
                _default_stages(2048),
                _default_communicators(8, 4),
            )
            plan = scan(Path(directory), 8)

        self.assertEqual(plan["group_sizes"], [4, 8])
        self.assertEqual(plan["unresolved_pg_names"], [])
        self.assertEqual(plan["membership_conflicts"], [])
        self.assertEqual(plan["warnings"], [])
        groups = {entry["pg_name"]: entry for entry in plan["groups"]}
        self.assertEqual(groups["1"]["members"], [0, 1, 2, 3])
        self.assertEqual(groups["2"]["members"], list(range(8)))
        # Every rank declares both groups, but only members issue the
        # subgroup collective.
        self.assertEqual(groups["1"]["declared_by_rank_count"], 8)
        self.assertEqual(groups["1"]["used_by_rank_count"], 4)
        self.assertEqual(groups["1"]["collectives"]["ALL_REDUCE"]["op_count"], 4)
        self.assertEqual(groups["2"]["collectives"]["ALL_TO_ALL"]["op_count"], 8)
        self.assertEqual(
            plan["required_schedules"],
            [
                {"signature": "n4-r0_3", "collective": "ALL_REDUCE"},
                {"signature": "n8-r0_7", "collective": "ALL_TO_ALL"},
            ],
        )

    def test_membership_conflict_is_reported(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            generate_synthetic_trace(
                root / "chakra", 4, _default_stages(), _default_communicators(4, 2)
            )
            # Rewrite rank 3 so it declares a different membership for group 1.
            disagreeing = Path(directory) / "other"
            generate_synthetic_trace(
                disagreeing / "chakra",
                4,
                _default_stages(),
                [Communicator("1", (0, 2)), Communicator("2", (0, 1, 2, 3))],
            )
            (root / "chakra.3.et").write_bytes(
                (disagreeing / "chakra.3.et").read_bytes()
            )
            plan = scan(root, 4)

        self.assertTrue(plan["membership_conflicts"])
        self.assertIn("process group 1", plan["membership_conflicts"][0])

    def test_signature_is_canonical_and_distinguishes_placements(self) -> None:
        self.assertEqual(signature([0, 1, 2, 3]), "n4-r0_3")
        self.assertEqual(signature([3, 1, 0, 2]), "n4-r0_3")
        self.assertEqual(signature(range(64, 128)), "n64-r64_127")
        # Same size, different placement must not collide.
        scattered = signature([0, 2, 4, 6])
        self.assertNotEqual(scattered, signature([0, 1, 2, 3]))
        self.assertTrue(scattered.startswith("n4-h"))
        self.assertEqual(scattered, signature([6, 4, 2, 0]))


if __name__ == "__main__":
    unittest.main()
