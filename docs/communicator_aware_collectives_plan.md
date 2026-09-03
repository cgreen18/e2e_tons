# Communicator-Aware Topology-Aware Collectives in End-to-End Traces

**Status:** Design accepted 2026-09-03; implementation not started.
**Supersedes nothing.** Extends
[`end_to_end_simulation_plan.md`](end_to_end_simulation_plan.md) and
[`pmcf_pdtt_simulation_plan.md`](pmcf_pdtt_simulation_plan.md) to the
end-to-end trace milestone.

## Goal

When ASTRA-sim replays a real or synthetic Chakra trace and reaches a
collective node, it must execute the pre-computed topology-aware collective
(pMCF or direct) for that node's *communicator*, parameterized by that node's
byte count -- instead of the built-in ring algorithm. Traces are never
rewritten into sends and receives, so workload-level kernel dependencies are
preserved exactly.

## What already works

`CustomAlgorithm` is the intended vehicle and most of it exists.

- It loads one Chakra ET per position in the communicator
  (`<prefix>.<pos_in_comm>.et`) rather than per global rank.
- `convert_algo_rank_to_real_rank` maps algo rank `i` to
  `comm_group->involved_NPUs[i]`, so a `k`-rank algorithm already runs on an
  arbitrary `k`-member subgroup.
- `get_msg_size` already rescales every send from the invoking workload node's
  `data_size` using the symbolic `msg_chunk_idx`, `msg_chunk_cnt`, and
  `workload_chunk_cnt` attributes emitted by `tons_collectives.chakra`.
- Collective-node dependencies remain in the workload DAG; the schedule DAG is
  a separate feeder. Kernel-level dependencies are therefore unaffected.

The isolated 111-job sweep already exercises this path at full communicator
size.

## Gaps

### G1. No per-communicator selection

`CollectiveImplLookup` supports exactly two custom keys: per Chakra node id
(`per-node-custom-implementation`) and per `ComType`
(`<coll>-implementation-custom`). Neither can express "this ET for this
communicator". The trace harness currently forbids custom schedules outright
(`tons_sim/trace_pipeline.py`) and runs native ring.

### G2. `CollectivePlan` cache ignores the lookup key

`CommunicatorGroup::get_collective_plan` caches `comm_plans[comm_type]` on
first use and returns it for every later call, discarding `workload_node_id`.
Any node- or communicator-keyed selection silently pins to whichever collective
of that type ran first. This is a blocking correctness bug for this feature.

### G3. Algo-rank binding is topology-blind

`involved_NPUs` is sorted ascending, so algo rank `i` binds to the `i`-th
lowest member id. A topology-aware algorithm is valid only on the placement it
was solved for. Reusing one size-`k` algorithm across differently placed
`k`-member groups is correct as *communication*, but is only topology-aware
where the placements are congruent.

### G4. Missing collective types

MoE traces contain `BROADCAST` and `REDUCE` nodes. ASTRA has no
custom-collective key for either, so they always fall back to native ring.

### G5. Restricted-demand A2A cannot relay through non-members

pMCF compiles causal store-and-forward hops. A `k`-rank collective ET can only
name algo ranks `0..k-1`, and ASTRA instantiates a collective algorithm only on
ranks that execute the collective node, so a non-member router can never be
woken up to forward a chunk. Running "the full-size algorithm with a restricted
demand matrix" therefore requires either restricting relays to members, or
activating non-participant ranks inside a collective.

**Decision:** default to member-only relays (option A below). Record the
alternative honestly rather than claiming full-fabric relaying.

- **A (chosen).** Restrict pMCF candidate paths to those whose intermediate
  nodes are all group members. Self-contained, uses existing machinery,
  topology-aware over the member-induced subgraph. Loses relay capacity that
  non-members could have supplied.
- **B (documented, not built).** Emit a 128-rank ET with identity rank mapping
  and wake non-member ranks as relays. Requires new `Sys`/`Workload` support
  for non-participant activation and a per-algorithm rank-mapping mode.
- **C (rejected).** Direct endpoint sends only, routed by the backend. This is
  already the `direct` A2A mode and gives pMCF nothing to contribute.

## Design decisions

| Decision | Choice |
| --- | --- |
| Selection key | Exact member set (canonical signature of the sorted rank list), not group size |
| A2A scope | Synthetic 128-node trace; plus `BROADCAST`/`REDUCE` custom keys. 256-rank models out of scope |
| Memoization | Deferred. Correctness first; revisit only if simulation time is the binding constraint |
| Subgroup relays | Member-only (G5 option A) |

Exact-member keying is required because restricted-demand A2A depends on
*which* ranks participate, not how many. It costs nothing extra for AG/AR/RS,
where several groups may register the same ET prefix.

## Subgroup AG/AR/RS: how to build the `k`-node map

`l3ss_tree` consumes an unweighted `k x k` binary adjacency matrix. The
member-induced subgraph of an arbitrary `k`-member group may be disconnected,
so three constructions are implemented behind
`--subgroup-graph {induced,proximity,canonical-k}`:

- `induced` -- physical induced subgraph. Most faithful; fails when disconnected.
- `proximity` (**default**) -- members `u,v` adjacent iff physical hop distance
  `d(u,v) <= D`, for the smallest `D` making the graph connected. Degrades to
  `induced` when `D = 1` suffices. Non-adjacent member sends are routed by the
  backend over the bundle's selected routes, exactly as `direct` A2A already is.
- `canonical-k` -- the `k`-node instance of the same topology family, i.e. the
  literal reading of "run the algorithm for the smaller size and replicate it".

The default is a judgment call, not a measured result. Compare the three on the
synthetic trace before committing to one.

## Work breakdown

### M1. Communicator pre-scan (`tools/chakra_comm_groups.py`)

Read every rank's trace and emit the set of communicators to pre-compute for.

- Extract PyTorch process-group metadata directly (the `pg_name` -> `ranks`
  JSON that `Workload::issue_pytorch_pg_metadata` parses), *not* the
  `record_param_comms` correlation heuristic used by
  `tools/chakra_collective_profile.py`. That heuristic resolves only 2.7% of
  Llama7B collectives.
- Tally, per `(comm_type, pg_name)`, the operation count and byte statistics.
- Gate on cross-rank agreement: every rank must report the same membership for
  every group it participates in. Disagreement is a blocking error.
- Output `comm_group_plan.json`: distinct groups (canonical member tuple), the
  collective types each is used for, and byte ranges.

Known target sizes from the committed profile: MoE8x13B `{2,4,8,16,128}`,
Llama7B `{128}`. Neither 128-rank model contains any all-to-all.

### M2. Subgroup collective synthesis (`tons_collectives/subgroup.py`)

For each `(topology, collective, member set)` emit MSCCL XML, verify it, and
lower it to a Chakra ET prefix.

- A2A `direct`: all-pairs among members only.
- A2A `pmcf`: commodities restricted to ordered member pairs; link capacities
  over the full topology; candidate paths filtered to member-only intermediates
  per G5-A.
- AG/AR/RS: `l3ss_tree` over the `k`-node map from the construction above.
- Reuse the existing `verify_schedule` and `lower_msccl_to_chakra` gates. Every
  generated schedule must verify before it can be registered.

### M3. ASTRA-sim: communicator-keyed selection

- Fix G2: key the `CollectivePlan` cache on the full lookup key, not `ComType`.
- Add a `per-communicator-custom-implementation` system-config key: a YAML
  mapping from canonical member signature to ET prefix, per collective type.
- Add `broadcast-implementation-custom` and `reduce-implementation-custom` (G4).
- Regenerate `third_party_patches/astra-sim-tons.patch` per
  `third_party_patches/README.md`.

### M4. Synthetic acceptance trace

128-rank Chakra trace: compute -> 64-member all-reduce -> compute ->
128-member all-to-all -> compute. Emitted with explicit process-group metadata
so M1 resolves both communicators cleanly.

This is the initial metric of success and is independent of the outstanding
real-trace deadlock recorded in `CHANGELOG.md` on 2026-09-03.

### M5. Harness and sweep

Extend `tons_sim/trace_pipeline.py` to accept custom per-communicator schedules
(currently hard-rejected), then run the synthetic trace across all three
128-node topologies, both backends, and pMCF/direct.

## Sequencing

M1 and M4 are independent and come first. M2 depends on M1's group list. M3 is
independent of M1/M2/M4 and touches only the ASTRA submodule, so it can proceed
in parallel in a separate clone per `docs/TRANSFER_AGENT_ROLES.md`. M5 is
serialized behind M2, M3, and M4.

## Out of scope

- Memoizing collective simulation time (deferred by decision).
- 256-rank topology bundles and the MoE8x70B all-to-all groups.
- The real-trace replay deadlock; the synthetic trace does not depend on it.
