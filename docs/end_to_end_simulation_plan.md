# TONS 128-Router Isolated-Collective Simulation Plan

## Summary

Implement a reproducible ASTRA-sim pipeline for the 128-router PT and TONS
topologies, beginning with isolated all-gather, all-reduce, reduce-scatter,
and all-to-all workloads. Run congestion-unaware simulations for functional
verification, then congestion-aware simulations for topology/routing
comparisons.

Use:

- PT topology corrected to `pt_2c_128r_6p_4x4x8`, with destination-based
  dimension-order routing.
- TONS `asc_lp_sym_2c_128r_6p_4x4x8_4x4x4`, with CPL-safe destination-based
  MCLB routing and its two-VC allocation.
- A 1 GHz paper-derived network profile: 128 GB/s per directed link, 25 ns
  source injection, 75 ns electrical edges, and 50 ns optical edges.
- An identical direct all-pairs schedule as the routing control, followed by a
  fixed-route-aware pipelined all-to-all schedule as the topology-aware
  collective result.
- No model/public AI trace in this milestone; retain interfaces needed to add
  them later.

Create a root `CHANGELOG.md`; every agent appends one brief dated entry
describing its changes and verification.

## Implementation Changes

### Topology artifacts and documentation

- Add `docs/topology_and_routing_artifacts.md` explaining topology meanings,
  artifact generation, file formats, canonical bundles, validation, and the
  limits of analytical VC/deadlock modeling.
- Extend the topology bundle API with explicit `pt-dor-128` and `tons-128`
  definitions. Correct the stale PT average-hop test.
- Validate maps, routes, next hops, route/VC alignment, and selected-route
  channel dependency acyclicity.
- Generate deterministic Graph edge-property CSVs by classifying the 576
  fixed intra-cube directed links as electrical and the remaining 192 as
  optical.
- Update the root README with the actual directory structure, dependency
  environments, generated artifacts, and commands.

### Collective synthesis and Chakra lowering

- Complete and test the current `l3ss_tree` work for deterministic
  round-robin all-gather, all-reduce, and reduce-scatter MSCCL schedules.
- Add identical direct and fixed-route-pipelined all-to-all schedules, using
  eight subchunks for primary runs.
- Extend MSCCL-to-Chakra lowering for relayed buffer offsets and unambiguous
  logical workload-chunk metadata.
- Port and correct ASTRA symbolic collective-size handling.
- Add schedule verification and isolated Chakra workload generation.

### Analytical backends and structured results

- Preserve and finish the existing Graph backend prototype for adjacency,
  source-major routes, per-edge properties, and injection latency.
- Preserve the intended congestion-unaware and congestion-aware timing
  equations.
- Add optional structured per-run, per-rank, and per-directed-link statistics
  without changing default CLI behavior.

### Experiment harness and plots

- Add a manifest-driven `prepare`, `run`, and `analyze` CLI.
- Sweep 128 ranks, eight subchunks, 1 MiB/16 MiB/256 MiB, both topologies,
  both analytical backends, AG/AR/RS, and both A2A modes after a one-subchunk
  congestion-unaware smoke test.
- Produce normalized summaries, throughput/speedup plots, and congestion-aware
  utilization/queue plots. Show the `128/72 = 1.7778` route-load reference for
  all-to-all.
- Ignore full generated XML/ET/log artifacts while retaining manifests, small
  fixtures, summaries, and plots.

## Interfaces

- Graph YAML uses a `graph` block for adjacency, routes, edge properties, and
  injection latency alongside the existing one-element topology arrays.
- Schedule ET nodes carry symbolic logical chunk index/count attributes, with
  old hard-coded `comm_size` schedules supported as a fallback.
- Experiment manifests identify topology bundle, route policy, network
  profile, collective, schedule, ranks, bytes, subchunks, backend, output, and
  seed.
- Structured simulator results use bytes and nanoseconds.

## Test and Acceptance Plan

- Run Python topology/VC tests, Rust workspace tests, schedule/converter tests,
  analytical C++ tests, and the root suite.
- Use small line/diamond fixtures for Graph validation, analytical timing,
  shared-link FIFO behavior, and counters.
- Run small XML-to-ET-to-ASTRA end-to-end fixtures before full bundles.
- Require both 128-router bundles to validate before performance runs.
- At 16 MiB and 256 MiB, require TONS A2A throughput to exceed PT and AG/AR/RS
  throughput to remain within 5%. Treat reduce-scatter as a new hypothesis.
- Treat reversed A2A trends, incomplete ranks, schedule mismatches, and failed
  analytical oracles as blocking failures.

## Assumptions

- `28r` was a typo; this is a 128-versus-128 comparison.
- PT DOR and TONS CPL-safe MCLB intentionally pair each topology with its
  selected routing policy.
- At 1 GHz, cycles equal nanoseconds; router latency is folded into every edge
  and injection is charged once per message.
- Packetization, finite buffers, credits, VC arbitration, and packet-level
  deadlock/fault simulation are deferred.
- Real AI traces are deferred until the isolated-collective pipeline passes.
