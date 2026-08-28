# TONS Isolated-Collective Simulation: Project Status

## Goal

Deliver a reproducible ASTRA-sim analytical experiment for 128 ranks that
compares PT, PDTT, and TONS topologies. It must synthesize and lower isolated
all-gather (AG), all-reduce (AR), reduce-scatter (RS), and all-to-all (A2A)
schedules; run congestion-unaware and congestion-aware Graph backends; verify
timing/queue accounting; and produce retained summaries and plots.

The network profile is 1 GHz with 128 GB/s directed links, 25 ns source
injection, 75 ns electrical edges, and 50 ns optical edges. Primary sizes are
1 MiB, 16 MiB, and 256 MiB per rank, with eight subchunks. AI-trace ingestion
is explicitly deferred to a later milestone.

## What is complete and trustworthy

The original PT-versus-TONS milestone is implemented and was run successfully.

- Topology bundle validation, Graph YAML generation, route/VC checks, and the
  topology/routing reference document are present. See
  [topology artifacts](topology_and_routing_artifacts.md).
- `pt-dor-128` uses destination-based dimension-order routes. `tons-128` uses
  its CPL-safe destination-based MCLB routes and two-VC allocation.
- The implementation supports deterministic l3ss-tree AG/AR/RS schedules,
  direct/all-pairs A2A, and fixed-route pipelined A2A. XML is lowered to Chakra
  ETs and independently schedule-verified before simulation.
- The ASTRA Graph analytical prototypes support arbitrary adjacency, fixed
  source-major routes, heterogeneous edge properties, injection latency, and
  optional structured statistics. The backend source/build is under the
  dirty `simul/astra-sim` submodule.
- A manifest-driven `prepare`, `run`, and `analyze` harness exists in
  `tons_sim/pipeline.py`. The original manifest is
  `experiments/tons_128/manifest.json`.
- The prior complete baseline includes 62 jobs: two one-subchunk smoke jobs and
  60 primary jobs over PT/TONS, two backends, AG/AR/RS, direct A2A, and
  fixed-route-pipeline A2A. Its normalized summaries and ten plots are in
  `experiments/tons_128/results/`.
- The root Python suite most recently passed: `16 tests` with
  `venv_py12/bin/python -m unittest discover -s tests -v`. Rust workspace and
  ASTRA build/test checks passed before the ongoing pMCF changes.

The baseline findings are internally consistent with the model:

- AG is identical between PT and TONS. AR and RS differ by less than 0.05%; no
  TONS advantage was established for those schedules.
- Congestion-aware direct A2A gives TONS/PT throughput ratios of 1.7364 at
  16 MiB and 1.7546 at 256 MiB. Fixed-route pipeline gives 1.7353 and 1.7946.
  The relevant selected-route load reference is `128 / 72 = 1.7778`.
- Congestion-unaware direct A2A has intentionally unrealistic, very high
  throughput because independent all-pairs messages overlap without shared-link
  serialization. It is a route-ingestion/timing control, not a congestion
  prediction. Pipeline A2A bounds each directed link per epoch and is the
  meaningful congestion-unaware scheduled control.

## Work added for the current pMCF/PDTT extension

The extension plan is recorded in
[pMCF and PDTT plan](pmcf_pdtt_simulation_plan.md). The following code is
present but has not completed end-to-end acceptance.

- `pdtt-128` is defined in `tons_topology/validation.py`, and the topology
  validation and Graph-config CLIs accept it. The canonical PDTT artifacts
  resolve to `pdtt_2c_128r_6p_4x4x8` with CPL-safe candidate paths,
  destination-based MCLB selected paths, next-hop table, allowed turns, and
  VC matrix.
- PDTT validation was manually successful: 128 routers, 768 directed links,
  degree six, 16,256 routes, average selected-route hop count 3.47133, and
  maximum selected-route load 74. The Graph edge classifier reports 576
  electrical and 192 optical directed links.
- The manifest now lists PT, PDTT, and TONS and adds `pmcf` with `highs`,
  CPL-safe candidates, and seed 1. Its default topology root is the committed
  `topology_fixtures/tons_128` fixture, rather than the former 433 GB local
  scratch symlink.
- `tons_collectives/pmcf.py` implements a path-based maximum-concurrent-flow
  A2A solver. It reads canonical `.rallpaths`, solves fractional routing,
  quantizes it deterministically to eight subchunks, and compiles causal
  store-and-forward XML. Its compiler is shared with the fixed-route pipeline.
- A HiGHS column-generation implementation replaced the infeasible monolithic
  LP construction. The installed Gurobi uses a size-limited license and
  rejected the complete 128-router model. pMCF is exported from
  `tons_collectives`, available in `tools/collective_schedule.py pmcf-a2a`, and
  wired into manifest preparation and plots.
- A three-topology plot layout splits A2A panels by `direct`,
  `fixed-route-pipeline`, and `pmcf`, preventing direct/unaware artifacts from
  flattening the scheduled curves.

## Critical unfinished state

**Do not present pMCF or PDTT results yet.** No successful three-topology
prepare/run/analyze cycle has been completed, and existing retained result
files are the earlier PT/TONS baseline only.

The full PT pMCF solve was intentionally stopped when this handoff was
requested. It was using the HiGHS finite-candidate column-generation solver
over 462,749 CPL-safe PT candidate paths. It reached iteration 10 with about
21,936 active columns and four improving commodities; it had not converged or
written a certified report. Any partial generated files from that attempt must
be treated as invalid.

The most important technical gate is solver correctness. The current objective
remained `0.00471698113208` through the observed iterations. That may be a
valid degenerate optimum, but it could also indicate an error in dual-sign or
reduced-cost pricing. Before an expensive full run, compare column generation
against the direct LP on a small multi-path/diamond fixture and add that test.
The existing line fixture only validates a trivial single-path case and cannot
detect this problem.

Additional open items:

- complete/certify the PT, PDTT, and TONS pMCF solves and make their reports
  fully reproducible;
- run the expanded 111-job sweep (three smoke jobs plus 108 primary jobs);
- inspect whether the generalized acceptance logic in `tons_sim/pipeline.py`
  expresses the intended PDTT comparison policy before relying on it;
- rerun all relevant Python, Rust, converter, ASTRA, and end-to-end tests after
  pMCF integration; and
- regenerate and promote only validated normalized results and plots.

## Principal files

| Area | Files |
| --- | --- |
| Topology and PDTT bundles | `tons_topology/validation.py`, `tools/validate_topology_bundle.py`, `tools/generate_graph_config.py` |
| pMCF solver/compiler | `tons_collectives/pmcf.py`, `tons_collectives/a2a.py`, `tons_collectives/__init__.py`, `tools/collective_schedule.py` |
| Harness and analysis | `tons_sim/pipeline.py`, `experiments/tons_128/manifest.json` |
| pMCF test start point | `tests/test_collectives.py` |
| Existing results | `experiments/tons_128/results/` |
| Original/full scope | `docs/end_to_end_simulation_plan.md`, `docs/pmcf_pdtt_simulation_plan.md` |

## Working-tree guidance

`git status` shows many untracked project directories plus modified nested
repositories/submodules. This is expected for the shared research checkout.
Preserve unrelated edits, use narrow `apply_patch` changes, and never reset or
clean the checkout. Generated data belongs under ignored `generated/tons_128/`;
small fixtures, manifest edits, normalized summaries, and final plots are the
only experiment outputs intended to be retained in source control.
