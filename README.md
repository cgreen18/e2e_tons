# TONS end-to-end simulation

This repository builds isolated collective schedules and simulates them on the
128-router Parallel Torus and TONS topologies with ASTRA-sim's analytical
backends. The current milestone covers all-gather, all-reduce,
reduce-scatter, and all-to-all; model/public AI trace composition is deferred.

The experiment design is in
[`docs/end_to_end_simulation_plan.md`](docs/end_to_end_simulation_plan.md), and
the topology bundle formats are documented in
[`docs/topology_and_routing_artifacts.md`](docs/topology_and_routing_artifacts.md).
For a new lead agent, use the ready-to-paste
[`docs/ORCHESTRATOR_START_PROMPT.md`](docs/ORCHESTRATOR_START_PROMPT.md).

## Directory structure

- `adorn_code/`: topology, routing, MCLB, and VC generation/analysis tools.
- `topology_fixtures/tons_128/`: committed canonical PT, PDTT, and TONS
  128-router maps, routes, candidate paths, next-hop tables, and VC artifacts.
- `topologies_and_routing/`: optional local symlink to the complete external
  topology corpus (433 GB); it is not required for the committed 128-router
  experiment and is intentionally not versioned.
- `tons_topology/`: dependency-free bundle validation and Graph config export.
- `coll_synth/l3ss_tree/`: Rust tree schedule synthesis for AG/AR/RS.
- `tons_collectives/`: direct/pipelined A2A generation, verification, Chakra ET
  writing, and workload generation.
- `simul/collectiveapi/`: upstream MSCCL-to-Chakra converter, extended for
  logical chunks and relay offsets.
- `simul/astra-sim/`: ASTRA-sim plus Graph analytical backend changes.
- `tons_sim/` and `tools/run_experiment.py`: manifest-driven experiment stages.
- `experiments/tons_128/`: committed primary experiment manifest.
- `experiments/tons_128/results/`: accepted normalized summary and final plots
  for the committed primary sweep.
- `tests/`: small topology, schedule, and adapter fixtures/tests.
- `generated/`: ignored XML, ET, logs, and raw run results.
- `docs/tons.pdf`: TONS paper.

## Dependencies

Use Python 3.12 from `venv_py12/`; the system Python is 3.6 and cannot parse
this code. The topology validator and ET writer use only the standard library.
Plotting uses the installed Matplotlib stack. Topology synthesis tools may also
need NetworkX, OR-Tools, or a licensed Gurobi environment.

Rust/Cargo builds `l3ss_tree`. ASTRA-sim requires CMake, a C++17 compiler, and
`protoc`; its CMake build fetches yaml-cpp, and analytical unit tests also fetch
GoogleTest. Chakra's Python packaging pins a different protobuf major than the
root topology environment, so keep converter/Chakra dependencies in a separate
environment if using the upstream converter. The local ET writer avoids that
conflict for this pipeline.

On the configured cluster, solver jobs requiring Gurobi should use SLURM
account `mithuna` and partition `cpu` (`-A mithuna -p cpu`).

## Reproducible commands

Validate the three committed 128-router bundles:

```bash
venv_py12/bin/python tools/validate_topology_bundle.py --bundle pt-dor-128 --root topology_fixtures/tons_128
venv_py12/bin/python tools/validate_topology_bundle.py --bundle pdtt-128 --root topology_fixtures/tons_128
venv_py12/bin/python tools/validate_topology_bundle.py --bundle tons-128 --root topology_fixtures/tons_128
```

Build deterministic round-robin collective synthesis and ASTRA:

```bash
module load protobuf/3.18.0
cargo build --release --manifest-path coll_synth/l3ss_tree/Cargo.toml
simul/astra-sim/build/astra_analytical/build.sh
```

Prepare, run, and analyze the complete sweep:

```bash
module load protobuf/3.18.0
venv_py12/bin/python tools/run_experiment.py prepare experiments/tons_128/manifest.json
venv_py12/bin/python tools/run_experiment.py run generated/tons_128/prepared/prepared.json
venv_py12/bin/python tools/run_experiment.py analyze generated/tons_128/prepared/prepared.json
```

`prepare` validates bundles and schedules and generates Graph YAML/edge CSV,
MSCCL XML, schedule ETs, one-node workload ETs, system configs, and a fully
expanded job manifest. `run` executes the one-subchunk congestion-unaware smoke
jobs first, records commands/revisions/stdout/stderr/JSON, and stops on any
incomplete rank. `analyze` writes normalized CSV/JSON, blocking acceptance
results, and final plots.

To inspect commands without running simulators:

```bash
venv_py12/bin/python tools/run_experiment.py run generated/tons_128/prepared/prepared.json --dry-run
```

Generated XML, ET, logs, and raw result trees are excluded from source control.
The accepted normalized summary and selected final plots are retained in
`experiments/tons_128/results/`; rerunnable XML, ET, logs, and raw result trees
remain under ignored `generated/`.

## Third-party source provenance

The root repository pins public submodules. Modifications to ASTRA-sim,
its analytical backend, and CollectiveAPI are additionally stored as explicit
patches in `third_party_patches/` so the root commit remains reconstructible
online while forks are pending. Apply the patches after recursive submodule
initialization as described in `third_party_patches/README.md`.
