# pMCF and PDTT 128-Router Simulation Plan

## Scope

Extend the isolated-collective experiment with the 128-router PDTT topology and
a third all-to-all schedule mode, `pmcf`. Retain the existing 1 GHz network
profile, 128 ranks, eight subchunks, 1/16/256 MiB per rank, and both ASTRA
analytical backends.

## Routing and schedule policies

- PT direct traffic uses destination-based DOR; TONS and PDTT direct traffic
  use their CPL-safe destination-based MCLB routes and VC artifacts.
- Fixed-route-pipeline traffic follows each bundle's selected `.paths` file.
- pMCF solves a path-based maximum concurrent-flow LP over each canonical
  bundle's CPL-safe candidate `.rallpaths`. Its fractional solution is
  deterministically quantized into eight subchunks and compiled into causal,
  unit-link-capacity, store-and-forward epochs. Thus pMCF embeds routes into
  adjacent-router sends and does not depend on the selected route table for
  end-to-end path choice.
- Use the unrestricted HiGHS LP backend by default; retain Gurobi as an
  optional backend where an unrestricted license is available.
- AG/AR/RS continue to use deterministic map-informed `l3ss_tree` schedules.

## Implementation

- Add and validate a canonical `pdtt-128` topology bundle, Graph configuration,
  routing/VC alignment, and topology documentation.
- Extend the pMCF XML compiler with stable logical-flow metadata and expose it
  through the manifest preparation stage.
- Verify and lower pMCF XML into Chakra ETs with the same completeness and DAG
  checks as the existing schedules.
- Generalize the experiment analysis from a PT/TONS pair to PT/PDTT/TONS,
  including pMCF curves and topology-relative speedups.
- Split all-to-all plots by schedule mode so congestion-unaware direct traffic
  cannot visually flatten the scheduled modes.

## Execution and acceptance

- Run topology, schedule, lowering, Python, Rust, and ASTRA regression tests.
- Prepare and execute all three topologies, both backends, all collective sizes,
  and direct/fixed-route-pipeline/pMCF all-to-all modes.
- Require every rank to complete. Require TONS and PDTT results to be reported
  against PT for all three A2A modes; pMCF scheduled runs must have no shared-link
  queues, and direct congestion-aware runs must show queue accounting.
- Preserve raw XML/ET/log/statistics under ignored `generated/`; retain accepted
  normalized summaries and final plots under `experiments/tons_128/results/`.
