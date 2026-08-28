# Orchestration and Agent Roles

The next session may use agents, but they share one working directory. The
orchestrator owns integration, generated-result promotion, and all edits to
shared coordination files. Assign agents narrow, non-overlapping file scopes.

## Orchestrator

Responsibilities:

- read the transfer set, preserve the dirty checkout, and maintain the execution
  order in [the next-session plan](TRANSFER_EXECUTION_PLAN.md);
- assign no more than one active editor per file group;
- require evidence (commands, test result, changed files) in every handoff;
- inspect and integrate each change before starting expensive runs;
- decide whether any result may replace/promote the existing PT/TONS baseline;
- append its own brief dated `CHANGELOG.md` entry at meaningful milestones.

The orchestrator must not launch the full 111-job sweep until the pMCF small
fixture proves that the solver is exact and the three pMCF reports are
certified.

## Recommended bounded assignments

| Role | Safe file scope | Deliverable and gate |
| --- | --- | --- |
| pMCF correctness engineer | `tons_collectives/pmcf.py`, `tests/test_collectives.py`, pMCF-only fixtures | A multi-path direct-LP vs column-generation regression test; certified finite-candidate pricing and deterministic results. This is first priority. |
| Topology/PDTT reviewer | `tons_topology/`, `tools/validate_topology_bundle.py`, `tools/generate_graph_config.py`, topology docs | Evidence that PDTT artifact selection, route/VC checks, graph classification, naming, and route-load metrics are correct. Avoid edits to pMCF/harness files. |
| Schedule/lowering reviewer | `tons_collectives/a2a.py`, converter/verification tests, small ET fixtures | Confirm pMCF quantization produces legal causal hop-by-hop traffic with offsets, logical chunks, matching endpoints, and complete rank DAGs. Coordinate before editing `a2a.py`, which the solver also imports. |
| Harness/analysis engineer | `tons_sim/pipeline.py`, manifest, plotting/analysis tests | Review job count, acceptance semantics, backward compatibility, three-topology plot labeling, and staged execution. Do not begin the full run before solver certification. |
| Simulation runner | no source edits while running; explicit `generated/tons_128/` outputs only | Capture prepare/run/analyze commands and structured logs, monitor rank completion, and report anomalies. A long pMCF solve should also be run by a single owner. |

If only three agents are available, combine topology review with schedule/lowering
review only after agreeing they will not edit overlapping files. Keep pMCF and
harness separate.

## Coordination protocol

1. The orchestrator gives each agent a precise outcome, allowed file list,
   prohibited shared files, and required validation command.
2. Agents inspect first and use narrow patches. They never reset, clean, or
   revert another agent’s changes.
3. Agents append one short dated line to `CHANGELOG.md` only after their changes
   and verification are complete. If several agents finish together, serialize
   their append operations.
4. On handoff, each agent reports: changed paths, exact test/command, result,
   remaining uncertainty, and whether generated output is reusable.
5. The orchestrator resolves test failures and reviews diffs before the next
   phase. Do not merge by overwriting shared files.

## Sequencing constraints

- Topology validation and harness/plot review may proceed in parallel with the
  pMCF small-fixture work.
- pMCF compiler/lowering changes require coordination because `pmcf.py` uses
  helpers from `a2a.py`.
- The full pMCF solves begin only after the solver test is merged.
- Manifest `prepare` begins only after all three pMCF reports exist and are
  certified.
- Primary simulation, result promotion, and final plots are serialized behind
  successful prepare and smoke runs.

## Definition of a useful agent report

An agent report should state the conclusion first, then concrete evidence. For
example: “The diamond test proves the column-generation objective equals the
direct LP within `1e-9`; modified `pmcf.py` and `test_collectives.py`; command
`…` passed; no generated schedules were promoted.” This gives the orchestrator
enough information to safely continue without reconstructing the entire task.
