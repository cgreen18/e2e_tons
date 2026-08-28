# TONS Simulation Project Transfer

This set is the starting point for the next orchestration session. It records
the state of the 128-router isolated-collective work at the handoff on
2026-08-28. Read the documents in this order:

1. [Project status](TRANSFER_PROJECT_STATUS.md) — what is complete, what is
   deliberately unfinished, and which existing results are trustworthy.
2. [Execution plan](TRANSFER_EXECUTION_PLAN.md) — the ordered path to finish
   pMCF, PDTT, simulations, analysis, and acceptance.
3. [Agent roles](TRANSFER_AGENT_ROLES.md) — safe parallelization boundaries
   for an orchestrator that shares one checkout among agents.

The authoritative scope documents remain
[the original plan](end_to_end_simulation_plan.md) and
[the pMCF/PDTT extension](pmcf_pdtt_simulation_plan.md). This transfer set
does not replace them; it makes their current implementation state explicit.

## Safety and provenance

The checkout was already broadly dirty and includes nested repositories and
submodules. Treat all pre-existing changes as user-owned. Do not use `git
reset`, `git clean`, or checkout/revert operations to make it appear clean.
Generated schedules, ET files, logs, and partial pMCF files live under
`generated/tons_128/` and are disposable only when their exact targets have
been inspected. The accepted two-topology baseline is retained separately in
`experiments/tons_128/results/`.

Every agent that changes files must append one dated, brief entry to the root
[change log](../CHANGELOG.md).
