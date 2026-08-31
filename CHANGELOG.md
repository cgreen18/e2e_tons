# Change Log

Brief implementation notes are appended here by each collaborating agent.

## 2026-08-27

- **root:** Saved the agreed isolated-collective implementation plan and
  initialized the shared change log.
- **root:** Added canonical PT-DOR/TONS bundle validation, deterministic Graph
  adapters, topology artifact documentation, and 128-router checks.
- **root:** Completed AG/AR/RS and direct/pipelined A2A schedule export,
  symbolic Chakra lowering, schedule/workload verification, and small ET tests.
- **root:** Finished arbitrary Graph timing, structured simulator statistics,
  manifest prepare/run/analyze tooling, plots, C++ fixtures, and end-to-end
  analytical verification.
- **root:** Fixed manifest preparation to create the l3ss XML parent directory
  before schedule export; completed 128-rank preparation and both smoke runs.
- **root:** Completed the 60-run primary sweep; all acceptance checks passed
  and normalized summaries plus ten plots were generated and retained as
  versionable experiment results.
- **root:** Added the pMCF/PDTT experiment plan before implementation.

## 2026-08-28

- **root:** Added the project-transfer status, execution, and agent-role
  documents; recorded that pMCF/PDTT results remain unfinished and untrusted.
- **root:** Added a committed canonical 128-router topology fixture and
  third-party patch escrow so the root status can be reconstructed online.

## 2026-08-29

- **root:** Added a ready-to-paste orchestrator prompt with handoff paths,
  milestone publication discipline, Codex CLI agent setup, and trace-provenance
  guidance.

## 2026-08-31

- **root:** Fixed the pMCF column-generation reduced-cost sign. SciPy/HiGHS
  returns marginals as d(objective)/d(rhs), so capacity duals are non-positive
  and commodity duals non-negative; pricing on the unnegated sum selected the
  *longest* inactive paths and declared convergence while improving columns
  remained. Added the mandatory `diamond6` K(2,4) multi-path fixture and a
  direct finite-LP cross-check: the old code certified 0.2 where the true
  optimum is 4/11 = 0.363636. Added an independent primal feasibility/concurrency
  certification and recorded solver version, active-path count, iterations,
  seed, and certification status in the pMCF report. `venv_py12/bin/python -m
  unittest discover -s tests -v` passes 19 tests (was 16).
