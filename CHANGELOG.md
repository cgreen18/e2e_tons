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
- **root:** Added `tools/slurm/pmcf_solve.sbatch` so the 128-router pMCF LPs run
  on compute nodes (`-A mithuna -p cpu`) instead of a login node, switched the
  manifest to the unrestricted Gurobi backend, pinned Gurobi to deterministic
  dual simplex so the quantizer sees a reproducible optimal basis, and added a
  `--report` flag to `tools/collective_schedule.py pmcf-a2a`. Note the token
  license is *not* picked up automatically on compute nodes: without
  `GRB_LICENSE_FILE` gurobipy falls back to its size-limited pip license, so
  the batch script sets it explicitly. Removed the invalid PT pMCF artifacts
  from the aborted run (objective 1/212 with exactly one positive path per
  commodity, i.e. the unimproved seed routing). Solves are in flight; no pMCF
  result is accepted yet.
- **PDTT topology review:** Verified the committed PDTT metrics, artifact
  alignment, CDG safety, and link roles; activated fixture-backed regressions
  and rejected CPL candidates without a consistent allowed VC assignment.
- **harness review:** Hardened harness acceptance so a reversed A2A trend is
  blocking rather than a recorded pass, required certified pre-solved pMCF
  artifacts instead of solving inside prepare, and added synthetic 111-job,
  queue-policy, and three-panel plot tests.
- **root:** Certified all three canonical pMCF solves with unrestricted Gurobi
  13.0.2 on compute nodes, all `exact-monolithic-lp`, all schedules verified at
  128 ranks with sends equal to receives: PT 7.804878049e-03 over 462,749
  candidates (149 s), PDTT 1.360978203e-02 over 171,960 (328 s), TONS
  1.399825022e-02 over 152,917 (259 s). pMCF A2A ratios versus PT are PDTT
  1.7438 and TONS 1.7935, slightly above the 128/74 = 1.7297 and 128/72 =
  1.7778 route-load references because pMCF also beats each bundle's own
  selected-route bound. Derived the all-to-all cut bounds that explain the
  numbers: PT is bisection-limited at exactly 1/128 (32 directed links cross
  between halves of 64), so pMCF reaches 99.90% of PT's physical ceiling and
  cannot improve on DOR, while PDTT and TONS have twice that cut (1/64). Also
  found that 3,544 of PT's 16,256 selected DOR routes are absent from its
  CPL-safe candidate set, which is why PT's optimum sits just below 1/128.
  Retained the evidence in `experiments/tons_128/results/pmcf_certification.json`.
- **root:** Ran the ASTRA analytical C++ fixtures: 6 congestion-unaware and 7
  congestion-aware tests pass, including the TONS Graph multi-hop timing,
  disjoint-transfer, shared-link FIFO/counter, heterogeneous-link, and
  non-physical-route cases. They must be run from a directory where their
  hardcoded `../../input/` fixture path resolves, and they need the spack GCC 12
  toolchain; the system GCC 8 fails to link `std::filesystem`. Note the Rust
  `l3ss_tree` workspace builds but contains no tests at all (16 targets, 0
  tests), so "Rust workspace tests passed" is not evidence of anything.
