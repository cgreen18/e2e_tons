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
- **root:** Completed and accepted the full three-topology sweep. Fixed the
  batch-stage script: `module` is a login-shell function, so `#!/bin/bash` plus
  `module load protobuf/3.18.0 || true` silently no-opped and the first ASTRA
  job died with exit 127 on a missing `libprotobuf.so.3.18.0.0`; the script now
  uses a login shell, does not swallow the module failure, and refuses to start
  if either analytical binary has an unresolved shared library. prepare expanded
  to exactly 111 jobs (10 min 9 s), run completed all 111 with every rank
  complete (18 min 35 s), analyze passed with zero failures across 48 checks.
  Congestion-aware A2A at 256 MiB gives TONS/PT 1.7546 direct, 1.7946
  fixed-route-pipeline, 1.6674 pMCF, and PDTT/PT 1.7047, 1.7669, 1.6180. pMCF
  raises absolute throughput on every topology (+11.4% PT, +2.0% PDTT, +3.5%
  TONS versus fixed-route-pipeline) but compresses the ratio because PT gains
  most; the measured ratios track store-and-forward epoch depth to within 0.5%,
  not the LP objective. AG/AR/RS stay within 0.04% of PT. Promoted the
  normalized summaries, ten plots, and an acceptance report to
  `experiments/tons_128/results/`, retaining the earlier PT/TONS baseline
  unmodified under `results/baseline_pt_tons/`.
- **root:** Trace ingestion remains OUTSTANDING and is not part of this
  milestone. The public host `http://storage2.spcl.ethz.ch/traces/astra-sim-traces/`
  was confirmed reachable (HTTP 200) but nothing was downloaded, no
  `trace_sources.json` provenance manifest exists yet, and no trace fixture or
  composition code has been written.

## 2026-09-02

- **ASTRA-sim statistics:** Added deterministic per-rank collective count,
  byte, and busy-time breakdowns to structured JSON, including `UNKNOWN`
  fallback accounting; proved the schema with a FeederV3 C++ regression and a
  4-rank analytical AllReduce run while retaining all 6 congestion-unaware and
  all 7 congestion-aware backend test passes. Added explicit `METADATA_NODE`
  classification while excluding metadata from operator timing and wall-time
  accounting, enabling real PyTorch-derived Chakra traces to pass statistics
  collection without changing existing reported values.
- **trace experiment harness:** Added the manifest-driven 12-job real-Chakra
  trace prepare/run/analyze path with strict 128-rank validation, native-ring
  topology controls, normalized collective/communication/end-to-end speedups,
  congestion plots, legacy-statistics handling, and ten focused regressions;
  the full 39-test Python suite passes without skips.
- **Chakra ET rewriter:** Added a streaming, deterministic trace rewriter with
  composable dangling-dependency repair and CPU NCCL launcher promotion,
  parallel rank processing, JSON provenance, synthetic coverage, and real
  MoE8x13B/MoE8x70B rank-0 regressions for communication type/size recovery;
  added on-demand binding generation from the authoritative submodule schema
  so all protobuf tests execute under the root Python test command.
- **root:** Repaired both 128-rank trace sets with the committed rewriter and
  verified the tool reproduces them byte-identically (sha256 match on rank 0),
  rebuilt both analytical binaries with the new per-collective statistics
  writer, and confirmed the main checkout's ASTRA-sim working tree regenerates
  the committed escrow patch exactly. Added `--only <run_id>` to the trace
  runner plus `tools/slurm/trace_run.sbatch`, so the 12-job matrix runs as a
  SLURM array instead of serially: each job replays 37-89 million Chakra nodes
  across 128 ranks and takes over an hour, which would have made a sequential
  sweep take most of a day. Added `tools/slurm/astra_build.sbatch` and
  `tools/slurm/chakra_rewrite.sbatch` so builds and whole-model trace rewrites
  stay off the login node.
