# PT / PDTT / TONS 128-Router Isolated-Collective Acceptance Report

Date: 2026-08-31. Commit: see `git log` for the commit carrying this file.

## Verdict

**Accepted.** All 111 jobs completed, every rank completed, and the harness
acceptance stage reported `passed: true` with zero failures across 48 checks.

This supersedes the earlier two-topology PT/TONS baseline, which is retained
unmodified under [`baseline_pt_tons/`](baseline_pt_tons) for comparison.

## What was run

- 111 jobs: 3 one-subchunk congestion-unaware smoke cases (one per topology)
  plus 108 primary cases = 3 topologies x 2 backends x 3 sizes x 6
  schedule/collective combinations.
- Topologies `pt-dor-128`, `pdtt-128`, `tons-128`; backends congestion-unaware
  and congestion-aware; 1 MiB, 16 MiB, 256 MiB per rank; 128 ranks; 8
  subchunks; seed 1.
- Stages ran on compute nodes (`-A mithuna -p cpu`): prepare 10 min 9 s,
  run 18 min 35 s, analyze 5 s. All exited `0:0`.

## Solver certification

All three pMCF instances were solved exactly by unrestricted Gurobi 13.0.2
under deterministic dual simplex, each certified `exact-monolithic-lp` and
independently re-checked for unit directed-link capacity and equal
per-commodity flow. Details in [`pmcf_certification.json`](pmcf_certification.json).

| topology | pMCF optimum | candidates | positive paths | epochs | solve |
| --- | --- | --- | --- | --- | --- |
| pt | 7.804878049e-03 | 462,749 | 16,499 | 1,044 | 149 s |
| pdtt | 1.360978203e-02 | 171,960 | 16,896 | 645 | 328 s |
| tons | 1.399825022e-02 | 152,917 | 16,896 | 627 | 259 s |

## Headline results: congestion-aware all-to-all at 256 MiB per rank

| schedule mode | PT GB/s | PDTT GB/s | TONS GB/s | PDTT/PT | TONS/PT |
| --- | --- | --- | --- | --- | --- |
| direct | 127.61 | 217.54 | 223.90 | 1.7047 | 1.7546 |
| fixed-route-pipeline | 107.51 | 189.96 | 192.95 | 1.7669 | 1.7946 |
| pmcf | 119.75 | 193.75 | 199.66 | 1.6180 | 1.6674 |

Route-load references: PT/PDTT `128/74 = 1.7297`, PT/TONS `128/72 = 1.7778`.

AG/AR/RS stay within 0.04% of PT for both comparison topologies at both large
sizes, far inside the plan's 5% band. All-gather is exactly equal.

## The pMCF result requires explanation, not celebration

pMCF has the best LP objective of the three schedule modes, yet its measured
topology-versus-PT *ratio* is the lowest. That is not a defect, and the reason
is measurable rather than speculative.

**pMCF improves absolute throughput on every topology.** Against
fixed-route-pipeline at 256 MiB: PT 107.51 -> 119.75 GB/s (+11.4%), PDTT
189.96 -> 193.75 (+2.0%), TONS 192.95 -> 199.66 (+3.5%). It is a strictly
better schedule everywhere. It compresses the *ratio* only because PT, whose
fixed DOR routing was the most load-imbalanced, has the most to gain from
optimal multipath routing.

**The measured speedup is governed by store-and-forward epoch depth, not by
the LP objective.** Both scheduled modes place one subchunk per directed link
per epoch, so the makespan tracks the schedule's critical-path depth:

| mode | epochs PT | epochs PDTT | epochs TONS | PT/TONS epochs | TONS/PT measured | PT/PDTT epochs | PDTT/PT measured |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed-route-pipeline | 1,170 | 659 | 649 | 1.8028 | 1.7946 | 1.7754 | 1.7669 |
| pmcf | 1,044 | 645 | 627 | 1.6651 | 1.6674 | 1.6186 | 1.6180 |

The epoch ratio predicts the simulated ratio to within 0.5%. The plan
anticipated exactly this: the fractional LP, eight-subchunk quantization, path
lengths, and store-and-forward epochs can all move the measured number away
from a route-load ratio.

## Why PT cannot be beaten by more than it is

PT all-to-all is bisection-limited. Maximizing over every contiguous torus
slab, PT's tightest all-to-all cut bound is exactly `1/128`: 32 directed links
cross between two halves of 64 routers, and 4,096 crossing commodities give
`t <= 32/4096`. PT's selected DOR routing already attains that ceiling, so
pMCF reaches 99.90% of PT's physical limit and *cannot* improve on DOR in the
LP sense. PDTT and TONS each have twice that cut bound (`1/64`), and that
doubled bisection -- not routing cleverness -- is the origin of their
all-to-all advantage.

The remaining 0.1% PT shortfall is accounted for: 3,544 of PT's 16,256
selected DOR routes are absent from its CPL-safe candidate set, so the pMCF
feasible region genuinely excludes part of the DOR solution.

## Acceptance checks

- All 111 runs complete; every run reports 128 of 128 ranks. `run` aborts on
  any incomplete rank and did not.
- All 18 schedules pass independent verification at 128 ranks with sends equal
  to receives and zero errors.
- Congestion-aware direct A2A has real shared-link FIFO queue accounting:
  417,174 (PT), 446,811 (PDTT), 433,657 (TONS) queued chunks at 256 MiB.
- Congestion-aware fixed-route-pipeline and pMCF have exactly zero queued
  chunks, zero total queue wait, and zero maximum queue wait, as required for
  modes that bound each directed link to one subchunk per epoch.
- A2A comparisons are blocking on a reversed trend; none reversed.
- 48 recorded checks, 0 failed.

## Caveats and what is explicitly not claimed

- Congestion-unaware **direct** A2A remains an unrealistic control, not a
  congestion prediction: independent all-pairs messages overlap without
  shared-link serialization, which is why its ratios (1.2073 at 16 MiB, 1.0495
  at 256 MiB) are near unity and should not be read as a topology result. The
  scheduled modes are the meaningful congestion-unaware controls.
- Reduce-scatter equality with PT is a **new experimental hypothesis** from
  this model, not a reproduced paper result.
- Packetization, finite buffers, credits, VC arbitration, and packet-level
  deadlock are not modelled. Bundle VC validation is a prerequisite check, not
  a claim that the analytical backend simulates VC flow control.
- No AI-trace workload is included. Trace ingestion remains outstanding; see
  the change log.
- The Rust `l3ss_tree` workspace builds but contains no tests, so it carries no
  automated evidence.
