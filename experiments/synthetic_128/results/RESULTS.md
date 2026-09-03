# Communicator-Aware Custom Collectives: Synthetic 128-Rank Validation

**Date:** 2026-09-03. **Status:** functional validation only. These numbers are
*not* a performance claim; see "What these numbers do and do not show".

## Workload

One synthetic Chakra trace, 128 ranks, emitted by
`tools/generate_synthetic_trace.py`:

    compute -> all-reduce over process group 1 (ranks 0-63)
            -> compute -> all-to-all over process group 2 (all 128 ranks)
            -> compute

Each collective carries 16 MiB; each compute stage replays 100 us. Ranks 64-127
have no all-reduce node, exactly as a real trace would record.

## Matrix

3 topologies (PT, PDTT, TONS) x 2 backends x 3 schedule modes = 18 runs.
`ring` is ASTRA's native implementation and is the control. `direct` and `pmcf`
load pre-computed topology-aware schedules selected by exact communicator
membership.

## Result

All 18 runs completed with 128/128 ranks complete.

| run | AR ranks | AR mean (us) | A2A ranks | A2A mean (us) | end (us) |
| --- | --- | --- | --- | --- | --- |
| pt.ring.Aware | 64 | 393.7 | 128 | 12132.1 | 12637.1 |
| pt.direct.Aware | 64 | 72.4 | 128 | 166.2 | 503.8 |
| pt.pmcf.Aware | 64 | 72.4 | 128 | 271.6 | 610.2 |
| pdtt.ring.Aware | 64 | 526.9 | 128 | 25147.6 | 25740.5 |
| pdtt.direct.Aware | 64 | 123.2 | 128 | 126.7 | 492.5 |
| pdtt.pmcf.Aware | 64 | 116.6 | 128 | 199.1 | 566.8 |
| tons.ring.Aware | 64 | 733.1 | 128 | 25273.2 | 25989.1 |
| tons.direct.Aware | 64 | 113.7 | 128 | 127.4 | 488.5 |
| tons.pmcf.Aware | 64 | 111.7 | 128 | 198.5 | 561.4 |

The full 18-row table including both backends is in `summary.csv`.

## What these numbers establish

1. **Communicator restriction is honoured.** Every run reports the all-reduce
   on exactly 64 ranks and the all-to-all on exactly 128. Ranks outside process
   group 1 never execute it.
2. **Byte parameterization works.** Both collectives report 16 MiB per rank,
   taken from the workload node rather than baked into the schedule.
3. **The right schedule is selected per communicator.** The 64-rank all-reduce
   time differs across topologies (PT 72.4 us, PDTT 116.6 us, TONS 111.7 us),
   so a topology-specific reduced sub-map really is driving synthesis.
4. **pMCF has no shared-link queueing.** `pmcf` congestion-aware and
   congestion-unaware end times are identical to the nanosecond on all three
   topologies (PT 610249 ns, PDTT 566753 ns, TONS 561442 ns). This is the
   acceptance property required by `docs/TRANSFER_EXECUTION_PLAN.md` section 5:
   a schedule that admits one subchunk per directed link per epoch cannot
   queue. `direct` correctly does differ between backends.

## What these numbers do and do not show

They do **not** support a topology comparison or a speedup claim.

- The congestion-unaware `direct` all-to-all is a known unrealistic control,
  not a congestion prediction (`docs/TRANSFER_PROJECT_STATUS.md`). Its 41.8 us
  figure must not be read as performance.
- One message size (16 MiB), one trace shape, and trivial compute. The accepted
  isolated sweep in `experiments/tons_128/results/` remains the only source for
  topology comparisons.
- `ring` here is a *correctness* control showing that a different algorithm is
  actually being substituted. The gap against it is not a validated speedup:
  native ring over a 128-rank all-to-all is a deliberately poor baseline.
- `direct` beating `pmcf` in this configuration is plausible (store-and-forward
  epoch depth dominates at a 128 KiB per-pair message) but is unexplained and
  was not investigated.

## Reproduce

```bash
venv_py12/bin/python tools/generate_synthetic_trace.py \
  --output-prefix generated/synthetic_128/traces/chakra --ranks 128 --subgroup-ranks 64
venv_py12/bin/python tools/chakra_comm_groups.py \
  --source generated/synthetic_128/traces --model-ranks 128 --jobs 8 \
  --output generated/synthetic_128/comm_groups.json
venv_py12/bin/python tools/build_communicator_schedules.py \
  --plan generated/synthetic_128/comm_groups.json \
  --topology-map topology_fixtures/tons_128/topo_maps/pt_2c_128r_6p_4x4x8.map \
  --output-root generated/synthetic_128/pt/direct --subchunks 8 --alltoall-mode direct
venv_py12/bin/python tools/analyze_synthetic_runs.py
```
