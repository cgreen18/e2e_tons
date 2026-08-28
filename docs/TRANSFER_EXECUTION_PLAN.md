# Next-Session Execution Plan

This is the ordered implementation and verification plan for finishing the
pMCF and PDTT extension. The plan intentionally separates mathematical
correctness from expensive 128-router simulation.

## 0. Establish the checkout and baseline

1. Read [project status](TRANSFER_PROJECT_STATUS.md), the two scope plans, and
   `CHANGELOG.md`.
2. Inspect `git status --short`; do not clean, reset, or overwrite unrelated
   work.
3. Re-run the quick Python baseline:

   ```bash
   venv_py12/bin/python -m unittest discover -s tests -v
   ```

4. Verify all three topology bundles before generating workloads:

   ```bash
   venv_py12/bin/python tools/validate_topology_bundle.py --bundle pt-dor-128 --root topology_fixtures/tons_128
   venv_py12/bin/python tools/validate_topology_bundle.py --bundle pdtt-128 --root topology_fixtures/tons_128
   venv_py12/bin/python tools/validate_topology_bundle.py --bundle tons-128 --root topology_fixtures/tons_128
   ```

These validations cover map connectivity/degree, route ordering and physical
hops, destination next hops, route/VC alignment, and selected route-plus-VC
channel-dependency acyclicity. They do not model packet-level VC flow control.

## 1. Make pMCF correct before scaling it

Owner scope: `tons_collectives/pmcf.py`, `tests/test_collectives.py`, and
small fixture files only.

1. Read `_solve_highs` closely. Its intended formulation is a path-based
maximum concurrent-flow LP:

   - one nonnegative fraction per candidate path;
   - sum of path fractions equals the concurrent-flow value for each ordered
     source/destination commodity;
   - each directed link has total fractional usage at most one;
   - maximize the common concurrent-flow value.

2. Add a diamond/multi-path fixture with at least two legal paths for one or
   more commodities. Solve it two ways: the current column-generation method
   and a direct finite LP built only for the small fixture. Assert equal
   objective within tolerance and valid link capacities. This is mandatory;
   the current three-node line test cannot expose a pricing mistake.
3. Check reduced-cost signs against the equality-constraint dual convention
   returned by SciPy/HiGHS. Keep a final scan over every finite candidate path
   and fail rather than claim optimality if an improving column remains.
4. Improve convergence only after correctness is established. Safe options are
   adding every negative path for a commodity near convergence, increasing the
   deterministic per-commodity batch, caching parsed candidate paths, or using
   an unrestricted commercial solver. Keep ordering deterministic under seed 1.
5. Ensure the JSON report records solver name/version, candidate/active/positive
   path counts, objective, quantized maximum link load, epoch count, seed, and
   a certification status.
6. Rerun the existing line fixture plus the new diamond test and schedule
   verifier/lowering tests.

The current Gurobi installation rejected the full model because its license is
size-limited. Do not silently fall back to an approximate solution. HiGHS
column generation is intended to solve exactly over the finite `.rallpaths`
candidate set; an alternative solver is acceptable only if it provides the
same exactness/certification.

## 2. Solve and verify pMCF schedules

After the small-fixture proof passes, run the three canonical solves. Start
with the smallest candidate set (TONS), then PDTT, then PT. PT has 462,749
CPL-safe candidates and can take materially longer; run it in a persistent
allocated job if the local shell is not suitable.

Example direct invocation (use an output directory that is known to be
generated and ignored):

```bash
venv_py12/bin/python tools/collective_schedule.py pmcf-a2a \
  --topology topologies_and_routing/topo_maps/pt_2c_128r_6p_4x4x8.map \
  --candidates topologies_and_routing/allpath_lists/pt_2c_128r_6p_4x4x8_turns_allowed_cpl_safe_destbased.rallpaths \
  --subchunks 8 --solver highs --seed 1 \
  --output generated/tons_128/prepared/schedules/pt/alltoall-pmcf.xml
```

For every result, inspect the report, run `verify_schedule`, lower XML to
Chakra ETs, and confirm each rank can complete. The compiler must retain
per-chunk source/destination buffer offsets and logical chunk index/count.
Scheduled pMCF sends must be causal store-and-forward hops, not direct endpoint
sends disguised as pMCF.

## 3. Prepare the full expanded experiment

Once all pMCF reports are certified, clear only stale files under the explicit
`generated/tons_128/` target if necessary, then invoke the manifest harness.
The expected job count is 111:

- 3 one-subchunk congestion-unaware smoke cases (one per topology), and
- 108 primary cases = 3 topologies × 2 backends × 3 sizes × 6 schedule/collective
  combinations (AG, AR, RS, and A2A direct, fixed-route-pipeline, pMCF).

Use the existing CLI/help in `tons_sim/pipeline.py`; do not guess its options.
The preparation stage must validate bundles, generate Graph configs, XML,
schedule ETs, workload ETs, verifier reports, and job records. A prepare
failure is a blocking correctness failure, not an artifact to bypass.

## 4. Run and analyze

Run only after `prepare` succeeds. The harness must capture manifest inputs,
revisions, command, stdout/stderr, and structured simulator statistics, and it
must fail if any rank does not complete.

Analyze into normalized CSV/JSON and plots. Preserve raw output under
`generated/`; promote only accepted compact results under
`experiments/tons_128/results/`. Do not overwrite the existing PT/TONS baseline
until the new outputs have passed review; retain it as a separate comparison if
needed.

The analysis should include:

- throughput versus size for every collective/backend;
- topology/PT speedups, with PT/PDTT reference `128/74 = 1.7297` and PT/TONS
  reference `128/72 = 1.7778` on A2A comparisons;
- separate A2A panels for `direct`, `fixed-route-pipeline`, and `pmcf`;
- congestion-aware link utilization and queue-wait distributions; and
- timing-oracle/acceptance summaries.

## 5. Acceptance review

Require all of the following before making performance claims:

- all ranks complete; Graph route/map and XML/ET checks pass;
- congestion-unaware timing matches the transfer oracle;
- congestion-aware direct A2A has valid shared-link FIFO queue accounting;
- fixed-route-pipeline and pMCF, which impose one subchunk per directed link
  per epoch, have no artificial shared-link queues;
- AG/AR/RS comparisons stay within the plan’s 5% band. Label RS equality as a
  new experimental hypothesis rather than a reproduced paper result;
- A2A comparisons are reported honestly for both PDTT and TONS versus PT.
  A reversed trend, incomplete rank, schedule mismatch, failed oracle, or
  uncertified pMCF solution is blocking.

Do not require the pMCF measured speedup to equal a route-load ratio exactly:
the fractional LP, eight-subchunk quantization, path lengths, store-and-forward
epochs, and ASTRA timing model can create deviation. Report it and explain it.

## Suggested final test order

1. Python topology/collective tests, including the new pMCF diamond test.
2. Rust `l3ss_tree` workspace tests and MSCCL/Chakra converter tests.
3. ASTRA analytical C++ fixture tests and both binary builds.
4. Small 4–8 rank end-to-end fixtures through XML → schedule ET → workload
   invocation → both analytical binaries.
5. All three 128-router bundle validations.
6. `prepare`, one smoke per topology, then the primary sweep and `analyze`.

Append a brief dated result to `CHANGELOG.md` after every completed phase.
