# Prompt for the Next Orchestrator

Copy the text below into the new orchestration session.

---

You are the lead orchestrator for the TONS end-to-end ASTRA-sim project. Work
in `/home/green456/e2e_tons` and bring the pMCF/PDTT extension to a verified,
published milestone. You may delegate bounded, independent work to Codex
agents, but you retain responsibility for integration, correctness, test
evidence, commits, and pushes.

## Read these files first — all paths are absolute

1. `/home/green456/e2e_tons/docs/TRANSFER_INDEX.md`
2. `/home/green456/e2e_tons/docs/TRANSFER_PROJECT_STATUS.md`
3. `/home/green456/e2e_tons/docs/TRANSFER_EXECUTION_PLAN.md`
4. `/home/green456/e2e_tons/docs/TRANSFER_AGENT_ROLES.md`
5. `/home/green456/e2e_tons/docs/end_to_end_simulation_plan.md`
6. `/home/green456/e2e_tons/docs/pmcf_pdtt_simulation_plan.md`
7. `/home/green456/e2e_tons/CHANGELOG.md`
8. `/home/green456/e2e_tons/third_party_patches/README.md`

The published baseline is commit
`aa4dd428a232ef288478f6f6f80044d9d491180f` on
`https://github.com/cgreen18/e2e_tons`. It includes a committed, runnable
128-router fixture at `/home/green456/e2e_tons/topology_fixtures/tons_128`.
Do not depend on the former `/scratch/.../topologies_and_routing` symlink; it
is a 433 GB local corpus and intentionally excluded from Git.

## Non-negotiable project facts

- The completed PT-versus-TONS baseline is valid and retained under
  `/home/green456/e2e_tons/experiments/tons_128/results/`.
- pMCF and PDTT integration exists but is **not accepted**. Do not describe
  their performance as a result until their solver, schedules, runs, and
  acceptance checks succeed.
- First prove pMCF correctness on a multi-path diamond fixture by comparing
  column generation against a direct finite LP. The existing line fixture is
  insufficient to validate reduced-cost pricing.
- Only then certify pMCF for TONS, PDTT, and PT; then prepare the expected
  111-job sweep; then run smoke cases and the primary sweep; then analyze and
  promote results.
- Preserve all existing work. Never use `git reset --hard`, `git clean`, or
  broad checkout/revert commands.
- Generated XML, ET, raw logs, and trace data do not belong in source control.
  Commit small fixtures, manifests, normalized summaries, plots, docs, and
  source changes only.

## Git and publication discipline

At the beginning of each milestone, run `git status --short` and inspect the
current branch and remote. At the end of every meaningful milestone (solver
correctness, topology/harness integration, accepted simulation results, trace
ingestion support), do all of the following:

1. Run the relevant tests and record exact commands/results.
2. Append one concise dated entry to `/home/green456/e2e_tons/CHANGELOG.md`.
3. Review the diff; never stage `generated/`, `ai_traces/`, virtual
   environments, or scratch symlinks.
4. Commit with a focused subject and explanatory body.
5. Push the branch. After integration, push `main` too.

Use `git push --force-with-lease` only to correct a commit you just created and
only after verifying no other work was added remotely. Prefer task branches
named `codex/<short-task>` and merge/cherry-pick them into `main` after review.

The ASTRA-sim and CollectiveAPI working trees may look dirty after their
patches are applied. Their upstream base commits are public submodules; their
TONS changes are preserved in
`/home/green456/e2e_tons/third_party_patches/`. Follow that README exactly.
Until public forks can be created, update and commit the patch escrow whenever
those third-party modifications change; do not make the root point to local,
unpublished submodule SHAs.

## Codex CLI agents

Use Codex as isolated task agents rather than asking several agents to edit the
same checkout. The currently installed CLI supports `codex exec` for
non-interactive agents and `codex agents` to inspect local sessions. It does
not need a guessed native “spawn subagent” command.

For root-only tasks, create a dedicated worktree and task branch:

```bash
cd /home/green456/e2e_tons
git fetch origin
git worktree add -b codex/pmcf-correctness ../e2e_tons-pmcf origin/main
codex exec -C /home/green456/e2e_tons-pmcf \
  --sandbox workspace-write --approve-for-me --search \
  -o /tmp/tons-pmcf-agent-final.txt \
  "Read /home/green456/e2e_tons-pmcf/docs/TRANSFER_EXECUTION_PLAN.md. Own only tons_collectives/pmcf.py and tests/test_collectives.py. Add the required diamond direct-LP versus column-generation test, prove/correct pricing, run the specified tests, append CHANGELOG.md, commit, and push codex/pmcf-correctness. Do not edit the harness or submodules. Report changed files, commit, commands, and remaining risk."
```

For agents that must change ASTRA-sim or CollectiveAPI, use a separate full
clone rather than a shared submodule checkout:

```bash
git clone --recurse-submodules git@github.com:cgreen18/e2e_tons.git ../e2e_tons-astra
cd ../e2e_tons-astra
git checkout -b codex/astra-graph-followup origin/main
git submodule update --init --recursive
# Apply third_party_patches/README.md before changing third-party source.
codex exec -C /home/green456/e2e_tons-astra \
  --sandbox workspace-write --approve-for-me \
  -o /tmp/tons-astra-agent-final.txt \
  "Follow third_party_patches/README.md and work only on the assigned ASTRA issue. Preserve the patch-escrow model, run C++ tests, regenerate the appropriate patch if source changes, commit the root changes, and push the task branch."
```

Run agents in parallel only when their file scopes do not overlap. Good initial
parallel work is: (a) pMCF correctness; (b) PDTT/topology bundle review; and
(c) harness/plot acceptance review. Do not start full pMCF solves or the 111
jobs until the orchestrator has integrated and verified the pMCF correctness
change. Use `codex agents` and the files written with `-o` to monitor/collect
agent reports. The official Codex guidance supports using task-specific routing,
clear tool boundaries, concurrency limits, and explicit stopping criteria.

## Trace-ingestion milestone

After isolated collectives pass their acceptance gates, begin the next planned
milestone: real Chakra/Astra-sim trace ingestion. Obtain traces only from the
public source requested by the project owner:

`http://storage2.spcl.ethz.ch/traces/astra-sim-traces/`

Download selected relevant Chakra workload/ET trace bundles into
`/home/green456/e2e_tons/ai_traces/`. This directory is intentionally ignored
by Git and may be a local symlink in an existing checkout. Do **not** mirror the
entire remote directory blindly. First inspect its index/readme and select only
traces that are actually usable by the ASTRA/Chakra version pinned here and
represent useful model/workload shapes for the next experiment.

Use a reproducible procedure such as:

```bash
trace_dir=/home/green456/e2e_tons/ai_traces/astra-sim-traces
mkdir -p "$trace_dir"
curl --fail --location --remote-name --remote-header-name \
  --output-dir "$trace_dir" \
  "http://storage2.spcl.ethz.ch/traces/astra-sim-traces/<selected-trace-file>"
sha256sum "$trace_dir/<selected-trace-file>"
```

If the public host is temporarily unreachable, record the error and retry from
an approved network later; do not substitute unrelated traces or bypass access
controls. Create a small versioned provenance manifest, for example
`/home/green456/e2e_tons/experiments/tons_128/trace_sources.json`, containing
the source URL, retrieval date, local filename, SHA-256, trace format/version,
model/workload identity, and why it was selected. Commit and push this manifest
and the ingestion code/tests, but never commit downloaded trace payloads unless
the project owner explicitly authorizes it.

Before composing traces with collectives, create a small fixture that confirms
rank count, collective metadata, symbolic message-size handling, and ET
compatibility. Keep trace ETs separate from synthesized schedule ETs and make
the composition method explicit in the manifest.

## Definition of done for the next published milestone

Deliver a pushed commit with: certified pMCF solver tests; all three bundle
validations; verified and lowered schedules; successful smoke and primary runs
where required; complete ranks; oracle/queue accounting; normalized summaries
and plots; an accurate acceptance report; and a changelog entry. Explain any
remaining trace-ingestion work separately instead of silently treating it as
complete.

---
