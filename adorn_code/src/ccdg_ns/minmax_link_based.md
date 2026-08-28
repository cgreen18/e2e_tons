# Agent Guide: `src/ccdg_ns/minmax_link_based.py`

Single-file Gurobi MILP for **joint topology synthesis, CDG-based deadlock-free routing, and min-max link load** (`L_max`). Replaces the MCF `lambda` objective (see `ccdg_model.py`) with minimizing maximum normalized edge load.

**~2,400 lines, self-contained** (stdlib + Gurobi only). Related but separate siblings in `src/ccdg_ns/`:

| File | Role |
|------|------|
| `ccdg_model.py` | MCF formulation (maximize `lambda`) |
| `ccdg_optimizer.py` | MCF solve/harden helpers |
| `ccdg_io.py` | Shared I/O utilities |
| `complete_cdg_generation.py` | CLI wrapper for MCF path |
| `minmax_link_based_good.py` | Older snapshot; prefer `minmax_link_based.py` |

---

## Problem

Given `n` nodes, radix `r`, `n_vcs` virtual channels, capacity matrix `C`, demand matrix `D`, and commodities `S = {(s,d)}`:

1. Choose a directed physical topology (out-degree ≤ radix per node).
2. Choose CDG turn edges (VC-aware) forming an acyclic dependency graph.
3. Route integer flow for each commodity satisfying exact demand.
4. **Minimize** `L_max` = max over directed links `(i,j)` of total load on `(i,j)`.

Optional **phase 2** (`--hierarchical-objectives`): fix optimal `L_max`, then minimize sum of all directed link loads.

---

## Architecture & Control Flow

```
main()
  → parse_problem_arguments()  → problem_args, file_args, solver_params
  → run_optimization()
       ├─ _run_single_source_optimization()     # one-shot, all commodities
       ├─ run_per_source_optimization()        # sequential per source
       └─ run_per_flow_optimization()          # sequential per (s,d)
  → print_results() / write_results()
```

Each solve iteration:

```
CCDGModel(...) → build_model_()
CCDGOptimizer(...) → solve() → extract_resultant_values() → harden_results()
  [sequential modes] accumulate flow counts, merge hardened topo/turns/paths
```

### Core classes

**`CCDGModel`** (~lines 101–545) — builds the Gurobi MILP.

**`CCDGOptimizer`** (~lines 549–1040) — configures Gurobi, solves, extracts values, hardens topology/paths.

---

## CDG Encoding

Physical directed link `(i → j)` on VC `l` → **CDG node** `u` with label `(i, j, l)`.

Physical turn `i → k → j` with VCs `l0, l1` → **CDG edge** `(u, v)` where  
`u = (i,k,l0)`, `v = (k,j,l1)`.

Key maps (built in `_create_cdg_u_to_topo_ijv_maps_`):

| Map | Meaning |
|-----|---------|
| `cdg_u_to_topo_ijv_map[u]` | `(i, j, l)` |
| `topo_ijv_to_cdg_u_map[(i,j,l)]` | `u` |
| `ss_to_u_conn_map[s]` | CDG nodes at super-source `s` (first hop from `s`) |
| `sd_to_u_conn_map[d]` | CDG nodes at sink for destination `d` |
| `u_to_v_turns`, `v_to_u_turns` | CDG adjacency |

Turn enumeration skips U-turns and (by default) VC changes between hops unless `--allow-vc-trans`.

---

## Decision Variables

| Variable | Name pattern | Type | Meaning |
|----------|--------------|------|---------|
| `m_{i,j}` | `vars_topo_adj_mat` | binary (or [0,1]) | Physical edge `i→j` exists |
| `c_{u,v}` | `vars_turn_adj_mat` | binary (or [0,1]) | CDG turn edge exists |
| `fuv_{s,u,v}` | `vars_uv_flow[s]` | integer (default) | Flow on CDG turn for source `s` |
| `fss_{s,u}` | `vars_ss_flow[s]` | integer | Injection at super-source |
| `s_{s,u}` | `vars_sink[s]` | integer | Absorption at destination CDG nodes |
| `o_u` | `vars_o` | continuous | MTZ ordering |
| `L_max` | `var_lmax` | continuous | Objective (max link load) |

Hardened edges from prior sequential iterations are **pinned** to 1 in `_vars_topo_adj_mat_` / `_vars_turn_adj_mat_`.

---

## Key Constraints

| Constraint | Method | Summary |
|------------|--------|---------|
| Radix | `_constr_radix_` | `Σ_j m_{i,j} ≤ radix` |
| Symmetric links | `_constr_symmetric_links_` | `m_{i,j} = m_{j,i}` if `--symmetric-links` |
| CDG↔topology | `_constr_cdg_to_topo_mapping_` | Turn requires both incident physical edges |
| Turn capacity | `_constr_flow_to_cdg_mapping_` | `known_scaled + new ≤ (known_scaled + M) · c_{u,v}` |
| Min-max load | `_constr_physical_link_capacity_` | See **Known load** below |
| Flow conservation | `_constr_flow_conservation_` | Standard CDG balance per source |
| Production | `_constr_super_source_production_` | `Σ fss = Σ_d demand[s,d]` |
| Consumption | `_constr_ccdg_node_consumption_` | `Σ sink = demand[s,d]` per `(s,d)` |
| Acyclicity | `_constr_mtz_acyclicality_` | MTZ on CDG turn graph |
| Lower bound | `_constr_heuristic_lower_bound_` | `L_max ≥ max(hop_bound, injection_bound, known_lmax)` |

Physical link load on `(i,j)` aggregates sink + outgoing turn flow across all VCs and sources.

`_constr_global_valid_inequalities_` exists but is **commented out** in `build_model_`.

---

## Known Load (Sequential Modes Only)

Between iterations, `known_load_links` and `known_load_turns` store **flow counts** (not absolute load).

- Accumulated via `accumulate_link_flow_counts` / `accumulate_turn_flow_counts`: adds `W_k / demand[s,d]` per path edge/turn.
- In the next model, scaled load = `count × λ` where `λ = iteration_flow_scale = max(demand[s,d])` over commodities in **this** subproblem.

Constraints use:

```text
total = known_count·λ + new_flow
total ≤ L_max
total ≤ (known_count·λ + M) · m_ij        # M = total_network_demand of subproblem
```

**Critical:** Big-M must include `known_scaled`; otherwise prior load blocks new flow on the same edge when `M=1`.

Final absolute loads are recomputed from merged paths via `absolute_link_load_from_paths`.

---

## Solve Modes

### 1. One-shot (default)

All commodities in one MILP. `run_optimization` → `_run_single_source_optimization`.

### 2. `--per-source-solves`

Loop: one source `s` with all its destinations per iteration. State carried forward:

- `hardened_topo_edges`, `hardened_turn_edges`
- `known_load_links`, `known_load_turns` (flow counts)
- `known_lmax` (passed into heuristic lower bound; tracks max solver `L_max` so far)
- `merged_paths`, routing/VC tables

After each solve: harden paths/topo/turns, accumulate counts.

**`--safe`**: iteration 1 only — `expand_spanning_tree_turns_at_root` adds reverse turns + root-pivot turns for spanning-tree CDG completeness.

### 3. `--per-flow-solves`

Same as per-source but one `(s,d)` per iteration. Mutually exclusive with per-source.

Requires **binary** topo/turn vars and **integral** flow (enforced in `main()`).

### Priority ordering (`--prioritized-per-source hops|throughput`)

- Iteration 1: lowest source id (per-source) or lowest `(s,d)` (per-flow).
- Later iterations:
  - **hops**: highest avg BFS hop distance to destinations (sources) or shortest-hop metric (flows).
  - **throughput**: lowest proxy score — see `NEXT_SOURCE_THROUGHPUT_PROXY` (default `"combined"`).
- Per-flow also tiers by connectivity: both-unconnected → one-unconnected → both-connected (incident edge only, not reachability).

---

## Solve Phases (`CCDGOptimizer.solve`)

1. **Phase 1**: minimize `L_max`.
2. **Phase 2** (if `--hierarchical-objectives`): add `L_max ≤ L_max* + ε`, minimize `total_link_load_expr()`.

On infeasibility: writes `iis.ilp` and raises `RuntimeError`.

Gurobi params (only if set): `time_limit`, `threads`, `mip_gap` (fraction), `obj_gap_pct` (percent → `MIPGap/100`). `mip_gap` and `obj_gap_pct` are mutually exclusive.

---

## Path Extraction & Hardening

Post-solve flow may **split** at CDG nodes. Do not assume tree-like paths.

**`_decompose_paths_for_source`**: iterative min-flow stripping from sinks back to super-source; produces one or more paths per `(s,d)` with weights `W_k`.

**`harden_paths`**: validates `Σ W_k = demand[s,d]`, builds:
- `paths_by_sd[(s,d)]` — list of `{phys_nodes, vcs, cdg_nodes, W_k, allocation, ...}`
- `route_table[r][s][d][path_idx]` — next hop
- `vc_table[r][s][d][path_idx]` — VC on hop
- `current_load` — absolute per-edge load this iteration

**`harden_topology`**: takes solver `m_{i,j}` values, respects radix cap when building adjacency matrix.

**`extract_hardened_topology_from_paths`** / **`extract_hardened_turns_from_paths`**: edges/turns actually used by paths (used for pinning in next iteration).

---

## Throughput Proxies

Module constant `NEXT_SOURCE_THROUGHPUT_PROXY` (line ~1426): `"link"` | `"turn"` | `"moore"` | `"cdg_penalty"` | `"combined"`.

Used only for **ordering** remaining sources/flows; does not affect optimality of individual solves.

---

## CLI Reference

```bash
python src/ccdg_ns/minmax_link_based.py \
  --n_nodes N --radix R --n_vcs V \
  [--capacity C | --capacity-matrix FILE] \
  [--demand D | --demand-matrix FILE] \
  [--commodities-file FILE] \
  [--out BASENAME] [--out-json FILE] [--write-model] \
  [--per-source-solves | --per-flow-solves] \
  [--prioritized-per-source hops|throughput] \
  [--hierarchical-objectives] [--safe] \
  [--symmetric-links] [--allow-vc-trans] \
  [--continuous-flow] [--continuous-topo-edges] [--continuous-turn-edges] \
  [--debug-viz-dir DIR] \
  [--time_limit SEC] [--mip_gap FRAC | --obj-gap-pct PCT] [--threads N] [--silent]
```

Default commodities: all-to-all except self. Default out name: `ccdg_minmax_{n}n_{radix}r_{n_vcs}vcs`.

---

## Output Files

Written under `topologies_and_routing/`:

| Path | Content |
|------|---------|
| `topo_maps/{base}.map` | `n×n` adjacency (space-separated rows) |
| `topo_maps/{base}.graphml` | (if implemented in write path — map is primary) |
| `routepath_lists/{base}.paths` | Header + JSON phys node lists per path |
| `routepath_lists/{base}.paths.jsonl` | Full path records |
| `nr_lists/{base}.nrl2` | `(s, d, cur, next)` tuples |
| `vc_mats/{base}.vcmat2` | `(s, d, cur, vc)` tuples |

`--write-model` → `files/models/{base}.lp`. `build_model_` also unconditionally writes `model.lp` in CWD (may want to remove for production).

---

## Debug Artifacts (`--debug-viz-dir`)

Per iteration (stem `{base}_iter{NN}_src{S}[_dst{D}]`):

| File | Content |
|------|---------|
| `.lp` | Gurobi model |
| `_after_solve.txt` | Flows, topo, turns from solver |
| `_after_harden.txt` | Cumulative hardened state, loads |
| `_topo_*.png`, `_cdg_*.png` | Matplotlib visualizations |

---

## File Section Map

| Lines (approx) | Section |
|----------------|---------|
| 40–97 | Matrix/commodity I/O helpers |
| 101–545 | `CCDGModel` |
| 549–1040 | `CCDGOptimizer` |
| 1046–1340 | Debug viz & per-iter debug writers |
| 1349–1422 | `print_results`, `write_results` |
| 1426–1858 | Priority proxies & flow selection |
| 1861–2233 | `run_*_optimization` loops |
| 2247–2389 | CLI & `main` |

---

## Common Pitfalls for Agents

1. **Sequential infeasibility** — Often `known_load` Big-M or missing bridge topology; check hardened components vs next `(s,d)`.
2. **`known_lmax` vs actual load** — Heuristic lower bound uses injection/Moore bounds; can force `L_max ≥ 5` while true cumulative load is 1. May over-constrain later iterations.
3. **Connectivity tier** — Checks incident edges, not `s→d` reachability on hardened graph.
4. **Split flow** — Use `_decompose_paths_for_source`; do not assume single predecessor per CDG node.
5. **`--safe` only iter 1** — Spanning-tree expansion is not repeated.
6. **Per-flow radix exhaustion** — Early flows can saturate a node's out-degree before later flows need those edges.
7. **Do not confuse** with `ccdg_model.py` MCF `lambda` — different objective and no sequential state.

---

## Typical Extension Points

| Goal | Where to change |
|------|-----------------|
| New constraint | `CCDGModel._constr_*`, call from `build_model_` |
| New objective / tie-break | `build_model_`, `CCDGOptimizer.solve` phase 2 block |
| New sequential ordering metric | Add proxy in ~1590–1750, wire in `_source_throughput_proxy` or `_flow_throughput_metric` |
| Reachability-aware priority | `_flow_connectivity_tier` / `_select_next_flow` |
| Fix `known_lmax` tracking | Per-source/per-flow loops where `known_lmax = max(known_lmax, iter_lmax)` |
| Remove debug `model.lp` write | `CCDGModel.build_model_` line ~539 |
| Extract to modules | Natural split: `model.py`, `optimizer.py`, `sequential.py`, `cli.py` (currently monolithic by design) |

---

## Minimal Usage Examples

On this cluster, Gurobi is only licensed on **SLURM compute nodes** — do not run the solver directly on a login node. Always `source setup.sh` first (see **Running on the cluster** below).

```bash
# From repo root, inside an srun/sbatch shell after setup.sh:
python src/ccdg_ns/minmax_link_based.py --n_nodes 10 --radix 4 --n_vcs 2

# Sequential per-flow with hop priority and debug
python src/ccdg_ns/minmax_link_based.py --n_nodes 20 --radix 4 --n_vcs 2 \
  --per-flow-solves --prioritized-per-source hops --debug-viz-dir debug_run/

# Lexicographic: min L_max then min total load
python src/ccdg_ns/minmax_link_based.py --n_nodes 10 --radix 4 --n_vcs 2 \
  --hierarchical-objectives --obj-gap-pct 1
```

---

## Running on the Cluster (SLURM)

This codebase targets the **Negishi / Purdue RCAC** environment. Gurobi must run on a compute node allocated through SLURM; running `python ... minmax_link_based.py` on a login node will fail or hang on license checkout.

### Environment setup (`setup.sh`)

From the **repository root**, source the setup script in every interactive or batch session before invoking Python:

```bash
source setup.sh
```

`setup.sh` (repo root) does the following:

- `module load gurobi/10.0` and sets `GRB_LICENSE_FILE` / `LD_LIBRARY_PATH` for the cluster Gurobi install
- Activates `venv_py12/bin/activate` (gurobipy matched to Gurobi 12)
- Loads `gcc` and prints the detected Gurobi version

All SLURM job scripts in this repo call `source setup.sh` after `cd $SLURM_SUBMIT_DIR`.

### Interactive run (one problem)

Request a compute shell, then run inside it:

```bash
srun --account mithuna -p cpu --cpus-per-task=64 --pty bash -ic \
  "source setup.sh && python src/ccdg_ns/minmax_link_based.py --n_nodes 10 --radix 4 --n_vcs 2"
```

Use `--threads 64` (or match `--cpus-per-task`) if you want Gurobi to use all allocated cores.

### Batch submission (recommended for sweeps)

Scripts follow the repo convention:

| Role | Path |
|------|------|
| **Job script** (SLURM + command) | `slurm/job_scripts/minmax_link_based/generic_minmax_link_based` |
| **Run script** (loops + `sbatch`) | `slurm/run_scripts/run_minmax_link_based.sh` |

Submit from repo root:

```bash
./slurm/run_scripts/run_minmax_link_based.sh
```

The run script:

- Submits to `--account mithuna -p cpu` with `--cpus-per-task=64` and a 14-day time limit
- Writes logs to `slurm/outputs/minmax_{n}n_{r}r_{v}vcs_{date}.out`
- Exports `N_NODES`, `RADIX`, `N_VCS`, and optional extra CLI via `CLAS` to the job script

Edit **`configs`** in the run script for problem sizes (`"n_nodes radix n_vcs"` per line) and **`clas_list`** for extra flags, e.g.:

```bash
clas_list=(
    ""
    "--threads 64"
    "--per-flow-solves --prioritized-per-source hops"
)
```

Override the Python entrypoint if needed:

```bash
MINMAX_SCRIPT=src/ccdg_ns/minmax_link_based.py ./slurm/run_scripts/run_minmax_link_based.sh
```

### Job script contract

`generic_minmax_link_based` expects these environment variables (set by the run script):

| Variable | Required | Meaning |
|----------|----------|---------|
| `N_NODES` | yes | `--n_nodes` |
| `RADIX` | yes | `--radix` |
| `N_VCS` | yes | `--n_vcs` |
| `CLAS` | no | Extra CLI tokens appended to the Python command |
| `MINMAX_SCRIPT` | no | Default `src/ccdg_ns/minmax_link_based.py` |

Command executed on the compute node:

```bash
srun python ${MINMAX_SCRIPT} --n_nodes ${N_NODES} --radix ${RADIX} --n_vcs ${N_VCS} ${CLAS}
```

### Checking jobs and output

```bash
squeue -u $USER          # queue status (or slist on this cluster)
tail -f slurm/outputs/minmax_10n_4r_2vcs_*.out
```

Solver outputs (topology, paths, routing) land under `topologies_and_routing/` as described in **Output Files** above. Infeasible models may write `iis.ilp` and `model.lp` in the job working directory (repo root when submitted from there).

---

## Dependencies

- **Gurobi** with valid license (`gurobipy`) — on this cluster, use only on SLURM compute nodes after `source setup.sh`
- **Cluster modules/venv** — `setup.sh` loads Gurobi 10.0 module and `venv_py12`
- **matplotlib** (only if `--debug-viz-dir` is used)
- **SLURM** — batch/interactive jobs via `sbatch` / `srun` (see **Running on the Cluster**)
- Python 3.x, stdlib (`argparse`, `collections`, `json`, `math`, `os`, `random`, `sys`, `xml.etree`, `datetime`)

---

This document is intended as a **single onboarding context** for an agent working on or porting `minmax_link_based.py`. For line-level edits, jump to the section map and grep for constraint name prefixes (`lmax_`, `phys_cap_`, `f2c_`, `mtz_`, `flow_con_`).
