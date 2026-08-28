# Topology and routing artifacts

This repository treats the files in `topologies_and_routing/` as one
cross-validated bundle. A topology map alone is not enough to reproduce a
result: the selected routes, destination next hops, and VC allocation must have
the same stem and ordering.

## The two 128-router designs

Routers use the TPU-style coordinate mapping
`router = x + 4*y + 16*z` over a `4x4x8` global grid. The final filename
component `4x4x4` on the TONS design is the canonical (minimum) symmetry cube;
there are two such cubes along z. Each router has six bidirectional ports, so a
map contains 768 directed edges. Of these, 576 are fixed, non-wrapping,
one-coordinate intra-cube mesh edges (electrical). The remaining 192 directed
edges are reconfigurable inter-cube links (optical).

- `pt_2c_128r_6p_4x4x8` is the two-cube Parallel Torus baseline. The selected
  bundle uses dimension-order routing with deterministic tie breaking and a
  destination-based next-hop function.
- `asc_lp_sym_2c_128r_6p_4x4x8_4x4x4` is the symmetric, link-partitioned TONS
  topology optimized for approximate sparsest cut (`asc`). `sym` records that
  a canonical cube is expanded by symmetry; `2c`, `128r`, and `6p` mean two
  cubes, 128 routers, and six ports per router.

The link classification is derived from coordinates, not from filename text.
`tools/generate_graph_config.py` deterministically writes ASTRA Graph edge CSVs
with 128 GB/s on every directed edge, 75 ns electrical latency, 50 ns optical
latency, and a one-time 25 ns injection latency.

## Artifact pipeline

The intended generation flow is:

1. A topology generator such as `adorn_code/src/aasc_tpuv4_all.py` writes the
   adjacency `.map`.
2. `adorn_code/python_scripts/allowedturns_vcs_sym.py` constructs a safe
   allowed-turn relation. PT DOR uses `dor.py`/`dor_vc.py` instead.
3. `allowedturns_routing.py` enumerates physical candidate paths that obey the
   turn relation and, with `--destination_based`, the destination next-hop
   constraint.
4. `adorn_code/src/new_mclb_unfactored.py` selects one candidate per ordered
   pair while minimizing maximum directed-link load (MCLB).
5. `convert_pathlist.py` or the routing helpers expand paths into `.nrl2`
   next-hop records.
6. `vnallocator*.py`, `dor_vc.py`, or the combined routing flow assigns VCs and
   writes `.vcmat2`. `destbased_routing_manager.py` is the state-machine
   orchestrator used by newer destination-based runs.

The repository has several generations of these scripts. Some historical
wrapper commands referenced by comments are absent, and multiple legacy
scripts still write to hard-coded `/scratch/negishi/green456` locations.
Therefore the committed artifacts are authoritative for this milestone; do
not assume a legacy command is relocatable without auditing its output paths.

## Record formats and ordering

| Extension | Record | Ordering/meaning |
| --- | --- | --- |
| `.map` | whitespace-separated 0/1 square matrix | row `u`, column `v` is directed edge `u->v` |
| `.rallpaths` | one space-separated node sequence | candidate simple paths; no self paths; grouped by source/destination by the generator |
| `.paths` | one header, then one Python/JSON node list | exactly `N*N` records in source-major order at index `s*N+d`, including `[s]` self paths |
| `.nrl2` | `(path_source, path_destination, current, next)` | expansion of every non-self selected-path hop, in `.paths` order |
| `.allowvcturns` | `((u,v,vc0),(v,w,vc1)) : Boolean` | complete table over physical non-U-turn channel/VC pairs |
| `.vcmat2` | `(path_source, path_destination, current, vc)` | one self record and then one record per selected hop, aligned with `.paths` |

Destination-based means that `(current, final_destination)` determines one
`next` router independent of the flow source. VC values are part of the
channel identity used for deadlock checking; they do not change the physical
path.

## Canonical bundles and measured properties

PT DOR:

- map: `topo_maps/pt_2c_128r_6p_4x4x8.map`
- route/next hop/VC stem:
  `pt_2c_128r_6p_4x4x8_dor_dim_tiebreak_destbased`

TONS:

- map: `topo_maps/asc_lp_sym_2c_128r_6p_4x4x8_4x4x4.map`
- allowed turns: map stem plus `_turns_allowed_cpl_safe.allowvcturns`
- candidates: map stem plus
  `_turns_allowed_cpl_safe_destbased.rallpaths`
- route/next-hop stem: candidate stem plus `_new_mclb_destbased`
- VC: route stem plus `_olb.vcmat2`

Validation measures 16,256 non-self ordered flows. PT DOR has 65,536 selected
hops (4.031496 average) and maximum directed-channel load 128. TONS has 54,864
hops (3.375 average), maximum load 72, 152,917 candidates, and two VCs. Thus
the fixed-route all-to-all load reference is `128/72 = 1.7778`.

Run:

```bash
venv_py12/bin/python tools/validate_topology_bundle.py --bundle pt-dor-128 --root topologies_and_routing
venv_py12/bin/python tools/validate_topology_bundle.py --bundle tons-128 --root topologies_and_routing
venv_py12/bin/python tools/generate_graph_config.py --bundle pt-dor-128 --root topologies_and_routing --output-dir generated/networks/pt
venv_py12/bin/python tools/generate_graph_config.py --bundle tons-128 --root topologies_and_routing --output-dir generated/networks/tons
```

The validator checks map connectivity/degree, source-major route positions,
simple physical paths, destination next hops, candidate inclusion, route/VC
alignment, allowed turns, and acyclicity of both the allowed and selected
route-plus-VC channel dependency graphs.

Analytical ASTRA consumes fixed application-level routes and per-edge timing.
It does not simulate credits, finite packet buffers, VC arbitration, or
packet-level deadlock. The VC validation is therefore a prerequisite check,
not a claim that the analytical backend models VC flow control.
