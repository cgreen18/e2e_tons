#!/usr/bin/env python3
"""
Translate a chosen path set (pathlist) into an MSCCL-style XML plan.

Paths are decomposed into epochs via the same pMCF-style compiler as
gen_coll_comm_a2a_tpuv4_sym.py, with each flow over a path assigned bw/flow = 1.0.

Pathlist format: use .paths files (same as convert_pathlist.py):
  - First line is a header (skipped by default for .paths).
  - One path per (source, destination) pair per line, each line a JSON array of
    node IDs (e.g. [0, 1], [0, 2, 3]).
  - Self-paths (s == d) are ignored; pathlist must cover all (s,d) with s != d.

Example:
  topology: four_mesh.map
  pathlist: topologies_and_routing/routepath_lists/four_mesh_naive_random.paths

  python_scripts/pathlist_to_xml.py \\
    --pathlist topologies_and_routing/routepath_lists/four_mesh_naive_random.paths \\
    --topology four_mesh.map \\
    --xml four_mesh_naive_random.xml

Topology (adj_list) is inferred from path edges if --topology is not given.
"""

import argparse
import json
import math
import os
import sys
import random
import time
from collections import defaultdict

# #region agent log
DEBUG_LOG_PATH = "/home/green456/adorn/.cursor/debug.log"
def _dbg(hypothesis_id, location, message, data=None):
    try:
        with open(DEBUG_LOG_PATH, "a") as f:
            f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": hypothesis_id, "location": location, "message": message, "data": data or {}, "timestamp": int(time.time() * 1000)}) + "\n")
    except Exception:
        pass
# #endregion

# Optional: JSONL support (like convert_pathlist.py)
try:
    import orjson
except ImportError:
    orjson = None
import json


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Import tpuv4_symmetry for symmetric allocation
PYTHON_SCRIPTS_DIR = os.path.normpath(os.path.join(BASE_DIR))
if PYTHON_SCRIPTS_DIR not in sys.path:
    sys.path.append(PYTHON_SCRIPTS_DIR)
try:
    from tpuv4_symmetry import TPUv4_Symmetry
except ImportError:
    TPUv4_Symmetry = None



print(f'imported gen_coll_comm_a2a_tpuv4_sym',flush=True)
# Larger buffer for faster sequential read (bytes)
_READ_BUFFER_SIZE = 4 * 1024 * 1024  # 4 MiB


def stream_pathlist(path, skip_header=False):
    """Yield one path (list of int node IDs) per line. Fast path for .paths (binary + orjson)."""
    is_paths_file = path.endswith(".paths")
    # Fast path: .paths files are JSON-only; binary mode + orjson is much faster
    if is_paths_file and orjson is not None:
        with open(path, "rb", buffering=_READ_BUFFER_SIZE) as inf:
            if skip_header:
                next(inf, None)
            for line in inf:
                line = line.strip()
                if not line:
                    continue
                yield orjson.loads(line)
        return

    # General path: text mode, JSON or space-separated fallback
    with open(path, "r", buffering=_READ_BUFFER_SIZE) as inf:
        if skip_header:
            next(inf, None)
        for line in inf:
            line = line.strip()
            if not line:
                continue
            try:
                if orjson is not None:
                    yield orjson.loads(line)
                else:
                    yield json.loads(line)
                continue
            except (ValueError, TypeError, json.JSONDecodeError):
                pass
            try:
                yield [int(x) for x in line.split()]
            except ValueError:
                raise ValueError(f"Path line is neither JSON array nor space-separated ints: {line[:80]!r}...")


def load_pathlist_to_path_dict(pathlist_filepath, skip_header=False, canon_set=None):
    """
    Load pathlist into path_dict: (s, d) -> [path1, path2, ...].
    Each path is a list of node IDs from s to d.
    Skips self-paths (s == d) and single-node lines.
    """
    path_dict = defaultdict(list)
    for path in stream_pathlist(pathlist_filepath, skip_header=skip_header):

        s, d = path[0], path[-1]

        if s not in canon_set:
            continue

        if len(path) < 2:
            continue
        if s == d:
            continue

        path_dict[(s, d)].append(path)
        if s%100 == 0 and d==0:
            print(f's={s}, d={d}')
    return dict(path_dict)


def adj_list_from_path_dict(path_dict, n_nodes=None):
    """
    Build adjacency list from edges that appear in path_dict.
    If n_nodes is None, use max(node_id over all paths) + 1.
    """
    edges = set()
    max_node = -1
    for paths in path_dict.values():
        for path in paths:
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                edges.add((u, v))
                max_node = max(max_node, u, v)
    if n_nodes is None:
        n_nodes = max_node + 1
    adj_list = [[] for _ in range(n_nodes)]
    for u, v in edges:
        if u < n_nodes:
            adj_list[u].append(v)
    for u in range(n_nodes):
        adj_list[u] = sorted(set(adj_list[u]))
    return adj_list


def ingest_map(path_name, assert_uniform_capacity):

    if True:
        print(f'Ingesting r map ({path_name})')

    adj_list = []
    adj_mat = []

    with open(path_name, 'r') as inf:
        for row in inf:
            r_conns = row.split(' ')
            if '\n' in r_conns:
                r_conns.remove('\n')

            # deal with approximate values (from MIP)
            try:
                r_conns = [int(elem) for elem in r_conns]
            except:
                r_conns = [int(float(elem)) for elem in r_conns]

            adj_mat.append(r_conns)
            adj_list.append( [r for r, c in enumerate(r_conns) if c > 0] )

    if assert_uniform_capacity:
        n_nodes = len(adj_mat)
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i==j: continue
                adj_mat[i][j] = min(adj_mat[i][j],1)

    return adj_mat, adj_list


def write_msccl_xml_from_link_epochs(adj_list, epochs, out_xml_path, n_chunks=1, n_channels=1,
                                     algo_name="alltoall_compiled", proto="Simple",
                                     add_self_copy=False,
                                     i_chunks_override=None,
                                     s_chunks_override=None,
                                     o_chunks_override=None):
    """
    Emit MSCCL-style XML for the compiled per-link epochs schedule.

    We create, per GPU g:
      - one recv TB per in-neighbor u (recv=u)
      - one send TB per out-neighbor v (send=v)
    And add step(s) with s=t for each epoch transfer.

    Note: dep fields are disabled (hasdep=0). Ordering is enforced by step time s=t.
    """
    n_nodes = len(adj_list)
    in_nei = _build_inverse_adj_list_neighbors(adj_list)

    max_in_deg = max((len(in_nei[v]) for v in range(n_nodes)), default=0)
    s_chunks_default = max(2, 2 * max_in_deg)

    i_chunks_default = n_nodes * n_chunks
    o_chunks_default = n_nodes * n_chunks

    # Allow compiler to override buffer sizes (useful when scratch is allocated dynamically).
    s_chunks = int(s_chunks_override) if s_chunks_override is not None else int(s_chunks_default)
    i_chunks = int(i_chunks_override) if i_chunks_override is not None else int(i_chunks_default)
    o_chunks = int(o_chunks_override) if o_chunks_override is not None else int(o_chunks_default)


    # TB id maps
    recv_tb_id = [dict() for _ in range(n_nodes)]
    send_tb_id = [dict() for _ in range(n_nodes)]

    # tb_steps[g][tbid] -> list[str]
    tb_steps = [defaultdict(list) for _ in range(n_nodes)]
    tb_info = [dict() for _ in range(n_nodes)]  # tbid -> {send,recv,chan}

    for g in range(n_nodes):
        tbid = 0
        for u in sorted(set(in_nei[g])):
            chan = _choose_channel(u, g, n_channels)
            recv_tb_id[g][u] = tbid
            tb_info[g][tbid] = {"send": -1, "recv": u, "chan": chan}
            tbid += 1
        for v in sorted(set(adj_list[g])):
            chan = _choose_channel(g, v, n_channels)
            send_tb_id[g][v] = tbid
            tb_info[g][tbid] = {"send": v, "recv": -1, "chan": chan}
            tbid += 1

    # Fill steps
    for t, transfers in enumerate(epochs):
        for tr in transfers:
            u = tr["u"]; v = tr["v"]

            tbid_s = send_tb_id[u].get(v, None)
            if tbid_s is not None:
                tb_steps[u][tbid_s].append(
                    f'      <step s="{t}" type="s" srcbuf="{tr["srcbuf"]}" srcoff="{tr["srcoff"]}" '
                    f'dstbuf="{tr["dstbuf"]}" dstoff="{tr["dstoff"]}" cnt="{tr["cnt"]}" '
                    f'depid="-1" deps="-1" hasdep="0"/>'
                )

            tbid_r = recv_tb_id[v].get(u, None)
            if tbid_r is not None:
                tb_steps[v][tbid_r].append(
                    f'      <step s="{t}" type="r" srcbuf="{tr["srcbuf"]}" srcoff="{tr["srcoff"]}" '
                    f'dstbuf="{tr["dstbuf"]}" dstoff="{tr["dstoff"]}" cnt="{tr["cnt"]}" '
                    f'depid="-1" deps="-1" hasdep="0"/>'
                )

    # Add self copies at the end (one per subchunk of self, or just once if n_chunks==1)
    if add_self_copy:
        t0 = len(epochs)
        for g in range(n_nodes):
            # host copy on first send TB else first recv TB
            host_tbid = None
            if len(send_tb_id[g]) > 0:
                host_tbid = min(send_tb_id[g].values())
            elif len(recv_tb_id[g]) > 0:
                host_tbid = min(recv_tb_id[g].values())
            else:
                continue

            for q in range(n_chunks):
                tb_steps[g][host_tbid].append(
                    f'      <step s="{t0 + q}" type="cpy" srcbuf="i" srcoff="{g * n_chunks + q}" '
                    f'dstbuf="o" dstoff="{g * n_chunks + q}" cnt="1" depid="-1" deps="-1" hasdep="0"/>'
                )

    # Write XML
    lines = []
    lines.append(
        f'<algo name="{algo_name}" proto="{proto}" nchannels="{n_channels}" '
        f'nchunksperloop="{i_chunks}" ngpus="{n_nodes}" coll="alltoall" inplace="0">'
    )

    for g in range(n_nodes):
        lines.append(f'  <gpu id="{g}" i_chunks="{i_chunks}" o_chunks="{o_chunks}" s_chunks="{s_chunks}">')

        for tbid in sorted(tb_info[g].keys()):
            info = tb_info[g][tbid]
            lines.append(f'    <tb id="{tbid}" send="{info["send"]}" recv="{info["recv"]}" chan="{info["chan"]}">')
            for step_line in tb_steps[g].get(tbid, []):
                lines.append(step_line)
            lines.append(f'    </tb>')

        lines.append(f'  </gpu>')

    lines.append(f'</algo>')

    with open(out_xml_path, "w") as outf:
        outf.write("\n".join(lines))


def verify_epochs_demand_satisfaction(adj_list, epochs, n_chunks, eps=1e-9, check_capacity=True, capacity=1.0):
    """
    Verify that the compiled per-link epochs schedule satisfies all-to-all demand:
    for each (s,d) with s != d, exactly n_chunks are injected at source s (srcbuf i, srcoff for dest d)
    and exactly n_chunks are delivered at destination d (dstbuf o, dstoff for source s).
    
    Also verify edge capacity constraints: for each edge (u,v) and epoch t, total transfer count <= capacity.

    Uses the same layout as compilers: injection srcoff = d*n_chunks + q, delivery dstoff = s*n_chunks + q.
    Returns (demand_ok, capacity_ok) tuple, both True if all checks pass.
    """
    n_nodes = len(adj_list)
    injections = defaultdict(lambda: 0)
    deliveries = defaultdict(lambda: 0)

    for t, transfers in enumerate(epochs):
        for tr in transfers:
            cnt = tr.get("cnt", 1)
            u, v = tr["u"], tr["v"]
            if tr.get("srcbuf") == "i":
                srcoff = tr.get("srcoff", 0)
                if isinstance(srcoff, (int, float)):
                    d = int(srcoff) // n_chunks
                    s = u
                    if s != d and 0 <= d < n_nodes:
                        injections[(s, d)] += cnt
            if tr.get("dstbuf") == "o":
                dstoff = tr.get("dstoff", 0)
                if isinstance(dstoff, (int, float)):
                    s = int(dstoff) // n_chunks
                    d = v
                    if s != d and 0 <= s < n_nodes:
                        deliveries[(s, d)] += cnt

    expected = n_chunks
    violations = []
    for s in range(n_nodes):
        for d in range(n_nodes):
            if s == d:
                continue
            inj = injections.get((s, d), 0)
            deliv = deliveries.get((s, d), 0)
            if inj != expected or deliv != expected:
                violations.append(((s, d), inj, deliv, expected))

    demand_ok = True
    if violations:
        print(f"ERROR: Epoch demand verification failed: {len(violations)} (s,d) pairs with wrong injection/delivery count")
        for ((s, d), inj, deliv, exp) in violations[:15]:
            print(f"  ({s},{d}): injections={inj} deliveries={deliv} expected={exp}")
        if len(violations) > 15:
            print(f"  ... and {len(violations) - 15} more")
        demand_ok = False
    else:
        print("Epoch demand verification passed: all (s,d) demands satisfied in compiled schedule")
    
    capacity_ok = True
    if check_capacity:
        capacity_violations = []
        for t, transfers in enumerate(epochs):
            edge_usage = defaultdict(int)
            for tr in transfers:
                u, v = tr["u"], tr["v"]
                cnt = tr.get("cnt", 1)
                edge_usage[(u, v)] += cnt
            
            for (u, v), usage in edge_usage.items():
                if usage > capacity + eps:
                    capacity_violations.append((t, (u, v), usage, capacity))
        
        if capacity_violations:
            print(f"ERROR: Epoch capacity verification failed: {len(capacity_violations)} edge capacity violations")
            for (epoch, (u, v), usage, cap) in capacity_violations[:15]:
                print(f"  epoch {epoch}, edge ({u},{v}): usage={usage} capacity={cap}")
            if len(capacity_violations) > 15:
                print(f"  ... and {len(capacity_violations) - 15} more")
            capacity_ok = False
        else:
            print("Epoch capacity verification passed: all edge capacities respected in compiled schedule")
    
    return demand_ok, capacity_ok

def flow_paths_to_frac_bw_one_per_path(path_dict: dict):
    """Assign 1.0 bw/flow to each path: flow_paths_to_frac_bw[(s,d)] = [1.0, 1.0, ...]."""
    return {(s, d): [1.0] * len(paths) for (s, d), paths in path_dict.items()}

# _ prefix means helper
def _build_inverse_adj_list_neighbors(adj_list):
    n_nodes = len(adj_list)
    in_nei = [[] for _ in range(n_nodes)]
    for u, nbrs in enumerate(adj_list):
        for v in nbrs:
            in_nei[v].append(u)
    # sort. helpful for XML generation
    for v in range(n_nodes):
        in_nei[v] = sorted(set(in_nei[v]))
    return in_nei

def _choose_channel(u, v, n_channels):
    # Deterministic and symmetric: both endpoints can compute the same channel id.
    # (This matters once n_channels > 1.)
    if n_channels <= 1:
        return 0
    return (u + v) % n_channels


def _quantize_pmcf_paths(path_dict, flow_paths_to_frac_bw, n_chunks, n_nodes, eps=1e-12):
    """
    Turn pMCF's fractional per-path bandwidths into an integer assignment of n_chunks subchunks per (s,d).

    Returns:
        chunk_assignments: list of (s, d, q, path)
            where q in [0, n_chunks) is the subchunk index for commodity (s,d),
            and path is the chosen node list.
    """

    chunk_assignments = []

    for (s,d), paths in path_dict.items():

        chosen = [0] * n_chunks

        for q, pidx in enumerate(chosen[:n_chunks]):
            pidx = int(pidx)
            pidx = max(0, min(pidx, len(paths) - 1))
            path = paths[pidx]
            chunk_assignments.append((s, d, q, path))

    return chunk_assignments

def _static_scratch_slot(s, d, q, n_chunks, n_nodes):
    """Static scratch slot for flow (s,d) subchunk q: n_chunks*(s + n_nodes*d) + q."""
    return n_chunks * (s + n_nodes * d) + q


def compile_pmcf_to_link_epochs(adj_list, path_dict, flow_paths_to_frac_bw,
                                n_chunks=1, n_channels=1, max_epochs=None, seed=0):
    """
    Greedy store-and-forward compiler:
      - One hop per epoch per subchunk
      - Unit capacity per directed link per epoch (<=1 subchunk transfer)
      - Static scratch: slot = n_chunks*(s + n_nodes*d) + q per (s,d,q)
    """
    rnd = random.Random(seed)
    n_nodes = len(adj_list)

    # Build subchunk assignments
    chunk_assignments = _quantize_pmcf_paths(path_dict, flow_paths_to_frac_bw, n_chunks, n_nodes)

    print(f"chunk_assignments = {chunk_assignments[:3]}")

    states = []
    for (s, d, q, path) in chunk_assignments:
        st = {
            "s": s, "d": d, "q": q, "path": path,
            "pos": 0, "done": False, "prev_node": None, "t_recv": None, "t_ready": 0,
        }
        states.append(st)

    epochs = []
    remaining = sum(1 for st in states if not st["done"])
    active = set(range(len(states)))
    t = 0

    while remaining > 0:
        if t % 10 == 0:
            print(f"on epoch {t}")
        if max_epochs is not None and t >= max_epochs:
            print(f'WARNING: hit max_epochs cap ({max_epochs}), stopping early with {remaining} subchunks incomplete')
            break
        used_links = set()
        transfers = []
        ready = [i for i in active if states[i]["t_ready"] <= t]
        rnd.shuffle(ready)

        for idx in ready:
            st = states[idx]
            path = st["path"]
            pos = st["pos"]
            if pos >= len(path) - 1:
                st["done"] = True
                active.discard(idx)
                continue
            u = path[pos]
            v = path[pos + 1]
            if (u, v) in used_links:
                continue
            used_links.add((u, v))

            # Static src/dest/scratch: slot = n_chunks*(s + n_nodes*d) + q
            slot = _static_scratch_slot(st["s"], st["d"], st["q"], n_chunks, n_nodes)
            if u == st["s"] and pos == 0:
                srcbuf = "i"
                srcoff = st["d"] * n_chunks + st["q"]
            else:
                srcbuf = "s"
                srcoff = slot
            if v == st["d"] and pos == len(path) - 2:
                dstbuf = "o"
                dstoff = st["s"] * n_chunks + st["q"]
            else:
                dstbuf = "s"
                dstoff = slot

            chan = _choose_channel(u, v, n_channels)
            transfers.append({
                "u": u, "v": v, "chan": chan,
                "srcbuf": srcbuf, "srcoff": srcoff, "dstbuf": dstbuf, "dstoff": dstoff, "cnt": 1,
            })
            st["pos"] += 1
            if st["pos"] >= len(path) - 1:
                st["done"] = True
                active.discard(idx)
                remaining -= 1
            else:
                st["prev_node"] = u
                st["t_recv"] = t
                st["t_ready"] = t + 1
        epochs.append(transfers)
        t += 1

    expected = n_nodes * (n_nodes - 1) * n_chunks
    inj_cnt = sum(tr.get("cnt", 0) for tr_list in epochs for tr in tr_list if tr.get("srcbuf") == "i")
    out_cnt = sum(tr.get("cnt", 0) for tr_list in epochs for tr in tr_list if tr.get("dstbuf") == "o")
    if inj_cnt != expected or out_cnt != expected:
        raise RuntimeError(f'pMCF compiler A2A mismatch: inj_cnt={inj_cnt} out_cnt={out_cnt} expected={expected}')

    max_scratch_slot = max(
        (tr.get("srcoff", 0) if tr.get("srcbuf") == "s" else 0) or (tr.get("dstoff", 0) if tr.get("dstbuf") == "s" else 0)
        for tr_list in epochs for tr in tr_list
    )
    s_chunks = max(2, max_scratch_slot + 1)
    meta = {
        "i_chunks": n_nodes * n_chunks,
        "o_chunks": n_nodes * n_chunks,
        "s_chunks": s_chunks,
        "epochs": len(epochs),
    }
    return epochs, meta



def compile_pmcf_to_link_epochs_sym(adj_list, path_dict, flow_paths_to_frac_bw_canonical,
                                    canonical_sources, my_tpuv4_symmetry,
                                    n_chunks=1, n_channels=1, max_epochs=None, seed=0):
    """
    Symmetric version: iterate over canonical flows (canonical source, any destination).
    For each candidate hop, evaluate link usage for all equivalent flows; if all those
    edges are free, allocate all equivalent transfers in the same epoch (no separate
    expansion step). Static scratch: slot = n_chunks*(s + n_nodes*d) + q.
    
    Args:
        adj_list: adjacency list
        path_dict: full path dict (will be filtered to canonical sources)
        flow_paths_to_frac_bw_canonical: canonical flow dict keyed by (sc, d_prime) where sc is canonical, d_prime is any destination
        canonical_sources: list of canonical source nodes
        my_tpuv4_symmetry: symmetry object
        n_chunks: chunks per flow
        n_channels: channels
        max_epochs: max epochs limit
        seed: random seed
    
    Returns:
        epochs: list of epochs, each with full transfers for all equivalent sources
        meta: metadata dict
    """
    rnd = random.Random(seed)
    n_nodes = len(adj_list)
    canon_set = set(canonical_sources)
    
    # Filter to canonical sources only
    path_dict_canonical = {k: v for k, v in path_dict.items() if k[0] in canon_set}
    flow_paths_to_frac_bw_canonical_filtered = {k: v for k, v in flow_paths_to_frac_bw_canonical.items() if k[0] in canon_set}
    
    # Build subchunk assignments for canonical sources only
    chunk_assignments = _quantize_pmcf_paths(path_dict_canonical, flow_paths_to_frac_bw_canonical_filtered, n_chunks, n_nodes)

    print(f"chunk_assignments = {chunk_assignments[:3]}")
    
    # Per-subchunk state (only canonical sources)
    states = []
    for (s, d, q, path) in chunk_assignments:
        if s not in canon_set:
            continue  # Should not happen after filtering, but be safe
        st = {
            "s": s,  # canonical source
            "d": d,  # destination (can be any node)
            "q": q,
            "path": path,
            "pos": 0,
            "done": False,
            "prev_node": None,
            "t_recv": None,
            "t_ready": 0,
        }
        states.append(st)
    
    # Greedy epoch packing (canonical sources only)
    epochs_canonical = []
    remaining = sum(1 for st in states if not st["done"])
    active = set(range(len(states)))
    t = 0
    
    while remaining > 0:
        if t % 10 == 0:
            print(f"on epoch {t} (canonical)")
        
        if max_epochs is not None and t >= max_epochs:
            print(f'WARNING: hit max_epochs cap ({max_epochs}), stopping early with {remaining} subchunks incomplete')
            break
        
        used_links = set()
        transfers = []
        ready = [i for i in active if states[i]["t_ready"] <= t]
        rnd.shuffle(ready)
        
        for idx in ready:
            st = states[idx]
            sc, d, q = st["s"], st["d"], st["q"]
            path = st["path"]
            pos = st["pos"]
            if pos >= len(path) - 1:
                st["done"] = True
                active.discard(idx)
                continue
            
            u = path[pos]
            v = path[pos+1]
            is_first_hop = (u == sc and pos == 0)
            is_last_hop = (v == d and pos == len(path) - 2)
            
            # All equivalent flows: (s, d') use edge (uc, vc). Check all edges free before allocating.
            equivalents = my_tpuv4_symmetry.get_all_noncanonical_equivalents(sc)
            equivalent_edges = []  # list of (s, d_prime, uc, vc)
            for s in equivalents:
                if s == sc:
                    uc, vc, d_prime = u, v, d
                else:
                    sc_to_s_tform = my_tpuv4_symmetry.calc_transform_delta(sc, s)
                    uc = my_tpuv4_symmetry.apply_transformation(u, sc_to_s_tform)
                    vc = my_tpuv4_symmetry.apply_transformation(v, sc_to_s_tform)
                    d_prime = my_tpuv4_symmetry.apply_transformation(d, sc_to_s_tform)
                equivalent_edges.append((s, d_prime, uc, vc))
            
            edges_for_equivalents = {(uc, vc) for (_, _, uc, vc) in equivalent_edges}
            if edges_for_equivalents & used_links:
                continue
            used_links |= edges_for_equivalents
            
            for s, d_prime, uc, vc in equivalent_edges:
                if s == d_prime:
                    continue
                slot = _static_scratch_slot(s, d_prime, q, n_chunks, n_nodes)
                if is_first_hop:
                    srcbuf = "i"
                    srcoff = d_prime * n_chunks + q
                else:
                    srcbuf = "s"
                    srcoff = slot
                if is_last_hop:
                    dstbuf = "o"
                    dstoff = s * n_chunks + q
                else:
                    dstbuf = "s"
                    dstoff = slot
                chan = _choose_channel(uc, vc, n_channels)
                transfers.append({
                    "u": uc, "v": vc, "chan": chan,
                    "srcbuf": srcbuf, "srcoff": srcoff,
                    "dstbuf": dstbuf, "dstoff": dstoff,
                    "cnt": 1,
                })
            
            # Advance canonical state once
            st["pos"] += 1
            if st["pos"] >= len(path) - 1:
                st["done"] = True
                active.discard(idx)
                remaining -= 1
            else:
                st["prev_node"] = u
                st["t_recv"] = t
                st["t_ready"] = t + 1
        
        epochs_canonical.append(transfers)
        t += 1
    
    # No expansion: epochs already contain all equivalent transfers
    epochs_expanded = epochs_canonical
    
    print(f"Verifying")
    # Verification: ensure true all-to-all (excluding self) of n_chunks per (s,d)
    expected = n_nodes * (n_nodes - 1) * n_chunks
    inj_cnt = 0
    out_cnt = 0
    for tr_list in epochs_expanded:
        for tr in tr_list:
            if tr.get('srcbuf') == 'i':
                inj_cnt += tr.get('cnt', 0)
            if tr.get('dstbuf') == 'o':
                out_cnt += tr.get('cnt', 0)
    if inj_cnt != expected or out_cnt != expected:
        raise RuntimeError(f'pMCF compiler A2A mismatch: inj_cnt={inj_cnt} out_cnt={out_cnt} expected={expected}')
    
    # Compute scratch size: flat buffer needs max slot index
    max_scratch_slot = max(
        (tr.get("srcoff", 0) if tr.get("srcbuf") == "s" else 0) or
        (tr.get("dstoff", 0) if tr.get("dstbuf") == "s" else 0)
        for epoch in epochs_expanded for tr in epoch
    )
    s_chunks = max(2, max_scratch_slot + 1) if max_scratch_slot >= 0 else 2
    
    meta = {
        "i_chunks": n_nodes * n_chunks,
        "o_chunks": n_nodes * n_chunks,
        "s_chunks": s_chunks,
        "epochs": len(epochs_expanded),
        "epochs_canonical": len(epochs_canonical),
    }
    return epochs_expanded, meta


def simple_compile_pmcf_to_link_epochs_sym(adj_list, path_dict, flow_paths_to_frac_bw_canonical,
                                          canonical_sources, my_tpuv4_symmetry,
                                          n_chunks=1, n_channels=1, max_epochs=None, seed=0):
    """
    Path-iterate symmetric compiler: iterate over canonical paths and edges, greedily
    allocate each hop to the first timestep where all equivalent edges are available.
    Same transfer format and buffer semantics as compile_pmcf_to_link_epochs_sym.
    """
    rnd = random.Random(seed)
    n_nodes = len(adj_list)
    canon_set = set(canonical_sources)

    path_dict_canonical = {k: v for k, v in path_dict.items() if k[0] in canon_set}
    flow_paths_to_frac_bw_canonical_filtered = {k: v for k, v in flow_paths_to_frac_bw_canonical.items() if k[0] in canon_set}
    chunk_assignments = _quantize_pmcf_paths(path_dict_canonical, flow_paths_to_frac_bw_canonical_filtered, n_chunks, n_nodes)

    used_edges_at_time = defaultdict(set)
    transfers_at_time = defaultdict(list)

    order = list(range(len(chunk_assignments)))
    rnd.shuffle(order)
    n_total_chunks = len(chunk_assignments)
    t_start = time.time()
    for iter_num, idx in enumerate(order):
        if iter_num % 1000 == 0 and iter_num > 0:
            t_now = time.time()
            sec_per_iter = (t_now - t_start) / iter_num
            print(f"on iter {iter_num} / {n_total_chunks} ({100*round(iter_num/n_total_chunks,1)}%)")
            print(f"\texpected time remaining: {round((n_total_chunks - iter_num)*sec_per_iter/3600, 1)} hours ({round(sec_per_iter, 3)} seconds per iter)")
        sc, d, q, path = chunk_assignments[idx]
        if sc not in canon_set:
            continue
        if len(path) < 2:
            continue
        ready_t = 0
        for pos in range(len(path) - 1):
            u, v = path[pos], path[pos + 1]
            is_first_hop = (pos == 0)
            is_last_hop = (pos == len(path) - 2)
            equivalents = my_tpuv4_symmetry.get_all_noncanonical_equivalents(sc)
            equivalent_edges = []
            for s in equivalents:
                if s == sc:
                    uc, vc, d_prime = u, v, d
                else:
                    sc_to_s_tform = my_tpuv4_symmetry.calc_transform_delta(sc, s)
                    uc = my_tpuv4_symmetry.apply_transformation(u, sc_to_s_tform)
                    vc = my_tpuv4_symmetry.apply_transformation(v, sc_to_s_tform)
                    d_prime = my_tpuv4_symmetry.apply_transformation(d, sc_to_s_tform)
                equivalent_edges.append((s, d_prime, uc, vc))
            edges_for_equivalents = {(uc, vc) for (_, _, uc, vc) in equivalent_edges}
            t = ready_t
            while True:
                if max_epochs is not None and t >= max_epochs:
                    print(f'WARNING: simple_compile_pmcf_to_link_epochs_sym hit max_epochs ({max_epochs}), skipping remaining hops for chunk (sc={sc}, d={d}, q={q})')
                    break
                if not (edges_for_equivalents & used_edges_at_time[t]):
                    break
                t += 1
            if max_epochs is not None and t >= max_epochs:
                break
            used_edges_at_time[t] |= edges_for_equivalents
            for s, d_prime, uc, vc in equivalent_edges:
                if s == d_prime:
                    continue
                slot = _static_scratch_slot(s, d_prime, q, n_chunks, n_nodes)
                if is_first_hop:
                    srcbuf = "i"
                    srcoff = d_prime * n_chunks + q
                else:
                    srcbuf = "s"
                    srcoff = slot
                if is_last_hop:
                    dstbuf = "o"
                    dstoff = s * n_chunks + q
                else:
                    dstbuf = "s"
                    dstoff = slot
                chan = _choose_channel(uc, vc, n_channels)
                transfers_at_time[t].append({
                    "u": uc, "v": vc, "chan": chan,
                    "srcbuf": srcbuf, "srcoff": srcoff,
                    "dstbuf": dstbuf, "dstoff": dstoff,
                    "cnt": 1,
                })
            ready_t = t + 1

    epochs = [transfers_at_time[t] for t in sorted(transfers_at_time)]
    expected = n_nodes * (n_nodes - 1) * n_chunks
    inj_cnt = sum(tr.get("cnt", 0) for epoch in epochs for tr in epoch if tr.get("srcbuf") == "i")
    out_cnt = sum(tr.get("cnt", 0) for epoch in epochs for tr in epoch if tr.get("dstbuf") == "o")
    if inj_cnt != expected or out_cnt != expected:
        raise RuntimeError(f'pMCF compiler A2A mismatch: inj_cnt={inj_cnt} out_cnt={out_cnt} expected={expected}')
    max_scratch_slot = max(
        (tr.get("srcoff", 0) if tr.get("srcbuf") == "s" else 0) or
        (tr.get("dstoff", 0) if tr.get("dstbuf") == "s" else 0)
        for epoch in epochs for tr in epoch
    )
    s_chunks = max(2, max_scratch_slot + 1) if max_scratch_slot >= 0 else 2
    meta = {
        "i_chunks": n_nodes * n_chunks,
        "o_chunks": n_nodes * n_chunks,
        "s_chunks": s_chunks,
        "epochs": len(epochs),
    }
    return epochs, meta


def main():
    parser = argparse.ArgumentParser(
        description="Translate a chosen path set (pathlist) into an MSCCL XML plan (pMCF-style epochs, 1.0 bw/flow per path)."
    )
    parser.add_argument("--pathlist", type=str, required=True,
                        help="Pathlist file (.paths: JSON per line, one path per (s,d); same format as convert_pathlist.py)")
    parser.add_argument("--xml", type=str, required=True, help="Output XML file path")
    parser.add_argument("--topology", type=str, default=None,
                        help="Topology file (adjacency matrix, e.g. four_mesh.map). If not set, topology is inferred from path edges.")
    parser.add_argument("--skip_header", action="store_true", default=None,
                        help="Skip first line of pathlist (default: True for .paths files)")
    parser.add_argument("--no_skip_header", action="store_false", dest="skip_header",
                        help="Do not skip first line")
    parser.add_argument("--n_chunks", type=int, default=1, help="Chunks per (s,d) for all-to-all (default 1)")
    parser.add_argument("--n_channels", type=int, default=1, help="Channels in XML (default 1)")
    parser.add_argument("--max_epochs", type=int, default=None, help="Max epochs (default: no limit)")
    parser.add_argument("--verify", action="store_true", help="Run epoch demand verification after compile")
    parser.add_argument("--symmetric", action="store_true",
                        help="Use symmetry: canonical sources/commodities only, then expand")
    parser.add_argument("--mc_dims", nargs=3, type=int, metavar=("MCX", "MCY", "MCZ"),
                        help="Mega-cube dimensions for symmetry (required if --symmetric)")
    parser.add_argument("--xyzc_dims", nargs=4, type=int, metavar=("X", "Y", "Z", "C"),
                        help="Global x, y, z, cube dimensions (required if --symmetric)")
    parser.add_argument("--sym_type", type=str, choices=["trans", "refl-trans"], default="trans",
                        help="Symmetry type (default: trans)")
    parser.add_argument("--n_nodes", type=int, required=True)
    args = parser.parse_args()

    print(f'args = {args}',flush=True)
    
    # Validate symmetric arguments
    symmetric = args.symmetric
    if symmetric:
        if not args.mc_dims or not args.xyzc_dims:
            print('--symmetric requires --mc_dims <mcx> <mcy> <mcz> and --xyzc_dims <x> <y> <z> <c>')
            sys.exit(1)
        if TPUv4_Symmetry is None:
            print('tpuv4_symmetry module not found; cannot use --symmetric')
            sys.exit(1)


    n_nodes = args.n_nodes

    # Create symmetry object if symmetric mode is enabled
    my_tpuv4_symmetry = None
    canonical_sources = None
    canon_set = set(range(n_nodes))
    if symmetric:
        xyzc_dims = tuple(args.xyzc_dims)
        mc_dims = tuple(args.mc_dims)
        sym_type = args.sym_type
        my_tpuv4_symmetry = TPUv4_Symmetry(xyzc_dims, mc_dims=mc_dims, sym_type=sym_type)
        if args.topology:
            # Verify symmetry matches topology
            my_tpuv4_symmetry.verify_symmetry_for_topology(adj_mat)
        canonical_sources = my_tpuv4_symmetry.get_canonical_nodes()
        canon_set = set(canonical_sources)
        print(f"Symmetry verified; canonical sources: {len(canonical_sources)}")


    skip_header = args.skip_header if args.skip_header is not None else args.pathlist.endswith(".paths")
    
    print(f'Loading pathlist from {args.pathlist}')
    path_dict = load_pathlist_to_path_dict(args.pathlist, skip_header=skip_header, canon_set=canon_set)
    if not path_dict:
        print("No paths loaded; pathlist is empty or invalid.")
        sys.exit(1)

    if args.topology:
        adj_mat, adj_list = ingest_map(args.topology, assert_uniform_capacity=True)
        n_nodes = len(adj_list)
    else:
        n_nodes = max(max(p[0], p[-1]) for paths in path_dict.values() for p in paths) + 1
        adj_list = adj_list_from_path_dict(path_dict, n_nodes)
        adj_mat = None  # Not needed if topology not provided


    print(f"Checking for all-to-all paths")
    # Require all-to-all: check based on symmetric mode
    missing = []
    # if symmetric:
    #     # For symmetric: only check canonical sources have paths to all destinations
    #     canon_set = set(canonical_sources)
    #     for sc in canonical_sources:
    #         for d in range(n_nodes):
    #             if sc == d:
    #                 continue
    #             if (sc, d) not in path_dict or not path_dict[(sc, d)]:
    #                 missing.append((sc, d))
    #     if missing:
    #         print(f"Pathlist missing canonical (sc,d) pairs: {len(missing)} (e.g. {missing[:5]}).")
    #         sys.exit(1)
    #     # Note: path_dict is kept full (not filtered) because compile_pmcf_to_link_epochs_sym
    #     # expects full path_dict and filters internally
    #     canon_path_count = sum(1 for (s, d) in path_dict.keys() if s in canon_set)
    #     print(f"Canonical source paths in pathlist: {canon_path_count} (s,d) pairs")
    # else:
    #     # Original all-to-all check for all (s,d)
    #     for s in range(n_nodes):
    #         for d in range(n_nodes):
    #             if s == d:
    #                 continue
    #             if (s, d) not in path_dict or not path_dict[(s, d)]:
    #                 missing.append((s, d))
    #     if missing:
    #         print(f"Pathlist is not complete all-to-all: missing (s,d) for {len(missing)} pairs (e.g. {missing[:5]}).")
    #         sys.exit(1)

    print(f"Ingested pathlist")

    print(f"Pathlist: {args.pathlist} -> n_nodes={n_nodes}, (s,d) pairs={len(path_dict)}")

    # Compile epochs: use symmetric compiler if symmetric mode is enabled
    if symmetric:
        # Filter to canonical sources only for flow dict creation
        canon_set = set(canonical_sources)
        path_dict_canonical = {k: v for k, v in path_dict.items() if k[0] in canon_set}

        # keys are (s,d), values are [1.0]
        # For canonical sources only, assign 1.0 per path
        flow_paths_to_frac_bw_canonical = flow_paths_to_frac_bw_one_per_path(path_dict_canonical)
        
        # print(f"flow_paths_to_frac_bw_canonical = {flow_paths_to_frac_bw_canonical}")


        print("Compiling path set to link epochs (pMCF-sym-style, canonical sources only, then expand)...")
        # epochs, meta = compile_pmcf_to_link_epochs_sym(
        epochs, meta = simple_compile_pmcf_to_link_epochs_sym(

            adj_list=adj_list,
            path_dict=path_dict,  # Should have all canonical (sc,d) pairs
            flow_paths_to_frac_bw_canonical=flow_paths_to_frac_bw_canonical,
            canonical_sources=canonical_sources,
            my_tpuv4_symmetry=my_tpuv4_symmetry,
            n_chunks=args.n_chunks,
            n_channels=args.n_channels,
            max_epochs=args.max_epochs,
        )

    else:

        # keys are (s,d), values are [1.0]
        flow_paths_to_frac_bw = flow_paths_to_frac_bw_one_per_path(path_dict)
        # print(f"flow_paths_to_frac_bw = {flow_paths_to_frac_bw}")


        print("Compiling path set to link epochs (pMCF-style, 1.0 bw/flow per path)...")
        epochs, meta = compile_pmcf_to_link_epochs(
            adj_list=adj_list,
            path_dict=path_dict,
            flow_paths_to_frac_bw=flow_paths_to_frac_bw,
            n_chunks=args.n_chunks,
            n_channels=args.n_channels,
            max_epochs=args.max_epochs,
        )
    print(f"Epochs: {len(epochs)} (meta: {meta})")

    write_msccl_xml_from_link_epochs(
        adj_list=adj_list,
        epochs=epochs,
        out_xml_path=args.xml,
        n_chunks=args.n_chunks,
        n_channels=args.n_channels,
    )


    print(f"Wrote MSCCL XML to: {args.xml}")

    if args.verify:
        verify_epochs_demand_satisfaction(adj_list, epochs, args.n_chunks)


if __name__ == "__main__":
    main()
