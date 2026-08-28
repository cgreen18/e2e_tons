# Copyright (c) 2026 Purdue University
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met: redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer;
# redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution;
# neither the name of the copyright holders nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Author(s): Conor Green

# Gurobi libs
import gurobipy as gp
from gurobipy import GRB

# regular libs
import argparse
import csv
import fcntl
import pickle
import random
import math
import collections
import os
import sys
import multiprocessing
import time
from collections import deque, defaultdict

# checkpoint dir for decomp master/child
MCF_DECOMP_CHECKPOINT_DIR = "/scratch/negishi/green456/mcf_decomp_checkpoints"

# pipd
import networkx as nx

# symmetry (TPUv4 canonical set)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# running log for decomp stats (topo_name, num_epochs, num_nodes, num_links)
A2A_STATS_CSV = os.path.normpath(os.path.join(BASE_DIR, "..", "files", "a2a_stats.csv"))
sys.path.append(os.path.join(BASE_DIR, "..", "python_scripts"))
try:
    from tpuv4_symmetry import TPUv4_Symmetry
except ImportError:
    TPUv4_Symmetry = None

# constants
VERBOSE = False # for all

# Regular Functions
# --------------------------------------------------------------------------------

def append_a2a_stats(topo_name, num_epochs, num_nodes, num_links, filepath=None):
    """Append a row to files/a2a_stats.csv: topo_name, num_epochs, num_nodes, num_links.
    Uses an exclusive file lock to avoid concurrency issues when multiple scripts finish at once.
    """
    if filepath is None:
        filepath = A2A_STATS_CSV
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "a+", newline="") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.seek(0)
            content = f.read()
            write_header = len(content) == 0
            f.seek(0, 2)  # end of file
            w = csv.writer(f)
            if write_header:
                w.writerow(["topo_name", "num_epochs", "num_nodes", "num_links"])
            w.writerow([topo_name, num_epochs, num_nodes, num_links])
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

def get_shape(nested_list):
    if isinstance(nested_list, list):
        return [len(nested_list)] + get_shape(nested_list[0])
    else:
        return []

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

def print_path(p):

    print(f'path {p[0]} to {p[-1]} (len {len(p)-1}): ',end='')

    l = len(p)
    for i in range(l-1):
        e = p[i]
        print(f'{e}->',end='')
    print(f'{p[-1]}')

def stream_raw_paths(filepath):
    with open(filepath, "rb", buffering=1024*1024) as inf:
        for bline in inf:
            bline = bline.strip()
            if not bline:
                # skip empty
                continue
            # split bytes into byte tokens, convert each to int
            row = [int(tok) for tok in bline.split()]
            yield row

def ingest_path_list(path_name, lb_flow=None, ub_flow=None, src_set=None, override_ingest_all=False):
    global VERBOSE
    if VERBOSE:
        print(f'Ingesting path list {path_name}')

    line_num = 0
    allpath_dict = defaultdict(list)
    for path in stream_raw_paths(path_name):

        s = path[0]
        d = path[-1]

        skip = False
        above_upper = False
        if lb_flow:
            (lb_s, lb_d) = lb_flow
            if s < lb_s:
                skip = True
                # print(f's<lb_s')
            elif s==lb_s and d < lb_d:
                skip = True
                # print(f's==lb_s and d < lb_d')
        if ub_flow:
            (ub_s, ub_d) = ub_flow
            if s > ub_s:
                skip = True
                above_upper = True
                # print(f's > ub_s')
            elif s==ub_s and d >= ub_d:
                skip = True
                above_upper = True
                # print(f's==ub_s and d > ub_d')
        
        if src_set:
            if s not in src_set:
                skip = True
        

        if above_upper:
            return allpath_dict

        if skip and not override_ingest_all:
            continue
        
        allpath_dict[(s,d)].append(path)

        line_num += 1
        if line_num % 1_000_000 == 0:
            print(f'read {line_num}')

    return allpath_dict

# Graph Algorithms
# --------------------------------------------------------------------------------

def create_nwx_G_from_adj_mat(adj_mat):

    n_nodes = len(adj_mat)
    # directed =  False

    G = nx.DiGraph()

    for src in range(n_nodes):
        for dest in range(n_nodes):

            if(src == dest):
                continue

            # if not directed and src > dest:
            #     continue

            if(adj_mat[src][dest] < 1):
                continue

            # print(f'connecting {src} -> {dest}')

            G.add_edge(src,dest)

    return G

def create_nwx_G_from_adj_list(adj_list):

    G = nx.DiGraph()

    for src, dests in enumerate(adj_list):
        for dest in dests:

            G.add_edge(src,dest)

    return G

def nwx_all_shortest_paths(adj_list):

    n_nodes = len(adj_list)

    G = create_nwx_G_from_adj_list(adj_list)

    all_paths = {}
    for src in range(n_nodes):
        for dest in range(n_nodes):

            if(src == dest):
                continue

            short_path_list = list(nx.all_shortest_paths(G,src,dest))

            # input(f'{src}->{dest} : {short_path_list}')
            all_paths[(src,dest)] = short_path_list

        # print(f'completed src {src}')

    print(f'Completed all short path creation')

    return all_paths

def nwx_diameter(adj_list):

    n_nodes = len(adj_list)

    G = create_nwx_G_from_adj_list(adj_list)

    # compute max over all-pairs shortest paths
    maxdist = 0
    for src in range(n_nodes):
        lens = nx.single_source_shortest_path_length(G, src)
        if len(lens) != n_nodes:
            raise RuntimeError("Graph is not strongly connected; cannot infer lmax/diameter.")
        mymax = max(lens.values())
        if mymax > maxdist:
            maxdist = mymax

    return int(maxdist)

# Collective Functions
# --------------------------------------------------------------------------------

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



# Compilation helpers
# --------------------------------------------------------------------------------

class _ScratchPairAllocator:
    def __init__(self):
        self.free = []
        self.next_id = 0
        self.max_slot = 0
        self.pending_free = []  # slots freed this epoch become available next epoch

    def alloc(self):
        if self.free:
            sl = self.free.pop()
        else:
            sl = self.next_id
            self.next_id += 1
        if sl + 1 > self.max_slot:
            self.max_slot = sl + 1
        return sl

    def free_later(self, sl):
        self.pending_free.append(sl)

    def advance_epoch(self):
        if self.pending_free:
            self.free.extend(self.pending_free)
            self.pending_free = []


def _precompute_rev_bfs_dists(adj_list):
    # dist_to_dst[dst][u] = shortest hops from u to dst (directed)
    n = len(adj_list)
    in_nei = _build_inverse_adj_list_neighbors(adj_list)
    dist_to_dst = [None for _ in range(n)]
    for dst in range(n):
        dist = [None for _ in range(n)]
        q = collections.deque([dst])
        dist[dst] = 0
        while q:
            v = q.popleft()
            dv = dist[v]
            for u in in_nei[v]:
                if dist[u] is None:
                    dist[u] = dv + 1
                    q.append(u)
        dist_to_dst[dst] = dist
    return dist_to_dst


def _compile_edgeflow_tokens_to_link_epochs(adj_list, flow_dict,
                                            n_chunks=1, n_channels=1,
                                            max_epochs=None, seed=0, eps=1e-15,
                                            verbose=False, assert_alltoall=True):
    """
    Store-and-forward compiler for *edge-flow* solutions (link-based MCF, decomposed child flows, or tsMCF aggregated flows).

    Input:
      flow_dict[(s,d)][(u,v)] = nonnegative flow value (rate / fraction).
      For compilation, only the *relative* split across outgoing edges matters; we normalize at each node.

    Output (same format as compile_pmcf_to_link_epochs):
      epochs: list[list[transfer_dict]]
      meta: dict with sizes and verification counts

    All-to-all assertion:
      If assert_alltoall, we require exactly n_chunks for every ordered pair (s,d) with s!=d to be delivered.
    """
    rnd = random.Random(seed)
    n = len(adj_list)

    # Precompute distances to reduce cycling (directed). This does not enumerate paths.
    dist_to_dst = _precompute_rev_bfs_dists(adj_list)

    # Prepare per-node per-prev scratch allocators
    in_nei = _build_inverse_adj_list_neighbors(adj_list)
    pair_alloc = [defaultdict(_ScratchPairAllocator) for _ in range(n)]

    # Token representation: one token == one subchunk of commodity (s,d,q)
    # Location is implicit by which tokens_at[u] list contains it.
    tokens_at = [defaultdict(list) for _ in range(n)]
    total_tokens = 0

    for s in range(n):
        for d in range(n):
            if s == d:
                continue
            for q in range(n_chunks):
                tok = {"s": s, "d": d, "q": q, "prev": None, "slot": None, "ready": 0, "hops": 0}
                tokens_at[s][(s, d)].append(tok)
                total_tokens += 1

    expected = n * (n - 1) * n_chunks

    if assert_alltoall and total_tokens != expected:
        raise RuntimeError(f"token init mismatch: got {total_tokens} expected {expected}")

    print(f"Created and verified tokens")

    # Helper: propose next hop for a token at node u
    def propose_next(u, tok):
        s = tok["s"]; d = tok["d"]
        # If already at destination, no move needed.
        if u == d:
            return None

        # Eligible outgoing edges where flow is positive
        edge_map = flow_dict.get((s, d), None)
        if edge_map is None:
            return None

        # gather candidates
        cand = []
        weights = []
        du = dist_to_dst[d][u]
        for v in adj_list[u]:
            w = edge_map.get((u, v), 0.0)
            if w > eps:
                # Prefer progress if possible
                dv = dist_to_dst[d][v]
                progress = (du is not None and dv is not None and dv < du)
                cand.append((v, progress, w))
                weights.append(w)

        if not cand:
            return None

        # If any progress edges exist, restrict to them
        if du is not None:
            prog = [(v, w) for (v, pr, w) in cand if pr]
            if prog:
                vs = [v for v, _ in prog]
                ws = [w for _, w in prog]
                # weighted choice
                tot = sum(ws)
                r = rnd.random() * tot
                acc = 0.0
                for v, w in zip(vs, ws):
                    acc += w
                    if acc >= r:
                        return v
                return vs[-1]

        # otherwise choose among all candidates by weights
        tot = sum(weights)
        r = rnd.random() * tot
        acc = 0.0
        for (v, _pr, w) in cand:
            acc += w
            if acc >= r:
                return v
        return cand[-1][0]

    # Main greedy epoch packing
    epochs = []
    delivered = 0
    t = 0
    stalled_epochs = 0

    # For verification: count injections and output placements
    inj_cnt = 0
    out_cnt = 0

    while delivered < expected:
        if t % 10 == 0:
            print(f"on epoch {t}")
        if max_epochs is not None and t >= max_epochs:
            raise RuntimeError(f"compile hit max_epochs={max_epochs} with delivered={delivered}/{expected}")

        # enable reuse of scratch freed in prior epoch
        for v in range(n):
            for prev, alloc in pair_alloc[v].items():
                alloc.advance_epoch()

        # Per-edge at most one transfer this epoch
        transfers = []
        used = set()

        # For each node, build proposals from ready tokens
        proposals = [defaultdict(list) for _ in range(n)]  # proposals[u][v] -> list[tok]
        ready_counts = 0

        for u in range(n):
            for (s, d), lst in tokens_at[u].items():
                # build a list of indices of tokens ready
                for tok in lst:
                    if tok["ready"] > t:
                        continue
                    ready_counts += 1
                    v = propose_next(u, tok)
                    if v is None:
                        continue
                    proposals[u][v].append(tok)

        # Try to schedule one token per edge (u,v)
        # Iterate edges in randomized order for fairness
        all_edges = [(u, v) for u in range(n) for v in adj_list[u]]
        rnd.shuffle(all_edges)

        progressed = 0

        for (u, v) in all_edges:
            if (u, v) in used:
                continue
            if not proposals[u].get(v, None):
                continue

            # pick one token to send on (u,v)
            tok_list = proposals[u][v]
            tok = tok_list.pop()  # LIFO is fine; proposals are randomized enough via edge shuffle
            # Remove tok from tokens_at[u][(s,d)] list
            key = (tok["s"], tok["d"])
            try:
                tokens_at[u][key].remove(tok)
            except ValueError:
                continue

            # Determine src buffer and offset
            if u == tok["s"] and tok["prev"] is None and tok["slot"] is None:
                srcbuf = "i"
                srcoff = tok["d"] * n_chunks + tok["q"]
                inj_cnt += 1
            else:
                srcbuf = "s"
                # slot is local to (u, prev)
                base_tag = ("S", tok["prev"], tok["slot"])
                srcoff = base_tag

            # Determine dst buffer and offset
            if v == tok["d"]:
                dstbuf = "o"
                dstoff = tok["s"] * n_chunks + tok["q"]
                out_cnt += 1
                delivered += 1
                tok_next = None
            else:
                dstbuf = "s"
                sl = pair_alloc[v][u].alloc()
                dstoff = ("S", u, sl)
                tok_next = {"s": tok["s"], "d": tok["d"], "q": tok["q"], "prev": u, "slot": sl, "ready": t + 1, "hops": tok["hops"] + 1}

            # Free the source scratch slot, available next epoch
            if tok["prev"] is not None and tok["slot"] is not None:
                pair_alloc[u][tok["prev"]].free_later(tok["slot"])

            chan = _choose_channel(u, v, n_channels)
            transfers.append({
                "u": u, "v": v, "chan": chan,
                "srcbuf": srcbuf, "srcoff": srcoff,
                "dstbuf": dstbuf, "dstoff": dstoff,
                "cnt": 1,
            })
            used.add((u, v))
            progressed += 1

            if tok_next is not None:
                tokens_at[v][(tok_next["s"], tok_next["d"])].append(tok_next)

        epochs.append(transfers)
        t += 1


        if progressed == 0:
            stalled_epochs += 1
            # If we're completely stalled, abort with a useful error
            if stalled_epochs >= 4:
                # Identify some undelivered commodities
                undel = []
                for u in range(n):
                    for (s, d), lst in tokens_at[u].items():
                        if lst:
                            undel.append((s, d, u, len(lst)))
                            if len(undel) >= 10:
                                break
                    if len(undel) >= 10:
                        break
                raise RuntimeError(f"compiler stalled at epoch {t} with delivered={delivered}/{expected}. Example undelivered: {undel}")
        else:
            stalled_epochs = 0

    # Finalize scratch layout: assign per-node bases for each incoming neighbor and resolve placeholders
    ring_sz = [dict() for _ in range(n)]
    total_needed_per_v = [0 for _ in range(n)]
    for v in range(n):
        base = 0
        for prev in sorted(set(in_nei[v])):
            alloc = pair_alloc[v].get(prev, None)
            need = alloc.max_slot if alloc is not None else 0
            need = max(2, int(need))
            ring_sz[v][prev] = (base, need)
            base += need
        total_needed_per_v[v] = base

    s_chunks = max(2, max(total_needed_per_v) if total_needed_per_v else 2)

    for transfers in epochs:
        for tr in transfers:
            u = tr["u"]; v = tr["v"]
            if tr["srcbuf"] == "s":
                _tag, prev, sl = tr["srcoff"]
                base, _need = ring_sz[u][prev]
                tr["srcoff"] = base + sl
            if tr["dstbuf"] == "s":
                _tag, prev, sl = tr["dstoff"]
                base, _need = ring_sz[v][prev]
                tr["dstoff"] = base + sl

    while epochs and len(epochs[-1]) == 0:
        epochs.pop()

    if assert_alltoall:
        if inj_cnt != expected:
            raise RuntimeError(f"compiler injection mismatch: inj_cnt={inj_cnt} expected={expected}")
        if out_cnt != expected:
            raise RuntimeError(f"compiler output mismatch: out_cnt={out_cnt} expected={expected}")

    meta = {
        "i_chunks": n * n_chunks,
        "o_chunks": n * n_chunks,
        "s_chunks": s_chunks,
        "epochs": len(epochs),
        "expected_tokens": expected,
        "inj_cnt": inj_cnt,
        "out_cnt": out_cnt,
    }
    return epochs, meta

def _quantize_pmcf_paths(path_dict, flow_paths_to_frac_bw, n_chunks, n_nodes, eps=1e-12):
    """
    Turn pMCF's fractional per-path bandwidths into an integer assignment of n_chunks subchunks per (s,d).

    Returns:
        chunk_assignments: list of (s, d, q, path)
            where q in [0, n_chunks) is the subchunk index for commodity (s,d),
            and path is the chosen node list.
    """

    chunk_assignments = []

    # IMPORTANT: to ensure a true all-to-all (excluding self), we must assign n_chunks for every (s,d), s!=d,
    # even if some (s,d) is missing due to sparse extraction / tolerance.
    for s in range(n_nodes):
        for d in range(n_nodes):
            if s == d:
                continue

            paths = path_dict.get((s, d), None)
            if paths is None or len(paths) == 0:
                raise RuntimeError(f"pMCF compiler: missing path(s) for commodity {s}->{d}")

            path_fracs = flow_paths_to_frac_bw.get((s, d), [])
            # If missing or empty, fall back to path 0 for all chunks.
            if not path_fracs:
                chosen = [0] * n_chunks
            else:
                # path_fracs is a list[float] aligned with path_dict[(s,d)]
                total = sum(max(0.0, float(bw)) for bw in path_fracs)
                if total <= eps:
                    chosen = [0] * n_chunks
                else:
                    raw = [(p, (max(0.0, float(bw)) / total) * n_chunks) for p, bw in enumerate(path_fracs)]
                    floors = [(p, int(math.floor(val + 1e-15))) for p, val in raw]
                    used = sum(f for _, f in floors)
                    rem = n_chunks - used

                    remainders = sorted([(p, val - math.floor(val + 1e-15)) for p, val in raw],
                                        key=lambda x: x[1], reverse=True)

                    alloc = {p: f for p, f in floors}
                    if remainders:
                        for i in range(rem):
                            alloc[remainders[i % len(remainders)][0]] += 1
                    else:
                        # degenerate: allocate everything to 0
                        alloc[0] = alloc.get(0, 0) + rem

                    chosen = []
                    for p in sorted(alloc.keys()):
                        cnt = alloc[p]
                        chosen += [p] * cnt

                    chosen = chosen[:n_chunks]
                    if len(chosen) < n_chunks:
                        chosen += [0] * (n_chunks - len(chosen))

            for q, pidx in enumerate(chosen[:n_chunks]):
                pidx = int(pidx)
                pidx = max(0, min(pidx, len(paths) - 1))
                path = paths[pidx]
                chunk_assignments.append((s, d, q, path))

    return chunk_assignments
def compile_pmcf_to_link_epochs(adj_list, path_dict, flow_paths_to_frac_bw,
                                n_chunks=1, n_channels=1, max_epochs=None, seed=0):
    """
    Greedy store-and-forward compiler:
      - One hop per epoch per subchunk
      - Unit capacity per directed link per epoch (<=1 subchunk transfer)
      - Uses scratch at intermediate nodes (ping-pong per in-neighbor)

    Returns:
        epochs: list[ list[transfer_dict] ], where each transfer_dict has:
            u, v, chan, srcbuf, srcoff, dstbuf, dstoff, cnt
        meta: dict of sizes, epoch count, etc.
    """
    rnd = random.Random(seed)

    n_nodes = len(adj_list)
    in_nei = _build_inverse_adj_list_neighbors(adj_list)
    max_in_deg = max((len(in_nei[v]) for v in range(n_nodes)), default=0)

    # scratch = 2 slots per in-neighbor (ping-pong), min 2
    s_chunks = max(2, 2 * max_in_deg)

    def scratch_slot(v, prev_u, t_recv):
        # map prev_u to index in in_nei[v]
        try:
            idx = in_nei[v].index(prev_u)
        except ValueError:
            idx = 0
        return 2 * idx + (t_recv % 2)

    # Build subchunk assignments
    chunk_assignments = _quantize_pmcf_paths(path_dict, flow_paths_to_frac_bw, n_chunks, n_nodes)

    # Per-subchunk state
    # Each subchunk advances along its chosen path one hop per epoch when scheduled.
    states = []
    for (s, d, q, path) in chunk_assignments:
        st = {
            "s": s,
            "d": d,
            "q": q,
            "path": path,
            "pos": 0,           # edge index along path (u=path[pos] -> v=path[pos+1])
            "done": False,
            "prev_node": None,  # the node that sent it into current node (for scratch lookup)
            "t_recv": None,     # epoch when it arrived at current node
            "t_ready": 0,       # earliest epoch it can be forwarded
        }
        states.append(st)

    # Greedy epoch packing
    epochs = []
    remaining = sum(1 for st in states if not st["done"])
    t = 0

    while remaining > 0:
        if max_epochs is not None and t >= max_epochs:
            print(f'WARNING: hit max_epochs cap ({max_epochs}), stopping early with {remaining} subchunks incomplete')
            break

        used_links = set()   # (u,v)
        transfers = []

        # randomize order to break ties deterministically via seed
        order = list(range(len(states)))
        rnd.shuffle(order)

        for idx in order:
            st = states[idx]
            if st["done"]:
                continue
            if st["t_ready"] > t:
                continue

            path = st["path"]
            pos = st["pos"]
            if pos >= len(path) - 1:
                st["done"] = True
                continue

            u = path[pos]
            v = path[pos+1]

            if (u,v) in used_links:
                continue
            used_links.add((u,v))

            # Determine src/dst buffer/offsets
            if u == st["s"] and pos == 0:
                # first hop reads from input
                srcbuf = "i"
                srcoff = st["d"] * n_chunks + st["q"]
            else:
                # intermediate hop reads from scratch at u, indexed by who sent it to u
                srcbuf = "s"
                srcoff = scratch_slot(u, st["prev_node"], st["t_recv"])

            if v == st["d"] and pos == len(path) - 2:
                # last hop writes to output
                dstbuf = "o"
                dstoff = st["s"] * n_chunks + st["q"]
            else:
                # intermediate arrival writes to scratch at v, indexed by sender u
                dstbuf = "s"
                dstoff = scratch_slot(v, u, t)

            chan = _choose_channel(u, v, n_channels)

            transfers.append({
                "u": u, "v": v, "chan": chan,
                "srcbuf": srcbuf, "srcoff": srcoff,
                "dstbuf": dstbuf, "dstoff": dstoff,
                "cnt": 1,
            })

            # Advance state
            st["pos"] += 1
            if st["pos"] >= len(path) - 1:
                st["done"] = True
                remaining -= 1
            else:
                st["prev_node"] = u
                st["t_recv"] = t
                st["t_ready"] = t + 1

        epochs.append(transfers)
        t += 1


    # Verification: ensure true all-to-all (excluding self) of n_chunks per (s,d)
    expected = n_nodes * (n_nodes - 1) * n_chunks
    inj_cnt = 0
    out_cnt = 0
    for tr_list in epochs:
        for tr in tr_list:
            if tr.get('srcbuf') == 'i':
                inj_cnt += tr.get('cnt', 0)
            if tr.get('dstbuf') == 'o':
                out_cnt += tr.get('cnt', 0)
    if inj_cnt != expected or out_cnt != expected:
        raise RuntimeError(f'pMCF compiler A2A mismatch: inj_cnt={inj_cnt} out_cnt={out_cnt} expected={expected}')

    meta = {
        "i_chunks": n_nodes * n_chunks,
        "o_chunks": n_nodes * n_chunks,
        "s_chunks": s_chunks,
        "epochs": len(epochs),
        "max_in_deg": max_in_deg,
    }
    return epochs, meta

def _largest_remainder_int_alloc(weights, total, eps=1e-12):
    """
    Largest remainder method to turn nonnegative weights into integer allocations summing to total.

    Args:
        weights: list[float] nonnegative
        total: int >= 0

    Returns:
        alloc: list[int] same length as weights, sum == total
    """
    n = len(weights)
    if n == 0:
        return []
    if total <= 0:
        return [0] * n

    wsum = sum(max(0.0, w) for w in weights)
    if wsum <= eps:
        # uniform fallback
        base = total // n
        rem = total - base * n
        alloc = [base] * n
        for i in range(rem):
            alloc[i] += 1
        return alloc

    raw = [(i, (max(0.0, w) / wsum) * total) for i, w in enumerate(weights)]
    floors = [(i, int(math.floor(val + 1e-15))) for i, val in raw]
    used = sum(v for _, v in floors)
    rem = total - used

    remainders = sorted([(i, val - math.floor(val + 1e-15)) for i, val in raw],
                        key=lambda x: x[1], reverse=True)

    alloc = [0] * n
    for i, v in floors:
        alloc[i] = v
    for k in range(rem):
        alloc[remainders[k % len(remainders)][0]] += 1

    return alloc

def compile_tsmcf_to_link_epochs(adj_list, util_by_t, flow_dict,
                                 n_chunks=1, n_channels=1, seed=0,
                                 eps=1e-15, max_epochs=None):
    """
    Compile tsMCF's time-expanded solution into per-link epochs suitable for XML.

    This compiler enforces a true all-to-all (excluding self) of n_chunks per (s,d) by:
      1) Aggregating edge flows over time: f_agg[(s,d)][(u,v)] = sum_t f[(s,d)][(u,v,t)]
      2) Using the same edge-flow token compiler as link-based/decomposed.

    This preserves the routing structure of tsMCF without generating explicit paths.
    """
    n_nodes = len(adj_list)

    # Aggregate over time
    flow_agg = defaultdict(dict)
    for (s, d), steps in flow_dict.items():
        for (u, v, t), frac in steps.items():
            if abs(frac) <= eps:
                continue
            prev = flow_agg[(s, d)].get((u, v), 0.0)
            flow_agg[(s, d)][(u, v)] = prev + float(frac)

    epochs, meta = _compile_edgeflow_tokens_to_link_epochs(
        adj_list=adj_list,
        flow_dict=flow_agg,
        n_chunks=n_chunks,
        n_channels=n_channels,
        max_epochs=max_epochs,
        seed=seed,
        eps=eps,
        verbose=False,
        assert_alltoall=True,
    )

    # Attach original ts diagnostics
    meta["ts_sum_util"] = float(sum(util_by_t.values())) if util_by_t else None
    meta["ts_lmax"] = max(util_by_t.keys()) if util_by_t else None

    return epochs, meta
def _shortest_path_bfs(adj_list, s, d):
    if s == d:
        return [s]
    n = len(adj_list)
    prev = [-1] * n
    q = deque([s])
    prev[s] = s
    while q:
        u = q.popleft()
        for v in adj_list[u]:
            if prev[v] != -1:
                continue
            prev[v] = u
            if v == d:
                q.clear()
                break
            q.append(v)
    if prev[d] == -1:
        return None
    path = [d]
    cur = d
    while cur != s:
        cur = prev[cur]
        path.append(cur)
    path.reverse()
    return path


def _decompose_flow_to_paths(adj_list, edge_flow, s, d, eps=1e-12, max_paths=None):
    residual = {e: float(val) for e, val in edge_flow.items() if float(val) > eps}
    paths = []
    weights = []

    def build_res_adj():
        res_adj = [[] for _ in range(len(adj_list))]
        for (u, v), val in residual.items():
            if val > eps:
                res_adj[u].append(v)
        return res_adj

    while True:
        if max_paths is not None and len(paths) >= max_paths:
            break

        res_adj = build_res_adj()

        stack = [s]
        parent = {s: None}
        iters = {s: 0}

        found = False
        while stack:
            u = stack[-1]
            if u == d:
                found = True
                break
            neigh = res_adj[u]
            i = iters[u]
            if i >= len(neigh):
                stack.pop()
                continue
            v = neigh[i]
            iters[u] = i + 1
            if v in parent:
                continue
            parent[v] = u
            iters[v] = 0
            stack.append(v)

        if not found:
            break

        path_nodes = []
        cur = d
        while cur is not None:
            path_nodes.append(cur)
            cur = parent[cur]
        path_nodes.reverse()

        bottleneck = float('inf')
        edges = []
        for i in range(len(path_nodes) - 1):
            e = (path_nodes[i], path_nodes[i + 1])
            edges.append(e)
            bottleneck = min(bottleneck, residual.get(e, 0.0))

        if bottleneck <= eps or bottleneck == float('inf'):
            break

        for e in edges:
            residual[e] = residual.get(e, 0.0) - bottleneck
            if residual[e] <= eps:
                residual.pop(e, None)

        paths.append(path_nodes)
        weights.append(bottleneck)

    return paths, weights


def link_flow_to_path_fractions(adj_list, flow_dict, eps=1e-12, max_paths_per_sd=None):
    n = len(adj_list)
    path_dict = {}
    flow_paths_to_frac_bw = {}

    for (s, d), edge_map in flow_dict.items():
        if s == d:
            continue
        paths, weights = _decompose_flow_to_paths(adj_list, edge_map, s, d, eps=eps, max_paths=max_paths_per_sd)
        if not paths:
            sp = _shortest_path_bfs(adj_list, s, d)
            if sp is None:
                continue
            paths = [sp]
            weights = [1.0]
        path_dict[(s, d)] = paths
        flow_paths_to_frac_bw[(s, d)] = weights

    return path_dict, flow_paths_to_frac_bw


def compile_link_mcf_to_link_epochs(adj_list, flow_dict,
                                    n_chunks=1, n_channels=1, max_epochs=None, seed=0,
                                    eps=1e-15, verbose=False):
    """
    Compile link-based MCF edge-flow solution directly into per-link epochs (no path generation).

    flow_dict[(s,d)][(u,v)] = flow rate on directed edge (u,v) for commodity (s,d)
    """
    epochs, meta = _compile_edgeflow_tokens_to_link_epochs(
        adj_list=adj_list,
        flow_dict=flow_dict,
        n_chunks=n_chunks,
        n_channels=n_channels,
        max_epochs=max_epochs,
        seed=seed,
        eps=eps,
        verbose=verbose,
        assert_alltoall=True,
    )
    return epochs, meta

def _sort_child_flow_dict(child_flow_dict):
    """
    Sort child_flow_dict by (s, d) keys in numerical order.
    Returns a new dict (Python 3.7+ maintains insertion order).
    """
    sorted_items = sorted(child_flow_dict.items(), key=lambda x: (x[0][0], x[0][1]))
    return dict(sorted_items)

def compile_decomposed_to_link_epochs(adj_list, max_thru, child_flow_dict,
                                      n_chunks=1, n_channels=1,
                                      max_epochs=None, seed=0, eps=1e-15, verbose=False):
    """
    Compile decomposed link-based MCF child edge-flow solution directly into per-link epochs (no path generation).

    child_flow_dict[(s,d)][(u,v)] = flow rate on directed edge (u,v) for commodity (s,d)

    Note: max_thru is not required for correctness of full-demand compilation; only relative flow splits matter.
    """
    epochs, meta = _compile_edgeflow_tokens_to_link_epochs(
        adj_list=adj_list,
        flow_dict=child_flow_dict,
        n_chunks=n_chunks,
        n_channels=n_channels,
        max_epochs=max_epochs,
        seed=seed,
        eps=eps,
        verbose=verbose,
        assert_alltoall=True,
    )
    return epochs, meta

def verify_epochs_demand_satisfaction(adj_list, epochs, n_chunks, eps=1e-9):
    """
    Verify that the compiled per-link epochs schedule satisfies all-to-all demand:
    for each (s,d) with s != d, exactly n_chunks are injected at source s (srcbuf i, srcoff for dest d)
    and exactly n_chunks are delivered at destination d (dstbuf o, dstoff for source s).

    Uses the same layout as compilers: injection srcoff = d*n_chunks + q, delivery dstoff = s*n_chunks + q.
    Returns True if all demands satisfied, False otherwise.
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

    if violations:
        print(f"ERROR: Epoch demand verification failed: {len(violations)} (s,d) pairs with wrong injection/delivery count")
        for ((s, d), inj, deliv, exp) in violations[:15]:
            print(f"  ({s},{d}): injections={inj} deliveries={deliv} expected={exp}")
        if len(violations) > 15:
            print(f"  ... and {len(violations) - 15} more")
        return False

    print("Epoch demand verification passed: all (s,d) demands satisfied in compiled schedule")
    return True

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

# Gurobi Functions
# --------------------------------------------------------------------------------

def setup_solver_params(args):

    # solver params
    solver_params = {}

    if args.time_limit is not None:
        solver_params.update({'TimeLimit':args.time_limit*60})
    if args.threads is not None:
        solver_params.update({'Threads':args.threads})
    if args.concurrent_mip is not None:
        solver_params.update({'ConcurrentMIP':args.concurrent_mip})
    if args.mip_focus is not None:
        solver_params.update({'MIPFocus':args.mip_focus})
    if args.heuristic_ratio is not None:
        solver_params.update({'Heuristics':args.heuristic_ratio})
    if args.symmetry_detection is not None:
        solver_params.update({'Symmetry':args.symmetry_detection})
    if args.barrier_iter_limit is not None:
        solver_params.update({'BarIterLimit':args.barrier_iter_limit})
    if args.iter_limit is not None:
        solver_params.update({'IterationLimit':args.iter_limit})
    if args.cut_passes is not None:
        solver_params.update({'CutPasses':args.cut_passes})
    if args.method is not None:
        solver_params.update({'Method':args.method})
    if args.node_method is not None:
        solver_params.update({'NodeMethod':args.node_method})
    if args.crossover is not None:
        solver_params.update({'Crossover':args.crossover})
    if args.crossover_basis is not None:
        solver_params.update({'CrossoverBasis':args.crossover_basis})
    if args.no_rel_heur_time is not None:
        solver_params.update({'NoRelHeurTime':args.no_rel_heur_time})
    if args.presolve is not None:
        solver_params.update({'Presolve':args.presolve})
    if args.presparsify is not None:
        solver_params.update({'PreSparsify':args.presparsify})
    if args.cuts is not None:
        solver_params.update({'Cuts':args.cuts})
    if args.scale_flag is not None:
        solver_params.update({'ScaleFlag':args.scale_flag})
    if args.feas_tol is not None:
        solver_params.update({'FeasibilityTol':args.feas_tol})
    if args.opt_tol is not None:
        solver_params.update({'OptimalityTol':args.opt_tol})
    if args.bar_conv_tol is not None:
        solver_params.update({'BarConvTol':args.bar_conv_tol})
    if args.markowitz_tol is not None:
        solver_params.update({'MarkowitzTol':args.markowitz_tol})
    if args.psd_tol is not None:
        solver_params.update({'PSDTol':args.psd_tol})
    if args.int_feas_tol is not None:
        solver_params.update({'IntFeasTol':args.int_feas_tol})
    if args.mip_gap is not None:
        solver_params.update({'MIPGap':args.mip_gap})
    if args.numeric_focus is not None:
        solver_params.update({'NumericFocus':args.numeric_focus})
    if args.dual_reductions is not None:
        solver_params.update({'DualReductions':args.dual_reductions})
    if args.predual is not None:
        solver_params.update({'PreDual':args.predual})
    if args.degen_moves is not None:
        solver_params.update({'DegenMoves':args.degen_moves})
    if args.output_flag is not None:
        solver_params.update({'OutputFlag':args.output_flag})

    return solver_params

def decomposed_link_based_mcf_master(adj_list, solver_params=None):

    # Implements paper's decomposed link-based master LP (eqs. 6-9).
    # This solves for source-grouped link flows f'[s,(u,v)] and a single max_thru F.

    n_nodes = len(adj_list)
    capacity = 1.0

    inv_adj_list = _build_inverse_adj_list_neighbors(adj_list)

    print(f'n_nodes = {n_nodes} (master)')

    try:
        model_base_name = "link_based_mcf_decomp_master"
        m = gp.Model(model_base_name)

        # Variables
        # --------------------------------------------------------------------------------
        max_thru = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name='max_thru')

        # fprime[s][(u,v)]
        fprime = defaultdict(dict)
        for s in range(n_nodes):
            for u in range(n_nodes):
                for v in adj_list[u]:
                    myvarname = f'v_fp_{s}s_{u}u_{v}v'
                    fprime[s][(u,v)] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=myvarname)

        print(f"Created variables")

        # Constraints
        # --------------------------------------------------------------------------------

        # (7) Edge capacity: sum_s f'[s,e] <= cap(e)
        for u in range(n_nodes):
            for v in adj_list[u]:
                cap_sum = gp.LinExpr()
                for s in range(n_nodes):
                    cap_sum += fprime[s][(u,v)]
                myconstrname = f'c_cap_edge_{u}u_{v}v'
                m.addConstr(cap_sum <= capacity, myconstrname)

        print(f"Created constraint (7)")

        # (8) For each source s and node u != s:
        #     F + sum_out f'[s,u->*] <= sum_in f'[s,*->u]
        for s in range(n_nodes):
            for u in range(n_nodes):
                if u == s:
                    continue

                out_sum = gp.LinExpr()
                for v in adj_list[u]:
                    out_sum += fprime[s][(u,v)]

                in_sum = gp.LinExpr()
                for w in inv_adj_list[u]:
                    in_sum += fprime[s][(w,u)]

                myconstrname = f'c_flowcons_{s}s_{u}u'
                m.addConstr(max_thru + out_sum <= in_sum, myconstrname)

        print(f"Created constraint (8)")


        # Objective
        # --------------------------------------------------------------------------------
        m.setObjective(max_thru, GRB.MAXIMIZE)

        # Solve
        # --------------------------------------------------------------------------------
        if solver_params:
            for k, v in solver_params.items():
                try:
                    m.setParam(k, v)
                except Exception:
                    pass

        m.optimize()

        max_thru_val = float(max_thru.X)
        print(f"max_thru (master) : {max_thru_val}")
        print(f"	(aka obj: {m.ObjVal:g} )")

        fprime_vals = defaultdict(dict)
        for s, e_map in fprime.items():
            for e, var in e_map.items():
                val = float(var.X)
                if abs(val) > 0.0:
                    fprime_vals[s][e] = val

        # Verify master capacity constraints
        if not verify_master_capacity(adj_list, fprime_vals, capacity):
            print("WARNING: Master LP solution violates capacity constraints")

        return max_thru_val, fprime_vals

    except gp.GurobiError as e:
        print(f"Error code {e.errno}: {e}")
        return None, None

    except AttributeError:
        print("Encountered an attribute error")
        return None, None

def decomposed_link_based_mcf_master_sym(adj_list, canonical_sources, my_tpuv4_symmetry, solver_params=None):
    """
    Symmetric master: only fprime[s] for s in canonical_sources.
    Capacity constraint (7) sums over all nodes' symmetric copies on each edge.
    """
    n_nodes = len(adj_list)
    capacity = 1.0
    canon_set = set(canonical_sources)
    inv_adj_list = _build_inverse_adj_list_neighbors(adj_list)

    print(f'n_nodes = {n_nodes} (master sym, {len(canon_set)} canonical sources)')

    try:
        model_base_name = "link_based_mcf_decomp_master_sym"
        m = gp.Model(model_base_name)

        # Variables: only for canonical sources
        max_thru = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name='max_thru')
        fprime = defaultdict(dict)
        for s in canon_set:
            for u in range(n_nodes):
                for v in adj_list[u]:
                    myvarname = f'v_fp_{s}s_{u}u_{v}v'
                    fprime[s][(u,v)] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=myvarname)

        print("Created variables (canonical sources only)")

        # (7) Edge capacity: total flow on (u,v) = sum over all nodes s of
        #     fprime[canonical_equiv(s)][(transform of (u,v) under s->canonical)]
        for u in range(n_nodes):
            for v in adj_list[u]:
                cap_sum = gp.LinExpr()
                for s in range(n_nodes):
                    sc = my_tpuv4_symmetry.canonical_equivalence_map[s]
                    t_s = my_tpuv4_symmetry.canonical_transformations[s]
                    u_c = my_tpuv4_symmetry.apply_transformation(u, t_s)
                    v_c = my_tpuv4_symmetry.apply_transformation(v, t_s)
                    if (u_c, v_c) in fprime.get(sc, {}):
                        cap_sum += fprime[sc][(u_c, v_c)]
                myconstrname = f'c_cap_edge_{u}u_{v}v'
                m.addConstr(cap_sum <= capacity, myconstrname)

        print("Created constraint (7) sym")

        # (8) For each canonical source s and node u != s
        for s in canon_set:
            for u in range(n_nodes):
                if u == s:
                    continue
                out_sum = gp.LinExpr()
                for v in adj_list[u]:
                    out_sum += fprime[s][(u,v)]
                in_sum = gp.LinExpr()
                for w in inv_adj_list[u]:
                    in_sum += fprime[s][(w,u)]
                myconstrname = f'c_flowcons_{s}s_{u}u'
                m.addConstr(max_thru + out_sum <= in_sum, myconstrname)

        print("Created constraint (8) sym")
        m.setObjective(max_thru, GRB.MAXIMIZE)

        if solver_params:
            for k, v in solver_params.items():
                try:
                    m.setParam(k, v)
                except Exception:
                    pass

        m.optimize()

        max_thru_val = float(max_thru.X)
        print(f"max_thru (master sym): {max_thru_val}")

        fprime_vals = defaultdict(dict)
        for s, e_map in fprime.items():
            for e, var in e_map.items():
                val = float(var.X)
                if abs(val) > 0.0:
                    fprime_vals[s][e] = val

        # Verify master capacity constraints
        if not verify_master_capacity(adj_list, fprime_vals, capacity):
            print("WARNING: Master LP solution violates capacity constraints")

        return max_thru_val, fprime_vals

    except gp.GurobiError as e:
        print(f"Error code {e.errno}: {e}")
        return None, None
    except AttributeError:
        print("Encountered an attribute error")
        return None, None

def expand_fprime_symmetry(fprime_vals, my_tpuv4_symmetry, adj_list):
    """
    Expand master fprime from canonical sources only to all N sources.
    For each source s: f_s[(u,v)] = f_s'[(u',v')] where s' = canonical(s) and
    (u',v') is the transform of (u,v) from s to s' (canonical view of edge).
    """
    n_nodes = len(adj_list)
    fprime_full = defaultdict(dict)
    for s in range(n_nodes):
        sc = my_tpuv4_symmetry.canonical_equivalence_map[s]
        tform_s_to_sc = my_tpuv4_symmetry.calc_transform_delta(s, sc)
        for u in range(n_nodes):
            for v in adj_list[u]:
                u_c = my_tpuv4_symmetry.apply_transformation(u, tform_s_to_sc)
                v_c = my_tpuv4_symmetry.apply_transformation(v, tform_s_to_sc)
                if (u_c, v_c) in fprime_vals.get(sc, {}):
                    fprime_full[s][(u, v)] = fprime_vals[sc][(u_c, v_c)]
    return dict(fprime_full)

def decomposed_link_based_mcf_child(adj_list, source, max_thru, fprime_s, solver_params=None):

    # Implements paper's child LP for a fixed source s (eqs. 10-14).
    # Given f'[s,(u,v)] from the master LP, this extracts per-destination flows
    # f[(s,d),(u,v)] with the same throughput max_thru, using minimum-total-flow.

    print(f"Child {source} started")


    n_nodes = len(adj_list)
    inv_adj_list = _build_inverse_adj_list_neighbors(adj_list)

    # materialize capacity for each edge; edges not in fprime_s have capacity 0
    cap = defaultdict(float)
    for (u,v), val in fprime_s.items():
        cap[(u,v)] = float(val)

    try:
        model_base_name = f"link_based_mcf_decomp_child_s{source}"
        m = gp.Model(model_base_name)

        # Variables
        # --------------------------------------------------------------------------------

        # f_d[(u,v)] for each dest d != source
        f = defaultdict(dict)
        for d in range(n_nodes):
            if d == source:
                continue
            for u in range(n_nodes):
                for v in adj_list[u]:
                    myvarname = f'v_f_{source}s_{d}d_{u}u_{v}v'
                    f[d][(u,v)] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=myvarname)

        print("Created variables")


        # Constraints
        # --------------------------------------------------------------------------------

        # (11) Edge capacity: sum_d f[d,e] <= f'[s,e]
        for u in range(n_nodes):
            for v in adj_list[u]:
                cap_sum = gp.LinExpr()
                for d in range(n_nodes):
                    if d == source:
                        continue
                    cap_sum += f[d][(u,v)]

                myconstrname = f'c_cap_edge_{source}s_{u}u_{v}v'
                m.addConstr(cap_sum <= cap[(u,v)], myconstrname)

        print("Created constraint (11)")

        # (12/13) Flow conservation (equalities) for each destination d:
        #   - at source: net outflow == max_thru
        #   - at sink d: net inflow == max_thru
        #   - at intermediate nodes: net flow == 0
        #
        # This prevents "flow disappearance" that can strand tokens during compilation.
        for d in range(n_nodes):
            if d == source:
                continue

            for u in range(n_nodes):

                out_sum = gp.LinExpr()
                for v in adj_list[u]:
                    out_sum += f[d][(u,v)]

                in_sum = gp.LinExpr()
                for w in inv_adj_list[u]:
                    in_sum += f[d][(w,u)]

                if u == source:
                    # supply at source
                    myconstrname = f'c_flowbal_src_{source}s_{d}d_{u}u'
                    m.addConstr(out_sum - in_sum == max_thru, myconstrname)

                elif u == d:
                    # demand at sink
                    myconstrname = f'c_flowbal_sink_{source}s_{d}d_{u}u'
                    m.addConstr(in_sum - out_sum == max_thru, myconstrname)

                else:
                    # conservation at intermediate nodes
                    myconstrname = f'c_flowbal_mid_{source}s_{d}d_{u}u'
                    m.addConstr(out_sum - in_sum == 0.0, myconstrname)

        print("Created constraint (12/13)")

        # # (12) Flow conservation inequality at intermediate nodes (u != source, u != d):
        # #      outflow(d,u) <= inflow(d,u)
        # for d in range(n_nodes):
        #     if d == source:
        #         continue

        #     for u in range(n_nodes):
        #         if u == source or u == d:
        #             continue

        #         out_sum = gp.LinExpr()
        #         for v in adj_list[u]:
        #             out_sum += f[d][(u,v)]

        #         in_sum = gp.LinExpr()
        #         for w in inv_adj_list[u]:
        #             in_sum += f[d][(w,u)]

        #         myconstrname = f'c_flowcons_{source}s_{d}d_{u}u'
        #         m.addConstr(out_sum <= in_sum, myconstrname)

        # # (13) Demand at sink d: inflow_to_d >= max_thru
        # for d in range(n_nodes):
        #     if d == source:
        #         continue

        #     in_to_d = gp.LinExpr()
        #     for w in inv_adj_list[d]:
        #         in_to_d += f[d][(w,d)]

        #     myconstrname = f'c_dem_sink_{source}s_{d}d'
        #     m.addConstr(in_to_d >= max_thru, myconstrname)

        # Objective
        # --------------------------------------------------------------------------------
        # (10) Minimize total flow (encourages short paths, discourages cycles)
        obj = gp.LinExpr()
        for d, e_map in f.items():
            for e, var in e_map.items():
                obj += var
        m.setObjective(obj, GRB.MINIMIZE)

        # Solve
        # --------------------------------------------------------------------------------
        if solver_params:
            for k, v in solver_params.items():
                try:
                    m.setParam(k, v)
                except Exception:
                    pass

        m.optimize()

        # Extract solution (sparse)
        flow_dict = defaultdict(dict)
        for d, e_map in f.items():
            for e, var in e_map.items():
                val = float(var.X)
                if abs(val) > 0.0:
                    flow_dict[(source, d)][e] = val

        # Verify child capacity constraints
        if not verify_child_capacity(adj_list, source, flow_dict, fprime_s):
            print(f"WARNING: Child {source} LP solution violates allocated capacity constraints")

        print(f"Child {source} completed")

        return flow_dict

    except gp.GurobiError as e:
        print(f"Error code {e.errno}: {e}")
        return None

    except AttributeError:
        print("Encountered an attribute error")
        return None

def _child_worker(args):
    (adj_list, s, max_thru, fprime_s, solver_params) = args

    flows_s = decomposed_link_based_mcf_child(
        adj_list,
        source=s,
        max_thru=max_thru,
        fprime_s=fprime_s,
        solver_params=solver_params,
    )

    return (s, flows_s)

def _child_worker_sym(args):
    (adj_list, s, max_thru, fprime_s, my_tpuv4_symmetry, canonical_sources, solver_params) = args
    flows_s = decomposed_link_based_mcf_child_sym(
        adj_list,
        source=s,
        max_thru=max_thru,
        fprime_s=fprime_s,
        my_tpuv4_symmetry=my_tpuv4_symmetry,
        canonical_sources=canonical_sources,
        solver_params=solver_params,
    )
    return (s, flows_s)

def decomposed_link_based_mcf_child_sym(adj_list, source, max_thru, fprime_s, my_tpuv4_symmetry, canonical_sources, solver_params=None):
    """
    Child LP for fixed source s with only canonical destination variables.
    Variables: f[d_c][(u,v)] for canonical d_c != canonical(source).
    For each destination d (all N-1): flow balance is enforced by expressing
    balance for d in terms of balance_d_c at transformed nodes (u',v',w' under d->d_c).
    Capacity: sum over all d of f(s,d)(u,v) <= fprime_s(u,v), with
    f(s,d)(u,v) = f[d_c](T_d_to_dc(u), T_d_to_dc(v)) when d in class(d_c).
    """
    print(f"Child (sym) {source} started")
    n_nodes = len(adj_list)
    inv_adj_list = _build_inverse_adj_list_neighbors(adj_list)
    canon_set = set(canonical_sources)
    sc = my_tpuv4_symmetry.canonical_equivalence_map[source]
    # canonical destinations for this source: all canonical except source's class rep
    canonical_dests = [dc for dc in canon_set if dc != sc]

    cap = defaultdict(float)
    for (u, v), val in fprime_s.items():
        cap[(u, v)] = float(val)

    try:
        m = gp.Model(f"link_based_mcf_decomp_child_sym_s{source}")

        # Variables: only for canonical destinations d_c
        f = defaultdict(dict)
        for d_c in canonical_dests:
            for u in range(n_nodes):
                for v in adj_list[u]:
                    myvarname = f'v_f_{source}s_{d_c}dc_{u}u_{v}v'
                    f[d_c][(u, v)] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=myvarname)
        print(f"Created variables (canonical dests only: {len(canonical_dests)})")

        # (11) Edge capacity: for each (u,v), sum over all d of f(s,d)(u,v)
        #      where f(s,d)(u,v) = f[d_c](u',v') with (u',v') = transform (u,v) under d->d_c
        for u in range(n_nodes):
            for v in adj_list[u]:
                cap_sum = gp.LinExpr()
                for d in range(n_nodes):
                    if d == source:
                        continue
                    d_c = my_tpuv4_symmetry.canonical_equivalence_map[d]
                    tform_d_to_dc = my_tpuv4_symmetry.calc_transform_delta(d, d_c)
                    u_c = my_tpuv4_symmetry.apply_transformation(u, tform_d_to_dc)
                    v_c = my_tpuv4_symmetry.apply_transformation(v, tform_d_to_dc)
                    if (u_c, v_c) in f.get(d_c, {}):
                        cap_sum += f[d_c][(u_c, v_c)]
                myconstrname = f'c_cap_edge_{source}s_{u}u_{v}v'
                m.addConstr(cap_sum <= cap[(u, v)], myconstrname)
        print("Created constraint (11) sym")

        # Flow conservation: for each destination d (all N-1), express in terms of canonical vars
        # For canonical d_c: balance at u = max_thru if u==source, -max_thru if u==d_c, else 0
        # For non-canonical d: balance_d_c(u') = max_thru if u==source, -max_thru if u==d, else 0
        #   where u' = transform(u) under d->d_c
        for d in range(n_nodes):
            if d == source:
                continue
            d_c = my_tpuv4_symmetry.canonical_equivalence_map[d]
            tform_d_to_dc = my_tpuv4_symmetry.calc_transform_delta(d, d_c)
            for u in range(n_nodes):
                u_c = my_tpuv4_symmetry.apply_transformation(u, tform_d_to_dc)
                out_sum = gp.LinExpr()
                for v in adj_list[u]:
                    v_c = my_tpuv4_symmetry.apply_transformation(v, tform_d_to_dc)
                    if (u_c, v_c) in f.get(d_c, {}):
                        out_sum += f[d_c][(u_c, v_c)]
                in_sum = gp.LinExpr()
                for w in inv_adj_list[u]:
                    w_c = my_tpuv4_symmetry.apply_transformation(w, tform_d_to_dc)
                    if (w_c, u_c) in f.get(d_c, {}):
                        in_sum += f[d_c][(w_c, u_c)]
                rhs = 0.0
                if u == source:
                    rhs = max_thru
                elif u == d:
                    rhs = -max_thru
                myconstrname = f'c_flowbal_{source}s_{d}d_{u}u'
                m.addConstr(out_sum - in_sum == rhs, myconstrname)
        print("Created constraint (12/13) sym")

        obj = gp.LinExpr()
        for d_c, e_map in f.items():
            for e, var in e_map.items():
                obj += var
        m.setObjective(obj, GRB.MINIMIZE)

        if solver_params:
            for k, v in solver_params.items():
                try:
                    m.setParam(k, v)
                except Exception:
                    pass
        m.optimize()

        flow_dict = defaultdict(dict)
        for d_c, e_map in f.items():
            for e, var in e_map.items():
                val = float(var.X)
                if abs(val) > 0.0:
                    flow_dict[(source, d_c)][e] = val

        # Capacity is verified after expand_child_flow_dict_destinations_symmetry
        print(f"Child (sym) {source} completed")
        return flow_dict

    except gp.GurobiError as e:
        print(f"Error code {e.errno}: {e}")
        return None
    except AttributeError:
        print("Encountered an attribute error")
        return None

def decomposed_link_based_mcf(adj_list, solver_params=None, parallel_child=False, n_procs=None):

    # Wrapper: solve master, then solve all child LPs (one per source).
    # Returns:
    #   max_thru, fprime_vals, child_flow_dict
    #
    # child_flow_dict[(s,d)][(u,v)] = flow

    n_nodes = len(adj_list)
    if n_procs is None:
        n_procs = max(1, min(n_nodes, os.cpu_count() or 1))



    max_thru, fprime_vals = decomposed_link_based_mcf_master(adj_list, solver_params=solver_params)
    if max_thru is None:
        return None, None, None



    child_flow_dict = defaultdict(dict)

    if not parallel_child:
        for s in range(n_nodes):
            flows_s = decomposed_link_based_mcf_child(
                adj_list,
                source=s,
                max_thru=max_thru,
                fprime_s=fprime_vals.get(s, {}),
                solver_params=solver_params,
            )
            if flows_s:
                for k, v in flows_s.items():
                    child_flow_dict[k] = v

        return max_thru, fprime_vals, _sort_child_flow_dict(child_flow_dict)

    # adjust before children
    if parallel_child:
        solver_params["Threads"] /= n_procs
        solver_params["Threads"] = max(1,solver_params["Threads"])
    old_output_flag = solver_params["OutputFlag"]
    # solver_params["OutputFlag"] = 0

    tasks = []
    for s in range(n_nodes):
        tasks.append((adj_list, s, max_thru, fprime_vals.get(s, {}), solver_params))

    print(f"Starting {n_procs} children")

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=n_procs) as pool:
        results = pool.map(_child_worker, tasks)

    print(f"All children complete")

    # adjust after children
    if parallel_child:
        solver_params["Threads"] *= n_procs
    solver_params["OutputFlag"] = old_output_flag

    for (s, flows_s) in results:
        if flows_s:
            for k, v in flows_s.items():
                child_flow_dict[k] = v

    return max_thru, fprime_vals, _sort_child_flow_dict(child_flow_dict)

def decomposed_link_based_mcf_sym(adj_list, canonical_sources, my_tpuv4_symmetry, solver_params=None, parallel_child=False, n_procs=None):
    """
    Symmetric decomposed MCF: master and children only for canonical sources.
    Returns max_thru, fprime_vals (canonical only), child_flow_dict (canonical (sc,dc) only).
    """
    n_nodes = len(adj_list)
    canon_set = set(canonical_sources)
    if n_procs is None:
        n_procs = max(1, min(len(canon_set), os.cpu_count() or 1))

    max_thru, fprime_vals = decomposed_link_based_mcf_master_sym(
        adj_list, canonical_sources, my_tpuv4_symmetry, solver_params=solver_params
    )
    if max_thru is None:
        return None, None, None

    child_flow_dict = defaultdict(dict)

    if not parallel_child:
        for s in canonical_sources:
            flows_s = decomposed_link_based_mcf_child(
                adj_list,
                source=s,
                max_thru=max_thru,
                fprime_s=fprime_vals.get(s, {}),
                solver_params=solver_params,
            )
            if flows_s:
                for k, v in flows_s.items():
                    child_flow_dict[k] = v
        return max_thru, fprime_vals, _sort_child_flow_dict(child_flow_dict)

    # parallel children (canonical sources only)
    if solver_params is None:
        solver_params = {}
    if parallel_child:
        solver_params["Threads"] = solver_params.get("Threads", 32) / n_procs
        solver_params["Threads"] = max(1, solver_params["Threads"])
    old_output_flag = solver_params.get("OutputFlag", 1)

    tasks = []
    for s in canonical_sources:
        tasks.append((adj_list, s, max_thru, fprime_vals.get(s, {}), solver_params))

    print(f"Starting {n_procs} children (canonical sources only)")

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=n_procs) as pool:
        results = pool.map(_child_worker, tasks)

    if parallel_child:
        solver_params["Threads"] *= n_procs
    solver_params["OutputFlag"] = old_output_flag

    for (s, flows_s) in results:
        if flows_s:
            for k, v in flows_s.items():
                child_flow_dict[k] = v
    return max_thru, fprime_vals, _sort_child_flow_dict(child_flow_dict)

def decomposed_link_based_mcf_sym_full_children(adj_list, canonical_sources, my_tpuv4_symmetry, max_thru, fprime_vals, solver_params=None, parallel_child=False, n_procs=None):
    """
    After master_sym: expand fprime to all N sources, run all N children with child_sym
    (canonical destinations only per child), return child_flow_dict with keys (s, d_c).
    """
    n_nodes = len(adj_list)
    if n_procs is None:
        n_procs = max(1, min(n_nodes, os.cpu_count() or 1))
    fprime_full = expand_fprime_symmetry(fprime_vals, my_tpuv4_symmetry, adj_list)
    child_flow_dict = defaultdict(dict)

    if not parallel_child:
        for s in range(n_nodes):
            flows_s = decomposed_link_based_mcf_child_sym(
                adj_list,
                source=s,
                max_thru=max_thru,
                fprime_s=fprime_full.get(s, {}),
                my_tpuv4_symmetry=my_tpuv4_symmetry,
                canonical_sources=canonical_sources,
                solver_params=solver_params,
            )
            if flows_s:
                for k, v in flows_s.items():
                    child_flow_dict[k] = v
        return max_thru, fprime_vals, _sort_child_flow_dict(child_flow_dict)

    if solver_params is None:
        solver_params = {}
    if parallel_child:
        solver_params = dict(solver_params)
        solver_params["Threads"] = solver_params.get("Threads", 32) / n_procs
        solver_params["Threads"] = max(1, solver_params["Threads"])
    old_output_flag = solver_params.get("OutputFlag", 1)

    tasks = []
    for s in range(n_nodes):
        tasks.append((adj_list, s, max_thru, fprime_full.get(s, {}), my_tpuv4_symmetry, canonical_sources, solver_params))

    print(f"Starting {n_procs} children (all sources, canonical dests per child)")
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=n_procs) as pool:
        results = pool.map(_child_worker_sym, tasks)

    if parallel_child:
        solver_params["Threads"] *= n_procs
    solver_params["OutputFlag"] = old_output_flag

    for (s, flows_s) in results:
        if flows_s:
            for k, v in flows_s.items():
                child_flow_dict[k] = v
    return max_thru, fprime_vals, _sort_child_flow_dict(child_flow_dict)

def tsMCF(adj_list, lmax=None, adj_mat=None, solver_params=None):

    n_nodes = len(adj_list)

    inv_adj_list = _build_inverse_adj_list_neighbors(adj_list)

    # pick lmax
    if lmax is None:

        diameter = nwx_diameter(adj_list)
        lmax = max(1, diameter)

    # define edge capacities (if present)
    if adj_mat:
        edge_cap = adj_mat
    else:
        edge_cap = [[0 for _ in range(n_nodes)] for __ in range(n_nodes)]
        for u in range(n_nodes):
            for v in adj_list[u]:
                edge_cap[u][v] = 1

    print(f'n_nodes = {n_nodes}  lmax/max_epochs = {lmax}')

    try:
        model_base_name = "tsMCF"
        m = gp.Model(model_base_name)

        # Variables
        # --------------------------------------------------------------------------------

        # U[t] (1-indexed)
        U = {}
        for t in range(1, lmax + 1):
            myvarname = f'v_util_{t}t'
            U[t] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=myvarname)

        # f[(s,d)][(u,v)][t]
        f = defaultdict(lambda: defaultdict(dict))

        for s in range(n_nodes):
            for d in range(n_nodes):
                if s == d:
                    continue

                for u in range(n_nodes):
                    for v in adj_list[u]:
                        for t in range(1, lmax + 1):
                            myvarname = f'v_f_{s}s_{d}d_{u}u_{v}v_{t}t'
                            # f[(s, d)][(u, v)][t] = m.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=myvarname)
                            f[(s, d)][(u, v)][t] = m.addVar(lb=0.0, ub=edge_cap[u][v], vtype=GRB.CONTINUOUS, name=myvarname)

        print(f"Completed variables")

        # Constraints
        # --------------------------------------------------------------------------------

        print(f"Starting constraints")

        # (16)
        for t in range(1, lmax + 1):
            for u in range(n_nodes):
                for v in adj_list[u]:
                    cap_sum = gp.LinExpr()
                    for s in range(n_nodes):
                        for d in range(n_nodes):
                            if s == d:
                                continue
                            cap_sum += f[(s, d)][(u, v)][t]
                    myconstrname = f'c_edgeutil_{u}u_{v}v_{t}t'
                    m.addConstr(cap_sum <= edge_cap[u][v]*U[t], myconstrname)

        print(f"Completed utilization cap")


        # (17) helpers
        out_step = defaultdict(lambda: defaultdict(dict))
        in_step  = defaultdict(lambda: defaultdict(dict))
        for s in range(n_nodes):
            for d in range(n_nodes):
                if s == d:
                    continue

                for u in range(n_nodes):
                    for t in range(1, lmax + 1):
                        out_expr = gp.LinExpr()
                        for v in adj_list[u]:
                            out_expr += f[(s, d)][(u, v)][t]
                        out_step[(s, d)][u][t] = out_expr

                        in_expr = gp.LinExpr()
                        for w in inv_adj_list[u]:
                            in_expr += f[(s, d)][(w, u)][t]
                        in_step[(s, d)][u][t] = in_expr

        # (17)
        for s in range(n_nodes):
            for d in range(n_nodes):
                if s == d:
                    continue

                for u in range(n_nodes):
                    if u == s or u == d:
                        continue

                    prefix_out = gp.LinExpr()
                    prefix_in  = gp.LinExpr()  # prefix_in is Σ_{t'' < t} in_step

                    for t in range(1, lmax + 1):
                        prefix_out += out_step[(s, d)][u][t]

                        # constraint for this t uses prefix_in for times < t
                        myconstrname = f'c_causal_{s}s_{d}d_{u}u_{t}t'
                        m.addConstr(prefix_out <= prefix_in, myconstrname)

                        # then extend prefix_in to include in_step at t (for next iteration)
                        prefix_in += in_step[(s, d)][u][t]

        print(f"Completed t+1 data greater than all received data in [0,t]")


        # (18)
        for s in range(n_nodes):
            for d in range(n_nodes):
                if s == d:
                    continue

                for u in range(n_nodes):
                    if u == s or u == d:
                        continue

                    tot_out = gp.LinExpr()
                    tot_in  = gp.LinExpr()
                    for t in range(1, lmax + 1):
                        tot_out += out_step[(s, d)][u][t]
                        tot_in  += in_step[(s, d)][u][t]

                    myconstrname = f'c_conserve_{s}s_{d}d_{u}u'
                    m.addConstr(tot_out == tot_in, myconstrname)

        print(f"Completed t+1 data equal to received data at t")


        # destination forwarding for its own commodity
        for s in range(n_nodes):
            for d in range(n_nodes):
                if s == d:
                    continue

                u = d
                tot_out = gp.LinExpr()
                for t in range(1, lmax + 1):
                    tot_out += out_step[(s, d)][u][t]
                myconstrname = f'c_dest_no_fwd_{s}s_{d}d'
                m.addConstr(tot_out == 0.0, myconstrname)

        print(f"Completed no tx own commodity")


        # disallow source from receiving its own commodity.
        for s in range(n_nodes):
            for d in range(n_nodes):
                if s == d:
                    continue

                u = s
                tot_in = gp.LinExpr()
                for t in range(1, lmax + 1):
                    tot_in += in_step[(s, d)][u][t]
                myconstrname = f'c_src_no_in_{s}s_{d}d'
                m.addConstr(tot_in == 0.0, myconstrname)

        print(f"Completed no rx own commodity")


        # (19)
        for s in range(n_nodes):
            for d in range(n_nodes):
                if s == d:
                    continue

                src_out = gp.LinExpr()
                for t in range(1, lmax + 1):
                    src_out += out_step[(s, d)][s][t]

                dst_in = gp.LinExpr()
                for t in range(1, lmax + 1):
                    dst_in += in_step[(s, d)][d][t]

                myconstrname = f'c_dem_src_{s}s_{d}d'
                m.addConstr(src_out == 1.0, myconstrname)

                myconstrname = f'c_dem_dst_{s}s_{d}d'
                m.addConstr(dst_in == 1.0, myconstrname)

        print(f"Completed in/out equal 1")


        # Objective
        # --------------------------------------------------------------------------------
        obj = gp.quicksum(U[t] for t in range(1, lmax + 1))
        m.setObjective(obj, GRB.MINIMIZE)

        # Solve
        # --------------------------------------------------------------------------------
        if solver_params:
            for k, v in solver_params.items():
                try:
                    m.setParam(k, v)
                except Exception:
                    pass

        m.optimize()

        total_time = float(obj.getValue())
        util_by_t = {t: float(U[t].X) for t in range(1, lmax + 1)}

        print(f"total_time (sum util_t) : {total_time}")
        print(f"\t(aka obj: {m.ObjVal:g} )")

        # Extract solution (sparse)
        # --------------------------------------------------------------------------------
        flow_dict = defaultdict(dict)
        for (s, d), edge_map in f.items():
            for (u, v), t_map in edge_map.items():
                for t, var in t_map.items():
                    val = var.X
                    if abs(val) > 0.0:
                        flow_dict[(s, d)][(u, v, t)] = float(val)

        return total_time, util_by_t, flow_dict

    except gp.GurobiError as e:
        print(f"Error code {e.errno}: {e}")
        return None, None, None

    except AttributeError:
        print("Encountered an attribute error")
        return None, None, None

def link_based_mcf(adj_list, solver_params=None):

    n_nodes = len(adj_list)
    capacity = 1.0

    # inverse adj_list (ie incoming links)
    inv_adj_list = [[] for _ in range(n_nodes)]
    for u in range(n_nodes):
        for v in adj_list[u]:
            inv_adj_list[v].append(u)

    print(f'n_nodes = {n_nodes}')

    try:
        model_base_name = "link_based_mcf"
        m = gp.Model(model_base_name)

        # Variables
        # --------------------------------------------------------------------------------
        max_thru = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name='max_thru')

        # f[(s,d)][(u,v)]
        f = defaultdict(dict)

        # Create link variables for each commodity on each directed link
        for s in range(n_nodes):
            for d in range(n_nodes):
                if s == d:
                    continue
                for u in range(n_nodes):
                    for v in adj_list[u]:
                        myvarname = f'v_f_{s}s_{d}d_{u}u_{v}v'
                        f[(s,d)][(u,v)] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=myvarname)

        # Constraints
        # --------------------------------------------------------------------------------

        # (2)
        for u in range(n_nodes):
            for v in adj_list[u]:
                cap_sum = gp.LinExpr()
                for s in range(n_nodes):
                    for d in range(n_nodes):
                        if s == d:
                            continue
                        cap_sum += f[(s,d)][(u,v)]
                myconstrname = f'c_cap_edge_{u}u_{v}v'
                m.addConstr(cap_sum <= capacity, myconstrname)

        # (3)
        for s in range(n_nodes):
            for d in range(n_nodes):
                if s == d:
                    continue
                for u in range(n_nodes):
                    if u == s or u == d:
                        continue

                    out_sum = gp.LinExpr()
                    for v in adj_list[u]:
                        out_sum += f[(s,d)][(u,v)]

                    in_sum = gp.LinExpr()
                    for w in inv_adj_list[u]:
                        in_sum += f[(s,d)][(w,u)]

                    myconstrname = f'c_flowcons_{s}s_{d}d_{u}u'
                    m.addConstr(out_sum <= in_sum, myconstrname)

        # (4)
        for s in range(n_nodes):
            for d in range(n_nodes):
                if s == d:
                    continue

                in_to_d = gp.LinExpr()
                for w in inv_adj_list[d]:
                    in_to_d += f[(s,d)][(w,d)]

                myconstrname = f'c_dem_sink_{s}s_{d}d'
                m.addConstr(in_to_d >= max_thru, myconstrname)

        # Objective
        # --------------------------------------------------------------------------------
        m.setObjective(max_thru, GRB.MAXIMIZE)

        # Solve
        # --------------------------------------------------------------------------------
        if solver_params:
            for k, v in solver_params.items():
                try:
                    m.setParam(k, v)
                except Exception:
                    pass
        m.optimize()

        max_thru_val = max_thru.X
        print(f"max_thru : {max_thru_val}")
        print(f"\t(aka obj: {m.ObjVal:g} )")

        # Extract solution (sparse)
        # --------------------------------------------------------------------------------
        flow_dict = defaultdict(dict)
        for key_sd, edge_map in f.items():
            for e, var in edge_map.items():
                val = var.X
                if abs(val) > 0.0:
                    flow_dict[key_sd][e] = val

        return max_thru_val, flow_dict

    except gp.GurobiError as e:
        print(f"Error code {e.errno}: {e}")
        return None, None

    except AttributeError:
        print("Encountered an attribute error")
        return None, None

def pMCF(adj_list, path_dict, solver_params=None):

    # Constants
    # --------------------------------------------------------------------------------
    n_nodes = len(adj_list)

    demand = 1.0
    capacity = 1.0

    # construct path set given edge
    # edge_paths[(i,j)][n] is nth path signature that crosses edge (i,j)
    edge_paths = defaultdict(list)

    for sr in range(n_nodes):
        for dr in range(n_nodes):
            if sr==dr: continue
            # flow (sr,dr)
            for p, path in enumerate(path_dict[(sr,dr)]):
                path_len = len(path)
                for n in range(path_len-1):
                    i = path[n]
                    j = path[n+1]

                    path_signature = (sr,dr,p)
                    edge_paths[(i,j)].append(path_signature)

    print(f'n_nodes = {n_nodes}')

    try:
        # Create a new model
        model_base_name = "pMCF"
        m = gp.Model(model_base_name)


        # Variables
        # --------------------------------------------------------------------------------

        max_thru = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name='max_thru')

        path_keys = [(sr, dr, p) for sr in range(n_nodes) for dr in range(n_nodes) if sr != dr for p in range(len(path_dict[(sr, dr)]))]
        path_flow = m.addVars(path_keys, lb=0.0, vtype=GRB.CONTINUOUS, name="v_path_flow")

        if VERBOSE:
            print(f'path_flow tupledict size = {path_flow.size()}')
            for k in list(path_flow.keys())[:10]:
                print(f'key = {k}')

        # Constraints
        # --------------------------------------------------------------------------------

        edge_list = [(i, j) for i in range(n_nodes) for j in adj_list[i]]
        m.addConstrs(
            (gp.quicksum(path_flow[s, d, p] for (s, d, p) in edge_paths.get((i, j), [])) <= capacity for (i, j) in edge_list),
            name="c_cap_edge"
        )

        sd_pairs = [(sr, dr) for sr in range(n_nodes) for dr in range(n_nodes) if sr != dr]
        m.addConstrs(
            (path_flow.sum(sr, dr, '*') >= max_thru * demand for (sr, dr) in sd_pairs),
            name="c_dem_flow"
        )

        # Objectives
        # --------------------------------------------------------------------------------

        m.setObjective(max_thru, GRB.MAXIMIZE)

        # Params and Model Output
        write_model = False
        if write_model:
            out_model_name = f'files/models/{model_base_name}.lp'
            m.write(out_model_name)


        # Solve
        # --------------------------------------------------------------------------------
        if solver_params:
            for k, v in solver_params.items():
                try:
                    m.setParam(k, v)
                except Exception:
                    pass

        m.optimize()

        # Output
        # --------------------------------------------------------------------------------

        max_thru_val = max_thru.X
        print(f"max_thru : {max_thru_val}")

        print(f"\t(aka obj: {m.ObjVal:g} )")

        # indexed by path signature : flow_paths_to_frac_bw[(sr, dr)][p]
        flow_paths_to_frac_bw = defaultdict(list)
        for (sr, dr) in sd_pairs:
            flow_paths_to_frac_bw[(sr, dr)] = [path_flow[sr, dr, p].X for p in range(len(path_dict[(sr, dr)]))]

        for sr in range(n_nodes):
            for dr in range(n_nodes):
                if sr==dr: continue
                for p, path in enumerate(path_dict[(sr,dr)]):
                    print(f"flow {sr}->{dr} : bw {flow_paths_to_frac_bw[(sr,dr)][p]} to path {path}")

        return max_thru_val, flow_paths_to_frac_bw

    # Error Handling
    # --------------------------------------------------------------------------------

    except AttributeError:
        print("Encountered an attribute error")

    except gp.GurobiError as e:
        print(f"Error code {e.errno}: {e}")

def _build_edge_paths_canonical(path_dict, canonical_pairs):
    """
    For each (sc,dc) in canonical_pairs, build edge -> list of path indices.
    Returns: dict (sc,dc) -> dict (u,v) -> list of p such that path_dict[(sc,dc)][p] uses (u,v).
    """
    edge_paths_canonical = defaultdict(lambda: defaultdict(list))
    for (sc, dc) in canonical_pairs:
        paths = path_dict.get((sc, dc), [])
        for p, path in enumerate(paths):
            for n in range(len(path) - 1):
                u, v = path[n], path[n + 1]
                edge_paths_canonical[(sc, dc)][(u, v)].append(p)
    return edge_paths_canonical


def _build_cap_contrib_pmcf_sym_reordered(adj_list, edge_paths_canonical, canonical_pairs, n_nodes, my_tpuv4_symmetry):
    """
    Build cap_contrib using canonical-first loop order (faster when canonical structure is small).
    Adds (sc, dc, p) once per (sc, dc), s, p when (i,j) is valid (no mult_d).
    """
    cap_contrib = defaultdict(list)
    adj_set = [set(adj_list[i]) for i in range(n_nodes)]
    for (sc, dc) in canonical_pairs:
        for (i_c, j_c), path_indices in edge_paths_canonical.get((sc, dc), {}).items():
            for s in my_tpuv4_symmetry.reverse_canonical_equivalence_map.get(sc, [sc]):
                sc_to_s_tform = my_tpuv4_symmetry.calc_transform_delta(sc, s)
                i = my_tpuv4_symmetry.apply_transformation(i_c, sc_to_s_tform)
                j = my_tpuv4_symmetry.apply_transformation(j_c, sc_to_s_tform)
                if j in adj_set[i]:
                    for p in path_indices:
                        cap_contrib[(i, j)].append((sc, dc, p))
    return cap_contrib


def _build_cap_contrib_pmcf_sym_reordered_with_mult_d(adj_list, edge_paths_canonical, canonical_pairs, n_nodes, my_tpuv4_symmetry):
    """
    Build cap_contrib using canonical-first loop with mult_d multiplicity (for comparison only).
    Adds (sc, dc, p) mult_d times per (sc, dc), s, p where mult_d = |equivalence class of dc|.
    This over-counts vs the original loop (correct multiplicity is 1 per s); use only to
    confirm that mult_d variant does not match original. Production uses
    _build_cap_contrib_pmcf_sym_reordered (no mult_d).
    """
    cap_contrib = defaultdict(list)
    adj_set = [set(adj_list[i]) for i in range(n_nodes)]
    for (sc, dc) in canonical_pairs:
        mult_d = len(my_tpuv4_symmetry.reverse_canonical_equivalence_map.get(dc, [dc]))
        for (i_c, j_c), path_indices in edge_paths_canonical.get((sc, dc), {}).items():
            for s in my_tpuv4_symmetry.reverse_canonical_equivalence_map.get(sc, [sc]):
                tform = my_tpuv4_symmetry.calc_transform_delta(s, sc)
                i = my_tpuv4_symmetry.apply_reverse_transformation(i_c, tform)
                j = my_tpuv4_symmetry.apply_reverse_transformation(j_c, tform)
                if j in adj_set[i]:
                    for _ in range(mult_d):
                        for p in path_indices:
                            cap_contrib[(i, j)].append((sc, dc, p))
    return cap_contrib


def _validate_cap_contrib_equivalence(cap_a, cap_b):
    """
    Return True if cap_a and cap_b are equivalent multisets (same (i,j) keys and same
    sorted list of (sc, dc, p) per key). Use to verify reordered cap_contrib matches original.
    """
    keys_a, keys_b = set(cap_a.keys()), set(cap_b.keys())
    if keys_a != keys_b:
        return False
    for key in keys_a:
        if sorted(cap_a[key]) != sorted(cap_b[key]):
            return False
    return True


def pMCF_sym(adj_list, path_dict, canonical_sources, my_tpuv4_symmetry, solver_params=None):
    """
    pMCF with symmetry: only solve for canonical (sc, dc) commodities.
    Capacity on each edge (i,j) sums flow over all (s,d) via transform (i,j) -> (i_c,j_c) in canonical view.
    """
    n_nodes = len(adj_list)
    demand = 1.0
    capacity = 1.0
    canon_set = set(canonical_sources)
    canonical_pairs = [(sc, dc) for sc in canonical_sources for dc in range(n_nodes) if sc != dc]

    edge_paths_canonical = _build_edge_paths_canonical(path_dict, canonical_pairs)

    print(f"pMCF_sym: n_nodes={n_nodes}, {len(canon_set)} canonical sources, {len(canonical_pairs)} canonical (s,d) pairs")

    t0 = time.time()
    # # Precompute for each physical edge (i,j) the list of (sc, dc, p) that contribute.
    # # Approach 2: use reordered (canonical-first) builder and verify against original.
    # cap_contrib_original = defaultdict(list)
    # s_to_canonical = [my_tpuv4_symmetry.get_canonical_equivalent(s) for s in range(n_nodes)]
    # edge_paths_flat = {}
    # for (sc, dc), edge_to_p in edge_paths_canonical.items():
    #     for (i_c, j_c), path_indices in edge_to_p.items():
    #         edge_paths_flat[(sc, dc, i_c, j_c)] = path_indices
    # for i in range(n_nodes):
    #     for j in adj_list[i]:
    #         for s in range(n_nodes):
    #             sc, s_to_sc_tform = s_to_canonical[s]
    #             for d in range(n_nodes):
    #                 if s == d:
    #                     continue
    #                 dc = my_tpuv4_symmetry.apply_transformation(d, s_to_sc_tform)
    #                 i_c = my_tpuv4_symmetry.apply_transformation(i, s_to_sc_tform)
    #                 j_c = my_tpuv4_symmetry.apply_transformation(j, s_to_sc_tform)
    #                 path_indices = edge_paths_flat.get((sc, dc, i_c, j_c), [])
    #                 for p in path_indices:
    #                     cap_contrib_original[(i, j)].append((sc, dc, p))
    # cap_contrib = _build_cap_contrib_pmcf_sym_reordered(
    #     adj_list, edge_paths_canonical, canonical_pairs, n_nodes, my_tpuv4_symmetry
    # )
    # if not _validate_cap_contrib_equivalence(cap_contrib_original, cap_contrib):
    #     raise RuntimeError("pMCF_sym: cap_contrib reordered != original; aborting.")
    # # Optional: confirm mult_d variant does not match original (it over-counts by |class(dc)|).
    # cap_contrib_mult_d = _build_cap_contrib_pmcf_sym_reordered_with_mult_d(
    #     adj_list, edge_paths_canonical, canonical_pairs, n_nodes, my_tpuv4_symmetry
    # )
    # assert not _validate_cap_contrib_equivalence(cap_contrib_original, cap_contrib_mult_d), (
    #     "pMCF_sym: mult_d variant unexpectedly matched original (expected over-count)."
    # )

    # # after all is verified
    # cap_contrib = _build_cap_contrib_pmcf_sym_reordered_with_mult_d(
    #     adj_list, edge_paths_canonical, canonical_pairs, n_nodes, my_tpuv4_symmetry
    # )
    cap_contrib = _build_cap_contrib_pmcf_sym_reordered(
        adj_list, edge_paths_canonical, canonical_pairs, n_nodes, my_tpuv4_symmetry
    )
    print(f"Completed pre-compute")
    print(f"\tin {time.time() - t0}")

    try:
        m = gp.Model("pMCF_sym")

        max_thru = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="max_thru")
        path_keys_sym = [(sc, dc, p) for (sc, dc) in canonical_pairs for p in range(len(path_dict[(sc, dc)]))]
        path_flow = m.addVars(path_keys_sym, lb=0.0, vtype=GRB.CONTINUOUS, name="v_path_flow")

        edge_list = [(i, j) for i in range(n_nodes) for j in adj_list[i]]
        m.addConstrs(
            (gp.quicksum(path_flow[sc, dc, p] for (sc, dc, p) in cap_contrib.get((i, j), [])) <= capacity for (i, j) in edge_list),
            name="c_cap_edge"
        )

        m.addConstrs(
            (path_flow.sum(sc, dc, '*') >= max_thru * demand for (sc, dc) in canonical_pairs),
            name="c_dem_flow"
        )

        m.setObjective(max_thru, GRB.MAXIMIZE)

        if solver_params:
            for k, v in solver_params.items():
                try:
                    m.setParam(k, v)
                except Exception:
                    pass
        m.optimize()

        max_thru_val = float(max_thru.X)
        print(f"max_thru (pMCF_sym): {max_thru_val}")

        flow_paths_to_frac_bw_canonical = defaultdict(list)
        for (sc, dc) in canonical_pairs:
            flow_paths_to_frac_bw_canonical[(sc, dc)] = [float(path_flow[sc, dc, p].X) for p in range(len(path_dict[(sc, dc)]))]

        return max_thru_val, dict(flow_paths_to_frac_bw_canonical)

    except gp.GurobiError as e:
        print(f"Gurobi error in pMCF_sym: {e}")
        return None, None
    except AttributeError:
        print("Attribute error in pMCF_sym")
        return None, None

def expand_pmcf_flow_symmetry(flow_paths_to_frac_bw_canonical, path_dict, canonical_sources, my_tpuv4_symmetry, n_nodes):
    """
    Expand canonical path flows to full (s,d). For each (sc,dc), for each s in class(sc),
    (s,d) with d = transform(dc, sc->s) gets the same path flow values and transformed paths.
    Returns (flow_paths_to_frac_bw_full, path_dict_full) for all (s,d), s != d.
    """
    flow_paths_to_frac_bw_full = defaultdict(list)
    path_dict_full = defaultdict(list)
    for (sc, dc) in flow_paths_to_frac_bw_canonical.keys():
        if sc == dc:
            continue
        paths_sc_dc = path_dict.get((sc, dc), [])
        flows = flow_paths_to_frac_bw_canonical[(sc, dc)]
        for s in my_tpuv4_symmetry.reverse_canonical_equivalence_map.get(sc, [sc]):
            sc_to_s_tform = my_tpuv4_symmetry.calc_transform_delta(sc, s)
            d = my_tpuv4_symmetry.apply_transformation(dc, sc_to_s_tform)
            if s == d:
                continue
            flow_paths_to_frac_bw_full[(s, d)] = list(flows)
            path_dict_full[(s, d)] = [
                [my_tpuv4_symmetry.apply_transformation(n, sc_to_s_tform) for n in path]
                for path in paths_sc_dc
            ]
    return dict(flow_paths_to_frac_bw_full), dict(path_dict_full)

# Main(s)
# --------------------------------------------------------------------------------

def handle_path_based(adj_list, apl_name=None, translate_to_xml=False, n_chunks=1, n_channels=1, max_epochs=None, solver_params=None,
                     symmetric=False, my_tpuv4_symmetry=None):

    if not apl_name:
        path_dict = nwx_all_shortest_paths(adj_list)
        print(f"Completed all paths calculation")
    else:
        path_dict = ingest_path_list(apl_name)
        print(f"Completed all paths ingestion")

    print(f"Completed min hop paths")

    if symmetric and my_tpuv4_symmetry is not None:
        canonical_sources = my_tpuv4_symmetry.get_canonical_nodes()
        max_thru, flow_paths_to_frac_bw_canonical = pMCF_sym(
            adj_list, path_dict, canonical_sources, my_tpuv4_symmetry, solver_params=solver_params
        )
        if max_thru is None:
            return None, None, None
        n_nodes = len(adj_list)
        flow_paths_to_frac_bw, path_dict = expand_pmcf_flow_symmetry(
            flow_paths_to_frac_bw_canonical, path_dict, canonical_sources, my_tpuv4_symmetry, n_nodes
        )
    else:
        max_thru, flow_paths_to_frac_bw = pMCF(adj_list, path_dict, solver_params=solver_params)

    verify_pmcf(adj_list, path_dict, flow_paths_to_frac_bw, max_thru)

    if not translate_to_xml:
        return max_thru, None, None

    epochs, meta = compile_pmcf_to_link_epochs(
                adj_list=adj_list,
                path_dict=path_dict,
                flow_paths_to_frac_bw=flow_paths_to_frac_bw,
                n_chunks=n_chunks,
                n_channels=n_channels,
                max_epochs=max_epochs
            )

    return max_thru, epochs, meta

def handle_link_based(adj_list, translate_to_xml=False, n_chunks=1, n_channels=1, max_epochs=None, solver_params=None):

    max_thru, flow_dict = link_based_mcf(adj_list, solver_params=solver_params)

    if not translate_to_xml:
        return max_thru, None, None

    eps = max(1e-12, 1e-9 * max_thru)

    epochs, meta = compile_link_mcf_to_link_epochs(
        adj_list=adj_list,
        flow_dict=flow_dict,
        n_chunks=n_chunks,
        n_channels=n_channels,
        max_epochs=max_epochs,
        eps=eps
    )

    return max_thru, epochs, meta

def handle_timestepped(adj_list, adj_mat, translate_to_xml=False, n_chunks=1, n_channels=1, max_epochs=None, solver_params=None):

    total_time, util_by_t, flow_dict = tsMCF(adj_list, max_epochs, adj_mat=adj_mat, solver_params=solver_params)
    max_flow = 1/total_time
    n_times = len(util_by_t)

    print(f"total_time : {total_time} => max_flow {1/total_time}")
    print(f"utilization:")
    for t, util in util_by_t.items():
        print(f"\ttime {t} => utilization {util}")

    # print(f"flow dict:")
    # for flow, steps in flow_dict.items():
    #     print(f"\tsrc, dest {flow}")
    #     for (i,j,t), val in steps.items():
    #         print(f"\t\tat time {t}, edge {(i,j)} w/ throughput/flow : {val}")

    # n_nodes = len(adj_list)
    # print(f"edges by time:")
    # for u, conns in enumerate(adj_list):
    #     for v in conns:
    #         print(f"edge {(u,v)}")
    #         for t in range(n_times+1):
    #             for s in range(n_nodes):
    #                 for d in range(n_nodes):
    #                     if (u,v,t) in flow_dict[(s,d)]:
    #                         print(f"\ttime {t}")

    if not translate_to_xml:
        return max_flow, None, None

    eps = max(1e-12, 1e-9 * max_flow)

    epochs, meta = compile_tsmcf_to_link_epochs(
        adj_list=adj_list,
        util_by_t=util_by_t,
        flow_dict=flow_dict,
        n_chunks=n_chunks,
        n_channels=n_channels,
        max_epochs=max_epochs,
        eps=eps
    )

    # attach for debug / reporting if desired
    meta["total_time"] = float(total_time)

    return max_flow, epochs, meta

def verify_pmcf(adj_list, path_dict, flow_paths_to_frac_bw, max_thru, demand=1.0, capacity=1.0, eps=1e-12):
    """
    Verify that pMCF path-flow solution satisfies all demands and edge capacities.

    - Demand: for each (s,d), s != d, sum of path flows >= max_thru * demand.
    - Capacity: for each edge (i,j), sum of path flows using (i,j) <= capacity.

    Returns True if all checks pass, False otherwise.
    """
    n_nodes = len(adj_list)
    required = max_thru * demand

    # Demand satisfaction: every (s,d) must have total path flow >= required
    demand_violations = []
    missing = []
    for s in range(n_nodes):
        for d in range(n_nodes):
            if s == d:
                continue
            path_flows = flow_paths_to_frac_bw.get((s, d), None)
            if path_flows is None or len(path_flows) == 0:
                missing.append((s, d))
                continue
            total = sum(max(0.0, float(f)) for f in path_flows)
            if total < required - eps:
                demand_violations.append(((s, d), total, required, required - total))

    if missing:
        print(f"ERROR: pMCF verification: missing commodities: {len(missing)}")
        for (s, d) in missing[:10]:
            print(f"  ({s},{d})")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")
    if demand_violations:
        print(f"ERROR: pMCF verification: demand violations: {len(demand_violations)}")
        for (s, d), total, req, shortfall in demand_violations[:10]:
            print(f"  ({s},{d}): total={total:.6e} < required={req:.6e} (shortfall={shortfall:.6e})")
        if len(demand_violations) > 10:
            print(f"  ... and {len(demand_violations) - 10} more violations")

    # Edge capacity: for each edge, sum of path flows through it <= capacity
    edge_flow = defaultdict(float)
    for (s, d), path_list in path_dict.items():
        if s == d:
            continue
        flows = flow_paths_to_frac_bw.get((s, d), [])
        for p, path in enumerate(path_list):
            if p >= len(flows):
                continue
            val = max(0.0, float(flows[p]))
            for n in range(len(path) - 1):
                i, j = path[n], path[n + 1]
                edge_flow[(i, j)] += val

    cap_violations = []
    for u in range(n_nodes):
        for v in adj_list[u]:
            total = edge_flow.get((u, v), 0.0)
            if total > capacity + eps:
                cap_violations.append(((u, v), total, capacity, total - capacity))

    if cap_violations:
        print(f"ERROR: pMCF verification: capacity violations: {len(cap_violations)}")
        for (u, v), total, cap, excess in cap_violations[:10]:
            print(f"  Edge ({u},{v}): total={total:.6e} > capacity={cap:.6e} (excess={excess:.6e})")
        if len(cap_violations) > 10:
            print(f"  ... and {len(cap_violations) - 10} more violations")

    if missing or demand_violations or cap_violations:
        return False

    print("pMCF verification passed: all demands satisfied and edge capacities respected")
    return True

def verify_master_capacity(adj_list, fprime_vals, capacity=1.0, eps=1e-9):
    """
    Verify that master LP solution respects edge capacity constraints:
    sum_s fprime[s][(u,v)] <= capacity for all edges (u,v).
    """
    n_nodes = len(adj_list)
    violations = []
    
    for u in range(n_nodes):
        for v in adj_list[u]:
            total_flow = 0.0
            for s in range(n_nodes):
                total_flow += fprime_vals.get(s, {}).get((u, v), 0.0)
            
            if total_flow > capacity + eps:
                violations.append(((u, v), total_flow, capacity, total_flow - capacity))
    
    if violations:
        print(f"ERROR: Master capacity violations: {len(violations)}")
        for (u, v), total, cap, excess in violations[:10]:
            print(f"  Edge ({u},{v}): total={total:.6e} > capacity={cap:.6e} (excess={excess:.6e})")
        if len(violations) > 10:
            print(f"  ... and {len(violations) - 10} more violations")
        return False
    
    print(f"Master capacity check passed: all edges within capacity")
    return True

def verify_child_capacity(adj_list, source, flows_s, fprime_s, eps=1e-9):
    """
    Verify that child LP solution for source s respects allocated capacity:
    sum_d f[(s,d)][(u,v)] <= fprime[s][(u,v)] for all edges (u,v).
    
    Args:
        adj_list: adjacency list
        source: source node s
        flows_s: dict {(s,d): {(u,v): flow_value}} for all destinations d
        fprime_s: dict {(u,v): allocated_capacity} from master LP
        eps: tolerance for floating point comparison
    """
    violations = []
    
    for u in range(len(adj_list)):
        for v in adj_list[u]:
            total_flow = 0.0
            # Sum flow over all destinations d for this source s
            for (s, d), emap in flows_s.items():
                if s == source:
                    total_flow += emap.get((u, v), 0.0)
            
            allocated = fprime_s.get((u, v), 0.0)
            if total_flow > allocated + eps:
                violations.append(((u, v), total_flow, allocated, total_flow - allocated))
    
    if violations:
        print(f"ERROR: Child {source} capacity violations: {len(violations)}")
        for (u, v), total, alloc, excess in violations[:10]:
            print(f"  Edge ({u},{v}): total={total:.6e} > allocated={alloc:.6e} (excess={excess:.6e})")
        if len(violations) > 10:
            print(f"  ... and {len(violations) - 10} more violations")
        return False
    
    return True

def verify_final_capacity(adj_list, child_flow_dict, capacity=1.0, eps=1e-9):
    """
    Verify that final child_flow_dict respects edge capacity constraints:
    sum_{s,d} f[(s,d)][(u,v)] <= capacity for all edges (u,v).
    """
    n_nodes = len(adj_list)
    violations = []
    
    for u in range(n_nodes):
        for v in adj_list[u]:
            total_flow = 0.0
            for (s, d), emap in child_flow_dict.items():
                total_flow += emap.get((u, v), 0.0)
            
            if total_flow > capacity + eps:
                violations.append(((u, v), total_flow, capacity, total_flow - capacity))
    
    if violations:
        print(f"ERROR: Final capacity violations: {len(violations)}")
        for (u, v), total, cap, excess in violations[:10]:
            print(f"  Edge ({u},{v}): total={total:.6e} > capacity={cap:.6e} (excess={excess:.6e})")
        if len(violations) > 10:
            print(f"  ... and {len(violations) - 10} more violations")
        return False
    
    print(f"Final capacity check passed: all edges within capacity")
    return True

def verify_decomposed(adj_list, child_flow_dict):
    eps = 1e-12  # you may tune this (see notes below)
    n_nodes = len(adj_list)

    n_nodes = len(adj_list)
    missing = []
    for s in range(n_nodes):
        for d in range(n_nodes):
            if s == d: 
                continue
            if (s, d) not in child_flow_dict or not child_flow_dict[(s, d)]:
                missing.append((s, d))
    print("missing commodities:", len(missing))

    dead_ends = []  # (s,d,u,in_sum,out_sum)
    for (s, d), emap in child_flow_dict.items():
        # precompute inflow/outflow per node for this commodity
        in_by_u  = [0.0] * n_nodes
        out_by_u = [0.0] * n_nodes

        for (u, v), val in emap.items():
            if val <= eps:
                continue
            out_by_u[u] += val
            in_by_u[v]  += val

        # source must have some outgoing support
        if out_by_u[s] <= eps:
            dead_ends.append((s, d, s, in_by_u[s], out_by_u[s]))
            continue

        # any node (except destination) with inflow must be able to forward
        for u in range(n_nodes):
            if u == d:
                continue
            if in_by_u[u] > eps and out_by_u[u] <= eps:
                dead_ends.append((s, d, u, in_by_u[u], out_by_u[u]))

    print("true dead ends:", len(dead_ends))
    print("sample:", dead_ends[:20])

def expand_child_flow_dict_destinations_symmetry(child_flow_dict_canonical_dests, my_tpuv4_symmetry, n_nodes):
    """
    Expand child_flow_dict from (s, d_c) only to (s, d) for all d.
    Input: keys (s, d_c) for canonical d_c. Output: keys (s, d) for all d != s.
    Flow (s,d) on physical (u,v) = flow (s,d_c) on (u_c,v_c) where (u_c,v_c) is d_c-view of (u,v).
    So (u,v) = transform of (u_c,v_c) under d_c -> d; iterate emap (u_c,v_c) and set (u,v) = tform_dc_to_d(u_c,v_c).
    """
    full_child_flow_dict = defaultdict(dict)
    for (s, d_c), emap in child_flow_dict_canonical_dests.items():
        if s == d_c:
            continue
        for d in my_tpuv4_symmetry.reverse_canonical_equivalence_map.get(d_c, [d_c]):
            if s == d:
                continue
            tform_dc_to_d = my_tpuv4_symmetry.calc_transform_delta(d_c, d)
            for (u_c, v_c), val in emap.items():
                u = my_tpuv4_symmetry.apply_transformation(u_c, tform_dc_to_d)
                v = my_tpuv4_symmetry.apply_transformation(v_c, tform_dc_to_d)
                full_child_flow_dict[(s, d)][(u, v)] = val
    return _sort_child_flow_dict(full_child_flow_dict)

def expand_child_flow_dict_symmetry(child_flow_dict_canonical, my_tpuv4_symmetry, n_nodes):
    """
    Expand child_flow_dict from canonical (sc, dc) only to full (s, d) for all s,d.
    For each (s,d), (sc,dc) = canonical commodity; flow (s,d) on (u,v) = flow (sc,dc) on (u',v')
    where (u',v') = transform of (u,v) under s->sc; so (u,v) = reverse_transform((u',v')).
    Must include canonical pairs (s=sc) so verify_symmetry_for_child_flow_dict can look up (sc,dc).
    """
    full_child_flow_dict = defaultdict(dict)
    for (sc, dc), emap in child_flow_dict_canonical.items():
        if sc == dc:
            continue
        # All nodes s that map to sc (including sc itself) get flow on transformed edge
        for s in my_tpuv4_symmetry.reverse_canonical_equivalence_map.get(sc, [sc]):
            # Transform from canonical (sc) view to s view: sc -> s
            sc_to_s_tform = my_tpuv4_symmetry.calc_transform_delta(sc, s)
            d = my_tpuv4_symmetry.apply_transformation(dc, sc_to_s_tform)
            if s == d:
                continue
            for (u_c, v_c), val in emap.items():
                u = my_tpuv4_symmetry.apply_transformation(u_c, sc_to_s_tform)
                v = my_tpuv4_symmetry.apply_transformation(v_c, sc_to_s_tform)
                full_child_flow_dict[(s, d)][(u, v)] = val
    return _sort_child_flow_dict(full_child_flow_dict)

def verify_symmetry_for_child_flow_dict(child_flow_dict, my_tpuv4_symmetry, eps=1e-9):
    """Check that each (s,d) flow matches the canonical (sc,dc) flow under the symmetry transform."""
    for (s, d), sd_flows in child_flow_dict.items():
        sc, sc_tform = my_tpuv4_symmetry.get_canonical_equivalent(s)
        dc = my_tpuv4_symmetry.apply_transformation(d, sc_tform)
        sc_dc_flows = child_flow_dict.get((sc, dc), {})
        for (u, v), val in sd_flows.items():
            uc = my_tpuv4_symmetry.apply_transformation(u, sc_tform)
            vc = my_tpuv4_symmetry.apply_transformation(v, sc_tform)
            canon_val = sc_dc_flows.get((uc, vc), 0.0)
            assert abs(val - canon_val) <= eps, (
                f"Symmetry mismatch (s,d)=({s},{d}) edge ({u},{v}) val={val} -> canonical (sc,dc)=({sc},{dc}) edge ({uc},{vc}) canon_val={canon_val}"
            )
    print("Success! Child flow dict is symmetric")

def handle_decomposed(adj_list, translate_to_xml=False, parallel_child=False, n_procs=None, n_chunks=1, n_channels=1, max_epochs=None, solver_params=None,
                     symmetric=False, my_tpuv4_symmetry=None):
    if symmetric and my_tpuv4_symmetry is not None:
        canonical_sources = my_tpuv4_symmetry.get_canonical_nodes()
        max_thru, fprime_vals = decomposed_link_based_mcf_master_sym(
            adj_list, canonical_sources, my_tpuv4_symmetry, solver_params=solver_params
        )
        if max_thru is None:
            return None, None, None
        n_nodes = len(adj_list)
        max_thru, fprime_vals, child_flow_dict_canonical_dests = decomposed_link_based_mcf_sym_full_children(
            adj_list,
            canonical_sources=canonical_sources,
            my_tpuv4_symmetry=my_tpuv4_symmetry,
            max_thru=max_thru,
            fprime_vals=fprime_vals,
            solver_params=solver_params,
            parallel_child=parallel_child,
            n_procs=n_procs,
        )
        child_flow_dict = expand_child_flow_dict_destinations_symmetry(
            child_flow_dict_canonical_dests, my_tpuv4_symmetry, n_nodes
        )
        verify_symmetry_for_child_flow_dict(child_flow_dict, my_tpuv4_symmetry)
    else:
        max_thru, fprime_vals, child_flow_dict = decomposed_link_based_mcf(adj_list, parallel_child=parallel_child, n_procs=n_procs, solver_params=solver_params)

    eps = 1e-9
    print(f"max_thru = {max_thru}")
    # print(f"fprime_vals:")
    # n_nodes = len(adj_list)
    # for s, s_flows in fprime_vals.items():
    #     print(f"\tsource {s}:")
    #     for (u,v), val in s_flows.items():
    #         if val < eps:
    #             continue
    #         print(f"\t\tedge {(u,v)} : flow {val}")
    # print(f"child_flow_dict:")
    # for (s,d), sd_flows in child_flow_dict.items():
    #     print(f"\tsrc, dest {s}->{d}:")
    #     for (u,v), val in sd_flows.items():
    #         val = child_flow_dict[(s,d)][(u,v)]
    #         if val < eps:
    #             continue
    #         print(f"\t\tedge {(u,v)} : flow {val}")

    # Verify final capacity constraints
    if not verify_final_capacity(adj_list, child_flow_dict):
        print("WARNING: Final child_flow_dict violates capacity constraints")

    verify_decomposed(adj_list, child_flow_dict)

    if not translate_to_xml:
        return max_thru, None, None

    eps = max(1e-12, 1e-9 * max_thru)
    
    epochs, meta = compile_decomposed_to_link_epochs(adj_list=adj_list, max_thru=max_thru, child_flow_dict=child_flow_dict, n_chunks=n_chunks, n_channels=n_channels, max_epochs=max_epochs, eps=eps )

    print(f"epochs: {epochs}")
    # print(f"meta: {meta}")

    return max_thru, epochs, meta

def main():

    parser = argparse.ArgumentParser(description="Compute all to all schedule(s) according to 'Efficiet all-to-all...' (Basu et. al. HPDC 2024)")
    parser.add_argument("--topology",type=str,help=".map file to evaluate",default="files/map_files/example_6r_25ll.map")
    parser.add_argument("--allpath_list",'-apl',type=str,help="shortcut or override path creation")
    parser.add_argument("--algorithm",'-alg',type=str,help='MCF variant',choices=["pMCF","link","tsMCF","decomp"],default="pMCF")
    parser.add_argument("--max_epochs", type=int, default=None,help="Max number of epochs. If not set then diameter used. Default: 1")
    parser.add_argument("--n_chunks",'-c', type=int, default=1,help="Number of equal-sized chunks per flow to quantize. Default: 1")
    parser.add_argument("--xml", type=str,help="Output XML file path. If not present than no XML generated")
    parser.add_argument('--n_channels', type=int, default=1,help='Number of channels to encode in the XML')
    parser.add_argument("--uniform_capacity", action="store_true",help="Enforce uniform edge capacity")
    parser.add_argument('--n_procs', type=int, default=1,help='Number of processes for children in decomposed MCF')

    # symmetry (for decomp only)
    parser.add_argument('--symmetric', action='store_true', help='Use symmetry: canonical sources/commodities only, then expand')
    parser.add_argument('--mc_dims', nargs=3, type=int, metavar=('MCX', 'MCY', 'MCZ'), help='Mega-cube dimensions for symmetry (required if --symmetric)')
    parser.add_argument('--xyzc_dims', nargs=4, type=int, metavar=('X', 'Y', 'Z', 'C'), help='Global x, y, z, cube dimensions (required if --symmetric)')
    parser.add_argument('--sym_type', type=str, choices=['trans', 'refl-trans'], default='trans', help='Symmetry type (default: trans)')

    # decomp checkpoints: master only, single child, or restore all and optionally write XML
    parser.add_argument('--checkpoint_master', action='store_true', help='Run master only, write checkpoint, exit')
    parser.add_argument('--checkpoint_child', type=int, default=None, metavar='SOURCE', help='Load master checkpoint, run child for SOURCE, write checkpoint, exit')
    parser.add_argument('--restore_all_checkpoints', action='store_true', help='Load master + all child checkpoints, reconstruct child_flow_dict (expand if symmetric), optionally write XML')

    # direct Gurobi solver params
    parser.add_argument('--time_limit',type=int,help='TimeLimit: time limit in minutes. Default: inf, min: 0')
    parser.add_argument('--threads', type=int, default=32,help='Threads: number of threads. Default: 32, min: 0')
    parser.add_argument('--concurrent_mip',type=int,help='ConcurrentMIP: number of concurrent MIP solvers. Default: 0 (auto), min: 0')
    parser.add_argument('--heuristic_ratio',type=float,help='Heuristics: heuristic effort [0,1]. Default: 0.05, range: [0, 1]')
    parser.add_argument('--mip_focus',type=int,help='MIPFocus: MIP solution focus. Default: 0, range: 0-3 (0=balanced, 1=feasible, 2=optimality, 3=bound)')
    parser.add_argument('--symmetry_detection',type=int,help='Symmetry: symmetry detection. Default: -1 (auto), range: -1 to 2 (0=off, 1=conservative, 2=aggressive)')
    parser.add_argument('--barrier_iter_limit',type=int,help='BarIterLimit: barrier iteration limit. Default: 1000, min: 0')
    parser.add_argument('--iter_limit',type=int,help='IterationLimit: simplex iteration limit. Default: inf, min: 0')
    parser.add_argument('--cut_passes',type=int,help='CutPasses: cutting plane pass limit. Default: -1 (auto), min: -1')
    parser.add_argument('--method',type=int,help='Method: LP algorithm. Default: -1 (auto), range: -1 to 5 (0=primal, 1=dual, 2=barrier, 3=concurrent, 4/5=deterministic)')
    parser.add_argument('--node_method',type=int,help='NodeMethod: MIP node LP method. Default: -1 (auto), range: -1 to 2 (0=primal, 1=dual, 2=barrier)')
    parser.add_argument('--crossover',type=int,help='Crossover: barrier crossover strategy. Default: -1 (auto), range: -1 to 2 (0=none, 1=conservative, 2=aggressive)')
    parser.add_argument('--crossover_basis',type=int,help='CrossoverBasis: crossover basis type. Default: 0, range: 0-2')
    parser.add_argument('--no_rel_heur_time',type=int,help='NoRelHeurTime: time limit for no-relaxation heuristic (sec). Default: -1 (auto), min: -1')
    parser.add_argument('--presolve',type=int,help='Presolve: presolve aggressiveness. Default: -1 (auto), range: -1 to 2 (0=off, 1=conservative, 2=aggressive)')
    parser.add_argument('--presparsify',type=int,help='PreSparsify: presolve sparsify. Default: -1 (auto), range: -1 to 2')
    parser.add_argument('--cuts',type=int,help='Cuts: cutting plane aggressiveness. Default: -1 (auto), range: -1 to 3')
    parser.add_argument('--scale_flag',type=int,help='ScaleFlag: scaling. Default: -1 (auto), range: -1 to 3')
    parser.add_argument('--feas_tol',type=float,help='FeasibilityTol: constraint feasibility tolerance. Default: 1e-6, range: [1e-9, 1e-2]')
    parser.add_argument('--opt_tol',type=float,help='OptimalityTol: dual feasibility / reduced-cost tolerance. Default: 1e-6, range: [1e-9, 1e-2]')
    parser.add_argument('--bar_conv_tol',type=float,help='BarConvTol: barrier convergence tolerance. Default: 1e-8, range: [1e-12, 1e-2]')
    parser.add_argument('--markowitz_tol',type=float,help='MarkowitzTol: simplex pivot tolerance (numerical stability). Default: 0.0078125, range: [1e-4, 0.999]')
    parser.add_argument('--psd_tol',type=float,help='PSDTol: positive semidefinite tolerance (barrier/QP). Default: 1e-6, range: [1e-9, 1e-2]')
    parser.add_argument('--int_feas_tol',type=float,help='IntFeasTol: integer feasibility tolerance (MIP). Default: 1e-5, range: [1e-9, 1e-1]')
    parser.add_argument('--mip_gap',type=float,help='MIPGap: relative MIP optimality gap tolerance. Default: 1e-4, range: [0, inf)')
    parser.add_argument('--numeric_focus',type=int,choices=[0,1,2,3],help='NumericFocus: numerical care level. Default: 0, range: 0-3 (1=conservative, 2=aggressive, 3=very aggressive)')
    parser.add_argument('--dual_reductions',type=int,choices=[0,1],help='DualReductions: enable dual reductions. Default: 1, values: 0=off, 1=on (0 can help with numerical issues)')
    parser.add_argument('--degen_moves',type=int,help='DegenMoves: degenerate simplex moves. Default: -1 (auto), range: -1 to 3')
    parser.add_argument('--write_presolved',action='store_true',help='presolve and write (presolved) model out as multiple/all formats')
    parser.add_argument('--read_presolved',type=str,help='read a presolved model of given name')
    parser.add_argument('--predual',type=int,help='PreDual: presolve dualization. Default: -1 (auto), range: -1 to 2')
    parser.add_argument('--output_flag',type=int,default=1,help='OutputFlag: solver log verbosity. Default: 1, values: 0=off, 1=on')

    args = parser.parse_args()

    map_filename = args.topology
    alg = args.algorithm
    xml_out_path = args.xml
    n_chunks = args.n_chunks
    n_channels = args.n_channels
    max_epochs = args.max_epochs
    uniform_capacity = args.uniform_capacity
    n_procs = args.n_procs
    symmetric = getattr(args, 'symmetric', False)

    if symmetric:
        if alg not in ('decomp', 'pMCF'):
            print('--symmetric requires --algorithm decomp or pMCF; forcing algorithm=decomp')
            alg = 'decomp'
        if not getattr(args, 'mc_dims', None) or not getattr(args, 'xyzc_dims', None):
            print('--symmetric requires --mc_dims <mcx> <mcy> <mcz> and --xyzc_dims <x> <y> <z> <c>')
            sys.exit(1)
        if TPUv4_Symmetry is None:
            print('tpuv4_symmetry module not found; cannot use --symmetric')
            sys.exit(1)

    parallelize = True if n_procs > 1 else False
    translate_to_xml = True if xml_out_path is not None else False

    solver_params = setup_solver_params(args)

    print(f"solver_params={solver_params}")

    # TODO solver params

    adj_mat, adj_list = ingest_map(map_filename, uniform_capacity)
    n_nodes = len(adj_list)

    print(f"Completed topology ingestion")

    my_tpuv4_symmetry = None
    if symmetric:
        xyzc_dims = tuple(args.xyzc_dims)
        mc_dims = tuple(args.mc_dims)
        sym_type = getattr(args, 'sym_type', 'trans')
        (x_dim, y_dim, z_dim, cube_dim) = xyzc_dims
        if (x_dim % 2 > 0 or y_dim % 2 > 0 or z_dim % 2 > 0) and sym_type != 'trans':
            print("Dimensions require 'trans' symmetry. Exiting...")
            sys.exit(1)
        my_tpuv4_symmetry = TPUv4_Symmetry(xyzc_dims, mc_dims=mc_dims, sym_type=sym_type)
        my_tpuv4_symmetry.verify_symmetry_for_topology(adj_mat)
        print(f"Symmetry verified; canonical sources: {len(my_tpuv4_symmetry.get_canonical_nodes())}")

    # Topology basename for checkpoint filenames (no path, no extension)
    topology_basename = os.path.splitext(os.path.basename(map_filename))[0]
    os.makedirs(MCF_DECOMP_CHECKPOINT_DIR, exist_ok=True)

    # --- checkpoint_master: run master only, write pkl, exit ---
    if getattr(args, 'checkpoint_master', False):
        if symmetric and my_tpuv4_symmetry is not None:
            canonical_sources = my_tpuv4_symmetry.get_canonical_nodes()
            max_thru, fprime_vals = decomposed_link_based_mcf_master_sym(
                adj_list, canonical_sources, my_tpuv4_symmetry, solver_params=solver_params
            )
            payload = {
                "max_thru": max_thru,
                "fprime_vals": dict(fprime_vals),
                "n_nodes": n_nodes,
                "symmetric": True,
                "topology_basename": topology_basename,
                "canonical_sources": list(canonical_sources),
                "xyzc_dims": tuple(args.xyzc_dims),
                "mc_dims": tuple(args.mc_dims),
                "sym_type": getattr(args, 'sym_type', 'trans'),
            }
        else:
            max_thru, fprime_vals = decomposed_link_based_mcf_master(adj_list, solver_params=solver_params)
            payload = {
                "max_thru": max_thru,
                "fprime_vals": dict(fprime_vals),
                "n_nodes": n_nodes,
                "symmetric": False,
                "topology_basename": topology_basename,
            }
        if max_thru is None:
            print("Master solve failed; not checkpointing.")
            sys.exit(1)
        chkpt_path = os.path.join(MCF_DECOMP_CHECKPOINT_DIR, f"{topology_basename}_master_chkpt.pkl")
        with open(chkpt_path, "wb") as f:
            pickle.dump(payload, f)
        print(f"Wrote master checkpoint: {chkpt_path}")
        sys.exit(0)

    # --- checkpoint_child <int>: load master, run child for SOURCE, write pkl, exit ---
    if getattr(args, 'checkpoint_child', None) is not None:
        source = args.checkpoint_child
        master_path = os.path.join(MCF_DECOMP_CHECKPOINT_DIR, f"{topology_basename}_master_chkpt.pkl")
        if not os.path.isfile(master_path):
            print(f"Master checkpoint not found: {master_path}")
            sys.exit(1)
        with open(master_path, "rb") as f:
            master_payload = pickle.load(f)
        max_thru = master_payload["max_thru"]
        fprime_vals = master_payload["fprime_vals"]
        n_nodes_chk = master_payload["n_nodes"]
        if n_nodes_chk != n_nodes:
            print(f"Checkpoint n_nodes={n_nodes_chk} != topology n_nodes={n_nodes}")
            sys.exit(1)
        if source < 0 or source >= n_nodes:
            print(f"checkpoint_child source={source} out of range [0, {n_nodes})")
            sys.exit(1)
        symmetric_chk = master_payload.get("symmetric", False)
        if symmetric_chk:
            canonical_sources = master_payload["canonical_sources"]
            xyzc_dims = master_payload["xyzc_dims"]
            mc_dims = master_payload["mc_dims"]
            sym_type = master_payload.get("sym_type", "trans")
            my_tpuv4_symmetry_chk = TPUv4_Symmetry(xyzc_dims, mc_dims=mc_dims, sym_type=sym_type)
            fprime_full = expand_fprime_symmetry(fprime_vals, my_tpuv4_symmetry_chk, adj_list)
            flows_s = decomposed_link_based_mcf_child_sym(
                adj_list, source=source, max_thru=max_thru,
                fprime_s=fprime_full.get(source, {}),
                my_tpuv4_symmetry=my_tpuv4_symmetry_chk,
                canonical_sources=canonical_sources,
                solver_params=solver_params,
            )
        else:
            flows_s = decomposed_link_based_mcf_child(
                adj_list, source=source, max_thru=max_thru,
                fprime_s=fprime_vals.get(source, {}), solver_params=solver_params,
            )
        if not flows_s:
            print(f"Child for source {source} failed.")
            sys.exit(1)
        child_payload = {"flows_s": flows_s, "source": source}
        child_path = os.path.join(MCF_DECOMP_CHECKPOINT_DIR, f"{topology_basename}_child{source}_chkpt.pkl")
        with open(child_path, "wb") as f:
            pickle.dump(child_payload, f)
        print(f"Wrote child checkpoint: {child_path}")
        sys.exit(0)

    # --- restore_all_checkpoints: load master + all children, reconstruct, optionally XML ---
    if getattr(args, 'restore_all_checkpoints', False):
        master_path = os.path.join(MCF_DECOMP_CHECKPOINT_DIR, f"{topology_basename}_master_chkpt.pkl")
        if not os.path.isfile(master_path):
            print(f"Master checkpoint not found: {master_path}")
            sys.exit(1)
        with open(master_path, "rb") as f:
            master_payload = pickle.load(f)
        max_thru = master_payload["max_thru"]
        n_nodes_chk = master_payload["n_nodes"]
        symmetric_restore = master_payload["symmetric"]
        if n_nodes_chk != n_nodes:
            print(f"Checkpoint n_nodes={n_nodes_chk} != topology n_nodes={n_nodes}")
            sys.exit(1)
        # Symmetric: we now run all N children (canonical-dest only per child), so expect all sources
        expected_sources = list(range(n_nodes))
        child_flow_dict = defaultdict(dict)
        for s in expected_sources:
            child_path = os.path.join(MCF_DECOMP_CHECKPOINT_DIR, f"{topology_basename}_child{s}_chkpt.pkl")
            if not os.path.isfile(child_path):
                print(f"Child checkpoint not found: {child_path}")
                sys.exit(1)
            with open(child_path, "rb") as f:
                child_payload = pickle.load(f)
            for k, v in child_payload["flows_s"].items():
                child_flow_dict[k] = v
        if symmetric_restore:
            xyzc_dims = master_payload["xyzc_dims"]
            mc_dims = master_payload["mc_dims"]
            sym_type = master_payload.get("sym_type", "trans")
            my_tpuv4_symmetry_restore = TPUv4_Symmetry(xyzc_dims, mc_dims=mc_dims, sym_type=sym_type)
            child_flow_dict = expand_child_flow_dict_destinations_symmetry(dict(child_flow_dict), my_tpuv4_symmetry_restore, n_nodes)
        else:
            child_flow_dict = _sort_child_flow_dict(child_flow_dict)
        
        # Verify final capacity constraints
        if not verify_final_capacity(adj_list, child_flow_dict):
            print("WARNING: Final child_flow_dict violates capacity constraints")
        
        verify_decomposed(adj_list, child_flow_dict)
        if xml_out_path:
            eps = max(1e-12, 1e-9 * max_thru)
            epochs, meta = compile_decomposed_to_link_epochs(
                adj_list=adj_list, max_thru=max_thru, child_flow_dict=child_flow_dict,
                n_chunks=n_chunks, n_channels=n_channels, max_epochs=max_epochs, eps=eps,
            )
            write_msccl_xml_from_link_epochs(
                adj_list=adj_list, epochs=epochs, out_xml_path=xml_out_path,
                n_chunks=n_chunks, n_channels=n_channels,
                i_chunks_override=(meta.get("i_chunks") if meta else None),
                o_chunks_override=(meta.get("o_chunks") if meta else None),
                s_chunks_override=(meta.get("s_chunks") if meta else None),
            )
            verify_epochs_demand_satisfaction(adj_list, epochs, n_chunks)
            print(f'Wrote MSCCL XML to: {xml_out_path}')
            n_links = sum(len(adj_list[u]) for u in range(n_nodes))
            append_a2a_stats(topology_basename, len(epochs), n_nodes, n_links)
        else:
            print("Restored all checkpoints (no --xml, skipping write).")
        sys.exit(0)

    if alg == "pMCF":
        max_thru, epochs, meta = handle_path_based(
            adj_list,
            apl_name=args.allpath_list,
            translate_to_xml=translate_to_xml,
            n_chunks=n_chunks,
            n_channels=n_channels,
            max_epochs=max_epochs,
            solver_params=solver_params,
            symmetric=symmetric,
            my_tpuv4_symmetry=my_tpuv4_symmetry,
        )
    elif alg == "link":
        max_thru, epochs, meta = handle_link_based(
            adj_list,
            translate_to_xml=translate_to_xml,
            n_chunks=n_chunks,
            n_channels=n_channels,
            max_epochs=max_epochs,
            solver_params=solver_params,
        )
    elif alg == "tsMCF":
        max_thru, epochs, meta = handle_timestepped(
            adj_list,
            adj_mat,
            translate_to_xml=translate_to_xml,
            n_chunks=n_chunks,
            n_channels=n_channels,
            max_epochs=max_epochs,
            solver_params=solver_params,
        )
    elif alg == "decomp":
        max_thru, epochs, meta = handle_decomposed(
            adj_list,
            translate_to_xml=translate_to_xml,
            parallel_child=parallelize,
            n_procs=n_procs,
            n_chunks=n_chunks,
            n_channels=n_channels,
            max_epochs=max_epochs,
            solver_params=solver_params,
            symmetric=symmetric,
            my_tpuv4_symmetry=my_tpuv4_symmetry,
        )

    if xml_out_path:
        write_msccl_xml_from_link_epochs(
            adj_list=adj_list,
            epochs=epochs,
            out_xml_path=xml_out_path,
            n_chunks=n_chunks,
            n_channels=n_channels,
            i_chunks_override=(meta.get("i_chunks") if meta else None),
            o_chunks_override=(meta.get("o_chunks") if meta else None),
            s_chunks_override=(meta.get("s_chunks") if meta else None),
        )
        verify_epochs_demand_satisfaction(adj_list, epochs, n_chunks)
        print(f'Wrote MSCCL compiled (per-link epochs) all-to-all XML to: {xml_out_path}')
        print(f'  epochs={len(epochs)}  i_chunks=o_chunks={n_nodes*n_chunks}  s_chunks={meta["s_chunks"]}')
    if alg == "decomp" and epochs is not None:
        n_links = sum(len(adj_list[u]) for u in range(n_nodes))
        append_a2a_stats(topology_basename, len(epochs), n_nodes, n_links)


# Script Stuff
# --------------------------------------------------------------------------------

if __name__ == '__main__':

    main()