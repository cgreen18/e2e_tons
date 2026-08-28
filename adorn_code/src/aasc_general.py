#!/usr/bin/env python3
"""
Radix-only (optionally directed) symmetric topology generator (LP or MILP).

Default behavior (undirected):
  - Synthesizes a simple undirected r-regular graph on n_nodes.
  - Symmetric adjacency: adj[i][j] == adj[j][i]
  - Variables are stored only for i<j (upper triangle), but degree constraints
    count both sides.

Directed mode (--directed):
  - Synthesizes a simple directed graph with no self-loops.
  - Enforces BOTH out-degree == radix and in-degree == radix for every node
    (i.e., a directed-regular digraph).
  - Variables are stored for all ordered pairs (i,j) with i != j.
  - Output adjacency is NOT symmetrized.

Supports LP relaxation (default) or MILP (use --binary_r_map).

Map file format:
  - Text adjacency matrix with whitespace-delimited numeric entries.
  - One row per node, n_nodes entries per row.

Examples:
  Undirected LP:     python aasc_general.py --n_nodes 256 --radix 6
  Undirected MILP:   python aasc_general.py --n_nodes 256 --radix 6 --binary_r_map
  Directed MILP:     python aasc_general.py --n_nodes 256 --radix 6 --binary_r_map --directed
"""

import argparse
import os
import sys
import math

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception as e:
    print("ERROR: gurobipy is required to run this script.", file=sys.stderr)
    raise


class UpperTriMatrix:
    def __init__(self, n, init_val=None):
        """
        Store only the upper-triangular (i < j) entries of an n x n matrix.
        Each entry can hold an arbitrary Python object.

        NOTE: __getitem__ symmetrizes indices (i,j) -> (min, max).
        """
        self.n = n
        self.size = n * (n - 1) // 2
        self.data = [init_val] * self.size  # Python objects

    def _row_start(self, i):
        return i * (self.n - 1) - (i * (i - 1)) // 2

    def _index(self, i, j):
        if i == j:
            raise IndexError("Diagonal elements are not stored")
        if i > j:
            i, j = j, i
        return self._row_start(i) + (j - i - 1)

    def __getitem__(self, key):
        i, j = key
        return self.data[self._index(i, j)]

    def __setitem__(self, key, value):
        i, j = key
        self.data[self._index(i, j)] = value


class OffDiagMatrix:
    def __init__(self, n, init_val=None):
        """
        Store all off-diagonal (i != j) entries of an n x n matrix (ordered pairs).
        Each entry can hold an arbitrary Python object.

        Index mapping:
          idx(i,j) = i*(n-1) + (j if j < i else j-1)
        """
        self.n = n
        self.size = n * (n - 1)
        self.data = [init_val] * self.size

    def _index(self, i, j):
        if i == j:
            raise IndexError("Diagonal elements are not stored")
        if not (0 <= i < self.n and 0 <= j < self.n):
            raise IndexError("Index out of bounds")
        return i * (self.n - 1) + (j if j < i else (j - 1))

    def __getitem__(self, key):
        i, j = key
        return self.data[self._index(i, j)]

    def __setitem__(self, key, value):
        i, j = key
        self.data[self._index(i, j)] = value


class SymTriDict(dict):
    @staticmethod
    def _n(key):
        i, j, k = key
        return (i, j, k)

    def __getitem__(self, key):
        return super().__getitem__(self._n(key))

    def __setitem__(self, key, val):
        super().__setitem__(self._n(key), val)

    def __delitem__(self, key):
        return super().__delitem__(self._n(key))

    def __contains__(self, key):
        return super().__contains__(self._n(key))


def parse_args():
    ap = argparse.ArgumentParser(description="Generate regular graphs (radix-only), LP or MILP.")
    ap.add_argument("--n_nodes", type=int, required=True, help="Number of nodes (n).")
    ap.add_argument("--radix", type=int, required=True, help="Exact degree per node (k).")

    ap.add_argument("--directed", action="store_true",
                    help="Generate a directed regular digraph (in-degree == out-degree == radix).")

    ap.add_argument("--binary_r_map", action="store_true",
                    help="Use MILP with binary edges. Default is LP relaxation with 0<=x<=1.")
    ap.add_argument("--out", type=str, default=None,
                    help="Output .map path (adjacency matrix). Default: ./radix_<n>_r<r>_{lp|milp}_{undir|dir}.map")

    ap.add_argument("--start_map", type=str, default=None,
                    help="Optional initial solution adjacency matrix (.map). Values are used as starts for x(i,j).")
    ap.add_argument("--edge_threshold", type=float, default=0.5,
                    help="Threshold to interpret fractional edges when printing degree stats (default: 0.5).")

    ap.add_argument("--objective", choices=["random", "none"], default="random",
                    help="Objective used to break symmetry: random (default) or none (feasibility only).")
    ap.add_argument("--seed", type=int, default=1, help="RNG seed (used for random objective).")

    ap.add_argument("--time_limit", type=float, default=None, help="Gurobi time limit in seconds.")
    ap.add_argument("--mip_gap", type=float, default=None, help="Gurobi MIPGap (MILP only).")
    ap.add_argument("--threads", type=int, default=None, help="Gurobi Threads.")
    ap.add_argument("--logfile", type=str, default=None, help="Optional Gurobi log file path.")
    ap.add_argument("--verbose", action="store_true", help="Verbose prints.")
    return ap.parse_args()


def ingest_adj_mat(path, n, assert_square=True):
    mat = []
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            row = [float(x) for x in line.strip().split()]
            mat.append(row)

    if assert_square:
        if len(mat) != n:
            raise ValueError(f"start_map has {len(mat)} rows, expected {n}")
        for r, row in enumerate(mat):
            if len(row) != n:
                raise ValueError(f"start_map row {r} has {len(row)} cols, expected {n}")

    return mat


def output_adj_mat(adj_mat, n, out_path, directed=False, assert_binary=False, quiet=False):
    """
    adj_mat:
      - undirected: UpperTriMatrix containing (i<j) values
      - directed:   OffDiagMatrix containing (i!=j) values
    """
    out_lines = []
    for sr in range(n):
        this_line = []
        for dr in range(n):
            if sr == dr:
                val = 0
            else:
                if directed:
                    val = adj_mat[sr, dr]
                else:
                    # symmetrize for output
                    val = adj_mat[(sr, dr)] if sr < dr else adj_mat[(dr, sr)]
            val = int(val)
            if assert_binary:
                val = min(1, round(val))
            this_line.append(f"{val} ")
        this_line.append("\n")
        out_lines.append(this_line)

    with open(out_path, "w+") as of:
        for line_list in out_lines:
            of.write("".join(line_list))

    if not quiet:
        print(f"Wrote out adj mat to {out_path}")



def _prepare_milp_dump_callback(model, var_r_map, n, directed, base_out_path):
    """
    Installs a MIP callback that dumps the *current incumbent* adjacency matrix to disk
    whenever Gurobi finds a new incumbent (MIPSOL callback).

    The callback prints:
      - incumbent objective value at that node
      - path of the dumped adjacency matrix

    Dumps are written under:
        <dirname(base_out_path)>/mip_dumps_<basename_without_ext>/

    Notes:
      - This runs ONLY for MILP (binary edges). For LP there is no MIP callback.
      - The dumped adjacency matrix is derived from the incumbent solution, with
        entries rounded to {0,1} via threshold 0.5.
    """
    dump_root = os.path.dirname(os.path.abspath(base_out_path))
    base = os.path.splitext(os.path.basename(base_out_path))[0]
    dump_dir = os.path.join(dump_root, f"mip_dumps_{base}")
    os.makedirs(dump_dir, exist_ok=True)

    # Build a stable list of variables to query in the callback
    edge_vars = []
    if directed:
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                edge_vars.append((i, j, var_r_map[(i, j)]))
    else:
        for i in range(n):
            for j in range(i + 1, n):
                edge_vars.append((i, j, var_r_map[(i, j)]))

    model._dump_cb_directed = bool(directed)
    model._dump_cb_n = int(n)
    model._dump_cb_dir = dump_dir
    model._dump_cb_base = base
    model._dump_cb_edge_vars = edge_vars
    model._dump_cb_count = 0

    def _cb(m, where):
        if where == GRB.Callback.MIPSOL:
            try:
                obj = m.cbGet(GRB.Callback.MIPSOL_OBJ)
            except Exception:
                obj = None

            vars_only = [t[2] for t in m._dump_cb_edge_vars]
            vals = m.cbGetSolution(vars_only)

            # Reconstruct adjacency container with 0/1 ints
            if m._dump_cb_directed:
                adj = OffDiagMatrix(m._dump_cb_n, init_val=0)
                for (i, j, _), v in zip(m._dump_cb_edge_vars, vals):
                    adj[(i, j)] = 1 if float(v) >= 0.5 else 0
            else:
                adj = UpperTriMatrix(m._dump_cb_n, init_val=0)
                for (i, j, _), v in zip(m._dump_cb_edge_vars, vals):
                    adj[(i, j)] = 1 if float(v) >= 0.5 else 0

            m._dump_cb_count += 1
            out_path = os.path.join(m._dump_cb_dir, f"{m._dump_cb_base}_inc_{m._dump_cb_count:05d}.map")

            # Write without extra prints (we print our own line below)
            output_adj_mat(adj, m._dump_cb_n, out_path,
                           directed=m._dump_cb_directed,
                           assert_binary=True,
                           quiet=True)

            if obj is None:
                print(f"[MIPSOL] wrote adjacency to {out_path}")
            else:
                print(f"[MIPSOL] obj={obj} wrote adjacency to {out_path}")
            try:
                sys.stdout.flush()
            except Exception:
                pass

    return _cb
def build(n, r, binary, start_mat, args):
    if r < 0:
        raise ValueError("radix must be >= 0")
    if n <= 0:
        raise ValueError("n_nodes must be > 0")
    if r >= n:
        raise ValueError("radix must be <= n_nodes-1")
    if (not args.directed) and ((n * r) % 2 != 0):
        # Necessary for existence of an undirected r-regular simple graph.
        raise ValueError(f"No simple undirected r-regular graph exists when n*r is odd (n={n}, r={r}).")

    model = gp.Model("radix_only")

    if args.logfile:
        model.Params.LogFile = args.logfile
    if args.time_limit is not None:
        model.Params.TimeLimit = float(args.time_limit)
    if args.threads is not None:
        model.Params.Threads = int(args.threads)
    if binary and args.mip_gap is not None:
        model.Params.MIPGap = float(args.mip_gap)
    if not args.verbose:
        model.Params.OutputFlag = 0

    # model.Params.Crossover = 0
    # model.Params.Method = 2
    model.Params.MIPFocus = 1

    # <= (n/r)*log_{r}(n)
    # unity_wye_ub = 1 / ((n * math.log(n, r) ) / r)
    # tri_ineq_wye_ub = unity_wye_ub / n

    # should be less than or equal to sparsest cut
    # cant be more than (# links) / (bisection demand)
    unity_wye_ub = 1.0 #((r/2) / (n))
    tri_ineq_wye_ub = unity_wye_ub # / n


    print(f"Assuming unity wye upper bound of {unity_wye_ub}")
    print(f"Assuming tri ineq wye upper bound of {tri_ineq_wye_ub}")



    vtype = GRB.BINARY if binary else GRB.CONTINUOUS

    # Edge variables x
    if args.directed:
        # Ordered pairs (i,j), i != j
        var_r_map = OffDiagMatrix(n)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                var_r_map[(i, j)] = model.addVar(lb=0.0, ub=1.0, vtype=vtype, name=f"x_{i}_{j}")
    else:
        # Upper-triangular x[i,j] for i<j
        var_r_map = UpperTriMatrix(n)
        for i in range(n):
            for j in range(i + 1, n):
                var_r_map[(i, j)] = model.addVar(lb=0.0, ub=1.0, vtype=vtype, name=f"x_{i}_{j}")

    # These triangle/wye variables and constraints are part of your existing formulation.
    # They are kept structurally identical; in directed mode, we simply apply A_transpose
    # across all ordered (a,b) with a!=b instead of only a<b.
    var_tri_ineq_wyes = SymTriDict()
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if i == k or j == k:
                    continue
                var_tri_ineq_wyes[(i, j, k)] = model.addVar(lb=0.0, ub=tri_ineq_wye_ub, name=f"t_{i}_{j}_{k}")

    var_unity_wye = model.addVar(lb=0.0, ub=unity_wye_ub, name="u")
    if args.verbose:
        print("Completed variables")

    # Degree constraints
    if args.directed:
        # out-degree == r, in-degree == r
        for i in range(n):
            expr_out = gp.quicksum(var_r_map[(i, j)] for j in range(n) if j != i)
            model.addConstr(expr_out == r, name=f"outdeg_{i}")
        for j in range(n):
            expr_in = gp.quicksum(var_r_map[(i, j)] for i in range(n) if i != j)
            model.addConstr(expr_in == r, name=f"indeg_{j}")
    else:
        # undirected degree == r
        for i in range(n):
            expr = gp.quicksum(var_r_map[(i, j)] for j in range(n) if j != i)
            model.addConstr(expr == r, name=f"deg_{i}")

    # A_transpose constraints
    if args.directed:
        pair_iter = ((a, b) for a in range(n) for b in range(n) if a != b)
    else:
        pair_iter = ((a, b) for a in range(n) for b in range(a + 1, n))

    for a, b in pair_iter:
        pos_lhs_vars = [var_unity_wye]
        neg_lhs_vars = [var_r_map[(a, b)]]

        # i->k
        # ----
        pos_lhs_vars += [var_tri_ineq_wyes[(a, j, b)] for j in range(n) if j != a and j != b]
        pos_lhs_vars += [var_tri_ineq_wyes[(i, b, a)] for i in range(n) if i != a and i != b]
        neg_lhs_vars += [var_tri_ineq_wyes[(a, b, k)] for k in range(n) if k != a and k != b]

        model.addConstr(gp.quicksum(pos_lhs_vars) - gp.quicksum(neg_lhs_vars) <= 0, name=f"At_{a}_{b}")

    # Optional starts
    if start_mat is not None:
        if args.directed:
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    val = float(start_mat[i][j])
                    if val < 0.0:
                        val = 0.0
                    if val > 1.0:
                        val = 1.0
                    var_r_map[(i, j)].Start = val
        else:
            for i in range(n):
                for j in range(i + 1, n):
                    val = float(start_mat[i][j])
                    if val < 0.0:
                        val = 0.0
                    if val > 1.0:
                        val = 1.0
                    var_r_map[(i, j)].Start = val

    if args.verbose:
        print("completed constraints")

    wye_obj_expr = gp.LinExpr()
    wye_obj_expr += var_unity_wye
    model.setObjective(wye_obj_expr, GRB.MAXIMIZE)

    return model, var_r_map


def solve(model, callback=None):
    if callback is None:
        model.optimize()
    else:
        model.optimize(callback)
    print(f"Gurobi ended with status {model.Status}")
    try:
        objval = model.objVal
        print(f"Solve w/ obj {objval}")
    except Exception:
        pass
    return model


def score_solution(model, var_r_map, n, directed=False):
    if directed:
        adj = OffDiagMatrix(n, init_val=0)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                var = var_r_map[(i, j)]
                adj[(i, j)] = float(var.X)
        return adj

    # undirected
    adj = UpperTriMatrix(n, init_val=0)
    for i in range(n):
        for j in range(i + 1, n):
            var = var_r_map[(i, j)]
            adj[(i, j)] = float(var.X)
    return adj


def main():
    args = parse_args()

    n = int(args.n_nodes)
    r = int(args.radix)

    start_mat = None
    if args.start_map:
        start_mat = ingest_adj_mat(args.start_map, n)

    # Greedy rounding iterations:
    #   - undirected: total edges = n*r/2
    #   - directed:   total arcs  = n*r   (out-degree == r)
    n_iters = (n * r) if args.directed else ((n * r) // 2)
    if bool(args.binary_r_map):
        n_iters = 1

    chosen_conns = set()
    model, var_r_map = build(n, r, bool(args.binary_r_map), start_mat, args)
    # model.write("general.lp")

    mode = "milp" if args.binary_r_map else "lp"
    dir_tag = "dir" if args.directed else "undir"
    out_path = args.out
    if out_path is None:
        out_path = os.path.abspath(f"radix_{n}_r{r}_{mode}_{dir_tag}.map")
    else:
        out_path = os.path.abspath(out_path)

    # MILP: install a callback that dumps the incumbent adjacency matrix
    cb = None
    if bool(args.binary_r_map):
        cb = _prepare_milp_dump_callback(model, var_r_map, n, directed=args.directed, base_out_path=out_path)

    for iter_val in range(n_iters):
        print("")
        print(f"Iteration : {iter_val} / {n_iters}")

        model = solve(model, callback=cb)
        adj = score_solution(model, var_r_map, n, directed=args.directed)

        highest_score = -1.0
        best_conn = None

        if args.directed:
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if (i, j) in chosen_conns:
                        continue
                    val = adj[(i, j)]
                    if val > highest_score:
                        highest_score = val
                        best_conn = (i, j)
        else:
            for i in range(n):
                for j in range(i + 1, n):
                    if (i, j) in chosen_conns:
                        continue
                    val = adj[(i, j)]
                    if val > highest_score:
                        highest_score = val
                        best_conn = (i, j)

        if best_conn is None:
            raise RuntimeError("Failed to select a best connection (no candidates left?)")

        print(f"Chose conn {best_conn} w/ val {highest_score}")

        a, b = best_conn
        chosen_conns.add(best_conn)

        if args.directed:
            # Fix only the chosen arc
            adj[(a, b)] = 1
            model.addConstr(var_r_map[(a, b)] == 1)
        else:
            # Fix undirected edge (a,b)
            adj[(min(a, b), max(a, b))] = 1
            model.addConstr(var_r_map[(min(a, b), max(a, b))] == 1)

        model.update()

    output_adj_mat(adj, n, out_path, directed=args.directed, assert_binary=bool(args.binary_r_map))


if __name__ == "__main__":
    main()
