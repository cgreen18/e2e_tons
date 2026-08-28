#!/usr/bin/env python3
import argparse
import os
import random
import sys

def sinkhorn_doubly_stochastic_zero_diag(n, seed=1, max_iter=20000, tol=1e-12):
    """
    Generate an n x n nonnegative matrix with diagonal exactly 0 that is
    (approximately) doubly stochastic: row sums ~1 and col sums ~1.

    Uses alternating row/column scaling (Sinkhorn-Knopp) while clamping diag to 0.
    """
    if n <= 1:
        raise ValueError("n must be >= 2")

    rng = random.Random(seed)

    # Initialize with strictly positive off-diagonal entries
    # (avoid zeros off-diagonal so scaling stays well-defined)
    M = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                M[i][j] = 0.0
            else:
                # log-uniform-ish spread helps avoid conditioning issues
                # but any positive distribution works
                M[i][j] = 10.0 ** rng.uniform(-2.0, 2.0)

    # Sinkhorn iterations: scale rows then columns, keep diagonal clamped to 0
    for it in range(max_iter):
        # Row scaling to make each row sum to 1
        max_row_err = 0.0
        for i in range(n):
            s = sum(M[i])
            if s <= 0.0:
                raise RuntimeError(f"Row {i} sum is nonpositive; cannot scale.")
            inv = 1.0 / s
            for j in range(n):
                if i != j:
                    M[i][j] *= inv
            # diag stays 0
            row_sum = sum(M[i])
            max_row_err = max(max_row_err, abs(row_sum - 1.0))

        # Column scaling to make each column sum to 1
        max_col_err = 0.0
        col_sums = [0.0] * n
        for j in range(n):
            s = 0.0
            for i in range(n):
                s += M[i][j]
            col_sums[j] = s

        for j in range(n):
            s = col_sums[j]
            if s <= 0.0:
                raise RuntimeError(f"Column {j} sum is nonpositive; cannot scale.")
            inv = 1.0 / s
            for i in range(n):
                if i != j:
                    M[i][j] *= inv
            # diag stays 0

        # Recompute errors after column scaling
        for i in range(n):
            max_row_err = max(max_row_err, abs(sum(M[i]) - 1.0))
        for j in range(n):
            s = 0.0
            for i in range(n):
                s += M[i][j]
            max_col_err = max(max_col_err, abs(s - 1.0))

        if max(max_row_err, max_col_err) <= tol:
            return M, it + 1

    return M, max_iter


def write_matrix_space_delimited(M, out_path):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        for row in M:
            f.write(" ".join(f"{x:.17g}" for x in row) + "\n")


def main():
    ap = argparse.ArgumentParser(
        description="Generate an n x n doubly-stochastic demand matrix with zero diagonal."
    )
    ap.add_argument("-n", type=int, required=True, help="Number of nodes (n >= 2).")
    ap.add_argument("-o", type=str, required=True, help="Output file path.")
    ap.add_argument("--seed", type=int, default=1, help="RNG seed (default: 1).")
    ap.add_argument("--tol", type=float, default=1e-12, help="Convergence tolerance (default: 1e-12).")
    ap.add_argument("--max_iter", type=int, default=20000, help="Maximum Sinkhorn iterations (default: 20000).")
    ap.add_argument("--verify", action="store_true", help="Print max row/col sum error to stderr.")
    args = ap.parse_args()

    M, iters = sinkhorn_doubly_stochastic_zero_diag(
        args.n, seed=args.seed, max_iter=args.max_iter, tol=args.tol
    )

    write_matrix_space_delimited(M, args.o)

    if args.verify:
        row_err = max(abs(sum(r) - 1.0) for r in M)
        col_err = 0.0
        n = args.n
        for j in range(n):
            s = 0.0
            for i in range(n):
                s += M[i][j]
            col_err = max(col_err, abs(s - 1.0))
        # Diagonal check
        diag_max = max(abs(M[i][i]) for i in range(n))
        print(
            f"Wrote {args.n}x{args.n} matrix to {args.o} "
            f"(iters={iters}, max_row_err={row_err:.3e}, max_col_err={col_err:.3e}, max_diag={diag_max:.3e})",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
