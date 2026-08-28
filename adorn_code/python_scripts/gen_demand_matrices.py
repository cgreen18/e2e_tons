#!/usr/bin/env python3
import argparse
import math
import os
import random
import sys

def is_power_of_two(n):
    return n > 0 and (n & (n - 1)) == 0

def bits_floor_log2(n):
    return int(math.floor(math.log2(n)))

def int_to_bits(x, bits):
    # LSB-first list, matching boost::dynamic_bitset indexing in your C++
    return [(x >> i) & 1 for i in range(bits)]

def bits_to_int(b):
    x = 0
    for i, bit in enumerate(b):
        if bit:
            x |= (1 << i)
    return x

def perm_bitcomplement(n):
    bits = bits_floor_log2(n)
    p = []
    for src in range(n):
        b = int_to_bits(src, bits)
        db = [1 - bi for bi in b]
        dest = bits_to_int(db) % n
        p.append(dest)
    return p

def perm_bitreverse(n):
    bits = bits_floor_log2(n)
    p = []
    for src in range(n):
        b = int_to_bits(src, bits)
        db = [0] * bits
        for i in range(bits):
            db[i] = b[bits - 1 - i]
        dest = bits_to_int(db) % n
        p.append(dest)
    return p

def perm_bitshuffle(n):
    bits = bits_floor_log2(n)
    p = []
    for src in range(n):
        b = int_to_bits(src, bits)
        last = b[bits - 1]
        db = [0] * bits
        # dest_binary = src_binary << 1; dest_binary[0] = last_bit;
        db[0] = last
        for i in range(1, bits):
            db[i] = b[i - 1]
        dest = bits_to_int(db) % n
        p.append(dest)
    return p

def perm_bittranspose(n):
    bits = bits_floor_log2(n)
    shift = bits // 2
    p = []
    for src in range(n):
        b = int_to_bits(src, bits)
        db = [0] * bits
        for i in range(bits):
            db[i] = b[(i + shift) % bits]
        dest = bits_to_int(db) % n
        p.append(dest)
    return p

def repair_fixed_points_to_derangement(p):
    """
    Given a permutation p over [0..n-1], repair fixed points (p[i]==i) by
    cycling their images among themselves. Result is still a permutation,
    and removes all fixed points as long as there are 0 or >=2 fixed points.
    """
    fixed = [i for i, d in enumerate(p) if d == i]
    if len(fixed) == 0:
        return p
    if len(fixed) == 1:
        # This should not happen for these standard patterns at n=2^k (k>=1),
        # but handle it anyway by swapping with some other element.
        i = fixed[0]
        j = 0 if i != 0 else 1
        p2 = p[:]
        p2[i], p2[j] = p2[j], p2[i]
        if p2[i] == i or p2[j] == j:
            # fallback: find another j
            for j in range(len(p)):
                if j != i and p2[j] != j:
                    p2[i], p2[j] = p2[j], p2[i]
                    break
        return p2

    p2 = p[:]
    # rotate images among fixed points: f0->f1, f1->f2, ..., f_{m-1}->f0
    for t in range(len(fixed)):
        p2[fixed[t]] = fixed[(t + 1) % len(fixed)]
    return p2

def permutation_matrix_from_perm(p):
    n = len(p)
    M = [[0.0] * n for _ in range(n)]
    for i, j in enumerate(p):
        if i == j:
            raise RuntimeError("Permutation has fixed point; diagonal would be nonzero.")
        M[i][j] = 1.0
    return M

def sinkhorn_doubly_stochastic_zero_diag(W, max_iter=20000, tol=1e-12):
    n = len(W)
    M = [[0.0] * n for _ in range(n)]
    # copy and clamp diagonal to 0
    for i in range(n):
        for j in range(n):
            M[i][j] = 0.0 if i == j else max(0.0, float(W[i][j]))

    # Ensure off-diagonal positivity to avoid dead rows/cols.
    # (tiny epsilon keeps Sinkhorn stable without materially changing pattern)
    eps = 1e-15
    for i in range(n):
        for j in range(n):
            if i != j and M[i][j] == 0.0:
                M[i][j] = eps

    for _ in range(max_iter):
        # Row scale
        max_row_err = 0.0
        for i in range(n):
            s = sum(M[i])
            inv = 1.0 / s
            for j in range(n):
                if i != j:
                    M[i][j] *= inv
            max_row_err = max(max_row_err, abs(sum(M[i]) - 1.0))

        # Col scale
        max_col_err = 0.0
        col_sums = [0.0] * n
        for j in range(n):
            s = 0.0
            for i in range(n):
                s += M[i][j]
            col_sums[j] = s

        for j in range(n):
            inv = 1.0 / col_sums[j]
            for i in range(n):
                if i != j:
                    M[i][j] *= inv

        # Check errors
        for i in range(n):
            max_row_err = max(max_row_err, abs(sum(M[i]) - 1.0))
        for j in range(n):
            s = 0.0
            for i in range(n):
                s += M[i][j]
            max_col_err = max(max_col_err, abs(s - 1.0))

        if max(max_row_err, max_col_err) <= tol:
            break

    # Final clamp diagonal exactly
    for i in range(n):
        M[i][i] = 0.0
    return M

def make_hotspot_weight_matrix(n, hotspots, hotspot_boost):
    # Start from uniform off-diagonal baseline; boost hotspot columns
    W = [[1.0] * n for _ in range(n)]
    for i in range(n):
        W[i][i] = 0.0
    hs = set(hotspots)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if j in hs:
                W[i][j] *= hotspot_boost
    return W

def write_matrix_space_delimited(M, out_path):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        for row in M:
            f.write(" ".join(f"{x:.17g}" for x in row) + "\n")

def max_row_col_err(M):
    n = len(M)
    row_err = max(abs(sum(M[i]) - 1.0) for i in range(n))
    col_err = 0.0
    for j in range(n):
        s = 0.0
        for i in range(n):
            s += M[i][j]
        col_err = max(col_err, abs(s - 1.0))
    diag_max = max(abs(M[i][i]) for i in range(n))
    return row_err, col_err, diag_max

def main():
    ap = argparse.ArgumentParser(
        description="Generate doubly-stochastic demand matrices (diag=0) for standard traffic patterns."
    )
    ap.add_argument("-n", type=int, required=True, help="Number of nodes.")
    ap.add_argument("-o", type=str, required=True, help="Output file path.")
    ap.add_argument("--pattern", type=str, required=True,
                    choices=["hotspot", "bitcomplement", "bitreverse", "bitshuffle", "bittranspose"],
                    help="Traffic pattern.")
    ap.add_argument("--tol", type=float, default=1e-12, help="Sinkhorn tolerance (hotspot only).")
    ap.add_argument("--max_iter", type=int, default=20000, help="Sinkhorn max iters (hotspot only).")
    ap.add_argument("--verify", action="store_true", help="Print max row/col sum error to stderr.")

    # Hotspot controls (since hotspot is not a permutation)
    ap.add_argument("--hotspots", type=str, default="0",
                    help="Comma-separated hotspot destination node IDs (default: 0).")
    ap.add_argument("--hotspot_boost", type=float, default=10.0,
                    help="Multiplicative boost for hotspot columns before Sinkhorn (default: 10.0).")

    args = ap.parse_args()
    n = args.n
    if n < 2:
        raise SystemExit("Error: n must be >= 2")

    pattern = args.pattern

    if pattern in ["bitcomplement", "bitreverse", "bitshuffle", "bittranspose"]:
        if not is_power_of_two(n):
            raise SystemExit(
                f"Error: {pattern} in your simulator uses bits=floor(log2(n)) and dest%=n; "
                f"to match a true permutation (and get a doubly-stochastic matrix), n must be a power of two. Got n={n}."
            )

        if pattern == "bitcomplement":
            p = perm_bitcomplement(n)
        elif pattern == "bitreverse":
            p = perm_bitreverse(n)
        elif pattern == "bitshuffle":
            p = perm_bitshuffle(n)
        else:  # bittranspose
            p = perm_bittranspose(n)

        # Repair fixed points so diagonal is 0 while staying a permutation.
        p = repair_fixed_points_to_derangement(p)
        M = permutation_matrix_from_perm(p)

    else:  # hotspot
        hotspots = []
        for tok in args.hotspots.split(","):
            tok = tok.strip()
            if tok:
                v = int(tok)
                if v < 0 or v >= n:
                    raise SystemExit(f"Error: hotspot node {v} out of range [0,{n-1}]")
                hotspots.append(v)
        if not hotspots:
            raise SystemExit("Error: hotspots list is empty")

        W = make_hotspot_weight_matrix(n, hotspots, args.hotspot_boost)
        M = sinkhorn_doubly_stochastic_zero_diag(W, max_iter=args.max_iter, tol=args.tol)

    write_matrix_space_delimited(M, args.o)

    if args.verify:
        row_err, col_err, diag_max = max_row_col_err(M)
        print(
            f"Wrote {n}x{n} {pattern} demand matrix to {args.o} "
            f"(max_row_err={row_err:.3e}, max_col_err={col_err:.3e}, max_diag={diag_max:.3e})",
            file=sys.stderr
        )

if __name__ == "__main__":
    main()
