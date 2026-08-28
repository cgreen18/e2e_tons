
import argparse

def node_id(r: int, c: int, n: int) -> int:
    return r * n + c

def main():
    ap = argparse.ArgumentParser(
        description="Generate an n×n 2D torus adjacency matrix (space-delimited 0/1)."
    )
    ap.add_argument("n", type=int, help="Torus dimension (n×n). Must be >= 2.")
    ap.add_argument(
        "--self-loops",
        action="store_true",
        help="Set diagonal to 1 (default: 0).",
    )
    args = ap.parse_args()

    n = args.n
    if n < 2:
        raise SystemExit("Error: n must be >= 2.")

    N = n * n
    A = [[0] * N for _ in range(N)]

    # Add undirected edges: right/left and down/up with wraparound
    for r in range(n):
        for c in range(n):
            u = node_id(r, c, n)

            v_right = node_id(r, (c + 1) % n, n)
            v_left  = node_id(r, (c - 1) % n, n)
            v_down  = node_id((r + 1) % n, c, n)
            v_up    = node_id((r - 1) % n, c, n)

            for v in (v_right, v_left, v_down, v_up):
                A[u][v] = 1
                A[v][u] = 1  # keep symmetric (safe even if already set)

    if args.self_loops:
        for i in range(N):
            A[i][i] = 1

    # Print space-delimited 0/1 matrix
    for i in range(N):
        print(" ".join(str(x) for x in A[i]))

if __name__ == "__main__":
    main()