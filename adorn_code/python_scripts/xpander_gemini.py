import numpy as np
import random
import argparse

def generate_xpander_lift(n, d):
    """
    Constructs an Xpander by performing a k-lift on a base clique K_{d+1}.
    """
    base_nodes = d + 1
    k = n // base_nodes
    if n % base_nodes != 0:
        print(f"Warning: n ({n}) is not a multiple of d+1 ({base_nodes}).")
        print("The resulting graph will have slightly uneven meta-nodes.")

    # Assign nodes to meta-nodes (k copies of each base vertex)
    meta_nodes = []
    nodes = list(range(n))
    random.shuffle(nodes)
    for i in range(base_nodes):
        size = k + (1 if i < n % base_nodes else 0)
        meta_nodes.append(nodes[:size])
        nodes = nodes[size:]

    adj = np.zeros((n, n), dtype=int)

    # For every edge in the base clique K_{d+1}, insert a matching
    for i in range(base_nodes):
        for j in range(i + 1, base_nodes):
            m1 = meta_nodes[i][:]
            m2 = meta_nodes[j][:]
            
            # Ensure we have enough nodes for a matching
            random.shuffle(m1)
            random.shuffle(m2)
            
            # The paper's k-lift: for every edge (u,v) in G, 
            # insert a matching between the k copies of u and k copies of v.
            size = min(len(m1), len(m2))
            for idx in range(size):
                u, v = m1[idx], m2[idx]
                adj[u, v] = 1
                adj[v, u] = 1
                
    return adj

def get_spectral_gap(adj):
    # Calculate eigenvalues to check expansion quality
    eigenvalues = np.linalg.eigvalsh(adj)
    # The largest eigenvalue for a d-regular graph is d.
    # We want the second largest (lambda_2) to be as small as possible.
    sorted_eig = sorted(eigenvalues, reverse=True)
    return sorted_eig[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, required=True, help="Total nodes")
    parser.add_argument("-d", type=int, required=True, help="Radix/Degree")
    parser.add_argument("-o", "--output", type=str, default="xpander.txt")
    args = parser.parse_args()

    # Strategy: Generate several and pick the best (smallest lambda_2)
    best_adj = None
    min_lambda = float('inf')
    
    print("Generating candidates and optimizing for spectral gap...")
    for _ in range(5): # Check 5 candidates
        adj = generate_xpander_lift(args.n, args.d)
        gap_metric = get_spectral_gap(adj)
        if gap_metric < min_lambda:
            min_lambda = gap_metric
            best_adj = adj
            
    print(f"Selected topology with lambda_2: {min_lambda:.4f} (Goal: < 2*sqrt(d-1) for Ramanujan)")

    with open(args.output, "w") as f:
        for row in best_adj:
            f.write(" ".join(map(str, row)) + "\n")
    print(f"Adjacency matrix saved to {args.output}")

if __name__ == "__main__":
    main()