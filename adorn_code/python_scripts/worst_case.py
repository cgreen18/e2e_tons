import numpy as np
from scipy.optimize import linear_sum_assignment
import sys

def load_adjacency_matrix(file_path):
    """
    Loads a space-delimited text file into a numpy matrix.
    Row i, Column j represents the bandwidth of link (i, j).
    """
    try:
        return np.loadtxt(file_path)
    except Exception as e:
        print(f"Error loading matrix: {e}")
        sys.exit(1)

def compute_worst_case_throughput(adj_matrix):
    """
    Determines the worst-case throughput for an input graph.
    """
    n = len(adj_matrix)
    
    # Identify all channels with a bandwidth greater than 0
    channels = [(i, j) for i in range(n) for j in range(n) if adj_matrix[i][j] > 0]
    
    # ---------------------------------------------------------
    # 1. Define Oblivious Routing Function (pi)
    # ---------------------------------------------------------
    # We construct an incremental routing function using Floyd-Warshall 
    # to find the deterministic shortest-path route for every pair.
    dist = np.full((n, n), np.inf)
    next_hop = np.full((n, n), -1, dtype=int)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                dist[i][j] = 0
            elif adj_matrix[i][j] > 0:
                dist[i][j] = 1 # Assuming unit distance for hop-count routing
                next_hop[i][j] = j
                
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    next_hop[i][j] = next_hop[i][k]

    # ---------------------------------------------------------
    # 2. Bipartite Graph Construction & Max-Weight Matching
    # ---------------------------------------------------------
    worst_case_ideal_throughput = np.inf
    bottleneck_channel = None

    # The maximum-weight matching is repeated over the set of all channels
    for c in channels:
        u, v = c
        b_c = adj_matrix[u][v]
        
        # Build the bipartite graph weights for this specific channel
        weight_matrix = np.zeros((n, n))
        for s in range(n):
            for d in range(n):
                if s == d: continue
                
                # Trace the oblivious route from source s to destination d
                curr = s
                uses_channel = False
                hops = 0 
                
                while curr != d and curr != -1 and hops < n:
                    nxt = next_hop[curr][d]
                    if curr == u and nxt == v:
                        uses_channel = True
                        break
                    curr = nxt
                    hops += 1
                    
                if uses_channel:
                    # Weight edge from source s to destination d by channel load
                    weight_matrix[s][d] = 1.0 
                    
        # Find maximum-weight matching to evaluate gamma_{c,max}(pi)
        # Note: linear_sum_assignment finds the *minimum* cost, so we negate the weights
        row_ind, col_ind = linear_sum_assignment(-weight_matrix)
        gamma_c_max = weight_matrix[row_ind, col_ind].sum()
        
        # ---------------------------------------------------------
        # 3. Calculate Ideal Worst-Case Throughput Bound
        # ---------------------------------------------------------
        if gamma_c_max > 0:
            # Scale bandwidth by the smallest fraction needed to saturate the channel
            channel_throughput = b_c / gamma_c_max
            if channel_throughput < worst_case_ideal_throughput:
                worst_case_ideal_throughput = channel_throughput
                bottleneck_channel = c
                
    return worst_case_ideal_throughput, bottleneck_channel

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python worst_case_throughput.py <adjacency_matrix.txt>")
        sys.exit(1)
        
    file_path = sys.argv[1]
    adj_matrix = load_adjacency_matrix(file_path)
    
    print("Calculating worst-case throughput...")
    throughput, channel = compute_worst_case_throughput(adj_matrix)
    
    if channel:
        print(f"Worst-case ideal throughput: {throughput:.4f}")
        print(f"Bottleneck Channel (source, dest): {channel}")
    else:
        print("No traffic loads found; throughput bound is infinite.")
