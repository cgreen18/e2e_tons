# std
import os
import math

# pipd
from ortools.sat.python import cp_model
import networkx as nx
import matplotlib.pyplot as plt


def visualize_synthesis(num_nodes, edges, schedule, output_dir="output_images"):
    """
    Generates PNGs for the synthesized physical topology and each time segment.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    G_base = nx.Graph()
    G_base.add_nodes_from(range(num_nodes))
    G_base.add_edges_from(edges)
    
    pos = nx.circular_layout(G_base)
    
    plt.figure(figsize=(8, 8))
    plt.title(f"Synthesized Physical Topology ({num_nodes} Nodes)")
    nx.draw(G_base, pos, with_labels=True, node_color='lightblue', 
            node_size=800, font_size=12, font_weight='bold', edge_color='gray')
    plt.savefig(f"{output_dir}/00_physical_topology.png")
    plt.close()

    time_segments = {}
    for s_time, e_time, c, u, v in schedule:
        window = (s_time, e_time)
        if window not in time_segments:
            time_segments[window] = []
        time_segments[window].append((u, v, c))
    
    sorted_windows = sorted(time_segments.keys())
    
    for i, (s_time, e_time) in enumerate(sorted_windows):
        transfers = time_segments[(s_time, e_time)]
        
        G_time = nx.DiGraph()
        G_time.add_nodes_from(range(num_nodes))
        
        edge_labels = {}
        for u, v, c in transfers:
            G_time.add_edge(u, v)
            if (u, v) in edge_labels:
                edge_labels[(u, v)] += f", {c}"
            else:
                edge_labels[(u, v)] = f"{c}"
                
        plt.figure(figsize=(8, 8))
        plt.title(f"Schedule Segment {i+1}: Time {s_time} to {e_time}")
        
        nx.draw_networkx_nodes(G_time, pos, node_color='lightgreen', node_size=800)
        nx.draw_networkx_labels(G_time, pos, font_size=12, font_weight='bold')
        nx.draw_networkx_edges(G_time, pos, edge_color='blue', arrows=True, 
                               arrowstyle='-|>', arrowsize=20, width=2)
        nx.draw_networkx_edge_labels(G_time, pos, edge_labels=edge_labels, 
                                     font_color='red', font_weight='bold')
        
        inactive_edges = [e for e in G_base.edges if e not in G_time.edges and (e[1], e[0]) not in G_time.edges]
        nx.draw_networkx_edges(G_base, pos, edgelist=inactive_edges, edge_color='lightgray', style='dashed')
        
        plt.savefig(f"{output_dir}/segment_{i+1:02d}_time_{s_time}_to_{e_time}.png")
        plt.close()
        
    print(f"Visualizations saved to ./{output_dir}/")

def calculate_theoretical_min(num_nodes, radix, bandwidth=1):
    """
    Calculates the absolute minimum makespan for an all-to-all based on the 
    best possible theoretical graph diameter and volume (Moore Bound).
    """
    # 1. Calculate minimum hop volume from the perspective of one node
    unreached_nodes = num_nodes - 1
    current_distance = 1
    nodes_at_current_distance = radix
    total_hops_for_one_node = 0
    
    while unreached_nodes > 0:
        # We can only reach as many nodes as are left, or the capacity of this "ring"
        nodes_to_reach = min(unreached_nodes, nodes_at_current_distance)
        total_hops_for_one_node += nodes_to_reach * current_distance
        
        unreached_nodes -= nodes_to_reach
        current_distance += 1
        nodes_at_current_distance *= (radix - 1)
        
    # Total volume of all chunks traveling their shortest possible paths
    total_network_hop_volume = total_hops_for_one_node * num_nodes
    
    # 2. Calculate maximum physical network capacity per time step
    # Max undirected edges = floor((N * R) / 2)
    max_undirected_edges = (num_nodes * radix) // 2
    
    # Because links are bidirectional, each undirected edge yields 2 directed links.
    max_directed_links = max_undirected_edges * 2
    max_capacity_per_step = max_directed_links * bandwidth
    
    # 3. The Bound: Volume / Capacity
    hop_bound = math.ceil(total_network_hop_volume / max_capacity_per_step)
    
    # We must also respect the standard injection bound (whichever is higher)
    injection_bound = math.ceil((num_nodes - 1) / radix)
    
    return max(hop_bound, injection_bound)

def calculate_theoretical_max(num_nodes, bandwidth, duration):
    """
    Calculates a safe theoretical maximum makespan by evaluating the
    worst-case connected topology (a line graph) and its bisection bandwidth.
    """
    # 1. Calculate traffic crossing the exact middle of a line graph
    half_1 = num_nodes // 2
    half_2 = num_nodes - half_1
    max_bisection_traffic = half_1 * half_2
    
    # 2. Time required to serialize that traffic across a single link
    serialization_time = math.ceil(max_bisection_traffic / bandwidth) * duration
    
    # 3. Add propagation delay (longest path in the line graph)
    propagation_delay = (num_nodes - 1) * duration
    
    # Safe upper bound
    return serialization_time + propagation_delay

def calculate_reasonable_max(num_nodes, radix, bandwidth=1, duration=1):
    """
    Calculates a tight, realistic upper bound for an all-to-all makespan by evaluating
    the Expected Hop Volume of a well-connected Random Regular Graph (RRG).
    """
    # 1. Expected Average Path Length (approximated by diameter of RRG)
    if radix > 2:
        expected_avg_path = math.ceil(math.log(num_nodes) / math.log(radix - 1))
    else:
        expected_avg_path = num_nodes // 2  # Fallback for ring graphs
        
    # 2. Total Expected Hop Volume
    total_chunks = num_nodes * (num_nodes - 1)
    expected_hop_volume = total_chunks * expected_avg_path
    
    # 3. Maximum Directed Network Capacity per step
    max_capacity = num_nodes * radix * bandwidth
    
    # 4. Total Time = (Serialization of Volume) + (Pipeline Drain / Propagation)
    serialization_time = math.ceil(expected_hop_volume / max_capacity) * duration
    propagation_delay = expected_avg_path * duration
    
    return serialization_time + propagation_delay

def synthesize_alltoall(num_nodes, radix, bandwidth, duration, max_time):
    model = cp_model.CpModel()
    nodes = range(num_nodes)

    theoretical_min = calculate_theoretical_min(num_nodes, radix, bandwidth=bandwidth)
    theoretical_max = calculate_theoretical_max(num_nodes, bandwidth=bandwidth, duration=duration)
    reasonable_max = calculate_reasonable_max(num_nodes, radix, bandwidth=bandwidth, duration=duration)
    max_time = reasonable_max
    print(f"Injection/Ejection theoretical minimum time steps: {theoretical_min}")
    print(f"Injection/Ejection theoretical maximum time steps: {theoretical_max}")
    print(f"Injection/Ejection reasonable maximum time steps: {reasonable_max}")

    # Define All-to-All Chunks as (Source, Destination) pairs
    chunks = [(s, d) for s in nodes for d in nodes if s != d]
    
    # --- 1. Variables ---
    y = {}        
    x = {}        
    start = {}    
    end = {}      
    interval = {} 
    A = {}        

    for u in nodes:
        for v in nodes:
            if u != v:
                y[u, v] = model.NewBoolVar(f'y_{u}_{v}')
                for s, d in chunks:
                    x[u, v, s, d] = model.NewBoolVar(f'x_{u}_{v}_{s}_{d}')
                    start[u, v, s, d] = model.NewIntVar(0, max_time, f's_{u}_{v}_{s}_{d}')
                    end[u, v, s, d] = model.NewIntVar(0, max_time, f'e_{u}_{v}_{s}_{d}')
                    
                    interval[u, v, s, d] = model.NewOptionalIntervalVar(
                        start[u, v, s, d], duration, end[u, v, s, d], x[u, v, s, d], f'i_{u}_{v}_{s}_{d}'
                    )

    for v in nodes:
        for s, d in chunks:
            A[v, s, d] = model.NewIntVar(0, max_time, f'A_{v}_{s}_{d}')

    M = model.NewIntVar(0, max_time, 'makespan')

    # --- 2. Topology Constraints ---
    for u in nodes:
        model.Add(sum(y[u, v] for v in nodes if u != v) <= radix)
        for v in nodes:
            if u < v:
                model.Add(y[u, v] == y[v, u])

    # --- 3. Network Flow & Scheduling Constraints ---
    for s, d in chunks:
        # Initial supply: Source acquires its own distinct chunk at time 0
        model.Add(A[s, s, d] == 0) 

    for u in nodes:
        for v in nodes:
            if u != v:
                intervals_on_link = [interval[u, v, s, d] for s, d in chunks]
                demands = [1 for _ in chunks] 
                model.AddCumulative(intervals_on_link, demands, bandwidth)

                for s, d in chunks:
                    model.AddImplication(x[u, v, s, d], y[u, v])
                    model.Add(start[u, v, s, d] >= A[u, s, d]).OnlyEnforceIf(x[u, v, s, d])
                    model.Add(A[v, s, d] == end[u, v, s, d]).OnlyEnforceIf(x[u, v, s, d])

    # --- 4. Demand & Routing constraints (Flow Conservation) ---
    for s, d in chunks:
        for v in nodes:
            incoming = sum(x[u, v, s, d] for u in nodes if u != v)
            outgoing = sum(x[v, w, s, d] for w in nodes if w != v)
            
            if v == s:
                model.Add(outgoing == 1)
                model.Add(incoming == 0)
            elif v == d:
                model.Add(incoming == 1)
                model.Add(outgoing == 0)
            else:
                model.Add(incoming == outgoing)
                # Bound to 1 to prevent isolated loops traversing intermediate nodes
                model.Add(incoming <= 1)
                
        # Makespan must wait for the specific destination to acquire the chunk
        model.Add(M >= A[d, s, d])

    # --- 5. Speedups ---
    # The injection lower bound applies to all-to-all just as it does all-gather
    # theoretical_min = math.ceil((num_nodes - 1) / radix)
    model.Add(M >= theoretical_min)

    for v in range(1, radix + 1):
        model.Add(y[0, v] == 1)

    # --- 6. Objective ---
    model.Minimize(M)

    # --- 7. Solve ---
    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = True
    solver.parameters.symmetry_level = 3
    print("Solving...")
    status = solver.Solve(model)

    # --- 8. Output ---
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(f"\nOptimal Makespan: {solver.ObjectiveValue()}")
        print("\nSynthesized Topology:")
        physical_edges = []
        for u in nodes:
            for v in nodes:
                if u < v and solver.Value(y[u, v]):
                    print(f"  Edge: Node {u} <--> Node {v}")
                    physical_edges.append((u, v))

        print("\n--- Execution Schedule (Chronological) ---")
        schedule = []
        for s, d in chunks:
            for u in nodes:
                for v in nodes:
                    if u != v and solver.Value(x[u, v, s, d]):
                        s_time = solver.Value(start[u, v, s, d])
                        e_time = solver.Value(end[u, v, s, d])
                        # Format chunk label as Source->Destination
                        chunk_label = f"{s}->{d}"
                        schedule.append((s_time, e_time, chunk_label, u, v))
        
        schedule.sort()
        for s_time, e_time, chunk_label, u, v in schedule:
            print(f"Time {s_time} to {e_time} | Chunk {chunk_label} | Node {u} -> Node {v}")
    
        visualize_synthesis(num_nodes, physical_edges, schedule)
    else:
        print("No solution found within the max_time bound.")

# Recommend testing with smaller sizes first (e.g., N=6) due to O(N^4) variables
synthesize_alltoall(num_nodes=20, radix=4, bandwidth=1, duration=1, max_time=200)