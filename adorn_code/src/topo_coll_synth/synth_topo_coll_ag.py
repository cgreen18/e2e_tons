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

    # 1. Visualize the Base Physical Topology
    G_base = nx.Graph()
    G_base.add_nodes_from(range(num_nodes))
    G_base.add_edges_from(edges)
    
    # Use a circular layout for symmetry, or spring_layout for complex graphs
    pos = nx.circular_layout(G_base)
    
    plt.figure(figsize=(8, 8))
    plt.title(f"Synthesized Physical Topology ({num_nodes} Nodes)")
    nx.draw(G_base, pos, with_labels=True, node_color='lightblue', 
            node_size=800, font_size=12, font_weight='bold', edge_color='gray')
    plt.savefig(f"{output_dir}/00_physical_topology.png")
    plt.close()

    # 2. Group Schedule by Time Segments
    # Group transfers by their (start, end) time window
    time_segments = {}
    for s_time, e_time, c, u, v in schedule:
        window = (s_time, e_time)
        if window not in time_segments:
            time_segments[window] = []
        time_segments[window].append((u, v, c))
    
    # 3. Visualize Each Time Segment
    # Sort the time windows chronologically
    sorted_windows = sorted(time_segments.keys())
    
    for i, (s_time, e_time) in enumerate(sorted_windows):
        transfers = time_segments[(s_time, e_time)]
        
        # Create a directed graph for this specific time slice
        G_time = nx.DiGraph()
        G_time.add_nodes_from(range(num_nodes))
        
        edge_labels = {}
        for u, v, c in transfers:
            G_time.add_edge(u, v)
            # If multiple chunks travel the same link (rare in this model but possible), append them
            if (u, v) in edge_labels:
                edge_labels[(u, v)] += f", C{c}"
            else:
                edge_labels[(u, v)] = f"C{c}"
                
        plt.figure(figsize=(8, 8))
        plt.title(f"Schedule Segment {i+1}: Time {s_time} to {e_time}")
        
        # Draw base nodes
        nx.draw_networkx_nodes(G_time, pos, node_color='lightgreen', node_size=800)
        nx.draw_networkx_labels(G_time, pos, font_size=12, font_weight='bold')
        
        # Draw active directed edges
        nx.draw_networkx_edges(G_time, pos, edge_color='blue', arrows=True, 
                               arrowstyle='-|>', arrowsize=20, width=2)
        
        # Draw edge labels (the chunk IDs)
        nx.draw_networkx_edge_labels(G_time, pos, edge_labels=edge_labels, 
                                     font_color='red', font_weight='bold')
        
        # Draw inactive physical edges in light gray background context
        inactive_edges = [e for e in G_base.edges if e not in G_time.edges and (e[1], e[0]) not in G_time.edges]
        nx.draw_networkx_edges(G_base, pos, edgelist=inactive_edges, edge_color='lightgray', style='dashed')
        
        plt.savefig(f"{output_dir}/segment_{i+1:02d}_time_{s_time}_to_{e_time}.png")
        plt.close()
        
    print(f"Visualizations saved to ./{output_dir}/")

def synthesize_allgather(num_nodes, radix, bandwidth, duration, max_time):
    model = cp_model.CpModel()
    nodes = range(num_nodes)
    
    # --- 1. Variables ---
    y = {}        # y[u,v] = 1 if physical edge exists
    x = {}        # x[u,v,c] = 1 if chunk c is routed over u->v
    start = {}    # start[u,v,c] = Start time of transfer
    end = {}      # end[u,v,c] = End time of transfer
    interval = {} # interval[u,v,c] = Optional interval for CP-SAT scheduling
    A = {}        # A[v,c] = Time node v acquires chunk c

    for u in nodes:
        for v in nodes:
            if u != v:
                y[u, v] = model.NewBoolVar(f'y_{u}_{v}')
                for c in nodes:
                    x[u, v, c] = model.NewBoolVar(f'x_{u}_{v}_{c}')
                    start[u, v, c] = model.NewIntVar(0, max_time, f's_{u}_{v}_{c}')
                    end[u, v, c] = model.NewIntVar(0, max_time, f'e_{u}_{v}_{c}')
                    
                    # Tie start, duration, and end together ONLY IF x is true
                    interval[u, v, c] = model.NewOptionalIntervalVar(
                        start[u, v, c], duration, end[u, v, c], x[u, v, c], f'i_{u}_{v}_{c}'
                    )

    for v in nodes:
        for c in nodes:
            A[v, c] = model.NewIntVar(0, max_time, f'A_{v}_{c}')

    M = model.NewIntVar(0, max_time, 'makespan')

    # --- 2. Topology Constraints (Physical Rules) ---
    for u in nodes:
        # Radix limit per node
        model.Add(sum(y[u, v] for v in nodes if u != v) <= radix)
        for v in nodes:
            if u < v:
                # Symmetry (bidirectional links)
                model.Add(y[u, v] == y[v, u])

    # --- 3. Network Flow & Scheduling Constraints ---
    for c in nodes:
        # Initial supply: Node c gets chunk c at time 0
        model.Add(A[c, c] == 0) 

    for u in nodes:
        for v in nodes:
            if u != v:
                # Link Capacity Constraint using AddCumulative
                intervals_on_link = [interval[u, v, c] for c in nodes]
                demands = [1 for _ in nodes] # 1 unit of bandwidth per chunk
                model.AddCumulative(intervals_on_link, demands, bandwidth)

                for c in nodes:
                    # Channel Enablement: Cannot route if edge doesn't exist
                    model.AddImplication(x[u, v, c], y[u, v])
                    
                    # Precedence: Node u must have chunk c before starting send
                    model.Add(start[u, v, c] >= A[u, c]).OnlyEnforceIf(x[u, v, c])
                    
                    # Arrival: Node v acquires chunk c exactly when transfer ends
                    model.Add(A[v, c] == end[u, v, c]).OnlyEnforceIf(x[u, v, c])

    # --- 4. Demand & Routing constraints ---
    for v in nodes:
        for c in nodes:
            if v != c:
                # Spanning tree: Node v must receive chunk c from exactly one neighbor
                model.AddExactlyOne(x[u, v, c] for u in nodes if u != v)
            
            # Makespan must wait for all nodes to acquire all chunks
            model.Add(M >= A[v, c])

    # --- 5. Speedups ---
    # Calculate the theoretical minimum time steps required
    # (Assuming bandwidth=1 and duration=1 for this calculation)
    theoretical_min = math.ceil((num_nodes - 1) / radix)

    # Inject the bound into the makespan variable directly
    model.Add(M >= theoretical_min)
    print(f"Theoretical minimum time steps required: {theoretical_min}")
    # quit()

    # Force Node 0 to connect to exactly 'radix' specific nodes (e.g., nodes 1, 2, 3, 4)
    # This prevents the solver from generating isomorphic rotations of the same graph.
    for v in range(1, radix + 1):
        model.Add(y[0, v] == 1)

    # --- 6. Objective ---
    model.Minimize(M)

    # --- 7. Solve ---
    solver = cp_model.CpSolver()
    # Enable solver logging
    solver.parameters.log_search_progress = True
    print("Solving...")
    status = solver.Solve(model)

    # --- 8. Output ---
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(f"Optimal Makespan: {solver.ObjectiveValue()}")
        print("\nSynthesized Topology:")
        physical_edges = []
        for u in nodes:
            for v in nodes:
                if u < v and solver.Value(y[u, v]):
                    print(f"  Edge: Node {u} <--> Node {v} : {solver.Value(y[u,v])}")
                    physical_edges.append((u, v))

        print("\n--- Execution Schedule (Chronological) ---")
        schedule = []
        for c in nodes:
            for u in nodes:
                for v in nodes:
                    # If the solver chose to route chunk c over link u->v
                    if u != v and solver.Value(x[u, v, c]):
                        s_time = solver.Value(start[u, v, c])
                        e_time = solver.Value(end[u, v, c])
                        schedule.append((s_time, e_time, c, u, v))
        
        # Sort by start time, then end time, then chunk
        schedule.sort()
        
        for s_time, e_time, c, u, v in schedule:
            print(f"Time {s_time:d} to {e_time:d} | Chunk {c} | Node {u} -> Node {v}")
    
        # Call the visualizer
        visualize_synthesis(num_nodes, physical_edges, schedule)
    
    else:
        print("No solution found within the max_time bound.")

# Test with 4 nodes, radix 2 (should synthesize a ring topology)
synthesize_allgather(num_nodes=10, radix=4, bandwidth=1, duration=1, max_time=200)
