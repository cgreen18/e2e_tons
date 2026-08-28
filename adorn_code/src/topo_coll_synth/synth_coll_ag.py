# std
import argparse
import math
import os


# pipd
from ortools.sat.python import cp_model
import networkx as nx



def visualize_synthesis(num_nodes, edges, schedule, output_dir="output_images"):
    """
    Generates PNGs for the synthesized physical topology and each time segment.
    """
    import matplotlib.pyplot as plt

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Visualize the Base Physical Topology
    G_base = nx.Graph()
    G_base.add_nodes_from(range(num_nodes))
    G_base.add_edges_from(edges)
    
    # Use a circular layout for symmetry, or spring_layout for complex graphs
    pos = nx.circular_layout(G_base)
    
    plt.figure(figsize=(8, 8))
    plt.title(f"Input Physical Topology ({num_nodes} Nodes)")
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


def read_unit_capacity_topology(path_name):
    """
    Reads a whitespace-delimited unit-capacity adjacency matrix.
    """
    print(f"Ingesting topology ({path_name})")

    adj_mat = []
    with open(path_name, "r") as inf:
        for lineno, row in enumerate(inf, start=1):
            row = row.strip()
            if not row:
                continue

            values = []
            for colno, elem in enumerate(row.split(), start=1):
                try:
                    value = float(elem)
                except ValueError as exc:
                    raise ValueError(
                        f"{path_name}:{lineno}:{colno}: invalid adjacency value {elem!r}"
                    ) from exc

                rounded = int(round(value))
                if not math.isclose(value, rounded) or rounded not in (0, 1):
                    raise ValueError(
                        f"{path_name}:{lineno}:{colno}: expected unit-capacity 0/1 value, got {elem!r}"
                    )
                values.append(rounded)

            adj_mat.append(values)

    if not adj_mat:
        raise ValueError(f"Empty adjacency matrix: {path_name}")

    num_nodes = len(adj_mat)
    for row_idx, row in enumerate(adj_mat):
        if len(row) != num_nodes:
            raise ValueError(
                f"{path_name}: expected a square matrix, row {row_idx} has "
                f"{len(row)} entries instead of {num_nodes}"
            )
        if row[row_idx] != 0:
            raise ValueError(f"{path_name}: diagonal entry ({row_idx},{row_idx}) must be 0")

    return adj_mat


def topology_arcs(adj_mat):
    return [
        (u, v)
        for u, row in enumerate(adj_mat)
        for v, value in enumerate(row)
        if u != v and value == 1
    ]


def topology_edges(adj_mat):
    edges = set()
    for u, row in enumerate(adj_mat):
        for v, value in enumerate(row):
            if u != v and value == 1:
                edges.add(tuple(sorted((u, v))))
    return sorted(edges)


def max_out_degree(adj_mat):
    return max((sum(row) for row in adj_mat), default=0)


def add_exactly_one_or_false(model, variables):
    if variables:
        model.AddExactlyOne(variables)
    else:
        model.Add(0 == 1)


def synthesize_allgather(
    topology_path,
    bandwidth=1,
    duration=1,
    max_time=200,
    output_dir="output_images",
    visualize=True,
    log_search_progress=False,
):
    if bandwidth <= 0:
        raise ValueError("bandwidth must be > 0")
    if duration <= 0:
        raise ValueError("duration must be > 0")
    if max_time < 0:
        raise ValueError("max_time must be >= 0")

    adj_mat = read_unit_capacity_topology(topology_path)
    num_nodes = len(adj_mat)
    fixed_arcs = topology_arcs(adj_mat)
    incoming = {
        v: [u for u in range(num_nodes) if adj_mat[u][v] == 1]
        for v in range(num_nodes)
    }
    radix = max_out_degree(adj_mat)

    model = cp_model.CpModel()
    nodes = range(num_nodes)
    
    # --- 1. Variables ---
    x = {}        # x[u,v,c] = 1 if chunk c is routed over fixed arc u->v
    start = {}    # start[u,v,c] = Start time of transfer
    end = {}      # end[u,v,c] = End time of transfer
    interval = {} # interval[u,v,c] = Optional interval for CP-SAT scheduling
    A = {}        # A[v,c] = Time node v acquires chunk c

    for u, v in fixed_arcs:
        for c in nodes:
            x[u, v, c] = model.NewBoolVar(f'x_{u}_{v}_{c}')
            start[u, v, c] = model.NewIntVar(0, max_time, f's_{u}_{v}_{c}')
            end[u, v, c] = model.NewIntVar(0, max_time, f'e_{u}_{v}_{c}')

            # Tie start, duration, and end together ONLY IF x is true.
            interval[u, v, c] = model.NewOptionalIntervalVar(
                start[u, v, c],
                duration,
                end[u, v, c],
                x[u, v, c],
                f'i_{u}_{v}_{c}',
            )

    for v in nodes:
        for c in nodes:
            A[v, c] = model.NewIntVar(0, max_time, f'A_{v}_{c}')

    M = model.NewIntVar(0, max_time, 'makespan')

    # --- 2. Network Flow & Scheduling Constraints ---
    for c in nodes:
        # Initial supply: Node c gets chunk c at time 0
        model.Add(A[c, c] == 0) 

    for u, v in fixed_arcs:
        # Link Capacity Constraint using AddCumulative
        intervals_on_link = [interval[u, v, c] for c in nodes]
        demands = [1 for _ in nodes] # 1 unit of bandwidth per chunk
        model.AddCumulative(intervals_on_link, demands, bandwidth)

        for c in nodes:
            # Precedence: Node u must have chunk c before starting send
            model.Add(start[u, v, c] >= A[u, c]).OnlyEnforceIf(x[u, v, c])

            # Arrival: Node v acquires chunk c exactly when transfer ends
            model.Add(A[v, c] == end[u, v, c]).OnlyEnforceIf(x[u, v, c])

    # --- 3. Demand & Routing constraints ---
    for v in nodes:
        for c in nodes:
            if v != c:
                # Spanning tree: Node v must receive chunk c from exactly one neighbor
                add_exactly_one_or_false(model, [x[u, v, c] for u in incoming[v]])
            
            # Makespan must wait for all nodes to acquire all chunks
            model.Add(M >= A[v, c])

    # --- 4. Speedups ---
    # Calculate the theoretical minimum time steps required
    # (Assuming bandwidth=1 and duration=1 for this calculation)
    theoretical_min = math.ceil((num_nodes - 1) / radix) if radix > 0 else 0

    # Inject the bound into the makespan variable directly
    model.Add(M >= theoretical_min)
    print(f"Theoretical minimum time steps required: {theoretical_min}")
    print(f"Fixed topology: {num_nodes} nodes, {len(fixed_arcs)} directed arcs, max out-degree {radix}")

    # --- 5. Objective ---
    model.Minimize(M)

    # --- 6. Solve ---
    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = log_search_progress
    print("Solving...")
    status = solver.Solve(model)

    # --- 7. Output ---
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(f"Optimal Makespan: {solver.ObjectiveValue()}")
        print("\nInput Topology Arcs:")
        for u, v in fixed_arcs:
            print(f"  Arc: Node {u} -> Node {v}")

        print("\n--- Execution Schedule (Chronological) ---")
        schedule = []
        for c in nodes:
            for u, v in fixed_arcs:
                # If the solver chose to route chunk c over fixed arc u->v
                if solver.Value(x[u, v, c]):
                    s_time = solver.Value(start[u, v, c])
                    e_time = solver.Value(end[u, v, c])
                    schedule.append((s_time, e_time, c, u, v))
        
        # Sort by start time, then end time, then chunk
        schedule.sort()
        
        for s_time, e_time, c, u, v in schedule:
            print(f"Time {s_time:d} to {e_time:d} | Chunk {c} | Node {u} -> Node {v}")
    
        if visualize:
            visualize_synthesis(num_nodes, topology_edges(adj_mat), schedule, output_dir=output_dir)
    
    else:
        print("No solution found within the max_time bound.")
    return status


def parse_args():
    parser = argparse.ArgumentParser(
        description="Synthesize an all-gather collective schedule on a fixed unit-capacity topology."
    )
    parser.add_argument("--topology", required=True, help="Path to a whitespace-delimited 0/1 adjacency matrix.")
    parser.add_argument("--bandwidth", type=int, default=1, help="Per-directed-arc transfer capacity.")
    parser.add_argument("--duration", type=int, default=1, help="Duration of each chunk transfer.")
    parser.add_argument("--max-time", type=int, default=200, help="Maximum makespan bound.")
    parser.add_argument("--output-dir", default="output_images", help="Directory for visualization PNGs.")
    parser.add_argument("--no-visualize", action="store_true", help="Skip PNG visualization output.")
    parser.add_argument("--log-search-progress", action="store_true", help="Enable verbose CP-SAT search logging.")
    return parser.parse_args()


def main():
    args = parse_args()
    synthesize_allgather(
        topology_path=args.topology,
        bandwidth=args.bandwidth,
        duration=args.duration,
        max_time=args.max_time,
        output_dir=args.output_dir,
        visualize=not args.no_visualize,
        log_search_progress=args.log_search_progress,
    )


if __name__ == "__main__":
    main()
