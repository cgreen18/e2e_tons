import sys
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
# print(matplotlib.get_backend())
# quit()

# Flag to control whether to plot in separate subplots or single plot
SEPARATE_PLOTS = True #False

TOPO_FONT_SIZE = 16
TIME_FONT_SIZE = 12
AXIS_FONT_SIZE = 12
LEGEND_FONT_SIZE = 10
TITLE_FONT_SIZE = 14
SUBTITLE_FONT_SIZE = 12
XLABEL_FONT_SIZE = 12
YLABEL_FONT_SIZE = 12

LEGEND_LOC = 'lower center'

in_name = sys.argv[1]
known_values_file = None
if len(sys.argv) > 2:
    known_values_file = sys.argv[2]

base_in_name = in_name.replace('_milp_timeline.csv', '').split('/')[-1]
out_png_name = f'files/timeline_graphs/{base_in_name}_milp_timeline.png'


time_vals = []
obj_vals = []
avg_hop_vals = []
pct_diff_vals = []
best_bound_vals = []

n_unconnected = 0

with open(in_name, 'r') as inf:
    for line in inf:
        if "elapsed_time" in line:
            continue
        split_line = line.strip().split(',')

        print(f'split_line = {split_line}')

        elapsed_time = float(split_line[0])
        avg_hops = float(split_line[1])
        objective = float(split_line[2])
        best_bound = float(split_line[3])
        pct_diff = float(split_line[4])

        if avg_hops > 100:
            n_unconnected += 1

        time_vals.append(elapsed_time)
        obj_vals.append(objective)
        avg_hop_vals.append(avg_hops)
        pct_diff_vals.append(pct_diff)
        best_bound_vals.append(best_bound)

        print(f'time {elapsed_time}, avg hops {avg_hops}, obj {objective}, pct_diff {pct_diff}')

# Convert time to minutes
time_vals = [t / 60.0 for t in time_vals]

# Filter valid values for plotting
# Average hops > 100 should not be plotted
# Approximate SC or bound > 1.0 should not be plotted
valid_time_hops = []
valid_avg_hops = []
valid_time_obj = []
valid_obj_vals = []
valid_time_bound = []
valid_bound_vals = []

for i in range(len(time_vals)):
    # Filter avg hops (must be <= 100)
    if avg_hop_vals[i] <= 100:
        valid_time_hops.append(time_vals[i])
        valid_avg_hops.append(avg_hop_vals[i])
    
    # Filter objective (must be <= 1.0)
    if obj_vals[i] <= 1.0 and obj_vals[i] > 0.0:
        valid_time_obj.append(time_vals[i])
        valid_obj_vals.append(obj_vals[i])
    
    # Filter best bound (must be <= 1.0)
    if best_bound_vals[i] <= 1.0 and best_bound_vals[i] > 0.0:
        valid_time_bound.append(time_vals[i])
        valid_bound_vals.append(best_bound_vals[i])

# Create figure(s) based on SEPARATE_PLOTS flag
if SEPARATE_PLOTS:
    # Create two separate subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot avg_hops in first subplot (only valid values)
    if len(valid_avg_hops) > 0:
        ax1.plot(valid_time_hops, valid_avg_hops, 
                 marker='o', color='tab:blue', label='Avg Hops', linewidth=3, markersize=6)
    ax1.set_ylabel('Average Hops', color='tab:blue', fontsize=YLABEL_FONT_SIZE)
    ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=AXIS_FONT_SIZE)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Average Hops', fontsize=SUBTITLE_FONT_SIZE, pad=10)
    
    # Plot objective and bound in second subplot
    if len(valid_obj_vals) > 0:
        ax2.plot(valid_time_obj, valid_obj_vals, 
                 marker='s', color='tab:red', label='Approximate SC', linewidth=3, markersize=8)
    if len(valid_bound_vals) > 0:
        # Plot best bound as a line (red to match approximate SC axis)
        ax2.plot(valid_time_bound, valid_bound_vals, 
                 marker='*', color='tab:red', label='Best Bound', linewidth=3, markersize=10)#, linestyle='--')
    ax2.set_ylabel('Approximate Sparsest Cut / Best Bound', color='tab:red', fontsize=YLABEL_FONT_SIZE)
    ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=AXIS_FONT_SIZE)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Approximate SC and Best Bound', fontsize=SUBTITLE_FONT_SIZE, pad=10)
    ax2.set_xlabel('Time (min)', fontsize=XLABEL_FONT_SIZE)
    ax2.tick_params(axis='x', labelsize=AXIS_FONT_SIZE)
    
    # Overall title
    fig.suptitle('MILP Timeline', fontsize=TITLE_FONT_SIZE, y=0.995)
else:
    # Create figure with two y-axes for lines (original behavior)
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Plot avg_hops on left y-axis (only valid values)
    if len(valid_avg_hops) > 0:
        ax1.plot(valid_time_hops, valid_avg_hops, 
                 marker='o', color='tab:blue', label='Avg Hops', linewidth=3, markersize=6)
    ax1.set_ylabel('Average Hops', color='tab:blue', fontsize=YLABEL_FONT_SIZE)
    ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=AXIS_FONT_SIZE)
    ax1.grid(True, alpha=0.3)
    
    # Create right y-axis for objective and bound
    ax2 = ax1.twinx()
    if len(valid_obj_vals) > 0:
        ax2.plot(valid_time_obj, valid_obj_vals, 
                 marker='*', color='tab:red', label='Approximate SC', linewidth=3, markersize=8)
    if len(valid_bound_vals) > 0:
        # Plot best bound as a line (red to match approximate SC axis)
        ax2.plot(valid_time_bound, valid_bound_vals, 
                 marker='s', color='tab:red', label='Best Bound', linewidth=3, markersize=6, linestyle='--')
    ax2.set_ylabel('Approximate Sparsest Cut / Best Bound', color='tab:red', fontsize=YLABEL_FONT_SIZE)
    ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=AXIS_FONT_SIZE)
    
    # Labels and layout
    ax1.set_xlabel('Time (min)', fontsize=XLABEL_FONT_SIZE)
    ax1.tick_params(axis='x', labelsize=AXIS_FONT_SIZE)
    plt.title('MILP Timeline: Avg Hops, Approximate SC, and Best Bound', fontsize=TITLE_FONT_SIZE, pad=20)
# Read and plot known values if provided
if known_values_file:
    try:
        # Get y-axis ranges to check for overlap
        if SEPARATE_PLOTS:
            y_min_hops = min(valid_avg_hops) if valid_avg_hops else 0
            y_max_hops = max(valid_avg_hops) if valid_avg_hops else 100
            y_min_sc = min(valid_obj_vals + valid_bound_vals) if (valid_obj_vals + valid_bound_vals) else 0
            y_max_sc = max(valid_obj_vals + valid_bound_vals) if (valid_obj_vals + valid_bound_vals) else 1.0
        else:
            y_min_hops = min(valid_avg_hops) if valid_avg_hops else 0
            y_max_hops = max(valid_avg_hops) if valid_avg_hops else 100
            y_min_sc = min(valid_obj_vals + valid_bound_vals) if (valid_obj_vals + valid_bound_vals) else 0
            y_max_sc = max(valid_obj_vals + valid_bound_vals) if (valid_obj_vals + valid_bound_vals) else 1.0
        
        # Get x-axis limits before extension for text placement
        all_times = time_vals if time_vals else [0, 1]
        x_min = min(all_times) if all_times else 0
        x_max = max(all_times) if all_times else 1
        
        with open(known_values_file, 'r') as inf:
            for line in inf:
                if line.strip() == "":
                    continue
                split_line = line.strip().split(',')
                if len(split_line) >= 3:
                    # Try to parse as float - if it fails, assume it's a header and skip
                    try:
                        float(split_line[1])
                        float(split_line[2])
                    except ValueError:
                        # This line can't be parsed as numbers, likely a header - skip it
                        continue
                    
                    topology = split_line[0].strip()
                    avg_hops_val = float(split_line[1])
                    approx_sc_val = float(split_line[2])
                    
                    # Read standard deviations (columns 4 and 5) if present
                    avg_hops_std = 0.0
                    approx_sc_std = 0.0
                    if len(split_line) >= 4:
                        try:
                            avg_hops_std = float(split_line[3])
                        except (ValueError, IndexError):
                            pass
                    if len(split_line) >= 5:
                        try:
                            approx_sc_std = float(split_line[4])
                        except (ValueError, IndexError):
                            pass
                    
                    # Only plot if values are valid
                    if avg_hops_val <= 100:
                        if avg_hops_std > 0:
                            # Plot standard deviation band
                            upper_hops = avg_hops_val + avg_hops_std
                            lower_hops = avg_hops_val - avg_hops_std
                            # Plot upper and lower bounds
                            ax1.axhline(y=upper_hops, color='tab:blue', linestyle='--', linewidth=1.5, alpha=0.5)
                            ax1.axhline(y=lower_hops, color='tab:blue', linestyle='--', linewidth=1.5, alpha=0.5)
                            # Shade between the bounds
                            x_range_plot = x_max - x_min
                            ax1.fill_between([x_min, x_max], lower_hops, upper_hops, 
                                            color='tab:blue', alpha=0.15)
                            # Plot center line
                            ax1.axhline(y=avg_hops_val, color='tab:blue', linestyle='--', linewidth=2.5, alpha=0.7)
                        else:
                            # Plot single horizontal dashed line for avg hops (blue to match avg hops axis)
                            ax1.axhline(y=avg_hops_val, color='tab:blue', linestyle='--', linewidth=2.5, alpha=0.7)
                    
                    if approx_sc_val <= 1.0:
                        if approx_sc_std > 0:
                            # Plot standard deviation band
                            upper_sc = approx_sc_val + approx_sc_std
                            lower_sc = approx_sc_val - approx_sc_std
                            # Plot upper and lower bounds
                            ax2.axhline(y=upper_sc, color='tab:red', linestyle='--', linewidth=1.5, alpha=0.5)
                            ax2.axhline(y=lower_sc, color='tab:red', linestyle='--', linewidth=1.5, alpha=0.5)
                            # Shade between the bounds
                            x_range_plot = x_max - x_min
                            ax2.fill_between([x_min, x_max], lower_sc, upper_sc, 
                                            color='tab:red', alpha=0.15)
                            # Plot center line
                            ax2.axhline(y=approx_sc_val, color='tab:red', linestyle='--', linewidth=2.5, alpha=0.7)
                        else:
                            # Plot single horizontal dashed line for approx SC (red to match SC axis)
                            ax2.axhline(y=approx_sc_val, color='tab:red', linestyle='--', linewidth=2.5, alpha=0.7)
    except Exception as e:
        print(f"Warning: Could not read known values file {known_values_file}: {e}")

# Extend x-axis by 10% on the right side to accommodate topology labels
if known_values_file:
    if SEPARATE_PLOTS:
        xlim1 = ax1.get_xlim()
        xlim2 = ax2.get_xlim()
        x_range1 = xlim1[1] - xlim1[0]
        x_range2 = xlim2[1] - xlim2[0]
        ax1.set_xlim(xlim1[0], xlim1[1] + x_range1 * 0.1)
        ax2.set_xlim(xlim2[0], xlim2[1] + x_range2 * 0.1)
    else:
        xlim = ax1.get_xlim()
        x_range = xlim[1] - xlim[0]
        ax1.set_xlim(xlim[0], xlim[1] + x_range * 0.1)

# Add topology text labels in the extended zone
if known_values_file:
    try:
        # Calculate text position in extended zone (after original x_max)
        all_times = time_vals if time_vals else [0, 1]
        x_min = min(all_times) if all_times else 0
        x_max = max(all_times) if all_times else 1
        x_range = x_max - x_min
        # Place text at 5% into the extended zone (x_max + 5% of original range)
        x_text_pos = x_max + x_range * 0.05
        
        with open(known_values_file, 'r') as inf:
            for line in inf:
                if line.strip() == "":
                    continue
                split_line = line.strip().split(',')
                if len(split_line) >= 3:
                    try:
                        float(split_line[1])
                        float(split_line[2])
                    except ValueError:
                        continue
                    
                    topology = split_line[0].strip()
                    avg_hops_val = float(split_line[1])
                    approx_sc_val = float(split_line[2])
                    
                    # Read standard deviations if present
                    avg_hops_std = 0.0
                    approx_sc_std = 0.0
                    if len(split_line) >= 4:
                        try:
                            avg_hops_std = float(split_line[3])
                        except (ValueError, IndexError):
                            pass
                    if len(split_line) >= 5:
                        try:
                            approx_sc_std = float(split_line[4])
                        except (ValueError, IndexError):
                            pass
                    
                    # Add text labels in extended zone
                    if avg_hops_val <= 100:
                        y_text_pos = avg_hops_val
                        y_range_hops = y_max_hops - y_min_hops if y_max_hops > y_min_hops else 1
                        text_offset = y_range_hops * 0.02
                        if avg_hops_val + text_offset <= y_max_hops * 1.05:
                            y_text_pos = avg_hops_val + text_offset
                        else:
                            y_text_pos = avg_hops_val - text_offset
                        ax1.text(x_text_pos, y_text_pos, topology, fontsize=TOPO_FONT_SIZE, color='black', 
                                alpha=0.8, verticalalignment='bottom' if y_text_pos > avg_hops_val else 'top')
                    
                    if approx_sc_val <= 1.0:
                        y_text_pos = approx_sc_val
                        y_range_sc = y_max_sc - y_min_sc if y_max_sc > y_min_sc else 0.1
                        text_offset = y_range_sc * 0.02
                        if approx_sc_val + text_offset <= y_max_sc * 1.05:
                            y_text_pos = approx_sc_val + text_offset
                        else:
                            y_text_pos = approx_sc_val - text_offset
                        ax2.text(x_text_pos, y_text_pos, topology, fontsize=TOPO_FONT_SIZE, color='black', 
                                alpha=0.8, verticalalignment='bottom' if y_text_pos > approx_sc_val else 'top')
    except Exception as e:
        print(f"Warning: Could not add topology labels: {e}")

# Add legends (known values are NOT included in legend)
if SEPARATE_PLOTS:
    # Separate legends for each subplot
    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1, labels1, loc=LEGEND_LOC, bbox_to_anchor=(1.02, 1), fontsize=LEGEND_FONT_SIZE, frameon=True)
    
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines2, labels2, loc=LEGEND_LOC, bbox_to_anchor=(1.02, 1), fontsize=LEGEND_FONT_SIZE, frameon=True)
else:
    # Combined legends outside the plot (original behavior)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc=LEGEND_LOC, 
               bbox_to_anchor=(1.02, 1), fontsize=LEGEND_FONT_SIZE, frameon=True)

fig.tight_layout()

# Save figure
plt.savefig(out_png_name, dpi=150, bbox_inches='tight')
print(f'Wrote to {out_png_name}')

plt.show()
