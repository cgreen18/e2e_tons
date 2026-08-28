import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ============================================================================
# CONFIGURABLE PARAMETERS
# ============================================================================

# File path
CSV_FILE_PATH = 'files/dor_vc_balance.csv'

# Group labels (major groups - routing methods)
MAJOR_GROUP_LABELS = ['DOR', 'AT']

# VC labels and colors
VC_LABELS = ['VC 0', 'VC 1']
VC_COLORS = ['tab:blue', 'tab:red']  # Blue for VC 0, Red for VC 1

# Rename dictionary for all labels (original: display_name)
LABEL_RENAMES = {
    # Major groups (routing)
    'DOR': 'DOR',  # Change as needed
    'AT': 'AT',  # Change as needed
    # Topologies (will be extracted from data)
    'pt_8x8x16': 'PT 8x8x16',  # Capitalize prefix and remove underscore
    'pt_4x16x16': 'PT 4x16x16',  # Capitalize prefix and remove underscore
    'pt_4x8x32': 'PT 4x8x32',  # Capitalize prefix and remove underscore
    'pt_4x4x64': 'PT 4x4x64',  # Capitalize prefix and remove underscore
    'pdtt_8x8x16': 'PDTT 8x8x16',  # Capitalize prefix and remove underscore
    # VC labels
    'VC 0': 'VC 0',  # Change as needed
    'VC 1': 'VC 1',  # Change as needed
}

# Bar chart configuration
BAR_WIDTH = 0.35  # Width of each bar
BAR_SPACING = 0.1  # Spacing between bars within a subgroup
SUBGROUP_SPACING = 0.3  # Spacing between subgroups (topologies)
MAJOR_GROUP_SPACING = 0.5  # Spacing between major groups

# Figure size
FIGURE_WIDTH = 6
FIGURE_HEIGHT = 3.2

# Font sizes
TITLE_FONT_SIZE = 16
AXIS_LABEL_FONT_SIZE = 14
TICK_LABEL_FONT_SIZE = 12
TICK_LABEL_ROTATION = 45  # Rotation angle for x-axis labels (degrees)
LEGEND_FONT_SIZE = 13
GROUP_LABEL_FONT_SIZE = 13
GROUP_LABEL_Y_OFFSET = 0.2  # Offset above plot (as fraction of y-axis range) - for y-axis extension

# Legend configuration
LEGEND_LOCATION = 'upper center'  # Options: 'best', 'upper right', 'lower left', etc.
LEGEND_BBOX_TO_ANCHOR = (0.5, 0.96)  # (x, y) position for legend - (0.5, 1.0) is center top of axes
LEGEND_BBOX_Y_OFFSET = 0.01  # Additional offset above plot (in axes coordinates, 0-1)
LEGEND_FRAME_ON = True
LEGEND_FRAME_ALPHA = 0.9

# Grid configuration
GRID_ENABLED = True
GRID_ALPHA = 0.3
GRID_LINESTYLE = '--'
GRID_AXIS = 'y'  # 'x', 'y', or 'both'

# Axes labels
X_AXIS_LABEL = 'Topology'
Y_AXIS_LABEL = 'Relative Hops per VC'

# Title
PLOT_TITLE = 'DOR VC Balance'

# Output file (None to display, or provide path to save)
OUTPUT_FILE = 'files/dor_vc_balance_plot.png'

# Edge color for bars (None for no edge, or specify color)
BAR_EDGE_COLOR = None
BAR_EDGE_WIDTH = 0.5

# ============================================================================
# MAIN SCRIPT
# ============================================================================

def main():
    # Read CSV file
    df = pd.read_csv(CSV_FILE_PATH)
    
    # Apply renames to data
    df['routing'] = df['routing'].map(lambda x: LABEL_RENAMES.get(x, x))
    df['topology'] = df['topology'].map(lambda x: LABEL_RENAMES.get(x, x))
    
    # Get unique topologies (sorted to maintain consistent order)
    topologies = sorted(df['topology'].unique())
    
    # Prepare data structure
    # Structure: {routing: {topology: {VC: value}}}
    data = {}
    for _, row in df.iterrows():
        routing = row['routing']
        topology = row['topology']
        
        if routing not in data:
            data[routing] = {}
        if topology not in data[routing]:
            data[routing][topology] = {}
        
        data[routing][topology]['VC 0'] = row['vc0']
        data[routing][topology]['VC 1'] = row['vc1']
    
    # Calculate positions
    num_major_groups = len(MAJOR_GROUP_LABELS)
    num_subgroups_per_major = len(topologies)
    num_bars_per_subgroup = len(VC_LABELS)  # 2 bars: VC0 and VC1
    
    # Calculate x positions for each major group
    x_positions = []
    x_labels = []
    x_major_group_positions = []
    
    current_x = 0
    for major_idx, major_group in enumerate(MAJOR_GROUP_LABELS):
        major_group_start = current_x
        major_group_renamed = LABEL_RENAMES.get(major_group, major_group)
        
        for subgroup_idx, topology in enumerate(topologies):
            # Center position for this subgroup
            subgroup_center = current_x
            
            # Calculate positions for 2 bars within this subgroup
            # Order: VC0, VC1
            bar_positions = []
            bar_offset = -(BAR_WIDTH + BAR_SPACING) / 2  # Start from left
            
            for vc_idx, vc_label in enumerate(VC_LABELS):
                bar_positions.append(subgroup_center + bar_offset)
                bar_offset += BAR_WIDTH + BAR_SPACING
            
            x_positions.append(bar_positions)
            topology_renamed = LABEL_RENAMES.get(topology, topology)
            x_labels.append(topology_renamed)
            
            # Move to next subgroup
            current_x += num_bars_per_subgroup * BAR_WIDTH + (num_bars_per_subgroup - 1) * BAR_SPACING + SUBGROUP_SPACING
        
        # Record major group center position for label
        major_group_center = (major_group_start + current_x - SUBGROUP_SPACING) / 2
        x_major_group_positions.append((major_group_center, major_group_renamed))
        
        # Add spacing between major groups
        current_x += MAJOR_GROUP_SPACING
    
    # Create figure
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    
    # Plot bars
    bar_idx = 0
    for major_group in MAJOR_GROUP_LABELS:
        major_group_renamed = LABEL_RENAMES.get(major_group, major_group)
        for topology in topologies:
            topology_renamed = LABEL_RENAMES.get(topology, topology)
            positions = x_positions[bar_idx]
            pos_idx = 0
            
            for vc_idx, vc_label in enumerate(VC_LABELS):
                vc_label_renamed = LABEL_RENAMES.get(vc_label, vc_label)
                # Get value using renamed keys
                value = data[major_group_renamed][topology_renamed][vc_label]
                
                # Plot bar
                ax.bar(
                    positions[pos_idx],
                    value,
                    width=BAR_WIDTH,
                    color=VC_COLORS[vc_idx],
                    edgecolor=BAR_EDGE_COLOR,
                    linewidth=BAR_EDGE_WIDTH,
                    label=vc_label_renamed if bar_idx == 0 and vc_idx == 0 else '',
                    alpha=1.0
                )
                
                pos_idx += 1
            
            bar_idx += 1
    
    # Get y-axis limits right after plotting bars (before legend/other elements)
    ylim = ax.get_ylim()
    y_range = ylim[1] - ylim[0]
    
    # Adjust y-axis slightly to accommodate group labels
    new_ylim_top = ylim[1] + y_range * GROUP_LABEL_Y_OFFSET
    ax.set_ylim(ylim[0], new_ylim_top)
    
    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = []
    for vc_idx, vc_label in enumerate(VC_LABELS):
        vc_label_renamed = LABEL_RENAMES.get(vc_label, vc_label)
        legend_elements.append(
            Patch(facecolor=VC_COLORS[vc_idx], 
                  alpha=1.0,
                  label=vc_label_renamed)
        )
    
    # Set x-axis with rotated labels
    # Position ticks at the center of each subgroup (between the two bars)
    x_tick_positions = [np.mean(positions) for positions in x_positions]
    ax.set_xticks(x_tick_positions)
    ax.set_xticklabels(x_labels, fontsize=TICK_LABEL_FONT_SIZE, 
                       rotation=TICK_LABEL_ROTATION, ha='right')
    
    # Configure axes
    # ax.set_xlabel(X_AXIS_LABEL, fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel(Y_AXIS_LABEL, fontsize=AXIS_LABEL_FONT_SIZE)
    # ax.set_title(PLOT_TITLE, fontsize=TITLE_FONT_SIZE)
    
    # Configure grid
    if GRID_ENABLED:
        ax.grid(True, alpha=GRID_ALPHA, linestyle=GRID_LINESTYLE, axis=GRID_AXIS)
    
    # Configure legend above the plot
    ax.legend(handles=legend_elements, 
              loc='lower center',
              bbox_to_anchor=(LEGEND_BBOX_TO_ANCHOR[0], LEGEND_BBOX_TO_ANCHOR[1] + LEGEND_BBOX_Y_OFFSET),
              frameon=LEGEND_FRAME_ON, framealpha=LEGEND_FRAME_ALPHA,
              fontsize=LEGEND_FONT_SIZE, ncol=2)
    
    # Position labels right above the top of the plot bars
    # Use a small offset in data coordinates
    label_y_position = ylim[1] + y_range * 0.02  # Very small offset, just above the top
    
    # Add major group labels right above the plot (after y-axis is set)
    for major_center, major_label in x_major_group_positions:
        ax.text(major_center, label_y_position, major_label,
                ha='center', va='bottom', fontsize=GROUP_LABEL_FONT_SIZE,
                weight='bold')
    
    # Adjust layout to accommodate legend above
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space at top for legend
    
    # Save or display
    if OUTPUT_FILE:
        plt.savefig(OUTPUT_FILE, dpi=600, bbox_inches='tight')
        print(f"Plot saved to {OUTPUT_FILE}")
    else:
        plt.show()

if __name__ == '__main__':
    main()
