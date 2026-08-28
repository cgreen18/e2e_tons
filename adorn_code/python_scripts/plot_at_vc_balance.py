import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ============================================================================
# CONFIGURABLE PARAMETERS
# ============================================================================

# File path
CSV_FILE_PATH = 'files/at_vc_balance.csv'

# Only plot symmetric results (no symmetry grouping)
SYMMETRY_FILTER = 'symmetric'  # Only rows with this symmetry value are plotted

# Turn prioritization as outer groups (labeled above)
MAJOR_GROUP_LABELS = ['cpl', 'apl', 'topo']  # Subgroups become outer groups

# Balancing states (labeled below each bar, rotated)
BALANCING_STATES = ['unbalanced', 'load balanced']
BALANCING_LABEL_ROTATION = 45  # Degrees for "Unbalanced" / "Load Balanced" labels below bars
BALANCING_LABEL_Y_AXES = -0.12  # Y position for balancing labels (axes coords, < 0 = below plot)

# VC labels and colors
VC_LABELS = ['VC 0', 'VC 1']
VC_COLORS = ['tab:blue', 'tab:red']  # Blue for VC 0, Red for VC 1

# Rename dictionary for all labels (original: display_name)
LABEL_RENAMES = {
    # Major groups
    'non-symmetric': 'Non-Symmetric',  # Change as needed
    'symmetric': 'Symmetric',  # Change as needed
    # Subgroups
    'cpl': 'CPL',  # Change as needed
    'apl': 'APL',  # Change as needed
    'topo': 'Random',  # Change as needed
    # Balancing states
    'unbalanced': 'Unbalanced',  # Change as needed
    'load balanced': 'Load Balanced',  # Change as needed
    # VC labels
    'VC 0': 'VC 0',  # Change as needed
    'VC 1': 'VC 1',  # Change as needed
}

# Hatching pattern for unbalanced bars
UNBALANCED_HATCH = '///'  # Options: '/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*', '///', '\\\\\\', etc.

# Bar chart configuration
BAR_WIDTH = 0.2  # Width of each bar
BAR_SPACING = 0.05  # Spacing between bars within a group
SUBGROUP_SPACING = 0.3  # Spacing between turn-prioritization groups
MAJOR_GROUP_SPACING = 0.5  # Spacing between major groups (unused when only one level)

# Figure size
FIGURE_WIDTH = 5
FIGURE_HEIGHT = 4

# Font sizes
TITLE_FONT_SIZE = 16
AXIS_LABEL_FONT_SIZE = 14
TICK_LABEL_FONT_SIZE = 12
LEGEND_FONT_SIZE = 11
GROUP_LABEL_FONT_SIZE = 13
GROUP_LABEL_Y_OFFSET = 0.15  # Offset above plot (as fraction of y-axis range) - for y-axis extension

# Legend configuration
LEGEND_LOCATION = 'upper center'  # Options: 'best', 'upper right', 'lower left', etc.
LEGEND_BBOX_TO_ANCHOR = (0.5, 1.0)  # (x, y) position for legend - (0.5, 1.0) is center top of axes
LEGEND_BBOX_Y_OFFSET = 0.02  # Additional offset above plot (in axes coordinates, 0-1)
LEGEND_FRAME_ON = True
LEGEND_FRAME_ALPHA = 0.9

# Grid configuration
GRID_ENABLED = True
GRID_ALPHA = 0.3
GRID_LINESTYLE = '--'
GRID_AXIS = 'y'  # 'x', 'y', or 'both'

# Axes labels
X_AXIS_LABEL = 'Turn Prioritization'
Y_AXIS_LABEL = 'Relative Hops per VC'

# Title
PLOT_TITLE = 'AT VC Balance'

# Output file (None to display, or provide path to save)
OUTPUT_FILE = 'files/at_vc_balance_plot.png'

# Edge color for bars (None for no edge, or specify color)
BAR_EDGE_COLOR = None
BAR_EDGE_WIDTH = 0.5

# ============================================================================
# MAIN SCRIPT
# ============================================================================

def main():
    # Read CSV file and keep only symmetric rows
    df = pd.read_csv(CSV_FILE_PATH)
    df = df[df['symmetry'] == SYMMETRY_FILTER].copy()
    
    # Apply renames to data
    df['turn prioritization'] = df['turn prioritization'].map(lambda x: LABEL_RENAMES.get(x, x))
    df['balancing'] = df['balancing'].map(lambda x: LABEL_RENAMES.get(x, x))
    
    # Prepare data structure: {turn_prioritization: {balancing: {VC: value}}}
    data = {}
    for _, row in df.iterrows():
        turn_prior = row['turn prioritization']
        balancing = row['balancing']
        if turn_prior not in data:
            data[turn_prior] = {}
        if balancing not in data[turn_prior]:
            data[turn_prior][balancing] = {}
        data[turn_prior][balancing]['VC 0'] = row['VC 0']
        data[turn_prior][balancing]['VC 1'] = row['VC 1']
    
    # One major group per turn prioritization; each has 4 bars
    num_major_groups = len(MAJOR_GROUP_LABELS)
    num_bars_per_group = len(BALANCING_STATES) * len(VC_LABELS)  # 4 bars
    
    x_positions = []  # list of 4 bar x positions per group
    x_major_group_positions = []  # (center_x, label) for labels above
    
    current_x = 0
    for major_group in MAJOR_GROUP_LABELS:
        major_group_renamed = LABEL_RENAMES.get(major_group, major_group)
        group_center = current_x
        
        bar_positions = []
        bar_offset = -(BAR_WIDTH * 2 + BAR_SPACING) / 2
        for _ in BALANCING_STATES:
            for _ in VC_LABELS:
                bar_positions.append(group_center + bar_offset)
                bar_offset += BAR_WIDTH + BAR_SPACING
        
        x_positions.append(bar_positions)
        x_major_group_positions.append((group_center, major_group_renamed))
        
        current_x += num_bars_per_group * BAR_WIDTH + (num_bars_per_group - 1) * BAR_SPACING + SUBGROUP_SPACING
    
    # Create figure
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    
    # Set x-axis limits so the plot area doesn't expand to include rotated labels
    x_min = min(pos for positions in x_positions for pos in positions) - BAR_WIDTH
    x_max = max(pos for positions in x_positions for pos in positions) + BAR_WIDTH
    ax.set_xlim(x_min, x_max)
    
    # Plot bars
    for bar_idx, turn_prior in enumerate(MAJOR_GROUP_LABELS):
        turn_prior_renamed = LABEL_RENAMES.get(turn_prior, turn_prior)
        positions = x_positions[bar_idx]
        pos_idx = 0
        for balancing in BALANCING_STATES:
            balancing_renamed = LABEL_RENAMES.get(balancing, balancing)
            for vc_idx, vc_label in enumerate(VC_LABELS):
                value = data[turn_prior_renamed][balancing_renamed][vc_label]
                is_unbalanced = balancing == 'unbalanced'
                ax.bar(
                    positions[pos_idx],
                    value,
                    width=BAR_WIDTH,
                    color=VC_COLORS[vc_idx],
                    edgecolor=BAR_EDGE_COLOR,
                    linewidth=BAR_EDGE_WIDTH,
                    hatch=UNBALANCED_HATCH if is_unbalanced else None,
                    alpha=1.0,
                )
                pos_idx += 1
    
    # Y-axis: extend slightly for labels above
    ylim = ax.get_ylim()
    y_range = ylim[1] - ylim[0]
    ax.set_ylim(ylim[0], ylim[1] + y_range * GROUP_LABEL_Y_OFFSET)
    label_y_position = ylim[1] + y_range * 0.02
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = []
    for balancing in BALANCING_STATES:
        balancing_renamed = LABEL_RENAMES.get(balancing, balancing)
        is_unbalanced = balancing == 'unbalanced'
        for vc_idx, vc_label in enumerate(VC_LABELS):
            vc_label_renamed = LABEL_RENAMES.get(vc_label, vc_label)
            legend_elements.append(
                Patch(
                    facecolor=VC_COLORS[vc_idx],
                    hatch=UNBALANCED_HATCH if is_unbalanced else None,
                    alpha=1.0,
                    label=f'{balancing_renamed} {vc_label_renamed}',
                )
            )
    ax.legend(
        handles=legend_elements,
        loc='lower center',
        bbox_to_anchor=(LEGEND_BBOX_TO_ANCHOR[0], LEGEND_BBOX_TO_ANCHOR[1] + LEGEND_BBOX_Y_OFFSET),
        frameon=LEGEND_FRAME_ON,
        framealpha=LEGEND_FRAME_ALPHA,
        fontsize=LEGEND_FONT_SIZE,
        ncol=4,
    )
    
    # Turn prioritization labels above the plot
    for major_center, major_label in x_major_group_positions:
        ax.text(major_center, label_y_position, major_label,
                ha='center', va='bottom', fontsize=GROUP_LABEL_FONT_SIZE, weight='bold')
    
    # Rotated "Unbalanced" / "Load Balanced" below each bar; last letter under the bar (ha='right')
    # x in data coords, y in axes coords so labels sit below the plot
    trans = ax.get_xaxis_transform()
    for bar_idx, positions in enumerate(x_positions):
        pos_idx = 0
        for balancing in BALANCING_STATES:
            balancing_renamed = LABEL_RENAMES.get(balancing, balancing)
            for _ in VC_LABELS:
                x_pos = positions[pos_idx]
                ax.text(x_pos, BALANCING_LABEL_Y_AXES, balancing_renamed,
                        ha='right', va='top', fontsize=TICK_LABEL_FONT_SIZE,
                        rotation=BALANCING_LABEL_ROTATION, rotation_mode='anchor',
                        transform=trans)
                pos_idx += 1
    
    # Hide default x-axis ticks and tick labels
    ax.set_xticks([])
    ax.set_xlabel(None)
    ax.set_ylabel(Y_AXIS_LABEL, fontsize=AXIS_LABEL_FONT_SIZE)
    
    if GRID_ENABLED:
        ax.grid(True, alpha=GRID_ALPHA, linestyle=GRID_LINESTYLE, axis=GRID_AXIS)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if OUTPUT_FILE:
        plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {OUTPUT_FILE}")
    else:
        plt.show()

if __name__ == '__main__':
    main()
