import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ============================================================================
# CONFIGURABLE PARAMETERS
# ============================================================================

# File path
CSV_FILE_PATH = 'files/analytical_size_sweep.csv'

# Column renaming
COLUMN_RENAMES = {
    'asc': 'TONS LP',
    'pt': 'PT',
    'pdtt': 'PDTT',
    'theoretical bound': 'Theoretical Limit'
}

# Plot order (list of column names after renaming)
PLOT_ORDER = ['PT', 'PDTT', 'TONS LP', 'Theoretical Limit']

# Scatter plot columns (columns that should be plotted as scatter instead of line)
SCATTER_COLUMNS = ['PDTT']

# Axes labels
X_AXIS_LABEL = 'Number of Nodes'
Y_AXIS_LABEL = 'Per Source Throughput'

# Figure size
FIGURE_WIDTH = 6
FIGURE_HEIGHT = 3

# Font sizes
TITLE_FONT_SIZE = 16
AXIS_LABEL_FONT_SIZE = 14
TICK_LABEL_FONT_SIZE = 12
LEGEND_FONT_SIZE = 12

# Legend configuration
LEGEND_LOCATION = 'best'  # Options: 'best', 'upper right', 'lower left', etc.
LEGEND_FRAME_ON = True
LEGEND_FRAME_ALPHA = 0.9

# Line styles and markers
LINE_STYLES = {
    'PT': '-',
    'PDTT': 'o',  # Marker for scatter
    'TONS LP': '-',
    'Theoretical Limit': '--'
}

# Colors (can use named colors or hex codes)
COLORS = {
    'PT': 'tab:blue',  # Blue
    'PDTT': 'tab:orange',  # Orange
    'TONS LP': 'tab:green',  # Green
    'Theoretical Limit': 'tab:red'  # Red
}

# Line widths
LINE_WIDTH = 2
MARKER_SIZE = 8

# Marker style for line plots
LINE_MARKER = 'o'  # Circle/dot marker

# Axis limits
X_AXIS_MIN = 0
X_AXIS_MAX = 8300

# Grid configuration
GRID_ENABLED = True
GRID_ALPHA = 0.3
GRID_LINESTYLE = '--'

# Title
PLOT_TITLE = 'Analytical Size Sweep'

# Output file (None to display, or provide path to save)
OUTPUT_FILE ='files/analytical_size_sweep_plot.png'

# ============================================================================
# MAIN SCRIPT
# ============================================================================

def main():
    # Read CSV file
    df = pd.read_csv(CSV_FILE_PATH)
    
    # Rename columns
    df = df.rename(columns=COLUMN_RENAMES)
    
    # Get the size column (x-axis)
    size_col = df.columns[0]  # Assuming first column is size
    x_data = df[size_col]
    
    # Create figure
    plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    
    # Plot each column in the specified order
    for col_name in PLOT_ORDER:
        if col_name not in df.columns:
            print(f"Warning: Column '{col_name}' not found in data. Skipping.")
            continue
        
        y_data = df[col_name]
        
        # Check if this column should be a scatter plot
        if col_name in SCATTER_COLUMNS:
            # Filter out NaN values for scatter plot
            mask = ~pd.isna(y_data)
            plt.scatter(
                x_data[mask],
                y_data[mask],
                label=col_name,
                color=COLORS.get(col_name, None),
                marker=LINE_STYLES.get(col_name, 'o'),
                s=MARKER_SIZE ** 2,
                zorder=3  # Put scatter points on top
            )
        else:
            # Line plot
            plt.plot(
                x_data,
                y_data,
                label=col_name,
                color=COLORS.get(col_name, None),
                linestyle=LINE_STYLES.get(col_name, '-'),
                linewidth=LINE_WIDTH,
                marker=LINE_MARKER,
                markersize=MARKER_SIZE
            )
    
    # Configure axes
    plt.xlabel(X_AXIS_LABEL, fontsize=AXIS_LABEL_FONT_SIZE)
    plt.ylabel(Y_AXIS_LABEL, fontsize=AXIS_LABEL_FONT_SIZE)
    # plt.title(PLOT_TITLE, fontsize=TITLE_FONT_SIZE)
    plt.xlim(X_AXIS_MIN, X_AXIS_MAX)
    
    # Configure ticks
    plt.tick_params(labelsize=TICK_LABEL_FONT_SIZE)
    
    # Configure grid
    if GRID_ENABLED:
        plt.grid(True, alpha=GRID_ALPHA, linestyle=GRID_LINESTYLE)
    
    # Configure legend
    plt.legend(
        loc=LEGEND_LOCATION,
        frameon=LEGEND_FRAME_ON,
        framealpha=LEGEND_FRAME_ALPHA,
        fontsize=LEGEND_FONT_SIZE,
        ncol=2
    )
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or display
    if OUTPUT_FILE:
        plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {OUTPUT_FILE}")
    else:
        plt.show()

if __name__ == '__main__':
    main()
