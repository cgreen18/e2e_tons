"""
Plot AT VC balance: symmetric entries only.
Two major groups (Unbalanced, Load Balanced), each with turn prioritization
subgroups (CPL, APL, Topo), each with two bars (VC 0, VC 1).
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ============================================================================
# CONFIGURABLE PARAMETERS
# ============================================================================

CSV_FILE_PATH = "files/at_vc_balance.csv"
SYMMETRY_FILTER = "symmetric"

# Major groups (balancing); labels shown above the bars
MAJOR_GROUP_LABELS = ["unbalanced", "load balanced"]

# Turn prioritization order within each major group
TURN_PRIORITIZATION_ORDER = ["cpl", "apl", "topo"]

# VC columns and display
VC_COLUMNS = ["VC 0", "VC 1"]
VC_COLORS = ["tab:blue", "tab:red"]  # Blue, Red

# Rename for display (optional)
LABEL_RENAMES = {
    "unbalanced": "Unbalanced",
    "load balanced": "Load Balanced",
    "cpl": "CPL",
    "apl": "APL",
    "topo": "Random",
    "VC 0": "VC 0",
    "VC 1": "VC 1",
}

# Bar layout
BAR_WIDTH = 0.35
BAR_SPACING = 0.05
SUBGROUP_SPACING = 0.4
MAJOR_GROUP_SPACING = 0.6

# Figure
FIGURE_WIDTH = 6
FIGURE_HEIGHT = 2.2

# Axes margins (fraction of figure: 0–1). Use these so the plot area keeps a fixed
# fraction of the figure and does not get compressed when FIGURE_WIDTH is reduced.
SUBPLOT_LEFT = 0.08
SUBPLOT_RIGHT = 0.96
SUBPLOT_BOTTOM = 0.12
SUBPLOT_TOP = 0.92
SUBPLOT_WSPACE = 0.2
SUBPLOT_HSPACE = 0.2

# Fonts
TITLE_FONT_SIZE = 16
AXIS_LABEL_FONT_SIZE = 14
TICK_LABEL_FONT_SIZE = 14
GROUP_LABEL_FONT_SIZE = 13
LEGEND_FONT_SIZE = 13

# Group label position (above bars, as fraction of y range above top of bars)
GROUP_LABEL_Y_OFFSET = 0.15

# Legend (above the plot)
LEGEND_LOCATION = "lower center"
LEGEND_BBOX_TO_ANCHOR = (0.5, 0.96)   # center, top of axes
LEGEND_BBOX_Y_OFFSET = 0.02          # extra offset above (axes coords)
LEGEND_TOP_MARGIN = -0.02             # reduce axes top by this much to fit legend
LEGEND_FRAME_ON = True
LEGEND_FRAME_ALPHA = 1.0

# Grid
GRID_ENABLED = True
GRID_ALPHA = 0.3
GRID_LINESTYLE = "--"
GRID_AXIS = "y"

# Axes
X_AXIS_LABEL = "Turn prioritization"
X_AXIS_LABEL_PAD = 22          # distance of x-axis label from axis (points); increase to lower
Y_AXIS_LABEL = "Relative hops per VC"

# Output
OUTPUT_FILE = "files/at_vc_balance_plot.png"

# Bar appearance
BAR_EDGE_COLOR = None
BAR_EDGE_WIDTH = 0.5

# X-axis tick and label placement (tick and label between and below the two VC bars)
# Labels are drawn manually at the exact midpoint of each VC0–VC1 pair
X_TICK_LABEL_HA = "center"  # horizontal alignment for manual labels
X_TICK_DIRECTION = "out"    # tick marks point out (below axis)
X_TICK_LENGTH = 4           # tick length in points
# Y position for bar labels (CPL, APL, Topo) in axes coords; 0=axis, negative=below
X_LABEL_Y_AXES = -0.08

# ============================================================================
# MAIN
# ============================================================================


def main():
    df = pd.read_csv(CSV_FILE_PATH)
    df = df[df["symmetry"] == SYMMETRY_FILTER].copy()

    # Optional renames for display
    df["balancing"] = df["balancing"].map(lambda x: LABEL_RENAMES.get(x, x))
    df["turn prioritization"] = df["turn prioritization"].map(
        lambda x: LABEL_RENAMES.get(x, x)
    )

    # Data: {balancing: {turn_prior: [vc0_val, vc1_val]}}
    data = {}
    for _, row in df.iterrows():
        bal = row["balancing"]
        tp = row["turn prioritization"]
        if bal not in data:
            data[bal] = {}
        data[bal][tp] = [row[VC_COLUMNS[0]], row[VC_COLUMNS[1]]]

    # Use renamed major group labels for lookup
    major_labels_display = [LABEL_RENAMES.get(g, g) for g in MAJOR_GROUP_LABELS]
    tp_order_display = [LABEL_RENAMES.get(t, t) for t in TURN_PRIORITIZATION_ORDER]

    num_bars_per_subgroup = len(VC_COLUMNS)
    x_positions = []  # list of bar x positions per subgroup
    x_tick_positions = []
    x_tick_labels = []
    major_group_centers = []  # (x_center, label) for labels above bars

    current_x = 0.0
    for major in major_labels_display:
        group_start = current_x
        for tp in tp_order_display:
            # Two bars: VC0, VC1 (x is left edge of each bar)
            w = BAR_WIDTH
            gap = BAR_SPACING
            total = 2 * w + gap
            bar_x = [current_x, current_x + w + gap]
            x_positions.append(bar_x)
            # Tick/label exactly between the two bars: midpoint of bar centers
            center_vc0 = bar_x[0] + w / 2
            center_vc1 = bar_x[1] + w / 2
            x_tick_positions.append((center_vc0 + center_vc1) / 2)
            x_tick_labels.append(tp)
            current_x += total + SUBGROUP_SPACING
        mid = (group_start + current_x - SUBGROUP_SPACING) / 2
        major_group_centers.append((mid, major))
        current_x += MAJOR_GROUP_SPACING

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

    # Plot bars
    idx = 0
    for major in major_labels_display:
        for tp in tp_order_display:
            bar_x = x_positions[idx]
            vals = data[major][tp]
            for i, (x, v) in enumerate(zip(bar_x, vals)):
                ax.bar(
                    x,
                    v,
                    width=BAR_WIDTH,
                    align="edge",  # x is left edge of bar so label midpoint matches visual
                    color=VC_COLORS[i],
                    edgecolor=BAR_EDGE_COLOR,
                    linewidth=BAR_EDGE_WIDTH,
                )
            idx += 1

    # X-axis: ticks at midpoint of each VC0–VC1 pair; labels drawn manually at same midpoint
    ax.set_xticks(x_tick_positions)
    ax.set_xticklabels([])  # no default labels
    ax.tick_params(axis="x", direction=X_TICK_DIRECTION, length=X_TICK_LENGTH, bottom=True, top=False)
    trans = ax.get_xaxis_transform()  # x in data coords, y in axes coords (0=bottom)
    for bar_x, label in zip(x_positions, x_tick_labels):
        midpoint = (bar_x[0] + bar_x[1] + BAR_WIDTH) / 2.0
        ax.text(
            midpoint,
            X_LABEL_Y_AXES,
            label,
            ha=X_TICK_LABEL_HA,
            va="top",
            fontsize=TICK_LABEL_FONT_SIZE,
            transform=trans,
        )
    ax.set_xlabel(X_AXIS_LABEL, fontsize=AXIS_LABEL_FONT_SIZE, labelpad=X_AXIS_LABEL_PAD)
    ax.set_ylabel(Y_AXIS_LABEL, fontsize=AXIS_LABEL_FONT_SIZE)

    # Y-axis: extend slightly and add major group labels above the bars
    ylim = ax.get_ylim()
    y_range = ylim[1] - ylim[0]
    ax.set_ylim(ylim[0], ylim[1] + y_range * GROUP_LABEL_Y_OFFSET)
    label_y = ylim[1] + y_range * 0.02
    for x_center, label in major_group_centers:
        ax.text(
            x_center,
            label_y,
            label,
            ha="center",
            va="bottom",
            fontsize=GROUP_LABEL_FONT_SIZE,
            weight="bold",
        )

    # X limits so plot is not thin
    all_x = [x for bar_x in x_positions for x in bar_x]
    ax.set_xlim(min(all_x) - BAR_WIDTH, max(all_x) + BAR_WIDTH + MAJOR_GROUP_SPACING)

    # Legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=VC_COLORS[0], label=LABEL_RENAMES.get(VC_COLUMNS[0], VC_COLUMNS[0])),
        Patch(facecolor=VC_COLORS[1], label=LABEL_RENAMES.get(VC_COLUMNS[1], VC_COLUMNS[1])),
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(LEGEND_BBOX_TO_ANCHOR[0], LEGEND_BBOX_TO_ANCHOR[1] + LEGEND_BBOX_Y_OFFSET),
        frameon=LEGEND_FRAME_ON,
        framealpha=LEGEND_FRAME_ALPHA,
        fontsize=LEGEND_FONT_SIZE,
        ncol=2,
    )

    if GRID_ENABLED:
        ax.grid(True, alpha=GRID_ALPHA, linestyle=GRID_LINESTYLE, axis=GRID_AXIS)

    # Fixed margins; leave room at top for legend above plot
    fig.subplots_adjust(
        left=SUBPLOT_LEFT,
        right=SUBPLOT_RIGHT,
        bottom=SUBPLOT_BOTTOM,
        top=SUBPLOT_TOP - LEGEND_TOP_MARGIN,
        wspace=SUBPLOT_WSPACE,
        hspace=SUBPLOT_HSPACE,
    )

    if OUTPUT_FILE:
        plt.savefig(OUTPUT_FILE, dpi=600, bbox_inches="tight")
        print(f"Plot saved to {OUTPUT_FILE}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
