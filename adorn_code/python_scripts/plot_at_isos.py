#!/usr/bin/env python3
"""
Plot at_isos.csv as two clustered bar charts (SC and average hops).
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator, NullFormatter
import numpy as np

CSV_PATH = "files/paper_results/at_isos.csv"
OUT_SC = "./files/paper_results/at_isos_sc.png"
OUT_AVGHOPS = "./files/paper_results/at_isos_avghops.png"
OUT_COMBINED = "./files/paper_results/at_isos_combined.png"

# If True, plot 1/x for all throughput (SC) values x.
INVERSE_THROUGHPUT = True

# If False, omit the reference/Topology bar; group and bar spacing adjusted accordingly.
PLOT_REFERENCE = False #True

# Spacing between groups, as a multiple of bar width (e.g. 1.5 or 2)
GROUP_GAP_BAR_MULTIPLE = 1.5
X_PAD = 0.2  # padding beyond bars for xlim (avoid clipping)

# Figure size
FIG_WIDTH = 8
FIG_HEIGHT = 2.75

# Y-axis limits (ymin, ymax) per plot
SC_YLIM = (0.975, 1.25)
AVGHOPS_YLIM = (0.95, 1.075)

# Y-axis minor tick increment (left = SC, right = avg hops)
SC_Y_MINOR_TICK_INTERVAL = 0.05
AVGHOPS_Y_MINOR_TICK_INTERVAL = 0.025

# Font sizes
X_TICKS_FONT_SIZE = 16
Y_TICKS_FONT_SIZE = 16
X_AXIS_LABEL_FONT_SIZE = 15
Y_AXIS_LABEL_FONT_SIZE = 16
LEGEND_FONT_SIZE = 15

# Columns: 0=group, 1=alg, 6=~SC/throughput (rel unconstrained), 7=avg hops (rel unconstrained)
COL_GROUP, COL_ALG = 0, 1
COL_SC, COL_AVGHOPS = 4,5 #6, 7

# Color mapping: Topology = tab:red, then 4 tableau colors for algs
ALG_COLORS = {
    "Topology": "tab:red",
    "Unconstrained": "tab:blue",
    "CPL": "tab:orange",
    "APL": "tab:green",
    "Random": "tab:purple",
}


def load_data():
    df = pd.read_csv(CSV_PATH, header=None, skiprows=3)
    # Forward-fill group
    df[COL_GROUP] = df[COL_GROUP].replace("", np.nan).ffill()
    # First row has no group; use "reference" or similar
    df.loc[df[COL_GROUP].isna(), COL_GROUP] = "reference"
    df[COL_GROUP] = df[COL_GROUP].fillna("reference")
    # Ensure numeric
    df[COL_SC] = pd.to_numeric(df[COL_SC], errors="coerce")
    df[COL_AVGHOPS] = pd.to_numeric(df[COL_AVGHOPS], errors="coerce")
    if not PLOT_REFERENCE:
        df = df[df[COL_GROUP] != "reference"].copy().reset_index(drop=True)
    return df


def _legend_handles():
    algs = ["Topology", "Unconstrained", "CPL", "APL", "Random"]
    if not PLOT_REFERENCE:
        algs = [a for a in algs if a != "Topology"]
    return [
        mpatches.Patch(facecolor=ALG_COLORS[a], label=a, edgecolor="none")
        for a in algs
    ]


def draw_clustered(ax, df, value_col, ylabel=None, draw_legend=True, ylim=None, y_minor_tick_interval=None):
    """Draw clustered bars on ax. Returns x_base (right edge of last group) for xlim."""
    groups = df[COL_GROUP].unique().tolist()
    bar_width = 0.18
    bar_step = bar_width * 1.2
    group_gap = GROUP_GAP_BAR_MULTIPLE * bar_width
    x_base = 0.0
    group_centers = []

    for g_idx, grp in enumerate(groups):
        subset = df[df[COL_GROUP] == grp].reset_index(drop=True)
        n_bars = len(subset)
        total_span = (n_bars - 1) * bar_step + bar_width
        x_start = x_base

        for i in range(n_bars):
            alg = subset.iloc[i][COL_ALG].strip()
            val = subset.iloc[i][value_col]
            x = x_start + i * bar_step
            facecolor = ALG_COLORS.get(alg, "tab:gray")
            ax.bar(x, val, bar_width, color=facecolor, edgecolor="none")

        group_centers.append(x_base + total_span / 2)
        x_base += total_span
        if g_idx < len(groups) - 1:
            x_base += group_gap

    ax.set_xticks(group_centers)
    ax.set_xticklabels(
        [str(g) for g in groups],
        fontsize=X_AXIS_LABEL_FONT_SIZE,
        rotation=0,
        ha="center",
    )
    ax.tick_params(axis="x", labelsize=X_AXIS_LABEL_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=Y_TICKS_FONT_SIZE)
    ax.set_ylabel(ylabel or value_col, fontsize=Y_AXIS_LABEL_FONT_SIZE)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if y_minor_tick_interval is not None:
        ax.yaxis.set_minor_locator(MultipleLocator(y_minor_tick_interval))
        ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="both", alpha=0.3)
    if draw_legend:
        handles = _legend_handles()
        ax.legend(
            handles=handles,
            loc="upper right",
            ncol=2 ,#len(handles),
            fontsize=LEGEND_FONT_SIZE,
        )
    return x_base


def plot_single(df, value_col, out_path, ylabel=None, ylim=None):
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    if ylim is None:
        ylim = SC_YLIM if value_col == COL_SC else AVGHOPS_YLIM
    y_minor = SC_Y_MINOR_TICK_INTERVAL if value_col == COL_SC else AVGHOPS_Y_MINOR_TICK_INTERVAL
    x_base = draw_clustered(ax, df, value_col, ylabel, draw_legend=True, ylim=ylim, y_minor_tick_interval=y_minor)
    ax.set_xlim(-X_PAD, x_base + X_PAD)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_combined(df):
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(FIG_WIDTH, FIG_HEIGHT))
    sc_ylabel = "Relative Channel Load" if INVERSE_THROUGHPUT else "Relative Throughput"

    sc_ylim = None if INVERSE_THROUGHPUT else SC_YLIM
    if sc_ylim is None:
        sc_ylim = SC_YLIM
    x_base = draw_clustered(
        ax_l,
        df,
        COL_SC,
        ylabel=sc_ylabel,
        draw_legend=False,
        ylim=sc_ylim,
        y_minor_tick_interval=SC_Y_MINOR_TICK_INTERVAL,
    )
    draw_clustered(
        ax_r,
        df,
        COL_AVGHOPS,
        ylabel="Relative Average Hops",
        draw_legend=False,
        ylim=AVGHOPS_YLIM,
        y_minor_tick_interval=AVGHOPS_Y_MINOR_TICK_INTERVAL,
    )
    ax_r.yaxis.tick_right()
    ax_r.yaxis.set_label_position("right")
    for ax in (ax_l, ax_r):
        ax.set_xlim(-X_PAD, x_base + X_PAD)
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    handles = _legend_handles()
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.875),
        ncol=2,#len(handles),
        fontsize=LEGEND_FONT_SIZE,
        frameon=True,
        columnspacing=1.0
    )
    fig.savefig(OUT_COMBINED, dpi=300, bbox_inches="tight")
    print(f"Wrote to {OUT_COMBINED}")
    plt.close(fig)


def main():
    df = load_data()
    if INVERSE_THROUGHPUT:
        df = df.copy()
        df[COL_SC] = 1.0 / df[COL_SC]
    sc_ylabel = "1 / Relative Maximum Channel Load" if INVERSE_THROUGHPUT else "Relative Maximum Channel Load"
    sc_ylim = None if INVERSE_THROUGHPUT else SC_YLIM
    plot_single(df, COL_SC, OUT_SC, ylabel=sc_ylabel, ylim=sc_ylim)
    plot_single(df, COL_AVGHOPS, OUT_AVGHOPS, ylabel="Relative Average Hops")
    plot_combined(df)


if __name__ == "__main__":
    main()
#  plot_combined(df)


if __name__ == "__main__":
    main()
