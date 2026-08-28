#!/usr/bin/env python3
"""
Plot collective communication analytical results from ag_ar and a2a CSV files.
Three stacked subplots: AllGather (ag), AllReduce (ar), All-to-All (a2a).
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from pathlib import Path

# -----------------------------------------------------------------------------
# Style constants (all caps)
# -----------------------------------------------------------------------------
FONT_SIZE = 16
FONT_SIZE_TITLE = 18
FONT_SIZE_LABEL = 18
FONT_SIZE_LEGEND = 13
FONT_SIZE_LEGEND_A2A = 10
LEGEND_LOC = "best"
LEGEND_LOC_AG_AR = "lower right"  # AllGather and AllReduce
LINE_WIDTH = 2
LINE_WIDTH_DASHED = 2
MARKER_SIZE = 7
MARKER_SIZE_STAR = 10
FIGURE_WIDTH = 8
FIGURE_HEIGHT = 6
SUBPLOT_HSPACE = 0.35
DOT_MARKER = "o"
STAR_MARKER = "*"
LINESTYLE_SOLID = "-"
LINESTYLE_DASHED = "--"

# Topology type order and labels
TOPOLOGY_TYPES = ["pt", "pdtt", "asc"]
TOPOLOGY_LABELS = {"pt": "PT", "pdtt": "PDTT", "asc": "TONS"}

# Paths (relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
A2A_CSV = PROJECT_ROOT / "files" / "a2a_results_final.csv"
AG_AR_CSV = PROJECT_ROOT / "files" / "ag_ar_results_final.csv"

# Column names (ag_ar uses "link-based utilization (%)" for theoretical)
COL_UTIL = "utilization (%)"
COL_THEORETICAL_AG_AR = "link-based utilization (%)"  # in ag_ar_results_final.csv
COL_THEORETICAL_A2A = "theoretical utilization (%)"  # in a2a_results_final.csv

# Single theoretical line for AllGather/AllReduce (all 100%)
THEORETICAL_UTIL_PCT = 100.0
THEORETICAL_LINE_COLOR = "gray"
THEORETICAL_LABEL = "Theoretical"

# Y-axis limits and ticks: AllGather and AllReduce
AG_AR_YMIN = 96
AG_AR_YMAX = 100
AG_AR_MAJOR_TICK = 2
AG_AR_MINOR_TICK = 1

# Y-axis limits and ticks: All-to-All
A2A_YMIN = 0
A2A_YMAX = 40
A2A_MAJOR_TICK = 10
A2A_MINOR_TICK = 5

# All-to-All legend
A2A_LEGEND_NCOL = 3


def _topology_type_from_name(name):
    """Return topology type 'pt', 'pdtt', or 'asc' from ag_ar name column."""
    if name.startswith("pt_"):
        return "pt"
    if name.startswith("pdtt_"):
        return "pdtt"
    if name.startswith("asc"):
        return "asc"
    return None


def _topology_type_from_a2a(topology):
    """Return topology type from a2a Topology column (e.g. pt_4x4x8 -> pt)."""
    if topology.startswith("pt_"):
        return "pt"
    if topology.startswith("pdtt_"):
        return "pdtt"
    if topology.startswith("asc"):
        return "asc"
    return None


def _get_a2a_keys(a2a):
    """Set of (Size, topology_type) that exist in a2a (used as key for what to plot)."""
    keys = set()
    for _, row in a2a.iterrows():
        tt = _topology_type_from_a2a(row["Topology"])
        if tt is not None:
            keys.add((int(row["Size"]), tt))
    return keys


def _one_row_per_size_type(df_ag_ar, coll_comm, a2a_keys):
    """
    From ag_ar, keep one row per (size, topology_type) that is in a2a_keys.
    Returns DataFrame with columns size, topology_type, utilization (%), theoretical utilization (%).
    """
    sub = df_ag_ar[df_ag_ar["coll_comm"] == coll_comm].copy()
    sub["topology_type"] = sub["name"].map(_topology_type_from_name)
    sub = sub[sub["topology_type"].notna()]
    sub["size"] = sub["size"].astype(int)

    # Keep only (size, topology_type) that appear in a2a
    rows = []
    for (sz, tt), group in sub.groupby(["size", "topology_type"]):
        if (sz, tt) not in a2a_keys:
            continue
        first = group.iloc[0]
        rows.append({
            "size": sz,
            "topology_type": tt,
            COL_UTIL: first[COL_UTIL],
            COL_THEORETICAL_AG_AR: first[COL_THEORETICAL_AG_AR],
        })
    return pd.DataFrame(rows)


def _prepare_a2a(a2a):
    """Add topology_type and return sorted; handle empty utilization."""
    a2a = a2a.copy()
    a2a["Size"] = pd.to_numeric(a2a["Size"], errors="coerce").astype("Int64")
    a2a["topology_type"] = a2a["Topology"].map(_topology_type_from_a2a)
    a2a[COL_UTIL] = pd.to_numeric(a2a[COL_UTIL], errors="coerce")
    a2a[COL_THEORETICAL_A2A] = pd.to_numeric(
        a2a[COL_THEORETICAL_A2A], errors="coerce"
    )
    return a2a


def main():
    a2a = pd.read_csv(A2A_CSV)
    ag_ar = pd.read_csv(AG_AR_CSV)

    a2a_keys = _get_a2a_keys(a2a)
    a2a_plot = _prepare_a2a(a2a)

    # Tableau palette (matplotlib tab10 first 6; we use 3 for pt, pdtt, asc)
    tableau_colors = plt.cm.tab10.colors
    COLOR_BY_TYPE = {
        "pt": tableau_colors[0],
        "pdtt": tableau_colors[1],
        "asc": tableau_colors[2],
    }

    fig, axes = plt.subplots(3, 1, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), sharex=True)
    fig.subplots_adjust(hspace=SUBPLOT_HSPACE)

    # ----- Subplot 0: AllGather (ag) -----
    ax = axes[0]
    df_ag = _one_row_per_size_type(ag_ar, "ag", a2a_keys)
    for tt in TOPOLOGY_TYPES:
        d = df_ag[df_ag["topology_type"] == tt].sort_values("size")
        if d.empty:
            continue
        c = COLOR_BY_TYPE[tt]
        d = d.dropna(subset=[COL_UTIL])
        if not d.empty:
            if tt == "pdtt":
                ax.scatter(
                    d["size"],
                    d[COL_UTIL],
                    color=c,
                    s=MARKER_SIZE ** 2,
                    marker=DOT_MARKER,
                    label=f"{TOPOLOGY_LABELS[tt]} utilization",
                    zorder=3,
                )
            else:
                ax.plot(
                    d["size"],
                    d[COL_UTIL],
                    color=c,
                    linestyle=LINESTYLE_SOLID,
                    linewidth=LINE_WIDTH,
                    marker=DOT_MARKER,
                    markersize=MARKER_SIZE,
                    label=f"{TOPOLOGY_LABELS[tt]} utilization",
                )
    ax.set_title("AllGather", fontsize=FONT_SIZE_TITLE)
    ax.legend(loc=LEGEND_LOC_AG_AR, fontsize=FONT_SIZE_LEGEND)
    ax.set_ylim(AG_AR_YMIN, AG_AR_YMAX)
    ax.yaxis.set_major_locator(mtick.MultipleLocator(AG_AR_MAJOR_TICK))
    ax.yaxis.set_minor_locator(mtick.MultipleLocator(AG_AR_MINOR_TICK))
    ax.tick_params(axis="both", labelsize=FONT_SIZE)
    ax.grid(True, alpha=0.3, which="major")
    ax.grid(True, alpha=0.3, which="minor")

    # ----- Subplot 1: AllReduce (ar) -----
    ax = axes[1]
    df_ar = _one_row_per_size_type(ag_ar, "ar", a2a_keys)
    for tt in TOPOLOGY_TYPES:
        d = df_ar[df_ar["topology_type"] == tt].sort_values("size")
        if d.empty:
            continue
        c = COLOR_BY_TYPE[tt]
        d = d.dropna(subset=[COL_UTIL])
        if not d.empty:
            if tt == "pdtt":
                ax.scatter(
                    d["size"],
                    d[COL_UTIL],
                    color=c,
                    s=MARKER_SIZE ** 2,
                    marker=DOT_MARKER,
                    label=f"{TOPOLOGY_LABELS[tt]} utilization",
                    zorder=3,
                )
            else:
                ax.plot(
                    d["size"],
                    d[COL_UTIL],
                    color=c,
                    linestyle=LINESTYLE_SOLID,
                    linewidth=LINE_WIDTH,
                    marker=DOT_MARKER,
                    markersize=MARKER_SIZE,
                    label=f"{TOPOLOGY_LABELS[tt]} utilization",
                )
    ax.set_ylabel("Link Bandwidth Utilization (%)", fontsize=FONT_SIZE_LABEL)
    ax.set_title("AllReduce", fontsize=FONT_SIZE_TITLE)
    ax.legend(loc=LEGEND_LOC_AG_AR, fontsize=FONT_SIZE_LEGEND)
    ax.set_ylim(AG_AR_YMIN, AG_AR_YMAX)
    ax.yaxis.set_major_locator(mtick.MultipleLocator(AG_AR_MAJOR_TICK))
    ax.yaxis.set_minor_locator(mtick.MultipleLocator(AG_AR_MINOR_TICK))
    ax.tick_params(axis="both", labelsize=FONT_SIZE)
    ax.grid(True, alpha=0.3, which="major")
    ax.grid(True, alpha=0.3, which="minor")

    # ----- Subplot 2: All-to-All (a2a); PDTT as scatter -----
    ax = axes[2]
    for tt in TOPOLOGY_TYPES:
        d = a2a_plot[a2a_plot["topology_type"] == tt].dropna(
            subset=[COL_UTIL, "Size"]
        ).sort_values("Size")
        if d.empty:
            continue
        c = COLOR_BY_TYPE[tt]
        if tt == "pdtt":
            # PDTT: scatter only (no line through points)
            ax.scatter(
                d["Size"],
                d[COL_UTIL],
                color=c,
                s=MARKER_SIZE ** 2,
                marker=DOT_MARKER,
                label=f"{TOPOLOGY_LABELS[tt]} utilization",
                zorder=3,
            )
            d2 = a2a_plot[
                (a2a_plot["topology_type"] == tt)
                & (a2a_plot[COL_THEORETICAL_A2A].notna())
            ].dropna(subset=["Size"])
            if not d2.empty:
                ax.scatter(
                    d2["Size"],
                    d2[COL_THEORETICAL_A2A],
                    color=c,
                    s=MARKER_SIZE_STAR ** 2,
                    marker=STAR_MARKER,
                    label=f"{TOPOLOGY_LABELS[tt]} theoretical",
                    zorder=3,
                )
        else:
            if not d.empty:
                ax.plot(
                    d["Size"],
                    d[COL_UTIL],
                    color=c,
                    linestyle=LINESTYLE_SOLID,
                    linewidth=LINE_WIDTH,
                    marker=DOT_MARKER,
                    markersize=MARKER_SIZE,
                    label=f"{TOPOLOGY_LABELS[tt]} utilization",
                )
            d2 = a2a_plot[
                (a2a_plot["topology_type"] == tt)
                & (a2a_plot[COL_THEORETICAL_A2A].notna())
            ].dropna(subset=["Size"]).sort_values("Size")
            if not d2.empty:
                ax.plot(
                    d2["Size"],
                    d2[COL_THEORETICAL_A2A],
                    color=c,
                    linestyle=LINESTYLE_DASHED,
                    linewidth=LINE_WIDTH_DASHED,
                    marker=STAR_MARKER,
                    markersize=MARKER_SIZE_STAR,
                    label=f"{TOPOLOGY_LABELS[tt]} theoretical",
                )
    ax.set_xlabel("Number of nodes", fontsize=FONT_SIZE_LABEL)
    ax.set_title("All-to-All", fontsize=FONT_SIZE_TITLE)
    ax.legend(loc=LEGEND_LOC, fontsize=FONT_SIZE_LEGEND_A2A, ncol=A2A_LEGEND_NCOL)
    ax.set_ylim(A2A_YMIN, A2A_YMAX)
    ax.yaxis.set_major_locator(mtick.MultipleLocator(A2A_MAJOR_TICK))
    ax.yaxis.set_minor_locator(mtick.MultipleLocator(A2A_MINOR_TICK))
    ax.tick_params(axis="both", labelsize=FONT_SIZE)
    ax.grid(True, alpha=0.3, which="major")
    ax.grid(True, alpha=0.3, which="minor")

    plt.tight_layout()
    out_path = PROJECT_ROOT / "files" / "coll_comm_analytical_plot.png"
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
