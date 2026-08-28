#!/usr/bin/env python3
"""
Plot topology_sweep_results.csv as a line chart (kautz as scatter):
  x-axis: size (number of nodes)
  y-axis: objective value * number of nodes (MCF or AASC objective scaled)
  each series: a topology variant (e.g. genkautz r4, aasc lp 2 trans r4)
  kautz: scatter plot with star markers; others: line plot
"""

import argparse
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Label renaming: when True, apply display names (Genkautz, Kautz, ADORN, Random, Xpander)
RENAME_LABELS = True

# Plot theoretical bound: per-source <= r / log_r(N) as dashed line(s), last in legend
PLOT_THEO = False # True

# Legend/series order: kautz, milp, lp, genkautz, xpander, random
LABEL_ORDER = {"kautz": 0, "aasc_milp": 1, "aasc_lp": 2, "genkautz": 3, "xpander": 4, "random": 5}

# Colors by topology_type (fallback "gray" for unknown)
COLOR_DICT = {
    "kautz": "tab:green",
    "aasc_milp": "tab:red",
    "aasc_lp": "tab:green",
    "genkautz": "tab:blue",
    "xpander": "tab:purple",
    "random": "tab:orange",
}

# Marker by topology_type (fallback "o" for unknown; matplotlib: o, s, ^, v, *, D, P, X, etc.)
MARKER_DICT = {
    "kautz": "*",
    "aasc_milp": "D",
    "aasc_lp": "D",
    "genkautz": "o",
    "xpander": "o",
    "random": "o",
}

# Default paths (relative to project root)
DEFAULT_CSV = "topology_sweep_results.csv"
DEFAULT_OUT = "files/paper_results/arbitrary_sweep.png"

# Plotting parameters (all caps)
FIG_WIDTH = 5
FIG_HEIGHT = 3
X_LABEL_FONT_SIZE = 14
Y_LABEL_FONT_SIZE = 14
TITLE_FONT_SIZE = 12
TICK_FONT_SIZE = 9
LEGEND_FONT_SIZE = 12
LEGEND_BBOX_ANCHOR = (0.5, 1.01)
LEGEND_LOC = 'lower center'
LEGEND_NCOLS = 3
MARKER_SIZE_LINE = 4
MARKER_SIZE_SCATTER = 80
GRID_ALPHA = 0.3
X_LIM_LEFT = 0
X_LIM_RIGHT = 85
DPI = 400
BBOX_INCHES = "tight"
X_AXIS_LABEL = "Number of nodes"
Y_AXIS_LABEL = "Per Source Throughput"

# Theoretical bound line (when PLOT_THEO)
THEO_LINESTYLE = "--"
THEO_COLOR = "gray"


def _rename_label(s):
    """Apply display renames: genkautz→Genkautz, kautz→Kautz, lp/milp→ADORN, random→Random, xpander→Xpander.
    Uses word boundaries; genkautz before kautz so 'Genkautz' is not altered."""
    if not s:
        return s
    # Order: genkautz first so kautz doesn't match inside it
    s = re.sub(r"\bgenkautz\b", "GenKautz", s, flags=re.IGNORECASE)
    s = re.sub(r"\bkautz\b", "Kautz", s, flags=re.IGNORECASE)
    s = re.sub(r"\blp\b", "TONS", s, flags=re.IGNORECASE)
    s = re.sub(r"\bmilp\b", "TONS", s, flags=re.IGNORECASE)
    s = re.sub(r"\brandom\b", "Jellyfish", s, flags=re.IGNORECASE)
    s = re.sub(r"\bxpander\b", "Xpander", s, flags=re.IGNORECASE)
    s = s.replace(" r4", "")
    return s


def line_label(row):
    """Build a display label for each series: topology type + trans + radix."""
    tt = row["topology_type"]
    radix = int(row["radix"])
    trans = row.get("trans")
    if pd.isna(trans) or trans == "" or trans is None:
        trans = None
    else:
        try:
            trans = int(float(trans))
        except (ValueError, TypeError):
            trans = None

    if tt in ("aasc_lp", "aasc_milp"):
        base = "aasc lp" if tt == "aasc_lp" else "aasc milp"
        out = f"{base} {trans} trans r{radix}" if trans is not None else f"{base} (no trans) r{radix}"
    elif tt == "genkautz":
        out = f"genkautz r{radix}"
    elif tt == "xpander":
        out = f"xpander r{radix}"
    elif tt == "random":
        out = f"random r{radix}"
    elif tt == "kautz":
        out = f"kautz r{radix}"
    else:
        out = f"{tt} r{radix}"
    if RENAME_LABELS:
        out = _rename_label(out)
    return out


def load_and_prepare(csv_path):
    """Load CSV and add line_label (trans may already be a column)."""
    df = pd.read_csv(csv_path)

    if "trans" not in df.columns:
        # Parse trans from topology_name for aasc (e.g. aasc_lp_10_r3_2trans -> 2)
        def get_trans(name, tt):
            if tt not in ("aasc_lp", "aasc_milp"):
                return np.nan
            if "_trans" not in name:
                return np.nan
            try:
                part = name.split("_trans")[0]
                return int(part.split("_")[-1])
            except Exception:
                return np.nan

        df["trans"] = df.apply(lambda r: get_trans(r["topology_name"], r["topology_type"]), axis=1)

    df["line_label"] = df.apply(line_label, axis=1)
    df["sort_key"] = df["topology_type"].map(LABEL_ORDER).fillna(99).astype(int)

    # Drop rows with no objective (so lines don't connect through missing data)
    df = df[df["objective_mcf"].notna()].copy()
    df["objective_mcf"] = pd.to_numeric(df["objective_mcf"], errors="coerce")
    df = df[df["objective_mcf"].notna()]
    df["y_plot"] = df["objective_mcf"] * df["number_of_nodes"]
    return df


def plot_lines(df, out_path, title=None, radix=None, topology_filter=None, best_per_category=False, exclude=None, max_nodes=80):
    """Plot one line per (line_label) with x=number_of_nodes, y=objective_mcf*n_nodes.
    Kautz series are drawn as scatter (no lines); others as line plot.
    If best_per_category: collapse aasc_lp to one 'lp' series (max at each n),
    aasc_milp to one 'milp' series (max at each n); genkautz, kautz, xpander, random unchanged.
    exclude: list of topology_type values to exclude. max_nodes: do not plot points with number_of_nodes > max_nodes."""
    if df.empty:
        print("No data to plot.")
        return

    df = df[df["number_of_nodes"] <= max_nodes].copy()
    if exclude:
        df = df[~df["topology_type"].isin(exclude)].copy()
    if radix is not None:
        df = df[df["radix"] == radix].copy()
    if topology_filter:
        df = df[df["line_label"].str.contains(topology_filter, case=False, na=False)].copy()

    if df.empty:
        print("No data after filters.")
        return

    # Radixes for theoretical bound (before best_per_category may drop radix from df)
    radixes_theo = sorted(df["radix"].unique().tolist()) if "radix" in df.columns else []

    # % difference of best aasc (ADORN) vs theoretical bound r / log_r(N)
    df_aasc = df[df["topology_type"].isin(["aasc_lp", "aasc_milp"])]
    if not df_aasc.empty and "radix" in df_aasc.columns:
        best = df_aasc.groupby(["number_of_nodes", "radix"], as_index=False)["y_plot"].max()
        best["bound"] = best.apply(
            lambda row: (
                row["radix"] / (np.log(row["number_of_nodes"]) / np.log(row["radix"]))
                if row["number_of_nodes"] >= 2 and row["radix"] >= 2
                else np.nan
            ),
            axis=1,
        )
        best = best.dropna(subset=["bound"])
        if not best.empty:
            best["pct_diff"] = (best["y_plot"] - best["bound"]) / best["bound"] * 100
            mean_pct = best["pct_diff"].mean()
            min_pct = best["pct_diff"].min()
            max_pct = best["pct_diff"].max()
            print(f"Best ADORN vs bound: mean {mean_pct:.2f}% (min {min_pct:.2f}%, max {max_pct:.2f}%)")

    if best_per_category:
        mask_lp = df["topology_type"] == "aasc_lp"
        mask_milp = df["topology_type"] == "aasc_milp"
        mask_other = ~(mask_lp | mask_milp)
        pieces = [df.loc[mask_other, ["number_of_nodes", "y_plot", "line_label", "sort_key", "topology_type"]]]
        if mask_lp.any():
            df_lp = df.loc[mask_lp].groupby("number_of_nodes", as_index=False)["y_plot"].max()
            df_lp["line_label"] = _rename_label("lp") if RENAME_LABELS else "lp"
            df_lp["sort_key"] = LABEL_ORDER["aasc_lp"]
            df_lp["topology_type"] = "aasc_lp"
            pieces.append(df_lp[["number_of_nodes", "y_plot", "line_label", "sort_key", "topology_type"]])
        if mask_milp.any():
            df_milp = df.loc[mask_milp].groupby("number_of_nodes", as_index=False)["y_plot"].max()
            df_milp["line_label"] = _rename_label("milp") if RENAME_LABELS else "milp"
            df_milp["sort_key"] = LABEL_ORDER["aasc_milp"]
            df_milp["topology_type"] = "aasc_milp"
            pieces.append(df_milp[["number_of_nodes", "y_plot", "line_label", "sort_key", "topology_type"]])
        df = pd.concat(pieces, ignore_index=True)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    # Plot in desired order (kautz, milp, lp, genkautz, xpander, random) so legend order matches
    order_df = df[["sort_key", "line_label", "topology_type"]].drop_duplicates(subset=["sort_key", "line_label"]).sort_values("sort_key")

    for _, row in order_df.iterrows():
        lb = row["line_label"]
        tt = row["topology_type"]
        sub = df[(df["line_label"] == lb) & (df["topology_type"] == tt)].sort_values("number_of_nodes")
        if sub.empty:
            sub = df[df["line_label"] == lb].sort_values("number_of_nodes")
        if sub.empty:
            continue
        color_val = COLOR_DICT.get(tt, "gray")
        marker_val = MARKER_DICT.get(tt, "o")
        # Only topology_type "kautz" gets scatter; genkautz and others are lines
        if tt == "kautz":
            ax.scatter(
                sub["number_of_nodes"].values,
                sub["y_plot"].values,
                marker=marker_val,
                s=MARKER_SIZE_SCATTER,
                label=lb,
                color=color_val,
                zorder=5,
            )
        else:
            ax.plot(
                sub["number_of_nodes"].values,
                sub["y_plot"].values,
                marker=marker_val,
                markersize=MARKER_SIZE_LINE,
                label=lb,
                color=color_val,
            )

    # Theoretical bound: per-source <= r / log_r(N); plot last (dashed, no markers)
    if PLOT_THEO and radixes_theo:
        n_vals = np.arange(2, max_nodes + 1, dtype=float)
        n_vals = np.maximum(n_vals, 2.0)  # avoid log(1)=0
        for r in radixes_theo:
            if r < 2:
                continue
            log_r_n = np.log(n_vals) / np.log(r)
            y_theo = r / log_r_n  # per-source bound
            ax.plot(
                n_vals,
                y_theo,
                linestyle=THEO_LINESTYLE,
                color=THEO_COLOR,
                label=f"Bound r{r}",
                zorder=1,
            )

    ax.set_xlabel(X_AXIS_LABEL, fontsize=X_LABEL_FONT_SIZE)
    ax.set_ylabel(Y_AXIS_LABEL, fontsize=Y_LABEL_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    if title:
        ax.set_title(title, fontsize=TITLE_FONT_SIZE)
        
    ax.legend(bbox_to_anchor=LEGEND_BBOX_ANCHOR, loc=LEGEND_LOC, fontsize=LEGEND_FONT_SIZE, ncol=LEGEND_NCOLS)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.set_xlim(X_LIM_LEFT, X_LIM_RIGHT)
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches=BBOX_INCHES)
    plt.close()
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot topology sweep CSV as line chart")
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV, help="Input CSV path")
    parser.add_argument("--out", type=str, default=DEFAULT_OUT, help="Output image path")
    parser.add_argument("--title", type=str, default=None, help="Plot title")
    parser.add_argument("--radix", type=int, default=None, help="Restrict to one radix (e.g. 4)")
    parser.add_argument("--filter", type=str, default=None, help="Substring filter on line label (e.g. aasc, genkautz)")
    parser.add_argument("--exclude", type=str, nargs="*", default=[], help="Exclude topology types (e.g. aasc_lp aasc_milp)")
    parser.add_argument("--max-nodes", type=int, default=80, help="Do not plot points with number_of_nodes > this (default 80)")
    parser.add_argument("--best-per-category", action="store_true", help="Collapse aasc_lp to 'lp' (max at each n), aasc_milp to 'milp'; others unchanged")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        # Assume project root is parent of python_scripts/
        root = Path(__file__).resolve().parent.parent
        csv_path = root / csv_path
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        return

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parent.parent / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_and_prepare(csv_path)
    plot_lines(
        df, out_path,
        title=args.title,
        radix=args.radix,
        topology_filter=args.filter,
        best_per_category=args.best_per_category,
        exclude=args.exclude if args.exclude else None,
        max_nodes=args.max_nodes,
    )


if __name__ == "__main__":
    main()
