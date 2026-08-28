#!/usr/bin/env python3
"""
Best Sample Analysis:
- Performs multiple "outer" trials
- For each outer trial, generates multiple topologies and computes metrics
- Records the best (highest SC, lowest avg hops) from each outer trial
- Performs statistical analysis on these best samples

This script wraps random_topo_analysis.py functionality to analyze
the distribution of "best" topologies from multiple independent searches.
"""

import argparse
import math
import os
import sys
import time
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
# SciPy is strongly recommended for robust stats tests/intervals.
from scipy import stats
SCIPY_OK = True

# Matplotlib only used if --plots is set.
try:
    import matplotlib.pyplot as plt
    MPL_OK = True
except Exception:
    MPL_OK = False

# Import functions from random_topo_analysis
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# We'll need to import the analysis functions
# For now, we'll duplicate the necessary statistical functions
# or import them if they're available

VERBOSE = False
INF = 10**10


# -----------------------------
# Statistical functions (from random_topo_analysis)
# -----------------------------

def describe(arr: np.ndarray) -> dict:
    arr = np.asarray(arr, dtype=float)
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size >= 2 else float("nan"),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
    }


def t_confidence_interval_mean(arr: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    """
    1-alpha CI for mean using Student-t:
        mean +/- t_{1-alpha/2, n-1} * s/sqrt(n)
    """
    arr = np.asarray(arr, dtype=float)
    n = arr.size
    if n < 2 or not SCIPY_OK:
        return (float("nan"), float("nan"))
    mean = float(np.mean(arr))
    s = float(np.std(arr, ddof=1))
    tcrit = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
    half = tcrit * s / math.sqrt(n)
    return (mean - half, mean + half)


def bootstrap_ci_mean(arr: np.ndarray, alpha: float = 0.05, n_boot: int = 20000,
                      rng: Optional[np.random.Generator] = None) -> Tuple[float, float]:
    """
    Percentile bootstrap CI for mean. Robust to non-normality.
    """
    arr = np.asarray(arr, dtype=float)
    n = arr.size
    if n < 2:
        return (float("nan"), float("nan"))
    if rng is None:
        rng = np.random.default_rng(0)

    # Vectorized bootstrap
    idx = rng.integers(0, n, size=(n_boot, n))
    samples = arr[idx]
    means = np.mean(samples, axis=1)

    lo = float(np.percentile(means, 100 * (alpha / 2)))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return (lo, hi)


def normality_tests(arr: np.ndarray) -> dict:
    """
    Runs multiple normality tests.
    """
    arr = np.asarray(arr, dtype=float)
    out = {}
    n = arr.size

    if not SCIPY_OK or n < 3:
        out["note"] = "SciPy not available or sample too small; skipping tests."
        return out

    # Shapiro-Wilk: recommended for n <= ~5000
    if n <= 5000:
        w, p = stats.shapiro(arr)
        out["shapiro_W"] = float(w)
        out["shapiro_p"] = float(p)
    else:
        out["shapiro_note"] = "n > 5000: skipping Shapiro-Wilk (common practice)."

    # D'Agostino & Pearson K^2 test: requires n >= 8
    if n >= 8:
        k2, p = stats.normaltest(arr)
        out["dagostino_k2"] = float(k2)
        out["dagostino_p"] = float(p)

    # Anderson-Darling
    ad = stats.anderson(arr, dist="norm")
    out["anderson_stat"] = float(ad.statistic)
    out["anderson_crit_5pct"] = float(ad.critical_values[list(ad.significance_level).index(5.0)])

    # Jarque–Bera
    jb, p = stats.jarque_bera(arr)
    out["jarque_bera"] = float(jb)
    out["jarque_p"] = float(p)

    return out


def ci_halfwidth(ci: Tuple[float, float]) -> float:
    if any(math.isnan(x) for x in ci):
        return float("nan")
    return 0.5 * (ci[1] - ci[0])


def ensure_dir(path: str) -> None:
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def save_plots(arr: np.ndarray, name: str, out_dir: str) -> None:
    """
    Saves histogram and QQ plot.
    """
    if not MPL_OK:
        return
    ensure_dir(out_dir)

    # Histogram
    plt.figure()
    plt.hist(arr, bins="auto")
    plt.title(f"{name} histogram")
    plt.xlabel(name)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}_hist.png"), dpi=200)
    plt.close()

    # QQ plot (requires SciPy)
    if SCIPY_OK:
        plt.figure()
        stats.probplot(arr, dist="norm", plot=plt)
        plt.title(f"{name} QQ plot vs Normal")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{name}_qq.png"), dpi=200)
        plt.close()


@dataclass
class BestSampleConfig:
    # Outer trial parameters
    n_outer_trials: int
    n_inner_generations: int
    
    # Topology parameters
    xyzc_dims: tuple
    symmetric: bool
    mc_dims: Optional[tuple]
    sym_type: str
    one_leg: bool
    
    # Statistical parameters
    alpha: float
    bootstrap_n: int
    seed: int
    
    # Output parameters
    out_dir: str
    plots: bool
    
    # Time limit
    time_limit_minutes: Optional[float]
    
    # Inner trial time limit (per outer trial)
    inner_time_limit_minutes: Optional[float]


def parse_args() -> BestSampleConfig:
    ap = argparse.ArgumentParser(
        description="Analyze best samples from multiple topology generation trials"
    )
    
    # Outer trial parameters
    ap.add_argument("--n-outer-trials", type=int, default=50,
                    help="Number of outer trials (each finds a best topology)")
    ap.add_argument("--n-inner-generations", type=int, default=100,
                    help="Number of topology generations per outer trial")
    
    # Topology parameters
    ap.add_argument('--xyzc_dims', nargs='+', type=int, required=True,
                    help='Global system dimensions (x, y, z, cube_dim). Space-separated integers')
    ap.add_argument('--symmetric', action='store_true',
                    help='Generate symmetric topologies')
    ap.add_argument('--mc_dims', nargs='+', type=int,
                    help='Mega cube dimensions (x, y, z). Required if --symmetric is set')
    ap.add_argument('--sym_type', type=str, default='trans',
                    choices=['trans', 'refl-trans'],
                    help='Symmetry type (default: trans)')
    ap.add_argument('--one_leg', action='store_true',
                    help='Only consider tri inequalities where (i,k) in E')
    
    # Statistical parameters
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="Significance level (default: 0.05 -> 95% CI)")
    ap.add_argument("--bootstrap", type=int, default=20000,
                    help="Bootstrap resamples for CI")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for reproducibility")
    
    # Output parameters
    ap.add_argument("--out-dir", type=str, default="best_sample_stats",
                    help="Output directory")
    ap.add_argument("--plots", action="store_true",
                    help="Save histograms and QQ plots")
    
    # Time limits
    ap.add_argument("--time-limit", type=float, default=None,
                    help="Total time limit in minutes")
    ap.add_argument("--inner-time-limit", type=float, default=None,
                    help="Time limit per outer trial in minutes")
    
    args = ap.parse_args()
    
    # Validate symmetric parameters
    mc_dims = None
    if args.symmetric:
        if args.mc_dims is None:
            raise ValueError("--mc_dims is required when --symmetric is set")
        if len(args.mc_dims) != 3:
            raise ValueError("--mc_dims must have exactly 3 values (x, y, z)")
        mc_dims = tuple(args.mc_dims)
        # Validate mc_dims are compatible with xyzc_dims
        if args.xyzc_dims:
            cube_dim = args.xyzc_dims[3]
            for d in mc_dims:
                if d % cube_dim != 0:
                    raise ValueError(f"mc_dims values must be multiples of cube_dim ({cube_dim})")
    
    return BestSampleConfig(
        n_outer_trials=args.n_outer_trials,
        n_inner_generations=args.n_inner_generations,
        xyzc_dims=tuple(args.xyzc_dims),
        symmetric=args.symmetric,
        mc_dims=mc_dims,
        sym_type=args.sym_type,
        one_leg=args.one_leg,
        alpha=args.alpha,
        bootstrap_n=args.bootstrap,
        seed=args.seed,
        out_dir=args.out_dir,
        plots=args.plots,
        time_limit_minutes=args.time_limit,
        inner_time_limit_minutes=args.inner_time_limit
    )


def run_single_outer_trial(trial_num: int, cfg: BestSampleConfig) -> Tuple[float, float]:
    """
    Run a single outer trial: generate n_inner_generations topologies
    and return the best (highest SC, lowest avg hops).
    
    Returns:
        (best_sc, best_avg_hops)
    """
    # Import here to avoid circular imports
    from random_topo_analysis import (
        random_gen, calc_avg_hops, calc_approx_sc
    )
    
    best_sc = -float('inf')
    best_avg_hops = float('inf')
    
    rng = np.random.default_rng(cfg.seed + trial_num)
    
    # Set random seed for this trial
    import random
    random.seed(cfg.seed + trial_num)
    np.random.seed(cfg.seed + trial_num)
    
    time_limit_seconds = None
    if cfg.inner_time_limit_minutes is not None:
        time_limit_seconds = cfg.inner_time_limit_minutes * 60.0
    
    t0 = time.time()
    
    for i in range(cfg.n_inner_generations):
        # Check time limit
        if time_limit_seconds is not None:
            elapsed = time.time() - t0
            if elapsed >= time_limit_seconds:
                print(f"  Inner trial {trial_num}: time limit reached after {i} generations")
                break
        
        # Generate topology
        topo = random_gen(
            cfg.xyzc_dims,
            symmetric=cfg.symmetric,
            mc_dims=cfg.mc_dims,
            sym_type=cfg.sym_type
        )
        
        # Compute metrics
        avg_hops = calc_avg_hops(topo)
        sc = calc_approx_sc(
            topo,
            one_leg=cfg.one_leg,
            xyzc_dims=cfg.xyzc_dims,
            symmetric=cfg.symmetric,
            mc_dims=cfg.mc_dims,
            sym_type=cfg.sym_type
        )
        
        # Update best
        if sc > best_sc:
            best_sc = sc
        if avg_hops < best_avg_hops:
            best_avg_hops = avg_hops
    
    return (best_sc, best_avg_hops)


def summarize_metric(name: str, arr: np.ndarray, alpha: float, bootstrap_n: int,
                    rng: np.random.Generator) -> dict:
    d = describe(arr)
    ci_t = t_confidence_interval_mean(arr, alpha=alpha)
    ci_b = bootstrap_ci_mean(arr, alpha=alpha, n_boot=bootstrap_n, rng=rng)
    nt = normality_tests(arr)

    d.update({
        "ci_t_lo": float(ci_t[0]),
        "ci_t_hi": float(ci_t[1]),
        "ci_t_half": float(ci_halfwidth(ci_t)),
        "ci_boot_lo": float(ci_b[0]),
        "ci_boot_hi": float(ci_b[1]),
        "ci_boot_half": float(ci_halfwidth(ci_b)),
    })
    d["normality"] = nt
    return d


def print_report(best_sc_arr: np.ndarray, best_hops_arr: np.ndarray,
                 cfg: BestSampleConfig) -> None:
    rng = np.random.default_rng(cfg.seed + 12345)

    sc_stats = summarize_metric("best_sc", best_sc_arr, cfg.alpha, cfg.bootstrap_n, rng)
    hops_stats = summarize_metric("best_avg_hops", best_hops_arr, cfg.alpha, cfg.bootstrap_n, rng)

    def fmt_ci(stats_dict: dict, which: str) -> str:
        lo = stats_dict[f"ci_{which}_lo"]
        hi = stats_dict[f"ci_{which}_hi"]
        half = stats_dict[f"ci_{which}_half"]
        return f"[{lo:.6g}, {hi:.6g}] (half-width={half:.6g})"

    # Collect all output lines
    output_lines = []

    def add_line(line: str):
        """Add a line to both output list and print to stdout"""
        output_lines.append(line)
        print(line)

    add_line("\n=== Best Sample Statistical Report ===")
    add_line(f"Outer trials: {cfg.n_outer_trials}")
    add_line(f"Inner generations per trial: {cfg.n_inner_generations}")
    add_line(f"Total topology generations: {cfg.n_outer_trials * cfg.n_inner_generations}")
    add_line(f"Confidence: {(1.0 - cfg.alpha)*100:.1f}%  (alpha={cfg.alpha})")
    add_line(f"Bootstrap resamples: {cfg.bootstrap_n}")

    add_line("\n--- Best Approximate Sparsest Cut (from each outer trial) ---")
    add_line(f"mean={sc_stats['mean']:.6g}  std={sc_stats['std']:.6g}  "
          f"min={sc_stats['min']:.6g}  max={sc_stats['max']:.6g}")
    add_line(f"median={sc_stats['median']:.6g}  p05={sc_stats['p05']:.6g}  p95={sc_stats['p95']:.6g}")
    if SCIPY_OK:
        add_line(f"95% CI (t):       {fmt_ci(sc_stats, 't')}")
    add_line(f"95% CI (bootstrap): {fmt_ci(sc_stats, 'boot')}")
    add_line(f"Normality tests: {sc_stats['normality']}")

    add_line("\n--- Best Average Hops (from each outer trial) ---")
    add_line(f"mean={hops_stats['mean']:.6g}  std={hops_stats['std']:.6g}  "
          f"min={hops_stats['min']:.6g}  max={hops_stats['max']:.6g}")
    add_line(f"median={hops_stats['median']:.6g}  p05={hops_stats['p05']:.6g}  p95={hops_stats['p95']:.6g}")
    if SCIPY_OK:
        add_line(f"95% CI (t):       {fmt_ci(hops_stats, 't')}")
    add_line(f"95% CI (bootstrap): {fmt_ci(hops_stats, 'boot')}")
    add_line(f"Normality tests: {hops_stats['normality']}")

    # Plots
    if cfg.plots:
        if not MPL_OK:
            add_line("\n[plots] Matplotlib not available; skipping plot generation.")
        else:
            save_plots(best_sc_arr, "best_sc", cfg.out_dir)
            save_plots(best_hops_arr, "best_avg_hops", cfg.out_dir)
            add_line(f"\nSaved plots to: {cfg.out_dir}/")

    # Write all output to file
    stat_summary_path = os.path.join(cfg.out_dir, "stat_summary.txt")
    with open(stat_summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
        f.write("\n")
    print(f"\nStatistics summary written to: {stat_summary_path}")


def run_trials(cfg: BestSampleConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run all outer trials and collect best samples.
    """
    ensure_dir(cfg.out_dir)
    
    best_sc_list: List[float] = []
    best_hops_list: List[float] = []
    
    csv_path = os.path.join(cfg.out_dir, "best_samples.csv")
    
    # Write CSV header
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("outer_trial,best_sc,best_avg_hops,elapsed_sec\n")
    
    t0 = time.time()
    time_limit_seconds = None
    if cfg.time_limit_minutes is not None:
        time_limit_seconds = cfg.time_limit_minutes * 60.0
        print(f"Total time limit set to {cfg.time_limit_minutes} minutes ({time_limit_seconds} seconds)")
    
    for trial in range(cfg.n_outer_trials):
        # Check total time limit
        elapsed = time.time() - t0
        if time_limit_seconds is not None and elapsed >= time_limit_seconds:
            print(f"Total time limit reached. Stopping after {trial} outer trials.")
            break
        
        print(f"Outer trial {trial + 1}/{cfg.n_outer_trials}")
        trial_start = time.time()
        
        best_sc, best_hops = run_single_outer_trial(trial, cfg)
        
        best_sc_list.append(best_sc)
        best_hops_list.append(best_hops)
        
        trial_elapsed = time.time() - trial_start
        total_elapsed = time.time() - t0
        
        print(f"  Best SC: {best_sc:.6g}, Best Avg Hops: {best_hops:.6g}")
        print(f"  Trial time: {trial_elapsed:.2f}s, Total time: {total_elapsed:.2f}s")
        
        # Write to CSV
        with open(csv_path, "a", encoding="utf-8") as f:
            f.write(f"{trial},{best_sc},{best_hops},{total_elapsed}\n")
    
    return (np.array(best_sc_list, dtype=float), np.array(best_hops_list, dtype=float))


def main() -> int:
    cfg = parse_args()

    if cfg.plots and not MPL_OK:
        print("Warning: --plots requested but matplotlib is not available.", file=sys.stderr)

    if not SCIPY_OK:
        print("Warning: SciPy not available. Normality tests and t-based CI will be limited.",
              file=sys.stderr)

    best_sc_arr, best_hops_arr = run_trials(cfg)
    print_report(best_sc_arr, best_hops_arr, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
