#!/usr/bin/env python3
"""
Analyze collected best values from multiple runs of random_topo_analysis.py

This script collects best SC and best avg hops values from multiple runs
(where each run had a time limit and generated multiple topologies),
then performs statistical analysis on these best samples.

Usage:
    python analyze_random_topo_bests.py <base_dir>
    
Example:
    python analyze_random_topo_bests.py random_topo_best_results_4x8x8x4_20250122_120000
"""

import argparse
import glob
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats

SCIPY_OK = True

# Matplotlib for plotting
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend by default
    import matplotlib.pyplot as plt
    MPL_OK = True
except ImportError:
    MPL_OK = False
    print("Warning: matplotlib not available. Plotting will be disabled.", file=sys.stderr)


def describe(arr: np.ndarray) -> dict:
    """Compute descriptive statistics for an array."""
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


def bootstrap_ci_mean(arr: np.ndarray, alpha: float = 0.05, n_boot: int = 20000) -> Tuple[float, float]:
    """
    Percentile bootstrap CI for mean. Robust to non-normality.
    """
    arr = np.asarray(arr, dtype=float)
    n = arr.size
    if n < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(0)
    idx = rng.integers(0, n, size=(n_boot, n))
    samples = arr[idx]
    means = np.mean(samples, axis=1)
    lo = float(np.percentile(means, 100 * (alpha / 2)))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return (lo, hi)


def collect_best_values(base_dir: str, config: str) -> Tuple[List[float], List[float]]:
    """
    Collect best values from all runs for a given configuration.
    
    Returns:
        (best_hops_list, best_sc_list)
    """
    best_hops_list: List[float] = []
    best_sc_list: List[float] = []
    
    # Find all best_values.csv files for this configuration
    pattern = os.path.join(base_dir, config, "run_*", "best_values.csv")
    best_files = sorted(glob.glob(pattern))
    
    if len(best_files) == 0:
        return best_hops_list, best_sc_list
    
    for best_file in best_files:
        try:
            with open(best_file, 'r') as f:
                lines = f.readlines()
                if len(lines) >= 2:  # Header + data
                    parts = lines[1].strip().split(',')
                    if len(parts) >= 2:
                        best_hops_list.append(float(parts[0]))
                        best_sc_list.append(float(parts[1]))
        except Exception as e:
            print(f"  Error reading {best_file}: {e}", file=sys.stderr)
    
    return best_hops_list, best_sc_list


def analyze_configuration(base_dir: str, config: str, alpha: float = 0.05, 
                          bootstrap_n: int = 20000) -> Tuple[List[float], List[float]]:
    """
    Analyze best values for a single configuration.
    """
    print(f"\n{'='*60}")
    print(f"Analyzing configuration: {config}")
    print(f"{'='*60}")
    
    best_hops_list, best_sc_list = collect_best_values(base_dir, config)
    
    if len(best_hops_list) == 0:
        print(f"  Warning: No best_values.csv files found for {config}")
        return [], []
    
    print(f"  Found {len(best_hops_list)} runs")
    
    best_hops_arr = np.array(best_hops_list)
    best_sc_arr = np.array(best_sc_list)
    
    # Statistics for best avg hops
    hops_stats = describe(best_hops_arr)
    hops_ci = bootstrap_ci_mean(best_hops_arr, alpha=alpha, n_boot=bootstrap_n)
    
    print(f"\n  Best Average Hops (from {len(best_hops_list)} runs):")
    print(f"    mean={hops_stats['mean']:.6g}  std={hops_stats['std']:.6g}")
    print(f"    min={hops_stats['min']:.6g}  max={hops_stats['max']:.6g}")
    print(f"    median={hops_stats['median']:.6g}")
    print(f"    95% CI: [{hops_ci[0]:.6g}, {hops_ci[1]:.6g}]")
    
    # Statistics for best SC
    sc_stats = describe(best_sc_arr)
    sc_ci = bootstrap_ci_mean(best_sc_arr, alpha=alpha, n_boot=bootstrap_n)
    
    print(f"\n  Best Approximate Sparsest Cut (from {len(best_sc_list)} runs):")
    print(f"    mean={sc_stats['mean']:.6g}  std={sc_stats['std']:.6g}")
    print(f"    min={sc_stats['min']:.6g}  max={sc_stats['max']:.6g}")
    print(f"    median={sc_stats['median']:.6g}")
    print(f"    95% CI: [{sc_ci[0]:.6g}, {sc_ci[1]:.6g}]")
    
    # Write collected bests to CSV
    output_csv = os.path.join(base_dir, config, "collected_bests.csv")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, 'w') as f:
        f.write("run_id,best_avg_hops,best_approx_sc\n")
        for i, (h, s) in enumerate(zip(best_hops_list, best_sc_list)):
            f.write(f"{i},{h},{s}\n")
    print(f"\n  Collected bests written to: {output_csv}")
    
    return best_hops_list, best_sc_list


def load_reference_values(csv_path: str) -> Dict[str, Tuple[float, float]]:
    """
    Load reference values from CSV file.
    
    CSV format:
        config_name,min_avg_hops,max_sc
        nonsym_no_oneleg,2.5,0.8
        ...
    
    Returns:
        Dictionary mapping config name to (min_avg_hops, max_sc)
    """
    ref_values: Dict[str, Tuple[float, float]] = {}
    
    if not os.path.isfile(csv_path):
        print(f"Warning: Reference CSV file not found: {csv_path}", file=sys.stderr)
        return ref_values
    
    try:
        with open(csv_path, 'r') as f:
            lines = f.readlines()
            # Skip header if present
            start_idx = 1 if len(lines) > 0 and 'config' in lines[0].lower() else 0
            
            for line in lines[start_idx:]:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) >= 3:
                    config_name = parts[0].strip()
                    min_hops = float(parts[1].strip())
                    max_sc = float(parts[2].strip())
                    ref_values[config_name] = (min_hops, max_sc)
    except Exception as e:
        print(f"Error reading reference CSV: {e}", file=sys.stderr)
    
    return ref_values


def create_box_plots(base_dir: str, configs: List[str], 
                     all_hops_data: Dict[str, List[float]], 
                     all_sc_data: Dict[str, List[float]],
                     ref_values: Dict[str, Tuple[float, float]] = None) -> None:
    """
    Create box and whisker plots for best avg hops and best SC.
    """
    if not MPL_OK:
        print("Warning: matplotlib not available. Skipping plots.", file=sys.stderr)
        return
    
    # Prepare data for plotting
    hops_data = [all_hops_data.get(config, []) for config in configs]
    sc_data = [all_sc_data.get(config, []) for config in configs]
    
    # Filter out empty configurations
    valid_configs = [config for config, data in zip(configs, hops_data) if len(data) > 0]
    valid_hops_data = [data for data in hops_data if len(data) > 0]
    valid_sc_data = [data for data in sc_data if len(data) > 0]
    
    if len(valid_configs) == 0:
        print("Warning: No data available for plotting.", file=sys.stderr)
        return
    
    # Create nicer labels for configurations
    config_labels = {
        "nonsym_no_oneleg": "Non-sym\nNo one-leg",
        "nonsym_oneleg": "Non-sym\nOne-leg",
        "sym_no_oneleg": "Symmetric\nNo one-leg",
        "sym_oneleg": "Symmetric\nOne-leg"
    }
    labels = [config_labels.get(config, config) for config in valid_configs]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Best Average Hops
    bp1 = ax1.boxplot(valid_hops_data, labels=labels, patch_artist=True,
                      showmeans=True, meanline=True)
    ax1.set_title('Best Average Hops (Lower is Better)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Average Hops', fontsize=12)
    ax1.set_xlabel('Configuration', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Color the boxes
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    for patch, color in zip(bp1['boxes'], colors[:len(bp1['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Add reference lines for avg hops (min values)
    if ref_values:
        for i, config in enumerate(valid_configs):
            if config in ref_values:
                min_hops_ref, _ = ref_values[config]
                # Position line centered above this configuration's box
                # Box positions are 1-indexed: 1, 2, 3, 4...
                x_pos = i + 1
                ax1.hlines(y=min_hops_ref, xmin=x_pos - 0.3, xmax=x_pos + 0.3,
                          color='red', linestyle='--', linewidth=2, alpha=0.7,
                          label='Reference min' if i == 0 else '')
    
    # Plot 2: Best Approximate Sparsest Cut
    bp2 = ax2.boxplot(valid_sc_data, labels=labels, patch_artist=True,
                      showmeans=True, meanline=True)
    ax2.set_title('Best Approximate Sparsest Cut (Higher is Better)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Approximate Sparsest Cut', fontsize=12)
    ax2.set_xlabel('Configuration', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Color the boxes
    for patch, color in zip(bp2['boxes'], colors[:len(bp2['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Add reference lines for SC (max values)
    if ref_values:
        for i, config in enumerate(valid_configs):
            if config in ref_values:
                _, max_sc_ref = ref_values[config]
                # Position line centered above this configuration's box
                x_pos = i + 1
                ax2.hlines(y=max_sc_ref, xmin=x_pos - 0.3, xmax=x_pos + 0.3,
                          color='red', linestyle='--', linewidth=2, alpha=0.7,
                          label='Reference max' if i == 0 else '')
    
    plt.tight_layout()
    
    # Save plots
    hops_plot_path = os.path.join(base_dir, "best_avg_hops_boxplot.png")
    sc_plot_path = os.path.join(base_dir, "best_sc_boxplot.png")
    combined_plot_path = os.path.join(base_dir, "best_values_boxplots.png")
    
    # Save combined plot first (before creating individual plots)
    plt.savefig(combined_plot_path, dpi=200, bbox_inches='tight')
    
    # Save individual plots
    fig1, ax1_alone = plt.subplots(figsize=(8, 6))
    bp1_alone = ax1_alone.boxplot(valid_hops_data, labels=labels, patch_artist=True,
                                   showmeans=True, meanline=True)
    ax1_alone.set_title('Best Average Hops (Lower is Better)', fontsize=14, fontweight='bold')
    ax1_alone.set_ylabel('Average Hops', fontsize=12)
    ax1_alone.set_xlabel('Configuration', fontsize=12)
    ax1_alone.grid(True, alpha=0.3, axis='y')
    for patch, color in zip(bp1_alone['boxes'], colors[:len(bp1_alone['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Add reference lines for avg hops (min values)
    if ref_values:
        for i, config in enumerate(valid_configs):
            if config in ref_values:
                min_hops_ref, _ = ref_values[config]
                x_pos = i + 1
                ax1_alone.hlines(y=min_hops_ref, xmin=x_pos - 0.3, xmax=x_pos + 0.3,
                                color='red', linestyle='--', linewidth=2, alpha=0.7,
                                label='Reference min' if i == 0 else '')
    
    plt.tight_layout()
    plt.savefig(hops_plot_path, dpi=200, bbox_inches='tight')
    plt.close(fig1)
    
    fig2, ax2_alone = plt.subplots(figsize=(8, 6))
    bp2_alone = ax2_alone.boxplot(valid_sc_data, labels=labels, patch_artist=True,
                                   showmeans=True, meanline=True)
    ax2_alone.set_title('Best Approximate Sparsest Cut (Higher is Better)', fontsize=14, fontweight='bold')
    ax2_alone.set_ylabel('Approximate Sparsest Cut', fontsize=12)
    ax2_alone.set_xlabel('Configuration', fontsize=12)
    ax2_alone.grid(True, alpha=0.3, axis='y')
    for patch, color in zip(bp2_alone['boxes'], colors[:len(bp2_alone['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Add reference lines for SC (max values)
    if ref_values:
        for i, config in enumerate(valid_configs):
            if config in ref_values:
                _, max_sc_ref = ref_values[config]
                x_pos = i + 1
                ax2_alone.hlines(y=max_sc_ref, xmin=x_pos - 0.3, xmax=x_pos + 0.3,
                                color='red', linestyle='--', linewidth=2, alpha=0.7,
                                label='Reference max' if i == 0 else '')
    
    plt.tight_layout()
    plt.savefig(sc_plot_path, dpi=200, bbox_inches='tight')
    plt.close(fig2)
    
    print(f"\nPlots saved:")
    print(f"  Best avg hops: {hops_plot_path}")
    print(f"  Best SC: {sc_plot_path}")
    print(f"  Combined: {combined_plot_path}")
    
    # Try to show in GUI
    try:
        # Check if we're in an environment that supports GUI
        if 'DISPLAY' in os.environ or sys.platform == 'darwin':
            # Try to switch to interactive backend
            try:
                matplotlib.use('TkAgg')
            except:
                try:
                    matplotlib.use('Qt5Agg')
                except:
                    pass
            plt.show(block=False)
            print("Plots displayed in GUI window.")
        else:
            print("No display available. Plots saved to files (see paths above).")
    except Exception as e:
        print(f"Could not display plots in GUI: {e}")
        print("Plots saved to files (see paths above).")
    finally:
        # Close the figure to free memory
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze collected best values from multiple random_topo_analysis.py runs"
    )
    parser.add_argument("base_dir", type=str,
                       help="Base directory containing configuration subdirectories")
    parser.add_argument("--alpha", type=float, default=0.05,
                       help="Significance level for confidence intervals (default: 0.05)")
    parser.add_argument("--bootstrap", type=int, default=20000,
                       help="Number of bootstrap resamples (default: 20000)")
    parser.add_argument("--configs", nargs="+", 
                       default=["nonsym_no_oneleg", "nonsym_oneleg", "sym_no_oneleg", "sym_oneleg"],
                       help="Configuration names to analyze (default: all 4)")
    parser.add_argument("--plots", action="store_true",
                       help="Generate box and whisker plots")
    parser.add_argument("--ref-csv", type=str, default=None,
                       help="CSV file with reference values (config_name,min_avg_hops,max_sc)")
    
    args = parser.parse_args()
    
    base_dir = args.base_dir
    
    if not os.path.isdir(base_dir):
        print(f"Error: Directory not found: {base_dir}", file=sys.stderr)
        return 1
    
    print(f"Analyzing best values from: {base_dir}")
    print(f"Configurations: {', '.join(args.configs)}")
    
    # Collect all data first (for plotting)
    all_hops_data: Dict[str, List[float]] = {}
    all_sc_data: Dict[str, List[float]] = {}
    
    for config in args.configs:
        best_hops_list, best_sc_list = analyze_configuration(
            base_dir, config, alpha=args.alpha, bootstrap_n=args.bootstrap
        )
        if len(best_hops_list) > 0:
            all_hops_data[config] = best_hops_list
            all_sc_data[config] = best_sc_list
    
    # Load reference values if provided
    ref_values = None
    if args.ref_csv:
        ref_values = load_reference_values(args.ref_csv)
        if ref_values:
            print(f"\nLoaded reference values from: {args.ref_csv}")
            for config, (min_hops, max_sc) in ref_values.items():
                print(f"  {config}: min_hops={min_hops:.6g}, max_sc={max_sc:.6g}")
    
    # Create plots if requested
    if args.plots:
        create_box_plots(base_dir, args.configs, all_hops_data, all_sc_data, ref_values)
    
    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"{'='*60}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
