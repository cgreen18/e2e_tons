# Best Sample Analysis

## Overview

`best_sample_analysis.py` performs a two-level statistical analysis:
1. **Outer trials**: Multiple independent searches for "best" topologies
2. **Inner generations**: Each outer trial generates multiple topologies and selects the best

For each outer trial, the script records:
- **Best Approximate Sparsest Cut (SC)**: The highest SC value found
- **Best Average Hops**: The lowest average hops value found

Then it performs statistical analysis on these "best" samples to understand:
- The distribution of best-case performance
- How consistently good topologies can be found
- Confidence intervals for best-case metrics

## Use Case

This script is useful when you want to understand:
- **What is the best topology quality achievable?** (not just average)
- **How much variation is there in finding good topologies?**
- **What is the expected best-case performance?** (e.g., "if I run 100 searches, what's the best SC I can expect?")

## Requirements

- Python 3.x
- numpy
- scipy (recommended)
- matplotlib (optional, for plots)
- All dependencies from `random_topo_analysis.py` (gurobipy, networkx, etc.)

## Basic Usage

### Required Arguments

- `--xyzc_dims`: Global system dimensions (x, y, z, cube_dim)
  - Format: space-separated integers, e.g., `--xyzc_dims 8 8 8 4`

### Basic Example

```bash
python best_sample_analysis.py \
    --xyzc_dims 8 8 8 4 \
    --n-outer-trials 50 \
    --n-inner-generations 100
```

This runs 50 outer trials, each generating 100 topologies and selecting the best.

## Command Line Arguments

### Trial Parameters

- `--n-outer-trials <int>`: Number of outer trials (default: 50)
  - Each outer trial finds one "best" topology
- `--n-inner-generations <int>`: Number of topology generations per outer trial (default: 100)
  - More generations = better chance of finding good topologies

### Topology Parameters

- `--xyzc_dims <x> <y> <z> <cube_dim>`: Global system dimensions (required)
- `--symmetric`: Generate symmetric topologies
- `--mc_dims <x> <y> <z>`: Mega cube dimensions (required if `--symmetric`)
- `--sym_type <type>`: Symmetry type - `trans` or `refl-trans` (default: `trans`)
- `--one_leg`: Only consider triangle inequalities where (i,k) is an edge

### Statistical Parameters

- `--alpha <float>`: Significance level (default: 0.05 → 95% CI)
- `--bootstrap <int>`: Bootstrap resamples (default: 20000)
- `--seed <int>`: Random seed for reproducibility (default: 0)

### Output Parameters

- `--out-dir <path>`: Output directory (default: `best_sample_stats`)
- `--plots`: Generate histogram and QQ plots

### Time Limits

- `--time-limit <float>`: Total time limit in minutes (stops all trials)
- `--inner-time-limit <float>`: Time limit per outer trial in minutes

## Example Workflows

### Quick Analysis (10 outer trials, 50 inner generations)
```bash
python best_sample_analysis.py \
    --xyzc_dims 8 8 8 4 \
    --n-outer-trials 10 \
    --n-inner-generations 50 \
    --out-dir quick_best_analysis
```

### Comprehensive Analysis
```bash
python best_sample_analysis.py \
    --xyzc_dims 8 8 8 4 \
    --n-outer-trials 100 \
    --n-inner-generations 200 \
    --bootstrap 50000 \
    --plots \
    --out-dir comprehensive_best_analysis
```

### Symmetric Topology Analysis
```bash
python best_sample_analysis.py \
    --xyzc_dims 8 8 8 4 \
    --symmetric \
    --mc_dims 4 4 4 \
    --sym_type trans \
    --n-outer-trials 50 \
    --n-inner-generations 100 \
    --one_leg \
    --out-dir symmetric_best_analysis
```

### Time-Limited Run
```bash
python best_sample_analysis.py \
    --xyzc_dims 8 8 8 4 \
    --n-outer-trials 50 \
    --n-inner-generations 100 \
    --time-limit 120 \
    --inner-time-limit 2 \
    --out-dir timed_best_analysis
```

## Output Files

The script generates the following files in the output directory:

1. **`best_samples.csv`**: Raw data with columns:
   - `outer_trial`: Outer trial number (0-indexed)
   - `best_sc`: Best approximate sparsest cut from this trial
   - `best_avg_hops`: Best (lowest) average hops from this trial
   - `elapsed_sec`: Total elapsed time in seconds

2. **`stat_summary.txt`**: Complete statistical report including:
   - Summary statistics for best SC and best avg hops
   - Confidence intervals for both metrics
   - Normality test results

3. **Plots** (if `--plots` is set):
   - `best_sc_hist.png`: Histogram of best SC values
   - `best_sc_qq.png`: QQ plot for best SC
   - `best_avg_hops_hist.png`: Histogram of best avg hops values
   - `best_avg_hops_qq.png`: QQ plot for best avg hops

## Understanding the Output

### Statistical Report

The `stat_summary.txt` file contains statistics for two metrics:

1. **Best Approximate Sparsest Cut**:
   - Distribution of the best SC found in each outer trial
   - Higher values indicate better connectivity
   - Mean tells you the expected best SC if you run one search
   - Max tells you the absolute best found across all trials

2. **Best Average Hops**:
   - Distribution of the best (lowest) avg hops found in each outer trial
   - Lower values indicate shorter paths
   - Mean tells you the expected best avg hops if you run one search
   - Min tells you the absolute best found across all trials

### Interpreting Results

**Example Output:**
```
Best Approximate Sparsest Cut:
  mean=0.85  std=0.12  min=0.65  max=1.05
  95% CI (bootstrap): [0.82, 0.88] (half-width=0.03)
```

This means:
- If you run one search (100 inner generations), you can expect to find a topology with SC ≈ 0.85
- With 95% confidence, the true expected best SC is between 0.82 and 0.88
- The best topology found across all trials had SC = 1.05
- The worst "best" topology had SC = 0.65

### Key Insights

1. **Mean vs Max/Min**: 
   - Mean tells you what to expect from a single search
   - Max/Min tell you the absolute best/worst found

2. **Standard Deviation**:
   - High std indicates high variability in finding good topologies
   - Low std indicates consistent ability to find good topologies

3. **Confidence Intervals**:
   - Narrow intervals indicate precise estimates
   - Useful for comparing different topology generation strategies

## Relationship to random_topo_analysis.py

- **`random_topo_analysis.py`**: Analyzes the distribution of all generated topologies
  - Answers: "What is the average topology quality?"
  
- **`best_sample_analysis.py`**: Analyzes the distribution of best topologies from multiple searches
  - Answers: "What is the best topology quality achievable?"

## Performance Considerations

- **Computation Time**: Each outer trial generates `n_inner_generations` topologies
  - Total topologies generated = `n_outer_trials × n_inner_generations`
  - Example: 50 outer × 100 inner = 5,000 topologies
  
- **Symmetric Generation**: Using `--symmetric` significantly speeds up computation
  - Recommended for large topologies
  
- **Time Limits**: Use `--inner-time-limit` to cap individual trial time
  - Useful when inner generations vary significantly in time

## Tips

1. **Start Small**: Begin with small `n_outer_trials` and `n_inner_generations` to estimate runtime
2. **Balance**: More inner generations = better best samples, but slower
3. **Reproducibility**: Use `--seed` for reproducible results
4. **Parallelization**: Consider running multiple instances with different seeds in parallel
5. **Comparison**: Run with different parameters to compare strategies

## Example: Comparing Strategies

```bash
# Strategy 1: Non-symmetric
python best_sample_analysis.py \
    --xyzc_dims 8 8 8 4 \
    --n-outer-trials 50 \
    --n-inner-generations 100 \
    --out-dir strategy1

# Strategy 2: Symmetric
python best_sample_analysis.py \
    --xyzc_dims 8 8 8 4 \
    --symmetric \
    --mc_dims 4 4 4 \
    --n-outer-trials 50 \
    --n-inner-generations 100 \
    --out-dir strategy2
```

Compare the `stat_summary.txt` files to see which strategy finds better topologies.
