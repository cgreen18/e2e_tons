# Random Topology Analysis

## Overview

`random_topo_analysis.py` performs statistical analysis on randomly generated topologies. For each generated topology, it computes:
- **Average hops**: Average shortest path length between all pairs of routers
- **Approximate sparsest cut**: A metric measuring network connectivity quality

The script collects these metrics across multiple topology generations and provides comprehensive statistical analysis including:
- Descriptive statistics (mean, std, min, max, median, percentiles)
- Confidence intervals (both t-based and bootstrap)
- Normality tests
- Optional plots (histograms and QQ plots)

## Requirements

- Python 3.x
- numpy
- networkx
- scipy (recommended for robust statistics)
- gurobipy (for approximate sparsest cut calculation)
- matplotlib (optional, for plots)
- tpuv4_symmetry module (optional, for symmetric topology generation)

## Basic Usage

### Required Arguments

- `--xyzc_dims`: Global system dimensions (x, y, z, cube_dim)
  - Format: space-separated integers, e.g., `--xyzc_dims 8 8 8 4`
  - Example: `8 8 8 4` means 8×8×8 routers with cube dimension 4

### Basic Example

```bash
python random_topo_analysis.py \
    --xyzc_dims 8 8 8 4 \
    --n 200 \
    --out-dir results
```

This generates 200 random topologies and analyzes their average hops and approximate sparsest cut.

## Command Line Arguments

### Sampling Parameters

- `--n <int>`: Number of topology generations (default: 200)
  - Ignored if `--stop` is enabled
- `--max-n <int>`: Maximum trials for sequential stopping (default: 5000)
- `--seed <int>`: Random seed for reproducibility (default: 0)

### Sequential Stopping

- `--stop`: Enable sequential stopping based on confidence interval tolerances
- `--tol-hops-abs <float>`: Stop when CI half-width for avg hops ≤ tolerance
- `--tol-sc-abs <float>`: Stop when CI half-width for approx SC ≤ tolerance
- `--tol-sc-rel <float>`: Stop when relative CI half-width ≤ tolerance × mean

Example with sequential stopping:
```bash
python random_topo_analysis.py \
    --xyzc_dims 8 8 8 4 \
    --max-n 5000 \
    --stop \
    --tol-hops-abs 0.02 \
    --tol-sc-rel 0.02
```

### Statistical Analysis

- `--alpha <float>`: Significance level for confidence intervals (default: 0.05 → 95% CI)
- `--bootstrap <int>`: Number of bootstrap resamples (default: 20000)

### Symmetric Topology Generation

- `--symmetric`: Generate symmetric topologies using canonical cube approach
- `--mc_dims <x> <y> <z>`: Mega cube dimensions (required if `--symmetric` is set)
  - Must be multiples of cube_dim
- `--sym_type <type>`: Symmetry type - `trans` or `refl-trans` (default: `trans`)

Example with symmetric generation:
```bash
python random_topo_analysis.py \
    --xyzc_dims 8 8 8 4 \
    --symmetric \
    --mc_dims 4 4 4 \
    --sym_type trans \
    --n 200
```

### Approximate Sparsest Cut Options

- `--one_leg`: Only consider triangle inequalities where (i,k) is an edge
  - Reduces problem size but may affect accuracy

### Output Options

- `--out-dir <path>`: Output directory (default: `topo_stats_out`)
- `--csv <path>`: CSV output path (default: `samples.csv` relative to out-dir)
- `--plots`: Generate histogram and QQ plots
- `--time-limit <float>`: Time limit in minutes (script stops after this time)

## Output Files

The script generates the following files in the output directory:

1. **`samples.csv`**: Raw data with columns:
   - `trial`: Trial number
   - `avg_hops`: Average hops for this topology
   - `approx_sc`: Approximate sparsest cut value
   - `elapsed_sec`: Elapsed time in seconds

2. **`stat_summary.txt`**: Complete statistical report including:
   - Summary statistics for both metrics
   - Confidence intervals
   - Normality test results

3. **`time_stats.csv`**: Cumulative time statistics:
   - `topo_gen_cumul_time`: Total time spent generating topologies
   - `avg_hops_cumul_time`: Total time computing average hops
   - `approx_sc_cumul_time`: Total time computing approximate sparsest cut

4. **Plots** (if `--plots` is set):
   - `avg_hops_hist.png`: Histogram of average hops
   - `avg_hops_qq.png`: QQ plot for average hops
   - `approx_sc_hist.png`: Histogram of approximate sparsest cut
   - `approx_sc_qq.png`: QQ plot for approximate sparsest cut

## Example Workflows

### Quick Analysis (100 topologies)
```bash
python random_topo_analysis.py \
    --xyzc_dims 8 8 8 4 \
    --n 100 \
    --out-dir quick_analysis
```

### Comprehensive Analysis with Plots
```bash
python random_topo_analysis.py \
    --xyzc_dims 8 8 8 4 \
    --n 500 \
    --bootstrap 50000 \
    --plots \
    --out-dir comprehensive_analysis
```

### Symmetric Topology Analysis
```bash
python random_topo_analysis.py \
    --xyzc_dims 8 8 8 4 \
    --symmetric \
    --mc_dims 4 4 4 \
    --sym_type trans \
    --n 200 \
    --one_leg \
    --out-dir symmetric_analysis
```

### Time-Limited Run
```bash
python random_topo_analysis.py \
    --xyzc_dims 8 8 8 4 \
    --time-limit 60 \
    --out-dir timed_run
```

## Understanding the Output

### Statistical Report

The `stat_summary.txt` file contains:

1. **Summary Statistics**:
   - `mean`: Sample mean
   - `std`: Sample standard deviation
   - `min`, `max`: Minimum and maximum values
   - `median`: Median value
   - `p05`, `p25`, `p75`, `p95`: Percentiles

2. **Confidence Intervals**:
   - `CI (t)`: Student-t based confidence interval
   - `CI (bootstrap)`: Bootstrap-based confidence interval (more robust)
   - Both include half-width for easy interpretation

3. **Normality Tests**:
   - Shapiro-Wilk test (for n ≤ 5000)
   - D'Agostino & Pearson K² test
   - Anderson-Darling test
   - Jarque-Bera test

### Interpreting Results

- **Average Hops**: Lower is better (shorter paths)
- **Approximate Sparsest Cut**: Higher is better (better connectivity)
- **Confidence Intervals**: Narrower intervals indicate more precise estimates
- **Normality Tests**: Low p-values suggest non-normal distributions (may affect CI interpretation)

## Notes

- The script uses Gurobi for approximate sparsest cut calculation, which requires a license
- Symmetric topology generation significantly reduces computation time for large topologies
- Sequential stopping can save time when you have specific precision requirements
- Time limits are checked before each iteration, so actual runtime may slightly exceed the limit
