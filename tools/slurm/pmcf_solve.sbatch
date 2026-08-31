#!/bin/bash
# Solve one canonical pMCF all-to-all instance on a compute node.
#
# The finite-candidate maximum-concurrent-flow LP is large (up to ~463k path
# columns for PT) and must not run on a login node.  The unrestricted Gurobi
# license is served by license.rcac.purdue.edu, but it is NOT picked up
# automatically: without GRB_LICENSE_FILE gurobipy falls back to its
# size-limited pip license and rejects the model, so the variable is set here
# explicitly.
#
# Usage:
#   sbatch --job-name=pmcf-tons --output=<log> \
#     tools/slurm/pmcf_solve.sbatch <map> <rallpaths> <out.xml> <report.json>
#SBATCH -A mithuna
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 08:00:00

set -euo pipefail

REPO=/home/green456/e2e_tons
export GRB_LICENSE_FILE=/apps/spack/negishi/apps/gurobi/10.0.1-gcc-8.5.0-q4gd3h5/license/gurobi.lic
export PYTHONPATH="$REPO"

map=$1; candidates=$2; output=$3; report=$4

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} cpus=${SLURM_CPUS_PER_TASK:-?}"
echo "map=$map"
echo "candidates=$candidates"
echo "output=$output"
echo "report=$report"
date -Is

cd "$REPO"
mkdir -p "$(dirname "$output")" "$(dirname "$report")"

"$REPO"/venv_py12/bin/python tools/collective_schedule.py pmcf-a2a \
  --topology "$map" \
  --candidates "$candidates" \
  --subchunks 8 \
  --solver gurobi \
  --threads "${SLURM_CPUS_PER_TASK:-8}" \
  --seed 1 \
  --output "$output" \
  --report "$report"

date -Is
echo "--- report ---"
cat "$report"
