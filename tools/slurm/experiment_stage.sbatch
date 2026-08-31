#!/bin/bash -l
# Run one manifest-driven experiment stage on a compute node.
#
# Usage:
#   sbatch --job-name=... --output=... tools/slurm/experiment_stage.sbatch \
#     prepare experiments/tons_128/manifest.json
#   sbatch ... tools/slurm/experiment_stage.sbatch \
#     run generated/tons_128/prepared/prepared.json
#   sbatch ... tools/slurm/experiment_stage.sbatch \
#     analyze generated/tons_128/prepared/prepared.json
#SBATCH -A mithuna
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 08:00:00

set -euo pipefail

REPO=/home/green456/e2e_tons
export PYTHONPATH="$REPO"
# prepare's pMCF preflight only reads the certified reports, but keep the
# license reachable so any Gurobi import in this stage behaves the same as the
# solve jobs.
export GRB_LICENSE_FILE=/apps/spack/negishi/apps/gurobi/10.0.1-gcc-8.5.0-q4gd3h5/license/gurobi.lic

cd "$REPO"

# `module` is a shell function that only exists in a login shell, hence the
# `-l` in the shebang.  Do NOT swallow a failure here: without protobuf on the
# library path the ASTRA binaries die with exit 127 on the first job and the
# whole sweep aborts partway through.
module load protobuf/3.18.0

for binary in \
  simul/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware \
  simul/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware
do
  if ldd "$binary" | grep -q "not found"; then
    echo "FATAL: $binary has unresolved shared libraries:" >&2
    ldd "$binary" | grep "not found" >&2
    exit 1
  fi
done
echo "both analytical binaries resolve their shared libraries"

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} stage=$1"
date -Is
"$REPO"/venv_py12/bin/python tools/run_experiment.py "$@"
date -Is
echo "stage $1 finished"
