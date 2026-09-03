#!/bin/bash -l
# Profile all ranks of the four original Chakra model traces on a compute node.
#
# This is a pure-Python reader.  The login shell initializes the normal Negishi
# environment, but no ASTRA binary is launched and no protobuf shared-library
# module is needed or loaded.
#SBATCH -A mithuna
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem=96G
#SBATCH -t 12:00:00

set -euo pipefail

REPO=/home/green456/e2e_tons-etrw
TRACE_ROOT=/home/green456/e2e_tons/ai_traces
PYTHON=/home/green456/e2e_tons/venv_py12/bin/python
export PYTHONPATH="$REPO"

cd "$REPO"
mkdir -p generated
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} cpus=${SLURM_CPUS_PER_TASK:-?}"
echo "trace_root=$TRACE_ROOT"
date -Is
"$PYTHON" tools/chakra_collective_profile.py \
  --trace-root "$TRACE_ROOT" \
  --output-csv generated/chakra_collective_profile.csv \
  --output-json generated/chakra_collective_profile.json \
  --jobs 16
date -Is
