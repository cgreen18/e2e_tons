#!/bin/bash -l
# Rewrite a whole Chakra trace directory on a compute node.
# Usage: sbatch tools/slurm/chakra_rewrite.sbatch <src_dir> <dst_dir> <ranks> <passes...>
#SBATCH -A mithuna
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --mem=96G
#SBATCH -t 06:00:00
set -euo pipefail
cd /home/green456/e2e_tons
export PYTHONPATH=/home/green456/e2e_tons
src=$1; dst=$2; ranks=$3; shift 3
echo "host=$(hostname) src=$src dst=$dst ranks=0..$((ranks-1)) passes=$*"
date -Is
venv_py12/bin/python tools/chakra_et_rewrite.py "$src" "$dst" \
  --passes "$@" --rank-range "0-$((ranks-1))" --jobs 16
date -Is
