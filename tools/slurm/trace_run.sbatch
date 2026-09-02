#!/bin/bash -l
# Run ONE prepared trace job per array task, so the 12-job matrix runs
# concurrently instead of serially. Each job replays tens of millions of Chakra
# nodes across 128 ranks and takes well over an hour, so a sequential sweep
# would take most of a day.
#
# Usage:
#   sbatch --array=0-11 tools/slurm/trace_run.sbatch generated/traces_128/prepared/prepared.json
#
# `module` is a login-shell function, hence the -l; the load must NOT be
# swallowed or the ASTRA binaries die with exit 127 on a missing libprotobuf.
#SBATCH -A mithuna
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=200G
#SBATCH -t 20:00:00
set -euo pipefail
cd /home/green456/e2e_tons
module load protobuf/3.18.0
export PYTHONPATH=/home/green456/e2e_tons

prepared=$1
run_id=$(venv_py12/bin/python -c "
import json,sys
jobs=json.load(open('$prepared'))['jobs']
print(jobs[int('${SLURM_ARRAY_TASK_ID:-0}')]['run_id'])
")
echo "host=$(hostname) array_task=${SLURM_ARRAY_TASK_ID:-0} run_id=$run_id"
date -Is
venv_py12/bin/python tools/run_trace_experiment.py run "$prepared" --only "$run_id"
date -Is
echo "run_id=$run_id finished"
