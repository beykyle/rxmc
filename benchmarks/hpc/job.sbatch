#!/bin/bash
# One benchmark task: the command line comes after the script.  submit_all.sh
# submits each kind of run as an array job, e.g.
#   sbatch --array=0-3 --job-name=ref_omp --time=3:00:00 job.sbatch \
#       python benchmarks/hpc/task.py ref omp
# and task.py runs the piece $SLURM_ARRAY_TASK_ID stands for.

#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=benchmarks/results/hpc/logs/%x-%A_%a.out
# --- edit: your site's partition and account ----------------------------------
##SBATCH --partition=CHANGE_ME
##SBATCH --account=CHANGE_ME
# ------------------------------------------------------------------------------
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source benchmarks/hpc/env.sh
echo "$(date -Is) $(git rev-parse --short HEAD) task ${SLURM_ARRAY_TASK_ID:-} $*"
"$@"
echo "$(date -Is) done"
