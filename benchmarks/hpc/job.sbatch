#!/bin/bash
# One benchmark task: the command line comes after the script.  submit.py
# submits each kind of run as an array job, e.g.
#   sbatch --array=0-3 --job-name=ref_omp --time=3:00:00 job.sbatch \
#       python benchmarks/hpc/task.py ref omp
# and task.py runs the piece $SLURM_ARRAY_TASK_ID stands for.

#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=benchmarks/results/hpc/logs/%x-%A_%a.out
# Site settings (account, partition, node constraint, venv) are not set here:
# submit.py passes them on the sbatch command line (its --account, --partition,
# --constraint, --venv and --sbatch options, or an @file of them).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source benchmarks/hpc/env.sh
echo "$(date -Is) $(git rev-parse --short HEAD) task ${SLURM_ARRAY_TASK_ID:-} $*"
if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v srun >/dev/null; then
    # bind the task to its allocated cores: XLA does not read OMP_NUM_THREADS,
    # and binding keeps all its threads on the task's cores, so tasks sharing
    # a node do not collide (and the wall times stay comparable)
    srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" --cpu-bind=cores "$@"
else
    "$@"
fi
echo "$(date -Is) done"
