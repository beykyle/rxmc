#!/bin/bash
# One benchmark job: the command line comes after the script, e.g.
#   sbatch --job-name=ref_omp --time=08:00:00 job.sbatch \
#       python benchmarks/bench_samplers.py --reference --problems omp ...
# submit_all.sh submits every job of the refresh this way.
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=benchmarks/results/hpc/logs/%x-%j.out
# --- edit: your site's partition and account ----------------------------------
##SBATCH --partition=CHANGE_ME
##SBATCH --account=CHANGE_ME
# ------------------------------------------------------------------------------
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source benchmarks/hpc/env.sh
echo "$(date -Is) $(git rev-parse --short HEAD) $*"
"$@"
echo "$(date -Is) done"
