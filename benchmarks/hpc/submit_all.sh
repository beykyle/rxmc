#!/bin/bash
# Submit the benchmark refresh as array jobs, one task per independent run:
#   bash benchmarks/hpc/submit_all.sh            # everything
#   bash benchmarks/hpc/submit_all.sh --dry-run  # print the sbatch lines only
# Run from anywhere inside the repository.  Each task writes
# benchmarks/results/hpc/<kind>/<problem>/<index>.json; assemble.py and
# merge_results.py put them together afterwards (README.md).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
mkdir -p benchmarks/results/hpc/logs
DRY=${1:-}
PY=${PYTHON:-python}

submit() {  # kind problem walltime-per-task
    local kind=$1 problem=$2 time=$3 n
    n=$($PY benchmarks/hpc/task.py "$kind" "$problem" --count)
    local cmd=(sbatch --array="0-$((n - 1))" --job-name="${kind}_${problem}"
               --time="$time" benchmarks/hpc/job.sbatch
               python benchmarks/hpc/task.py "$kind" "$problem")
    if [[ "$DRY" == "--dry-run" ]]; then echo "${cmd[*]}"; else "${cmd[@]}"; fi
}

# per-task walltimes, about twice the local estimates in README.md
for p in omp alpha_gp n40ca_gp; do
    case $p in
        omp)      ref=4:00:00 run=2:00:00 sweep=3:00:00 ;;
        alpha_gp) ref=6:00:00 run=3:00:00 sweep=4:00:00 ;;
        n40ca_gp) ref=2:00:00 run=1:00:00 sweep=1:00:00 ;;
    esac
    submit ref "$p" "$ref"      # 3 pocomc seeds at 8192/4096 + dynesty
    submit run "$p" "$run"      # one task per sampler
    submit sweep "$p" "$sweep"  # one task per (setting, seed)
done
submit seed gp 1:00:00          # one task per (seed set, rung)
