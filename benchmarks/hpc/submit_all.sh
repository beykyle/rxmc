#!/bin/bash
# Submit every job of the benchmark refresh, from the repository root:
#   bash benchmarks/hpc/submit_all.sh            # all of them
#   bash benchmarks/hpc/submit_all.sh --dry-run  # print the sbatch lines only
# Every job writes its own file under benchmarks/results/hpc/, so they can run
# at once; merge them afterwards with benchmarks/merge_results.py.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
OUT=benchmarks/results/hpc
mkdir -p "$OUT/logs"
DRY=${1:-}

submit() {  # name walltime command...
    local name=$1 time=$2
    shift 2
    if [[ "$DRY" == "--dry-run" ]]; then
        echo sbatch --job-name="$name" --time="$time" benchmarks/hpc/job.sbatch "$@"
    else
        sbatch --job-name="$name" --time="$time" benchmarks/hpc/job.sbatch "$@"
    fi
}

PROBLEMS=(omp alpha_gp n40ca_gp)
SAMPLERS=("emcee" "dynesty" "pocomc" "NUTS" "ptemcee" "HMC" "PyMC NUTS" "PyMC HMC")
# walltimes, generous: about twice the local estimates in README.md
declare -A REF_TIME=([omp]=10:00:00 [alpha_gp]=14:00:00 [n40ca_gp]=6:00:00)
declare -A RUN_TIME=([omp]=6:00:00 [alpha_gp]=8:00:00 [n40ca_gp]=3:00:00)
declare -A SWEEP_TIME=([omp]=6:00:00 [alpha_gp]=10:00:00 [n40ca_gp]=3:00:00)

for p in "${PROBLEMS[@]}"; do
    submit "ref_$p" "${REF_TIME[$p]}" \
        python benchmarks/bench_samplers.py --reference --problems "$p" \
        --references "$OUT/ref_$p.json"
    for s in "${SAMPLERS[@]}"; do
        tag=${s// /_}
        submit "run_${p}_$tag" "${RUN_TIME[$p]}" \
            python benchmarks/bench_samplers.py --problems "$p" --only "$s" \
            --out "$OUT/samplers_${p}_$tag.json" --references "$OUT/none.json"
    done
    submit "sweep_$p" "${SWEEP_TIME[$p]}" \
        python benchmarks/sweep_pocomc.py "$p" --out "$OUT/sweep_$p.json"
done
submit gp_seed_check 3:00:00 \
    python benchmarks/gp_seed_check.py --out "$OUT/gp_seed_check.json"
