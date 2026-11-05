# Running the benchmark refresh on a cluster

The long jobs of the sampler benchmark run independently, one SLURM job
each, writing a file of its own under `benchmarks/results/hpc/`.  Nothing
here writes the committed `results/samplers.json` or `results/references.json`;
`benchmarks/merge_results.py` does that afterwards.

## What runs

Ten SLURM array jobs, one task per independent run (72 in all).  Each task
writes `benchmarks/results/hpc/<kind>/<problem>/<index>.json` and nothing
else, so they can all run at once.  `python benchmarks/hpc/task.py KIND
PROBLEM --list` shows what each index of a job is.

| array job | tasks | each task |
|---|---|---|
| `ref_<problem>` | 4 | one pocomc run at 8192/4096 (three seeds), or the dynesty run (2000 live points) that must agree with their pool |
| `run_<problem>` | 8 | one sampler at the benchmark's budget: emcee, dynesty, pocomc, NUTS, ptemcee, HMC, PyMC NUTS, PyMC HMC |
| `sweep_<problem>` | 9 | pocomc at one of 1024/512, 2048/1024, 4096/2048 and one of three seeds |
| `seed_gp` | 9 | one rung of `gp_discrepancy`'s ladder at one of three other seed sets |

The problems are `omp` (exactly the log-space problem of
`local_optical_model_calibration`), `alpha_gp` (the power-of-q problem of
`alpha_ca_error_model_comparison`) and `n40ca_gp` (the GP rung of
`gp_discrepancy`'s cross-section ladder).

A sampler run is not split further: its chains are scored together (R-hat
needs all of them), and the benchmark times a sampler as a user runs it.
What needs every piece (pooling a reference and checking dynesty against
it, the sweep and seed-check tables) happens afterwards, in `assemble.py`.

**Rough cost per task** (an 8-core laptop): a reference pocomc run 1 to
2 h, dynesty up to about 1 h, a sampler run up to about 30 min (`omp`; the
PyMC runs are the slowest), a sweep run 5 to 40 min, a seed-check run a few
minutes.  `submit_all.sh` asks for about twice that.

## Steps

1. Check out the commit to run, and build an environment with the extras:

   ```bash
   git clone https://github.com/beykyle/rxmc.git && cd rxmc
   git checkout <commit>
   python3.12 -m venv ~/rxmc-venv && source ~/rxmc-venv/bin/activate
   pip install -e ".[validation,samplers]"
   ```

   `validation` brings the samplers the benchmark drives (blackjax, emcee,
   dynesty, pocomc) and `samplers` adds ptemcee and PyMC.  If torch would pull CUDA
   wheels you don't want, install it from the CPU index first.

2. Edit `benchmarks/hpc/env.sh` (modules, the venv path) and the
   `--partition`/`--account` lines of `benchmarks/hpc/job.sbatch`.

3. With the environment active (`submit_all.sh` asks `task.py` how many
   tasks each job has):

   ```bash
   bash benchmarks/hpc/submit_all.sh --dry-run   # look first
   bash benchmarks/hpc/submit_all.sh
   ```

   Logs go to `benchmarks/results/hpc/logs/<job>-<array id>_<task>.out`.  A
   failed or timed-out task is rerun alone, e.g.
   `sbatch --array=5 --time=6:00:00 --job-name=run_omp benchmarks/hpc/job.sbatch python benchmarks/hpc/task.py run omp`.

4. Bring back `benchmarks/results/hpc/` (everything but `logs/`): commit it
   on a branch and push, or copy it into the same place locally.  Then:

   ```bash
   python benchmarks/hpc/assemble.py      # pools the references; joins sweeps, seed check
   python benchmarks/merge_results.py benchmarks/results/hpc/references.json \
       benchmarks/results/hpc/run/*/*.json
   python benchmarks/bench_samplers.py --report benchmarks/results/samplers.json --markdown
   python benchmarks/sweep_pocomc.py --report benchmarks/results/hpc/sweep_alpha_gp.json
   ```

   A reference whose dynesty corroboration disagrees is saved with
   `"corroborated": false`; look at it before trusting anything scored
   against it.
