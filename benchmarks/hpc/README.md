# Running the benchmark refresh on a cluster

The long jobs of the sampler benchmark run independently, one SLURM job
each, writing a file of its own under `benchmarks/results/hpc/`.  Nothing
here writes the committed `results/samplers.json` or `results/references.json`;
`benchmarks/merge_results.py` does that afterwards.

## What runs

For each of `omp`, `alpha_gp` and `n40ca_gp`:
- `omp`: exactly the log-space problem of `local_optical_model_calibration`;
- `alpha_gp`: the power-of-q problem of `alpha_ca_error_model_comparison`;
- `n40ca_gp`: the GP rung of `gp_discrepancy`'s cross-section ladder.

| job | what | output |
|---|---|---|
| `ref_<problem>` | the reference: three pooled pocomc runs at 8192/4096, then one dynesty run (2000 live points) that must agree with the pool | `ref_<problem>.json` |
| `run_<problem>_<sampler>` | one sampler at the benchmark's budget (emcee, dynesty, pocomc, NUTS, ptemcee, HMC, PyMC NUTS, PyMC HMC) | `samplers_<problem>_<sampler>.json` |
| `sweep_<problem>` | pocomc at 1024/512, 2048/1024 and 4096/2048, three seeds each | `sweep_<problem>.json` |
| `gp_seed_check` | `gp_discrepancy`'s ladder at its settings, three other sets of seeds | `gp_seed_check.json` |

31 jobs in all.  The sampler and sweep jobs need no reference: they are
scored against it when the results are reported.  The sweeps and the seed
check save every run as it ends and resume where they stopped if resubmitted.

**Rough cost** (from an 8-core laptop): a reference about 3 to 6 h; a
sampler job up to about 30 min (`omp`; the PyMC runs are the slowest);
a sweep 1.5 to 4 h; the seed check about 30 min.  `submit_all.sh` asks for
about twice that.

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

3. From the repository root:

   ```bash
   bash benchmarks/hpc/submit_all.sh --dry-run   # look first
   bash benchmarks/hpc/submit_all.sh
   ```

   Logs go to `benchmarks/results/hpc/logs/`.  A job that runs out of time
   can be resubmitted with a longer `--time`; sweeps and the seed check resume.

4. Bring back `benchmarks/results/hpc/*.json`: commit them on a branch and
   push it, or copy them into the same directory locally.  Then:

   ```bash
   python benchmarks/merge_results.py benchmarks/results/hpc/ref_*.json \
       benchmarks/results/hpc/samplers_*.json
   python benchmarks/bench_samplers.py --report benchmarks/results/samplers.json --markdown
   python benchmarks/sweep_pocomc.py --report benchmarks/results/hpc/sweep_alpha_gp.json
   ```

   A reference whose dynesty corroboration disagrees is saved with
   `"corroborated": false`; look at it before trusting anything scored
   against it.
