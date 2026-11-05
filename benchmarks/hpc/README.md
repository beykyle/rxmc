# Running the benchmark on a cluster

Every independent run of the sampler benchmark can be a SLURM array task of
its own.  Each task writes `benchmarks/results/hpc/<kind>/<problem>/<index>.json`
and nothing else, so any number can run at once; nothing here touches the
committed `results/samplers.json` or `results/references.json` until you
merge (step 4).

## The pieces

| file | what it does |
|---|---|
| `task.py` | runs one task: `task.py KIND PROBLEM INDEX` (INDEX defaults to `$SLURM_ARRAY_TASK_ID`); `--list` shows what each index of a job is, `--count` how many there are |
| `submit.py` | submits any subset as array jobs, one per (kind, problem) |
| `job.sbatch` | the SLURM script every task runs in: resources, then `env.sh`, then the task |
| `env.sh` | site settings: modules, the venv, thread counts |
| `assemble.py` | after the runs: pools references, joins the sweep and seed-check pieces |

The kinds of task:

| kind | problems | tasks per problem |
|---|---|---|
| `ref` | any | `omp`, `alpha_gp`, `n40ca_gp`: 6 (three pocomc runs at 8192/4096, pooled into the reference, and three dynesty runs, rslice with 1000 live points, pooled into an independent check of it); every other problem: 1 (quadrature or a long NUTS run) |
| `run` | any | one per sampler the problem runs (4, or 8 for the three optical models) |
| `sweep` | any | 9: pocomc at 1024/512, 2048/1024 and 4096/2048, three seeds each |
| `seed` | `gp` only | 9: `gp_discrepancy`'s ladder, one rung at one of three other seed sets |

A sampler run is not split further: its chains are scored together (R-hat
needs all of them), and the benchmark times a sampler as a user runs it.

## Selecting what to run

```bash
# the refresh of the three optical-model problems (78 tasks in 10 arrays)
python benchmarks/hpc/submit.py --problems omp alpha_gp n40ca_gp --kinds ref run sweep seed
# every problem (default kinds: ref run sweep)
python benchmarks/hpc/submit.py --all
# a subset: two samplers on omp; one sweep setting; one seed set of the gp check
python benchmarks/hpc/submit.py --problems omp --kinds run --samplers NUTS pocomc
python benchmarks/hpc/submit.py --problems alpha_gp --kinds sweep --settings 1024/512
python benchmarks/hpc/submit.py --kinds seed --seed-sets 20
```

Add `--dry-run` to see the `sbatch` lines first.  A selection keeps each
task's array index, so its result lands in the same file whichever subset
produced it, and a failed task is rerun by selecting it again (or with
`sbatch --array=<index> ...` from the dry run).  `--time` overrides the
per-task walltimes in `submit.py`'s `TIME` table, which are about twice an
8-core laptop's: a population-reference pocomc run 1 to 2 h, a dynesty run
about 1 h, a sampler run
up to about 30 min on `omp` (PyMC is the slowest), a sweep run 5 to 40 min.

## Steps

1. Check out the commit to run and build an environment:

   ```bash
   git clone https://github.com/beykyle/rxmc.git && cd rxmc
   git checkout <commit>
   python3.12 -m venv ~/rxmc-venv && source ~/rxmc-venv/bin/activate
   pip install -e ".[validation,samplers]"
   ```

   `validation` brings the samplers the benchmark drives (blackjax, emcee,
   dynesty, pocomc) and `samplers` adds ptemcee and PyMC.  If torch would
   pull CUDA wheels you don't want, install it from the CPU index first.

2. Edit `benchmarks/hpc/env.sh` (modules, the venv path) and the
   `--partition`/`--account` lines of `benchmarks/hpc/job.sbatch`.  Each task
   runs through `srun --cpu-bind=cores`, so its threads (XLA's included,
   which ignore `OMP_NUM_THREADS`) stay on its own cores and the wall times
   stay comparable when tasks share a node; if your site forbids `srun`
   inside a batch job, take that line out and rely on its cgroups.

3. With the environment active, submit (`submit.py` above).  Logs go to
   `benchmarks/results/hpc/logs/<job>-<array id>_<task>.out`.

4. Bring back `benchmarks/results/hpc/` (everything but `logs/`): commit it
   on a branch and push, or copy it into the same place locally.  Then:

   ```bash
   python benchmarks/hpc/assemble.py      # references; sweep and seed-check tables
   python benchmarks/merge_results.py benchmarks/results/hpc/references.json \
       benchmarks/results/hpc/run/*/*.json
   python benchmarks/bench_samplers.py --report benchmarks/results/samplers.json --markdown
   python benchmarks/sweep_pocomc.py --report benchmarks/results/hpc/sweep_alpha_gp.json
   ```

   A population reference whose dynesty pool disagrees with it, on log Z
   (beyond twice the combined error), on the mode share (by more than 0.03)
   or on the moments (by more than 0.2 sd), is saved with
   `"corroborated": false` and the failed checks named; one whose own pocomc
   runs disagree is saved with `"accepted": false`.  Look at either before
   trusting anything scored against it.
