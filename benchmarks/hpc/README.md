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
| `ref` | any | `omp`: 6 (three pocomc runs at 8192/4096, pooled into the reference, and three dynesty runs, rslice with 1000 live points, pooled into an independent check of it); `alpha_gp`, `n40ca_gp`: 53 (the same, with fifty dynesty runs: single runs scatter too much for three to check the pool); every other problem: 1 (quadrature or a long NUTS run) |
| `run` | any | one per sampler the problem runs (4, or 8 for the three optical models) |
| `sweep` | any | `gp`, `modes`, `elastic`: 21, seven settings at three seeds each (dynesty rwalk with 200 and 500 live points, rslice with 250 and 500; pocomc at 512/256, 1024/512 and 2048/1024); every other problem: 9, pocomc at 1024/512, 2048/1024 and 4096/2048 |
| `seed` | `gp` only | 9: `gp_discrepancy`'s ladder, one rung at one of three other seed sets |

A sampler run is not split further: its chains are scored together (R-hat
needs all of them), and the benchmark times a sampler as a user runs it.

## Selecting what to run

```bash
# the three optical-model problems, every kind (172 tasks in 10 arrays)
python benchmarks/hpc/submit.py --problems omp alpha_gp n40ca_gp --kinds ref run sweep seed
# every problem (default kinds: ref run sweep)
python benchmarks/hpc/submit.py --all
# a subset: two samplers on omp; one sweep setting; one seed set of the gp check
python benchmarks/hpc/submit.py --problems omp --kinds run --samplers NUTS pocomc
python benchmarks/hpc/submit.py --problems alpha_gp --kinds sweep --settings 1024/512
python benchmarks/hpc/submit.py --kinds seed --seed-sets 20
python benchmarks/hpc/submit.py --problems gp --kinds sweep --settings "dynesty rslice 250"
# by array index (for any kind): the dynesty runs added to alpha_gp's reference
python benchmarks/hpc/submit.py --problems alpha_gp --kinds ref --indices 12-52
```

Add `--dry-run` to see the `sbatch` lines first, and `@<site file>` (step 2)
for your site's account, partition and the like.  A selection keeps each
task's array index, so its result lands in the same file whichever subset
produced it, and a failed task is rerun by selecting it again (or with
`sbatch --array=<index> ...` from the dry run).  `--time` overrides the
per-task walltimes in `submit.py`'s `TIME` table, which are about twice an
8-core laptop's or a cluster run's: a population-reference pocomc or
dynesty run up to 50 min on `alpha_gp` (25 on `n40ca_gp`), a sampler run
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

2. Put your site's settings in a file outside the repository, one argument
   per line, for `submit.py` to read as `@file` (nothing in the repository
   needs editing):

   ```
   # ~/rxmc-hpc/site.args
   --account=my-account
   --partition=my-partition
   --constraint=my-node-feature
   --venv=~/rxmc-venv
   --sbatch=--qos=my-qos
   ```

   Every line is optional: `--account`, `--partition` and `--constraint` go to
   `sbatch` as they are, `--venv` is exported to the job as `RXMC_VENV` (else
   it is inherited from your shell, else `~/rxmc-venv`), and `--sbatch`
   passes any other `sbatch` option.  Each task
   runs through `srun --cpu-bind=cores`, so its threads (XLA's included,
   which ignore `OMP_NUM_THREADS`) stay on its own cores and the wall times
   stay comparable when tasks share a node; if your site forbids `srun`
   inside a batch job, take that line out of `job.sbatch` and rely on its
   cgroups.  A node constraint naming one CPU generation keeps the wall times
   comparable across tasks too.

3. With the environment active, submit (`submit.py` above), adding
   `@$HOME/rxmc-hpc/site.args` to each command line.  Logs go to
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
