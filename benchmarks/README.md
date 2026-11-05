# Benchmarks

Two benchmarks of the JAX rxmc, and the tools to run them locally or on a
cluster.  [`docs/benchmarks.md`](../docs/benchmarks.md) has the set-up, the
scoring and the results; this file is a map of the directory.

## The problems: `problems.py`

`PROBLEMS[name]()` returns a compiled `rxmc.Problem` and a fixed set of
parameter points.  They span the cost regimes and posterior geometries the
benchmarks care about:

| problem | parameters | what it exercises |
|---|---|---|
| `linear` | 2 | a line with reported errors: per-call overhead |
| `modes` | 9 | free normalisations, offsets and a shared mode: the Woodbury path, a funnel |
| `gp`, `gp500` | 4 | a dense GP discrepancy over 100 or 500 points: Cholesky-bound, a funnel |
| `elastic` | 5 | `ElasticXS` at four energies: the reaction solver, one mode |
| `ias` | 4 | an `IsobaricAnalogPN` prediction |
| `hierarchy` | 33 | 30 lines with non-centred intercepts: high dimension |
| `omp` | 9 | exactly `local_optical_model_calibration`'s log-space p + 40Ca problem: two separated islands |
| `alpha_gp` | 9 | exactly `alpha_ca_error_model_comparison`'s power-of-q GP problem: two families of potentials and GP hyperparameters |
| `n40ca_gp` | 7 | exactly the GP rung of `gp_discrepancy`'s cross-section ladder (`n40ca_ladder()` builds all three rungs) |

## The density benchmark: `bench_log_posterior.py`, `compare.py`

Times `log_posterior`, its gradient and the batched densities on every
problem; `compare.py` sets two saved runs side by side (the numpy rxmc
against the JAX one: `results/numpy-jitr3.json`, `results/jax-jitr4.json`).

## The sampler benchmark

| file | what it does |
|---|---|
| `bench_samplers.py` | runs the samplers (emcee, dynesty, pocomc and NUTS everywhere; ptemcee, HMC and PyMC on the optical models) at the budgets in `BUDGETS`, scores them against the references, and prints the tables (`--report ... --markdown`); `--reference` computes references |
| `reference.py` | the reference posteriors: exact quadrature (`linear`, `gp`, `hierarchy`), a long NUTS run (`modes`, `elastic`), or three pooled pocomc runs checked against pooled dynesty runs, three or nine (`omp`, `alpha_gp`, `n40ca_gp`) |
| `sweep_pocomc.py` | pocomc's resolution swept on a problem, several seeds per setting, and the table of it |
| `gp_seed_check.py` | `gp_discrepancy`'s ladder rerun with other seeds |
| `merge_results.py` | merges per-job result files into `results/samplers.json` and `results/references.json` |
| `results/` | the committed results the docs are built from |
| `hpc/` | running any subset of the above as SLURM array jobs: see [`hpc/README.md`](hpc/README.md) |

Locally:

```bash
python benchmarks/bench_samplers.py --out benchmarks/results/samplers.json --problems linear gp
python benchmarks/bench_samplers.py --out benchmarks/results/samplers.json --problems omp --only pocomc
python benchmarks/bench_samplers.py --report benchmarks/results/samplers.json --markdown
python benchmarks/bench_samplers.py --reference --problems modes
python benchmarks/sweep_pocomc.py n40ca_gp --out sweep.json --settings 1024/512 --seeds 0
```

The three optical-model problems take hours per sampler and reference; run
them on a cluster.
