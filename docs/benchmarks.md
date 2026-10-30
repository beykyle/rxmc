# Benchmarks

Two benchmarks live in `benchmarks/`: one times the densities, the other
compares samplers on them.  Both build their problems from
`benchmarks/problems.py` and write JSON to `benchmarks/results/`, which is
checked in, so the tables below can be regenerated without a rerun.

## Set-up

From the repository root:

```bash
uv sync --managed-python --extra examples --extra samplers
```

- `--managed-python` uses uv's own CPython, which ships the Python headers.
  PyMC's PyTensor needs them (and a C++ compiler) to compile its graphs; with
  a system Python that lacks `pyconfig.h` it falls back to pure Python and
  the PyMC timings suffer.
- The `samplers` extra adds ptemcee, PyMC and pocomc.  pocomc brings torch,
  which `uv` takes from PyTorch's CPU index (`[tool.uv.sources]` in
  `pyproject.toml`), avoiding about 3 GB of CUDA wheels.  With pip, install
  torch first: `pip install torch --index-url https://download.pytorch.org/whl/cpu`.
- Run on an otherwise idle machine; every number is a wall time.

## The problems

| name | parameters | what it exercises |
|---|---|---|
| `linear` | 2 | a line with reported errors (recipe 1): per-call overhead |
| `modes` | 9 | six comparisons with free normalisations, an offset each and one shared mode (recipes 4, 37): the Woodbury path |
| `gp`, `gp500` | 4 | a dense GP discrepancy over 100 or 500 points (recipe 7): Cholesky-bound |
| `elastic` | 5 | `ElasticXS` at four energies with a proportional model error (recipes 15, 26): the reaction solver |
| `ias` | 4 | an `IsobaricAnalogPN` prediction at one energy |
| `hierarchy` | 33 | 30 lines sharing a slope, their intercepts non-centred about a common mean (recipe 38): high dimension |
| `omp` | 9 | the log-space p + ⁴⁰Ca optical model of `local_optical_model_calibration`: EXFOR O1199007 at 35 MeV, seven potential parameters and two inferred relative errors |

## Densities: `bench_log_posterior.py`

```bash
python benchmarks/bench_log_posterior.py --out benchmarks/results/jax-jitr4.json
python benchmarks/bench_log_posterior.py --out ... --problems linear gp --emcee 500
python benchmarks/compare.py benchmarks/results/numpy-jitr3.json benchmarks/results/jax-jitr4.json
```

For each problem it records `log_posterior` at fixed parameter points (a
value regression), the first-call latency (compilation), the steady-state
time per call, the gradient's cost, the throughput of `log_posterior_batch`
at batch sizes 1, 32 and 256, and, with `--emcee N`, the wall time of 32
walkers for `N` steps.  Timings are medians over repeated calls after
warm-up.  `compare.py` prints two runs side by side, with time ratios and
the largest relative difference of the values.

The script runs against whichever rxmc is importable.  The baseline
`numpy-jitr3.json` is the numpy rxmc before the JAX rewrite (`main`), on
jitr 3.0.1, whose gradient is a central finite difference of `2 ndim`
calls; to reproduce it, install that rxmc and jitr in a separate
environment and run this branch's script there.

Against that baseline (same machine, CPU):

| problem | `log_posterior` | gradient (FD baseline → autodiff) | vmap, 256 rows | emcee 32×500 |
|---|---|---|---|---|
| line, reported errors | 0.16 → 0.10 ms | 0.66 → 0.11 ms | 6.2k → 1.06M θ/s | 2.9 → 1.9 s |
| 6 comparisons, 13 modes | 1.38 → 0.20 ms | 30.4 → 0.39 ms | 0.5k → 41k θ/s | |
| GP, N = 100 | 0.79 → 0.27 ms | 6.4 → 1.2 ms | 1.3k → 4.4k θ/s | 10.3 → 4.6 s |
| GP, N = 500 | 10.8 → 5.3 ms | 90 → 32 ms | 87 → 147 θ/s | |
| elastic, 4 energies | 7.0 → 2.6 ms | 70 → 6.0 ms | 142 → 331 θ/s | 106 → 45 s |
| (p,n) IAS, 1 energy | 5.2 → 2.1 ms | 43 → 4.1 ms | 191 → 450 θ/s | |

The first call of a new problem pays compilation, 0.1 to 1 s.  Values agree
to 1e-13 except for the reaction models, where jitr v4 solves every partial
wave up to `lmax` (elastic agrees to 1e-6).

## Samplers: `bench_samplers.py`

```bash
# everything (about 4.5 h)
python benchmarks/bench_samplers.py --out benchmarks/results/samplers.json
# some problems
python benchmarks/bench_samplers.py --out benchmarks/results/samplers.json --problems linear gp
# re-run some samplers into the saved file, keeping the others' results
python benchmarks/bench_samplers.py --out benchmarks/results/samplers.json --problems omp --only pocomc NUTS
# print a saved run, or the Markdown table below
python benchmarks/bench_samplers.py --report benchmarks/results/samplers.json --markdown
# recompute the reference posteriors (about 3 h, most of it omp)
python benchmarks/bench_samplers.py --reference
```

emcee, dynesty, pocomc and NUTS run on every problem; `omp` also runs
ptemcee, HMC and PyMC's NUTS and HMC.  The results are written after each
problem, so an interrupted run keeps what it finished.

### The samplers and their starts

Each sampler starts the way it is usually used, and pays for the search
that finds its start: the search's wall time and evaluations are added to
its own.

| sampler | start | budget on `omp` |
|---|---|---|
| [emcee](https://emcee.readthedocs.io/) | a ball about the MLE (`rx.optimize.find_mle`, Nelder–Mead) | 32 walkers × 3000 steps |
| [ptemcee](https://github.com/NuclearTalent/ptemcee_2025) | the same ball, at every temperature | 32 walkers × 8 temperatures × 2000 steps, adaptive ladder |
| [dynesty](https://dynesty.readthedocs.io/) | the prior | rwalk, 500 live points, to Δlog Z = 0.5 |
| [pocomc](https://pocomc.readthedocs.io/) | the prior (the unit cube, with `log_likelihood(prior_transform(u))`) | `n_effective` = 1024, `n_active` = 512 (twice its defaults) |
| NUTS ([blackjax](https://blackjax-devs.github.io/blackjax/)) | a ball about the MAP (`rx.optimize.find_map`, L-BFGS-B) | 2 chains × (300 warm-up + 1000 draws) |
| HMC (blackjax) | the same | as NUTS, 20 leapfrog steps |
| NUTS, HMC ([PyMC](https://www.pymc.io/)) | the same | as blackjax; HMC with path length 2 |

A ball is `to_theta(to_unconstrained(start) + 0.01 N(0, I))`, inside the
support.  The gradient samplers run on `log_posterior_unconstrained`
(recipe 41); PyMC calls it through a PyTensor Op whose gradient is
`jax.grad`.  On `hierarchy` NUTS starts from prior draws instead: a
hierarchy's density peaks in the neck of its funnel, so the MAP is a corner
the posterior hardly visits.  The other problems' budgets are in `BUDGETS`
at the top of the script.

dynesty's and pocomc's settings are the cheapest of a sweep that are
correct (below) on the optical models over three seeds.  At their defaults
both were biased on `omp`, by about a quarter of a posterior standard
deviation:

| setting | `gp` bias / width | `modes` | `elastic` | `omp` | `omp` wall time |
|---|---|---|---|---|---|
| dynesty rwalk, 200 live points (default) | 0.54 / 0.45 | | 0.02 / 0.04 | 0.26 / 0.22 | 192 s |
| dynesty rwalk, 500 | 0.21 / 0.11 | 0.13 / 0.22 | 0.04 / 0.02 | 0.05 / 0.08 | 460–900 s |
| dynesty rslice, 250 | 0.08 / 0.09 | 0.22 / 0.19 | 0.02 / 0.03 | 0.13 / 0.08 | 1682 s |
| dynesty rslice, 500 | 0.12 / 0.07 | 0.13 / 0.08 | 0.05 / 0.03 | 0.71 / 1.05 | 3661 s |
| pocomc 512/256 (default) | 0.63 / 0.51 | | 0.02 / 0.02 | 0.26 / 0.15 | 225 s |
| pocomc 1024/512 | 0.61 / 0.42 | 0.59 / 0.30 | 0.02 / 0.03 | 0.05 / 0.09 | 260–380 s |
| pocomc 2048/1024 | 0.35 / 0.13 | 0.16 / 0.08 | 0.02 / 0.01 | 0.03 / 0.06 | 537 s |

The chosen rows' `omp` columns are the worst of three seeds.  No setting
tried is correct on `gp` and `modes`, whose posteriors are funnels (see the
discussion); more resolution helps there, but slowly.

### How they are scored

- **Calls.**  A call is one evaluation of the log density and, for the
  gradient samplers and the MAP search, its gradient.  Each run's calls are
  split by phase:

  | phase | what it is |
  |---|---|
  | search | the MLE or MAP search that found the start (Nelder–Mead steps are value calls, L-BFGS-B steps value-and-gradient calls) |
  | warm-up | emcee's and ptemcee's discarded burn-in (every rung of ptemcee's ladder), NUTS's and HMC's step-size and mass-matrix adaptation, pocomc's tempering from the prior until β reaches 1 |
  | sampling | the calls that produced the kept draws: post-burn-in steps, post-warm-up trajectories, dynesty's whole run, pocomc's iterations at β = 1 |
  | evidence | pocomc's importance sampling of its normalizing flow for log Z |

- **Time per call.**  Wall time over calls: what a call costs on this
  machine, sampler overhead included.  emcee, ptemcee and pocomc hand over
  many points at once and evaluate them in one `jax.vmap` call
  (`log_posterior_batch`; `log_prior_batch` and `log_likelihood_batch`
  through ptemcee's `mapper`; `log_likelihood_batch` on
  `prior_transform_batch` for pocomc), which removes the per-point Python
  overhead.  On a reaction model the solver dominates and batching buys
  nothing: on this CPU a vmapped `elastic` row costs about 3.9 ms against
  2.6 ms for a single call.  The batch densities run in chunks of at most
  `BATCH_CHUNK` = 256 rows, since a reaction model holds every row's
  partial-wave solutions at once (about 3 MB a row for `elastic`).  dynesty cannot batch: each live point's replacement is a
  sequential walk, so it parallelises only across processes (`pool`,
  `queue_size`), which the benchmark does not use.  pocomc's time also
  includes training its normalizing flow.  A gradient call costs a few
  value calls; blackjax evaluates `value_and_grad` once, while PyMC
  evaluates the value and the gradient as two Ops, repeating the forward
  pass, and adds PyTensor's own overhead.
- **ESS.**  The smallest bulk effective sample size over the parameters,
  from `blackjax.diagnostics` across chains (the walkers, for the
  ensembles).  It is reported per second of wall time, start search and
  compilation included, and per thousand calls.  ESS per call does not
  depend on the machine; ESS per second is what a user waits for.
- **Bias and width.**  Every run is scored against a reference posterior
  (`benchmarks/reference.py`, saved in `benchmarks/results/references.json`):
  its bias is the largest |mean − reference mean| over the parameters, and
  its width error the largest |log(sd / reference sd)|, both in reference
  standard deviations.  Above `TOLERANCE` = 0.1 the column is marked ✗.
  Single runs of the reference-quality samplers differ by up to about 0.09,
  so the threshold sits at the level of their Monte Carlo noise.
- **The ESS of weighted draws is optimistic.**  For dynesty and pocomc it is
  [Kish's](https://en.wikipedia.org/wiki/Effective_sample_size) ESS of the
  importance weights.  That ignores the correlation between particles that
  share an ancestor (pocomc's resampling, dynesty's random walks from live
  points), so it overstates their ESS, and pocomc stops when Kish's ESS
  reaches its `n_total` target, so its ESS sits near that target by
  construction.  The bias and width columns are the check that matters:
  pocomc on `gp` reports an ESS of 4400 with means 0.6 sd off.
- **Correct.**  A run is correct if it converged (next) and its bias and
  width are both within the tolerance; ESS is reported only for correct
  runs.
- **Convergence.**  A run has not converged (✗, and no ESS) if its split
  [R-hat](https://en.wikipedia.org/wiki/Gelman-Rubin_statistic) exceeds
  1.01 or, for an ensemble, its kept chain is shorter than 50 times emcee's
  integrated autocorrelation time τ.  The second rule matters: walkers that
  swap or stretch toward one another can agree, and pass R-hat, long before
  they have explored the posterior.
- **Evidence.**  log Z for dynesty, pocomc and ptemcee (thermodynamic
  integration over its ladder), against the exact value where there is one.

### The references

| problem | reference | how |
|---|---|---|
| `linear`, `gp`, `hierarchy` | exact moments and log Z | the log posterior is quadratic in all but a few parameters (none; the GP's log variance and log length scale; the hierarchy's τ).  At each node of a grid over those, one Newton step over the rest lands on the conditional mode, and the Hessian there is the conditional precision, both exact for a quadratic; the conditional normalisers weight the nodes. |
| `modes`, `elastic`, `omp` | moments | one long blackjax NUTS run, 4 chains × (1000 warm-up + 3000 draws) at target acceptance 0.95, from a ball about the MAP with a seed of its own, accepted only with R-hat ≤ 1.01 and no divergences (R-hat 1.002, 1.000, 1.002; min ESS 1551, 3905, 2594).  Chains started from prior draws can stick in a minor mode: on `modes`, one with a log posterior 40 below the main one. |

The benchmarked NUTS row is scored against a run that shares its algorithm,
not its draws; the other gradient samplers on `omp` (blackjax HMC, PyMC's
NUTS) agree with the reference to 0.09 sd, and on `gp` and `hierarchy`,
where the reference is exact, NUTS's means agree to 0.08.

### Results

| problem | sampler | wall time | calls | search | warm-up | sampling | evidence | ms/call | bias | width | min ESS | ESS/s | ESS/1k calls | R-hat | τ (steps) | log Z |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| linear (2) | emcee | 2 s | 96771 | 1% | 33% | 66% | 0% | 0.02 | 0.00 | 0.00 | 1855 | 1120.64 | 19.16 | 1.008 | 28 |  |
| linear (2) | dynesty | 34 s | 93609 | 0% | 0% | 100% | 0% | 0.37 | 0.02 | 0.03 | 1605 | 46.75 | 17.15 | — |  | 0.7 ± 0.24 |
| linear (2) | pocomc | 27 s | 53760 | 0% | 70% | 23% | 8% | 0.50 | 0.02 | 0.01 | 4292 | 159.70 | 79.84 | — |  | 0.6 ± 0.06 |
| linear (2) | NUTS | 6 s | 62145 | 0% | 49% | 51% | 0% | 0.09 | 0.03 | 0.01 | 726 | 124.34 | 11.69 | 1.000 |  |  |
| linear (2) | reference: quadrature |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0.64 (exact) |
| gp (4) | emcee | 18 s | 98389 | 2% | 33% | 65% | 0% | 0.18 | 0.09 | 0.03 | — | — | — | 1.053 ✗ | 86 (< 50 τ) |  |
| gp (4) | dynesty | 71 s | 139598 | 0% | 0% | 100% | 0% | 0.51 | 0.20 ✗ | 0.11 ✗ | — | — | — | — |  | 80.3 ± 0.28 |
| gp (4) | pocomc | 73 s | 99840 | 0% | 71% | 25% | 4% | 0.73 | 0.60 ✗ | 0.42 ✗ | — | — | — | — |  | 80.2 ± 0.04 |
| gp (4) | NUTS | 206 s | 245453 | 0% | 68% | 32% | 0% | 0.84 | 0.08 | 0.23 ✗ | — | — | — | 1.003 |  |  |
| gp (4) | reference: quadrature |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 80.48 (exact) |
| elastic (5) | emcee | 163 s | 51409 | 7% | 31% | 62% | 0% | 3.17 | 0.13 ✗ | 0.04 | — | — | — | 1.062 ✗ | 43 (< 50 τ) |  |
| elastic (5) | dynesty | 646 s | 232479 | 0% | 0% | 100% | 0% | 2.78 | 0.04 | 0.02 | 2399 | 3.71 | 10.32 | — |  | 495.9 ± 0.34 |
| elastic (5) | pocomc | 312 s | 84480 | 0% | 81% | 15% | 5% | 3.69 | 0.02 | 0.03 | 4501 | 14.42 | 53.28 | — |  | 496.0 ± 0.01 |
| elastic (5) | NUTS | 320 s | 50006 | 0% | 51% | 48% | 0% | 6.39 | 0.04 | 0.02 | 621 | 1.94 | 12.42 | 1.003 |  |  |
| elastic (5) | reference: NUTS, 4 x 3000 draws |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| hierarchy (33) | emcee | 18 s | 440232 | 7% | 31% | 62% | 0% | 0.04 | 17.47 ✗ | 2.12 ✗ | — | — | — | 1.045 ✗ | 578 (< 50 τ) |  |
| hierarchy (33) | dynesty | 4256 s | 1926706 | 0% | 0% | 100% | 0% | 2.21 | 0.05 | 0.11 ✗ | — | — | — | — |  | 76.2 ± 0.55 |
| hierarchy (33) | pocomc | 3410 s | 663040 | 0% | 92% | 7% | 1% | 5.14 | 0.39 ✗ | 0.21 ✗ | — | — | — | — |  | 73.9 ± 0.17 |
| hierarchy (33) | NUTS | 27 s | 215227 | 0% | 56% | 44% | 0% | 0.13 | 0.06 | 0.04 | 231 | 8.41 | 1.07 | 1.004 |  |  |
| hierarchy (33) | reference: quadrature |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 73.97 (exact) |
| omp (9) | emcee | 183 s | 110400 | 13% | 29% | 58% | 0% | 1.66 | 10.80 ✗ | 0.93 ✗ | — | — | — | 1.648 ✗ | 125 (< 50 τ) |  |
| omp (9) | dynesty | 464 s | 232476 | 0% | 0% | 100% | 0% | 2.00 | 0.05 | 0.08 | 3191 | 6.88 | 13.72 | — |  | -28.4 ± 0.30 |
| omp (9) | pocomc | 275 s | 96768 | 0% | 80% | 16% | 4% | 2.85 | 0.05 | 0.09 | 4512 | 16.39 | 46.63 | — |  | -28.1 ± 0.02 |
| omp (9) | NUTS | 995 s | 126631 | 1% | 54% | 45% | 0% | 7.86 | 0.06 | 0.05 | 510 | 0.51 | 4.03 | 1.001 |  |  |
| omp (9) | ptemcee | 892 s | 526400 | 3% | 32% | 65% | 0% | 1.69 | 0.73 ✗ | 1.18 ✗ | — | — | — | 1.002 | 126 (< 50 τ) | -34.1 ± 3.80 |
| omp (9) | HMC | 423 s | 52863 | 2% | 23% | 76% | 0% | 8.01 | 0.08 | 0.08 | 131 | 0.31 | 2.48 | 1.009 |  |  |
| omp (9) | PyMC NUTS | 1291 s | 105965 | 1% | 44% | 55% | 0% | 12.18 | 0.09 | 0.07 | 482 | 0.37 | 4.55 | 1.001 |  |  |
| omp (9) | PyMC HMC | 1314 s | 108262 | 1% | 79% | 20% | 0% | 12.14 | 0.15 ✗ | 0.13 ✗ | — | — | — | 1.005 |  |  |
| omp (9) | reference: NUTS, 4 x 3000 draws |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| modes (9) | emcee | 11 s | 109533 | 12% | 29% | 58% | 0% | 0.10 | 3.03 ✗ | 0.66 ✗ | — | — | — | 1.174 ✗ | 261 (< 50 τ) |  |
| modes (9) | dynesty | 92 s | 222314 | 0% | 0% | 100% | 0% | 0.41 | 0.13 ✗ | 0.22 ✗ | — | — | — | — |  | 103.2 ± 0.29 |
| modes (9) | pocomc | 130 s | 91136 | 0% | 62% | 33% | 4% | 1.43 | 0.59 ✗ | 0.29 ✗ | — | — | — | — |  | 102.7 ± 0.08 |
| modes (9) | NUTS | 62 s | 340473 | 0% | 69% | 31% | 0% | 0.18 | 0.05 | 0.04 | 604 | 9.72 | 1.77 | 1.004 |  |  |
| modes (9) | reference: NUTS, 4 x 3000 draws |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

Which sampler is correct, and fastest among those that are, on each problem:

| problem | correct | fastest correct |
|---|---|---|
| `linear` | all four | emcee, 2 s (NUTS 6 s) |
| `gp` | none at these budgets | emcee is closest (0.09 / 0.03), but its chain is 35 τ, not 50 |
| `modes` | NUTS | NUTS, 62 s |
| `elastic` | dynesty, pocomc, NUTS | pocomc, 312 s (NUTS 320 s, dynesty 646 s) |
| `hierarchy` | NUTS | NUTS, 27 s |
| `omp` | dynesty, pocomc, NUTS, HMC, PyMC NUTS | pocomc, 275 s (dynesty 464 s, NUTS 995 s) |

- **Funnels defeat the samplers that start from the prior.**  `gp`, `modes`
  and `hierarchy` each have a scale parameter that, when small, pins the
  others into a narrow neck: the GP's log variance against its bound, the
  shared normalisation mode's width, the hierarchy's τ.  dynesty and pocomc
  under-visit the neck, so their scale parameters come out biased and their
  widths too wide; more resolution narrows the error only slowly (the sweep
  above), and on `hierarchy` it costs an hour (dynesty misses the width by
  0.11 after 71 min, pocomc is 0.39 off after 57 min).  NUTS, adapting a
  step size to the neck, is correct on `modes` and `hierarchy` in about a
  minute.  On `gp` it is not: 31 of its transitions diverge in the neck and
  its widths come out 0.23 too narrow, while its means agree.  emcee's
  means and widths on `gp` are within the tolerance, but its 3000 steps are
  35 τ; a longer chain is the cheap fix.
- **On the optical models the prior-started samplers win.**  The `omp`
  posterior is correlated and has local modes, but no funnel: pocomc is
  correct at a quarter of NUTS's wall time and eleven times its ESS per
  thousand calls, and it gives log Z.  That needed twice pocomc's default
  resolution; at its defaults it, and dynesty at 200 live points, were 0.26
  sd off.
- **The ensembles do not converge in these budgets** outside `linear`.  emcee
  is short of 50 τ everywhere else, and on `omp` and `hierarchy` it is
  10 and 17 sd off; ptemcee's adaptive ladder passes R-hat on `omp` but is
  0.73 off, with τ = 126 steps against 2000 kept.
- **Evidence.**  Against the exact log Z, dynesty is within its error on
  `linear` and `gp` (80.3 ± 0.28 against 80.48) but 4 errors high on
  `hierarchy` (76.2 ± 0.55 against 73.97); pocomc is within its error on
  `linear` and `hierarchy` but 7 errors low on `gp` (80.2 ± 0.04), where its
  posterior is biased too.  Like Kish's ESS, pocomc's log Z error leaves out
  the correlation between its particles.
- **PyMC against blackjax.**  The same algorithms on the same density:
  PyMC's NUTS is correct on `omp` at 1.3 times blackjax's wall time, since
  its call costs 12 ms against 8 (the forward pass repeated for the
  gradient Op, and PyTensor's overhead).  Both HMCs use a fixed path, and
  PyMC's (length 2) misses the tolerance where blackjax's (20 steps) does
  not.
- **Cost per call.**  A reaction-model call costs 2–4 ms whatever the
  sampler, since the solver dominates; a gradient call 6–8 ms.  On the
  cheap densities the batched samplers (emcee, pocomc) pay a few hundredths
  of a millisecond per point, and dynesty, which cannot batch, several times
  more.  pocomc spends 60–90% of its calls tempering from the prior to the
  posterior, and its per-call time includes training its flow.
