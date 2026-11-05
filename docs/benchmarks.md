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
| `omp` | 9 | exactly the log-space p + ⁴⁰Ca problem of `local_optical_model_calibration`: EXFOR C0076002 (Gross et al. 1967) at 35.8 MeV, seven potential parameters and two median-one log-space errors; its posterior has two separated islands |
| `alpha_gp` | 9 | exactly the power-of-$q$ problem of `alpha_ca_error_model_comparison`: ⁴⁴Ca(α,α) at 29 MeV, a four-parameter potential, two log-space errors and a Matérn GP whose amplitude grows with $q$; two families of potentials and GP hyperparameters at once |
| `n40ca_gp` | 7 | exactly the "GP, growing amplitude" rung of `gp_discrepancy`'s cross-section ladder: mock n + ⁴⁰Ca data fitted by a volume-only potential with a GP discrepancy |

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
# everything (a day or more: the optical-model problems are meant for a cluster, see below)
python benchmarks/bench_samplers.py --out benchmarks/results/samplers.json
# some problems
python benchmarks/bench_samplers.py --out benchmarks/results/samplers.json --problems linear gp
# re-run some samplers into the saved file, keeping the others' results
python benchmarks/bench_samplers.py --out benchmarks/results/samplers.json --problems omp --only pocomc NUTS
# print a saved run, or the Markdown table below
python benchmarks/bench_samplers.py --report benchmarks/results/samplers.json --markdown
# recompute the reference posteriors (many hours for the three optical-model problems)
python benchmarks/bench_samplers.py --reference
```

emcee, dynesty, pocomc and NUTS run on every problem; `omp`, `alpha_gp`
and `n40ca_gp` also run ptemcee, HMC and PyMC's NUTS and HMC.  The results
are written after each problem, so an interrupted run keeps what it
finished.

The three optical-model problems, their references and the pocomc
settings sweeps (`benchmarks/sweep_pocomc.py`) take many hours; they are
meant to run as independent cluster jobs, each writing a file of its own
that `benchmarks/merge_results.py` merges afterwards.
`benchmarks/hpc/README.md` is the runbook.

### The samplers and their starts

Each sampler starts the way it is usually used, and pays for the search
that finds its start: the search's wall time and evaluations are added to
its own.

| sampler | start | budget on `omp` |
|---|---|---|
| [emcee](https://emcee.readthedocs.io/) | a ball about the MAP (`rx.optimize.find_map`, L-BFGS-B) | 32 walkers × 3000 steps |
| [ptemcee](https://github.com/NuclearTalent/ptemcee_2025) | the same ball, at every temperature | 32 walkers × 8 temperatures × 2000 steps, adaptive ladder |
| [dynesty](https://dynesty.readthedocs.io/) | the prior | rwalk, 500 live points, to Δlog Z = 0.5 |
| [pocomc](https://pocomc.readthedocs.io/) | the prior (the unit cube, with `log_likelihood(prior_transform(u))`) | `n_effective` = 4096, `n_active` = 2048 (eight times its defaults; 1024 and 512 on the other problems) |
| NUTS ([blackjax](https://blackjax-devs.github.io/blackjax/)) | the same ball as emcee | 2 chains × (300 warm-up + 1000 draws) |
| HMC (blackjax) | the same | as NUTS, 20 leapfrog steps |
| NUTS, HMC ([PyMC](https://www.pymc.io/)) | the same | as blackjax; HMC with path length 2 |

A ball is `to_theta(to_unconstrained(start) + 0.01 N(0, I))`, inside the
support.  The gradient samplers run on `log_posterior_unconstrained`
(recipe 41); PyMC calls it through a PyTensor Op whose gradient is
`jax.grad`.  Every chain sampler starts about the MAP rather than the
maximum of the likelihood, because on `hierarchy` the likelihood depends on
`mu + tau * eta_j` alone, so its maximum is a ridge, not a point.  On
`hierarchy` emcee and NUTS start from prior draws instead, because its MAP
is far from the bulk of the posterior: the non-centred mode shrinks the
`eta_j` toward their prior and inflates τ to compensate, putting τ at 1.27
against a posterior of 0.31 ± 0.04.  The other problems' budgets are in
`BUDGETS` at the top of the script.

dynesty's and pocomc's settings are the cheapest of a sweep that are
correct (below) over three seeds: on `elastic` for every problem but `omp`,
and on `omp` for pocomc there.  Bias / width, the worst over the seeds run:

| setting | `gp` | `modes` | `elastic` |
|---|---|---|---|
| dynesty rwalk, 200 live points (default) | 0.54 / 0.45 | | 0.02 / 0.04 |
| dynesty rwalk, 500 | 0.21 / 0.11 | 0.13 / 0.22 | 0.04 / 0.02 |
| dynesty rslice, 250 | 0.08 / 0.09 | 0.22 / 0.19 | 0.02 / 0.03 |
| dynesty rslice, 500 | 0.12 / 0.07 | 0.13 / 0.08 | 0.05 / 0.03 |
| pocomc 512/256 (default) | 0.63 / 0.51 | | 0.02 / 0.02 |
| pocomc 1024/512 | 0.61 / 0.42 | 0.59 / 0.30 | 0.02 / 0.03 |
| pocomc 2048/1024 | 0.35 / 0.13 | 0.16 / 0.08 | 0.02 / 0.01 |

No setting tried is correct on `gp` and `modes`, whose posteriors are
funnels (see the discussion); more resolution helps there, but slowly.

`omp` needs much more.  About 85 % of its posterior sits in a narrow
island about the MAP and the rest in a broader one, and a run that
under-samples the narrow island puts too much weight on the other: at
1024/512 pocomc put 37 to 83 % of its draws away from the MAP, and its
log Z was low by up to 1.8 as a result.  Each seed below is a separate run;
the rows are scored against the reference (next section):

| setting | seeds | bias / width, each seed | wall time |
|---|---|---|---|
| dynesty rwalk, 500 | 1 | 0.31 / 0.17 | 563–750 s |
| dynesty rwalk, 1500 | 2 | 0.37 / 0.27, 0.12 / 0.15 | 1682–2358 s |
| pocomc 1024/512 | 3 | 1.90 / 0.43, 1.84 / 0.49, 0.78 / 0.42 | 287–425 s |
| pocomc 2048/1024 | 3 | 0.11 / 0.15, 0.20 / 0.17, 0.94 / 0.44 | 432–475 s |
| pocomc 4096/2048 | 4 | 0.04 / 0.03, 0.06 / 0.06, 0.06 / 0.07, 0.04 / 0.05 | 741–1064 s |

4096/2048 is the cheapest setting correct on every seed scored, but it is
not infallible: a first attempt at the reference pooled three runs at that
setting, and one was 0.19 from their pool.  A user should run it more than
once and compare the runs.  dynesty's random walks are not reliable here at
any setting tried; its evidence is (−6.8 ± 0.29 at 500 live points,
against −6.63).

### How they are scored

- **Calls.**  A call is one evaluation of the log density and, for the
  gradient samplers and the MAP search, its gradient.  Each run's calls are
  split by phase:

  | phase | what it is |
  |---|---|
  | search | the MAP search that found the start (each L-BFGS-B step is one value-and-gradient call) |
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
- **ESS.**  The smallest rank-normalised bulk effective sample size over
  the parameters ([Vehtari et al. 2021](https://arxiv.org/abs/1903.08008),
  `blackjax.diagnostics.ess_bulk`) across chains (the walkers, for the
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
- **Convergence.**  A run has not converged (✗, and no ESS) if its
  rank-normalised split [R-hat](https://arxiv.org/abs/1903.08008)
  (`blackjax.diagnostics.rhat`) exceeds
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
| `modes`, `elastic` | moments | one long blackjax NUTS run, 4 chains × (1000 warm-up + 3000 draws) at target acceptance 0.95, from a ball about the MAP with a seed of its own, accepted only with a rank-normalised R-hat ≤ 1.01 and no divergences (R-hat 1.001, 1.001; min bulk ESS 2475, 3920).  Chains started from prior draws can stick in a minor mode: on `modes`, one with a log posterior 40 below the main one. |
| `omp`, `alpha_gp`, `n40ca_gp` | moments and log Z | three pocomc runs at 8192/4096, sixteen times its defaults, each with a seed of its own, pooled; accepted only if every run is within 0.05 of the pool; then checked against three dynesty runs (rslice, 1000 live points; nine on `alpha_gp`, whose single runs put 0.55 to 0.91 of their mass in the deeper family), pooled the same way, on log Z (within twice the combined error), the mode share (within 0.03) and the moments (within 0.2 sd).  A NUTS reference cannot work on `omp`: its chains stay in the island they start in.  All three are accepted (their pocomc runs agree to 0.018, 0.012 and 0.042); `omp` and `n40ca_gp` are corroborated, `alpha_gp` is not (below). |

The benchmarked NUTS row on `modes` and `elastic`, and the pocomc row on
`omp`, are scored against runs that share their algorithm, not their draws.
On `gp` and `hierarchy`, where the reference is exact, NUTS's means agree
to 0.08.  On the three optical-model problems the reference is checked by dynesty, an
independent algorithm:

| problem | log Z, pocomc / dynesty | moments, bias / width | mode share, pocomc / dynesty | corroborated |
|---|---|---|---|---|
| `omp` | −6.65 ± 0.04 / −6.40 ± 0.25 | 0.04 / 0.07 | 0.131 / 0.121 away from the MAP island | yes |
| `alpha_gp` | −84.53 ± 0.01 / −84.58 ± 0.59 | 0.16 / 0.03 | 0.740 / 0.787 in the deeper family | **no**: the share, by 0.047 |
| `n40ca_gp` | −11.34 ± 0.01 / −11.84 ± 0.28 | 0.15 / 0.09 | (one mode) | yes |

`alpha_gp`'s reference is kept, marked `corroborated: false` (the tables say
so), because the disagreement is dynesty's noise as far as can be told.
Single dynesty runs put 0.55 to 0.91 of the mass in the deeper family, so
even nine of them pooled carry a standard error of 0.035 on the share, and
the gap is 1.3 of those; with three runs the gap was 0.056 the other way
(0.684).  The pocomc runs agree on 0.739 to 0.743, and the sweep's 4096/2048
runs on 0.71 to 0.74.  log Z and the moments, which are what the bias and
width are scored on, agree within their bars.  What it does not rule out is
a bias pocomc shares across its own runs; the 0.03 bar would take about
fifty dynesty runs at these settings to test.

### Results

> The `omp`, `alpha_gp` and `n40ca_gp` rows ran on a cluster (8 cores of an
> AMD EPYC "amd20" node per sampler, `benchmarks/hpc/`), the others in an
> earlier run on one local machine: compare wall times within a problem, not
> across these two groups; calls and ESS per thousand calls do not depend on
> the machine.
> The discussion below still quotes the earlier `omp` parametrisation's
> numbers where it names `omp`, and the `omp` sweep above is of that version.

| problem | sampler | wall time | calls | search | warm-up | sampling | evidence | ms/call | bias | width | min ESS | ESS/s | ESS/1k calls | R-hat | τ (steps) | log Z |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| linear (2) | emcee | 6 s | 384075 | 0% | 33% | 67% | 0% | 0.01 | 0.00 | 0.00 | 7619 | 1324.66 | 19.84 | 1.006 | 33 |  |
| linear (2) | dynesty | 34 s | 93609 | 0% | 0% | 100% | 0% | 0.37 | 0.02 | 0.03 | 1605 | 46.75 | 17.15 | — |  | 0.7 ± 0.24 |
| linear (2) | pocomc | 27 s | 53760 | 0% | 70% | 23% | 8% | 0.50 | 0.02 | 0.01 | 4292 | 159.70 | 79.84 | — |  | 0.6 ± 0.06 |
| linear (2) | NUTS | 6 s | 62145 | 0% | 49% | 51% | 0% | 0.09 | 0.03 | 0.01 | 732 | 132.75 | 11.77 | 1.003 |  |  |
| linear (2) | reference: quadrature |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0.64 (exact) |
| gp (4) | emcee | 18 s | 96329 | 0% | 33% | 66% | 0% | 0.19 | 0.07 | 0.30 ✗ | — | — | — | 1.062 ✗ | 82 (< 50 τ) |  |
| gp (4) | dynesty | 71 s | 139598 | 0% | 0% | 100% | 0% | 0.51 | 0.20 ✗ | 0.11 ✗ | — | — | — | — |  | 80.3 ± 0.28 |
| gp (4) | pocomc | 73 s | 99840 | 0% | 71% | 25% | 4% | 0.73 | 0.60 ✗ | 0.42 ✗ | — | — | — | — |  | 80.2 ± 0.04 |
| gp (4) | NUTS | 211 s | 245453 | 0% | 68% | 32% | 0% | 0.86 | 0.08 | 0.23 ✗ | — | — | — | 1.018 ✗ |  |  |
| gp (4) | reference: quadrature |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 80.48 (exact) |
| elastic (5) | emcee | 162 s | 48212 | 0% | 33% | 66% | 0% | 3.35 | 0.07 | 0.03 | — | — | — | 1.071 ✗ | 41 (< 50 τ) |  |
| elastic (5) | dynesty | 646 s | 232479 | 0% | 0% | 100% | 0% | 2.78 | 0.04 | 0.02 | 2399 | 3.71 | 10.32 | — |  | 495.9 ± 0.34 |
| elastic (5) | pocomc | 312 s | 84480 | 0% | 81% | 15% | 5% | 3.69 | 0.02 | 0.03 | 4501 | 14.42 | 53.28 | — |  | 496.0 ± 0.01 |
| elastic (5) | NUTS | 324 s | 48984 | 0% | 53% | 47% | 0% | 6.62 | 0.06 | 0.02 | 673 | 2.07 | 13.73 | 1.003 |  |  |
| elastic (5) | reference: NUTS, 4 x 3000 draws |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| hierarchy (33) | emcee | 11 s | 408000 | 0% | 33% | 67% | 0% | 0.03 | 1.43 ✗ | 0.58 ✗ | — | — | — | 1.504 ✗ | 518 (< 50 τ) |  |
| hierarchy (33) | dynesty | 4256 s | 1926706 | 0% | 0% | 100% | 0% | 2.21 | 0.05 | 0.11 ✗ | — | — | — | — |  | 76.2 ± 0.55 |
| hierarchy (33) | pocomc | 3410 s | 663040 | 0% | 92% | 7% | 1% | 5.14 | 0.39 ✗ | 0.21 ✗ | — | — | — | — |  | 73.9 ± 0.17 |
| hierarchy (33) | NUTS | 41 s | 406667 | 0% | 30% | 70% | 0% | 0.10 | 0.08 | 0.01 | 536 | 13.14 | 1.32 | 1.006 |  |  |
| hierarchy (33) | reference: quadrature |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 73.97 (exact) |
| modes (9) | emcee | 10 s | 96478 | 0% | 33% | 66% | 0% | 0.10 | 0.36 ✗ | 0.24 ✗ | — | — | — | 1.319 ✗ | 177 (< 50 τ) |  |
| modes (9) | dynesty | 92 s | 222314 | 0% | 0% | 100% | 0% | 0.41 | 0.13 ✗ | 0.22 ✗ | — | — | — | — |  | 103.2 ± 0.29 |
| modes (9) | pocomc | 130 s | 91136 | 0% | 62% | 33% | 4% | 1.43 | 0.59 ✗ | 0.29 ✗ | — | — | — | — |  | 102.7 ± 0.08 |
| modes (9) | NUTS | 61 s | 340473 | 0% | 69% | 31% | 0% | 0.18 | 0.05 | 0.04 | 851 | 13.88 | 2.50 | 1.008 |  |  |
| modes (9) | reference: NUTS, 4 x 3000 draws |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| alpha_gp (9) | emcee | 166 s | 97744 | 2% | 33% | 65% | 0% | 1.70 | 1.32 ✗ | 1.48 ✗ | — | — | — | 1.605 ✗ | 164 (< 50 τ) |  |
| alpha_gp (9) | dynesty | 584 s | 274264 | 0% | 0% | 100% | 0% | 2.13 | 0.30 ✗ | 0.26 ✗ | — | — | — | — |  | -84.6 ± 0.31 |
| alpha_gp (9) | pocomc | 1048 s | 442368 | 0% | 99% | 0% | 1% | 2.37 | 0.03 | 0.01 | 7625 | 7.27 | 17.24 | — |  | -84.5 ± 0.01 |
| alpha_gp (9) | NUTS | 780 s | 219832 | 1% | 49% | 50% | 0% | 3.55 | 0.60 ✗ | 1.43 ✗ | — | — | — | 1.008 |  |  |
| alpha_gp (9) | ptemcee | 859 s | 513744 | 0% | 33% | 66% | 0% | 1.67 | 1.39 ✗ | 7.85 ✗ | — | — | — | 1.447 ✗ | 121 (< 50 τ) | -89.4 ± 4.87 |
| alpha_gp (9) | HMC | 203 s | 53744 | 3% | 22% | 74% | 0% | 3.78 | 0.57 ✗ | 1.42 ✗ | — | — | — | 1.014 ✗ |  |  |
| alpha_gp (9) | PyMC NUTS | 3037 s | 140769 | 1% | 43% | 55% | 0% | 21.58 | 0.59 ✗ | 1.43 ✗ | — | — | — | 1.009 |  |  |
| alpha_gp (9) | PyMC HMC | 868 s | 148213 | 1% | 71% | 28% | 0% | 5.85 | 0.58 ✗ | 1.47 ✗ | — | — | — | 1.002 |  |  |
| alpha_gp (9) | reference: pocomc 8192/4096, 3 runs (NOT CORROBORATED) |  |  |  |  |  |  |  |  |  |  |  |  |  |  | -84.53 ± 0.01 |
| n40ca_gp (7) | emcee | 41 s | 96778 | 1% | 33% | 66% | 0% | 0.42 | 0.05 | 0.09 | — | — | — | 1.045 ✗ | 65 (< 50 τ) |  |
| n40ca_gp (7) | dynesty | 284 s | 278094 | 0% | 0% | 100% | 0% | 1.02 | 0.06 | 0.08 | 3843 | 13.55 | 13.82 | — |  | -11.9 ± 0.34 |
| n40ca_gp (7) | pocomc | 336 s | 280576 | 0% | 99% | 0% | 1% | 1.20 | 0.02 | 0.03 | 7180 | 21.37 | 25.59 | — |  | -11.4 ± 0.01 |
| n40ca_gp (7) | NUTS | 94 s | 95455 | 1% | 49% | 51% | 0% | 0.99 | 0.10 ✗ | 0.08 | — | — | — | 1.005 |  |  |
| n40ca_gp (7) | ptemcee | 225 s | 512778 | 0% | 33% | 67% | 0% | 0.44 | 0.14 ✗ | 0.15 ✗ | — | — | — | 1.008 | 9 | -17.4 ± 21.77 |
| n40ca_gp (7) | HMC | 56 s | 52778 | 1% | 23% | 76% | 0% | 1.06 | 0.08 | 0.07 | 722 | 12.89 | 13.67 | 1.003 |  |  |
| n40ca_gp (7) | PyMC NUTS | 139 s | 84674 | 1% | 44% | 55% | 0% | 1.64 | 0.10 | 0.07 | 626 | 4.49 | 7.39 | 1.004 |  |  |
| n40ca_gp (7) | PyMC HMC | 106 s | 61151 | 1% | 64% | 35% | 0% | 1.74 | 0.05 | 0.10 | 655 | 6.16 | 10.71 | 1.003 |  |  |
| n40ca_gp (7) | reference: pocomc 8192/4096, 3 runs |  |  |  |  |  |  |  |  |  |  |  |  |  |  | -11.34 ± 0.01 |
| omp (9) | emcee | 183 s | 96876 | 1% | 33% | 66% | 0% | 1.89 | 1.26 ✗ | 1.29 ✗ | — | — | — | 1.593 ✗ | 170 (< 50 τ) |  |
| omp (9) | dynesty | 565 s | 255013 | 0% | 0% | 100% | 0% | 2.21 | 0.20 ✗ | 0.12 ✗ | — | — | — | — |  | -6.4 ± 0.32 |
| omp (9) | pocomc | 807 s | 311296 | 0% | 99% | 0% | 1% | 2.59 | 0.07 | 0.07 | 6815 | 8.44 | 21.89 | — |  | -6.7 ± 0.01 |
| omp (9) | NUTS | 626 s | 161531 | 1% | 51% | 49% | 0% | 3.87 | 0.91 ✗ | 0.35 ✗ | — | — | — | 1.694 ✗ |  |  |
| omp (9) | ptemcee | 927 s | 512876 | 0% | 33% | 67% | 0% | 1.81 | 1.35 ✗ | 7.34 ✗ | — | — | — | 1.535 ✗ | 97 (< 50 τ) | -12.0 ± 2.34 |
| omp (9) | HMC | 214 s | 52876 | 2% | 23% | 76% | 0% | 4.04 | 0.31 ✗ | 0.75 ✗ | — | — | — | 1.005 |  |  |
| omp (9) | PyMC NUTS | 613 s | 106554 | 1% | 45% | 54% | 0% | 5.76 | 0.32 ✗ | 0.68 ✗ | — | — | — | 1.006 |  |  |
| omp (9) | PyMC HMC | 727 s | 130313 | 1% | 80% | 19% | 0% | 5.58 | 0.30 ✗ | 0.79 ✗ | — | — | — | 1.007 |  |  |
| omp (9) | reference: pocomc 8192/4096, 3 runs |  |  |  |  |  |  |  |  |  |  |  |  |  |  | -6.65 ± 0.04 |

Which sampler is correct, and fastest among those that are, on each problem:

| problem | correct | fastest correct |
|---|---|---|
| `linear` | all four | NUTS, 6 s (emcee 6 s, at 12 000 steps, with ten times the ESS per second) |
| `gp` | none at these budgets | emcee's means are closest (0.07), but its widths are 0.30 off and its kept chain is 25 τ, not 50 |
| `modes` | NUTS | NUTS, 61 s |
| `elastic` | dynesty, pocomc, NUTS | pocomc, 312 s (NUTS 324 s, dynesty 646 s) |
| `hierarchy` | NUTS | NUTS, 41 s, at 3000 draws per chain |
| `omp` | pocomc | pocomc, 807 s, at 4096/2048 |
| `alpha_gp` | pocomc (against a reference not corroborated on its mode share) | pocomc, 1048 s, at 4096/2048 |
| `n40ca_gp` | dynesty, pocomc, HMC, PyMC NUTS and HMC | HMC, 56 s (PyMC HMC 106 s, PyMC NUTS 139 s, dynesty 284 s, pocomc 336 s) |

- **Funnels defeat the samplers that start from the prior.**  `gp`, `modes`
  and `hierarchy` each have a scale parameter that, when small, pins the
  others into a narrow neck: the GP's log variance against its bound, the
  shared normalisation mode's width, the hierarchy's τ.  dynesty and pocomc
  under-visit the neck, and come out wrong in ways that differ by problem:
  on `gp` both bias the log variance (0.20 and 0.60) and widen the
  posterior; on `modes` pocomc biases the shared width `log_s` too (0.59)
  and widens `m` (0.29), while dynesty is 0.13 off in `b` and `log_s` alike
  and too narrow in `m` (0.22); on `hierarchy` both come out too narrow
  (0.11 and 0.21), and pocomc's worst biases fall on the school effects
  (0.39), τ close behind (0.37).  More
  resolution narrows the error only slowly (the sweep above), and on
  `hierarchy` it costs an hour (dynesty misses the width by 0.11 after
  71 min, pocomc is 0.39 off after 57 min).  NUTS, adapting a
  step size to the neck, is correct on `modes` and `hierarchy` in about a
  minute (on `hierarchy` once its chains are 3000 draws long: at 1000 the
  rank-normalised R-hat was 1.021).  On `gp` it is not: 31 of its
  transitions diverge in the neck, its widths come out 0.23 too narrow and
  its R-hat is 1.018, while its means agree.  emcee on `gp`, started at the
  MAP, has its means within the tolerance but its widths 0.30 off, and its
  kept chain is 25 τ.
- **Separated modes defeat the samplers that start at the MAP, and R-hat
  seldom tells.**  The `omp` posterior has two islands, and the gradient
  samplers, started from a ball about the MAP, stay in the one they start
  in: blackjax's NUTS and HMC and PyMC's NUTS are 0.28 to 0.30 sd off in
  their means and 0.69 to 0.77 in their widths, while their rank-normalised
  R-hats are 1.004 to 1.006.  Chains that never visit an island agree with
  one another about it.  PyMC's HMC is the exception: its R-hat of 1.152
  says its chains disagree (its min ESS is 9), and its errors shrink (its bias to 0.14, about half the others', its width error to
  0.18, about a quarter) but do not vanish.  Only pocomc, at eight times its default resolution, is correct; at
  lower resolutions it weighs the islands differently from seed to seed
  (the sweep above).  On `elastic`, which has one mode, NUTS, pocomc and
  dynesty are all correct.
- **The ensembles do not converge in these budgets** outside `linear`, where
  emcee needs 12 000 steps for the rank-normalised R-hat to pass (1.019 at
  3000, 1.010 at 6000).  emcee is short of 50 τ everywhere else, and on
  `omp` and `hierarchy` it is 1.0 and 1.4 sd off, with R-hats of 1.68 and
  1.50; ptemcee on `omp`, started at the MAP, is 12 τ long with an R-hat of
  1.56, and is 1.3 sd off, with a log Z of −11.0 ± 2.5 against −6.63.
- **Evidence.**  Against the exact log Z, dynesty is within its error on
  `linear` and `gp` (80.3 ± 0.28 against 80.48) but 4 errors high on
  `hierarchy` (76.2 ± 0.55 against 73.97); pocomc is within its error on
  `hierarchy`, 1.2 errors low on `linear` and 6 errors low on `gp` (80.2 ± 0.04), where its
  posterior is biased too.  On `omp`, against the reference's −6.63 ± 0.01,
  pocomc and dynesty are within their errors (−6.6 ± 0.01, −6.8 ± 0.29),
  dynesty's evidence right even where its posterior is not; ptemcee, which
  has not converged, is 1.7 errors low.  Like Kish's ESS, pocomc's log Z error leaves out the
  correlation between its particles.
- **PyMC against blackjax.**  The same algorithms on the same density, and
  NUTS gives the same answer on `omp`, wrong in the same way: PyMC's NUTS
  takes 1.2 times blackjax's wall time, since its call costs 13 ms against 9
  (the forward pass repeated for the gradient Op, and PyTensor's overhead).
- **Cost per call.**  A reaction-model call costs 1.8–3.7 ms whatever the
  sampler, since the solver dominates; a gradient call 6.6 to 8.9 ms in
  blackjax.  On the cheap densities emcee, which evaluates its walkers in
  one batch, pays a few hundredths of a millisecond per point (0.015 to
  0.19 ms); dynesty, which cannot batch, pays 3 to 84 times more, and
  pocomc more still (0.5 to 5 ms), because its per-call time includes
  training its normalizing flow.  pocomc spends 62–99 % of its calls
  tempering from the prior to the posterior.
