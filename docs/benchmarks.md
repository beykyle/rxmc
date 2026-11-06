# Benchmarks

Two benchmarks live in `benchmarks/`: one times the densities, the other
compares samplers on them.  Both build their problems from
`benchmarks/problems.py` and write JSON to `benchmarks/results/`, which is
checked in, so the tables below (all but the dynesty and pocomc settings
sweep) can be regenerated without a rerun.

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
| `omp` | 9 | exactly the log-space p + $^{40}$Ca problem of `local_optical_model_calibration`: EXFOR C0076002 (Gross et al. 1967) at 35.8 MeV, seven potential parameters and two median-one log-space errors; its posterior has two separated islands |
| `alpha_gp` | 9 | exactly the power-of-$q$ problem of `alpha_ca_error_model_comparison`: $^{44}$Ca($\alpha$,$\alpha$) at 29 MeV, a four-parameter potential, two log-space errors and a Matérn GP whose amplitude grows with $q$; two families of potentials and GP hyperparameters at once |
| `n40ca_gp` | 7 | exactly the "GP, growing amplitude" rung of `gp_discrepancy`'s cross-section ladder: mock n + $^{40}$Ca data fitted by a volume-only potential with a GP discrepancy |

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
`numpy-jitr3.json` is the numpy rxmc 0.x (tag `v0.1.0`) on jitr 3.0.1,
whose gradient is a central finite difference of `2 ndim` calls; to
reproduce it, install that rxmc and jitr in a separate environment and run
this script there.

Against that baseline (same machine, CPU):

| problem | `log_posterior` | gradient (FD baseline to autodiff) | vmap, 256 rows | emcee $32 \times 500$ |
|---|---|---|---|---|
| line, reported errors | 0.16 to 0.10 ms | 0.66 to 0.11 ms | 6.2k to 1.06M evaluations/s | 2.9 to 1.9 s |
| 6 comparisons, 13 modes | 1.38 to 0.20 ms | 30.4 to 0.39 ms | 0.5k to 41k evaluations/s | |
| GP, N = 100 | 0.79 to 0.27 ms | 6.4 to 1.2 ms | 1.3k to 4.4k evaluations/s | 10.3 to 4.6 s |
| GP, N = 500 | 10.8 to 5.3 ms | 90 to 32 ms | 87 to 147 evaluations/s | |
| elastic, 4 energies | 7.0 to 2.6 ms | 70 to 6.0 ms | 142 to 331 evaluations/s | 106 to 45 s |
| (p,n) IAS, 1 energy | 5.2 to 2.1 ms | 43 to 4.1 ms | 191 to 450 evaluations/s | |

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
| [emcee](https://emcee.readthedocs.io/) | a ball about the MAP (`rx.optimize.find_map`, L-BFGS-B) | 32 walkers $\times$ 3000 steps |
| [ptemcee](https://github.com/NuclearTalent/ptemcee_2025) | the same ball, at every temperature | 32 walkers $\times$ 8 temperatures $\times$ 2000 steps, adaptive ladder |
| [dynesty](https://dynesty.readthedocs.io/) | the prior | rwalk, 500 live points, to $\Delta\log Z$ = 0.5 |
| [pocomc](https://pocomc.readthedocs.io/) | the prior (the unit cube, with `log_likelihood(prior_transform(u))`) | `n_effective` = 4096, `n_active` = 2048 (eight times its defaults; 1024 and 512 on the other problems) |
| NUTS ([blackjax](https://blackjax-devs.github.io/blackjax/)) | the same ball as emcee | 2 chains $\times$ (300 warm-up + 1000 draws) |
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
`eta_j` toward their prior and inflates $\tau$ to compensate, putting $\tau$ at 1.27
against a posterior of $0.31 \pm 0.04$.  The other problems' budgets are in
`BUDGETS` at the top of the script.

dynesty runs rwalk with 500 live points, and pocomc 1024/512, on every
problem but the three optical-model ones, where pocomc runs at 4096/2048
(below).  Every setting of this sweep, the defaults included, is correct on
`elastic`; none is correct on the funnels, `gp` and `modes`, on more than
one seed of three.  Bias / width, the worst over seeds 100, 101 and 102
(`sweep_pocomc.py --grid` on `benchmarks/results/hpc/sweep_{gp,modes,elastic}.json`,
run on the cluster):

| setting | `gp` | `modes` | `elastic` |
|---|---|---|---|
| dynesty rwalk, 200 live points (default) | 0.26 / 0.41 ✗ | 0.70 / 0.45 ✗ | 0.04 / 0.07 |
| dynesty rwalk, 500 | 0.42 / 0.38 ✗ | 0.22 / 0.17 ✗ | 0.04 / 0.02 |
| dynesty rslice, 250 | 0.35 / 0.39 ✗ | 0.30 / 0.34 ✗ | 0.06 / 0.04 |
| dynesty rslice, 500 | 0.36 / 0.29 ✗ | 0.15 / 0.14 ✗ | 0.06 / 0.04 |
| pocomc 512/256 (default) | 0.69 / 0.59 ✗ | 0.53 / 0.30 ✗ | 0.03 / 0.02 |
| pocomc 1024/512 | 0.81 / 0.51 ✗ | 0.42 / 0.25 ✗ | 0.02 / 0.02 |
| pocomc 2048/1024 | 0.53 / 0.58 ✗ | 0.17 / 0.11 ✗ | 0.03 / 0.02 |

On the funnels a run's error scatters from seed to seed as much as from
setting to setting: dynesty rslice with 250 live points is 0.35 / 0.39,
0.19 / 0.16 and 0.06 / 0.29 on `gp`, so one run says little about a
setting.  (An earlier sweep on one local machine, whose files were not
kept, found that setting correct on `gp`; these seeds do not reproduce it.)
More resolution helps on `modes`, where pocomc 2048/1024 is correct on
one seed (0.09 / 0.06) and dynesty rslice's worst halves from 250 to 500
live points, but not visibly on `gp` (see the discussion).

The three optical-model problems need much more.  `omp` has two islands,
about 87 % of its posterior in a narrow one about the MAP and the rest in a
broader one; `alpha_gp` has two families of potentials, about 74 % of its
posterior in the deeper one; `n40ca_gp` has one mode.  A run that
under-samples a narrow island puts too much weight on the other: at
1024/512 pocomc put 23 to 85 % of its draws away from `omp`'s MAP island,
and its log Z was low by up to 2.2 as a result.  Each seed below (100, 101,
102) is a separate run, scored against the reference (next section; ? marks
an error the reference cannot resolve); the share is of `omp`'s minor
island (0.131 in the reference) and of `alpha_gp`'s deeper family (0.740):

| problem | pocomc | bias / width, each seed | share | log Z | wall time |
|---|---|---|---|---|---|
| `omp` | 1024/512 | 1.99 / 0.49 ✗, 0.34 / 0.20 ✗, 1.93 / 0.31 ✗ | 0.82, 0.23, 0.85 | -8.86, -6.67, -8.81 | 231–348 s |
| `omp` | 2048/1024 | 0.19 / 0.06 ✗, 0.25 / 0.19 ✗, 0.13 / 0.03 ? | 0.17, 0.20, 0.15 | -6.62, -6.69, -6.68 | 406–477 s |
| `omp` | 4096/2048 | 0.17 / 0.12 ✗, 0.07 / 0.05 ?, 0.04 / 0.05 ? | 0.16, 0.15, 0.14 | -6.66, -6.64, -6.69 | 824–889 s |
| `alpha_gp` | 1024/512 | 0.47 / 0.81 ✗, 0.11 / 0.05 ?, 0.30 / 0.07 ✗ | 0.96, 0.76, 0.66 | -84.75, -84.55, -84.54 | 376–431 s |
| `alpha_gp` | 2048/1024 | 0.14 / 0.04 ?, 0.10 / 0.07 ?, 0.10 / 0.05 ? | 0.72, 0.79, 0.75 | -84.53, -84.13, -84.57 | 528–632 s |
| `alpha_gp` | 4096/2048 | 0.04 / 0.01, 0.02 / 0.01, 0.07 / 0.02 ? | 0.73, 0.74, 0.71 | -84.50, -84.52, -84.55 | 928–1089 s |
| `n40ca_gp` | 1024/512 | 0.29 / 0.05 ✗, 0.40 / 0.13 ✗, 0.16 / 0.14 ? |  | -11.33, -11.40, -11.32 | 129–150 s |
| `n40ca_gp` | 2048/1024 | 0.05 / 0.09 ?, 0.31 / 0.12 ✗, 0.12 / 0.06 ? |  | -11.35, -11.40, -11.36 | 190–206 s |
| `n40ca_gp` | 4096/2048 | 0.10 / 0.05 ?, 0.03 / 0.08 ?, 0.09 / 0.10 ? |  | -11.36, -11.35, -11.34 | 305–338 s |

(Wall times on 8 cores of a cluster node; references -6.65, -84.53,
-11.34.)  4096/2048, the benchmark's setting on all three, is wrong on one
seed of nine, `omp`'s seed 100 (0.17 / 0.12), so a user should run it more
than once and compare the runs.  Every lower setting is wrong on at least
one seed of each problem except 2048/1024 on `alpha_gp`, which its
reference cannot tell from 4096/2048; the mode share separates them a
little (0.72 to 0.79 against 0.71 to 0.74, the reference 0.740).  Below
4096/2048 the evidence holds up better than the moments: at 2048/1024
log Z is within 0.06 of the reference on `omp` and `n40ca_gp` while the
moments are up to 0.31 sd off.

dynesty is wrong on the two problems with separated modes at the
benchmark's setting (0.20 / 0.12 on `omp`, 0.30 / 0.26 on `alpha_gp`), and
even the references' check, rslice with 1000 live points, scatters from
run to run (next section).  On `n40ca_gp`, with one mode, it is within the
reference's resolution.

### How they are scored

- **Calls.**  A call is one evaluation of the log density and, for the
  gradient samplers and the MAP search, its gradient.  Each run's calls are
  split by phase:

  | phase | what it is |
  |---|---|
  | search | the MAP search that found the start (each L-BFGS-B step is one value-and-gradient call) |
  | warm-up | emcee's and ptemcee's discarded burn-in (every rung of ptemcee's ladder), NUTS's and HMC's step-size and mass-matrix adaptation, pocomc's tempering from the prior until $\beta$ reaches 1 |
  | sampling | the calls that produced the kept draws: post-burn-in steps, post-warm-up trajectories, dynesty's whole run, pocomc's iterations at $\beta$ = 1 |
  | evidence | pocomc's importance sampling of its normalizing flow for log Z |

- **Time per call.**  Wall time over calls: what a call costs on this
  machine, sampler overhead included.  emcee, ptemcee and pocomc hand over
  many points at once and evaluate them in one `jax.vmap` call
  (`log_posterior_batch`; `log_prior_batch` and `log_likelihood_batch`
  through ptemcee's `mapper`; `log_likelihood_batch` on
  `prior_transform_batch` for pocomc), which removes the per-point Python
  overhead.  On a reaction model the solver dominates and batching buys
  nothing: on this CPU a vmapped `elastic` row costs about 3.0 ms against
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
  its bias is the largest $|\mu - \mu_\mathrm{ref}|$ over the parameters,
  and its width error the largest $|\log(\sigma / \sigma_\mathrm{ref})|$,
  both in reference standard deviations.  The tolerance is `TOLERANCE` = 0.1.
  A reference checked against an independent algorithm (next section) has a
  resolution $r$, the larger of the bias and width gaps between the two: an
  error more than $r$ above the tolerance is marked ✗, and one within $r$ of
  it, on either side, ? (the reference cannot tell).  $r$ is 0 for the
  exact and NUTS references, 0.07 on `omp`, 0.05 on `alpha_gp` and 0.12 on
  `n40ca_gp`.
- **The ESS of weighted draws is optimistic.**  For dynesty and pocomc it is
  [Kish's](https://en.wikipedia.org/wiki/Effective_sample_size) ESS of the
  importance weights.  That ignores the correlation between particles that
  share an ancestor (pocomc's resampling, dynesty's random walks from live
  points), so it overstates their ESS, and pocomc stops when Kish's ESS
  reaches its `n_total` target, so its ESS sits near that target by
  construction.  The bias and width columns are the check that matters:
  pocomc on `gp` reports an ESS of about 4500 with means 0.6 sd off.
- **Correct.**  A run is correct if it converged (next) and neither error
  is marked ✗; ESS is reported only for correct runs.  A ? means correct as
  far as the reference can tell.
- **Convergence.**  A run has not converged (✗, and no ESS) if its
  rank-normalised split [R-hat](https://arxiv.org/abs/1903.08008)
  (`blackjax.diagnostics.rhat`) exceeds
  1.01 or, for an ensemble, its kept chain is shorter than 50 times emcee's
  integrated autocorrelation time $\tau$.  The second rule matters: walkers that
  swap or stretch toward one another can agree, and pass R-hat, long before
  they have explored the posterior.
- **Evidence.**  log Z for dynesty, pocomc and ptemcee (thermodynamic
  integration over its ladder), against the exact value where there is one.

### The references

| problem | reference | how |
|---|---|---|
| `linear`, `gp`, `hierarchy` | exact moments and log Z | the log posterior is quadratic in all but a few parameters (none; the GP's log variance and log length scale; the hierarchy's $\tau$).  At each node of a grid over those, one Newton step over the rest lands on the conditional mode, and the Hessian there is the conditional precision, both exact for a quadratic; the conditional normalisers weight the nodes. |
| `modes`, `elastic` | moments | one long blackjax NUTS run, 4 chains $\times$ (1000 warm-up + 3000 draws) at target acceptance 0.95, from a ball about the MAP with a seed of its own, accepted only with a rank-normalised R-hat $\le$ 1.01 and no divergences (R-hat 1.001, 1.001; min bulk ESS 2475, 3920).  Chains started from prior draws can stick in a minor mode: on `modes`, one with a log posterior 40 below the main one. |
| `omp`, `alpha_gp`, `n40ca_gp` | moments and log Z | three pocomc runs at 8192/4096, sixteen times its defaults, each with a seed of its own, pooled; accepted only if every run is within 0.05 of the pool; then checked against dynesty runs (rslice, 1000 live points; three on `omp`, fifty on `alpha_gp` and `n40ca_gp`), pooled the same way (log Z's error is half the runs' range or the largest quoted error, whichever is larger), on log Z (within twice the combined error), the mode share (within 0.03) and the moments (within 0.2 sd).  A NUTS reference cannot work on `omp`: its chains stay in the island they start in.  All three are accepted (their pocomc runs agree to 0.018, 0.012 and 0.042); all three are corroborated (below). |

The benchmarked NUTS rows on `modes` and `elastic`, and the pocomc rows on
the three optical-model problems, are scored against runs that share their
algorithm, not their draws.  On `gp` and `hierarchy`, where the reference is
exact, NUTS's means agree to 0.08.  On the optical-model problems dynesty,
an independent algorithm, checks the reference, and the larger of the two
moment gaps is the reference's resolution $r$:

| problem | log Z, pocomc / dynesty | moments, bias / width | mode share, pocomc / dynesty | corroborated |
|---|---|---|---|---|
| `omp` | -6.65 ± 0.04 / -6.40 ± 0.25 | 0.04 / 0.07 | 0.131 / 0.121 away from the MAP island | yes |
| `alpha_gp` | -84.53 ± 0.01 / -84.50 ± 0.82 | 0.05 / 0.02 | 0.740 / 0.756 in the deeper family | yes |
| `n40ca_gp` | -11.34 ± 0.01 / -11.53 ± 0.87 | 0.07 / 0.12 | (one mode) | yes |

`alpha_gp` needed fifty dynesty runs.  Single runs put 0.46 to 0.91 of the
mass in the deeper family, so nine pooled (0.787) carried a standard error
of 0.035 on the share and could not test the 0.03 bar; fifty carry 0.015,
and their 0.756 is 0.016 from pocomc's 0.740 (whose three runs agree on
0.739 to 0.743, and the sweep's 4096/2048 runs on 0.71 to 0.74).  The
moment gap fell with it, from 0.16 at nine runs to 0.05: it was dynesty's
noise, not pocomc's bias.  `n40ca_gp` has fifty too, which move its gap
from 0.15 / 0.09 at three runs to 0.07 / 0.12.

The log Z check is looser than it looks at fifty runs: a pool's error is
half the runs' range, which grows with their number (0.82 and 0.87 above).
Against the standard error of the fifty runs' mean, dynesty agrees on
`alpha_gp` (-84.50 ± 0.05 against -84.53) but not on `n40ca_gp`, where it is
0.18 below pocomc (-11.53 ± 0.06 against -11.34), about three standard
errors; the benchmark's dynesty run is low there too.  The moments and the
share are unaffected.

### Results

> The `omp`, `alpha_gp` and `n40ca_gp` rows ran on a cluster (8 cores of an
> AMD EPYC "amd20" node per sampler, `benchmarks/hpc/`), the others on one
> local machine: compare wall times within a problem, not
> across these two groups; calls and ESS per thousand calls do not depend on
> the machine.

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
| alpha_gp (9) | reference: pocomc 8192/4096, 3 runs |  |  |  |  |  |  |  |  |  |  |  |  |  |  | -84.53 ± 0.01 |
| n40ca_gp (7) | emcee | 41 s | 96778 | 1% | 33% | 66% | 0% | 0.42 | 0.05 ? | 0.09 ? | — | — | — | 1.045 ✗ | 65 (< 50 τ) |  |
| n40ca_gp (7) | dynesty | 284 s | 278094 | 0% | 0% | 100% | 0% | 1.02 | 0.06 ? | 0.08 ? | 3843 | 13.55 | 13.82 | — |  | -11.9 ± 0.34 |
| n40ca_gp (7) | pocomc | 336 s | 280576 | 0% | 99% | 0% | 1% | 1.20 | 0.02 ? | 0.03 ? | 7180 | 21.37 | 25.59 | — |  | -11.4 ± 0.01 |
| n40ca_gp (7) | NUTS | 94 s | 95455 | 1% | 49% | 51% | 0% | 0.99 | 0.10 ? | 0.08 ? | 491 | 5.20 | 5.15 | 1.005 |  |  |
| n40ca_gp (7) | ptemcee | 225 s | 512778 | 0% | 33% | 67% | 0% | 0.44 | 0.14 ? | 0.15 ? | 4418 | 19.65 | 8.62 | 1.008 | 9 | -17.4 ± 21.77 |
| n40ca_gp (7) | HMC | 56 s | 52778 | 1% | 23% | 76% | 0% | 1.06 | 0.08 ? | 0.07 ? | 722 | 12.89 | 13.67 | 1.003 |  |  |
| n40ca_gp (7) | PyMC NUTS | 139 s | 84674 | 1% | 44% | 55% | 0% | 1.64 | 0.10 ? | 0.07 ? | 626 | 4.49 | 7.39 | 1.004 |  |  |
| n40ca_gp (7) | PyMC HMC | 106 s | 61151 | 1% | 64% | 35% | 0% | 1.74 | 0.05 ? | 0.10 ? | 655 | 6.16 | 10.71 | 1.003 |  |  |
| n40ca_gp (7) | reference: pocomc 8192/4096, 3 runs |  |  |  |  |  |  |  |  |  |  |  |  |  |  | -11.34 ± 0.01 |
| omp (9) | emcee | 183 s | 96876 | 1% | 33% | 66% | 0% | 1.89 | 1.26 ✗ | 1.29 ✗ | — | — | — | 1.593 ✗ | 170 (< 50 τ) |  |
| omp (9) | dynesty | 565 s | 255013 | 0% | 0% | 100% | 0% | 2.21 | 0.20 ✗ | 0.12 ? | — | — | — | — |  | -6.4 ± 0.32 |
| omp (9) | pocomc | 807 s | 311296 | 0% | 99% | 0% | 1% | 2.59 | 0.07 ? | 0.07 ? | 6815 | 8.44 | 21.89 | — |  | -6.7 ± 0.01 |
| omp (9) | NUTS | 626 s | 161531 | 1% | 51% | 49% | 0% | 3.87 | 0.91 ✗ | 0.35 ✗ | — | — | — | 1.694 ✗ |  |  |
| omp (9) | ptemcee | 927 s | 512876 | 0% | 33% | 67% | 0% | 1.81 | 1.35 ✗ | 7.34 ✗ | — | — | — | 1.535 ✗ | 97 (< 50 τ) | -12.0 ± 2.34 |
| omp (9) | HMC | 214 s | 52876 | 2% | 23% | 76% | 0% | 4.04 | 0.31 ✗ | 0.75 ✗ | — | — | — | 1.005 |  |  |
| omp (9) | PyMC NUTS | 613 s | 106554 | 1% | 45% | 54% | 0% | 5.76 | 0.32 ✗ | 0.68 ✗ | — | — | — | 1.006 |  |  |
| omp (9) | PyMC HMC | 727 s | 130313 | 1% | 80% | 19% | 0% | 5.58 | 0.30 ✗ | 0.79 ✗ | — | — | — | 1.007 |  |  |
| omp (9) | reference: pocomc 8192/4096, 3 runs |  |  |  |  |  |  |  |  |  |  |  |  |  |  | -6.65 ± 0.04 |

Which sampler is correct (? included), and fastest among those that are,
on each problem:

| problem | correct | fastest correct |
|---|---|---|
| `linear` | all four | emcee and NUTS, 6 s each; emcee with ten times the ESS per second |
| `gp` | none at these budgets | emcee's means are closest (0.07), but its widths are 0.30 off and its kept chain is 25 $\tau$, not 50 |
| `modes` | NUTS | NUTS, 61 s |
| `elastic` | dynesty, pocomc, NUTS | pocomc, 312 s (NUTS 324 s, dynesty 646 s) |
| `hierarchy` | NUTS | NUTS, 41 s, at 3000 draws per chain |
| `omp` | pocomc (?) | pocomc, 807 s, at 4096/2048 |
| `alpha_gp` | pocomc | pocomc, 1048 s, at 4096/2048 |
| `n40ca_gp` | all but emcee, each within the reference's resolution (?) | HMC, 56 s (NUTS 94 s, PyMC HMC 106 s, PyMC NUTS 139 s, ptemcee 225 s, dynesty 284 s, pocomc 336 s); with log Z, pocomc (dynesty's is 1.7 errors low) |

- **Funnels defeat the samplers that start from the prior.**  `gp`, `modes`
  and `hierarchy` each have a scale parameter that, when small, pins the
  others into a narrow neck: the GP's log variance against its bound, the
  shared normalisation mode's width, the hierarchy's $\tau$.  dynesty and
  pocomc under-visit the neck and come out wrong, in the means or the widths
  depending on the problem (on `gp` both bias the log variance, by 0.20 and
  0.60; on `hierarchy` both come out too narrow, by 0.11 and 0.21).  More
  resolution narrows the error only slowly, and on `hierarchy` it costs an
  hour (dynesty misses the width by 0.11 after 71 min, pocomc is 0.39 off
  after 57 min).  NUTS, adapting a step size to the neck, is correct on
  `modes` and `hierarchy` in about a minute (on `hierarchy` with 3000 draws
  per chain; 1000 give an R-hat of 1.021).  On `gp` it is not: 31 of its
  transitions diverge in the neck and its widths come out 0.23 too narrow,
  while its means agree.  emcee on `gp`, started at the MAP, has its means
  within the tolerance but its widths 0.30 off, and its kept chain is
  25 $\tau$.
- **Separated modes defeat the samplers that start at the MAP, and R-hat
  seldom tells.**  On `omp` and `alpha_gp` the gradient samplers, started
  from a ball about the MAP, stay in the island or family they start in.
  On `omp`, blackjax's HMC and PyMC's NUTS and HMC are 0.30 to 0.32 sd off
  in their means and 0.68 to 0.79 in their widths, while their
  rank-normalised R-hats are 1.005 to 1.007; on `alpha_gp` all four are
  0.57 to 0.60 off in their means and 1.42 to 1.47 in their widths, with
  R-hats of 1.002 to 1.014.  Chains that never visit an island agree with
  one another about it.  blackjax's NUTS on `omp` is the exception: its two
  chains disagree (R-hat 1.69, min ESS 3), and it is 0.91 off.  Only
  pocomc, at eight times its default resolution, is not shown wrong on
  both; at lower resolutions it weighs the islands differently from seed to
  seed (the sweep above).  On the one-mode problems the gradient samplers do
  well: on `n40ca_gp` every converged sampler is within the reference's
  resolution, and on `elastic` NUTS, pocomc and dynesty are correct.
- **The ensembles do not converge in these budgets** outside `linear`, where
  emcee needs 12 000 steps for the rank-normalised R-hat to pass (3000
  and 6000 steps give 1.019 and 1.010).  emcee is short of 50 $\tau$ everywhere else, and on
  `omp`, `alpha_gp` and `hierarchy` it is 1.3, 1.3 and 1.4 sd off, with
  R-hats of 1.59, 1.61 and 1.50.  ptemcee, started at the MAP, is 14 and 11 $\tau$
  long on `omp` and `alpha_gp`, with R-hats of 1.54 and 1.45, and 1.4 sd
  off on both; its $\beta$ = 1 walkers have collapsed in one parameter each
  (`log_sigma_B` on `omp`, `log_A_q` on `alpha_gp`), to an sd 1/1500 and
  1/2600 of the reference's (width errors of 7.3 and 7.9).  On `n40ca_gp` it
  converges (149 $\tau$, R-hat 1.008), 0.14 / 0.15 from the reference, within
  its resolution.
- **Evidence.**  Against the exact log Z, dynesty is within its error on
  `linear` and `gp` ($80.3 \pm 0.28$ against 80.48) but 4 errors high on
  `hierarchy` ($76.2 \pm 0.55$ against 73.97); pocomc is within its error on
  `hierarchy`, 1.2 errors low on `linear` and 6 errors low on `gp` ($80.2 \pm 0.04$), where its
  posterior is biased too.  Against the optical-model references (pooled
  pocomc, checked by dynesty), dynesty's evidence is right even where its
  posterior is not: $-6.42 \pm 0.32$ against -6.65 on `omp`, $-84.61 \pm 0.31$
  against -84.53 on `alpha_gp`, though 1.7 errors low on `n40ca_gp`
  ($-11.92 \pm 0.34$ against -11.34).  ptemcee's thermodynamic integration is
  2.3, 1.0 and 0.3 errors low on the three ($-11.95 \pm 2.34$, $-89.36 \pm 4.87$,
  $-17.4 \pm 21.8$), errors too wide to say much.  Like Kish's ESS, pocomc's
  log Z error leaves out the correlation between its particles.
- **Cost per call.**  On a reaction model the solver dominates whatever the
  sampler: a call costs 1.7 to 2.6 ms on `omp` and `alpha_gp` and 0.4 to
  1.2 ms on `n40ca_gp` (the cluster), and 2.8 to 3.7 ms on `elastic` (the
  local machine).  A gradient call costs 3.6 to 4.0 ms in blackjax on `omp`
  and `alpha_gp`, and 5.6 to 5.8 ms in PyMC on `omp`, which repeats the
  forward pass for its gradient Op; the two give the same answers (NUTS on
  `alpha_gp` is 0.60 / 1.43 off in blackjax and 0.59 / 1.43 in PyMC).
  PyMC's NUTS on `alpha_gp`, at 21.6 ms a call, is an unexplained outlier.
  On the cheap densities emcee, evaluating its walkers in one batch, pays
  0.015 to 0.19 ms per point; dynesty, which cannot batch, pays 3 to 84
  times more, and pocomc 0.5 to 5 ms, its per-call time including the
  training of its normalizing flow.  pocomc spends 62 to 99 % of its calls
  tempering from the prior to the posterior.

## `gp_discrepancy` under other seeds: `gp_seed_check.py`

The notebook fits its three-rung cross-section ladder with pocomc at
4096/2048 and seeds 10, 11 and 12, one run per rung.  `gp_seed_check.py`
reruns the ladder at the same settings with three other seed sets (20 + i,
30 + i, 40 + i for rung i) to see how much of what the notebook quotes is
the seed's (`results/hpc/gp_seed_check.json`; wall times on one local
machine):

| seeds | log Z: bare | proportional error | GP, growing amplitude | $\Delta\log Z$: proportional - bare | GP - proportional |
|---|---|---|---|---|---|
| notebook (10-12) | -467.21 ± 0.01 | 91.12 ± 0.01 | 103.91 ± 0.04 | 558.3 | 12.8 |
| 20-22 | -467.20 ± 0.01 | 91.11 ± 0.01 | 103.92 ± 0.02 | 558.3 | 12.8 |
| 30-32 | -467.22 ± 0.01 | 91.14 ± 0.02 | 103.97 ± 0.07 | 558.4 | 12.8 |
| 40-42 | -467.19 ± 0.01 | 91.14 ± 0.01 | 103.93 ± 0.02 | 558.3 | 12.8 |

The volume parameters' medians, and the notebook's 68 % intervals in
brackets:

| rung | seeds | $V_v$ | $W_v$ | $R_v$ | $a_v$ |
|---|---|---|---|---|---|
| bare | notebook | 26.82 [26.41, 27.25] | 16.63 [16.39, 16.86] | 4.431 [4.409, 4.452] | 0.659 [0.653, 0.665] |
| bare | 20, 30, 40 | 26.81, 26.82, 26.81 | 16.62, 16.62, 16.62 | 4.432, 4.431, 4.432 | 0.659, 0.659, 0.659 |
| proportional error | notebook | 29.98 [27.49, 32.80] | 18.18 [16.82, 19.66] | 4.277 [4.153, 4.391] | 0.690 [0.655, 0.725] |
| proportional error | 20, 30, 40 | 30.06, 30.21, 29.91 | 18.21, 18.27, 18.12 | 4.272, 4.265, 4.278 | 0.690, 0.692, 0.689 |
| GP, growing amplitude | notebook | 43.80 [42.46, 45.36] | 13.64 [12.69, 14.60] | 4.288 [4.221, 4.347] | 0.487 [0.466, 0.511] |
| GP, growing amplitude | 20, 30, 40 | 43.70, 43.74, 43.60 | 13.65, 13.65, 13.69 | 4.294, 4.290, 4.298 | 0.485, 0.486, 0.484 |

Every number the notebook quotes holds under other seeds: each rung's log Z
moves by at most 0.06 from the notebook's, the evidence for the GP over the
proportional error is 12.8 in every set, and every median moves by at most
0.16 of its 68 % interval's half-width (the GP rung's $R_v$).  The GP
rung's medians in every set, the notebook's included, lie within 0.12
posterior standard deviations of `n40ca_gp`'s reference.
