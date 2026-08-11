# rxmc

`rxmc` is an orchestration layer for Bayesian calibration of reaction models to
large data sets with flexible, composable covariance modeling.

It is built around two complementary workflows:

1. **External-sampler orchestration** via `rxmc.config.CalibrationConfig`
   for drivers such as
   [`black-box-bayes`](https://github.com/beykyle/black-box-bayes/).
2. **In-package end-to-end prototyping** via `rxmc.walker.Walker`
   for smaller problems where you want to run the full MCMC workflow locally.

The package composes:

- curated experimental data as `Observation` objects,
- model predictions via `PhysicalModel`,
- uncertainty declared as additive covariance `Term`s (statistical,
  systematic, unknown-noise, and Gaussian-process discrepancy modes) via
  `rxmc.covariance`,
- maximal blocks of mutually-correlated data via `Constraint`,
- and full calibration problems via `Evidence`.

## Quickstart

```python
import numpy as np
from scipy import stats
import rxmc

# measured data: pure data plus (optional) reported systematics as metadata
obs = rxmc.observation.Observation(
    x=x, y=y, y_stat_err=y_err, y_sys_err_normalization=0.04
)

# a constraint owns one multivariate likelihood over its stacked observations;
# every correlated mode is an explicit covariance term - nothing is folded in
# silently
(support,) = rxmc.covariance.stacked_supports([obs])
constraint = rxmc.constraint.Constraint(
    [obs], model, extra_terms=obs.systematic_terms(support)
)
evidence = rxmc.evidence.Evidence([constraint])

# calibrate with the in-package Gibbs walker (or wrap in CalibrationConfig
# for emcee / dynesty)
prior = stats.multivariate_normal(mean=prior_mean, cov=prior_cov)
walker = rxmc.walker.Walker(
    rxmc.param_sampling.BatchedAdaptiveMetropolisSampler(
        params=model.params,
        starting_location=prior.mean,
        prior=prior,
        initial_proposal_cov=prior.cov / 100,
    ),
    evidence,
    rng=np.random.default_rng(1),
)
walker.walk(n_steps=10_000, burnin=1_000, batch_size=1_000)
```

> **Note — behavior change from pre-0.1 versions:** an `Observation`'s
> reported systematic errors are never folded into the covariance
> automatically. The default constraint covariance is the statistical diagonal
> only; systematics enter explicitly, e.g. via
> `obs.systematic_terms(support)` passed to `Constraint(extra_terms=...)`.


## Installation

### Development / local use

```bash
git clone git@github.com:beykyle/rxmc.git
cd rxmc
pip install -ve .
```

It is strongly recommended to use an isolated environment.

### `venv`

```bash
python -m venv .rxmc
source .rxmc/bin/activate
pip install -r requirements.txt
pip install -ve .
```

### `uv`:

```bash
uv env create
uv env use python
uv install -e .
```

### Optional extras

Install the example notebook runtime dependencies with:

```bash
pip install -ve '.[examples]'
```

Install the full validation toolchain with:

```bash
pip install -ve '.[validation]'
```

## Supported workflow 1: external samplers with `CalibrationConfig`

`CalibrationConfig` packages a calibration problem into a flat parameter space
for external drivers. It exposes the interface expected by
`black-box-bayes`-style tooling:

- `ndim`
- `starting_location(nwalkers)`
- `log_posterior(theta)`
- `log_likelihood(theta)`
- `prior_transform(u)`
- `log_posterior_batch(thetas)` (optional convenience interface)
- `parameter_names`

Typical flow:

1. Build `Observation` objects from your measurements.
2. Define a `PhysicalModel`.
3. Declare correlated uncertainty as covariance `Term`s (and pick a
   likelihood functional: Gaussian, Student-t, or chi-squared).
4. Combine them into `Constraint` objects and then `Evidence`.
5. Wrap the problem in `ParameterConfig` and `CalibrationConfig`.
6. Hand the resulting object to an external sampler.

This is the recommended path for larger production calibrations.

## Supported workflow 2: in-package MCMC with `Walker`

`Walker` is the smaller-scale, in-package path. It coordinates:

- one sampler for the physical-model parameters, and
- optional additional samplers for parametric likelihood sectors.

It alternates between these sectors in a Gibbs-style workflow and is useful
for:

- prototyping new likelihood models,
- validating new observation/model compositions,
- and running smaller end-to-end inference problems without introducing an
  external orchestration layer.

## Core concepts

### `Observation`

Pure measured data — `x`, `y`, and the statistical error on `y` — plus the
measurement's reported systematic magnitudes retained as inert metadata
(`y_sys_err_normalization`, `y_sys_err_offset`). It contributes only its
statistical diagonal by default; `obs.systematic_terms(support)` turns the
metadata into explicit covariance terms when you ask.

### `PhysicalModel`

Maps model parameters to predicted observables for a given `Observation`.
`ScaledModel` / `PerObservationScaledModel` wrap any model with latent
normalization parameters (Kennedy–O'Hagan style).

### Covariance `Term`s (`rxmc.covariance`)

Every uncertainty beyond the statistical diagonal is an explicit additive
contribution to the constraint's stacked covariance. Factory helpers cover the
common modes:

- `normalization_term` / `offset_term` — correlated systematics, fixed
  magnitude or free nuisance,
- `noise_term` / `noise_fraction_term` — unknown statistical noise,
- `model_error_term` — uncorrelated model error,
- `discrepancy_term` — Gaussian-process model discrepancy using sklearn
  kernels.

A term whose support spans several observations *couples* them (correlated
datasets); referencing the same `Parameter` object in two terms *shares* one
sampled value between them.

### Likelihood functionals

`GaussianLikelihood` (default), `StudentT` (heavy-tailed, with a
degrees-of-freedom parameter), and `Chi2` are thin functionals over the same
stacked covariance.

### `Constraint`

The maximal block of mutually-correlated data: observations, a physical model,
a covariance assembled from terms, and a likelihood functional.

### `Evidence`

Aggregates multiple independent constraints that share the same physical-model
parameterization.

## Examples and tutorials

The `examples/` directory contains richer notebooks and demos. The most useful
entry points are:

- `examples/linear_calibration_demo.ipynb` for the basic workflow,
- `examples/systematic_err_demo.ipynb` for the error-model catalog and
  systematic-error handling,
- `examples/measurement_to_calibration.ipynb` for the EXFOR-measurement →
  calibration path (units, retained systematics, guardrails),
- `examples/30s_optical_potential_calibration.ipynb` for a realistic optical
  potential calibration example,
- `examples/correlated_observations.ipynb` for correlated datasets and shared
  systematics (including across cross-section experiments),
- `examples/gp_discrepancy.ipynb` for Gaussian-process model discrepancy,
- `examples/robust_likelihoods.ipynb` for Student-t vs Gaussian likelihoods,
- `examples/normalization_inference.ipynb` for normalization-focused modeling,
- `examples/sampling_algos.ipynb` for sampling comparisons.

## Documentation

The full API reference and rendered example notebooks are hosted at
**https://beykyle.github.io/rxmc/**.

To build the documentation locally:

```bash
pip install -ve '.[docs]'
cd docs && make html
# then open docs/_build/html/index.html
```

## Testing

Run the full validation matrix with:

```bash
python -m isort --check-only src test
python -m black --check src test
python -m ruff check src test
python -m nbqa isort --check examples/*.ipynb
python -m black --check --ipynb examples/*.ipynb
python -m ruff check examples/*.ipynb
python -m pytest
```

If you want to apply the formatting fixes locally instead of only checking them:

```bash
python -m isort src test
python -m black src test
python -m ruff check --fix src test
python -m nbqa isort examples/*.ipynb
python -m black --ipynb examples/*.ipynb
```

Run only the unit tests with:

```bash
python -m pytest test
```

Run only the notebooks with:

```bash
python -m pytest examples
```

