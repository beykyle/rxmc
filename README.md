# rxmc

`rxmc` is an orchestration layer for Bayesian calibration of reaction models to
large data sets with flexible likelihood modeling.

It is built around two complementary workflows:

1. **External-sampler orchestration** via `rxmc.config.CalibrationConfig`
   for drivers such as
   [`black-box-bayes`](https://github.com/beykyle/black-box-bayes/).
2. **In-package end-to-end prototyping** via `rxmc.walker.Walker`
   for smaller problems where you want to run the full MCMC workflow locally.

The package composes:

- curated experimental data as `Observation` objects,
- model predictions via `PhysicalModel`,
- statistical assumptions via `LikelihoodModel`,
- independent data-model pairings via `Constraint`,
- and full calibration problems via `Evidence`.


## Installation

### Development / local use

```bash
git clone git@github.com:beykyle/rxmc.git
cd rxmc
pip install -ve .
```

It is strongly recommended to use an isolated environment.

### `conda` / `mamba`

```bash
conda env create -f environment.yml
conda activate rxmc
pip install -ve . --no-deps
```

```bash
mamba env create -f environment.yml
mamba activate rxmc
pip install -ve . --no-deps
```

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
3. Choose a `LikelihoodModel`.
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

Represents measured data and its covariance structure, including:

- statistical errors,
- systematic normalization errors,
- systematic offset errors,
- or fixed covariance matrices.

### `PhysicalModel`

Maps model parameters to predicted observables for a given `Observation`.

### `LikelihoodModel`

Encodes how predictions are compared to observations. Built-in options include:

- Gaussian covariance-based likelihoods,
- unknown noise models,
- unknown normalization / normalization-error models,
- unknown model-error terms,
- Student-t likelihoods,
- and a Gaussian-process discrepancy model using sklearn kernels.

### `Constraint`

Pairs observations, a physical model, and a likelihood model.

### `Evidence`

Aggregates multiple independent constraints that share the same physical-model
parameterization.

## Examples and tutorials

The `examples/` directory contains richer notebooks and demos. The most useful
entry points are:

- `examples/linear_calibration_demo.ipynb` for the basic workflow,
- `examples/systematic_err_demo.ipynb` for likelihood comparisons and
  systematic-error handling,
- `examples/30s_optical_potential_calibration.ipynb` for a realistic optical
  potential calibration example,
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

