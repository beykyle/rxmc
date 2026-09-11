# rxmc

`rxmc` calibrates reaction models to experimental data by Bayesian
inference, with the error model declared as part of the problem: which
errors are statistical and which are correlated, whether a normalisation is
inferred or marginalised, whether the model is allowed a discrepancy, which
points are held out.  A reviewer can read the declaration and write down the
likelihood.  The library owns no sampler: a compiled problem exposes the
densities and the prior transform that emcee, dynesty or black-box-bayes
need.

##  Documentation

The documentation website, including API reference is https://beykyle.github.io/rxmc/.

## Quickstart

Declare, compile, hand to a sampler, read the chain back by name.

```python
import emcee
import numpy as np
from scipy import stats

import rxmc as rx

# a model with parameters and priors
m = rx.Parameter("m", prior=stats.norm(0.0, 5.0))
b = rx.Parameter("b", prior=stats.norm(0.0, 5.0))
line = rx.Model(lambda x, m, b: m * x + b, [m, b])

# data with reported statistical errors
rng = np.random.default_rng(0)
x = np.linspace(0.0, 1.0, 20)
data = rx.Dataset(x, 0.6 * x + 2.0 + rng.normal(0.0, 0.1, x.size), np.full(x.size, 0.1))

# one comparison, one likelihood, one compiled problem
problem = rx.Problem([rx.Constraint([rx.Comparison(data, line)])])
print(problem.names)  # ['m', 'b']

sampler = emcee.EnsembleSampler(16, problem.ndim, problem.log_posterior)
sampler.run_mcmc(problem.sample_prior(16, rng=1), 1000)
samples = sampler.get_chain(discard=300, flat=True)
print(samples[:, problem.columns(m)].mean(), samples[:, problem.columns(b)].mean())

# the posterior predictive on the data points, with the error model
draws = rx.diagnostics.predictive_draws(problem, samples[::20], n_rep=2)
print(rx.diagnostics.coverage_curve(draws, data.y, [0.68]))
```

The error model is a sum of covariance *terms* on the constraint.  A
normalisation the experiment did not report, inferred alongside the model:

```python
from rxmc import terms as T

log_eta = rx.Parameter("log_eta", prior=stats.norm(-2.0, 1.0))
problem = rx.Problem([rx.Constraint([rx.Comparison(data, line)], terms=[T.normalization(log_eta)])])
print(problem.names)  # ['m', 'b', 'log_eta']
```

## A reaction model

A jitr optical potential is a `Model` whose solver is built from the
dataset's kinematics.  EXFOR measurements arrive through
`from_measurement`, which converts units and keeps the reported systematics
as inert metadata until asked for.  Optical-model posteriors are correlated
and sometimes multimodal, so drive them with nested sampling.

```python
from types import SimpleNamespace

import dynesty
import jitr
from jitr.optical_potentials.potential_forms import thomas_safe, woods_saxon_safe

reaction = jitr.reactions.ElasticReaction(target=(40, 20), projectile=(1, 0))
R = 1.2 * 40 ** (1 / 3)


def central(r, V, W, a):
    return -(V + 1j * W) * woods_saxon_safe(r, R, a)


def spin_orbit(r, Vso, Rso, aso):
    return Vso * thomas_safe(r, Rso, aso) / jitr.utils.constants.WAVENUMBER_PION**2


params = [
    rx.Parameter("V", prior=stats.norm(48.0, 5.0), bounds=(0.0, np.inf)),
    rx.Parameter("W", prior=stats.norm(4.0, 3.0), bounds=(0.0, np.inf)),
    rx.Parameter("a", prior=stats.norm(0.65, 0.1), bounds=(0.3, 1.2)),
]
omp = rx.reactions.ElasticXS(
    "dXS/dA", central, spin_orbit, lambda ws, *v: (tuple(v), (6.0, R, 0.45)), params, lmax=10
)

# an EXFOR-shaped measurement (exfor_tools.Distribution has these fields); here mock data
angles = np.linspace(10.0, 150.0, 12)
truth = omp.bind(np.deg2rad(angles), {"reaction": reaction, "Elab": 14.1})(48.0, 4.0, 0.65)
measurement = SimpleNamespace(
    x=angles, y=1e3 * truth * (1 + rng.normal(0, 0.05, angles.size)), Einc=14.1,
    quantity="dXS/dA", y_units="mb/sr", statistical_err=1e3 * truth * 0.05,
    systematic_norm_err=0.04, systematic_offset_err=None, subentry="mock",
)
d = rx.from_measurement(measurement, reaction=reaction)
comp = rx.Comparison(d, omp)
problem = rx.Problem([rx.Constraint([comp], terms=comp.reported_terms())])

sampler = dynesty.NestedSampler(problem.log_likelihood, problem.prior_transform, problem.ndim, nlive=50)
sampler.run_nested(dlogz=5.0, print_progress=False)
print(problem.names, sampler.results.logz[-1])
```

## Where to go next

- [`docs/recipes.md`](docs/recipes.md): every supported use case with its
  spelling and expected behaviour.  Each recipe is a test under
  `test/recipes/`.
- [`examples/`](examples/): nine notebooks, the tutorials for the recipes,
  from a line to an error-model comparison on real α + ⁴⁴Ca data and a
  hierarchical calibration.
- [`docs/design.md`](docs/design.md): the maintainer's description of the
  library, its rules and its testing tiers.
- The documentation website, including API reference, at
  https://beykyle.github.io/rxmc/.

## Installation

Python 3.12 or later; the runtime dependencies are `numpy`, `scipy`,
`jitr >= 3.0` and `exfor-tools`.  Until the 1.0 pre-releases are on PyPI
(`pip install --pre rxmc`), install from the branch:

```bash
git clone -b rewrite git@github.com:beykyle/rxmc.git
cd rxmc
python -m venv .venv && source .venv/bin/activate
pip install -e '.[examples]'        # or '.[validation]' to run the tests
```

## Validation

```bash
python -m isort --check-only src test && python -m black --check src test && python -m ruff check src test
python -m pytest            # fast tier
python -m pytest -m slow    # converged tier: required on pushes and PRs to main
python -m pytest -n 4 --nbmake --nbmake-timeout=3600 examples   # the notebooks, same workflow
sphinx-build -W docs docs/_build/html
```

The three Python blocks of this README are executed by `test/test_readme.py`.
