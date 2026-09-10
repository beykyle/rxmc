# rxmc

`rxmc` is a library for Bayesian calibration of reaction models to
experimental data, with the error model — statistical and systematic,
experimental and theoretical — declared explicitly as part of the problem
and calibrated with external samplers (emcee, dynesty, black-box-bayes).

**The 1.0 rewrite is in progress on this branch.**

- [`docs/groundup_design.md`](docs/groundup_design.md) is the design and
  plan of record: the API skeleton, what is harvested from 0.x, the gaps,
  the milestones, and the release path.
- [`docs/recipes.md`](docs/recipes.md) lists every supported use case with
  its spelling and expected behaviour.  Every recipe is a test under
  `test/recipes/`; the richest are tutorial notebooks.

The 0.x package is preserved at tag `v0.1.0` and on branch `legacy/0.x`.

## Installation

```bash
git clone -b rewrite git@github.com:beykyle/rxmc.git
cd rxmc
python -m venv .venv && source .venv/bin/activate
pip install -e '.[validation]'
```

Python ≥ 3.12; `jitr >= 3.0` from PyPI.

## Validation

```bash
python -m isort --check-only src test && python -m black --check src test && python -m ruff check src test
python -m pytest            # fast tier
python -m pytest -m slow    # converged tier, run nightly in CI
sphinx-build -W docs docs/_build/html
```
