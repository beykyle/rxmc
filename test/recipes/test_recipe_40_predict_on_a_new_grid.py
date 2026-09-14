"""Recipe 40: predict on a new grid, error model included.

I have a posterior, and I want predictions at ``x`` I never measured — a fine
plotting grid, an extrapolation — carrying the uncertainty my error model
declares, not only the spread of the model curves.
"""

import numpy as np
import pytest
from scipy import stats

from common import TRUE, line, oracle_samples
from rxmc import Comparison, Constraint, Dataset, Parameter, Problem
from rxmc import terms as T
from rxmc.diagnostics import coverage_curve, predictive_draws
from rxmc.predictive import grid_draws

SIGMA = 0.1
X_FINE = np.linspace(-1.0, 4.0, 40)  # past the data on both sides


def data(n=200, seed=4):
    rng = np.random.default_rng(seed)
    x = np.linspace(0.5, 2.5, n)
    y = TRUE[0] * x + TRUE[1] + rng.normal(0.0, SIGMA, n)
    return Dataset(x, y, np.full(n, SIGMA), label="d")


def problems():
    """The same line with the reported errors, and with a constant inferred noise."""
    d = data()
    model = line()
    comp = Comparison(d, model)
    reported = Problem([Constraint([comp])])
    log_sigma = Parameter("log_sigma", prior=stats.norm(np.log(0.2), 1.0))
    inferred = Problem(
        [Constraint([comp], terms=[T.noise(log_sigma)], statistical=False)]
    )
    return reported, inferred, model, d


def chain(reported, n=400):
    """Exact posterior rows for (m, b), with the noise at its true value."""
    mb = oracle_samples(reported, n, rng=0)
    return np.column_stack([mb, np.full(n, np.log(SIGMA))])


def test_the_error_model_travels_to_the_grid_and_the_model_band_does_not_widen():
    reported, inferred, model, _ = problems()
    rows = chain(reported)
    pred = model.bind(X_FINE)
    curve = grid_draws(inferred, pred, X_FINE, rows, model_only=True, levels=(16, 84))
    full = grid_draws(inferred, pred, X_FINE, rows, n_rep=4, rng=1, levels=(16, 84))
    w_curve, w_full = curve[1] - curve[0], full[1] - full[0]
    # a measurement at any x scatters by sigma about a curve known far better
    np.testing.assert_allclose(np.sqrt(w_full**2 - w_curve**2) / 2, SIGMA, rtol=0.15)
    # and the curve's own uncertainty fans out away from the data
    assert w_curve[0] > 3 * w_curve[len(X_FINE) // 2]


def test_at_the_data_the_model_band_under_covers_and_the_full_one_does_not():
    reported, inferred, _, d = problems()
    rows = chain(reported)
    levels = np.array([0.5, 0.68, 0.9])
    c = inferred.constraints[0]
    y = c.y[c.active]
    full = predictive_draws(inferred, rows, n_rep=4, rng=2, return_draws=True)
    curve = predictive_draws(inferred, rows, model_only=True, return_draws=True)
    np.testing.assert_allclose(coverage_curve(full, y, levels), levels, atol=0.08)
    assert np.all(coverage_curve(curve, y, levels) < 0.5 * levels)


def test_reported_errors_have_no_value_off_the_measured_points():
    reported, _, model, _ = problems()
    rows = chain(reported)[:, :2]
    pred = model.bind(X_FINE)
    with pytest.raises(ValueError, match="reported statistical errors"):
        grid_draws(reported, pred, X_FINE, rows)
    # saying so explicitly is allowed: the model alone
    band = grid_draws(reported, pred, X_FINE, rows, terms=[])
    np.testing.assert_allclose(
        band, grid_draws(reported, pred, X_FINE, rows, model_only=True)
    )
