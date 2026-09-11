"""Recipe 17: check the posterior predictive.

I want to know whether my error model is calibrated: do 68 % intervals
contain 68 % of the points, and how wide are they?
"""

import numpy as np
import pytest
from scipy import stats

from common import TRUE, line, line_data, oracle_samples
from rxmc import Comparison, Constraint, Dataset, Parameter, Problem
from rxmc import terms as T
from rxmc import transforms as tf
from rxmc.diagnostics import (
    compare_logz,
    coverage_curve,
    coverage_error,
    logz_summary,
    predictive_draws,
    sharpness,
)


def test_draws_are_ym_plus_correlated_noise_in_comparison_space():
    d = line_data()
    log_eps = Parameter("log_eps", prior=stats.norm(-2, 1))
    p = Problem([Constraint([Comparison(d, line())], terms=[T.noise(log_eps)])])
    theta = np.array([*TRUE, np.log(0.2)])
    draws = predictive_draws(p, theta, n_rep=20000, rng=0)
    ym = TRUE[0] * d.x + TRUE[1]
    np.testing.assert_allclose(draws.mean(0), ym, atol=0.01)
    np.testing.assert_allclose(
        np.cov(draws.T), p.constraints[0].matrix(theta), atol=0.01
    )
    # model_only: the predictions themselves, no covariance
    np.testing.assert_allclose(predictive_draws(p, theta, model_only=True)[0], ym)


def big_dataset(err_scale=1.0, seed=3, n=200):
    rng = np.random.default_rng(seed)
    x = np.linspace(0.5, 2.5, n)
    y = TRUE[0] * x + TRUE[1] + rng.normal(0, 0.1, n)
    return Dataset(x, y, err_scale * 0.1 * np.ones(n), label="big")


def test_coverage_is_nominal_for_the_right_error_model_and_low_for_an_overconfident_one():
    levels = np.array([0.5, 0.68, 0.9])
    d = big_dataset()
    p = Problem([Constraint([Comparison(d, line())])])
    s = oracle_samples(p, 400, rng=0)  # exact posterior rows stand in for a chain
    draws = predictive_draws(p, s, n_rep=4, rng=1)
    y_active = p.constraints[0].y[p.constraints[0].active]
    np.testing.assert_allclose(
        coverage_curve(draws, y_active, levels), levels, atol=0.1
    )
    assert coverage_error(draws, y_active, levels) < 0.1
    # the same data with the errors claimed five times smaller
    tight = Problem([Constraint([Comparison(big_dataset(err_scale=0.2), line())])])
    draws_tight = predictive_draws(
        tight, oracle_samples(tight, 400, rng=0), n_rep=4, rng=1
    )
    assert np.all(coverage_curve(draws_tight, y_active, levels) < levels - 0.2)


def test_sharpness_in_physical_units_for_a_log_fit():
    d = line_data()
    comp = Comparison(d, line(), space=tf.log)
    log_eps = Parameter("log_eps", prior=stats.norm(-2, 1))
    p = Problem([Constraint([comp], terms=[T.noise(log_eps)], statistical=False)])
    draws = predictive_draws(p, np.array([*TRUE, np.log(0.1)]), n_rep=2000, rng=2)
    width_log = sharpness(draws)
    width = sharpness(draws, transform=np.exp)
    assert np.all(width > 0) and np.all(width_log > 0)
    # a constant width in log space is a width proportional to y in physical units
    np.testing.assert_allclose(width_log, width_log.mean(), rtol=0.1)
    np.testing.assert_allclose(
        width / comp.data.y, (width / comp.data.y).mean(), rtol=0.15
    )


def test_evidence_bookkeeping():
    m, e, n = logz_summary([-10.0, -10.4], [0.1, 0.1])
    assert (m, n) == pytest.approx((-10.2, 2)) and e == pytest.approx(0.2)
    assert compare_logz((-10.0, 0.5), (-10.4, 0.5))["verdict"] == "tie"
    assert compare_logz((-10.0, 0.1), (-12.0, 0.1))["verdict"] == "a"
