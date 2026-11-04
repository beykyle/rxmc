"""rx.optimize: the posterior mode and the maximum of the likelihood."""

import warnings

import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

import rxmc as rx
from rxmc import Comparison, Constraint, Dataset, Model, Parameter, Problem


def line(prior_scale=5.0, bounds=(-np.inf, np.inf)):
    m = Parameter("m", prior=stats.norm(0, prior_scale))
    b = Parameter("b", prior=stats.norm(0, prior_scale), bounds=bounds)
    return Model(lambda x, m, b: m * x + b, [m, b])


def data(seed=0, n=8):
    rng = np.random.default_rng(seed)
    x = np.linspace(0.5, 2.5, n)
    return Dataset(x, 2.0 * x + 1.0 + rng.normal(0, 0.1, n), 0.1 * np.ones(n))


def weighted_least_squares(d, prior_scale=None):
    X = np.column_stack([d.x, np.ones_like(d.x)])
    A = X.T @ (X / d.y_err[:, None] ** 2)
    if prior_scale is not None:  # a normal prior adds its precision
        A = A + np.eye(2) / prior_scale**2
    return np.linalg.solve(A, X.T @ (d.y / d.y_err**2))


def test_map_is_the_analytic_posterior_mode():
    d = data()
    p = Problem([Constraint([Comparison(d, line(prior_scale=0.5))])])
    fit = rx.optimize.find_map(p, n_starts=4, rng=0)
    np.testing.assert_allclose(fit.x, weighted_least_squares(d, 0.5), rtol=1e-6)
    assert fit.fun == pytest.approx(p.log_posterior(fit.x))
    assert len(fit.ends) == 4 and fit.ends[0] == pytest.approx(fit.fun)
    assert fit.nfev > 0 and fit.wall > 0


def test_mle_ignores_the_prior():
    d = data()
    p = Problem([Constraint([Comparison(d, line(prior_scale=0.5))])])
    fit = rx.optimize.find_mle(p, n_starts=4, rng=0)
    np.testing.assert_allclose(fit.x, weighted_least_squares(d), rtol=1e-6)
    assert fit.fun == pytest.approx(p.log_likelihood(fit.x))


def test_an_optimum_on_a_bound_is_found_inside():
    # the intercept wants 1.0; its bound caps it at 0.8
    p = Problem([Constraint([Comparison(data(), line(bounds=(0.0, 0.8)))])])
    for find in (rx.optimize.find_map, rx.optimize.find_mle):
        fit = find(p, n_starts=3, rng=1)
        b = fit.x[p.columns(p.params[1])][0]
        assert 0.0 < b < 0.8 and b == pytest.approx(0.8, abs=1e-4)
        assert np.isfinite(fit.fun)


def test_likelihood_gradient_matches_finite_differences():
    p = Problem([Constraint([Comparison(data(), line())])])
    theta = np.array([1.7, 1.3])
    value, grad = p.value_and_grad_log_likelihood(theta)
    assert value == pytest.approx(p.log_likelihood(theta))
    h = 1e-5
    fd = [
        (p.log_likelihood(theta + h * e) - p.log_likelihood(theta - h * e)) / (2 * h)
        for e in np.eye(2)
    ]
    np.testing.assert_allclose(p.grad_log_likelihood(theta), fd, rtol=1e-5)
    np.testing.assert_allclose(grad, fd, rtol=1e-5)


def test_a_search_that_does_not_converge_says_so():
    p = Problem([Constraint([Comparison(data(), line())])])
    with pytest.warns(RuntimeWarning, match="did not converge"):
        fit = rx.optimize.find_mle(p, n_starts=3, rng=0, options={"maxiter": 1})
    assert fit.n_success == 0
    # a converged search reports every start and stays quiet
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fit = rx.optimize.find_map(p, n_starts=3, rng=0)
    assert fit.n_success == 3


def test_a_search_on_an_infinite_plateau_is_no_success():
    # the model is undefined for m > 3: searches that start there stop at once
    m = Parameter("m", prior=stats.uniform(0, 10))
    b = Parameter("b", prior=stats.norm(0, 5))
    model = Model(lambda x, m, b: jnp.where(m > 3, jnp.nan, m * x + b), [m, b])
    p = Problem([Constraint([Comparison(data(), model)])])
    fit = rx.optimize.find_mle(p, n_starts=8, rng=1)
    assert np.isfinite(fit.fun) and fit.fun == fit.ends[0]
    assert fit.n_success == int(np.isfinite(fit.ends).sum()) < 8
