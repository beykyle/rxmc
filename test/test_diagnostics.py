"""Posterior checks on (problem, samples): draws, coverage, held-out scores."""

from unittest.mock import patch

import numpy as np
import pytest
from scipy import stats
from scipy.special import logsumexp

from helpers import manual_mvn_loglike
from rxmc import Comparison, Constraint, Dataset, Model, Parameter, Problem, Term
from rxmc.covariance import StructuredCovariance
from rxmc.diagnostics import (
    compare_logz,
    coverage_curve,
    coverage_error,
    heldout_log_predictive,
    log_posterior_predictive,
    logz_summary,
    predictive_draws,
    sharpness,
)
from rxmc.likelihood import StudentT
from rxmc.terms import noise
from rxmc.transforms import log

X = np.linspace(0.0, 4.0, 5)
Y = 1.0 + 2.0 * X
ERR = np.full(5, 0.3)


def poly(order):
    """``polynomial(order)`` with wide priors, so problems compile."""
    params = [Parameter(f"a{i}", prior=stats.norm(0, 10)) for i in range(order + 1)]
    return Model(lambda x, *a: sum(ai * x**i for i, ai in enumerate(a)), params)


def line_problem(terms=(), masks=None, **kw):
    d = Dataset(X, Y, ERR, label="d")
    c = Constraint([Comparison(d, poly(1))], terms=terms, masks=masks, **kw)
    return Problem([c]), c


class TestPredictiveDraws:
    def test_draw_covariance_recovers_sigma(self):
        p, _ = line_problem([noise(Parameter("log_eps", prior=stats.norm(-2, 1)))])
        row = np.array([1.0, 2.0, np.log(0.4)])
        draws = predictive_draws(p, row, n_rep=40000, rng=0)
        assert draws.shape == (40000, 5)
        np.testing.assert_allclose(draws.mean(axis=0), Y, atol=0.02)
        np.testing.assert_allclose(np.cov(draws.T), np.diag(ERR**2 + 0.16), atol=0.02)

    def test_model_only_returns_ym_and_assembles_nothing(self):
        p, _ = line_problem()
        with patch.object(StructuredCovariance, "matrix") as m:
            d = predictive_draws(p, [[1.0, 2.0], [0.0, 1.0]], model_only=True)
        m.assert_not_called()
        np.testing.assert_allclose(d[0], Y)
        np.testing.assert_allclose(d[1], X)

    def test_one_row_masks_and_width_check(self):
        p, _ = line_problem(masks=[np.array([True, False, True, False, True])])
        d = predictive_draws(p, [1.0, 2.0], n_rep=3, rng=1)
        assert d.shape == (3, 3)
        with pytest.raises(ValueError, match=r"\(n, 2\)"):
            predictive_draws(p, [[1.0, 2.0, 3.0]])

    def test_student_t_draws_follow_the_multivariate_t(self):
        p, _ = line_problem(likelihood=StudentT(Parameter("nu", bounds=(1, 30))))
        draws = predictive_draws(p, [1.0, 2.0, 6.0], n_rep=100000, rng=0)
        # a multivariate t with scale diag(ERR**2) has covariance nu/(nu-2) times it
        np.testing.assert_allclose(draws.var(axis=0), 1.5 * ERR**2, rtol=0.05)
        # one mixing scale per draw: the points' |residuals| move together
        r = np.abs(draws - Y)
        assert np.corrcoef(r[:, 0], r[:, 1])[0, 1] > 0.05

    def test_tiny_variances_not_inflated(self):
        d = Dataset(X, Y, np.full(5, 1e-9), label="tiny")
        p = Problem([Constraint([Comparison(d, poly(1))])])
        draws = predictive_draws(p, [1.0, 2.0], n_rep=2000, rng=3)
        assert np.all(draws.std(axis=0) < 1e-8)


class TestCoverageSharpness:
    def test_coverage_near_nominal_for_matching_draws(self):
        rng = np.random.default_rng(0)
        draws = rng.normal(0.0, 1.0, (4000, 2000))
        y = rng.normal(0.0, 1.0, 2000)
        levels = np.array([0.5, 0.9])
        np.testing.assert_allclose(coverage_curve(draws, y, levels), levels, atol=0.03)
        assert coverage_error(draws, y, levels) < 0.03
        assert np.all(coverage_curve(0.3 * draws, y, levels) < levels - 0.2)

    def test_sharpness_width(self):
        rng = np.random.default_rng(0)
        draws = rng.normal(0.0, 1.0, (20000, 3))
        np.testing.assert_allclose(sharpness(draws), 2 * 0.9945, atol=0.05)
        np.testing.assert_allclose(sharpness(np.zeros((10, 2)), transform=np.exp), 0.0)
        np.testing.assert_allclose(
            sharpness(draws, percentiles=(2.5, 97.5)), 2 * 1.96, atol=0.15
        )


class TestHeldout:
    def setup_method(self):
        d = Dataset(
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([3.1, 4.8, 7.2, 9.1]),
            np.array([0.2, 0.2, 0.3, 0.3]),
            label="d",
        )
        self.d = d
        self.model = poly(1)

    def problems(self, terms, cut=2.5):
        c = Constraint([Comparison(self.d, self.model)], terms=terms)
        fit = c.masked_where(lambda x: x < cut)
        return Problem([fit]), Problem([fit.complement()]), Problem([c])

    def test_marginal_matches_manual_and_partitions_without_a_spanning_term(self):
        fit, held, full = self.problems([Term(0.1 * np.ones(4), kind="diag")])
        samples = np.array([[1.0, 2.0], [1.2, 1.9]])
        lp = heldout_log_predictive(held, samples)
        for i, (a0, a1) in enumerate(samples):
            ym = a0 + a1 * self.d.x[2:]
            cov = np.diag(self.d.y_err[2:] ** 2 + 0.01)
            assert lp[i] == pytest.approx(manual_mvn_loglike(self.d.y[2:], ym, cov))
        np.testing.assert_allclose(
            lp + [fit.log_likelihood(s) for s in samples],
            [full.log_likelihood(s) for s in samples],
        )
        # the conditional equals the marginal when nothing spans the split
        np.testing.assert_allclose(heldout_log_predictive(held, samples, given=fit), lp)
        np.testing.assert_allclose(
            predictive_draws(held, samples, given=fit, model_only=True),
            predictive_draws(held, samples, model_only=True),
        )

    def test_conditional_under_a_spanning_matrix_term(self):
        rng = np.random.default_rng(5)
        A = rng.normal(size=(4, 4))
        K = A @ A.T / 4  # couples every point with every other
        fit, held, full = self.problems([Term(K, kind="matrix")])
        theta = np.array([1.0, 2.0])
        S = np.diag(self.d.y_err**2) + K
        ym = theta[0] + theta[1] * self.d.x
        F, H = slice(0, 2), slice(2, 4)
        r = self.d.y[F] - ym[F]
        mean = ym[H] + S[H, F] @ np.linalg.solve(S[F, F], r)
        cov = S[H, H] - S[H, F] @ np.linalg.solve(S[F, F], S[F, H])
        lp = heldout_log_predictive(held, theta, given=fit)
        assert lp[0] == pytest.approx(manual_mvn_loglike(self.d.y[H], mean, cov))
        # and it is not the marginal
        assert lp[0] != pytest.approx(heldout_log_predictive(held, theta)[0])
        draws = predictive_draws(held, theta, n_rep=40000, rng=0, given=fit)
        np.testing.assert_allclose(draws.mean(axis=0), mean, atol=0.02)
        np.testing.assert_allclose(np.cov(draws.T), cov, atol=0.03)
        # the conditional is exactly the joint over the full data divided by the fit
        joint = full.log_likelihood(theta) - fit.log_likelihood(theta)
        assert lp[0] == pytest.approx(joint)

    def test_conditional_reads_the_columns_of_every_constraint(self):
        # two constraints with a parameter each: the second's conditional must
        # read b's column, not the first one
        rng = np.random.default_rng(5)
        A = rng.normal(size=(4, 4))
        K = A @ A.T / 4
        cs = []
        for s, label in ((1.0, "d1"), (3.0, "d2")):
            d = Dataset(self.d.x, s * self.d.y, self.d.y_err, label=label)
            p = Parameter(f"s{label}", prior=stats.norm(0, 10))
            m = Model(lambda x, s: s * x, [p])
            cs.append(Constraint([Comparison(d, m)], terms=[Term(K, kind="matrix")]))
        fits = [c.masked_where(lambda x: x < 2.5) for c in cs]
        fit, held = Problem(fits), Problem([f.complement() for f in fits])
        full = Problem(cs)
        theta = np.array([1.0, 3.0])
        lp = heldout_log_predictive(held, theta, given=fit)
        assert lp[0] == pytest.approx(
            full.log_likelihood(theta) - fit.log_likelihood(theta)
        )
        S = np.diag(self.d.y_err**2) + K
        ym = theta[1] * self.d.x
        F, H = slice(0, 2), slice(2, 4)
        mean = ym[H] + S[H, F] @ np.linalg.solve(S[F, F], 3.0 * self.d.y[F] - ym[F])
        draws = predictive_draws(held, theta, constraint=1, given=fit, model_only=True)
        np.testing.assert_allclose(draws[0], mean)

    def test_given_needs_no_marginal_priors_and_keeps_doubly_masked_rows_out(self):
        # the coefficients have only a joint prior, and a y = 0 point (not
        # finite in log space) is masked out of both the fit and the held-out view
        a0, a1 = Parameter("a0"), Parameter("a1")
        model = Model(lambda x, a0, a1: np.exp(a0 + a1 * x), [a0, a1])
        d = Dataset(np.arange(5.0), [0.0, 2.0, 3.0, 5.0, 8.0], np.full(5, 0.2))
        K = 0.05 * np.exp(-0.5 * np.subtract.outer(d.x, d.x) ** 2)
        c = Constraint([Comparison(d, model, space=log)], terms=[Term(K)])
        prior = [([a0, a1], stats.multivariate_normal(np.zeros(2), 4 * np.eye(2)))]
        fit = c.masked([np.array([False, True, True, False, False])])
        held = c.masked([np.array([False, False, False, True, True])])
        both = c.masked([np.array([False, True, True, True, True])])
        p_fit, p_held = Problem([fit], priors=prior), Problem([held], priors=prior)
        p_both = Problem([both], priors=prior)
        theta = np.array([0.5, 0.4])
        lp = heldout_log_predictive(p_held, theta, given=p_fit)
        assert lp[0] == pytest.approx(
            p_both.log_likelihood(theta) - p_fit.log_likelihood(theta)
        )

    def test_given_is_validated(self):
        fit, held, full = self.problems([])
        with pytest.raises(ValueError, match="overlap"):
            heldout_log_predictive(fit, [1.0, 2.0], given=fit)
        other = Problem([Constraint([Comparison(self.d, poly(1))])])
        with pytest.raises(ValueError, match="same comparison objects"):
            heldout_log_predictive(held, [1.0, 2.0], given=other)
        c = Constraint(
            [Comparison(self.d, self.model)],
            likelihood=StudentT(Parameter("nu", prior=stats.uniform(1, 30))),
        ).masked_where(lambda x: x < 2.5)
        fit_t, held_t = Problem([c]), Problem([c.complement()])
        with pytest.raises(ValueError, match="StudentT"):
            heldout_log_predictive(held_t, [1.0, 2.0, 1.0], given=fit_t)

    def test_log_posterior_predictive(self):
        lp = np.array([-1.0, -2.0, -0.5])
        assert log_posterior_predictive(lp) == pytest.approx(logsumexp(lp) - np.log(3))
        logw = np.array([0.0, -np.inf, 0.0])
        assert log_posterior_predictive(lp, logw) == pytest.approx(
            logsumexp(lp[[0, 2]]) - np.log(2)
        )


class TestLogZ:
    def test_summary_single_and_replicates(self):
        assert logz_summary([-10.0], [0.3]) == (-10.0, 0.3, 1)
        assert logz_summary([-10.0, -12.0], [0.3, 0.3]) == (-11.0, 1.0, 2)
        assert logz_summary([-10.0, -10.2], [0.5, 0.5])[1] == pytest.approx(0.5)

    def test_compare(self):
        r = compare_logz((-10.0, 0.5), (-15.0, 0.5))
        assert r["verdict"] == "a" and r["dlogZ"] == pytest.approx(5.0)
        assert r["err"] == pytest.approx(np.hypot(0.5, 0.5))
        assert compare_logz((-15.0, 0.5), (-10.0, 0.5))["verdict"] == "b"
        assert compare_logz((-10.0, 1.0), (-11.0, 1.0))["verdict"] == "tie"

    def test_log_jacobian_lives_on_the_problem(self):
        y = np.array([2.0, 3.0])
        d = Dataset(np.array([0.0, 1.0]), y, np.ones(2), label="d")
        p = Problem([Constraint([Comparison(d, poly(0), space=log)])])
        assert p.log_jacobian() == pytest.approx(-np.sum(np.log(y)))
        assert stats.norm(0, 1).cdf(0) == 0.5
