"""Tests for the sampler-agnostic ``rxmc.model_comparison`` utilities."""

import unittest
from unittest.mock import patch

import numpy as np
from scipy import stats
from scipy.special import logsumexp
from sklearn.gaussian_process.kernels import RBF

from helpers import manual_mvn_loglike
from rxmc.config import CalibrationConfig, ParameterConfig
from rxmc.constraint import Constraint
from rxmc.covariance import ConstraintCovariance, Term, kernel_term, noise_term
from rxmc.evidence import Evidence
from rxmc.model_comparison import (
    compare_logz,
    coverage_curve,
    coverage_error,
    heldout_log_predictive,
    log_jacobian,
    log_posterior_predictive,
    logz_summary,
    predictive_draws,
    sharpness,
    split_samples,
)
from rxmc.observation import Observation
from rxmc.params import Parameter
from rxmc.physical_model import Polynomial
from rxmc.priors import IndependentPrior
from rxmc.transforms import log


class TestPredictiveDraws(unittest.TestCase):
    def setUp(self):
        self.pm = Polynomial(order=1)
        self.x = np.linspace(0.0, 4.0, 5)
        self.y = 1.0 + 2.0 * self.x
        self.err = np.full(5, 0.3)
        self.obs = Observation(self.x, self.y, y_stat_err=self.err)

    def test_draw_covariance_recovers_sigma(self):
        eps = Parameter("log eps")
        c = Constraint([self.obs], self.pm, extra_terms=[noise_term(eps)])
        theta = np.array([[1.0, 2.0]])
        cov = np.array([[np.log(0.4)]])
        draws = predictive_draws(
            c, theta, cov, n_rep=40000, rng=np.random.default_rng(0)
        )
        self.assertEqual(draws.shape, (40000, 5))
        np.testing.assert_allclose(draws.mean(axis=0), self.y, atol=0.02)
        S = np.cov(draws.T)
        np.testing.assert_allclose(S, np.diag(self.err**2 + 0.16), atol=0.02)

    def test_model_only_returns_ym(self):
        c = Constraint([self.obs], self.pm)
        theta = np.array([[1.0, 2.0], [0.0, 1.0]])
        d = predictive_draws(c, theta, model_only=True)
        np.testing.assert_allclose(d[0], self.y)
        np.testing.assert_allclose(d[1], self.x)

    def test_model_only_skips_covariance_assembly(self):
        kernel = RBF(length_scale=1.0)
        c = Constraint(
            [self.obs.masked([True, True, False, True, True])],
            self.pm,
            extra_terms=[kernel_term(kernel)],
        )
        ym, _ = c.predict_and_covariance((1.0, 2.0), (0.0,))
        with patch.object(ConstraintCovariance, "matrix") as m:
            d = predictive_draws(c, [1.0, 2.0], [0.0], model_only=True)
        m.assert_not_called()
        self.assertEqual(d.shape, (1, 4))
        np.testing.assert_allclose(d[0], ym)

    def test_one_dimensional_row_is_one_sample(self):
        c = Constraint([self.obs], self.pm)
        d = predictive_draws(c, [1.0, 2.0], n_rep=3, rng=np.random.default_rng(2))
        self.assertEqual(d.shape, (3, 5))
        eps = Parameter("log eps")
        cp = Constraint([self.obs], self.pm, extra_terms=[noise_term(eps)])
        with self.assertRaises(ValueError):
            predictive_draws(cp, [[1.0, 2.0], [1.0, 2.0]], [[0.0]])

    def test_tiny_variances_not_inflated(self):
        obs = Observation(self.x, self.y, y_stat_err=np.full(5, 1e-4))
        c = Constraint([obs], self.pm)
        draws = predictive_draws(
            c, [1.0, 2.0], n_rep=40000, rng=np.random.default_rng(3)
        )
        var = draws.var(axis=0)
        np.testing.assert_allclose(var, 1e-8, rtol=0.03)

    def test_requires_cov_samples_when_parametric(self):
        c = Constraint([self.obs], self.pm, extra_terms=[noise_term(Parameter("e"))])
        with self.assertRaises(ValueError):
            predictive_draws(c, np.array([[1.0, 2.0]]))

    def test_respects_mask(self):
        c = Constraint([self.obs.masked([True, False, True, False, True])], self.pm)
        d = predictive_draws(
            c, np.array([[1.0, 2.0]]), n_rep=3, rng=np.random.default_rng(1)
        )
        self.assertEqual(d.shape, (3, 3))


class TestCoverageSharpness(unittest.TestCase):
    def test_coverage_near_nominal_for_matching_draws(self):
        rng = np.random.default_rng(0)
        n_pts = 2000
        draws = rng.normal(0.0, 1.0, (4000, n_pts))
        y = rng.normal(0.0, 1.0, n_pts)
        levels = np.array([0.5, 0.9])
        cov = coverage_curve(draws, y, levels)
        np.testing.assert_allclose(cov, levels, atol=0.03)
        self.assertLess(coverage_error(draws, y, levels), 0.03)
        # overconfident draws under-cover
        cov_narrow = coverage_curve(0.3 * draws, y, levels)
        self.assertTrue(np.all(cov_narrow < levels - 0.2))

    def test_sharpness_width(self):
        rng = np.random.default_rng(0)
        draws = rng.normal(0.0, 1.0, (20000, 3))
        w = sharpness(draws)
        np.testing.assert_allclose(w, 2 * 0.9945, atol=0.05)
        w_exp = sharpness(np.zeros((10, 2)), transform=np.exp)
        np.testing.assert_allclose(w_exp, 0.0)
        w95 = sharpness(draws, percentiles=(2.5, 97.5))
        np.testing.assert_allclose(w95, 2 * 1.96, atol=0.15)


class TestHeldout(unittest.TestCase):
    def test_heldout_log_predictive_matches_manual(self):
        pm = Polynomial(order=1)
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([3.1, 4.8, 7.2, 9.1])
        err = np.array([0.2, 0.2, 0.3, 0.3])
        obs = Observation(x, y, y_stat_err=err).masked_where(lambda x: x < 2.5)
        fit = Constraint(
            [obs], pm, extra_terms=[Term(np.array(0.1 * np.ones(4)), kind="diag")]
        )
        held = fit.complement()
        samples = np.array([[1.0, 2.0], [1.2, 1.9]])
        lp = heldout_log_predictive(held, samples)
        for i, (a0, a1) in enumerate(samples):
            ym = a0 + a1 * x[2:]
            cov = np.diag(err[2:] ** 2 + 0.01)
            self.assertAlmostEqual(lp[i], manual_mvn_loglike(y[2:], ym, cov))

    def test_log_posterior_predictive(self):
        lp = np.array([-1.0, -2.0, -0.5])
        self.assertAlmostEqual(log_posterior_predictive(lp), logsumexp(lp) - np.log(3))
        logw = np.array([0.0, -np.inf, 0.0])
        self.assertAlmostEqual(
            log_posterior_predictive(lp, logw), logsumexp(lp[[0, 2]]) - np.log(2)
        )


class TestLogZ(unittest.TestCase):
    def test_summary_single_and_replicates(self):
        m, e, n = logz_summary([-10.0], [0.3])
        self.assertEqual((m, e, n), (-10.0, 0.3, 1))
        m, e, n = logz_summary([-10.0, -12.0], [0.3, 0.3])
        self.assertEqual((m, e, n), (-11.0, 1.0, 2))  # half-range dominates
        m, e, n = logz_summary([-10.0, -10.2], [0.5, 0.5])
        self.assertAlmostEqual(e, 0.5)  # reported error dominates

    def test_compare(self):
        r = compare_logz((-10.0, 0.5), (-15.0, 0.5))
        self.assertEqual(r["verdict"], "a")
        self.assertAlmostEqual(r["dlogZ"], 5.0)
        self.assertAlmostEqual(r["err"], np.hypot(0.5, 0.5))
        self.assertEqual(compare_logz((-15.0, 0.5), (-10.0, 0.5))["verdict"], "b")
        self.assertEqual(compare_logz((-10.0, 1.0), (-11.0, 1.0))["verdict"], "tie")

    def test_log_jacobian(self):
        y = np.array([2.0, 3.0])
        obs = Observation(np.array([0.0, 1.0]), y, transform=log)
        c = Constraint(
            [obs], Polynomial(order=0), extra_terms=[noise_term(Parameter("e"))]
        )
        self.assertAlmostEqual(log_jacobian(c), -np.sum(np.log(y)))


class TestSplitSamples(unittest.TestCase):
    def test_split_rows(self):
        pm = Polynomial(order=1)
        obs = Observation(
            np.arange(4.0), 1.0 + 2.0 * np.arange(4.0), y_stat_err=np.full(4, 0.1)
        )
        eps = Parameter("log eps")
        c = Constraint([obs], pm, extra_terms=[noise_term(eps)])
        ev = Evidence([c])
        mprior = IndependentPrior([stats.norm(0, 1), stats.norm(0, 1)])
        lprior = IndependentPrior([stats.norm(-2, 1)])
        config = CalibrationConfig(
            ev,
            ParameterConfig(pm.params, mprior, mprior),
            [ParameterConfig(list(c.params), lprior, lprior)],
        )
        samples = np.array([[1.0, 2.0, -1.0], [0.5, 1.5, -2.0]])
        m, covs = split_samples(config, samples)
        np.testing.assert_allclose(m, samples[:, :2])
        self.assertEqual(len(covs), 1)
        np.testing.assert_allclose(covs[0], samples[:, 2:])
        # single row
        m1, _ = split_samples(config, samples[0])
        self.assertEqual(m1.shape, (1, 2))


if __name__ == "__main__":
    unittest.main()
