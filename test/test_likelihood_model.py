"""Tests for the stacked likelihood functionals and their Term-based covariances."""

import unittest

import numpy as np
from scipy.special import gammaln

from helpers import manual_mvn_loglike
from rxmc.constraint import Constraint
from rxmc.covariance import (
    DenseTerm,
    model_error_term,
    noise_fraction_term,
    noise_term,
    normalization_term,
)
from rxmc.likelihood_model import (
    Chi2,
    StudentT,
    mahalanobis_distance_sqr_cholesky,
)
from rxmc.observation import Observation
from rxmc.params import Parameter
from rxmc.physical_model import Polynomial


class LikelihoodTestBase(unittest.TestCase):
    def setUp(self):
        self.x = np.array([1.0, 2.0, 3.0])
        self.y = np.array([2.0, 4.0, 7.0])
        self.stat = np.array([0.1, 0.2, 0.3])
        self.obs = Observation(self.x, self.y, y_stat_err=self.stat)
        self.pm = Polynomial(order=1)
        self.model_params = (1.0, 1.5)
        self.ym = self.pm.evaluate(self.obs, *self.model_params)


class TestGaussianStatisticalOnly(LikelihoodTestBase):
    def test_matches_manual_mvn(self):
        c = Constraint([self.obs], self.pm)
        cov = np.diag(self.stat**2)
        expected = manual_mvn_loglike(self.y, self.ym, cov)
        self.assertAlmostEqual(c.log_likelihood(self.model_params), expected)

    def test_constraint_is_non_parametric(self):
        c = Constraint([self.obs], self.pm)
        self.assertEqual(c.n_params, 0)
        self.assertTrue(c.covariance.block_diagonal)


class TestUnknownNoise(LikelihoodTestBase):
    def test_constant_noise(self):
        eps = 0.05
        p = Parameter("log eps")
        c = Constraint([self.obs], self.pm, extra_terms=[noise_term(np.arange(3), p)])
        cov = np.diag(self.stat**2) + np.diag(np.full(3, eps**2))
        expected = manual_mvn_loglike(self.y, self.ym, cov)
        self.assertEqual(c.n_params, 1)
        self.assertAlmostEqual(
            c.log_likelihood(self.model_params, (np.log(eps),)), expected
        )

    def test_noise_fraction(self):
        eps = 0.05
        p = Parameter("log eps")
        c = Constraint(
            [self.obs], self.pm, extra_terms=[noise_fraction_term(np.arange(3), p)]
        )
        cov = np.diag(self.stat**2) + np.diag((eps * self.ym) ** 2)
        expected = manual_mvn_loglike(self.y, self.ym, cov)
        self.assertAlmostEqual(
            c.log_likelihood(self.model_params, (np.log(eps),)), expected
        )


class TestUnknownNormalizationError(LikelihoodTestBase):
    def test_eta(self):
        eta = 0.07
        p = Parameter("log eta")
        c = Constraint(
            [self.obs],
            self.pm,
            extra_terms=[normalization_term(np.arange(3), parameter=p)],
        )
        cov = np.diag(self.stat**2) + eta**2 * np.outer(self.ym, self.ym)
        expected = manual_mvn_loglike(self.y, self.ym, cov)
        self.assertAlmostEqual(
            c.log_likelihood(self.model_params, (np.log(eta),)), expected
        )


class TestUnknownModelError(LikelihoodTestBase):
    def test_averaging(self):
        gamma = 0.1
        p = Parameter("log gamma")
        c = Constraint(
            [self.obs],
            self.pm,
            extra_terms=[model_error_term(np.arange(3), p, averaging=True)],
        )
        z = 0.5 * (self.y + self.ym)
        cov = np.diag(self.stat**2) + np.diag((gamma * z) ** 2)
        expected = manual_mvn_loglike(self.y, self.ym, cov)
        self.assertAlmostEqual(
            c.log_likelihood(self.model_params, (np.log(gamma),)), expected
        )


class TestFixedCovariance(LikelihoodTestBase):
    def test_dense_term_fixed_full_covariance(self):
        cov = np.array([[0.04, 0.01, 0.0], [0.01, 0.09, 0.02], [0.0, 0.02, 0.16]])
        obs = Observation(self.x, self.y)  # no stat err -> zeros
        c = Constraint([obs], self.pm, extra_terms=[DenseTerm(np.arange(3), cov)])
        self.assertTrue(c.covariance.is_constant)
        expected = manual_mvn_loglike(self.y, self.ym, cov)
        self.assertAlmostEqual(c.log_likelihood(self.model_params), expected)

    def test_cholesky_cached(self):
        cov = np.diag([0.04, 0.09, 0.16])
        obs = Observation(self.x, self.y)
        c = Constraint([obs], self.pm, extra_terms=[DenseTerm(np.arange(3), cov)])
        L1, _ = c.covariance.cholesky(None)
        L2, _ = c.covariance.cholesky(None)
        self.assertIs(L1, L2)


class TestStudentT(LikelihoodTestBase):
    def test_student_t_value(self):
        nu = 5.0
        c = Constraint([self.obs], self.pm, likelihood=StudentT())
        self.assertEqual(c.n_params, 1)
        self.assertEqual(c.params[0].name, "degrees_of_freedom")
        ll = c.log_likelihood(self.model_params, (nu,))

        cov = np.diag(self.stat**2)
        d2, logdet = mahalanobis_distance_sqr_cholesky(self.y, self.ym, cov)
        n = 3
        expected = (
            gammaln((n + nu) / 2)
            - gammaln(nu / 2)
            - 0.5 * n * np.log(np.pi * nu)
            - 0.5 * logdet
            - 0.5 * (nu + n) * np.log1p(d2 / nu)
        )
        self.assertAlmostEqual(ll, expected)


class TestChi2(LikelihoodTestBase):
    def test_chi2_drops_logdet(self):
        c = Constraint([self.obs], self.pm, likelihood=Chi2())
        cov = np.diag(self.stat**2)
        d2, _ = mahalanobis_distance_sqr_cholesky(self.y, self.ym, cov)
        self.assertAlmostEqual(c.log_likelihood(self.model_params), -0.5 * d2)


class TestMahalanobisDistanceCholesky(unittest.TestCase):
    def test_diagonal(self):
        y = np.array([1.0, 2.0, 3.0])
        ym = np.array([1.1, 1.8, 3.2])
        cov = np.diag([0.1, 0.2, 0.3])
        d2, logdet = mahalanobis_distance_sqr_cholesky(y, ym, cov)
        self.assertAlmostEqual(d2, np.sum((y - ym) ** 2 / np.diag(cov)))
        self.assertAlmostEqual(logdet, np.log(np.prod(np.diag(cov))))


if __name__ == "__main__":
    unittest.main()
