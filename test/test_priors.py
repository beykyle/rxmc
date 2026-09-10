import unittest

import numpy as np
import scipy.stats

from rxmc.config import ParameterConfig
from rxmc.params import Parameter
from rxmc.priors import (
    IndependentPrior,
    TruncatedNormalPrior,
    as_prior,
    clip_unit_cube,
)


class TestIndependentPrior(unittest.TestCase):
    def setUp(self):
        self.dists = [
            scipy.stats.norm(loc=0.0, scale=1.0),
            scipy.stats.uniform(loc=0.0, scale=5.0),
            scipy.stats.norm(loc=2.0, scale=0.5),
        ]
        self.prior = IndependentPrior(self.dists, seed=0)

    def test_dim(self):
        self.assertEqual(self.prior.dim, 3)

    def test_logpdf_scalar(self):
        """1-D input returns a finite float."""
        lp = self.prior.logpdf(np.array([0.0, 2.5, 2.0]))
        self.assertIsInstance(lp, float)
        self.assertTrue(np.isfinite(lp))

    def test_logpdf_batch(self):
        """2-D input returns a 1-D array of the right length."""
        theta = np.array([[0.0, 1.0, 2.0], [1.0, 4.0, 2.5]])
        lp = self.prior.logpdf(theta)
        self.assertEqual(lp.shape, (2,))
        self.assertTrue(np.all(np.isfinite(lp)))

    def test_logpdf_out_of_support(self):
        """Points outside the support of a bounded marginal return -inf."""
        # uniform(0, 5): value -1 is outside support
        lp = self.prior.logpdf(np.array([0.0, -1.0, 2.0]))
        self.assertEqual(lp, -np.inf)

    def test_logpdf_wrong_dim_raises(self):
        with self.assertRaises(ValueError):
            self.prior.logpdf(np.array([0.0, 1.0]))

    def test_rvs_shape(self):
        samples = self.prior.rvs(10)
        self.assertEqual(samples.shape, (10, 3))

    def test_rvs_reproducible(self):
        p1 = IndependentPrior(self.dists, seed=42)
        p2 = IndependentPrior(self.dists, seed=42)
        np.testing.assert_array_equal(p1.rvs(5), p2.rvs(5))

    def test_prior_transform_median(self):
        """prior_transform at u=0.5 returns the per-component median."""
        u = np.full(3, 0.5)
        theta = self.prior.prior_transform(u)
        expected = np.array([d.ppf(0.5) for d in self.dists])
        np.testing.assert_allclose(theta, expected, atol=1e-10)

    def test_prior_transform_roundtrip(self):
        """logpdf(prior_transform(u)) is finite for interior u."""
        u = np.array([0.3, 0.7, 0.5])
        theta = self.prior.prior_transform(u)
        lp = self.prior.logpdf(theta)
        self.assertTrue(np.isfinite(lp))

    def test_compatible_with_parameter_config(self):
        """IndependentPrior works as the prior argument to ParameterConfig."""
        params = [Parameter(f"p{i}") for i in range(3)]
        config = ParameterConfig(
            params=params,
            prior=self.prior,
            initial_proposal_distribution=self.prior,
        )
        self.assertEqual(config.ndim, 3)

        lp = config.prior_logpdf(np.array([0.0, 2.5, 2.0]))
        self.assertTrue(np.isfinite(lp))

        x0 = config.x0(4)
        self.assertEqual(x0.shape, (4, 3))

        theta = config.prior_transform(np.full(3, 0.5))
        self.assertEqual(theta.shape, (3,))
        self.assertTrue(np.all(np.isfinite(theta)))


class TestTruncatedNormalPrior(unittest.TestCase):
    def setUp(self):
        self.prior = TruncatedNormalPrior(
            mu=[0.0, 1.0],
            sigma=[1.0, 1.0],
            lower=[-5.0, -4.0],
            upper=[5.0, 6.0],
            seed=0,
        )

    def test_dim(self):
        self.assertEqual(self.prior.dim, 2)

    def test_logpdf_scalar(self):
        lp = self.prior.logpdf(np.array([0.0, 1.0]))
        self.assertIsInstance(lp, float)
        self.assertTrue(np.isfinite(lp))

    def test_logpdf_batch(self):
        theta = np.array([[0.0, 1.0], [1.0, 2.0]])
        lp = self.prior.logpdf(theta)
        self.assertEqual(lp.shape, (2,))
        self.assertTrue(np.all(np.isfinite(lp)))

    def test_logpdf_out_of_support(self):
        lp = self.prior.logpdf(np.array([10.0, 1.0]))
        self.assertEqual(lp, -np.inf)

    def test_rvs_shape(self):
        samples = self.prior.rvs(8)
        self.assertEqual(samples.shape, (8, 2))

    def test_rvs_within_bounds(self):
        samples = self.prior.rvs(500)
        self.assertTrue(np.all(samples[:, 0] >= -5.0))
        self.assertTrue(np.all(samples[:, 0] <= 5.0))
        self.assertTrue(np.all(samples[:, 1] >= -4.0))
        self.assertTrue(np.all(samples[:, 1] <= 6.0))

    def test_prior_transform_median(self):
        """Symmetric bounds → median maps to mu."""
        u = np.full(2, 0.5)
        theta = self.prior.prior_transform(u)
        np.testing.assert_allclose(theta, [0.0, 1.0], atol=1e-6)

    def test_prior_transform_roundtrip(self):
        u = np.array([0.2, 0.8])
        theta = self.prior.prior_transform(u)
        lp = self.prior.logpdf(theta)
        self.assertTrue(np.isfinite(lp))


class TestUnitCubeClipping(unittest.TestCase):
    def test_clip_unit_cube(self):
        u = clip_unit_cube([0.0, 0.5, 1.0])
        eps = np.finfo(float).eps
        np.testing.assert_allclose(u, [eps, 0.5, 1.0 - eps])
        self.assertEqual(u.dtype, float)
        self.assertTrue(np.all(u > 0.0) and np.all(u < 1.0))

    def test_independent_prior_boundary_is_finite(self):
        # an unbounded marginal's ppf is +-inf at exactly 0 / 1
        prior = IndependentPrior([scipy.stats.norm(0, 1), scipy.stats.norm(0, 1)])
        theta = prior.prior_transform([0.0, 1.0])
        self.assertTrue(np.all(np.isfinite(theta)))
        self.assertLess(theta[0], 0.0)
        self.assertGreater(theta[1], 0.0)

    def test_truncated_normal_boundary_is_finite(self):
        prior = TruncatedNormalPrior(mu=[0.0], sigma=[1.0], lower=[-2.0], upper=[3.0])
        theta = prior.prior_transform([0.0])
        self.assertTrue(np.all(np.isfinite(theta)))
        np.testing.assert_allclose(theta, [-2.0], atol=1e-6)

    def test_as_prior_wraps_lists_only(self):
        dists = [scipy.stats.norm(0, 1)]
        wrapped = as_prior(dists)
        self.assertIsInstance(wrapped, IndependentPrior)
        self.assertIs(wrapped.distributions[0], dists[0])
        prior = TruncatedNormalPrior(mu=[0.0], sigma=[1.0], lower=[-1.0], upper=[1.0])
        self.assertIs(as_prior(prior), prior)


if __name__ == "__main__":
    unittest.main()
