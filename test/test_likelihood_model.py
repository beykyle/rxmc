import unittest

import numpy as np
from rxmc.likelihood_model import (
    FixedCovarianceLikelihood,
    LikelihoodModel,
    UnknownModelError,
    UnknownNoiseErrorModel,
    UnknownNoiseFractionErrorModel,
    UnknownNormalizationErrorModel,
    UnknownNormalizationModel,
    mahalanobis_distance_sqr_cholesky,
)
from rxmc.observation import FixedCovarianceObservation, Observation


class TestFixedCovarianceLikelihoodDiagCovariance(unittest.TestCase):
    def setUp(self):
        self.observation = FixedCovarianceObservation(
            x=np.array([0.0, 1.0, 2.0]),
            y=np.array([10.0, 15.0, 20.0]),
            covariance=np.array(
                [1.0, 1.0, 1.0],
            ),
        )
        self.delta = np.array([1.0, -1.0, 0.0])
        self.ym = self.observation.y + self.delta
        self.ym_same = self.observation.y
        self.likelihood = FixedCovarianceLikelihood()

    def test_covariance(self):
        cov = self.likelihood.covariance(self.observation, self.ym)
        np.testing.assert_array_equal(cov, self.observation.cov)

    def test_chi2(self):
        chi2_value = self.likelihood.chi2(self.observation, self.ym)
        expected_chi2 = np.dot(self.delta, self.delta)
        self.assertAlmostEqual(chi2_value, expected_chi2)

    def test_log_likelihood(self):
        log_likelihood_value = self.likelihood.log_likelihood(self.observation, self.ym)
        expected_log_likelihood = -0.5 * (
            np.dot(self.delta, self.delta) + np.log(1) + 3 * np.log(2 * np.pi)
        )
        self.assertAlmostEqual(log_likelihood_value, expected_log_likelihood)

    def test_chi2_same_(self):
        chi2_value = self.likelihood.chi2(self.observation, self.ym_same)
        expected_chi2 = 0.0
        self.assertAlmostEqual(chi2_value, expected_chi2)

    def test_log_likelihood_same_(self):
        log_likelihood_value = self.likelihood.log_likelihood(
            self.observation, self.ym_same
        )
        expected_log_likelihood = -0.5 * (np.log(1) + 3 * np.log(2 * np.pi))
        self.assertAlmostEqual(log_likelihood_value, expected_log_likelihood)


class TestFixedCovarianceLikelihood(unittest.TestCase):
    def setUp(self):
        # positive definite covariance matrix
        self.cov = np.array(
            [
                [1.0, 0.5, 0.3],
                [0.5, 1.0, 0.2],
                [0.3, 0.2, 1.0],
            ]
        )
        self.observation = FixedCovarianceObservation(
            x=np.array([0.0, 1.0, 2.0]),
            y=np.array([10.0, 15.0, 20.0]),
            covariance=self.cov,
        )
        self.delta = np.array([1.0, -1.0, 0.0])
        self.ym = self.observation.y - self.delta
        self.likelihood = FixedCovarianceLikelihood()
        self.expected_chi2 = self.delta @ np.linalg.inv(self.cov) @ self.delta
        self.expected_log_likelihood = -0.5 * (
            self.expected_chi2 + np.log(np.linalg.det(self.cov)) + 3 * np.log(2 * np.pi)
        )

    def test_covariance(self):
        cov = self.likelihood.covariance(self.observation, self.ym)
        np.testing.assert_array_equal(cov, self.observation.cov)

    def test_chi2(self):
        chi2_value = self.likelihood.chi2(self.observation, self.ym)
        self.assertAlmostEqual(chi2_value, self.expected_chi2)

    def test_log_likelihood(self):
        log_likelihood_value = self.likelihood.log_likelihood(self.observation, self.ym)
        self.assertAlmostEqual(log_likelihood_value, self.expected_log_likelihood)


class TestLikelihoodModel(unittest.TestCase):
    def setUp(self):
        self.observation = Observation(
            x=np.array([0.0, 1.0, 2.0]),
            y=np.array([10.0, 15.0, 20.0]),
            y_stat_err=np.array([0.1, 0.1, 0.1]),
            y_sys_err_normalization=0.04,
            y_sys_err_offset=0.2,
        )
        self.ym = self.observation.y + np.array([1.0, -1.0, 0.0])
        self.delta = self.observation.y - self.ym
        self.likelihood = LikelihoodModel()
        self.expected_covariance = (
            self.observation.statistical_covariance
            + self.observation.systematic_offset_covariance
            + self.observation.systematic_normalization_covariance
            * np.outer(self.ym, self.ym)
        )
        self.expected_chi2 = (
            self.delta.T @ np.linalg.inv(self.expected_covariance) @ self.delta
        )
        self.expected_log_likelihood = -0.5 * (
            self.expected_chi2
            + np.log(np.linalg.det(self.expected_covariance))
            + 3 * np.log(2 * np.pi)
        )

    def test_covariance(self):
        cov = self.likelihood.covariance(self.observation, self.ym)
        self.assertEqual(cov.shape, (3, 3))
        np.testing.assert_allclose(cov, self.expected_covariance)

    def test_chi2(self):
        chi2_value = self.likelihood.chi2(self.observation, self.ym)
        self.assertAlmostEqual(chi2_value, self.expected_chi2)

    def test_log_likelihood(self):
        log_likelihood_value = self.likelihood.log_likelihood(self.observation, self.ym)
        self.assertAlmostEqual(log_likelihood_value, self.expected_log_likelihood)


class TestUnknownNoiseFrac(unittest.TestCase):
    def setUp(self):
        self.observation = Observation(
            x=np.array([0.0, 1.0, 2.0]),
            y=np.array([10.0, 15.0, 20.0]),
            y_stat_err=np.array([0.1, 0.1, 0.1]),
            y_sys_err_normalization=0.00,
            y_sys_err_offset=0.0,
        )
        self.ym = self.observation.y + np.array([1.0, -1.0, 0.0])
        self.delta = self.observation.y - self.ym
        self.likelihood = UnknownNoiseFractionErrorModel()
        self.noise_fraction = 0.312
        self.expected_covariance = (
            np.diag((self.noise_fraction * self.ym) ** 2)
            + self.observation.systematic_offset_covariance
            + self.observation.systematic_normalization_covariance
            * np.outer(self.ym, self.ym)
        )
        self.expected_chi2 = (
            self.delta.T @ np.linalg.inv(self.expected_covariance) @ self.delta
        )
        self.expected_log_likelihood = -0.5 * (
            self.expected_chi2
            + np.log(np.linalg.det(self.expected_covariance))
            + 3 * np.log(2 * np.pi)
        )

    def test_covariance(self):
        cov = self.likelihood.covariance(
            self.observation, self.ym, np.log(self.noise_fraction)
        )
        self.assertEqual(cov.shape, (3, 3))
        np.testing.assert_array_almost_equal(cov, self.expected_covariance)

    def test_chi2(self):
        chi2_value = self.likelihood.chi2(
            self.observation, self.ym, np.log(self.noise_fraction)
        )
        self.assertAlmostEqual(chi2_value, self.expected_chi2)

    def test_log_likelihood(self):
        log_likelihood_value = self.likelihood.log_likelihood(
            self.observation, self.ym, np.log(self.noise_fraction)
        )
        self.assertAlmostEqual(log_likelihood_value, self.expected_log_likelihood)


class TestUnknownNoise(unittest.TestCase):
    def setUp(self):
        self.observation = Observation(
            x=np.array([0.0, 1.0, 2.0]),
            y=np.array([10.0, 15.0, 20.0]),
            y_stat_err=np.array([0.1, 0.1, 0.1]),
            y_sys_err_normalization=0.04,
            y_sys_err_offset=0.2,
        )
        self.delta = np.array([1.0, -1.0, 0.0])
        self.ym = self.observation.y + self.delta
        self.likelihood = UnknownNoiseErrorModel()
        self.noise = 0.312
        self.expected_covariance = (
            np.diag(np.ones_like(self.ym) * self.noise**2)
            + self.observation.systematic_offset_covariance
            + self.observation.systematic_normalization_covariance
            * np.outer(self.ym, self.ym)
        )
        self.expected_chi2 = (
            self.delta.T @ np.linalg.inv(self.expected_covariance) @ self.delta
        )
        self.expected_log_likelihood = -0.5 * (
            self.expected_chi2
            + np.log(np.linalg.det(self.expected_covariance))
            + 3 * np.log(2 * np.pi)
        )

    def test_covariance(self):
        cov = self.likelihood.covariance(self.observation, self.ym, np.log(self.noise))
        self.assertEqual(cov.shape, (3, 3))
        np.testing.assert_array_almost_equal(cov, self.expected_covariance)

    def test_chi2(self):
        chi2_value = self.likelihood.chi2(self.observation, self.ym, np.log(self.noise))
        self.assertAlmostEqual(chi2_value, self.expected_chi2)

    def test_log_likelihood(self):
        log_likelihood_value = self.likelihood.log_likelihood(
            self.observation, self.ym, np.log(self.noise)
        )
        self.assertAlmostEqual(log_likelihood_value, self.expected_log_likelihood)


class TestUnknownNormalizationError(unittest.TestCase):
    def setUp(self):
        self.observation = Observation(
            x=np.array([0.0, 1.0, 2.0]),
            y=np.array([10.0, 15.0, 20.0]),
            y_stat_err=np.array([0.1, 0.1, 0.1]),
            y_sys_err_normalization=0.04,
            y_sys_err_offset=0.2,
        )
        self.ym = self.observation.y + np.array([1.0, -1.0, 0.0])
        self.delta = self.observation.y - self.ym
        self.likelihood = UnknownNormalizationErrorModel()
        self.normalization_err = 0.312
        self.expected_covariance = (
            self.observation.statistical_covariance
            + self.observation.systematic_offset_covariance
            + self.normalization_err**2 * np.outer(self.ym, self.ym)
        )
        self.expected_chi2 = (
            self.delta.T @ np.linalg.inv(self.expected_covariance) @ self.delta
        )
        self.expected_log_likelihood = -0.5 * (
            self.expected_chi2
            + np.log(np.linalg.det(self.expected_covariance))
            + 3 * np.log(2 * np.pi)
        )

    def test_covariance(self):
        cov = self.likelihood.covariance(
            self.observation, self.ym, np.log(self.normalization_err)
        )
        self.assertEqual(cov.shape, (3, 3))
        np.testing.assert_array_almost_equal(cov, self.expected_covariance)

    def test_chi2(self):
        chi2_value = self.likelihood.chi2(
            self.observation, self.ym, np.log(self.normalization_err)
        )
        self.assertAlmostEqual(chi2_value, self.expected_chi2)

    def test_log_likelihood(self):
        log_likelihood_value = self.likelihood.log_likelihood(
            self.observation, self.ym, np.log(self.normalization_err)
        )
        self.assertAlmostEqual(log_likelihood_value, self.expected_log_likelihood)


class TestUnknownNormalization(unittest.TestCase):
    def setUp(self):
        self.observation = Observation(
            x=np.array([0.0, 1.0, 2.0]),
            y=np.array([10.0, 15.0, 20.0]),
            y_stat_err=np.array([0.1, 0.1, 0.1]),
        )
        self.normalization = 0.9
        self.likelihood = UnknownNormalizationModel()

        self.ym = self.observation.y + np.array([1.0, -1.0, 0.0])
        self.delta = self.observation.y - self.ym * self.normalization
        self.expected_covariance = self.observation.statistical_covariance
        self.expected_chi2 = (
            self.delta.T @ np.linalg.inv(self.expected_covariance) @ self.delta
        )
        self.expected_log_likelihood = -0.5 * (
            self.expected_chi2
            + np.log(np.linalg.det(self.expected_covariance))
            + 3 * np.log(2 * np.pi)
        )

    def test_covariance(self):
        cov = self.likelihood.covariance(
            self.observation, self.ym, np.log(self.normalization)
        )
        self.assertEqual(cov.shape, (3, 3))
        np.testing.assert_array_almost_equal(cov, self.expected_covariance)

    def test_chi2(self):
        chi2_value = self.likelihood.chi2(
            self.observation, self.ym, np.log(self.normalization)
        )
        self.assertAlmostEqual(chi2_value, self.expected_chi2)

    def test_log_likelihood(self):
        log_likelihood_value = self.likelihood.log_likelihood(
            self.observation, self.ym, np.log(self.normalization)
        )
        self.assertAlmostEqual(log_likelihood_value, self.expected_log_likelihood)


class TestMahalanobisDistanceCholesky(unittest.TestCase):
    def test_mahalanobis_distance_sqr(self):
        # Test case: Simple 2D example
        y = np.array([2.0, 3.0])
        ym = np.array([0.0, 0.0])
        cov = np.array([[1.0, 0.5], [0.5, 1.0]])

        residual = y - ym

        expected_mahalanobis_distance_sqr = residual.T @ np.linalg.inv(cov) @ residual
        expected_log_det = np.log(0.75)

        mahalanobis, log_det = mahalanobis_distance_sqr_cholesky(y, ym, cov)

        self.assertAlmostEqual(mahalanobis, expected_mahalanobis_distance_sqr, places=5)
        self.assertAlmostEqual(log_det, expected_log_det, places=5)

    def test_mahalanobis_distance_sqr_0(self):
        # Test case: Simple 2D example
        y = np.array([2.0, 3.0])
        ym = y
        cov = np.array([[1.0, 0.5], [0.5, 1.0]])

        expected_mahalanobis_distance_sqr = 0
        expected_log_det = np.log(0.75)

        mahalanobis, log_det = mahalanobis_distance_sqr_cholesky(y, ym, cov)

        self.assertAlmostEqual(mahalanobis, expected_mahalanobis_distance_sqr, places=5)
        self.assertAlmostEqual(log_det, expected_log_det, places=5)

    def test_mahalanobis_distance_sqr_near_singular(self):
        # Test case: Simple 2D example
        y = np.array([1, 2, 3])
        ym = np.array([1.1, 1.9, 3.1])

        # Ill-conditioned covariance matrix
        cov = np.array([[1.0, 0.999, 0.999], [0.999, 1.0, 0.999], [0.999, 0.999, 1.0]])
        residual = y - ym

        expected_mahalanobis_distance_sqr = residual.T @ np.linalg.inv(cov) @ residual
        expected_log_det = np.log(np.linalg.det(cov))
        mahalanobis, log_det = mahalanobis_distance_sqr_cholesky(y, ym, cov)

        self.assertAlmostEqual(mahalanobis, expected_mahalanobis_distance_sqr, places=5)
        self.assertAlmostEqual(log_det, expected_log_det, places=5)


class TestUnknownModelError(unittest.TestCase):
    def setUp(self):
        # Setup Observation instance with mock data
        self.observation = Observation(
            x=np.array([0.0, 1.0, 2.0]),
            y=np.array([10.0, 15.0, 20.0]),
            y_stat_err=np.array([0.1, 0.1, 0.1]),
            y_sys_err_normalization=0.04,
            y_sys_err_offset=0.2,
        )

        # Model prediction deviations from observations
        self.ym = self.observation.y + np.array([1.0, -1.0, 0.0])
        self.delta = self.observation.y - self.ym

        # Initialize KnownModelError class
        self.likelihood = UnknownModelError()

        # Free parameter: fractional uncorrelated error
        self.frac_err = 0.312

        # Expected covariance calculations
        self.expected_covariance = (
            self.observation.statistical_covariance
            + self.observation.systematic_offset_covariance
            + self.observation.systematic_normalization_covariance
            * np.outer(self.ym, self.ym)
            + self.frac_err**2 * np.diag(0.5**2 * (self.ym + self.observation.y) ** 2)
        )

        # Expected chi-squared value
        self.expected_chi2 = (
            self.delta.T @ np.linalg.inv(self.expected_covariance) @ self.delta
        )

        # Expected log-likelihood value
        self.expected_log_likelihood = -0.5 * (
            self.expected_chi2
            + np.log(np.linalg.det(self.expected_covariance))
            + 3 * np.log(2 * np.pi)
        )

    def test_covariance(self):
        cov = self.likelihood.covariance(
            self.observation, self.ym, np.log(self.frac_err)
        )
        self.assertEqual(cov.shape, (3, 3))
        np.testing.assert_array_almost_equal(cov, self.expected_covariance)

    def test_chi2(self):
        chi2_value = self.likelihood.chi2(
            self.observation, self.ym, np.log(self.frac_err)
        )
        self.assertAlmostEqual(chi2_value, self.expected_chi2)

    def test_log_likelihood(self):
        log_likelihood_value = self.likelihood.log_likelihood(
            self.observation, self.ym, np.log(self.frac_err)
        )
        self.assertAlmostEqual(log_likelihood_value, self.expected_log_likelihood)


if __name__ == "__main__":
    unittest.main()
