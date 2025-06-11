import unittest
import numpy as np
from rxmc.observation import Observation, FixedCovarianceObservation
from rxmc.likelihood_model import (
    LikelihoodModel,
    FixedCovarianceLikelihood,
    LikelihoodWithSystematicError,
    ParametricLikelihoodModel,
    UnknownNoiseErrorModel,
    UnknownNoiseFractionErrorModel,
    UnknownNormalizationErrorModel,
    mahalanobis_distance_cholesky,
    log_likelihood,
)


class TestLikelihoodModel(unittest.TestCase):

    def setUp(self):
        # Mock observation and model prediction
        self.observation = Observation(
            x=np.array([0.0, 1.0, 2.0]), y=np.array([10.0, 15.0, 20.0])
        )
        # Create the LikelihoodModel instance
        self.likelihood_model = LikelihoodModel()

    def test_residual(self):
        delta = np.array([1.0, -1.0, 0.0])
        ym = self.observation.y - delta
        residual = self.likelihood_model.residual(self.observation, ym)
        np.testing.assert_array_equal(residual, delta)


class TestFixedCovarianceLikelihood(unittest.TestCase):

    def setUp(self):
        # Mock observation and model prediction
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

        # Create the FixedCovarianceLikelihood instance
        self.likelihood_instance = FixedCovarianceLikelihood()

    def test_covariance(self):
        cov = self.likelihood_instance.covariance(self.observation, self.ym)
        np.testing.assert_array_equal(cov, self.observation.covariance)

    def test_chi2(self):
        chi2_value = self.likelihood_instance.chi2(self.observation, self.ym)
        expected_chi2 = np.linalg.norm(self.delta)
        self.assertAlmostEqual(chi2_value, expected_chi2)

    def test_logpdf(self):
        logpdf_value = self.likelihood_instance.logpdf(self.observation, self.ym)
        expected_logpdf = -0.5 * (
            np.dot(self.delta, self.delta) + np.log(3) + 3 * np.log(2 * np.pi)
        )
        self.assertAlmostEqual(logpdf_value, expected_logpdf)

    def test_chi2_same_(self):
        chi2_value = self.likelihood_instance.chi2(self.observation, self.ym_same)
        expected_chi2 = 0.0
        self.assertAlmostEqual(chi2_value, expected_chi2)

    def test_logpdf_same_(self):
        logpdf_value = self.likelihood_instance.logpdf(self.observation, self.ym_same)
        expected_logpdf = 0.0
        self.assertAlmostEqual(logpdf_value, expected_logpdf)


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
        self.likelihood_instance = FixedCovarianceLikelihood()
        self.expected_chi2 = self.delta @ np.linalg.inv(self.cov) @ self.delta
        self.expected_logpdf = -0.5 * (
            self.expected_chi2 + np.log(np.linalg.det(self.cov)) + 3 * np.log(2 * np.pi)
        )

    def test_covariance(self):
        cov = self.likelihood_instance.covariance(self.observation, self.ym)
        np.testing.assert_array_equal(cov, self.observation.covariance)

    def test_chi2(self):
        chi2_value = self.likelihood_instance.chi2(self.observation, self.ym)
        self.assertAlmostEqual(chi2_value, self.expected_chi2)

    def test_logpdf(self):
        logpdf_value = self.likelihood_instance.logpdf(self.observation, self.ym)
        self.assertAlmostEqual(logpdf_value, self.expected_logpdf)


class TestLikelihoodWithSystematicError(unittest.TestCase):

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
        self.likelihood = LikelihoodWithSystematicError(0.01)
        self.expected_covariance = (
            np.diag(self.observation.y_stat_err**2)
            + self.likelihood.fractional_uncorrelated_error**2 * np.diag(self.ym**2)
            + self.observation.y_sys_err_normalization**2 * np.outer(self.ym, self.ym)
            + self.observation.y_sys_err_offset**2 * np.ones((3, 3))
        )
        self.expected_chi2 = (
            self.delta.T @ np.linalg.inv(self.expected_covariance) @ self.delta
        )
        self.expected_logpdf = -0.5 * (
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

    def test_logpdf(self):
        logpdf_value = self.likelihood.logpdf(self.observation, self.ym)
        self.assertAlmostEqual(logpdf_value, self.expected_logpdf)


class TestUnknownNoiseFrac(unittest.TestCase):

    def setup(self):
        self.param = Parameter(
            "noise",
        )
        self.likelihood_params = [self.param]
        self.observation = Observation(
            x=np.array([0.0, 1.0, 2.0]),
            y=np.array([10.0, 15.0, 20.0]),
            y_stat_err=np.array([0.1, 0.1, 0.1]),
            y_sys_err_normalization=0.04,
            y_sys_err_offset=0.2,
        )
        self.ym = self.observation.y + np.array([1.0, -1.0, 0.0])
        self.delta = self.observation.y - ym
        self.likelihood = UnknownNoiseFractionErrorModel(self.likelihood_params, 0.01)
        self.noise_fraction = 0.312
        self.expected_covariance = (
            np.diag(self.noise_fraction * 2 * self.observation.y**2)
            + self.likelihood.fractional_uncorrelated_error**2 * np.diag(self.ym**2)
            + self.observation.y_sys_err_normalization**2 * np.outer(self.ym, self.ym)
            + self.observation.y_sys_err_offset**2 * np.ones((3, 3))
        )
        self.expected_chi2 = self.delta.t @ self.expected_covariance @ self.delta
        self.expected_logpdf = -0.5 * (
            self.expected_chi2
            + np.log(np.linalg.det(self.expected_covariance))
            + 3 * np.log(2 * np.pi)
        )

        def test_covariance(self):
            cov = self.likelihood.covariance(
                self.observation, self.ym, self.noise_fraction
            )
            self.assertequal(cov.shape, (3, 3))
            np.testing.assert_array_almost_equal(cov, self.expected_covariance)

        def test_chi2(self):
            chi2_value = self.likelihood.chi2(
                self.observation, self.ym, self.noise_fraction
            )
            self.assertalmostequal(chi2_value, self.expected_chi2)

        def test_logpdf(self):
            logpdf_value = self.likelihood.logpdf(
                self.observation, self.ym, self.noise_fraction
            )
            self.assertalmostequal(logpdf_value, self.expected_logpdf)


class TestUnknownNoise(unittest.TestCase):

    def setup(self):
        self.param = Parameter(
            "noise",
        )
        self.likelihood_params = [self.param]
        self.observation = Observation(
            x=np.array([0.0, 1.0, 2.0]),
            y=np.array([10.0, 15.0, 20.0]),
            y_stat_err=np.array([0.1, 0.1, 0.1]),
            y_sys_err_normalization=0.04,
            y_sys_err_offset=0.2,
        )
        self.ym = self.observation.y + np.array([1.0, -1.0, 0.0])
        self.ym = self.observation.y - self.delta
        self.likelihood = UnknownNoiseErrorModel(self.likelihood_params, 0.01)
        self.noise = 0.312
        self.expected_covariance = (
            np.diag(self.noise**2)
            + self.likelihood.fractional_uncorrelated_error**2 * np.diag(self.ym**2)
            + self.observation.y_sys_err_normalization**2 * np.outer(self.ym, self.ym)
            + self.observation.y_sys_err_offset**2 * np.ones((3, 3))
        )
        self.expected_chi2 = self.delta.t @ self.expected_covariance @ self.delta
        self.expected_logpdf = -0.5 * (
            self.expected_chi2
            + np.log(np.linalg.det(self.expected_covariance))
            + 3 * np.log(2 * np.pi)
        )

        def test_covariance(self):
            cov = self.likelihood.covariance(self.observation, self.ym, self.noise)
            self.assertequal(cov.shape, (3, 3))
            np.testing.assert_array_almost_equal(cov, self.expected_covariance)

        def test_chi2(self):
            chi2_value = self.likelihood.chi2(self.observation, self.ym, self.noise)
            self.assertalmostequal(chi2_value, self.expected_chi2)

        def test_logpdf(self):
            logpdf_value = self.likelihood.logpdf(self.observation, self.ym, self.noise)
            self.assertalmostequal(logpdf_value, self.expected_logpdf)


class TestUnknownNormalization(unittest.TestCase):

    def setup(self):
        self.param = Parameter(
            "noise",
        )
        self.likelihood_params = [self.param]
        self.observation = Observation(
            x=np.array([0.0, 1.0, 2.0]),
            y=np.array([10.0, 15.0, 20.0]),
            y_stat_err=np.array([0.1, 0.1, 0.1]),
            y_sys_err_normalization=0.04,
            y_sys_err_offset=0.2,
        )
        self.ym = self.observation.y + np.array([1.0, -1.0, 0.0])
        self.delta = self.observation.y - self.ym
        self.likelihood = UnknownNormalizationErrorModel(self.likelihood_params, 0.01)
        self.normalization_err = 0.312
        self.expected_covariance = (
            np.diag(self.observation.y_stat_err**2)
            + self.likelihood.fractional_uncorrelated_error**2 * np.diag(self.ym**2)
            + self.normalization_err**2 * np.outer(self.ym, self.ym)
            + self.observation.y_sys_err_offset**2 * np.ones((3, 3))
        )
        self.expected_chi2 = self.delta.t @ self.expected_covariance @ self.delta
        self.expected_logpdf = -0.5 * (
            self.expected_chi2
            + np.log(np.linalg.det(self.expected_covariance))
            + 3 * np.log(2 * np.pi)
        )

        def test_covariance(self):
            cov = self.likelihood.covariance(
                self.observation, self.ym, self.normalization_err
            )
            self.assertequal(cov.shape, (3, 3))
            np.testing.assert_array_almost_equal(cov, self.expected_covariance)

        def test_chi2(self):
            chi2_value = self.likelihood.chi2(
                self.observation, self.ym, self.normalization_err
            )
            self.assertalmostequal(chi2_value, self.expected_chi2)

        def test_logpdf(self):
            logpdf_value = self.likelihood.logpdf(
                self.observation, self.ym, self.normalization_err
            )
            self.assertalmostequal(logpdf_value, self.expected_logpdf)


if __name__ == "__main__":
    unittest.main()
