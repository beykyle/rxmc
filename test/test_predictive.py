"""Tests for the predictive-uncertainty helpers (:mod:`rxmc.predictive`)."""

import unittest

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from rxmc.predictive import (
    _gp_posterior_mean_var,
    gp_posterior_predictive,
    predictive_band,
    total_predictive_band,
)


def make_kernel():
    return ConstantKernel(1.0) * RBF(length_scale=0.7) + WhiteKernel(1e-6)


class TestGPPosteriorPredictive(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.X_train = np.sort(rng.uniform(-2, 2, 12))
        self.residuals = np.sin(self.X_train) + 0.05 * rng.standard_normal(12)
        self.X_pred = np.linspace(-2.5, 2.5, 25)
        self.kernel = make_kernel()
        self.theta = self.kernel.theta  # log-space

    def test_matches_sklearn_gpr(self):
        noise_var = 0.01
        # sklearn GPR with the SAME fixed kernel (no hyperparameter optimisation)
        gpr = GaussianProcessRegressor(
            kernel=self.kernel.clone_with_theta(self.theta),
            optimizer=None,
            alpha=noise_var,
            normalize_y=False,
        )
        gpr.fit(self.X_train[:, None], self.residuals)
        mean_sk, cov_sk = gpr.predict(self.X_pred[:, None], return_cov=True)

        mean, cov = gp_posterior_predictive(
            self.kernel,
            self.theta,
            self.X_train,
            self.residuals,
            self.X_pred,
            train_noise_var=noise_var,
        )
        np.testing.assert_allclose(mean, mean_sk, atol=1e-6)
        np.testing.assert_allclose(cov, cov_sk, atol=1e-6)

    def test_noiseless_interpolation(self):
        # a noiseless kernel (no WhiteKernel) interpolates the residuals exactly
        kernel = ConstantKernel(1.0) * RBF(length_scale=0.7)
        mean, cov = gp_posterior_predictive(
            kernel,
            kernel.theta,
            self.X_train,
            self.residuals,
            self.X_train,
            train_noise_var=0.0,
            jitter=1e-10,
        )
        # exact interpolation up to numerical conditioning of the Gram matrix
        np.testing.assert_allclose(mean, self.residuals, atol=5e-3)
        self.assertLess(np.max(np.abs(np.diag(cov))), 1e-3)

    def test_2d_inputs(self):
        rng = np.random.default_rng(1)
        Xtr = rng.uniform(-1, 1, (8, 2))
        r = rng.standard_normal(8)
        Xp = rng.uniform(-1, 1, (5, 2))
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(1e-6)
        mean, cov = gp_posterior_predictive(kernel, kernel.theta, Xtr, r, Xp)
        self.assertEqual(mean.shape, (5,))
        self.assertEqual(cov.shape, (5, 5))


class TestPredictiveBand(unittest.TestCase):
    def test_percentiles(self):
        draws = np.arange(101)[:, None] * np.ones((1, 3))  # 0..100 on each column
        band = predictive_band(draws, levels=(16, 50, 84))
        self.assertEqual(band.shape, (3, 3))
        np.testing.assert_allclose(band[1], [50, 50, 50])
        np.testing.assert_allclose(band[0], [16, 16, 16])
        np.testing.assert_allclose(band[2], [84, 84, 84])


class TestTotalPredictiveBand(unittest.TestCase):
    def test_shape_and_finite(self):
        rng = np.random.default_rng(2)
        kernel = make_kernel()

        def mean_fn(x, m, b):
            return m * x + b

        x_train = np.linspace(0, 1, 10)
        y_train = 0.5 * x_train + 0.2 + np.sin(3 * x_train)
        x_pred = np.linspace(-0.2, 1.2, 30)

        n_kparams = len(kernel.theta)
        n_model = 2
        # fake posterior chain: [m, b, *log_theta]
        chain = np.column_stack(
            [
                0.5 + 0.05 * rng.standard_normal(50),
                0.2 + 0.05 * rng.standard_normal(50),
                np.tile(kernel.theta, (50, 1)),
            ]
        )
        self.assertEqual(chain.shape[1], n_model + n_kparams)

        band = total_predictive_band(
            mean_fn,
            kernel,
            x_train,
            y_train,
            x_pred,
            chain,
            n_model,
            noise_std=0.05,
            levels=(16, 84),
            n_draws=40,
            rng=rng,
        )
        self.assertEqual(band.shape, (2, len(x_pred)))
        self.assertTrue(np.all(np.isfinite(band)))
        self.assertTrue(np.all(band[1] >= band[0]))

    def _mean_fn(self, x, m, b):
        return m * x + b

    def test_raises_on_kernel_theta_width_mismatch(self):
        kernel = make_kernel()
        n_theta = len(kernel.theta)
        x_train = np.linspace(0, 1, 8)
        y_train = 0.5 * x_train + 0.2
        # an EXTRA nuisance column beyond the kernel theta -> ambiguous trailing slice
        chain = np.zeros((10, 2 + n_theta + 1))
        chain[:, 2 : 2 + n_theta] = kernel.theta
        with self.assertRaises(ValueError):
            total_predictive_band(
                self._mean_fn,
                kernel,
                x_train,
                y_train,
                np.linspace(0, 1, 5),
                chain,
                2,
                noise_std=0.05,
                n_draws=5,
                rng=np.random.default_rng(0),
            )

    def test_explicit_theta_cols(self):
        kernel = make_kernel()
        n_theta = len(kernel.theta)
        x_train = np.linspace(0, 1, 8)
        y_train = 0.5 * x_train + 0.2
        # layout: [m, b, log_eps, *kernel_theta]; kernel theta is NOT the only tail
        chain = np.zeros((10, 2 + 1 + n_theta))
        chain[:, 3:] = kernel.theta
        theta_cols = np.arange(3, 3 + n_theta)
        band = total_predictive_band(
            self._mean_fn,
            kernel,
            x_train,
            y_train,
            np.linspace(0, 1, 5),
            chain,
            2,
            theta_cols=theta_cols,
            noise_std=0.05,
            n_draws=5,
            rng=np.random.default_rng(0),
        )
        self.assertEqual(band.shape, (2, 5))
        self.assertTrue(np.all(np.isfinite(band)))


class TestDiagonalMeanVar(unittest.TestCase):
    def test_matches_full_covariance_diagonal(self):
        rng = np.random.default_rng(4)
        kernel = make_kernel()
        X_train = np.sort(rng.uniform(-2, 2, 10))
        residuals = np.sin(X_train)
        X_pred = np.linspace(-2.5, 2.5, 20)
        mean_full, cov_full = gp_posterior_predictive(
            kernel, kernel.theta, X_train, residuals, X_pred, train_noise_var=0.01
        )
        mean, var = _gp_posterior_mean_var(
            kernel, kernel.theta, X_train, residuals, X_pred, train_noise_var=0.01
        )
        np.testing.assert_allclose(mean, mean_full, atol=1e-9)
        np.testing.assert_allclose(var, np.diag(cov_full), atol=1e-9)


if __name__ == "__main__":
    unittest.main()
