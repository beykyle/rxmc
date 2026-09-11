"""GP conditioning and the total predictive band of a problem."""

import numpy as np
import pytest
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from rxmc import Comparison, Constraint, Dataset, Model, Parameter, Problem, Term
from rxmc.predictive import (
    gp_posterior_predictive,
    predictive_band,
    total_predictive_band,
)
from rxmc.terms import constant_amplitude, kernel, noise
from rxmc.transforms import log


def make_kernel():
    return ConstantKernel(1.0) * RBF(length_scale=0.7) + WhiteKernel(1e-6)


class TestGPPosteriorPredictive:
    def setup_method(self):
        rng = np.random.default_rng(0)
        self.X_train = np.sort(rng.uniform(-2, 2, 12))
        self.residuals = np.sin(self.X_train) + 0.05 * rng.standard_normal(12)
        self.X_pred = np.linspace(-2.5, 2.5, 25)
        self.kernel = make_kernel()
        self.theta = self.kernel.theta

    def test_matches_sklearn_gpr(self):
        gpr = GaussianProcessRegressor(
            kernel=self.kernel.clone_with_theta(self.theta),
            optimizer=None,
            alpha=0.01,
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
            train_noise_var=0.01,
        )
        np.testing.assert_allclose(mean, mean_sk, atol=1e-6)
        np.testing.assert_allclose(cov, cov_sk, atol=1e-6)

    def test_noiseless_interpolation_and_2d_inputs(self):
        k = ConstantKernel(1.0) * RBF(length_scale=0.7)
        mean, cov = gp_posterior_predictive(
            k, k.theta, self.X_train, self.residuals, self.X_train
        )
        # the nugget keeps a near-singular K invertible; interpolation to 1e-2
        np.testing.assert_allclose(mean, self.residuals, atol=1e-2)
        assert np.all(np.diag(cov) < 1e-2)
        X2 = np.column_stack([self.X_train, self.X_train**2])
        mean2, cov2 = gp_posterior_predictive(
            k, k.theta, X2, self.residuals, X2[:5], train_noise_var=0.01
        )
        assert mean2.shape == (5,) and cov2.shape == (5, 5)

    def test_predictive_band_percentiles(self):
        draws = np.tile(np.arange(101.0)[:, None], (1, 3))
        band = predictive_band(draws)
        np.testing.assert_allclose(band, [[16] * 3, [50] * 3, [84] * 3])


# ----------------------------------------------------------------------------
# total_predictive_band on a Problem
# ----------------------------------------------------------------------------


def line():
    m = Parameter("m", prior=stats.norm(0, 10))
    b = Parameter("b", prior=stats.norm(0, 10))
    return Model(lambda x, m, b: m * x + b, [m, b])


X = np.linspace(0.0, 1.0, 10)
Y = 0.5 * X + 0.2 + 0.3 * np.sin(3 * X)  # a smooth defect on a line
X_PRED = np.linspace(-0.2, 1.2, 30)


def problem(terms_first, space=None, err=0.05, fixed=True):
    """A line with a GP (hyperparameters fixed or free) and a noise nuisance."""
    d = Dataset(X, Y, np.full(10, err), label="d")
    model = line()
    comp = Comparison(d, model) if space is None else Comparison(d, model, space=space)
    if fixed:
        gp = kernel(RBF(0.3, "fixed"))
    else:
        gp = kernel(RBF(0.3), params=[Parameter("log_ell", prior=stats.norm(-1, 1))])
    eps = noise(Parameter("log_eps", prior=stats.norm(-3, 1)))
    terms = [eps, gp] if terms_first == "noise" else [gp, eps]
    p = Problem([Constraint([comp], terms=terms)])
    return p, gp, model, comp


class TestTotalPredictiveBand:
    def chain(self, p, n=40, seed=2):
        rng = np.random.default_rng(seed)
        rows = np.zeros((n, p.ndim))
        rows[:, p.names.index("m")] = 0.5 + 0.05 * rng.standard_normal(n)
        rows[:, p.names.index("b")] = 0.2 + 0.05 * rng.standard_normal(n)
        rows[:, p.names.index("log_eps")] = np.log(0.05)
        if "log_ell" in p.names:
            rows[:, p.names.index("log_ell")] = np.log(0.3)
        return rows

    def test_shape_finite_and_columns_from_the_problem(self):
        p1, gp1, model1, _ = problem("noise", fixed=False)
        p2, gp2, model2, _ = problem("kernel", fixed=False)
        assert p1.names != p2.names  # the nuisance and kernel columns swapped
        bands = []
        for p, gp, model in ((p1, gp1, model1), (p2, gp2, model2)):
            band = total_predictive_band(
                p, gp, model.bind(X_PRED, {}), X_PRED, self.chain(p), rng=0
            )
            assert band.shape == (2, 30) and np.all(np.isfinite(band))
            assert np.all(band[1] >= band[0])
            bands.append(band)
        np.testing.assert_allclose(bands[0], bands[1])  # no column arithmetic

    def test_conditioned_band_passes_through_the_data(self):
        p, gp, model, comp = problem("noise", err=1e-4)
        chain = np.tile([0.5, 0.2, np.log(1e-4)], (5, 1))
        band = total_predictive_band(
            p, gp, model.bind(X, {}), X, chain, levels=(50,), rng=1
        )
        np.testing.assert_allclose(band[0], Y, atol=2e-3)

    def test_physical_band_under_log_space(self):
        p, gp, model, comp = problem("noise", space=log)
        # 26 rows put the 16th and 84th percentiles on order statistics, so the
        # monotone exp commutes with the percentile
        chain = self.chain(p, n=26)
        pred = model.bind(X_PRED, {})
        band_log = total_predictive_band(p, gp, pred, X_PRED, chain, rng=3)
        band_phys = total_predictive_band(
            p, gp, pred, X_PRED, chain, rng=3, physical=True
        )
        np.testing.assert_allclose(band_phys, np.exp(band_log))

    def test_amplitude_matches_a_scaled_kernel(self):
        d = Dataset(X, Y, np.full(10, 0.05), label="d")
        A = 0.7
        m1, m2 = line(), line()
        lA = Parameter("log_A", prior=stats.norm(0, 1))
        gp_amp = kernel(
            RBF(0.3, "fixed"), amplitude=constant_amplitude, amplitude_params=(lA,)
        )
        gp_fix = kernel(ConstantKernel(A**2, "fixed") * RBF(0.3, "fixed"))
        p1 = Problem([Constraint([Comparison(d, m1)], terms=[gp_amp])])
        p2 = Problem([Constraint([Comparison(d, m2)], terms=[gp_fix])])
        chain1 = np.tile([0.5, 0.2, np.log(A)], (8, 1))
        chain2 = np.tile([0.5, 0.2], (8, 1))
        b1 = total_predictive_band(
            p1, gp_amp, m1.bind(X_PRED, {}), X_PRED, chain1, rng=4
        )
        b2 = total_predictive_band(
            p2, gp_fix, m2.bind(X_PRED, {}), X_PRED, chain2, rng=4
        )
        np.testing.assert_allclose(b1, b2, atol=1e-8)

    def test_explicit_noise_and_errors(self):
        p, gp, model, comp = problem("noise")
        chain = self.chain(p, n=6)
        pred = model.bind(X_PRED, {})
        band = total_predictive_band(
            p, gp, pred, X_PRED, chain, train_noise_var=0.05**2, noise_std=0.1, rng=5
        )
        assert band.shape == (2, 30)
        with pytest.raises(TypeError, match="KernelTerm"):
            total_predictive_band(
                p, Term(np.ones(10), kind="diag"), pred, X_PRED, chain
            )
        with pytest.raises(ValueError, match="not part of any constraint"):
            total_predictive_band(p, kernel(RBF(1.0, "fixed")), pred, X_PRED, chain)
        with pytest.raises(ValueError, match=r"\(n, 3\)"):
            total_predictive_band(p, gp, pred, X_PRED, chain[:, :2])
