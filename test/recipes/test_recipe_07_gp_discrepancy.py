"""Recipe 7: absorb model deficiency with a Gaussian process.

My model is missing physics.  I want a smooth correlated discrepancy, in
angle or in momentum transfer, learned from the residuals.
"""

import numpy as np
from scipy import stats
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern

from common import TRUE, line, linear_posterior
from rxmc import Comparison, Constraint, Dataset, KernelTerm, Parameter, Problem
from rxmc import terms as T
from rxmc.predictive import total_predictive_band
from rxmc.reactions import momentum_transfer

X = np.linspace(0.3, 2.5, 12)


def defect_data(seed=0, err=0.05):
    """A line with a smooth defect the line cannot follow."""
    rng = np.random.default_rng(seed)
    y = TRUE[0] * X + TRUE[1] + 0.3 * np.sin(3 * X) + rng.normal(0, err, len(X))
    return Dataset(X, y, err * np.ones(len(X)), label="d", meta={"k": 2.7})


def test_one_parameter_per_free_hyperparameter_in_log_theta():
    d = defect_data()
    comp = Comparison(d, line())
    log_A = Parameter("log_A", prior=stats.norm(0, 2))
    gp = T.kernel(
        Matern(1.0, nu=2.5),
        on=comp,
        coords=lambda x: x / np.pi,
        amplitude=T.constant_amplitude,
        amplitude_params=(log_A,),
    )
    assert isinstance(gp, KernelTerm)
    p = Problem([Constraint([comp], terms=[gp])])
    assert p.names == ["m", "b", "discrepancy_length_scale", "log_A"]
    # the derived parameter compiles with a uniform prior over sklearn's bounds
    np.testing.assert_allclose(p.bounds[2], np.log([1e-5, 1e5]))
    theta = np.array([*TRUE, np.log(0.4), np.log(0.2)])
    u = (X / np.pi)[:, None]
    K = 0.04 * Matern(0.4, nu=2.5)(u) + 1e-10 * np.eye(12)
    np.testing.assert_allclose(p.constraints[0].matrix(theta), np.diag(d.y_err**2) + K)


def test_momentum_transfer_coordinates_and_a_running_amplitude():
    d = defect_data()
    comp = Comparison(d, line())
    log_A = Parameter("log_A", prior=stats.norm(0, 2))
    r = Parameter("r", prior=stats.norm(0, 1))
    q = lambda x: momentum_transfer(x, d.meta["k"])  # noqa: E731
    gp_q = T.kernel(
        RBF(1.0),
        on=comp,
        coords=q,
        amplitude=lambda c, lA, r: np.exp(lA) * c.x ** (r / 2),
        amplitude_params=(log_A, r),
    )
    p = Problem([Constraint([comp], terms=[gp_q])])
    assert p.names == ["m", "b", "discrepancy_length_scale", "log_A", "r"]
    theta = np.array([*TRUE, np.log(2.0), np.log(0.3), 1.5])
    qx = 2 * 2.7 * np.sin(X / 2)
    a = 0.3 * qx ** (1.5 / 2)
    K = np.outer(a, a) * RBF(2.0)(qx[:, None]) + 1e-10 * np.eye(12)
    np.testing.assert_allclose(p.constraints[0].matrix(theta), np.diag(d.y_err**2) + K)


def test_the_band_finds_the_kernel_columns_itself():
    d = defect_data()
    model = line()
    comp = Comparison(d, model)
    log_eps = Parameter("log_eps", prior=stats.norm(-3, 1))
    gp = T.kernel(RBF(0.5), on=comp)
    p = Problem([Constraint([comp], terms=[T.noise(log_eps), gp])])
    # a chain in problem.names order with the nuisance column between the
    # model parameters and the kernel hyperparameter
    rng = np.random.default_rng(1)
    chain = np.column_stack(
        [
            TRUE[0] + 0.02 * rng.standard_normal(30),
            TRUE[1] + 0.02 * rng.standard_normal(30),
            np.full(30, np.log(0.05)),
            np.full(30, np.log(0.5)),
        ]
    )
    x_fine = np.linspace(0.0, 3.0, 50)
    band = total_predictive_band(
        p, gp, model.bind(x_fine, d.meta), x_fine, chain, rng=0
    )
    assert band.shape == (2, 50) and np.all(np.isfinite(band))
    assert np.all(band[1] > band[0])
    # inside the data the band is narrow, outside it relaxes to the prior width
    inside = (x_fine > 0.5) & (x_fine < 2.3)
    assert np.median((band[1] - band[0])[inside]) < np.median(
        (band[1] - band[0])[~inside]
    )


def test_the_discrepancy_relaxes_the_model_parameters_toward_the_truth():
    d = defect_data()
    bare = Problem([Constraint([Comparison(d, line())])])
    gp = T.kernel(ConstantKernel(0.3**2, "fixed") * RBF(0.5, "fixed"))
    with_gp = Problem([Constraint([Comparison(d, line())], terms=[gp])])
    # both posteriors are exact (linear model, fixed covariances)
    mean_bare, cov_bare, _ = linear_posterior(bare)
    mean_gp, cov_gp, _ = linear_posterior(with_gp)
    bias_bare = np.abs(mean_bare - TRUE)
    bias_gp = np.abs(mean_gp - TRUE)
    assert np.all(bias_gp < bias_bare)
    # and the bare fit is overconfident: its error bars exclude the truth
    assert np.any(bias_bare > 3 * np.sqrt(np.diag(cov_bare)))
    assert np.all(bias_gp < 3 * np.sqrt(np.diag(cov_gp)))
