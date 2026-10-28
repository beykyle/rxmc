"""Gradients through ``Problem.log_posterior``: JAX autodiff against central
finite differences, one problem per family of the error model.

The families follow the recipes: reported errors (01), unknown noise (02),
a free normalisation and offset (04), a latent scale (06), a GP discrepancy
(07), heavy tails (09), log space (10), a hyperprior (24), a model error
proportional to the prediction (26), and the two jitr reaction models.
"""

import jax
import jax.numpy as jnp
import jax.scipy.stats as jst
import jitr
import numpy as np
import pytest
import tinygp
from jitr.optical_potentials.potential_forms import (
    coulomb_charged_sphere,
    thomas_safe,
    woods_saxon_safe,
)
from scipy import stats

from rxmc import (
    Comparison,
    Constraint,
    Dataset,
    Model,
    Parameter,
    Problem,
    StudentT,
)
from rxmc import terms as T
from rxmc import transforms as tf
from rxmc.reactions import ElasticXS, IsobaricAnalogPN

X = np.linspace(0.5, 2.5, 10)


def line():
    m = Parameter("m", prior=stats.norm(0, 5))
    b = Parameter("b", prior=stats.norm(0, 5))
    return Model(lambda x, m, b: m * x + b, [m, b])


def data(seed=0, n=10, label="d", offset=0.0):
    rng = np.random.default_rng(seed)
    x = X[:n]
    y = 2.0 * x + 1.0 + offset + rng.normal(0, 0.1, n)
    return Dataset(x, y, 0.1 * np.ones(n), label=label)


def fd_grad(f, theta, h=1e-6):
    g = np.empty_like(theta)
    for i in range(len(theta)):
        e = np.zeros_like(theta)
        e[i] = h * max(1.0, abs(theta[i]))
        g[i] = (f(theta + e) - f(theta - e)) / (2 * e[i])
    return g


def reported():
    return Problem([Constraint([Comparison(data(), line())])]), [2.1, 0.9]


def unknown_noise():
    log_eps = Parameter("log_eps", prior=stats.norm(-2, 1))
    c = Constraint([Comparison(data(), line())], terms=[T.noise(log_eps)])
    return Problem([c]), [2.1, 0.9, -1.5]


def normalisation_and_offset():
    eta = Parameter("log_eta", prior=stats.norm(-3, 1))
    om = Parameter("log_om", prior=stats.norm(-3, 1))
    c = Constraint(
        [Comparison(data(), line())],
        terms=[T.normalization(eta), T.offset(om), T.systematic(eta, T.x_basis())],
    )
    return Problem([c]), [2.1, 0.9, -2.5, -2.0]


def latent_scale():
    rho = Parameter("log_rho", prior=stats.norm(0, 0.1))
    c = Constraint([Comparison(data(), line() | tf.scale(rho))])
    return Problem([c]), [2.1, 0.9, 0.05]


def gp_discrepancy():
    params = T.kernel_params(["log_var", "log_ell"], [(-12.0, 2.0), (-3.0, 1.0)])
    gp = T.kernel(
        lambda v, e: jnp.exp(v) * tinygp.kernels.Matern52(scale=jnp.exp(e)), params
    )
    c = Constraint([Comparison(data(), line())], terms=[gp])
    return Problem([c]), [2.1, 0.9, -3.0, -0.5]


def heavy_tails():
    nu = Parameter("nu", prior=stats.gamma(a=2.0, scale=10.0), bounds=(1.0, np.inf))
    c = Constraint([Comparison(data(), line())], likelihood=StudentT(nu))
    return Problem([c]), [2.1, 0.9, 7.0]


def log_space():
    log_eps = Parameter("log_eps", prior=stats.norm(-2, 1))
    c = Constraint(
        [Comparison(data(), line(), space=tf.log)],
        terms=[T.noise(log_eps)],
    )
    return Problem([c]), [2.1, 0.9, -2.0]


class Hierarchy:
    """``log_rho_i ~ N(0, tau)`` with a half-normal hyperprior on ``tau``."""

    def logpdf(self, v):
        *r, lt = v
        tau = jnp.exp(lt)
        halfnorm = jnp.log(2.0) + jst.norm.logpdf(tau, scale=0.3)
        return jst.norm.logpdf(jnp.stack(r), scale=tau).sum() + halfnorm + lt


def hyperprior():
    rhos = [Parameter(f"log_rho_{i}") for i in range(3)]
    log_tau = Parameter("log_tau")
    model = line()
    comps = [
        Comparison(data(i, 5, f"d{i}"), model | tf.scale(rho))
        for i, rho in enumerate(rhos)
    ]
    p = Problem([Constraint(comps)], priors=[(rhos + [log_tau], Hierarchy())])
    return p, [2.1, 0.9, 0.02, -0.03, 0.01, np.log(0.1)]


def proportional_model_error():
    log_c = Parameter("log_c", prior=stats.norm(-3, 1))
    c = Constraint(
        [Comparison(data(), line())],
        terms=[T.proportional_error(log_c, averaging=True)],
    )
    return Problem([c]), [2.1, 0.9, -2.5]


A, Z = 40, 20
R = 1.2 * A ** (1 / 3)
MSO = 1.0 / jitr.utils.constants.WAVENUMBER_PION


def central(r, Vv, Wv, Rv, av):
    return -(Vv + 1j * Wv) * woods_saxon_safe(r, Rv, av)


def spin_orbit(r, Vso, Rso, aso):
    return Vso * MSO**2 * thomas_safe(r, Rso, aso)


def omp_params():
    return [
        Parameter("Vv", bounds=(40.0, 56.0)),
        Parameter("Wv", bounds=(1.0, 8.0)),
        Parameter("Rv", bounds=(0.9 * R, 1.1 * R)),
        Parameter("av", bounds=(0.5, 0.8)),
    ]


def elastic():
    rxn = jitr.reactions.ElasticReaction(target=(A, Z), projectile=(1, 0))
    model = ElasticXS(
        "dXS/dA",
        central,
        spin_orbit,
        lambda ws, *x: (tuple(x), (6.0, R, 0.45)),
        omp_params(),
        lmax=10,
    )
    meta = {"reaction": rxn, "Elab": 14.1}
    angles = np.linspace(0.3, 2.6, 8)
    y = np.asarray(model.bind(angles, meta)(48.0, 3.5, R, 0.7))
    d = Dataset(angles, 1.05 * y, 0.05 * y, label="n", meta=meta)
    log_c = Parameter("log_c", bounds=(-6.0, 0.0))
    comp = Comparison(d, model, space=tf.log)
    c = Constraint([comp], terms=[T.noise(log_c)])
    return Problem([c]), [47.0, 3.8, 1.02 * R, 0.68, -3.0]


def ias():
    A, Z = 48, 20
    R = 1.2 * A ** (1 / 3)
    rxn = jitr.reactions.Reaction(
        target=(A, Z), projectile=(1, 1), product=(1, 0), residual=(A, Z + 1)
    )
    model = IsobaricAnalogPN(
        coulomb_charged_sphere,
        central,
        spin_orbit,
        central,
        spin_orbit,
        lambda ws, Vv, Wv, Rv, av: (
            (Z, R),
            (Vv + 4.0, Wv, Rv, av),
            (6.0, R, 0.45),
            (Vv - 4.0, Wv, Rv, av),
            (6.0, R, 0.45),
        ),
        [Parameter(n, prior=stats.norm(v, 5.0)) for n, v in zip("VWRa", (48, 4, R, 1))],
        lmax=10,
    )
    meta = {"reaction": rxn, "Elab": 25.0, "ExIAS": 6.7}
    angles = np.linspace(0.3, 2.6, 6)
    y = np.asarray(model.bind(angles, meta)(48.0, 3.5, R, 0.7))
    d = Dataset(angles, 1.05 * y, 0.05 * y, label="pn", meta=meta)
    return Problem([Constraint([Comparison(d, model)])]), [47.0, 3.8, 1.02 * R, 0.68]


FAMILIES = {
    "reported errors (01)": reported,
    "unknown noise (02)": unknown_noise,
    "normalisation and offset (04)": normalisation_and_offset,
    "latent scale (06)": latent_scale,
    "GP discrepancy (07)": gp_discrepancy,
    "heavy tails (09)": heavy_tails,
    "log space (10)": log_space,
    "hyperprior (24)": hyperprior,
    "proportional model error (26)": proportional_model_error,
    "elastic reaction": elastic,
    "(p,n) reaction": ias,
}


@pytest.mark.parametrize("family", list(FAMILIES))
def test_autodiff_gradient_matches_finite_differences(family):
    problem, theta = FAMILIES[family]()
    theta = np.asarray(theta, dtype=float)
    assert np.isfinite(problem.log_posterior(theta))
    g = problem.grad_log_posterior(theta)
    np.testing.assert_allclose(
        g, fd_grad(problem.log_posterior, theta), rtol=1e-4, atol=1e-5
    )
    # jax.grad of the public method is the same gradient
    np.testing.assert_allclose(jax.grad(problem.log_posterior)(theta), g, rtol=1e-12)


def test_value_and_grad_and_hessian_agree_with_the_scalar_methods():
    problem, theta = gp_discrepancy()
    theta = np.asarray(theta)
    v, g = problem.value_and_grad_log_posterior(theta)
    assert v == pytest.approx(problem.log_posterior(theta))
    np.testing.assert_allclose(g, problem.grad_log_posterior(theta))
    H = problem.hessian_log_posterior(theta)
    np.testing.assert_allclose(H, H.T, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(
        H[0], fd_grad(lambda t: problem.grad_log_posterior(t)[0], theta), rtol=1e-4
    )


def test_batch_is_the_vectorised_scalar_and_outside_the_prior_is_minus_inf():
    problem, theta = latent_scale()
    thetas = problem.sample_prior(16, rng=0)
    np.testing.assert_allclose(
        problem.log_posterior_batch(thetas),
        [problem.log_posterior(t) for t in thetas],
    )
    bounded, _ = elastic()
    out = np.array([60.0, 3.8, R, 0.68, -3.0])  # Vv above its bound
    assert bounded.log_posterior(out) == -np.inf
    # the forward model is not run there, so the gradient is clean
    assert np.all(bounded.grad_log_posterior(out) == 0.0)


def test_the_compiled_densities_are_traced_once():
    problem, theta = unknown_noise()
    for t in problem.sample_prior(5, rng=0):
        problem.log_posterior(t)
        problem.grad_log_posterior(t)
    # one compiled executable per density, whatever theta (and its dtype) is
    problem.log_posterior([2, 1, -2])
    assert problem._log_posterior_j._cache_size() == 1
    assert problem._vg_j._cache_size() == 1


def test_gradient_is_finite_where_the_covariance_is_singular():
    """A rejected theta inside the prior has the prior's gradient alone."""
    s = Parameter("s", prior=stats.norm(0, 5))
    noise = T.Term(lambda c, s: s * jnp.ones(len(c)), [s], kind="diag")
    # no reported errors, so s = 0 leaves a zero diagonal
    diag = Problem(
        [Constraint([Comparison(data(), line())], terms=[noise], statistical=False)]
    )
    params = T.kernel_params(["log_var"], [(-12.0, 2.0)])
    gp = T.kernel(
        lambda v: jnp.exp(v) * tinygp.kernels.ExpSquared(scale=100.0),
        params,
        jitter=0.0,
    )
    # and a smooth kernel alone, without its nugget, is singular on 10 points
    dense = Problem(
        [Constraint([Comparison(data(), line())], terms=[gp], statistical=False)]
    )
    for problem, theta in [(diag, [1.0, 0.0, 0.0]), (dense, [2.1, 0.9, -3.0])]:
        theta = np.asarray(theta)
        assert problem.log_posterior(theta) == -np.inf
        v, g = problem.value_and_grad_log_posterior(theta)
        assert v == -np.inf and np.all(np.isfinite(g))
        np.testing.assert_allclose(g, jax.grad(problem.log_prior)(theta))
        assert np.all(np.isfinite(problem.hessian_log_posterior(theta)))


def test_bounded_custom_joint_has_a_finite_gradient_out_of_bounds():
    class LogNormalPair:
        def logpdf(self, v):
            return jnp.sum(jst.norm.logpdf(jnp.log(v)) - jnp.log(v))

    m = Parameter("m", bounds=(0.0, np.inf))
    b = Parameter("b", bounds=(0.0, np.inf))
    model = Model(lambda x, m, b: m * x + b, [m, b])
    p = Problem(
        [Constraint([Comparison(data(), model)])], priors=[([m, b], LogNormalPair())]
    )
    out = np.array([-1.0, 1.0])
    assert p.log_prior(out) == -np.inf
    assert np.all(jax.grad(p.log_prior)(out) == 0.0)
    assert np.all(p.grad_log_posterior(out) == 0.0)
    assert np.isfinite(p.log_posterior([2.0, 1.0]))
