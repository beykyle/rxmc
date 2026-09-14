"""Recipe 38: the classic normal hierarchical model (eight schools).

Several groups each report an estimate y_j with a known standard error
sigma_j.  I believe the group effects theta_j are drawn from a common
distribution N(mu, tau^2) and want to learn mu, tau, and the shrunken
theta_j.
"""

import numpy as np
import pytest
from scipy import stats

from rxmc import Comparison, Constraint, Dataset, Model, Parameter, Problem
from rxmc import terms as T

Y = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
S = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])
J = len(Y)
SCHOOLS = Dataset(np.arange(J), Y, S, label="schools")


def marginalised():
    mu, log_tau = Parameter("mu", prior=stats.norm(0, 25)), Parameter(
        "log_tau", prior=stats.norm(1, 1)
    )
    model = Model(lambda x, mu: np.full(len(x), mu), [mu])
    return (
        Problem([Constraint([Comparison(SCHOOLS, model)], terms=[T.noise(log_tau)])]),
        mu,
        log_tau,
    )


def non_centred(mu, log_tau):
    etas = [Parameter(f"eta_{j}", prior=stats.norm(0, 1)) for j in range(J)]
    school = Model(
        lambda x, mu, lt, *eta: mu + np.exp(lt) * np.asarray(eta), [mu, log_tau, *etas]
    )
    return Problem([Constraint([Comparison(SCHOOLS, school)])]), etas


def test_marginalised_form_has_two_columns_and_the_right_covariance():
    p, mu, log_tau = marginalised()
    assert p.names == ["mu", "log_tau"]
    theta = np.array([5.0, np.log(4.0)])
    np.testing.assert_allclose(np.diag(p.constraints[0].matrix(theta)), S**2 + 16.0)


def test_non_centred_form_marginalises_to_the_same_likelihood():
    pm, mu, log_tau = marginalised()
    pn, etas = non_centred(mu, log_tau)
    assert pn.names == ["mu", "log_tau", *[f"eta_{j}" for j in range(J)]]
    assert np.all(
        np.isfinite(pn.prior_transform(np.full(J + 2, 0.5)))
    )  # marginals only
    # integrate the etas out numerically for one school and compare to the
    # marginalised likelihood of that school: N(y_j | mu, s_j^2 + tau^2)
    mu_v, tau = 5.0, 4.0
    eta = np.linspace(-6, 6, 4001)
    j = 0
    integrand = stats.norm(mu_v + tau * eta, S[j]).pdf(Y[j]) * stats.norm(0, 1).pdf(eta)
    marg = np.log(np.trapezoid(integrand, eta))
    assert marg == pytest.approx(
        stats.norm(mu_v, np.hypot(S[j], tau)).logpdf(Y[j]), abs=1e-6
    )


def test_centred_form_is_a_joint_block_including_the_hyperparameters():
    class SchoolHierarchy:
        def logpdf(self, v):
            *th, mu, lt = v
            return (
                stats.norm(mu, np.exp(lt)).logpdf(th).sum()
                + stats.norm(0, 25).logpdf(mu)
                + stats.norm(1, 1).logpdf(lt)
            )

    thetas = [Parameter(f"theta_{j}") for j in range(J)]
    mu, log_tau = Parameter("mu"), Parameter("log_tau")
    model = Model(lambda x, *th: np.asarray(th), thetas)
    p = Problem(
        [Constraint([Comparison(SCHOOLS, model)])],
        priors=[(thetas + [mu, log_tau], SchoolHierarchy())],
    )
    assert p.names == [*[f"theta_{j}" for j in range(J)], "mu", "log_tau"]
    theta = np.array([*Y, 5.0, np.log(4.0)])
    assert np.isfinite(p.log_posterior(theta))
    with pytest.raises(NotImplementedError):
        p.prior_transform(
            np.full(J + 2, 0.5)
        )  # no unit-cube map for a custom joint without one


def test_a_held_out_school_keeps_its_eta():
    pm, mu, log_tau = marginalised()
    pn, etas = non_centred(mu, log_tau)
    c = pn.constraints[0].source
    fit = c.masked([np.arange(J) < J - 1])
    pf = Problem([fit])
    assert pf.names == pn.names
    s = pf.sample_prior(20, rng=0)
    assert np.std(s[:, -1]) > 0.5  # eta of the held-out school is drawn from its prior
