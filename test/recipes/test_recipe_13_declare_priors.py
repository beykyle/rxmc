"""Recipe 13: declare priors.

I want each nuisance parameter to carry its own prior, the optical potential
to have a correlated multivariate normal prior, and everything to work with
nested sampling.
"""

import numpy as np
import pytest
from scipy import stats

from common import line_data
from rxmc import Comparison, Constraint, Model, Parameter, Problem
from rxmc import terms as T


def make(m_kw, b_kw, priors=None, terms=()):
    m, b = Parameter("m", **m_kw), Parameter("b", **b_kw)
    model = Model(lambda x, m, b: m * x + b, [m, b])
    c = Constraint([Comparison(line_data(), model)], terms=list(terms))
    return Problem([c], priors=priors(model) if priors else ()), model


def test_marginal_uniform_and_truncated_marginal():
    p, _ = make(
        dict(prior=stats.norm(2, 1), bounds=(0.0, 4.0)), dict(bounds=(0.0, 3.0))
    )
    tn = stats.truncnorm(-2, 2, loc=2, scale=1)
    assert p.log_prior([2.5, 1.0]) == pytest.approx(tn.logpdf(2.5) - np.log(3.0))
    assert p.log_prior([2.5, 3.5]) == -np.inf
    assert np.all(np.isfinite(p.prior_transform([0.0, 1.0])))
    assert np.all(np.isfinite(p.prior_transform([1.0, 0.0])))


def test_joint_multivariate_normal_over_the_model():
    mu, cov = np.array([2.0, 1.0]), np.array([[0.4, 0.1], [0.1, 0.2]])
    p, model = make(
        {}, {}, priors=lambda mdl: [(mdl.params, stats.multivariate_normal(mu, cov))]
    )
    assert p.log_prior([2.0, 1.0]) == pytest.approx(
        stats.multivariate_normal(mu, cov).logpdf([2.0, 1.0])
    )
    theta = p.prior_transform([0.5, 0.5])
    np.testing.assert_allclose(theta, mu)  # the median of the whitening map is the mean
    assert p.sample_prior(3, rng=0).shape == (3, 2)


def test_every_slot_is_covered_exactly_once():
    with pytest.raises(ValueError, match="'b' has no prior"):
        make(dict(prior=stats.norm()), {})
    with pytest.raises(ValueError, match="'m' has its own prior"):
        make(
            dict(prior=stats.norm()),
            {},
            priors=lambda mdl: [(mdl.params, stats.multivariate_normal(np.zeros(2)))],
        )


def test_nuisance_parameters_carry_their_own_priors():
    log_eps = Parameter("log_eps", prior=stats.halfnorm(scale=1))
    p, _ = make(
        dict(prior=stats.norm(0, 5)),
        dict(prior=stats.norm(0, 5)),
        terms=[T.noise(log_eps)],
    )
    assert p.names[-1] == "log_eps"
    assert p.log_prior([1.0, 1.0, -0.5]) == -np.inf  # halfnorm support
    assert np.isfinite(p.log_prior([1.0, 1.0, 0.5]))
