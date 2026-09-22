"""Recipe 24: per-dataset parameters drawn from a sampled hyperprior.

Each dataset has its own normalisation, and I want to learn how spread out
those normalisations are, rather than fix the spread.
"""

import numpy as np
import pytest
from scipy import stats

from common import TRUE, line, line_data
from rxmc import Comparison, Constraint, Parameter, Problem
from rxmc import transforms as tf


class RhoHierarchy:
    def logpdf(self, v):
        *r, lt = v
        return (
            stats.norm(0, np.exp(lt)).logpdf(r).sum()
            + stats.halfnorm(scale=0.3).logpdf(np.exp(lt))
            + lt
        )

    def prior_transform(self, u):
        lt = np.log(stats.halfnorm(scale=0.3).ppf(u[-1]))
        return np.append(stats.norm(0, np.exp(lt)).ppf(u[:-1]), lt)


def build(priors):
    rhos = [Parameter(f"log_rho_{i}") for i in range(3)]
    log_tau = Parameter("log_tau")
    model = line()
    comps = [
        Comparison(line_data(i, 5, f"d{i}"), model | tf.scale(rho))
        for i, rho in enumerate(rhos)
    ]
    return Problem([Constraint(comps)], priors=priors(rhos, log_tau)), rhos, log_tau


def test_tau_is_a_column_and_the_block_covers_children_and_hyper():
    p, rhos, log_tau = build(lambda rhos, lt: [(rhos + [lt], RhoHierarchy())])
    assert p.names == ["m", "b", "log_rho_0", "log_rho_1", "log_rho_2", "log_tau"]
    theta = np.array([*TRUE, 0.1, -0.1, 0.0, np.log(0.2)])
    expected = stats.norm(0, 5).logpdf(TRUE).sum() + RhoHierarchy().logpdf(
        [0.1, -0.1, 0.0, np.log(0.2)]
    )
    assert p.log_prior(theta) == pytest.approx(expected)


def test_forgetting_the_block_is_a_compile_error():
    with pytest.raises(ValueError, match="'log_rho_0' has no prior"):
        build(lambda rhos, lt: [])


def test_prior_transform_draws_the_hyperparameter_first():
    p, *_ = build(lambda rhos, lt: [(rhos + [lt], RhoHierarchy())])
    u = np.array([0.5, 0.5, 0.9, 0.1, 0.5, 0.5])
    theta = p.prior_transform(u)
    lt = theta[-1]
    assert lt == pytest.approx(np.log(stats.halfnorm(scale=0.3).ppf(0.5)))
    assert theta[2] == pytest.approx(stats.norm(0, np.exp(lt)).ppf(0.9))
    assert p.sample_prior(4, rng=0).shape == (4, 6)
