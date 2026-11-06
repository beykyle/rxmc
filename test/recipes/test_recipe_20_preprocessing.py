"""Recipe 20: preprocess instead of asking for a feature.

I want to mean-subtract, standardise, or project both data and model onto
principal components of a prior predictive ensemble.
"""

import numpy as np
import pytest
from scipy import stats

from common import TRUE, line, line_data
from helpers import manual_mvn_loglike
from rxmc import Comparison, Constraint, Dataset, Model, Parameter, Problem, Term
from rxmc import terms as T


def test_projection_onto_principal_components():
    d = line_data()
    rng = np.random.default_rng(0)
    ens = np.array(
        [TRUE[0] * d.x + TRUE[1] + rng.normal(0, 0.3, d.n) for _ in range(50)]
    )
    mu = ens.mean(0)
    A = np.linalg.svd(ens - mu, full_matrices=False)[2][:2]  # (k, N)
    native = line().bind(d.x)
    model = line()
    proj = Model(lambda x_pc, *theta: A @ (native(*theta) - mu), model.params)
    d_pc = Dataset(np.arange(2), A @ (d.y - mu), np.zeros(2), label=f"{d.label} PC")
    stat = Term(A @ np.diag(d.y_err**2) @ A.T, on=d_pc)
    log_eps = Parameter("log_eps", prior=stats.norm(-2, 1))
    p = Problem(
        [
            Constraint(
                [Comparison(d_pc, proj)],
                terms=[stat, T.noise(log_eps, on=d_pc)],
                statistical=False,
            )
        ]
    )
    theta = np.array([*TRUE, np.log(0.1)])
    ym_pc = A @ (TRUE[0] * d.x + TRUE[1] - mu)
    S = A @ np.diag(d.y_err**2) @ A.T + 0.01 * np.eye(2)
    assert p.log_likelihood(theta) == pytest.approx(
        manual_mvn_loglike(d_pc.y, ym_pc, S)
    )


def test_mean_subtraction_as_preprocessing_or_as_a_pointwise_space():
    from dataclasses import replace

    from rxmc import transforms as tf

    d = line_data()
    mu = 0.5 * np.ones(d.n)
    shifted = replace(d, y=d.y - mu)
    shift = Model(lambda x: -mu, [])
    p_pre = Problem([Constraint([Comparison(shifted, line() + shift)])])
    centre = tf.Transform(lambda a: a - mu, derivative=np.ones_like)
    p_space = Problem([Constraint([Comparison(d, line(), space=centre)])])
    theta = np.array(TRUE)
    assert p_pre.log_likelihood(theta) == pytest.approx(p_space.log_likelihood(theta))
    assert p_space.log_jacobian() == 0.0  # a shift has unit Jacobian
