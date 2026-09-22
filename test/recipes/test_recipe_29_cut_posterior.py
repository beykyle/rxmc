"""Recipe 29: cut (modular) posterior by multiple imputation.

One module of my model, say a systematic-error parameter or a GP
hyperparameter, should be learned from its own data only and not be
contaminated by the primary data, which I trust less.
"""

import functools

import numpy as np
import pytest
from scipy import stats

from common import TRUE, line, line_data
from rxmc import Comparison, Constraint, Model, Parameter, Problem


def test_stage_two_fixes_the_module_by_closure():
    # stage 1: phi (a background level) from its own data
    phi = Parameter("phi", prior=stats.norm(0, 1))
    aux = Model(lambda x, phi: phi * np.ones_like(x), [phi])
    d_aux = line_data(3, 4, "aux")
    stage1 = Problem([Constraint([Comparison(d_aux, aux)])])
    assert stage1.names == ["phi"]
    phis = stage1.sample_prior(3, rng=0)[:, 0]

    def f(x, m, b, phi):
        return m * x + b + phi

    m, b = line().params
    d = line_data()
    for value in phis:
        model_t = Model(functools.partial(f, phi=value), [m, b])
        stage2 = Problem([Constraint([Comparison(d, model_t)])])
        assert stage2.names == ["m", "b"]  # phi is not a column of stage 2
        np.testing.assert_allclose(
            stage2.predict(np.array(TRUE))[0][0], TRUE[0] * d.x + TRUE[1] + value
        )


def test_power_weighted_modules_are_one_problem_with_two_weights():
    phi = Parameter("phi", prior=stats.norm(0, 1))
    m, b = line().params
    full = Model(lambda x, m, b, phi: m * x + b + phi, [m, b, phi])
    aux = Model(lambda x, phi: phi * np.ones_like(x), [phi])
    c_aux = Constraint([Comparison(line_data(3, 4, "aux"), aux)], weight=1.0)
    c_pri = Constraint([Comparison(line_data(), full)], weight=0.3)
    p = Problem([c_aux, c_pri])
    assert p.names == ["phi", "m", "b"]
    theta = np.array([0.1, *TRUE])
    assert p.log_likelihood(theta) == pytest.approx(
        1.0 * p.constraints[0].log_likelihood(theta)
        + 0.3 * p.constraints[1].log_likelihood(theta)
    )
