"""Recipe 2: infer an unknown noise level.

My data have no usable error bars, or I do not trust them.  I want to infer
the noise magnitude alongside the model.
"""

import numpy as np
import pytest
from scipy import stats

from common import TRUE, line, line_data
from rxmc import Comparison, Constraint, Parameter, Problem
from rxmc import terms as T


def test_statistical_false_replaces_and_default_adds():
    d = line_data()
    log_eps = Parameter("log_eps", prior=stats.norm(-2, 2))
    theta = np.array([*TRUE, np.log(0.3)])
    replaced = Problem(
        [
            Constraint(
                [Comparison(d, line())], terms=[T.noise(log_eps)], statistical=False
            )
        ]
    )
    added = Problem([Constraint([Comparison(d, line())], terms=[T.noise(log_eps)])])
    np.testing.assert_allclose(np.diag(replaced.constraints[0].matrix(theta)), 0.3**2)
    np.testing.assert_allclose(
        np.diag(added.constraints[0].matrix(theta)), d.y_err**2 + 0.3**2
    )


def test_prediction_scaled_noise_changes_with_the_model_parameters():
    d = line_data()
    log_eps = Parameter("log_eps", prior=stats.norm(-2, 2))
    for term in (T.noise_fraction(log_eps), T.model_error(log_eps)):
        p = Problem(
            [Constraint([Comparison(d, line())], terms=[term], statistical=False)]
        )
        S1 = p.constraints[0].matrix(np.array([*TRUE, -1.0]))
        S2 = p.constraints[0].matrix(np.array([TRUE[0] * 2, TRUE[1], -1.0]))
        assert not np.allclose(S1, S2)


def test_noise_growing_along_x():
    d = line_data()
    log_eps, slope = Parameter("log_eps", prior=stats.norm(-2, 2)), Parameter(
        "slope", prior=stats.norm(0, 1)
    )
    t = T.noise(log_eps, basis=T.exp_growth(np.pi), basis_params=(slope,))
    p = Problem([Constraint([Comparison(d, line())], terms=[t], statistical=False)])
    assert p.names == ["m", "b", "log_eps", "slope"]
    S = p.constraints[0].matrix(np.array([*TRUE, np.log(0.2), 1.0]))
    np.testing.assert_allclose(np.diag(S), (0.2 * np.exp(d.x / np.pi)) ** 2)
    assert p.log_posterior(np.array([*TRUE, np.log(0.2), 1.0])) == pytest.approx(
        p.log_prior([*TRUE, np.log(0.2), 1.0])
        + p.log_likelihood([*TRUE, np.log(0.2), 1.0])
    )
