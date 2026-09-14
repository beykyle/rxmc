"""Recipe 10: compare in log space.

Cross sections span orders of magnitude.  I want the Gaussian to live in log
space, with the model still written in physical units.
"""

import numpy as np
import pytest
from scipy import stats

from common import TRUE, line, line_data
from rxmc import Comparison, Constraint, Dataset, Parameter, Problem
from rxmc import terms as T
from rxmc import transforms as tf


def test_delta_method_and_jacobian():
    d = line_data()
    comp = Comparison(d, line(), space=tf.log)
    np.testing.assert_allclose(comp.y, np.log(d.y))
    np.testing.assert_allclose(comp.y_err, d.y_err / d.y)
    log_eps = Parameter("log_eps", prior=stats.norm(-2, 1))
    p = Problem([Constraint([comp], terms=[T.noise(log_eps)], statistical=False)])
    assert p.log_jacobian() == pytest.approx(-np.sum(np.log(d.y)))
    theta = np.array([*TRUE, np.log(0.1)])
    np.testing.assert_allclose(p.predict(theta)[0][0], np.log(TRUE[0] * d.x + TRUE[1]))


def test_non_positive_prediction_and_data():
    d = line_data()
    p = Problem([Constraint([Comparison(d, line(), space=tf.log)])])
    assert p.log_likelihood([-5.0, 0.0]) == -np.inf and p.chi2([-5.0, 0.0]) == np.inf
    neg = Dataset(d.x, d.y - 3.0, d.y_err, label="neg")  # negative at the first points
    with pytest.raises(ValueError, match="neg"):
        Problem([Constraint([Comparison(neg, line(), space=tf.log)])])


def test_constant_noise_in_log_space_is_fractional_noise_in_linear_space():
    d = line_data()
    log_eps = Parameter("log_eps", prior=stats.norm(-2, 1))
    p = Problem(
        [
            Constraint(
                [Comparison(d, line(), space=tf.log)],
                terms=[T.noise(log_eps)],
                statistical=False,
            )
        ]
    )
    theta = np.array([*TRUE, np.log(0.1)])
    S = p.constraints[0].matrix(theta)
    np.testing.assert_allclose(np.diag(S), 0.01)  # constant in log space
