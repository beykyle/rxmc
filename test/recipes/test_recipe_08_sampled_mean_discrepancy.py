"""Recipe 8: sample a mean discrepancy explicitly.

Instead of marginalising the discrepancy, I want to sample an additive
correction with a parametric shape.
"""

import numpy as np
from scipy import stats

from common import TRUE, line, line_data
from rxmc import Comparison, Constraint, Model, Parameter, Problem
from rxmc import transforms as tf


def test_additive_correction_lists_its_parameters_after_the_model():
    phi = [Parameter(f"c{i}", prior=stats.norm(0, 1)) for i in range(2)]
    delta = Model(lambda x, c0, c1: c0 + c1 * x**2, phi)
    d = line_data()
    p = Problem([Constraint([Comparison(d, line() + delta)])])
    assert p.names == ["m", "b", "c0", "c1"]
    theta = np.array([*TRUE, 0.5, -0.1])
    np.testing.assert_allclose(
        p.predict(theta)[0][0], TRUE[0] * d.x + TRUE[1] + 0.5 - 0.1 * d.x**2
    )


def test_composition_precedence():
    c0 = Parameter("c0", prior=stats.norm(0, 1))
    rho = Parameter("log_rho", prior=stats.norm(0, 0.1))
    delta = Model(lambda x, c0: c0 * np.ones_like(x), [c0])
    d = line_data()
    base = TRUE[0] * d.x + TRUE[1]
    sum_scaled = Problem(
        [Constraint([Comparison(d, (line() + delta) | tf.scale(rho))])]
    )
    model_scaled = Problem(
        [Constraint([Comparison(d, (line() | tf.scale(rho)) + delta)])]
    )
    np.testing.assert_allclose(
        sum_scaled.predict([*TRUE, 0.5, np.log(2.0)])[0][0], 2.0 * (base + 0.5)
    )
    np.testing.assert_allclose(
        model_scaled.predict([*TRUE, np.log(2.0), 0.5])[0][0], 2.0 * base + 0.5
    )


def test_multiplicative_correction():
    g = Parameter("g", prior=stats.norm(0, 1))
    factor = Model(lambda x, g: np.exp(g * x), [g])
    d = line_data()
    p = Problem([Constraint([Comparison(d, line() * factor)])])
    np.testing.assert_allclose(
        p.predict([*TRUE, 0.3])[0][0], (TRUE[0] * d.x + TRUE[1]) * np.exp(0.3 * d.x)
    )
