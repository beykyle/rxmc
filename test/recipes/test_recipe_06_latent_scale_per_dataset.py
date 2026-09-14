"""Recipe 6: one latent scale per dataset.

Each dataset has its own unknown normalisation.  I want a Kennedy-O'Hagan
scale factor on the model prediction, one per dataset, sampled with the
model parameters.
"""

import numpy as np
from scipy import stats

from common import TRUE, line, line_data
from rxmc import Comparison, Constraint, Parameter, Problem
from rxmc import transforms as tf


def test_scales_follow_the_model_parameters_and_scale_the_mean():
    model = line()
    rhos = [Parameter(f"log_rho_{i}", prior=stats.norm(0, 0.1)) for i in range(2)]
    comps = [
        Comparison(line_data(i, 5, f"d{i}"), model | tf.scale(rho))
        for i, rho in enumerate(rhos)
    ]
    p = Problem([Constraint(comps)])
    assert p.names == ["m", "b", "log_rho_0", "log_rho_1"]
    theta = np.array([*TRUE, np.log(1.1), np.log(0.9)])
    pred = p.predict(theta)[0]
    x = comps[0].data.x
    np.testing.assert_allclose(pred[0], 1.1 * (TRUE[0] * x + TRUE[1]))
    np.testing.assert_allclose(pred[1], 0.9 * (TRUE[0] * x + TRUE[1]))


def test_masked_view_keeps_the_same_columns():
    model = line()
    rho = Parameter("log_rho", prior=stats.norm(0, 0.1))
    c = Constraint([Comparison(line_data(), model | tf.scale(rho))])
    assert Problem([c.masked_where(lambda x: x < 1.5)]).names == Problem([c]).names


def test_global_scale_on_every_comparison():
    model = line()
    rho = Parameter("log_rho", prior=stats.norm(0, 0.1))
    scaled = model | tf.scale(rho)
    comps = [Comparison(line_data(i, 5, f"d{i}"), scaled) for i in range(2)]
    p = Problem([Constraint(comps)])
    assert p.names == ["m", "b", "log_rho"]
