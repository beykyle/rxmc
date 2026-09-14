"""Recipe 12: temper the likelihood.

I have many points and worry the posterior is overconfident, or I want a
power posterior.
"""

import numpy as np
import pytest

from common import TRUE, line_problem, map_estimate
from oracle import linear_gaussian


def test_weight_scales_the_likelihood_only():
    p1, _ = line_problem()
    pw, _ = line_problem(weight=0.25)
    theta = np.array(TRUE)
    assert pw.log_likelihood(theta) == pytest.approx(0.25 * p1.log_likelihood(theta))
    assert pw.log_prior(theta) == pytest.approx(p1.log_prior(theta))


def test_tempered_posterior_equals_the_oracle_with_inflated_errors():
    w = 0.25
    pw, d = line_problem(weight=w)
    Xd = np.column_stack([d.x, np.ones_like(d.x)])
    mean, _, _ = linear_gaussian(
        Xd, d.y, np.diag(d.y_err**2) / w, [0.0, 0.0], 25.0 * np.eye(2)
    )
    np.testing.assert_allclose(map_estimate(pw, TRUE), mean, atol=1e-4)
