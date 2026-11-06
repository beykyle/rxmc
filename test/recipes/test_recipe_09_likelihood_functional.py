"""Recipe 9: choose the likelihood functional.

I want the same covariance under a different functional of the residual: the
Gaussian (the default), or a pure chi-squared that drops the log-determinant.
"""

import numpy as np
import pytest

from common import TRUE, line, line_data
from rxmc import Comparison, Constraint, Problem
from rxmc.likelihood import Chi2


def test_same_covariance_different_functional():
    d = line_data()
    gauss = Problem([Constraint([Comparison(d, line())])])
    chi2 = Problem([Constraint([Comparison(d, line())], likelihood=Chi2())])
    theta = np.array(TRUE)
    assert chi2.log_likelihood(theta) == pytest.approx(-0.5 * gauss.chi2(theta))
    # the functional adds no parameters
    assert chi2.names == gauss.names == ["m", "b"]
