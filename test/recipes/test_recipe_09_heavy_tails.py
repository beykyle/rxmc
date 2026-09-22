"""Recipe 9: heavy tails.

A few points are gross outliers.  I do not want them to drag the fit; I want
a likelihood that widens instead of breaking.
"""

import numpy as np
import pytest
from scipy.special import gammaln

from common import TRUE, line, line_data
from helpers import mahalanobis
from rxmc import Comparison, Constraint, Dataset, Parameter, Problem
from rxmc.likelihood import Chi2, StudentT


def test_student_t_parameter_and_closed_form():
    d = line_data()
    nu = Parameter("nu", bounds=(1.0, 100.0))
    p = Problem([Constraint([Comparison(d, line())], likelihood=StudentT(nu=nu))])
    assert p.names == ["m", "b", "nu"] and p.bounds[2].tolist() == [1.0, 100.0]
    theta = np.array([*TRUE, 5.0])
    ym = TRUE[0] * d.x + TRUE[1]
    d2, logdet = mahalanobis(d.y, ym, np.diag(d.y_err**2))
    n, v = 8, 5.0
    expected = (
        gammaln((n + v) / 2)
        - gammaln(v / 2)
        - 0.5 * n * np.log(np.pi * v)
        - 0.5 * logdet
        - 0.5 * (v + n) * np.log1p(d2 / v)
    )
    assert p.log_likelihood(theta) == pytest.approx(expected)
    assert p.chi2(theta) == pytest.approx(d2)  # the distance ignores nu


def test_same_covariance_different_functional():
    d = line_data()
    gauss = Problem([Constraint([Comparison(d, line())])])
    chi2 = Problem([Constraint([Comparison(d, line())], likelihood=Chi2())])
    theta = np.array(TRUE)
    assert chi2.log_likelihood(theta) == pytest.approx(-0.5 * gauss.chi2(theta))
    assert StudentT().params[0].name == "nu"


def test_student_t_downweights_an_outlier():
    d = line_data()
    y = d.y.copy()
    y[3] += 3.0  # a gross outlier
    bad = Dataset(d.x, y, d.y_err)
    nu = Parameter("nu", bounds=(1.0, 100.0))
    theta_true = np.array([*TRUE, 2.0])
    theta_pulled = np.array([TRUE[0], TRUE[1] + 0.4, 2.0])
    t = Problem([Constraint([Comparison(bad, line())], likelihood=StudentT(nu=nu))])
    g = Problem([Constraint([Comparison(bad, line())])])
    # the Gaussian prefers moving toward the outlier more strongly than the t does
    gain_g = g.log_likelihood(theta_pulled[:2]) - g.log_likelihood(theta_true[:2])
    gain_t = t.log_likelihood(theta_pulled) - t.log_likelihood(theta_true)
    assert gain_g > gain_t
