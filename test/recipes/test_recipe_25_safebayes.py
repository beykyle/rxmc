"""Recipe 25: SafeBayes, learn the tempering exponent.

I suspect my model is misspecified and want the tempering exponent chosen
by the data rather than by hand: minimise the prequential log-loss of the
eta-generalised posterior over prefixes of the data.
"""

from dataclasses import replace

import numpy as np
import pytest
from scipy import stats

from common import TRUE, line, line_data
from rxmc import Comparison, Constraint, Parameter, Problem
from rxmc import terms as T


def test_replace_weight_and_masked_prefixes_keep_the_columns():
    d = line_data()
    log_eps = Parameter("log_eps", prior=stats.norm(-2, 1))
    c = Constraint([Comparison(d, line())], terms=[T.noise(log_eps)])
    order = np.arange(d.n)
    prefix = lambda i: [np.isin(np.arange(d.n), order[:i])]  # noqa: E731
    names = Problem([c]).names
    for eta in (1.0, 0.5, 0.25):
        for i in range(2, d.n):
            p = Problem([replace(c.masked(prefix(i)), weight=eta)])
            assert p.names == names
            theta = np.array([*TRUE, np.log(0.2)])
            assert p.log_likelihood(theta) == pytest.approx(
                eta * Problem([c.masked(prefix(i))]).log_likelihood(theta)
            )


def test_prefix_difference_is_the_conditional_density_of_the_next_point():
    d = line_data()
    log_w = Parameter("log_omega", prior=stats.norm(-2, 1))
    c = Constraint(
        [Comparison(d, line())], terms=[T.offset(log_w)]
    )  # correlated: conditional != marginal
    theta = np.array([*TRUE, np.log(0.3)])
    ym = TRUE[0] * d.x + TRUE[1]
    Sigma = np.diag(d.y_err**2) + 0.09 * np.ones((d.n, d.n))
    i = 4
    before = Problem([c.masked([np.arange(d.n) < i])])
    after = Problem([c.masked([np.arange(d.n) < i + 1])])
    diff = after.log_likelihood(theta) - before.log_likelihood(theta)
    # closed-form Gaussian conditional of point i given points < i
    S_aa, S_ab, S_bb = Sigma[i, i], Sigma[i, :i], Sigma[:i, :i]
    r = d.y[:i] - ym[:i]
    mu_c = ym[i] + S_ab @ np.linalg.solve(S_bb, r)
    var_c = S_aa - S_ab @ np.linalg.solve(S_bb, S_ab)
    assert diff == pytest.approx(stats.norm(mu_c, np.sqrt(var_c)).logpdf(d.y[i]))


def test_weight_is_a_float_by_type():
    with pytest.raises((TypeError, ValueError)):
        Constraint([Comparison(line_data(), line())], weight=Parameter("eta"))
