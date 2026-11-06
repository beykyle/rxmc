"""Recipe 36: discrepancy on a physically constrained basis.

I know the shape the model defect can take, say a few Legendre modes in
angle, and want the discrepancy restricted to that basis.
"""

import numpy as np
from scipy import stats
from scipy.special import eval_legendre

from common import TRUE, line, line_data
from rxmc import Comparison, Constraint, Model, Parameter, Problem
from rxmc import terms as T


def test_marginalised_modes_give_a_low_rank_covariance():
    d = line_data()
    comp = Comparison(d, line())
    modes = [
        T.systematic(
            Parameter(f"log_s{k}", prior=stats.norm(-3, 1)),
            basis=lambda c, k=k: eval_legendre(k, np.cos(c.x)),
            on=comp,
        )
        for k in range(1, 4)
    ]
    p = Problem([Constraint([comp], terms=modes)])
    assert p.names[2:] == ["log_s1", "log_s2", "log_s3"]
    theta = np.array([*TRUE, np.log(0.3), np.log(0.2), np.log(0.1)])
    S = p.constraints[0].matrix(theta) - np.diag(d.y_err**2)
    assert np.linalg.matrix_rank(S, tol=1e-10) == 3
    assert not p.constraints[0].covariance.dense


def test_sampled_form_lists_the_coefficients_after_the_model():
    d = line_data()
    coeffs = [Parameter(f"c{k}", prior=stats.norm(0, 0.1)) for k in range(1, 4)]
    delta = Model(
        lambda x, *cs: sum(
            ck * eval_legendre(k, np.cos(x)) for k, ck in enumerate(cs, 1)
        ),
        coeffs,
    )
    p = Problem([Constraint([Comparison(d, line() + delta)])])
    assert p.names == ["m", "b", "c1", "c2", "c3"]
    theta = np.array([*TRUE, 0.1, 0.0, 0.0])
    np.testing.assert_allclose(
        p.predict(theta)[0][0], TRUE[0] * d.x + TRUE[1] + 0.1 * np.cos(d.x)
    )
