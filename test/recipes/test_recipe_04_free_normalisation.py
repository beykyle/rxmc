"""Recipe 4: infer a normalisation or offset the experiment did not report.

I suspect an unreported normalisation (or background offset) and want its
magnitude as a nuisance parameter.
"""

import numpy as np
from scipy import stats

from common import TRUE, line, line_data
from rxmc import Comparison, Constraint, Parameter, Problem
from rxmc import terms as T


def test_free_normalisation_adds_one_rank_one_mode():
    d = line_data()
    log_eta = Parameter("log_eta", prior=stats.norm(-3, 1))
    p = Problem(
        [
            Constraint(
                [Comparison(d, line())], terms=[T.normalization(parameter=log_eta)]
            )
        ]
    )
    assert p.names == ["m", "b", "log_eta"]
    theta = np.array([*TRUE, np.log(0.05)])
    ym = TRUE[0] * d.x + TRUE[1]
    np.testing.assert_allclose(
        p.constraints[0].matrix(theta), np.diag(d.y_err**2) + 0.05**2 * np.outer(ym, ym)
    )


def test_free_offset_and_shaped_mode():
    d = line_data()
    log_w = Parameter("log_omega", prior=stats.norm(-3, 1))
    log_s = Parameter("log_s", prior=stats.norm(-3, 1))
    p = Problem(
        [
            Constraint(
                [Comparison(d, line())],
                terms=[
                    T.offset(parameter=log_w),
                    T.systematic(log_s, basis=T.x_basis(np.pi)),
                ],
            )
        ]
    )
    theta = np.array([*TRUE, np.log(0.1), np.log(0.2)])
    u = d.x / np.pi
    ref = np.diag(d.y_err**2) + 0.01 * np.ones((8, 8)) + 0.04 * np.outer(u, u)
    np.testing.assert_allclose(p.constraints[0].matrix(theta), ref)
