"""Recipe 5: share an error model between datasets, or couple them.

Two datasets.  Case B: each has its own independent normalisation
measurement, but I believe the two magnitudes are the same.  Case A: both
were normalised against the same uncertain flux, so their errors are
correlated.
"""

import numpy as np
import pytest
from scipy import stats

from common import TRUE, line, line_data
from rxmc import Comparison, Constraint, Parameter, Problem
from rxmc import terms as T


@pytest.fixture
def two():
    model = line()
    c1 = Comparison(line_data(0, 5, "d1"), model)
    c2 = Comparison(line_data(1, 6, "d2"), model)
    return c1, c2


def test_case_b_two_spellings_agree_and_stay_block_diagonal(two):
    c1, c2 = two
    log_eta = Parameter("log_eta", prior=stats.norm(-3, 1))
    one = Problem(
        [
            Constraint(
                [c1, c2],
                terms=[
                    T.normalization(log_eta, on=c1),
                    T.normalization(log_eta, on=c2),
                ],
            )
        ]
    )
    split = Problem(
        [
            Constraint([c1], terms=[T.normalization(log_eta)]),
            Constraint([c2], terms=[T.normalization(log_eta)]),
        ]
    )
    assert one.ndim == split.ndim == 3
    theta = np.array([*TRUE, np.log(0.05)])
    assert one.log_likelihood(theta) == pytest.approx(split.log_likelihood(theta))
    S = one.constraints[0].matrix(theta)
    assert np.all(S[:5, 5:] == 0.0)


def test_case_a_couples_the_blocks_and_differs_from_case_b(two):
    c1, c2 = two
    log_eta = Parameter("log_eta", prior=stats.norm(-3, 1))
    a = Problem([Constraint([c1, c2], terms=[T.normalization(log_eta, on=[c1, c2])])])
    b = Problem(
        [
            Constraint(
                [c1, c2],
                terms=[
                    T.normalization(log_eta, on=c1),
                    T.normalization(log_eta, on=c2),
                ],
            )
        ]
    )
    assert a.ndim == b.ndim
    theta = np.array([*TRUE, np.log(0.05)])
    S = a.constraints[0].matrix(theta)
    assert np.any(S[:5, 5:] != 0.0)
    assert a.log_likelihood(theta) != pytest.approx(b.log_likelihood(theta))
    assert not a.constraints[
        0
    ].covariance.dense  # a mode across blocks stays structured


def test_sharing_is_by_object_not_by_name(two):
    c1, c2 = two
    e1, e2 = Parameter("log_eta", prior=stats.norm()), Parameter(
        "log_eta", prior=stats.norm()
    )
    with pytest.raises(ValueError, match="duplicate parameter name"):
        Problem(
            [
                Constraint(
                    [c1, c2],
                    terms=[T.normalization(e1, on=c1), T.normalization(e2, on=c2)],
                )
            ]
        )
