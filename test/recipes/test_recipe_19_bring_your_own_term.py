"""Recipe 19: bring my own covariance or term.

I have a full covariance matrix from a correlated measurement, or a noise
model no helper expresses.
"""

import numpy as np
import pytest
from scipy import stats

from common import TRUE, line, line_data
from helpers import manual_mvn_loglike
from rxmc import Comparison, Constraint, Parameter, Problem, Term
from rxmc import terms as T


def test_fixed_matrix_and_fixed_diagonal_inside_a_problem():
    d = line_data()
    C = 0.01 * np.exp(-np.abs(d.x[:, None] - d.x[None, :]) / 0.5)
    comp = Comparison(d, line())
    p = Problem([Constraint([comp], terms=[Term(C, on=comp)], statistical=False)])
    theta = np.array(TRUE)
    ym = TRUE[0] * d.x + TRUE[1]
    assert p.log_likelihood(theta) == pytest.approx(manual_mvn_loglike(d.y, ym, C))
    sig = 0.2 * np.ones(d.n)
    p2 = Problem(
        [Constraint([comp], terms=[Term(sig, kind="diag", on=comp)], statistical=False)]
    )
    assert p2.log_likelihood(theta) == pytest.approx(
        manual_mvn_loglike(d.y, ym, np.diag(sig**2))
    )


def test_custom_callable_term_matches_the_helper():
    d = line_data()
    e, sl = Parameter("log_e", prior=stats.norm(-2, 1)), Parameter(
        "slope", prior=stats.norm(0, 1)
    )
    custom = Term(
        lambda c, e, l: np.exp(e) * np.exp(l * c.x / np.pi), (e, sl), kind="diag"
    )
    helper = T.noise(e, basis=T.exp_growth(np.pi), basis_params=(sl,))
    pc = Problem(
        [Constraint([Comparison(d, line())], terms=[custom], statistical=False)]
    )
    ph = Problem(
        [Constraint([Comparison(d, line())], terms=[helper], statistical=False)]
    )
    theta = np.array([*TRUE, np.log(0.2), 1.1])
    assert pc.names == ph.names
    assert pc.log_likelihood(theta) == pytest.approx(ph.log_likelihood(theta))


def test_wrong_shape_for_the_support_is_rejected_at_construction():
    d = line_data()
    comp = Comparison(d, line())
    with pytest.raises(ValueError, match="expects shape"):
        Constraint([comp], terms=[Term(np.ones(3), kind="diag", on=comp)])
    with pytest.raises(ValueError, match="symmetric"):
        Term(np.array([[1.0, 0.5], [0.0, 1.0]]))
