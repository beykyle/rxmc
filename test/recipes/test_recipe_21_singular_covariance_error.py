"""Recipe 21: get a useful error, not a LinAlgError.

My measurement reports no statistical error and I forgot to add any term.
"""

import numpy as np
import pytest
from scipy import stats

from common import TRUE, line
from rxmc import Comparison, Constraint, Dataset, Parameter, Problem
from rxmc import terms as T


def test_singular_covariance_is_reported_by_label_with_remedies():
    d = Dataset(
        np.linspace(0.5, 2.5, 6),
        TRUE[0] * np.linspace(0.5, 2.5, 6) + 1.0,
        np.zeros(6),
        label="E1234-002",
    )
    with pytest.raises(
        ValueError, match="E1234-002.*zero statistical error.*reported_terms"
    ):
        Problem([Constraint([Comparison(d, line())])])


def test_the_remedies_work():
    x = np.linspace(0.5, 2.5, 6)
    d = Dataset(
        x,
        TRUE[0] * x + 1.0,
        np.zeros(6),
        norm_err=0.05,
        offset_err=0.1,
        label="E1234-002",
    )
    comp = Comparison(d, line())
    Problem(
        [
            Constraint(
                [comp],
                terms=comp.reported_terms()
                + [T.noise(Parameter("log_eps", prior=stats.norm()))],
            )
        ]
    )
    Problem(
        [
            Constraint(
                [comp],
                terms=[T.statistical(0.1 * np.ones(6), on=comp)],
                statistical=False,
            )
        ]
    )
    # modes alone are rank two and still singular on six points
    with pytest.raises(ValueError, match="singular"):
        Problem([Constraint([comp], terms=comp.reported_terms())])


def test_other_compile_time_errors_are_named():
    x = np.linspace(0.5, 2.5, 6)
    d = Dataset(x, TRUE[0] * x + 1.0, 0.1 * np.ones(6), label="d")
    comp = Comparison(d, line())
    stray = Comparison(Dataset(x, x, 0.1 * np.ones(6), label="stray"), line())
    with pytest.raises(ValueError, match="stray"):
        Constraint([comp], terms=[T.noise(Parameter("e"), on=stray)])
    with pytest.raises(ValueError, match="duplicate parameter name"):
        Problem(
            [
                Constraint(
                    [comp],
                    terms=[
                        T.noise(Parameter("e", prior=stats.norm())),
                        T.noise(Parameter("e", prior=stats.norm())),
                    ],
                )
            ]
        )
    with pytest.raises(ValueError, match="'e' has no prior"):
        Problem([Constraint([comp], terms=[T.noise(Parameter("e"))])])
