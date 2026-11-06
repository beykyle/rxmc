"""Recipe 26: unaccounted-for model error per data type (KDUQ).

I am calibrating a global optical potential to many datasets of several
observable types.  I want one fractional "unaccounted-for" uncertainty per
type, sampled with the potential, added in quadrature to the reported
errors and scaled with the average of datum and prediction.
"""

import numpy as np
import pytest
from scipy import stats

from common import TRUE, line, line_data
from rxmc import Comparison, Constraint, Parameter, Problem
from rxmc import terms as T


def build():
    types = ("dxs", "ay")
    delta = {t: Parameter(f"delta_{t}", prior=stats.halfnorm(scale=0.2)) for t in types}
    datasets = [line_data(i, 5, f"d{i}", meta={"type": types[i % 2]}) for i in range(4)]
    model = line()
    comps = [Comparison(d, model) for d in datasets]
    terms = [
        T.proportional_error(delta[d.meta["type"]], averaging=True, log=False, on=c)
        for d, c in zip(datasets, comps)
    ]
    return types, delta, datasets, comps, terms


def test_one_delta_per_type_shared_by_object():
    types, delta, datasets, comps, terms = build()
    p = Problem([Constraint(comps, terms=terms)])
    assert p.names == ["m", "b", "delta_dxs", "delta_ay"]
    theta = np.array([*TRUE, 0.1, 0.3])
    S = p.constraints[0].matrix(theta)
    for i, d in enumerate(datasets):
        ym = TRUE[0] * d.x + TRUE[1]
        dT = 0.1 if d.meta["type"] == "dxs" else 0.3
        expected = d.y_err**2 + (dT * 0.5 * (d.y + ym)) ** 2
        np.testing.assert_allclose(np.diag(S)[5 * i : 5 * i + 5], expected)


def test_democratic_and_federal_scalings_are_tempering_weights():
    types, delta, datasets, comps, terms = build()
    theta = np.array([*TRUE, 0.1, 0.3])
    n_params, n_data = 2, 20
    plain = Problem([Constraint(comps, terms=terms)])
    dem = Problem([Constraint(comps, terms=terms, weight=n_params / n_data)])
    assert dem.log_likelihood(theta) == pytest.approx(
        n_params / n_data * plain.log_likelihood(theta)
    )
    fed = Problem(
        [
            Constraint(
                [c for c, d in zip(comps, datasets) if d.meta["type"] == t],
                terms=[tt for tt, d in zip(terms, datasets) if d.meta["type"] == t],
                weight=n_params / (len(types) * 10),
            )
            for t in types
        ]
    )
    assert fed.names == plain.names  # each delta lives in its own type constraint
    assert np.isfinite(fed.log_likelihood(theta))
