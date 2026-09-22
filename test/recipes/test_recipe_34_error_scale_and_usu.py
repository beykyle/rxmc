"""Recipe 34: global error scale factor and unrecognised sources of uncertainty.

Repeated measurements scatter more than their stated errors.  I want a global
scale on the reported errors, or a fully correlated unknown component per
experimental technique.
"""

import numpy as np
from scipy import stats

from common import TRUE, line, line_data
from rxmc import Comparison, Constraint, Parameter, Problem, Term
from rxmc import terms as T


def test_global_scale_multiplies_every_stated_error():
    d = line_data()
    comp = Comparison(d, line())
    log_s = Parameter("log_s", prior=stats.norm(0, 0.5))
    scaled = Term(lambda c, ls: np.exp(ls) * comp.y_err, (log_s,), kind="diag", on=comp)
    p = Problem([Constraint([comp], terms=[scaled], statistical=False)])
    theta = np.array([*TRUE, np.log(1.5)])
    np.testing.assert_allclose(
        np.diag(p.constraints[0].matrix(theta)), 1.5**2 * d.y_err**2
    )


def test_usu_offset_couples_exactly_the_comparisons_of_one_technique():
    model = line()
    techs = ("tof", "tof", "act")
    datasets = [
        line_data(i, 4, f"d{i}", meta={"technique": t}) for i, t in enumerate(techs)
    ]
    comps = [Comparison(d, model) for d in datasets]
    log_delta = {
        t: Parameter(f"log_usu_{t}", prior=stats.norm(-3, 1)) for t in ("tof", "act")
    }
    usu = [
        T.offset(
            log_delta[t],
            on=[c for c, d in zip(comps, datasets) if d.meta["technique"] == t],
        )
        for t in ("tof", "act")
    ]
    p = Problem([Constraint(comps, terms=usu)])
    assert p.names == ["m", "b", "log_usu_tof", "log_usu_act"]
    theta = np.array([*TRUE, np.log(0.2), np.log(0.1)])
    S = p.constraints[0].matrix(theta)
    assert np.allclose(S[:4, 4:8], 0.04)  # the two tof datasets are fully correlated
    assert np.all(S[:8, 8:] == 0.0)  # and uncorrelated with the activation one
    assert not p.constraints[0].covariance.dense
