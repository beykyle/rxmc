"""Recipe 23: a discrepancy correlated across energies and angles.

I believe the model's defect varies smoothly in both energy and angle.  I
want one GP over (E, theta) that correlates the datasets at different
energies.
"""

import numpy as np
import pytest
from scipy import stats
from sklearn.gaussian_process.kernels import RBF, Matern

from common import TRUE, line
from helpers import manual_mvn_loglike
from rxmc import Comparison, Constraint, Dataset, Parameter, Problem, Term


def test_one_matrix_term_spanning_the_comparisons():
    x = np.linspace(0.5, 2.5, 4)
    ds = [
        Dataset(
            x, TRUE[0] * x + TRUE[1], 0.1 * np.ones(4), label=f"E{E}", meta={"Elab": E}
        )
        for E in (10.0, 20.0)
    ]
    model = line()
    comps = [Comparison(d, model) for d in ds]
    kE, kt = RBF(10.0), Matern(0.3, nu=2.5)
    lE, lt, log_A = (
        Parameter(n, prior=stats.norm(0, 1)) for n in ("log_lE", "log_ltheta", "log_A")
    )

    def fn(c, lE, lt, lA):
        E, t = c.meta("Elab")[:, None], c.x[:, None]
        K = kE.clone_with_theta([lE])(E) * kt.clone_with_theta([lt])(t)
        return np.exp(2 * lA) * K + 1e-10 * np.eye(len(c))

    md = Term(fn, (lE, lt, log_A), kind="matrix", on=comps)
    p = Problem([Constraint(comps, terms=[md])])
    assert p.names == ["m", "b", "log_lE", "log_ltheta", "log_A"]
    assert p.constraints[0].covariance.dense  # a matrix across comparisons is dense
    theta = np.array([*TRUE, np.log(10.0), np.log(0.3), np.log(0.2)])
    E = np.repeat([10.0, 20.0], 4)[:, None]
    t = np.tile(x, 2)[:, None]
    ref = (
        np.diag(np.full(8, 0.01))
        + 0.04 * RBF(10.0)(E) * Matern(0.3, nu=2.5)(t)
        + 1e-10 * np.eye(8)
    )
    y, ym = np.concatenate([d.y for d in ds]), np.tile(TRUE[0] * x + TRUE[1], 2)
    assert p.log_likelihood(theta) == pytest.approx(manual_mvn_loglike(y, ym, ref))
    S = p.constraints[0].matrix(theta)
    assert np.any(S[:4, 4:] != 0.0)  # the energies really are correlated
