"""Recipe 37: correlated systematics between observables of one measurement.

One experiment reports both a cross section and an analysing power, and they
share a normalisation or an angle calibration.
"""

import jitr
import numpy as np
from jitr.optical_potentials.potential_forms import thomas_safe, woods_saxon_safe
from scipy import stats

from rxmc import Comparison, Constraint, Dataset, Parameter, Problem
from rxmc import terms as T
from rxmc.reactions import ElasticXS

MSO = 1.0 / jitr.utils.constants.WAVENUMBER_PION
R = 1.2 * 40 ** (1 / 3)
N_CA = jitr.reactions.ElasticReaction(target=(40, 20), projectile=(1, 0))


def central(r, Vv, Wv, Rv, av):
    return -(Vv + 1j * Wv) * woods_saxon_safe(r, Rv, av)


def spin_orbit(r, Vso, Rso, aso):
    return Vso * MSO**2 * thomas_safe(r, Rso, aso)


def test_two_observables_one_constraint_one_spanning_mode():
    params = [Parameter(n, prior=stats.norm(0, 100)) for n in ("Vv", "Wv", "Rv", "av")]
    pot = (central, spin_orbit, lambda ws, *x: (tuple(x), (6.0, R, 0.45)))
    xs_model = ElasticXS("dXS/dA", *pot, params, lmax=10)
    ay_model = ElasticXS("Ay", *pot, params, lmax=10)
    theta = np.array([48.0, 3.5, R, 0.7])
    x = np.linspace(0.3, 2.5, 4)
    kin = {"reaction": N_CA, "Elab": 14.1}
    m_xs, m_ay = {**kin, "quantity": "dXS/dA"}, {**kin, "quantity": "Ay"}
    d_xs = Dataset(
        x, xs_model.bind(x, kin)(*theta), 0.01 * np.ones(4), label="xs", meta=m_xs
    )
    d_ay = Dataset(
        x, ay_model.bind(x, kin)(*theta), 0.02 * np.ones(4), label="ay", meta=m_ay
    )
    comp_xs, comp_ay = Comparison(d_xs, xs_model), Comparison(d_ay, ay_model)
    log_eta = Parameter("log_eta", prior=stats.norm(-3, 1))
    log_dtheta = Parameter("log_dtheta", prior=stats.norm(-4, 1))

    def dy_dtheta(c):
        # angle-calibration mode: the slope of each prediction in angle.  A
        # spanning term sees the gathered stack, so the finite difference must
        # not straddle the seam: split by the per-point dataset metadata.
        u = np.empty(len(c))
        for q in np.unique(c.meta("quantity")):
            rows = c.meta("quantity") == q
            u[rows] = np.gradient(c.ym[rows], c.x[rows])
        return u

    c = Constraint(
        [comp_xs, comp_ay],
        terms=[
            T.normalization(log_eta, on=comp_xs),  # the ratio observable is unaffected
            T.systematic(log_dtheta, basis=dy_dtheta, on=[comp_xs, comp_ay]),
        ],
    )
    p = Problem([c])
    assert p.names == ["Vv", "Wv", "Rv", "av", "log_eta", "log_dtheta"]
    full = np.array([*theta, np.log(0.05), np.log(0.01)])
    S = p.constraints[0].matrix(full)
    ym_xs, ym_ay = p.predict(full)[0]
    u = 0.01 * np.concatenate([np.gradient(ym_xs, x), np.gradient(ym_ay, x)])
    expected = np.diag(np.concatenate([d_xs.y_err, d_ay.y_err]) ** 2) + np.outer(u, u)
    expected[:4, :4] += 0.05**2 * np.outer(ym_xs, ym_xs)
    np.testing.assert_allclose(S, expected)
    assert np.any(S[:4, 4:] != 0.0)  # the angle mode couples the two observables
    assert not p.constraints[0].covariance.dense
    assert p.chi2(full) == 0.0
