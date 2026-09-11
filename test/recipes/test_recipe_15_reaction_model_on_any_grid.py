"""Recipe 15: evaluate a reaction model on any grid.

I want the model on the data angles for the likelihood and on a fine grid
for plotting, with the solver set up once per grid.
"""

import jitr
import numpy as np
import pytest
from jitr.optical_potentials.potential_forms import thomas_safe, woods_saxon_safe
from scipy import stats

from rxmc import Comparison, Constraint, Dataset, Parameter, Problem
from rxmc.reactions import ElasticXS

MSO = 1.0 / jitr.utils.constants.WAVENUMBER_PION
R = 1.2 * 40 ** (1 / 3)
N_CA = jitr.reactions.ElasticReaction(target=(40, 20), projectile=(1, 0))


def central(r, Vv, Wv, Rv, av):
    return -(Vv + 1j * Wv) * woods_saxon_safe(r, Rv, av)


def spin_orbit(r, Vso, Rso, aso):
    return Vso * MSO**2 * thomas_safe(r, Rso, aso)


def test_data_grid_for_the_likelihood_and_a_fine_grid_for_plotting():
    omp = ElasticXS(
        "dXS/dA",
        central,
        spin_orbit,
        lambda ws, *x: (tuple(x), (6.0, R, 0.45)),
        [Parameter(n, prior=stats.norm(0, 100)) for n in ("Vv", "Wv", "Rv", "av")],
        lmax=10,
    )
    theta = np.array([48.0, 3.5, R, 0.7])
    x = np.linspace(0.3, 2.5, 5)
    meta = {"reaction": N_CA, "Elab": 14.1}
    truth = omp.bind(x, meta)(*theta)
    d = Dataset(x, truth, 0.05 * truth, label="mock", meta=meta)
    p = Problem([Constraint([Comparison(d, omp)])])
    assert p.chi2(theta) == pytest.approx(0.0, abs=1e-20)
    # the same model on a fine grid, read back through problem.columns
    fine = omp.bind(np.deg2rad(np.linspace(0.5, 179.5, 60)), d.meta)
    sample = np.concatenate([theta, [0.0]])  # a chain row with an extra column
    y_fine = fine(*sample[p.columns(omp.params)])
    assert y_fine.shape == (60,) and np.all(np.isfinite(y_fine)) and np.all(y_fine > 0)
    assert len(omp._cache) == 1  # one basis for both grids
    # cross sections are in b/sr: jitr's mb/sr divided by 1000
    ws = omp.workspace(x, meta)
    r = ws.radial_grid()
    direct = ws.xs(central(r, *theta), spin_orbit(r, 6.0, R, 0.45), None).dsdo
    np.testing.assert_allclose(truth, direct / 1000.0)
