"""Real EXFOR excitation functions through from_measurement and IntegratedXS.

Importing exfor_tools opens (or, without one, downloads) the EXFOR database,
so this is slow-tier and skips unless a local database is configured with
``X43I_DATAPATH``.
"""

import os

import jitr
import numpy as np
import pytest
from jitr.optical_potentials import kduq
from scipy import stats

from rxmc import Comparison, Constraint, Parameter, Problem, from_measurement
from rxmc.reactions import IntegratedXS

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        "X43I_DATAPATH" not in os.environ, reason="no local EXFOR database"
    ),
]

NEUTRON, PROTON = (1, 0), (1, 1)


def measurements(projectile, target, process):
    """The points from 10 to 200 MeV, with a reported error, of every
    parsable excitation function, each as a mapping of the measurement's
    fields."""
    import exfor_tools
    from exfor_tools.reaction import Reaction, query_for_reaction

    reaction = Reaction(target=target, projectile=projectile, process=process)
    out = []
    for entry in query_for_reaction(reaction, "XS"):
        try:
            parsed = exfor_tools.ExforEntry(
                entry=entry, reaction=reaction, quantity="XS", vocal=False
            )
        except Exception:  # a few entries do not parse
            continue
        for m in parsed.measurements:
            # points with no reported error need an error model of their own
            keep = (m.x >= 10) & (m.x <= 200) & (np.asarray(m.statistical_err) > 0)
            if keep.sum() < 2:
                continue
            fields = {k: v for k, v in vars(m).items() if k not in ("notes",)}
            for k in ("x", "x_err", "y", "statistical_err"):
                fields[k] = np.asarray(fields[k])[keep]
            for k in ("systematic_offset_err",):  # scalar or per point
                if np.ndim(fields[k]):
                    fields[k] = np.asarray(fields[k])[keep]
            out.append(fields)
    return out


def test_a_real_energy_distribution_object():
    from exfor_tools.distribution import EnergyDistribution

    m = EnergyDistribution(
        "TEST0001",
        "XS",
        "MeV",
        "millibarn",
        np.array([10.0, 20.0, 40.0]),
        np.zeros(3),
        np.array([1100.0, 1050.0, 900.0]),
        np.array([20.0, 20.0, 18.0]),
        0.03,
        0.0,
    )
    reaction = jitr.reactions.ElasticReaction(target=(58, 28), projectile=PROTON)
    d = from_measurement(m, reaction=reaction)
    np.testing.assert_allclose(d.x, [10.0, 20.0, 40.0])
    np.testing.assert_allclose(d.y, [1.1, 1.05, 0.9])
    assert d.norm_err == 0.03 and d.meta["reaction"] is reaction


@pytest.mark.parametrize(
    "projectile, target, process, quantity",
    [(NEUTRON, (208, 82), "tot", "total"), (PROTON, (58, 28), "non", "reaction")],
)
def test_kd03_against_exfor(projectile, target, process, quantity):
    """One model bound to every measurement of the reaction describes the
    10-200 MeV data at KD03 to a few percent."""
    reaction = jitr.reactions.ElasticReaction(target=target, projectile=projectile)
    data = [
        from_measurement(m, reaction=reaction, quantity=quantity)
        for m in measurements(projectile, target, process)
    ]
    assert len(data) >= 2
    omp = kduq.KDUQ(projectile)
    params = [
        Parameter(name, prior=stats.norm(v, 0.1 * abs(v) + 1e-3))
        for name, v in zip(omp.params, kduq.get_kd03(projectile))
    ]
    model = IntegratedXS.from_omp(quantity, omp, params, n_energies=40)
    problem = Problem([Constraint([Comparison(d, model)]) for d in data])
    theta = kduq.get_kd03(projectile)
    assert np.isfinite(problem.log_posterior(theta))
    # model / data: most sets agree to 1-3%; a few are off (n + 208Pb
    # 13735002 near 17.5 MeV; p + 58Ni O0081003, 4 of the 11 proton points,
    # ~12% below KD03), hence the loose bounds
    deviation = np.concatenate(
        [np.abs(model.bind(d.x, d.meta)(*theta) / d.y - 1) for d in data]
    )
    assert np.median(deviation) < 0.06 and np.percentile(deviation, 90) < 0.16
