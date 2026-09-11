"""The jitr-backed reaction models: real solves with small settings.

The first solve in a process pays numba's compilation (several seconds);
every later solve is milliseconds and a second angular grid on the same
basis is free.
"""

import dill
import jitr
import numpy as np
import pytest
from jitr.optical_potentials.potential_forms import (
    coulomb_charged_sphere,
    thomas_safe,
    woods_saxon_safe,
)
from scipy import stats

import rxmc.reactions.elastic as elastic_module
import rxmc.reactions.ias as ias_module
from rxmc import Comparison, Constraint, Dataset, Parameter, Problem, reactions
from rxmc.reactions import ElasticXS, IsobaricAnalogPN, momentum_transfer, rutherford
from rxmc.transforms import scale

MSO = 1.0 / jitr.utils.constants.WAVENUMBER_PION
A, Z = 40, 20
R = 1.2 * A ** (1 / 3)
ANGLES = np.linspace(0.2, 2.6, 6)
N_CA = jitr.reactions.ElasticReaction(target=(A, Z), projectile=(1, 0))
P_CA = jitr.reactions.ElasticReaction(target=(A, Z), projectile=(1, 1))


def central(r, Vv, Wv, Rv, av):
    return -(Vv + 1j * Wv) * woods_saxon_safe(r, Rv, av)


def spin_orbit(r, Vso, Rso, aso):
    return Vso * MSO**2 * thomas_safe(r, Rso, aso)


def omp(quantity="dXS/dA", **kw):
    names = ("Vv", "Wv", "Rv", "av")
    return ElasticXS(
        quantity,
        central,
        spin_orbit,
        lambda ws, *x: (tuple(x), (6.0, R, 0.45)),
        [Parameter(n, prior=stats.norm(0, 100)) for n in names],
        lmax=10,
        **kw,
    )


THETA = (48.0, 3.5, R, 0.7)


def meta(reaction=N_CA, Elab=14.1):
    return {"reaction": reaction, "Elab": Elab}


class TestElasticXS:
    def test_cross_section_is_finite_positive_in_barns(self):
        pred = omp().bind(ANGLES, meta())
        y = pred(*THETA)
        assert y.shape == (6,) and np.all(np.isfinite(y)) and np.all(y > 0)
        # b/sr: jitr's mb/sr divided by 1000
        ws = omp().workspace(ANGLES, meta())
        r = ws.radial_grid()
        direct = ws.xs(central(r, *THETA), spin_orbit(r, 6.0, R, 0.45), None).dsdo
        np.testing.assert_allclose(y, direct / 1000.0)

    def test_ratio_to_rutherford_and_analysing_power(self):
        ratio = omp("dXS/dRuth").bind(ANGLES, meta(P_CA, 14.1))(*THETA)
        assert np.all(np.isfinite(ratio)) and np.all(ratio > 0)
        ay = omp("Ay").bind(ANGLES, meta())(*THETA)
        assert np.all(np.abs(ay) <= 1.0)
        with pytest.raises(ValueError, match="quantity"):
            omp("sigma_tot")

    def test_basis_is_cached_per_energy_and_fine_grid_is_free(self):
        model = omp()
        model.bind(ANGLES, meta())
        fine = model.bind(np.linspace(0.05, 3.1, 200), meta())
        assert len(model._cache) == 1
        model.bind(ANGLES, meta(Elab=20.0))
        assert len(model._cache) == 2
        y = fine(*THETA)
        assert y.shape == (200,) and np.all(np.isfinite(y))

    def test_missing_kinematics_names_what_is_needed(self):
        with pytest.raises(ValueError, match="meta\\['reaction'\\].*from_measurement"):
            omp().bind(ANGLES, {})
        with pytest.raises(ValueError, match="meta\\['Elab'\\]"):
            omp().bind(ANGLES, {"reaction": N_CA})

    def test_composes_like_any_model(self):
        rho = Parameter("log_rho", prior=stats.norm(0, 0.1))
        scaled = omp() | scale(rho)
        base = omp().bind(ANGLES, meta())(*THETA)
        np.testing.assert_allclose(
            scaled.bind(ANGLES, meta())(*THETA, np.log(2.0)), 2.0 * base
        )

    def test_solver_settings_reach_the_basis(self, monkeypatch):
        seen = {}
        real = elastic_module._basis

        def spy(reaction, Elab, lmax, wavelengths_beyond_range, zeros_per_node):
            seen.update(lmax=lmax, wbr=wavelengths_beyond_range, zpn=zeros_per_node)
            return real(reaction, Elab, lmax, wavelengths_beyond_range, zeros_per_node)

        monkeypatch.setattr(elastic_module, "_basis", spy)
        omp(wavelengths_beyond_range=3.5, zeros_per_node=9).bind(ANGLES, meta())
        assert seen == {"lmax": 10, "wbr": 3.5, "zpn": 9}

    def test_dill_round_trip_of_a_reaction_problem(self):
        d = Dataset(
            ANGLES, omp().bind(ANGLES, meta())(*THETA), 0.01 * np.ones(6), meta=meta()
        )
        p = Problem([Constraint([Comparison(d, omp())])])
        q = dill.loads(dill.dumps(p))
        theta = np.array(THETA)
        assert q.log_posterior(theta) == pytest.approx(p.log_posterior(theta))


class TestIsobaricAnalogPN:
    def test_cross_section_with_a_lane_term(self):
        A, Z = 48, 20
        R = 1.2 * A ** (1 / 3)
        rxn = jitr.reactions.Reaction(
            target=(A, Z), projectile=(1, 1), product=(1, 0), residual=(A, Z + 1)
        )
        model = IsobaricAnalogPN(
            coulomb_charged_sphere,
            central,
            spin_orbit,
            central,
            spin_orbit,
            # the (p,n) transition is driven by the difference between the
            # proton and neutron potentials (the Lane term): make them distinct
            lambda ws, Vv, Wv, Rv, av: (
                (Z, R),
                (Vv + 4.0, Wv, Rv, av),
                (6.0, R, 0.45),
                (Vv - 4.0, Wv, Rv, av),
                (6.0, R, 0.45),
            ),
            [Parameter(n) for n in ("Vv", "Wv", "Rv", "av")],
            lmax=10,
        )
        pred = model.bind(
            np.linspace(0.2, 2.6, 5), {"reaction": rxn, "Elab": 25.0, "ExIAS": 6.7}
        )
        y = pred(48.0, 3.5, R, 0.7)
        assert (
            y.shape == (5,)
            and np.all(np.isfinite(y))
            and np.all(y >= 0)
            and y.max() > 0
        )
        with pytest.raises(ValueError, match="ExIAS"):
            model.bind(ANGLES, {"reaction": rxn, "Elab": 25.0})

    def test_solver_settings_forwarded(self, monkeypatch):
        seen = {}

        def fake(reaction, Elab, ExIAS, angles_rad, **kw):
            seen.update(kw)
            raise RuntimeError("stop")

        monkeypatch.setattr(ias_module, "set_up_solver", fake)
        model = IsobaricAnalogPN(
            *([central] * 5),
            lambda ws, *x: ((),) * 5,
            [Parameter("V")],
            lmax=7,
            wavelengths_beyond_range=3.5,
            zeros_per_node=9,
        )
        with pytest.raises(RuntimeError):
            model.bind(ANGLES, {"reaction": object(), "Elab": 1.0, "ExIAS": 1.0})
        assert seen == {"lmax": 7, "wavelengths_beyond_range": 3.5, "zeros_per_node": 9}


def test_closed_forms():
    kin = P_CA.kinematics(14.1)
    x = np.array([0.5, 1.0, 2.0])
    np.testing.assert_allclose(momentum_transfer(x, kin.k), 2.0 * kin.k * np.sin(x / 2))
    np.testing.assert_allclose(
        rutherford(kin, x), 10 * kin.eta**2 / (4 * kin.k**2 * np.sin(x / 2) ** 4)
    )
    assert np.all(rutherford(N_CA.kinematics(14.1), x) == 0.0)
    assert reactions.ElasticXS is ElasticXS
