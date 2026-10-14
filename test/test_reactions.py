"""The jitr-backed reaction models: real solves with small settings.

The first solve of each workspace shape in a process pays JAX's compilation
(about a second); every later solve is milliseconds and a second angular grid
on the same basis shares its solver.
"""

import dill
import jitr
import numpy as np
import pytest
from jitr.optical_potentials import kduq
from jitr.optical_potentials.potential_forms import (
    coulomb_charged_sphere,
    thomas_safe,
    woods_saxon_safe,
)
from jitr.xs.coupled import Potential
from scipy import stats

import rxmc.reactions.elastic as elastic_module
import rxmc.reactions.ias as ias_module
from rxmc import Comparison, Constraint, Dataset, Parameter, Problem, reactions
from rxmc.reactions import (
    ElasticXS,
    IntegratedXS,
    IsobaricAnalogPN,
    momentum_transfer,
    rutherford,
)
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
        r = ws.r
        potential = Potential(central(r, *THETA), spin_orbit(r, 6.0, R, 0.45))
        np.testing.assert_allclose(y, ws.xs(potential).dsdo[0] / 1000.0)

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

    def test_ratio_to_rutherford_needs_a_charged_projectile(self):
        with pytest.raises(ValueError, match="neutral projectile"):
            omp("dXS/dRuth").bind(ANGLES, meta(N_CA))

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


class TestIntegratedXS:
    E = np.array([5.0, 20.0, 60.0])
    PB = jitr.reactions.ElasticReaction(target=(208, 82), projectile=(1, 0))
    NI = jitr.reactions.ElasticReaction(target=(58, 28), projectile=(1, 1))

    def model(self, quantity, projectile):
        return IntegratedXS.from_omp(
            quantity, kduq.KDUQ(projectile), lmax=30, zeros_per_node=8
        )

    @pytest.mark.parametrize("quantity, rxn", [("total", "PB"), ("reaction", "NI")])
    def test_matches_jitr_directly(self, quantity, rxn):
        reaction = getattr(self, rxn)
        projectile = (1, reaction.projectile.Z)
        params = kduq.get_kd03(projectile)
        model = self.model(quantity, projectile)
        y = model.bind(self.E, {"reaction": reaction})(*params)
        assert y.shape == (3,) and np.all(y > 0)
        ws = model.workspace(self.E, {"reaction": reaction})
        terms = kduq.KDUQ(projectile)(ws.r, reaction, ws.kinematics, *params)
        coulomb = terms[2] if reaction.projectile.Z else None
        xs = ws.xs(Potential(terms[0], terms[1], coulomb))
        direct = xs.t if quantity == "total" else xs.rxn
        np.testing.assert_allclose(y, np.asarray(direct) / 1000.0, rtol=1e-12)
        # one energy at a time, through ElasticXS's single-energy workspaces
        for e, Elab in enumerate(self.E):
            ws1 = ElasticXS.from_omp(
                "dXS/dA", kduq.KDUQ(projectile), lmax=30, zeros_per_node=8
            ).workspace(ANGLES, {"reaction": reaction, "Elab": Elab})
            xs1 = ws1.xs(
                Potential(
                    *kduq.KDUQ(projectile)(ws1.r, reaction, ws1.kinematics, *params)
                )
                if reaction.projectile.Z
                else Potential(
                    *kduq.KDUQ(projectile)(ws1.r, reaction, ws1.kinematics, *params)[:2]
                )
            )
            one = xs1.t if quantity == "total" else xs1.rxn
            np.testing.assert_allclose(y[e], one[0] / 1000.0, rtol=5e-4)

    def test_from_omp_matches_explicit_callables(self):
        params = kduq.get_kd03((1, 0))
        explicit = ElasticXS(
            "dXS/dA",
            lambda r, ws, v: kduq.KDUQ((1, 0))(r, ws.reaction, ws.kinematics, *v)[0],
            lambda r, ws, v: kduq.KDUQ((1, 0))(r, ws.reaction, ws.kinematics, *v)[1],
            lambda ws, *v: ((ws, v), (ws, v)),
            [Parameter(n) for n in kduq.KDUQ((1, 0)).params],
        )
        wrapped = ElasticXS.from_omp("dXS/dA", kduq.KDUQ((1, 0)))
        assert [p.name for p in wrapped.params] == kduq.KDUQ((1, 0)).params
        np.testing.assert_allclose(
            wrapped.bind(ANGLES, meta())(*params),
            explicit.bind(ANGLES, meta())(*params),
            rtol=1e-12,
        )

    def test_grids_and_projectiles_are_checked(self):
        with pytest.raises(ValueError, match="infinite"):
            self.model("total", (1, 1)).bind(self.E, {"reaction": self.NI})
        with pytest.raises(ValueError, match="positive"):
            self.model("total", (1, 0)).bind(-self.E, {"reaction": self.PB})
        with pytest.raises(ValueError, match="meta\\['reaction'\\]"):
            self.model("total", (1, 0)).bind(self.E, {})
        with pytest.raises(ValueError, match="quantity"):
            IntegratedXS.from_omp("dXS/dA", kduq.KDUQ((1, 0)))


class TestIsobaricAnalogU1:
    A, Z = 48, 20
    RXN = jitr.reactions.Reaction(
        target=(48, 20), projectile=(1, 1), product=(1, 0), residual=(48, 21)
    )
    META = {"reaction": RXN, "Elab": 25.0, "ExIAS": 6.7}
    X = np.linspace(0.2, 2.6, 5)
    RA = 1.2 * 48 ** (1 / 3)

    def args(self, ws, Vv, Wv, Rv, av):
        return (
            (self.Z, self.RA),
            (Vv + 4.0, Wv, Rv, av),
            (6.0, self.RA, 0.45),
            (Vv - 4.0, Wv, Rv, av),
            (6.0, self.RA, 0.45),
        )

    def model(self, args, **kw):
        return IsobaricAnalogPN(
            coulomb_charged_sphere,
            central,
            spin_orbit,
            central,
            spin_orbit,
            args,
            [Parameter(n) for n in ("Vv", "Wv", "Rv", "av")],
            lmax=10,
            **kw,
        )

    def test_explicit_isovector_difference_is_the_default(self):
        theta = (48.0, 3.5, self.RA, 0.7)
        factor = jitr.xs.quasielastic_pn.isovector_factor(self.RXN)

        def U1_central(r, Vv, Wv, Rv, av):
            # -(U_n - U_p) * factor, with U_p - U_n = -8 MeV of real depth
            return -factor * (
                central(r, Vv - 4.0, Wv, Rv, av) - central(r, Vv + 4.0, Wv, Rv, av)
            )

        for method in ("dwba", "cc"):
            default = self.model(self.args, method=method).bind(self.X, self.META)
            explicit = self.model(
                lambda ws, *v: (*self.args(ws, *v), tuple(v), ()),
                method=method,
                U1_central=U1_central,
            ).bind(self.X, self.META)
            np.testing.assert_allclose(explicit(*theta), default(*theta), rtol=1e-10)
            doubled = self.model(
                lambda ws, *v: (*self.args(ws, *v), tuple(v), ()),
                method=method,
                U1_central=lambda r, *v: 2 * U1_central(r, *v),
                U1_spin_orbit=lambda r: 0 * r,
            ).bind(self.X, self.META)
            assert not np.allclose(doubled(*theta), default(*theta))

    def test_args_count_and_method_are_checked(self):
        with pytest.raises(ValueError, match="method"):
            self.model(self.args, method="born")
        pred = self.model(self.args, U1_central=central).bind(self.X, self.META)
        with pytest.raises(ValueError, match="7 argument tuples"):
            pred(48.0, 3.5, self.RA, 0.7)


def test_from_omp_shares_parameters_through_a_mapping():
    protons = {name: Parameter(name) for name in kduq.KDUQ((1, 1)).params}
    neutrons = {name: Parameter("n_" + name) for name in kduq.KDUQ((1, 0)).params}
    reaction = IntegratedXS.from_omp("reaction", kduq.KDUQ((1, 1)), protons)
    elastic = ElasticXS.from_omp("dXS/dA", kduq.KDUQ((1, 1)), protons)
    total = IntegratedXS.from_omp("total", kduq.KDUQ((1, 0)), neutrons)
    assert all(a is b for a, b in zip(reaction.params, elastic.params, strict=True))
    assert not set(map(id, total.params)) & set(map(id, elastic.params))
    with pytest.raises(ValueError, match="no entry"):
        ElasticXS.from_omp("dXS/dA", kduq.KDUQ((1, 1)), {"x": Parameter("x")})


def test_integrated_xs_interpolates_a_dense_excitation_function():
    PB = jitr.reactions.ElasticReaction(target=(208, 82), projectile=(1, 0))
    x = np.sort(np.random.default_rng(0).uniform(2.0, 150.0, 300))
    params = kduq.get_kd03((1, 0))
    coarse = IntegratedXS.from_omp("total", kduq.KDUQ((1, 0)), n_energies=40)
    assert coarse.workspace(x, {"reaction": PB}).kinematics.Elab.size == 40
    y = coarse.bind(x, {"reaction": PB})(*params)
    every = IntegratedXS.from_omp("total", kduq.KDUQ((1, 0))).bind(
        x[::30], {"reaction": PB}
    )
    assert y.shape == x.shape
    np.testing.assert_allclose(y[::30], every(*params), rtol=2e-3)
