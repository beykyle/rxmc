"""Real-solver smoke tests for the jitr-backed reaction models.

These are the only tests that exercise the actual solver pipeline
(everything else mocks ``set_up_solver``); they pin output shape,
finiteness, and positivity with deliberately small solver settings.
"""

import unittest

import jitr
import numpy as np
from jitr.optical_potentials.potential_forms import (
    coulomb_charged_sphere,
    thomas_safe,
    woods_saxon_safe,
)

from rxmc.elastic_diffxs_model import ElasticDifferentialXSModel
from rxmc.elastic_diffxs_observation import ElasticDifferentialXSObservation
from rxmc.ias_pn_model import IsobaricAnalogPNXSModel
from rxmc.ias_pn_observation import IsobaricAnalogPNObservation
from rxmc.params import Parameter

MSO = 1.0 / jitr.utils.constants.WAVENUMBER_PION


def central(r, Vv, Wv, Rv, av):
    return -(Vv + 1j * Wv) * woods_saxon_safe(r, Rv, av)


def spin_orbit(r, Vso, Rso, aso):
    return Vso * MSO**2 * thomas_safe(r, Rso, aso)


class TestElasticDifferentialXSModel(unittest.TestCase):
    def test_evaluate_and_visualization_smoke(self):
        R = 1.2 * 40 ** (1 / 3)
        rxn = jitr.reactions.ElasticReaction(target=(40, 20), projectile=(1, 0))
        model = ElasticDifferentialXSModel(
            "dXS/dA",
            interaction_central=central,
            interaction_spin_orbit=spin_orbit,
            calculate_interaction_from_params=lambda ws, *x: (
                tuple(x),
                (6.0, R, 0.45),
            ),
            params=[Parameter(n) for n in ("Vv", "Wv", "Rv", "av")],
        )
        obs = ElasticDifferentialXSObservation(
            x=np.linspace(10.0, 150.0, 6),
            y=np.ones(6),
            Elab=14.1,
            reaction=rxn,
            quantity="dXS/dA",
            measurement_quantity="dXS/dA",
            y_units="barn / steradian",
            lmax=10,
        )

        y = model.evaluate(obs, 48.0, 3.5, R, 0.7)
        self.assertEqual(y.shape, (obs.n_data_pts,))
        self.assertTrue(np.all(np.isfinite(y)))
        self.assertTrue(np.all(y > 0))

        y_vis = model.visualizable_model_prediction(obs, 48.0, 3.5, R, 0.7)
        self.assertEqual(y_vis.shape, obs.visualization_workspace.angles.shape)
        self.assertTrue(np.all(np.isfinite(y_vis)))


class TestIsobaricAnalogPNXSModel(unittest.TestCase):
    def test_evaluate_and_visualization_smoke(self):
        A, Z = 48, 20
        R = 1.2 * A ** (1 / 3)
        rxn = jitr.reactions.Reaction(
            target=(A, Z),
            projectile=(1, 1),
            product=(1, 0),
            residual=(A, Z + 1),
        )
        model = IsobaricAnalogPNXSModel(
            U_p_coulomb=coulomb_charged_sphere,
            U_p_central=central,
            U_p_spin_orbit=spin_orbit,
            U_n_central=central,
            U_n_spin_orbit=spin_orbit,
            # the (p,n) IAS transition is driven by the *difference* between
            # the proton and neutron potentials (the Lane term) — make them
            # distinct or the cross section vanishes
            calculate_params=lambda ws, Vv, Wv, Rv, av: (
                (Z, R),  # p Coulomb: zz product, charge radius
                (Vv + 4.0, Wv, Rv, av),  # p central
                (6.0, R, 0.45),  # p spin-orbit
                (Vv - 4.0, Wv, Rv, av),  # n central
                (6.0, R, 0.45),  # n spin-orbit
            ),
            params=[Parameter(n) for n in ("Vv", "Wv", "Rv", "av")],
        )
        obs = IsobaricAnalogPNObservation(
            x=np.linspace(10.0, 150.0, 5),
            y=np.ones(5),
            Elab=25.0,
            reaction=rxn,
            ExIAS=6.7,
            y_units="barn / steradian",
            lmax=10,
        )

        y = model.evaluate(obs, 48.0, 3.5, R, 0.7)
        self.assertEqual(y.shape, (obs.n_data_pts,))
        self.assertTrue(np.all(np.isfinite(y)))
        self.assertTrue(np.all(y >= 0))
        self.assertGreater(y.max(), 0)

        y_vis = model.visualizable_model_prediction(obs, 48.0, 3.5, R, 0.7)
        self.assertEqual(y_vis.shape, obs.visualization_workspace.angles.shape)
        self.assertTrue(np.all(np.isfinite(y_vis)))


if __name__ == "__main__":
    unittest.main()
