"""
Physical model for elastic differential cross sections.

:class:`ElasticDifferentialXSModel` wraps a ``jitr`` optical-model solver to
predict elastic differential cross sections (dXS/dΩ, dXS/dRuth, or analysing
power Ay) given a parametric central and spin-orbit interaction.
"""

from typing import Callable

import jitr
import numpy as np

from .elastic_diffxs_observation import ElasticDifferentialXSObservation
from .physical_model import PhysicalModel


class ElasticDifferentialXSModel(PhysicalModel):
    """
    A model that predicts the elastic differential xs for a given reaction.
    """

    def __init__(
        self,
        quantity: str,
        interaction_central: Callable[..., np.ndarray],
        interaction_spin_orbit: Callable[..., np.ndarray] | None,
        calculate_interaction_from_params: Callable[
            [jitr.xs.elastic.DifferentialWorkspace, tuple], tuple
        ],
        params: list = [],
        model_name: str = None,
        interaction_coulomb: Callable[..., np.ndarray] | None = None,
        transform=None,
    ):
        """
        Parameters
        ----------
        quantity : str
            Observable to compute: ``"dXS/dA"``, ``"dXS/dRuth"``, or ``"Ay"``.
        interaction_central : callable
            ``f(r, *args) -> np.ndarray`` returning the central interaction
            potential on the radial grid ``r`` (fm), in MeV.
        interaction_spin_orbit : callable or None
            ``f(r, *args) -> np.ndarray`` returning the spin-orbit potential on
            ``r``.  ``None`` for a spin-orbit-free model.
        calculate_interaction_from_params : callable
            ``f(workspace, *params) -> (central_args, spin_orbit_args)`` or
            ``-> (central_args, spin_orbit_args, coulomb_args)`` mapping model
            parameters to the argument tuples expected by the interaction
            callables.
        params : list of Parameter, optional
            Parameters of the model.  Defaults to ``[]``.
        model_name : str, optional
            Human-readable model name.  Defaults to ``"ElasticDifferentialXSModel"``.
        interaction_coulomb : callable, optional
            ``f(r, *args) -> np.ndarray`` returning the Coulomb potential on
            ``r``.  When ``None`` the Coulomb interaction inside the channel
            radius must be folded into ``interaction_central``.
        transform : Transform or callable, optional
            Parametric model-side transform applied to the prediction; see
            :class:`~rxmc.physical_model.PhysicalModel`.
        """
        self.model_name = model_name or "ElasticDifferentialXSModel"

        self.quantity = quantity
        self.interaction_central = interaction_central
        self.interaction_spin_orbit = interaction_spin_orbit
        self.interaction_coulomb = interaction_coulomb
        self.calculate_interaction_from_params = calculate_interaction_from_params

        if self.quantity == "dXS/dA":
            self.extractor = extract_dXS_dA
        elif self.quantity == "dXS/dRuth":
            self.extractor = extract_dXS_dRuth
        elif self.quantity == "Ay":
            self.extractor = extract_Ay
        else:
            raise ValueError(
                f"Unknown quantity {quantity!r}; expected 'dXS/dA', 'dXS/dRuth' "
                "or 'Ay'."
            )

        super().__init__(params, transform=transform)

    def _xs(self, ws, params):
        """Evaluate the potentials on ``ws.radial_grid()`` and solve."""
        args = self.calculate_interaction_from_params(ws, *params)
        if len(args) == 2:
            (central_args, spin_orbit_args), coulomb_args = args, ()
        elif len(args) == 3:
            central_args, spin_orbit_args, coulomb_args = args
        else:
            raise ValueError(
                "calculate_interaction_from_params must return 2 or 3 argument "
                f"tuples, got {len(args)}"
            )
        r = ws.radial_grid()
        central = self.interaction_central(r, *central_args)
        spin_orbit = (
            None
            if self.interaction_spin_orbit is None
            else self.interaction_spin_orbit(r, *spin_orbit_args)
        )
        coulomb = (
            None
            if self.interaction_coulomb is None
            else self.interaction_coulomb(r, *coulomb_args)
        )
        return ws.xs(central, spin_orbit, coulomb)

    def evaluate(
        self,
        observation: ElasticDifferentialXSObservation,
        *params: tuple,
    ) -> np.ndarray:
        """
        Evaluate the model on the constraint angular grid.

        Parameters
        ----------
        observation : ElasticDifferentialXSObservation
            Observation containing the reaction data and pre-built workspace.
        *params : float
            Physical-model parameter values.

        Returns
        -------
        np.ndarray
            Predicted observable on ``observation.constraint_workspace.angles``.
        """
        if observation.quantity != self.quantity:
            raise ValueError(
                f"Observation quantity {observation.quantity} does not match "
                f"model quantity {self.quantity}."
            )
        ws = observation.constraint_workspace
        xs = self._xs(ws, params)
        if observation.compound_correction is not None:
            if observation.quantity not in ["dXS/dA", "dXS/dRuth"]:
                raise ValueError(
                    "Compound correction can only be applied to dXS/dA and dXS/dRuth."
                )
            xs.dsdo += observation.compound_correction
            xs.t += 2 * np.pi * np.trapz(observation.compound_correction, ws.angles)
        return self.extractor(xs, ws)

    def visualizable_model_prediction(
        self,
        observation: ElasticDifferentialXSObservation,
        *params: tuple,
    ) -> np.ndarray:
        """
        Evaluate the model on the visualisation angular grid.

        Parameters
        ----------
        observation : ElasticDifferentialXSObservation
            Observation containing the reaction data and pre-built workspace.
        *params : float
            Full model parameter values (physical parameters followed by any
            transform parameters).

        Returns
        -------
        np.ndarray
            Predicted observable on ``observation.visualization_workspace.angles``.
        """
        if observation.quantity != self.quantity:
            raise ValueError(
                f"Observation quantity {observation.quantity} does not match "
                f"model quantity {self.quantity}."
            )
        base, values = self.split_params(params)
        ws = observation.visualization_workspace
        xs = self._xs(ws, base)
        if observation.compound_correction is not None:
            cn = np.interp(
                ws.angles,
                observation.constraint_workspace.angles,
                observation.compound_correction,
            )
            if observation.quantity not in ["dXS/dA", "dXS/dRuth"]:
                raise ValueError(
                    "Compound correction can only be applied to dXS/dA and dXS/dRuth."
                )
            xs.dsdo += cn
            xs.t += 2 * np.pi * np.trapz(cn, ws.angles)
        return self.apply_transform(observation, self.extractor(xs, ws), values)


def extract_dXS_dA(
    xs: jitr.xs.elastic.ElasticXS, ws: jitr.xs.elastic.DifferentialWorkspace
) -> np.ndarray:
    """Extracts dXS/dA in b/Sr"""
    return xs.dsdo / 1000


def extract_dXS_dRuth(
    xs: jitr.xs.elastic.ElasticXS, ws: jitr.xs.elastic.DifferentialWorkspace
) -> np.ndarray:
    """Extracts dXS/dRuth (dimensionlesss)"""
    return xs.dsdo / ws.rutherford


def extract_Ay(
    xs: jitr.xs.elastic.ElasticXS, ws: jitr.xs.elastic.DifferentialWorkspace
) -> np.ndarray:
    """Extracts Ay (dimensionless)"""
    return xs.Ay
