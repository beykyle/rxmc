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
        interaction_central: Callable[[float, tuple], complex],
        interaction_spin_orbit: Callable[[float, tuple], complex],
        calculate_interaction_from_params: Callable[
            [jitr.xs.elastic.DifferentialWorkspace, tuple], tuple
        ],
        params: list = [],
        model_name: str = None,
    ):
        """
        Parameters
        ----------
        quantity : str
            Observable to compute: ``"dXS/dA"``, ``"dXS/dRuth"``, or ``"Ay"``.
        interaction_central : callable
            ``f(r, args) -> complex`` returning the central interaction potential.
        interaction_spin_orbit : callable
            ``f(r, args) -> complex`` returning the spin-orbit potential.
        calculate_interaction_from_params : callable
            ``f(workspace, *params) -> (central_args, spin_orbit_args)``
            mapping model parameters to the argument tuples expected by the
            interaction callables.
        params : list of Parameter, optional
            Parameters of the model.  Defaults to ``[]``.
        model_name : str, optional
            Human-readable model name.  Defaults to ``"ElasticDifferentialXSModel"``.
        """
        self.model_name = model_name or "ElasticDifferentialXSModel"

        self.quantity = quantity
        self.interaction_central = interaction_central
        self.interaction_spin_orbit = interaction_spin_orbit
        self.calculate_interaction_from_params = calculate_interaction_from_params

        if self.quantity == "dXS/dA":
            self.extractor = extract_dXS_dA
        elif self.quantity == "dXS/dRuth":
            self.extractor = extract_dXS_dRuth
        elif self.quantity == "Ay":
            self.extractor = extract_Ay

        super().__init__(params)

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
        central_params, spin_orbit_params = self.calculate_interaction_from_params(
            ws, *params
        )
        xs = ws.xs(
            self.interaction_central,
            self.interaction_spin_orbit,
            args_central=central_params,
            args_spin_orbit=spin_orbit_params,
        )
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
            Physical-model parameter values.

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
        ws = observation.visualization_workspace
        central_params, spin_orbit_params = self.calculate_interaction_from_params(
            ws, *params
        )
        xs = ws.xs(
            self.interaction_central,
            self.interaction_spin_orbit,
            args_central=central_params,
            args_spin_orbit=spin_orbit_params,
        )
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
        return self.extractor(xs, ws)


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
