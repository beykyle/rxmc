from typing import Callable
import numpy as np

import jitr

from .physical_model import PhysicalModel
from .elastic_diffxs_observation import ElasticDifferentialXSObservation


class ElasticDifferentialXSModel(PhysicalModel):
    """
    A model that predicts the elastic differential xs for a given reaction.

    Should be used within an ElasticDifferentialXSConstraint rather than a
    regular Constraint.

    Rather than taking in an `Observation` object, this model takes in a
    `jitr.xs.elastic.DifferentialWorkspace` object, which contains all
    the relevant precomputed quantities needed to calculate the differential
    cross section using the provided interaction parameters.

    The model is initialized with two interactions: one for the central
    potential and one for the spin-orbit interaction. The
    `calculate_interaction_from_params` function is used to extract
    the parameters needed for each interaction.
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
        Initialize the ElasticDifferentialXSModel with the interactions and
        a function to calculate the subparameters.
        Parameters:
        ----------
        quantity : str
            The type of quantity to be calculated (e.g., "dXS/dA",
            "dXS/dRuth", "Ay").
        interaction_central : Callable
            Function that returns the central interaction potential for a given
            energy and parameters.
        interaction_spin_orbit : Callable
            Function that returns the spin-orbit interaction potential for a
            given energy and parameters.
        calculate_interaction_from_params : Callable
            Function that takes in a workspace, and the model parameters, and
            returns the parameters needed for the central and spin-orbit
            interactions. Should return a tuple of size two, the first element
            being the tuple of parameters taken in by `interaction_central` and
            the second element being the tuple of parameters taken in by
            `interaction_spin_orbit`.
        params: list[Parameter] = []
            A list of Parameter objects that define the model's parameters.
            Each Parameter should have a name and a dtype.
        model_name : str, optional
            Name of the model, used for identification purposes.
            Defaults to None.
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
        Evaluate the model at the given parameters.

        Parameters:
        ----------
        observation : ElasticDifferentialXSObservation
            The observation containing the reaction data and workspace.
        params : tuple
            The parameters of the physical model.

        Returns:
        -------
        np.ndarray
            An array, containing the evaluated differential data on the
            angular grid corresponding to the
            `observation.constraint_workspace`.
        """
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
        return self.extractor(xs, ws)

    def visualizable_model_prediction(
        self,
        observation: ElasticDifferentialXSObservation,
        *params: tuple,
    ) -> np.ndarray:
        """
        Visualize the model at the given parameters.

        Parameters:
        ----------
        observation : ElasticDifferentialXSObservation
            The observation containing the reaction data and workspace.
        params : tuple
            The parameters of the physical model.

        Returns:
        -------
        np.ndarray
            An array, containing the evaluated differential data on the
            angular grid corresponding to the
            `observation.visualization_workspace`.
        """
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
