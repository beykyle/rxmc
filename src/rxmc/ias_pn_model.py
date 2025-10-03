from typing import Callable

import jitr
import numpy as np

from .ias_pn_observation import IsobaricAnalogPNObservation
from .physical_model import PhysicalModel


class IsobaricAnalogPNXSModel(PhysicalModel):
    """ """

    def __init__(
        self,
        U_p_coulomb: Callable[[float, tuple], complex],
        U_p_central: Callable[[float, tuple], complex],
        U_p_spin_orbit: Callable[[float, tuple], complex],
        U_n_central: Callable[[float, tuple], complex],
        U_n_spin_orbit: Callable[[float, tuple], complex],
        calculate_params: Callable[[jitr.xs.quasielastic_pn.Workspace, tuple], tuple],
        params: list = [],
        model_name: str = None,
    ):
        """
        Parameters:
        ----------
        calculate_interaction_from_params : Callable
        params: list[Parameter] = []
        model_name : str, optional
            Name of the model, used for identification purposes.
            Defaults to None.
        """
        self.model_name = model_name or "ElasticDifferentialXSModel"
        self.U_p_coulomb = U_p_coulomb
        self.U_p_central = U_p_central
        self.U_p_spin_orbit = U_p_spin_orbit
        self.U_n_central = U_n_central
        self.U_n_spin_orbit = U_n_spin_orbit
        self.calculate_params = calculate_params

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
        observation :
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
        (
            args_p_coulomb,
            args_p_central,
            args_p_spin_orbit,
            args_n_central,
            args_n_spin_orbit,
        ) = self.calculate_params(ws, *params)
        xs = ws.xs(
            self.U_p_coulomb,
            self.U_p_central,
            self.U_p_spin_orbit,
            self.U_n_central,
            self.U_n_spin_orbit,
            args_p_coulomb=args_p_coulomb,
            args_p_central=args_p_central,
            args_p_spin_orbit=args_p_spin_orbit,
            args_n_central=args_n_central,
            args_n_spin_orbit=args_n_spin_orbit,
        )
        return xs

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
        (
            args_p_coulomb,
            args_p_central,
            args_p_spin_orbit,
            args_n_central,
            args_n_spin_orbit,
        ) = self.calculate_params(ws, *params)
        xs = ws.xs(
            self.U_p_coulomb,
            self.U_p_central,
            self.U_p_spin_orbit,
            self.U_n_central,
            self.U_n_spin_orbit,
            args_p_coulomb=args_p_coulomb,
            args_p_central=args_p_central,
            args_p_spin_orbit=args_p_spin_orbit,
            args_n_central=args_n_central,
            args_n_spin_orbit=args_n_spin_orbit,
        )
        return xs
