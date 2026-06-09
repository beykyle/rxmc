"""
Physical model for isobaric-analog-state (p,n) differential cross sections.

:class:`IsobaricAnalogPNXSModel` wraps a ``jitr`` quasielastic-pn solver to
predict (p,n) IAS differential cross sections given five parametric interaction
potentials (proton Coulomb, proton central, proton spin-orbit, neutron central,
neutron spin-orbit).
"""

from typing import Callable

import jitr
import numpy as np

from .ias_pn_observation import IsobaricAnalogPNObservation
from .physical_model import PhysicalModel


class IsobaricAnalogPNXSModel(PhysicalModel):
    """
    A model that predicts the (p,n) IAS differential xs for a given reaction.
    This model requires five interaction potentials:
    - Proton Coulomb potential: U_p_coulomb
    - Proton central potential: U_p_central
    - Proton spin-orbit potential: U_p_spin_orbit
    - Neutron central potential: U_n_central
    - Neutron spin-orbit potential: U_n_spin_orbit

    Each potential takes in an arbitrary tuple of params, which are
    calculated from the model parameters via the `calculate_params` function.

    The `calculate_params` function should have the signature:
    (ws: jitr.xs.quasielastic_pn.Workspace, *params: tuple) -> tuple
    and return a tuple of five elements, each being a tuple of parameters
    to be passed to the corresponding potential function in the order listed above.
    """

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
        Parameters
        ----------
        U_p_coulomb : callable
            ``f(r, args) -> complex`` — proton Coulomb potential.
        U_p_central : callable
            ``f(r, args) -> complex`` — proton central potential.
        U_p_spin_orbit : callable
            ``f(r, args) -> complex`` — proton spin-orbit potential.
        U_n_central : callable
            ``f(r, args) -> complex`` — neutron central potential.
        U_n_spin_orbit : callable
            ``f(r, args) -> complex`` — neutron spin-orbit potential.
        calculate_params : callable
            ``f(workspace, *params) -> (args_p_coulomb, args_p_central,
            args_p_spin_orbit, args_n_central, args_n_spin_orbit)``
            mapping model parameters to the argument tuples expected by each
            potential callable.
        params : list of Parameter, optional
            Parameters of the model.  Defaults to ``[]``.
        model_name : str, optional
            Human-readable model name.  Defaults to ``"IsobaricAnalogPNXSModel"``.
        """
        self.model_name = model_name or "IsobaricAnalogPNXSModel"
        self.U_p_coulomb = U_p_coulomb
        self.U_p_central = U_p_central
        self.U_p_spin_orbit = U_p_spin_orbit
        self.U_n_central = U_n_central
        self.U_n_spin_orbit = U_n_spin_orbit
        self.calculate_params = calculate_params

        super().__init__(params)

    def evaluate(
        self,
        observation: IsobaricAnalogPNObservation,
        *params: tuple,
    ) -> np.ndarray:
        """
        Evaluate the model on the constraint angular grid.

        Parameters
        ----------
        observation : IsobaricAnalogPNObservation
            Observation containing the pre-built workspace.
        *params : float
            Physical-model parameter values, consumed by *calculate_params*.

        Returns
        -------
        np.ndarray
            Predicted (p,n) IAS differential cross section in b/sr on
            ``observation.constraint_workspace.angles``.
        """
        ws = observation.constraint_workspace
        (
            args_p_coulomb,
            args_p_central,
            args_p_spin_orbit,
            args_n_central,
            args_n_spin_orbit,
        ) = self.calculate_params(ws, *params)
        xs = (
            ws.xs(
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
            / 1000
        )
        return xs

    def visualizable_model_prediction(
        self,
        observation: IsobaricAnalogPNObservation,
        *params: tuple,
    ) -> np.ndarray:
        """
        Evaluate the model on the visualisation angular grid.

        Parameters
        ----------
        observation : IsobaricAnalogPNObservation
            Observation containing the pre-built workspace.
        *params : float
            Physical-model parameter values, consumed by *calculate_params*.

        Returns
        -------
        np.ndarray
            Predicted (p,n) IAS differential cross section in b/sr on
            ``observation.visualization_workspace.angles``.
        """
        ws = observation.visualization_workspace
        (
            args_p_coulomb,
            args_p_central,
            args_p_spin_orbit,
            args_n_central,
            args_n_spin_orbit,
        ) = self.calculate_params(ws, *params)
        xs = (
            ws.xs(
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
            / 1000
        )
        return xs
