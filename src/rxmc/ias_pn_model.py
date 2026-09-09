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
from .observation_from_measurement import MB_PER_B
from .physical_model import PhysicalModel


def _require_observation(observation) -> None:
    """Reject observations this model cannot evaluate on.

    Both reaction observations report ``quantity == "dXS/dA"``, so a string
    check cannot tell them apart; the class carries the solver workspace the
    model needs.
    """
    if not isinstance(observation, IsobaricAnalogPNObservation):
        raise ValueError(
            "IsobaricAnalogPNXSModel requires an IsobaricAnalogPNObservation, "
            f"got {type(observation).__name__}"
        )


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

    The ``calculate_params`` function should have the signature:
    ``(ws: jitr.xs.quasielastic_pn.Workspace, *params: tuple) -> tuple``
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
        transform=None,
    ):
        """
        Parameters
        ----------
        U_p_coulomb : callable
            ``f(r, *args) -> np.ndarray`` on the radial grid ``r``: the proton Coulomb
            potential.
        U_p_central : callable
            ``f(r, *args) -> np.ndarray`` on the radial grid ``r``: the proton central
            potential.
        U_p_spin_orbit : callable
            ``f(r, *args) -> np.ndarray`` on the radial grid ``r``: the proton spin-orbit
            potential.
        U_n_central : callable
            ``f(r, *args) -> np.ndarray`` on the radial grid ``r``: the neutron central
            potential.
        U_n_spin_orbit : callable
            ``f(r, *args) -> np.ndarray`` on the radial grid ``r``: the neutron spin-orbit
            potential.
        calculate_params : callable
            ``f(workspace, *params) -> (args_p_coulomb, args_p_central,
            args_p_spin_orbit, args_n_central, args_n_spin_orbit)``
            mapping model parameters to the argument tuples expected by each
            potential callable.
        params : list of Parameter, optional
            Parameters of the model.  Defaults to ``[]``.
        model_name : str, optional
            Human-readable model name.  Defaults to ``"IsobaricAnalogPNXSModel"``.
        transform : Transform or callable, optional
            Parametric model-side transform applied to the prediction; see
            :class:`~rxmc.physical_model.PhysicalModel`.
        """
        self.model_name = model_name or "IsobaricAnalogPNXSModel"
        self.U_p_coulomb = U_p_coulomb
        self.U_p_central = U_p_central
        self.U_p_spin_orbit = U_p_spin_orbit
        self.U_n_central = U_n_central
        self.U_n_spin_orbit = U_n_spin_orbit
        self.calculate_params = calculate_params

        super().__init__(params, transform=transform)

    def _xs(self, ws, params) -> np.ndarray:
        """Evaluate the five potentials on ``ws.radial_grid()`` and solve (b/sr)."""
        (
            args_p_coulomb,
            args_p_central,
            args_p_spin_orbit,
            args_n_central,
            args_n_spin_orbit,
        ) = self.calculate_params(ws, *params)
        r = ws.radial_grid()
        return (
            ws.xs(
                self.U_p_coulomb(r, *args_p_coulomb),
                self.U_p_central(r, *args_p_central),
                self.U_p_spin_orbit(r, *args_p_spin_orbit),
                self.U_n_central(r, *args_n_central),
                self.U_n_spin_orbit(r, *args_n_spin_orbit),
            )
            / MB_PER_B  # jitr reports mb/sr; internal unit is b/sr
        )

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
            Physical-model (base) parameter values, consumed by
            *calculate_params*; transform parameters are split off by
            ``__call__``.

        Returns
        -------
        np.ndarray
            Predicted (p,n) IAS differential cross section in b/sr on
            ``observation.constraint_workspace.angles``.
        """
        _require_observation(observation)
        return self._xs(observation.constraint_workspace, params)

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
        _require_observation(observation)
        base, values = self.split_params(params)
        xs = self._xs(observation.visualization_workspace, base)
        return self.apply_transform(observation, xs, values)
