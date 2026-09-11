"""
(p,n) isobaric-analog-state differential cross sections from ``jitr``.

:class:`IsobaricAnalogPN` is a :class:`~rxmc.model.Model` that overrides only
``bind``: given a grid of angles and a dataset's kinematics
(``meta["reaction"]``, ``meta["Elab"]``, ``meta["ExIAS"]``) it builds the
``jitr`` quasielastic (p,n) workspace on that grid and returns a
:class:`~rxmc.model.Predictor` for the cross section in b/sr.  The (p,n)
transition is driven by the difference between the proton and neutron
potentials (the Lane term), so the five potentials are declared separately.
"""

from __future__ import annotations

from typing import Callable

import jitr
import numpy as np

from ..model import Model, Predictor
from ..units import DEFAULT_LMAX, MB_PER_B, check_angle_grid

__all__ = ["IsobaricAnalogPN", "set_up_solver"]


def set_up_solver(
    reaction,
    Elab: float,
    ExIAS: float,
    angles_rad: np.ndarray,
    lmax: int = DEFAULT_LMAX,
    wavelengths_beyond_range: float = 2.0,
    zeros_per_node: int = 5,
):
    """Set up the ``jitr`` (p,n) workspace for a reaction at one energy.

    Parameters
    ----------
    reaction : jitr.reactions.Reaction
    Elab : float
        Laboratory energy of the incoming proton (MeV).
    ExIAS : float
        Excitation energy of the isobaric analog state in the residual (MeV).
    angles_rad : np.ndarray
        Angles in radians the cross section is wanted on.
    lmax, wavelengths_beyond_range, zeros_per_node
        Solver settings (see :func:`rxmc.reactions.elastic.set_up_solver`).

    Returns
    -------
    (jitr.xs.quasielastic_pn.Workspace, kinematics_entrance, kinematics_exit)
    """
    kinematics_entrance = reaction.kinematics(Elab=Elab)
    kinematics_exit = reaction.kinematics_exit(
        kinematics_entrance, residual_excitation_energy=ExIAS
    )
    k = kinematics_entrance.k
    interaction_range_fm = jitr.utils.interaction_range(reaction.target.A) + 2
    a = k * interaction_range_fm + wavelengths_beyond_range * 2 * np.pi
    channel_radius_fm = a / k
    N = jitr.utils.suggested_basis_size(a, zeros_per_node)
    ws = jitr.xs.quasielastic_pn.Workspace(
        reaction,
        kinematics_entrance,
        kinematics_exit,
        jitr.rmatrix.Solver(N),
        np.asarray(angles_rad, dtype=float),
        lmax,
        channel_radius_fm,
        tmatrix_abs_tol=1e-8,
    )
    return ws, kinematics_entrance, kinematics_exit


class IsobaricAnalogPN(Model):
    """The (p,n) IAS differential cross section in b/sr.

    Parameters
    ----------
    U_p_coulomb, U_p_central, U_p_spin_orbit, U_n_central, U_n_spin_orbit : callable
        ``f(r, *args) -> np.ndarray`` on the radial grid ``r`` (fm), in MeV.
    args_from_params : callable
        ``f(workspace, *values) -> (args_p_coulomb, args_p_central,
        args_p_spin_orbit, args_n_central, args_n_spin_orbit)``.
    params : sequence of Parameter
    lmax, wavelengths_beyond_range, zeros_per_node
        Solver settings, forwarded to :func:`set_up_solver`.
    """

    def __init__(
        self,
        U_p_coulomb: Callable,
        U_p_central: Callable,
        U_p_spin_orbit: Callable,
        U_n_central: Callable,
        U_n_spin_orbit: Callable,
        args_from_params: Callable,
        params,
        *,
        lmax: int = DEFAULT_LMAX,
        wavelengths_beyond_range: float = 2.0,
        zeros_per_node: int = 5,
    ):
        super().__init__(None, params)
        self.potentials = (
            U_p_coulomb,
            U_p_central,
            U_p_spin_orbit,
            U_n_central,
            U_n_spin_orbit,
        )
        self.args_from_params = args_from_params
        self.lmax = lmax
        self.wavelengths_beyond_range = wavelengths_beyond_range
        self.zeros_per_node = zeros_per_node

    def workspace(self, x, meta):
        meta = meta or {}
        try:
            reaction, Elab, ExIAS = (
                meta["reaction"],
                float(meta["Elab"]),
                float(meta["ExIAS"]),
            )
        except KeyError as err:
            raise ValueError(
                f"{type(self).__name__} needs meta[{err.args[0]!r}] to bind: build "
                "the dataset with from_measurement(..., ExIAS=), or pass "
                "meta={'reaction': ..., 'Elab': ..., 'ExIAS': ...}"
            ) from None
        x = np.asarray(x, dtype=float)
        check_angle_grid(x, "x")
        ws, _, _ = set_up_solver(
            reaction,
            Elab,
            ExIAS,
            x,
            lmax=self.lmax,
            wavelengths_beyond_range=self.wavelengths_beyond_range,
            zeros_per_node=self.zeros_per_node,
        )
        return ws

    def bind(self, x, meta=None) -> Predictor:
        ws = self.workspace(x, meta)
        potentials, args_from_params = self.potentials, self.args_from_params
        r = ws.radial_grid()

        def predict(*values):
            args = args_from_params(ws, *values)
            if len(args) != 5:
                raise ValueError(
                    f"args_from_params must return 5 argument tuples, got {len(args)}"
                )
            return ws.xs(*(U(r, *a) for U, a in zip(potentials, args))) / MB_PER_B

        return Predictor(self.params, x, predict)
