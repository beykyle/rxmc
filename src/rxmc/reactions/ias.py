"""
(p,n) isobaric-analog-state differential cross sections from ``jitr``.

:class:`IsobaricAnalogPN` is a :class:`~rxmc.model.Model` that overrides only
``bind``: given a grid of angles and a dataset's kinematics
(``meta["reaction"]``, ``meta["Elab"]``, ``meta["ExIAS"]``) it builds the
``jitr`` quasielastic (p,n) workspace on that grid and returns a
:class:`~rxmc.model.Predictor` for the cross section in b/sr.  The (p,n)
transition is driven by the isovector (Lane) potential ``U1``: by default the
difference between the proton and neutron potentials, or, term by term, any
potential of its own.
"""

from __future__ import annotations

from typing import Callable

import jitr
import numpy as np
from jitr.xs.coupled import Potential

from ..model import Model, Predictor
from ..units import DEFAULT_LMAX, MB_PER_B, check_angle_grid
from .elastic import mesh

__all__ = ["IsobaricAnalogPN", "set_up_solver"]

METHODS = ("dwba", "cc")


def set_up_solver(
    reaction,
    Elab: float,
    ExIAS: float,
    angles_rad: np.ndarray,
    lmax: int | None = DEFAULT_LMAX,
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
        The workspace holds one energy: its results have a leading axis of 1.
    """
    kinematics_entrance = reaction.kinematics(np.array([float(Elab)]))
    kinematics_exit = reaction.kinematics_exit(
        kinematics_entrance, residual_excitation_energy=ExIAS
    )
    radius, N, lmax = mesh(
        reaction, kinematics_entrance, lmax, wavelengths_beyond_range, zeros_per_node
    )
    ws = jitr.xs.quasielastic_pn.Workspace(
        reaction,
        kinematics_entrance,
        kinematics_exit,
        radius,
        lmax,
        N,
        angles=np.asarray(angles_rad, dtype=float),
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
        args_p_spin_orbit, args_n_central, args_n_spin_orbit)``, followed by
        ``(args_U1_central, args_U1_spin_orbit)`` when either ``U1`` callable
        is given (``()`` for one that is not).
    params : sequence of Parameter
    method : {"dwba", "cc"}
        First-order DWBA, or the coupled (Lane) channels.
    U1_central, U1_spin_orbit : callable, optional
        ``f(r, *args)``, the isovector potential driving the transition.  A
        term that is not given is the isovector difference of the proton and
        neutron potentials, :math:`-(U_n - U_p)\\sqrt{|N-Z|}/(N-Z-1)`; pass a
        callable returning zeros to switch a term off.
    lmax, wavelengths_beyond_range, zeros_per_node
        Solver settings, forwarded to :func:`set_up_solver`.  ``lmax=None``
        (the default) takes ``jitr.utils.suggest_lmax`` at the energy.
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
        method: str = "dwba",
        U1_central: Callable | None = None,
        U1_spin_orbit: Callable | None = None,
        lmax: int | None = DEFAULT_LMAX,
        wavelengths_beyond_range: float = 2.0,
        zeros_per_node: int = 5,
    ):
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")
        super().__init__(None, params)
        self.potentials = (
            U_p_coulomb,
            U_p_central,
            U_p_spin_orbit,
            U_n_central,
            U_n_spin_orbit,
        )
        self.U1 = (U1_central, U1_spin_orbit)
        self.method = method
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
        potentials, U1, args_from_params = (
            self.potentials,
            self.U1,
            self.args_from_params,
        )
        n_args = 5 if U1 == (None, None) else 7
        observables = getattr(ws, self.method)
        r = ws.r

        def predict(*values):
            args = args_from_params(ws, *values)
            if len(args) != n_args:
                raise ValueError(
                    f"args_from_params must return {n_args} argument tuples, "
                    f"got {len(args)}"
                )
            cou, pc, pso, nc, nso = (U(r, *a) for U, a in zip(potentials, args[:5]))
            Up = Potential(central=pc, spin_orbit=pso, coulomb=cou)
            Un = Potential(central=nc, spin_orbit=nso)
            isovector = None
            if n_args == 7:
                c, so = (None if U is None else U(r, *a) for U, a in zip(U1, args[5:]))
                isovector = Potential(central=c, spin_orbit=so)
            return np.asarray(observables(Up, Un, isovector).dsdo[0]) / MB_PER_B

        return Predictor(self.params, x, predict, meta)
