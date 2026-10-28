"""
Elastic differential cross sections from the ``jitr`` optical-model solver.

:class:`ElasticXS` is a :class:`~rxmc.model.Model` that overrides only
``bind``: given a grid of angles and a dataset's kinematics
(``meta["reaction"]``, ``meta["Elab"]``) it builds the ``jitr`` R-matrix
workspace on that grid and returns a :class:`~rxmc.model.Predictor` that
evaluates the potentials on the radial mesh, solves, and extracts one of
``"dXS/dA"`` (b/sr), ``"dXS/dRuth"`` or ``"Ay"``.  The model owns the
solver; the data does not.

``jitr``'s solver is written in JAX, so the prediction is differentiable in
the potential parameters: the potentials and ``args_from_params`` must be
written with ``jax.numpy`` (``jitr``'s potential forms are).  The spin-orbit
potential multiplies ``l.sigma`` (``2 l.s``) for a spin-1/2 projectile, as in
``jitr``.

The expensive part of a workspace (the mesh, the asymptotics and the
compiled solve) depends only on the kinematics and the solver settings, so it
is cached per model instance and shared by every grid at one energy: the data
grid, a plotting grid, a held-out view (``Workspace.at_angles``).  The cache is
dropped on pickling; a bound predictor carries its own workspace and pickles
with ``dill``.
"""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp
import jitr
import numpy as np
from jitr.xs.coupled import Potential

from ..model import Model, Predictor
from ..units import DEFAULT_LMAX, MB_PER_B, check_angle_grid

__all__ = [
    "ElasticXS",
    "set_up_solver",
    "rutherford",
    "momentum_transfer",
    "extract_dXS_dA",
    "extract_dXS_dRuth",
    "extract_Ay",
]

QUANTITIES = ("dXS/dA", "dXS/dRuth", "Ay")


def rutherford(kinematics, angles_rad) -> np.ndarray:
    r"""The Rutherford cross section in mb/sr on ``angles_rad``.

    :math:`10\,\eta^2 / (4 k^2 \sin^4(\theta/2))` with ``k`` in fm⁻¹; a closed
    form of the kinematics, so no workspace is needed.  Zero for a neutral
    projectile (``eta == 0``).
    """
    angles_rad = np.asarray(angles_rad, dtype=float)
    sin2 = np.sin(angles_rad / 2.0) ** 2
    return 10.0 * kinematics.eta**2 / (4.0 * kinematics.k**2 * sin2**2)


def momentum_transfer(angles_rad, k):
    r"""Momentum transfer :math:`q = 2k\sin(\theta/2)` (fm⁻¹) on the angles.

    For a :func:`~rxmc.terms.kernel` term in :math:`q`-space pass it as the
    term's coordinate transform: ``coords=lambda x: momentum_transfer(x, k)``.
    """
    return 2.0 * float(k) * jnp.sin(jnp.asarray(angles_rad, dtype=float) / 2.0)


def _basis(reaction, Elab, lmax, wavelengths_beyond_range, zeros_per_node):
    """The angle-free workspace and kinematics for one reaction at one energy."""
    kinematics = reaction.kinematics(Elab)
    k = kinematics.k
    interaction_range_fm = jitr.utils.interaction_range(reaction.target.A) + 2
    a = k * interaction_range_fm + wavelengths_beyond_range * 2 * np.pi
    channel_radius_fm = a / k
    N = jitr.utils.suggested_basis_size(a, zeros_per_node)
    ws = jitr.xs.elastic.Workspace(reaction, kinematics, channel_radius_fm, lmax, N)
    return ws, kinematics


def set_up_solver(
    reaction,
    Elab: float,
    angles_rad: np.ndarray,
    lmax: int = DEFAULT_LMAX,
    wavelengths_beyond_range: float = 2.0,
    zeros_per_node: int = 5,
):
    """Set up a ``jitr`` differential workspace for a reaction at one energy.

    Parameters
    ----------
    reaction : jitr.reactions.Reaction
    Elab : float
        Laboratory energy in MeV.
    angles_rad : np.ndarray
        Angles in radians the observables are wanted on.
    lmax : int
        Maximum partial wave.
    wavelengths_beyond_range : float
        Number of wavelengths beyond the interaction range used to set the
        channel radius.
    zeros_per_node : int
        Basis-function zeros per node in the R-matrix solver.

    Returns
    -------
    (jitr.xs.elastic.Workspace, jitr.utils.kinematics.ChannelKinematics)
    """
    ws, kinematics = _basis(
        reaction, Elab, lmax, wavelengths_beyond_range, zeros_per_node
    )
    return ws.at_angles(np.asarray(angles_rad, dtype=float)), kinematics


# jitr results carry a leading energy axis; a workspace here has one energy


def extract_dXS_dA(xs, ws):
    """dXS/dA in b/sr (``jitr`` reports mb/sr)."""
    return xs.dsdo[0] / MB_PER_B


def extract_dXS_dRuth(xs, ws):
    """dXS/dRuth (dimensionless)."""
    return xs.dsdo[0] / ws.rutherford[0]


def extract_Ay(xs, ws):
    """The analysing power (dimensionless)."""
    return xs.Ay[0]


_EXTRACT = {"dXS/dA": extract_dXS_dA, "dXS/dRuth": extract_dXS_dRuth, "Ay": extract_Ay}


class ElasticXS(Model):
    """Elastic differential cross section, ratio to Rutherford, or analysing power.

    Parameters
    ----------
    quantity : {"dXS/dA", "dXS/dRuth", "Ay"}
    central : callable
        ``f(r, *args) -> array``, the central potential on the radial mesh
        ``r`` (fm), in MeV, written with ``jax.numpy``.
    spin_orbit : callable or None
        ``f(r, *args) -> array``, the spin-orbit potential (it multiplies
        ``l.sigma`` for a spin-1/2 projectile); ``None`` for a spin-orbit-free
        model, and for a spin-0 projectile.
    args_from_params : callable
        ``f(workspace, *values) -> (central_args, spin_orbit_args)`` or
        ``-> (central_args, spin_orbit_args, coulomb_args)``: the argument
        tuples for the potential callables at the sampled values.
    params : sequence of Parameter
    coulomb : callable, optional
        ``f(r, *args) -> array``, the Coulomb potential.  When ``None``
        the Coulomb interaction inside the channel radius must be folded into
        ``central``.
    lmax, wavelengths_beyond_range, zeros_per_node
        Solver settings, forwarded to :func:`set_up_solver`.
    """

    def __init__(
        self,
        quantity: str,
        central: Callable,
        spin_orbit: Callable | None,
        args_from_params: Callable,
        params,
        coulomb: Callable | None = None,
        *,
        lmax: int = DEFAULT_LMAX,
        wavelengths_beyond_range: float = 2.0,
        zeros_per_node: int = 5,
    ):
        if quantity not in QUANTITIES:
            raise ValueError(f"quantity must be one of {QUANTITIES}, got {quantity!r}")
        super().__init__(None, params)
        self.quantity = quantity
        self.central = central
        self.spin_orbit = spin_orbit
        self.coulomb = coulomb
        self.args_from_params = args_from_params
        self.lmax = lmax
        self.wavelengths_beyond_range = wavelengths_beyond_range
        self.zeros_per_node = zeros_per_node
        self._cache: dict = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache"] = {}
        return state

    def _kinematics_of(self, meta):
        meta = meta or {}
        try:
            return meta["reaction"], float(meta["Elab"])
        except KeyError as err:
            raise ValueError(
                f"{type(self).__name__} needs meta[{err.args[0]!r}] to bind: build "
                "the dataset with from_measurement, or pass meta={'reaction': ..., "
                "'Elab': ...}"
            ) from None

    def workspace(self, x, meta):
        """The ``jitr`` workspace on the angles ``x`` for this dataset's kinematics."""
        reaction, Elab = self._kinematics_of(meta)
        key = (
            id(reaction),
            Elab,
            self.lmax,
            self.wavelengths_beyond_range,
            self.zeros_per_node,
        )
        if key not in self._cache:
            self._cache[key] = _basis(
                reaction,
                Elab,
                self.lmax,
                self.wavelengths_beyond_range,
                self.zeros_per_node,
            )
        basis, _ = self._cache[key]
        x = np.asarray(x, dtype=float)
        check_angle_grid(x, "x")
        return basis.at_angles(x)

    def bind(self, x, meta=None) -> Predictor:
        ws = self.workspace(x, meta)
        if self.quantity == "dXS/dRuth" and ws.rutherford is None:
            raise ValueError(
                "dXS/dRuth needs the Rutherford cross section, which a neutral "
                "projectile does not have: compare dXS/dA instead"
            )
        extract = _EXTRACT[self.quantity]
        central, spin_orbit, coulomb = self.central, self.spin_orbit, self.coulomb
        args_from_params = self.args_from_params
        r = ws.r

        def predict(*values):
            args = args_from_params(ws, *values)
            if len(args) == 2:
                (c_args, so_args), cou_args = args, ()
            elif len(args) == 3:
                c_args, so_args, cou_args = args
            else:
                raise ValueError(
                    "args_from_params must return 2 or 3 argument tuples, "
                    f"got {len(args)}"
                )
            xs = ws.xs(
                Potential(
                    central=central(r, *c_args),
                    spin_orbit=None if spin_orbit is None else spin_orbit(r, *so_args),
                    coulomb=None if coulomb is None else coulomb(r, *cou_args),
                )
            )
            return extract(xs, ws)

        return Predictor(self.params, x, predict, meta)
