"""
Elastic differential cross sections from a ``jitr`` optical-model solver.

:class:`ElasticXS` is a :class:`~rxmc.model.Model` that overrides only
``bind``: given a grid of angles and a dataset's kinematics
(``meta["reaction"]``, ``meta["Elab"]``) it builds the ``jitr`` R-matrix
workspace on that grid and returns a :class:`~rxmc.model.Predictor` that
evaluates the potentials on the radial grid, solves, and extracts one of
``"dXS/dA"`` (b/sr), ``"dXS/dRuth"`` or ``"Ay"``.  The model owns the
solver; the data does not.

The expensive part of a workspace (the mesh, the partial waves and their
asymptotics) depends only on the kinematics and the solver settings, so it is
cached per model instance and shared by every grid at one energy: the data
grid, a plotting grid, a held-out view (``Workspace.at_angles``).  The cache is
dropped on pickling; a bound predictor carries its own workspace and pickles
with ``dill``.
"""

from __future__ import annotations

from typing import Callable, Mapping

import jitr
import numpy as np
from jitr.xs.coupled import Potential

from ..model import Model, Parameter, Predictor
from ..units import DEFAULT_LMAX, MB_PER_B, check_angle_grid

__all__ = [
    "ElasticXS",
    "OpticalModel",
    "mesh",
    "omp_parameters",
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


def momentum_transfer(angles_rad, k) -> np.ndarray:
    r"""Momentum transfer :math:`q = 2k\sin(\theta/2)` (fm⁻¹) on the angles.

    For a :func:`~rxmc.terms.kernel` term in :math:`q`-space pass it as the
    term's coordinate transform: ``coords=lambda x: momentum_transfer(x, k)``.
    """
    return 2.0 * float(k) * np.sin(np.asarray(angles_rad, dtype=float) / 2.0)


def mesh(reaction, kinematics, lmax, wavelengths_beyond_range, zeros_per_node):
    """Channel radius (fm), basis size and ``lmax`` for one or more energies.

    The radius reaches ``wavelengths_beyond_range`` wavelengths past the
    interaction range at the highest energy (the R-matrix matching is exact
    anywhere outside the potential, so lower energies need no more room), and
    the basis resolves that energy.  ``lmax=None`` takes
    ``jitr.utils.suggest_lmax``: every partial wave up to ``lmax`` is solved,
    so it must be large enough for the highest energy.
    """
    k = np.atleast_1d(kinematics.k)
    interaction_range_fm = jitr.utils.interaction_range(reaction.target.A) + 2
    radius = interaction_range_fm + wavelengths_beyond_range * 2 * np.pi / k.max()
    N = jitr.utils.suggested_basis_size(k.max() * radius, zeros_per_node)
    if lmax is None:
        lmax = jitr.utils.suggest_lmax(k, radius)
    return radius, N, lmax


def _basis(reaction, Elab, lmax, wavelengths_beyond_range, zeros_per_node):
    """The angle-free elastic workspace and kinematics for one energy."""
    kinematics = reaction.kinematics(np.array([float(Elab)]))
    radius, N, lmax = mesh(
        reaction, kinematics, lmax, wavelengths_beyond_range, zeros_per_node
    )
    return jitr.xs.elastic.Workspace(reaction, kinematics, radius, lmax, N), kinematics


def set_up_solver(
    reaction,
    Elab: float,
    angles_rad: np.ndarray,
    lmax: int | None = DEFAULT_LMAX,
    wavelengths_beyond_range: float = 2.0,
    zeros_per_node: int = 5,
):
    """Set up a ``jitr`` elastic workspace for a reaction at one energy.

    Parameters
    ----------
    reaction : jitr.reactions.Reaction
    Elab : float
        Laboratory energy in MeV.
    angles_rad : np.ndarray
        Angles in radians the observables are wanted on.
    lmax : int or None
        Maximum partial wave; ``None`` for ``jitr.utils.suggest_lmax``.
    wavelengths_beyond_range : float
        Number of wavelengths beyond the interaction range used to set the
        channel radius.
    zeros_per_node : int
        Basis-function zeros per node in the R-matrix solver.

    Returns
    -------
    (jitr.xs.elastic.Workspace, kinematics)
        The workspace holds one energy: its results have a leading axis of 1.
    """
    ws, kinematics = _basis(
        reaction, Elab, lmax, wavelengths_beyond_range, zeros_per_node
    )
    return ws.at_angles(np.asarray(angles_rad, dtype=float)), kinematics


def extract_dXS_dA(xs, ws) -> np.ndarray:
    """dXS/dA in b/sr (``jitr`` reports mb/sr)."""
    return np.asarray(xs.dsdo[0]) / MB_PER_B


def extract_dXS_dRuth(xs, ws) -> np.ndarray:
    """dXS/dRuth (dimensionless)."""
    return np.asarray(xs.dsdo[0] / ws.rutherford[0])


def extract_Ay(xs, ws) -> np.ndarray:
    """The analysing power (dimensionless)."""
    return np.asarray(xs.Ay[0])


_EXTRACT = {"dXS/dA": extract_dXS_dA, "dXS/dRuth": extract_dXS_dRuth, "Ay": extract_Ay}


def omp_parameters(omp, params=None, prefix: str = "") -> list[Parameter]:
    """One :class:`~rxmc.Parameter` per entry of ``omp.params``: from a
    sequence (in that order) or a mapping of name to parameter, or new
    prior-less parameters named ``prefix + name``."""
    if params is None:
        return [Parameter(prefix + name) for name in omp.params]
    if isinstance(params, Mapping):
        missing = [name for name in omp.params if name not in params]
        if missing:
            raise ValueError(f"params has no entry for {missing}")
        return [params[name] for name in omp.params]
    params = list(params)
    if len(params) != len(omp.params):
        raise ValueError(
            f"{type(omp).__name__} has {len(omp.params)} parameters, got {len(params)}"
        )
    return params


class OpticalModel(Model):
    """A ``jitr`` model driven by central, spin-orbit and Coulomb potentials:
    the machinery :class:`ElasticXS` and :class:`~rxmc.reactions.IntegratedXS`
    share.

    The potentials come either from callables ``f(r, *args)`` with the
    arguments from ``args_from_params(workspace, *values)``, or, through
    :meth:`from_omp`, from a ``jitr`` optical model evaluated with the
    workspace kinematics.  Subclasses set the observables they can extract,
    :attr:`QUANTITIES`, and override ``bind``.
    """

    QUANTITIES: tuple[str, ...] = ()
    #: default of ``wavelengths_beyond_range``
    WAVELENGTHS_BEYOND_RANGE = 2.0

    def __init__(
        self,
        quantity: str,
        central: Callable | None,
        spin_orbit: Callable | None,
        args_from_params: Callable | None,
        params,
        coulomb: Callable | None = None,
        *,
        lmax: int | None = DEFAULT_LMAX,
        wavelengths_beyond_range: float | None = None,
        zeros_per_node: int = 5,
    ):
        if quantity not in self.QUANTITIES:
            raise ValueError(
                f"quantity must be one of {self.QUANTITIES}, got {quantity!r}"
            )
        super().__init__(None, params)
        self.quantity = quantity
        self.central = central
        self.spin_orbit = spin_orbit
        self.coulomb = coulomb
        self.args_from_params = args_from_params
        self.omp = None
        self.lmax = lmax
        if wavelengths_beyond_range is None:
            wavelengths_beyond_range = self.WAVELENGTHS_BEYOND_RANGE
        self.wavelengths_beyond_range = wavelengths_beyond_range
        self.zeros_per_node = zeros_per_node
        self._cache: dict = {}

    @classmethod
    def from_omp(cls, quantity: str, omp, params=None, **settings):
        """The model with the potentials of a ``jitr`` optical model.

        Parameters
        ----------
        quantity : str
            As for the constructor.
        omp : jitr.optical_potentials.omp.SingleChannelOpticalModel
            Evaluated as ``omp(r, reaction, kinematics, *values)`` with the
            workspace mesh and kinematics, e.g.
            ``jitr.optical_potentials.kduq.KDUQ(projectile)``.
        params : sequence of Parameter, or mapping of name to Parameter, optional
            One per entry of ``omp.params``, in that order, or a mapping from
            those names to parameters.  Pass the same mapping to every model
            that shares the potential (e.g. KDUQ for protons, behind elastic,
            reaction and (p,n) data), so that they share the
            :class:`~rxmc.Parameter` objects.  Different potentials need
            different parameters even where their names coincide: KDUQ for
            neutrons and for protons reuse names such as ``Ef_0`` and
            ``w1_0`` for different values, so give each its own mapping (with
            parameters named apart, e.g. ``Parameter("n_" + name)``).
            Defaults to new prior-less parameters named after ``omp.params``.
        **settings
            Solver settings (``lmax``, ``wavelengths_beyond_range``,
            ``zeros_per_node``).
        """
        model = cls(quantity, None, None, None, omp_parameters(omp, params), **settings)
        model.omp = omp
        return model

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache"] = {}
        return state

    def potential(self, ws, values) -> Potential:
        """The ``jitr`` potential on the workspace mesh at ``values``."""
        r = ws.r
        if self.omp is not None:
            central, spin_orbit, coulomb = self.omp(
                r, ws.reaction, ws.kinematics, *values
            )
            return Potential(
                central=central,
                spin_orbit=spin_orbit if ws.s > 0 else None,
                coulomb=coulomb if np.any(coulomb) else None,
            )
        assert self.args_from_params is not None and self.central is not None
        args = self.args_from_params(ws, *values)
        if len(args) == 2:
            (c_args, so_args), cou_args = args, ()
        elif len(args) == 3:
            c_args, so_args, cou_args = args
        else:
            raise ValueError(
                f"args_from_params must return 2 or 3 argument tuples, got {len(args)}"
            )
        spin_orbit = None if self.spin_orbit is None else self.spin_orbit(r, *so_args)
        if ws.s == 0 and spin_orbit is not None and not np.any(spin_orbit):
            spin_orbit = None  # a zero spin-orbit term, for a spin-0 projectile
        return Potential(
            central=self.central(r, *c_args),
            spin_orbit=spin_orbit,
            coulomb=None if self.coulomb is None else self.coulomb(r, *cou_args),
        )

    def settings(self) -> tuple:
        """``(lmax, wavelengths_beyond_range, zeros_per_node)``."""
        return self.lmax, self.wavelengths_beyond_range, self.zeros_per_node


class ElasticXS(OpticalModel):
    """Elastic differential cross section, ratio to Rutherford, or analysing power.

    Parameters
    ----------
    quantity : {"dXS/dA", "dXS/dRuth", "Ay"}
    central : callable
        ``f(r, *args) -> np.ndarray``, the central potential on the radial grid
        ``r`` (fm), in MeV.
    spin_orbit : callable or None
        ``f(r, *args) -> np.ndarray``, the spin-orbit potential; ``None`` for a
        spin-orbit-free model.
    args_from_params : callable
        ``f(workspace, *values) -> (central_args, spin_orbit_args)`` or
        ``-> (central_args, spin_orbit_args, coulomb_args)``: the argument
        tuples for the potential callables at the sampled values.
    params : sequence of Parameter
    coulomb : callable, optional
        ``f(r, *args) -> np.ndarray``, the Coulomb potential.  When ``None``
        the Coulomb interaction inside the channel radius must be folded into
        ``central``.
    lmax, wavelengths_beyond_range, zeros_per_node
        Solver settings, forwarded to :func:`set_up_solver`.  ``lmax=None``
        (the default) takes ``jitr.utils.suggest_lmax`` at the energy.

    :meth:`~OpticalModel.from_omp` builds the model from a ``jitr`` optical
    model instead of potential callables.
    """

    QUANTITIES = QUANTITIES

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
        """The ``jitr`` elastic workspace on ``x`` for this dataset's kinematics."""
        reaction, Elab = self._kinematics_of(meta)
        key = (id(reaction), Elab, *self.settings())
        if key not in self._cache:
            self._cache[key] = _basis(reaction, Elab, *self.settings())
        base, _ = self._cache[key]
        x = np.asarray(x, dtype=float)
        check_angle_grid(x, "x")
        return base.at_angles(x)

    def bind(self, x, meta=None) -> Predictor:
        ws = self.workspace(x, meta)
        if self.quantity == "dXS/dRuth" and ws.rutherford is None:
            raise ValueError(
                "dXS/dRuth needs the Rutherford cross section, which a neutral "
                "projectile does not have: compare dXS/dA instead"
            )
        extract = _EXTRACT[self.quantity]

        def predict(*values):
            return extract(ws.xs(self.potential(ws, values)), ws)

        return Predictor(self.params, x, predict, meta)
