"""
Total and reaction cross sections as functions of energy, from ``jitr``.

:class:`IntegratedXS` binds to a grid of incident laboratory energies (MeV):
one angle-free ``jitr`` elastic workspace solves every energy of the grid in a
single call, so a whole excitation function costs about as much as one
energy.  The potentials are evaluated on all energies at once, as ``(NE, N)``
arrays, or as ``(N,)`` arrays when they do not depend on energy.
"""

from __future__ import annotations

import jitr
import numpy as np
from scipy.interpolate import CubicSpline

from ..model import Predictor
from ..units import MB_PER_B, check_energy_grid
from .elastic import OpticalModel, mesh

__all__ = ["IntegratedXS"]


class IntegratedXS(OpticalModel):
    r"""The total or reaction cross section (b) on a grid of lab energies.

    Parameters
    ----------
    quantity : {"total", "reaction"}
        ``"total"`` (neutral projectiles only: the Coulomb amplitude makes it
        infinite otherwise) or ``"reaction"`` (the absorption cross section,
        the nonelastic cross section of nucleon scattering).
    central, spin_orbit, args_from_params, params, coulomb
        As for :class:`~rxmc.reactions.ElasticXS`; ``args_from_params(ws,
        *values)`` sees the workspace over all energies, whose
        ``ws.kinematics.Elab`` is the grid.
    lmax, wavelengths_beyond_range, zeros_per_node
        Solver settings, applied at the highest energy of the grid (see
        :func:`~rxmc.reactions.elastic.mesh`); ``wavelengths_beyond_range``
        defaults to 3 and ``lmax=None`` takes ``jitr.utils.suggest_lmax`` for
        the grid.
    n_energies : int, optional
        Solve on this many log-spaced energies spanning ``x`` and interpolate
        (a cubic spline in :math:`\log E`) to ``x``, when ``x`` has more
        points.  The cross section is smooth, so a few dozen energies serve an
        excitation function measured at thousands; the cost of an evaluation
        no longer grows with the number of data points.

    :meth:`~rxmc.reactions.elastic.OpticalModel.from_omp` builds the model
    from a ``jitr`` optical model, evaluated on every energy at once.
    """

    QUANTITIES = ("total", "reaction")
    # one radius serves the whole grid: three wavelengths at the highest energy
    # keep the lowest energies converged to ~1e-4 (two give ~1e-3)
    WAVELENGTHS_BEYOND_RANGE = 3.0

    def __init__(self, *args, n_energies: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_energies = n_energies

    def solver_energies(self, x) -> np.ndarray:
        """The energies the workspace solves on: ``x``, or ``n_energies``
        log-spaced energies spanning it."""
        x = np.asarray(x, dtype=float)
        if self.n_energies is None or x.size <= self.n_energies:
            return x
        return np.geomspace(x.min(), x.max(), self.n_energies)

    def workspace(self, x, meta):
        """The angle-free ``jitr`` elastic workspace over the energies ``x``
        (over :meth:`solver_energies` of them)."""
        meta = meta or {}
        if "reaction" not in meta:
            raise ValueError(
                f"{type(self).__name__} needs meta['reaction'] to bind: build the "
                "dataset with from_measurement, or pass meta={'reaction': ...}"
            )
        reaction = meta["reaction"]
        x = np.asarray(x, dtype=float)
        check_energy_grid(x, "x")
        if self.quantity == "total" and reaction.projectile.Z * reaction.target.Z:
            raise ValueError(
                "the total cross section of a charged projectile is infinite: "
                "compare the reaction cross section instead"
            )
        x = self.solver_energies(x)
        key = (id(reaction), tuple(x), *self.settings())
        if key not in self._cache:
            kinematics = reaction.kinematics(x)
            radius, N, lmax = mesh(reaction, kinematics, *self.settings())
            self._cache[key] = jitr.xs.elastic.Workspace(
                reaction, kinematics, radius, lmax, N
            )
        return self._cache[key]

    def bind(self, x, meta=None) -> Predictor:
        ws = self.workspace(x, meta)
        field = "t" if self.quantity == "total" else "rxn"

        x = np.asarray(x, dtype=float)
        grid = np.asarray(ws.kinematics.Elab)
        interpolate = grid.shape != x.shape or np.any(grid != x)

        def predict(*values):
            xs = ws.xs(self.potential(ws, values))
            y = np.asarray(getattr(xs, field)) / MB_PER_B
            return CubicSpline(np.log(grid), y)(np.log(x)) if interpolate else y

        return Predictor(self.params, x, predict, meta)
