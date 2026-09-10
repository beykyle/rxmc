"""
The dataset: pure data.

A :class:`Dataset` holds an independent variable, a dependent variable, its
statistical error, the measurement's *reported* systematic magnitudes as
inert metadata, and whatever kinematics a model needs to bind to it.  It has
no comparison transform, no mask and no solver state: those belong to the
:class:`~rxmc.constraint.Comparison`, the :class:`~rxmc.constraint.Constraint`
and the :class:`~rxmc.model.Model` respectively.

``x`` is opaque to the library.  A model may ignore it and close over a
predictor on another grid (recipe 20), and a covariance term sees it only
through its own ``coords`` transform.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

__all__ = ["Dataset"]


def _error_spec(value, n, name):
    """``None``, a float, or a float array of shape ``(n,)``."""
    if value is None:
        return None
    if np.ndim(value) == 0:
        return float(value)
    v = np.asarray(value, dtype=float)
    if v.shape != (n,):
        raise ValueError(
            f"{name} must be a scalar or have shape ({n},), got shape {v.shape}"
        )
    return v


@dataclass(eq=False, frozen=True)
class Dataset:
    """Experimental data in physical units.

    Parameters
    ----------
    x : array_like
        Independent variable; any dtype or shape whose first dimension is the
        number of points.  Angles in radians for reaction data.
    y : array_like
        Dependent variable, shape ``(n,)``.
    y_err : array_like
        Statistical (uncorrelated) error on ``y``, shape ``(n,)``, non-negative.
    norm_err : float or array_like, optional
        Reported *fractional* normalisation uncertainty; inert until
        :meth:`~rxmc.constraint.Comparison.reported_terms` asks for it.
    offset_err : float or array_like, optional
        Reported *absolute* offset uncertainty, in the units of ``y``; inert
        likewise.
    label : str, optional
        Human-readable identifier used in error messages.
    meta : mapping, optional
        Kinematics and provenance a model or a term may read: ``reaction``,
        ``Elab``, ``ExIAS``, ``quantity``, ``k``, ...  Copied on construction.
    """

    x: Any
    y: Any
    y_err: Any
    norm_err: Any = None
    offset_err: Any = None
    label: str = ""
    meta: Mapping = field(default_factory=dict, repr=False)

    def __post_init__(self):
        x = np.asarray(self.x)
        y = np.asarray(self.y, dtype=float)
        if y.ndim != 1:
            raise ValueError(f"y must be 1-D, got shape {y.shape}")
        n = y.shape[0]
        if x.ndim == 0 or x.shape[0] != n:
            raise ValueError(
                f"x must have {n} points along its first dimension, got shape "
                f"{x.shape}"
            )
        y_err = np.asarray(self.y_err, dtype=float)
        if y_err.shape != (n,):
            raise ValueError(f"y_err must have shape ({n},), got {y_err.shape}")
        if np.any(y_err < 0):
            raise ValueError("y_err must be non-negative")
        set_ = object.__setattr__
        set_(self, "x", x)
        set_(self, "y", y)
        set_(self, "y_err", y_err)
        set_(self, "norm_err", _error_spec(self.norm_err, n, "norm_err"))
        set_(self, "offset_err", _error_spec(self.offset_err, n, "offset_err"))
        set_(self, "meta", dict(self.meta))

    @property
    def n(self) -> int:
        """Number of points."""
        return self.y.shape[0]

    def __repr__(self):
        label = f"{self.label!r}, " if self.label else ""
        return f"Dataset({label}n={self.n})"
