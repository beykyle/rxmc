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

__all__ = ["Dataset", "from_measurement"]


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


# ----------------------------------------------------------------------------
# EXFOR measurements
# ----------------------------------------------------------------------------

_QUANTITY_KIND = {
    "dXS/dA": "differential",
    "dXS/dRuth": "dimensionless",
    "Ay": "dimensionless",
}


def from_measurement(
    measurement, *, reaction=None, quantity=None, ExIAS=None
) -> Dataset:
    """A :class:`Dataset` from an ``exfor_tools`` measurement, in internal units.

    Reads ``x`` (degrees), ``y``, ``Einc``, ``quantity``, ``y_units``,
    ``statistical_err``, ``systematic_norm_err``, ``systematic_offset_err`` and
    ``subentry`` from ``measurement`` (any object with those attributes).  Angles
    are stored in radians, cross sections in b/sr, ratios and analysing powers
    as they are.  Every dimensionful error (statistical, absolute offset) is
    converted with the data; the fractional normalisation error passes through
    untouched.  The kinematics a reaction model needs to bind land in
    ``meta``: ``reaction``, ``Elab``, ``quantity``, ``k``, ``eta`` and, for the
    (p,n) channel, ``ExIAS``.

    Parameters
    ----------
    measurement : object
        An ``exfor_tools.distribution.Distribution`` or anything shaped like it.
    reaction : jitr.reactions.Reaction, optional
        Needed for the kinematics in ``meta`` and for any conversion between
        ``dXS/dA`` and ``dXS/dRuth`` (the Rutherford cross section is a closed
        form of the kinematics).
    quantity : {"dXS/dA", "dXS/dRuth", "Ay"}, optional
        The quantity the dataset should hold; defaults to the measured one.
    ExIAS : float, optional
        Excitation energy of the isobaric analog state (MeV), for (p,n) data.
    """
    from .units import MB_PER_B, check_angle_grid, parse_unit

    measured = measurement.quantity
    target = measured if quantity is None else quantity
    for q in (measured, target):
        if q not in _QUANTITY_KIND:
            raise ValueError(
                f"unknown quantity {q!r}; expected one of {list(_QUANTITY_KIND)}"
            )
    factor, kind = parse_unit(measurement.y_units)
    if kind != _QUANTITY_KIND[measured]:
        raise ValueError(
            f"measurement quantity {measured!r} needs {_QUANTITY_KIND[measured]} units, "
            f"got {measurement.y_units!r}"
        )
    x = np.deg2rad(np.asarray(measurement.x, dtype=float))
    label = getattr(measurement, "subentry", None) or ""
    check_angle_grid(x, f"x of {label or 'measurement'}")
    Elab = float(measurement.Einc)

    meta = {"quantity": target, "Elab": Elab, "subentry": label or None}
    kinematics = None
    if reaction is not None:
        kinematics = reaction.kinematics(Elab)
        meta.update(reaction=reaction, k=float(kinematics.k), eta=float(kinematics.eta))
    if ExIAS is not None:
        meta["ExIAS"] = float(ExIAS)

    if measured == target:
        norm = factor  # into b/sr for a cross section, 1 for a ratio
    elif {measured, target} == {"dXS/dA", "dXS/dRuth"}:
        if kinematics is None:
            raise ValueError(
                f"converting {measured!r} to {target!r} needs the Rutherford cross "
                "section: pass reaction="
            )
        if not kinematics.eta > 0:
            raise ValueError(
                f"converting {measured!r} to {target!r} needs a charged projectile "
                f"(eta = {kinematics.eta})"
            )
        from .reactions.elastic import rutherford

        ruth_b = rutherford(kinematics, x) / MB_PER_B
        norm = factor / ruth_b if measured == "dXS/dA" else ruth_b
    else:
        raise ValueError(
            f"cannot convert measurement quantity {measured!r} to {target!r}"
        )

    def scaled(v):
        return None if v is None else np.asarray(v, dtype=float) * norm

    y_err = measurement.statistical_err
    return Dataset(
        x,
        np.asarray(measurement.y, dtype=float) * norm,
        scaled(np.zeros_like(x) if y_err is None else y_err),
        norm_err=measurement.systematic_norm_err,
        offset_err=scaled(measurement.systematic_offset_err),
        label=label,
        meta=meta,
    )
