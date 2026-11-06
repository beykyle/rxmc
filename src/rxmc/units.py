"""
The unit contract, without a unit library.

Cross sections are stored internally in b/sr (:data:`XS_UNIT`); ``jitr``
reports cross sections and the Rutherford cross section in mb/sr
(:data:`RUTHERFORD_UNIT`), so model outputs are divided by :data:`MB_PER_B`.
Angles are stored in radians.

Measurement unit labels come from a small fixed vocabulary: ``x4i3`` converts
every EXFOR cross section to barns while parsing and ``exfor_tools`` labels the
result ``"barns/ster"``, ``"b"`` or ``"unitless"``.  :func:`parse_unit` maps
that vocabulary (plus the obvious spellings) to a factor into the internal
unit and a quantity kind, and rejects anything else loudly.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "DEFAULT_LMAX",
    "XS_UNIT",
    "RUTHERFORD_UNIT",
    "MB_PER_B",
    "parse_unit",
    "check_angle_grid",
]

#: Default maximum partial wave for the reaction solvers.
DEFAULT_LMAX = 20

#: Internal cross-section unit: every ``y`` in b/sr.
XS_UNIT = "b/sr"

#: Unit ``jitr`` reports cross sections (and the Rutherford cross section) in.
RUTHERFORD_UNIT = "mb/sr"

#: Millibarn per barn; divides ``jitr`` output to land in :data:`XS_UNIT`.
MB_PER_B = 1000.0

# label (lower case, no spaces) -> (factor into the internal unit, kind)
_UNITS = {
    # differential cross sections, internal unit b/sr
    "barns/ster": (1.0, "differential"),
    "barn/steradian": (1.0, "differential"),
    "b/sr": (1.0, "differential"),
    "millibarn/steradian": (1e-3, "differential"),
    "mb/sr": (1e-3, "differential"),
    "microbarn/steradian": (1e-6, "differential"),
    "micro-b/sr": (1e-6, "differential"),
    "ub/sr": (1e-6, "differential"),
    # integral cross sections, internal unit b
    "barns": (1.0, "integral"),
    "barn": (1.0, "integral"),
    "b": (1.0, "integral"),
    "millibarn": (1e-3, "integral"),
    "mb": (1e-3, "integral"),
    "microbarn": (1e-6, "integral"),
    "micro-b": (1e-6, "integral"),
    "ub": (1e-6, "integral"),
    # ratios and analysing powers
    "no-dim": (1.0, "dimensionless"),
    "unitless": (1.0, "dimensionless"),
    "dimensionless": (1.0, "dimensionless"),
    "": (1.0, "dimensionless"),
}


def parse_unit(label: str) -> tuple[float, str]:
    """``(factor, kind)`` for a measurement unit label.

    ``factor`` multiplies a value in ``label`` to give the internal unit of its
    ``kind``: b/sr for ``"differential"``, b for ``"integral"``, and 1 for
    ``"dimensionless"``.  Matching ignores case and spaces.

    Raises
    ------
    ValueError
        For a label outside the vocabulary, listing what is accepted.
    """
    key = "".join(str(label).split()).lower()
    try:
        return _UNITS[key]
    except KeyError:
        raise ValueError(
            f"unknown unit label {label!r}; accepted labels are "
            f"{sorted(k for k in _UNITS if k)}"
        ) from None


def check_angle_grid(angles_rad: np.ndarray, name: str) -> None:
    """Reject a grid that is not 1-D, not finite, or not inside ``[0, pi]`` radians."""
    angles_rad = np.asarray(angles_rad)
    if angles_rad.ndim != 1:
        raise ValueError(f"{name} must be 1D, is {angles_rad.ndim}D")
    if not np.all(np.isfinite(angles_rad)):
        raise ValueError(f"{name} must be finite")
    if angles_rad.size and (angles_rad.min() < 0 or angles_rad.max() > np.pi):
        raise ValueError(f"{name} must be on [0, pi] radians")
