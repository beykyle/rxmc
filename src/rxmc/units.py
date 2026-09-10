"""
The one unit registry and the unit contract.

``pint`` refuses to combine quantities from different registries, so every
module that touches units imports :data:`ureg` from here.  The contract:
cross sections are stored internally in b/sr (:data:`XS_UNIT`); ``jitr``
reports cross sections and the Rutherford cross section in mb/sr
(:data:`RUTHERFORD_UNIT`), so model outputs are divided by :data:`MB_PER_B`.
Angles are stored in radians.
"""

import numpy as np
from pint import UnitRegistry

__all__ = [
    "ureg",
    "DEFAULT_LMAX",
    "XS_UNIT",
    "RUTHERFORD_UNIT",
    "MB_PER_B",
    "check_angle_grid",
]

#: The one unit registry for the package.
ureg = UnitRegistry()

#: Default maximum partial wave for the reaction solvers.
DEFAULT_LMAX = 20

#: Internal cross-section unit: every ``y`` in b/sr.
XS_UNIT = ureg.barn / ureg.steradian

#: Unit ``jitr`` reports cross sections (and the Rutherford cross section) in.
RUTHERFORD_UNIT = ureg.millibarn / ureg.steradian

#: Millibarn per barn; divides ``jitr`` output to land in :data:`XS_UNIT`.
MB_PER_B = float((1 * ureg.barn).to(ureg.millibarn).magnitude)


def check_angle_grid(angles_rad: np.ndarray, name: str) -> None:
    """Reject a grid that is not 1-D or not inside ``[0, pi]`` radians."""
    angles_rad = np.asarray(angles_rad)
    if angles_rad.ndim != 1:
        raise ValueError(f"{name} must be 1D, is {angles_rad.ndim}D")
    if angles_rad.size and (angles_rad.min() < 0 or angles_rad.max() > np.pi):
        raise ValueError(f"{name} must be on [0, pi] radians")
