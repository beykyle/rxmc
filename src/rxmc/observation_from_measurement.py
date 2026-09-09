"""
Helpers shared by the reaction-observation classes.

The reaction observations (:class:`~rxmc.elastic_diffxs_observation.ElasticDifferentialXSObservation`
and :class:`~rxmc.ias_pn_observation.IsobaricAnalogPNObservation`) are now plain
:class:`~rxmc.observation.Observation` subclasses carrying **statistical error
only**; any correlated systematic is composed explicitly as a
:class:`~rxmc.covariance.Term` in the :class:`~rxmc.constraint.Constraint`.  This
module holds what the reaction observations *and* the reaction models share:
the angle-grid validation, the single ``pint`` unit registry, and the unit
convention (cross sections are stored internally in b/sr; ``jitr`` returns
mb/sr, so model outputs are divided by :data:`MB_PER_B`).
"""

import numpy as np
from pint import UnitRegistry

#: The one unit registry for the package.  ``pint`` refuses to combine
#: quantities from different registries, so every module must use this one.
ureg = UnitRegistry()

#: Default maximum partial wave for the reaction solvers.
DEFAULT_LMAX = 20

#: Internal cross-section unit: every ``y`` in b/sr.
XS_UNIT = ureg.barn / ureg.steradian

#: Unit ``jitr`` reports cross sections (and the Rutherford cross section) in.
RUTHERFORD_UNIT = ureg.millibarn / ureg.steradian

#: Millibarn per barn; divides ``jitr`` output to land in :data:`XS_UNIT`.
MB_PER_B = float((1 * ureg.barn).to(ureg.millibarn).magnitude)


def normalized_error_kwargs(
    norm, y_stat_err, y_sys_err_normalization, y_sys_err_offset
) -> dict:
    """Error keywords for ``Observation.__init__`` in internal (norm-divided) units.

    Encodes the unit contract shared by the reaction observations: dimensionful
    errors (statistical, absolute offset) are divided by ``norm`` (a scalar, or
    a per-point array); the fractional normalisation error is dimensionless and
    passed through untouched.
    """
    return {
        "y_stat_err": (
            None if y_stat_err is None else np.asarray(y_stat_err, dtype=float) / norm
        ),
        "y_sys_err_normalization": y_sys_err_normalization,
        "y_sys_err_offset": (
            None
            if y_sys_err_offset is None
            else np.asarray(y_sys_err_offset, dtype=float) / norm
        ),
    }


def measurement_kwargs(measurement) -> dict:
    """The ``Observation``-side constructor keywords carried by an
    ``exfor_tools`` :class:`~exfor_tools.distribution.Distribution`.

    Shared by the reaction observations' ``from_measurement`` classmethods; the
    reaction-specific arguments (``reaction``, ``quantity``/``ExIAS``, solver
    settings, ``transform``, ``mask``) are passed alongside.
    """
    return {
        "x": measurement.x,
        "y": measurement.y,
        "Elab": measurement.Einc,
        "y_units": measurement.y_units,
        "y_stat_err": measurement.statistical_err,
        "y_sys_err_normalization": measurement.systematic_norm_err,
        "y_sys_err_offset": measurement.systematic_offset_err,
        "dataset_label": getattr(measurement, "subentry", None),
    }


def check_angle_grid(angles_rad: np.ndarray, name: str):
    if len(angles_rad.shape) > 1:
        raise ValueError(f"{name} must be 1D, is {len(angles_rad.shape)}D")
    if angles_rad[0] < 0 or angles_rad[-1] > np.pi:
        raise ValueError(f"{name} must be on [0,pi)")
