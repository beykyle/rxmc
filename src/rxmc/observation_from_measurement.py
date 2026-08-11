"""
Helpers shared by the reaction-observation classes.

The reaction observations (:class:`~rxmc.elastic_diffxs_observation.ElasticDifferentialXSObservation`
and :class:`~rxmc.ias_pn_observation.IsobaricAnalogPNObservation`) are now plain
:class:`~rxmc.observation.Observation` subclasses carrying **statistical error
only**; any correlated systematic is composed explicitly as a
:class:`~rxmc.covariance.Term` in the :class:`~rxmc.constraint.Constraint`.  This
module just holds the angle-grid validation they share.
"""

import numpy as np


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


def check_angle_grid(angles_rad: np.ndarray, name: str):
    if len(angles_rad.shape) > 1:
        raise ValueError(f"{name} must be 1D, is {len(angles_rad.shape)}D")
    if angles_rad[0] < 0 or angles_rad[-1] > np.pi:
        raise ValueError(f"{name} must be on [0,pi)")
