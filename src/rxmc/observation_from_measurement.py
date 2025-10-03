from typing import Type

import jitr
import numpy as np
from exfor_tools.distribution import Distribution
from pint import UnitRegistry

from .observation import FixedCovarianceObservation, Observation


def set_up_observation(
    ObservationClass: Type[Observation],
    measurement: Distribution,
    normalization: np.ndarray,
    x=None,
    include_sys_norm_err=True,
    include_sys_offset_err=True,
    include_statistical_err=True,
):
    r"""
    Set up an `Observation` from a `Distribution`.

    This function converts a `Distribution` into an `Observation` object,
    normalizing the y-values and handling systematic and statistical errors.

    Parameters
    ----------
    ObservationClass : Type[Observation]
        The class type of the `Observation` to be created. It must be a
        subclass of `Observation`, such as `FixedCovarianceObservation`.
    measurement : Distribution
        The measurement data containing x, y, and associated errors.
    normalization : np.ndarray
        Normalization factor which the y-values and all dimensionfull
        errors (e.g. all others than normalization errors) will be divided by.
    include_sys_norm_err : bool, optional
        Whether to include systematic normalization errors, by default True.
    include_sys_offset_err : bool, optional
        Whether to include systematic offset errors, by default True.
    include_statistical_err : bool, optional
        Whether to include statistical errors, by default True.
    x : np.ndarray, optional
        Custom x-values to use instead of the measurement's x-values.
    Returns
    -------
        args: tuple
            args for the ObservationClass initializer
        kwargs: dict
            kwargs for the ObservationClass initializer
        y_stat_err: np.ndarray
            Statistical errors normalized by the normalization factor.
    """

    x = x if x is not None else measurement.x
    y = measurement.y / normalization
    y_stat_err = (
        measurement.statistical_err / normalization
        if include_statistical_err
        else np.zeros_like(y)
    )

    y_sys_err_offset = None
    y_sys_err_offset_mask = None
    if include_sys_offset_err:
        y_sys_err_offset = measurement.systematic_offset_err / normalization
        # check if systematic errors are common to all angles
        if not (
            np.isscalar(y_sys_err_offset)
            or np.allclose(y_sys_err_offset, y_sys_err_offset[0])
        ):
            raise ValueError(
                f"Error while parsing measurement from subentry {measurement.subentry}:\n"
                "Systematic offset errors must be scalar or constant."
            )
        else:
            y_sys_err_offset = (
                y_sys_err_offset
                if np.isscalar(y_sys_err_offset)
                else y_sys_err_offset[0]
            )
            y_sys_err_offset_mask = None

    y_sys_err_normalization = None
    y_sys_err_normalization_mask = None
    if include_sys_norm_err:
        y_sys_err_normalization = measurement.systematic_norm_err
        # check if systematic errors are common to all angles
        ratio = y_sys_err_normalization
        if not (np.isscalar(ratio) or np.allclose(ratio, ratio[0])):
            raise ValueError(
                f"Error while parsing measurement from subentry {measurement.subentry}:\n"
                "Systematic normalization errors must be scalar or constant."
            )
        else:
            y_sys_err_normalization_mask = None
            y_sys_err_normalization = ratio if np.isscalar(ratio) else ratio[0]

    if ObservationClass is Observation:
        # If the base class is Observation, we can directly return it
        args = (x, y)
        kwargs = {
            "y_stat_err": y_stat_err,
            "y_sys_err_offset": y_sys_err_offset,
            "y_sys_err_offset_mask": y_sys_err_offset_mask,
            "y_sys_err_normalization": y_sys_err_normalization,
            "y_sys_err_normalization_mask": y_sys_err_normalization_mask,
        }
        return args, kwargs, y_stat_err
    elif ObservationClass is FixedCovarianceObservation:
        if include_sys_norm_err:
            raise ValueError(
                "FixedCovarianceObservation does not support systematic normalization errors."
            )
        covariance = np.diag(y_stat_err**2)
        if y_sys_err_offset is not None and include_sys_offset_err:
            covariance += np.outer(y_sys_err_offset, y_sys_err_offset)
        args = (x, y, covariance)
        return args, {}, y_stat_err
    else:
        # if a new ObservationClass is written, a case for it must be added here
        raise NotImplementedError(
            f"ObservationClass {ObservationClass} is not implemented."
        )


def check_angle_grid(angles_rad: np.ndarray, name: str):
    if len(angles_rad.shape) > 1:
        raise ValueError(f"{name} must be 1D, is {len(angles_rad.shape)}D")
    if angles_rad[0] < 0 or angles_rad[-1] > np.pi:
        raise ValueError(f"{name} must be on [0,pi)")
