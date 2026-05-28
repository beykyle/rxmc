from typing import Any, Type

import numpy as np
from exfor_tools.distribution import Distribution

from .observation import FixedCovarianceObservation, Observation


def set_up_observation(
    ObservationClass: Type[Observation],
    y: np.ndarray,
    normalization: np.ndarray,
    x: np.ndarray,
    y_stat_err=None,
    y_sys_err_normalization=None,
    y_sys_err_offset=None,
    dataset_label: str | None = None,
    include_sys_norm_err: bool = True,
    include_sys_offset_err: bool = True,
    include_statistical_err: bool = True,
):
    r"""
    Set up an `Observation` from explicit measurement data.

    This function normalizes raw measurement arrays and handles systematic and
    statistical errors before constructing the requested `Observation` type.

    Parameters
    ----------
    ObservationClass : Type[Observation]
        The class type of the `Observation` to be created. It must be a
        subclass of `Observation`, such as `FixedCovarianceObservation`.
    y : np.ndarray
        Measured dependent-variable data.
    normalization : np.ndarray
        Normalization factor which the y-values and all dimensionfull
        errors (e.g. all others than normalization errors) will be divided by.
    x : np.ndarray
        Independent-variable grid for the observation.
    y_stat_err : np.ndarray, optional
        Statistical errors associated with `y`.
    y_sys_err_normalization : float or array-like, optional
        Systematic normalization error(s) associated with `y`.
    y_sys_err_offset : float or array-like, optional
        Systematic offset error(s) associated with `y`.
    dataset_label : str, optional
        Human-readable dataset identifier used in validation errors.
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

    label = dataset_label or "dataset"
    y = np.asarray(y) / normalization
    y_stat_err = (
        np.asarray(y_stat_err) / normalization
        if include_statistical_err and y_stat_err is not None
        else np.zeros_like(y)
    )

    y_sys_err_offset_value = None
    y_sys_err_offset_mask = None
    if include_sys_offset_err and y_sys_err_offset is not None:
        y_sys_err_offset_value = np.asarray(y_sys_err_offset) / normalization
        offset_is_scalar = (
            np.isscalar(y_sys_err_offset) or np.ndim(y_sys_err_offset_value) == 0
        )
        # check if systematic errors are common to all angles
        if not (
            offset_is_scalar
            or np.allclose(y_sys_err_offset_value, y_sys_err_offset_value[0])
        ):
            raise ValueError(
                f"Error while parsing measurement from {label}:\n"
                "Systematic offset errors must be scalar or constant."
            )
        else:
            y_sys_err_offset_value = (
                np.asarray(y_sys_err_offset_value).item()
                if offset_is_scalar
                else y_sys_err_offset_value[0]
            )
            y_sys_err_offset_mask = None

    y_sys_err_normalization_value = None
    y_sys_err_normalization_mask = None
    if include_sys_norm_err and y_sys_err_normalization is not None:
        y_sys_err_normalization_value = y_sys_err_normalization
        # check if systematic errors are common to all angles
        ratio = y_sys_err_normalization_value
        ratio_is_scalar = np.isscalar(ratio) or np.ndim(ratio) == 0
        if not (ratio_is_scalar or np.allclose(ratio, ratio[0])):
            raise ValueError(
                f"Error while parsing measurement from {label}:\n"
                "Systematic normalization errors must be scalar or constant."
            )
        else:
            y_sys_err_normalization_mask = None
            y_sys_err_normalization_value = (
                np.asarray(ratio).item() if ratio_is_scalar else ratio[0]
            )

    if ObservationClass is Observation:
        # If the base class is Observation, we can directly return it
        args = (x, y)
        kwargs = {
            "y_stat_err": y_stat_err,
            "y_sys_err_offset": y_sys_err_offset_value,
            "y_sys_err_offset_mask": y_sys_err_offset_mask,
            "y_sys_err_normalization": y_sys_err_normalization_value,
            "y_sys_err_normalization_mask": y_sys_err_normalization_mask,
        }
        return args, kwargs, y_stat_err
    elif ObservationClass is FixedCovarianceObservation:
        if include_sys_norm_err:
            raise ValueError(
                "FixedCovarianceObservation does not support systematic normalization errors."
            )
        covariance = np.diag(y_stat_err**2)
        if y_sys_err_offset_value is not None and include_sys_offset_err:
            covariance += np.outer(y_sys_err_offset_value, y_sys_err_offset_value)
        args = (x, y, covariance)
        return args, {}, y_stat_err
    else:
        # if a new ObservationClass is written, a case for it must be added here
        raise NotImplementedError(
            f"ObservationClass {ObservationClass} is not implemented."
        )


def set_up_observation_from_measurement(
    ObservationClass: Type[Observation],
    measurement: Distribution | Any,
    normalization: np.ndarray,
    x=None,
    include_sys_norm_err=True,
    include_sys_offset_err=True,
    include_statistical_err=True,
):
    r"""
    Convenience wrapper for EXFOR-style `Distribution` inputs.
    """
    x = x if x is not None else measurement.x
    return set_up_observation(
        ObservationClass,
        x=x,
        y=measurement.y,
        y_stat_err=measurement.statistical_err,
        y_sys_err_normalization=measurement.systematic_norm_err,
        y_sys_err_offset=measurement.systematic_offset_err,
        dataset_label=getattr(measurement, "subentry", None),
        normalization=normalization,
        include_sys_norm_err=include_sys_norm_err,
        include_sys_offset_err=include_sys_offset_err,
        include_statistical_err=include_statistical_err,
    )


def check_angle_grid(angles_rad: np.ndarray, name: str):
    if len(angles_rad.shape) > 1:
        raise ValueError(f"{name} must be 1D, is {len(angles_rad.shape)}D")
    if angles_rad[0] < 0 or angles_rad[-1] > np.pi:
        raise ValueError(f"{name} must be on [0,pi)")
