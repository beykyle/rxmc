from typing import Optional, Union
import numpy as np


class Observation:
    """
    A class to represent an observation with associated errors.

    Attributes:
    ----------
    x : np.ndarray
        The independent variable data.
    y : np.ndarray
        The dependent variable data.
    y_stat_err : np.ndarray
        The statistical errors associated with the dependent variable data.
    y_sys_err_offset : np.ndarray
        The systematic errors associated with an offset to the dependent
        variable data.
    y_sys_err_normalization : np.ndarray
        The systematic errors associated with a normalization of the
        dependent variable data.
    n_data_pts : int
        The number of data points in the observation.
    stat_cov : np.ndarray
        The statistical covariance matrix, which is diagonal with the
        squares of the statistical errors.
    sys_cov_offset : np.ndarray
        The systematic covariance matrix for the offset errors, which is
        the outer product of the systematic error offsets.
    fractional_sys_cov_normalization : np.ndarray
        The fractional systematic covariance matrix for the normalization
        errors. When multiplied element-wise with the outer product of the
        true y-values, it gives the systematic covariance matrix for the
        normalization errors.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        y_stat_err: Optional[np.ndarray] = None,
        y_sys_err_offset: Optional[Union[np.ndarray, float, int]] = None,
        y_sys_err_normalization: Optional[np.ndarray] = None,
    ):
        """
        Initialize the DataHandler with data points and their statistical and systematic errors.

        Parameters
        ----------
        x : np.ndarray
            The x data points.
        y : np.ndarray
            The y data points.
        y_stat_err : Optional[np.ndarray], optional
            The statistical errors in y, defaults to an array of zeros with the same shape as y if not provided.
        y_sys_err_offset : Optional[Union[np.ndarray, float, int]], optional
            The systematic offset errors in y. Can be a scalar, array-like, or None.
            Defaults to scalar 0 if not provided.
        y_sys_err_normalization : Optional[np.ndarray], optional
            The systematic normalization errors in y, not applied in current logic, can be handled similarly.
        """
        self.n_data_pts = x.shape[0]
        self.x = x
        self.y = y
        if self.x.shape != self.y.shape:
            raise ValueError(
                "x and y must have the same shape, they have shapes "
                f"{x.shape} and {y.shape}"
            )

        self.y_sys_err_normalization = ensure_array_like(y_sys_err_normalization,
                                                         reference_shape=self.y.shape)
        self.y_stat_err = ensure_array_like(y_stat_err, reference_shape=self.y.shape)
        self.y_sys_err_offset = ensure_array_like(y_sys_err_offset,
                                                  reference_shape=self.y.shape)
        self.stat_cov = np.diag(self.y_stat_err**2)
        self.sys_cov_offset = np.outer(self.y_sys_err_offset, self.y_sys_err_offset)
        self.fractional_sys_cov_normalization = (
            np.ones_like(self.systematic_covariance_offset)
            * self.y_sys_err_normalization**2
        )

    def residual(self, ym: np.ndarray):
        assert ym.shape == self.y.shape
        return self.y - ym

    def num_pts_within_interval(
        self,
        ylow: np.ndarray,
        yhigh: np.ndarray,
        xlim=None,
    ):
        """
        Returns the number of points in y that fall between ylow and yhigh,
        useful for calculating emperical coverages

        Parameters
        ----------
        ylow : np.ndarray, same shape as self.y
        yhigh : np.ndarray, same shape as self.y
        xlim : tuple, optional
            If provided, only consider points where self.x is within
            this range. Defaults to None, meaning all points are
            considered.

        Returns
        -------
        int
            The number of points in self.y (within xlim) that fall
            within the specified interval defined by ylow and yhigh.
        """
        mask = np.ones_like(self.y, dtype=bool)
        if xlim is not None:
            xlow, xhigh = xlim
            mask = np.logical_and(self.x >= xlow, self.x < xhigh)
        return int(
            np.sum(
                np.logical_and(
                    self.observation.y[mask] >= ylow[mask],
                    self.observation.y[mask] < yhigh[mask],
                )
            )
        )


class CombinedObservation(Observation):
    """
    A class to represent a combined observation, which is a collection of
    multiple observations.q

    Useful in the case that multiple observations are made that share
    a global systematic error in offset or normalization, but also
    have their own individual systematic errors.

    Attributes
    ----------
    observations : list[Observation]
        A list of Observation objects that are combined into this
        CombinedObservation.
    """

    def __init__(
        self,
        observations: list[Observation],
        global_sys_err_offset: float = 0.0,
        global_sys_err_normalization: float = 0.0,
    ):
        """
        Initializes the CombinedObservation with a list of observations.

        Parameters
        ----------
        observations : list[Observation]
            A list of Observation objects to be combined.
        global_sys_err_offset : float, optional
            A global systematic error offset applied to all observations.
            Defaults to 0.0.
        global_sys_err_normalization : float, optional
            A global systematic error normalization applied to all observations.
            Defaults to 0.0.
        """
        self.observations = observations
        x = np.concatenate([obs.x for obs in observations])
        y = np.concatenate([obs.y for obs in observations])
        y_stat_err = np.concatenate([obs.y_stat_err for obs in observations])
        y_sys_err_offset = np.concatenate(
            [obs.y_sys_err_offset for obs in observations]
        )  + global_sys_err_offset

        super().__init__(
            x,
            y,
            y_stat_err=y_stat_err,
            y_sys_err_offset=global_sys_err_offset,
            y_sys_err_normalization=global_sys_err_normalization,
        )


class FixedCovarianceObservation(Observation):
    """
    A class to represent an observation with fixed covariance. That is, the
    covariance matrix for the Multivariate Gaussian likelihood for a model
    prediction ym is known a priori and does not change with the model
    prediction.

    The simplest such case is when the covariance is a diagonal matrix
    containing the reported statistical variances for each data point in y.

    In the case that the covariance is a vector, it is interpreted as the
    diagonal of the covariance matrix, and the likelihood reduces to the
    standard form using the chi-squared statistic. In the case that the
    covariance is a full matrix, this corresponds to the generalised
    chi-squared statistic.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        covariance: np.ndarray,
    ):
        """
        Initializes the FixedCovarianceObservation with data and a fixed
        covariance matrix.

        Parameters
        ----------
        x : np.ndarray
            The independent variable data.
        y : np.ndarray
            The dependent variable data.
        covariance : np.ndarray
            The fixed covariance matrix associated with the observation.
        """
        super().__init__(x, y)
        self.covariance = covariance
        if covariance.shape == (self.y.shape[0],):
            self.covariance = np.diag(covariance)
        elif covariance.shape == (self.y.shape[0], self.y.shape[0]):
            self.covariance = covariance
        else:
            raise ValueError(
                f"Incompatible covariance matrix shape "
                f"{covariance.shape} for Constraint with "
                f"{self.y.shape[0]} data points"
            )

        self.cov_inv = np.linalg.inv(self.covariance)
        sign, self.log_det = np.linalg.slogdet(self.covariance)
        if sign != +1:
            raise ValueError("Invalid covariance matrix! Must be positive definite.")


def ensure_array_like(target: Optional[Union[np.ndarray, float, int]], reference_shape: tuple) -> np.ndarray:
    """
    Ensure that the target variable is an array-like structure matching the shape of a reference shape.

    Parameters
    ----------
    target : Optional[Union[np.ndarray, float, int]]
        Variable that could be scalar, array-like, or None.
    reference_shape : tuple
        The shape that the output array should match.

    Returns
    -------
    np.ndarray
        An array matching the reference shape with the content of target.
    """
    if target is None:
        # Default to a scalar 0
        return np.zeros(reference_shape)
    elif np.isscalar(target):
        # If it is a scalar, broadcast to the reference shape
        return np.full(reference_shape, target)
    else:
        # Assume numpy will broadcast if input is an array-like
        target_array = np.array(target)
        if target_array.shape != reference_shape:
            raise ValueError(
                f"Input array must have shape {reference_shape}, but has shape {target_array.shape}."
            )
        return target_array
