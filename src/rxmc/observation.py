import numpy as np
from collections.abc import Iterable


class Observation:
    """
    A class to represent an observation with statistical errors,
    as well as systematic errors associated with a common normalization
    and offset of all or some of the data points of the the dependent
    variable y.

    Attributes:
    ----------
    x : np.ndarray
        The independent variable data.
    y : np.ndarray
        The dependent variable data.
    statistical_covariance : np.ndarray
        The covariance matrix representing the statistical errors of y.
    systematic_offset_covariance : np.ndarray
        The covariance matrix representing systematic errors associated with
        the offset of y.
    systematic_normalization_covariance : np.ndarray
        The fractional covariance matrix representing systematic errors
        associated with the normalization of y.
    n_data_pts : int
        The number of data points in the observation.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        y_stat_err=None,
        y_sys_err_normalization=None,
        y_sys_err_normalization_mask=None,
        y_sys_err_offset=None,
        y_sys_err_offset_mask=None,
    ):
        r"""
        x : np.ndarray
            The independent variable data.
        y : np.ndarray
            The dependent variable data.
        y_stat_err : np.ndarray, optional
            The statistical error associated with y. Defaults to an array of
            zeros with the same shape as y.
        y_sys_err_normalization : float or array-like, optional
            The fractional systematic error associated with normalization of y.
            Defaults to 0.0.  If array-like object is passed in, that implies
            that there are multiple systematic errors associated with
            normalization, each corresponding to an entry in
            `y_sys_err_normalization_mask`.
        y_sys_err_normalization_mask : list of np.ndarray, optional
            Masks for the systematic errors associated with normalization of y.
            Each mask should have the same shape as y, and the systematic error
            associated with normalization will only apply to the points where
            the mask is True. Defaults to None, meaning no systematic errors
            associated with normalization, or equivalently, a single
            systematic error for all points.
        y_sys_err_offset : float or array-like, optional
            The systematic error associated with the offset of y. Defaults to
            0.0. If array-like object is passed in, that implies that there
            a multiple systematic errors associated with normalization, each
            corresponding to an entry in `y_sys_err_normalization_mask`.
        y_sys_err_offset_mask : list of np.ndarray, optional
            Masks for the systematic errors associated with the offset of y.
            Each mask should have the same shape as y, and the systematic error
            associated with the offset will only apply to the points where
            the mask is True. Defaults to None, meaning no systematic errors
            associated with the offset, or equivalently, a single systematic
            error for all points.
        """
        self.n_data_pts = x.shape[0]
        self.x = x
        self.y = y
        if self.x.shape != self.y.shape:
            raise ValueError(
                "x and y mustr have the same shape, they have shapes "
                f" {x.shape} and {y.shape}"
            )
        y_stat_err = y_stat_err if y_stat_err is not None else np.zeros_like(y)
        if y_stat_err.shape != self.y.shape:
            raise ValueError(
                "y_stat_err must have the same shape as y, "
                f"it has shape {y_stat_err.shape} and y has shape {self.y.shape}"
            )
        self.statistical_covariance = np.diag(y_stat_err**2)

        # systematic errors in normalization
        self.systematic_normalization_covariance = np.zeros_like(
            self.statistical_covariance
        )
        if y_sys_err_normalization is not None:
            if is_array_like(y_sys_err_normalization):
                if y_sys_err_normalization_mask is None:
                    raise ValueError(
                        "If y_sys_err_normalization is array-like, "
                        "y_sys_err_normalization_mask must also be provided."
                    )
                if len(y_sys_err_normalization) != len(y_sys_err_normalization_mask):
                    raise ValueError(
                        "If y_sys_err_normalization is array-like, "
                        "y_sys_err_normalization_mask must have the same length."
                    )
                for sys_err, mask in zip(
                    y_sys_err_normalization, y_sys_err_normalization_mask
                ):
                    if mask.shape != self.y.shape:
                        raise ValueError(
                            "each mask in y_sys_err_normalization_mask must have the same shape as y"
                        )
                    sys_err = sys_err * mask.astype(float)
                    self.systematic_normalization_covariance += np.outer(
                        sys_err, sys_err
                    )
            elif is_scalar_like(y_sys_err_normalization):
                sys_err = y_sys_err_normalization * np.ones_like(y)
                self.systematic_normalization_covariance += np.outer(sys_err, sys_err)

        # systematic errors in offset
        self.systematic_offset_covariance = np.zeros_like(self.statistical_covariance)
        if y_sys_err_offset is not None:
            if is_array_like(y_sys_err_offset):
                if y_sys_err_offset_mask is None:
                    raise ValueError(
                        "If y_sys_err_offset is array-like, "
                        "y_sys_err_offset_mask must also be provided."
                    )
                if len(y_sys_err_offset) != len(y_sys_err_offset_mask):
                    raise ValueError(
                        "If y_sys_err_offset is array-like, "
                        "y_sys_err_offset_mask must have the same length."
                    )
                for sys_err, mask in zip(y_sys_err_offset, y_sys_err_offset_mask):
                    if mask.shape != self.y.shape:
                        raise ValueError(
                            "each mask in y_sys_err_offset_mask must have the same shape as y"
                        )
                    sys_err = sys_err * mask.astype(float)
                    self.systematic_offset_covariance += np.outer(sys_err, sys_err)
            elif is_scalar_like(y_sys_err_offset):
                sys_err = y_sys_err_offset * np.ones_like(y)
                self.systematic_offset_covariance += np.outer(sys_err, sys_err)

    def covariance(self, y):
        """
        Returns the default covariance matrix for the observation,
        which is the sum of the statistical and systematic offset covariance
        matrices, and the fractional normalization covariance matrix
        multiplied by the outer product of y with itself.

        Parameters
        ----------
        y : np.ndarray
            The dependent variable data for which to compute the covariance.
        """
        return (
            self.statistical_covariance
            + self.systematic_offset_covariance
            + self.systematic_normalization_covariance**2 * np.outer(y, y)
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
                    self.y[mask] >= ylow[mask],
                    self.y[mask] < yhigh[mask],
                )
            )
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
        if covariance.shape == (self.y.shape[0],):
            self.cov = np.diag(covariance)
        elif covariance.shape == (self.y.shape[0], self.y.shape[0]):
            self.cov = covariance
        else:
            raise ValueError(
                f"Incompatible covariance matrix shape "
                f"{covariance.shape} for Constraint with "
                f"{self.y.shape[0]} data points"
            )

        self.cov_inv = np.linalg.inv(self.cov)
        sign, self.log_det = np.linalg.slogdet(self.cov)
        if sign != +1:
            raise ValueError("Invalid covariance matrix! Must be positive definite.")

    def covariance(self, y):
        """
        Returns the fixed covariance matrix for the observation,
        which is constant and does not depend on y.

        Parameters
        ----------
        y : np.ndarray
            The dependent variable data for which to compute the covariance.
        """
        return self.cov


def is_array_like(obj):
    return isinstance(obj, Iterable) and not isinstance(obj, (str, bytes))


def is_scalar_like(obj):
    return np.isscalar(obj) or isinstance(obj, (int, float, complex))
