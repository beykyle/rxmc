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
    y_stat_err : np.ndarray, optional
        The statistical error associated with y. Defaults to an array of
        zeros with the same shape as y.
    y_sys_err_normalization : float, optional
        The systematic error normalization associated with y. Defaults to 0.0.
    y_sys_err_offset : float, optional
        The systematic error offset associated with y. Defaults to 0.0.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        y_stat_err=None,
        y_sys_err_normalization=0.0,
        y_sys_err_offset=0.0,
    ):
        self.n_data_pts = x.shape[0]
        self.x = x
        self.y = y
        if self.x.shape != self.y.shape:
            raise ValueError(
                "x and y mustr have the same shape, they have shapes "
                f" {x.shape} and {y.shape}"
            )
        self.y_stat_err = y_stat_err if y_stat_err is not None else np.zeros_like(y)
        self.y_sys_err_normalization = y_sys_err_normalization
        self.y_sys_err_offset = y_sys_err_offset

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
