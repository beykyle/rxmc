from collections import OrderedDict

import scipy as sc
import numpy as np

from .model import Model


def mahalanobis_distance_cholesky(y, ym, cov):
    """
    Returns the Mahalanobis distance between y and ym according to
    covariance matrix cov, as well as the log determinant of cov, using
    Cholesky decomposition to factorize cov
    """
    L = sc.linalg.cholesky(cov, lower=True)
    z = sc.linalg.solve_triangular(L, y - ym, lower=True)
    mahalanobis = np.dot(z, z)
    log_det = 2 * np.sum(np.log(np.diag(L)))

    return mahalanobis, log_det


def log_likelihood(mahalanobis: float, log_det: float, n: int):
    """
    Returns the log likelihood of a multivariate normal with given
    mahalanobis distance, log determinant of covariance matrix,
    and dimension n
    """
    return -0.5 * (mahalanobis + log_det + n * np.log(2 * np.pi))


class Constraint:
    """
    Represents experimental data y, which is assumed to be
    a random variate distributed according to a multivariate normal
    around y with an arbitrary covariance matrix, along with some
    parameteric Model which produces predictions for y given a
    set of parameters.

    In this  base class, the covariance matrix is unspecified. Derived
    classes must implement the eval_model_and_covariance() method, or
    otherwise override the chi2() and logpdf() methods.


    Parameters
    ----------
    y : np.ndarray
        Experimental data output
    x : np.ndarray
        Experimental data input
    model: Model
        The parametric model. Must be a callable which takes in an
        OrderedDict of parameters and outputs a np.ndarray of the
        same shape as y
    """

    def __init__(self, y: np.ndarray, model: Model):
        self.y = y
        self.model = model
        self.x = model.x
        if self.x.shape != self.y.shape:
            raise ValueError(
                "Incompatible x and y shapes: " f"{self.x.shape} and {self.y.shape}"
            )
        self.n_data_pts = y.shape[0]

    def residual(self, params: OrderedDict):
        """
        Calculate the residuals.

        Parameters
        ----------
        params : OrderedDict
            parameters of model

        Returns
        -------
        np.ndarray
            Residuals.
        """
        ym = self.model(params)
        return self.y - ym

    def num_pts_within_interval(self, ylow: np.ndarray, yhigh: np.ndarray):
        """
        Returns the number of points in y that fall between ylow and yhigh,
        useful for calculating emperical coverages

        Parameters
        ----------
        ylow : np.ndarray, same shape as self.y
        yhigh : np.ndarray, same shape as self.y

        Returns
        -------
        int
        """
        return int(np.sum(np.logical_and(self.y >= ylow, self.y < yhigh)))

    def model(self, params: OrderedDict):
        return self.model(params)

    def eval_model_and_covariance(self, params: OrderedDict):
        pass

    def chi2(self, params: OrderedDict):
        """
        Calculate the generalised chi-squared statistic. This is the
        Malahanobis distance between y and model(params).

        Parameters
        ----------
        params : OrderedDict
            parameters of model

        Returns
        -------
        float
            Chi-squared statistic.
        """
        ym, cov = self.eval_model_and_covariance(params)
        mahalanobis, _ = mahalanobis_distance_cholesky(self.y, ym, cov)
        return mahalanobis

    def logpdf(self, params: OrderedDict):
        """
        Returns the logpdf that the Model, given params, reproduces y

        Parameters
        ----------
        params : OrderedDict
            parameters of model

        Returns
        -------
        float
        """
        ym, cov = self.eval_model_and_covariance(params)
        mahalanobis, log_det = mahalanobis_distance_cholesky(self.y, ym, cov)
        return log_likelihood(mahalanobis, log_det, self.n_data_pts)


class FixedCovarianceConstraint(Constraint):
    """
    A special case of Constraint in which the covariance matrix is known a
    priori; e.g. is not dependent on the Model or its params

    Parameters
    ----------
    covariance : np.ndarray
        Covariance matrix.
    cov_inc : np.ndarray
        inverse of covariance matrix.
    log_det : np.ndarray
        log determinant of covariance matrix
    """

    def __init__(self, y: np.ndarray, covariance: np.ndarray, model: Model):
        super().__init__(y, model)
        if covariance.shape == (self.n_data_pts,):
            self.covariance = np.diag(covariance)
        elif covariance.shape == (self.n_data_pts, self.n_data_pts):
            self.covariance = covariance
        else:
            raise ValueError(
                f"Incompatible covariance matrix shape "
                f"{covariance.shape} for Constraint with "
                f"{self.n_data_pts} data points"
            )

        self.cov_inv = np.linalg.inv(self.covariance)
        sign, self.log_det = np.linalg.slogdet(self.covariance)
        if sign != +1:
            raise ValueError("Invalid covariance matrix! Must be positive definite.")

    def eval_model_and_covariance(self, params: OrderedDict):
        ym = self.model(params)
        return ym, self.covariance

    def chi2(self, params: OrderedDict):
        """
        Calculate the generalised chi-squared statistic. This is the
        Malahanobis distance between y and model(params).

        Parameters
        ----------
        params : OrderedDict
            parameters of model

        Returns
        -------
        float
            Chi-squared statistic.
        """
        delta = self.residual(params)
        return delta.T @ self.cov_inv @ delta

    def logpdf(self, params: OrderedDict):
        """
        Returns the logpdf that the Model, given params, reproduces y

        Parameters
        ----------
        params : OrderedDict
            parameters of model

        Returns
        -------
        float
        """
        mahalanobis = self.chi2(params)
        return -0.5 * (mahalanobis + self.log_det + self.n_data_pts * np.log(2 * np.pi))


class ConstraintWithKnownError(Constraint):
    """Constraint in which the systematic error is a constant fraction of y"""

    def __init__(
        self,
        y: np.ndarray,
        model: Model,
        sys_err_frac: float,
        diag_err_frac: float,
        model_independent_covariance=None,
    ):
        super().__init__(y, model)
        if model_independent_covariance is None:
            model_independent_covariance = np.zeros(
                (len(self.y), len(self.y)), dtype=float
            )
        self.model_independent_covariance = model_independent_covariance
        self.sys_err_frac = sys_err_frac
        self.diag_err_frac = diag_err_frac

    def eval_model_and_covariance(self, params: OrderedDict):
        ym = self.model(params)
        cov = (
            self.model_independent_covariance
            + self.sys_err_frac * np.outer(ym, ym)
            + self.diag_err_frac * np.diag(ym**2)
        )
        return ym, cov


class ConstraintWithUnknownError(Constraint):
    """
    Constraint for arbitrary covariance model, in which a callable
    `get_model_y_and_cov` is passed in to handle determination of
    the covariance and model prediction for a given parameter
    """

    def __init__(
        self,
        y: np.ndarray,
        model: Model,
        get_model_y_and_cov,
        model_independent_covariance=None,
    ):
        super().__init__(y, model)
        if model_independent_covariance is None:
            model_independent_covariance = np.zeros(
                (len(self.y), len(self.y)), dtype=float
            )
        self.get_model_y_and_cov = get_model_y_and_cov

    def eval_model_and_covariance(self, params: OrderedDict):
        ym, cov = self.get_model_y_and_cov(self, params)
        return ym, cov
