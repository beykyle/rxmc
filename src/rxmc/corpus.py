import numpy as np

from .params import Parameter
from .constraint import Constraint


class Corpus:
    """
    A class to represent a generic collection of independent constraints.

    Attributes
    ----------
    constraints : list of Constraints
        A list of constraints with the same underlying physical model
        parameterization.
    params: list of Parameters
    y : np.ndarray
        Combined y values from all constraints.
    x : np.ndarray
        Combined x values from all constraints.
    n_data_pts : int
        Total number of data points.
    n_dof : int
        Total number of degrees of freedom in calibration.
    nparams : int
        Total number of free parameters in the model.
    weights : np.ndarray
        Weights for each constraint (should sum to 1)

    Methods
    -------
    residual(params)
        Computes the residuals for the given parameters.
    chi2(params)
        Computes the chi-squared value for the given parameters.
    empirical_coverage(ylow, yhigh, method='count')
        Computes the empirical coverage within the given interval.
    """

    def __init__(
        self,
        constraints: list[Constraint],
        params: list[Parameter],
        weights: np.ndarray = None,
    ):
        self.constraints = constraints
        self.params = params
        self.n_params = len(params)
        self.n_data_pts = sum(c.observation.n_data_pts for c in constraints)
        self.n_dof = self.n_data_pts - self.n_params
        if self.n_dof < 0:
            raise ValueError(
                f"Model under-constrained! {self.n_params} free parameters"
                f"and {self.n_data_pts} data points"
            )

        if weights is None:
            weights = np.ones((len(self.constraints),), dtype=float)
        elif weights.shape != (len(self.constraints),):
            raise ValueError(
                "weights must be a 1D array with the same shape as constraints"
            )
        self.weights = weights * len(self.constraints)
        if not np.isclose(np.sum(weights), len(self.constraints)):
            raise ValueError("weights must sum to 1")

    def model(self, params):
        """
        Compute the model output for each constraint, given params


        Parameters
        ----------
        params : OrderedDict or np.ndarray
            The parameters of the physical model

        Returns
        -------
        list
        """
        return [c.model(params) for c in self.constraints]

    def residual(self, params):
        """
        Compute the residuals for the given parameters.

        Parameters
        ----------

        Returns
        -------
        np.ndarray
            Residuals for the given parameters.
        """
        return np.hstack(
            [constraint.observation.residual(params) for constraint in self.constraints]
        )

    def chi2(self, params):
        """
        Compute the weighted chi-squared value for the given parameters.

        Parameters
        ----------
        params : OrderedDict or np.ndarray
            The parameters of the physical model

        Returns
        -------
        float
            Chi-squared value for the given parameters.
        """
        return sum(
            constraint.chi2(params) * weight
            for weight, constraint in zip(self.weights, self.constraints)
        )

    def logpdf(self, params):
        """
        Returns the log-pdf that the Model, given params, reproduces y

        Parameters
        ----------
        params : OrderedDict or np.ndarray
            The parameters of the physical model

        Returns
        -------
        float
        """
        return sum(
            constraint.logpdf(params) * weight
            for weight, constraint in zip(self.weights, self.constraints)
        )

    def num_pts_within_interval(self, ylow: np.ndarray, yhigh: np.ndarray, xlim=None):
        """
        Compute the empirical coverage within the given interval by summing
        the number of points within the interval.

        Parameters
        ----------
        ylow : np.ndarray
            Lower bounds of the interval.
        yhigh : np.ndarray
            Upper bounds of the interval.
        xlim : tuple, optional
            If provided, only consider points where self.x is within
            this range. Defaults to None, meaning all points are
            considered.

        Returns
        -------
        float
            Empirical coverage within the given interval.
        """
        return (
            sum(
                constraint.num_pts_within_interval(ylow, yhigh, xlim=xlim)
                for constraint in self.constraints
            )
            / self.n_data_pts
        )
