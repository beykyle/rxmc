from collections import OrderedDict

import numpy as np

from .constraint import Constraint


class Corpus:
    """
    A class to represent a generic collection of independent constraints.

    Attributes
    ----------
    model_name : str
        Name of model
    corpus_name : str
        label for corpus of constraints
    constraints : list of Constraint
        A list of constraints with the same model.
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
    params : list of str
        The parameter names
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
        params: list,
        model_name: str,
        corpus_name: str,
        weights: np.ndarray = None,
    ):
        self.model_name = model_name
        self.corpus_name = corpus_name
        self.constraints = constraints
        self.y = np.hstack([constraint.y for constraint in self.constraints])
        self.x = np.hstack([constraint.x for constraint in self.constraints])
        self.params = params
        self.n_params = len(params)
        self.n_data_pts = self.y.size
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

    def residual(self, params: OrderedDict):
        """
        Compute the residuals for the given parameters.

        Parameters
        ----------
        params : OrderedDict
            Parameters for which to compute the residuals.

        Returns
        -------
        np.ndarray
            Residuals for the given parameters.
        """
        return np.hstack(
            [constraint.residual(params) for constraint in self.constraints]
        )

    def chi2(self, params: OrderedDict):
        """
        Compute the weighted chi-squared value for the given parameters.

        Parameters
        ----------
        params : OrderedDict
            Parameters for which to compute the chi-squared value.

        Returns
        -------
        float
            Chi-squared value for the given parameters.
        """
        return sum(
            constraint.chi2(params) * weight
            for weight, constraint in zip(self.weights, self.constraints)
        )

    def logpdf(self, params: OrderedDict):
        """
        Returns the log-pdf that the Model, given params, reproduces y

        Parameters
        ----------
        params : OrderedDict
            parameters of model

        Returns
        -------
        float
        """
        return sum(
            constraint.logpdf(params) * weight
            for weight, constraint in zip(self.weights, self.constraints)
        )

    def num_pts_within_interval(self, ylow: np.ndarray, yhigh: np.ndarray):
        """
        Compute the empirical coverage within the given interval by summing
        the number of points within the interval.

        Parameters
        ----------
        ylow : np.ndarray
            Lower bounds of the interval.
        yhigh : np.ndarray
            Upper bounds of the interval.

        Returns
        -------
        float
            Empirical coverage within the given interval.
        """
        return (
            sum(
                constraint.num_pts_within_interval(ylow, yhigh)
                for constraint in self.constraints
            )
            / self.n_data_pts
        )

    def model(self, params: OrderedDict):
        """
        Compute the model output for each constraint, given params


        Parameters
        ----------
        params : OrderedDict
            parameters of model

        Returns
        -------
        list
        """
        return [c.model(params) for c in self.constraints]
