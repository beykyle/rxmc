import numpy as np

from .params import Parameter
from .constraint import Constraint

# TODO implement a version in which each constraint has its own
# likelihood model and parameters


class Corpus:
    """
    A collection of constraints that can be used to fit a common
    physical model.
    """

    def __init__(
        self,
        constraints: list[Constraint],
        weights: np.ndarray = None,
    ):
        self.constraints = constraints
        for constraint in self.constraints:
            self.model_params = self.constraints[0].model.params
            self.likelihood_params = self.constraints[0].likelihood_model.params
        for constraint in constraints:
            if constraint.model.params != self.model_params:
                raise ValueError(
                    "All constraints must use the same physical model parameters"
                )
            if constraint.likelihood_model.params != self.likelihood_params:
                raise ValueError(
                    "All constraints must use the same likelihood model parameters"
                )

        self.n_params = len(self.model_params) + len(self.likelihood_params)
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

    def model(self, model_params):
        """
        Compute the model output for each constraint, given model_params


        Parameters
        ----------
        model_params : tuple
            The parameters of the physical model.

        Returns
        -------
        list
        """
        return np.hstack([c.model(*model_params) for c in self.constraints])

    def residual(self, model_params):
        """
        Compute the residuals for the given parameters.

        Parameters
        ----------
        model_params : tuple
            The parameters of the physical model.

        Returns
        -------
        np.ndarray
            Residuals for the given parameters.
        """
        return np.hstack(
            [c.observation.residual(c.model(*model_params)) for c in self.constraints]
        )

    def chi2(self, model_params, likelihood_params=None):
        """
        Compute the weighted chi-squared value for the given parameters.

        Parameters
        ----------
        model_params : tuple
            The parameters of the physical model.
        likelihood_params : tuple, optional
            Additional parameters for the likelihood model, if any.

        Returns
        -------
        float
            Chi-squared value for the given parameters.
        """
        return sum(
            c.chi2(model_params, likelihood_params) * w
            for w, c in zip(self.weights, self.constraints)
        )

    def logpdf(self, model_params, likelihood_params=None):
        """
        Returns the log-pdf that the Model, given params, reproduces y

        Parameters
        ----------
        model_params : tuple
            The parameters of the physical model.
        likelihood_params : tuple, optional
            Additional parameters for the likelihood model, if any.

        Returns
        -------
        float
        """
        return sum(
            c.logpdf(model_params, likelihood_params) * w
            for w, c in zip(self.weights, self.constraints)
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
                c.num_pts_within_interval(ylow, yhigh, xlim=xlim)
                for c in self.constraints
            )
            / self.n_data_pts
        )
