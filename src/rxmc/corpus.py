import numpy as np

from .params import Parameter
from .constraint import Constraint


class Corpus:
    """
    A collection of independent `Constraint`s that can be used to fit a common
    physical model.

    Each `Constraint` represents a set of `Observation`s and a `LikelihoodModel`.
    Each `Constraint` must share the same `PhysicalModel` parameters, but may
    have different `LikelihoodModel` parameters.

    This class is designed to aggregate multiple constraints, so that, for a
    given set of physical model parameters (and, optionally, likelihood model
    parameters), a log likelihood can be computed.

    Optionally, weights can be assigned to each constraint, which will
    scale the contribution of each constraint to the total log likelihood.
    """

    def __init__(
        self,
        constraints: list[Constraint],
        weights: np.ndarray = None,
        likelihood_params: list[Parameter] = None,
    ):
        self.constraints = constraints
        for constraint in self.constraints:
            self.model_params = self.constraints[0].physical_model.params
            self.likelihood_params = []
            self.n_likelihood_params = 0
        for constraint in constraints:
            if constraint.physical_model.params != self.model_params:
                raise ValueError(
                    "All constraints must use the same physical model parameters"
                )
            self.likelihood_params.append(constraint.likelihood.params)
            self.n_likelihood_params += constraint.likelihood.n_params

        self.n_params = len(self.model_params) + len(self.likelihood_params)
        self.n_data_pts = sum(
            sum(obs.n_data_pts for obs in c.observations) for c in constraints
        )
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
        else:
            if not np.isclose(np.sum(weights), len(self.constraints)):
                raise ValueError("weights must sum to number of constraints")
        self.weights = weights

    def logpdf(self, model_params, likelihood_params: list[tuple] = None):
        """
        Returns the log-pdf that the PhysicalModel predictions, given
        model_params, reproduce the observations in the constraints
        according to the LikelihoodModel, given the likelihood_params.

        Parameters
        ----------
        model_params : tuple
            The parameters of the physical model.
        likelihood_params : list[tuple], optional
            A list of tuples containing additional parameters
            for the likelihood model for each constraint, in the order
            of self.constraints. Defaults to None, meaning none of the
            constraints have additional likelihood parameters. In the case
            that some constraints have additional likelihood parameters,
            and some don't, the list must have the same length as
            self.constraints, with entries containing tupples corresponding
            to constraints taking in parameters and () (empty tuple) for
            those that do not.

        Returns
        -------
        float
        """
        likelihood_params = likelihood_params or [()] * len(self.constraints)
        return sum(
            c.logpdf(model_params, lp) * w
            for w, c, lp in zip(self.weights, self.constraints, likelihood_params)
        )

    def logpdf_conditional_model_params(self, ym: list, likelihood_params: list[tuple]):
        """
        Returns the log-pdf that the model predictions ym, for the
        likelihood_params provided, reproduces the observations in the
        constraints.

        This is useful when the likelihood is parametric and the model is
        computationally expensive to evaluate. In this case, by using Gibb's
        sampling, in which the MCMC chain is broken into batches, where
        each batch consists of first sampling the model parameters conditional
        on a fixed likelihood parameter sample, and secondly sampling the
        likelihood parameters conditional on the ym, using this method.

        Parameters
        ----------
        ym : list
            A list of model predictions corresponding to the observations
            in each constraint.
        likelihood_params : list[tuple], optional
            A list of tuples containing additional parameters
            for the likelihood model for each constraint, in the order
            of self.constraints. Defaults to None, meaning none of the
            constraints have additional likelihood parameters. In the case
            that some constraints have additional likelihood parameters,
            and some don't, the list must have the same length as
            self.constraints, with entries containing tupples corresponding
            to constraints taking in parameters and () (empty tuple) for
            those that do not.

        Returns
        -------
        float
        """
        return sum(
            c.logpdf_conditional_model_params(y, lp) * w
            for w, c, lp, y in zip(
                self.weights, self.constraints, likelihood_params, ym
            )
        )

    def predict(self, *model_params):
        """
        Returns the model predictions for the given model parameters.

        Parameters
        ----------
        *model_params : tuple
            The parameters of the physical model.

        Returns
        -------
        list
            A list of model predictions for each constraint.
        """
        return [c.predict(*model_params) for c in self.constraints]
