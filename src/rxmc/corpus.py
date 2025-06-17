import numpy as np

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
        constraints: list[Constraint] = [],
        parametric_constraints: list[Constraint] = [],
        weights: np.ndarray = None,
    ):
        if len(constraints) > 0:
            self.model_params = constraints[0].physical_model.params
        elif len(parametric_constraints) > 0:
            self.model_params = parametric_constraints[0].physical_model.params
        else:
            raise ValueError(
                "Either 'constraints' or 'parametric_constraints' must not be empty"
            )

        self.constraints = constraints
        self.parametric_constraints = parametric_constraints
        self.n_likelihood_params = 0

        # check the regular constraints
        for constraint in self.constraints:
            if constraint.physical_model.params != self.model_params:
                raise ValueError(
                    "All constraints must use the same physical model parameters"
                )
            if constraint.likelihood.n_params > 0:
                raise ValueError(
                    "Constraint with parametric likelihood model "
                    "found in the `constraints` list; should be "
                    "in the `parametric_constraints` list"
                )

        # check the parametric constraints
        for constraint in self.parametric_constraints:
            if constraint.physical_model.params != self.model_params:
                raise ValueError(
                    "All constraints must use the same physical model parameters"
                )
            if constraint.likelihood.n_params == 0:
                raise ValueError(
                    "Constraint with out parametric likelihood "
                    "model found in the `parametric_constraints` "
                    "list; should be in the `constraints` list"
                )
            self.n_likelihood_params += constraint.likelihood.n_params

        self.n_params = len(self.model_params) + self.n_likelihood_params
        self.n_data_pts = sum(
            sum(obs.n_data_pts for obs in c.observations)
            for c in constraints + parametric_constraints
        )
        self.n_dof = self.n_data_pts - self.n_params
        if self.n_dof < 0:
            raise ValueError(
                f"Model under-constrained! {self.n_params} free parameters "
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

    def log_likelihood(self, model_params, likelihood_params: list[tuple] = []):
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
            for the likelihood model for each parameteric constraint,
            corresponding to the order of `self.parametric_constraints`.
            Defaults to None, meaning none of the constraints have
            parametric likelihoods.

        Returns
        -------
        float
            the log likelihood
        """
        assert len(likelihood_params) == len(self.parametric_constraints)

        # sum log likelihood over regular constraints
        ll = sum(
            c.log_likelihood(model_params) * w
            for w, c in zip(self.weights, self.constraints)
        )
        # also sum log likelihood over constraints that take in parameters
        # for the likelihood model, as well as just the physical model
        ll += sum(
            c.log_likelihood(model_params, lp) * w
            for w, c, lp in zip(
                self.weights, self.parametric_constraints, likelihood_params
            )
        )
        return ll
