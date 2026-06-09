"""
Evidence: aggregate of independent constraints for Bayesian calibration.

An :class:`Evidence` object collects multiple :class:`~rxmc.constraint.Constraint`
objects that share the same physical-model parameters.  It computes a joint log
likelihood by summing the individual constraint log likelihoods (optionally
weighted), supporting both fixed and parametric likelihood models via a
Gibbs-style decomposition.
"""

import numpy as np

from .constraint import Constraint


class Evidence:
    """A collection of independent constraints sharing a common physical model.

    Each :class:`~rxmc.constraint.Constraint` represents a set of observations
    paired with a likelihood model.  All constraints must share the same
    physical-model parameters, but may have different likelihood-model parameters.

    Optional per-constraint weights scale the contribution of each constraint to
    the total log likelihood.

    Parameters
    ----------
    constraints : list of Constraint, optional
        Constraints whose likelihood models have no free parameters.
    parametric_constraints : list of Constraint, optional
        Constraints whose likelihood models have free parameters.
    weights : np.ndarray, optional
        1-D array of weights for *constraints*.  Defaults to all ones.
    weights_parametric : np.ndarray, optional
        1-D array of weights for *parametric_constraints*.  Defaults to all ones.

    Raises
    ------
    ValueError
        If both *constraints* and *parametric_constraints* are empty.
    ValueError
        If any constraint uses different physical-model parameters than the first.
    ValueError
        If a non-parametric constraint appears in *parametric_constraints*, or
        vice versa.
    ValueError
        If the number of data points is less than the number of free parameters
        (under-constrained model).
    ValueError
        If *weights* or *weights_parametric* do not match the corresponding list
        length.
    """

    def __init__(
        self,
        constraints: list[Constraint] | None = None,
        parametric_constraints: list[Constraint] | None = None,
        weights: np.ndarray = None,
        weights_parametric: np.ndarray = None,
    ):
        constraints = constraints or []
        parametric_constraints = parametric_constraints or []

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

        for constraint in self.parametric_constraints:
            if constraint.physical_model.params != self.model_params:
                raise ValueError(
                    "All constraints must use the same physical model parameters"
                )
            if constraint.likelihood.n_params == 0:
                raise ValueError(
                    "Constraint without parametric likelihood "
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
        self.weights = weights

        if weights_parametric is None:
            weights_parametric = np.ones(
                (len(self.parametric_constraints),), dtype=float
            )
        elif weights_parametric.shape != (len(self.parametric_constraints),):
            raise ValueError(
                "weights_parametric must be a 1D array with the same shape as parametric_constraints"
            )
        self.weights_parametric = weights_parametric

    def log_likelihood(
        self, model_params, likelihood_params: list[tuple] | None = None
    ):
        """Weighted sum of log likelihoods over all constraints.

        Parameters
        ----------
        model_params : tuple
            Parameters of the physical model.
        likelihood_params : list of tuple, optional
            One tuple of likelihood parameters per entry in
            *parametric_constraints*.  Defaults to an empty list.

        Returns
        -------
        float
            Total weighted log likelihood.
        """
        likelihood_params = likelihood_params or []
        if len(likelihood_params) != len(self.parametric_constraints):
            raise ValueError(
                f"Expected {len(self.parametric_constraints)} likelihood parameter "
                f"tuples, got {len(likelihood_params)}"
            )

        ll = sum(
            c.log_likelihood(model_params) * w
            for w, c in zip(self.weights, self.constraints)
        )
        ll += sum(
            c.log_likelihood(model_params, lp) * w
            for w, c, lp in zip(
                self.weights_parametric,
                self.parametric_constraints,
                likelihood_params,
            )
        )
        return ll
