"""
Evidence: aggregate of independent constraints for Bayesian calibration.

An :class:`Evidence` object collects multiple :class:`~rxmc.constraint.Constraint`
objects that share the same physical-model parameters.  It computes a joint log
likelihood by summing the individual constraint log likelihoods (optionally
weighted).  Each constraint owns its own (possibly correlated) covariance over the
stack of its observations; constraints are assumed independent of one another, so
``Evidence`` is a plain sum.  Parametric constraints (those with free covariance /
likelihood parameters) are auto-detected via ``constraint.n_params > 0``.
"""

import numpy as np

from .constraint import Constraint


class Evidence:
    """A collection of independent constraints sharing a common physical model.

    Parameters
    ----------
    constraints : list of Constraint
        All constraints.  Those with ``n_params > 0`` are exposed (in order) as
        :attr:`parametric_constraints`.
    weights : np.ndarray, optional
        1-D array of per-constraint weights.  Defaults to all ones.

    Raises
    ------
    ValueError
        If *constraints* is empty, if any constraint uses different
        physical-model parameters than the first, if the model is
        under-constrained, or if *weights* does not match the constraint count.

    Attributes
    ----------
    constraints : list of Constraint
        All constraints.
    parametric_constraints : list of Constraint
        The subset with ``n_params > 0``, in the order they appear.
    parametric_indices : list of int
        Global index into :attr:`constraints` for each entry of
        :attr:`parametric_constraints` (e.g. to look up its :attr:`weights`
        entry).
    """

    def __init__(
        self,
        constraints: list[Constraint] | None = None,
        weights: np.ndarray = None,
    ):
        constraints = list(constraints or [])
        if len(constraints) == 0:
            raise ValueError("'constraints' must not be empty")

        self.constraints = constraints
        self.model_params = constraints[0].physical_model.params
        for constraint in self.constraints:
            if constraint.physical_model.params != self.model_params:
                raise ValueError(
                    "All constraints must use the same physical model parameters"
                )

        self._validate_constraint_params()

        parametric = [(i, c) for i, c in enumerate(self.constraints) if c.n_params > 0]
        self.parametric_indices = [i for i, _ in parametric]
        self.parametric_constraints = [c for _, c in parametric]
        self.n_likelihood_params = sum(c.n_params for c in self.parametric_constraints)

        self.n_params = len(self.model_params) + self.n_likelihood_params
        self.n_data_pts = sum(c.n_data_pts for c in self.constraints)
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

    def _validate_constraint_params(self):
        """Reject cross-constraint parameter sharing and duplicate names.

        Covariance/likelihood parameters are constraint-scoped (see
        ``covariance_refactor.md`` §8): the same ``Parameter`` object in two
        constraints would silently be sampled as two independent values.
        Names must also be unique across the whole Evidence — they label
        sampler columns, priors, and corner-plot axes.
        """
        seen_id = {}  # id(p) -> constraint index
        seen_name = {}  # p.name -> constraint index
        for ci, c in enumerate(self.constraints):
            for p in c.params:
                if id(p) in seen_id:
                    raise ValueError(
                        f"Parameter '{p.name}' is the same object in constraints "
                        f"{seen_id[id(p)]} and {ci}. Covariance/likelihood "
                        "parameters are constraint-scoped and cannot be shared "
                        "across constraints. To model a systematic shared "
                        "between datasets, place those datasets in ONE "
                        "Constraint with a cross-block coupling term."
                    )
                seen_id[id(p)] = ci
                if p.name in seen_name:
                    raise ValueError(
                        f"Duplicate parameter name '{p.name}' in constraints "
                        f"{seen_name[p.name]} and {ci}. Parameter names label "
                        "sampler columns and must be unique across the "
                        "Evidence; rename one (e.g. suffix it with the dataset "
                        "label)."
                    )
                seen_name[p.name] = ci

    def log_likelihood(self, model_params, cov_params: list | None = None):
        """Weighted sum of log likelihoods over all constraints.

        Parameters
        ----------
        model_params : tuple
            Physical-model parameters.
        cov_params : list of tuple, optional
            One tuple of constraint parameters per entry in
            :attr:`parametric_constraints` (in that order).  Defaults to ``[]``.

        Returns
        -------
        float
            Total weighted log likelihood.
        """
        cov_params = cov_params or []
        if len(cov_params) != len(self.parametric_constraints):
            raise ValueError(
                f"Expected {len(self.parametric_constraints)} constraint parameter "
                f"tuples, got {len(cov_params)}"
            )

        ll = 0.0
        pidx = 0
        for w, c in zip(self.weights, self.constraints):
            cp = ()
            if c.n_params > 0:
                cp = cov_params[pidx]
                pidx += 1
            ll += c.log_likelihood(model_params, cp) * w
        return ll

    def weighted_marginal_log_likelihood(self, lm_index, ym, *cov_params):
        """Weighted marginal log likelihood of one parametric constraint.

        Applies the same :attr:`weights` entry that :meth:`log_likelihood`
        uses for this constraint, so Gibbs-style conditional updates target
        the same joint distribution as the model block.

        Parameters
        ----------
        lm_index : int
            Index into :attr:`parametric_constraints`.
        ym : list of np.ndarray
            One prediction array per observation of that constraint.
        *cov_params : float
            The constraint's parameters, in its ``params`` order.
        """
        w = self.weights[self.parametric_indices[lm_index]]
        c = self.parametric_constraints[lm_index]
        return w * c.marginal_log_likelihood(ym, *cov_params)
