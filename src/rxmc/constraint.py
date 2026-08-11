"""
Constraint: the maximal block of mutually-correlated data.

A :class:`Constraint` pairs one or more :class:`~rxmc.observation.Observation`
objects with a :class:`~rxmc.physical_model.PhysicalModel` and a likelihood
functional (:class:`~rxmc.likelihood_model.GaussianLikelihood` by default).  It
owns **one** multivariate distribution over the *stacked* vector of all its
observations, whose covariance is a
:class:`~rxmc.covariance.ConstraintCovariance` assembled from
:class:`~rxmc.covariance.Term` s.

Each observation *i* occupies a contiguous slice of the stacked vector.  The
default covariance is the concatenation of every observation's statistical
diagonal (strictly block-diagonal — reproducing the old summed independent
likelihoods).  Correlated modes — a dataset's own normalisation/offset
systematic, an unknown-noise term, or a cross-dataset coupling — are supplied as
``extra_terms``.
"""

import numpy as np

from .covariance import ConstraintCovariance, StackContext, stacked_supports
from .likelihood_model import GaussianLikelihood
from .observation import Observation
from .physical_model import PhysicalModel


class Constraint:
    """Pair observations with a physical model and a stacked covariance.

    Parameters
    ----------
    observations : list of Observation
        The observed data that the model will attempt to reproduce.  Together
        they form one stacked vector ``y = [y1; y2; ...]``.
    physical_model : PhysicalModel
        Model that predicts the observed data.
    likelihood : object, optional
        Likelihood functional of ``(d2, logdet, n, *like_params)``.  Defaults to
        :class:`~rxmc.likelihood_model.GaussianLikelihood`.
    extra_terms : sequence of Term, optional
        Additional covariance contributions beyond the statistical diagonals —
        local systematics or cross-block couplings.
    include_statistical_term : bool, optional
        When ``True`` (default) each observation's statistical diagonal
        (``obs.statistical_term``) is added automatically.  Set ``False`` to omit
        it and compose the *entire* covariance from ``extra_terms`` — e.g. to let
        an unknown-noise term (:func:`~rxmc.covariance.noise_term`) *replace* the
        reported statistics rather than add to them.

    Attributes
    ----------
    covariance : ConstraintCovariance
        The stacked covariance.
    params : tuple of Parameter
        Free parameters of this constraint: covariance params followed by
        likelihood params (e.g. Student-t ``nu``).
    n_params : int
        ``len(params)``.
    """

    def __init__(
        self,
        observations: list[Observation],
        physical_model: PhysicalModel,
        likelihood=None,
        extra_terms=(),
        include_statistical_term: bool = True,
    ):
        self.observations = observations
        self.physical_model = physical_model
        self.likelihood = likelihood if likelihood is not None else GaussianLikelihood()

        supports = stacked_supports(observations)
        self._supports = supports
        self.n_data_pts = sum(o.n_data_pts for o in observations)

        # x and y are invariant per constraint; stack them once.  Frozen
        # because they are shared across every likelihood evaluation.
        self._x_stacked = np.concatenate([o.x for o in observations])
        self._y_stacked = np.concatenate([o.y for o in observations])
        self._x_stacked.setflags(write=False)
        self._y_stacked.setflags(write=False)

        if include_statistical_term:
            terms = [obs.statistical_term(s) for obs, s in zip(observations, supports)]
        else:
            terms = []
        terms += list(extra_terms)
        self.covariance = ConstraintCovariance(terms, self.n_data_pts, blocks=supports)

        self.params = tuple(self.covariance.params) + tuple(self.likelihood.params)
        self.n_params = len(self.params)
        self._n_cov_params = self.covariance.n_params

        self._validate_parameter_names()
        if self.covariance.is_constant:
            self._validate_constant_covariance()

    def _validate_parameter_names(self):
        """Reject ambiguous parameter names within this constraint.

        Sharing one sampled value between terms works by referencing the *same*
        ``Parameter`` object (identity); two distinct objects with one name would
        silently become two sampler columns with identical labels.
        """
        model_names = {p.name for p in self.physical_model.params}
        seen = set()
        for p in self.params:
            if p.name in seen:
                raise ValueError(
                    f"Constraint has multiple distinct parameters named "
                    f"'{p.name}'. To share one sampled value between terms, "
                    "pass the SAME Parameter object to each term; otherwise "
                    "give each parameter a unique name."
                )
            seen.add(p.name)
            if p.name in model_names:
                raise ValueError(
                    f"Constraint parameter '{p.name}' collides with a "
                    "physical-model parameter of the same name; rename the "
                    "covariance/likelihood parameter."
                )

    def _validate_constant_covariance(self):
        """Fail fast on a singular constant covariance (also warms the cache).

        A routine trigger is an EXFOR measurement reporting no statistical
        error: ``from_measurement`` then yields an all-zero ``y_stat_err``, and
        without an extra covariance term the stacked covariance is singular.
        Catching it here names the offending dataset instead of surfacing an
        opaque ``LinAlgError`` deep inside a sampler.
        """
        try:
            self.covariance.cholesky(None)
        except np.linalg.LinAlgError as err:
            labels = [
                o.label or f"observation {i}" for i, o in enumerate(self.observations)
            ]
            Sigma = self.covariance.matrix(None)
            zero_rows = np.flatnonzero(np.diag(Sigma) == 0.0)
            offenders = [
                label
                for label, s in zip(labels, self._supports)
                if np.isin(s, zero_rows).any()
            ]
            msg = (
                f"Constraint covariance over [{', '.join(labels)}] is singular "
                "(Cholesky factorization failed)."
            )
            if offenders:
                msg += (
                    f" The covariance diagonal is zero on rows belonging to "
                    f"{offenders}: these datasets report zero statistical error "
                    "and no other covariance term covers their points."
                )
            msg += (
                " Remedies: pass the dataset's reported systematics as terms "
                "(extra_terms=[*obs.systematic_terms(support)], with supports "
                "from rxmc.covariance.stacked_supports(observations)), add a "
                "noise_term or DenseTerm covering those points, or compose the "
                "full covariance explicitly with include_statistical_term=False."
            )
            raise ValueError(msg) from err

    # ------------------------------------------------------------------
    # Stacking
    # ------------------------------------------------------------------

    def _stack(self, model_params):
        ym = [self.physical_model(o, *model_params) for o in self.observations]
        return self._stack_from_predictions(ym)

    def _stack_from_predictions(self, ym: list):
        if len(ym) != len(self.observations):
            raise ValueError(
                f"expected {len(self.observations)} prediction arrays, got {len(ym)}"
            )
        ym_arrays = []
        for o, y in zip(self.observations, ym):
            y = np.asarray(y)
            if y.shape != o.y.shape:
                raise ValueError(
                    f"prediction shape {y.shape} does not match observation shape "
                    f"{o.y.shape}"
                )
            ym_arrays.append(y)
        return StackContext(
            x=self._x_stacked,
            y=self._y_stacked,
            ym=np.concatenate(ym_arrays),
            supports=self._supports,
        )

    def _split(self, params):
        params = tuple(params)
        if len(params) != self.n_params:
            names = ", ".join(p.name for p in self.params) or "none"
            raise ValueError(
                f"Constraint expects {self.n_params} parameter(s) [{names}], "
                f"got {len(params)}"
            )
        return params[: self._n_cov_params], params[self._n_cov_params :]

    # ------------------------------------------------------------------
    # Likelihood
    # ------------------------------------------------------------------

    def _evaluate(self, ctx, cov_params, statistic):
        cov_part, like_part = self._split(cov_params)
        d2, logdet = self.covariance.stacked_distance(ctx, cov_part)
        return statistic(d2, logdet, self.n_data_pts, *like_part)

    def log_likelihood(self, model_params, cov_params=()):
        """Log likelihood of the stacked observations given the model.

        Parameters
        ----------
        model_params : tuple
            Physical-model parameters.
        cov_params : tuple, optional
            Constraint parameters: covariance params followed by likelihood
            params, in :attr:`params` order.
        """
        ctx = self._stack(model_params)
        return self._evaluate(ctx, cov_params, self.likelihood.log_likelihood)

    def marginal_log_likelihood(self, ym: list, *cov_params):
        """Log likelihood from pre-computed predictions (Gibbs hook).

        Parameters
        ----------
        ym : list of np.ndarray
            One prediction array per observation (no physical-model re-eval).
        *cov_params : float
            Constraint parameters, in :attr:`params` order.
        """
        ctx = self._stack_from_predictions(ym)
        return self._evaluate(ctx, cov_params, self.likelihood.log_likelihood)

    def chi2(self, model_params, cov_params=()):
        """Generalised chi-squared (Mahalanobis distance) over the stack.

        ``cov_params`` is the full constraint tuple in :attr:`params` order,
        including likelihood params (e.g. Student-t ``nu``) even though the
        chi-squared statistic ignores them.
        """
        ctx = self._stack(model_params)
        return self._evaluate(ctx, cov_params, self.likelihood.chi2)

    def predict(self, *model_params):
        """Generate predictions for each observation."""
        return [self.physical_model(obs, *model_params) for obs in self.observations]

    def covariance_matrix(self, model_params, cov_params=()):
        """Assemble the stacked covariance matrix Σ at a parameter point.

        Convenience accessor (e.g. for visualising the off-diagonal block
        structure of correlated observations).

        Parameters
        ----------
        model_params : tuple
            Physical-model parameters (needed for prediction-scaled terms).
        cov_params : tuple, optional
            Constraint parameters: covariance params followed by likelihood
            params, in :attr:`params` order (matching :meth:`log_likelihood`).

        Returns
        -------
        np.ndarray, shape (n_data_pts, n_data_pts)
            A fresh copy (safe to mutate; never aliases the internal cache).
        """
        ctx = self._stack(model_params)
        cov_part, _ = self._split(cov_params)
        return np.array(self.covariance.matrix(ctx, *cov_part))

    # ------------------------------------------------------------------
    # Coverage diagnostics
    # ------------------------------------------------------------------

    def num_pts_within_interval(
        self, ylow: list[np.ndarray], yhigh: list[np.ndarray], xlim=None
    ):
        """Count data points that fall within a predictive interval."""
        return sum(
            obs.num_pts_within_interval(ylow[i], yhigh[i], xlim)
            for i, obs in enumerate(self.observations)
        )

    def empirical_coverage(
        self, ylow: list[np.ndarray], yhigh: list[np.ndarray], xlim=None
    ):
        """Fraction of data points within a predictive interval."""
        return self.num_pts_within_interval(ylow, yhigh, xlim) / self.n_data_pts
