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

Each observation's comparison-space ``transform`` is applied to the model
prediction here, so the residual ``y - ym`` is formed in that space.  Masks —
point-level on the observations, observation-level via ``mask=`` — select the
*active* rows; terms are always authored over the full stack.
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
    mask : sequence of bool or of int, optional
        Which *observations* are active (all by default): a boolean per
        observation, or the indices of the active ones.  An integer array of
        length ``len(observations)`` holding only 0/1 is ambiguous and
        rejected; pass a bool array or explicit indices.  Combined with each
        observation's own point ``mask`` to give :attr:`active`.

    Attributes
    ----------
    covariance : ConstraintCovariance
        The stacked covariance.
    params : tuple of Parameter
        Free parameters of this constraint: covariance params followed by
        likelihood params (e.g. Student-t ``nu``).
    n_params : int
        ``len(params)``.
    active : np.ndarray
        Stacked indices of the active points.
    n_data_pts : int
        Number of *active* points (the ``n`` of the likelihood).
    n_data_pts_total : int
        Length of the full stack.
    """

    def __init__(
        self,
        observations: list[Observation],
        physical_model: PhysicalModel,
        likelihood=None,
        extra_terms=(),
        include_statistical_term: bool = True,
        mask=None,
    ):
        self.observations = list(observations)
        self.physical_model = physical_model
        self.likelihood = likelihood if likelihood is not None else GaussianLikelihood()

        observations = self.observations
        supports = stacked_supports(observations)
        self._supports = supports
        self.n_data_pts_total = sum(o.n_data_pts for o in observations)

        self.observation_mask = self._observation_mask(mask)
        self.active = np.concatenate(
            [
                s[o.mask] if keep else np.zeros(0, dtype=int)
                for o, s, keep in zip(observations, supports, self.observation_mask)
            ]
        ).astype(int)
        self.n_data_pts = int(self.active.size)

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
        self.covariance = ConstraintCovariance(
            terms, self.n_data_pts_total, blocks=supports, active=self.active
        )

        self.params = tuple(self.covariance.params) + tuple(self.likelihood.params)
        self.n_params = len(self.params)
        self._n_cov_params = self.covariance.n_params

        self._validate_parameter_names()
        if self.covariance.is_constant:
            self._validate_constant_covariance()

    def _observation_mask(self, mask):
        n = len(self.observations)
        if mask is None:
            return np.ones(n, dtype=bool)
        m = np.asarray(mask)
        if m.dtype == bool:
            if m.shape != (n,):
                raise ValueError(f"mask must have one entry per observation ({n})")
            return m
        idx = np.asarray(m, dtype=int)
        if n > 1 and idx.shape == (n,) and np.isin(idx, (0, 1)).all():
            raise ValueError(
                f"ambiguous observation mask: an integer array of length {n} with "
                "only 0/1 entries could be a boolean mask or a list of indices; "
                "pass a bool array or integer indices"
            )
        out = np.zeros(n, dtype=bool)
        out[idx] = True
        return out

    # ------------------------------------------------------------------
    # Masked views
    # ------------------------------------------------------------------

    def masked(self, mask=None, point_masks=None):
        """A new constraint over the same observations/model/terms with new masks.

        Parameters
        ----------
        mask : sequence of bool or of int, optional
            Observation-level mask (see the constructor); ``None`` keeps this
            constraint's.
        point_masks : sequence of array_like of bool, optional
            One point mask per observation (``None`` entries keep that
            observation's current mask).

        Notes
        -----
        The new constraint shares the ``Term``/``Parameter`` objects with this
        one, so its parameter vector is identical — it is a *view* for
        evaluating the same likelihood on a different subset (e.g. held-out
        scoring), not an independent constraint to place in the same
        :class:`~rxmc.evidence.Evidence`.  Sharing the terms is safe because
        both constraints stack the same observations in the same order, so the
        terms' bound supports and cached ``x``-dependent values stay valid.
        """
        observations = list(self.observations)
        if point_masks is not None:
            if len(point_masks) != len(observations):
                raise ValueError("point_masks must have one entry per observation")
            observations = [
                o if pm is None else o.masked(pm)
                for o, pm in zip(observations, point_masks)
            ]
        # hand over the already-built term list (statistical diagonals included)
        # so the view shares the exact same Term/Parameter objects
        return Constraint(
            observations,
            self.physical_model,
            likelihood=self.likelihood,
            extra_terms=self.covariance.terms,
            include_statistical_term=False,
            mask=self.observation_mask if mask is None else mask,
        )

    def complement(self):
        """The held-out counterpart: every currently inactive point becomes active
        and every active point inactive.

        An observation excluded wholesale at the constraint level is therefore
        restored in full; an active observation whose point mask keeps every
        point is dropped wholesale.  See :meth:`masked` for the sharing caveat.
        """
        # an observation dropped wholesale at the constraint level is restored
        # in full; an active one has its point mask flipped
        point_masks = [
            ~o.mask if keep else np.ones(o.n_data_pts, dtype=bool)
            for o, keep in zip(self.observations, self.observation_mask)
        ]
        obs_mask = [
            not keep or o.n_active < o.n_data_pts
            for o, keep in zip(self.observations, self.observation_mask)
        ]
        return self.masked(mask=np.array(obs_mask), point_masks=point_masks)

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
        ctx = StackContext.constant(self._x_stacked, self._y_stacked, self._supports)
        cov = self.covariance
        try:
            # warm whichever factorisation the likelihood path will use
            if cov.uses_block_path:
                cov.block_cholesky(ctx)
            else:
                cov.cholesky(ctx)
        except np.linalg.LinAlgError as err:
            labels = [
                o.label or f"observation {i}" for i, o in enumerate(self.observations)
            ]
            Sigma = self.covariance.matrix(ctx)
            zero_rows = self.active[np.diag(Sigma)[self.active] == 0.0]
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
                "(extra_terms=[*obs.systematic_terms()]; for a multi-observation "
                "constraint place them with support= from "
                "rxmc.covariance.stacked_supports(observations)), add a "
                "noise_term or a fixed Term covering those points, or compose "
                "the full covariance explicitly with include_statistical_term=False."
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
            y = np.asarray(y, dtype=float)
            if y.shape != o.y.shape:
                raise ValueError(
                    f"prediction shape {y.shape} does not match observation shape "
                    f"{o.y.shape}"
                )
            ym_arrays.append(o.transform(y))
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

    def _evaluate(self, ctx, cov_params, statistic, *, invalid=-np.inf):
        """Evaluate ``statistic(d2, logdet, n, *like_params)`` on the stack.

        ``invalid`` is returned when the prediction is not finite on the active
        points (e.g. a non-positive prediction under a log comparison space):
        ``-inf`` for a log likelihood (default), ``+inf`` for a chi-squared.
        """
        cov_part, like_part = self._split(cov_params)
        if not np.all(np.isfinite(ctx.ym[self.active])):
            return invalid
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
        return self._evaluate(ctx, cov_params, self.likelihood.chi2, invalid=np.inf)

    def predict(self, *model_params, raw=False):
        """Predictions for each observation (all points, comparison space).

        With ``raw=True`` the predictions are returned in physical space (the
        model's own output, before each observation's ``transform``).
        """
        ym = [self.physical_model(obs, *model_params) for obs in self.observations]
        if raw:
            return ym
        return [o.transform(y) for o, y in zip(self.observations, ym)]

    def _stack_and_covariance(self, model_params, cov_params, active_only):
        """``(ctx, Sigma)`` at a parameter point; ``Sigma`` is a fresh copy."""
        ctx = self._stack(model_params)
        cov_part, _ = self._split(cov_params)
        if active_only:
            return ctx, np.array(self.covariance.active_matrix(ctx, *cov_part))
        return ctx, np.array(self.covariance.matrix(ctx, *cov_part))

    def predict_and_covariance(self, model_params, cov_params=()):
        """Stacked prediction and covariance on the active points, one model call.

        Returns
        -------
        (np.ndarray, np.ndarray)
            ``(ym, Sigma)`` with ``ym`` of length ``n_data_pts`` (comparison
            space) and ``Sigma`` a fresh ``(n_data_pts, n_data_pts)`` array.
        """
        ctx, Sigma = self._stack_and_covariance(model_params, cov_params, True)
        return ctx.ym[self.active], Sigma

    @property
    def y(self) -> np.ndarray:
        """Stacked observed data on the active points (comparison space)."""
        return self._y_stacked[self.active]

    @property
    def x(self) -> np.ndarray:
        """Stacked independent variable on the active points."""
        return self._x_stacked[self.active]

    @property
    def log_jacobian(self) -> float:
        """Sum of the observations' comparison-space log-Jacobians (active points)."""
        return float(
            sum(
                o.log_jacobian
                for o, keep in zip(self.observations, self.observation_mask)
                if keep
            )
        )

    def covariance_matrix(self, model_params, cov_params=(), active_only=True):
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
        active_only : bool, optional
            Restrict to the active points (default); ``False`` returns the
            full stacked matrix.

        Returns
        -------
        np.ndarray
            Shape ``(n_data_pts, n_data_pts)`` (active points) or
            ``(n_data_pts_total, n_data_pts_total)`` when ``active_only=False``.
            A fresh copy (safe to mutate; never aliases the internal cache).
        """
        _, Sigma = self._stack_and_covariance(model_params, cov_params, active_only)
        return Sigma

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
            if self.observation_mask[i]
        )

    def empirical_coverage(
        self, ylow: list[np.ndarray], yhigh: list[np.ndarray], xlim=None
    ):
        """Fraction of active data points within a predictive interval.

        ``nan`` when the constraint has no active points.
        """
        if self.n_data_pts == 0:
            return float("nan")
        return self.num_pts_within_interval(ylow, yhigh, xlim) / self.n_data_pts
