"""
Configuration helpers for calibration workflows.

``CalibrationConfig`` is the main integration surface for external samplers
such as black-box-bayes.  It exposes a flat parameterization of an
``Evidence`` instance together with the sampler-facing methods needed to
evaluate the posterior and generate starting locations.

Prior protocol
--------------
Both ``ParameterConfig`` and ``CalibrationConfig`` accept any prior object
that implements:

* ``logpdf(x: ndarray) -> float`` — log-density at a parameter vector of
  shape ``(ndim,)``.
* ``rvs(n: int) -> ndarray`` — draw ``n`` samples; returned array must have
  shape ``(n, ndim)`` or ``(n,)`` for scalar parameters.

This covers ``scipy.stats`` frozen distributions (both univariate and
multivariate), the built-in :class:`~rxmc.priors.TruncatedNormalPrior`, and
any user-supplied class that satisfies the same interface.

Alternatively, a **list** of univariate ``scipy.stats`` frozen distributions
(one per parameter) may be passed.  This form and any prior class that
implements ``prior_transform(u)`` both support the Dynesty-compatible
:meth:`CalibrationConfig.prior_transform`.
"""

from typing import List, Optional

import numpy as np

from rxmc.evidence import Evidence
from rxmc.params import Parameter


class ParameterConfig:
    """Configuration for a single sector of parameters.

    Bundles a list of :class:`~rxmc.params.Parameter` objects with a prior
    distribution and an initial-proposal distribution.  Instances are passed
    to :class:`CalibrationConfig` to describe the model-parameter sector and
    each likelihood-parameter sector.

    Parameters
    ----------
    params : list of Parameter
        Ordered list of parameters in this sector.
    prior : prior object or list of rv_continuous
        Prior distribution.  May be any object that exposes ``logpdf`` and
        ``rvs`` (e.g. a frozen ``scipy.stats`` multivariate distribution,
        :class:`~rxmc.priors.TruncatedNormalPrior`, or any user-defined class
        with the same interface), **or** a list of frozen univariate
        ``scipy.stats`` distributions — one per parameter.
    initial_proposal_distribution : prior object or list of rv_continuous
        Starting-location proposal distribution.  Accepts the same forms as
        ``prior``.

    Raises
    ------
    ValueError
        If ``params`` is empty.
    ValueError
        If the dimensionality implied by ``prior`` or
        ``initial_proposal_distribution`` does not match ``len(params)``.
    """

    def __init__(
        self,
        params: List[Parameter],
        prior,
        initial_proposal_distribution,
    ):
        self.params = params
        self.ndim = len(params)
        self.prior = prior
        self.initial_proposal_distribution = initial_proposal_distribution

        if self.ndim == 0:
            raise ValueError("Parameter list cannot be empty")

        self._validate_prior_dim(self.prior, "prior")
        self._validate_prior_dim(
            self.initial_proposal_distribution, "initial_proposal_distribution"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_dim(dist) -> Optional[int]:
        """Return the dimensionality of a prior object, or None if unknown."""
        if hasattr(dist, "dim") and isinstance(dist.dim, int):
            return dist.dim
        if hasattr(dist, "mean"):
            return int(np.size(dist.mean))
        return None

    def _validate_prior_dim(self, dist, name: str) -> None:
        if isinstance(dist, list):
            if len(dist) != self.ndim:
                raise ValueError(
                    f"{name} list length ({len(dist)}) does not match "
                    f"number of parameters ({self.ndim})"
                )
        else:
            dim = self._infer_dim(dist)
            if dim is not None and dim != self.ndim:
                raise ValueError(
                    f"{name} dimensionality ({dim}) does not match "
                    f"number of parameters ({self.ndim})"
                )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def x0(self, nwalkers: int) -> np.ndarray:
        """Draw initial walker positions from the proposal distribution.

        Parameters
        ----------
        nwalkers : int
            Number of walkers (rows) to generate.

        Returns
        -------
        ndarray, shape (nwalkers, ndim)
            One initial position per walker.
        """
        dist = self.initial_proposal_distribution
        if isinstance(dist, list):
            samples = [d.rvs(nwalkers) for d in dist]
            return np.column_stack(samples)
        samples = np.atleast_1d(dist.rvs(nwalkers))
        return samples.reshape(nwalkers, -1)

    def prior_logpdf(self, x: np.ndarray) -> float:
        """Evaluate the log prior density at a parameter vector.

        Parameters
        ----------
        x : ndarray, shape (ndim,)
            Parameter vector for this sector.

        Returns
        -------
        float
            Log prior probability at ``x``.
        """
        x = np.atleast_1d(x)
        if isinstance(self.prior, list):
            logpdfs = [dist.logpdf(x[i]) for i, dist in enumerate(self.prior)]
            return float(np.sum(logpdfs))
        return float(self.prior.logpdf(x))

    def prior_transform(self, u: np.ndarray) -> np.ndarray:
        """Map unit-cube coordinates to physical parameters for this sector.

        Supports two forms:

        * **List prior** — each element must expose a ``ppf`` method (all
          frozen ``scipy.stats`` univariate distributions do).
        * **Joint prior with ``prior_transform``** — the prior object must
          implement ``prior_transform(u) -> ndarray`` itself (e.g.
          :class:`~rxmc.priors.TruncatedNormalPrior`).

        Parameters
        ----------
        u : ndarray, shape (ndim,)
            Unit-cube coordinates, each in ``[0, 1)``.

        Returns
        -------
        ndarray, shape (ndim,)
            Physical parameter vector for this sector.

        Raises
        ------
        NotImplementedError
            If the prior is neither a list nor exposes ``prior_transform``.
        """
        if isinstance(self.prior, list):
            return np.array([dist.ppf(u[i]) for i, dist in enumerate(self.prior)])
        if hasattr(self.prior, "prior_transform"):
            return self.prior.prior_transform(u)
        raise NotImplementedError(
            "Prior transform requires either a list of distributions with ppf "
            "or a prior object that implements prior_transform(u)."
        )


class CalibrationConfig:
    """End-to-end configuration for Bayesian calibration.

    Combines an :class:`~rxmc.evidence.Evidence` object with
    :class:`ParameterConfig` instances for the physical-model parameters and
    any parametric likelihood parameters.  The result is a flat
    parameterization suitable for black-box samplers (emcee, Dynesty,
    black-box-bayes, etc.).

    Parameters
    ----------
    evidence : Evidence
        Aggregated experimental constraints.
    model_config : ParameterConfig
        Prior and proposal for the physical-model parameters.
    likelihood_configs : list of ParameterConfig, optional
        One :class:`ParameterConfig` per parametric constraint in
        ``evidence.parametric_constraints``.  Omit or pass ``None`` when
        there are no parametric likelihood models.
    likelihood_scaling : float, optional
        Multiplicative scale applied to the total log-likelihood before
        adding the log-prior.  Useful for tempering or importance
        re-weighting.  Defaults to ``1.0``.

    Raises
    ------
    ValueError
        If ``evidence`` contains no constraints.
    ValueError
        If the model parameters in ``model_config`` do not match those in
        the evidence constraints.
    ValueError
        If the number or parameter lists of ``likelihood_configs`` do not
        match ``evidence.parametric_constraints``.

    Attributes
    ----------
    ndim : int
        Total number of free parameters (model + all likelihood sectors).
    dimensions : ndarray
        Array of sector sizes ``[model_ndim, lc0_ndim, lc1_ndim, ...]``.
    indices : ndarray
        Cumulative split indices derived from ``dimensions``; used by
        :meth:`split_parameters`.
    """

    def __init__(
        self,
        evidence: Evidence,
        model_config: ParameterConfig,
        likelihood_configs: Optional[list] = None,
        likelihood_scaling: Optional[float] = None,
    ):
        self.evidence = evidence
        self.model_config = model_config
        self.likelihood_configs = likelihood_configs or []
        self.ndim = model_config.ndim + sum(lc.ndim for lc in self.likelihood_configs)
        self.likelihood_scaling = (
            1.0 if likelihood_scaling is None else likelihood_scaling
        )

        if (
            len(self.evidence.constraints) == 0
            and len(self.evidence.parametric_constraints) == 0
        ):
            raise ValueError("Evidence must have at least one constraint")
        if np.any(
            [
                c.physical_model.params != self.model_config.params
                for c in self.evidence.constraints
                + self.evidence.parametric_constraints
            ]
        ):
            raise ValueError(
                "Model parameters do not match those in the evidence constraints"
            )
        if len(self.likelihood_configs) != len(self.evidence.parametric_constraints):
            raise ValueError(
                "Likelihood configurations do not match the likelihood models"
                "in the evidence constraints"
            )
        for lc, c in zip(self.likelihood_configs, self.evidence.parametric_constraints):
            if lc.params != c.likelihood.params:
                raise ValueError(
                    "Likelihood parameters do not match those in the evidence constraints"
                )

        self.dimensions = np.array(
            [self.model_config.ndim] + [lc.ndim for lc in self.likelihood_configs]
        )
        self.indices = np.cumsum(self.dimensions)

    # ------------------------------------------------------------------
    # Structural properties
    # ------------------------------------------------------------------

    @property
    def parameter_configs(self) -> list:
        """All parameter sectors in flat sampler order: model first, then likelihoods."""
        return [self.model_config] + self.likelihood_configs

    @property
    def parameters(self) -> List[Parameter]:
        """All :class:`~rxmc.params.Parameter` objects in flat sampler order.

        The order matches the flat parameter vector consumed by
        :meth:`log_posterior`, :meth:`log_likelihood`, and
        :meth:`starting_location`.
        """
        return [param for pc in self.parameter_configs for param in pc.params]

    @property
    def parameter_names(self) -> List[str]:
        """Flat parameter names in sampler order.

        Convenience accessor equivalent to
        ``[p.name for p in self.parameters]``.  Provided for compatibility
        with external drivers such as black-box-bayes that export inference
        results keyed by name.
        """
        return [p.name for p in self.parameters]

    @property
    def prior(self) -> list:
        """Prior distribution objects in parameter-sector order.

        Returns one entry per sector: the model prior first, followed by one
        entry per likelihood sector.  Each entry is whatever was passed as
        ``prior`` to the corresponding :class:`ParameterConfig` — a list of
        univariate distributions, a multivariate distribution, or a custom
        prior object.
        """
        return [pc.prior for pc in self.parameter_configs]

    # ------------------------------------------------------------------
    # Parameter manipulation
    # ------------------------------------------------------------------

    def split_parameters(self, x) -> tuple:
        """Split a flat parameter vector into model and likelihood sub-vectors.

        Parameters
        ----------
        x : ndarray, shape (ndim,)
            Flat parameter vector in sampler order.

        Returns
        -------
        xmodel : ndarray
            Model parameter sub-vector of shape ``(model_config.ndim,)``.
        xlikelihoods : list of ndarray
            One sub-vector per likelihood sector.
        """
        parts = np.split(x, self.indices[:-1])
        return parts[0], parts[1:]

    # ------------------------------------------------------------------
    # Posterior evaluation
    # ------------------------------------------------------------------

    def log_prior(self, x) -> float:
        """Evaluate the joint log prior at a flat parameter vector.

        Parameters
        ----------
        x : ndarray, shape (ndim,)
            Flat parameter vector in sampler order.

        Returns
        -------
        float
            Sum of log prior densities across all sectors.
        """
        xmodel, xlikelihoods = self.split_parameters(x)
        lprior = self.model_config.prior_logpdf(xmodel)
        lprior += sum(
            lc.prior_logpdf(xl) for lc, xl in zip(self.likelihood_configs, xlikelihoods)
        )
        return lprior

    def log_likelihood(self, x) -> float:
        """Evaluate the scaled log likelihood at a flat parameter vector.

        Parameters
        ----------
        x : ndarray, shape (ndim,)
            Flat parameter vector in sampler order.

        Returns
        -------
        float
            ``likelihood_scaling * evidence.log_likelihood(xmodel, xlikelihoods)``.
        """
        xmodel, xlikelihoods = self.split_parameters(x)
        return self.likelihood_scaling * self.evidence.log_likelihood(
            xmodel, xlikelihoods
        )

    def log_posterior(self, x) -> float:
        """Evaluate the log posterior at a flat parameter vector.

        Returns ``-inf`` immediately if either the prior or likelihood is
        non-finite, avoiding unnecessary model evaluations.

        Parameters
        ----------
        x : ndarray, shape (ndim,)
            Flat parameter vector in sampler order.

        Returns
        -------
        float
            ``log_prior(x) + log_likelihood(x)``, or ``-inf`` if either term
            is non-finite.
        """
        lp = self.log_prior(x)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.log_likelihood(x)
        if not np.isfinite(ll):
            return -np.inf
        return lp + ll

    def log_posterior_batch(self, thetas: np.ndarray) -> np.ndarray:
        """Evaluate the log posterior for a batch of parameter vectors.

        Convenience wrapper for external orchestration layers (e.g.
        black-box-bayes).  The current implementation is serial but
        preserves the advertised interface for future parallelisation.

        Parameters
        ----------
        thetas : ndarray, shape (..., ndim)
            Batch of parameter vectors.

        Returns
        -------
        ndarray, shape (n,)
            Log posterior value for each row of ``thetas``.

        Raises
        ------
        ValueError
            If the trailing dimension of ``thetas`` is not ``ndim``.
        """
        thetas = np.atleast_2d(np.asarray(thetas, dtype=float))
        if thetas.shape[-1] != self.ndim:
            raise ValueError(
                f"Expected thetas with trailing dimension {self.ndim}, "
                f"got shape {thetas.shape}"
            )
        return np.array([self.log_posterior(theta) for theta in thetas], dtype=float)

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------

    def starting_location(self, nwalkers: int) -> np.ndarray:
        """Generate initial walker positions across all parameter sectors.

        Parameters
        ----------
        nwalkers : int
            Number of walkers.

        Returns
        -------
        ndarray, shape (nwalkers, ndim)
            Concatenated initial positions from each sector's proposal
            distribution.
        """
        x0_model = self.model_config.x0(nwalkers)
        x0_likelihoods = [
            lc.x0(nwalkers).reshape(nwalkers, lc.ndim) for lc in self.likelihood_configs
        ]
        return np.hstack([x0_model] + x0_likelihoods)

    def prior_transform(self, u) -> np.ndarray:
        """Map unit-cube coordinates to physical parameters (Dynesty interface).

        Delegates to :meth:`ParameterConfig.prior_transform` for each sector,
        which in turn either calls ``ppf`` on each element of a list prior or
        calls ``prior_transform`` directly on a joint prior object.

        Parameters
        ----------
        u : array-like, shape (ndim,)
            Unit-cube coordinates, each in ``[0, 1)``.

        Returns
        -------
        theta : ndarray, shape (ndim,)
            Physical parameter vector.

        Raises
        ------
        NotImplementedError
            If any sector's prior does not support a transform (see
            :meth:`ParameterConfig.prior_transform`).
        ValueError
            If ``u`` does not have length ``ndim``.
        """
        u = np.asarray(u, dtype=float)
        if u.shape[-1] != self.ndim:
            raise ValueError(f"Expected u with length {self.ndim}, got shape {u.shape}")

        eps = np.finfo(float).eps
        u = np.clip(u, eps, 1.0 - eps)

        theta = np.empty_like(u)
        offset = 0
        for pc in self.parameter_configs:
            sector = u[offset : offset + pc.ndim]
            theta[offset : offset + pc.ndim] = pc.prior_transform(sector)
            offset += pc.ndim

        return theta

    # ------------------------------------------------------------------
    # Prediction and conditional posterior
    # ------------------------------------------------------------------

    def predict(self, xmodel) -> list:
        """Generate model predictions for every constraint.

        Parameters
        ----------
        xmodel : ndarray, shape (model_config.ndim,)
            Physical model parameter vector.

        Returns
        -------
        list of ndarray
            Predicted observable for each constraint, in the order
            ``evidence.constraints + evidence.parametric_constraints``.
        """
        constraints = self.evidence.constraints + self.evidence.parametric_constraints
        return [c.predict(*xmodel) for c in constraints]

    def conditional_posterior(self, x_lm, lm_index: int, ym) -> float:
        """Log posterior for one likelihood sector, conditioned on observed data.

        Evaluates ``marginal_log_likelihood(ym, *x_lm) + prior_logpdf(x_lm)``
        for the likelihood sector at ``lm_index``.  Useful for Gibbs-style
        updates where the likelihood parameters are sampled separately from
        the physical model parameters.

        Parameters
        ----------
        x_lm : ndarray, shape (likelihood_configs[lm_index].ndim,)
            Parameter vector for the target likelihood sector.
        lm_index : int
            Index into ``likelihood_configs`` (and
            ``evidence.parametric_constraints``).
        ym : ndarray
            Predicted observable used as the conditioning data.

        Returns
        -------
        float
            Log posterior for this likelihood sector.
        """
        return self.evidence.parametric_constraints[lm_index].marginal_log_likelihood(
            ym, *x_lm
        ) + self.likelihood_configs[lm_index].prior_logpdf(x_lm)
