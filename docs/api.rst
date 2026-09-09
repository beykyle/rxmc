API Reference
=============

Configuration
-------------

High-level configuration objects for assembling a calibration problem and
handing it to an external sampler (emcee, dynesty, etc.).

.. autosummary::
   :toctree: generated/
   :nosignatures:

   rxmc.config.CalibrationConfig
   rxmc.config.ParameterConfig

Priors
------

Prior distribution classes that satisfy the generic prior protocol required by
:class:`~rxmc.config.ParameterConfig`.  Any user-defined class with ``logpdf``,
``rvs``, and (optionally) ``prior_transform`` methods can be used directly.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   rxmc.priors.IndependentPrior
   rxmc.priors.TruncatedNormalPrior

Core building blocks
--------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   rxmc.constraint.Constraint
   rxmc.evidence.Evidence
   rxmc.observation.Observation
   rxmc.params.Parameter
   rxmc.physical_model.PhysicalModel
   rxmc.physical_model.Polynomial

Transforms
----------

One low-level, numpy-style transform type shared by observations (the
comparison space, e.g. ``transform=log``), models (parametric transforms such
as a latent normalisation) and covariance terms (coordinate transforms).

.. autosummary::
   :toctree: generated/
   :nosignatures:

   rxmc.transforms.Transform
   rxmc.transforms.as_transform
   rxmc.transforms.scale
   rxmc.transforms.per_observation_scaling

Covariance terms
----------------

The stacked covariance of a :class:`~rxmc.constraint.Constraint` is assembled
additively from :class:`~rxmc.covariance.Term` objects — a single generic type:
a numpy-style callable of a :class:`~rxmc.covariance.TermContext` (the term's
local ``x``/``y``/``ym``) and its parameters, plus a ``kind``
(``"diag"``/``"mode"``/``"matrix"``).  The factory helpers build the common
terms in one line; anything else is a direct ``Term(fn, params, kind=...)``.
A :class:`~rxmc.covariance.StackContext` bundles the stacked ``x``/``y``/``ym``
that a :class:`~rxmc.covariance.ConstraintCovariance` is evaluated on.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   rxmc.covariance.Term
   rxmc.covariance.TermContext
   rxmc.covariance.statistical_term
   rxmc.covariance.normalization_term
   rxmc.covariance.offset_term
   rxmc.covariance.noise_term
   rxmc.covariance.noise_fraction_term
   rxmc.covariance.model_error_term
   rxmc.covariance.systematic_term
   rxmc.covariance.kernel_term
   rxmc.covariance.ones
   rxmc.covariance.ym
   rxmc.covariance.averaging
   rxmc.covariance.x_basis
   rxmc.covariance.exp_growth
   rxmc.covariance.constant_amplitude
   rxmc.covariance.exp_growth_amplitude
   rxmc.covariance.stacked_supports
   rxmc.covariance.ConstraintCovariance
   rxmc.covariance.StackContext

Likelihood functionals
----------------------

Thin functionals of the pre-computed Mahalanobis statistics
``(d2, logdet, n)``; all covariance modeling lives on the
:class:`~rxmc.covariance.ConstraintCovariance`.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   rxmc.likelihood_model.Likelihood
   rxmc.likelihood_model.GaussianLikelihood
   rxmc.likelihood_model.StudentT
   rxmc.likelihood_model.Chi2
   rxmc.likelihood_model.mahalanobis_distance_sqr_cholesky
   rxmc.likelihood_model.log_likelihood

Predictive utilities
--------------------

Posterior-predictive helpers, including Gaussian-process discrepancy
propagation.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   rxmc.predictive.predictive_band
   rxmc.predictive.gp_posterior_predictive
   rxmc.predictive.total_predictive_band

Model comparison
----------------

Sampler-agnostic posterior-predictive checks, held-out scoring, and
nested-sampling evidence bookkeeping.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   rxmc.model_comparison.predictive_draws
   rxmc.model_comparison.coverage_curve
   rxmc.model_comparison.coverage_error
   rxmc.model_comparison.sharpness
   rxmc.model_comparison.heldout_log_predictive
   rxmc.model_comparison.log_posterior_predictive
   rxmc.model_comparison.logz_summary
   rxmc.model_comparison.compare_logz
   rxmc.model_comparison.log_jacobian
   rxmc.model_comparison.split_samples

Sampling
--------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   rxmc.walker.Walker
   rxmc.param_sampling.Sampler
   rxmc.param_sampling.MetropolisHastingsSampler
   rxmc.param_sampling.AdaptiveMetropolisSampler
   rxmc.param_sampling.BatchedAdaptiveMetropolisSampler
   rxmc.proposal.ProposalDistribution
   rxmc.proposal.NormalProposalDistribution
   rxmc.proposal.HalfNormalProposalDistribution
   rxmc.proposal.LogspaceNormalProposalDistribution

Sampling algorithms
-------------------

Low-level sampling functions used internally by the sampler classes.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   rxmc.metropolis_hastings.metropolis_hastings
   rxmc.adaptive_metropolis.adaptive_metropolis

Domain-specific models
----------------------

Reaction-physics observation and model classes for elastic differential
cross sections and isobaric-analog (p,n) cross sections.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   rxmc.elastic_diffxs_observation.ElasticDifferentialXSObservation
   rxmc.elastic_diffxs_observation.momentum_transfer
   rxmc.elastic_diffxs_model.ElasticDifferentialXSModel
   rxmc.ias_pn_observation.IsobaricAnalogPNObservation
   rxmc.ias_pn_model.IsobaricAnalogPNXSModel
