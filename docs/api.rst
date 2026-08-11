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
   rxmc.physical_model.ScaledModel
   rxmc.physical_model.PerObservationScaledModel

Covariance terms
----------------

The stacked covariance of a :class:`~rxmc.constraint.Constraint` is assembled
additively from :class:`~rxmc.covariance.Term` objects.  The factory helpers
are the primary authoring API; the term primitives underneath are available for
custom modes.  Context-dependent terms and basis callables receive a
``StackContext`` bundling the stacked ``x``/``y``/``ym`` arrays and block
supports (see the :mod:`rxmc.covariance` module docstring).

.. autosummary::
   :toctree: generated/
   :nosignatures:

   rxmc.covariance.statistical_term
   rxmc.covariance.normalization_term
   rxmc.covariance.offset_term
   rxmc.covariance.noise_term
   rxmc.covariance.noise_fraction_term
   rxmc.covariance.model_error_term
   rxmc.covariance.discrepancy_term
   rxmc.covariance.stacked_supports
   rxmc.covariance.Term
   rxmc.covariance.DenseTerm
   rxmc.covariance.DiagonalTerm
   rxmc.covariance.RankOneTerm
   rxmc.covariance.KernelTerm
   rxmc.covariance.ConstraintCovariance

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
   rxmc.elastic_diffxs_model.ElasticDifferentialXSModel
   rxmc.ias_pn_observation.IsobaricAnalogPNObservation
   rxmc.ias_pn_model.IsobaricAnalogPNXSModel
