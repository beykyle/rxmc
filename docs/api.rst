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
   rxmc.observation.FixedCovarianceObservation
   rxmc.params.Parameter
   rxmc.physical_model.PhysicalModel
   rxmc.physical_model.Polynomial

Likelihood models
-----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   rxmc.likelihood_model.LikelihoodModel
   rxmc.likelihood_model.FixedCovarianceLikelihood
   rxmc.likelihood_model.Chi2LikelihoodModel
   rxmc.likelihood_model.ParametricLikelihoodModel
   rxmc.likelihood_model.UnknownNoiseErrorModel
   rxmc.likelihood_model.UnknownNoiseFractionErrorModel
   rxmc.likelihood_model.UnknownNormalizationModel
   rxmc.likelihood_model.UnknownNormalizationErrorModel
   rxmc.likelihood_model.UnknownModelError
   rxmc.likelihood_model.StudentTLikelihoodModel
   rxmc.correlated_discrepancy_likelihood_model.SklearnKernelGPDiscrepancyModel

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
