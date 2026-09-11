API reference
=============

The public surface, in the order the design document introduces it.  Every
name below is importable from ``rxmc`` or the module shown.

Building blocks
---------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   rxmc.params.Parameter
   rxmc.data.Dataset
   rxmc.data.from_measurement
   rxmc.model.Model
   rxmc.model.Predictor
   rxmc.model.polynomial
   rxmc.constraint.Comparison
   rxmc.constraint.Constraint
   rxmc.problem.Problem

Transforms
----------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   rxmc.transforms.Transform
   rxmc.transforms.as_transform
   rxmc.transforms.identity
   rxmc.transforms.log
   rxmc.transforms.exp
   rxmc.transforms.scale

Covariance terms
----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   rxmc.terms.Term
   rxmc.terms.TermContext
   rxmc.terms.KernelTerm
   rxmc.terms.statistical
   rxmc.terms.offset
   rxmc.terms.normalization
   rxmc.terms.noise
   rxmc.terms.noise_fraction
   rxmc.terms.model_error
   rxmc.terms.systematic
   rxmc.terms.kernel
   rxmc.terms.ones
   rxmc.terms.ym
   rxmc.terms.averaging
   rxmc.terms.x_basis
   rxmc.terms.exp_growth
   rxmc.terms.constant_amplitude
   rxmc.terms.exp_growth_amplitude

Likelihood functionals
----------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   rxmc.likelihood.Likelihood
   rxmc.likelihood.Gaussian
   rxmc.likelihood.StudentT
   rxmc.likelihood.Chi2

The structured covariance
-------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   rxmc.covariance.StructuredCovariance
   rxmc.covariance.chol_logdet
   rxmc.problem.CompiledConstraint
   rxmc.problem.ParameterIndex

Diagnostics
-----------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   rxmc.diagnostics.predictive_draws
   rxmc.diagnostics.coverage_curve
   rxmc.diagnostics.coverage_error
   rxmc.diagnostics.sharpness
   rxmc.diagnostics.heldout_log_predictive
   rxmc.diagnostics.log_posterior_predictive
   rxmc.diagnostics.logz_summary
   rxmc.diagnostics.compare_logz

Predictive
----------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   rxmc.predictive.gp_posterior_predictive
   rxmc.predictive.predictive_band
   rxmc.predictive.total_predictive_band

Reactions
---------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   rxmc.reactions.elastic.ElasticXS
   rxmc.reactions.ias.IsobaricAnalogPN
   rxmc.reactions.elastic.rutherford
   rxmc.reactions.elastic.momentum_transfer
   rxmc.reactions.elastic.set_up_solver
   rxmc.reactions.ias.set_up_solver

Units
-----

.. autosummary::
   :toctree: generated/
   :nosignatures:

   rxmc.units.parse_unit
   rxmc.units.check_angle_grid
