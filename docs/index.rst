rxmc
====

``rxmc`` is an orchestration layer for Bayesian calibration of reaction models
to large data sets with flexible likelihood modeling.

It is built around two complementary workflows:

1. **External-sampler orchestration** via :class:`~rxmc.config.CalibrationConfig`
   for drivers such as `black-box-bayes <https://github.com/beykyle/black-box-bayes/>`_.
2. **In-package end-to-end prototyping** via :class:`~rxmc.walker.Walker`
   for smaller problems where you want to run the full MCMC workflow locally.

The package composes curated experimental data (:class:`~rxmc.observation.Observation`),
model predictions (:class:`~rxmc.physical_model.PhysicalModel`), statistical assumptions
(:class:`~rxmc.likelihood_model.LikelihoodModel`), independent data-model pairings
(:class:`~rxmc.constraint.Constraint`), and full calibration problems
(:class:`~rxmc.evidence.Evidence`).

.. toctree::
   :maxdepth: 1
   :caption: Contents

   installation
   api
   examples
