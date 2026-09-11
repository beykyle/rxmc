rxmc
====

``rxmc`` calibrates reaction models to experimental data by Bayesian
inference, with the error model, statistical and systematic, experimental
and theoretical, declared explicitly as part of the problem, and the
calibration driven by external samplers (emcee, dynesty, black-box-bayes).

- :doc:`recipes` states every supported use case with its spelling and the
  behaviour to expect; each recipe is a test.
- :doc:`examples` are the tutorials for the recipes.
- :doc:`design` is the maintainer's description of the library.
- :doc:`api` is the reference.

The 1.0 rewrite lives on the ``rewrite`` branch until its release; the 0.x
package is preserved at tag ``v0.1.0`` and on branch ``legacy/0.x``.

.. toctree::
   :maxdepth: 1
   :caption: Contents

   installation
   recipes
   examples
   design
   api
