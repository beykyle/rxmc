Examples
========

The notebooks below are the tutorials for the recipes in :doc:`recipes`;
each names the recipes it teaches in its first cell.  They are rendered with
their committed outputs, which are full runs (``python examples/run.py``);
the converged-tier CI workflow only checks that they still run, with every
sampler budget cut tenfold.
To run them yourself, install the example dependencies first::

   pip install -e '.[examples]'
   jupyter lab examples

Calibration basics
------------------

.. toctree::
   :maxdepth: 1

   examples/linear_calibration.ipynb
   examples/normalization_and_covariance_structure.ipynb

Beyond the Gaussian
-------------------

.. toctree::
   :maxdepth: 1

   examples/gp_discrepancy.ipynb
   examples/robust_likelihoods.ipynb

Reactions and studies
---------------------

.. toctree::
   :maxdepth: 1

   examples/local_optical_model_calibration.ipynb
   examples/alpha_ca_error_model_comparison.ipynb
   examples/hierarchical_calibration.ipynb

Samplers
--------

.. toctree::
   :maxdepth: 1

   examples/gradient_based_sampling.ipynb
