Examples
========

The notebooks below are the tutorials for the recipes in :doc:`recipes`;
each names the recipes it teaches in its first cell.  They are rendered with
their committed outputs and re-executed by the converged-tier CI workflow.
To run them yourself, install the example dependencies first::

   pip install -e '.[examples]'
   jupyter lab examples

Calibration basics
------------------

.. toctree::
   :maxdepth: 1

   examples/linear_calibration.ipynb
   examples/error_models.ipynb
   examples/sharing_error_models.ipynb
   examples/normalization_and_covariance_structure.ipynb

Beyond the Gaussian
-------------------

.. toctree::
   :maxdepth: 1

   examples/gp_discrepancy.ipynb
   examples/robust_likelihoods.ipynb
   examples/error_scale_and_usu.ipynb

Reactions and studies
---------------------

.. toctree::
   :maxdepth: 1

   examples/local_optical_model_calibration.ipynb
   examples/alpha_ca_error_model_comparison.ipynb
   examples/hierarchical_calibration.ipynb
