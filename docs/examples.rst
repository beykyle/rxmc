Examples
========

The following notebooks demonstrate the main workflows and features of ``rxmc``.
They are rendered with their pre-computed outputs; to run them locally, install
the example dependencies first:

.. code-block:: bash

   pip install -ve '.[examples]'
   jupyter lab

Basic calibration
-----------------

.. toctree::
   :maxdepth: 1

   examples/linear_calibration_demo.ipynb
   examples/systematic_err_demo.ipynb

Realistic nuclear physics calibration
--------------------------------------

.. toctree::
   :maxdepth: 1

   examples/30s_optical_potential_calibration.ipynb
   examples/calibration_config_emcee_dynesty.ipynb

Advanced topics
---------------

.. toctree::
   :maxdepth: 1

   examples/normalization_inference.ipynb
   examples/sampling_algos.ipynb
   examples/overconfidence.ipynb
