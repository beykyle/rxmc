Installation
============

``rxmc`` requires Python 3.12 or later.  The runtime dependencies are
``numpy``, ``scipy``, ``jax``, ``tinygp``, ``jitr`` and ``exfor-tools``,
installed automatically.  ``jitr`` is taken from its ``jax-core`` branch, the
JAX solver of the next jitr release, until that release is on PyPI.
Importing ``rxmc`` turns on JAX's 64-bit mode for the process.  Until the 1.0 pre-releases appear on PyPI (``pip install
--pre rxmc``), install from GitHub:

.. code-block:: bash

   git clone git@github.com:beykyle/rxmc.git
   cd rxmc
   python -m venv .venv && source .venv/bin/activate
   pip install -e '.[examples]'

Extras:

===============  =============================================================
``examples``     emcee, dynesty, corner, matplotlib, scikit-learn, dill,
                 jupyter: everything the notebooks use
``validation``   ``examples`` plus pytest, nbmake, nbqa, ruff, black, isort
``docs``         sphinx, the pydata theme, myst-nb
===============  =============================================================

The 0.x package, which 1.0 replaces without backwards compatibility, is
preserved at tag `v0.1.0 <https://github.com/beykyle/rxmc/tree/v0.1.0>`_ and
on branch `legacy/0.x <https://github.com/beykyle/rxmc/tree/legacy/0.x>`_.
Pin it with:

.. code-block:: bash

   pip install git+https://github.com/beykyle/rxmc@v0.1.0
