Installation
============

``rxmc`` requires Python 3.12 or later.  The runtime dependencies are
``numpy``, ``scipy``, ``jitr >= 3.0`` and ``exfor-tools``, installed
automatically.  Until the 1.0 pre-releases appear on PyPI (``pip install
--pre rxmc``), install from the branch:

.. code-block:: bash

   git clone -b rewrite git@github.com:beykyle/rxmc.git
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

The 0.x package is at tag ``v0.1.0`` and on branch ``legacy/0.x``.
