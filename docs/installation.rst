Installation
============

``rxmc`` requires Python 3.12 or later.  The runtime dependencies are
``numpy``, ``scipy``, ``jax``, ``tinygp``, ``jitr`` and ``exfor-tools``,
installed automatically.  ``jitr`` is taken from its ``jax-core`` branch, the
JAX solver of the next jitr release, until that release is on PyPI; the rxmc
1.0 pre-releases wait on it too, since PyPI does not accept a git dependency.
Importing ``rxmc`` turns on JAX's 64-bit mode for the process.  For a GPU,
install a CUDA build of ``jax`` first (see the JAX installation guide).
Install from GitHub:

.. code-block:: bash

   git clone git@github.com:beykyle/rxmc.git
   cd rxmc
   python -m venv .venv && source .venv/bin/activate
   pip install -e '.[examples]'

Extras:

===============  =============================================================
``examples``     emcee, dynesty, blackjax, corner, matplotlib, dill,
                 jupyter: everything the notebooks use
``validation``   ``examples`` plus pytest, nbmake, nbqa, ruff, black, isort,
                 and scikit-learn (a reference for the kernel tests)
``docs``         sphinx, the pydata theme, myst-nb
``samplers``     ptemcee, pocomc, PyMC: the extra samplers of
                 ``benchmarks/bench_samplers.py`` (pocomc brings torch; with
                 ``uv`` it comes from PyTorch's CPU wheels, with ``pip``
                 install ``torch`` from ``--index-url
                 https://download.pytorch.org/whl/cpu`` first)
===============  =============================================================

The 0.x package, which 1.0 replaces without backwards compatibility, is
preserved at tag `v0.1.0 <https://github.com/beykyle/rxmc/tree/v0.1.0>`_ and
on branch `legacy/0.x <https://github.com/beykyle/rxmc/tree/legacy/0.x>`_.
Pin it with:

.. code-block:: bash

   pip install git+https://github.com/beykyle/rxmc@v0.1.0
