Installation
============

``rxmc`` requires Python 3.12 or later.  Core dependencies are listed in
``requirements.txt`` and are installed automatically.  The 1.0 pre-releases
will be published to PyPI as ``v1.0.0a1``, ``b1``, ``rc1`` and installable
with ``pip install --pre rxmc``; until then install from the branch:

.. code-block:: bash

   git clone -b rewrite git@github.com:beykyle/rxmc.git
   cd rxmc
   python -m venv .venv && source .venv/bin/activate
   pip install -e '.[validation]'      # or '.[examples]' for the notebooks only

The 0.x package is at tag ``v0.1.0`` and on branch ``legacy/0.x``.
