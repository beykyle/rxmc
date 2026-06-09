Installation
============

Requirements
------------

``rxmc`` requires Python 3.10 or later. Core dependencies are listed in
``requirements.txt`` and are installed automatically.

Development / local use
-----------------------

.. code-block:: bash

   git clone git@github.com:beykyle/rxmc.git
   cd rxmc
   pip install -ve .

It is strongly recommended to use an isolated environment.

``conda`` / ``mamba``
---------------------

.. code-block:: bash

   conda env create -f environment.yml
   conda activate rxmc
   pip install -ve . --no-deps

``venv``
--------

.. code-block:: bash

   python -m venv .rxmc
   source .rxmc/bin/activate
   pip install -r requirements.txt
   pip install -ve .

``uv``
------

.. code-block:: bash

   uv env create
   uv env use python
   uv install -e .

Optional extras
---------------

Install example notebook runtime dependencies:

.. code-block:: bash

   pip install -ve '.[examples]'

Install the full validation toolchain (formatting, linting, testing):

.. code-block:: bash

   pip install -ve '.[validation]'

Install documentation build dependencies:

.. code-block:: bash

   pip install -ve '.[docs]'
