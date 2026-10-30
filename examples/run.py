"""Execute the example notebooks in place: the committed outputs, full runs.

::

    python examples/run.py                    # every notebook, one at a time
    python examples/run.py gp_discrepancy local_optical_model_calibration

The documentation shows the notebooks as committed, so a changed notebook is
rerun with this script before it is committed.  The notebooks run one after
another, so their wall times (printed, and listed in design.md's notebook
table) do not overlap.  CI runs them quick instead (``RXMC_EXAMPLES_QUICK=1``,
see ``budget.py``) and never writes their outputs back.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

HERE = Path(__file__).resolve().parent


def run(name: str, timeout: int = 7200) -> float:
    """Execute ``examples/<name>.ipynb`` in place; its wall time in seconds."""
    path = HERE / f"{name}.ipynb"
    nb = nbformat.read(path, as_version=4)
    ep = ExecutePreprocessor(timeout=timeout, record_timing=False)
    t0 = time.perf_counter()
    ep.preprocess(nb, {"metadata": {"path": str(HERE)}})
    wall = time.perf_counter() - t0
    nbformat.write(nb, path)
    return wall


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("names", nargs="*", help="notebooks without .ipynb (default: all)")
    args = ap.parse_args(argv)
    if os.environ.get("RXMC_EXAMPLES_QUICK") == "1":
        sys.exit("RXMC_EXAMPLES_QUICK=1 is set: committed notebooks are full runs")
    names = args.names or sorted(p.stem for p in HERE.glob("*.ipynb"))
    for name in names:
        print(f"{name}: {run(name):.0f} s", flush=True)


if __name__ == "__main__":
    main()
