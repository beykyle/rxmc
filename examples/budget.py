"""Sampler budgets for the example notebooks.

The committed notebooks are full runs, made with ``python examples/run.py``,
and the documentation shows them as committed.  CI only checks that every
cell still runs: it sets ``RXMC_EXAMPLES_QUICK=1``, which cuts every budget
wrapped in :func:`budget` tenfold and caps NUTS's trees
(:data:`QUICK_NUTS`), so its numbers mean nothing.

``import budget`` works because a notebook runs with ``examples/`` as its
working directory, under Jupyter and under ``pytest --nbmake`` alike.
"""

from __future__ import annotations

import os

__all__ = ["QUICK", "BANNER", "QUICK_NUTS", "budget"]

QUICK = os.environ.get("RXMC_EXAMPLES_QUICK") == "1"
#: NUTS's extra keyword arguments in a quick run, empty otherwise: trees of at
#: most 2**5 leapfrog steps, since a step size adapted in a few dozen steps is
#: small and its trees grow deep (on a reaction model, five times the time)
QUICK_NUTS = dict(max_num_doublings=5) if QUICK else {}
#: printed at import in a quick run; test_notebooks_index refuses a committed
#: notebook whose outputs contain it
BANNER = "quick run: sampler budgets cut tenfold, results not meaningful"
if QUICK:
    print(BANNER)


def budget(n: int, floor: int = 50) -> int:
    """``n``, or ``n // 10`` (at least ``floor``) in a quick run."""
    return max(n // 10, min(floor, n)) if QUICK else n
