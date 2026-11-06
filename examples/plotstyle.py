"""Shared plot styling for the example notebooks.

The notebooks call :func:`use` once, near their imports, and then draw with
the ordinary matplotlib API.  :func:`band` and :func:`corner_kwargs` are the
two things every notebook was re-implementing by hand.

``import plotstyle`` works because a notebook runs with ``examples/`` as its
working directory, under Jupyter and under ``pytest --nbmake`` alike.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

__all__ = ["COLOURS", "HATCHES", "use", "band", "corner_kwargs", "label_at"]

# Wong's colourblind-safe qualitative palette (Nature Methods 8, 441 (2011)).
COLOURS = [
    "#0072b2",  # blue
    "#d55e00",  # vermillion
    "#009e73",  # bluish green
    "#cc79a7",  # reddish purple
    "#e69f00",  # orange
    "#56b4e9",  # sky blue
    "#525252",  # grey
]

# For overlapping bands that must stay apart in greyscale.
HATCHES = ["///", "\\\\\\", "...", "xxx", "+++"]


def use() -> None:
    """Apply the example notebooks' rcParams."""
    mpl.rcParams.update(
        {
            "figure.figsize": (6.4, 4.0),
            "figure.dpi": 110,
            "savefig.dpi": 110,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.prop_cycle": mpl.cycler(color=COLOURS),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "lines.linewidth": 1.8,
            "lines.markersize": 4,
            "errorbar.capsize": 0,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    )


def band(ax, x, lo, hi, *, color=None, hatch=None, label=None, alpha=None, **kwargs):
    """Fill between ``lo`` and ``hi``: a predictive band.

    Pass ``hatch`` (see :data:`HATCHES`) when several bands overlap and the
    difference must survive in greyscale; the fill is then lighter and the
    edge carries the colour.
    """
    hatched = hatch is not None
    if alpha is None:
        alpha = 0.18 if hatched else 0.28
    return ax.fill_between(
        np.asarray(x, dtype=float),
        np.asarray(lo, dtype=float),
        np.asarray(hi, dtype=float),
        color=color,
        alpha=alpha,
        hatch=hatch,
        edgecolor=color if hatched else None,
        linewidth=0.8 if hatched else 0.0,
        label=label,
        **kwargs,
    )


def corner_kwargs(**overrides) -> dict:
    """Defaults for ``corner.corner``; keyword arguments override them."""
    kwargs = {
        "levels": (0.39, 0.68, 0.95),
        "smooth": 0.8,
        "bins": 32,
        "color": COLOURS[0],
        "plot_datapoints": False,
        "fill_contours": True,
        "show_titles": True,
        "title_fmt": ".2f",
        "title_kwargs": {"fontsize": 9},
        "label_kwargs": {"fontsize": 10},
        "truth_color": COLOURS[6],
    }
    kwargs.update(overrides)
    return kwargs


def label_at(ax, x, y, text, *, color=None, **kwargs):
    """A small text label on a curve, for datasets offset in ``y``."""
    return ax.text(
        x,
        y,
        text,
        color=color,
        fontsize=8,
        va="bottom",
        ha="left",
        **kwargs,
    )


def _demo() -> None:  # pragma: no cover - a visual check, not run by the tests
    use()
    x = np.linspace(0, 1, 50)
    fig, ax = plt.subplots()
    for i, (colour, hatch) in enumerate(zip(COLOURS, HATCHES)):
        band(ax, x, i + x, i + 1.5 * x, color=colour, hatch=hatch, label=f"band {i}")
    ax.legend()
    plt.show()
