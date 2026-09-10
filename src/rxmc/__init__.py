"""rxmc: Bayesian calibration of reaction models with composable error models.

The 1.0 rewrite is in progress on this branch; see ``docs/groundup_design.md``
for the design and ``docs/recipes.md`` for the supported use cases.
"""

try:
    from .__version__ import __version__ as __version__
except ImportError:  # pragma: no cover - source checkout without a build
    __version__ = "0+unknown"

__all__ = ["__version__"]
