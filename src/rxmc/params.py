"""
Parameter definitions.

The :class:`Parameter` class describes a single scalar model parameter —
its name, data type, physical unit, LaTeX label, and optional bounds.
"""

import numpy as np


class Parameter:
    """A single scalar model parameter.

    Parameters
    ----------
    name : str
        Human-readable name of the parameter.
    dtype : type, optional
        Data type of the parameter value.  Defaults to ``float``.
    unit : str, optional
        Physical unit string (e.g. ``"MeV"``).  Defaults to ``""``.
    latex_name : str, optional
        LaTeX representation used in plots and documentation.  Defaults to
        ``name`` when not supplied.
    bounds : tuple of float, optional
        ``(lower, upper)`` bounds for the parameter.  Defaults to
        ``(-np.inf, np.inf)``.
    """

    def __init__(
        self, name, dtype=float, unit="", latex_name=None, bounds=(-np.inf, np.inf)
    ):
        self.name = name
        self.dtype = dtype
        self.unit = unit
        self.bounds = bounds
        self.latex_name = latex_name if latex_name else name

    def __eq__(self, other):
        if not isinstance(other, Parameter):
            return False
        return (
            self.name == other.name
            and self.dtype == other.dtype
            and self.unit == other.unit
            and self.latex_name == other.latex_name
            and self.bounds == other.bounds
        )
