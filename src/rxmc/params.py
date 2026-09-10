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
        ``(-np.inf, np.inf)``.  Stored as a tuple of floats.

    Notes
    -----
    Equality and hashing are by *value* (all five fields), so equal
    parameters are interchangeable as dict keys and set members.  Sharing one
    sampled value between covariance terms is by object *identity* (see
    :mod:`rxmc.covariance`); two equal-but-distinct parameters are two
    parameters.
    """

    def __init__(
        self, name, dtype=float, unit="", latex_name=None, bounds=(-np.inf, np.inf)
    ):
        self.name = name
        self.dtype = dtype
        self.unit = unit
        bounds = tuple(float(b) for b in bounds)
        if len(bounds) != 2:
            raise ValueError(f"bounds must be (lower, upper), got {bounds!r}")
        self.bounds = bounds
        self.latex_name = latex_name if latex_name else name

    def _key(self):
        return (self.name, self.dtype, self.unit, self.latex_name, self.bounds)

    def __eq__(self, other):
        if not isinstance(other, Parameter):
            return False
        return self._key() == other._key()

    def __hash__(self):
        return hash(self._key())

    def __repr__(self):
        return (
            f"Parameter({self.name!r}, dtype={self.dtype.__name__}, "
            f"unit={self.unit!r}, latex_name={self.latex_name!r}, "
            f"bounds={self.bounds!r})"
        )
