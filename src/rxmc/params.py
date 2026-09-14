"""The sampled scalar.

A :class:`Parameter` is a declaration: a name, optional bounds, an optional
marginal prior, and display metadata.  Its identity *is* the object: passing
the same ``Parameter`` to two places, anywhere in a problem, shares one sampled
value between them.  Two distinct objects with the same name are two
parameters, and :class:`~rxmc.problem.Problem` rejects the duplicate name.

Prior rules, enforced once at compile:

* ``prior`` given: the marginal, truncated to ``bounds``;
* finite ``bounds`` and no ``prior``: uniform on the bounds;
* neither: the parameter must be covered by a joint prior passed to
  ``Problem``, otherwise compile fails naming it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import inf

__all__ = ["Parameter"]


@dataclass(eq=False, frozen=True)
class Parameter:
    """A single scalar model parameter.

    Parameters
    ----------
    name : str
        Non-empty; used for chain columns, plots and error messages.
    bounds : (float, float), optional
        Support of the parameter, ``(-inf, inf)`` by default.  ``lo < hi``.
    prior : object, optional
        A frozen univariate distribution exposing ``logpdf``, ``cdf``, ``ppf``
        and ``rvs`` (any ``scipy.stats`` frozen distribution).  Left ``None``
        when the parameter is covered by a joint prior.
    unit : str, optional
        Physical unit string for display.
    latex : str, optional
        LaTeX label for plots; ``name`` when omitted.
    """

    name: str
    bounds: tuple[float, float] = (-inf, inf)
    prior: object | None = field(default=None, repr=False)
    unit: str = ""
    latex: str | None = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"name must be a non-empty string, got {self.name!r}")
        try:
            lo, hi = (float(b) for b in self.bounds)
        except (TypeError, ValueError):
            raise ValueError(
                f"bounds must be (lower, upper), got {self.bounds!r}"
            ) from None
        if not lo < hi:
            raise ValueError(f"bounds must satisfy lower < upper, got {(lo, hi)!r}")
        object.__setattr__(self, "bounds", (lo, hi))

    @property
    def label(self) -> str:
        """The LaTeX label, falling back to the name."""
        return self.latex if self.latex is not None else self.name

    def __repr__(self):
        prior = ", prior=set" if self.prior is not None else ""
        return f"Parameter({self.name!r}, bounds={self.bounds!r}{prior})"
