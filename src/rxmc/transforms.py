"""
Low-level transforms shared across rxmc.

A :class:`Transform` is a numpy-style callable ``fn(a, *values) -> array`` with an
optional tuple of :class:`~rxmc.params.Parameter` s (``values`` are their sampled
values) and optional analytic ``derivative``/``inverse``.  The same type serves
three roles:

* the comparison space of a :class:`~rxmc.constraint.Comparison`
  (e.g. ``space=log``; parameter-free, so the delta-method errors and the
  log-Jacobian are constants),
* a mean transform composed onto a :class:`~rxmc.model.Model` with ``|``
  (e.g. :func:`scale` for a latent normalisation),
* the coordinate transform of a covariance :class:`~rxmc.terms.Term`
  (e.g. angle to momentum transfer).

Anything callable is accepted wherever a ``Transform`` is expected and is wrapped
as a parameter-free transform.  Transforms compose with ``|``: ``(f | g)(a)`` is
``g(f(a))``, parameters concatenated in that order.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from .params import Parameter

__all__ = ["Transform", "as_transform", "identity", "log", "exp", "scale"]


class Transform:
    """A numpy-style transform with optional parameters.

    Parameters
    ----------
    fn : callable
        ``fn(a, *values) -> np.ndarray``.
    params : sequence of Parameter, optional
        Parameters whose sampled values are passed as ``*values``.
    derivative : callable, optional
        ``derivative(a, *values) -> np.ndarray``, :math:`\\partial fn/\\partial a`
        elementwise.  Used for delta-method error propagation and Jacobians.
    inverse : Transform or callable, optional
        The inverse transform (parameter-free transforms only).
    name : str, optional
        Human-readable name.
    """

    def __init__(
        self,
        fn: Callable,
        params: Sequence[Parameter] = (),
        *,
        derivative: Callable | None = None,
        inverse=None,
        name: str | None = None,
    ):
        if not callable(fn):
            raise TypeError("fn must be callable")
        self.fn = fn
        self.params = tuple(params)
        for p in self.params:
            if not isinstance(p, Parameter):
                raise TypeError(f"params must be Parameter objects, got {p!r}")
        self.derivative_fn = derivative
        self._inverse = inverse
        self._inverse_factory = None
        self.name = name or getattr(fn, "__name__", "transform")

    @property
    def n_params(self) -> int:
        return len(self.params)

    @property
    def is_identity(self) -> bool:
        """Whether this is the module-level :data:`identity` singleton.

        An object-identity test, not a semantic one: ``Transform(lambda a: a)``
        or ``log | exp`` are not recognised and take no identity fast path.
        """
        return self is identity

    @property
    def inverse(self) -> "Transform | None":
        """The inverse transform, or ``None`` if unknown.

        Only meaningful for parameter-free transforms (a parametric inverse
        would need the same values, which are not carried along).
        """
        if self._inverse is None and self._inverse_factory is not None:
            self._inverse = self._inverse_factory()
        if self._inverse is None:
            return None
        return as_transform(self._inverse)

    def __call__(self, a, *values):
        if len(values) != self.n_params:
            raise ValueError(
                f"transform {self.name!r} expects {self.n_params} value(s), "
                f"got {len(values)}"
            )
        a = np.asarray(a, dtype=float)
        return np.asarray(self.fn(a, *values), dtype=float)

    def derivative(self, a, *values):
        """Elementwise derivative :math:`\\partial fn/\\partial a` at ``a``.

        Falls back to a central finite difference when no analytic derivative
        was supplied.
        """
        a = np.asarray(a, dtype=float)
        if self.derivative_fn is not None:
            return np.asarray(self.derivative_fn(a, *values), dtype=float)
        # a step relative to a, so data far below 1 (b/sr) keep their precision
        h = 1e-6 * np.where(a == 0.0, 1.0, np.abs(a))
        return (self(a + h, *values) - self(a - h, *values)) / (2 * h)

    def __or__(self, other) -> "Transform":
        """``(f | g)(a) = g(f(a))`` with parameters ``f.params + g.params``."""
        f, g = self, as_transform(other)
        nf = f.n_params

        def fn(a, *values):
            return g(f(a, *values[:nf]), *values[nf:])

        def derivative(a, *values):
            b = f(a, *values[:nf])
            return g.derivative(b, *values[nf:]) * f.derivative(a, *values[:nf])

        out = Transform(
            fn, f.params + g.params, derivative=derivative, name=f"{f.name}|{g.name}"
        )
        if not (f.params or g.params):
            # lazy: composing eagerly would recurse for mutually inverse pairs
            def _inverse():
                if f.inverse is None or g.inverse is None:
                    return None
                return g.inverse | f.inverse

            out._inverse_factory = _inverse
        return out

    def __repr__(self):
        names = ", ".join(p.name for p in self.params)
        return f"Transform({self.name}{', params=(' + names + ')' if names else ''})"


def as_transform(t) -> Transform:
    """Coerce ``None`` (identity), a callable, or a :class:`Transform`."""
    if t is None:
        return identity
    if isinstance(t, Transform):
        return t
    if callable(t):
        return Transform(t)
    raise TypeError(f"expected a Transform or callable, got {type(t).__name__}")


# ----------------------------------------------------------------------------
# Parameter-free transforms
# ----------------------------------------------------------------------------


def _identity(a):
    return a


def _safe_log(a):
    """``log(a)`` for an ndarray ``a``, ``-inf`` where ``a <= 0`` (no warnings)."""
    out = np.full(np.shape(a), -np.inf, dtype=float)
    pos = a > 0
    out[pos] = np.log(a[pos])
    return out


def _reciprocal(a):
    with np.errstate(divide="ignore"):
        return 1.0 / a


identity = Transform(_identity, derivative=np.ones_like, name="identity")
identity._inverse = identity

exp = Transform(np.exp, derivative=np.exp, name="exp")
log = Transform(_safe_log, derivative=_reciprocal, inverse=exp, name="log")
"""Natural log; ``-inf`` (no warnings) where the argument is not positive."""
exp._inverse = log


# ----------------------------------------------------------------------------
# Parametric transforms
# ----------------------------------------------------------------------------


def scale(parameter: Parameter | None = None, log: bool = True, name=None) -> Transform:
    r"""A latent multiplicative normalisation, ``rho * a``.

    The Kennedy & O'Hagan forward-model scale: it changes the *mean*, not the
    covariance, so it is composed onto the model (``model | scale(rho)``) and
    the prediction, not the data, is scaled.

    Parameters
    ----------
    parameter : Parameter, optional
        The scale parameter.  Defaults to ``log_rho`` (or ``rho`` when
        ``log=False``).
    log : bool, optional
        If ``True`` (default) the sampled value is :math:`\log\rho` and the
        prediction is scaled by ``exp(value)``; otherwise by ``value``.
    name : str, optional
        Name for the default parameter.
    """
    if parameter is None:
        name = name or ("log_rho" if log else "rho")
        parameter = Parameter(
            name, unit="dimensionless", latex=r"\log{\rho}" if log else r"\rho"
        )
    if log:
        return Transform(
            lambda a, v: np.exp(v) * a,
            (parameter,),
            derivative=lambda a, v: np.full_like(a, np.exp(v)),
            name="scale",
        )
    return Transform(
        lambda a, v: v * a,
        (parameter,),
        derivative=lambda a, v: np.full_like(a, v),
        name="scale",
    )
