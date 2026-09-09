"""
Low-level transforms shared across rxmc.

A :class:`Transform` is a numpy-style callable ``fn(a, *values) -> array`` with an
optional tuple of :class:`~rxmc.params.Parameter` s (``values`` are their sampled
values) and optional analytic ``derivative``/``inverse``.  The same type serves
three roles:

* the comparison-space transform of an :class:`~rxmc.observation.Observation`
  (e.g. ``transform=log`` to compare in log space; parameter-free),
* a parametric model-side transform on a
  :class:`~rxmc.physical_model.PhysicalModel` (e.g. :func:`scale` for a latent
  normalisation, :func:`per_observation_scaling` for one per dataset),
* the coordinate transform of a covariance :class:`~rxmc.covariance.Term`
  (e.g. angle to momentum transfer).

Anything callable is accepted wherever a ``Transform`` is expected and is wrapped
as a parameter-free transform.  Transforms compose with ``|``: ``(f | g)(a)`` is
``g(f(a))``, parameters concatenated in that order.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from .params import Parameter


class Transform:
    """A numpy-style transform with optional parameters.

    Parameters
    ----------
    fn : callable
        ``fn(a, *values) -> np.ndarray``; when ``contextual`` is ``True``,
        ``fn(context, a, *values)`` where ``context`` is whatever the owner
        passes (the :class:`~rxmc.observation.Observation` for model transforms).
    params : sequence of Parameter, optional
        Parameters whose sampled values are passed as ``*values``.
    contextual : bool, optional
        Whether ``fn`` takes the owner's context as its first argument.
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
        contextual: bool = False,
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
        self.contextual = bool(contextual)
        self.derivative_fn = derivative
        self._inverse = inverse
        self._inverse_factory = None
        self.name = name or getattr(fn, "__name__", "transform")

    @property
    def n_params(self) -> int:
        return len(self.params)

    @property
    def is_identity(self) -> bool:
        return self is identity

    @property
    def inverse(self) -> "Transform | None":
        """The inverse transform, or ``None`` if unknown."""
        if self._inverse is None and self._inverse_factory is not None:
            self._inverse = self._inverse_factory()
        if self._inverse is None:
            return None
        return as_transform(self._inverse)

    def __call__(self, a, *values, context=None):
        if len(values) != self.n_params:
            raise ValueError(
                f"transform {self.name!r} expects {self.n_params} value(s), "
                f"got {len(values)}"
            )
        a = np.asarray(a, dtype=float)
        if self.contextual:
            return np.asarray(self.fn(context, a, *values), dtype=float)
        return np.asarray(self.fn(a, *values), dtype=float)

    def derivative(self, a, *values, context=None):
        """Elementwise derivative :math:`\\partial fn/\\partial a` at ``a``.

        Falls back to a central finite difference when no analytic derivative
        was supplied.
        """
        a = np.asarray(a, dtype=float)
        if self.derivative_fn is not None:
            if self.contextual:
                return np.asarray(self.derivative_fn(context, a, *values), dtype=float)
            return np.asarray(self.derivative_fn(a, *values), dtype=float)
        h = 1e-6 * np.maximum(np.abs(a), 1.0)
        fp = self(a + h, *values, context=context)
        fm = self(a - h, *values, context=context)
        return (fp - fm) / (2 * h)

    def __or__(self, other) -> "Transform":
        """``(f | g)(a) = g(f(a))`` with parameters ``f.params + g.params``."""
        f, g = self, as_transform(other)
        nf = f.n_params
        contextual = f.contextual or g.contextual

        def fn(*args):
            if contextual:
                context, a, *values = args
            else:
                context, (a, *values) = None, args
            b = f(a, *values[:nf], context=context)
            return g(b, *values[nf:], context=context)

        def derivative(*args):
            if contextual:
                context, a, *values = args
            else:
                context, (a, *values) = None, args
            b = f(a, *values[:nf], context=context)
            return g.derivative(b, *values[nf:], context=context) * f.derivative(
                a, *values[:nf], context=context
            )

        out = Transform(
            fn,
            f.params + g.params,
            contextual=contextual,
            derivative=derivative,
            name=f"{f.name}|{g.name}",
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
    r"""A latent multiplicative normalisation :math:`\rho\, y`.

    The Kennedy & O'Hagan forward-model scale: it changes the *mean*, not the
    covariance, so it belongs on the model
    (``PhysicalModel(params, transform=scale())``).

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
            name,
            float,
            unit="dimensionless",
            latex_name=r"\log{\rho}" if log else r"\rho",
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


def _root(observation):
    """The identity key of an observation (its root; itself for other objects)."""
    return getattr(observation, "identity", observation)


def per_observation_scaling(
    observations, parameters=None, log: bool = True, prefix: str | None = None
) -> Transform:
    r"""One latent normalisation :math:`\rho_i` per dataset, routed by identity.

    Contextual: when the owning model is evaluated on observation :math:`i`
    (matched by identity of ``obs.identity``, so masked views made with
    :meth:`~rxmc.observation.Observation.masked` route to their root's scale),
    the prediction is scaled by :math:`\rho_i`.  Parameters are ordered as
    ``observations``.

    Parameters
    ----------
    observations : sequence of Observation
        The datasets, each assigned one scale parameter.
    parameters : sequence of Parameter, optional
        One per observation.  Defaults to ``{prefix}_{i}``.
    log : bool, optional
        Sample :math:`\log\rho_i` (default) or :math:`\rho_i`.
    prefix : str, optional
        Default-parameter name prefix; ``log_rho``/``rho`` by ``log``.
    """
    observations = list(observations)
    index = {id(_root(o)): i for i, o in enumerate(observations)}
    if len(index) != len(observations):
        raise ValueError("observations must be distinct objects (routing by identity)")
    if prefix is None:
        prefix = "log_rho" if log else "rho"
    if parameters is None:
        parameters = [
            Parameter(
                f"{prefix}_{i}",
                float,
                unit="dimensionless",
                latex_name=(rf"\log{{\rho_{{{i}}}}}" if log else rf"\rho_{{{i}}}"),
            )
            for i in range(len(observations))
        ]
    parameters = tuple(parameters)
    if len(parameters) != len(observations):
        raise ValueError("need exactly one parameter per observation")

    def _value(context, values):
        i = index.get(id(_root(context)))
        if i is None:
            raise KeyError(
                "observation was not registered with this per_observation_scaling"
            )
        v = values[i]
        return np.exp(v) if log else v

    def fn(context, a, *values):
        return _value(context, values) * a

    def derivative(context, a, *values):
        return np.full_like(a, _value(context, values))

    t = Transform(
        fn,
        parameters,
        contextual=True,
        derivative=derivative,
        name="per_observation_scaling",
    )
    t.observations = observations  # keep ids alive
    return t
