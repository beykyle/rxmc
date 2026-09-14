"""
Models and predictors.

A :class:`Model` is a spec of "observables from parameters": a callable
``fn(x, *values)`` in physical space plus the :class:`~rxmc.params.Parameter` s
it consumes.  A :class:`Predictor` is that model **bound to a grid**: ``bind``
is where anything expensive or grid-dependent is built (a reaction model
builds its solver workspace there and overrides only ``bind``).

Models compose, and the composition happens on the bound predictors so a
reaction model and a plain function add without special cases:

* ``model | transform`` applies a mean transform after the prediction
  (:func:`~rxmc.transforms.scale` for a latent normalisation);
* ``model + other`` is an additive, ``x``-dependent correction (a sampled
  mean discrepancy);
* ``model * other`` is a multiplicative one (an additive discrepancy in log
  space; ``scale`` is its constant case).

Parameters are concatenated left to right.  A parameter object appearing on
both sides is one sampled value, as everywhere.
"""

from __future__ import annotations

import operator
from typing import Callable, Sequence

import numpy as np
from numpy.polynomial import polynomial as P

from .params import Parameter
from .transforms import as_transform

__all__ = ["Model", "Predictor", "polynomial"]


def _check_params(params) -> tuple:
    params = tuple(params)
    for p in params:
        if not isinstance(p, Parameter):
            raise TypeError(f"params must be Parameter objects, got {p!r}")
    return params


class Predictor:
    """A model bound to a grid: ``predictor(*values) -> y`` in physical space.

    Parameters
    ----------
    params : sequence of Parameter
        The values ``__call__`` expects, in order.
    x : array_like
        The grid the prediction is made on.
    fn : callable
        ``fn(*values) -> np.ndarray`` on that grid.
    meta : mapping, optional
        The dataset metadata the model was bound with; a term evaluated at the
        predictor's grid (:func:`~rxmc.predictive.grid_draws`) reads
        it through ``c.meta(key)``.
    """

    def __init__(self, params: Sequence[Parameter], x, fn: Callable, meta=None):
        self.params = _check_params(params)
        self.x = np.asarray(x)
        self._fn = fn
        self.meta = meta

    def __call__(self, *values) -> np.ndarray:
        if len(values) != len(self.params):
            raise ValueError(
                f"predictor expects {len(self.params)} value(s), got {len(values)}"
            )
        return np.asarray(self._fn(*values), dtype=float)

    def __repr__(self):
        names = ", ".join(p.name for p in self.params)
        return f"Predictor(params=({names}), n={len(self.x)})"


class Model:
    """A parametric model of an observable, in physical space.

    Parameters
    ----------
    fn : callable, optional
        ``fn(x, *values) -> np.ndarray`` on a grid ``x``.  Subclasses that
        build their prediction in :meth:`bind` may omit it.
    params : sequence of Parameter, optional
        The parameters ``fn`` consumes, in order.
    """

    def __init__(self, fn: Callable | None = None, params: Sequence[Parameter] = ()):
        if fn is not None and not callable(fn):
            raise TypeError("fn must be callable")
        self.fn = fn
        self.params = _check_params(params)

    def bind(self, x, meta=None) -> Predictor:
        """The model on the grid ``x``; ``meta`` carries the dataset's kinematics.

        The generic model closes over ``x``.  Reaction models override this to
        build their solver on ``x`` from ``meta``.
        """
        if self.fn is None:
            raise TypeError(f"{type(self).__name__} must override bind()")
        fn = self.fn
        return Predictor(self.params, x, lambda *values: fn(x, *values), meta)

    def __or__(self, transform) -> "Model":
        return _Transformed(self, as_transform(transform))

    def __add__(self, other) -> "Model":
        return _Combined(self, other, operator.add, "+")

    def __mul__(self, other) -> "Model":
        return _Combined(self, other, operator.mul, "*")

    def __repr__(self):
        names = ", ".join(p.name for p in self.params)
        return f"{type(self).__name__}(params=({names}))"


class _Transformed(Model):
    """``inner | transform``: the transform's parameters follow the model's."""

    def __init__(self, inner: Model, transform):
        self.inner, self.transform = inner, transform
        super().__init__(None, inner.params + transform.params)

    def bind(self, x, meta=None) -> Predictor:
        pred, t, n = self.inner.bind(x, meta), self.transform, len(self.inner.params)
        return Predictor(self.params, x, lambda *v: t(pred(*v[:n]), *v[n:]), meta)


class _Combined(Model):
    """``left op right`` on the bound predictors; parameters left then right."""

    def __init__(self, left: Model, right: Model, op, symbol: str):
        if not isinstance(right, Model):
            raise TypeError(f"can only combine a Model with a Model, got {right!r}")
        self.left, self.right, self.op, self.symbol = left, right, op, symbol
        super().__init__(None, left.params + right.params)

    def bind(self, x, meta=None) -> Predictor:
        lp, rp = self.left.bind(x, meta), self.right.bind(x, meta)
        n, op = len(self.left.params), self.op
        return Predictor(self.params, x, lambda *v: op(lp(*v[:n]), rp(*v[n:])), meta)


def polynomial(order: int) -> Model:
    """A polynomial of the given order with parameters ``a0`` to ``a<order>``.

    ``y = a_0 + a_1 x + a_2 x^2 + ...``; the parameters carry no prior.
    """
    params = [Parameter(f"a{i}", latex=f"a_{i}") for i in range(order + 1)]
    return Model(lambda x, *a: P.polyval(np.asarray(x, dtype=float), a), params)
