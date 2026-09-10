"""
Covariance terms: the additive pieces of a constraint's covariance.

A constraint owns one multivariate distribution over the stacked residual of
its comparisons; its covariance is a sum of :class:`Term` s.  There is exactly
**one** term type.  A ``Term`` is a numpy-style callable ``fn(c, *values) ->
array`` of a :class:`TermContext` ``c`` (the term's view of ``x``, ``y`` and the
prediction ``ym`` on its support, with ``x`` passed through an optional
coordinate :class:`~rxmc.transforms.Transform`) and its parameter values, plus a
``kind`` that says how the returned array enters the covariance:

``"diag"``
    ``fn`` returns a standard-deviation vector ``v``; ``Sigma_ii += v_i**2``.
``"mode"``
    ``fn`` returns a vector ``v``; ``Sigma += outer(v, v)`` (one correlated mode).
``"matrix"``
    ``fn`` returns a full symmetric block ``M``; ``Sigma_block += M``.

A term is a stateless declaration.  Where it applies is its ``on``: a
comparison, a dataset, a sequence of them, or ``None`` for the whole
constraint; :class:`~rxmc.problem.Problem` resolves that to rows when it
compiles.  The same term may be placed in several constraints.

Two mechanisms are expressed here (see ``docs/groundup_design.md``):

* **Correlating comparisons** — a ``mode`` or ``matrix`` term whose ``on``
  spans several comparisons writes off-diagonal blocks, coupling the data.
* **Sharing a parameter** — terms declare the :class:`~rxmc.params.Parameter`
  objects they consume *by identity*: pass the same object to two terms and
  they share one sampled value.

The factory helpers (:func:`statistical`, :func:`offset`, :func:`normalization`,
:func:`noise`, :func:`noise_fraction`, :func:`model_error`, :func:`systematic`,
:func:`kernel`) are one-line conveniences that build the common terms; anything
they cannot express is a direct ``Term(fn, params, kind=...)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .params import Parameter
from .transforms import as_transform, identity

__all__ = [
    "KINDS",
    "TermContext",
    "Term",
    "as_2d",
    "ones",
    "ym",
    "averaging",
    "x_basis",
    "exp_growth",
    "constant_amplitude",
    "exp_growth_amplitude",
    "statistical",
    "offset",
    "normalization",
    "noise",
    "noise_fraction",
    "model_error",
    "systematic",
    "kernel",
]

KINDS = ("diag", "mode", "matrix")


def as_2d(X) -> np.ndarray:
    """Promote a 1-D input grid to a single-column 2-D array (sklearn kernels)."""
    X = np.asarray(X, dtype=float)
    return X[:, None] if X.ndim == 1 else X


@dataclass(eq=False, frozen=True)
class TermContext:
    """A term's view of the stack on its own support.

    ``x`` are the coordinates on the support, already passed through the term's
    ``coords`` transform (so ``x`` may be 2-D); ``y`` and ``ym`` are the data and
    the model prediction on the support in comparison space.  ``ym`` is ``None``
    while a *constant* term is evaluated before any prediction exists, so a
    mis-declared constant term fails loudly.  ``len(c)`` is the number of
    points; :meth:`meta` gives per-point dataset metadata (``c.meta("Elab")``).
    """

    x: np.ndarray
    y: np.ndarray
    ym: np.ndarray | None = None
    _meta: Mapping[str, np.ndarray] | None = None

    def __len__(self) -> int:
        return len(self.y)

    def meta(self, key: str) -> np.ndarray:
        """The owning dataset's ``meta[key]``, one value per point of the support."""
        if self._meta is None or key not in self._meta:
            raise KeyError(
                f"no per-point metadata {key!r} on this term's support; put it in "
                "Dataset.meta"
            )
        return self._meta[key]


@dataclass(eq=False, frozen=True)
class Term:
    """One additive contribution to a constraint's covariance.

    Parameters
    ----------
    fn : callable or array_like
        ``fn(c, *values) -> np.ndarray`` with ``c`` a :class:`TermContext` and
        ``values`` the sampled values of ``params`` (in order).  A plain array is
        a fixed contribution (``constant=True`` implied): a standard-deviation
        vector for ``kind="diag"``, a mode vector for ``"mode"``, or a symmetric
        block for ``"matrix"``.
    params : sequence of Parameter, optional
        Parameters consumed by ``fn``, matched *by identity* across terms.
    kind : {"diag", "mode", "matrix"}
        How the returned array enters the covariance (see module docstring).
    on : Comparison, Dataset, sequence of them, or None
        Where the term applies; ``None`` means the whole constraint.  Resolved
        by :class:`~rxmc.problem.Problem`.
    coords : Transform or callable, optional
        Coordinate transform applied to ``x`` before ``fn`` sees it (e.g. angle
        to momentum transfer).  Its parameters, if any, are appended to
        :attr:`params`.
    constant : bool, optional
        Declare that a *callable* ``fn`` reads neither ``c.ym`` nor any
        parameter, so the contribution can be evaluated once at compile.
        ``c.x`` and ``c.y`` may be read freely.  A constant term is evaluated
        with ``c.ym is None``, so a mis-declared term fails loudly.  Implied
        for an array ``fn``; an error together with ``params``.
    """

    fn: Any
    params: tuple[Parameter, ...] = ()
    kind: str = "matrix"
    on: Any = None
    coords: Any = identity
    constant: bool = False

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")
        coords = as_transform(self.coords)
        fn_params = tuple(self.params)
        for p in fn_params + coords.params:
            if not isinstance(p, Parameter):
                raise TypeError(f"params must be Parameter objects, got {p!r}")
        object.__setattr__(self, "coords", coords)
        object.__setattr__(self, "params", fn_params + coords.params)
        object.__setattr__(self, "_n_fn_params", len(fn_params))
        if callable(self.fn):
            if self.constant and self.params:
                raise ValueError(
                    "a constant term cannot have parameters (constant means the "
                    "term reads neither ym nor any parameter)"
                )
            return
        if self.params:
            raise ValueError("an array-valued term cannot have parameters")
        a = np.asarray(self.fn, dtype=float)
        if self.kind == "matrix":
            if a.ndim != 2 or a.shape[0] != a.shape[1]:
                raise ValueError(f"matrix term expects a square block, got {a.shape}")
            if not np.allclose(a, a.T):
                raise ValueError("matrix term must be symmetric")
        elif a.ndim != 1:
            raise ValueError(f"{self.kind} term expects a vector, got shape {a.shape}")
        object.__setattr__(self, "fn", a)
        object.__setattr__(self, "constant", True)

    # -- structure ----------------------------------------------------------

    @property
    def is_constant(self) -> bool:
        """Whether the term can be evaluated once, before any prediction."""
        return bool(self.constant)

    @property
    def couples_offdiagonal(self) -> bool:
        """Whether the term can write off-diagonal entries (``kind != "diag"``)."""
        return self.kind != "diag"

    def expected_shape(self, n: int) -> tuple:
        return (n, n) if self.kind == "matrix" else (n,)

    # -- evaluation -----------------------------------------------------------

    def context(self, x, y, ym=None, meta=None, *values) -> TermContext:
        """The :class:`TermContext` this term sees on its support at ``values``."""
        self._check_count(values)
        x = np.asarray(x, dtype=float)
        if not self.coords.is_identity:
            x = self.coords(x, *values[self._n_fn_params :])
        return TermContext(
            x=x,
            y=np.asarray(y, dtype=float),
            ym=None if ym is None else np.asarray(ym, dtype=float),
            _meta=meta,
        )

    def value(self, x, y, ym=None, *values, meta=None) -> np.ndarray:
        """The raw array ``fn`` returns (std vector, mode vector, or block)."""
        n = len(y)
        if not callable(self.fn):
            if self.fn.shape != self.expected_shape(n):
                raise ValueError(
                    f"{self.kind} term expects shape {self.expected_shape(n)}, "
                    f"got {self.fn.shape}"
                )
            return self.fn
        c = self.context(x, y, ym, meta, *values)
        v = np.asarray(self.fn(c, *values[: self._n_fn_params]), dtype=float)
        if v.shape != self.expected_shape(n):
            raise ValueError(
                f"{self.kind} term fn returned shape {v.shape}, "
                f"expected {self.expected_shape(n)}"
            )
        return v

    def _check_count(self, values):
        if len(values) != len(self.params):
            raise ValueError(f"expected {len(self.params)} params, got {len(values)}")

    def __repr__(self):
        names = ", ".join(p.name for p in self.params)
        return f"Term(kind={self.kind!r}, params=({names}), on={self.on!r})"


# ----------------------------------------------------------------------------
# Standard bases and amplitudes (numpy-style callables over a TermContext)
# ----------------------------------------------------------------------------


def ones(c: TermContext) -> np.ndarray:
    """Constant unit basis."""
    return np.ones(len(c))


def ym(c: TermContext) -> np.ndarray:
    """The model prediction — the prediction-scaled basis."""
    return c.ym


def averaging(c: TermContext) -> np.ndarray:
    """``0.5 * (y + ym)`` — the averaging model-error basis."""
    return 0.5 * (c.y + c.ym)


_AVERAGING = averaging  # factories take an ``averaging`` flag that shadows the name


def x_basis(scale: float = 1.0):
    """Basis ``x / scale`` (e.g. ``x_basis(np.pi)`` for ``theta/180`` on radians)."""

    def basis(c: TermContext) -> np.ndarray:
        return np.asarray(c.x, dtype=float) / scale

    return basis


def exp_growth(scale: float = 1.0, base=ones):
    """Parametric basis ``base(c) * exp(slope * x / scale)``.

    Takes one basis parameter, ``slope``; use with
    ``noise(..., basis=exp_growth(np.pi), basis_params=(slope,))``.
    """

    def basis(c: TermContext, slope: float) -> np.ndarray:
        return base(c) * np.exp(slope * np.asarray(c.x, dtype=float) / scale)

    return basis


def constant_amplitude(c: TermContext, log_amplitude: float) -> np.ndarray:
    """Kernel amplitude ``exp(log_amplitude)``, constant over the support."""
    return np.full(len(c), np.exp(log_amplitude))


def exp_growth_amplitude(scale: float = 1.0):
    """Kernel amplitude ``exp(log_amplitude) * exp(slope * x / scale)`` (two params)."""

    def amplitude(c: TermContext, log_amplitude: float, slope: float) -> np.ndarray:
        return np.exp(log_amplitude) * np.exp(
            slope * np.asarray(c.x, dtype=float) / scale
        )

    return amplitude


# ----------------------------------------------------------------------------
# Term factory helpers
# ----------------------------------------------------------------------------


def _masked(magnitude, mask=None):
    """A scalar/array magnitude with an optional mask.

    Scalar-like values include 0-d ndarrays (e.g. ``np.array(0.05)`` as stored by
    ``exfor_tools`` distributions), not just Python scalars.  An array magnitude
    must have exactly the support's length; this is checked by :func:`_full`
    when the term is evaluated.
    """
    v = float(magnitude) if np.ndim(magnitude) == 0 else np.asarray(magnitude, float)
    if mask is not None:
        v = v * np.asarray(mask, dtype=float)
    return v


def _full(v, n):
    """``v`` as a length-``n`` vector.

    Scalars are broadcast; arrays must already have shape ``(n,)`` — a length-1
    array is *not* treated as a scalar, so a magnitude or mask of the wrong
    length fails loudly instead of being spread over the support.
    """
    v = np.asarray(v, dtype=float)
    if v.ndim == 0:
        return np.full(n, float(v))
    if v.shape != (n,):
        raise ValueError(
            f"magnitude/basis has shape {v.shape} but the term's support has "
            f"length {n}"
        )
    return v


def _coefficient(parameter, log):
    """Parameter -> multiplicative coefficient ``exp(theta)`` (``log``) or ``theta``."""
    if parameter is None:
        return (), (lambda values: 1.0)
    if log:
        return (parameter,), (lambda values: np.exp(values[0]))
    return (parameter,), (lambda values: values[0])


def _scaled_term(kind, parameter, log, basis, basis_params=(), *, on=None, coords=None):
    """``c * basis(ctx, *basis_values)`` as a term of the given kind."""
    cparams, coef = _coefficient(parameter, log)
    basis_params = tuple(basis_params)
    nc = len(cparams)

    def fn(c, *values):
        b = basis(c, *values[nc:]) if callable(basis) else basis
        return coef(values[:nc]) * _full(b, len(c))

    return Term(fn, cparams + basis_params, kind=kind, on=on, coords=coords)


def statistical(y_err, on=None) -> Term:
    """Always-on, genuinely uncorrelated statistical diagonal ``diag(y_err**2)``."""
    return Term(np.asarray(y_err, dtype=float), kind="diag", on=on)


def offset(parameter=None, magnitude=None, mask=None, log=True, on=None) -> Term:
    """A correlated absolute-offset systematic ``outer(omega, omega)``.

    With ``parameter`` (first, like every nuisance factory) it is a free
    magnitude (``c = exp(theta)`` when ``log``); with ``magnitude=`` it is a
    fixed, data-given rank-one mode.
    """
    if magnitude is None and parameter is None:
        raise ValueError("offset requires a magnitude and/or a parameter")
    mag = _masked(1.0 if magnitude is None else magnitude, mask=mask)

    def basis(c):
        return _full(mag, len(c))

    if parameter is None:
        return Term(basis, kind="mode", on=on, constant=True)
    return _scaled_term("mode", parameter, log, basis, on=on)


def normalization(parameter=None, magnitude=None, mask=None, log=True, on=None) -> Term:
    """A correlated normalisation systematic ``outer(eta * ym, eta * ym)``.

    With ``parameter`` (first, like every nuisance factory) the magnitude eta
    is a free nuisance (``c = exp(theta)`` when ``log``); with ``magnitude=`` it
    is a fixed fractional normalisation uncertainty.  In both cases the mode scales with the model *prediction* ``ym``,
    never with the data: that is what keeps the fit free of Peelle's Pertinent
    Puzzle (recipe 27).
    """
    if magnitude is None and parameter is None:
        raise ValueError("normalization requires a magnitude and/or a parameter")
    mag = _masked(1.0 if magnitude is None else magnitude, mask=mask)

    def basis(c):
        return _full(mag, len(c)) * c.ym

    return _scaled_term("mode", parameter, log, basis, on=on)


def noise(
    parameter, log=True, basis=None, basis_params=(), on=None, coords=None
) -> Term:
    """Unknown statistical noise ``diag((epsilon * basis)**2)``.

    ``basis`` defaults to ones (constant noise); pass any
    ``basis(c, *basis_values)`` — e.g. :func:`exp_growth` with
    ``basis_params=(slope,)`` for noise growing along ``x``.

    This term is **additive**: a constraint with ``statistical=True`` (the
    default) already adds each comparison's reported diagonal, so the assembled
    covariance is ``diag(y_err**2 + epsilon**2)``.  To make the inferred noise
    *replace* the reported statistics pass ``statistical=False`` to the
    constraint.
    """
    return _scaled_term(
        "diag",
        parameter,
        log,
        ones if basis is None else basis,
        basis_params,
        on=on,
        coords=coords,
    )


def noise_fraction(parameter, log=True, on=None) -> Term:
    """Unknown fractional noise ``diag((epsilon * ym)**2)``.

    **Additive** on top of the reported statistical diagonal (see :func:`noise`
    for how to get replace-semantics instead).
    """
    return _scaled_term("diag", parameter, log, ym, on=on)


def model_error(parameter, averaging=True, log=True, on=None) -> Term:
    """Unknown uncorrelated model error ``diag((gamma * z)**2)``.

    ``z = 0.5 * (y + ym)`` when ``averaging`` (stabilises when ``ym`` is near zero),
    else ``z = ym``.
    """
    basis = _AVERAGING if averaging else ym
    return _scaled_term("diag", parameter, log, basis, on=on)


def systematic(
    parameter, basis, log=True, basis_params=(), on=None, coords=None
) -> Term:
    """A correlated mode ``outer(s * u, s * u)`` with a user basis ``u = basis(c, ...)``.

    :func:`offset` and :func:`normalization` are its ``ones``/``ym`` special
    cases; use e.g. ``basis=x_basis(np.pi)`` for a mode growing with angle.
    """
    return _scaled_term(
        "mode", parameter, log, basis, basis_params, on=on, coords=coords
    )


def _kernel_params(kernel, prefix) -> list:
    """One :class:`~rxmc.params.Parameter` per free kernel hyperparameter element."""
    params = []
    for hp in kernel.hyperparameters:
        if hp.fixed:
            continue
        if hp.n_elements == 1:
            params.append(Parameter(f"{prefix}_{hp.name}", latex=hp.name))
        else:
            params.extend(
                Parameter(f"{prefix}_{hp.name}_{i}", latex=f"{hp.name}[{i}]")
                for i in range(hp.n_elements)
            )
    return params


def _n_free_elements(kernel) -> int:
    return sum(hp.n_elements for hp in kernel.hyperparameters if not hp.fixed)


def kernel(
    kernel,
    coords=None,
    amplitude=None,
    amplitude_params=(),
    jitter=1e-10,
    prefix="discrepancy",
    params=None,
    on=None,
) -> Term:
    """A Gaussian-process kernel ``a a^T * K(x, x; theta)`` over the support.

    One :class:`~rxmc.params.Parameter` is derived per *free* kernel
    hyperparameter **element** (sampled in sklearn's log-theta space): an
    anisotropic hyperparameter (``n_elements > 1``) contributes that many
    parameters.  ``amplitude_params`` follow the kernel parameters.

    Parameters
    ----------
    kernel : sklearn-style kernel
        Duck-typed on ``hyperparameters``, ``theta``, ``clone_with_theta`` and
        ``__call__``.
    coords : Transform or callable, optional
        Coordinate transform of ``x`` the kernel is evaluated in (e.g. angle to
        momentum transfer).  Default: ``x`` itself.
    amplitude : callable or array, optional
        ``amplitude(c, *amplitude_values) -> vector a``; the block becomes
        ``outer(a, a) * K``.  Like the kernel, it sees the *transformed*
        coordinate: ``c.x`` is ``coords(x)``.  See :func:`constant_amplitude`,
        :func:`exp_growth_amplitude`.
    amplitude_params : sequence of Parameter, optional
        Parameters consumed by ``amplitude``.
    jitter : float, optional
        Added to the diagonal after scaling, for numerical stability.
    prefix : str, optional
        Name prefix of the derived kernel parameters.  Two kernel terms with
        derived names and the same prefix fail to compile on the duplicate
        name: pass distinct prefixes, or ``params=`` to share one kernel.
    params : sequence of Parameter, optional
        The hyperparameter objects themselves, one per free element in
        ``kernel.theta`` order.  Pass the same objects to several ``kernel``
        calls to share hyperparameters between comparisons.
    on : optional
        See :class:`Term`.
    """
    nk = _n_free_elements(kernel)
    if params is None:
        kparams = _kernel_params(kernel, prefix)
    else:
        kparams = list(params)
        if len(kparams) != nk:
            raise ValueError(
                f"kernel has {nk} free hyperparameter element(s) but params= "
                f"has {len(kparams)}"
            )
    amplitude_params = tuple(amplitude_params)
    fixed_K = nk == 0 and not as_transform(coords).params

    def fn(c, *values):
        if fixed_K:
            K = np.asarray(kernel(as_2d(c.x)), dtype=float)
        else:
            K = kernel.clone_with_theta(np.asarray(values[:nk], dtype=float))(
                as_2d(c.x)
            )
        if amplitude is not None:
            a = amplitude(c, *values[nk:]) if callable(amplitude) else amplitude
            a = np.broadcast_to(np.asarray(a, dtype=float), (len(c),))
            K = np.outer(a, a) * K
        else:
            K = np.array(K, dtype=float)
        K[np.diag_indices_from(K)] += jitter
        return K

    return Term(
        fn,
        tuple(kparams) + amplitude_params,
        kind="matrix",
        on=on,
        coords=coords,
        constant=nk == 0 and not amplitude_params and not callable(amplitude),
    )
