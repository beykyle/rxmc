"""
Covariance terms: the additive pieces of a constraint's covariance.

A constraint owns one multivariate distribution over the stacked residual of
its comparisons; its covariance is a sum of :class:`Term` s.  There is exactly
**one** term type.  A ``Term`` is a ``jax.numpy`` callable ``fn(c, *values) ->
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

A term whose ``fn`` is a function of the context is defined wherever the model
is, so :func:`~rxmc.predictive.grid_draws` can evaluate it on a grid that was
never measured.  A term that is an array — the reported statistical errors, a
fixed covariance, a per-point ``magnitude=`` — exists only at the measured rows.

Two mechanisms are expressed here (see ``docs/groundup_design.md``):

* **Correlating comparisons** — a ``mode`` or ``matrix`` term whose ``on``
  spans several comparisons writes off-diagonal blocks, coupling the data.
* **Sharing a parameter** — terms declare the :class:`~rxmc.params.Parameter`
  objects they consume *by identity*: pass the same object to two terms and
  they share one sampled value.

The factory helpers (:func:`statistical`, :func:`offset`, :func:`normalization`,
:func:`noise`, :func:`proportional_error`, :func:`systematic`,
:func:`kernel`) are one-line conveniences that build the common terms; anything
they cannot express is a direct ``Term(fn, params, kind=...)``.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Mapping

import jax.numpy as jnp
import numpy as np
from tinygp.kernels import Kernel

from .params import Parameter
from .transforms import as_transform, identity

__all__ = [
    "KINDS",
    "TermContext",
    "Term",
    "KernelTerm",
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
    "proportional_error",
    "systematic",
    "kernel",
    "kernel_params",
]

KINDS = ("diag", "mode", "matrix")


def as_2d(X):
    """Promote a 1-D input grid to a single-column 2-D array."""
    X = jnp.asarray(X, dtype=float)
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

    ``x`` and ``y`` are numpy (a function of them alone, such as a Legendre
    basis from scipy, is a constant) unless a parametric ``coords`` transform
    makes ``x`` traced; ``ym`` and the term's parameter values are traced by
    JAX, so anything computed from them uses ``jax.numpy``.

    A term spanning several comparisons sees their rows *gathered* into one
    stack, in constraint order.  :attr:`segments` are the slices of that stack
    belonging to each comparison, :attr:`labels` their labels, and
    :meth:`split` cuts any support-length array along them, so a basis that
    differentiates or smooths along the grid can stay within one comparison::

        def slope(c):  # angle-calibration mode: the slope of each prediction
            return jnp.concatenate([jnp.gradient(ym, x) for x, ym in
                                    zip(c.split(c.x), c.split(c.ym))])
    """

    x: Any
    y: Any
    ym: Any = None
    _meta: Mapping[str, np.ndarray] | None = None
    _segments: tuple | None = None
    _labels: tuple | None = None

    def __len__(self) -> int:
        return len(self.y)

    @property
    def segments(self) -> tuple:
        """Slices of the support, one per comparison it spans, in constraint order.

        A term evaluated outside a problem (as in tests) has one segment.
        """
        return (slice(0, len(self)),) if self._segments is None else self._segments

    @property
    def labels(self) -> tuple:
        """The label of each segment's comparison (``str(comparison.data.label)``)."""
        return ("",) * len(self.segments) if self._labels is None else self._labels

    def split(self, a) -> list:
        """``a[s] for s in segments``: views of a support-length array per comparison."""
        if a.shape[0] != len(self):
            raise ValueError(
                f"split expects an array of length {len(self)}, got shape {a.shape}"
            )
        return [a[s] for s in self.segments]

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
        ``fn(c, *values) -> array`` with ``c`` a :class:`TermContext` and
        ``values`` the sampled values of ``params`` (in order), written with
        ``jax.numpy`` so gradients flow through it.  A plain array is
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
        :attr:`params`.  A transform needs numeric ``x``; without one, ``fn``
        sees ``x`` exactly as the dataset holds it (any dtype).
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
    _appended: Any = None  # the coords parameters appended to params last time

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")
        coords = as_transform(self.coords)
        fn_params = tuple(self.params)
        # dataclasses.replace re-runs this with the previous coords' parameters
        # already appended to params: take them off before appending the current
        k = len(self._appended or ())
        if k and len(fn_params) >= k:
            if all(a is b for a, b in zip(fn_params[-k:], self._appended)):
                fn_params = fn_params[:-k]
        for p in fn_params + coords.params:
            if not isinstance(p, Parameter):
                raise TypeError(f"params must be Parameter objects, got {p!r}")
        object.__setattr__(self, "coords", coords)
        object.__setattr__(self, "params", fn_params + coords.params)
        object.__setattr__(self, "_appended", coords.params)
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

    def context(
        self, x, y, ym=None, meta=None, *values, segments=None, labels=None
    ) -> TermContext:
        """The :class:`TermContext` this term sees on its support at ``values``.

        ``segments``/``labels`` describe the comparisons the support spans (see
        :attr:`TermContext.segments`); omitted, the support is one segment.
        """
        self._check_count(values)
        # the data stay numpy constants (x is opaque, any dtype, unless a
        # coordinate transform needs it numeric), so a basis of the data alone
        # may use numpy or scipy even under jit
        if not self.coords.is_identity:
            x = self.coords(x, *values[self._n_fn_params :])
        elif np.asarray(x).dtype.kind in "fiu":
            x = np.asarray(x, dtype=float)
        return TermContext(
            x=x,
            y=np.asarray(y, dtype=float),
            ym=None if ym is None else jnp.asarray(ym),
            _meta=meta,
            _segments=None if segments is None else tuple(segments),
            _labels=None if labels is None else tuple(labels),
        )

    def value(self, x, y, ym=None, *values, meta=None, segments=None, labels=None):
        """The raw array ``fn`` returns (std vector, mode vector, or block)."""
        n = len(y)
        if not callable(self.fn):
            if self.fn.shape != self.expected_shape(n):
                raise ValueError(
                    f"{self.kind} term expects shape {self.expected_shape(n)}, "
                    f"got {self.fn.shape}"
                )
            return jnp.asarray(self.fn)
        c = self.context(x, y, ym, meta, *values, segments=segments, labels=labels)
        v = jnp.asarray(self.fn(c, *values[: self._n_fn_params]), dtype=float)
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


@dataclass(eq=False, frozen=True)
class KernelTerm(Term):
    """A :func:`~rxmc.terms.kernel` term that also carries what GP conditioning needs.

    :func:`~rxmc.predictive.gp_predictive_draws` reads these to condition the
    discrepancy at new points; the covariance machinery treats a
    ``KernelTerm`` exactly as a ``matrix`` :class:`Term`.

    Parameters
    ----------
    build : callable
        ``build(*kernel_values) -> tinygp kernel``, as passed to :func:`kernel`
        (a fixed kernel is wrapped as a parameter-free builder).
    n_kernel : int
        Number of kernel parameters: ``params[:n_kernel]`` are ``build``'s,
        ``params[n_kernel:]`` the amplitude's, then any coordinate-transform
        parameters.
    amplitude : callable or array or None
        The amplitude as passed to :func:`kernel`.
    jitter : float
        The diagonal nugget of every kernel block, relative to the kernel's
        mean variance before the amplitude scales it.
    """

    build: Any = None
    n_kernel: int = 0
    amplitude: Any = None
    jitter: float = 0.0


# ----------------------------------------------------------------------------
# Standard bases and amplitudes (jax.numpy callables over a TermContext)
# ----------------------------------------------------------------------------


def ones(c: TermContext):
    """Constant unit basis."""
    return jnp.ones(len(c))


def ym(c: TermContext):
    """The model prediction — the prediction-scaled basis."""
    return c.ym


def averaging(c: TermContext):
    """``0.5 * (y + ym)`` — the averaging model-error basis."""
    return 0.5 * (c.y + c.ym)


_AVERAGING = averaging  # factories take an ``averaging`` flag that shadows the name


def x_basis(scale: float = 1.0):
    """Basis ``x / scale`` (e.g. ``x_basis(np.pi)`` for ``theta/180`` on radians)."""

    def basis(c: TermContext):
        return jnp.asarray(c.x, dtype=float) / scale

    return basis


def exp_growth(scale: float = 1.0, base=ones):
    """Parametric basis ``base(c) * exp(slope * x / scale)``.

    Takes one basis parameter, ``slope``; use with
    ``noise(..., basis=exp_growth(np.pi), basis_params=(slope,))``.
    """

    def basis(c: TermContext, slope: float):
        return base(c) * jnp.exp(slope * jnp.asarray(c.x, dtype=float) / scale)

    return basis


def constant_amplitude(c: TermContext, log_amplitude: float):
    """Kernel amplitude ``exp(log_amplitude)``, constant over the support."""
    return jnp.full(len(c), jnp.exp(log_amplitude))


def exp_growth_amplitude(scale: float = 1.0):
    """Kernel amplitude ``exp(log_amplitude) * exp(slope * x / scale)`` (two params)."""

    def amplitude(c: TermContext, log_amplitude: float, slope: float):
        return jnp.exp(log_amplitude) * jnp.exp(
            slope * jnp.asarray(c.x, dtype=float) / scale
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
    v = jnp.asarray(v, dtype=float)
    if v.ndim == 0:
        return jnp.full(n, v)
    if v.shape != (n,):
        raise ValueError(
            f"magnitude/basis has shape {v.shape} but the term's support has length {n}"
        )
    return v


def _coefficient(parameter, log):
    """Parameter -> multiplicative coefficient ``exp(theta)`` (``log``) or ``theta``."""
    if parameter is None:
        return (), (lambda values: 1.0)
    if log:
        return (parameter,), (lambda values: jnp.exp(values[0]))
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


def proportional_error(parameter, averaging=False, log=True, on=None) -> Term:
    """An unknown uncorrelated error proportional to the prediction.

    ``diag((c * z)**2)`` with ``c = exp(theta)`` when ``log`` (else ``theta``)
    and ``z = ym``, or ``z = 0.5 * (y + ym)`` when ``averaging`` (KDUQ's
    spelling, which stays finite when ``ym`` is near zero).  Because it scales
    with the prediction, the covariance changes with the model parameters.

    **Additive** on top of the reported statistical diagonal (see :func:`noise`
    for how to get replace-semantics instead).
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


def kernel_params(names, bounds=None, prefix="discrepancy") -> list:
    """One :class:`~rxmc.params.Parameter` per kernel hyperparameter.

    A convenience for :func:`kernel`: ``kernel_params(["log_var", "log_ell"],
    [(-12, 2), (-3, 1)])`` gives ``discrepancy_log_var`` and
    ``discrepancy_log_ell`` with uniform priors on those bounds.  tinygp
    kernels carry no hyperparameter metadata, so the parameters (and the
    space they are sampled in) are declared here and the builder passed to
    :func:`kernel` maps them to the kernel.  With ``bounds=None`` the
    parameters carry no prior: give bounds, or cover them with a joint block
    in ``Problem(priors=)``.
    """
    names = list(names)
    bounds = [None] * len(names) if bounds is None else list(bounds)
    if len(bounds) != len(names):
        raise ValueError(f"{len(names)} names but {len(bounds)} bounds")
    return [
        Parameter(
            f"{prefix}_{n}" if prefix else n,
            **({} if b is None else {"bounds": tuple(b)}),
            latex=n,
        )
        for n, b in zip(names, bounds)
    ]


def kernel(
    build,
    params=(),
    coords=None,
    amplitude=None,
    amplitude_params=(),
    jitter=1e-10,
    on=None,
) -> Term:
    """A Gaussian-process kernel ``a a^T * K(x, x; theta)`` over the support.

    Parameters
    ----------
    build : callable or tinygp kernel
        ``build(*values) -> tinygp.kernels.Kernel`` with ``values`` the sampled
        values of ``params``; e.g. ``lambda log_var, log_ell: jnp.exp(log_var)
        * tinygp.kernels.Matern52(scale=jnp.exp(log_ell))``.  The builder owns
        the parameterisation (log space, a sum of kernels, an anisotropic
        metric), so anything tinygp expresses is a kernel term.  A kernel
        object itself is a fixed kernel (``params`` must be empty).
    params : sequence of Parameter
        The hyperparameters ``build`` consumes, in order; see
        :func:`kernel_params`.  Pass the same objects to several ``kernel``
        calls to share hyperparameters between comparisons.
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
        ``jitter * mean(diag K)`` is added to the kernel's diagonal *before*
        the amplitude scales it, for numerical stability: each point's nugget
        is then that fraction of its own variance, so it never dominates a
        small variance (a cross section in b/sr, or the small end of a growing
        amplitude).
    on : optional
        See :class:`Term`.
    """
    kparams = tuple(params)
    if not callable(build) or isinstance(build, Kernel):
        if kparams:
            raise ValueError(
                "a fixed kernel takes no params; pass a builder "
                "build(*values) -> kernel to sample its hyperparameters"
            )
        fixed = build
        build = lambda: fixed  # noqa: E731
    _check_arity(build, kparams)
    nk = len(kparams)
    amplitude_params = tuple(amplitude_params)

    def fn(c, *values):
        X = as_2d(c.x)
        K = build(*values[:nk])(X, X)
        nugget = jitter * jnp.maximum(jnp.mean(jnp.diag(K)), np.finfo(float).tiny)
        K = K + nugget * jnp.eye(len(c))
        if amplitude is None:
            return K
        a = amplitude(c, *values[nk:]) if callable(amplitude) else amplitude
        a = jnp.broadcast_to(jnp.asarray(a, dtype=float), (len(c),))
        return jnp.outer(a, a) * K

    return KernelTerm(
        fn,
        kparams + amplitude_params,
        kind="matrix",
        on=on,
        coords=coords,
        constant=(
            nk == 0
            and not as_transform(coords).params
            and not amplitude_params
            and not callable(amplitude)
        ),
        build=build,
        n_kernel=nk,
        amplitude=amplitude,
        jitter=jitter,
    )


def _check_arity(build, params) -> None:
    """Fail at declaration, by name, when the builder cannot take the params
    (inside jit it would fail on the first density call, naming nothing)."""
    try:
        sig = inspect.signature(build)
    except ValueError:  # a builtin without a signature: nothing to check
        return
    try:
        sig.bind(*params)
    except TypeError:
        raise ValueError(
            f"the kernel builder takes {sig} but {len(params)} kernel param(s) "
            f"were given {[p.name for p in params]}"
        ) from None
