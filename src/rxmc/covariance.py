"""
Stacked-covariance model for a :class:`~rxmc.constraint.Constraint`.

A constraint owns one multivariate-normal distribution over the *stacked* vector
of all its observations, ``y = [y1; y2; ...]``.  The covariance of that MVN is
built additively from :class:`Term` objects, each of which writes its
contribution into a sub-block of the stacked covariance matrix selected by an
index array ``support`` (``None`` = the whole constraint).

Two mechanisms are expressed here (see ``docs/design.md``):

* **Correlating observations (A)** — a term whose ``support`` spans more than one
  observation block writes off-diagonal blocks, coupling the data.  A
  block-diagonal covariance (every term local to one block) reproduces the old
  per-observation summed likelihood as a special case.
* **Sharing a parameter (B)** — terms declare the :class:`~rxmc.params.Parameter`
  objects they consume *by identity*.  :class:`ConstraintCovariance` deduplicates
  them (gather, not slice): two terms referencing the *same* ``Parameter`` object
  share one entry in the sampled vector.

There is exactly **one** term type.  A :class:`Term` is a numpy-style callable
``fn(c, *values) -> array`` of a :class:`TermContext` ``c`` (the term's local view
of ``x``, ``y`` and ``ym`` on its support, with ``x`` passed through an optional
coordinate :class:`~rxmc.transforms.Transform`) and its parameter values, plus a
``kind`` that says how the returned array enters the covariance:

``"diag"``
    ``fn`` returns a standard-deviation vector ``v``; ``Sigma_ii += v_i**2``.
``"mode"``
    ``fn`` returns a vector ``v``; ``Sigma += outer(v, v)`` (one correlated mode).
``"matrix"``
    ``fn`` returns a full block ``M``; ``Sigma_block += M``.

The factory helpers (:func:`statistical_term`, :func:`normalization_term`,
:func:`offset_term`, :func:`noise_term`, :func:`noise_fraction_term`,
:func:`model_error_term`, :func:`systematic_term`, :func:`kernel_term`) are
one-line conveniences that build the common terms; anything they cannot express
is a direct ``Term(fn, params, kind=...)``.
"""

from dataclasses import dataclass

import numpy as np
import scipy as sc

from .params import Parameter
from .transforms import as_transform

__all__ = [
    "StackContext",
    "TermContext",
    "Term",
    "ConstraintCovariance",
    "stacked_supports",
    "chol_logdet",
    "as_2d",
    "ones",
    "ym",
    "averaging",
    "x_basis",
    "exp_growth",
    "constant_amplitude",
    "exp_growth_amplitude",
    "statistical_term",
    "offset_term",
    "normalization_term",
    "noise_term",
    "noise_fraction_term",
    "model_error_term",
    "systematic_term",
    "kernel_term",
    "discrepancy_term",
]

KINDS = ("diag", "mode", "matrix")


@dataclass(frozen=True)
class StackContext:
    """Bundle of stacked arrays passed to every :meth:`Term.add_to`.

    Attributes
    ----------
    x : np.ndarray
        Stacked independent variable, ``np.concatenate`` over observations.
    y : np.ndarray
        Stacked observed data (in each observation's comparison space).
    ym : np.ndarray or None
        Stacked model prediction (same space as ``y``).  ``None`` when a
        *constant* covariance is assembled before any model evaluation (see
        :meth:`constant`); constant terms never read it.
    supports : tuple of np.ndarray
        One contiguous index array per observation block, in stacking order.
    """

    x: np.ndarray
    y: np.ndarray
    ym: np.ndarray | None
    supports: tuple

    @classmethod
    def constant(cls, x, y, supports) -> "StackContext":
        """A stack with no model prediction, for assembling constant terms."""
        return cls(x=x, y=y, ym=None, supports=tuple(supports))


@dataclass(frozen=True)
class TermContext:
    """A term's local view of the stack on its own support.

    ``x`` are the coordinates on the support — ``ctx.x[support]`` passed
    through the term's ``coords`` transform (so ``x`` may be 2-D); ``y`` and
    ``ym`` are the observed data and model prediction on the support (``ym`` is
    ``None`` when a constant covariance is assembled without a model
    prediction), and ``support`` the stacked indices this view corresponds to.
    ``len(c)`` is the number of points.
    """

    x: np.ndarray
    y: np.ndarray
    ym: np.ndarray | None
    support: np.ndarray

    def __len__(self):
        return len(self.support)


def stacked_supports(observations) -> tuple:
    """One contiguous index array per observation, in stacking order.

    The block layout of the stacked vector ``y = [y1; y2; ...]``:
    ``Constraint`` uses this internally; callers only need it to place a term on
    a *subset* of a constraint's observations (``support=None`` covers all).
    """
    supports, b = [], 0
    for obs in observations:
        supports.append(np.arange(b, b + obs.n_data_pts))
        b += obs.n_data_pts
    return tuple(supports)


def chol_logdet(Sigma):
    """Lower Cholesky factor and log-determinant of a positive-definite matrix."""
    L = sc.linalg.cholesky(Sigma, lower=True)
    return L, 2.0 * float(np.sum(np.log(np.diag(L))))


def as_2d(X) -> np.ndarray:
    """Promote a 1-D input grid to a single-column 2-D array (sklearn kernels)."""
    X = np.asarray(X, dtype=float)
    return X[:, None] if X.ndim == 1 else X


# ----------------------------------------------------------------------------
# The term
# ----------------------------------------------------------------------------


class Term:
    """One additive contribution to the stacked covariance.

    Parameters
    ----------
    fn : callable or array_like
        ``fn(c, *values) -> np.ndarray`` with ``c`` a :class:`TermContext` and
        ``values`` the sampled values of ``params`` (in order).  A plain array is a
        fixed contribution (``constant=True`` implied): a standard-deviation
        vector for ``kind="diag"``, a mode vector for ``"mode"``, or a symmetric
        block for ``"matrix"``.
    params : sequence of Parameter, optional
        Parameters consumed by ``fn``, matched *by identity* across terms
        (pass the same object to two terms to share one sampled value).
    kind : {"diag", "mode", "matrix"}
        How the returned array enters the covariance (see module docstring).
    support : array_like of int, optional
        Indices into the stacked vector.  ``None`` (default) means the whole
        constraint; it is resolved when the term is added to a
        :class:`ConstraintCovariance`.
    coords : Transform or callable, optional
        Coordinate transform applied to ``x[support]`` before ``fn`` sees it
        (e.g. angle to momentum transfer).  Its parameters, if any, are appended
        to :attr:`params`.
    constant : bool, optional
        Declare that a *callable* ``fn`` does not read ``c.ym`` (the model
        prediction) and has no parameters, so the contribution can be evaluated
        once and cached.  ``c.x`` and ``c.y`` are invariant per constraint and
        may be read freely (e.g. a fixed-hyperparameter kernel over ``x``).  A
        constant term is first evaluated with ``c.ym is None``, so a
        mis-declared term fails loudly.  Ignored (``True``) for array ``fn``.
    """

    def __init__(
        self,
        fn,
        params=(),
        *,
        kind="matrix",
        support=None,
        coords=None,
        constant=False,
    ):
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        self.kind = kind
        self.coords = as_transform(coords)
        fn_params = tuple(params)
        for p in fn_params + self.coords.params:
            if not isinstance(p, Parameter):
                raise TypeError(f"params must be Parameter objects, got {p!r}")
        self._n_fn_params = len(fn_params)
        self.params = fn_params + self.coords.params
        self.support = None
        self._cache = None
        self._x_cache = None

        if callable(fn):
            self.fn = fn
            self._array = None
            self.is_constant = bool(constant) and not self.params
        else:
            self.fn = None
            self._array = np.asarray(fn, dtype=float)
            if self.params:
                raise ValueError("an array-valued term cannot have parameters")
            self.is_constant = True
        if support is not None:
            self._set_support(np.asarray(support, dtype=int))

    # -- structure ----------------------------------------------------------

    @property
    def couples_offdiagonal(self) -> bool:
        """Whether the term can write off-diagonal entries (``kind != "diag"``)."""
        return self.kind != "diag"

    @property
    def bound(self) -> bool:
        return self.support is not None

    def bind(self, N: int) -> None:
        """Resolve ``support=None`` to the whole stack of length ``N``.

        Idempotent; a term constructed with an explicit support is untouched.
        """
        if self.support is None:
            self._set_support(np.arange(int(N)))

    def _set_support(self, ix: np.ndarray) -> None:
        self.support = ix
        n = len(ix)
        # supports from ``stacked_supports`` (and the whole stack) are
        # contiguous: index the block with slices instead of a gather/scatter
        if n and np.array_equal(ix, np.arange(ix[0], ix[0] + n)):
            sl = slice(int(ix[0]), int(ix[0]) + n)
            self._block = (sl, sl)
        else:
            self._block = np.ix_(ix, ix)
        if self._array is not None:
            self._validate_bound()

    @property
    def _expected_shape(self) -> tuple:
        n = len(self.support)
        return (n, n) if self.kind == "matrix" else (n,)

    def _validate_bound(self):
        a = self._array
        if a.shape != self._expected_shape:
            raise ValueError(
                f"{self.kind} term expects shape {self._expected_shape}, got {a.shape}"
            )
        if self.kind == "matrix" and not np.allclose(a, a.T):
            raise ValueError("matrix term must be symmetric")

    # -- evaluation -----------------------------------------------------------

    def _check_bound(self):
        if self.support is None:
            raise ValueError(
                "term support is unresolved; add it to a Constraint / "
                "ConstraintCovariance (which binds support=None to the whole "
                "stack) or pass support= explicitly"
            )

    def local_context(self, ctx: StackContext, theta=()) -> TermContext:
        """The :class:`TermContext` this term sees at ``theta``."""
        self._check_bound()
        ix = self.support
        x = self._coords_x(ctx, np.asarray(theta, dtype=float))
        ym = None if ctx.ym is None else ctx.ym[ix]
        return TermContext(x=x, y=ctx.y[ix], ym=ym, support=ix)

    def _coords_x(self, ctx, theta):
        """``coords(x[support])``; cached when the transform is parameter-free
        (``x`` is invariant per constraint)."""
        if self.coords.is_identity:
            return ctx.x[self.support]
        if self.coords.params:
            return self.coords(ctx.x[self.support], *theta[self._n_fn_params :])
        if self._x_cache is None:
            self._x_cache = self.coords(ctx.x[self.support])
            self._x_cache.setflags(write=False)
        return self._x_cache

    def value(self, ctx: StackContext, theta=()) -> np.ndarray:
        """The raw array ``fn`` returns (std vector, mode vector, or block)."""
        self._check_bound()
        if self._array is not None:
            return self._array
        if self.is_constant and self._cache is not None:
            return self._cache
        theta = np.asarray(theta, dtype=float)
        if len(theta) != len(self.params):
            raise ValueError(f"expected {len(self.params)} params, got {len(theta)}")
        c = self.local_context(ctx, theta)
        v = np.asarray(self.fn(c, *theta[: self._n_fn_params]), dtype=float)
        if v.shape != self._expected_shape:
            raise ValueError(
                f"{self.kind} term fn returned shape {v.shape}, "
                f"expected {self._expected_shape}"
            )
        if self.is_constant:
            v.setflags(write=False)
            self._cache = v
        return v

    def add_to(self, Sigma: np.ndarray, ctx: StackContext, theta) -> None:
        """Add this term's contribution to the stacked ``Sigma`` in place."""
        v = self.value(ctx, theta)
        if self.kind == "diag":
            ix = self.support
            Sigma[ix, ix] += v**2
        elif self.kind == "mode":
            Sigma[self._block] += np.outer(v, v)
        else:
            Sigma[self._block] += v

    def __repr__(self):
        names = ", ".join(p.name for p in self.params)
        sup = "all" if self.support is None else f"{len(self.support)} pts"
        return f"Term(kind={self.kind!r}, params=({names}), support={sup})"


# ----------------------------------------------------------------------------
# Constraint-local covariance: gather-by-identity over the stacked space
# ----------------------------------------------------------------------------


class ConstraintCovariance:
    """The stacked covariance of a constraint, assembled from :class:`Term` s.

    Parameters are routed *by identity*: the unique ``Parameter`` objects across
    all terms (first-seen order) form the flat parameter vector; each term gathers
    its own parameters from that vector.  Referencing the *same* ``Parameter``
    object in two terms makes them share one sampled value (case B).

    Parameters
    ----------
    terms : sequence of Term
        Additive covariance contributions.  Terms with ``support=None`` are
        bound to the whole stack here.
    N : int
        Dimension of the stacked vector.
    blocks : sequence of np.ndarray, optional
        One index array per observation block.  When given, :attr:`block_diagonal`
        is decided against the true block boundaries.  When omitted, only a
        covariance whose every term is strictly diagonal
        (``couples_offdiagonal == False``) is classified block-diagonal; any
        coupling-capable term conservatively forces the dense path — there is no
        guessing of block structure from support shape.
    active : array_like of int, optional
        Indices of the *active* (unmasked) rows of the stack.  The full
        ``N x N`` matrix is always assembled (terms are authored in the full
        space); factorisation and the Mahalanobis distance are restricted to
        ``active``.  ``None`` means all rows.

    ``terms`` and ``blocks`` are treated as immutable after construction:
    :attr:`block_diagonal` and :attr:`is_constant` are decided once, here.
    """

    def __init__(self, terms, N, blocks=None, active=None):
        self.terms = list(terms)
        self.N = int(N)
        for t in self.terms:
            if not isinstance(t, Term):
                raise TypeError(f"terms must be Term objects, got {type(t).__name__}")
            t.bind(self.N)
        self._blocks = (
            None if blocks is None else [np.asarray(b, dtype=int) for b in blocks]
        )
        if active is None:
            self.active = None
        else:
            active = np.asarray(active, dtype=int)
            self.active = None if active.size == self.N else active
        self.n_active = self.N if self.active is None else int(self.active.size)
        if self._blocks is not None and self.active is not None:
            self._active_blocks = [b[np.isin(b, self.active)] for b in self._blocks]
        else:
            self._active_blocks = self._blocks

        params, index_of = [], {}
        for t in self.terms:
            for p in t.params:
                if id(p) not in index_of:  # dedup by identity -> sharing
                    index_of[id(p)] = len(params)
                    params.append(p)
        self.params = tuple(params)
        self._gather = [
            np.array([index_of[id(p)] for p in t.params], dtype=int) for t in self.terms
        ]
        self._const_cache = None
        self._chol_cache = None
        self._block_chol_cache = None

        self.block_diagonal = all(
            not t.couples_offdiagonal or self._within_one_block(t.support)
            for t in self.terms
        )
        self.is_constant = self.n_params == 0 and all(t.is_constant for t in self.terms)

    def _within_one_block(self, support) -> bool:
        """True if ``support`` lies inside a single known observation block."""
        if self._blocks is None:
            return False
        support = np.asarray(support, dtype=int)
        return any(bool(np.isin(support, b).all()) for b in self._blocks)

    @property
    def n_params(self) -> int:
        return len(self.params)

    @property
    def blocks(self):
        """Observation block index arrays, or ``None`` if unknown."""
        return self._blocks

    @property
    def uses_block_path(self) -> bool:
        """Whether :meth:`stacked_distance` factors block by block
        (:meth:`block_cholesky`) rather than the whole active stack
        (:meth:`cholesky`)."""
        return (
            self.block_diagonal and self._blocks is not None and len(self._blocks) > 1
        )

    def matrix(self, ctx, *theta) -> np.ndarray:
        """Assemble the full stacked covariance matrix (all ``N`` rows).

        Parameters
        ----------
        ctx : StackContext
            Stacked arrays.  When :attr:`is_constant`, ``ctx.ym`` is never read
            (it may be ``None``, see :meth:`StackContext.constant`) and the
            result is cached.
        *theta : float
            One value per unique parameter, in :attr:`params` order.
        """
        if len(theta) != self.n_params:
            raise ValueError(f"expected {self.n_params} params, got {len(theta)}")
        if self.is_constant and self._const_cache is not None:
            return self._const_cache
        theta = np.asarray(theta, dtype=float)
        Sigma = np.zeros((self.N, self.N))
        for t, g in zip(self.terms, self._gather):
            t.add_to(Sigma, ctx, theta[g])
        if self.is_constant:
            # cached arrays are shared across calls; freeze so aliasing
            # bugs fail loudly instead of corrupting later evaluations
            Sigma.setflags(write=False)
            self._const_cache = Sigma
        return Sigma

    def active_matrix(self, ctx, *theta) -> np.ndarray:
        """The covariance restricted to the active rows/columns."""
        Sigma = self.matrix(ctx, *theta)
        if self.active is None:
            return Sigma
        return Sigma[np.ix_(self.active, self.active)]

    def cholesky(self, ctx, *theta):
        """Lower Cholesky factor and log-determinant of the active stacked covariance.

        Cached when :attr:`is_constant`, so a fixed covariance is factored once.

        Returns
        -------
        (np.ndarray, float)
            ``(L, logdet)`` with ``L`` lower-triangular and
            ``logdet = log det Sigma``.
        """
        if self.is_constant and self._chol_cache is not None:
            return self._chol_cache
        L, logdet = chol_logdet(self.active_matrix(ctx, *theta))
        result = (L, logdet)
        if self.is_constant:
            L.setflags(write=False)
            self._chol_cache = result
        return result

    def block_cholesky(self, ctx, *theta):
        """Per-block ``(L_i, logdet_i)`` factors, aligned with :attr:`blocks`.

        Only meaningful when :attr:`block_diagonal`; cached when
        :attr:`is_constant` so a constant block-diagonal covariance is factored
        once instead of on every likelihood evaluation.  Blocks are restricted to
        the active rows; a fully-masked block yields ``(empty, 0.0)``.

        Returns
        -------
        tuple of (np.ndarray, float)
            One ``(L_i, logdet_i)`` pair per block, ``L_i`` lower-triangular.
        """
        if self._blocks is None:
            raise ValueError("block_cholesky requires blocks to be set")
        if self.is_constant and self._block_chol_cache is not None:
            return self._block_chol_cache
        Sigma = self.matrix(ctx, *theta)
        factors = []
        for ix in self._active_blocks:
            if ix.size == 0:
                factors.append((np.zeros((0, 0)), 0.0))
                continue
            L, logdet = chol_logdet(Sigma[np.ix_(ix, ix)])
            L.setflags(write=False)
            factors.append((L, logdet))
        factors = tuple(factors)
        if self.is_constant:
            self._block_chol_cache = factors
        return factors

    def stacked_distance(self, ctx, params=()):
        r"""Squared Mahalanobis distance and log-determinant over the active residual.

        Owns the dispatch between the block-diagonal fast path (factor each
        block separately, :math:`O(\sum n_i^3)`, cached per block via
        :meth:`block_cholesky` when constant) and a single dense Cholesky over
        the full stack (cached via :meth:`cholesky` when constant).

        Returns
        -------
        (float, float)
            ``(d2, logdet)``.
        """
        params = tuple(params)
        r = ctx.y - ctx.ym
        if self.uses_block_path:
            factors = self.block_cholesky(ctx, *params)
            d2 = 0.0
            logdet = 0.0
            for ix, (L, ld) in zip(self._active_blocks, factors):
                if ix.size == 0:
                    continue
                z = sc.linalg.solve_triangular(L, r[ix], lower=True)
                d2 += float(np.dot(z, z))
                logdet += ld
            return d2, logdet

        L, logdet = self.cholesky(ctx, *params)
        if self.active is not None:
            r = r[self.active]
        z = sc.linalg.solve_triangular(L, r, lower=True)
        return float(np.dot(z, z)), logdet


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
    ``noise_term(..., basis=exp_growth(np.pi), basis_params=(slope,))``.
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
# Term factory helpers (the assembly-time builders)
# ----------------------------------------------------------------------------


def _masked(magnitude, mask=None):
    """A scalar/array magnitude with an optional mask.

    Scalar-like values include 0-d ndarrays (e.g. ``np.array(0.05)`` as stored by
    ``exfor_tools`` distributions), not just Python scalars.  An array magnitude
    is checked against the support length when the term is evaluated
    (``np.broadcast_to`` in the basis).
    """
    v = float(magnitude) if np.ndim(magnitude) == 0 else np.asarray(magnitude, float)
    if mask is not None:
        v = v * np.asarray(mask, dtype=float)
    return v


def _coefficient(parameter, log):
    """Parameter -> multiplicative coefficient ``exp(theta)`` (``log``) or ``theta``."""
    if parameter is None:
        return (), (lambda values: 1.0)
    if log:
        return (parameter,), (lambda values: np.exp(values[0]))
    return (parameter,), (lambda values: values[0])


def _scaled_term(
    kind, parameter, log, basis, basis_params=(), *, support=None, coords=None
):
    """``c * basis(ctx, *basis_values)`` as a term of the given kind."""
    cparams, coef = _coefficient(parameter, log)
    basis_params = tuple(basis_params)
    nc = len(cparams)

    def fn(c, *values):
        b = basis(c, *values[nc:]) if callable(basis) else basis
        return coef(values[:nc]) * np.broadcast_to(b, (len(c),))

    return Term(fn, cparams + basis_params, kind=kind, support=support, coords=coords)


def statistical_term(stat_err, support=None) -> Term:
    """Always-on, genuinely uncorrelated statistical diagonal ``diag(stat_err**2)``."""
    return Term(np.asarray(stat_err, dtype=float), kind="diag", support=support)


def offset_term(
    magnitude=None, parameter=None, mask=None, log=True, support=None
) -> Term:
    """A correlated absolute-offset systematic ``outer(omega, omega)``.

    With ``magnitude`` it is a fixed (data-given) rank-one mode; with ``parameter``
    it is a free nuisance magnitude (``c = exp(theta)`` when ``log``).
    """
    if magnitude is None and parameter is None:
        raise ValueError("offset_term requires a magnitude and/or a parameter")
    mag = _masked(1.0 if magnitude is None else magnitude, mask=mask)

    def basis(c):
        return np.broadcast_to(mag, (len(c),))

    if parameter is None:
        return Term(basis, kind="mode", support=support, constant=True)
    return _scaled_term("mode", parameter, log, basis, support=support)


def normalization_term(
    magnitude=None, parameter=None, mask=None, log=True, support=None
) -> Term:
    """A correlated normalisation systematic ``outer(eta * ym, eta * ym)``.

    With ``magnitude`` it is a fixed fractional normalisation uncertainty; with
    ``parameter`` the magnitude eta is a free nuisance (``c = exp(theta)`` when
    ``log``).  In both cases the mode scales with the model prediction ``ym``.
    """
    if magnitude is None and parameter is None:
        raise ValueError("normalization_term requires a magnitude and/or a parameter")
    mag = _masked(1.0 if magnitude is None else magnitude, mask=mask)

    def basis(c):
        return np.broadcast_to(mag, (len(c),)) * c.ym

    return _scaled_term("mode", parameter, log, basis, support=support)


def noise_term(
    parameter, log=True, basis=None, basis_params=(), support=None, coords=None
) -> Term:
    """Unknown statistical noise ``diag((epsilon * basis)**2)``.

    ``basis`` defaults to ones (constant noise); pass any
    ``basis(c, *basis_values)`` — e.g. :func:`exp_growth` with
    ``basis_params=(slope,)`` for noise growing along ``x``.

    This term is **additive**: a :class:`~rxmc.constraint.Constraint` already adds
    each observation's reported statistical diagonal, so the assembled covariance
    is ``diag(y_stat_err**2 + epsilon**2)``.  To make the inferred noise *replace*
    the reported statistics, build the
    ``Observation`` with zero ``y_stat_err`` or pass ``include_statistical_term=False``
    to the ``Constraint``.
    """
    return _scaled_term(
        "diag",
        parameter,
        log,
        ones if basis is None else basis,
        basis_params,
        support=support,
        coords=coords,
    )


def noise_fraction_term(parameter, log=True, support=None) -> Term:
    """Unknown fractional noise ``diag((epsilon * ym)**2)``.

    **Additive** on top of the reported statistical diagonal (see
    :func:`noise_term` for how to get replace-semantics instead).
    """
    return _scaled_term("diag", parameter, log, ym, support=support)


def model_error_term(parameter, averaging=True, log=True, support=None) -> Term:
    """Unknown uncorrelated model error ``diag((gamma * z)**2)``.

    ``z = 0.5 * (y + ym)`` when ``averaging`` (stabilises when ``ym`` is near zero),
    else ``z = ym``.
    """
    basis = _AVERAGING if averaging else ym
    return _scaled_term("diag", parameter, log, basis, support=support)


def systematic_term(
    parameter, basis, log=True, basis_params=(), support=None, coords=None
) -> Term:
    """A correlated mode ``outer(s * u, s * u)`` with a user basis ``u = basis(c, ...)``.

    :func:`offset_term` and :func:`normalization_term` are its ``ones``/``ym``
    special cases; use e.g. ``basis=x_basis(np.pi)`` for a mode growing with
    angle.
    """
    return _scaled_term(
        "mode", parameter, log, basis, basis_params, support=support, coords=coords
    )


def kernel_term(
    kernel,
    coords=None,
    amplitude=None,
    amplitude_params=(),
    jitter=1e-10,
    prefix="discrepancy",
    support=None,
) -> Term:
    """A Gaussian-process kernel ``a a^T * K(x, x; theta)`` over the support.

    One :class:`~rxmc.params.Parameter` is auto-derived per *free* kernel
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
        ``outer(a, a) * K``.  See :func:`constant_amplitude`,
        :func:`exp_growth_amplitude`.
    amplitude_params : sequence of Parameter, optional
        Parameters consumed by ``amplitude``.
    jitter : float, optional
        Added to the diagonal after scaling, for numerical stability.
    prefix : str, optional
        Name prefix of the auto-derived kernel parameters.
    support : array_like of int, optional
        See :class:`Term`.
    """
    kparams = []
    for hp in kernel.hyperparameters:
        if hp.fixed:
            continue
        if hp.n_elements == 1:
            kparams.append(Parameter(f"{prefix}_{hp.name}", float, latex_name=hp.name))
        else:
            kparams.extend(
                Parameter(
                    f"{prefix}_{hp.name}_{i}", float, latex_name=f"{hp.name}[{i}]"
                )
                for i in range(hp.n_elements)
            )
    nk = len(kparams)
    amplitude_params = tuple(amplitude_params)
    # with no free kernel hyperparameters and parameter-free coordinates,
    # K(x, x) is invariant per constraint: build it once
    fixed_K = nk == 0 and not as_transform(coords).params
    K_cache = []

    def fn(c, *values):
        if fixed_K:
            if not K_cache:
                K_cache.append(np.asarray(kernel(as_2d(c.x)), dtype=float))
            K = K_cache[0]
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
        support=support,
        coords=coords,
        constant=nk == 0 and not amplitude_params and not callable(amplitude),
    )


discrepancy_term = kernel_term
"""Alias of :func:`kernel_term` (a GP model-discrepancy term)."""
