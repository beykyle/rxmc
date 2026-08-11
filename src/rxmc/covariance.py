"""
Stacked-covariance model for a :class:`~rxmc.constraint.Constraint`.

A constraint owns one multivariate-normal distribution over the *stacked* vector
of all its observations, ``y = [y1; y2; ...]``.  The covariance of that MVN is
built additively from :class:`Term` objects, each of which writes its
contribution into a sub-block of the stacked covariance matrix selected by an
index array ``support``.

Two mechanisms are expressed here (see ``covariance_refactor.md``):

* **Correlating observations (A)** — a term whose ``support`` spans more than one
  observation block writes off-diagonal blocks, coupling the data.  A
  block-diagonal covariance (every term local to one block) reproduces the old
  per-observation summed likelihood as a special case.
* **Sharing a parameter (B)** — terms declare the :class:`~rxmc.params.Parameter`
  objects they consume *by identity*.  :class:`ConstraintCovariance` deduplicates
  them (gather, not slice): two terms referencing the *same* ``Parameter`` object
  share one entry in the sampled vector.

Term primitives
---------------
:class:`DenseTerm`
    A fixed sub-block (statistical diagonal, fixed offset, fixed full covariance).
:class:`DiagonalTerm`
    ``diag((c * basis)**2)`` — an uncorrelated (rank-zero) contribution.
:class:`RankOneTerm`
    ``outer(v, v)`` with ``v = c * basis`` — one correlated mode.
:class:`KernelTerm`
    A Gaussian-process kernel ``K(x, x; theta)`` over ``support``.

Factory helpers (:func:`statistical_term`, :func:`normalization_term`,
:func:`offset_term`, :func:`noise_term`, :func:`noise_fraction_term`,
:func:`model_error_term`, :func:`discrepancy_term`) map each capability of the old
``LikelihoodModel`` zoo to a single ``Term``.
"""

from dataclasses import dataclass

import numpy as np
import scipy as sc

from .params import Parameter

__all__ = [
    "StackContext",
    "Term",
    "DenseTerm",
    "DiagonalTerm",
    "RankOneTerm",
    "KernelTerm",
    "ConstraintCovariance",
    "stacked_supports",
    "chol_logdet",
    "ym_basis",
    "ones_basis",
    "averaging_basis",
    "statistical_term",
    "offset_term",
    "normalization_term",
    "noise_term",
    "noise_fraction_term",
    "model_error_term",
    "discrepancy_term",
]


@dataclass(frozen=True)
class StackContext:
    """Bundle of stacked arrays passed to every :meth:`Term.add_to`.

    Attributes
    ----------
    x : np.ndarray
        Stacked independent variable, ``np.concatenate`` over observations.
    y : np.ndarray
        Stacked observed data.
    ym : np.ndarray
        Stacked model prediction.
    supports : tuple of np.ndarray
        One contiguous index array per observation block, in stacking order.
    """

    x: np.ndarray
    y: np.ndarray
    ym: np.ndarray
    supports: tuple


def stacked_supports(observations) -> tuple:
    """One contiguous index array per observation, in stacking order.

    The block layout of the stacked vector ``y = [y1; y2; ...]``:
    ``Constraint`` uses this internally, and callers building ``extra_terms``
    use it to place a term on the right rows.
    """
    supports, b = [], 0
    for obs in observations:
        supports.append(np.arange(b, b + obs.n_data_pts))
        b += obs.n_data_pts
    return tuple(supports)


# ----------------------------------------------------------------------------
# Standard basis callables (shared modes / diagonal scalings)
# ----------------------------------------------------------------------------


def ym_basis(ctx: StackContext, support: np.ndarray) -> np.ndarray:
    """Model prediction on ``support`` — the prediction-scaled basis."""
    return ctx.ym[support]


def ones_basis(ctx: StackContext, support: np.ndarray) -> np.ndarray:
    """Constant unit basis on ``support``."""
    return np.ones(len(support))


def averaging_basis(ctx: StackContext, support: np.ndarray) -> np.ndarray:
    """``0.5 * (y + ym)`` on ``support`` — the averaging model-error basis."""
    return 0.5 * (ctx.y[support] + ctx.ym[support])


def chol_logdet(Sigma):
    """Lower Cholesky factor and log-determinant of a positive-definite matrix."""
    L = sc.linalg.cholesky(Sigma, lower=True)
    return L, 2.0 * float(np.sum(np.log(np.diag(L))))


def as_2d(X) -> np.ndarray:
    """Promote a 1-D input grid to a single-column 2-D array (sklearn kernels)."""
    X = np.asarray(X, dtype=float)
    return X[:, None] if X.ndim == 1 else X


# ----------------------------------------------------------------------------
# Term primitives
# ----------------------------------------------------------------------------


class Term:
    """One additive contribution to the stacked covariance.

    Subclasses set ``support`` (indices into the stacked vector) and ``params``
    (the :class:`~rxmc.params.Parameter` objects they consume, by identity) and
    implement :meth:`add_to`.

    ``couples_offdiagonal`` declares whether the term can write off-diagonal
    entries of its sub-block: terms that write only the diagonal (e.g.
    :class:`DiagonalTerm`) can never couple observation blocks, whatever their
    support.

    ``is_constant`` declares that the contribution depends on neither ``theta``
    nor ``ctx`` — a covariance whose every term is constant is factored once and
    cached by :class:`ConstraintCovariance`.
    """

    params: tuple = ()
    support: np.ndarray
    couples_offdiagonal: bool = True
    is_constant: bool = False

    def add_to(self, Sigma: np.ndarray, ctx: StackContext, theta: np.ndarray) -> None:
        raise NotImplementedError


class DenseTerm(Term):
    """A fixed sub-block written into ``Sigma[support, support]``.

    A 1-D ``matrix`` is treated as a diagonal (variance vector); a 2-D ``matrix``
    is used as-is.
    """

    is_constant = True

    def __init__(self, support, matrix):
        self.support = np.asarray(support, dtype=int)
        m = np.asarray(matrix, dtype=float)
        n = len(self.support)
        if m.shape not in ((n,), (n, n)):
            raise ValueError(
                f"DenseTerm matrix shape {m.shape} does not match support "
                f"length {n}: expected ({n},) or ({n}, {n})"
            )
        if m.ndim == 2 and not np.allclose(m, m.T):
            raise ValueError("DenseTerm 2-D matrix must be symmetric")
        self.couples_offdiagonal = m.ndim != 1
        self._m = m

    def add_to(self, Sigma, ctx, theta):
        ix = self.support
        if self._m.ndim == 1:
            Sigma[ix, ix] += self._m
        else:
            Sigma[np.ix_(ix, ix)] += self._m


class _ScaledBasisTerm(Term):
    """Shared ``v = c * basis`` machinery for scaled-basis terms.

    ``c = exp(theta)`` when ``log`` else ``theta`` (``c = 1`` when there is no
    parameter).  ``basis`` is an array, a callable ``basis(ctx, support)``, or
    ``None`` (ones).
    """

    def __init__(self, support, basis=None, parameter=None, log=True):
        self.support = np.asarray(support, dtype=int)
        self.basis = basis
        self.log = log
        self.params = (parameter,) if parameter is not None else ()
        # a callable basis reads ctx (e.g. ym); a parameter reads theta
        self.is_constant = not self.params and not callable(basis)

    def _vec(self, ctx, theta):
        if self.params:
            c = np.exp(theta[0]) if self.log else theta[0]
        else:
            c = 1.0
        if callable(self.basis):
            b = self.basis(ctx, self.support)
        elif self.basis is not None:
            b = np.asarray(self.basis, dtype=float)
        else:
            b = np.ones(len(self.support))
        return c * b


class DiagonalTerm(_ScaledBasisTerm):
    """``diag((c * basis)**2)`` on ``support`` — an uncorrelated contribution.

    See :class:`_ScaledBasisTerm` for the ``basis``/``parameter``/``log``
    semantics.
    """

    couples_offdiagonal = False

    def add_to(self, Sigma, ctx, theta):
        v = self._vec(ctx, theta)
        ix = self.support
        Sigma[ix, ix] += v**2


class RankOneTerm(_ScaledBasisTerm):
    """``outer(v, v)`` with ``v = c * basis`` — one correlated mode.

    A local systematic when ``support`` lies in a single observation block; a
    cross-block *coupling* (case A) when ``support`` spans blocks.  See
    :class:`_ScaledBasisTerm` for the ``basis``/``parameter``/``log`` semantics.
    """

    def add_to(self, Sigma, ctx, theta):
        v = self._vec(ctx, theta)
        ix = self.support
        Sigma[np.ix_(ix, ix)] += np.outer(v, v)


class KernelTerm(Term):
    """A Gaussian-process kernel ``K(x, x; theta)`` over ``support``.

    Subsumes the old ``SklearnKernelGPDiscrepancyModel``.  Auto-derives one
    :class:`~rxmc.params.Parameter` per *free* kernel hyperparameter **element**
    (sampled in sklearn's log-theta space): an anisotropic hyperparameter (one
    with ``n_elements > 1``, e.g. a vector ``length_scale``) contributes that many
    parameters, so ``len(self.params) == len(kernel.theta)``.  Local on one block,
    or a correlated discrepancy across blocks when ``support`` spans them.
    """

    def __init__(self, support, kernel, jitter=1e-10, prefix="discrepancy"):
        self.support = np.asarray(support, dtype=int)
        self.kernel = kernel
        self.jitter = float(jitter)
        params = []
        for hp in kernel.hyperparameters:
            if hp.fixed:
                continue
            if hp.n_elements == 1:
                params.append(
                    Parameter(f"{prefix}_{hp.name}", float, latex_name=hp.name)
                )
            else:
                params.extend(
                    Parameter(
                        f"{prefix}_{hp.name}_{i}",
                        float,
                        latex_name=f"{hp.name}[{i}]",
                    )
                    for i in range(hp.n_elements)
                )
        self.params = tuple(params)

    def add_to(self, Sigma, ctx, theta):
        ix = self.support
        X = as_2d(np.asarray(ctx.x)[ix])
        K = self.kernel.clone_with_theta(np.asarray(theta, dtype=float))(X)
        K[np.diag_indices_from(K)] += self.jitter
        Sigma[np.ix_(ix, ix)] += K


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
        Additive covariance contributions.
    N : int
        Dimension of the stacked vector.
    blocks : sequence of np.ndarray, optional
        One index array per observation block.  When given, :attr:`block_diagonal`
        is decided against the true block boundaries.  When omitted, only a
        covariance whose every term is strictly diagonal
        (``couples_offdiagonal == False``) is classified block-diagonal; any
        coupling-capable term conservatively forces the dense path — there is no
        guessing of block structure from support shape.

    ``terms`` and ``blocks`` are treated as immutable after construction:
    :attr:`block_diagonal` and :attr:`is_constant` are decided once, here.
    """

    def __init__(self, terms, N, blocks=None):
        self.terms = list(terms)
        self.N = int(N)
        self._blocks = (
            None if blocks is None else [np.asarray(b, dtype=int) for b in blocks]
        )

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

    def matrix(self, ctx, *theta) -> np.ndarray:
        """Assemble the stacked covariance matrix.

        Parameters
        ----------
        ctx : StackContext
            Stacked arrays; may be ``None`` only when :attr:`is_constant`.
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

    def cholesky(self, ctx, *theta):
        """Lower Cholesky factor and log-determinant of the full stacked covariance.

        Cached when :attr:`is_constant` (the old ``FixedCovarianceLikelihood`` fast
        path).

        Returns
        -------
        (np.ndarray, float)
            ``(L, logdet)`` with ``L`` lower-triangular and
            ``logdet = log det Sigma``.
        """
        if self.is_constant and self._chol_cache is not None:
            return self._chol_cache
        L, logdet = chol_logdet(self.matrix(ctx, *theta))
        result = (L, logdet)
        if self.is_constant:
            L.setflags(write=False)
            self._chol_cache = result
        return result

    def block_cholesky(self, ctx, *theta):
        """Per-block ``(L_i, logdet_i)`` factors, aligned with :attr:`blocks`.

        Only meaningful when :attr:`block_diagonal`; cached when
        :attr:`is_constant` so a constant block-diagonal covariance is factored
        once instead of on every likelihood evaluation.

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
        for ix in self._blocks:
            L, logdet = chol_logdet(Sigma[np.ix_(ix, ix)])
            L.setflags(write=False)
            factors.append((L, logdet))
        factors = tuple(factors)
        if self.is_constant:
            self._block_chol_cache = factors
        return factors

    def stacked_distance(self, ctx, params=()):
        r"""Squared Mahalanobis distance and log-determinant over the stacked residual.

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
        if self.block_diagonal and self._blocks is not None and len(self._blocks) > 1:
            factors = self.block_cholesky(ctx, *params)
            d2 = 0.0
            logdet = 0.0
            for ix, (L, ld) in zip(self._blocks, factors):
                z = sc.linalg.solve_triangular(L, ctx.y[ix] - ctx.ym[ix], lower=True)
                d2 += float(np.dot(z, z))
                logdet += ld
            return d2, logdet

        L, logdet = self.cholesky(ctx, *params)
        z = sc.linalg.solve_triangular(L, ctx.y - ctx.ym, lower=True)
        return float(np.dot(z, z)), logdet


# ----------------------------------------------------------------------------
# Term factory helpers (the assembly-time builders)
# ----------------------------------------------------------------------------


def _masked(magnitude, support, mask=None) -> np.ndarray:
    """Broadcast a scalar/array magnitude over ``support`` with an optional mask.

    Scalar-like values include 0-d ndarrays (e.g. ``np.array(0.05)`` as stored by
    ``exfor_tools`` distributions), not just Python scalars.
    """
    n = len(support)
    if np.ndim(magnitude) == 0:
        v = np.full(n, float(magnitude), dtype=float)
    else:
        v = np.asarray(magnitude, dtype=float)
        if v.shape != (n,):
            raise ValueError(
                f"magnitude shape {v.shape} does not match support length {n}"
            )
    if mask is not None:
        v = v * np.asarray(mask, dtype=float)
    return v


def statistical_term(support, stat_err) -> DenseTerm:
    """Always-on, genuinely uncorrelated statistical diagonal ``diag(stat_err**2)``."""
    return DenseTerm(support, np.asarray(stat_err, dtype=float) ** 2)


def offset_term(
    support, magnitude=None, parameter=None, mask=None, log=True
) -> RankOneTerm:
    """A correlated absolute-offset systematic ``outer(omega, omega)`` on ``support``.

    With ``magnitude`` it is a fixed (data-given) rank-one mode; with ``parameter``
    it is a free nuisance magnitude (``c = exp(theta)`` when ``log``).
    """
    if magnitude is None and parameter is None:
        raise ValueError("offset_term requires a magnitude and/or a parameter")
    basis = _masked(1.0 if magnitude is None else magnitude, support, mask)
    return RankOneTerm(support, basis=basis, parameter=parameter, log=log)


def normalization_term(
    support, magnitude=None, parameter=None, mask=None, log=True
) -> RankOneTerm:
    """A correlated normalisation systematic ``outer(eta * ym, eta * ym)``.

    With ``magnitude`` it is a fixed fractional normalisation uncertainty; with
    ``parameter`` the magnitude eta is a free nuisance (``UnknownNormalizationError``,
    ``c = exp(theta)`` when ``log``).  In both cases the mode scales with the model
    prediction ``ym`` on ``support``.
    """
    if magnitude is None and parameter is None:
        raise ValueError("normalization_term requires a magnitude and/or a parameter")
    if magnitude is None and mask is None:
        basis = ym_basis
    else:
        scale = _masked(1.0 if magnitude is None else magnitude, support, mask)

        def basis(ctx, support):
            return scale * ctx.ym[support]

    return RankOneTerm(support, basis=basis, parameter=parameter, log=log)


def noise_term(support, parameter, log=True) -> DiagonalTerm:
    """Unknown constant statistical noise ``diag(epsilon**2)`` (``UnknownNoise``).

    This term is **additive**: a :class:`~rxmc.constraint.Constraint` already adds
    each observation's reported statistical diagonal, so the assembled covariance
    is ``diag(y_stat_err**2 + epsilon**2)``.  To make the inferred noise *replace*
    the reported statistics (the old ``UnknownNoise`` semantics), build the
    ``Observation`` with zero ``y_stat_err`` or pass ``include_statistical_term=False``
    to the ``Constraint``.
    """
    return DiagonalTerm(support, basis=ones_basis, parameter=parameter, log=log)


def noise_fraction_term(support, parameter, log=True) -> DiagonalTerm:
    """Unknown fractional noise ``diag((epsilon * ym)**2)`` (``UnknownNoiseFraction``).

    **Additive** on top of the reported statistical diagonal (see
    :func:`noise_term` for how to get replace-semantics instead).
    """
    return DiagonalTerm(support, basis=ym_basis, parameter=parameter, log=log)


def model_error_term(support, parameter, averaging=True, log=True) -> DiagonalTerm:
    """Unknown uncorrelated model error ``diag((gamma * z)**2)`` (``UnknownModelError``).

    ``z = 0.5 * (y + ym)`` when ``averaging`` (stabilises when ``ym`` is near zero),
    else ``z = ym``.
    """
    basis = averaging_basis if averaging else ym_basis
    return DiagonalTerm(support, basis=basis, parameter=parameter, log=log)


def discrepancy_term(support, kernel, jitter=1e-10, prefix="discrepancy") -> KernelTerm:
    """A Gaussian-process discrepancy ``K(x, x; theta)`` over ``support``."""
    return KernelTerm(support, kernel, jitter=jitter, prefix=prefix)
