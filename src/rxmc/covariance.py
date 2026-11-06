"""
The structured covariance of a constraint.

The three term kinds describe a decomposition of the stacked covariance over
a constraint's comparisons (one block per comparison):

.. math::

    \\Sigma = \\mathrm{diag}(D) + \\mathrm{blockdiag}(M_b) + U U^T

``D`` collects every ``diag`` term (squared), ``M_b`` the ``matrix`` terms that
lie inside block ``b``, and each ``mode`` term is one column of ``U``, zero off
its support.  A mode spanning several blocks is a column with entries in
several blocks, and costs nothing beyond its rank.  With
``B = blockdiag(M_b + diag(D_b))`` and per-block Cholesky factors ``L_b``,

.. math::

    z = L^{-1} r, \\quad W = L^{-1} U, \\quad S = I_r + W^T W, \\\\
    d^2 = z^T z - (W^T z)^T S^{-1} (W^T z), \\quad
    \\log\\det\\Sigma = \\sum_b \\log\\det B_b + \\log\\det S

so the cost is :math:`O(\\sum_b n_b^3 + N r^2 + r^3)` instead of
:math:`O(N^3)`.  A ``matrix`` term whose support crosses blocks (a Gaussian
process over the union of two datasets) forces the dense path for that
constraint.  Masking selects rows before assembly.  Constant parts are
evaluated once; a covariance with no parametric part is factored once.

``B`` must be positive definite.  A block covered only by modes is singular in
``B`` even when ``Sigma`` is not; when every term is constant this is caught at
construction and reported with the block's label.

The layout (which rows each term writes, which block it lives in, the active
positions of every block) is fixed at construction and held as numpy index
arrays; only the values are computed per ``theta``, in ``jax.numpy``, so
:meth:`StructuredCovariance.distance` is traceable and differentiable.  A
parametric covariance that is not positive definite at some ``theta`` gives a
NaN ``logdet`` there rather than raising (:func:`chol_logdet`), with a finite
gradient; the compiled constraint reads that as zero density.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsla
import numpy as np

from .terms import Term

__all__ = ["StructuredCovariance", "chol_logdet"]


@jax.custom_jvp
def chol_logdet(Sigma):
    """Lower Cholesky factor and log-determinant of a positive-definite matrix.

    Where ``Sigma`` is not positive definite the factor is the identity and the
    log-determinant NaN, with a zero tangent: the caller reads the NaN as zero
    density, and the gradient there stays finite (a failed factor would
    otherwise send NaN back through the discarded branch).
    """
    Sigma = jnp.asarray(Sigma, dtype=float)
    L = jnp.linalg.cholesky(Sigma)
    ok = jnp.all(jnp.isfinite(L))
    L = jnp.where(ok, L, jnp.eye(Sigma.shape[-1]))
    return L, jnp.where(ok, 2.0 * jnp.sum(jnp.log(jnp.diag(L))), jnp.nan)


@chol_logdet.defjvp
def _chol_logdet_jvp(primals, tangents):
    # the standard Cholesky derivative, dL = L Phi(L^-1 dSigma L^-T), but on the
    # safe factor and a zeroed tangent where Sigma is not positive definite
    L, logdet = chol_logdet(*primals)
    dS = jnp.where(jnp.isfinite(logdet), tangents[0], 0.0)
    dS = 0.5 * (dS + dS.T)
    X = _solve_lower(L, _solve_lower(L, dS).T).T
    phi = jnp.tril(X) - 0.5 * jnp.diag(jnp.diag(X))
    return (L, logdet), (L @ phi, jnp.trace(X))


def _solve_lower(L, b):
    return jsla.solve_triangular(L, b, lower=True)


class _Entry:
    """One term placed on rows of the stack, with its gather into theta.

    Every index array here is numpy and fixed at construction: ``kept`` are
    the term's support positions that are active, ``at`` their positions in
    the active stack, and ``local`` (a matrix term inside one block) their
    positions among that block's active rows.
    """

    __slots__ = (
        "term",
        "rows",
        "gather",
        "pos",
        "keep",
        "kept",
        "at",
        "local",
        "block",
        "segments",
        "labels",
    )

    def __init__(self, term, rows, gather, active, offsets, block_pos, labels):
        self.term = term
        self.rows = np.asarray(rows, dtype=int)
        self.gather = np.asarray(gather, dtype=int)
        # positions of this term's rows inside the active stack (-1: inactive)
        lookup = np.full(int(offsets[-1].stop) if offsets else 0, -1, dtype=int)
        lookup[active] = np.arange(len(active))
        self.pos = lookup[self.rows]
        self.keep = self.pos >= 0
        self.kept = np.flatnonzero(self.keep)
        self.at = self.pos[self.keep]
        # the block a matrix term lives in, or None if it crosses blocks
        blocks = {
            i
            for i, o in enumerate(offsets)
            if np.any((self.rows >= o.start) & (self.rows < o.stop))
        }
        self.block = blocks.pop() if len(blocks) == 1 else None
        self.local = (
            np.searchsorted(block_pos[self.block], self.at)
            if term.kind == "matrix" and self.block is not None
            else None
        )
        # the term's rows are in stack order, so each block's rows are contiguous
        # within them: slices of the support per block, and the block labels
        counts = [
            int(np.count_nonzero((self.rows >= o.start) & (self.rows < o.stop)))
            for o in offsets
        ]
        stops = np.cumsum(counts)
        self.segments = tuple(
            slice(int(stop - n), int(stop)) for n, stop in zip(counts, stops) if n
        )
        self.labels = tuple(str(lab) for n, lab in zip(counts, labels) if n)


def _meta_rows(meta, rows):
    if meta is None:
        return None
    return {k: np.asarray(v)[rows] for k, v in meta.items()}


class StructuredCovariance:
    """``diag(D) + blockdiag(M_b) + U Uᵀ`` over the active rows of a constraint.

    Parameters
    ----------
    entries : sequence of (Term, rows, gather)
        ``rows`` are the stacked indices the term applies to
        (:meth:`~rxmc.constraint.Constraint.support`), ``gather`` the indices
        into the flat parameter vector giving the term's values in
        ``term.params`` order.
    x, y : array_like
        Stacked over the whole constraint, ``y`` in comparison space.
    offsets : sequence of slice
        One slice per comparison block.
    active : array_like of int
        Stacked indices of the active points, ascending.
    meta : mapping, optional
        ``{key: stacked array}`` of per-point metadata for ``TermContext.meta``.
    labels : sequence of str, optional
        One label per block, for error messages.
    """

    def __init__(self, entries, x, y, offsets, active, *, meta=None, labels=None):
        self.x = np.asarray(x)
        self.y = np.asarray(y, dtype=float)
        self.offsets = tuple(offsets)
        self.active = np.asarray(active, dtype=int)
        self.meta = meta
        self.labels = (
            list(labels)
            if labels is not None
            else [f"block {i}" for i in range(len(self.offsets))]
        )
        self.n_active = int(self.active.size)
        # active rows of each block, as positions in the active stack
        self.block_pos = []
        for o in self.offsets:
            inside = (self.active >= o.start) & (self.active < o.stop)
            self.block_pos.append(np.flatnonzero(inside))
        self.entries = []
        for term, rows, gather in entries:
            if not isinstance(term, Term):
                raise TypeError(f"entries must hold Term objects, got {term!r}")
            e = _Entry(
                term,
                rows,
                gather,
                self.active,
                self.offsets,
                self.block_pos,
                self.labels,
            )
            if np.any(e.keep):
                self.entries.append(e)
        self.dense = any(
            e.term.kind == "matrix" and e.block is None for e in self.entries
        )
        self.constant_entries = [e for e in self.entries if e.term.is_constant]
        self.parametric_entries = [e for e in self.entries if not e.term.is_constant]
        self.is_constant = not self.parametric_entries
        # theta must reach every gather; checked on its (static) shape, since a
        # jax gather out of range clamps instead of raising
        self.n_theta = max(
            (int(e.gather.max()) + 1 for e in self.entries if e.gather.size),
            default=0,
        )
        self._D0, self._U0, self._M0, self._X0 = self._pieces(
            self.constant_entries, None, None
        )
        self._cache = None
        # a singular constant covariance fails now, by label, rather than as a
        # zero density at every theta.  Modes never enter B, so on the
        # structured path every block no parametric diag or matrix term
        # touches has a constant B, checked on its own
        if self.is_constant and self.dense:
            if not _is_pd(self._dense_matrix(self._D0, self._U0, self._M0, self._X0)):
                raise ValueError(self._singular_message(self._D0))
        elif not self.dense:
            self._check_blocks(self._D0, self._M0, self._constant_blocks())
        if self.is_constant:
            self._cache = self._factor(self._D0, self._U0, self._M0, self._X0)

    # -- assembly -------------------------------------------------------------

    def _pieces(self, entries, ym, theta):
        """``(D, U_columns, M_blocks, cross_blocks)`` from a set of entries."""
        n = self.n_active
        D = jnp.zeros(n)
        U = []
        M = [None] * len(self.offsets)
        X = []  # (positions, matrix) for cross-block matrix terms (dense path)
        for e in entries:
            t = e.term
            values = () if theta is None else tuple(theta[e.gather])
            v = t.value(
                self.x[e.rows],
                self.y[e.rows],
                None if ym is None else ym[e.rows],
                *values,
                meta=_meta_rows(self.meta, e.rows),
                segments=e.segments,
                labels=e.labels,
            )
            if t.kind == "diag":
                D = D.at[e.at].add(v[e.kept] ** 2)
            elif t.kind == "mode":
                U.append(jnp.zeros(n).at[e.at].set(v[e.kept]))
            else:
                sub = v[np.ix_(e.kept, e.kept)]
                if e.block is None:
                    X.append((e.at, sub))
                else:
                    b = e.block
                    if M[b] is None:
                        nb = len(self.block_pos[b])
                        M[b] = jnp.zeros((nb, nb))
                    M[b] = M[b].at[np.ix_(e.local, e.local)].add(sub)
        return D, U, M, X

    def _check_theta(self, theta):
        theta = jnp.asarray(theta, dtype=float)
        if theta.ndim != 1 or theta.shape[0] < self.n_theta:
            raise ValueError(
                f"theta has shape {theta.shape} but the covariance's terms index "
                f"{self.n_theta} parameter(s)"
            )
        return theta

    def _assemble(self, ym, theta):
        """Constant pieces plus the parametric ones at ``theta``."""
        if self.is_constant:
            return self._D0, self._U0, self._M0, self._X0
        D, U, M, X = self._pieces(self.parametric_entries, ym, theta)
        D = D + self._D0
        U = list(self._U0) + U
        M = [
            a if b is None else (b if a is None else a + b) for a, b in zip(self._M0, M)
        ]
        return D, U, M, list(self._X0) + X

    def _dense_matrix(self, D, U, M, X):
        Sigma = jnp.diag(D)
        for b, Mb in enumerate(M):
            if Mb is not None:
                pos = self.block_pos[b]
                Sigma = Sigma.at[np.ix_(pos, pos)].add(Mb)
        for pos, sub in X:
            Sigma = Sigma.at[np.ix_(pos, pos)].add(sub)
        if U:
            Uc = jnp.column_stack(U)
            Sigma = Sigma + Uc @ Uc.T
        return Sigma

    # -- factorisation ---------------------------------------------------------

    def _factor(self, D, U, M, X):
        """Factor the covariance; the cached factors when constant."""
        if self._cache is not None:
            return self._cache
        if self.dense:
            L, logdet = chol_logdet(self._dense_matrix(D, U, M, X))
            return ("dense", L, logdet)
        blocks = []
        logdet = 0.0
        Ws = []
        Uc = jnp.column_stack(U) if U else None
        for b, pos in enumerate(self.block_pos):
            if pos.size == 0:
                blocks.append(None)
                continue
            B = jnp.diag(D[pos])
            if M[b] is not None:
                B = B + M[b]
            L, ld = chol_logdet(B)
            logdet = logdet + ld
            W = _solve_lower(L, Uc[pos]) if U else jnp.zeros((pos.size, 0))
            blocks.append((pos, L, W))
            Ws.append(W)
        if U:
            W = jnp.vstack(Ws) if Ws else jnp.zeros((0, len(U)))
            Ls, lds = chol_logdet(jnp.eye(len(U)) + W.T @ W)
            logdet = logdet + lds
        else:
            Ls = None
        return ("structured", blocks, Ls, logdet)

    def _constant_blocks(self) -> list[int]:
        """The blocks whose ``B`` no parametric diag or matrix entry touches."""
        varying = set()
        for e in self.parametric_entries:
            if e.term.kind == "mode":
                continue
            varying.update(
                b for b, bp in enumerate(self.block_pos) if np.isin(e.at, bp).any()
            )
        return [b for b in range(len(self.block_pos)) if b not in varying]

    def _check_blocks(self, D, M, blocks):
        for b in blocks:
            pos = self.block_pos[b]
            if pos.size:
                B = jnp.diag(D[pos]) + (0.0 if M[b] is None else M[b])
                if not _is_pd(B):
                    raise ValueError(self._singular_message(D, blocks))

    def _singular_message(self, D, blocks=None):
        D = np.asarray(D)
        blocks = range(len(self.block_pos)) if blocks is None else blocks
        offenders = [
            self.labels[b] for b in blocks if np.any(D[self.block_pos[b]] == 0.0)
        ]
        msg = "the constraint's covariance is singular on its active points"
        if not offenders:
            return msg + (
                " although its diagonal is nonzero: a matrix term dominates it "
                "(e.g. a kernel amplitude large against the diagonal)"
            )
        return msg + (
            f"; the diagonal is zero on rows of {offenders}: those comparisons "
            "have zero statistical error and no diagonal term covers their "
            "points (a block covered only by correlated modes is singular here "
            "even when the full covariance is not).  Remedies: "
            "comparison.reported_terms(), a noise term, a fixed Term covering "
            "those points, or statistical=False with an explicit covariance."
        )

    # -- public --------------------------------------------------------------

    def distance(self, ym, theta=()):
        r"""``(d2, logdet)`` of the residual ``y - ym`` on the active rows.

        ``ym`` is the full-length stacked prediction; ``theta`` the flat
        parameter vector the entries' gathers index into.  Returns two 0-d
        jax arrays; ``logdet`` is NaN where a parametric covariance is
        singular.
        """
        theta = self._check_theta(theta)
        r = (self.y - ym)[self.active]
        kind, *rest = self._factor(*self._assemble(ym, theta))
        if kind == "dense":
            L, logdet = rest
            z = _solve_lower(L, r)
            return z @ z, logdet
        blocks, Ls, logdet = rest
        d2 = 0.0
        w = None
        for blk in blocks:
            if blk is None:
                continue
            pos, L, W = blk
            z = _solve_lower(L, r[pos])
            d2 = d2 + z @ z
            w = W.T @ z if w is None else w + W.T @ z
        if Ls is not None and w is not None:
            s = _solve_lower(Ls, w)
            d2 = d2 - s @ s
        return jnp.asarray(d2), jnp.asarray(logdet)

    def matrix(self, ym, theta=(), entries=None):
        """The dense covariance on the active rows, for display and tests.

        ``entries`` restricts the sum to a subset of :attr:`entries` (the
        objects themselves, compared by identity): the covariance of *part* of
        the error model, as a predictive draw of the model plus its discrepancy
        but not the experimental errors needs.  Default: every entry.  A subset
        is assembled directly, without the constant-piece cache.
        """
        theta = self._check_theta(theta)
        if entries is None:
            return self._dense_matrix(*self._assemble(ym, theta))
        return self._dense_matrix(*self._pieces(entries, ym, theta))


def _is_pd(B) -> bool:
    """Whether a concrete matrix is positive definite (has a Cholesky factor)."""
    try:
        np.linalg.cholesky(np.asarray(B))
    except np.linalg.LinAlgError:
        return False
    return True
