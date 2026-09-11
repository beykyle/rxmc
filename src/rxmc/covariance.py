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
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla

from .terms import Term

__all__ = ["StructuredCovariance", "chol_logdet"]


class _SingularCovariance(ValueError):
    """A parametric covariance that is singular at the requested ``theta``.

    The compiled constraint reads it as zero density (``log_likelihood`` is
    ``-inf``) so a sampler stepping onto such a point moves on.
    """


def chol_logdet(Sigma):
    """Lower Cholesky factor and log-determinant of a positive-definite matrix."""
    L = sla.cholesky(np.asarray(Sigma, dtype=float), lower=True)
    return L, 2.0 * float(np.sum(np.log(np.diag(L))))


class _Entry:
    """One term placed on rows of the stack, with its gather into theta."""

    __slots__ = (
        "term",
        "rows",
        "gather",
        "pos",
        "keep",
        "block",
        "segments",
        "labels",
    )

    def __init__(self, term, rows, gather, active, offsets, labels):
        self.term = term
        self.rows = np.asarray(rows, dtype=int)
        self.gather = np.asarray(gather, dtype=int)
        # positions of this term's rows inside the active stack (-1: inactive)
        lookup = np.full(int(offsets[-1].stop) if offsets else 0, -1, dtype=int)
        lookup[active] = np.arange(len(active))
        self.pos = lookup[self.rows]
        self.keep = self.pos >= 0
        # the block a matrix term lives in, or None if it crosses blocks
        blocks = {
            i
            for i, o in enumerate(offsets)
            if np.any((self.rows >= o.start) & (self.rows < o.stop))
        }
        self.block = blocks.pop() if len(blocks) == 1 else None
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
        self.entries = []
        for term, rows, gather in entries:
            if not isinstance(term, Term):
                raise TypeError(f"entries must hold Term objects, got {term!r}")
            e = _Entry(term, rows, gather, self.active, self.offsets, self.labels)
            if np.any(e.keep):
                self.entries.append(e)
        self.n_active = int(self.active.size)
        # active rows of each block, as positions in the active stack
        self.block_pos = []
        for o in self.offsets:
            inside = (self.active >= o.start) & (self.active < o.stop)
            self.block_pos.append(np.flatnonzero(inside))
        self.dense = any(
            e.term.kind == "matrix" and e.block is None for e in self.entries
        )
        self.constant_entries = [e for e in self.entries if e.term.is_constant]
        self.parametric_entries = [e for e in self.entries if not e.term.is_constant]
        self.is_constant = not self.parametric_entries
        self._D0, self._U0, self._M0, self._X0 = self._pieces(
            self.constant_entries, None, None
        )
        self._cache = None
        if self.is_constant:
            self._factor(self._D0, self._U0, self._M0, self._X0)
        elif not self.dense and all(
            e.term.is_constant for e in self.entries if e.term.kind != "mode"
        ):
            # modes never enter B, so B is constant even with parametric modes:
            # fail now, by label, rather than at the first likelihood call
            self._check_blocks(self._D0, self._M0)

    # -- assembly -------------------------------------------------------------

    def _pieces(self, entries, ym, theta):
        """``(D, U_columns, M_blocks, cross_blocks)`` from a set of entries."""
        n = self.n_active
        D = np.zeros(n)
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
            pos, keep = e.pos[e.keep], e.keep
            if t.kind == "diag":
                D[pos] += v[keep] ** 2
            elif t.kind == "mode":
                col = np.zeros(n)
                col[pos] = v[keep]
                U.append(col)
            else:
                sub = v[np.ix_(keep, keep)]
                if e.block is None:
                    X.append((pos, sub))
                else:
                    b = e.block
                    if M[b] is None:
                        M[b] = np.zeros(
                            (len(self.block_pos[b]), len(self.block_pos[b]))
                        )
                    # positions of this term's rows within the block's active rows
                    local = np.searchsorted(self.block_pos[b], pos)
                    M[b][np.ix_(local, local)] += sub
        return D, U, M, X

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
        Sigma = np.diag(D)
        for b, Mb in enumerate(M):
            if Mb is not None:
                Sigma[np.ix_(self.block_pos[b], self.block_pos[b])] += Mb
        for pos, sub in X:
            Sigma[np.ix_(pos, pos)] += sub
        for col in U:
            Sigma += np.outer(col, col)
        return Sigma

    # -- factorisation ---------------------------------------------------------

    def _factor(self, D, U, M, X):
        """Factor the covariance; cached when constant."""
        if self.is_constant and self._cache is not None:
            return self._cache
        try:
            if self.dense:
                L, logdet = chol_logdet(self._dense_matrix(D, U, M, X))
                factors = ("dense", L, logdet)
            else:
                blocks = []
                logdet = 0.0
                Ws = []
                for b, pos in enumerate(self.block_pos):
                    if pos.size == 0:
                        blocks.append(None)
                        continue
                    B = np.diag(D[pos])
                    if M[b] is not None:
                        B = B + M[b]
                    L, ld = chol_logdet(B)
                    logdet += ld
                    W = (
                        sla.solve_triangular(
                            L, np.column_stack([c[pos] for c in U]), lower=True
                        )
                        if U
                        else np.zeros((pos.size, 0))
                    )
                    blocks.append((pos, L, W))
                    Ws.append(W)
                if U:
                    W = np.vstack(Ws) if Ws else np.zeros((0, len(U)))
                    Ls, lds = chol_logdet(np.eye(len(U)) + W.T @ W)
                    logdet += lds
                else:
                    Ls = None
                factors = ("structured", blocks, Ls, logdet)
        except np.linalg.LinAlgError as err:
            error = ValueError if self.is_constant else _SingularCovariance
            raise error(self._singular_message(D)) from err
        if self.is_constant:
            self._cache = factors
        return factors

    def _check_blocks(self, D, M):
        try:
            for b, pos in enumerate(self.block_pos):
                if pos.size:
                    B = np.diag(D[pos]) + (0.0 if M[b] is None else M[b])
                    sla.cholesky(B, lower=True)
        except np.linalg.LinAlgError as err:
            raise ValueError(self._singular_message(D)) from err

    def _singular_message(self, D):
        offenders = [
            self.labels[b]
            for b, pos in enumerate(self.block_pos)
            if np.any(D[pos] == 0.0)
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
        parameter vector the entries' gathers index into.
        """
        theta = np.asarray(theta, dtype=float)
        r = (self.y - np.asarray(ym, dtype=float))[self.active]
        kind, *rest = self._factor(*self._assemble(ym, theta))
        if kind == "dense":
            L, logdet = rest
            z = sla.solve_triangular(L, r, lower=True)
            return float(z @ z), float(logdet)
        blocks, Ls, logdet = rest
        d2 = 0.0
        w = None
        for blk in blocks:
            if blk is None:
                continue
            pos, L, W = blk
            z = sla.solve_triangular(L, r[pos], lower=True)
            d2 += float(z @ z)
            w = W.T @ z if w is None else w + W.T @ z
        if Ls is not None and w is not None:
            s = sla.solve_triangular(Ls, w, lower=True)
            d2 -= float(s @ s)
        return d2, float(logdet)

    def matrix(self, ym, theta=()):
        """The dense covariance on the active rows, for display and tests."""
        theta = np.asarray(theta, dtype=float)
        return self._dense_matrix(*self._assemble(ym, theta))
