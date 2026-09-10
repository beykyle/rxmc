"""Shared helpers for the test suite: dense references built by hand."""

import numpy as np


def mahalanobis(y, ym, cov):
    """``(d2, logdet)`` of a dense covariance by a numpy Cholesky factorisation."""
    L = np.linalg.cholesky(np.asarray(cov, dtype=float))
    z = np.linalg.solve(L, np.asarray(y, dtype=float) - np.asarray(ym, dtype=float))
    return float(z @ z), float(2.0 * np.sum(np.log(np.diag(L))))


def manual_mvn_loglike(y, ym, cov):
    """Reference dense multivariate-normal log likelihood."""
    d2, logdet = mahalanobis(y, ym, cov)
    return -0.5 * (d2 + logdet + len(y) * np.log(2 * np.pi))


def assemble_dense(terms, x, y, ym, values=()):
    """Reference assembly of whole-support terms into a dense covariance.

    ``values`` is one tuple of sampled values per term, in ``terms`` order
    (an empty tuple for a term without parameters).  This is the dense
    reference the structured covariance is checked against.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    values = tuple(values) if values else tuple(() for _ in terms)
    if len(values) != len(terms):
        raise ValueError("one value tuple per term")
    Sigma = np.zeros((n, n))
    for term, v in zip(terms, values):
        out = term.value(x, y, ym, *v)
        if term.kind == "diag":
            Sigma[np.diag_indices(n)] += out**2
        elif term.kind == "mode":
            Sigma += np.outer(out, out)
        else:
            Sigma += out
    return Sigma
