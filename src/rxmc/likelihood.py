"""
Likelihood functionals over the stacked residual of a constraint.

A constraint owns one multivariate distribution over the stacked residual
``y - ym`` of its comparisons, with covariance assembled from
:class:`~rxmc.terms.Term` s.  A *likelihood* here is a thin functional of the
pre-computed Mahalanobis statistics ``(d2, logdet, n)``, with no parameters
of its own:

* :class:`Gaussian` — the multivariate normal.
* :class:`Chi2` — drops the log-determinant normalisation (pure chi-squared).
"""

from __future__ import annotations

import numpy as np

__all__ = ["Likelihood", "Gaussian", "Chi2", "log_likelihood"]


class Likelihood:
    """A functional of the pre-computed Mahalanobis statistics ``(d2, logdet, n)``.

    Subclasses implement :meth:`log_likelihood`.  The chi-squared statistic
    is likelihood-independent (always the Mahalanobis distance).
    """

    def log_likelihood(self, d2, logdet, n):
        raise NotImplementedError

    def chi2(self, d2, logdet, n):
        return d2


class Gaussian(Likelihood):
    """Multivariate-normal likelihood over the stacked residual."""

    def log_likelihood(self, d2, logdet, n):
        return log_likelihood(d2, logdet, n)


class Chi2(Likelihood):
    """Generalised chi-squared functional: drops the log-det normalisation."""

    def log_likelihood(self, d2, logdet, n):
        return -0.5 * d2


def log_likelihood(d2, logdet, n: int):
    r"""Multivariate-normal log likelihood from pre-computed statistics.

    Parameters
    ----------
    d2 : float
        Squared Mahalanobis distance :math:`(y - y_m)^T \Sigma^{-1} (y - y_m)`.
    logdet : float
        :math:`\log \det \Sigma`.
    n : int
        Number of data points.
    """
    return -0.5 * (d2 + logdet + n * np.log(2 * np.pi))
