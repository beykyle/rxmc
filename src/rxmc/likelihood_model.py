"""
Likelihoods over the stacked residual of a :class:`~rxmc.constraint.Constraint`.

A constraint owns one multivariate distribution over the stacked vector of all its
observations; its covariance is a :class:`~rxmc.covariance.ConstraintCovariance`
assembled from :class:`~rxmc.covariance.Term` s.  A *likelihood* here is a thin
functional of the pre-computed Mahalanobis statistics ``(d2, logdet, n)`` plus its
own optional parameters:

* :class:`GaussianLikelihood` — the multivariate normal (parameter-free).
* :class:`StudentT` — a heavy-tailed variant carrying a degrees-of-freedom
  parameter ``nu``.
* :class:`Chi2` — drops the log-determinant normalisation (pure chi-squared).

All covariance parameters live on the :class:`~rxmc.covariance.ConstraintCovariance`;
the only likelihood-side parameter is ``StudentT``'s ``nu``.  The ``(d2, logdet)``
statistics themselves are computed by
:meth:`~rxmc.covariance.ConstraintCovariance.stacked_distance`.

Helper functions
----------------
:func:`mahalanobis_distance_sqr_cholesky`
    Squared Mahalanobis distance and log-determinant via Cholesky decomposition.
:func:`log_likelihood`
    Multivariate-normal log likelihood from pre-computed distance and log-det.
"""

import numpy as np
import scipy as sc
from scipy.special import gammaln

from .covariance import chol_logdet
from .params import Parameter

__all__ = [
    "Likelihood",
    "GaussianLikelihood",
    "StudentT",
    "Chi2",
    "mahalanobis_distance_sqr_cholesky",
    "log_likelihood",
]


class Likelihood:
    """A functional of the pre-computed Mahalanobis statistics ``(d2, logdet, n)``.

    Subclasses implement :meth:`log_likelihood` and declare any parameters via
    ``params``/``n_params``.  The chi-squared statistic is
    likelihood-independent (always the Mahalanobis distance).
    """

    params: tuple = ()
    n_params: int = 0

    def log_likelihood(self, d2, logdet, n, *like_params):
        raise NotImplementedError

    def chi2(self, d2, logdet, n, *like_params):
        return d2


class GaussianLikelihood(Likelihood):
    """Multivariate-normal likelihood over the stacked residual.

    Parameter-free — all uncertainty lives on the covariance terms.
    """

    def log_likelihood(self, d2, logdet, n, *like_params):
        return log_likelihood(d2, logdet, n)


class StudentT(Likelihood):
    r"""Multivariate Student-t likelihood with a degrees-of-freedom parameter.

    .. math::

        \log p = \ln\Gamma\!\Big(\tfrac{n+\nu}{2}\Big) - \ln\Gamma\!\Big(\tfrac{\nu}{2}\Big)
        - \tfrac{n}{2}\ln(\pi\nu) - \tfrac12 \ln\det\Sigma
        - \tfrac{\nu+n}{2}\,\ln\!\Big(1 + \tfrac{d^2}{\nu}\Big)
    """

    def __init__(self, nu_parameter: Parameter = None):
        self.nu_parameter = (
            nu_parameter
            if nu_parameter is not None
            else Parameter("degrees_of_freedom", float, latex_name=r"\nu")
        )
        self.params = (self.nu_parameter,)
        self.n_params = 1

    def log_likelihood(self, d2, logdet, n, nu):
        return (
            gammaln((n + nu) / 2.0)
            - gammaln(nu / 2.0)
            - 0.5 * n * np.log(np.pi * nu)
            - 0.5 * logdet
            - 0.5 * (nu + n) * np.log1p(d2 / nu)
        )


class Chi2(Likelihood):
    """Generalised chi-squared functional — drops the log-det normalisation."""

    def log_likelihood(self, d2, logdet, n, *like_params):
        return -0.5 * d2


# ----------------------------------------------------------------------------
# Math helpers
# ----------------------------------------------------------------------------


def mahalanobis_distance_sqr_cholesky(y, ym, cov):
    r"""Squared Mahalanobis distance and log-determinant via Cholesky factorisation.

    Parameters
    ----------
    y : array-like, shape (n,)
        Observation vector.
    ym : array-like, shape (n,)
        Model prediction vector.
    cov : array-like, shape (n, n)
        Positive-definite covariance matrix.

    Returns
    -------
    mahalanobis_sqr : float
        $(y - y_m)^T \Sigma^{-1} (y - y_m)$.
    log_det : float
        $\log \det \Sigma$.
    """
    L, log_det = chol_logdet(np.asarray(cov, dtype=float))
    z = sc.linalg.solve_triangular(L, np.asarray(y) - np.asarray(ym), lower=True)
    return np.dot(z, z), log_det


def log_likelihood(mahalanobis_sqr: float, log_det: float, n: int):
    r"""Multivariate-normal log likelihood from pre-computed statistics.

    Parameters
    ----------
    mahalanobis_sqr : float
        Squared Mahalanobis distance $(y - y_m)^T \Sigma^{-1} (y - y_m)$.
    log_det : float
        $\log \det \Sigma$.
    n : int
        Number of data points.

    Returns
    -------
    float
        Log likelihood value.
    """
    return -0.5 * (mahalanobis_sqr + log_det + n * np.log(2 * np.pi))
