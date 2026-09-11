"""
Likelihood functionals over the stacked residual of a constraint.

A constraint owns one multivariate distribution over the stacked residual
``y - ym`` of its comparisons, with covariance assembled from
:class:`~rxmc.terms.Term` s.  A *likelihood* here is a thin functional of the
pre-computed Mahalanobis statistics ``(d2, logdet, n)`` plus its own optional
parameters, which are ordinary :class:`~rxmc.params.Parameter` s:

* :class:`Gaussian` — the multivariate normal (parameter-free).
* :class:`StudentT` — a heavy-tailed variant carrying a degrees-of-freedom
  parameter ``nu``; one radial tail factor for the whole residual.
* :class:`Chi2` — drops the log-determinant normalisation (pure chi-squared).
"""

from __future__ import annotations

from math import inf

import numpy as np
from scipy import stats
from scipy.special import betaln, gammaln

from .params import Parameter

__all__ = ["Likelihood", "Gaussian", "StudentT", "Chi2", "log_likelihood"]


class Likelihood:
    """A functional of the pre-computed Mahalanobis statistics ``(d2, logdet, n)``.

    Subclasses implement :meth:`log_likelihood` and declare any parameters in
    ``params``.  The chi-squared statistic is likelihood-independent (always
    the Mahalanobis distance).
    """

    params: tuple[Parameter, ...] = ()

    def log_likelihood(self, d2, logdet, n, *values) -> float:
        raise NotImplementedError

    def chi2(self, d2, logdet, n, *values) -> float:
        return d2

    def predictive_scale(self, rng, size, *values) -> np.ndarray:
        """Per-draw multipliers of a correlated normal draw ``L z``: ones.

        A scale mixture of normals returns its mixing scales instead, so
        ``ym + scale * L z`` is a draw from the likelihood's own predictive.
        """
        return np.ones(size)


class Gaussian(Likelihood):
    """Multivariate-normal likelihood over the stacked residual (parameter-free)."""

    def log_likelihood(self, d2, logdet, n, *values) -> float:
        return log_likelihood(d2, logdet, n)


class StudentT(Likelihood):
    r"""Multivariate Student-t likelihood with a degrees-of-freedom parameter.

    .. math::

        \log p = \ln\Gamma\!\Big(\tfrac{n+\nu}{2}\Big) - \ln\Gamma\!\Big(\tfrac{\nu}{2}\Big)
        - \tfrac{n}{2}\ln(\pi\nu) - \tfrac12 \ln\det\Sigma
        - \tfrac{\nu+n}{2}\,\ln\!\Big(1 + \tfrac{d^2}{\nu}\Big)

    Parameters
    ----------
    nu : Parameter, optional
        The degrees of freedom.  Defaults to ``Parameter("nu",
        prior=gamma(a=2, scale=10), bounds=(1, inf))``: the Gamma(2, rate 0.1)
        prior of Juárez & Steel, "Model-based clustering of non-Gaussian panel
        data based on skew-t distributions", J. Bus. Econ. Stat. 28, 52 (2010),
        truncated to ``nu >= 1``.  It has most of its mass on heavy tails and a
        mean near 20, and a unit-cube map, so nested samplers take it as is.
        Two constraints using the default each derive a ``"nu"`` and the
        problem fails to compile on the duplicate name, so pass ``nu=`` to
        share one or to name them apart.
    """

    def __init__(self, nu: Parameter | None = None):
        if nu is None:
            nu = Parameter(
                "nu",
                prior=stats.gamma(a=2.0, scale=10.0),
                bounds=(1.0, inf),
                latex=r"\nu",
            )
        self.params = (nu,)

    def predictive_scale(self, rng, size, nu) -> np.ndarray:
        """``sqrt(nu / w)`` with ``w ~ chi2(nu)``: the multivariate-t mixing scale."""
        return np.sqrt(nu / rng.chisquare(nu, size))

    def log_likelihood(self, d2, logdet, n, nu) -> float:
        # lnG((n+nu)/2) - lnG(nu/2), without the cancellation at large nu
        return (
            gammaln(n / 2.0)
            - betaln(nu / 2.0, n / 2.0)
            - 0.5 * n * np.log(np.pi * nu)
            - 0.5 * logdet
            - 0.5 * (nu + n) * np.log1p(d2 / nu)
        )


class Chi2(Likelihood):
    """Generalised chi-squared functional: drops the log-det normalisation."""

    def log_likelihood(self, d2, logdet, n, *values) -> float:
        return -0.5 * d2


def log_likelihood(d2: float, logdet: float, n: int) -> float:
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
