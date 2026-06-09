"""
Built-in prior distribution classes for use with ``ParameterConfig``.

These classes satisfy the generic prior protocol expected by ``ParameterConfig``
and ``CalibrationConfig``: they expose ``logpdf(x)`` and ``rvs(n)`` methods
with consistent array conventions.  Any user-defined class that provides the
same two methods can be passed directly to ``ParameterConfig``.

Available classes
-----------------
:class:`IndependentPrior`
    Wraps an arbitrary list of ``scipy.stats`` frozen distributions.  The
    preferred way to combine per-parameter marginals into a joint prior.

:class:`TruncatedNormalPrior`
    Convenience class for the common case of independent truncated normals.

Dynesty / nested-sampling support
----------------------------------
When using a nested sampler (e.g. Dynesty) the sampler requires a
*prior transform* that maps unit-cube coordinates to physical parameters.
Prior classes that support this should implement::

    def prior_transform(self, u: ndarray) -> ndarray

where ``u`` has shape ``(ndim,)`` with each element in ``[0, 1)`` and the
return value is the corresponding physical parameter vector.  Both
:class:`IndependentPrior` and :class:`TruncatedNormalPrior` provide this
method via the ``ppf`` of each marginal distribution.
"""

import numpy as np
from scipy import stats


class TruncatedNormalPrior:
    """Independent truncated-normal prior for a vector of parameters.

    Each component ``j`` follows::

        theta_j ~ Normal(mu_j, sigma_j),  truncated to [lower_j, upper_j]

    The components are treated as independent, so the joint log-density is the
    sum of the marginal log-densities.

    Parameters
    ----------
    mu : array-like, shape (ndim,)
        Mean of the untruncated normal for each component.
    sigma : array-like, shape (ndim,)
        Standard deviation of the untruncated normal for each component.
    lower : array-like, shape (ndim,)
        Lower truncation bound for each component.
    upper : array-like, shape (ndim,)
        Upper truncation bound for each component.
    seed : int, optional
        Seed for the internal random-number generator used in :meth:`rvs`.
        Default is ``123``.
    """

    def __init__(self, mu, sigma, lower, upper, seed=123):
        self.mu = np.asarray(mu, dtype=float)
        self.sigma = np.asarray(sigma, dtype=float)
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        self.dim = len(self.mu)
        self.rng = np.random.default_rng(seed)

        self.a = (self.lower - self.mu) / self.sigma
        self.b = (self.upper - self.mu) / self.sigma

    def logpdf(self, theta):
        """Evaluate the joint log prior density.

        Parameters
        ----------
        theta : array-like
            Parameter vector of shape ``(ndim,)`` for a single evaluation, or
            ``(n, ndim)`` for a batch of ``n`` vectors.

        Returns
        -------
        float or ndarray
            Scalar log-density when ``theta`` has shape ``(ndim,)``;
            array of shape ``(n,)`` when ``theta`` has shape ``(n, ndim)``.

        Raises
        ------
        ValueError
            If the trailing dimension of ``theta`` does not equal ``self.dim``.
        """
        theta = np.asarray(theta, dtype=float)
        squeeze = theta.ndim == 1
        theta = np.atleast_2d(theta)

        if theta.shape[1] != self.dim:
            raise ValueError(
                f"Expected theta with {self.dim} columns, got {theta.shape[1]}"
            )

        logp = np.zeros(theta.shape[0])
        for j in range(self.dim):
            logp += stats.truncnorm.logpdf(
                theta[:, j],
                a=self.a[j],
                b=self.b[j],
                loc=self.mu[j],
                scale=self.sigma[j],
            )

        return float(logp[0]) if squeeze else logp

    def rvs(self, n):
        """Draw ``n`` independent samples from the prior.

        Parameters
        ----------
        n : int
            Number of samples to draw.

        Returns
        -------
        ndarray, shape (n, ndim)
            Matrix of samples, one row per draw.
        """
        out = np.zeros((n, self.dim))
        for j in range(self.dim):
            out[:, j] = stats.truncnorm.rvs(
                a=self.a[j],
                b=self.b[j],
                loc=self.mu[j],
                scale=self.sigma[j],
                size=n,
                random_state=self.rng,
            )
        return out

    def prior_transform(self, u):
        """Map unit-cube coordinates to physical parameters (Dynesty interface).

        Parameters
        ----------
        u : array-like, shape (ndim,)
            Unit-cube coordinates, each in ``[0, 1)``.

        Returns
        -------
        ndarray, shape (ndim,)
            Physical parameter vector obtained by applying the component-wise
            percent-point function of each truncated normal.
        """
        u = np.asarray(u, dtype=float)
        theta = np.empty_like(u)
        for j in range(self.dim):
            theta[j] = stats.truncnorm.ppf(
                u[j],
                a=self.a[j],
                b=self.b[j],
                loc=self.mu[j],
                scale=self.sigma[j],
            )
        return theta


class IndependentPrior:
    """Independent prior built from an arbitrary list of ``scipy.stats`` distributions.

    Each parameter component ``j`` is modelled by the corresponding frozen
    distribution ``distributions[j]``.  The components are treated as
    independent, so the joint log-density is the sum of the marginal
    log-densities, and sampling draws from each marginal separately.

    This class satisfies the generic prior protocol required by
    :class:`~rxmc.config.ParameterConfig`: it exposes ``logpdf``, ``rvs``,
    and ``prior_transform``, and carries a ``dim`` attribute used for
    automatic dimension validation.

    Parameters
    ----------
    distributions : list of frozen scipy.stats distributions
        One distribution per parameter component.  Any frozen univariate
        ``scipy.stats`` distribution works (e.g. ``stats.norm(0, 1)``,
        ``stats.uniform(0, 5)``, ``stats.truncnorm(...)``).
    seed : int, optional
        Seed for the internal :class:`numpy.random.Generator` used in
        :meth:`rvs` to ensure reproducibility.  Default is ``123``.

    Examples
    --------
    >>> from scipy import stats
    >>> from rxmc.priors import IndependentPrior
    >>> prior = IndependentPrior([stats.norm(0, 1), stats.uniform(0, 5)])
    >>> prior.dim
    2
    >>> prior.rvs(4).shape
    (4, 2)
    """

    def __init__(self, distributions: list, seed: int = 123):
        self.distributions = distributions
        self.dim = len(distributions)
        self.rng = np.random.default_rng(seed)

    def logpdf(self, theta):
        """Evaluate the joint log prior density.

        Parameters
        ----------
        theta : array-like
            Parameter vector of shape ``(ndim,)`` for a single evaluation, or
            ``(n, ndim)`` for a batch of ``n`` vectors.

        Returns
        -------
        float or ndarray
            Scalar log-density when ``theta`` has shape ``(ndim,)``;
            array of shape ``(n,)`` when ``theta`` has shape ``(n, ndim)``.

        Raises
        ------
        ValueError
            If the trailing dimension of ``theta`` does not equal ``self.dim``.
        """
        theta = np.asarray(theta, dtype=float)
        squeeze = theta.ndim == 1
        theta = np.atleast_2d(theta)

        if theta.shape[1] != self.dim:
            raise ValueError(
                f"Expected theta with {self.dim} columns, got {theta.shape[1]}"
            )

        logp = np.zeros(theta.shape[0])
        for j, dist in enumerate(self.distributions):
            logp += dist.logpdf(theta[:, j])

        return float(logp[0]) if squeeze else logp

    def rvs(self, n: int) -> np.ndarray:
        """Draw ``n`` independent samples from the prior.

        Parameters
        ----------
        n : int
            Number of samples to draw.

        Returns
        -------
        ndarray, shape (n, ndim)
            Matrix of samples, one row per draw.
        """
        out = np.zeros((n, self.dim))
        for j, dist in enumerate(self.distributions):
            out[:, j] = dist.rvs(size=n, random_state=self.rng)
        return out

    def prior_transform(self, u) -> np.ndarray:
        """Map unit-cube coordinates to physical parameters (Dynesty interface).

        Applies the percent-point function (inverse CDF) of each marginal
        distribution component-wise.

        Parameters
        ----------
        u : array-like, shape (ndim,)
            Unit-cube coordinates, each in ``[0, 1)``.

        Returns
        -------
        ndarray, shape (ndim,)
            Physical parameter vector.
        """
        u = np.asarray(u, dtype=float)
        theta = np.empty_like(u)
        for j, dist in enumerate(self.distributions):
            theta[j] = dist.ppf(u[j])
        return theta
