"""
Proposal distributions for Metropolis-Hastings MCMC.

Each class is a callable that accepts the current parameter vector and a
:class:`numpy.random.Generator` and returns a proposed parameter vector.
"""

import numpy as np
from scipy import stats


class ProposalDistribution:
    """Abstract base class for Metropolis-Hastings proposal distributions.

    Subclasses must implement :meth:`__call__`, which generates a new candidate
    parameter vector given the current state.
    """

    def __init__(self):
        pass

    def __call__(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Generate a proposed sample from the current state.

        Parameters
        ----------
        x : np.ndarray
            Current parameter vector.
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        np.ndarray
            Proposed parameter vector.

        Raises
        ------
        NotImplementedError
            Always — subclasses must implement this method.
        """
        raise NotImplementedError("This method should be overridden by subclasses")


class NormalProposalDistribution(ProposalDistribution):
    """Multivariate-normal proposal centred at the current state.

    Parameters
    ----------
    cov : np.ndarray, shape (ndim, ndim)
        Covariance matrix of the proposal distribution.
    """

    def __init__(self, cov: np.ndarray):
        self.cov = cov

    def __call__(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return stats.multivariate_normal.rvs(mean=x, cov=self.cov, random_state=rng)


class HalfNormalProposalDistribution(ProposalDistribution):
    """Half-normal proposal, useful for strictly non-negative parameters.

    Parameters
    ----------
    scale : float
        Scale parameter of the half-normal distribution.
    """

    def __init__(self, scale: float):
        self.scale = scale

    def __call__(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return stats.halfnorm.rvs(loc=x, scale=self.scale, random_state=rng)


class LogspaceNormalProposalDistribution(ProposalDistribution):
    """Normal proposal operating in log space, for strictly positive parameters.

    Proposes ``exp(log(x) + eps)`` where ``eps ~ Normal(0, scale)``, which
    preserves positivity while allowing multiplicative jumps of arbitrary size.

    Parameters
    ----------
    scale : float
        Standard deviation of the normal perturbation in log space.
    """

    def __init__(self, scale: float):
        self.scale = scale

    def __call__(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return np.exp(stats.norm.rvs(loc=np.log(x), scale=self.scale, random_state=rng))
