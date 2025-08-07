import numpy as np
from scipy import stats


class ProposalDistribution:
    """
    Base class for proposal distributions used in the `MetropolisHastingsSampler`. This
    class defines the interface for proposal distributions, which generate new samples
    based on the current sample in the Markov chain.
    """

    def __init__(self):
        pass

    def __call__(self, x, rng):
        """
        Generate a proposed sample based on the current sample `x`.
        Parameters:
            x (numpy.ndarray): Current sample from the Markov chain.
            rng (numpy.random.Generator): Random number generator instance.
        Returns:
            numpy.ndarray: Proposed sample.
        """
        raise NotImplementedError("This method should be overridden by subclasses")


class NormalProposalDistribution(ProposalDistribution):
    """
    Proposal distribution that generates new samples from a multivariate normal distribution
    centered at the current sample with a specified covariance matrix.
    """

    def __init__(self, cov):
        """
        Initialize the proposal distribution with a covariance matrix.
        Parameters:
            cov (numpy.ndarray): Covariance matrix for the multivariate normal distribution.
        """
        self.cov = cov

    def __call__(self, x, rng):
        return stats.multivariate_normal.rvs(mean=x, cov=self.cov, random_state=rng)


class HalfNormalProposalDistribution(ProposalDistribution):
    """
    Proposal distribution that generates new samples from a half-normal distribution.
    This is commonly used to ensure that the proposed samples are non-negative.
    """

    def __init__(self, scale):
        """
        Initialize the proposal distribution with a scale parameter.
        Parameters:
            scale (float): Scale parameter for the half-normal distribution.
        """
        self.scale = scale

    def __call__(self, x, rng):
        return stats.halfnorm.rvs(loc=x, scale=self.scale, random_state=rng)


class LogspaceNormalProposalDistribution(ProposalDistribution):
    """
    Proposal distribution that generates new samples from a normal distribution
    in log space. This is useful for parameters that are strictly positive.
    """

    def __init__(self, scale):
        """
        Initialize the proposal distribution with a scale parameter.
        Parameters:
            scale (float): Scale parameter for the log-normal distribution.
        """
        self.scale = scale

    def __call__(self, x, rng):
        return np.exp(stats.norm.rvs(loc=np.log(x), scale=self.scale, random_state=rng))
