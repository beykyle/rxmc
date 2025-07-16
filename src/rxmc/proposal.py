from scipy import stats


class ProposalDistribution:
    """
    Base class for proposal distributions used in Markov Chain Monte Carlo (MCMC) sampling.
    This class defines the interface for proposal distributions, which generate new samples
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
    This is commonly used in MCMC methods like Metropolis-Hastings.
    Parameters:
        cov (numpy.ndarray): Covariance matrix of the proposal distribution.
    """

    def __init__(self, cov):
        self.cov = cov

    def __call__(self, x, rng):
        return stats.multivariate_normal.rvs(mean=x, cov=self.cov, random_state=rng)
