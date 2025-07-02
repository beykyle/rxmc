from scipy import stats


class ProposalDistribution:
    def __init__(self, rng):
        self.rng = rng

    def __call__(self, x):
        raise NotImplementedError("This method should be overridden by subclasses")


class NormalProposalDistribution(ProposalDistribution):
    def __init__(self, cov, rng):
        self.cov = cov
        super().__init__(rng)

    def __call__(self, x):
        return stats.multivariate_normal.rvs(mean=x, cov=self.cov, random_state=self.rng)
