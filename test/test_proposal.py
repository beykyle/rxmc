import unittest

import numpy as np
from numpy.random import default_rng
from rxmc.proposal import (
    HalfNormalProposalDistribution,
    LogspaceNormalProposalDistribution,
    NormalProposalDistribution,
    ProposalDistribution,
)


class TestProposalDistributions(unittest.TestCase):
    def setUp(self):
        # Set up a random number generator
        self.rng = default_rng(42)
        self.current_sample = np.array([1.0, 2.0, 3.0])
        self.scale = 1.0
        self.cov = np.eye(3)

    def test_normal_proposal_distribution(self):
        normal_proposal = NormalProposalDistribution(cov=self.cov)
        proposed_sample = normal_proposal(self.current_sample, self.rng)

        # Check the output is of correct shape
        self.assertEqual(proposed_sample.shape, self.current_sample.shape)

        # Check the mean and covariance of the distribution approximately match the expected
        samples = [normal_proposal(self.current_sample, self.rng) for _ in range(10000)]
        samples_mean = np.mean(samples, axis=0)
        samples_cov = np.cov(np.array(samples).T)

        self.assertTrue(np.allclose(samples_mean, self.current_sample, atol=0.1))
        self.assertTrue(np.allclose(samples_cov, self.cov, atol=0.1))

    def test_half_normal_proposal_distribution(self):
        half_normal_proposal = HalfNormalProposalDistribution(scale=self.scale)
        proposed_sample = half_normal_proposal(self.current_sample, self.rng)

        # Check the output is of correct shape
        self.assertEqual(proposed_sample.shape, self.current_sample.shape)

        # Check that all values in proposed_sample are non-negative
        self.assertTrue(np.all(proposed_sample >= 0))

    def test_logspace_normal_proposal_distribution(self):
        logspace_normal_proposal = LogspaceNormalProposalDistribution(scale=self.scale)
        proposed_sample = logspace_normal_proposal(self.current_sample, self.rng)

        # Check the output is of correct shape
        self.assertEqual(proposed_sample.shape, self.current_sample.shape)

        # Check that all values in proposed_sample are strictly positive
        self.assertTrue(np.all(proposed_sample > 0))

    def test_proposal_distribution_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            proposal = ProposalDistribution()
            proposal(self.current_sample, self.rng)


if __name__ == "__main__":
    unittest.main()
