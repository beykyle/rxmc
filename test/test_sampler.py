import unittest

import numpy as np
import scipy.stats

from rxmc.adaptive_metropolis import adaptive_metropolis
from rxmc.constraint import Constraint
from rxmc.evidence import Evidence
from rxmc.likelihood_model import ParametricLikelihoodModel
from rxmc.observation import Observation
from rxmc.param_sampling import AdaptiveMetropolisSampler, MetropolisHastingsSampler
from rxmc.params import Parameter
from rxmc.physical_model import Polynomial
from rxmc.proposal import NormalProposalDistribution
from rxmc.walker import Walker


class TwoParameterLikelihood(ParametricLikelihoodModel):
    def __init__(self):
        super().__init__(
            [
                Parameter("log noise floor", float),
                Parameter("log noise slope", float),
            ]
        )

    def covariance(
        self,
        observation: Observation,
        ym: np.ndarray,
        log_noise_floor: float,
        log_noise_slope: float,
    ) -> np.ndarray:
        sigma = np.exp(log_noise_floor) + np.exp(log_noise_slope) * np.abs(ym)
        return np.diag(sigma**2)


class TestAdaptiveMetropolisSampler(unittest.TestCase):
    def test_sampling_runs_with_bounds(self):
        sampler = AdaptiveMetropolisSampler(
            params=[Parameter("x", bounds=(-1.0, 1.0))],
            prior=scipy.stats.norm(loc=0.0, scale=1.0),
            starting_location=np.array([0.0]),
            adapt_start=2,
            window_size=3,
        )

        sampler.sample(
            n_steps=5,
            starting_location=np.array([0.0]),
            rng=np.random.default_rng(123),
            log_posterior=lambda x: -0.5 * x[0] ** 2,
        )

        self.assertEqual(sampler.chain.shape, (5, 1))
        self.assertTrue(np.all(sampler.chain[:, 0] >= -1.0))
        self.assertTrue(np.all(sampler.chain[:, 0] <= 1.0))

    def test_multidimensional_regularization_stays_symmetric(self):
        x0 = np.array([0.5, -0.25])
        bounds = np.array([[-10.0, 10.0], [-10.0, 10.0]])
        previous_chain = np.array(
            [
                [0.2, -0.1],
                [0.3, -0.2],
                [0.4, -0.15],
            ]
        )
        epsilon_fraction = 1e-6
        history_mean = np.mean(previous_chain, axis=0)
        empirical_cov = np.atleast_2d(np.cov(previous_chain.T))
        proposal_cov = empirical_cov + epsilon_fraction * np.diag(history_mean**2)
        self.assertTrue(np.allclose(proposal_cov, proposal_cov.T))

        chain, logp_chain, accepted = adaptive_metropolis(
            x0=x0,
            bounds=bounds,
            n_steps=4,
            log_posterior=lambda x: -0.5 * np.dot(x, x),
            rng=np.random.default_rng(42),
            adapt_start=0,
            window_size=3,
            epsilon_fraction=epsilon_fraction,
            previous_chain=previous_chain,
        )

        self.assertEqual(chain.shape, (4, 2))
        self.assertEqual(logp_chain.shape, (4,))
        self.assertGreaterEqual(accepted, 0)


class TestWalker(unittest.TestCase):
    def test_walk_runs_end_to_end_with_multi_parameter_likelihood(self):
        model = Polynomial(1)
        observation = Observation(
            x=np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
            y=np.array([1.0, 2.1, 3.2, 4.0, 5.1]),
            y_stat_err=np.array([0.1, 0.1, 0.1, 0.1, 0.1]),
        )
        likelihood = TwoParameterLikelihood()
        evidence = Evidence(
            parametric_constraints=[
                Constraint(
                    observations=[observation],
                    physical_model=model,
                    likelihood_model=likelihood,
                )
            ]
        )

        model_sampler = MetropolisHastingsSampler(
            params=model.params,
            prior=scipy.stats.multivariate_normal(mean=[0.0, 1.0], cov=np.eye(2)),
            starting_location=np.array([0.9, 1.0]),
            proposal=NormalProposalDistribution(0.01 * np.eye(2)),
        )
        likelihood_sampler = MetropolisHastingsSampler(
            params=likelihood.params,
            prior=scipy.stats.multivariate_normal(mean=[-2.0, -2.0], cov=np.eye(2)),
            starting_location=np.array([-2.0, -2.0]),
            proposal=NormalProposalDistribution(0.01 * np.eye(2)),
        )

        walker = Walker(
            model_sampler=model_sampler,
            evidence=evidence,
            likelihood_samplers=[likelihood_sampler],
            rng=np.random.default_rng(321),
        )
        walker.walk(n_steps=5, burnin=2, batch_size=2, verbose=False)

        self.assertEqual(walker.model_sampler.chain.shape, (5, 2))
        self.assertEqual(walker.likelihood_samplers[0].chain.shape, (5, 2))
        self.assertEqual(walker.model_sampler.state.shape, (2,))
        self.assertEqual(walker.likelihood_samplers[0].state.shape, (2,))


if __name__ == "__main__":
    unittest.main()
