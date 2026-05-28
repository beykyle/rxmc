import unittest

import numpy as np
import scipy.stats

from rxmc.config import CalibrationConfig, ParameterConfig
from rxmc.constraint import Constraint
from rxmc.evidence import Evidence
from rxmc.likelihood_model import LikelihoodModel, UnknownModelError
from rxmc.observation import Observation
from rxmc.params import Parameter
from rxmc.physical_model import Polynomial


class TestParameterConfig(unittest.TestCase):
    def setUp(self):
        self.param1 = Parameter(name="param1")
        self.param2 = Parameter(name="param2")
        self.prior = scipy.stats.multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]])
        self.initial_proposal_dist = scipy.stats.multivariate_normal(
            mean=[0, 0], cov=[[1, 0], [0, 1]]
        )

    def test_initialization(self):
        """Test ParameterConfig initialization."""
        config = ParameterConfig(
            params=[self.param1, self.param2],
            prior=self.prior,
            initial_proposal_distribution=self.initial_proposal_dist,
        )
        self.assertEqual(config.ndim, 2)
        self.assertEqual(config.params, [self.param1, self.param2])

    def test_empty_parameters_raises_valueerror(self):
        """Test empty parameters list raises ValueError."""
        with self.assertRaises(ValueError):
            ParameterConfig(
                params=[],
                prior=self.prior,
                initial_proposal_distribution=self.initial_proposal_dist,
            )

    def test_single_parameter_x0_shape(self):
        config = ParameterConfig(
            params=[self.param1],
            prior=scipy.stats.multivariate_normal(mean=[0], cov=[[1]]),
            initial_proposal_distribution=scipy.stats.multivariate_normal(
                mean=[0], cov=[[1]]
            ),
        )
        x0 = config.x0(4)
        self.assertEqual(x0.shape, (4, 1))


class TestCalibrationConfig(unittest.TestCase):
    def setUp(self):
        # Evidence with one regular and one parametric constraint
        self.model = Polynomial(1)
        self.evidence = Evidence(
            constraints=[
                Constraint(
                    observations=[
                        Observation(
                            x=np.array([1.0, 2.0, 3.0]),
                            y=np.array([1.0, 2.0, 3.0]),
                            y_stat_err=np.array([0.1, 0.1, 0.1]),
                        )
                    ],
                    physical_model=self.model,
                    likelihood_model=LikelihoodModel(),
                )
            ],
            parametric_constraints=[
                Constraint(
                    observations=[
                        Observation(
                            x=np.array([6.0, 7.0, 8.0]),
                            y=np.array([6.3, 8.1, 9.6]),
                            y_stat_err=np.array([0.1, 0.1, 0.1]),
                        )
                    ],
                    physical_model=self.model,
                    likelihood_model=UnknownModelError(),
                ),
            ],
        )

        # Model Config
        model_prior = scipy.stats.multivariate_normal(
            mean=[0, 1],
            cov=[
                [
                    1,
                    0,
                ],
                [0, 1],
            ],
        )
        initial_proposal = model_prior
        self.model_config = ParameterConfig(
            params=self.model.params,
            prior=model_prior,
            initial_proposal_distribution=initial_proposal,
        )

        # Likelihood Config
        likelihood_prior = scipy.stats.multivariate_normal(mean=[0], cov=[[1]])
        initial_proposal = likelihood_prior
        self.likelihood_config = ParameterConfig(
            params=self.evidence.parametric_constraints[0].likelihood.params,
            prior=likelihood_prior,
            initial_proposal_distribution=initial_proposal,
        )

    def test_initialization(self):
        """Test CalibrationConfig initialization."""
        config = CalibrationConfig(
            evidence=self.evidence,
            model_config=self.model_config,
            likelihood_configs=[self.likelihood_config],
        )
        self.assertEqual(config.ndim, 3)

    def test_split_parameters(self):
        """Test splitting flat parameters into model and likelihood parameters."""
        config = CalibrationConfig(
            evidence=self.evidence,
            model_config=self.model_config,
            likelihood_configs=[self.likelihood_config],
        )
        x = np.array([1.0, 2.0, 0.0])
        model_params, likelihood_params = config.split_parameters(x)
        np.testing.assert_array_equal(model_params, [1.0, 2.0])
        np.testing.assert_array_equal(likelihood_params[0], [0.0])

    def test_black_box_bayes_interface(self):
        config = CalibrationConfig(
            evidence=self.evidence,
            model_config=self.model_config,
            likelihood_configs=[self.likelihood_config],
        )

        self.assertEqual(config.parameter_names, ["a0", "a1", "log fractional err"])

        x0_single = config.starting_location(1)
        self.assertEqual(x0_single.shape, (1, 3))

        x0_batch = config.starting_location(4)
        self.assertEqual(x0_batch.shape, (4, 3))

        theta = np.array([1.0, 2.0, 0.0])
        batched = config.log_posterior_batch(np.vstack([theta, theta]))
        self.assertEqual(batched.shape, (2,))
        np.testing.assert_allclose(
            batched,
            [config.log_posterior(theta), config.log_posterior(theta)],
        )

    def test_conditional_posterior_uses_parametric_constraint(self):
        config = CalibrationConfig(
            evidence=self.evidence,
            model_config=self.model_config,
            likelihood_configs=[self.likelihood_config],
        )

        xmodel = np.array([1.0, 1.0])
        ym = self.evidence.parametric_constraints[0].predict(*xmodel)
        x_lm = np.array([0.0])

        expected = self.evidence.parametric_constraints[0].marginal_log_likelihood(
            ym, *x_lm
        ) + self.likelihood_config.prior_logpdf(x_lm)
        self.assertAlmostEqual(config.conditional_posterior(x_lm, 0, ym), expected)

    def test_starting_location_with_single_parameter_sectors(self):
        model = Polynomial(0)
        likelihood = UnknownModelError()
        observation = Observation(
            x=np.array([1.0, 2.0]),
            y=np.array([1.0, 1.1]),
            y_stat_err=np.array([0.1, 0.1]),
        )
        evidence = Evidence(
            parametric_constraints=[
                Constraint(
                    observations=[observation],
                    physical_model=model,
                    likelihood_model=likelihood,
                )
            ]
        )
        model_config = ParameterConfig(
            params=model.params,
            prior=scipy.stats.multivariate_normal(mean=[0], cov=[[1]]),
            initial_proposal_distribution=scipy.stats.multivariate_normal(
                mean=[0], cov=[[1]]
            ),
        )
        likelihood_config = ParameterConfig(
            params=likelihood.params,
            prior=scipy.stats.multivariate_normal(mean=[0], cov=[[1]]),
            initial_proposal_distribution=scipy.stats.multivariate_normal(
                mean=[0], cov=[[1]]
            ),
        )
        config = CalibrationConfig(
            evidence=evidence,
            model_config=model_config,
            likelihood_configs=[likelihood_config],
        )

        x0 = config.starting_location(5)
        self.assertEqual(x0.shape, (5, 2))


if __name__ == "__main__":
    unittest.main()
