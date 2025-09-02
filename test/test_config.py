import unittest
import numpy as np
import scipy.stats
from rxmc.config import ParameterConfig, CalibrationConfig
from rxmc.params import Parameter
from rxmc.evidence import Evidence
from rxmc.constraint import Constraint
from rxmc.physical_model import Polynomial
from rxmc.likelihood_model import LikelihoodModel, UnknownModelError
from rxmc.observation import Observation


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


if __name__ == "__main__":
    unittest.main()
