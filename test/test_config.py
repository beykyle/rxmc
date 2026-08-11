import unittest

import numpy as np
import scipy.stats

from rxmc.config import CalibrationConfig, ParameterConfig
from rxmc.constraint import Constraint
from rxmc.covariance import model_error_term
from rxmc.evidence import Evidence
from rxmc.observation import Observation
from rxmc.params import Parameter
from rxmc.physical_model import Polynomial
from rxmc.priors import TruncatedNormalPrior


def gamma_parameter():
    """The UnknownModelError gamma, as a covariance parameter."""
    return Parameter(
        "log fractional err", float, latex_name=r"\gamma", unit="dimensionless"
    )


def model_error_constraint(observation, model, gamma):
    """A constraint with an UnknownModelError (averaging) covariance term."""
    support = np.arange(observation.n_data_pts)
    return Constraint(
        observations=[observation],
        physical_model=model,
        extra_terms=[model_error_term(support, gamma, averaging=True)],
    )


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

    def test_generic_prior_class(self):
        """TruncatedNormalPrior satisfies the generic prior protocol."""
        prior = TruncatedNormalPrior(
            mu=[0.0, 1.0],
            sigma=[1.0, 1.0],
            lower=[-5.0, -5.0],
            upper=[5.0, 5.0],
        )
        config = ParameterConfig(
            params=[self.param1, self.param2],
            prior=prior,
            initial_proposal_distribution=prior,
        )
        self.assertEqual(config.ndim, 2)

        x0 = config.x0(3)
        self.assertEqual(x0.shape, (3, 2))

        lp = config.prior_logpdf(np.array([0.0, 1.0]))
        self.assertTrue(np.isfinite(lp))

    def test_prior_transform_list(self):
        """List-of-distributions prior_transform uses ppf on each element."""
        prior_list = [scipy.stats.norm(0, 1), scipy.stats.norm(0, 1)]
        config = ParameterConfig(
            params=[self.param1, self.param2],
            prior=prior_list,
            initial_proposal_distribution=prior_list,
        )
        u = np.array([0.5, 0.5])
        theta = config.prior_transform(u)
        np.testing.assert_allclose(theta, [0.0, 0.0], atol=1e-10)

    def test_prior_transform_generic(self):
        """Generic prior with prior_transform method is called directly."""
        # Use symmetric bounds so the median (u=0.5) maps exactly to mu.
        prior = TruncatedNormalPrior(
            mu=[0.0, 1.0],
            sigma=[1.0, 1.0],
            lower=[-5.0, -4.0],
            upper=[5.0, 6.0],
        )
        config = ParameterConfig(
            params=[self.param1, self.param2],
            prior=prior,
            initial_proposal_distribution=prior,
        )
        u = np.array([0.5, 0.5])
        theta = config.prior_transform(u)
        np.testing.assert_allclose(theta, [0.0, 1.0], atol=1e-6)

    def test_prior_transform_unsupported_raises(self):
        """Non-list prior without prior_transform raises NotImplementedError."""
        config = ParameterConfig(
            params=[self.param1, self.param2],
            prior=self.prior,
            initial_proposal_distribution=self.initial_proposal_dist,
        )
        with self.assertRaises(NotImplementedError):
            config.prior_transform(np.array([0.5, 0.5]))


class TestCalibrationConfig(unittest.TestCase):
    def setUp(self):
        # Evidence with one regular and one parametric constraint
        self.model = Polynomial(1)
        self.gamma = gamma_parameter()
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
                ),
                model_error_constraint(
                    Observation(
                        x=np.array([6.0, 7.0, 8.0]),
                        y=np.array([6.3, 8.1, 9.6]),
                        y_stat_err=np.array([0.1, 0.1, 0.1]),
                    ),
                    self.model,
                    self.gamma,
                ),
            ],
        )

        # Model Config
        model_prior = scipy.stats.multivariate_normal(
            mean=[0, 1],
            cov=[[1, 0], [0, 1]],
        )
        self.model_config = ParameterConfig(
            params=self.model.params,
            prior=model_prior,
            initial_proposal_distribution=model_prior,
        )

        # Likelihood Config
        likelihood_prior = scipy.stats.multivariate_normal(mean=[0], cov=[[1]])
        self.likelihood_config = ParameterConfig(
            params=list(self.evidence.parametric_constraints[0].params),
            prior=likelihood_prior,
            initial_proposal_distribution=likelihood_prior,
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

    def test_parameters_property(self):
        """parameters returns all Parameter objects in flat sampler order."""
        config = CalibrationConfig(
            evidence=self.evidence,
            model_config=self.model_config,
            likelihood_configs=[self.likelihood_config],
        )
        params = config.parameters
        self.assertEqual(len(params), 3)
        self.assertEqual(params[0].name, "a0")
        self.assertEqual(params[1].name, "a1")
        self.assertEqual(params[2].name, "log fractional err")

    def test_prior_property(self):
        """prior returns one prior object per parameter sector."""
        config = CalibrationConfig(
            evidence=self.evidence,
            model_config=self.model_config,
            likelihood_configs=[self.likelihood_config],
        )
        priors = config.prior
        self.assertEqual(len(priors), 2)
        self.assertIs(priors[0], self.model_config.prior)
        self.assertIs(priors[1], self.likelihood_config.prior)

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

    def test_parametric_indices_map(self):
        # the parametric constraint is second in evidence.constraints
        self.assertEqual(self.evidence.parametric_indices, [1])

    def test_conditional_posterior_tempering(self):
        # the Gibbs conditional must apply likelihood_scaling and the
        # constraint's Evidence weight, matching log_posterior's tempering
        scaling = 0.5
        weights = np.array([1.0, 3.0])  # parametric constraint has weight 3
        evidence = Evidence(constraints=self.evidence.constraints, weights=weights)
        config = CalibrationConfig(
            evidence=evidence,
            model_config=self.model_config,
            likelihood_configs=[self.likelihood_config],
            likelihood_scaling=scaling,
        )

        xmodel = np.array([1.0, 1.0])
        ym = evidence.parametric_constraints[0].predict(*xmodel)
        x_lm = np.array([0.0])

        lp = self.likelihood_config.prior_logpdf(x_lm)
        ll = evidence.parametric_constraints[0].marginal_log_likelihood(ym, *x_lm)
        self.assertAlmostEqual(
            config.conditional_posterior(x_lm, 0, ym), lp + scaling * 3.0 * ll
        )

    def test_predict_parametric_aligns_with_conditional_posterior(self):
        # mixed evidence: constraints[0] is non-parametric, constraints[1] is the
        # parametric one. predict() is in constraints order (len 2) while
        # predict_parametric() is in parametric order (len 1) -> aligned with lm_index.
        config = CalibrationConfig(
            evidence=self.evidence,
            model_config=self.model_config,
            likelihood_configs=[self.likelihood_config],
        )
        xmodel = np.array([1.0, 1.0])

        all_preds = config.predict(xmodel)
        param_preds = config.predict_parametric(xmodel)
        self.assertEqual(len(all_preds), 2)
        self.assertEqual(len(param_preds), 1)
        # predict_parametric()[0] is the parametric constraint's prediction (== all_preds[1])
        np.testing.assert_allclose(param_preds[0][0], all_preds[1][0])
        np.testing.assert_allclose(
            param_preds[0][0],
            self.evidence.parametric_constraints[0].predict(*xmodel)[0],
        )

    def test_starting_location_with_single_parameter_sectors(self):
        model = Polynomial(0)
        gamma = gamma_parameter()
        observation = Observation(
            x=np.array([1.0, 2.0]),
            y=np.array([1.0, 1.1]),
            y_stat_err=np.array([0.1, 0.1]),
        )
        constraint = model_error_constraint(observation, model, gamma)
        evidence = Evidence(constraints=[constraint])
        model_config = ParameterConfig(
            params=model.params,
            prior=scipy.stats.multivariate_normal(mean=[0], cov=[[1]]),
            initial_proposal_distribution=scipy.stats.multivariate_normal(
                mean=[0], cov=[[1]]
            ),
        )
        likelihood_config = ParameterConfig(
            params=list(constraint.params),
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

    def test_prior_transform_via_generic_prior(self):
        """prior_transform works end-to-end with TruncatedNormalPrior sectors."""
        model = Polynomial(1)
        gamma = gamma_parameter()
        constraint = model_error_constraint(
            Observation(
                x=np.array([1.0, 2.0, 3.0, 4.0]),
                y=np.array([1.0, 2.0, 3.0, 4.0]),
                y_stat_err=np.array([0.1, 0.1, 0.1, 0.1]),
            ),
            model,
            gamma,
        )
        evidence = Evidence(constraints=[constraint])
        model_prior = TruncatedNormalPrior(
            mu=[0.0, 1.0], sigma=[1.0, 1.0], lower=[-5.0, -5.0], upper=[5.0, 5.0]
        )
        lm_prior = TruncatedNormalPrior(
            mu=[0.0], sigma=[1.0], lower=[-5.0], upper=[5.0]
        )
        config = CalibrationConfig(
            evidence=evidence,
            model_config=ParameterConfig(
                params=model.params,
                prior=model_prior,
                initial_proposal_distribution=model_prior,
            ),
            likelihood_configs=[
                ParameterConfig(
                    params=list(constraint.params),
                    prior=lm_prior,
                    initial_proposal_distribution=lm_prior,
                )
            ],
        )
        u = np.full(config.ndim, 0.5)
        theta = config.prior_transform(u)
        self.assertEqual(theta.shape, (config.ndim,))
        self.assertTrue(np.all(np.isfinite(theta)))


if __name__ == "__main__":
    unittest.main()
