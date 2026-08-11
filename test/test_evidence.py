import unittest

import numpy as np

from rxmc.constraint import Constraint
from rxmc.covariance import model_error_term
from rxmc.evidence import Evidence
from rxmc.observation import Observation
from rxmc.params import Parameter
from rxmc.physical_model import Polynomial


class TestEvidence(unittest.TestCase):

    def setUp(self):
        y = np.array([1.0, 2.0, 3.0])
        x = np.array([1.0, 2.0, 3.0])
        y_stat_err = np.array([0.1, 0.2, 0.3])
        self.y = y
        self.y_stat_err = y_stat_err
        self.observations = [Observation(x=x, y=y, y_stat_err=y_stat_err)]
        self.pm = Polynomial(order=1)
        self.constraints = [
            Constraint(observations=self.observations, physical_model=self.pm)
            for _ in range(4)
        ]
        self.weights = np.array([1.0, 1.0, 1.0, 1.0])
        self.evidence = Evidence(constraints=self.constraints, weights=self.weights)

        self.model_params = (3.0, 5.0)
        modely = self.pm.evaluate(self.observations[0], *self.model_params)
        delta = y - modely
        chi2 = np.sum((delta / y_stat_err) ** 2)
        N = self.observations[0].n_data_pts
        log_det = np.sum(np.log(y_stat_err**2))
        logl_single = -0.5 * (N * np.log(2 * np.pi) + log_det + chi2)
        self.expected_loglikelihood = 4 * logl_single

    def test_serial_execution(self):
        log_likelihood = self.evidence.log_likelihood(model_params=self.model_params)
        self.assertAlmostEqual(log_likelihood, self.expected_loglikelihood)

    def test_no_parametric_constraints_detected(self):
        self.assertEqual(len(self.evidence.parametric_constraints), 0)
        self.assertEqual(self.evidence.n_params, self.pm.n_params)

    def test_parametric_constraint_auto_detected(self):
        gamma = Parameter("log gamma")
        parametric = Constraint(
            observations=self.observations,
            physical_model=self.pm,
            extra_terms=[model_error_term(np.arange(3), gamma)],
        )
        evidence = Evidence(constraints=[self.constraints[0], parametric])
        self.assertEqual(len(evidence.parametric_constraints), 1)
        self.assertIs(evidence.parametric_constraints[0], parametric)
        self.assertEqual(evidence.n_params, self.pm.n_params + 1)

    def test_parametric_constraint_receives_its_params(self):
        gamma = Parameter("log gamma")
        parametric = Constraint(
            observations=self.observations,
            physical_model=self.pm,
            extra_terms=[model_error_term(np.arange(3), gamma)],
        )
        evidence = Evidence(constraints=[parametric])
        # one tuple per parametric constraint
        ll = evidence.log_likelihood(self.model_params, [(np.log(0.1),)])
        # directly via the constraint
        ll_direct = parametric.log_likelihood(self.model_params, (np.log(0.1),))
        self.assertAlmostEqual(ll, ll_direct)

    def test_wrong_cov_params_length_raises(self):
        gamma = Parameter("log gamma")
        parametric = Constraint(
            observations=self.observations,
            physical_model=self.pm,
            extra_terms=[model_error_term(np.arange(3), gamma)],
        )
        evidence = Evidence(constraints=[parametric])
        with self.assertRaises(ValueError):
            evidence.log_likelihood(self.model_params, [])


class TestCrossConstraintParameterValidation(unittest.TestCase):
    """Covariance/likelihood parameters are constraint-scoped (spec §8)."""

    def setUp(self):
        self.pm = Polynomial(order=1)
        self.obs = [
            Observation(
                x=np.array([1.0, 2.0, 3.0]),
                y=np.array([1.0, 2.0, 3.0]),
                y_stat_err=np.array([0.1, 0.2, 0.3]),
            )
        ]

    def _constraint(self, param):
        return Constraint(
            observations=self.obs,
            physical_model=self.pm,
            extra_terms=[model_error_term(np.arange(3), param)],
        )

    def test_same_parameter_object_in_two_constraints_raises(self):
        gamma = Parameter("log gamma")
        c1, c2 = self._constraint(gamma), self._constraint(gamma)
        with self.assertRaises(ValueError) as cm:
            Evidence(constraints=[c1, c2])
        self.assertIn("same object", str(cm.exception))

    def test_duplicate_name_across_constraints_raises(self):
        c1 = self._constraint(Parameter("log gamma"))
        c2 = self._constraint(Parameter("log gamma"))
        with self.assertRaises(ValueError) as cm:
            Evidence(constraints=[c1, c2])
        self.assertIn("Duplicate parameter name", str(cm.exception))

    def test_unique_names_accepted(self):
        c1 = self._constraint(Parameter("log gamma 1"))
        c2 = self._constraint(Parameter("log gamma 2"))
        ev = Evidence(constraints=[c1, c2])
        self.assertEqual(ev.n_likelihood_params, 2)

    def test_two_default_student_t_constraints_raise(self):
        from rxmc.likelihood_model import StudentT

        c1 = Constraint(self.obs, self.pm, likelihood=StudentT())
        c2 = Constraint(self.obs, self.pm, likelihood=StudentT())
        with self.assertRaises(ValueError):
            Evidence(constraints=[c1, c2])


if __name__ == "__main__":
    unittest.main()
