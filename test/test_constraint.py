"""Tests for the stacked Constraint: multi-observation stacking and case A/B."""

import unittest

import numpy as np
from sklearn.gaussian_process.kernels import RBF

from helpers import manual_mvn_loglike
from rxmc.constraint import Constraint
from rxmc.covariance import (
    Term,
    kernel_term,
    noise_term,
    normalization_term,
    systematic_term,
)
from rxmc.evidence import Evidence
from rxmc.likelihood_model import GaussianLikelihood, StudentT
from rxmc.observation import Observation
from rxmc.params import Parameter
from rxmc.physical_model import PerObservationScaledModel, Polynomial


class TestStackedConstraint(unittest.TestCase):
    def setUp(self):
        self.pm = Polynomial(order=1)
        self.model_params = (0.5, 1.2)
        self.obs1 = Observation(
            np.array([1.0, 2.0]), np.array([2.0, 3.0]), y_stat_err=np.array([0.1, 0.2])
        )
        self.obs2 = Observation(
            np.array([3.0, 4.0, 5.0]),
            np.array([5.0, 6.0, 8.0]),
            y_stat_err=np.array([0.3, 0.2, 0.4]),
        )

    def _stacked(self):
        y = np.concatenate([self.obs1.y, self.obs2.y])
        ym = np.concatenate(
            [
                self.pm.evaluate(self.obs1, *self.model_params),
                self.pm.evaluate(self.obs2, *self.model_params),
            ]
        )
        return y, ym

    def test_block_diagonal_equals_sum_of_blocks(self):
        c = Constraint([self.obs1, self.obs2], self.pm)
        self.assertEqual(c.n_data_pts, 5)
        self.assertTrue(c.covariance.block_diagonal)

        y, ym = self._stacked()
        stat = np.concatenate([self.obs1.y_stat_err, self.obs2.y_stat_err])
        cov = np.diag(stat**2)
        expected = manual_mvn_loglike(y, ym, cov)
        self.assertAlmostEqual(c.log_likelihood(self.model_params), expected)

    def test_block_diagonal_fast_path_matches_dense(self):
        # build a constraint and compare blockwise path against an explicit dense MVN
        c = Constraint([self.obs1, self.obs2], self.pm)
        y, ym = self._stacked()
        stat = np.concatenate([self.obs1.y_stat_err, self.obs2.y_stat_err])
        dense = manual_mvn_loglike(y, ym, np.diag(stat**2))
        self.assertAlmostEqual(c.log_likelihood(self.model_params), dense)

    def test_constant_block_diag_cached_factors_match_dense(self):
        # constant multi-block covariance: cold call and cache-warm call both
        # equal the manual dense MVN
        c = Constraint([self.obs1, self.obs2], self.pm)
        y, ym = self._stacked()
        stat = np.concatenate([self.obs1.y_stat_err, self.obs2.y_stat_err])
        expected = manual_mvn_loglike(y, ym, np.diag(stat**2))
        cold = c.log_likelihood(self.model_params)
        warm = c.log_likelihood(self.model_params)
        self.assertAlmostEqual(cold, expected)
        self.assertEqual(cold, warm)

    def test_case_A_cross_block_coupling(self):
        # one rank-one mode spanning both blocks couples the data (off-diagonal blocks)
        eta = 0.1
        p = Parameter("log eta")
        support = np.arange(5)
        coupling = systematic_term(p, basis=lambda c: c.ym, support=support)
        c = Constraint([self.obs1, self.obs2], self.pm, extra_terms=[coupling])

        self.assertFalse(c.covariance.block_diagonal)
        self.assertEqual(c.n_params, 1)

        y, ym = self._stacked()
        stat = np.concatenate([self.obs1.y_stat_err, self.obs2.y_stat_err])
        cov = np.diag(stat**2) + eta**2 * np.outer(ym, ym)
        expected = manual_mvn_loglike(y, ym, cov)
        self.assertAlmostEqual(
            c.log_likelihood(self.model_params, (np.log(eta),)), expected
        )

    def test_case_A_differs_from_independent(self):
        eta = 0.1
        p = Parameter("log eta")
        coupled = Constraint(
            [self.obs1, self.obs2],
            self.pm,
            extra_terms=[
                systematic_term(p, basis=lambda c: c.ym, support=np.arange(5))
            ],
        )
        # independent: two per-block normalization modes (no cross coupling)
        p1, p2 = Parameter("log eta 1"), Parameter("log eta 2")
        independent = Constraint(
            [self.obs1, self.obs2],
            self.pm,
            extra_terms=[
                normalization_term(parameter=p1, support=np.arange(2)),
                normalization_term(parameter=p2, support=np.arange(2, 5)),
            ],
        )
        ll_coupled = coupled.log_likelihood(self.model_params, (np.log(eta),))
        ll_indep = independent.log_likelihood(
            self.model_params, (np.log(eta), np.log(eta))
        )
        self.assertNotAlmostEqual(ll_coupled, ll_indep)


class TestPerDatasetScaledModel(unittest.TestCase):
    """Per-dataset latent rho expressed as identity-routed model parameters."""

    def setUp(self):
        self.base = Polynomial(order=1)
        self.obs1 = Observation(
            np.array([1.0, 2.0, 3.0]),
            np.array([2.0, 4.0, 6.0]),
            y_stat_err=np.array([0.1, 0.1, 0.1]),
        )
        self.obs2 = Observation(
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0, 8.0, 12.0]),
            y_stat_err=np.array([0.1, 0.1, 0.1]),
        )

    def test_routes_rho_by_identity(self):
        model = PerObservationScaledModel(self.base, [self.obs1, self.obs2])
        # params = [a0, a1, log_rho_0, log_rho_1]
        self.assertEqual(model.n_params, 4)
        mp = (0.0, 2.0, np.log(1.0), np.log(2.0))
        np.testing.assert_allclose(model.evaluate(self.obs1, *mp), [2.0, 4.0, 6.0])
        np.testing.assert_allclose(model.evaluate(self.obs2, *mp), [4.0, 8.0, 12.0])

    def test_shared_across_constraints_in_evidence(self):
        model = PerObservationScaledModel(self.base, [self.obs1, self.obs2])
        c1 = Constraint([self.obs1], model)
        c2 = Constraint([self.obs2], model)
        ev = Evidence([c1, c2])  # all constraints share one model instance
        self.assertEqual(
            [p.name for p in ev.model_params], ["a0", "a1", "log_rho_0", "log_rho_1"]
        )
        ll = ev.log_likelihood((0.0, 2.0, np.log(1.0), np.log(2.0)))
        self.assertTrue(np.isfinite(ll))

    def test_holds_observation_references(self):
        # id()-keyed routing must keep the registered objects alive so a
        # garbage-collected observation's id can never be recycled
        import gc

        model = PerObservationScaledModel(self.base, [self.obs1, self.obs2])
        gc.collect()
        self.assertIs(model.observations[0], self.obs1)
        self.assertIs(model.observations[1], self.obs2)
        mp = (0.0, 2.0, np.log(1.0), np.log(2.0))
        np.testing.assert_allclose(
            model.evaluate(model.observations[0], *mp), [2.0, 4.0, 6.0]
        )

    def test_unregistered_observation_raises(self):
        model = PerObservationScaledModel(self.base, [self.obs1])
        with self.assertRaises(KeyError):
            model.evaluate(self.obs2, 0.0, 1.0, 0.0)


class TestSharedParameterCaseB(unittest.TestCase):
    def test_shared_eta_one_param(self):
        pm = Polynomial(order=0)
        obs1 = Observation(np.array([1.0, 2.0]), np.array([3.0, 3.0]))
        obs2 = Observation(np.array([3.0, 4.0]), np.array([3.0, 3.0]))
        eta = Parameter("log eta")
        c = Constraint(
            [obs1, obs2],
            pm,
            extra_terms=[
                normalization_term(parameter=eta, support=np.arange(2)),
                normalization_term(parameter=eta, support=np.arange(2, 4)),
            ],
        )
        # one shared parameter, covariance stays block-diagonal
        self.assertEqual(c.n_params, 1)
        self.assertTrue(c.covariance.block_diagonal)


class TestParamCountValidation(unittest.TestCase):
    """The full constraint tuple is validated; surplus params no longer vanish."""

    def setUp(self):
        self.pm = Polynomial(order=1)
        self.mp = (0.5, 1.2)
        self.obs = Observation(
            np.array([1.0, 2.0]), np.array([2.0, 3.0]), y_stat_err=np.array([0.1, 0.2])
        )

    def test_surplus_params_raise(self):
        # reviewer repro: these used to be silently swallowed, returning the
        # same value as the no-param call
        c = Constraint([self.obs], self.pm)
        with self.assertRaises(ValueError) as cm:
            c.log_likelihood(self.mp, (0.3, 99.0, -5.0))
        self.assertIn("expects 0 parameter", str(cm.exception))

    def test_missing_params_raise(self):
        eta = Parameter("log eta")
        c = Constraint(
            [self.obs],
            self.pm,
            extra_terms=[normalization_term(parameter=eta, support=np.arange(2))],
        )
        with self.assertRaises(ValueError) as cm:
            c.log_likelihood(self.mp)
        self.assertIn("log eta", str(cm.exception))

    def test_correct_count_unchanged(self):
        eta = Parameter("log eta")
        c = Constraint(
            [self.obs],
            self.pm,
            extra_terms=[normalization_term(parameter=eta, support=np.arange(2))],
        )
        self.assertTrue(np.isfinite(c.log_likelihood(self.mp, (np.log(0.1),))))

    def test_studentt_chi2_full_tuple(self):

        eps = Parameter("log eps")
        student = Constraint(
            [self.obs],
            self.pm,
            likelihood=StudentT(),
            extra_terms=[noise_term(eps, support=np.arange(2))],
        )
        # covariance-only tuple is a deficit now
        with self.assertRaises(ValueError):
            student.chi2(self.mp, (np.log(0.1),))
        # full tuple works; nu is ignored by the statistic
        gauss = Constraint(
            [self.obs],
            self.pm,
            likelihood=GaussianLikelihood(),
            extra_terms=[noise_term(Parameter("log eps g"), support=np.arange(2))],
        )
        self.assertAlmostEqual(
            student.chi2(self.mp, (np.log(0.1), 4.0)),
            gauss.chi2(self.mp, (np.log(0.1),)),
        )

    def test_covariance_matrix_full_tuple_convention(self):

        eta = Parameter("log eta")
        c = Constraint(
            [self.obs],
            self.pm,
            likelihood=StudentT(),
            extra_terms=[normalization_term(parameter=eta, support=np.arange(2))],
        )
        # reviewer repro: forwarding the full sampled tuple used to crash
        S = c.covariance_matrix(self.mp, (np.log(0.1), 4.0))
        gauss = Constraint(
            [self.obs],
            self.pm,
            extra_terms=[
                normalization_term(parameter=Parameter("log eta"), support=np.arange(2))
            ],
        )
        np.testing.assert_allclose(S, gauss.covariance_matrix(self.mp, (np.log(0.1),)))
        # a partial (covariance-only) tuple is now rejected uniformly
        with self.assertRaises(ValueError):
            c.covariance_matrix(self.mp, (np.log(0.1),))


class TestParameterNameValidation(unittest.TestCase):
    def setUp(self):
        self.pm = Polynomial(order=1)
        self.obs = Observation(
            np.array([1.0, 2.0]), np.array([2.0, 3.0]), y_stat_err=np.array([0.1, 0.2])
        )

    def test_two_distinct_same_name_params_raise(self):
        # two equal-but-distinct objects are almost certainly intended sharing
        # gone wrong (sharing works by identity)
        with self.assertRaises(ValueError) as cm:
            Constraint(
                [self.obs],
                self.pm,
                extra_terms=[
                    normalization_term(
                        parameter=Parameter("log eta"), support=np.arange(2)
                    ),
                    normalization_term(
                        parameter=Parameter("log eta"), support=np.arange(2)
                    ),
                ],
            )
        self.assertIn("SAME Parameter object", str(cm.exception))

    def test_collision_with_model_param_name_raises(self):
        # Polynomial(order=1) has model params named a0, a1
        with self.assertRaises(ValueError) as cm:
            Constraint(
                [self.obs],
                self.pm,
                extra_terms=[
                    normalization_term(parameter=Parameter("a0"), support=np.arange(2))
                ],
            )
        self.assertIn("physical-model parameter", str(cm.exception))

    def test_likelihood_param_collision_raises(self):

        nu_clone = Parameter("nu")
        with self.assertRaises(ValueError):
            Constraint(
                [self.obs],
                self.pm,
                likelihood=StudentT(nu_parameter=Parameter("nu")),
                extra_terms=[
                    normalization_term(parameter=nu_clone, support=np.arange(2))
                ],
            )


class TestSingularCovarianceGuard(unittest.TestCase):
    """A constant singular covariance fails at construction with a clear error."""

    def setUp(self):
        self.pm = Polynomial(order=1)
        self.x = np.array([1.0, 2.0])
        self.y = np.array([2.0, 3.0])

    def test_zero_stat_err_raises_clear_error(self):
        obs = Observation(self.x, self.y)  # y_stat_err defaults to zeros
        with self.assertRaises(ValueError) as cm:
            Constraint([obs], self.pm)
        self.assertIn("singular", str(cm.exception))
        self.assertIn("observation 0", str(cm.exception))

    def test_offending_dataset_named_by_label(self):
        good = Observation(self.x, self.y, y_stat_err=np.array([0.1, 0.1]))
        bad = Observation(np.array([3.0, 4.0]), np.array([4.0, 5.0]), label="C1010-2-0")
        with self.assertRaises(ValueError) as cm:
            Constraint([good, bad], self.pm)
        msg = str(cm.exception)
        self.assertIn("C1010-2-0", msg)
        self.assertNotIn("observation 0'", msg)

    def test_zero_stat_err_with_covering_term_ok(self):
        obs = Observation(self.x, self.y)
        c = Constraint(
            [obs],
            self.pm,
            extra_terms=[Term(np.array([0.2, 0.2]), kind="diag", support=np.arange(2))],
        )
        self.assertTrue(np.isfinite(c.log_likelihood((0.5, 1.2))))

    def test_zero_stat_err_with_systematic_terms_ok(self):
        obs = Observation(
            self.x, self.y, y_sys_err_normalization=0.05, y_sys_err_offset=0.1
        )
        c = Constraint([obs], self.pm, extra_terms=obs.systematic_terms(np.arange(2)))
        self.assertTrue(np.isfinite(c.log_likelihood((0.5, 1.2))))

    def test_parametric_covariance_not_checked_eagerly(self):
        # replace-semantics: zero stat err + a free noise term must construct

        obs = Observation(self.x, self.y)
        p = Parameter("log eps")
        c = Constraint(
            [obs], self.pm, extra_terms=[noise_term(p, support=np.arange(2))]
        )
        self.assertEqual(c.n_params, 1)


class TestConstraintFixes(unittest.TestCase):
    def setUp(self):
        self.pm = Polynomial(order=1)
        self.params = (0.5, 1.2)
        self.obs = Observation(
            np.array([1.0, 2.0, 3.0]),
            np.array([2.0, 3.0, 5.0]),
            y_stat_err=np.array([0.1, 0.2, 0.3]),
        )

    def test_covariance_matrix_returns_copy(self):
        c = Constraint([self.obs], self.pm)
        before = c.log_likelihood(self.params)
        M = c.covariance_matrix(self.params)
        M += 1e6  # in-place mutation must not corrupt the constraint
        after = c.log_likelihood(self.params)
        self.assertAlmostEqual(before, after)

    def test_stack_shape_guard(self):
        c = Constraint([self.obs], self.pm)
        with self.assertRaises(ValueError):
            c.marginal_log_likelihood([np.array([1.0, 2.0])])  # wrong length

    def test_include_statistical_term_false_omits_diagonal(self):
        sup = np.arange(self.obs.n_data_pts)
        cov = np.diag([0.04, 0.04, 0.04])
        c = Constraint(
            [self.obs],
            self.pm,
            extra_terms=[Term(cov, support=sup)],
            include_statistical_term=False,
        )
        # only the supplied fixed Term survives (no statistical diagonal added)
        S = c.covariance_matrix(self.params)
        np.testing.assert_allclose(S, cov)


class TestScaledModel(unittest.TestCase):
    def test_scale_applied_and_params_prepended(self):
        from rxmc.physical_model import ScaledModel

        base = Polynomial(order=1)
        obs = Observation(
            np.array([1.0, 2.0, 3.0]),
            np.array([2.0, 4.0, 6.0]),
            y_stat_err=np.array([0.1, 0.1, 0.1]),
        )
        model = ScaledModel(base)
        self.assertEqual(model.n_params, 3)
        self.assertEqual(model.params[0].name, "log normalization")
        np.testing.assert_allclose(
            model.evaluate(obs, np.log(2.0), 0.0, 2.0),
            2.0 * base.evaluate(obs, 0.0, 2.0),
        )

    def test_linear_scale(self):
        from rxmc.params import Parameter
        from rxmc.physical_model import ScaledModel

        base = Polynomial(order=0)
        obs = Observation(np.array([1.0, 2.0]), np.array([3.0, 3.0]))
        model = ScaledModel(base, scale_parameter=Parameter("rho"), log=False)
        np.testing.assert_allclose(
            model.evaluate(obs, 1.5, 4.0), 1.5 * base.evaluate(obs, 4.0)
        )


class TestConstantTermsReadX(unittest.TestCase):
    """Constant terms (fixed kernels, x-dependent fixed diagonals) are evaluated
    once with the invariant x/y at Constraint construction."""

    def setUp(self):
        self.pm = Polynomial(order=1)
        self.mp = (0.1, 1.0)
        self.x = np.linspace(0.0, 1.0, 5)
        self.err = np.full(5, 0.1)
        self.obs = Observation(self.x, self.x + 0.1, y_stat_err=self.err)

    def test_fixed_kernel_term_in_constraint(self):
        kernel = RBF(length_scale=0.3, length_scale_bounds="fixed")
        c = Constraint(
            [self.obs], self.pm, extra_terms=[kernel_term(kernel, jitter=0.0)]
        )
        self.assertTrue(c.covariance.is_constant)
        self.assertEqual(c.n_params, 0)
        ym = self.pm(self.obs, *self.mp)
        cov = np.diag(self.err**2) + kernel(self.x[:, None])
        expected = manual_mvn_loglike(self.obs.y, ym, cov)
        self.assertAlmostEqual(c.log_likelihood(self.mp), expected)

    def test_x_dependent_constant_term_in_constraint(self):
        t = Term(lambda c: 0.1 * c.x, kind="diag", constant=True)
        c = Constraint([self.obs], self.pm, extra_terms=[t])
        self.assertTrue(c.covariance.is_constant)
        ym = self.pm(self.obs, *self.mp)
        cov = np.diag(self.err**2 + (0.1 * self.x) ** 2)
        self.assertAlmostEqual(
            c.log_likelihood(self.mp), manual_mvn_loglike(self.obs.y, ym, cov)
        )
        # the eager validation warmed the cache; a fresh copy is returned to callers
        S = c.covariance_matrix(self.mp)
        self.assertTrue(S.flags.writeable)
        np.testing.assert_allclose(S, cov)


if __name__ == "__main__":
    unittest.main()
