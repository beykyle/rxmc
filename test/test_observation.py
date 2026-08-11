import unittest

import numpy as np

from helpers import make_ctx
from rxmc.covariance import ConstraintCovariance, RankOneTerm
from rxmc.observation import Observation


class TestObservation(unittest.TestCase):

    def test_initialization(self):
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        observation = Observation(x, y)
        self.assertEqual(observation.n_data_pts, 3)
        np.testing.assert_array_equal(observation.x, x)
        np.testing.assert_array_equal(observation.y, y)
        np.testing.assert_array_equal(observation.y_stat_err, np.zeros_like(y))

    def test_invalid_initialization(self):
        x = np.array([1, 2, 3])
        y = np.array([4, 5])
        with self.assertRaises(ValueError):
            Observation(x, y)

    def test_stat_err_shape_validation(self):
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        with self.assertRaises(ValueError):
            Observation(x, y, y_stat_err=np.array([0.1, 0.2]))

    def test_statistical_term_is_diagonal_variance(self):
        x = np.array([1.0, 2.0])
        y = np.array([2.0, 4.0])
        y_stat_err = np.array([0.1, 0.2])
        observation = Observation(x, y, y_stat_err=y_stat_err)
        support = np.arange(2)
        term = observation.statistical_term(support)
        Sigma = np.zeros((2, 2))
        term.add_to(Sigma, None, np.array([]))
        np.testing.assert_array_almost_equal(Sigma, np.diag(y_stat_err**2))

    def test_statistical_term_writes_into_support_block(self):
        # an observation occupying the second block of a length-4 stack
        x = np.array([1.0, 2.0])
        y = np.array([2.0, 4.0])
        y_stat_err = np.array([0.3, 0.4])
        observation = Observation(x, y, y_stat_err=y_stat_err)
        support = np.array([2, 3])
        cov = ConstraintCovariance([observation.statistical_term(support)], N=4)
        Sigma = cov.matrix(None)
        expected = np.zeros((4, 4))
        expected[2, 2] = 0.3**2
        expected[3, 3] = 0.4**2
        np.testing.assert_array_almost_equal(Sigma, expected)

    def test_default_statistical_term_is_constant(self):
        observation = Observation(
            np.array([1.0, 2.0]), np.array([2.0, 4.0]), y_stat_err=np.array([0.1, 0.2])
        )
        cov = ConstraintCovariance([observation.statistical_term(np.arange(2))], N=2)
        self.assertTrue(cov.is_constant)
        self.assertTrue(cov.block_diagonal)
        self.assertEqual(cov.n_params, 0)

    def test_systematics_default_none_and_no_terms(self):
        obs = Observation(np.array([1.0, 2.0]), np.array([2.0, 4.0]))
        self.assertIsNone(obs.y_sys_err_normalization)
        self.assertIsNone(obs.y_sys_err_offset)
        self.assertEqual(obs.systematic_terms(np.arange(2)), [])

    def test_systematics_storage(self):
        obs = Observation(
            np.array([1.0, 2.0]),
            np.array([2.0, 4.0]),
            y_sys_err_normalization=0.03,
            y_sys_err_offset=np.array([0.1, 0.2]),
        )
        self.assertEqual(obs.y_sys_err_normalization, 0.03)
        np.testing.assert_allclose(obs.y_sys_err_offset, [0.1, 0.2])
        # 0-d ndarrays count as scalars
        obs2 = Observation(
            np.array([1.0, 2.0]),
            np.array([2.0, 4.0]),
            y_sys_err_normalization=np.array(0.03),
        )
        self.assertEqual(obs2.y_sys_err_normalization, 0.03)

    def test_systematics_bad_shape_raises(self):
        with self.assertRaises(ValueError):
            Observation(
                np.array([1.0, 2.0, 3.0]),
                np.array([2.0, 4.0, 6.0]),
                y_sys_err_offset=np.array([0.1, 0.2]),
            )

    def test_systematics_zero_magnitudes_skipped(self):
        obs = Observation(
            np.array([1.0, 2.0]),
            np.array([2.0, 4.0]),
            y_sys_err_normalization=0.0,
            y_sys_err_offset=0,
        )
        self.assertEqual(obs.systematic_terms(np.arange(2)), [])

    def test_systematic_terms_offset_then_normalization(self):
        obs = Observation(
            np.array([1.0, 2.0]),
            np.array([2.0, 4.0]),
            y_sys_err_normalization=0.05,
            y_sys_err_offset=0.2,
        )
        terms = obs.systematic_terms(np.arange(2))
        self.assertEqual(len(terms), 2)
        self.assertTrue(all(isinstance(t, RankOneTerm) for t in terms))

    def test_systematic_terms_recover_old_covariance(self):
        # statistical_term + systematic_terms matches the old auto-folded
        # Observation.covariance(ym)
        y = np.array([1.0, 2.0, 4.0])
        ym = np.array([1.2, 2.1, 3.5])
        stat = np.array([0.1, 0.2, 0.3])
        norm_frac = 0.05
        offset = 0.2
        obs = Observation(
            np.arange(3.0),
            y,
            y_stat_err=stat,
            y_sys_err_normalization=norm_frac,
            y_sys_err_offset=offset,
        )
        support = np.arange(3)
        cov = ConstraintCovariance(
            [obs.statistical_term(support), *obs.systematic_terms(support)], N=3
        )
        S = cov.matrix(make_ctx(np.arange(3.0), y, ym, [support]))
        old = (
            np.diag(stat**2)
            + np.outer(offset * np.ones(3), offset * np.ones(3))
            + norm_frac**2 * np.outer(ym, ym)
        )
        np.testing.assert_allclose(S, old)

    def test_num_pts_within_interval(self):
        x = np.array([1, 2, 3, 4])
        y = np.array([10, 12, 14, 16])
        ylow = np.array([9, 11, 13, 15])
        yhigh = np.array([11, 13, 15, 17])
        observation = Observation(x, y)
        num_pts = observation.num_pts_within_interval(ylow, yhigh)
        self.assertEqual(num_pts, 4)

    def test_num_pts_within_interval_out(self):
        x = np.array([1, 2, 3, 4])
        y = np.array([10, 15, 14, -12])
        ylow = np.array([9, 11, 13, 15])
        yhigh = np.array([11, 13, 15, 17])
        observation = Observation(x, y)
        num_pts = observation.num_pts_within_interval(ylow, yhigh)
        self.assertEqual(num_pts, 2)


if __name__ == "__main__":
    unittest.main()
