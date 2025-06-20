import unittest
import numpy as np
from rxmc.observation import Observation, FixedCovarianceObservation
from rxmc.elastic_diffxs_observation import ElasticDifferentialXSObservation


class TestObservation(unittest.TestCase):

    def test_initialization(self):
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        observation = Observation(x, y)
        self.assertEqual(observation.n_data_pts, 3)
        np.testing.assert_array_equal(observation.x, x)
        np.testing.assert_array_equal(observation.y, y)

    def test_invalid_initialization(self):
        x = np.array([1, 2, 3])
        y = np.array([4, 5])
        with self.assertRaises(ValueError):
            Observation(x, y)

    def test_statistical_covariance(self):
        x = np.array([1, 2])
        y = np.array([2, 4])
        y_stat_err = np.array([0.1, 0.2])
        observation = Observation(x, y, y_stat_err=y_stat_err)
        expected_covariance = np.diag(y_stat_err**2)
        np.testing.assert_array_almost_equal(
            observation.statistical_covariance, expected_covariance
        )

    def test_systematic_covariance(self):
        x = np.array([1, 2])
        y = np.array([2, 4])
        y_stat_err = np.array([0.1, 0.2])
        y_sys_err_norm = 0.3
        y_sys_err_offset = 0.001
        observation = Observation(
            x,
            y,
            y_stat_err=y_stat_err,
            y_sys_err_normalization=y_sys_err_norm,
            y_sys_err_offset=y_sys_err_offset,
        )
        np.testing.assert_array_almost_equal(
            observation.systematic_offset_covariance,
            np.outer(np.ones_like(y), np.ones_like(y)) * y_sys_err_offset**2,
        )
        np.testing.assert_array_almost_equal(
            observation.systematic_normalization_covariance,
            np.outer(np.ones_like(y), np.ones_like(y)) * y_sys_err_norm**2,
        )

    def test_full_covariance(self):
        x = np.array([1, 2])
        y = np.array([2, 4])
        y_stat_err = np.array([0.1, 0.2])
        y_sys_err_norm = 0.3
        y_sys_err_offset = 0.001
        observation = Observation(
            x,
            y,
            y_stat_err=y_stat_err,
            y_sys_err_normalization=y_sys_err_norm,
            y_sys_err_offset=y_sys_err_offset,
        )
        expected_covariance = (
            np.diag(y_stat_err**2)
            + np.outer(y, y) * y_sys_err_norm**2
            + np.outer(np.ones_like(y), np.ones_like(y)) * y_sys_err_offset**2
        )
        np.testing.assert_array_almost_equal(
            observation.covariance(y), expected_covariance
        )

    def test_full_covariance_with_masks_offset_only(self):
        x = np.array([1, 2, 3, 4])
        y = x**2
        y_stat_err = 0.119 * y
        # first and last points have a shared err in offset of 1,
        # no systematic error in offset for the other points
        y_sys_err_offset = [0.331]
        y_sys_err_offset_mask = [
            np.array([True, False, False, True]),
        ]
        # create the observation with the masks
        observation = Observation(
            x,
            y,
            y_stat_err=y_stat_err,
            y_sys_err_offset=y_sys_err_offset,
            y_sys_err_offset_mask=y_sys_err_offset_mask,
        )
        expected_covariance = (
            # statistical error
            np.diag(y_stat_err**2)
            # systematic error in offset for first and last points
            + np.outer(np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1])) * 0.331**2
        )

        np.testing.assert_array_almost_equal(
            observation.covariance(y), expected_covariance
        )

    def test_full_covariance_with_masks(self):
        x = np.array([1, 2, 3, 4])
        y = x**2
        y_stat_err = 0.119 * y
        # first two points have a shared 5% systematic error in normalization
        # all points have a shared 10% systematic error in normalization
        y_sys_err_norm = [0.05, 0.1]
        y_sys_err_normalization_mask = [
            np.array([True, True, False, False]),
            np.ones_like(y),
        ]
        # first and last points have a shared err in offset of 1,
        # no systematic error in offset for the other points
        y_sys_err_offset = [0.331]
        y_sys_err_offset_mask = [
            np.array([True, False, False, True]),
        ]
        # create the observation with the masks
        observation = Observation(
            x,
            y,
            y_stat_err=y_stat_err,
            y_sys_err_normalization=y_sys_err_norm,
            y_sys_err_normalization_mask=y_sys_err_normalization_mask,
            y_sys_err_offset=y_sys_err_offset,
            y_sys_err_offset_mask=y_sys_err_offset_mask,
        )
        expected_covariance = (
            # statistical error
            np.diag(y_stat_err**2)
            # systematic error in normalization
            + np.outer(y, y)
            * (
                # for all points
                0.1**2
                # plus for first two points
                + np.outer(np.array([1, 1, 0, 0]), np.array([1, 1, 0, 0])) * 0.05**2
            )
            # systematic error in offset for first and last points
            + np.outer(np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1])) * 0.331**2
        )

        np.testing.assert_array_almost_equal(
            observation.covariance(y), expected_covariance
        )

    def test_residual(self):
        x = np.array([1, 2])
        y = np.array([2, 4])
        ym = np.array([1.5, 3.5])
        observation = Observation(x, y)
        expected_residual = y - ym
        np.testing.assert_array_almost_equal(
            observation.residual(ym), expected_residual
        )

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


class TestFixedCovarianceObservation(unittest.TestCase):

    def test_fixed_covariance_initialization(self):
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        covariance = np.array([0.1, 0.2, 0.3])
        obs = FixedCovarianceObservation(x, y, covariance)
        expected_covariance = np.diag(covariance)
        np.testing.assert_array_almost_equal(obs.cov, expected_covariance)

    def test_fixed_covariance_initialization_general(self):
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        covariance = np.diag([1, 1, 1])
        obs = FixedCovarianceObservation(x, y, covariance)
        np.testing.assert_array_almost_equal(obs.cov, covariance)

    def test_invalid_fixed_covariance(self):
        x = np.array([1, 2])
        y = np.array([3, 4])
        covariance = np.array([0.1, 0.2, 0.3])
        with self.assertRaises(ValueError):
            FixedCovarianceObservation(x, y, covariance)

    def test_covariance_method(self):
        x = np.array([1, 2])
        y = np.array([3, 4])
        covariance = np.array([0.1, 0.2])
        obs = FixedCovarianceObservation(x, y, covariance)
        np.testing.assert_array_almost_equal(obs.covariance(y), np.diag(covariance))


if __name__ == "__main__":
    unittest.main()
