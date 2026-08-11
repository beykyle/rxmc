import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from rxmc.elastic_diffxs_observation import ElasticDifferentialXSObservation
from rxmc.ias_pn_observation import IsobaricAnalogPNObservation
from rxmc.observation import Observation


class DummyElasticWorkspace:
    def __init__(self, rutherford=1.0):
        self.rutherford = rutherford


class TestElasticDifferentialXSObservation(unittest.TestCase):
    @patch("rxmc.elastic_diffxs_observation.set_up_solver")
    def test_direct_construction_from_explicit_data(self, mock_set_up_solver):
        mock_set_up_solver.return_value = (
            DummyElasticWorkspace(),
            DummyElasticWorkspace(),
            object(),
        )

        angles_deg = np.array([15.0, 30.0, 45.0])
        y = np.array([1.2, 0.8, 0.4])
        y_stat_err = np.array([0.1, 0.1, 0.1])

        obs = ElasticDifferentialXSObservation(
            x=angles_deg,
            y=y,
            Elab=12.0,
            reaction=object(),
            quantity="dXS/dA",
            measurement_quantity="dXS/dA",
            y_units="barn / steradian",
            y_stat_err=y_stat_err,
            dataset_label="mock-elastic",
        )

        # it IS an Observation (statistical error only); systematics are composed
        # as Constraint extra_terms by the caller
        self.assertIsInstance(obs, Observation)
        np.testing.assert_allclose(obs.x, np.deg2rad(angles_deg))
        np.testing.assert_allclose(obs.y, y)
        np.testing.assert_allclose(obs.y_stat_err, y_stat_err)
        self.assertEqual(obs.subentry, "mock-elastic")
        self.assertIsNotNone(obs.statistical_term(np.arange(obs.n_data_pts)))

    @patch("rxmc.elastic_diffxs_observation.set_up_solver")
    def test_from_measurement_construction(self, mock_set_up_solver):
        mock_set_up_solver.return_value = (
            DummyElasticWorkspace(),
            DummyElasticWorkspace(),
            object(),
        )

        measurement = SimpleNamespace(
            x=np.array([20.0, 40.0]),
            y=np.array([2.0, 1.0]),
            Einc=8.0,
            quantity="dXS/dA",
            y_units="barn / steradian",
            statistical_err=np.array([0.2, 0.1]),
            systematic_norm_err=0.03,
            systematic_offset_err=0.02,
            subentry="elastic-subentry",
        )

        obs = ElasticDifferentialXSObservation.from_measurement(
            measurement=measurement,
            reaction=object(),
            quantity="dXS/dA",
        )

        np.testing.assert_allclose(obs.x, np.deg2rad(measurement.x))
        np.testing.assert_allclose(obs.y, measurement.y)
        np.testing.assert_allclose(obs.y_stat_err, measurement.statistical_err)
        self.assertEqual(obs.subentry, "elastic-subentry")
        # systematics are retained as inert metadata (norm == 1 here: b/sr)
        self.assertEqual(obs.norm, 1.0)
        self.assertEqual(obs.y_sys_err_normalization, 0.03)
        self.assertEqual(obs.y_sys_err_offset, 0.02)
        self.assertEqual(len(obs.systematic_terms(np.arange(obs.n_data_pts))), 2)

    @patch("rxmc.elastic_diffxs_observation.set_up_solver")
    def test_from_measurement_rutherford_array_norm(self, mock_set_up_solver):
        # dXS/dRuth requested from a dXS/dA measurement: norm is the per-angle
        # Rutherford cross section, so the absolute offset error becomes a
        # per-angle array in internal units, while the fractional normalization
        # error is untouched
        rutherford = np.array([2000.0, 500.0])  # mb/sr
        mock_set_up_solver.return_value = (
            DummyElasticWorkspace(rutherford=rutherford),
            DummyElasticWorkspace(rutherford=rutherford),
            object(),
        )

        measurement = SimpleNamespace(
            x=np.array([20.0, 40.0]),
            y=np.array([1800.0, 300.0]),
            Einc=8.0,
            quantity="dXS/dA",
            y_units="mb/sr",
            statistical_err=np.array([20.0, 10.0]),
            systematic_norm_err=np.array(0.03),  # 0-d, as exfor_tools stores it
            systematic_offset_err=5.0,  # mb/sr
            subentry="ruth-subentry",
        )

        obs = ElasticDifferentialXSObservation.from_measurement(
            measurement=measurement,
            reaction=object(),
            quantity="dXS/dRuth",
        )

        np.testing.assert_allclose(obs.norm, rutherford)
        np.testing.assert_allclose(obs.y, measurement.y / rutherford)
        np.testing.assert_allclose(
            obs.y_stat_err, measurement.statistical_err / rutherford
        )
        np.testing.assert_allclose(obs.y_sys_err_offset, 5.0 / rutherford)
        self.assertEqual(obs.y_sys_err_normalization, 0.03)

        # reaction-level regression: statistical + systematic terms recover the
        # old auto-folded covariance, in internal (normalized) units
        support = np.arange(2)
        from helpers import make_ctx
        from rxmc.covariance import ConstraintCovariance

        ym = np.array([0.9, 0.6])
        cov = ConstraintCovariance(
            [obs.statistical_term(support), *obs.systematic_terms(support)], N=2
        )
        S = cov.matrix(make_ctx(obs.x, obs.y, ym, [support]))
        omega = 5.0 / rutherford
        old = (
            np.diag((measurement.statistical_err / rutherford) ** 2)
            + np.outer(omega, omega)
            + 0.03**2 * np.outer(ym, ym)
        )
        np.testing.assert_allclose(S, old)

    @patch("rxmc.elastic_diffxs_observation.set_up_solver")
    def test_dxsda_from_dxsdruth_conversion(self, mock_set_up_solver):
        # the inverse branch: absolute dXS/dA requested from a Rutherford-ratio
        # measurement; norm = (1/mb->b) / rutherford, so obs.y is b/sr
        rutherford = np.array([2000.0, 500.0])  # mb/sr
        mock_set_up_solver.return_value = (
            DummyElasticWorkspace(rutherford=rutherford),
            DummyElasticWorkspace(rutherford=rutherford),
            object(),
        )

        y_ratio = np.array([0.9, 0.6])  # dimensionless dXS/dRuth
        obs = ElasticDifferentialXSObservation(
            x=np.array([20.0, 40.0]),
            y=y_ratio,
            Elab=8.0,
            reaction=object(),
            quantity="dXS/dA",
            measurement_quantity="dXS/dRuth",
            y_units="no-dim",
        )

        np.testing.assert_allclose(obs.norm, 1000.0 / rutherford)
        # y_ratio * rutherford[mb/sr] / 1000 = absolute xs in b/sr
        np.testing.assert_allclose(obs.y, y_ratio * rutherford / 1000.0)

    @patch("rxmc.elastic_diffxs_observation.set_up_solver")
    def test_incompatible_units_raise(self, mock_set_up_solver):
        mock_set_up_solver.return_value = (
            DummyElasticWorkspace(),
            DummyElasticWorkspace(),
            object(),
        )
        # dXS/dA measurement with non-cross-section units
        with self.assertRaises(ValueError):
            ElasticDifferentialXSObservation(
                x=np.array([15.0, 30.0]),
                y=np.array([1.0, 0.5]),
                Elab=12.0,
                reaction=object(),
                quantity="dXS/dA",
                measurement_quantity="dXS/dA",
                y_units="MeV",
            )
        # dimensionless quantity with dimensionful units
        with self.assertRaises(ValueError):
            ElasticDifferentialXSObservation(
                x=np.array([15.0, 30.0]),
                y=np.array([1.0, 0.5]),
                Elab=12.0,
                reaction=object(),
                quantity="Ay",
                measurement_quantity="Ay",
                y_units="mb/sr",
            )

    @patch("rxmc.elastic_diffxs_observation.set_up_solver")
    def test_quantity_mismatch_raises(self, mock_set_up_solver):
        mock_set_up_solver.return_value = (
            DummyElasticWorkspace(),
            DummyElasticWorkspace(),
            object(),
        )
        with self.assertRaises(ValueError):
            ElasticDifferentialXSObservation(
                x=np.array([15.0, 30.0]),
                y=np.array([1.0, 0.5]),
                Elab=12.0,
                reaction=object(),
                quantity="Ay",
                measurement_quantity="dXS/dA",
                y_units="barn / steradian",
            )


class TestIsobaricAnalogPNObservation(unittest.TestCase):
    @patch("rxmc.ias_pn_observation.set_up_solver")
    def test_direct_construction_from_explicit_data(self, mock_set_up_solver):
        mock_set_up_solver.return_value = (object(), object(), object(), object())

        angles_deg = np.array([10.0, 25.0, 50.0])
        y = np.array([0.4, 0.3, 0.2])
        y_stat_err = np.array([0.05, 0.05, 0.05])

        obs = IsobaricAnalogPNObservation(
            x=angles_deg,
            y=y,
            Elab=30.0,
            reaction=object(),
            ExIAS=5.0,
            y_units="barn / steradian",
            y_stat_err=y_stat_err,
            dataset_label="mock-ias",
        )

        self.assertIsInstance(obs, Observation)
        np.testing.assert_allclose(obs.x, np.deg2rad(angles_deg))
        np.testing.assert_allclose(obs.y, y)
        np.testing.assert_allclose(obs.y_stat_err, y_stat_err)
        self.assertEqual(obs.subentry, "mock-ias")

    @patch("rxmc.ias_pn_observation.set_up_solver")
    def test_from_measurement_construction(self, mock_set_up_solver):
        mock_set_up_solver.return_value = (object(), object(), object(), object())

        measurement = SimpleNamespace(
            x=np.array([5.0, 15.0]),
            y=np.array([0.9, 0.7]),
            Einc=18.0,
            y_units="barn / steradian",
            statistical_err=np.array([0.08, 0.07]),
            systematic_norm_err=0.02,
            systematic_offset_err=0.01,
            subentry="ias-subentry",
        )

        obs = IsobaricAnalogPNObservation.from_measurement(
            measurement=measurement,
            reaction=object(),
            ExIAS=4.5,
        )

        np.testing.assert_allclose(obs.x, np.deg2rad(measurement.x))
        np.testing.assert_allclose(obs.y, measurement.y)
        np.testing.assert_allclose(obs.y_stat_err, measurement.statistical_err)
        self.assertEqual(obs.subentry, "ias-subentry")
        # systematics retained; norm == 1 (measurement already in b/sr)
        self.assertEqual(obs.norm, 1.0)
        self.assertEqual(obs.y_sys_err_normalization, 0.02)
        self.assertEqual(obs.y_sys_err_offset, 0.01)
        self.assertEqual(len(obs.systematic_terms(np.arange(obs.n_data_pts))), 2)

    @patch("rxmc.ias_pn_observation.set_up_solver")
    def test_unit_conversion_divides_offset_not_normalization(self, mock_set_up_solver):
        mock_set_up_solver.return_value = (object(), object(), object(), object())

        obs = IsobaricAnalogPNObservation(
            x=np.array([5.0, 15.0]),
            y=np.array([900.0, 700.0]),
            Elab=18.0,
            reaction=object(),
            ExIAS=4.5,
            y_units="millibarn / steradian",
            y_stat_err=np.array([80.0, 70.0]),
            y_sys_err_normalization=0.02,
            y_sys_err_offset=10.0,
        )
        # mb -> b: norm = 1000
        self.assertEqual(obs.norm, 1000.0)
        np.testing.assert_allclose(obs.y, [0.9, 0.7])
        np.testing.assert_allclose(obs.y_stat_err, [0.08, 0.07])
        self.assertAlmostEqual(obs.y_sys_err_offset, 0.01)
        self.assertEqual(obs.y_sys_err_normalization, 0.02)


if __name__ == "__main__":
    unittest.main()
