import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from rxmc.elastic_diffxs_observation import ElasticDifferentialXSObservation
from rxmc.ias_pn_observation import IsobaricAnalogPNObservation


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
            y_sys_err_normalization=0.05,
            y_sys_err_offset=0.01,
            dataset_label="mock-elastic",
        )

        np.testing.assert_allclose(obs.x, np.deg2rad(angles_deg))
        np.testing.assert_allclose(obs.y, y)
        np.testing.assert_allclose(obs.y_stat_err, y_stat_err)
        self.assertEqual(obs.subentry, "mock-elastic")

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
            y_sys_err_normalization=0.04,
            y_sys_err_offset=0.01,
            dataset_label="mock-ias",
        )

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


if __name__ == "__main__":
    unittest.main()
