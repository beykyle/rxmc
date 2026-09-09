"""Tests for the low-level ``rxmc.transforms`` type."""

import unittest

import numpy as np

from rxmc.observation import Observation
from rxmc.params import Parameter
from rxmc.transforms import (
    Transform,
    as_transform,
    exp,
    identity,
    log,
    per_observation_scaling,
    scale,
)


class TestTransform(unittest.TestCase):
    def test_callable_is_wrapped_parameter_free(self):
        t = as_transform(np.sqrt)
        self.assertIsInstance(t, Transform)
        self.assertEqual(t.params, ())
        np.testing.assert_allclose(t([4.0, 9.0]), [2.0, 3.0])
        self.assertIs(as_transform(None), identity)
        self.assertIs(as_transform(t), t)

    def test_log_is_safe_and_invertible(self):
        y = np.array([1.0, 0.0, -2.0, np.e])
        out = log(y)
        self.assertEqual(out[0], 0.0)
        self.assertEqual(out[1], -np.inf)
        self.assertEqual(out[2], -np.inf)
        self.assertAlmostEqual(out[3], 1.0)
        np.testing.assert_allclose(log.derivative(np.array([2.0, 4.0])), [0.5, 0.25])
        self.assertIs(log.inverse, exp)
        self.assertIs(exp.inverse, log)
        np.testing.assert_allclose(exp(log(np.array([3.0, 7.0]))), [3.0, 7.0])

    def test_finite_difference_derivative_fallback(self):
        t = Transform(lambda a: a**3)
        np.testing.assert_allclose(
            t.derivative(np.array([1.0, 2.0])), [3.0, 12.0], rtol=1e-5
        )

    def test_compose_order_and_params(self):
        p = Parameter("c")
        shift = Transform(
            lambda a, c: a + c, (p,), derivative=lambda a, c: np.ones_like(a)
        )
        t = shift | log  # log(a + c)
        self.assertEqual(t.params, (p,))
        np.testing.assert_allclose(t(np.array([1.0]), 1.0), [np.log(2.0)])
        np.testing.assert_allclose(t.derivative(np.array([1.0]), 1.0), [0.5])
        self.assertIsNone(t.inverse)  # parametric -> no inverse
        u = log | exp
        np.testing.assert_allclose(u(np.array([2.0])), [2.0])
        self.assertIsNotNone(u.inverse)

    def test_wrong_value_count_raises(self):
        with self.assertRaises(ValueError):
            scale()(np.ones(2))


class TestScale(unittest.TestCase):
    def test_log_scale(self):
        t = scale()
        self.assertEqual(t.params[0].name, "log_rho")
        np.testing.assert_allclose(t(np.array([1.0, 2.0]), np.log(3.0)), [3.0, 6.0])
        np.testing.assert_allclose(
            t.derivative(np.array([1.0, 2.0]), np.log(3.0)), [3.0, 3.0]
        )

    def test_linear_scale_names(self):
        t = scale(log=False)
        self.assertEqual(t.params[0].name, "rho")
        np.testing.assert_allclose(t(np.array([1.0, 2.0]), 3.0), [3.0, 6.0])
        t2 = scale(Parameter("eta"), log=False)
        self.assertEqual(t2.params[0].name, "eta")


class TestPerObservationScaling(unittest.TestCase):
    def setUp(self):
        self.o1 = Observation(np.array([1.0]), np.array([1.0]))
        self.o2 = Observation(np.array([1.0]), np.array([1.0]))

    def test_routes_by_identity(self):
        t = per_observation_scaling([self.o1, self.o2])
        self.assertEqual([p.name for p in t.params], ["log_rho_0", "log_rho_1"])
        a = np.array([1.0, 2.0])
        np.testing.assert_allclose(
            t(a, np.log(2.0), np.log(5.0), context=self.o2), [5.0, 10.0]
        )
        np.testing.assert_allclose(
            t(a, np.log(2.0), np.log(5.0), context=self.o1), [2.0, 4.0]
        )
        with self.assertRaises(KeyError):
            t(a, 0.0, 0.0, context=Observation(np.array([1.0]), np.array([1.0])))

    def test_masked_view_routes_to_root(self):
        t = per_observation_scaling([self.o1, self.o2])
        view = self.o2.masked(np.array([False]))
        self.assertIs(view.identity, self.o2)
        a = np.array([1.0])
        np.testing.assert_allclose(t(a, 0.0, np.log(5.0), context=view), [5.0])
        # registering a view and its root is still a duplicate
        with self.assertRaises(ValueError):
            per_observation_scaling([self.o2, view])

    def test_missing_context_raises(self):
        t = per_observation_scaling([self.o1])
        with self.assertRaisesRegex(ValueError, "contextual"):
            t(np.array([1.0]), 0.0)

    def test_linear_and_custom_parameters(self):
        t = per_observation_scaling([self.o1], log=False)
        self.assertEqual(t.params[0].name, "rho_0")
        t2 = per_observation_scaling([self.o1], parameters=[Parameter("n")])
        self.assertEqual(t2.params[0].name, "n")
        with self.assertRaises(ValueError):
            per_observation_scaling([self.o1, self.o2], parameters=[Parameter("n")])
        with self.assertRaises(ValueError):
            per_observation_scaling([self.o1, self.o1])


if __name__ == "__main__":
    unittest.main()
