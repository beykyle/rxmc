import unittest

import numpy as np

from rxmc.params import Parameter


class TestParameterHashing(unittest.TestCase):
    def test_equal_parameters_hash_equal(self):
        a = Parameter("g", float, unit="MeV", latex_name="g", bounds=(0.0, 1.0))
        b = Parameter("g", float, unit="MeV", latex_name="g", bounds=(0.0, 1.0))
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))

    def test_unequal_parameters_differ(self):
        a = Parameter("g")
        self.assertNotEqual(a, Parameter("h"))
        self.assertNotEqual(a, Parameter("g", unit="MeV"))
        self.assertNotEqual(a, Parameter("g", bounds=(0.0, 1.0)))
        self.assertNotEqual(a, "g")

    def test_usable_in_set_and_dict(self):
        a = Parameter("a")
        b = Parameter("b")
        self.assertEqual(len({a, b, Parameter("a")}), 2)
        table = {a: 1, b: 2}
        self.assertEqual(table[Parameter("a")], 1)

    def test_bounds_coerced_to_float_tuple(self):
        for bounds in ([0, 2], np.array([0.0, 2.0]), (0, 2)):
            p = Parameter("x", bounds=bounds)
            self.assertEqual(p.bounds, (0.0, 2.0))
            self.assertIsInstance(p.bounds, tuple)
            self.assertTrue(all(isinstance(b, float) for b in p.bounds))

    def test_default_bounds_are_infinite(self):
        self.assertEqual(Parameter("x").bounds, (-np.inf, np.inf))

    def test_bad_bounds_length_raises(self):
        with self.assertRaises(ValueError):
            Parameter("x", bounds=(0.0, 1.0, 2.0))

    def test_repr_contains_fields(self):
        r = repr(Parameter("V", float, unit="MeV", latex_name=r"V_0", bounds=(0, 9)))
        self.assertIn("'V'", r)
        self.assertIn("MeV", r)
        self.assertIn("V_0", r)
        self.assertIn("(0.0, 9.0)", r)


if __name__ == "__main__":
    unittest.main()
