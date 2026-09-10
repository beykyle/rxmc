"""Tests for the low-level ``rxmc.transforms`` type."""

import numpy as np
import pytest

from rxmc import Parameter
from rxmc.transforms import Transform, as_transform, exp, identity, log, scale


class TestTransform:
    def test_callable_is_wrapped_parameter_free(self):
        t = as_transform(np.sqrt)
        assert isinstance(t, Transform)
        assert t.params == ()
        np.testing.assert_allclose(t([4.0, 9.0]), [2.0, 3.0])
        assert as_transform(None) is identity
        assert as_transform(t) is t

    def test_log_is_safe_and_invertible(self):
        y = np.array([1.0, 0.0, -2.0, np.e])
        out = log(y)
        assert out[0] == 0.0
        assert out[1] == -np.inf
        assert out[2] == -np.inf
        assert out[3] == pytest.approx(1.0)
        np.testing.assert_allclose(log.derivative(np.array([2.0, 4.0])), [0.5, 0.25])
        assert log.inverse is exp
        assert exp.inverse is log
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
        assert t.params == (p,)
        np.testing.assert_allclose(t(np.array([1.0]), 1.0), [np.log(2.0)])
        np.testing.assert_allclose(t.derivative(np.array([1.0]), 1.0), [0.5])
        assert t.inverse is None  # parametric -> no inverse
        u = log | exp
        np.testing.assert_allclose(u(np.array([2.0])), [2.0])
        assert u.inverse is not None

    def test_wrong_value_count_raises(self):
        with pytest.raises(ValueError, match="expects 1 value"):
            scale()(np.ones(2))

    def test_params_must_be_parameters(self):
        with pytest.raises(TypeError, match="Parameter"):
            Transform(lambda a, c: a + c, ("c",))

    def test_no_context_keyword(self):
        with pytest.raises(TypeError):
            identity(np.ones(2), context=object())


class TestScale:
    def test_log_scale(self):
        t = scale()
        assert t.params[0].name == "log_rho"
        np.testing.assert_allclose(t(np.array([1.0, 2.0]), np.log(3.0)), [3.0, 6.0])
        np.testing.assert_allclose(
            t.derivative(np.array([1.0, 2.0]), np.log(3.0)), [3.0, 3.0]
        )

    def test_linear_scale_names(self):
        t = scale(log=False)
        assert t.params[0].name == "rho"
        np.testing.assert_allclose(t(np.array([1.0, 2.0]), 3.0), [3.0, 6.0])
        t2 = scale(Parameter("eta"), log=False)
        assert t2.params[0].name == "eta"
