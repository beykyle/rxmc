"""Models, predictors and their composition."""

import numpy as np
import pytest

from rxmc import Model, Parameter, polynomial
from rxmc.model import Predictor
from rxmc.transforms import Transform, scale

x = np.linspace(0.0, 2.0, 5)
m, b = Parameter("m"), Parameter("b")
line = Model(lambda x, m, b: m * x + b, [m, b])


def test_generic_model_binds_to_x():
    pred = line.bind(x)
    assert isinstance(pred, Predictor)
    assert pred.params == (m, b)
    np.testing.assert_allclose(pred(2.0, 1.0), 2.0 * x + 1.0)
    with pytest.raises(ValueError, match="expects 2 value"):
        pred(2.0)
    with pytest.raises(TypeError, match="Parameter"):
        Model(lambda x, a: a * x, ["a"])


def test_polynomial():
    poly = polynomial(2)
    assert [p.name for p in poly.params] == ["a0", "a1", "a2"]
    np.testing.assert_allclose(
        poly.bind(x)(1.0, 2.0, 3.0), np.polyval([3.0, 2.0, 1.0], x)
    )


def test_scale_transform_appends_and_scales_after():
    rho = Parameter("log_rho")
    scaled = line | scale(rho)
    assert scaled.params == (m, b, rho)
    np.testing.assert_allclose(
        scaled.bind(x)(2.0, 1.0, np.log(3.0)), 3.0 * (2.0 * x + 1.0)
    )


def test_two_parameter_transform_in_order():
    c, d = Parameter("c"), Parameter("d")
    t = Transform(lambda a, c, d: c * a + d, (c, d))
    model = line | t
    assert model.params == (m, b, c, d)
    np.testing.assert_allclose(model.bind(x)(1.0, 0.0, 2.0, 5.0), 2.0 * x + 5.0)


def test_add_and_mul_compose_on_the_bound_grid():
    c0 = Parameter("c0")
    delta = Model(lambda x, c0: c0 * x**2, [c0])
    both = line + delta
    assert both.params == (m, b, c0)
    np.testing.assert_allclose(both.bind(x)(1.0, 1.0, 0.5), x + 1.0 + 0.5 * x**2)
    prod = line * delta
    np.testing.assert_allclose(prod.bind(x)(1.0, 1.0, 0.5), (x + 1.0) * 0.5 * x**2)
    with pytest.raises(TypeError, match="Model"):
        line + 3.0


def test_precedence_of_transform_and_addition():
    c0, rho = Parameter("c0"), Parameter("log_rho")
    delta = Model(lambda x, c0: c0 * np.ones_like(x), [c0])
    scale_sum = (line + delta) | scale(rho)
    scale_model = (line | scale(rho)) + delta
    v_sum = scale_sum.bind(x)(1.0, 0.0, 2.0, np.log(3.0))  # m, b, c0, rho
    v_model = scale_model.bind(x)(1.0, 0.0, np.log(3.0), 2.0)  # m, b, rho, c0
    np.testing.assert_allclose(v_sum, 3.0 * (x + 2.0))
    np.testing.assert_allclose(v_model, 3.0 * x + 2.0)


def test_mul_by_constant_equals_scale():
    rho = Parameter("log_rho")
    const = Model(lambda x, r: np.exp(r) * np.ones_like(x), [rho])
    np.testing.assert_allclose(
        (line * const).bind(x)(2.0, 1.0, 0.7),
        (line | scale(rho)).bind(x)(2.0, 1.0, 0.7),
    )


def test_shared_parameter_is_concatenated_not_deduplicated():
    # the same object on both sides is one slot at compile; the model just lists it twice
    both = line + Model(lambda x, m: m * np.ones_like(x), [m])
    assert both.params == (m, b, m)
    np.testing.assert_allclose(both.bind(x)(1.0, 0.0, 1.0), x + 1.0)


def test_model_may_ignore_x_and_close_over_another_predictor():
    native = line.bind(x)
    A = np.array([[1.0, 1.0, 1.0, 1.0, 1.0], [0.0, 1.0, 2.0, 3.0, 4.0]])
    proj = Model(lambda x_pc, *theta: A @ native(*theta), line.params)
    pc = proj.bind(np.arange(2))
    np.testing.assert_allclose(pc(2.0, 1.0), A @ (2.0 * x + 1.0))


def test_subclass_overrides_bind_and_composes():
    class Doubler(Model):
        def __init__(self):
            super().__init__(None, [m])

        def bind(self, x, meta=None):
            k = meta["k"]
            return Predictor(self.params, x, lambda mm: k * mm * np.asarray(x))

    model = Doubler() | scale(Parameter("log_rho"))
    pred = model.bind(x, {"k": 2.0})
    np.testing.assert_allclose(pred(1.5, 0.0), 3.0 * x)
    with pytest.raises(TypeError, match="override bind"):
        Model(None, [m]).bind(x)
