"""Unit tests for the stateless covariance terms (:mod:`rxmc.terms`)."""

import numpy as np
import pytest
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel

from helpers import STUDY_LEGEND, assemble_dense, study_form
from rxmc import Parameter
from rxmc.terms import (
    Term,
    TermContext,
    averaging,
    constant_amplitude,
    exp_growth,
    exp_growth_amplitude,
    kernel,
    model_error,
    noise,
    noise_fraction,
    normalization,
    offset,
    ones,
    statistical,
    systematic,
    x_basis,
    ym,
)
from rxmc.transforms import Transform


def dense(terms, x, y, ym_, values=()):
    return assemble_dense(terms, x, y, ym_, values)


# ----------------------------------------------------------------------------
# Term
# ----------------------------------------------------------------------------


class TestTermKinds:
    def test_diag_array_squares_std(self):
        t = Term(np.array([1.0, 2.0, 3.0]), kind="diag")
        S = dense([t], np.zeros(3), np.zeros(3), np.zeros(3))
        assert np.allclose(S, np.diag([1.0, 4.0, 9.0]))
        assert t.is_constant
        assert not t.couples_offdiagonal

    def test_mode_array_outer(self):
        v = np.array([1.0, 2.0])
        t = Term(v, kind="mode")
        assert np.allclose(dense([t], np.zeros(2), np.zeros(2), None), np.outer(v, v))
        assert t.couples_offdiagonal

    def test_matrix_array_passthrough(self):
        m = np.array([[2.0, 0.5], [0.5, 3.0]])
        t = Term(m)
        assert np.allclose(dense([t], np.zeros(2), np.zeros(2), None), m)

    def test_bad_kind_raises(self):
        with pytest.raises(ValueError, match="kind"):
            Term(np.ones(2), kind="rank1")

    def test_scalar_broadcast_raises(self):
        # a (1, 1) matrix on a length-3 support must not broadcast silently
        t = Term([[0.04]])
        with pytest.raises(ValueError, match="expects shape"):
            t.value(np.zeros(3), np.zeros(3), np.zeros(3))

    def test_wrong_length_vector_raises(self):
        with pytest.raises(ValueError, match="expects shape"):
            Term(np.ones(2), kind="diag").value(np.zeros(3), np.zeros(3), np.zeros(3))

    def test_asymmetric_matrix_raises(self):
        with pytest.raises(ValueError, match="symmetric"):
            Term(np.array([[1.0, 0.2], [0.0, 1.0]]))

    def test_non_square_matrix_raises(self):
        with pytest.raises(ValueError, match="square"):
            Term(np.ones((2, 3)))

    def test_array_with_params_raises(self):
        with pytest.raises(ValueError, match="array-valued"):
            Term(np.ones(2), (Parameter("p"),), kind="diag")

    def test_non_parameter_raises(self):
        with pytest.raises(TypeError, match="Parameter"):
            Term(lambda c, a: ones(c), ("a",), kind="diag")

    def test_callable_sees_context_and_values(self):
        seen = {}

        def fn(c, a, b):
            seen["c"] = c
            return a * c.ym + b * c.y

        pa, pb = Parameter("a"), Parameter("b")
        t = Term(fn, (pa, pb), kind="diag")
        x, y, ym_ = np.array([1.0, 2.0]), np.array([2.0, 3.0]), np.array([2.5, 3.5])
        v = t.value(x, y, ym_, 2.0, 1.0)
        c = seen["c"]
        assert isinstance(c, TermContext)
        assert np.allclose(c.x, x) and np.allclose(c.ym, ym_) and len(c) == 2
        assert np.allclose(v, 2.0 * ym_ + y)

    def test_callable_wrong_shape_raises(self):
        t = Term(lambda c: np.ones(len(c) + 1), kind="diag")
        with pytest.raises(ValueError, match="returned shape"):
            t.value(np.zeros(2), np.zeros(2), np.zeros(2))

    def test_wrong_param_count_raises(self):
        t = Term(lambda c, a: a * ones(c), (Parameter("a"),), kind="diag")
        with pytest.raises(ValueError, match="expected 1 params"):
            t.value(np.zeros(1), np.zeros(1), np.zeros(1))

    def test_constant_with_params_raises(self):
        with pytest.raises(ValueError, match="constant"):
            Term(lambda c, a: ones(c), (Parameter("a"),), kind="diag", constant=True)

    def test_constant_term_may_read_x_and_is_evaluated_without_ym(self):
        x = np.linspace(0.0, 2.0, 4)
        t = Term(lambda c: 0.1 * c.x, kind="diag", constant=True)
        assert t.is_constant
        np.testing.assert_allclose(t.value(x, np.zeros(4), None), 0.1 * x)
        bad = Term(lambda c: 0.1 * c.ym, kind="diag", constant=True)
        with pytest.raises(TypeError):
            bad.value(x, np.zeros(4), None)

    def test_meta_accessor(self):
        t = Term(lambda c: c.meta("E") * ones(c), kind="diag", constant=True)
        v = t.value(np.zeros(2), np.zeros(2), None, meta={"E": np.array([5.0, 5.0])})
        assert np.allclose(v, 5.0)
        with pytest.raises(KeyError, match="Dataset.meta"):
            t.value(np.zeros(2), np.zeros(2), None)

    def test_repr(self):
        assert "log_e" in repr(noise(Parameter("log_e")))


class TestTermCoords:
    def test_coords_callable_applied_to_x(self):
        t = Term(lambda c: c.x, kind="diag", coords=lambda x: 2 * x)
        assert np.allclose(t.value(np.array([1.0, 3.0]), np.zeros(2), None), [2.0, 6.0])

    def test_parametric_coords_params_appended(self):
        pk = Parameter("k")
        coords = Transform(lambda x, k: k * x, (pk,))
        pa = Parameter("a")
        t = Term(lambda c, a: a * c.x, (pa,), kind="diag", coords=coords)
        assert t.params == (pa, pk)
        v = t.value(np.array([1.0, 2.0]), np.zeros(2), None, 3.0, 2.0)  # a=3, k=2
        assert np.allclose(v, 6.0 * np.array([1.0, 2.0]))

    def test_coords_array_2d_reaches_kernel(self):
        k = RBF(length_scale=1.0)
        X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
        t = kernel(k, coords=lambda x: X, jitter=0.0)
        S = t.value(np.zeros(3), np.zeros(3), np.zeros(3), *k.theta)
        assert np.allclose(S, k(X))


# ----------------------------------------------------------------------------
# Factories
# ----------------------------------------------------------------------------


class TestFactories:
    def setup_method(self):
        self.x = np.array([0.5, 1.0, 1.5])
        self.y = np.array([1.0, 2.0, 3.0])
        self.ym = np.array([1.1, 1.9, 3.2])
        self.stat = np.array([0.1, 0.2, 0.3])

    def S(self, terms, values=()):
        return dense(terms, self.x, self.y, self.ym, values)

    def test_statistical_only(self):
        assert np.allclose(self.S([statistical(self.stat)]), np.diag(self.stat**2))

    def test_unknown_noise(self):
        assert np.allclose(
            self.S([noise(Parameter("e"))], [(np.log(0.4),)]), 0.16 * np.eye(3)
        )
        assert np.allclose(
            self.S([noise(Parameter("e"), log=False)], [(0.4,)]), 0.16 * np.eye(3)
        )

    def test_unknown_noise_fraction(self):
        S = self.S([noise_fraction(Parameter("e"))], [(np.log(0.4),)])
        assert np.allclose(S, np.diag((0.4 * self.ym) ** 2))

    def test_unknown_normalization_error(self):
        S = self.S([normalization(parameter=Parameter("n"))], [(np.log(0.05),)])
        assert np.allclose(S, 0.05**2 * np.outer(self.ym, self.ym))

    def test_unknown_model_error(self):
        S = self.S([model_error(Parameter("g"), averaging=True)], [(np.log(0.1),)])
        z = 0.5 * (self.y + self.ym)
        assert np.allclose(S, np.diag((0.1 * z) ** 2))
        S = self.S([model_error(Parameter("g"), averaging=False)], [(np.log(0.1),)])
        assert np.allclose(S, np.diag((0.1 * self.ym) ** 2))

    def test_fixed_normalization_systematic(self):
        for mag in (0.05, np.array(0.05)):
            S = self.S([normalization(magnitude=mag)])
            assert np.allclose(S, 0.05**2 * np.outer(self.ym, self.ym))

    def test_fixed_offset_systematic(self):
        t = offset(magnitude=0.2)
        assert t.is_constant
        assert np.allclose(self.S([t]), 0.04 * np.ones((3, 3)))
        v = np.array([0.1, 0.2, 0.3])
        assert np.allclose(self.S([offset(magnitude=v)]), np.outer(v, v))

    def test_fixed_offset_is_evaluated_without_ym(self):
        t = offset(magnitude=0.2)
        np.testing.assert_allclose(t.value(self.x, self.y, None), 0.2)

    def test_masked_magnitudes(self):
        m = np.array([1.0, 0.0, 1.0])
        S = self.S([offset(magnitude=0.2, mask=m)])
        assert np.allclose(S, np.outer(0.2 * m, 0.2 * m))
        S = self.S([normalization(parameter=Parameter("n"), mask=m)], [(np.log(0.5),)])
        v = 0.5 * m * self.ym
        assert np.allclose(S, np.outer(v, v))

    def test_length_one_magnitude_or_mask_raises(self):
        with pytest.raises(ValueError, match="shape"):
            self.S([offset(magnitude=np.array([0.2]))])
        with pytest.raises(ValueError, match="shape"):
            self.S([offset(magnitude=0.2, mask=np.array([1.0]))])
        with pytest.raises(ValueError, match="shape"):
            self.S(
                [normalization(parameter=Parameter("n"), mask=np.array([1.0]))],
                [(0.0,)],
            )

    def test_magnitude_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            self.S([offset(magnitude=np.ones(2))])

    def test_requires_magnitude_or_parameter(self):
        with pytest.raises(ValueError):
            offset()
        with pytest.raises(ValueError):
            normalization()

    def test_systematic_with_basis(self):
        S = self.S([systematic(Parameter("s"), basis=x_basis(2.0))], [(np.log(3.0),)])
        v = 3.0 * self.x / 2.0
        assert np.allclose(S, np.outer(v, v))

    def test_parametric_basis(self):
        e, sl = Parameter("log e"), Parameter("slope")
        t = noise(e, basis=exp_growth(np.pi), basis_params=(sl,))
        assert t.params == (e, sl)
        assert np.allclose(self.S([t], [(np.log(0.3), 0.0)]), 0.09 * np.eye(3))
        S = self.S([t], [(np.log(0.3), 2.0)])
        assert np.allclose(S, np.diag((0.3 * np.exp(2.0 * self.x / np.pi)) ** 2))
        t = noise(e, basis=exp_growth(np.pi, base=ym), basis_params=(sl,))
        S = self.S([t], [(np.log(0.3), 1.0)])
        assert np.allclose(S, np.diag((0.3 * self.ym * np.exp(self.x / np.pi)) ** 2))

    def test_old_observation_covariance_equivalence(self):
        off, norm = 0.2, 0.05
        terms = [
            statistical(self.stat),
            offset(magnitude=off),
            normalization(magnitude=norm),
        ]
        old = (
            np.diag(self.stat**2)
            + np.outer(off * np.ones(3), off * np.ones(3))
            + norm**2 * np.outer(self.ym, self.ym)
        )
        assert np.allclose(self.S(terms), old)

    def test_bases(self):
        c = TermContext(x=self.x, y=self.y, ym=self.ym)
        assert np.allclose(ones(c), 1.0)
        assert np.allclose(ym(c), self.ym)
        assert np.allclose(averaging(c), 0.5 * (self.y + self.ym))


class TestKernel:
    def test_params_match_theta_length_isotropic(self):
        k = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(1e-6)
        t = kernel(k)
        assert len(t.params) == len(k.theta)
        assert not t.is_constant

    def test_params_anisotropic(self):
        k = ConstantKernel(1.0) * RBF(length_scale=[1.0, 1.0])
        t = kernel(k)
        assert len(t.params) == len(k.theta)
        x2d = np.array([[0.0, 0.0], [1.0, 0.5], [2.0, 1.0]])
        S = t.value(x2d, np.zeros(3), np.zeros(3), *k.theta)
        assert np.all(np.isfinite(S))

    def test_fixed_kernel_is_constant(self):
        t = kernel(RBF(length_scale=1.0, length_scale_bounds="fixed"))
        assert t.params == () and t.is_constant

    def test_shared_hyperparameters_via_params(self):
        ell = Parameter("gp_length")
        t1, t2 = kernel(RBF(1.0), params=[ell]), kernel(RBF(1.0), params=[ell])
        assert t1.params == (ell,) and t2.params == (ell,)
        with pytest.raises(ValueError, match="free hyperparameter"):
            kernel(RBF(1.0), params=[ell, Parameter("extra")])

    def test_constant_amplitude_reproduces_constant_kernel(self):
        x = np.linspace(0.0, 2.0, 5)
        A, la = 0.7, Parameter("log A")
        t = kernel(
            RBF(1.0), amplitude=constant_amplitude, amplitude_params=(la,), jitter=0.0
        )
        assert [p.name for p in t.params] == ["discrepancy_length_scale", "log A"]
        S = dense([t], x, np.zeros(5), np.zeros(5), [(0.0, np.log(A))])
        ref = ConstantKernel(A**2, constant_value_bounds="fixed") * RBF(1.0)
        np.testing.assert_allclose(S, ref(x[:, None]))

    def test_exp_growth_amplitude_and_coords(self):
        x = np.linspace(0.1, 3.0, 4)
        la, sl = Parameter("log A"), Parameter("slope")

        def q(x):
            return 2.0 * np.sin(x / 2)

        t = kernel(
            Matern(1.0, nu=2.5),
            coords=q,
            amplitude=exp_growth_amplitude(np.pi),
            amplitude_params=(la, sl),
            jitter=0.0,
        )
        S = dense([t], x, np.zeros(4), np.zeros(4), [(np.log(0.5), np.log(0.3), 1.5)])
        a = 0.3 * np.exp(1.5 * q(x) / np.pi)  # the amplitude sees the transformed x
        np.testing.assert_allclose(
            S, np.outer(a, a) * Matern(0.5, nu=2.5)(q(x)[:, None])
        )

    def test_duplicate_coords_factorizable_with_jitter(self):
        x = np.array([0.0, 0.0, 1.0])
        t = kernel(RBF(1.0), jitter=1e-8)
        S = t.value(x, np.zeros(3), np.zeros(3), 0.0)
        assert np.all(np.isfinite(np.linalg.cholesky(S)))


# ----------------------------------------------------------------------------
# The alpha+Ca error-model ladder as one-line term lists
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("label", list(STUDY_LEGEND))
def test_study_forms(label):
    """Each error model of the motivating study (elastic alpha + Ca scattering
    data with no reported uncertainties, compared in log space) is one term
    list; the term values assemble to the hand-built dense covariance.  The
    labels are defined in ``helpers.STUDY_LEGEND``."""
    rng = np.random.default_rng(1)
    n = 12
    x = np.sort(rng.uniform(0.2, 3.0, n))  # radians
    y = rng.uniform(0.1, 1.5, n)
    ym_ = y + rng.normal(0.0, 0.1, n)
    form = study_form(label, x, y, ym_)
    assert form.description  # every label has a legend entry
    S = dense(form.terms, x, y, ym_, form.values)
    assert np.allclose(S, form.dense)


# ----------------------------------------------------------------------------
# Term-level halves of recipes (the rest of each recipe needs a Problem)
# ----------------------------------------------------------------------------


class TestRecipeHalves:
    def test_recipe_19_bring_your_own_term(self):
        x = np.linspace(0.0, 1.0, 4)
        C = np.exp(-np.abs(x[:, None] - x[None, :]))  # symmetric, PD
        fixed = Term(C)
        assert fixed.is_constant and np.allclose(fixed.value(x, np.zeros(4), None), C)
        sig = np.array([0.1, 0.2, 0.3, 0.4])
        stat = Term(sig, kind="diag")
        assert np.allclose(dense([stat], x, np.zeros(4), None), np.diag(sig**2))
        e, sl = Parameter("log_e"), Parameter("slope")
        custom = Term(
            lambda c, e, l: np.exp(e) * np.exp(l * c.x / np.pi), (e, sl), kind="diag"
        )
        helper = noise(e, basis=exp_growth(np.pi), basis_params=(sl,))
        v = (np.log(0.3), 1.1)
        np.testing.assert_allclose(
            custom.value(x, np.zeros(4), None, *v),
            helper.value(x, np.zeros(4), None, *v),
        )

    def test_recipe_27_normalization_mode_is_built_from_the_prediction(self):
        x = np.linspace(0.0, 1.0, 3)
        y, ym_ = np.array([1.0, 2.0, 3.0]), np.array([1.2, 1.8, 3.3])
        s = 0.1
        right = normalization(magnitude=s)
        wrong = Term(
            lambda c: s * c.y, kind="mode", constant=True
        )  # data-built: Peelle
        np.testing.assert_allclose(right.value(x, y, ym_), s * ym_)
        np.testing.assert_allclose(wrong.value(x, y, ym_), s * y)
        assert not np.allclose(right.value(x, y, ym_), wrong.value(x, y, ym_))
