"""Unit tests for the stateless covariance terms (:mod:`rxmc.terms`)."""

import jax.numpy as jnp
import numpy as np
import pytest
import tinygp
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern

from helpers import (
    LOG_LENGTH_BOUNDS,
    STUDY_LEGEND,
    assemble_dense,
    matern52,
    rbf,
    study_form,
)
from rxmc import Parameter
from rxmc.terms import (
    KernelTerm,
    Term,
    TermContext,
    averaging,
    constant_amplitude,
    exp_growth,
    exp_growth_amplitude,
    kernel,
    kernel_params,
    noise,
    normalization,
    offset,
    ones,
    proportional_error,
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
        X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
        t = kernel(*rbf(), coords=lambda x: X, jitter=0.0)
        S = t.value(np.zeros(3), np.zeros(3), np.zeros(3), 0.0)
        assert np.allclose(S, RBF(1.0)(X))

    def test_fixed_kernel_with_parametric_coords(self):
        s = Parameter("s")
        coords = Transform(lambda a, s: a * s, (s,))
        x = np.array([0.0, 0.5, 1.0])
        for amp in (None, 0.3):
            t = kernel(tinygp.kernels.ExpSquared(), coords=coords, amplitude=amp)
            assert t.params == (s,) and not t.is_constant
            K = t.value(x, np.zeros(3), np.zeros(3), 2.0)
            a2 = 1.0 if amp is None else amp**2
            np.testing.assert_allclose(K, a2 * RBF(1.0)(2.0 * x[:, None]), atol=1e-8)

    def test_replace_keeps_one_copy_of_the_coords_parameters(self):
        from dataclasses import replace

        le, s, u = Parameter("le"), Parameter("s"), Parameter("u")
        t = noise(le, coords=Transform(lambda a, s: a * s, (s,)))
        x, y = np.array([1.0, 2.0]), np.zeros(2)
        moved = replace(t, on=None)
        assert moved.params == (le, s)
        np.testing.assert_allclose(moved.value(x, y, None, np.log(0.2), 3.0), 0.2)
        assert replace(t, coords=Transform(lambda a, u: a + u, (u,))).params == (le, u)
        le2 = Parameter("le2")
        assert replace(t, params=(le2,)).params == (le2, s)
        lA = Parameter("log_A")
        k = kernel(
            tinygp.kernels.ExpSquared(),
            amplitude=constant_amplitude,
            amplitude_params=(lA,),
            coords=Transform(lambda a, s: a * s, (s,)),
        )
        k2 = replace(k, on=None)
        assert k2.params == (lA, s)
        np.testing.assert_allclose(
            k2.value(x, y, y, 0.0, 1.0), k.value(x, y, y, 0.0, 1.0)
        )

    def test_non_numeric_x_reaches_a_callable_term(self):
        x = np.empty(3, dtype=object)
        x[:] = [(10.0, 0.1), (10.0, 0.2), (20.0, 0.1)]  # (E, theta) pairs
        t = Term(lambda c: np.array([e for e, _ in c.x]) / 100.0, kind="diag")
        np.testing.assert_allclose(t.value(x, np.ones(3), None), [0.1, 0.1, 0.2])
        le = Parameter("le")
        np.testing.assert_allclose(
            noise(le).value(x, np.ones(3), None, np.log(0.3)), 0.3
        )


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

    def test_unknown_proportional_error(self):
        S = self.S([proportional_error(Parameter("e"))], [(np.log(0.4),)])
        assert np.allclose(S, np.diag((0.4 * self.ym) ** 2))

    def test_unknown_normalization_error(self):
        S = self.S([normalization(parameter=Parameter("n"))], [(np.log(0.05),)])
        assert np.allclose(S, 0.05**2 * np.outer(self.ym, self.ym))

    def test_averaged_proportional_error(self):
        S = self.S(
            [proportional_error(Parameter("g"), averaging=True)], [(np.log(0.1),)]
        )
        z = 0.5 * (self.y + self.ym)
        assert np.allclose(S, np.diag((0.1 * z) ** 2))
        S = self.S(
            [proportional_error(Parameter("g"), averaging=False)], [(np.log(0.1),)]
        )
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
    def test_builder_arity_mismatch_fails_at_construction(self):
        params = kernel_params(["v", "l"], [(-3.0, 1.0), (-3.0, 1.0)])
        build = lambda log_ell: tinygp.kernels.ExpSquared(  # noqa: E731
            scale=jnp.exp(log_ell)
        )
        with pytest.raises(
            ValueError, match=r"takes \(log_ell\).*\['discrepancy_v', 'discrepancy_l'\]"
        ):
            kernel(build, params)

    def test_jitter_is_relative_to_each_points_own_variance(self):
        """The nugget is added before the amplitude scales the kernel, so a
        growing amplitude does not over-nugget its small end."""
        x = np.linspace(0.0, 1.0, 5)
        a = np.exp(4.0 * x)
        far = tinygp.kernels.ExpSquared(scale=1e-3)  # off-diagonal ~ 0
        K = np.asarray(kernel(far, amplitude=a, jitter=1e-3).value(x, x))
        np.testing.assert_allclose(np.diag(K), (1.0 + 1e-3) * a**2)

    def test_builder_params_are_the_terms_params(self):
        build, params = rbf()
        t = kernel(build, params)
        assert t.params == tuple(params) and t.n_kernel == 1
        assert [p.name for p in t.params] == ["discrepancy_length_scale"]
        np.testing.assert_allclose(t.params[0].bounds, LOG_LENGTH_BOUNDS)
        assert not t.is_constant

    def test_a_sum_of_kernels_in_one_builder(self):
        # sklearn's ConstantKernel * RBF + WhiteKernel, written as one builder
        lv, le, lw = kernel_params(["log_var", "log_ell", "log_white"])
        t = kernel(
            lambda v, e, w: (
                jnp.exp(v) * tinygp.kernels.ExpSquared(scale=jnp.exp(e))
                + tinygp.kernels.Constant(jnp.exp(w)) * tinygp.kernels.Cosine(1e9)
            ),
            [lv, le, lw],
            jitter=0.0,
        )
        x = np.linspace(0.0, 2.0, 4)
        S = t.value(x, np.zeros(4), np.zeros(4), np.log(2.0), np.log(0.5), -20.0)
        ref = ConstantKernel(2.0) * RBF(0.5)
        np.testing.assert_allclose(S, ref(x[:, None]), atol=1e-8)

    def test_anisotropic_metric_on_2d_inputs(self):
        l0, l1 = kernel_params(["ell0", "ell1"])
        t = kernel(
            # one length scale per input dimension: scale the inputs
            lambda a, b: tinygp.transforms.Linear(
                jnp.exp(-jnp.stack([a, b])), tinygp.kernels.ExpSquared()
            ),
            [l0, l1],
            jitter=0.0,
        )
        x2d = np.array([[0.0, 0.0], [1.0, 0.5], [2.0, 1.0]])
        S = t.value(x2d, np.zeros(3), np.zeros(3), np.log(1.0), np.log(0.5))
        np.testing.assert_allclose(S, RBF([1.0, 0.5])(x2d))

    def test_fixed_kernel_is_constant(self):
        t = kernel(tinygp.kernels.ExpSquared(scale=1.0))
        assert t.params == () and t.is_constant
        with pytest.raises(ValueError, match="fixed kernel takes no params"):
            kernel(tinygp.kernels.ExpSquared(scale=1.0), [Parameter("ell")])

    def test_shared_hyperparameters_via_params(self):
        build, params = rbf()
        t1, t2 = kernel(build, params), kernel(build, params)
        assert t1.params == t2.params == tuple(params)

    def test_constant_amplitude_reproduces_constant_kernel(self):
        x = np.linspace(0.0, 2.0, 5)
        A, la = 0.7, Parameter("log A")
        t = kernel(
            *rbf(), amplitude=constant_amplitude, amplitude_params=(la,), jitter=0.0
        )
        assert [p.name for p in t.params] == ["discrepancy_length_scale", "log A"]
        S = dense([t], x, np.zeros(5), np.zeros(5), [(0.0, np.log(A))])
        ref = ConstantKernel(A**2, constant_value_bounds="fixed") * RBF(1.0)
        np.testing.assert_allclose(S, ref(x[:, None]))

    def test_exp_growth_amplitude_and_coords(self):
        x = np.linspace(0.1, 3.0, 4)
        la, sl = Parameter("log A"), Parameter("slope")

        def q(x):
            return 2.0 * jnp.sin(x / 2)

        t = kernel(
            *matern52(),
            coords=q,
            amplitude=exp_growth_amplitude(np.pi),
            amplitude_params=(la, sl),
            jitter=0.0,
        )
        S = dense([t], x, np.zeros(4), np.zeros(4), [(np.log(0.5), np.log(0.3), 1.5)])
        qx = np.asarray(q(x))
        a = 0.3 * np.exp(1.5 * qx / np.pi)  # the amplitude sees the transformed x
        np.testing.assert_allclose(S, np.outer(a, a) * Matern(0.5, nu=2.5)(qx[:, None]))

    def test_duplicate_coords_factorizable_with_jitter(self):
        x = np.array([0.0, 0.0, 1.0])
        t = kernel(*rbf(), jitter=1e-8)
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
            lambda c, e, l: jnp.exp(e) * jnp.exp(l * c.x / np.pi), (e, sl), kind="diag"
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


class TestSegments:
    """A term's view of the comparisons its support spans."""

    def test_one_segment_outside_a_problem(self):
        seen = {}

        def fn(c):
            seen["segments"], seen["labels"] = c.segments, c.labels
            seen["split"] = c.split(c.x)
            return np.ones(len(c))

        Term(fn, kind="mode").value(np.arange(3.0), np.zeros(3), np.zeros(3))
        assert seen["segments"] == (slice(0, 3),) and seen["labels"] == ("",)
        np.testing.assert_array_equal(seen["split"][0], np.arange(3.0))

    def test_explicit_segments_and_split(self):
        def fn(c):
            assert c.labels == ("a", "b")
            parts = c.split(c.ym)
            assert [len(p) for p in parts] == [2, 3]
            return np.concatenate([p - p.mean() for p in parts])

        t = Term(fn, kind="mode")
        ym = np.array([1.0, 3.0, 10.0, 20.0, 30.0])
        v = t.value(
            np.zeros(5),
            np.zeros(5),
            ym,
            segments=[slice(0, 2), slice(2, 5)],
            labels=["a", "b"],
        )
        np.testing.assert_allclose(v, [-1.0, 1.0, -10.0, 0.0, 10.0])

    def test_split_checks_length(self):
        c = TermContext(x=np.zeros(3), y=np.zeros(3))
        with pytest.raises(ValueError, match="length 3"):
            c.split(np.zeros(4))


class TestKernelTerm:
    """The kernel factory returns a Term that also carries its kernel."""

    def test_fields(self):
        build, params = rbf()
        lA = Parameter("log_A")
        t = kernel(build, params, amplitude=constant_amplitude, amplitude_params=(lA,))
        assert isinstance(t, KernelTerm) and isinstance(t, Term)
        assert t.build is build and t.n_kernel == 1
        assert t.amplitude is constant_amplitude
        assert [p.name for p in t.params] == ["discrepancy_length_scale", "log_A"]
        assert t.kind == "matrix" and t.jitter == 1e-10
        # a fixed kernel has no kernel parameters and is constant
        fixed = kernel(tinygp.kernels.ExpSquared(scale=1.0))
        assert fixed.n_kernel == 0 and fixed.is_constant and fixed.params == ()


def test_a_relative_term_is_refused_outside_linear_space():
    """A fraction of the prediction is a relative error only in linear space:
    in log space it would scale with log(y), so it is refused there."""
    from scipy import stats

    from rxmc import Comparison, Constraint, Dataset, Model
    from rxmc import transforms as tf

    x = np.linspace(0.0, 1.0, 5)
    d = Dataset(x, np.exp(-x), 0.05 * np.exp(-x))
    a = Parameter("a", prior=stats.norm(1.0, 0.1))
    model = Model(lambda x, a: a * jnp.exp(-x), [a])
    eps = Parameter("log_eps", prior=stats.norm(-3.0, 1.0))
    lin, log = Comparison(d, model), Comparison(d, model, space=tf.log)
    Constraint([lin], terms=[proportional_error(eps)])  # linear space: fine
    Constraint([log], terms=[noise(eps)])  # the relative error in log space
    for term in (proportional_error(eps), normalization(eps)):
        with pytest.raises(ValueError, match="relative error only in the identity"):
            Constraint([log], terms=[term])
    with pytest.raises(ValueError, match="relative"):  # on= picks the log one
        Constraint([lin, log], terms=[proportional_error(eps, on=log)])
    Constraint([lin, log], terms=[proportional_error(eps, on=lin)])
    with pytest.raises(ValueError, match="relative"):  # on=None spans both
        Constraint([lin, log], terms=[proportional_error(eps)])
    d2 = Dataset(x, np.exp(-x), 0.05 * np.exp(-x))
    with pytest.raises(ValueError, match="relative"):  # on= a dataset
        Constraint(
            [lin, Comparison(d2, model, space=tf.log)],
            terms=[normalization(eps, on=d2)],
        )
