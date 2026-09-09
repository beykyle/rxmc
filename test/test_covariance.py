"""Unit tests for the stacked-covariance core (:mod:`rxmc.covariance`)."""

import numpy as np
import pytest
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel

from helpers import make_ctx
from rxmc.covariance import (
    ConstraintCovariance,
    StackContext,
    Term,
    averaging,
    constant_amplitude,
    exp_growth,
    exp_growth_amplitude,
    kernel_term,
    model_error_term,
    noise_fraction_term,
    noise_term,
    normalization_term,
    offset_term,
    ones,
    statistical_term,
    systematic_term,
    x_basis,
    ym,
)
from rxmc.likelihood_model import mahalanobis_distance_sqr_cholesky
from rxmc.params import Parameter
from rxmc.transforms import Transform


def single_block_ctx(x, y, ym):
    n = len(x)
    return make_ctx(x, y, ym, [np.arange(n)])


def assemble(terms, ctx, theta=()):
    cov = ConstraintCovariance(terms, len(ctx.x), blocks=ctx.supports)
    return cov.matrix(ctx, *theta)


# ----------------------------------------------------------------------------
# Term
# ----------------------------------------------------------------------------


class TestTermKinds:
    def test_diag_array_squares_std(self):
        t = Term(np.array([1.0, 2.0, 3.0]), kind="diag", support=[0, 1, 2])
        S = np.zeros((3, 3))
        t.add_to(S, None, ())
        assert np.allclose(S, np.diag([1.0, 4.0, 9.0]))
        assert t.is_constant
        assert not t.couples_offdiagonal

    def test_mode_array_outer(self):
        v = np.array([1.0, 2.0])
        t = Term(v, kind="mode", support=[0, 1])
        S = np.zeros((2, 2))
        t.add_to(S, None, ())
        assert np.allclose(S, np.outer(v, v))
        assert t.couples_offdiagonal

    def test_matrix_array_passthrough_into_subblock(self):
        m = np.array([[2.0, 0.5], [0.5, 3.0]])
        t = Term(m, support=[2, 3])
        S = np.zeros((4, 4))
        t.add_to(S, None, ())
        expected = np.zeros((4, 4))
        expected[2:, 2:] = m
        assert np.allclose(S, expected)

    def test_bad_kind_raises(self):
        with pytest.raises(ValueError, match="kind"):
            Term(np.ones(2), kind="rank1", support=[0, 1])

    def test_scalar_broadcast_raises(self):
        # a (1, 1) matrix on a length-3 support used to broadcast silently
        with pytest.raises(ValueError, match="expects shape"):
            Term([[0.04]], support=np.arange(3))

    def test_wrong_length_vector_raises(self):
        with pytest.raises(ValueError, match="expects shape"):
            Term(np.ones(2), kind="diag", support=np.arange(3))

    def test_asymmetric_matrix_raises(self):
        with pytest.raises(ValueError, match="symmetric"):
            Term(np.array([[1.0, 0.2], [0.0, 1.0]]), support=[0, 1])

    def test_array_with_params_raises(self):
        with pytest.raises(ValueError, match="array-valued"):
            Term(np.ones(2), (Parameter("p"),), kind="diag", support=[0, 1])

    def test_callable_sees_local_context_and_values(self):
        seen = {}

        def fn(c, a, b):
            seen["c"] = c
            return a * c.ym + b * c.y

        pa, pb = Parameter("a"), Parameter("b")
        t = Term(fn, (pa, pb), kind="diag", support=[1, 2])
        ctx = single_block_ctx([0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [1.5, 2.5, 3.5])
        S = np.zeros((3, 3))
        t.add_to(S, ctx, (2.0, 1.0))
        c = seen["c"]
        assert np.allclose(c.x, [1.0, 2.0])
        assert np.allclose(c.ym, [2.5, 3.5])
        assert len(c) == 2
        v = 2.0 * np.array([2.5, 3.5]) + np.array([2.0, 3.0])
        assert np.allclose(np.diag(S), [0.0, *(v**2)])

    def test_callable_wrong_shape_raises(self):
        t = Term(lambda c: np.ones(len(c) + 1), kind="diag", support=[0, 1])
        ctx = single_block_ctx([0.0, 1.0], [0.0, 0.0], [0.0, 0.0])
        with pytest.raises(ValueError, match="returned shape"):
            t.add_to(np.zeros((2, 2)), ctx, ())

    def test_wrong_param_count_raises(self):
        t = Term(lambda c, a: a * ones(c), (Parameter("a"),), kind="diag", support=[0])
        ctx = single_block_ctx([0.0], [0.0], [0.0])
        with pytest.raises(ValueError, match="expected 1 params"):
            t.add_to(np.zeros((1, 1)), ctx, ())

    def test_constant_callable_cached(self):
        calls = []

        def fn(c):
            calls.append(1)
            return np.ones(len(c))

        t = Term(fn, kind="diag", support=[0, 1], constant=True)
        assert t.is_constant
        ctx = single_block_ctx([0.0, 1.0], [0.0, 0.0], [0.0, 0.0])
        t.add_to(np.zeros((2, 2)), ctx, ())
        t.add_to(np.zeros((2, 2)), ctx, ())
        assert len(calls) == 1

    def test_constant_flag_ignored_with_params(self):
        t = Term(lambda c, a: ones(c), (Parameter("a"),), kind="diag", constant=True)
        assert not t.is_constant


class TestTermCoords:
    def test_coords_callable_applied_to_x(self):
        t = Term(lambda c: c.x, kind="diag", support=[0, 1], coords=lambda x: 2 * x)
        ctx = single_block_ctx([1.0, 3.0], [0.0, 0.0], [0.0, 0.0])
        assert np.allclose(t.local_context(ctx).x, [2.0, 6.0])

    def test_parametric_coords_params_appended(self):
        pk = Parameter("k")
        coords = Transform(lambda x, k: k * x, (pk,))
        pa = Parameter("a")
        t = Term(
            lambda c, a: a * c.x, (pa,), kind="diag", support=[0, 1], coords=coords
        )
        assert t.params == (pa, pk)
        ctx = single_block_ctx([1.0, 2.0], [0.0, 0.0], [0.0, 0.0])
        S = np.zeros((2, 2))
        t.add_to(S, ctx, (3.0, 2.0))  # a=3, k=2 -> v = 3 * 2 * x
        assert np.allclose(np.diag(S), (6.0 * np.array([1.0, 2.0])) ** 2)

    def test_coords_array_2d_reaches_kernel(self):
        kernel = RBF(length_scale=1.0)
        X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
        t = kernel_term(kernel, coords=lambda x: X, jitter=0.0, support=np.arange(3))
        ctx = single_block_ctx(np.zeros(3), np.zeros(3), np.zeros(3))
        S = np.zeros((3, 3))
        t.add_to(S, ctx, kernel.theta)
        assert np.allclose(S, kernel(X))


class TestSupportNone:
    def test_bound_by_constraint_covariance(self):
        p = Parameter("log eps")
        t = noise_term(p)
        assert not t.bound
        cov = ConstraintCovariance([t], 3)
        assert t.bound and np.array_equal(t.support, np.arange(3))
        ctx = single_block_ctx(np.zeros(3), np.zeros(3), np.zeros(3))
        assert np.allclose(cov.matrix(ctx, np.log(2.0)), 4.0 * np.eye(3))

    def test_unbound_add_to_raises(self):
        t = noise_term(Parameter("p"))
        with pytest.raises(ValueError, match="unresolved"):
            t.add_to(np.zeros((2, 2)), None, (0.0,))

    def test_bind_idempotent_and_explicit_support_untouched(self):
        t = Term(np.ones(2), kind="diag", support=[1, 2])
        t.bind(5)
        assert np.array_equal(t.support, [1, 2])
        u = Term(np.ones(3), kind="diag")
        u.bind(3)
        u.bind(3)
        assert np.array_equal(u.support, np.arange(3))

    def test_array_length_checked_at_bind(self):
        t = Term(np.ones(2), kind="diag")
        with pytest.raises(ValueError, match="expects shape"):
            ConstraintCovariance([t], 3)

    def test_whole_stack_mode_block_diagonality(self):
        one = ConstraintCovariance(
            [offset_term(parameter=Parameter("w"))], 3, blocks=[np.arange(3)]
        )
        assert one.block_diagonal
        two = ConstraintCovariance(
            [offset_term(parameter=Parameter("w"))],
            4,
            blocks=[np.arange(2), np.arange(2, 4)],
        )
        assert not two.block_diagonal
        diag = ConstraintCovariance(
            [noise_term(Parameter("e"))], 4, blocks=[np.arange(2), np.arange(2, 4)]
        )
        assert diag.block_diagonal

    def test_non_term_raises(self):
        with pytest.raises(TypeError):
            ConstraintCovariance([np.eye(2)], 2)


# ----------------------------------------------------------------------------
# Gather-by-identity and structural properties
# ----------------------------------------------------------------------------


class TestGatherByIdentity:
    def test_shared_parameter_dedup(self):
        p = Parameter("log eps")
        cov = ConstraintCovariance(
            [noise_term(p, support=[0, 1]), noise_term(p, support=[2, 3])], 4
        )
        assert cov.n_params == 1

    def test_distinct_parameters_not_shared(self):
        cov = ConstraintCovariance(
            [
                noise_term(Parameter("a"), support=[0, 1]),
                noise_term(Parameter("b"), support=[2, 3]),
            ],
            4,
        )
        assert cov.n_params == 2

    def test_shared_value_fed_to_both(self):
        p = Parameter("log eps")
        cov = ConstraintCovariance(
            [noise_term(p, support=[0, 1]), noise_term(p, support=[2, 3])], 4
        )
        ctx = single_block_ctx(np.zeros(4), np.zeros(4), np.zeros(4))
        S = cov.matrix(ctx, np.log(3.0))
        assert np.allclose(S, 9.0 * np.eye(4))

    def test_first_seen_order_deterministic(self):
        a, b = Parameter("a"), Parameter("b")
        cov = ConstraintCovariance(
            [
                noise_term(b, support=[0]),
                noise_term(a, support=[1]),
                noise_term(b, support=[2]),
            ],
            3,
        )
        assert cov.params == (b, a)

    def test_wrong_param_count_raises(self):
        cov = ConstraintCovariance([noise_term(Parameter("a"))], 2)
        with pytest.raises(ValueError, match="expected 1 params"):
            cov.matrix(None)


class TestProperties:
    def test_is_constant_and_caching(self):
        cov = ConstraintCovariance(
            [statistical_term(np.array([1.0, 2.0]))], 2, blocks=[np.arange(2)]
        )
        assert cov.is_constant
        S1 = cov.matrix(None)
        S2 = cov.matrix(None)
        assert S1 is S2
        assert not S1.flags.writeable
        L, _ = cov.cholesky(None)
        assert not L.flags.writeable

    def test_nonconstant_matrix_writable(self):
        cov = ConstraintCovariance([noise_term(Parameter("a"))], 2)
        ctx = single_block_ctx(np.zeros(2), np.zeros(2), np.zeros(2))
        S = cov.matrix(ctx, 0.0)
        assert S.flags.writeable

    def test_block_cholesky_cached_and_requires_blocks(self):
        cov = ConstraintCovariance(
            [statistical_term(np.ones(4))], 4, blocks=[np.arange(2), np.arange(2, 4)]
        )
        f1 = cov.block_cholesky(None)
        f2 = cov.block_cholesky(None)
        assert f1 is f2
        with pytest.raises(ValueError):
            ConstraintCovariance([statistical_term(np.ones(2))], 2).block_cholesky(None)

    def test_cross_block_term_without_blocks_not_block_diagonal(self):
        cov = ConstraintCovariance([offset_term(parameter=Parameter("w"))], 4)
        assert not cov.block_diagonal
        diag = ConstraintCovariance([noise_term(Parameter("e"))], 4)
        assert diag.block_diagonal

    def test_stacked_distance_matches_dense(self):
        x = np.arange(4.0)
        y = np.array([1.0, 2.0, 3.0, 4.0])
        ymod = np.array([1.1, 1.9, 3.2, 3.8])
        ctx = make_ctx(x, y, ymod, [np.arange(2), np.arange(2, 4)])
        terms = [statistical_term(0.5 * np.ones(4)), noise_term(Parameter("e"))]
        block = ConstraintCovariance(terms, 4, blocks=ctx.supports)
        dense = ConstraintCovariance(terms, 4)
        assert block.block_diagonal
        d_b = block.stacked_distance(ctx, (np.log(0.3),))
        d_d = dense.stacked_distance(ctx, (np.log(0.3),))
        S = block.matrix(ctx, np.log(0.3))
        assert np.allclose(d_b, mahalanobis_distance_sqr_cholesky(y, ymod, S))
        assert np.allclose(d_d, d_b)


# ----------------------------------------------------------------------------
# Factories
# ----------------------------------------------------------------------------


class TestFactories:
    def setup_method(self):
        self.x = np.array([0.5, 1.0, 1.5])
        self.y = np.array([1.0, 2.0, 3.0])
        self.ym = np.array([1.1, 1.9, 3.2])
        self.stat = np.array([0.1, 0.2, 0.3])
        self.ctx = single_block_ctx(self.x, self.y, self.ym)

    def test_statistical_only(self):
        S = assemble([statistical_term(self.stat)], self.ctx)
        assert np.allclose(S, np.diag(self.stat**2))

    def test_unknown_noise(self):
        S = assemble([noise_term(Parameter("e"))], self.ctx, (np.log(0.4),))
        assert np.allclose(S, 0.16 * np.eye(3))
        S = assemble([noise_term(Parameter("e"), log=False)], self.ctx, (0.4,))
        assert np.allclose(S, 0.16 * np.eye(3))

    def test_unknown_noise_fraction(self):
        S = assemble([noise_fraction_term(Parameter("e"))], self.ctx, (np.log(0.4),))
        assert np.allclose(S, np.diag((0.4 * self.ym) ** 2))

    def test_unknown_normalization_error(self):
        S = assemble(
            [normalization_term(parameter=Parameter("n"))], self.ctx, (np.log(0.05),)
        )
        assert np.allclose(S, 0.05**2 * np.outer(self.ym, self.ym))

    def test_unknown_model_error_averaging(self):
        S = assemble(
            [model_error_term(Parameter("g"), averaging=True)], self.ctx, (np.log(0.1),)
        )
        z = 0.5 * (self.y + self.ym)
        assert np.allclose(S, np.diag((0.1 * z) ** 2))
        S = assemble(
            [model_error_term(Parameter("g"), averaging=False)],
            self.ctx,
            (np.log(0.1),),
        )
        assert np.allclose(S, np.diag((0.1 * self.ym) ** 2))

    def test_fixed_normalization_systematic(self):
        S = assemble([normalization_term(magnitude=0.05)], self.ctx)
        assert np.allclose(S, 0.05**2 * np.outer(self.ym, self.ym))
        S = assemble([normalization_term(magnitude=np.array(0.05))], self.ctx)
        assert np.allclose(S, 0.05**2 * np.outer(self.ym, self.ym))

    def test_fixed_offset_systematic(self):
        t = offset_term(magnitude=0.2)
        assert t.is_constant
        S = assemble([t], self.ctx)
        assert np.allclose(S, 0.04 * np.ones((3, 3)))
        S = assemble([offset_term(magnitude=np.array([0.1, 0.2, 0.3]))], self.ctx)
        v = np.array([0.1, 0.2, 0.3])
        assert np.allclose(S, np.outer(v, v))

    def test_fixed_offset_in_constant_covariance_ignores_ym(self):
        # a constant covariance never reads ym: it can be factored (eagerly, at
        # Constraint construction) with a placeholder ym and the cached factor
        # is reused afterwards
        cov = ConstraintCovariance(
            [statistical_term(self.stat), offset_term(magnitude=0.2)], 3
        )
        assert cov.is_constant
        L, logdet = cov.cholesky(
            StackContext.constant(self.ctx.x, self.ctx.y, [np.arange(3)])
        )
        assert np.all(np.isfinite(L))
        L2, logdet2 = cov.cholesky(self.ctx)
        assert L2 is L and logdet2 == logdet

    def test_masked_magnitudes(self):
        m = np.array([1.0, 0.0, 1.0])
        S = assemble([offset_term(magnitude=0.2, mask=m)], self.ctx)
        v = 0.2 * m
        assert np.allclose(S, np.outer(v, v))
        S = assemble(
            [normalization_term(parameter=Parameter("n"), mask=m)],
            self.ctx,
            (np.log(0.5),),
        )
        v = 0.5 * m * self.ym
        assert np.allclose(S, np.outer(v, v))

    def test_magnitude_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            assemble([offset_term(magnitude=np.ones(2))], self.ctx)

    def test_requires_magnitude_or_parameter(self):
        with pytest.raises(ValueError):
            offset_term()
        with pytest.raises(ValueError):
            normalization_term()

    def test_systematic_term_with_basis(self):
        s = Parameter("log s")
        S = assemble([systematic_term(s, basis=x_basis(2.0))], self.ctx, (np.log(3.0),))
        v = 3.0 * self.x / 2.0
        assert np.allclose(S, np.outer(v, v))

    def test_parametric_basis(self):
        e, l = Parameter("log e"), Parameter("slope")
        t = noise_term(e, basis=exp_growth(np.pi), basis_params=(l,))
        assert t.params == (e, l)
        S = assemble([t], self.ctx, (np.log(0.3), 0.0))
        assert np.allclose(S, 0.09 * np.eye(3))  # slope 0 == plain noise_term
        S = assemble([t], self.ctx, (np.log(0.3), 2.0))
        assert np.allclose(S, np.diag((0.3 * np.exp(2.0 * self.x / np.pi)) ** 2))
        # basis growing with ym in linear space
        t = noise_term(e, basis=exp_growth(np.pi, base=ym), basis_params=(l,))
        S = assemble([t], self.ctx, (np.log(0.3), 1.0))
        assert np.allclose(S, np.diag((0.3 * self.ym * np.exp(self.x / np.pi)) ** 2))

    def test_old_observation_covariance_equivalence(self):
        offset, norm = 0.2, 0.05
        terms = [
            statistical_term(self.stat),
            offset_term(magnitude=offset),
            normalization_term(magnitude=norm),
        ]
        S = assemble(terms, self.ctx)
        old = (
            np.diag(self.stat**2)
            + np.outer(offset * np.ones(3), offset * np.ones(3))
            + norm**2 * np.outer(self.ym, self.ym)
        )
        assert np.allclose(S, old)

    def test_bases(self):
        c = Term(lambda c: c.ym, kind="diag", support=np.arange(3)).local_context(
            self.ctx
        )
        assert np.allclose(ones(c), 1.0)
        assert np.allclose(ym(c), self.ym)
        assert np.allclose(averaging(c), 0.5 * (self.y + self.ym))


class TestKernelTerm:
    def test_params_match_theta_length_isotropic(self):
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(1e-6)
        term = kernel_term(kernel)
        assert len(term.params) == len(kernel.theta)
        assert not term.is_constant

    def test_params_anisotropic(self):
        kernel = ConstantKernel(1.0) * RBF(length_scale=[1.0, 1.0])
        term = kernel_term(kernel, support=np.arange(3))
        assert len(term.params) == len(kernel.theta)
        x2d = np.array([[0.0, 0.0], [1.0, 0.5], [2.0, 1.0]])
        ctx = make_ctx(x2d, np.zeros(3), np.zeros(3), [np.arange(3)])
        Sigma = np.zeros((3, 3))
        term.add_to(Sigma, ctx, kernel.theta)
        assert np.all(np.isfinite(Sigma))

    def test_cross_block_values(self):
        kernel = RBF(length_scale=1.0)
        term = kernel_term(kernel, jitter=0.0, support=np.arange(4))
        x = np.array([0.0, 1.0, 2.0, 3.0])
        ctx = make_ctx(x, np.zeros(4), np.zeros(4), [np.arange(2), np.arange(2, 4)])
        S = np.zeros((4, 4))
        term.add_to(S, ctx, kernel.theta)
        np.testing.assert_allclose(S, kernel(x[:, None]))
        assert np.any(S[:2, 2:] != 0.0)
        cov = ConstraintCovariance([term], N=4, blocks=[np.arange(2), np.arange(2, 4)])
        assert not cov.block_diagonal

    def test_fixed_kernel_is_constant(self):
        kernel = RBF(length_scale=1.0, length_scale_bounds="fixed")
        term = kernel_term(kernel)
        assert term.params == () and term.is_constant

    def test_constant_term_may_read_x(self):
        # constant means "independent of ym"; x is invariant and readable
        x = np.linspace(0.0, 2.0, 4)
        t = Term(lambda c: 0.1 * c.x, kind="diag", constant=True)
        cov = ConstraintCovariance([t], 4)
        assert cov.is_constant
        ctx = StackContext.constant(x, np.zeros(4), [np.arange(4)])
        np.testing.assert_allclose(cov.matrix(ctx), np.diag((0.1 * x) ** 2))
        # a mis-declared constant term (reads ym) fails loudly, not silently
        bad = ConstraintCovariance(
            [Term(lambda c: 0.1 * c.ym, kind="diag", constant=True)], 4
        )
        with pytest.raises(TypeError):
            bad.matrix(ctx)
        assert cov.matrix(single_block_ctx(x, np.zeros(4), np.ones(4))) is cov.matrix(
            ctx
        )

    def test_constant_amplitude_reproduces_constant_kernel(self):
        x = np.linspace(0.0, 2.0, 5)
        ctx = single_block_ctx(x, np.zeros(5), np.zeros(5))
        A = 0.7
        la = Parameter("log A")
        term = kernel_term(
            RBF(1.0), amplitude=constant_amplitude, amplitude_params=(la,), jitter=0.0
        )
        assert [p.name for p in term.params] == ["discrepancy_length_scale", "log A"]
        S = assemble([term], ctx, (0.0, np.log(A)))
        ref = ConstantKernel(A**2, constant_value_bounds="fixed") * RBF(1.0)
        np.testing.assert_allclose(S, ref(x[:, None]))

    def test_exp_growth_amplitude_and_coords(self):
        x = np.linspace(0.1, 3.0, 4)
        ctx = single_block_ctx(x, np.zeros(4), np.zeros(4))
        la, sl = Parameter("log A"), Parameter("slope")
        q = lambda x: 2.0 * np.sin(x / 2)  # noqa: E731
        term = kernel_term(
            Matern(1.0, nu=2.5),
            coords=q,
            amplitude=exp_growth_amplitude(np.pi),
            amplitude_params=(la, sl),
            jitter=0.0,
        )
        S = assemble([term], ctx, (np.log(0.5), np.log(0.3), 1.5))
        # note: amplitude sees the *transformed* coordinate
        a = 0.3 * np.exp(1.5 * q(x) / np.pi)
        ref = np.outer(a, a) * Matern(0.5, nu=2.5)(q(x)[:, None])
        np.testing.assert_allclose(S, ref)

    def test_duplicate_coords_factorizable_with_jitter(self):
        x = np.array([0.0, 0.0, 1.0])
        ctx = single_block_ctx(x, np.zeros(3), np.zeros(3))
        term = kernel_term(RBF(1.0), jitter=1e-8)
        cov = ConstraintCovariance([term, statistical_term(1e-3 * np.ones(3))], 3)
        L, _ = cov.cholesky(ctx, 0.0)
        assert np.all(np.isfinite(L))


# ----------------------------------------------------------------------------
# The alpha+Ca error-model ladder as one-line term lists (study self-checks)
# ----------------------------------------------------------------------------


class TestStudyForms:
    """Each error model of the alpha+Ca study is one term list; compare to the
    hand-rolled dense covariance from that study's ``error_covariance``."""

    def setup_method(self):
        rng = np.random.default_rng(1)
        n = 12
        self.x = np.sort(rng.uniform(0.2, 3.0, n))  # radians
        self.y = rng.uniform(0.1, 1.5, n)  # log-space "data" (any values)
        self.ym = self.y + rng.normal(0.0, 0.1, n)
        self.ctx = single_block_ctx(self.x, self.y, self.ym)
        self.X = np.pi
        self.log_err, self.log_slope = Parameter("log_err"), Parameter("log_err_slope")
        self.log_sys, self.log_amp = Parameter("log_sys"), Parameter("log_amp")
        self.err, self.slope, self.sys, self.amp = 0.05, 1.3, 0.04, 0.2
        self.k = 2.7

    def xdeg(self):
        return self.x / self.X  # theta / 180

    def test_L0(self):
        S = assemble([noise_term(self.log_err)], self.ctx, (np.log(self.err),))
        assert np.allclose(S, self.err**2 * np.eye(len(self.x)))

    def test_E0_linear_space(self):
        S = assemble([noise_fraction_term(self.log_err)], self.ctx, (np.log(self.err),))
        assert np.allclose(S, np.diag((self.err * self.ym) ** 2))

    def test_L1(self):
        terms = [
            noise_term(
                self.log_err, basis=exp_growth(self.X), basis_params=(self.log_slope,)
            )
        ]
        S = assemble(terms, self.ctx, (np.log(self.err), self.slope))
        sigma = self.err * np.exp(self.slope * self.xdeg())
        assert np.allclose(S, np.diag(sigma**2))

    def test_L2_rank_one_over_theta(self):
        terms = [
            noise_term(self.log_err),
            systematic_term(self.log_sys, basis=x_basis(self.X)),
        ]
        S = assemble(terms, self.ctx, (np.log(self.err), np.log(self.sys)))
        u = self.xdeg()
        assert np.allclose(
            S, self.err**2 * np.eye(len(u)) + self.sys**2 * np.outer(u, u)
        )

    def test_L2n_and_L2y(self):
        S = assemble(
            [noise_term(self.log_err), offset_term(parameter=self.log_sys)],
            self.ctx,
            (np.log(self.err), np.log(self.sys)),
        )
        assert np.allclose(S, self.err**2 * np.eye(len(self.x)) + self.sys**2)
        S = assemble(
            [noise_term(self.log_err), normalization_term(parameter=self.log_sys)],
            self.ctx,
            (np.log(self.err), np.log(self.sys)),
        )
        assert np.allclose(
            S,
            self.err**2 * np.eye(len(self.x))
            + self.sys**2 * np.outer(self.ym, self.ym),
        )

    def test_L12(self):
        terms = [
            noise_term(
                self.log_err, basis=exp_growth(self.X), basis_params=(self.log_slope,)
            ),
            systematic_term(self.log_sys, basis=x_basis(self.X)),
        ]
        S = assemble(terms, self.ctx, (np.log(self.err), self.slope, np.log(self.sys)))
        sigma = self.err * np.exp(self.slope * self.xdeg())
        u = self.xdeg()
        assert np.allclose(S, np.diag(sigma**2) + self.sys**2 * np.outer(u, u))

    def test_Lgp_matern_in_theta(self):
        ell = 0.3
        terms = [
            noise_term(self.log_err),
            kernel_term(
                Matern(1.0, nu=2.5),
                coords=lambda x: x / self.X,
                amplitude=constant_amplitude,
                amplitude_params=(self.log_amp,),
                jitter=0.0,
                prefix="gp",
            ),
        ]
        S = assemble(terms, self.ctx, (np.log(self.err), np.log(ell), np.log(self.amp)))
        u = self.xdeg()
        K = self.amp**2 * Matern(ell, nu=2.5)(u[:, None])
        assert np.allclose(S, self.err**2 * np.eye(len(u)) + K)

    def test_Lgpn_angle_growing_amplitude(self):
        ell = 0.3
        terms = [
            noise_term(self.log_err),
            kernel_term(
                Matern(1.0, nu=2.5),
                coords=lambda x: x / self.X,
                amplitude=exp_growth_amplitude(1.0),
                amplitude_params=(self.log_amp, self.log_slope),
                jitter=0.0,
            ),
        ]
        S = assemble(
            terms,
            self.ctx,
            (np.log(self.err), np.log(ell), np.log(self.amp), self.slope),
        )
        u = self.xdeg()
        a = self.amp * np.exp(self.slope * u)
        K = np.outer(a, a) * Matern(ell, nu=2.5)(u[:, None])
        assert np.allclose(S, self.err**2 * np.eye(len(u)) + K)

    def test_LKp_kernel_in_momentum_transfer(self):
        # b^2 I + s^2 11^T + a(q) a(q') RBF(|q - q'| / l_q), a = A q^(r/2)
        log_b, log_s, r_pow = Parameter("log_b"), Parameter("log_s"), Parameter("r")
        b, s, lq, r = 0.05, 0.05, 1.2, 0.8
        q = 2.0 * self.k * np.sin(self.x / 2)
        terms = [
            noise_term(log_b),
            offset_term(parameter=log_s),
            kernel_term(
                RBF(1.0),
                coords=lambda x: 2.0 * self.k * np.sin(x / 2),
                amplitude=lambda c, lA, r: np.exp(lA) * c.x ** (r / 2),
                amplitude_params=(self.log_amp, r_pow),
                jitter=0.0,
                prefix="gpq",
            ),
        ]
        S = assemble(
            terms, self.ctx, (np.log(b), np.log(s), np.log(lq), np.log(self.amp), r)
        )
        a = self.amp * q ** (r / 2)
        K = np.outer(a, a) * RBF(lq)(q[:, None])
        ref = b**2 * np.eye(len(q)) + s**2 * np.ones((len(q), len(q))) + K
        assert np.allclose(S, ref)

    def test_custom_term_direct(self):
        # anything the factories cannot say is a one-line Term
        e, l = Parameter("e"), Parameter("l")
        t = Term(
            lambda c, e, l: np.exp(e) * np.exp(l * c.x / np.pi), (e, l), kind="diag"
        )
        S = assemble([t], self.ctx, (np.log(self.err), self.slope))
        sigma = self.err * np.exp(self.slope * self.xdeg())
        assert np.allclose(S, np.diag(sigma**2))
