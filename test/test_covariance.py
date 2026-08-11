"""Unit tests for the stacked-covariance core (:mod:`rxmc.covariance`)."""

import numpy as np
import pytest

from helpers import make_ctx
from rxmc.covariance import (
    ConstraintCovariance,
    DenseTerm,
    DiagonalTerm,
    RankOneTerm,
    model_error_term,
    noise_fraction_term,
    noise_term,
    normalization_term,
    offset_term,
    statistical_term,
    ym_basis,
)
from rxmc.params import Parameter


def single_block_ctx(x, y, ym):
    n = len(x)
    return make_ctx(x, y, ym, [np.arange(n)])


class TestDenseTerm:
    def test_vector_is_diagonal(self):
        t = DenseTerm([0, 1, 2], np.array([1.0, 2.0, 3.0]))
        S = np.zeros((3, 3))
        t.add_to(S, None, np.array([]))
        assert np.allclose(S, np.diag([1.0, 2.0, 3.0]))

    def test_full_matrix_passthrough(self):
        m = np.array([[2.0, 0.5], [0.5, 3.0]])
        t = DenseTerm([0, 1], m)
        S = np.zeros((2, 2))
        t.add_to(S, None, np.array([]))
        assert np.allclose(S, m)

    def test_writes_into_subblock(self):
        t = DenseTerm([2, 3], np.array([1.0, 1.0]))
        S = np.zeros((4, 4))
        t.add_to(S, None, np.array([]))
        expected = np.zeros((4, 4))
        expected[2, 2] = expected[3, 3] = 1.0
        assert np.allclose(S, expected)

    def test_scalar_broadcast_raises(self):
        # a (1, 1) matrix on a length-3 support used to broadcast silently
        # into the whole block
        with pytest.raises(ValueError, match="does not match support"):
            DenseTerm(np.arange(3), [[0.04]])

    def test_wrong_length_vector_raises(self):
        with pytest.raises(ValueError, match="does not match support"):
            DenseTerm(np.arange(3), np.ones(2))

    def test_asymmetric_matrix_raises(self):
        with pytest.raises(ValueError, match="symmetric"):
            DenseTerm(np.arange(2), [[1.0, 0.5], [0.0, 1.0]])

    def test_couples_offdiagonal(self):
        # only terms that can write off-diagonal entries can couple blocks
        assert not DenseTerm([0, 1, 2], np.ones(3)).couples_offdiagonal
        assert DenseTerm([0, 1], np.array([[2.0, 0.5], [0.5, 3.0]])).couples_offdiagonal
        assert not DiagonalTerm([0, 1]).couples_offdiagonal
        assert RankOneTerm([0, 1], basis=np.ones(2)).couples_offdiagonal


class TestDiagonalTerm:
    def test_ones_basis_with_param(self):
        p = Parameter("log eps")
        t = DiagonalTerm([0, 1], basis=None, parameter=p, log=True)
        ctx = single_block_ctx(np.zeros(2), np.zeros(2), np.zeros(2))
        S = np.zeros((2, 2))
        t.add_to(S, ctx, np.array([np.log(0.5)]))
        assert np.allclose(S, np.diag([0.25, 0.25]))

    def test_basis_array_no_param(self):
        t = DiagonalTerm([0, 1], basis=np.array([2.0, 3.0]))
        S = np.zeros((2, 2))
        t.add_to(S, None, np.array([]))
        assert np.allclose(S, np.diag([4.0, 9.0]))

    def test_ym_basis_callable(self):
        p = Parameter("log eps")
        t = DiagonalTerm([0, 1], basis=ym_basis, parameter=p)
        ym = np.array([2.0, 4.0])
        ctx = single_block_ctx(np.zeros(2), np.zeros(2), ym)
        S = np.zeros((2, 2))
        t.add_to(S, ctx, np.array([np.log(0.1)]))
        assert np.allclose(S, np.diag((0.1 * ym) ** 2))


class TestRankOneTerm:
    def test_outer_product(self):
        v = np.array([1.0, 2.0, 3.0])
        t = RankOneTerm([0, 1, 2], basis=v)
        S = np.zeros((3, 3))
        t.add_to(S, None, np.array([]))
        assert np.allclose(S, np.outer(v, v))

    def test_normalization_mode_scales_with_ym(self):
        p = Parameter("log eta")
        t = RankOneTerm([0, 1], basis=ym_basis, parameter=p)
        ym = np.array([3.0, 5.0])
        ctx = single_block_ctx(np.zeros(2), np.zeros(2), ym)
        S = np.zeros((2, 2))
        eta = 0.07
        t.add_to(S, ctx, np.array([np.log(eta)]))
        assert np.allclose(S, np.outer(eta * ym, eta * ym))

    def test_cross_block_coupling_writes_offdiagonal(self):
        # support spans two blocks [0,1] and [2,3]
        v = np.array([1.0, 1.0, 1.0, 1.0])
        t = RankOneTerm([0, 1, 2, 3], basis=v)
        S = np.zeros((4, 4))
        t.add_to(S, None, np.array([]))
        # off-diagonal blocks coupling block 0 and block 1 are non-zero
        assert S[0, 2] != 0.0 and S[1, 3] != 0.0


class TestGatherByIdentity:
    def test_shared_parameter_dedup(self):
        # Case B: two block-local terms share ONE Parameter object
        eta = Parameter("log eta")
        t1 = normalization_term([0, 1], parameter=eta)
        t2 = normalization_term([2, 3], parameter=eta)
        cov = ConstraintCovariance([t1, t2], N=4)
        assert cov.n_params == 1
        assert cov.params == (eta,)

    def test_distinct_parameters_not_shared(self):
        e1 = Parameter("log eta 1")
        e2 = Parameter("log eta 2")
        t1 = normalization_term([0, 1], parameter=e1)
        t2 = normalization_term([2, 3], parameter=e2)
        cov = ConstraintCovariance([t1, t2], N=4)
        assert cov.n_params == 2
        assert cov.params == (e1, e2)

    def test_shared_value_fed_to_both(self):
        eta = Parameter("log eta")
        t1 = normalization_term([0, 1], parameter=eta)
        t2 = normalization_term([2, 3], parameter=eta)
        cov = ConstraintCovariance([t1, t2], N=4)
        ym = np.array([1.0, 1.0, 2.0, 2.0])
        ctx = make_ctx(np.zeros(4), np.zeros(4), ym, [np.arange(2), np.arange(2, 4)])
        e = 0.1
        S = cov.matrix(ctx, np.log(e))
        # both blocks scaled by the same eta
        assert np.allclose(S[:2, :2], np.outer(e * ym[:2], e * ym[:2]))
        assert np.allclose(S[2:, 2:], np.outer(e * ym[2:], e * ym[2:]))
        # block-local -> no cross coupling
        assert np.allclose(S[:2, 2:], 0.0)

    def test_first_seen_order_deterministic(self):
        a = Parameter("a")
        b = Parameter("b")
        t1 = DiagonalTerm([0], parameter=b)
        t2 = DiagonalTerm([1], parameter=a)
        cov = ConstraintCovariance([t1, t2], N=2)
        assert cov.params == (b, a)

    def test_wrong_param_count_raises(self):
        p = Parameter("p")
        cov = ConstraintCovariance([DiagonalTerm([0], parameter=p)], N=1)
        with pytest.raises(ValueError):
            cov.matrix(single_block_ctx(np.zeros(1), np.zeros(1), np.zeros(1)))


class TestProperties:
    def test_block_diagonal_true_for_local_terms(self):
        cov = ConstraintCovariance(
            [
                statistical_term([0, 1], np.ones(2)),
                statistical_term([2, 3], np.ones(2)),
            ],
            N=4,
            blocks=[np.arange(2), np.arange(2, 4)],
        )
        assert cov.block_diagonal

    def test_block_diagonal_false_for_coupling(self):
        cov = ConstraintCovariance(
            [RankOneTerm([0, 1, 2, 3], basis=np.ones(4))],
            N=4,
            blocks=[np.arange(2), np.arange(2, 4)],
        )
        assert not cov.block_diagonal

    def test_contiguous_cross_block_term_without_blocks_not_block_diagonal(self):
        # the old contiguity heuristic classified a coupling term with a
        # contiguous support as block-local; without explicit blocks the
        # classification must now be conservative (dense path)
        cov = ConstraintCovariance(
            [
                statistical_term([0, 1], np.ones(2)),
                statistical_term([2, 3], np.ones(2)),
                RankOneTerm(np.arange(4), basis=np.ones(4)),
            ],
            N=4,
        )
        assert not cov.block_diagonal

    def test_diagonal_only_terms_block_diagonal_without_blocks(self):
        # strictly-diagonal terms are block-diagonal under ANY partition
        cov = ConstraintCovariance(
            [statistical_term([0, 1], np.ones(2)), DiagonalTerm([2, 3])], N=4
        )
        assert cov.block_diagonal

    def test_diagonal_term_spanning_blocks_stays_block_diagonal(self):
        cov = ConstraintCovariance(
            [DiagonalTerm(np.arange(4))],
            N=4,
            blocks=[np.arange(2), np.arange(2, 4)],
        )
        assert cov.block_diagonal

    def test_is_constant(self):
        cov = ConstraintCovariance([statistical_term([0, 1], np.ones(2))], N=2)
        assert cov.is_constant
        p = Parameter("p")
        cov2 = ConstraintCovariance([DiagonalTerm([0, 1], parameter=p)], N=2)
        assert not cov2.is_constant

    def test_constant_matrix_cached(self):
        cov = ConstraintCovariance(
            [statistical_term([0, 1], np.array([2.0, 3.0]))], N=2
        )
        m1 = cov.matrix(None)
        m2 = cov.matrix(None)
        assert m1 is m2
        assert np.allclose(m1, np.diag([4.0, 9.0]))

    def test_cached_matrix_readonly(self):
        # the cached matrix is shared state: mutation must fail loudly
        cov = ConstraintCovariance([statistical_term([0, 1], np.ones(2))], N=2)
        m = cov.matrix(None)
        with pytest.raises(ValueError):
            m[0, 0] = 99.0

    def test_cached_cholesky_readonly(self):
        cov = ConstraintCovariance([statistical_term([0, 1], np.ones(2))], N=2)
        L, _ = cov.cholesky(None)
        with pytest.raises(ValueError):
            L[0, 0] = 99.0

    def test_block_cholesky_cached_and_readonly(self):
        cov = ConstraintCovariance(
            [
                statistical_term([0, 1], np.ones(2)),
                statistical_term([2, 3], np.ones(2)),
            ],
            N=4,
            blocks=[np.arange(2), np.arange(2, 4)],
        )
        f1 = cov.block_cholesky(None)
        f2 = cov.block_cholesky(None)
        assert f1 is f2
        with pytest.raises(ValueError):
            f1[0][0][0, 0] = 99.0

    def test_block_cholesky_requires_blocks(self):
        cov = ConstraintCovariance([statistical_term([0, 1], np.ones(2))], N=2)
        with pytest.raises(ValueError, match="blocks"):
            cov.block_cholesky(None)

    def test_block_cholesky_nonconstant_not_cached(self):
        p = Parameter("log eps")
        cov = ConstraintCovariance(
            [
                statistical_term([0, 1], np.ones(2)),
                DiagonalTerm([2, 3], parameter=p),
            ],
            N=4,
            blocks=[np.arange(2), np.arange(2, 4)],
        )
        ctx = make_ctx(
            np.zeros(4), np.zeros(4), np.zeros(4), [np.arange(2), np.arange(2, 4)]
        )
        f1 = cov.block_cholesky(ctx, 0.0)
        f2 = cov.block_cholesky(ctx, 0.0)
        assert f1 is not f2
        for (L1, d1), (L2, d2) in zip(f1, f2):
            assert np.allclose(L1, L2)
            assert d1 == d2

    def test_nonconstant_matrix_writable(self):
        p = Parameter("p")
        cov = ConstraintCovariance([DiagonalTerm([0, 1], parameter=p)], N=2)
        ctx = single_block_ctx(np.zeros(2), np.zeros(2), np.zeros(2))
        S = cov.matrix(ctx, 0.0)
        S[0, 0] = 99.0  # fresh array, caller-owned


class TestNoSilentFastPathWithoutBlocks:
    """A cross-block coupling built without explicit blocks must take the dense path.

    This pins the fix for the contiguity-heuristic bug: a RankOneTerm whose
    support is one contiguous run over two observation blocks used to be
    misclassified as block-local, and the fast path silently dropped the
    off-diagonal coupling.
    """

    def test_stacked_distance_matches_dense(self):
        from rxmc.likelihood_model import mahalanobis_distance_sqr_cholesky

        rng = np.random.default_rng(7)
        y = rng.normal(size=4)
        ym = y + 0.1 * rng.normal(size=4)
        cov = ConstraintCovariance(
            [
                statistical_term(np.arange(4), np.full(4, 0.5)),
                RankOneTerm(np.arange(4), basis=np.ones(4)),
            ],
            N=4,
        )
        ctx = make_ctx(np.arange(4.0), y, ym, [np.arange(2), np.arange(2, 4)])

        d2, logdet = cov.stacked_distance(ctx)
        Sigma = np.diag(np.full(4, 0.25)) + np.ones((4, 4))
        d2_dense, logdet_dense = mahalanobis_distance_sqr_cholesky(y, ym, Sigma)
        assert np.isclose(d2, d2_dense)
        assert np.isclose(logdet, logdet_dense)


class TestScalarLikeMagnitudes:
    """0-d ndarrays (as stored by exfor_tools distributions) count as scalars."""

    def test_zero_dim_offset_magnitude(self):
        support = np.arange(3)
        a = offset_term(support, magnitude=np.array(0.2))
        b = offset_term(support, magnitude=0.2)
        S1, S2 = np.zeros((3, 3)), np.zeros((3, 3))
        a.add_to(S1, None, np.array([]))
        b.add_to(S2, None, np.array([]))
        assert np.allclose(S1, S2)

    def test_zero_dim_normalization_magnitude(self):
        support = np.arange(3)
        ctx = single_block_ctx(np.zeros(3), np.zeros(3), np.array([1.0, 2.0, 3.0]))
        a = normalization_term(support, magnitude=np.array(0.05))
        b = normalization_term(support, magnitude=0.05)
        S1, S2 = np.zeros((3, 3)), np.zeros((3, 3))
        a.add_to(S1, ctx, np.array([]))
        b.add_to(S2, ctx, np.array([]))
        assert np.allclose(S1, S2)


class TestFactoryHelpersReproduceOldCovariance:
    """Each helper reproduces the matrix the old LikelihoodModel zoo built."""

    def setup_method(self):
        self.y = np.array([1.0, 2.0, 3.0])
        self.ym = np.array([1.1, 1.9, 3.2])
        self.stat = np.array([0.1, 0.2, 0.3])
        self.support = np.arange(3)
        self.ctx = single_block_ctx(np.arange(3.0), self.y, self.ym)

    def test_statistical_only(self):
        S = ConstraintCovariance(
            [statistical_term(self.support, self.stat)], N=3
        ).matrix(self.ctx)
        assert np.allclose(S, np.diag(self.stat**2))

    def test_unknown_noise(self):
        eps = 0.05
        p = Parameter("log eps")
        cov = ConstraintCovariance(
            [statistical_term(self.support, np.zeros(3)), noise_term(self.support, p)],
            N=3,
        )
        S = cov.matrix(self.ctx, np.log(eps))
        assert np.allclose(S, np.diag(np.full(3, eps**2)))

    def test_unknown_noise_fraction(self):
        eps = 0.05
        p = Parameter("log eps")
        cov = ConstraintCovariance([noise_fraction_term(self.support, p)], N=3)
        S = cov.matrix(self.ctx, np.log(eps))
        assert np.allclose(S, np.diag((eps * self.ym) ** 2))

    def test_unknown_normalization_error(self):
        eta = 0.07
        p = Parameter("log eta")
        cov = ConstraintCovariance([normalization_term(self.support, parameter=p)], N=3)
        S = cov.matrix(self.ctx, np.log(eta))
        assert np.allclose(S, eta**2 * np.outer(self.ym, self.ym))

    def test_unknown_model_error_averaging(self):
        gamma = 0.1
        p = Parameter("log gamma")
        cov = ConstraintCovariance(
            [model_error_term(self.support, p, averaging=True)], N=3
        )
        S = cov.matrix(self.ctx, np.log(gamma))
        z = 0.5 * (self.y + self.ym)
        assert np.allclose(S, np.diag((gamma * z) ** 2))

    def test_fixed_normalization_systematic(self):
        # data-given fractional normalisation (no free parameter)
        frac = 0.03
        cov = ConstraintCovariance(
            [normalization_term(self.support, magnitude=frac)], N=3
        )
        S = cov.matrix(self.ctx)
        expected = (frac**2) * np.outer(self.ym, self.ym)
        assert np.allclose(S, expected)

    def test_fixed_offset_systematic(self):
        off = 0.2
        cov = ConstraintCovariance([offset_term(self.support, magnitude=off)], N=3)
        S = cov.matrix(self.ctx)
        omega = off * np.ones(3)
        assert np.allclose(S, np.outer(omega, omega))


class TestOldObservationCovarianceEquivalence:
    """A stat+offset+normalization stack matches the old Observation.covariance(ym)."""

    def test_full_covariance(self):
        y = np.array([1.0, 2.0, 4.0])
        ym = np.array([1.2, 2.1, 3.5])
        stat = np.array([0.1, 0.2, 0.3])
        norm = 0.05
        offset = 0.2
        support = np.arange(3)
        ctx = make_ctx(np.arange(3.0), y, ym, [support])

        terms = [
            statistical_term(support, stat),
            offset_term(support, magnitude=offset),
            normalization_term(support, magnitude=norm),
        ]
        S = ConstraintCovariance(terms, N=3).matrix(ctx)

        # old Observation.covariance(ym):
        old = (
            np.diag(stat**2)
            + np.outer(offset * np.ones(3), offset * np.ones(3))
            + np.outer(norm * np.ones(3), norm * np.ones(3)) * np.outer(ym, ym)
        )
        assert np.allclose(S, old)


def test_kernel_term_params_match_theta_length_isotropic():
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

    from rxmc.covariance import KernelTerm

    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(1e-6)
    term = KernelTerm(np.arange(4), kernel)
    # one free Parameter per non-fixed hyperparameter element
    assert len(term.params) == len(kernel.theta)


def test_kernel_term_params_anisotropic():
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel

    from rxmc.covariance import KernelTerm

    # anisotropic: a vector length_scale is ONE hyperparameter with n_elements == 2
    kernel = ConstantKernel(1.0) * RBF(length_scale=[1.0, 1.0])
    term = KernelTerm(np.arange(3), kernel)
    assert len(term.params) == len(kernel.theta)  # would be 1 vs 2 before the fix

    # the gathered theta has the right length for clone_with_theta in add_to
    x2d = np.array([[0.0, 0.0], [1.0, 0.5], [2.0, 1.0]])
    ctx = make_ctx(x2d, np.zeros(3), np.zeros(3), [np.arange(3)])
    Sigma = np.zeros((3, 3))
    term.add_to(Sigma, ctx, kernel.theta)  # must not raise
    assert np.all(np.isfinite(Sigma))


def test_kernel_term_cross_block_values():
    # a kernel spanning two observation blocks writes the correct
    # off-diagonal coupling block (case A with a GP)
    from sklearn.gaussian_process.kernels import RBF

    from rxmc.covariance import KernelTerm

    kernel = RBF(length_scale=1.0)
    term = KernelTerm(np.arange(4), kernel, jitter=0.0)
    x = np.array([0.0, 1.0, 2.0, 3.0])
    ctx = make_ctx(x, np.zeros(4), np.zeros(4), [np.arange(2), np.arange(2, 4)])

    S = np.zeros((4, 4))
    term.add_to(S, ctx, kernel.theta)
    np.testing.assert_allclose(S, kernel(x[:, None]))
    assert np.any(S[:2, 2:] != 0.0)

    cov = ConstraintCovariance([term], N=4, blocks=[np.arange(2), np.arange(2, 4)])
    assert not cov.block_diagonal
