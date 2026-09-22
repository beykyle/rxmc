"""The structured (Woodbury) covariance against the dense reference."""

import numpy as np
import pytest
from sklearn.gaussian_process.kernels import RBF, Matern

from helpers import (
    STUDY_LEGEND,
    assemble_dense,
    index_params,
    mahalanobis,
    study_form,
)
from rxmc import Parameter
from rxmc.covariance import StructuredCovariance, chol_logdet
from rxmc.terms import Term, kernel, noise, normalization, offset, statistical


def build(terms, x, y, offsets, active=None, rows=None, labels=None):
    """Wire terms into a StructuredCovariance with identity-gathered params."""
    n = len(y)
    params, gathers = index_params(terms)
    rows = rows if rows is not None else [np.arange(n) for _ in terms]
    active = np.arange(n) if active is None else np.asarray(active, dtype=int)
    entries = list(zip(terms, rows, gathers))
    cov = StructuredCovariance(entries, x, y, offsets, active, labels=labels)
    return cov, params


def theta_for(params, values_by_term, terms):
    """Flat theta from per-term value tuples, honouring shared parameters."""
    theta = np.zeros(len(params))
    for t, v in zip(terms, values_by_term):
        for p, val in zip(t.params, v):
            theta[params.index(p)] = val
    return theta


def grid(n=12, seed=1):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0.2, 3.0, n))
    y = rng.uniform(0.1, 1.5, n)
    ym = y + rng.normal(0.0, 0.1, n)
    return x, y, ym


# ----------------------------------------------------------------------------
# Single block: every study form
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("label", list(STUDY_LEGEND))
def test_study_forms_match_dense(label):
    """See ``helpers.STUDY_LEGEND`` for what each label means."""
    x, y, ym = grid()
    form = study_form(label, x, y, ym)
    cov, params = build(form.terms, x, y, [slice(0, len(y))])
    theta = theta_for(params, form.values, form.terms)
    d2, logdet = cov.distance(ym, theta)
    d2_ref, logdet_ref = mahalanobis(y, ym, form.dense)
    assert d2 == pytest.approx(d2_ref) and logdet == pytest.approx(logdet_ref)
    np.testing.assert_allclose(cov.matrix(ym, theta), form.dense)
    assert not cov.dense


# ----------------------------------------------------------------------------
# Gather by identity
# ----------------------------------------------------------------------------


class TestGatherByIdentity:
    def setup_method(self):
        self.x, self.y, self.ym = grid(6)
        self.offsets = [slice(0, 3), slice(3, 6)]
        self.rows = [np.arange(3), np.arange(3, 6)]

    def test_shared_parameter_is_one_slot(self):
        p = Parameter("log eps")
        terms = [noise(p), noise(p)]
        cov, params = build(terms, self.x, self.y, self.offsets, rows=self.rows)
        assert params == (p,)
        np.testing.assert_allclose(cov.matrix(self.ym, [np.log(3.0)]), 9.0 * np.eye(6))

    def test_distinct_parameters_are_two_slots_in_first_seen_order(self):
        a, b = Parameter("a"), Parameter("b")
        terms = [noise(b), noise(a)]
        cov, params = build(terms, self.x, self.y, self.offsets, rows=self.rows)
        assert params == (b, a)
        S = cov.matrix(self.ym, [np.log(2.0), np.log(3.0)])
        np.testing.assert_allclose(np.diag(S), [4, 4, 4, 9, 9, 9])

    def test_wrong_theta_length_raises(self):
        cov, _ = build([noise(Parameter("a"))], self.x, self.y, [slice(0, 6)])
        with pytest.raises((IndexError, ValueError)):
            cov.distance(self.ym, [])


# ----------------------------------------------------------------------------
# Several blocks: modes across blocks stay structured, matrices crossing go dense
# ----------------------------------------------------------------------------


class TestMultiBlock:
    def setup_method(self):
        self.x, self.y, self.ym = grid(9, seed=2)
        self.offsets = [slice(0, 3), slice(3, 6), slice(6, 9)]
        self.b = [np.arange(0, 3), np.arange(3, 6), np.arange(6, 9)]
        self.eps, self.eta, self.omega = (
            Parameter("log_eps"),
            Parameter("log_eta"),
            Parameter("log_omega"),
        )
        self.stat = 0.1 * np.ones(9)

    def reference(self, terms, rows, values):
        return assemble_dense(terms, self.x, self.y, self.ym, values, rows)

    def test_case_a_modes_across_blocks(self):
        terms = [
            statistical(self.stat[:3]),
            statistical(self.stat[3:6]),
            statistical(self.stat[6:]),
            noise(self.eps),
            normalization(parameter=self.eta),  # on blocks 1 and 2 only (case A)
            offset(parameter=self.omega),  # on all
        ]
        rows = [*self.b, np.arange(9), np.arange(6), np.arange(9)]
        cov, params = build(terms, self.x, self.y, self.offsets, rows=rows)
        assert not cov.dense
        values = [(), (), (), (np.log(0.2),), (np.log(0.05),), (np.log(0.07),)]
        theta = theta_for(params, values, terms)
        ref = self.reference(terms, rows, values)
        d2, logdet = cov.distance(self.ym, theta)
        d2_ref, ld_ref = mahalanobis(self.y, self.ym, ref)
        assert d2 == pytest.approx(d2_ref) and logdet == pytest.approx(ld_ref)
        np.testing.assert_allclose(cov.matrix(self.ym, theta), ref)
        assert np.any(ref[:3, 3:6] != 0.0)  # the modes really couple blocks

    def test_rank_two_woodbury_with_block_local_kernel(self):
        gp = kernel(Matern(0.5, nu=2.5), jitter=0.0, prefix="gp")
        terms = [
            statistical(self.stat),
            offset(parameter=self.omega),
            normalization(parameter=self.eta),
            gp,
        ]
        rows = [np.arange(9), np.arange(9), np.arange(9), self.b[1]]
        cov, params = build(terms, self.x, self.y, self.offsets, rows=rows)
        assert not cov.dense
        values = [(), (np.log(0.07),), (np.log(0.05),), (np.log(0.4),)]
        theta = theta_for(params, values, terms)
        ref = self.reference(terms, rows, values)
        np.testing.assert_allclose(
            cov.distance(self.ym, theta), mahalanobis(self.y, self.ym, ref)
        )
        np.testing.assert_allclose(cov.matrix(self.ym, theta), ref)

    def test_cross_block_matrix_forces_dense_path(self):
        gp = kernel(RBF(1.0), jitter=0.0)
        terms = [statistical(self.stat), gp]
        rows = [np.arange(9), np.arange(6)]  # spans blocks 0 and 1
        cov, params = build(terms, self.x, self.y, self.offsets, rows=rows)
        assert cov.dense
        values = [(), (0.0,)]
        theta = theta_for(params, values, terms)
        ref = self.reference(terms, rows, values)
        np.testing.assert_allclose(
            cov.distance(self.ym, theta), mahalanobis(self.y, self.ym, ref)
        )
        local, _ = build(
            terms, self.x, self.y, self.offsets, rows=[np.arange(9), self.b[0]]
        )
        assert not local.dense

    def test_masks_restrict_to_active_rows(self):
        terms = [
            statistical(self.stat),
            noise(self.eps),
            normalization(parameter=self.eta),
        ]
        rows = [np.arange(9)] * 3
        active = np.array([0, 2, 3, 5, 7, 8])
        cov, params = build(
            terms, self.x, self.y, self.offsets, active=active, rows=rows
        )
        assert cov.n_active == 6
        values = [(), (np.log(0.2),), (np.log(0.05),)]
        theta = theta_for(params, values, terms)
        ref = self.reference(terms, rows, values)[np.ix_(active, active)]
        np.testing.assert_allclose(
            cov.distance(self.ym, theta),
            mahalanobis(self.y[active], self.ym[active], ref),
        )
        np.testing.assert_allclose(cov.matrix(self.ym, theta), ref)

    def test_fully_masked_block_is_skipped(self):
        terms = [statistical(self.stat), offset(parameter=self.omega)]
        rows = [np.arange(9)] * 2
        active = np.arange(3, 9)  # block 0 fully masked
        cov, params = build(
            terms, self.x, self.y, self.offsets, active=active, rows=rows
        )
        values = [(), (np.log(0.07),)]
        theta = theta_for(params, values, terms)
        ref = self.reference(terms, rows, values)[np.ix_(active, active)]
        np.testing.assert_allclose(
            cov.distance(self.ym, theta),
            mahalanobis(self.y[active], self.ym[active], ref),
        )


# ----------------------------------------------------------------------------
# Constant parts, caching and the singular check
# ----------------------------------------------------------------------------


class TestConstantAndSingular:
    def setup_method(self):
        self.x, self.y, self.ym = grid(6, seed=3)
        self.offsets = [slice(0, 3), slice(3, 6)]

    def test_constant_covariance_is_evaluated_and_factored_once(self):
        calls = []

        def fn(c):
            calls.append(1)
            return 0.1 * np.ones(len(c))

        t = Term(fn, kind="diag", constant=True)
        cov, _ = build([t], self.x, self.y, self.offsets)
        assert cov.is_constant
        d1 = cov.distance(self.ym, [])
        d2 = cov.distance(self.ym + 0.1, [])
        assert len(calls) == 1
        assert d1[1] == d2[1]  # same logdet from the cached factor
        assert d1[0] != d2[0]

    def test_prediction_dependent_parameter_free_term_is_not_constant(self):
        t = normalization(magnitude=0.05)
        cov, _ = build([statistical(0.1 * np.ones(6)), t], self.x, self.y, self.offsets)
        assert not cov.is_constant
        S1, S2 = cov.matrix(self.ym, []), cov.matrix(2 * self.ym, [])
        assert not np.allclose(S1, S2)

    def test_mode_only_block_is_singular_and_named(self):
        terms = [offset(magnitude=0.2), statistical(0.1 * np.ones(3))]
        rows = [np.arange(6), np.arange(3, 6)]
        with pytest.raises(ValueError, match="'first'.*zero statistical error"):
            build(
                terms,
                self.x,
                self.y,
                self.offsets,
                rows=rows,
                labels=["first", "second"],
            )
        # a diagonal term covering the block makes it legal
        terms = [offset(magnitude=0.2), statistical(0.1 * np.ones(6))]
        cov, _ = build(terms, self.x, self.y, self.offsets, rows=[np.arange(6)] * 2)
        assert cov.is_constant

    def test_modes_alone_fail_at_construction_even_when_parametric(self):
        # modes never enter the block factor B, so B is constant and checkable
        with pytest.raises(ValueError, match="singular"):
            build([offset(parameter=Parameter("w"))], self.x, self.y, self.offsets)

    def test_parametric_diagonal_defers_the_check(self):
        # B depends on theta here: nothing to check until the first evaluation
        cov, _ = build([noise(Parameter("e"))], self.x, self.y, self.offsets)
        assert not cov.is_constant
        d2, logdet = cov.distance(self.ym, [np.log(0.3)])
        assert np.isfinite(d2) and np.isfinite(logdet)


def test_chol_logdet_on_a_diagonal():
    L, logdet = chol_logdet(np.diag([1.0, 4.0, 9.0]))
    np.testing.assert_allclose(np.diag(L), [1.0, 2.0, 3.0])
    assert logdet == pytest.approx(np.log(36.0))


class TestSegments:
    """A spanning term sees the rows of each block it touches, in stack order."""

    def setup_method(self):
        self.x, self.y, self.ym = grid(9, seed=3)
        self.offsets = [slice(0, 3), slice(3, 6), slice(6, 9)]

    def capture(self, rows, active=None):
        seen = {}

        def fn(c):
            seen["segments"], seen["labels"] = c.segments, c.labels
            seen["x"] = c.split(c.x)
            return np.ones(len(c))

        terms = [statistical(0.1 * np.ones(9)), Term(fn, kind="mode")]
        cov, _ = build(
            terms,
            self.x,
            self.y,
            self.offsets,
            active=active,
            rows=[np.arange(9), rows],
            labels=["L0", "L1", "L2"],
        )
        cov.matrix(self.ym, np.zeros(0))
        return seen

    def test_whole_stack(self):
        seen = self.capture(np.arange(9))
        assert seen["segments"] == (slice(0, 3), slice(3, 6), slice(6, 9))
        assert seen["labels"] == ("L0", "L1", "L2")
        np.testing.assert_array_equal(seen["x"][1], self.x[3:6])

    def test_partial_support_skips_untouched_blocks(self):
        # blocks 0 and 2 only: the support is 6 rows in two segments
        seen = self.capture(np.r_[0:3, 6:9])
        assert seen["segments"] == (slice(0, 3), slice(3, 6))
        assert seen["labels"] == ("L0", "L2")
        np.testing.assert_array_equal(seen["x"][1], self.x[6:9])

    def test_masked_rows_stay_in_the_segment_view(self):
        # fn sees every row of its support (masking selects after evaluation),
        # so the segments describe the unmasked support
        seen = self.capture(np.arange(9), active=np.r_[0:2, 3:9])
        assert seen["segments"] == (slice(0, 3), slice(3, 6), slice(6, 9))
