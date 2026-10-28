"""The structured (Woodbury) covariance against the dense reference."""

import numpy as np
import pytest
from scipy import stats
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
from rxmc.terms import (
    Term,
    constant_amplitude,
    kernel,
    noise,
    normalization,
    offset,
    statistical,
)


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


class TestNonFinite:
    """A non-finite covariance is zero density at a theta, an error at compile."""

    def setup_method(self):
        from rxmc import Comparison, Dataset, Model

        x = np.linspace(0.0, 2.0, 6)
        self.m = Parameter("m", prior=stats.norm(0, 5))
        model = Model(lambda x, m: m * x, [self.m])
        self.comps = [
            Comparison(
                Dataset(x, 2.0 * x + 0.1 * k, 0.1 * np.ones(6), label=f"d{k}"), model
            )
            for k in range(2)
        ]

    def problem(self, terms, comps=None, masks=None):
        from rxmc import Constraint, Problem

        c = Constraint(comps or self.comps[:1], terms=terms)
        return Problem([c if masks is None else c.masked(masks)])

    def assert_no_density(self, p, theta):
        assert p.log_likelihood(theta) == -np.inf
        assert p.log_posterior(theta) == -np.inf
        assert p.chi2(theta) == np.inf

    def amplitude_kernel(self):
        la = Parameter("log_A", prior=stats.norm(0, 1))
        return kernel(
            RBF(0.5, "fixed"), amplitude=constant_amplitude, amplitude_params=(la,)
        )

    # -- the likelihood: -inf, and no numpy warnings ---------------------------

    @pytest.mark.filterwarnings("error")
    def test_overflowing_kernel_amplitude_structured_path(self):
        p = self.problem([self.amplitude_kernel()])
        assert not p.constraints[0].covariance.dense
        assert np.isfinite(p.log_likelihood(np.array([2.0, 0.0])))
        self.assert_no_density(p, np.array([2.0, 400.0]))  # exp(400)**2 overflows

    @pytest.mark.filterwarnings("error")
    def test_overflowing_kernel_amplitude_dense_path(self):
        p = self.problem([self.amplitude_kernel()], comps=self.comps)  # dense
        assert p.constraints[0].covariance.dense
        self.assert_no_density(p, np.array([2.0, 400.0]))

    @pytest.mark.filterwarnings("error")
    def test_finite_values_whose_square_overflows(self):
        # exp(400) is finite; the diagonal adds its square, which is not
        p = self.problem([noise(Parameter("log_eps", prior=stats.norm(0, 1)))])
        self.assert_no_density(p, np.array([2.0, 400.0]))

    @pytest.mark.filterwarnings("error")
    def test_overflowing_residual_on_the_woodbury_path(self):
        # every term finite; z@z and s@s overflow and d2 = inf - inf is NaN
        from rxmc import Comparison, Dataset, Model

        m = Parameter("m", prior=stats.norm(0, 200))
        x = np.linspace(0.0, 2.0, 6)
        comp = Comparison(
            Dataset(x, 2.0 * x, 0.1 * np.ones(6)),
            Model(lambda x, m: np.exp(m) * x, [m]),
        )
        p = self.problem([offset(magnitude=10.0)], comps=[comp])
        assert np.isfinite(p.log_likelihood(np.array([0.5])))
        self.assert_no_density(p, np.array([370.0]))

    def test_parametric_nan_is_zero_density_and_warns_once(self):
        import warnings

        s = Parameter("s", prior=stats.norm(0, 1))
        t = Term(lambda c, s: np.sqrt(s) * np.ones(len(c)), (s,), kind="diag")
        p = self.problem([t])
        assert np.isfinite(p.log_likelihood(np.array([2.0, 0.04])))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(10):
                self.assert_no_density(p, np.array([2.0, -1.0]))
        # one named warning; numpy's per-call "invalid value in sqrt" is silenced
        assert len(caught) == 1
        assert issubclass(caught[0].category, RuntimeWarning)
        assert "NaN" in str(caught[0].message) and "d0" in str(caught[0].message)

    # -- construction: a constant term is checked once and named ---------------

    def test_constant_term_with_nan_fails_at_compile_and_names_it(self):
        bad = np.full(6, 0.1)
        bad[2] = np.nan
        with pytest.raises(ValueError, match="kind='diag'.*non-finite.*declaration"):
            self.problem([Term(bad, kind="diag")])

    @pytest.mark.parametrize("kind", ["mode", "matrix"])
    def test_constant_mode_or_matrix_with_nan_names_it(self, kind):
        bad = np.full(6, 0.1) if kind == "mode" else 0.01 * np.eye(6)
        bad[(2,) if kind == "mode" else (2, 4)] = np.nan
        if kind == "matrix":
            bad[4, 2] = np.nan
        with pytest.raises(ValueError, match=f"kind='{kind}'.*non-finite"):
            self.problem([Term(bad, kind=kind)])

    def test_nan_on_a_masked_row_compiles(self):
        bad = np.full(6, 0.1)
        bad[2] = np.nan
        mask = np.ones(6, bool)
        mask[2] = False
        p = self.problem([Term(bad, kind="diag")], masks=[mask])
        assert np.isfinite(p.log_likelihood(np.array([2.0])))

    def test_constant_square_overflow_raises_beside_a_parametric_term(self):
        terms = [
            Term(np.full(6, 1e200), kind="diag"),
            noise(Parameter("log_eps", prior=stats.norm(0, 1))),
        ]
        with pytest.raises(ValueError, match="non-finite.*declaration"):
            self.problem(terms)

    @pytest.mark.filterwarnings("error")
    def test_constant_mode_whose_product_overflows_raises(self):
        # the mode column is finite; W^T W is not
        with pytest.raises(ValueError, match="overflow when combined"):
            self.problem([Term(np.full(6, 1e200), kind="mode")])
