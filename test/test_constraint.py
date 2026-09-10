"""Comparison and Constraint: comparison space, reported terms, masks, support."""

import numpy as np
import pytest

from helpers import assemble_dense
from rxmc import Dataset, Model, Parameter
from rxmc.constraint import Comparison, Constraint
from rxmc.likelihood import StudentT
from rxmc.terms import Term, noise, statistical
from rxmc.transforms import Transform, log, scale

m, b = Parameter("m"), Parameter("b")
line = Model(lambda x, m, b: m * x + b, [m, b])


def dataset(n=4, label="d", **kw):
    x = np.linspace(1.0, 2.0, n)
    return Dataset(x, 2.0 * x + 1.0, 0.1 * np.ones(n), label=label, **kw)


class TestComparison:
    def test_identity_space_is_the_default(self):
        d = dataset()
        c = Comparison(d, line)
        assert c.y is d.y and c.y_err is d.y_err
        assert c.log_jacobian() == 0.0
        assert c.n == 4
        np.testing.assert_allclose(c.predict(2.0, 1.0), d.y)

    def test_log_space_delta_method_and_jacobian(self):
        d = dataset()
        c = Comparison(d, line, space=log)
        np.testing.assert_allclose(c.y, np.log(d.y))
        np.testing.assert_allclose(c.y_err, d.y_err / d.y)
        assert c.log_jacobian() == pytest.approx(-np.sum(np.log(d.y)))
        mask = np.array([True, False, True, False])
        assert c.log_jacobian(mask) == pytest.approx(-np.sum(np.log(d.y[mask])))
        np.testing.assert_allclose(c.predict(2.0, 1.0), np.log(d.y))

    def test_non_positive_data_under_log_does_not_raise(self):
        d = Dataset([0.0, 1.0], [-1.0, 2.0], [0.1, 0.1])
        c = Comparison(d, line, space=log)
        assert not np.isfinite(c.y[0]) and np.isfinite(c.y[1])

    def test_parametric_space_rejected_and_callable_accepted(self):
        with pytest.raises(ValueError, match="parameter-free"):
            Comparison(dataset(), line, space=scale())
        c = Comparison(dataset(), line, space=np.sqrt)
        np.testing.assert_allclose(c.y, np.sqrt(dataset().y))

    def test_predictor_is_bound_with_meta(self):
        seen = {}

        class Probe(Model):
            def bind(self, x, meta=None):
                seen["meta"] = meta
                return line.bind(x)

        d = dataset(meta={"Elab": 10.0})
        Comparison(d, Probe(None, [m, b]))
        assert seen["meta"] == {"Elab": 10.0}

    def test_type_checks(self):
        with pytest.raises(TypeError, match="Dataset"):
            Comparison(np.ones(3), line)
        with pytest.raises(TypeError, match="Model"):
            Comparison(dataset(), lambda x: x)


class TestReportedTerms:
    def test_empty_and_zero_skipped(self):
        assert Comparison(dataset(), line).reported_terms() == []
        d = dataset(norm_err=0.0, offset_err=np.zeros(4))
        assert Comparison(d, line).reported_terms() == []

    def test_identity_recovers_the_old_folded_covariance(self):
        d = dataset(norm_err=0.05, offset_err=0.2)
        c = Comparison(d, line)
        terms = c.reported_terms()
        assert [t.kind for t in terms] == ["mode", "mode"]
        assert all(t.on is c for t in terms)
        ym = c.predict(1.9, 1.1)
        S = assemble_dense([statistical(c.y_err), *terms], d.x, c.y, ym)
        old = (
            np.diag(d.y_err**2)
            + np.outer(0.2 * np.ones(4), 0.2 * np.ones(4))
            + 0.05**2 * np.outer(ym, ym)
        )
        np.testing.assert_allclose(S, old)

    def test_delta_method_under_log(self):
        d = dataset(norm_err=0.05, offset_err=0.2)
        c = Comparison(d, line, space=log)
        offset, norm = c.reported_terms()
        ym = c.predict(1.9, 1.1)
        np.testing.assert_allclose(offset.value(d.x, c.y, ym), 0.2 / d.y)
        np.testing.assert_allclose(
            norm.value(d.x, c.y, ym), 0.05
        )  # constant in log space

    def test_normalisation_needs_an_inverse(self):
        d = dataset(norm_err=0.05)
        c = Comparison(d, line, space=Transform(lambda a: a**3))
        with pytest.raises(ValueError, match="no inverse"):
            c.reported_terms()
        d = dataset(offset_err=0.2)  # an offset alone is fine
        assert (
            len(Comparison(d, line, space=Transform(lambda a: a**3)).reported_terms())
            == 1
        )


class TestConstraint:
    def setup_method(self):
        self.d1, self.d2 = dataset(3, "d1"), dataset(4, "d2")
        self.c1, self.c2 = Comparison(self.d1, line), Comparison(self.d2, line)
        self.eps = Parameter("log_eps")

    def test_construction_and_validation(self):
        c = Constraint(iter([self.c1, self.c2]), terms=[noise(self.eps)])
        assert c.comparisons == (self.c1, self.c2) and c.n_total == 7
        assert c.offsets == (slice(0, 3), slice(3, 7))
        assert c.n_active == 7 and np.array_equal(c.active, np.arange(7))
        assert c.masks is None and c.weight == 1.0 and c.statistical
        with pytest.raises(ValueError, match="distinct"):
            Constraint([self.c1, self.c1])
        with pytest.raises(TypeError, match="Term"):
            Constraint([self.c1], terms=[np.eye(3)])
        with pytest.raises(ValueError, match="non-negative"):
            Constraint([self.c1], weight=-1.0)
        with pytest.raises(ValueError, match="at least one"):
            Constraint([])
        with pytest.raises(TypeError, match="Likelihood"):
            Constraint([self.c1], likelihood=object())
        assert isinstance(
            Constraint([self.c1], likelihood=StudentT()).likelihood, StudentT
        )

    def test_masks_validated(self):
        with pytest.raises(ValueError, match="one entry per comparison"):
            Constraint([self.c1, self.c2], masks=[np.ones(3, bool)])
        with pytest.raises(ValueError, match="shape"):
            Constraint([self.c1, self.c2], masks=[np.ones(3, bool), np.ones(3, bool)])

    def test_support_resolution(self):
        c = Constraint([self.c1, self.c2])
        assert np.array_equal(c.support(None), np.arange(7))
        assert np.array_equal(c.support(self.c2), np.arange(3, 7))
        assert np.array_equal(c.support(self.d1), np.arange(3))
        assert np.array_equal(c.support([self.c2, self.c1]), np.arange(7))
        assert np.array_equal(c.support([self.c2, self.d2]), np.arange(3, 7))
        stray = Comparison(dataset(2, "stray"), line)
        with pytest.raises(ValueError, match="stray"):
            c.support(stray)
        with pytest.raises(ValueError, match="does not reference"):
            c.support(dataset(2, "other"))
        with pytest.raises(TypeError, match="comparisons or datasets"):
            c.support(3)

    def test_term_support_checked_eagerly(self):
        stray = Comparison(dataset(2, "stray"), line)
        with pytest.raises(ValueError, match="stray"):
            Constraint([self.c1], terms=[noise(self.eps, on=stray)])
        with pytest.raises(ValueError, match="expects shape"):
            Constraint(
                [self.c1, self.c2], terms=[Term(np.ones(3), kind="diag", on=self.c2)]
            )
        Constraint(
            [self.c1, self.c2], terms=[Term(np.ones(3), kind="diag", on=self.c1)]
        )

    def test_masked_where_and_complement_partition(self):
        c = Constraint([self.c1, self.c2], terms=[noise(self.eps)])
        fit = c.masked_where(lambda x: x < 1.6)
        held = fit.complement()
        assert set(fit.active) | set(held.active) == set(range(7))
        assert set(fit.active).isdisjoint(held.active)
        assert fit.n_active + held.n_active == 7
        again = held.complement()
        assert all(np.array_equal(a, b) for a, b in zip(again.masks, fit.masks))
        # complement of an unmasked constraint deactivates everything
        assert c.complement().n_active == 0

    def test_views_share_objects(self):
        t = noise(self.eps)
        c = Constraint([self.c1, self.c2], terms=[t], weight=0.5, likelihood=StudentT())
        view = c.masked_where(lambda x: x > 1.5)
        assert view.comparisons == c.comparisons and view.terms == (t,)
        assert view.terms[0].params[0] is self.eps
        assert view.likelihood is c.likelihood and view.weight == 0.5

    def test_log_jacobian_over_active_points(self):
        c1 = Comparison(self.d1, line, space=log)
        c = Constraint([c1, self.c2])
        assert c.log_jacobian == pytest.approx(-np.sum(np.log(self.d1.y)))
        masked = c.masked([np.array([True, False, False]), np.ones(4, bool)])
        assert masked.log_jacobian == pytest.approx(-np.log(self.d1.y[0]))
