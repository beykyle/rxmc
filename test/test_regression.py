"""Regression pins carried over from 0.x.

The 0.x default covariance silently folded a dataset's reported
normalisation and offset systematics into the likelihood.  The 1.0 default
is statistical only, and the systematics become terms when asked for
(``comparison.reported_terms()``).  These pins record that the old number is
recovered exactly by asking, that the default differs, and that a mode
spanning two comparisons changes the likelihood the way a dense reference
says it should.
"""

import numpy as np
import pytest
from scipy import stats

from helpers import manual_mvn_loglike
from rxmc import Comparison, Constraint, Dataset, Model, Parameter, Problem
from rxmc import terms as T

X = np.array([1.0, 2.0, 3.0, 4.0])
Y = np.array([2.1, 3.9, 6.2, 7.8])
STAT = np.array([0.1, 0.15, 0.2, 0.25])
NORM, OFFSET = 0.05, 0.02
THETA = np.array([0.2, 1.9])  # a0, a1 of the 0.x Polynomial(order=1)
PINNED = 1.195784087817536


def line():
    a0 = Parameter("a0", prior=stats.norm(0, 10))
    a1 = Parameter("a1", prior=stats.norm(0, 10))
    return Model(lambda x, a0, a1: a0 + a1 * x, [a0, a1])


def old_covariance(ym):
    ones = np.ones_like(ym)
    return (
        np.diag(STAT**2) + OFFSET**2 * np.outer(ones, ones) + NORM**2 * np.outer(ym, ym)
    )


class TestSystematicDefaultBehaviourChange:
    def setup_method(self):
        self.d = Dataset(X, Y, STAT, norm_err=NORM, offset_err=OFFSET, label="pin")
        self.comp = Comparison(self.d, line())
        self.ym = THETA[0] + THETA[1] * X

    def test_old_value(self):
        assert manual_mvn_loglike(Y, self.ym, old_covariance(self.ym)) == pytest.approx(
            PINNED, abs=1e-9
        )

    def test_new_default_is_statistical_only_and_differs(self):
        p = Problem([Constraint([self.comp])])
        assert p.log_likelihood(THETA) == pytest.approx(
            manual_mvn_loglike(Y, self.ym, np.diag(STAT**2))
        )
        assert p.log_likelihood(THETA) != pytest.approx(PINNED)

    def test_reported_terms_recover_old_value(self):
        p = Problem([Constraint([self.comp], terms=self.comp.reported_terms())])
        assert p.log_likelihood(THETA) == pytest.approx(PINNED, abs=1e-9)


class TestSpanningMode:
    """A normalisation mode across two comparisons equals the dense reference."""

    def setup_method(self):
        self.model = line()
        self.d1 = Dataset(X, Y, STAT, label="one")
        self.d2 = Dataset(X + 4.0, Y + 7.6, STAT, label="two")
        self.comps = [Comparison(self.d1, self.model), Comparison(self.d2, self.model)]

    def test_independent_blocks_match_a_dense_cholesky(self):
        p = Problem(
            [
                Constraint(
                    self.comps,
                    terms=[T.normalization(magnitude=NORM, on=c) for c in self.comps],
                )
            ]
        )
        ym = np.concatenate([THETA[0] + THETA[1] * X, THETA[0] + THETA[1] * (X + 4.0)])
        y = np.concatenate([Y, Y + 7.6])
        S = np.diag(np.tile(STAT, 2) ** 2)
        S[:4, :4] += NORM**2 * np.outer(ym[:4], ym[:4])
        S[4:, 4:] += NORM**2 * np.outer(ym[4:], ym[4:])
        assert p.log_likelihood(THETA) == pytest.approx(manual_mvn_loglike(y, ym, S))
        assert not p.constraints[0].covariance.dense

    def test_a_mode_spanning_both_changes_the_likelihood(self):
        p_each = Problem(
            [
                Constraint(
                    self.comps,
                    terms=[T.normalization(magnitude=NORM, on=c) for c in self.comps],
                )
            ]
        )
        p_span = Problem(
            [
                Constraint(
                    self.comps, terms=[T.normalization(magnitude=NORM, on=self.comps)]
                )
            ]
        )
        S = p_span.constraints[0].matrix(THETA)
        assert np.all(S[:4, 4:] != 0.0)
        ym = np.concatenate([THETA[0] + THETA[1] * X, THETA[0] + THETA[1] * (X + 4.0)])
        y = np.concatenate([Y, Y + 7.6])
        dense = np.diag(np.tile(STAT, 2) ** 2) + NORM**2 * np.outer(ym, ym)
        assert p_span.log_likelihood(THETA) == pytest.approx(
            manual_mvn_loglike(y, ym, dense)
        )
        assert p_span.log_likelihood(THETA) != pytest.approx(
            p_each.log_likelihood(THETA)
        )
