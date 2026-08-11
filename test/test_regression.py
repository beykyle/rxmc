"""
Regression pins for the deliberate behaviour change of the covariance refactor.

The old default covariance silently folded a dataset's normalisation/offset
systematics into ``Observation.covariance``.  The new default is **statistical
only**; those systematics must be stated explicitly as covariance
:class:`~rxmc.covariance.Term` s.  These tests pin:

1. the before/after log-posterior gap (the default changed), and
2. that re-adding the explicit terms exactly recovers the old number.

They also pin the block-diagonal fast path against a dense Cholesky and a genuine
case-A cross-dataset coupling, so the new capabilities are recorded.
"""

import unittest

import numpy as np

from helpers import manual_mvn_loglike as manual_mvn
from rxmc.constraint import Constraint
from rxmc.covariance import normalization_term, offset_term
from rxmc.observation import Observation
from rxmc.physical_model import Polynomial


class TestSystematicDefaultBehaviourChange(unittest.TestCase):
    """systematic_err_demo / normalization_inference behaviour change."""

    def setUp(self):
        self.x = np.array([1.0, 2.0, 3.0, 4.0])
        self.y = np.array([2.1, 3.9, 6.2, 7.8])
        self.stat = np.array([0.1, 0.15, 0.2, 0.25])
        self.norm = 0.05  # fractional normalisation systematic
        self.offset = 0.02  # absolute offset systematic
        self.obs = Observation(self.x, self.y, y_stat_err=self.stat)
        self.pm = Polynomial(order=1)
        self.model_params = (0.2, 1.9)
        self.ym = self.pm.evaluate(self.obs, *self.model_params)

        # the matrix the OLD Observation.covariance(ym) produced
        ones = np.ones(4)
        self.old_cov = (
            np.diag(self.stat**2)
            + np.outer(self.offset * ones, self.offset * ones)
            + np.outer(self.norm * ones, self.norm * ones) * np.outer(self.ym, self.ym)
        )

    def test_old_value(self):
        # pinned old log-likelihood (auto-included systematics)
        old = manual_mvn(self.y, self.ym, self.old_cov)
        self.assertAlmostEqual(old, 1.195784087817536, places=9)

    def test_new_default_is_statistical_only_and_differs(self):
        c = Constraint([self.obs], self.pm)
        new_default = c.log_likelihood(self.model_params)
        stat_only = manual_mvn(self.y, self.ym, np.diag(self.stat**2))
        self.assertAlmostEqual(new_default, stat_only)
        # the default genuinely changed
        old = manual_mvn(self.y, self.ym, self.old_cov)
        self.assertNotAlmostEqual(new_default, old)

    def test_explicit_terms_recover_old_value(self):
        support = np.arange(4)
        c = Constraint(
            [self.obs],
            self.pm,
            extra_terms=[
                offset_term(support, magnitude=self.offset),
                normalization_term(support, magnitude=self.norm),
            ],
        )
        recovered = c.log_likelihood(self.model_params)
        old = manual_mvn(self.y, self.ym, self.old_cov)
        self.assertAlmostEqual(recovered, old)


class TestBlockDiagonalFastPathEquivalence(unittest.TestCase):
    """The block-diagonal fast path equals a dense Cholesky over the full stack."""

    def test_multi_block_matches_dense(self):
        pm = Polynomial(order=1)
        mp = (0.3, 1.1)
        obs = [
            Observation(
                np.array([1.0, 2.0]),
                np.array([1.5, 2.4]),
                y_stat_err=np.array([0.1, 0.2]),
            ),
            Observation(
                np.array([3.0, 4.0, 5.0]),
                np.array([3.2, 4.5, 5.9]),
                y_stat_err=np.array([0.2, 0.1, 0.3]),
            ),
            Observation(np.array([6.0]), np.array([7.1]), y_stat_err=np.array([0.15])),
        ]
        c = Constraint(obs, pm)
        self.assertTrue(c.covariance.block_diagonal)
        fast = c.log_likelihood(mp)

        y = np.concatenate([o.y for o in obs])
        ym = np.concatenate([pm.evaluate(o, *mp) for o in obs])
        stat = np.concatenate([o.y_stat_err for o in obs])
        dense = manual_mvn(y, ym, np.diag(stat**2))
        self.assertAlmostEqual(fast, dense, places=10)


class TestCaseACrossDatasetCorrelation(unittest.TestCase):
    """A correlated systematic shared across two datasets (case A)."""

    def test_off_diagonal_blocks_present_and_changes_likelihood(self):
        pm = Polynomial(order=1)
        mp = (0.5, 1.0)
        obs1 = Observation(
            np.array([1.0, 2.0]), np.array([1.6, 2.9]), y_stat_err=np.array([0.1, 0.1])
        )
        obs2 = Observation(
            np.array([3.0, 4.0]), np.array([3.4, 4.6]), y_stat_err=np.array([0.1, 0.1])
        )

        from rxmc.params import Parameter

        eta = Parameter("log eta")
        coupling = normalization_term(np.arange(4), parameter=eta)
        c = Constraint([obs1, obs2], pm, extra_terms=[coupling])

        # the assembled Sigma has non-zero cross-block (off-diagonal) entries
        ym = np.concatenate([pm.evaluate(obs1, *mp), pm.evaluate(obs2, *mp)])
        ctx = c._stack(mp)
        Sigma = c.covariance.matrix(ctx, np.log(0.1))
        self.assertFalse(c.covariance.block_diagonal)
        self.assertGreater(abs(Sigma[0, 2]), 0.0)

        # and it equals the explicit dense form
        cov = np.diag(np.full(4, 0.1**2)) + 0.1**2 * np.outer(ym, ym)
        expected = manual_mvn(np.concatenate([obs1.y, obs2.y]), ym, cov)
        self.assertAlmostEqual(c.log_likelihood(mp, (np.log(0.1),)), expected)


if __name__ == "__main__":
    unittest.main()
