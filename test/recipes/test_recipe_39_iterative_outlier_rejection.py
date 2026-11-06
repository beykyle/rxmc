"""Recipe 39: iterative outlier rejection.

A few points are gross outliers.  I want to reject them and refit, in an
outer loop of problems, until the mask stops moving.
"""

import numpy as np
import pytest

from common import TRUE, line, map_estimate
from rxmc import Comparison, Constraint, Dataset, Problem

X = np.linspace(0.5, 3.0, 12)
BAD = (3, 8)  # the points pushed off the line
PUSH = 12.0  # in units of the reported error
ERR = 0.1


def outlier_data(seed=0):
    """Twelve points on the line, two of them pushed far above it."""
    rng = np.random.default_rng(seed)
    y = TRUE[0] * X + TRUE[1] + rng.normal(0, ERR, X.size)
    y[list(BAD)] += PUSH * ERR
    return Dataset(X, y, np.full(X.size, ERR), label="outliers")


def reject(constraint, data, *, k=3.0, max_rounds=6):
    """The recipe: fit, mask the points more than ``k`` pulls away, refit."""
    mask = np.ones(data.n, dtype=bool)
    for rounds in range(1, max_rounds + 1):
        problem = Problem([constraint.masked([mask])])
        theta = map_estimate(problem, np.array(TRUE))
        pull = np.abs(data.y - problem.constraints[0].ym(theta)) / data.y_err
        keep = pull < k
        if np.array_equal(keep, mask):
            return mask, theta, rounds
        mask = keep
    raise AssertionError("the mask did not settle")


def _rms_from_truth(theta, rows):
    """How far the fitted line sits from the truth, over ``rows``."""
    offset = (theta[0] * X + theta[1]) - (TRUE[0] * X + TRUE[1])
    return float(np.sqrt(np.mean(offset[rows] ** 2)))


def test_the_loop_settles_on_the_planted_outliers():
    d = outlier_data()
    c = Constraint([Comparison(d, line())])
    mask, theta, rounds = reject(c, d)
    assert rounds > 1  # the first fit is dragged, so one pass is not enough
    assert np.array_equal(np.flatnonzero(~mask), np.array(BAD))
    ym = Problem([c.masked([mask])]).constraints[0].ym(theta)
    pull = np.abs(d.y - ym) / d.y_err
    assert np.max(pull[mask]) < 3.0 and np.min(pull[~mask]) > 10.0


def test_a_fit_that_keeps_the_outliers_is_dragged_away_from_the_truth():
    d = outlier_data()
    c = Constraint([Comparison(d, line())])
    mask, theta, _ = reject(c, d)
    naive = map_estimate(Problem([c]), np.array(TRUE))
    kept = np.flatnonzero(mask)
    assert _rms_from_truth(naive, kept) > 3 * _rms_from_truth(theta, kept)


def test_the_rejected_points_are_named_by_the_complement_and_keep_their_columns():
    d = outlier_data()
    c = Constraint([Comparison(d, line())])
    mask, _, _ = reject(c, d)
    kept, rejected = c.masked([mask]), c.masked([mask]).complement()
    assert np.array_equal(rejected.active, np.array(BAD))
    assert set(kept.active).isdisjoint(rejected.active)
    assert kept.n_active + rejected.n_active == d.n
    assert Problem([kept]).names == Problem([c]).names


def test_the_loop_is_reproducible():
    d = outlier_data()
    c = Constraint([Comparison(d, line())])
    first, theta_a, rounds_a = reject(c, d)
    second, theta_b, rounds_b = reject(c, d)
    assert np.array_equal(first, second) and rounds_a == rounds_b
    assert theta_a == pytest.approx(theta_b)
