"""Recipe 27: Peelle's Pertinent Puzzle, normalise the prediction, never the data.

My datasets carry a common fractional normalisation error and I want the fit
not to be biased low.
"""

import numpy as np
import pytest
from scipy import stats

from common import map_estimate
from rxmc import Comparison, Constraint, Dataset, Model, Parameter, Problem, Term
from rxmc import terms as T


def constant_fit(y, s, mode, stat=0.02, t0=None):
    """MAP of a constant fit with the normalisation mode built three ways."""
    n = len(y)
    d = Dataset(np.arange(n), y, stat * np.ones(n))
    c0 = Parameter("c0", prior=stats.norm(1.0, 100.0))  # effectively flat
    const = Model(lambda x, c: c * np.ones_like(x, dtype=float), [c0])
    comp = Comparison(d, const)
    if mode == "data":
        term = Term(s * y, kind="mode", on=comp)  # the covariance built from the data
    elif mode == "prediction":
        term = T.normalization(magnitude=s, on=comp)  # built from the live prediction
    else:
        term = Term(s * t0 * np.ones(n), kind="mode", on=comp)  # t0: a fixed reference
    p = Problem([Constraint([comp], terms=[term])])
    return map_estimate(p, [np.mean(y)])[0]


def test_data_built_mode_is_biased_low_and_prediction_built_is_not():
    # two measurements of the same quantity, in the spirit of Peelle's puzzle
    y, s = np.array([1.5, 1.0]), 0.2
    stat = 0.02
    c_bad = constant_fit(y, s, "data")
    c_ok = constant_fit(y, s, "prediction")
    # the analytic GLS with a data-built covariance
    Sigma = np.diag([stat**2] * 2) + s**2 * np.outer(y, y)
    w = np.linalg.solve(Sigma, np.ones(2))
    gls = w @ y / w.sum()
    assert c_bad == pytest.approx(gls, rel=1e-3)
    assert c_bad < y.min()  # below both data points: the puzzle
    assert y.min() < c_ok < y.max()


def test_the_three_spellings_order_as_the_recipe_says():
    # seeded replicates of n noisy points around a constant.  The data-built
    # mode biases the fit low (D'Agostini); the live prediction-built mode
    # removes that but its log-determinant still pulls the mode down; the t0
    # refit (mode frozen at a reference prediction) is unbiased.
    rng = np.random.default_rng(0)
    val, s, sigma = 2.0, 0.3, 0.2
    means = {}
    for n in (4, 12):
        est = []
        for _ in range(40):
            y = val + rng.normal(0, sigma, n)
            est.append(
                [
                    constant_fit(y, s, "data", stat=sigma),
                    constant_fit(y, s, "prediction", stat=sigma),
                    constant_fit(y, s, "t0", stat=sigma, t0=np.mean(y)),
                ]
            )
        means[n] = np.mean(est, axis=0)
        data_built, live, t0 = means[n]
        assert data_built < live < val
        assert abs(t0 - val) < 0.1
    assert means[12][0] < means[4][0]  # the data-built bias grows with n
