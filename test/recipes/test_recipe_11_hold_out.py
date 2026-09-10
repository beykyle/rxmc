"""Recipe 11: hold out data and score it.

I want to fit below an angular cut and score the prediction above it, or
compare error models by their held-out predictive density.
"""

import numpy as np
import pytest
from scipy import stats

from common import TRUE, line, line_data
from rxmc import Comparison, Constraint, Parameter, Problem
from rxmc import terms as T


def test_fit_and_held_out_share_columns_and_partition_the_likelihood():
    d = line_data()
    log_eps = Parameter("log_eps", prior=stats.norm(-2, 1))
    c = Constraint([Comparison(d, line())], terms=[T.noise(log_eps)])
    fit = c.masked_where(lambda x: x < 1.5)
    held = fit.complement()
    pf, ph, pa = Problem([fit]), Problem([held]), Problem([c])
    assert pf.names == ph.names == pa.names
    assert set(fit.active).isdisjoint(held.active)
    assert sorted([*fit.active, *held.active]) == list(range(d.n))
    theta = np.array([*TRUE, np.log(0.2)])
    assert pf.log_likelihood(theta) + ph.log_likelihood(theta) == pytest.approx(
        pa.log_likelihood(theta)
    )


def test_a_chain_from_the_fit_scores_the_held_out_problem_directly():
    d = line_data()
    c = Constraint([Comparison(d, line())])
    fit = c.masked_where(lambda x: x < 1.5)
    pf, ph = Problem([fit]), Problem([fit.complement()])
    samples = pf.sample_prior(5, rng=0)
    scores = np.array([ph.log_likelihood(s) for s in samples])
    assert scores.shape == (5,) and np.all(np.isfinite(scores))
