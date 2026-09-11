"""Recipe 30: leave-one-experiment-out prediction.

I want to know whether the calibrated model, with its discrepancy, predicts
an experiment it was not fit to, and with what tolerance.
"""

import numpy as np
import pytest
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

from common import TRUE, line, line_data, oracle_samples
from rxmc import Comparison, Constraint, Problem
from rxmc import terms as T
from rxmc.diagnostics import coverage_curve, heldout_log_predictive, predictive_draws

DATA = [line_data(seed=s, label=f"d{s}", err=0.1) for s in range(3)]


def split(i, terms=()):
    model = line()
    comps = [Comparison(d, model) for d in DATA]
    c = Constraint(comps, terms=[t(comps) for t in terms])
    fit_c = c.masked([np.full(d.n, j != i) for j, d in enumerate(DATA)])
    return Problem([fit_c]), Problem([fit_c.complement()]), Problem([c])


def test_held_out_draws_coverage_and_tolerance():
    for i in range(3):
        fit, held, _ = split(i)
        s = oracle_samples(fit, 300, rng=i)
        draws = predictive_draws(held, s, n_rep=4, rng=i)
        h = held.constraints[0]
        assert draws.shape == (1200, DATA[i].n) and h.n_active == DATA[i].n
        tol = np.percentile(np.abs(draws - draws.mean(0)), 90, axis=0)
        assert tol.shape == (DATA[i].n,) and np.all(tol > 0)
        cov68 = coverage_curve(draws, h.y[h.active], [0.68])[0]
        assert 0.3 <= cov68 <= 1.0  # eight points: coarse, but not empty
        # predictions on the held-out experiment centre near the truth
        np.testing.assert_allclose(
            draws.mean(0), TRUE[0] * DATA[i].x + TRUE[1], atol=0.15
        )


def spanning_gp(comps):
    return T.kernel(ConstantKernel(0.1**2, "fixed") * RBF(1.0, "fixed"), on=comps)


def test_a_discrepancy_fit_to_the_other_experiments_carries_into_the_prediction():
    fit, held, full = split(2, terms=[spanning_gp])
    theta = np.array(TRUE)
    # the marginal held-out block ignores what the fitted experiments taught the GP
    marginal = predictive_draws(held, theta, model_only=True)[0]
    conditional = predictive_draws(held, theta, model_only=True, given=fit)[0]
    np.testing.assert_allclose(marginal, TRUE[0] * DATA[2].x + TRUE[1])
    assert not np.allclose(conditional, marginal)
    # the conditional density is the joint divided by the fit, exactly
    lp = heldout_log_predictive(held, theta, given=fit)[0]
    assert lp == pytest.approx(full.log_likelihood(theta) - fit.log_likelihood(theta))
    # ...which the marginal is not, because the GP spans the split
    assert heldout_log_predictive(held, theta)[0] != pytest.approx(lp)
    s = oracle_samples(fit, 200, rng=0)
    draws = predictive_draws(held, s, n_rep=4, rng=0, given=fit)
    assert draws.shape == (800, DATA[2].n) and np.all(np.isfinite(draws))
