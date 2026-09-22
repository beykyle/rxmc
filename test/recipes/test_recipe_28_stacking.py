"""Recipe 28: stacking by leave-one-dataset-out.

Evidence weights assume the true model is among my candidates.  I would
rather weight models by how well they predict each dataset when it is left
out.
"""

import numpy as np
import pytest
from scipy import stats
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

from common import TRUE, line, line_data, line_design, linear_posterior, oracle_samples
from rxmc import Comparison, Constraint, Model, Parameter, Problem
from rxmc.diagnostics import heldout_log_predictive, log_posterior_predictive

DATA = [line_data(seed=s, label=f"d{s}", err=0.1) for s in range(3)]


def constant():
    b = Parameter("b", prior=stats.norm(0, 5))
    return Model(lambda x, b: b * np.ones_like(x), [b])


def constant_design(x):
    return np.ones((len(x), 1))


def loo_scores(model, design, rng=0):
    """One held-out log score per dataset, with exact posterior samples."""
    comps = [Comparison(d, model) for d in DATA]  # one model object, shared
    c = Constraint(comps)
    out = []
    for i in range(len(comps)):
        fit_c = c.masked(
            [np.full(comp.data.y.shape, j != i) for j, comp in enumerate(comps)]
        )
        fit, held = Problem([fit_c]), Problem([fit_c.complement()])
        s = oracle_samples(fit, 4000, rng=rng, design=design)
        out.append(log_posterior_predictive(heldout_log_predictive(held, s)))
    return np.array(out)


def test_held_out_scores_match_the_closed_form_marginal():
    model = line()
    comps = [Comparison(d, model) for d in DATA]
    c = Constraint(comps)
    fit_c = c.masked([np.full(d.n, j != 2) for j, d in enumerate(DATA)])
    fit, held = Problem([fit_c]), Problem([fit_c.complement()])
    s = oracle_samples(fit, 4000, rng=0)
    score = log_posterior_predictive(heldout_log_predictive(held, s))
    # p(y_h | y_fit) = N(X_h mean, X_h cov X_h^T + Sigma_h) for the linear model
    mean, cov, _ = linear_posterior(fit)
    h = held.constraints[0]
    Xh = np.column_stack([h.x[h.active], np.ones(h.n_active)])
    Sh = h.matrix(np.zeros(fit.ndim))
    closed = stats.multivariate_normal(Xh @ mean, Xh @ cov @ Xh.T + Sh).logpdf(
        h.y[h.active]
    )
    assert score == pytest.approx(closed, abs=0.15)
    assert h.n_active == DATA[2].n  # the whole third dataset was held out


def test_stacking_weights_prefer_the_model_that_predicts():
    S = np.stack(
        [loo_scores(line(), line_design), loo_scores(constant(), constant_design)]
    )
    assert S.shape == (2, 3)

    def objective(z):  # softmax keeps w on the simplex
        return -np.sum(logsumexp(np.log(softmax(z))[:, None] + S, axis=0))

    w = softmax(minimize(objective, np.zeros(2)).x)
    assert w.sum() == pytest.approx(1.0)
    assert w[0] > 0.95  # the line predicts the held-out datasets; the constant does not
    assert np.all(S[0] > S[1])
    assert (
        TRUE[0] != 0.0
    )  # the datasets really do have a slope for the constant to miss
