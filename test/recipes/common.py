"""Small synthetic problems shared by the recipe tests."""

import numpy as np
from scipy import stats

from rxmc import Comparison, Constraint, Dataset, Model, Parameter, Problem

X = np.linspace(0.5, 2.5, 8)
TRUE = (2.0, 1.0)


def line(prior_scale=5.0):
    """``y = m x + b`` with wide normal priors on both parameters."""
    m = Parameter("m", prior=stats.norm(0, prior_scale))
    b = Parameter("b", prior=stats.norm(0, prior_scale))
    return Model(lambda x, m, b: m * x + b, [m, b])


def line_data(seed=0, n=8, label="d", err=0.1, **kw):
    rng = np.random.default_rng(seed)
    x = X[:n]
    y = TRUE[0] * x + TRUE[1] + rng.normal(0, err, n)
    return Dataset(x, y, err * np.ones(n), label=label, **kw)


def line_problem(**constraint_kw):
    d = line_data()
    return Problem([Constraint([Comparison(d, line())], **constraint_kw)]), d


def map_estimate(problem, x0):
    """The posterior mode by a quasi-Newton search from ``x0``."""
    from scipy.optimize import minimize

    res = minimize(
        lambda t: -problem.log_posterior(t), np.asarray(x0, float), method="L-BFGS-B"
    )
    return res.x
