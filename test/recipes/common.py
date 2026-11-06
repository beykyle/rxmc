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


# ----------------------------------------------------------------------------
# Exact posteriors for problems whose model is linear in its parameters
# ----------------------------------------------------------------------------


def line_design(x):
    """Design matrix of :func:`line` in its parameter order ``(m, b)``."""
    x = np.asarray(x, dtype=float)
    return np.column_stack([x, np.ones_like(x)])


def linear_posterior(problem, design=line_design):
    """``(mean, cov, log_evidence)`` of a problem linear in its parameters.

    Every constraint must have a constant covariance and independent normal
    priors on the parameters (in ``problem.params`` order); ``design(x)`` maps
    a constraint's stacked ``x`` to the rows of the design matrix.  Wraps
    ``oracle.linear_gaussian`` over the active rows of all constraints.
    """
    from scipy.linalg import block_diag

    from oracle import linear_gaussian

    theta0 = np.zeros(problem.ndim)
    Xs, ys, Ss = [], [], []
    for c in problem.constraints:
        Xs.append(design(c.x)[c.active])
        ys.append(c.y[c.active])
        Ss.append(c.matrix(theta0))
    mu0 = np.array([p.prior.mean() for p in problem.params])
    C0 = np.diag([p.prior.var() for p in problem.params])
    return linear_gaussian(np.vstack(Xs), np.concatenate(ys), block_diag(*Ss), mu0, C0)


def oracle_samples(problem, n, rng, design=line_design):
    """``(n, ndim)`` exact posterior samples, a stand-in for a converged chain."""
    mean, cov, _ = linear_posterior(problem, design)
    return np.random.default_rng(rng).multivariate_normal(mean, cov, n)
