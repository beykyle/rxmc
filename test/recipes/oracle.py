"""Closed-form posterior of a linear-Gaussian model, the fast tier's oracle.

For ``y = X theta + eps`` with ``eps ~ N(0, Sigma)`` and ``theta ~ N(mu0, C0)``
the posterior is Gaussian with the mean and covariance below, and the log
evidence is that of ``y ~ N(X mu0, X C0 X^T + Sigma)``.  Recipes instantiated
with a linear model and Gaussian terms are checked against this without any
sampling.
"""

import numpy as np
from scipy import stats


def linear_gaussian(X, y, Sigma, mu0, C0):
    """``(mean, cov, log_evidence)`` of the posterior over ``theta``."""
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    Sigma, C0 = np.atleast_2d(Sigma), np.atleast_2d(C0)
    mu0 = np.asarray(mu0, dtype=float)
    Si = np.linalg.inv(Sigma)
    C0i = np.linalg.inv(C0)
    cov = np.linalg.inv(C0i + X.T @ Si @ X)
    mean = cov @ (C0i @ mu0 + X.T @ Si @ y)
    log_ev = stats.multivariate_normal(X @ mu0, X @ C0 @ X.T + Sigma).logpdf(y)
    return mean, cov, float(log_ev)
