"""Recipe 33: MAP and Laplace approximation.

I want a quick Gaussian approximation to the posterior, and to know when it
is good enough.
"""

import numpy as np
from scipy.optimize import minimize

from common import TRUE, line_problem
from oracle import linear_gaussian


def numerical_hessian(f, x, h=1e-4):
    x = np.asarray(x, float)
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            e_i, e_j = np.eye(n)[i] * h, np.eye(n)[j] * h
            H[i, j] = (
                f(x + e_i + e_j)
                - f(x + e_i - e_j)
                - f(x - e_i + e_j)
                + f(x - e_i - e_j)
            ) / (4 * h * h)
    return H


def test_laplace_covariance_equals_the_oracle_on_a_linear_gaussian_problem():
    p, d = line_problem()
    nll = lambda t: -p.log_posterior(t)  # noqa: E731
    res = minimize(nll, np.array(TRUE), method="L-BFGS-B", bounds=p.bounds)
    H = numerical_hessian(nll, res.x)
    cov = np.linalg.inv(H)
    Xd = np.column_stack([d.x, np.ones_like(d.x)])
    mean, cov_ref, _ = linear_gaussian(
        Xd, d.y, np.diag(d.y_err**2), [0.0, 0.0], 25.0 * np.eye(2)
    )
    np.testing.assert_allclose(res.x, mean, atol=1e-4)
    np.testing.assert_allclose(cov, cov_ref, rtol=1e-3)
    assert p.bounds.shape == (2, 2)  # bounds feed the optimiser
