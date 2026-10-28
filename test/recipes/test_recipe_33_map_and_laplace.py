"""Recipe 33: MAP and Laplace approximation.

I want a quick Gaussian approximation to the posterior, and to know when it
is good enough.
"""

import numpy as np

import rxmc as rx
from common import line_problem
from oracle import linear_gaussian


def test_laplace_covariance_equals_the_oracle_on_a_linear_gaussian_problem():
    p, d = line_problem()
    fit = rx.optimize.find_map(p, rng=0)
    cov = np.linalg.inv(-p.hessian_log_posterior(fit.x))
    Xd = np.column_stack([d.x, np.ones_like(d.x)])
    mean, cov_ref, _ = linear_gaussian(
        Xd, d.y, np.diag(d.y_err**2), [0.0, 0.0], 25.0 * np.eye(2)
    )
    np.testing.assert_allclose(fit.x, mean, rtol=1e-6)
    np.testing.assert_allclose(cov, cov_ref, rtol=1e-8)
    # every search from the prior ends at the one mode
    np.testing.assert_allclose(fit.ends, fit.fun, rtol=1e-8)
