"""Recipe 1: fit a model to data with reported statistical errors.

I have x, y, and a statistical error per point, and a model with a few
parameters.  I want the posterior.
"""

import numpy as np
import pytest

from common import TRUE, line_problem, map_estimate
from oracle import linear_gaussian


def test_covariance_is_the_statistical_diagonal_and_chi2_is_the_weighted_sum():
    p, d = line_problem()
    theta = np.array(TRUE)
    ym = TRUE[0] * d.x + TRUE[1]
    assert p.chi2(theta) == pytest.approx(np.sum(((d.y - ym) / d.y_err) ** 2))
    np.testing.assert_allclose(p.constraints[0].matrix(theta), np.diag(d.y_err**2))


def test_names_and_dimension_follow_the_declaration():
    p, _ = line_problem()
    assert p.ndim == 2 and p.names == ["m", "b"]


def test_map_matches_the_linear_gaussian_oracle():
    p, d = line_problem()
    Xd = np.column_stack([d.x, np.ones_like(d.x)])
    mean, cov, _ = linear_gaussian(
        Xd, d.y, np.diag(d.y_err**2), [0.0, 0.0], 25.0 * np.eye(2)
    )
    np.testing.assert_allclose(map_estimate(p, TRUE), mean, atol=1e-4)


def test_the_flat_interface_is_all_a_sampler_needs():
    p, _ = line_problem()
    p0 = p.sample_prior(4, rng=0)
    assert p0.shape == (4, 2)
    assert np.all(np.isfinite(p.prior_transform([0.5, 0.5])))
    assert np.isfinite(p.log_posterior(p0[0]))
