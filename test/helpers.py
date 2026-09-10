"""Shared helpers for the test suite."""

import numpy as np

from rxmc.covariance import StackContext
from rxmc.likelihood_model import log_likelihood, mahalanobis_distance_sqr_cholesky


def make_ctx(x, y, ym, supports) -> StackContext:
    """Build a StackContext from stacked arrays and block supports."""
    return StackContext(
        x=np.asarray(x),
        y=np.asarray(y),
        ym=np.asarray(ym),
        supports=tuple(np.asarray(s, dtype=int) for s in supports),
    )


def manual_mvn_loglike(y, ym, cov):
    """Reference dense multivariate-normal log likelihood."""
    d2, logdet = mahalanobis_distance_sqr_cholesky(y, ym, cov)
    return log_likelihood(d2, logdet, len(y))
