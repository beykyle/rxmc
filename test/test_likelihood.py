"""The likelihood functionals against their closed forms."""

import numpy as np
import pytest

from helpers import mahalanobis, manual_mvn_loglike
from rxmc.likelihood import Chi2, Gaussian, Likelihood, log_likelihood


@pytest.fixture
def stats():
    y = np.array([2.0, 4.0, 7.0])
    ym = np.array([2.5, 4.0, 6.5])
    cov = np.diag([0.1, 0.2, 0.3]) + 0.01
    d2, logdet = mahalanobis(y, ym, cov)
    return y, ym, cov, d2, logdet


def test_gaussian_matches_manual_mvn(stats):
    y, ym, cov, d2, logdet = stats
    assert Gaussian().log_likelihood(d2, logdet, 3) == pytest.approx(
        manual_mvn_loglike(y, ym, cov)
    )
    assert log_likelihood(d2, logdet, 3) == pytest.approx(
        manual_mvn_loglike(y, ym, cov)
    )
    assert Gaussian().chi2(d2, logdet, 3) == d2


def test_chi2_drops_logdet(stats):
    _, _, _, d2, logdet = stats
    assert Chi2().log_likelihood(d2, logdet, 3) == pytest.approx(-0.5 * d2)
    assert Chi2().chi2(d2, logdet, 3) == d2


def test_mahalanobis_helper_on_diagonal():
    y = np.array([1.0, 2.0, 3.0])
    ym = np.array([1.1, 1.8, 3.2])
    cov = np.diag([0.1, 0.2, 0.3])
    d2, logdet = mahalanobis(y, ym, cov)
    assert d2 == pytest.approx(np.sum((y - ym) ** 2 / np.diag(cov)))
    assert logdet == pytest.approx(np.log(np.prod(np.diag(cov))))


def test_a_likelihood_has_no_parameters():
    # a functional of (d2, logdet, n) only: nothing for the problem to sample
    for like in (Likelihood(), Gaussian(), Chi2()):
        assert not hasattr(like, "params")
