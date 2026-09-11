"""The likelihood functionals against their closed forms."""

import numpy as np
import pytest
from scipy.special import gammaln

from helpers import mahalanobis, manual_mvn_loglike
from rxmc import Parameter
from rxmc.likelihood import Chi2, Gaussian, StudentT, log_likelihood


@pytest.fixture
def stats():
    y = np.array([2.0, 4.0, 7.0])
    ym = np.array([2.5, 4.0, 6.5])
    cov = np.diag([0.1, 0.2, 0.3]) + 0.01
    d2, logdet = mahalanobis(y, ym, cov)
    return y, ym, cov, d2, logdet


def test_gaussian_matches_manual_mvn(stats):
    y, ym, cov, d2, logdet = stats
    assert Gaussian().params == ()
    assert Gaussian().log_likelihood(d2, logdet, 3) == pytest.approx(
        manual_mvn_loglike(y, ym, cov)
    )
    assert log_likelihood(d2, logdet, 3) == pytest.approx(
        manual_mvn_loglike(y, ym, cov)
    )
    assert Gaussian().chi2(d2, logdet, 3) == d2


def test_student_t_closed_form(stats):
    _, _, _, d2, logdet = stats
    nu, n = 5.0, 3
    expected = (
        gammaln((n + nu) / 2)
        - gammaln(nu / 2)
        - 0.5 * n * np.log(np.pi * nu)
        - 0.5 * logdet
        - 0.5 * (nu + n) * np.log1p(d2 / nu)
    )
    assert StudentT().log_likelihood(d2, logdet, n, nu) == pytest.approx(expected)


def test_student_t_default_and_explicit_parameter():
    default = StudentT()
    assert [p.name for p in default.params] == ["nu"]
    assert default.params[0].bounds == (1.0, np.inf)
    # Gamma(2, rate 0.1) (Juárez & Steel 2010): a proper prior, so it compiles
    assert default.params[0].prior.mean() == pytest.approx(20.0)
    p = Parameter("nu_a", bounds=(1.0, 100.0))
    assert StudentT(nu=p).params == (p,)
    # two defaults are two distinct parameters with one name (compile rejects)
    assert StudentT().params[0] is not default.params[0]


def test_student_t_tends_to_the_gaussian_at_huge_nu(stats):
    _, _, _, d2, logdet = stats
    gauss = Gaussian().log_likelihood(d2, logdet, 3)
    for nu in (1e15, 1e16):
        assert StudentT().log_likelihood(d2, logdet, 3, nu) == pytest.approx(
            gauss, abs=1e-6
        )


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
