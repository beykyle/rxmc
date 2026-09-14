"""Recipe 37: correlated normalisations between quantities of one experiment.

One experiment reports several physical quantities, each measured one or
more times, all multiplied by normalisations that were themselves measured
with correlated uncertainties.  I want the covariance across the quantities
built so that it does not bias the evaluation.

The analytic study (section II.A) and the numerical study (section II.B) of
Neudecker, Frühwirth, Kawano and Leeb, Nucl. Data Sheets 118, 364 (2014) are
recreated: quantity i is rho_i = alpha_i * eta_i, alpha_i measured as q_i
(once or several times, independent errors sigma_i), eta_i as N_i with a
correlated covariance B; the data are the products r_i = q_i N_i.  C_F is
the covariance built from the measured q (Peelle), C_I the one built from
the weighted means / the prediction.
"""

import numpy as np
from scipy import stats

from common import linear_posterior, map_estimate
from rxmc import Comparison, Constraint, Dataset, Model, Parameter, Problem, Term


def experiment(q, N, sigma_q, sigma_N, corr, prior=None):
    """Comparisons for the quantities with the products ``r = q N`` as data.

    The default prior is normal with a standard deviation of 1000: flat for the
    closed-form (fixed-covariance) posteriors, which the oracle needs to be
    Gaussian.  A prediction-built covariance has a posterior tail falling only
    like 1 / rho, so the sampled cases pass a bounded uniform prior instead.
    """
    comps, rhos = [], []
    for i, (qi, Ni, si) in enumerate(zip(q, N, sigma_q)):
        qi, si = np.asarray(qi, dtype=float), np.asarray(si, dtype=float)
        rho = Parameter(f"rho_{i}", prior=prior or stats.norm(0.0, 1e3))
        rhos.append(rho)
        d = Dataset(np.full(len(qi), i), qi * Ni, Ni * si, label=f"q{i}")
        comps.append(Comparison(d, Model(lambda x, r: np.full(len(x), r), [rho])))
    return comps, rhos


def normalisation_term(comps, frac, corr, from_data=False):
    """``outer(f * ym, f * ym) * corr`` across the quantities (C_I), or from y (C_F)."""
    corr = np.asarray(corr, dtype=float)

    def fn(c):
        pieces = c.split(c.y if from_data else c.ym)
        u = np.concatenate([f * piece for f, piece in zip(frac, pieces)])
        which = np.concatenate(
            [np.full(s.stop - s.start, k) for k, s in enumerate(c.segments)]
        )
        return np.outer(u, u) * corr[np.ix_(which, which)]

    return Term(fn, kind="matrix", on=comps, constant=from_data)


def one_hot(x):
    return np.eye(int(x.max()) + 1)[x.astype(int)]


def exact(q, N, sigma_q, sigma_N, corr):
    """Means, variances and covariances from the full information (eqs. 2-8)."""
    qbar, var_a = [], []
    for qi, si in zip(q, sigma_q):
        w = 1.0 / np.asarray(si, dtype=float) ** 2
        qbar.append(np.sum(w * qi) / np.sum(w))
        var_a.append(1.0 / np.sum(w))
    qbar, var_a, N, sN = map(np.asarray, (qbar, var_a, N, sigma_N))
    mean = qbar * N
    cov = np.outer(qbar * sN, qbar * sN) * np.asarray(corr)
    cov[np.diag_indices_from(cov)] += var_a * N**2
    return mean, cov, qbar, var_a


# section II.A: two quantities, the first measured twice
Q = [np.array([1.0, 1.5]), np.array([1.8])]
N = np.array([1.0, 1.1])
SIGMA_Q = [0.1 * Q[0], 0.1 * Q[1]]
SIGMA_N = 0.2 * N
C = 0.8
CORR = np.array([[1.0, C], [C, 1.0]])


def test_the_prediction_built_covariance_is_the_analytic_solution():
    comps, rhos = experiment(Q, N, SIGMA_Q, SIGMA_N, CORR)
    mean, cov, qbar, var_a = exact(Q, N, SIGMA_Q, SIGMA_N, CORR)
    # C_I in its fixed form: the weighted means stand in for the prediction
    fixed = Term(
        _fixed_matrix(qbar * N, SIGMA_N / N, CORR, [len(q) for q in Q]),
        kind="matrix",
        on=comps,
    )
    p_I = Problem([Constraint(comps, terms=[fixed])])
    mean_I, cov_I, _ = linear_posterior(p_I, design=one_hot)
    np.testing.assert_allclose(mean_I, mean, rtol=1e-6)  # eqs. 18, 19
    np.testing.assert_allclose(cov_I, cov, rtol=1e-6)  # eqs. 20-22
    # the live term reads the prediction: at the exact means it is that matrix
    live = normalisation_term(comps, SIGMA_N / N, CORR)
    p_live = Problem([Constraint(comps, terms=[live])])
    S = p_live.constraints[0].matrix(mean)
    np.testing.assert_allclose(S, p_I.constraints[0].matrix(mean))


def _fixed_matrix(rho, frac, corr, counts):
    u = np.concatenate([np.full(n, f * r) for n, f, r in zip(counts, frac, rho)])
    which = np.concatenate([np.full(n, k) for k, n in enumerate(counts)])
    return np.outer(u, u) * np.asarray(corr)[np.ix_(which, which)]


def test_the_data_built_covariance_is_peelles_puzzle_in_two_dimensions():
    comps, rhos = experiment(Q, N, SIGMA_Q, SIGMA_N, CORR)
    mean, cov, qbar, var_a = exact(Q, N, SIGMA_Q, SIGMA_N, CORR)
    p_F = Problem(
        [
            Constraint(
                comps,
                terms=[normalisation_term(comps, SIGMA_N / N, CORR, from_data=True)],
            )
        ]
    )
    mean_F, cov_F, _ = linear_posterior(p_F, design=one_hot)
    q1, q1p = Q[0]
    s1, s1p = SIGMA_Q[0]
    xi = (q1 - q1p) ** 2 * SIGMA_N[0] ** 2 * var_a[0] / (N[0] ** 2 * s1**2 * s1p**2)
    # eqs. 13-17 of the reference
    np.testing.assert_allclose(mean_F[0], qbar[0] * N[0] / (1 + xi), rtol=1e-6)
    np.testing.assert_allclose(
        mean_F[1],
        Q[1][0]
        * N[1]
        * (1 - C * xi * N[0] * SIGMA_N[1] / (N[1] * SIGMA_N[0]) / (1 + xi)),
        rtol=1e-6,
    )
    # eq. 15: only the normalisation part of var(rho_1) is deflated
    np.testing.assert_allclose(
        cov_F[0, 0],
        var_a[0] * N[0] ** 2 + SIGMA_N[0] ** 2 * qbar[0] ** 2 / (1 + xi),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        cov_F[1, 1],
        cov[1, 1] - C**2 * xi * Q[1][0] ** 2 * SIGMA_N[1] ** 2 / (1 + xi),
        rtol=1e-6,
    )
    np.testing.assert_allclose(cov_F[0, 1], cov[0, 1] / (1 + xi), rtol=1e-6)
    # the puzzle: both means are biased low, both variances too small
    assert xi > 0 and np.all(mean_F < mean) and np.all(np.diag(cov_F) < np.diag(cov))


# section II.B: the five-quantity numerical study, Table I of the reference
Q5 = [
    np.array([1.0, 1.5]),
    np.array([1.8]),
    np.array([2.2, 2.4]),
    np.array([1.9, 1.5]),
    np.array([1.4, 1.2]),
]
N5 = np.array([1.0, 1.1, 1.25, 1.15, 1.05])
SIGMA_Q5 = [0.1 * q for q in Q5]
SIGMA_N5 = 0.2 * N5
CORR5 = np.full((5, 5), C) + (1 - C) * np.eye(5)


def test_five_quantities_reproduce_figure_1_in_ordering():
    comps, rhos = experiment(Q5, N5, SIGMA_Q5, SIGMA_N5, CORR5)
    mean, cov, qbar, var_a = exact(Q5, N5, SIGMA_Q5, SIGMA_N5, CORR5)
    p_F = Problem(
        [
            Constraint(
                comps,
                terms=[normalisation_term(comps, SIGMA_N5 / N5, CORR5, from_data=True)],
            )
        ]
    )
    mean_F, cov_F, _ = linear_posterior(p_F, design=one_hot)
    assert np.all(mean_F < mean) and np.all(np.diag(cov_F) < np.diag(cov))
    fixed = Term(
        _fixed_matrix(mean, SIGMA_N5 / N5, CORR5, [len(q) for q in Q5]),
        kind="matrix",
        on=comps,
    )
    mean_I, cov_I, _ = linear_posterior(
        Problem([Constraint(comps, terms=[fixed])]), design=one_hot
    )
    np.testing.assert_allclose(mean_I, mean, rtol=1e-6)
    np.testing.assert_allclose(np.diag(cov_I), np.diag(cov), rtol=1e-6)


def test_the_live_prediction_built_term_carries_the_log_determinant_pull():
    """The live term is the generative model's marginal likelihood, not C_I.

    Its mode is pulled below the exact values by the log-determinant of a
    covariance that grows with the prediction (5 % in the two-quantity case,
    9 % here with 20 % normalisation errors), though far less than the
    data-built C_F; under a flat prior its mean is pulled the other way by
    the 1 / rho tail.  The two-step refit (a constant term from a first
    estimate, recipe 27) is what reproduces the reference exactly.
    """
    comps, rhos = experiment(
        Q5, N5, SIGMA_Q5, SIGMA_N5, CORR5, prior=stats.uniform(0, 10)
    )
    mean, cov, *_ = exact(Q5, N5, SIGMA_Q5, SIGMA_N5, CORR5)
    p_I = Problem(
        [Constraint(comps, terms=[normalisation_term(comps, SIGMA_N5 / N5, CORR5)])]
    )
    mode = map_estimate(p_I, mean)
    p_F = Problem(
        [
            Constraint(
                comps,
                terms=[normalisation_term(comps, SIGMA_N5 / N5, CORR5, from_data=True)],
            )
        ]
    )
    mean_F, *_ = linear_posterior(p_F, design=one_hot)
    pull = mode / mean - 1
    assert np.all(pull < 0) and np.all(pull > -0.12)
    assert np.all(np.abs(mode - mean) < np.abs(mean_F - mean))
    # the refit: a constant term built at the first estimate is exact again
    comps_flat, _ = experiment(Q5, N5, SIGMA_Q5, SIGMA_N5, CORR5)
    fixed = Term(
        _fixed_matrix(mean, SIGMA_N5 / N5, CORR5, [len(q) for q in Q5]),
        kind="matrix",
        on=comps_flat,
    )
    mean_I, *_ = linear_posterior(
        Problem([Constraint(comps_flat, terms=[fixed])]), design=one_hot
    )
    np.testing.assert_allclose(mean_I, mean, rtol=1e-6)
