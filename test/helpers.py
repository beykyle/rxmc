"""Shared helpers for the test suite: dense references built by hand.

``STUDY_LEGEND`` and :func:`study_form` build the error-model ladder of the
motivating study (elastic alpha + Ca scattering data with no reported
uncertainties, compared in log space).  The ladder is *defined* for users in
the table of recipe 18 (``docs/recipes.md``); the legend here is the test
suite's copy of that table, kept identical so the tests and the docs cannot
drift.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.gaussian_process.kernels import RBF, Matern

from rxmc import Parameter
from rxmc.terms import (
    Term,
    constant_amplitude,
    exp_growth,
    exp_growth_amplitude,
    kernel,
    noise,
    normalization,
    offset,
    proportional_error,
    systematic,
    x_basis,
)

# ----------------------------------------------------------------------------
# Dense references
# ----------------------------------------------------------------------------


def mahalanobis(y, ym, cov):
    """``(d2, logdet)`` of a dense covariance by a numpy Cholesky factorisation."""
    L = np.linalg.cholesky(np.asarray(cov, dtype=float))
    z = np.linalg.solve(L, np.asarray(y, dtype=float) - np.asarray(ym, dtype=float))
    return float(z @ z), float(2.0 * np.sum(np.log(np.diag(L))))


def manual_mvn_loglike(y, ym, cov):
    """Reference dense multivariate-normal log likelihood."""
    d2, logdet = mahalanobis(y, ym, cov)
    return -0.5 * (d2 + logdet + len(y) * np.log(2 * np.pi))


def assemble_dense(terms, x, y, ym, values=(), rows=None):
    """Reference assembly of terms into a dense covariance over the whole stack.

    ``values`` is one tuple of sampled values per term, in ``terms`` order (an
    empty tuple for a term without parameters).  ``rows`` is one index array
    per term giving the stacked rows it applies to; ``None`` means every term
    covers the whole stack.  This is the dense reference the structured
    covariance is checked against.
    """
    x, y = np.asarray(x), np.asarray(y, dtype=float)
    ym = None if ym is None else np.asarray(ym, dtype=float)
    n = len(y)
    values = tuple(values) if values else tuple(() for _ in terms)
    rows = tuple(rows) if rows is not None else tuple(np.arange(n) for _ in terms)
    if not len(values) == len(rows) == len(terms):
        raise ValueError("one value tuple and one row array per term")
    Sigma = np.zeros((n, n))
    for term, v, r in zip(terms, values, rows):
        r = np.asarray(r, dtype=int)
        out = term.value(x[r], y[r], None if ym is None else ym[r], *v)
        if term.kind == "diag":
            Sigma[r, r] += out**2
        elif term.kind == "mode":
            Sigma[np.ix_(r, r)] += np.outer(out, out)
        else:
            Sigma[np.ix_(r, r)] += out
    return Sigma


def index_params(terms):
    """``(params, gathers)``: unique parameters by identity, first seen, and one
    gather array per term.  A stand-in for ``ParameterIndex.add_all``."""
    params, slot = [], {}
    gathers = []
    for t in terms:
        g = []
        for p in t.params:
            if id(p) not in slot:
                slot[id(p)] = len(params)
                params.append(p)
            g.append(slot[id(p)])
        gathers.append(np.asarray(g, dtype=int))
    return tuple(params), gathers


# ----------------------------------------------------------------------------
# The alpha + Ca error-model ladder
# ----------------------------------------------------------------------------

#: Label -> what the error model is; the same table as recipe 18 in
#: docs/recipes.md.  All forms are covariances of the residual ``y - ym`` in
#: log space unless stated; ``theta`` is the scattering angle in radians and
#: ``u = theta / pi`` its normalised form.
STUDY_LEGEND = {
    "L0": "constant noise: sigma = err on every point",
    "E0": "fractional noise in linear space: sigma_i = err * ym_i",
    "L1": "noise growing with angle: sigma(theta) = err * exp(slope * u)",
    "L2": "L0 plus one correlated mode proportional to angle: sys * u",
    "L2n": "L0 plus a free correlated offset mode: sys * 1",
    "L2y": "L0 plus a free correlated normalisation mode: sys * ym",
    "L12": "L1 plus the angle mode of L2",
    "Lgp": "L0 plus a Matérn(5/2) Gaussian process in u with constant amplitude",
    "Lgpn": "L0 plus the Gaussian process with an angle-growing amplitude",
    "LKp": "noise, an offset mode, and an RBF Gaussian process in momentum transfer "
    "q = 2 k sin(theta/2) with amplitude A q^(r/2)",
    "custom": "the L1 form written as a direct two-parameter Term",
}


@dataclass
class StudyForm:
    label: str
    description: str
    terms: list
    values: list  # one tuple per term
    dense: np.ndarray  # the hand-built reference covariance


def study_form(label, x, y, ym, X=np.pi, k=2.7) -> StudyForm:
    """Build the labelled form on the grid ``x`` (radians) with data ``y``, ``ym``."""
    err, slope, sys, amp, ell = 0.05, 1.3, 0.04, 0.2, 0.3
    log_err, log_slope = Parameter("log_err"), Parameter("log_err_slope")
    log_sys, log_amp = Parameter("log_sys"), Parameter("log_amp")
    n, u = len(x), x / X
    eye = np.eye(n)
    L0 = noise(log_err)
    if label == "L0":
        return StudyForm(
            label, STUDY_LEGEND[label], [L0], [(np.log(err),)], err**2 * eye
        )
    if label == "E0":
        t = proportional_error(log_err)
        return StudyForm(
            label, STUDY_LEGEND[label], [t], [(np.log(err),)], np.diag((err * ym) ** 2)
        )
    if label == "L1":
        t = noise(log_err, basis=exp_growth(X), basis_params=(log_slope,))
        sigma = err * np.exp(slope * u)
        return StudyForm(
            label, STUDY_LEGEND[label], [t], [(np.log(err), slope)], np.diag(sigma**2)
        )
    if label == "L2":
        terms = [L0, systematic(log_sys, basis=x_basis(X))]
        dense = err**2 * eye + sys**2 * np.outer(u, u)
        return StudyForm(
            label, STUDY_LEGEND[label], terms, [(np.log(err),), (np.log(sys),)], dense
        )
    if label == "L2n":
        terms = [L0, offset(parameter=log_sys)]
        dense = err**2 * eye + sys**2 * np.ones((n, n))
        return StudyForm(
            label, STUDY_LEGEND[label], terms, [(np.log(err),), (np.log(sys),)], dense
        )
    if label == "L2y":
        terms = [L0, normalization(parameter=log_sys)]
        dense = err**2 * eye + sys**2 * np.outer(ym, ym)
        return StudyForm(
            label, STUDY_LEGEND[label], terms, [(np.log(err),), (np.log(sys),)], dense
        )
    if label == "L12":
        terms = [
            noise(log_err, basis=exp_growth(X), basis_params=(log_slope,)),
            systematic(log_sys, basis=x_basis(X)),
        ]
        sigma = err * np.exp(slope * u)
        dense = np.diag(sigma**2) + sys**2 * np.outer(u, u)
        return StudyForm(
            label,
            STUDY_LEGEND[label],
            terms,
            [(np.log(err), slope), (np.log(sys),)],
            dense,
        )
    if label == "Lgp":
        gp = kernel(
            Matern(1.0, nu=2.5),
            coords=lambda x: x / X,
            amplitude=constant_amplitude,
            amplitude_params=(log_amp,),
            jitter=0.0,
            prefix="gp",
        )
        dense = err**2 * eye + amp**2 * Matern(ell, nu=2.5)(u[:, None])
        return StudyForm(
            label,
            STUDY_LEGEND[label],
            [L0, gp],
            [(np.log(err),), (np.log(ell), np.log(amp))],
            dense,
        )
    if label == "Lgpn":
        gp = kernel(
            Matern(1.0, nu=2.5),
            coords=lambda x: x / X,
            amplitude=exp_growth_amplitude(1.0),
            amplitude_params=(log_amp, log_slope),
            jitter=0.0,
        )
        a = amp * np.exp(slope * u)
        dense = err**2 * eye + np.outer(a, a) * Matern(ell, nu=2.5)(u[:, None])
        return StudyForm(
            label,
            STUDY_LEGEND[label],
            [L0, gp],
            [(np.log(err),), (np.log(ell), np.log(amp), slope)],
            dense,
        )
    if label == "LKp":
        log_b, log_s, r_pow = Parameter("log_b"), Parameter("log_s"), Parameter("r")
        b, s, lq, r = 0.05, 0.05, 1.2, 0.8
        q = 2.0 * k * np.sin(x / 2)
        gp = kernel(
            RBF(1.0),
            coords=lambda x: 2.0 * k * np.sin(x / 2),
            amplitude=lambda c, lA, r: np.exp(lA) * c.x ** (r / 2),
            amplitude_params=(log_amp, r_pow),
            jitter=0.0,
            prefix="gpq",
        )
        a = amp * q ** (r / 2)
        dense = (
            b**2 * eye + s**2 * np.ones((n, n)) + np.outer(a, a) * RBF(lq)(q[:, None])
        )
        return StudyForm(
            label,
            STUDY_LEGEND[label],
            [noise(log_b), offset(parameter=log_s), gp],
            [(np.log(b),), (np.log(s),), (np.log(lq), np.log(amp), r)],
            dense,
        )
    if label == "custom":
        e, sl = Parameter("e"), Parameter("l")
        t = Term(
            lambda c, e, l: np.exp(e) * np.exp(l * c.x / np.pi), (e, sl), kind="diag"
        )
        sigma = err * np.exp(slope * u)
        return StudyForm(
            label, STUDY_LEGEND[label], [t], [(np.log(err), slope)], np.diag(sigma**2)
        )
    raise KeyError(label)
