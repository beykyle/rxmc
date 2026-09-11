"""Sampler-agnostic posterior checks on ``(problem, samples)``.

Everything here consumes a compiled :class:`~rxmc.problem.Problem` and a
matrix of posterior ``samples`` of shape ``(n, problem.ndim)`` in
``problem.names`` order (what emcee's ``get_chain(flat=True)``, dynesty's
``samples_equal()`` and black-box-bayes give) and never touches a sampler:

* :func:`predictive_draws` — draws from the posterior predictive
  ``N(ym(theta), Sigma(theta))`` on a constraint's active points, or the
  model-only predictive ``ym(theta)``.
* :func:`coverage_curve`, :func:`coverage_error`, :func:`sharpness` —
  empirical calibration and width of those draws against the data.
* :func:`heldout_log_predictive`, :func:`log_posterior_predictive` —
  out-of-sample scoring on a held-out problem (``Problem([fit.complement()])``).
* :func:`logz_summary`, :func:`compare_logz` — nested-sampling evidence
  bookkeeping with replicate-based errors and a conservative tie verdict.

Held-out scoring and a term that spans the split
------------------------------------------------
A held-out problem built from ``fit.complement()`` has the *marginal*
covariance of its active rows.  When no term couples the fitted and the
held-out rows that is the right density, and ``ll(fit) + ll(held) == ll(full)``.
When a term does span the split (a Gaussian process ``on=comps`` over
several experiments), the honest held-out density is the conditional
``p(y_held | y_fit, theta)`` under the full covariance; pass the fitted
problem as ``given=`` to :func:`heldout_log_predictive` and
:func:`predictive_draws` and they compute exactly that.  Without a spanning
term the conditional equals the marginal.

Draws and densities are in the comparison space of the constraint (a
``space=log`` comparison gives log-space draws; map them back with
``comparison.space.inverse``).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import scipy.linalg as sla
from scipy.special import logsumexp

from .likelihood import Gaussian
from .problem import Problem

__all__ = [
    "predictive_draws",
    "coverage_curve",
    "coverage_error",
    "sharpness",
    "heldout_log_predictive",
    "log_posterior_predictive",
    "logz_summary",
    "compare_logz",
]

_DEFAULT_LEVELS = np.linspace(0.02, 0.98, 49)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _rows(samples, ndim) -> np.ndarray:
    """Posterior samples as ``(n, ndim)``; a 1-D input is one row."""
    samples = np.asarray(samples, dtype=float)
    if samples.ndim == 1:
        samples = samples[None, :]
    if samples.ndim != 2 or samples.shape[1] != ndim:
        raise ValueError(
            f"samples must have shape (n, {ndim}) in problem.names order, got "
            f"{samples.shape}"
        )
    return samples


def _psd_factor(Sigma, jitter=1e-10) -> np.ndarray:
    """A factor ``L`` with ``L L^T = Sigma``.

    The lower Cholesky factor when ``Sigma`` is positive definite; otherwise
    the Cholesky factor of ``Sigma`` plus a jitter *relative* to its mean
    variance, and failing that a symmetric square root from the eigen
    decomposition (negative eigenvalues clipped to zero).  No jitter is added
    on the successful path, so draws are never inflated.
    """
    Sigma = np.asarray(Sigma, dtype=float)
    try:
        return np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        pass
    scale = max(float(np.mean(np.diag(Sigma))), np.finfo(float).tiny)
    try:
        return np.linalg.cholesky(Sigma + jitter * scale * np.eye(len(Sigma)))
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(Sigma)
        return V * np.sqrt(np.clip(w, 0.0, None))


class _Conditional:
    """``p(y_H | y_F, theta)`` for one held-out constraint given its fit."""

    def __init__(self, held: Problem, given: Problem, constraint: int):
        try:
            h, f = held.constraints[constraint], given.constraints[constraint]
        except IndexError:
            raise ValueError(
                f"held-out and fitted problems must both have constraint "
                f"{constraint}"
            ) from None
        if h.comparisons != f.comparisons:
            raise ValueError(
                "given= must be the fitted problem over the same comparison "
                "objects (Problem([fit]) with held = Problem([fit.complement()]))"
            )
        if held.names != given.names:
            raise ValueError(
                "held-out and fitted problems index different parameters: "
                f"{held.names} vs {given.names}"
            )
        if np.intersect1d(h.active, f.active).size:
            raise ValueError(
                "the held-out and fitted active points overlap; the held-out "
                "constraint should be the complement of the fitted one"
            )
        if not isinstance(h.likelihood, Gaussian):
            raise ValueError(
                "the conditional held-out density is Gaussian; the constraint "
                f"uses {type(h.likelihood).__name__}.  Score the marginal instead "
                "(omit given=) or use a Gaussian likelihood"
            )
        self.held, self.fit = h, f
        # the union of the two active sets, compiled once: every row active
        self.full = Problem([replace(h.source, masks=None)]).constraints[0]
        lookup = {int(r): i for i, r in enumerate(self.full.active)}
        self.iH = np.array([lookup[int(r)] for r in h.active], dtype=int)
        self.iF = np.array([lookup[int(r)] for r in f.active], dtype=int)
        self.yF = self.full.y[f.active]

    def __call__(self, theta):
        """``(mean, cov)`` of the held-out rows given the fitted data."""
        ym = self.full.ym(theta)
        S = self.full.covariance.matrix(ym, theta)
        SHH = S[np.ix_(self.iH, self.iH)]
        SHF = S[np.ix_(self.iH, self.iF)]
        SFF = S[np.ix_(self.iF, self.iF)]
        L = _psd_factor(SFF)
        r = self.yF - ym[self.fit.active]
        mean = ym[self.held.active] + SHF @ sla.cho_solve((L, True), r)
        cov = SHH - SHF @ sla.cho_solve((L, True), SHF.T)
        return mean, 0.5 * (cov + cov.T)


def _gaussian_logpdf(r, cov) -> float:
    L = _psd_factor(cov)
    z = sla.solve_triangular(L, r, lower=True)
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    return float(-0.5 * (z @ z + logdet + len(r) * np.log(2 * np.pi)))


# ----------------------------------------------------------------------------
# Posterior predictive
# ----------------------------------------------------------------------------


def predictive_draws(
    problem: Problem,
    samples,
    constraint: int = 0,
    *,
    n_rep: int = 1,
    rng=None,
    model_only: bool = False,
    given: Problem | None = None,
) -> np.ndarray:
    """Posterior-predictive draws on a constraint's active points.

    For each posterior row ``theta_i`` the constraint gives ``ym_i`` and
    ``Sigma_i``; ``n_rep`` draws ``ym_i + L_i z`` (``z ~ N(0, I)``) are taken.
    With ``model_only=True`` the rows are ``ym_i`` themselves and no
    covariance is assembled.

    Parameters
    ----------
    problem : Problem
    samples : array_like, shape (n, problem.ndim)
        Posterior rows in ``problem.names`` order (a 1-D array is one row).
    constraint : int, optional
        Index into ``problem.constraints``.
    n_rep : int, optional
        Draws per posterior row (ignored when ``model_only``).
    rng : numpy.random.Generator or seed, optional
    model_only : bool, optional
        Return the predictions ``ym_i`` on the active points instead of draws
        around them.
    given : Problem, optional
        The *fitted* problem when ``problem`` is its held-out complement and a
        term spans the two (module docstring): draws then come from the
        conditional ``N(mu_c, Sigma_c)`` of the held-out rows given the fitted
        data.

    Returns
    -------
    np.ndarray
        In comparison space: shape ``(n * n_rep, n_active)``, or
        ``(n, n_active)`` when ``model_only``.
    """
    rng = np.random.default_rng(rng)
    samples = _rows(samples, problem.ndim)
    n = samples.shape[0]
    c = problem.constraints[constraint]
    N = c.n_active
    cond = None if given is None else _Conditional(problem, given, constraint)

    if model_only:
        out = np.empty((n, N))
        for i in range(n):
            out[i] = cond(samples[i])[0] if cond else c.ym(samples[i])[c.active]
        return out

    out = np.empty((n * n_rep, N))
    for i in range(n):
        theta = samples[i]
        if cond:
            mu, Sigma = cond(theta)
        else:
            mu, Sigma = c.ym(theta)[c.active], c.matrix(theta)
        L = _psd_factor(Sigma)
        z = rng.standard_normal((n_rep, N))
        out[i * n_rep : (i + 1) * n_rep] = mu + z @ L.T
    return out


def coverage_curve(draws, y, levels=None) -> np.ndarray:
    """Empirical coverage of central predictive intervals at each nominal level.

    Parameters
    ----------
    draws : array_like, shape (n_draws, n_pts)
    y : array_like, shape (n_pts,)
        The data the draws are checked against (same space as ``draws``).
    levels : array_like, optional
        Nominal central-interval probabilities in (0, 1).  Defaults to 49
        levels from 0.02 to 0.98.

    Returns
    -------
    np.ndarray
        Fraction of points inside the central ``level`` interval, per level.
    """
    draws = np.asarray(draws, dtype=float)
    y = np.asarray(y, dtype=float)
    levels = _DEFAULT_LEVELS if levels is None else np.asarray(levels, dtype=float)
    out = np.empty(len(levels))
    for i, lv in enumerate(levels):
        lo, hi = np.percentile(draws, [50 * (1 - lv), 50 * (1 + lv)], axis=0)
        out[i] = np.mean((y >= lo) & (y <= hi))
    return out


def coverage_error(draws, y, levels=None) -> float:
    """``max |coverage(level) - level|``: a single calibration score."""
    levels = _DEFAULT_LEVELS if levels is None else np.asarray(levels, dtype=float)
    return float(np.max(np.abs(coverage_curve(draws, y, levels) - levels)))


def sharpness(draws, percentiles=(16, 84), transform=None) -> np.ndarray:
    """Per-point width of a central predictive interval.

    Parameters
    ----------
    draws : array_like, shape (n_draws, n_pts)
    percentiles : (float, float), optional
        Lower and upper percentiles (in 0-100) bounding the interval; the
        default is the central 68 %.  :func:`coverage_curve` takes interval
        *probabilities* in (0, 1) instead.
    transform : callable, optional
        Applied to the draws first (e.g. ``np.exp`` to report widths in
        physical units for a log comparison space).
    """
    draws = np.asarray(draws, dtype=float)
    if transform is not None:
        draws = transform(draws)
    lo, hi = np.percentile(draws, percentiles, axis=0)
    return hi - lo


# ----------------------------------------------------------------------------
# Held-out scoring
# ----------------------------------------------------------------------------


def heldout_log_predictive(
    heldout_problem: Problem, samples, *, given: Problem | None = None
) -> np.ndarray:
    """``log p(y_held | theta_i)`` for each posterior row.

    ``heldout_problem`` is typically ``Problem([fit.complement()])``: the same
    comparisons, terms and parameters, with the held-out points active.  The
    score is that problem's log likelihood at each row, so the constraints'
    ``weight`` and likelihood family apply.  With ``given=`` (the fitted
    problem) the score is instead the Gaussian conditional density of the
    held-out rows given the fitted data, constraint by constraint (module
    docstring).

    Returns
    -------
    np.ndarray, shape (n,)
    """
    samples = _rows(samples, heldout_problem.ndim)
    if given is None:
        return np.array([heldout_problem.log_likelihood(t) for t in samples])
    conds = [
        _Conditional(heldout_problem, given, i)
        for i in range(len(heldout_problem.constraints))
    ]
    out = np.empty(samples.shape[0])
    for k, theta in enumerate(samples):
        total = 0.0
        for c, cond in zip(heldout_problem.constraints, conds):
            if c.weight == 0.0:
                continue
            mean, cov = cond(theta)
            total += c.weight * _gaussian_logpdf(c.y[c.active] - mean, cov)
        out[k] = total
    return out


def log_posterior_predictive(logp_samples, logw=None) -> float:
    """``log E_post[p(y_held | theta)]``: the joint log posterior predictive
    density of the held-out block.

    A log-mean-exp over posterior samples of :func:`heldout_log_predictive`
    values; pass ``logw`` (unnormalised log importance weights, e.g.
    nested-sampling ``logwt``) for weighted samples.  This is the joint
    predictive of the whole held-out block, not the pointwise-summed ``elpd``
    of Vehtari et al.; divide by the number of held-out points for a
    per-point score.
    """
    logp = np.asarray(logp_samples, dtype=float)
    if logw is None:
        return float(logsumexp(logp) - np.log(len(logp)))
    logw = np.asarray(logw, dtype=float)
    return float(logsumexp(logp + logw) - logsumexp(logw))


# ----------------------------------------------------------------------------
# Evidence bookkeeping
# ----------------------------------------------------------------------------


def logz_summary(logz, logzerr) -> tuple[float, float, int]:
    """Replicate-aware evidence summary.

    Parameters
    ----------
    logz, logzerr : array_like
        ``log Z`` and its sampler-reported error for each replicate run (one
        value each is fine).  Add ``problem.log_jacobian()`` to ``logz`` first
        when comparing fits in different comparison spaces.

    Returns
    -------
    (float, float, int)
        ``(mean, err, n)`` with ``err = max(half-range across replicates, mean
        reported error)``: the sampler's own error is a lower bound.
    """
    logz = np.atleast_1d(np.asarray(logz, dtype=float))
    logzerr = np.atleast_1d(np.asarray(logzerr, dtype=float))
    half_range = 0.5 * (logz.max() - logz.min()) if logz.size > 1 else 0.0
    return float(logz.mean()), float(max(half_range, logzerr.mean())), int(logz.size)


def compare_logz(a, b, sigma: float = 2.0) -> dict:
    """``Delta log Z = a - b`` with a conservative tie verdict.

    Parameters
    ----------
    a, b : (mean, err) or (mean, err, n)
        As returned by :func:`logz_summary`; a trailing replicate count is
        accepted and ignored.
    sigma : float, optional
        A difference smaller than ``sigma * hypot(err_a, err_b)`` is a ``"tie"``.

    Returns
    -------
    dict
        ``{"dlogZ": ..., "err": ..., "verdict": "a" | "b" | "tie"}``.
    """
    ma, ea = a[0], a[1]
    mb, eb = b[0], b[1]
    d = float(ma - mb)
    err = float(np.hypot(ea, eb))
    if abs(d) < sigma * err:
        verdict = "tie"
    else:
        verdict = "a" if d > 0 else "b"
    return {"dlogZ": d, "err": err, "verdict": verdict}
