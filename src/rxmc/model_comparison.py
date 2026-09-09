"""
Sampler-agnostic model-comparison and predictive-checking utilities.

Everything here consumes a :class:`~rxmc.constraint.Constraint` plus posterior
*samples* (rows of model parameters and, optionally, of the constraint's
covariance/likelihood parameters) and never touches a sampler:

* :func:`predictive_draws` — draws from the posterior predictive
  ``N(ym(theta), Sigma(theta))`` on the constraint's active points (or the
  model-only predictive ``ym(theta)``).
* :func:`coverage_curve`, :func:`coverage_error`, :func:`sharpness` — empirical
  calibration and width of those draws against the data.
* :func:`heldout_log_predictive`, :func:`elpd` — out-of-sample scoring on a
  held-out constraint (e.g. ``constraint.complement()``).
* :func:`logz_summary`, :func:`compare_logz` — nested-sampling evidence
  bookkeeping with replicate-based errors and a conservative tie verdict.
* :func:`log_jacobian` — the comparison-space Jacobian needed to compare
  evidences across residual spaces (e.g. log-y versus linear-y fits).

Notes
-----
Drawing from ``N(ym, Sigma)`` in a *transformed* comparison space (an
observation with ``transform=log``) yields draws in that space; map them back
with the transform's inverse (``np.exp``) before comparing to raw data.
"""

from __future__ import annotations

import numpy as np
import scipy as sc
from scipy.special import logsumexp

__all__ = [
    "log_jacobian",
    "predictive_draws",
    "coverage_curve",
    "coverage_error",
    "sharpness",
    "heldout_log_predictive",
    "elpd",
    "logz_summary",
    "compare_logz",
    "split_samples",
]


def log_jacobian(constraint) -> float:
    """Comparison-space log-Jacobian of a constraint (sum over active points).

    ``log Z_raw = log Z_transformed + log_jacobian``: add it to the evidence of a
    fit performed in a transformed comparison space (e.g. ``transform=log``)
    before comparing with a fit in raw space.  Zero for identity transforms.
    """
    return float(constraint.log_jacobian)


def _psd_factor(Sigma, jitter=1e-10):
    """Lower factor ``L`` with ``L L^T = Sigma`` (Cholesky, eigen fallback)."""
    try:
        return sc.linalg.cholesky(Sigma + jitter * np.eye(len(Sigma)), lower=True)
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(Sigma)
        return V * np.sqrt(np.clip(w, 0.0, None))


def _rows(samples, n):
    samples = np.asarray(samples, dtype=float)
    if samples.ndim == 1:
        samples = samples[:, None]
    if samples.shape[0] != n:
        raise ValueError(f"expected {n} sample rows, got {samples.shape[0]}")
    return samples


def predictive_draws(
    constraint,
    model_samples,
    cov_samples=None,
    *,
    n_rep: int = 1,
    rng=None,
    model_only: bool = False,
) -> np.ndarray:
    """Posterior-predictive draws on the constraint's active points.

    For each posterior row ``theta_i`` the constraint gives ``ym_i`` and
    ``Sigma_i``; ``n_rep`` draws ``ym_i + L_i z`` (``z ~ N(0, I)``) are taken.
    With ``model_only=True`` the rows are ``ym_i`` (the model-only predictive,
    no error-model noise).

    Parameters
    ----------
    constraint : Constraint
        The constraint whose predictive is wanted (its active points).
    model_samples : array_like, shape (n, n_model_params)
        Posterior samples of the physical-model parameters.
    cov_samples : array_like, shape (n, constraint.n_params), optional
        Matching samples of the constraint's parameters (required when the
        constraint has any).
    n_rep : int, optional
        Draws per posterior row (ignored when ``model_only``).
    rng : numpy.random.Generator, optional
    model_only : bool, optional

    Returns
    -------
    np.ndarray, shape (n * n_rep, n_data_pts)
        Draws in the observations' comparison space.
    """
    rng = np.random.default_rng() if rng is None else rng
    model_samples = _rows(model_samples, len(np.asarray(model_samples)))
    n = model_samples.shape[0]
    if constraint.n_params:
        if cov_samples is None:
            raise ValueError("constraint has parameters; pass cov_samples")
        cov_samples = _rows(cov_samples, n)
    else:
        cov_samples = np.zeros((n, 0))

    N = constraint.n_data_pts
    if model_only:
        out = np.empty((n, N))
        for i in range(n):
            ym, _ = constraint.predict_and_covariance(
                tuple(model_samples[i]), tuple(cov_samples[i])
            )
            out[i] = ym
        return out

    out = np.empty((n * n_rep, N))
    for i in range(n):
        ym, Sigma = constraint.predict_and_covariance(
            tuple(model_samples[i]), tuple(cov_samples[i])
        )
        L = _psd_factor(Sigma)
        z = rng.standard_normal((n_rep, N))
        out[i * n_rep : (i + 1) * n_rep] = ym + z @ L.T
    return out


def coverage_curve(draws, y, levels=None) -> np.ndarray:
    """Empirical coverage of central predictive intervals at each nominal level.

    Parameters
    ----------
    draws : array_like, shape (n_draws, n_pts)
    y : array_like, shape (n_pts,)
        The data the draws are checked against (same space as ``draws``).
    levels : array_like, optional
        Nominal central-interval probabilities in (0, 1).  Defaults to
        ``np.linspace(0.02, 0.98, 49)``.

    Returns
    -------
    np.ndarray
        Fraction of points inside the central ``level`` interval, per level.
    """
    draws = np.asarray(draws, dtype=float)
    y = np.asarray(y, dtype=float)
    levels = np.linspace(0.02, 0.98, 49) if levels is None else np.asarray(levels)
    out = np.empty(len(levels))
    for i, lv in enumerate(levels):
        lo, hi = np.percentile(draws, [50 * (1 - lv), 50 * (1 + lv)], axis=0)
        out[i] = np.mean((y >= lo) & (y <= hi))
    return out


def coverage_error(draws, y, levels=None) -> float:
    """``max |coverage(level) - level|`` — a single calibration score."""
    levels = np.linspace(0.02, 0.98, 49) if levels is None else np.asarray(levels)
    return float(np.max(np.abs(coverage_curve(draws, y, levels) - levels)))


def sharpness(draws, levels=(16, 84), transform=None) -> np.ndarray:
    """Per-point width of the central predictive interval.

    Parameters
    ----------
    draws : array_like, shape (n_draws, n_pts)
    levels : (float, float), optional
        Percentiles of the interval; default the central 68 %.
    transform : callable, optional
        Applied to the draws first (e.g. ``np.exp`` to report widths in raw
        space for a log comparison space, or ``np.log10``).
    """
    draws = np.asarray(draws, dtype=float)
    if transform is not None:
        draws = transform(draws)
    lo, hi = np.percentile(draws, levels, axis=0)
    return hi - lo


def heldout_log_predictive(heldout_constraint, model_samples, cov_samples=None):
    """``log p(y_held | theta_i)`` for each posterior row.

    ``heldout_constraint`` is typically ``fit_constraint.complement()``: the same
    observations, terms and parameters, with the held-out points active.  The
    score is the constraint's own (marginal-block) log likelihood at each
    sample.

    Returns
    -------
    np.ndarray, shape (n,)
    """
    model_samples = _rows(model_samples, len(np.asarray(model_samples)))
    n = model_samples.shape[0]
    if heldout_constraint.n_params:
        if cov_samples is None:
            raise ValueError("constraint has parameters; pass cov_samples")
        cov_samples = _rows(cov_samples, n)
    else:
        cov_samples = np.zeros((n, 0))
    return np.array(
        [
            heldout_constraint.log_likelihood(
                tuple(model_samples[i]), tuple(cov_samples[i])
            )
            for i in range(n)
        ]
    )


def elpd(logp_samples, logw=None) -> float:
    """Expected log predictive density ``log E_post[p(y_held | theta)]``.

    A log-mean-exp over posterior samples; pass ``logw`` (unnormalised log
    importance weights, e.g. nested-sampling ``logwt``) for weighted samples.
    """
    logp = np.asarray(logp_samples, dtype=float)
    if logw is None:
        return float(logsumexp(logp) - np.log(len(logp)))
    logw = np.asarray(logw, dtype=float)
    return float(logsumexp(logp + logw) - logsumexp(logw))


def logz_summary(logz, logzerr):
    """Replicate-aware evidence summary.

    Parameters
    ----------
    logz, logzerr : array_like
        ``log Z`` and its sampler-reported error for each replicate run (one
        value each is fine).

    Returns
    -------
    (float, float, int)
        ``(mean, err, n)`` with ``err = max(half-range across replicates, mean
        reported error)`` — the sampler's own error is a lower bound.
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
        As returned by :func:`logz_summary`.
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


def split_samples(config, samples):
    """Split flat sampler rows into ``(model_samples, [cov_samples, ...])``.

    Row-wise :meth:`~rxmc.config.CalibrationConfig.split_parameters`: one
    covariance-sample block per parametric constraint, in
    ``config.evidence.parametric_constraints`` order.
    """
    samples = np.asarray(samples, dtype=float)
    if samples.ndim == 1:
        samples = samples[None, :]
    parts = np.split(samples, config.indices[:-1], axis=1)
    return parts[0], parts[1:]
