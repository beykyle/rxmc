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
* :func:`heldout_log_predictive`, :func:`log_posterior_predictive` —
  out-of-sample scoring on a held-out constraint (e.g.
  ``constraint.complement()``).
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
from scipy.special import logsumexp

__all__ = [
    "log_jacobian",
    "predictive_draws",
    "coverage_curve",
    "coverage_error",
    "sharpness",
    "heldout_log_predictive",
    "log_posterior_predictive",
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


_DEFAULT_LEVELS = np.linspace(0.02, 0.98, 49)


def _psd_factor(Sigma, jitter=1e-10):
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


def _rows(samples, n=None):
    """Posterior samples as a 2-D ``(n_samples, n_params)`` array.

    A 1-D input is one sample row (as in :func:`split_samples`).  When ``n`` is
    given the number of rows must match it.
    """
    samples = np.asarray(samples, dtype=float)
    if samples.ndim == 1:
        samples = samples[None, :]
    if n is not None and samples.shape[0] != n:
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
        Posterior samples of the physical-model parameters (a 1-D array is
        one sample).
    cov_samples : array_like, shape (n, constraint.n_params), optional
        Matching samples of the constraint's parameters (required when the
        constraint has any).
    n_rep : int, optional
        Draws per posterior row (ignored when ``model_only``).
    rng : numpy.random.Generator, optional
        Source of the standard-normal draws; a fresh default generator when
        omitted.
    model_only : bool, optional
        Return the predictions ``ym_i`` themselves instead of draws around
        them (no covariance is assembled).

    Returns
    -------
    np.ndarray
        Draws in the observations' comparison space: shape
        ``(n * n_rep, n_data_pts)``, or ``(n, n_data_pts)`` when ``model_only``.
    """
    rng = np.random.default_rng() if rng is None else rng
    model_samples = _rows(model_samples)
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
            ym = np.concatenate(constraint.predict(*model_samples[i]))
            out[i] = ym[constraint.active]
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
        Nominal central-interval probabilities in (0, 1).  Defaults to 49
        levels from 0.02 to 0.98 (``_DEFAULT_LEVELS``).

    Returns
    -------
    np.ndarray
        Fraction of points inside the central ``level`` interval, per level.
    """
    draws = np.asarray(draws, dtype=float)
    y = np.asarray(y, dtype=float)
    levels = _DEFAULT_LEVELS if levels is None else np.asarray(levels)
    out = np.empty(len(levels))
    for i, lv in enumerate(levels):
        lo, hi = np.percentile(draws, [50 * (1 - lv), 50 * (1 + lv)], axis=0)
        out[i] = np.mean((y >= lo) & (y <= hi))
    return out


def coverage_error(draws, y, levels=None) -> float:
    """``max |coverage(level) - level|`` — a single calibration score."""
    levels = _DEFAULT_LEVELS if levels is None else np.asarray(levels)
    return float(np.max(np.abs(coverage_curve(draws, y, levels) - levels)))


def sharpness(draws, percentiles=(16, 84), transform=None) -> np.ndarray:
    """Per-point width of a central predictive interval.

    Parameters
    ----------
    draws : array_like, shape (n_draws, n_pts)
    percentiles : (float, float), optional
        Lower and upper percentiles (in 0-100) bounding the interval; the
        default is the central 68 %.  Note :func:`coverage_curve` takes
        interval *probabilities* in (0, 1) instead.
    transform : callable, optional
        Applied to the draws first (e.g. ``np.exp`` to report widths in raw
        space for a log comparison space, or ``np.log10``).
    """
    draws = np.asarray(draws, dtype=float)
    if transform is not None:
        draws = transform(draws)
    lo, hi = np.percentile(draws, percentiles, axis=0)
    return hi - lo


def heldout_log_predictive(heldout_constraint, model_samples, cov_samples=None):
    """``log p(y_held | theta_i)`` for each posterior row.

    ``heldout_constraint`` is typically ``fit_constraint.complement()``: the same
    observations, terms and parameters, with the held-out points active.  The
    score is that constraint's log likelihood at each sample (a 1-D
    ``model_samples`` is one sample).

    Returns
    -------
    np.ndarray, shape (n,)
    """
    model_samples = _rows(model_samples)
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
        As returned by :func:`logz_summary`; a trailing replicate count is
        accepted and ignored (it is informational only).
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

    Returns
    -------
    (np.ndarray, list of np.ndarray)
        ``model_samples`` of shape ``(n, n_model_params)`` and one
        ``(n, constraint.n_params)`` array per parametric constraint.
    """
    samples = np.asarray(samples, dtype=float)
    if samples.ndim == 1:
        samples = samples[None, :]
    parts = np.split(samples, config.indices[:-1], axis=1)
    return parts[0], parts[1:]
