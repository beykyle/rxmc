"""Predicting a Gaussian-process discrepancy away from the data.

A :func:`~rxmc.terms.kernel` term only inflates the covariance *at the data
points* with ``K(X, X)``; it does not say what the discrepancy is at new
``x``.  :func:`gp_posterior_predictive` is the standard conditioning of a
zero-mean GP on observed residuals, and :func:`total_predictive_band` turns a
posterior chain into a predictive band on any grid that propagates the model
parameters, the conditioned discrepancy and, optionally, observation noise.
The band needs nothing but the problem, the kernel term and a predictor: the
training rows, the comparison space, the kernel hyperparameter and amplitude
columns and the conditioning noise all follow from the problem.

Kernels are duck-typed as in :func:`~rxmc.terms.kernel`: sklearn-style objects
with ``clone_with_theta``, ``__call__`` and ``diag``, ``theta`` in sklearn's
log space.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla

from .problem import Problem, _per_point
from .terms import KernelTerm, TermContext, as_2d

__all__ = ["gp_posterior_predictive", "predictive_band", "total_predictive_band"]


# ----------------------------------------------------------------------------
# GP conditioning
# ----------------------------------------------------------------------------


def _train_noise_matrix(train_noise_var, n) -> np.ndarray:
    if train_noise_var is None:
        return np.zeros((n, n))
    v = np.asarray(train_noise_var, dtype=float)
    if v.ndim == 0:
        return float(v) * np.eye(n)
    if v.ndim == 1:
        return np.diag(v)
    return v


def _condition(Ktt, Kst, r, N, jitter):
    """``(mean, v)`` with ``mean = Kst (Ktt + N)^-1 r`` and ``v = L^-1 Kst^T``.

    Callers form the posterior covariance ``Kss - v^T v`` or its diagonal
    ``diag(Kss) - sum(v**2, 0)``.
    """
    n = len(r)
    L = sla.cholesky(Ktt + N + jitter * np.eye(n), lower=True)
    alpha = sla.cho_solve((L, True), r)
    v = sla.solve_triangular(L, Kst.T, lower=True)
    return Kst @ alpha, v


def gp_posterior_predictive(
    kernel, theta, X_train, residuals, X_pred, *, train_noise_var=None, jitter=1e-10
):
    r"""Posterior mean and covariance of a GP discrepancy at ``X_pred``.

    Conditions a zero-mean GP with covariance ``kernel`` (rebuilt at ``theta``)
    on the observed ``residuals`` at ``X_train``:

    .. math::

        \bar f_* = K_{*t} (K_{tt} + N)^{-1} r, \qquad
        \mathrm{cov}_* = K_{**} - K_{*t} (K_{tt} + N)^{-1} K_{t*}

    Parameters
    ----------
    kernel : sklearn-style kernel
    theta : array_like
        Kernel hyperparameters in sklearn log-theta space.
    X_train, X_pred : array_like
        Training and prediction inputs (1-D promoted to a column).
    residuals : array_like, shape (n_train,)
        Observed minus model mean at ``X_train``.
    train_noise_var : float, array_like or matrix, optional
        Training-noise covariance ``N``: scalar, per-point vector, or full.
    jitter : float, optional
        Nugget added to the training covariance.

    Returns
    -------
    mean : np.ndarray, shape (n_pred,)
    cov : np.ndarray, shape (n_pred, n_pred)
    """
    k = kernel.clone_with_theta(np.asarray(theta, dtype=float))
    Xt, Xp = as_2d(X_train), as_2d(X_pred)
    r = np.asarray(residuals, dtype=float)
    Ktt, Kst, Kss = k(Xt), k(Xp, Xt), k(Xp)
    N = _train_noise_matrix(train_noise_var, len(r))
    mean, v = _condition(Ktt, Kst, r, N, jitter)
    return mean, Kss - v.T @ v


def predictive_band(draws, levels=(16, 50, 84)) -> np.ndarray:
    """Percentile band over ``draws`` of shape ``(n_draws, n_points)``.

    Returns ``(len(levels), n_points)``.
    """
    return np.percentile(np.asarray(draws, dtype=float), levels, axis=0)


# ----------------------------------------------------------------------------
# The total predictive band of a problem
# ----------------------------------------------------------------------------


def _locate(problem: Problem, term: KernelTerm):
    """The compiled constraint holding ``term`` and its covariance entry."""
    if not isinstance(term, KernelTerm):
        raise TypeError(
            "term must be the KernelTerm returned by rxmc.terms.kernel, got "
            f"{term!r}"
        )
    for c in problem.constraints:
        for e in c.covariance.entries:
            if e.term is term:
                return c, e
    raise ValueError("the kernel term is not part of any constraint of the problem")


def _one_space(constraint, rows):
    spaces = []
    for comp, o in zip(constraint.comparisons, constraint.offsets):
        if np.any((rows >= o.start) & (rows < o.stop)):
            if not any(comp.space is s for s in spaces):
                spaces.append(comp.space)
    if len(spaces) != 1:
        raise ValueError(
            "the comparisons a kernel term spans must share one comparison space "
            "to predict in it; found "
            f"{[s.name for s in spaces]}"
        )
    return spaces[0]


def total_predictive_band(
    problem: Problem,
    term: KernelTerm,
    predictor,
    x_pred,
    samples,
    *,
    noise_std: float = 0.0,
    train_noise_var=None,
    levels=(16, 84),
    n_draws: int = 400,
    rng=None,
    physical: bool = False,
) -> np.ndarray:
    r"""Predictive band on ``x_pred`` propagating model, discrepancy and noise.

    For each chain row ``theta`` the model's prediction on the training rows
    gives the residual ``y - ym`` in comparison space; the discrepancy GP of
    ``term`` (kernel hyperparameters, amplitude and coordinate transform at
    that row) is conditioned on it and predicted at ``x_pred``; one draw

    .. math::

        y_* = \mathrm{space}(\mathrm{predictor}(\theta)) + \bar f_*
              + \mathcal N\!\big(0,\ \mathrm{var}_* + \sigma^2\big)

    is taken, and the requested percentiles over the draws are returned.
    Only the GP's posterior *variance* enters per point.

    Parameters
    ----------
    problem : Problem
        The compiled problem the chain was sampled from.
    term : KernelTerm
        The discrepancy term, as declared in one of the problem's constraints.
    predictor : Predictor
        The model bound to ``x_pred`` (``model.bind(x_pred, meta)``); its
        columns are read from the problem, and the ``meta`` it was bound with
        is what a callable amplitude's ``c.meta(key)`` reads at ``x_pred``.
    x_pred : array_like
        The prediction grid (raw coordinates; the term's ``coords`` transform
        is applied for the kernel).
    samples : array_like, shape (n, problem.ndim)
        Chain rows in ``problem.names`` order.
    noise_std : float, optional
        Observation noise added at the prediction points, in comparison space.
    train_noise_var : float, array_like or matrix, optional
        Conditioning noise on the training rows.  Default: everything in the
        constraint's covariance at that row *except* the kernel term itself
        (statistical errors, noise and mode terms), which is the exact GP
        regression noise for the declared error model.
    levels : sequence of float, optional
        Percentiles of the band.
    n_draws : int, optional
        Rows subsampled from the chain (all rows when the chain is shorter).
    rng : numpy.random.Generator or seed, optional
    physical : bool, optional
        Map every draw back through the comparison space's inverse before
        taking percentiles (``space=log`` gives a band in physical units).

    Returns
    -------
    np.ndarray, shape (len(levels), len(x_pred))
    """
    rng = np.random.default_rng(rng)
    samples = np.asarray(samples, dtype=float)
    if samples.ndim == 1:
        samples = samples[None, :]
    if samples.shape[1] != problem.ndim:
        raise ValueError(
            f"samples must have shape (n, {problem.ndim}) in problem.names order, "
            f"got {samples.shape}"
        )
    if samples.shape[0] > n_draws:
        samples = samples[rng.choice(samples.shape[0], n_draws, replace=False)]

    c, entry = _locate(problem, term)
    rows = entry.rows
    keep = entry.keep  # the active rows of the term's support
    pos = entry.pos[keep]  # their positions in the active stack
    space = _one_space(c, rows)
    if physical and space.inverse is None:
        raise ValueError(f"comparison space {space.name!r} has no inverse")
    cols_term = problem.columns(term.params)
    cols_pred = problem.columns(predictor.params)
    nk, n_fn = term.n_kernel, term._n_fn_params
    x_pred = np.asarray(x_pred)
    meta = None if c.meta is None else {k: v[rows] for k, v in c.meta.items()}
    # the prediction points carry the metadata the predictor was bound with
    meta_p = (
        None
        if predictor.meta is None
        else {k: _per_point(v, len(x_pred)) for k, v in predictor.meta.items()}
    )

    out = np.empty((samples.shape[0], x_pred.shape[0]))
    for i, theta in enumerate(samples):
        values = tuple(theta[cols_term])
        kv, av, cv = values[:nk], values[nk:n_fn], values[n_fn:]
        ym = c.ym(theta)
        # the term's own block on its active rows, exactly as the likelihood saw it
        K_full = term.value(
            c.x[rows],
            c.y[rows],
            ym[rows],
            *values,
            meta=meta,
            segments=entry.segments,
            labels=entry.labels,
        )
        Ktt = K_full[np.ix_(keep, keep)]
        r = (c.y - ym)[rows][keep]
        if train_noise_var is None:
            N = c.covariance.matrix(ym, theta)[np.ix_(pos, pos)] - Ktt
        else:
            N = _train_noise_matrix(train_noise_var, len(r))
        # kernel and amplitude at the training and prediction coordinates
        k = (
            term.kernel.clone_with_theta(np.asarray(kv, dtype=float))
            if nk
            else term.kernel
        )
        ctx_t = term.context(
            c.x[rows], c.y[rows], ym[rows], meta, *values,
            segments=entry.segments, labels=entry.labels,
        )  # fmt: skip
        mu = space(predictor(*theta[cols_pred]))
        ctx_p = TermContext(x=term.coords(x_pred, *cv), y=mu, ym=mu, _meta=meta_p)
        a_t, a_p = _amplitudes(term, ctx_t, ctx_p, av)
        Kst = np.outer(a_p, a_t[keep]) * k(as_2d(ctx_p.x), as_2d(ctx_t.x)[keep])
        Kss = a_p**2 * np.asarray(k.diag(as_2d(ctx_p.x)), dtype=float)
        f_mean, v = _condition(Ktt, Kst, r, N, 0.0)
        f_var = np.clip(Kss - np.einsum("ij,ij->j", v, v), 0.0, None)
        y = mu + f_mean + rng.normal(0.0, np.sqrt(f_var + float(noise_std) ** 2))
        out[i] = space.inverse(y) if physical else y
    return predictive_band(out, levels)


def _amplitudes(term, ctx_t, ctx_p, values):
    a = term.amplitude
    if a is None:
        return np.ones(len(ctx_t)), np.ones(len(ctx_p))
    if callable(a):
        return (
            np.broadcast_to(np.asarray(a(ctx_t, *values), dtype=float), (len(ctx_t),)),
            np.broadcast_to(np.asarray(a(ctx_p, *values), dtype=float), (len(ctx_p),)),
        )
    a = np.asarray(a, dtype=float)
    if a.ndim == 0:
        return np.full(len(ctx_t), float(a)), np.full(len(ctx_p), float(a))
    raise ValueError(
        "a kernel amplitude given as an array is defined only on the training "
        "rows; use a callable amplitude(c, ...) to predict at new points"
    )
