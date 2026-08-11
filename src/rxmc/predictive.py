"""
Predictive-uncertainty helpers.

A :class:`~rxmc.covariance.KernelTerm` (like the GP discrepancy model it replaced)
only inflates the covariance *at the data points* with ``K(X, X)`` — it does not
propagate the discrepancy to new ``x``.  :func:`gp_posterior_predictive` performs
the standard Gaussian-process conditioning needed to predict the discrepancy (mean
and covariance) at new points, and :func:`total_predictive_band` turns a posterior
sample of ``[model params | kernel log-theta]`` into a data-space predictive band
that propagates model-parameter, discrepancy, and observation-noise uncertainty.

The kernel is duck-typed exactly as in :class:`~rxmc.covariance.KernelTerm`: a
scikit-learn-style object exposing ``clone_with_theta`` and ``__call__``, with
``theta`` in sklearn **log-theta** space.
"""

import numpy as np
import scipy as sc

from .covariance import as_2d

__all__ = [
    "gp_posterior_predictive",
    "total_predictive_band",
    "predictive_band",
]


def _train_noise_matrix(train_noise_var, n) -> np.ndarray:
    if train_noise_var is None:
        return np.zeros((n, n))
    v = np.asarray(train_noise_var, dtype=float)
    if v.ndim == 0:
        return float(v) * np.eye(n)
    if v.ndim == 1:
        return np.diag(v)
    return v


def gp_posterior_predictive(
    kernel,
    theta,
    X_train,
    residuals,
    X_pred,
    *,
    train_noise_var=None,
    jitter=1e-10,
):
    r"""Posterior mean and covariance of a GP discrepancy at ``X_pred``.

    Conditions a zero-mean GP with covariance ``kernel`` (rebuilt at ``theta``) on
    the observed ``residuals`` at ``X_train`` and returns its posterior at
    ``X_pred``:

    .. math::

        \bar f_* = K_{*t} (K_{tt} + N)^{-1} r, \qquad
        \mathrm{cov}_* = K_{**} - K_{*t} (K_{tt} + N)^{-1} K_{t*}

    where ``N`` is the training-noise covariance.

    Parameters
    ----------
    kernel : sklearn-style kernel
        Object with ``clone_with_theta`` and ``__call__`` (as for
        :class:`~rxmc.covariance.KernelTerm`).
    theta : array-like
        Kernel hyperparameters in sklearn **log-theta** space.
    X_train, X_pred : array-like
        Training and prediction inputs (1-D promoted to a column).
    residuals : array-like, shape (n_train,)
        Observed minus model-mean at ``X_train``.
    train_noise_var : float, array-like, or matrix, optional
        Training-noise variance: scalar (``var*I``), per-point vector
        (``diag``), or full covariance.  Defaults to none.
    jitter : float, optional
        Diagonal nugget added to the training covariance for stability.

    Returns
    -------
    mean : np.ndarray, shape (n_pred,)
    cov : np.ndarray, shape (n_pred, n_pred)
    """
    k = kernel.clone_with_theta(np.asarray(theta, dtype=float))
    Xp = as_2d(X_pred)
    mean, v = _gp_condition(
        k, X_train, residuals, Xp, train_noise_var=train_noise_var, jitter=jitter
    )
    Kss = np.asarray(k(Xp), dtype=float)
    cov = Kss - v.T @ v
    return mean, cov


def _gp_condition(k, X_train, residuals, Xp, *, train_noise_var, jitter):
    """Shared GP conditioning: returns (mean, v).

    ``k`` is an already-instantiated kernel (``clone_with_theta`` applied);
    ``Xp`` is the 2-D prediction grid.  ``mean = Kst @ alpha`` and
    ``v = L^{-1} Kst.T`` so callers form the full posterior covariance
    (``Kss - v.T @ v``) or only its diagonal (``diag(Kss) - sum(v**2, 0)``).
    """
    Xtr = as_2d(X_train)
    r = np.asarray(residuals, dtype=float)
    n = Xtr.shape[0]

    Ktt = np.asarray(k(Xtr), dtype=float)
    Kst = np.asarray(k(Xp, Xtr), dtype=float)

    Ktrain = Ktt + _train_noise_matrix(train_noise_var, n) + jitter * np.eye(n)
    L = sc.linalg.cholesky(Ktrain, lower=True)
    alpha = sc.linalg.cho_solve((L, True), r)
    mean = Kst @ alpha
    v = sc.linalg.solve_triangular(L, Kst.T, lower=True)
    return mean, v


def _gp_posterior_mean_var(
    kernel, theta, X_train, residuals, X_pred, *, train_noise_var=None, jitter=1e-10
):
    """Posterior mean and **diagonal variance** of a GP discrepancy at ``X_pred``.

    Equivalent to ``mean, diag(cov)`` from :func:`gp_posterior_predictive` but
    avoids building the full ``n_pred x n_pred`` covariance — O(n_pred * n_train)
    instead of O(n_pred^2 * n_train).  Uses ``kernel.diag(X_pred)`` for the prior
    variances.
    """
    k = kernel.clone_with_theta(np.asarray(theta, dtype=float))
    Xp = as_2d(X_pred)
    mean, v = _gp_condition(
        k, X_train, residuals, Xp, train_noise_var=train_noise_var, jitter=jitter
    )
    prior_var = np.asarray(k.diag(Xp), dtype=float)
    var = prior_var - np.einsum("ij,ij->j", v, v)
    return mean, var


def predictive_band(draws, levels=(16, 50, 84)) -> np.ndarray:
    """Percentile band over a matrix of predictive draws.

    Parameters
    ----------
    draws : array-like, shape (n_draws, n_points)
        Predictive samples (each row a function evaluated on a grid).
    levels : sequence of float, optional
        Percentile levels.

    Returns
    -------
    np.ndarray, shape (len(levels), n_points)
    """
    return np.percentile(np.asarray(draws, dtype=float), levels, axis=0)


def total_predictive_band(
    mean_fn,
    kernel,
    x_train,
    y_train,
    x_pred,
    draws,
    n_model_params,
    *,
    theta_cols=None,
    noise_std=0.0,
    train_noise_var=None,
    levels=(16, 84),
    n_draws=400,
    rng=None,
):
    r"""Data-space predictive band that propagates *total* uncertainty.

    For each posterior draw ``q`` it splits out the model parameters and the kernel
    log-theta, forms the residual ``y_train - mean_fn(x_train, *model_params)``,
    conditions the GP discrepancy at ``x_pred``, and samples

    .. math::

        y_*^{(s)} = \mathrm{mean\_fn}(x_*) + \bar f_*^{(s)}
                    + \mathcal N\!\big(0,\ \mathrm{var}_*^{(s)} + \sigma^2\big),

    returning the requested percentile band over the samples.  Only the GP's
    posterior *variance* is propagated per point (the off-diagonal posterior
    covariance is not used for the marginal band).

    Parameters
    ----------
    mean_fn : callable
        ``mean_fn(x, *model_params) -> y`` on a raw ``x`` array (e.g. a model's
        ``.y`` plotting helper).
    kernel : sklearn-style kernel
        The discrepancy kernel (as passed to :class:`~rxmc.covariance.KernelTerm`).
    x_train, y_train, x_pred : array-like
        Training inputs/outputs and the prediction grid.
    draws : array-like, shape (n_samples, n_draw_cols)
        Posterior chain rows.
    n_model_params : int
        Number of leading model parameters in each draw (consumed by ``mean_fn``).
    theta_cols : array-like of int, optional
        Column indices selecting the kernel log-theta within each draw row, in
        ``kernel.theta`` order.  When ``None`` (default) the kernel theta is taken
        as the trailing columns ``q[n_model_params:]`` and the row width is
        validated against ``len(kernel.theta)``.  Pass explicit columns when the
        draw also carries other covariance/likelihood nuisances.
    noise_std : float, optional
        Observation noise std-dev, used both to condition the GP and as the
        prediction-point noise (overridden for conditioning by
        ``train_noise_var`` if given).
    train_noise_var : float or array-like, optional
        Training-noise covariance for conditioning (defaults to ``noise_std**2``).
    levels : sequence of float, optional
        Percentile levels for the returned band.
    n_draws : int, optional
        Number of posterior rows to subsample (if the chain is longer).
    rng : np.random.Generator, optional

    Returns
    -------
    np.ndarray, shape (len(levels), len(x_pred))

    Raises
    ------
    ValueError
        If the kernel-theta selection does not have ``len(kernel.theta)`` columns.
    """
    rng = np.random.default_rng() if rng is None else rng
    draws = np.asarray(draws, dtype=float)
    if draws.shape[0] > n_draws:
        draws = draws[rng.choice(draws.shape[0], n_draws, replace=False)]

    n_theta = len(kernel.theta)
    if theta_cols is None:
        n_trailing = draws.shape[1] - n_model_params
        if n_trailing != n_theta:
            raise ValueError(
                f"draws have {n_trailing} columns after the {n_model_params} model "
                f"parameters, but the kernel has {n_theta} hyperparameters. Pass "
                "theta_cols to select the kernel log-theta columns explicitly."
            )
        theta_cols = np.arange(n_model_params, n_model_params + n_theta)
    else:
        theta_cols = np.asarray(theta_cols, dtype=int)
        if theta_cols.shape[0] != n_theta:
            raise ValueError(
                f"theta_cols selects {theta_cols.shape[0]} columns but the kernel "
                f"has {n_theta} hyperparameters."
            )

    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    x_pred = np.asarray(x_pred, dtype=float)
    pred_noise_var = float(noise_std) ** 2
    cond_noise = pred_noise_var if train_noise_var is None else train_noise_var

    samples = np.empty((draws.shape[0], x_pred.shape[0]))
    for i, q in enumerate(draws):
        model_params = q[:n_model_params]
        theta = q[theta_cols]
        residual = y_train - mean_fn(x_train, *model_params)
        disc_mean, disc_var = _gp_posterior_mean_var(
            kernel, theta, x_train, residual, x_pred, train_noise_var=cond_noise
        )
        disc_var = np.clip(disc_var, 0.0, np.inf)
        mu = mean_fn(x_pred, *model_params) + disc_mean
        samples[i] = mu + rng.normal(0.0, np.sqrt(disc_var + pred_noise_var))

    return predictive_band(samples, levels)
