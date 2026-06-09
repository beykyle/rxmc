"""
Adaptive Metropolis MCMC sampler with sliding-window covariance adaptation.

The :func:`adaptive_metropolis` function implements the AM algorithm of Haario,
Saksman & Tamminen (2001), extended with a sliding window so that old samples
do not dominate the estimated proposal covariance.
"""

from typing import Callable, Tuple

import numpy as np


def adaptive_metropolis(
    x0: np.ndarray,
    bounds: np.ndarray,
    n_steps: int,
    log_posterior: Callable[[np.ndarray], float],
    rng: np.random.Generator,
    adapt_start: int = 1000,
    window_size: int = 1000,
    epsilon_fraction: float = 1e-6,
    previous_chain: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Adaptive Metropolis algorithm with sliding-window covariance adaptation.

    Before *adapt_start* steps the proposal is a diagonal Gaussian scaled to
    1 % of ``|x0|``.  After *adapt_start* steps the proposal covariance is
    estimated from the last *window_size* samples and scaled by
    ``2.38² / ndim`` (the Gelman–Roberts–Gilks optimal scale).

    Parameters
    ----------
    x0 : np.ndarray, shape (ndim,)
        Initial parameter vector.
    bounds : np.ndarray, shape (ndim, 2)
        Parameter bounds; each row is ``[lower, upper]``.  Proposals outside
        these bounds are rejected outright.
    n_steps : int
        Number of MCMC steps to generate.
    log_posterior : callable
        Function ``f(x) -> float`` returning the log posterior at ``x``.
    rng : np.random.Generator
        Random number generator for reproducibility.
    adapt_start : int, optional
        Step index at which covariance adaptation begins.  Ignored when
        *previous_chain* is supplied.  Defaults to ``1000``.
    window_size : int, optional
        Number of past samples used for covariance estimation.
        Defaults to ``1000``.
    epsilon_fraction : float, optional
        Fraction of the mean diagonal element added to the covariance for
        numerical stability.  Defaults to ``1e-6``.
    previous_chain : np.ndarray, shape (m, ndim), optional
        Chain from a prior run to continue from.  When provided the new
        samples are appended and adaptation uses all available history.

    Returns
    -------
    chain : np.ndarray, shape (n_steps, ndim)
        Newly generated samples (does not include *previous_chain*).
    logp_chain : np.ndarray, shape (n_steps,)
        Log posterior values for the new samples.
    accepted : int
        Number of accepted proposals in this run.
    """
    dim = x0.size
    if previous_chain is not None and previous_chain.shape[0] > 0:
        start = previous_chain.shape[0]
        chain = np.concatenate((previous_chain, np.zeros((n_steps, dim))), axis=0)
    else:
        start = 0
        chain = np.zeros((n_steps, dim))

    logp_chain = np.zeros(n_steps)
    accepted = 0

    x = x0.copy()
    logp = log_posterior(x)
    scale = 2.38**2 / dim

    for i in range(start, start + n_steps):
        if i < adapt_start:
            proposal_scale = np.maximum(np.abs(x0), 1.0) * 0.01
            proposal_cov = np.diag(proposal_scale**2)
        else:
            start_idx = max(0, i - window_size)
            history_subset = chain[start_idx:i, ...]
            cov = np.atleast_2d(np.cov(history_subset.T))
            cov += epsilon_fraction * np.diag(np.mean(history_subset, axis=0) ** 2)
            proposal_cov = scale * cov

        x_new = rng.multivariate_normal(x, proposal_cov)
        if np.any(x_new < bounds[:, 0]) or np.any(x_new > bounds[:, 1]):
            chain[i, :] = x
            logp_chain[i - start] = logp
            continue
        logp_new = log_posterior(x_new)

        log_ratio = min(0, logp_new - logp)
        if np.log(rng.random()) < log_ratio:
            x = x_new
            logp = logp_new
            accepted += 1

        chain[i, :] = x
        logp_chain[i - start] = logp

    return chain[start:, ...], logp_chain, accepted
