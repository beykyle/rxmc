"""
Plain Metropolis-Hastings MCMC sampler.

The :func:`metropolis_hastings` function implements a single-chain MH kernel
with hard parameter bounds and a user-supplied proposal distribution.
"""

from typing import Callable, Tuple

import numpy as np


def metropolis_hastings(
    x0: np.ndarray,
    bounds: np.ndarray,
    n_steps: int,
    log_posterior: Callable[[np.ndarray], float],
    rng: np.random.Generator,
    propose: Callable[[np.ndarray, np.random.Generator], np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Metropolis-Hastings MCMC sampling.

    Proposals that fall outside *bounds* are rejected outright; otherwise the
    standard MH acceptance criterion is applied.

    Parameters
    ----------
    x0 : np.ndarray, shape (ndim,)
        Initial parameter vector.
    bounds : np.ndarray, shape (ndim, 2)
        Parameter bounds; each row is ``[lower, upper]``.
    n_steps : int
        Number of MCMC steps to generate.
    log_posterior : callable
        Function ``f(x) -> float`` returning the log posterior at ``x``.
    rng : np.random.Generator
        Random number generator for reproducibility.
    propose : callable
        Function ``g(x, rng) -> x_new`` that draws a candidate from the
        proposal distribution centred at ``x``.

    Returns
    -------
    chain : np.ndarray, shape (n_steps, ndim)
        Sampled parameter vectors.
    logp_chain : np.ndarray, shape (n_steps,)
        Log posterior values corresponding to each sample.
    accepted : int
        Number of accepted proposals.
    """
    chain = np.zeros((n_steps, x0.size))
    logp_chain = np.zeros((n_steps,))
    logp = float(log_posterior(x0))
    accepted = 0
    x = x0
    for i in range(n_steps):
        x_new = propose(x, rng)
        if np.any(x_new < bounds[:, 0]) or np.any(x_new > bounds[:, 1]):
            chain[i, ...] = x
            logp_chain[i] = logp
            continue
        logp_new = float(log_posterior(x_new))
        log_ratio = min(0, logp_new - logp)
        xi = np.log(rng.random())
        if xi < log_ratio:
            x = x_new
            logp = logp_new
            accepted += 1

        chain[i, ...] = x
        logp_chain[i] = logp

    return chain, logp_chain, accepted
