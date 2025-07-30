from typing import Callable, Tuple

import numpy as np


def adaptive_metropolis(
    x0: np.ndarray,
    n_steps: int,
    log_posterior: Callable[[np.ndarray], float],
    rng: np.random.Generator,
    adapt_start: int = 100,
    epsilon: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Adaptive Metropolis-Hastings MCMC.

    Parameters:
        x0 : np.ndarray
            Initial parameter values for the chain.
        n_steps : int
            Number of steps/samples to generate.
        log_posterior : Callable[[np.ndarray], float]
            Function to compute the log posterior probability of
            the parameters.
        rng : np.random.Generator
            Random number generator for reproducibility.
        adapt_start : int
            step at which adaptation begins.
        epsilon : float
            small term to regularize the covariance matrix.

    Returns:
        Tuple of:
            - chain: np.ndarray of samples
            - logp_chain: np.ndarray of log posterior values
            - accepted: int (number of accepted proposals)
    """
    dim = x0.size
    chain = np.zeros((n_steps, dim))
    logp_chain = np.zeros(n_steps)
    accepted = 0

    x = x0.copy()
    logp = log_posterior(x)
    cov = np.eye(dim)
    scale = 2.38**2 / dim
    history = [x.copy()]

    for i in range(n_steps):
        if i < adapt_start:
            proposal_cov = np.eye(dim)
        else:
            history_array = np.array(history)
            cov = np.cov(history_array.T) + epsilon * np.eye(dim)
            proposal_cov = scale * cov

        # Propose new point
        x_new = rng.multivariate_normal(x, proposal_cov)
        logp_new = log_posterior(x_new)
        log_ratio = min(0.0, logp_new - logp)
        xi = np.log(rng.random())
        if xi < log_ratio:
            x = x_new
            logp = logp_new
            accepted += 1

        history.append(x.copy())
        chain[i, :] = x
        logp_chain[i] = logp

    return chain, logp_chain, accepted
