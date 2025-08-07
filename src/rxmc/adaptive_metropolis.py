import numpy as np
from typing import Callable, Tuple


def adaptive_metropolis(
    x0: np.ndarray,
    n_steps: int,
    log_posterior: Callable[[np.ndarray], float],
    rng: np.random.Generator,
    adapt_start: int = 1000,
    window_size: int = 1000,
    epsilon_fraction: float = 1e-6,
    previous_chain: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Adaptive Metropolis algorithm with a sliding window covariance adaptation.
        Parameters:
        ---------
        x0 : np.ndarray
            Initial point in the parameter space.
        n_steps : int
            Number of MCMC steps to perform.
        log_posterior : Callable[[np.ndarray], float]
            Function to compute the log posterior probability.
        rng : np.random.Generator
            Random number generator for reproducibility.
        adapt_start : int
            Step at which adaptation starts.
        window_size : int
            Size of the sliding window for covariance estimation.
        epsilon_fraction : float
            Fraction of the mean diagonal element to add to the covariance matrix for stability.
        previous_chain : np.ndarray, optional
            Previous chain to continue from, if any. Defaults to None. If provided, the new
            chain will be appended to it, and adapt_start will be ignored.
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
            proposal_cov = np.diag((x0 * 0.01) ** 2)
        else:
            # Determine the index range for the sliding window
            start_idx = max(0, i - window_size)
            history_subset = chain[start_idx:i, ...]

            # Use the window of samples to compute the covariance
            cov = np.atleast_2d(np.cov(history_subset.T))
            cov += epsilon_fraction * (
                np.eye(dim) @ np.mean(history_subset, axis=0) ** 2
            )
            proposal_cov = scale * cov

        # Propose new point
        x_new = rng.multivariate_normal(x, proposal_cov)
        logp_new = log_posterior(x_new)

        # Acceptance probability
        log_ratio = min(0, logp_new - logp)
        if np.log(rng.random()) < log_ratio:
            x = x_new
            logp = logp_new
            accepted += 1

        chain[i, :] = x
        logp_chain[i - start] = logp

    return chain[start:, ...], logp_chain, accepted
