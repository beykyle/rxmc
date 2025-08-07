from typing import Callable, Tuple

import numpy as np


def metropolis_hastings(
    x0: np.ndarray,
    n_steps: int,
    log_posterior: Callable[[np.ndarray], float],
    rng: np.random.Generator,
    propose: Callable[[np.ndarray, np.random.Generator], np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Performs Metropolis-Hastings MCMC sampling.

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
        propose : Callable[[np.ndarray, np.random.Generator], np.ndarray]
            Function to propose new parameter values.
    Returns:
        tuple:
            - numpy.ndarray: The chain of samples generated.
            - numpy.ndarray: Log posteriors corresponding to the samples.
            - int: The number of accepted proposals.

    """
    chain = np.zeros((n_steps, x0.size))
    logp_chain = np.zeros((n_steps,))
    logp = log_posterior(x0)
    accepted = 0
    x = x0
    for i in range(n_steps):
        x_new = propose(x, rng)
        logp_new = log_posterior(x_new)
        # use sum log exp trick
        # https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
        log_ratio = min(0, logp_new - logp)
        xi = np.log(rng.random())
        if xi < log_ratio:
            x = x_new
            logp = logp_new
            accepted += 1

        chain[i, ...] = x
        logp_chain[i] = logp

    return chain, logp_chain, accepted
