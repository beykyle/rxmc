from typing import Callable, Tuple

import numpy as np

# signature for the sampling algorithm
# for type hints
SamplingAlgorithm = Callable[
    [
        np.ndarray,  # x0
        int,  # n_steps
        Callable[[np.ndarray], float],  # log_posterior
        Callable[[np.ndarray, np.random.Generator], np.ndarray],
        # proposal distribution
        np.random.Generator,  # rng
    ],
    # Return type:
    # chain: np.ndarray,
    # logp_chain: np.ndarray,
    # accepted: int
    Tuple[np.ndarray, np.ndarray, int],
]


def metropolis_hastings(
    x0: np.ndarray,
    n_steps: int,
    log_posterior: Callable[[np.ndarray], float],
    propose: Callable[[np.ndarray, np.random.Generator], np.ndarray],
    rng: np.random.Generator = None,
):
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
        propose : Callable[[np.ndarray, np.random.Generator], np.ndarray]
            Function to propose new parameter values.
        rng : np.random.Generator, optional
            Random number generator for reproducibility. Defaults to None,
            which uses the default RNG with seed 42.
    Returns:
        tuple:
            - numpy.ndarray: The chain of samples generated.
            - numpy.ndarray: Log posteriors corresponding to the samples.
            - int: The number of accepted proposals.

    """
    if rng is None:
        rng = np.random.default_rng(42)
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

    return np.array(chain), np.array(logp_chain), accepted
