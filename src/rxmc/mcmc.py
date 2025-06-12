import numpy as np


def metropolis_hastings(
    x0,
    n_steps,
    log_posterior,
    propose,
    rng=None,
):
    """
    Performs Metropolis-Hastings MCMC sampling.

    Parameters:
        x0 (numpy.ndarray): Initial parameter values for the chain.
        n_steps (int): Number of steps/samples to generate.
        log_posterior (callable): Function to calculate the log posterior
            of a sample.
        propose (callable): Proposal function for generating new samples.
        rng (numpy.random.Generator, optional): Random number generator,
            default is initialized with a seed of 42.

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
        x_new = propose(x)
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
