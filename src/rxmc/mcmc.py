from pathlib import Path
from scipy import stats
import numpy as np

from rxmc import corpus


def metropolis_hastings(
    x0,
    n_steps,
    log_likelihood,
    propose,
    rng=None,
):
    """
    Performs Metropolis-Hastings MCMC sampling.

    Parameters:
        x0 (numpy.ndarray): Initial parameter values for the chain.
        n_steps (int): Number of steps/samples to generate.
        log_likelihood (callable): Function to calculate the log likelihood
            of a sample.
        propose (callable): Proposal function for generating new samples.
        rng (numpy.random.Generator, optional): Random number generator,
            default is initialized with a seed of 42.

    Returns:
        tuple:
            - numpy.ndarray: The chain of samples generated.
            - numpy.ndarray: Log likelihoods corresponding to the samples.
            - int: The number of accepted proposals.

    """
    if rng is None:
        rng = np.random.default_rng(42)
    chain = np.zeros((n_steps, x0.size))
    logl_chain = np.zeros((n_steps,))
    logl = log_likelihood(x0)
    accepted = 0
    x = x0
    for i in range(n_steps):
        x_new = propose(x)
        logl_new = log_likelihood(x_new)
        # use sum log exp trick
        # https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
        log_ratio = min(0, logl_new - logl)
        xi = np.log(rng.random())
        if xi < log_ratio:
            x = x_new
            logl = logl_new
            accepted += 1

        chain[i, ...] = x
        logl_chain[i] = logl

    return np.array(chain), np.array(logl_chain), accepted


def run_chain(
    prior,
    corpus: corpus.Corpus,
    nsteps: int,
    burnin: int = 0,
    seed: int = 42,
    batch_size: int = None,
    rank: int = 0,
    proposal_cov_scale_factor: float = 100,
    verbose: bool = True,
    output: Path = None,
):
    """
    Runs the MCMC chain with the specified parameters.

    Parameters:
        prior (object): The prior distribution object with mean, cov attributes
            and a logpdf method.
        corpus (corpus.Corpus): The corpus object with a logpdf method.
        nsteps (int): Total number of steps for the MCMC chain.
        batch_size (int): Number of steps per batch.
        burnin (int): Number of initial steps to discard.
        seed (int): Random seed for generating random numbers.
        rank (int): MPI rank for the current process.
        proposal_cov_scale_factor (float): Scale factor for the proposal
            covariance.
        verbose (bool): Flag to print extra logging information.
        output (Path, optional): Output directory for saving chain batches.

    Returns:
        tuple:
            - numpy.ndarray: Log likelihood values.
            - numpy.ndarray: The MCMC chain of samples.
            - float: The acceptance fraction.

    Raises:
        ValueError:
            - If the prior does not have the required methods/attributes.
    """

    if not hasattr(prior, "logpdf") or not callable(getattr(prior, "logpdf")):
        raise ValueError(
            "prior must have a callable .logpdf(x) method that takes in "
            "a point x in parameter space as an array"
        )
    if not hasattr(prior, "mean"):
        raise ValueError("prior must have a .mean attribute")
    if not hasattr(prior, "cov"):
        raise ValueError("prior must have a .cov attribute")

    # batching
    if batch_size is not None:
        rem_burn = burnin % batch_size
        n_burn_batches = burnin // batch_size
        burn_batches = n_burn_batches * [batch_size] + (rem_burn > 0) * [rem_burn]
        rem = (nsteps - burnin) % batch_size
        n_full_batches = (nsteps - burnin) // batch_size
        batches = n_full_batches * [batch_size] + (rem > 0) * [rem]
    else:
        batches = [nsteps - burnin]
        burn_batches = [burnin]

    if burnin == 0:
        burn_batches = []

    # RNG
    seed = seed + rank
    rng = np.random.default_rng(seed)

    # likelihood
    prior = prior
    corpus = corpus

    def log_likelihood(x):
        return prior.logpdf(x) + corpus.logpdf(x)

    # proposal distribution
    proposal_cov = prior.cov / proposal_cov_scale_factor
    proposal_mean = np.zeros_like(prior.mean)

    def proposal(x):
        return x + stats.multivariate_normal.rvs(
            mean=proposal_mean, cov=proposal_cov, random_state=rng
        )

    # starting location
    x0 = proposal(prior.mean)

    # run burn-in
    for i, steps_in_batch in enumerate(burn_batches):
        batch_chain, _, _ = metropolis_hastings(
            x0,
            steps_in_batch,
            log_likelihood,
            proposal,
            rng=rng,
        )
        if verbose:
            print(
                f"Rank: {rank}. Burn-in batch {i+1}/{len(burn_batches)}"
                f" completed, {steps_in_batch} steps."
            )

    # update starter location to tail of burn-in
    if burnin > 0:
        x0 = batch_chain[-1]

    # run real steps
    chain = []
    logl = []
    accepted = 0

    for i, steps_in_batch in enumerate(batches):
        batch_chain, batch_logl, accepted_in_batch = metropolis_hastings(
            x0,
            steps_in_batch,
            log_likelihood,
            proposal,
            rng=rng,
        )

        # diagnostics
        accepted += accepted_in_batch
        chain.append(batch_chain)
        logl.append(batch_logl)
        x0 = batch_chain[-1]
        if verbose:
            print(
                f"Rank: {rank}. Batch: {i+1}/{len(batches)} completed, "
                f"{steps_in_batch} steps. "
                f"Acceptance frac: {accepted_in_batch/steps_in_batch:.3f}"
            )

        # update proposal distribution?

        # update unknown covariance factor estimate (Gibbs sampling)

        # write record of batch chain to disk
        if output is not None:
            np.save(Path(output) / f"chain_{rank}_{i}.npy", batch_chain)

    logl = np.concatenate(logl, axis=0)
    chain = np.concatenate(chain, axis=0)

    return logl, chain, accepted / (nsteps - burnin)
