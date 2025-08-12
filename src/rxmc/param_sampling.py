from typing import Callable

import numpy as np

from . import params, proposal
from .adaptive_metropolis import adaptive_metropolis
from .metropolis_hastings import metropolis_hastings


class Sampler:
    """
    Base class wrapping sampling algorithm and recording samples and
    statistics
    """

    def __init__(
        self,
        params: list[params.Parameter],
        prior,
        starting_location: np.ndarray,
        sampling_algorithm,
        args: tuple = None,
        kwargs: dict = None,
    ):
        """
        Initializes the Sampler with the provided parameters.
        Parameters:
        ----------
        params: list[params.Parameter]
            List of parameters to sample.
        prior: object
            Prior distribution object that has a method `logpdf`.
        starting_location: np.ndarray
            Initial parameter values for the chain.
        sampling_algorithm: Callable
            Function that implements the sampling algorithm.
        args: list
            Additional positional arguments to pass to the sampling algorithm.
        kwargs: dict
            Additional keyword arguments to pass to the sampling algorithm.
        """
        self.params = params
        self.starting_location = starting_location
        self.prior = prior
        self.sampling_algorithm = sampling_algorithm
        self.args = args if args is not None else ()
        self.kwargs = kwargs if kwargs is not None else {}
        self.batches_run = 0
        self.n_steps = []
        self.n_accepted = []
        self.chain = np.empty((0, starting_location.size))
        self.logp_chain = np.empty((0,))
        self.state = np.atleast_1d(starting_location)

        _validate_object(
            prior,
            "prior",
            required_methods=["logpdf"],
        )

    def record_batch(
        self, n_steps: int, n_accepted: int, chain: np.ndarray, logp_chain: np.ndarray
    ):
        """
        Records the batch of samples and acceptance statistics.

        Parameters:
        ----------
        n_steps: int
            Number of steps in the batch.
        n_accepted: int
            Number of accepted samples in the batch.
        chain: np.ndarray
            Array of sampled parameter vectors.
        logp_chain: np.ndarray
            Array of log posterior values corresponding to the sampled parameter vectors.
        """
        self.batches_run += 1
        self.n_steps.append(n_steps)
        self.n_accepted.append(n_accepted)
        self.chain = np.concatenate((self.chain, chain), axis=0)
        self.logp_chain = np.concatenate((self.logp_chain, logp_chain), axis=0)

    def sample(
        self,
        n_steps: int,
        starting_location: np.ndarray,
        rng: np.random.Generator,
        log_posterior: Callable[[np.ndarray], float],
        burn: bool = False,
    ):
        """
        Samples from the posterior distribution using the specified
        sampling algorithm, updating the state and recording the chain,
        log posterior values, and acceptance statistics.

        Parameters:
        ----------
        n_steps: int
            Number of steps to sample.
        starting_location: np.ndarray
            Initial parameter values for the chain.
        rng: np.random.Generator
            Random number generator for reproducibility.
        log_posterior: Callable[[np.ndarray], float]
            Function that computes the log posterior probability of
            a parameter vector.
        burn: bool
            If True, the samples are considered burn-in and will not
            be recorded in the chain, only the current state will be updated.
        """
        chain, logp_chain, accepted = self.sampling_algorithm(
            starting_location,
            n_steps,
            log_posterior,
            rng,
            *self.args,
            **self.kwargs,
        )
        self.state = np.atleast_1d(chain[-1, :])

        if not burn:
            self.record_batch(n_steps, accepted, chain, logp_chain)

    def most_recent_batch_acceptance_fraction(self) -> float:
        """
        Returns the acceptance fraction of the most recent batch run.
        """
        if self.batches_run == 0:
            return 0.0
        return self.n_accepted[-1] / self.n_steps[-1]

    def batch_acceptance_fractions(self) -> np.ndarray:
        """
        Returns the acceptance fraction of the sampler in each batch run.
        """
        if self.batches_run == 0:
            return np.array([0.0])
        return np.array(self.n_accepted) / np.array(self.n_steps)

    def overall_acceptance_fraction(self) -> float:
        """
        Returns the overall acceptance fraction of the sampler.
        """
        if self.batches_run == 0:
            return 0.0
        return sum(self.n_accepted) / sum(self.n_steps)


class MetropolisHastingsSampler(Sampler):
    """
    Metropolis-Hastings sampler. The ol' reliable.
    """

    def __init__(
        self,
        params: list[params.Parameter],
        prior,
        starting_location: np.ndarray,
        proposal: proposal.ProposalDistribution,
    ):
        """
        Parameters:
        ----------
        params: list[params.Parameter]
            List of parameters to sample.
        prior: object
            Prior distribution object that has a method `logpdf`.
        starting_location: np.ndarray
            Initial parameter values for the chain.
        proposal: proposal.ProposalDistribution
            Proposal distribution object that has a method `__call__` which
            takes in a parameter vector and an rng, returning a proposed
            parameter vector.
        """

        self.proposal = proposal
        super().__init__(
            params,
            prior,
            starting_location,
            metropolis_hastings,
            args=[self.proposal],
            kwargs={},
        )

        if not callable(proposal):
            raise ValueError(
                "The proposal must be a callable object that takes in a "
                "parameter vector and an rng returns a proposed parameter"
                " vector."
            )


class AdaptiveMetropolisSampler(Sampler):
    """
    Adaptive Metropolis sampler for MCMC, which adapts the proposal covariance
    based on the samples collected so far, with a sliding window.
    """

    def __init__(
        self,
        params: list[params.Parameter],
        prior,
        starting_location: np.ndarray,
        adapt_start: int = 100,
        window_size: int = 1000,
        epsilon_fraction: float = 1e-6,
    ):
        """
        Parameters:
        ----------
        params: list[params.Parameter]
            List of parameters to sample.
        prior: object
            Prior distribution object that has a method `logpdf`.
        starting_location: np.ndarray
            Initial parameter values for the chain.
        adapt_start: int
            Step at which adaptation begins.
        window_size: int
            Size of the sliding window for covariance estimation.
        epsilon: float
            Small term to regularize the covariance matrix.
        """
        super().__init__(
            params,
            prior,
            starting_location,
            adaptive_metropolis,
            args=[],
            kwargs={
                "adapt_start": adapt_start,
                "window_size": window_size,
                "epsilon_fraction": epsilon_fraction,
            },
        )

    def sample(
        self,
        n_steps: int,
        starting_location: np.ndarray,
        rng: np.random.Generator,
        log_posterior: Callable[[np.ndarray], float],
        burn: bool = False,
    ):
        """
        Overrides `Sampler.sample` method to provide the current
        chain to the adaptive_metropolis sampling algorithm.
        """
        chain, logp_chain, accepted = self.sampling_algorithm(
            starting_location,
            n_steps,
            log_posterior,
            rng,
            *self.args,
            **self.kwargs,
            previous_chain=self.chain,
        )
        self.state = np.atleast_1d(chain[-1, :])

        if not burn:
            self.record_batch(n_steps, accepted, chain, logp_chain)


class BatchedAdaptiveMetropolisSampler(Sampler):
    """
    Adaptive Metropolis sampler for MCMC, which adapts the proposal covariance
    based on the samples collected in the last batch
    """

    def __init__(
        self,
        params: list[params.Parameter],
        prior,
        starting_location: np.ndarray,
        initial_proposal_cov: np.ndarray,
        epsilon_fraction: float = 1e-6,
    ):
        """
        Parameters:
        ----------
        params: list[params.Parameter]
            List of parameters to sample.
        prior: object
            Prior distribution object that has a method `logpdf`.
        starting_location: np.ndarray
            Initial parameter values for the chain.
        initial_proposal_cov: np.ndarray
            Initial covariance matrix for the proposal distribution.
        epsilon: float
            Small term to regularize the covariance matrix.
        """
        self.proposal_cov = np.atleast_2d(initial_proposal_cov)
        self.proposal = proposal.NormalProposalDistribution(initial_proposal_cov)
        super().__init__(
            params,
            prior,
            starting_location,
            metropolis_hastings,
            args=[self.proposal],
        )
        ndim = starting_location.size
        self.scale = 2.38**2 / ndim
        self.epsilon_fraction = epsilon_fraction

    def sample(
        self,
        n_steps: int,
        starting_location: np.ndarray,
        rng: np.random.Generator,
        log_posterior: Callable[[np.ndarray], float],
        burn: bool = False,
    ):
        """
        Overrides `Sampler.sample` method to adapt the proposal
        covariance based on the samples collected in the last batch.
        """
        chain, logp_chain, accepted = self.sampling_algorithm(
            starting_location,
            n_steps,
            log_posterior,
            rng,
            *self.args,
            **self.kwargs,
        )
        self.state = np.atleast_1d(chain[-1, :])
        empirical_cov = np.atleast_2d(np.cov(chain.T))
        epsilon = self.epsilon_fraction * np.median(np.diag(empirical_cov))
        self.proposal_cov = (
            self.scale * empirical_cov + np.eye(empirical_cov.shape[0]) * epsilon
        )
        new_proposal = proposal.NormalProposalDistribution(self.proposal_cov)
        self.args = [new_proposal]
        if not burn:
            self.record_batch(n_steps, accepted, chain, logp_chain)


def _validate_object(obj, name: str, required_attributes=[], required_methods=[]):
    # Check for required attributes
    for attr in required_attributes:
        if not hasattr(obj, attr):
            raise ValueError(f"The {name} object must have a '{attr}' attribute.")

    # Check for required methods
    for method in required_methods:
        if not callable(getattr(obj, method, None)):
            raise ValueError(
                f"The {name} object must have a callable '{method}' method."
            )
