from typing import Callable

import numpy as np

from . import proposal
from . import params
from .metropolis_hastings import metropolis_hastings
from .adaptive_metropolis import adaptive_metropolis


class SamplingConfig:
    """Base class for sampling configurations"""

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
        Initializes the SamplingConfig with the provided parameters.
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

        _validate_object(
            prior,
            "prior",
            required_methods=["logpdf"],
        )

    def sample(
        self,
        n_steps: int,
        starting_location: np.ndarray,
        rng: np.random.Generator,
        log_posterior: Callable[[np.ndarray], float],
    ):
        """
        Samples from the posterior distribution using the specified sampling algorithm.
        This is the interface used by `Walker` to sample from the posterior distribution.

        Parameters:
        ----------
        n_steps: int
            Number of steps to sample.
        rng: np.random.Generator
            Random number generator for reproducibility.
        log_posterior: Callable[[np.ndarray], float]
            Function that computes the log posterior probability of a parameter vector.

        """
        return self.sampling_algorithm(
            starting_location,
            n_steps,
            log_posterior,
            rng,
            *self.args,
            **self.kwargs,
        )


class MetropolisHastingsSampler(SamplingConfig):
    """
    Metropolis-Hastings sampler for MCMC.
    """

    def __init__(
        self,
        params: list[params.Parameter],
        prior,
        starting_location: np.ndarray,
        proposal: proposal.ProposalDistribution,
    ):
        """
        Initializes the Metropolis-Hastings sampler with the provided parameters.

        Parameters:
        ----------
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
                "parameter vector and an rng returns a proposed parameter vector."
            )


class AdaptiveMetropolisSampler(SamplingConfig):
    """
    Adaptive Metropolis sampler for MCMC.
    """

    def __init__(
        self,
        params: list[params.Parameter],
        prior,
        starting_location: np.ndarray,
        adapt_start: int = 1000,
        window_size: int = 1000,
        epsilon_fraction: float = 1e-4,
    ):
        """
        Initializes the Adaptive Metropolis sampler with the provided parameters.

        Parameters:
        ----------
        adapt_start: int
            Step at which adaptation begins.
        epsilon: float
            Small term to regularize the covariance matrix.
        """
        super().__init__(
            params,
            prior,
            starting_location,
            adaptive_metropolis,
            args=[adapt_start, window_size, epsilon_fraction],
            kwargs={},
        )


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
