import numpy as np

from . import proposal
from . import params
from . import mcmc


class SamplingConfig:
    """
    Configuration for parameter sampling in a Bayesian inference context.
    This class encapsulates the parameters, starting location, proposal function,
    and prior distribution used for sampling.

    """

    def __init__(
        self,
        params: list[params.Parameter],
        starting_location: np.ndarray,
        proposal: proposal.ProposalDistribution,
        prior,
        sampling_algorithm: mcmc.SamplingAlgorithm = mcmc.metropolis_hastings,
    ):
        """
        Initializes the SamplingConfig with the provided parameters.
        Parameters:
        ----------
        params: list[params.Parameter]
            A list of Parameter objects defining the parameters to be sampled.
        starting_location: np.ndarray
            The initial parameter vector from which to start the sampling.
        proposal: proposal.ProposalDistribution
            A callable that defines the proposal distribution for sampling.
        prior: object
            An object representing the prior distribution, which must have a
            method `logpdf` for calculating the log probability density function.
        sampling_algorithm: mcmc.SamplingAlgorithm, optional
            The sampling algorithm to use for generating samples. Defaults to
            `mcmc.metropolis_hastings`.
        Raises:
        -------
        ValueError: If the proposal is not callable or if the prior does not
            have the required methods.
        """

        self.params = params
        self.starting_location = starting_location
        self.proposal = proposal
        self.prior = prior
        self.sampling_algorithm = sampling_algorithm

        _validate_object(
            prior,
            "prior",
            required_methods=["logpdf"],
        )

        if not callable(self.proposal):
            raise ValueError(
                "The proposal must be a callable object that takes in a "
                "parameter vector and an rng returns a proposed parameter vector."
            )

    def update_proposal(self, proposal: callable):
        """
        Update the proposal function used for sampling.

        Parameters:
        ----------
        proposal: callable
            A new proposal function that takes a parameter vector and returns
            a proposed parameter vector.
        """
        if not callable(proposal):
            raise ValueError(
                "The proposal must be a callable object that takes in a "
                "parameter vector and returns a proposed parameter vector."
            )
        self.proposal = proposal


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
