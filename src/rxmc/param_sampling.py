import numpy as np

from . import proposal
from . import params
from . import mcmc


class SamplingConfig:
    """
    Configuration for parameter sampling in a Bayesian inference context.
    This class encapsulates the parameters, starting location, proposal function,
    and prior distribution used for sampling.

    Parameters
    ----------
    params: list[params.Parameter]
        A list of Parameter objects representing the parameters to be sampled.
    starting_location: np.ndarray
        A numpy array representing the initial location in parameter space
        from which sampling will begin.
    proposal: proposal.ProposalDistribution
        An instance of a ProposalDistribution that defines how to propose
        new parameter values based on the current state.
    prior: object
        An object representing the prior distribution over the parameters.
        It must implement a 'logpdf' method, which takes in a parameter
        vector and returns the log pdf of the prior.
    sampling_algorithm: callable, optional
        A callable sampling algorithm to be used for sampling.
        Defaults to `mcmc.metropolis_hastings`.
    """

    def __init__(
        self,
        params: list[params.Parameter],
        starting_location: np.ndarray,
        proposal: proposal.ProposalDistribution,
        prior,
        sampling_algorithm: callable = mcmc.metropolis_hastings,
        rng: np.random.Generator = np.random.default_rng(),
    ):
        self.params = params
        self.starting_location = starting_location
        self.proposal = proposal
        self.proposal.rng = rng
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
                "parameter vector and returns a proposed parameter vector."
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
