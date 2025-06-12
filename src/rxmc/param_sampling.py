import numpy as np

from . import params
from . import mcmc


class SamplingConfig:
    """
    Configuration for parameter sampling in a Bayesian inference context.
    This class encapsulates the parameters, starting location, proposal function,
    and prior distribution used for sampling.
    It validates the provided prior and proposal function to ensure they meet
    the required interface.
    Parameters
    ----------
    params: list[params.Parameter]
        A list of Parameter objects representing the parameters to be sampled.
    starting_location: np.ndarray
        A numpy array representing the initial location in parameter space
        from which sampling will begin.
    proposal: callable
        A callable function that takes a parameter vector and returns
        a proposed parameter vector for sampling.
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
        proposal: callable,
        prior,
        sampling_algorithm: callable = mcmc.metropolis_hastings,
    ):
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
