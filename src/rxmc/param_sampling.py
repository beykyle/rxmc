from . import params
from .constraint import Constraint
from .corpus import Corpus

class Sampler:
    def __init__(
        self,
        params: list[params.Parameter],
        starting_location: np.ndarray,
        proposal: callable,
        prior,
    ):
        self.params = params
        self.starting_location = starting_location
        self.proposal = proposal
        self.prior = prior

        self._validate_object(
            prior,
            "prior",
            required_attributes=["mean"],
            required_methods=["rsv", "logpdf"],
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


class Walker:
    def __init__(self, physical_model_sampler: Sampler, corpus: Corpus,  likelihood_model_samplers: list[Sampler]=[], constraints: list[constraint.Constraint]=[]):
        """
        Initialize the SamplingConfig with a list of samplers.

        Parameters:
        ----------
        physical_model_samplers: Sampler
            A Sampler object for physical model parameters.
        corpus: Corpus
            A Corpus object containing the data for which the likelihood
            model is evaluated.
        likelihood_model_samplers: list[Sampler]
            A list of Sampler objects for likelihood model parameters.
        constraints: list[Constraint]
            A list of Constraint objects corresponding the the likelihood model
            Samplers
        """
        self.physical_model_sampler = physical_model_sampler
        self.corpus = corpus
        self.likelihood_model_samplers = likelihood_model_samplers
        self.constraints = constraints
        self.pm_params = [ self.physical_model_sampler.starting_location ]
        self.lm_params = [ sampler.starting_location for sampler in self.likelihood_model_samplers ]

    def log_likelihood(model_params, likelihood_params):
        return self.corpus.logpdf() # TODO

    def log_posterior(model_params, likelihood_params):
        return self.log_likelihood(model_params, likelihood_params) + self.log_prior(model_params, likelihood_params)

    def log_prior(model_params, likelihood_params):
        """
        Returns the log-prior probability of the model parameters and likelihood
        parameters.

        Parameters:
        ----------
        model_params: tuple
            The parameters of the physical model.
        likelihood_params: list[tuple]
            A list of tuples containing additional parameters for the likelihood
            model for each constraint.

        Returns:
        -------
        float
            The log-prior probability.
        """
        return self.physical_model_sampler.prior.logpdf(model_params) + \
               sum(sampler.prior.logpdf(likelihood_param) for sampler, likelihood_param in zip(self.likelihood_model_samplers, likelihood_params))

    def log_likelihood_conditionl_on_model_params(self, ym, likelihood_params):
        pass

    def run_chain(nsteps: int, nbatches: int, burnin: int):
        pass



def _validate_object(obj, name: str, required_attributes, required_methods):
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
