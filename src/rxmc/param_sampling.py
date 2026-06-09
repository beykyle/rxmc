"""
Sampler classes wrapping MCMC algorithms for parameter estimation.

:class:`Sampler` is the base class; it wraps any sampling algorithm function,
records the chain and acceptance statistics, and manages state across batches.
Three concrete subclasses are provided:

- :class:`MetropolisHastingsSampler` — plain MH with a fixed proposal.
- :class:`AdaptiveMetropolisSampler` — sliding-window covariance adaptation.
- :class:`BatchedAdaptiveMetropolisSampler` — per-batch covariance adaptation.
"""

from typing import Callable

import numpy as np

from . import params, proposal
from .adaptive_metropolis import adaptive_metropolis
from .metropolis_hastings import metropolis_hastings


class Sampler:
    """Base class wrapping a sampling algorithm with chain recording.

    Parameters
    ----------
    params : list of Parameter
        Parameters to sample.
    prior : object
        Prior distribution with a callable ``logpdf(x)`` method.
    starting_location : np.ndarray, shape (ndim,)
        Initial parameter vector.
    sampling_algorithm : callable
        Function implementing the sampling algorithm.  Must have the
        signature ``f(x0, bounds, n_steps, log_posterior, rng, *args, **kwargs)``
        and return ``(chain, logp_chain, n_accepted)``.
    args : tuple, optional
        Extra positional arguments passed to *sampling_algorithm*.
    kwargs : dict, optional
        Extra keyword arguments passed to *sampling_algorithm*.
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
        self.bounds = np.array([param.bounds for param in params])

        _validate_object(
            prior,
            "prior",
            required_methods=["logpdf"],
        )

    def record_batch(
        self, n_steps: int, n_accepted: int, chain: np.ndarray, logp_chain: np.ndarray
    ):
        """Append a completed batch to the running chain.

        Parameters
        ----------
        n_steps : int
            Number of steps in the batch.
        n_accepted : int
            Number of accepted proposals in the batch.
        chain : np.ndarray, shape (n_steps, ndim)
            Sampled parameter vectors.
        logp_chain : np.ndarray, shape (n_steps,)
            Log posterior values for the batch.
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
        """Run the sampling algorithm for one batch.

        Updates ``self.state`` to the last sample; records the batch unless
        *burn* is ``True``.

        Parameters
        ----------
        n_steps : int
            Number of steps to run.
        starting_location : np.ndarray, shape (ndim,)
            Starting parameter vector for this batch.
        rng : np.random.Generator
            Random number generator.
        log_posterior : callable
            Function ``f(x) -> float`` returning the log posterior at ``x``.
        burn : bool, optional
            If ``True``, discard samples (burn-in); only ``self.state`` is
            updated.  Defaults to ``False``.
        """
        chain, logp_chain, accepted = self.sampling_algorithm(
            starting_location,
            self.bounds,
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
        """Acceptance fraction of the most recent batch.

        Returns
        -------
        float
            Fraction of proposals accepted in the last batch, or ``0.0`` if
            no batches have been run.
        """
        if self.batches_run == 0:
            return 0.0
        return self.n_accepted[-1] / self.n_steps[-1]

    def batch_acceptance_fractions(self) -> np.ndarray:
        """Acceptance fraction for each completed batch.

        Returns
        -------
        np.ndarray
            Per-batch acceptance fractions, or ``[0.0]`` if no batches run.
        """
        if self.batches_run == 0:
            return np.array([0.0])
        return np.array(self.n_accepted) / np.array(self.n_steps)

    def overall_acceptance_fraction(self) -> float:
        """Overall acceptance fraction across all completed batches.

        Returns
        -------
        float
            Total accepted / total proposed, or ``0.0`` if no batches run.
        """
        if self.batches_run == 0:
            return 0.0
        return sum(self.n_accepted) / sum(self.n_steps)


class MetropolisHastingsSampler(Sampler):
    """Metropolis-Hastings sampler with a fixed proposal distribution.

    Parameters
    ----------
    params : list of Parameter
        Parameters to sample.
    prior : object
        Prior distribution with a callable ``logpdf(x)`` method.
    starting_location : np.ndarray, shape (ndim,)
        Initial parameter vector.
    proposal : ProposalDistribution
        Callable proposal distribution.  Must accept ``(x, rng)`` and return
        a proposed parameter vector.
    """

    def __init__(
        self,
        params: list[params.Parameter],
        prior,
        starting_location: np.ndarray,
        proposal: proposal.ProposalDistribution,
    ):
        if not callable(proposal):
            raise ValueError(
                "The proposal must be a callable object that takes in a "
                "parameter vector and an rng returns a proposed parameter"
                " vector."
            )
        self.proposal = proposal
        super().__init__(
            params,
            prior,
            starting_location,
            metropolis_hastings,
            args=[self.proposal],
            kwargs={},
        )


class AdaptiveMetropolisSampler(Sampler):
    """Metropolis sampler that adapts the proposal covariance with a sliding window.

    The proposal covariance is estimated from the last *window_size* samples
    after *adapt_start* steps have been collected.

    Parameters
    ----------
    params : list of Parameter
        Parameters to sample.
    prior : object
        Prior distribution with a callable ``logpdf(x)`` method.
    starting_location : np.ndarray, shape (ndim,)
        Initial parameter vector.
    adapt_start : int, optional
        Step at which adaptation begins.  Defaults to ``100``.
    window_size : int, optional
        Number of past samples used for covariance estimation.
        Defaults to ``1000``.
    epsilon_fraction : float, optional
        Small regularisation term for the proposal covariance.
        Defaults to ``1e-6``.
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
        """Run the adaptive sampler for one batch, passing history to the algorithm.

        Overrides :meth:`Sampler.sample` to forward the accumulated chain so
        the adaptive algorithm can estimate the covariance from all past samples.

        Parameters
        ----------
        n_steps : int
            Number of steps to run.
        starting_location : np.ndarray, shape (ndim,)
            Starting parameter vector.
        rng : np.random.Generator
            Random number generator.
        log_posterior : callable
            Function ``f(x) -> float`` returning the log posterior.
        burn : bool, optional
            If ``True``, discard samples.  Defaults to ``False``.
        """
        chain, logp_chain, accepted = self.sampling_algorithm(
            starting_location,
            self.bounds,
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
    """Metropolis sampler that updates the proposal covariance after each batch.

    After each completed (non-burn) batch the proposal covariance is replaced
    by the empirical covariance of that batch, scaled by ``2.38² / ndim``.

    Parameters
    ----------
    params : list of Parameter
        Parameters to sample.
    prior : object
        Prior distribution with a callable ``logpdf(x)`` method.
    starting_location : np.ndarray, shape (ndim,)
        Initial parameter vector.
    initial_proposal_cov : np.ndarray, shape (ndim, ndim)
        Initial proposal covariance matrix.
    epsilon_fraction : float, optional
        Small regularisation fraction added to the empirical covariance diagonal.
        Defaults to ``1e-6``.
    """

    def __init__(
        self,
        params: list[params.Parameter],
        prior,
        starting_location: np.ndarray,
        initial_proposal_cov: np.ndarray,
        epsilon_fraction: float = 1e-6,
    ):
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
        """Run the sampler for one batch, updating the proposal after recording.

        Overrides :meth:`Sampler.sample` to adapt the proposal covariance from
        the current batch's empirical covariance after each non-burn batch.

        Parameters
        ----------
        n_steps : int
            Number of steps to run.
        starting_location : np.ndarray, shape (ndim,)
            Starting parameter vector.
        rng : np.random.Generator
            Random number generator.
        log_posterior : callable
            Function ``f(x) -> float`` returning the log posterior.
        burn : bool, optional
            If ``True``, discard samples and skip covariance update.
            Defaults to ``False``.
        """
        chain, logp_chain, accepted = self.sampling_algorithm(
            starting_location,
            self.bounds,
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
    for attr in required_attributes:
        if not hasattr(obj, attr):
            raise ValueError(f"The {name} object must have a '{attr}' attribute.")

    for method in required_methods:
        if not callable(getattr(obj, method, None)):
            raise ValueError(
                f"The {name} object must have a callable '{method}' method."
            )
