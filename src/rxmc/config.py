"""
Configuration classes for parameters and calibration settings, useful for
Markov Chain Monte Carlo (MCMC) simulations using external libraries like emcee,
with the flexibility to handle multiple likelihood models and constraints, either
using a batched Metropolis-in-Gibbs approach or a full joint sampling approach.
"""

from typing import Optional

import numpy as np
import scipy.stats

from rxmc.evidence import Evidence
from rxmc.params import Parameter


class ParameterConfig:
    """Configuration for a set of parameters, including their prior and initial
    proposal distribution.

    Attributes
    ----------
    params : list[Parameter]
        List of Parameter objects defining the parameters.
    ndim : int
        Number of parameters.
    prior : scipy.stats distribution
        Prior distribution for the parameters.
    initial_proposal_distribution : scipy.stats distribution
        Initial proposal distribution for the parameters.
    """

    def __init__(
        self,
        params: list[Parameter],
        prior: scipy.stats._distn_infrastructure.rv_frozen,
        initial_proposal_distribution: scipy.stats._distn_infrastructure.rv_frozen,
    ):
        """
        Initialize the ParameterConfig with parameters, prior, and initial
        proposal distribution.

        Parameters
        ----------
        params : list[Parameter]
            List of Parameter objects defining the parameters.
        prior : scipy.stats distribution
            Prior distribution for the parameters.
        initial_proposal_distribution : scipy.stats distribution
            Initial proposal distribution for the parameters.

        Raises
        ------
        ValueError
            If params is empty.
        ValueError
            If the dimensions of the prior or initial proposal distribution do
            not match the number of parameters.
        """
        self.params = params
        self.ndim = len(params)
        self.prior = prior
        self.initial_proposal_distribution = initial_proposal_distribution
        if self.ndim == 0:
            raise ValueError("Parameter list cannot be empty")
        if self.prior.dim != self.ndim:
            raise ValueError(
                "Prior distribution dimensions do not match number of parameters"
            )
        if self.initial_proposal_distribution.dim != self.ndim:
            raise ValueError(
                "Initial proposal distribution dimensions do not match number of parameters"
            )

    def x0(self, nwalkers: int) -> np.ndarray:
        """
        Generate initial positions for walkers.

        Parameters
        ----------
        nwalkers : int
            Number of walkers to generate initial positions for.
        """
        return self.initial_proposal_distribution.rvs(nwalkers)


class CalibrationConfig:
    """Configuration for calibration, including evidence, model parameters,
    and likelihood parameters.
    Attributes
    ----------
    evidence : Evidence
        Evidence object containing experimental constraints
    model_config : ParameterConfig
        Configuration for the model parameters.
    likelihood_configs : list[ParameterConfig]
        List of configurations for each likelihood's parameters.
    ndim : int
        Total number of parameters (model + likelihoods).
    dimensions : np.ndarray
        Array of dimensions for model and likelihood parameters.
    indices : np.ndarray
        Cumulative indices for splitting parameter vectors.
    """

    def __init__(
        self,
        evidence: Evidence,
        model_config: ParameterConfig,
        likelihood_configs: Optional[list[ParameterConfig]] = None,
    ):
        """
        Initialize the CalibrationConfig with evidence, model configuration,
        and likelihood configurations.
        Parameters
        ----------
        evidence : Evidence
            Evidence object containing experimental constraints
        model_config : ParameterConfig
            Configuration for the model parameters.
        likelihood_configs : list[ParameterConfig]
            List of configurations for each likelihood's parameters.
        Raises
        ------
        ValueError
            If the evidence has no constraints.
        ValueError
            If the model parameters do not match those in the evidence constraints.
        ValueError
            If the likelihood configurations do not match the likelihood models
            in the evidence constraints.
        """
        self.evidence = evidence
        self.model_config = model_config
        self.likelihood_configs = likelihood_configs or []
        self.ndim = model_config.ndim + sum(lc.ndim for lc in self.likelihood_configs)

        if (
            len(self.evidence.constraints) == 0
            and len(self.evidence.parametric_constraints) == 0
        ):
            raise ValueError("Evidence must have at least one constraint")
        if np.any(
            [
                c.physical_model.params != self.model_config.params
                for c in self.evidence.constraints
                + self.evidence.parametric_constraints
            ]
        ):
            raise ValueError(
                "Model parameters do not match those in the evidence constraints"
            )
        if len(self.likelihood_configs) != len(self.evidence.parametric_constraints):
            raise ValueError(
                "Likelihood configurations do not match the likelihood models"
                "in the evidence constraints"
            )
        for lc, c in zip(self.likelihood_configs, self.evidence.parametric_constraints):
            if lc.params != c.likelihood.params:
                raise ValueError(
                    "Likelihood parameters do not match those in the evidence constraints"
                )

        self.dimensions = np.array(
            [self.model_config.ndim] + [lc.ndim for lc in self.likelihood_configs]
        )

        # indices used to un-flatten parameter vector into sub-vectors
        # corresponding to model and likelihood parameters
        self.indices = np.cumsum(self.dimensions)

    def split_parameters(self, x) -> tuple[np.ndarray, list[np.ndarray]]:
        """
        Split a flat parameter vector into model and likelihood parameter vectors.

        Parameters
        ----------
        x : np.ndarray
            Flat parameter vector of shape (ndim,).
        Returns
        -------
        tuple[np.ndarray, list[np.ndarray]]
            Tuple containing the model parameter vector and a list of likelihood
            parameter vectors.
        """
        parts = np.split(x, self.indices[:-1])
        return parts[0], parts[1:]

    def log_prior(self, x) -> float:
        """
        Compute the log prior probability of a flat parameter vector.
        Parameters
        ----------
        x : np.ndarray
            Flat parameter vector of shape (ndim,).
        Returns
        -------
        float
            Log prior probability of the parameter vector.
        """
        xmodel, xlikelihoods = self.split_parameters(x)
        lprior = self.model_config.prior.logpdf(xmodel)
        lprior += sum(
            lc.prior.logpdf(xlikelihood)
            for lc, xlikelihood in zip(self.likelihood_configs, xlikelihoods)
        )
        return lprior

    def log_likelihood(self, x) -> float:
        """
        Compute the log likelihood of a flat parameter vector.
        Parameters
        ----------
        x : np.ndarray
            Flat parameter vector of shape (ndim,).
        Returns
        -------
        float
            Log likelihood of the parameter vector.
        """
        xmodel, xlikelihoods = self.split_parameters(x)
        return self.evidence.log_likelihood(xmodel, xlikelihoods)

    def log_posterior(self, x) -> float:
        """
        Compute the log posterior probability of a flat parameter vector.
        Parameters
        ----------
        x : np.ndarray
            Flat parameter vector of shape (ndim,).
        Returns
        -------
        float
            Log posterior probability of the parameter vector.
        """
        lp = self.log_prior(x)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.log_likelihood(x)
        if not np.isfinite(ll):
            return -np.inf
        return lp + ll

    def predict(self, xmodel) -> list[np.ndarray]:
        """
        Generate predictions for each constraint given model parameters.
        Parameters
        ----------
        xmodel : np.ndarray
            Model parameter vector of shape (model_config.ndim,).
        Returns
        -------
        list[np.ndarray]
            List of predictions for each constraint.
        """
        return [constraint.predict(*xmodel) for constraint in self.evidence.constraints]

    def conditional_posterior(self, x_lm, lm_index, ym) -> float:
        """
        Compute the log posterior for the parameters of a specific likelihood
        model, conditional upon the the observaed data for the corresponding constraints
        Parameters
        ----------
        x_lm : np.ndarray
            Parameter vector for the likelihood model of shape
            (likelihood_configs[lm_index].ndim,).
        lm_index : int
            Index of the likelihood model sector.
        ym : np.ndarray
            Observed data for the corresponding constraint.
        Returns
        -------
        float
            Log posterior probability for the likelihood model sector.
        """
        return self.evidence.constraints[lm_index].marginal_log_likelihood(
            ym, *x_lm
        ) + self.likelihood_configs[lm_index].prior.logpdf(x_lm)

    def starting_location(self, nwalkers) -> np.ndarray:
        """
        Generate initial positions for walkers in the full parameter space.
        Parameters
        nwalkers : int
            Number of walkers to generate initial positions for.
        Returns
        -------
        np.ndarray
            Initial positions for walkers of shape (nwalkers, ndim).
        """
        x0_model = self.model_config.x0(nwalkers)
        x0_likelihoods = [
            lc.x0(nwalkers).reshape(nwalkers, lc.ndim) for lc in self.likelihood_configs
        ]
        x0 = np.hstack([x0_model] + x0_likelihoods)
        return x0
