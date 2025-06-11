import numpy as np

import jitr

from .likelihood_model import LikelihoodModel
from .elastic_diffxs_model import ElasticDifferentialXSModel
from .elastic_diff_xs_observation import ElasticDifferentialXSObservation
from .constraint import Constraint


# TODO allow for writing and reading the precomputed workspaces to/from disk


class ElasticDifferentialXSConstraint(Constraint):
    """
    A `Constraint` for elastic differential cross sections and analyzing
    powers.

    It composes a set of `ElasticDifferentialXSObservation` objects, each
    describing angular data (differential cross section, `dXS/dA`, differential
    cross section as a ratio to the Rutherford cross section `dXS/dRuth` or
    analyzing power `Ay`) for a specific reaction at a given  laboratory
    energy, with an `ElasticDifferentialXSModel` that computes the model
    predictions for the corresponding angular data, and a `LikelihoodModel`
    that calculates the likelihood of the model parameters given the observed
    data.
    """

    def __init__(
        self,
        observations: list[ElasticDifferentialXSObservation],
        physical_model: ElasticDifferentialXSModel,
        likelihood_model: LikelihoodModel,
    ):
        """
        Params:
        ----------
        observations: list[ElasticDifferentialXSObservation]
            List of ReactionObservations, each containing the reaction,
            laboratory energy, and measurements.
        physical_model: ElasticDifferentialXSModel
            A callable that takes a
            `jitr.xs.elastic.DifferentialWorkspace` and a tuple of params,
            and returns the model predictions for the corresponding
            observation.
        likelihood_model: LikelihoodModel
            The model used to calculate the likelihood of the model parameters
            given the observed data.
        """
        super().__init__(
            observations=observations,
            physical_model=physical_model,
            likelihood_model=likelihood_model,
        )

    def chi2_observation(self, obs, workspaces, *params):
        """
        Calculate the chi-squared statistic (or Mahalanobis distance) between
        the model prediction, given the parameters, and the observed data for
        a single observation.

        Parameters:
        ----------
        obs : ReactionObservation
            The observed data that the model will attempt to reproduce.
        workspaces : list[DifferentialWorkspace]
            The precomputed workspaces corresponding to the observation.
        params : tuple
            The parameters of the physical model

        Returns:
        -------
        float
            The chi-squared statistic.
        """

    def logpdf_observation(self, obs, workspaces, *params):
        """
        Calculate the log probability density function (logpdf) that the model
        predictions, given the parameters, reproduce the observed data for a
        single observation.

        Parameters:
        ----------
        obs : ReactionObservation
            The observed data that the model will attempt to reproduce.
        workspaces : list[DifferentialWorkspace]
            The precomputed workspaces corresponding to the observation.
        params : tuple
            The parameters of the physical model

        Returns:
        -------
        float
            The log probability density of the observation given the
            parameters.
        """
        # TODO concatenate model predictions into 1d array and pass into
        # LikelihoodModel
        ym = np.concat([self.model_observation(obs, ws, *params) for ws in workspaces])

    def model_observation(self, obs, workspaces, *params):
        """
        Compute the model output for a single observation, given params.

        Parameters:
        ----------
        obs : ReactionObservation
            The observed data that the model will attempt to reproduce.
        workspaces : list[DifferentialWorkspace]
            The precomputed workspaces corresponding to the observation.
        params : tuple
            The parameters of the physical model

        Returns:
        -------
        float
            The model prediction for the observed data.
        """
        return [self.physical_model(obs, ws, *params) for ws in workspaces]

    def model(self, *params):
        """
        Compute the model output for each observation, given params.

        Parameters:
        ----------
        params : tuple
            The parameters of the physical model

        Returns:
        -------
        float
            The model predictions for the observed data.
        """
        return [
            self.model_observation(obs, workspaces)
            for obs, workspaces in zip(self.observations, self.constraint_workspaces)
        ]

    def logpdf(self, params, likelihood_params=None):
        """
        Calculate the log probability density function (logpdf) that the model
        predictions, given the parameters, reproduce the observed data.

        Parameters:
        ----------
        observation : Observation
            The observed data that the model will attempt to reproduce.
        params : tuple
            The parameters of the physical model
        likelihood_params : tuple, optional
            Additional parameters for the likelihood model, if any.


        Returns:
        -------
        float
            The log probability density of the observation given the
            parameters.
        """
        return sum(
            self.logpdf_observation(obs, ws, *params)
            for obs, ws in zip(self.observations, self.constraint_workspaces)
        )

    def chi2(self, params, likelihood_params=None):
        """
        Calculate the chi-squared statistic (or Mahalanobis distance) between
        the model prediction, given the parameters, and the observed data.

        Parameters:
        ----------
        params : tuple
            The parameters of the physical model
        likelihood_params : tuple, optional
            Additional parameters for the likelihood model, if any.

        Returns:
        -------
        float
            The chi-squared statistic.
        """
        return sum(
            self.likelihood.chi2(
                obs, self.physical_model(obs, ws, *params), *likelihood_params
            )
            for obs, ws in zip(self.observations, self.constraint_workspaces)
        )

    def logpdf_and_ym(self, params, likelihood_params=None):
        """
        Calculate the log probability density function (logpdf) that the model
        predictions, given the parameters, reproduce the observed data, and
        returns it along with the model predictions.

        Parameters:
        ----------
        params : tuple
            The parameters of the physical model
        likelihood_params : tuple, optional
            Additional parameters for the likelihood model, if any.

        Returns:
        -------
        float
            The log probability density of the observation given the
            parameters.
        list
            The model predictions for the observed data.
        """
        ym = []
        logpdf = 0.0
        for obs, ws in zip(self.observations, self.constraint_workspaces):
            y_pred = self.physical_model(obs, ws, *params)
            ym.append(y_pred)
            logpdf += self.likelihood.logpdf(obs, y_pred, *likelihood_params)

        return logpdf, ym


