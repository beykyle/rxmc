import numpy as np

from .physical_model import PhysicalModel
from .likelihood_model import LikelihoodModel
from .observation import Observation


class Constraint:
    """
    A `Constraint` is the composition of one or more `Observation`s with
    a `PhysicalModel` that is able to make predictions of the observed data,
    along with a `LikelihoodModel` which defines the likelihood of the
    observation given the model predictions.

    This class is meant to be a box that takes in model params and spits out
    the log likelihood or other staisticsT
    """

    def __init__(
        self,
        observations: list[Observation],
        physical_model: PhysicalModel,
        likelihood_model: LikelihoodModel,
    ):
        """
        Initialize the Constraint with some Observations, a PhysicalModel, and

        Parameters:
        ----------
        observations: list[Observation]
            The observed data that the model will attempt to reproduce.
        physical_model : PhysicalModel
            The model that predicts the observed data
        likelihood_model : LikelihoodModel
            The model that defines the likelihood of the observation given the
            physical model prediction.
        """
        self.observations = observations
        self.physical_model = physical_model
        self.likelihood = likelihood_model
        self.n_data_pts = sum(obs.n_data_pts for obs in self.observations)

    def model(self, model_params):
        """
        Compute the model output for each observation, given model_params.

        Parameters:
        ----------
        model_params : tuple
            The parameters of the physical model
        """
        return [self.physical_model(obs, *model_params) for obs in self.observations]

    def logpdf(self, model_params, likelihood_params=None):
        """
        Calculate the log probability density function (logpdf) that the
        model predictions, given the parameters, reproduce the observed data.

        Parameters:
        ----------
        model_params : tuple
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
            self.likelihood.logpdf(
                obs, self.physical_model(obs, *model_params), *likelihood_params
            )
            for obs in self.observations
        )

    def chi2(self, model_params, likelihood_params=None):
        """
        Calculate the chi-squared statistic (or Mahalanobis distance) between
        the model prediction, given the parameters, and the observed data.

        Parameters:
        ----------
        model_params : tuple
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
                obs, self.physical_model(obs, *model_params), *likelihood_params
            )
            for obs in self.observations
        )

    def logpdf_and_ym(self, model_params, likelihood_params=None):
        """
        Calculate the log probability density function (logpdf) that the model
        predictions, given the parameters, reproduce the observed data, and
        returns it along with the model predictions.

        Parameters:
        ----------
        model_params : tuple
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
        for obs in self.observations:
            y_pred = self.physical_model(obs, *model_params)
            ym.append(y_pred)
            logpdf += self.likelihood.logpdf(obs, y_pred, *likelihood_params)

        return logpdf, ym

    def num_pts_within_interval(
        self, ylow: list[np.ndarray], yhigh: list[np.ndarray], xlim=None
    ):
        """
        Count the number of points within the specified interval for each
        observation.

        Parameters:
        ----------
        ylow : list[np.ndarray]
            Lower bounds of the intervals for each observation.
        yhigh : list[np.ndarray]
            Upper bounds of the intervals for each observation.
        xlim : tuple, optional
            Limits for the x-axis, if applicable.

        Returns:
        -------
        int
            The total number of points within the specified intervals across
            all observations.
        """
        return sum(
            obs.num_pts_within_interval(ylow[i], yhigh[i], xlim)
            for i, obs in enumerate(self.observations)
        )

    def empirical_coverage(
        self, ylow: list[np.ndarray], yhigh: list[np.ndarray], xlim=None
    ):
        """
        Calculate the empirical coverage of the model predictions within the
        specified intervals for each observation.

        Parameters:
        ----------
        ylow : list[np.ndarray]
            Lower bounds of the intervals for each observation.
        yhigh : list[np.ndarray]
            Upper bounds of the intervals for each observation.
        xlim : tuple, optional
            Limits for the x-axis, if applicable.

        Returns:
        -------
        float
            The empirical coverage across all observations.
        """
        return self.num_pts_within_interval(ylow, yhigh, xlim) / self.n_data_pts
