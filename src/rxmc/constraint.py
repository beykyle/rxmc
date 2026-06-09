"""
Constraint: the composition of observations, a physical model, and a likelihood.

A :class:`Constraint` pairs one or more :class:`~rxmc.observation.Observation`
objects with a :class:`~rxmc.physical_model.PhysicalModel` and a
:class:`~rxmc.likelihood_model.LikelihoodModel`.  Given model parameters it
computes the log likelihood, chi-squared statistic, and coverage statistics.
"""

import numpy as np

from .likelihood_model import LikelihoodModel
from .observation import Observation
from .physical_model import PhysicalModel


class Constraint:
    """Pair observations with a physical model and a likelihood model.

    A ``Constraint`` is the composition of one or more :class:`~rxmc.observation.Observation`
    objects with a :class:`~rxmc.physical_model.PhysicalModel` and a
    :class:`~rxmc.likelihood_model.LikelihoodModel`.  It acts as a box that
    accepts model parameters and returns the log likelihood or other statistics,
    acting as a constraint on those parameters.

    Parameters
    ----------
    observations : list of Observation
        The observed data that the model will attempt to reproduce.
    physical_model : PhysicalModel
        Model that predicts the observed data.
    likelihood_model : LikelihoodModel
        Model that defines the likelihood of the observations given the
        physical-model predictions.
    """

    def __init__(
        self,
        observations: list[Observation],
        physical_model: PhysicalModel,
        likelihood_model: LikelihoodModel,
    ):
        self.observations = observations
        self.physical_model = physical_model
        self.likelihood = likelihood_model
        self.n_data_pts = sum(obs.n_data_pts for obs in self.observations)

    def model(self, model_params):
        """Compute the physical model output for each observation.

        Parameters
        ----------
        model_params : tuple
            Parameters of the physical model.

        Returns
        -------
        list of np.ndarray
            Model predictions, one array per observation.
        """
        return [self.physical_model(obs, *model_params) for obs in self.observations]

    def log_likelihood(self, model_params, likelihood_params=()):
        """Total log likelihood over all observations.

        Parameters
        ----------
        model_params : tuple
            Parameters of the physical model.
        likelihood_params : tuple, optional
            Additional parameters for the likelihood model.

        Returns
        -------
        float
            Sum of log likelihoods across all observations.
        """
        return sum(
            self.likelihood.log_likelihood(
                obs, self.physical_model(obs, *model_params), *likelihood_params
            )
            for obs in self.observations
        )

    def marginal_log_likelihood(self, ym: list, *likelihood_params):
        """Log likelihood given pre-computed model predictions.

        Parameters
        ----------
        ym : list of np.ndarray
            Pre-computed model predictions for each observation.
        *likelihood_params : float
            Additional parameters for the likelihood model.

        Returns
        -------
        float
            Sum of log likelihoods across all observations.
        """
        return sum(
            self.likelihood.log_likelihood(obs, y, *likelihood_params)
            for obs, y in zip(self.observations, ym)
        )

    def chi2(self, model_params, likelihood_params=()):
        """Generalised chi-squared (Mahalanobis distance) summed over observations.

        Parameters
        ----------
        model_params : tuple
            Parameters of the physical model.
        likelihood_params : tuple, optional
            Additional parameters for the likelihood model.

        Returns
        -------
        float
            Total chi-squared statistic.
        """
        return sum(
            self.likelihood.chi2(
                obs, self.physical_model(obs, *model_params), *likelihood_params
            )
            for obs in self.observations
        )

    def predict(self, *model_params):
        """Generate predictions for each observation.

        Parameters
        ----------
        *model_params : float
            Parameters of the physical model.

        Returns
        -------
        list of np.ndarray
            Predicted values for each observation.
        """
        return [self.physical_model(obs, *model_params) for obs in self.observations]

    def num_pts_within_interval(
        self, ylow: list[np.ndarray], yhigh: list[np.ndarray], xlim=None
    ):
        """Count data points that fall within a predictive interval.

        Parameters
        ----------
        ylow : list of np.ndarray
            Lower bounds of the interval for each observation.
        yhigh : list of np.ndarray
            Upper bounds of the interval for each observation.
        xlim : tuple, optional
            ``(x_min, x_max)`` range to restrict the count.

        Returns
        -------
        int
            Total number of points within the interval across all observations.
        """
        return sum(
            obs.num_pts_within_interval(ylow[i], yhigh[i], xlim)
            for i, obs in enumerate(self.observations)
        )

    def empirical_coverage(
        self, ylow: list[np.ndarray], yhigh: list[np.ndarray], xlim=None
    ):
        """Fraction of data points within a predictive interval.

        Parameters
        ----------
        ylow : list of np.ndarray
            Lower bounds of the interval for each observation.
        yhigh : list of np.ndarray
            Upper bounds of the interval for each observation.
        xlim : tuple, optional
            ``(x_min, x_max)`` range to restrict the count.

        Returns
        -------
        float
            Empirical coverage fraction in ``[0, 1]``.
        """
        return self.num_pts_within_interval(ylow, yhigh, xlim) / self.n_data_pts
