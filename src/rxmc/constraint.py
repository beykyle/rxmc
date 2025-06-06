from .physical_model import PhysicalModel
from .likelihood_model import LikelihoodModel


class Constraint:
    """
    A `Constraint` is the composition of an `Observation` with a `PhysicalModel`
    that is able to make predictions of the observed data, along with a
    LikelihoodModel which defines the likelihood of the observation given
    the model predictions.
    """

    def __init__(
        self, physical_model: PhysicalModel, likelihood_model: LikelihoodModel
    ):
        """
        Initialize the Constraint with an Observation, PhysicalModel, and LikelihoodModel.

        Parameters:
        ----------
        physical_model : PhysicalModel
            The model that predicts the observed data, which contains an
            `observation` attribute.
        likelihood_model : LikelihoodModel
            The model that defines the likelihood of the observation given the
            physical model prediction.
        """
        self.observation = physical_model.observation
        self.model = physical_model
        self.likelihood = likelihood_model

    def logpdf(self, params):
        """
        Calculate the log probability density function (logpdf) that the model
        predictions, given the parameters, reproduce the observed data.

        Parameters:
        ----------
        params : OrderedDict or np.ndarray
            The parameters of the physical model

        Returns:
        -------
        float
            The log probability density of the observation given the parameters.
        """
        ym = self.model(params)
        return self.likelihood.logpdf(self.observation, ym)

    def chi2(self, params):
        """
        Calculate the chi-squared statistic (or Mahalanobis distance) between
        the model prediction, given the parameters, and the observed data.

        Parameters:
        ----------
        params : OrderedDict or np.ndarray
            The parameters of the physical model

        Returns:
        -------
        float
            The chi-squared statistic.
        """
        ym = self.model(params)
        return self.likelihood.chi2(self.observation, ym)
