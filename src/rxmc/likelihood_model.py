import scipy as sc
import numpy as np

from .observation import Observation, FixedCovarianceObservation
from .params import Parameter


class LikelihoodModel:
    """
    A class to represent a likelihood model for comparing an Observation
    to a PhysicalModel
    """

    def __init__(self):
        self.params = None
        pass

    def residual(self, observation: Observation, ym: np.ndarray):
        """
        Returns the residual between the model prediction ym and
        observation.y

        Parameters:
        ----------
        observation : Observation
            The observation object containing the observed data.
        ym : np.ndarray
            Model prediction for the observation.

        Returns
        -------
        np.ndarray
            Residual vector.
        """
        return observation.residual(ym)

    def covariance(self, observation: Observation, ym: np.ndarray):
        """
        Returns the covariance matrix determined by the likelihood model.

        Parameters
        ----------
        observation : Observation
            The observation object containing the observed data.
        ym : np.ndarray
            Model prediction for the observation.

        Returns
        -------
        np.ndarray
            Covariance matrix of the observation.
        """
        raise NotImplementedError(
            "This method should be implemented in subclasses to return the "
            "covariance matrix."
        )

    def chi2(self, observation: Observation, ym: np.ndarray):
        """
        Calculate the generalised chi-squared statistic. This is the
        Malahanobis distance between y and ym

        Parameters
        ----------
        observation : Observation
            The observation object containing the observed data.
        ym : np.ndarray
            Model prediction for the observation.

        Returns
        -------
        float
            Chi-squared statistic.
        """
        cov = self.covariance(observation, ym)
        mahalanobis, _ = mahalanobis_distance_cholesky(observation.y, ym, cov)
        return mahalanobis

    def logpdf(self, observation: Observation, ym: np.ndarray):
        """
        Returns the logpdf that ym reproduces y, given the covariance

        Parameters
        ----------
        ym : np.ndarray
            Model prediction for the observation.
        observation : Observation
            The observation object containing the observed data.

        Returns
        -------
        float
        """
        cov = self.covariance(observation, ym)
        mahalanobis, log_det = mahalanobis_distance_cholesky(observation.y, ym, cov)
        return log_likelihood(mahalanobis, log_det, observation.n_data_pts)


class FixedCovarianceLikelihood(LikelihoodModel):
    """
    A special LikelihoodModel to handle FixedCovarianceObservation objects,
    where the covariance matrix is fixed and does not depend on the
    parameters of the PhysicalModel.

    This allows for the use of precomputed inverse covariance matrices which
    can speed up the calculation of the chi-squared statistic and logpdf.
    """

    def __init__(self):
        pass

    def covariance(self, observation: FixedCovarianceObservation, ym: np.ndarray):
        """
        Returns the fixed covariance matrix in `observation`

        Parameters
        ----------
        ym : np.ndarray
            Model prediction for the observation.
        observation : FixedCovarianceObservation
            The observation object containing the observed data, which has
            attribute `covariance`.

        Returns
        -------
        np.ndarray
            Fixed covariance matrix.
        """
        return observation.covariance

    def chi2(self, observation: FixedCovarianceObservation, ym: np.ndarray):
        """
        Calculate the generalised chi-squared statistic. This is the
        Malahanobis distance between y and ym

        Parameters
        ----------
        params : OrderedDict
            parameters of model
        observation : FixedCovarianceObservation
            The observation object containing the observed data, which has
            attribute `cov_inv`.

        Returns
        -------
        float
            Chi-squared statistic.
        """
        # we overload this method to use precomputed inverse
        # covariance matrix
        delta = observation.residual(ym)
        return delta.T @ observation.cov_inv @ delta

    def logpdf(self, observation: FixedCovarianceObservation, ym: np.ndarray):
        """
        Returns the logpdf that ym reproduces y, given the fixed
        covariance matrix

        Parameters
        ----------
        params : OrderedDict
            parameters of model
        observation : FixedCovarianceObservation
            The observation object containing the observed data, which has
            attributes `cov_inv`, `n_data_pts` and `log_det`.

        Returns
        -------
        float
        """
        # we overload this method to use precomputed inverse
        # covariance matrix
        mahalanobis = self.chi2(observation, ym)
        return log_likelihood(mahalanobis, observation.log_det, observation.n_data_pts)


class LikelihoodWithSystematicError(LikelihoodModel):
    """
    LikelihoodModel in which the covariance matrix is dependent on the values
    of the PhysicalModel and its parameters, as is the case when systematic
    errors are present in the observation, following D'Agostini, G. (1993) 'On
    the use of the covariance matrix to fit correlated data'

    Note that this is equivalent to the alternative method to handle systematic
    errors described by Barlow, R (2021) 'Combining experiments with systematic
    errors', in which nuisance parameters are introduced corresponding to the
    normalization and additive bias of the observation.

    The advantage of this approach is that it does not require introducing
    nuisance parameters, but instead encodes the correlation between the data
    points in the observation in the covariance matrix directly.

    Attributes:
    ----------
    fractional_uncorrelated_error : float
        Fractional uncorrelated error in the model prediction. For example,
        if one expects the model to be correct to 1% in any given data point,
        then this should be set to 0.01. Default is 0.0.
    """

    def __init__(self, fractional_uncorrelated_error: float = 0.0):
        """
        Initializes the LikelihoodWithSystematicError, optionally with
        a fractional uncorrelated error.

        Parameters
        ----------
        fractional_uncorrelated_error : float, optional
            Fractional uncorrelated error in the model prediction. For example,
            if one expects the model to be correct to 1% in any given data
            point, then this should be set to 0.01. Default is 0.0.
        """
        self.fractional_uncorrelated_error = fractional_uncorrelated_error

    def covariance(self, observation: Observation, ym: np.ndarray):
        """
        Returns the following covariance matrix:
            \[
                \Sigma_{ij} = \sigma^2_{i}^{stat} \delta_{ij}
                            + \Sigma_{ij}^{sys}
                            + \gamma^2 y_m^2(x_i, \alpha)
            \]
        where sigma^2_{i}^{stat} is the statistical variance of the i-th
        observation, (`observation.y_stat_err`) and $\gamma$ is the
        fractional uncorrelated error (`self.fractional_uncorrelated_error`).

        Here, Sigma_{ij}^{sys} is the systematic covariance matrix:
            \[
                \Sigma_{ij}^{sys} = \eta**2 y_m(x_i, \alpha) y_m(x_j, \alpha) + \omega,
            \]
        where $\eta$ is the uncertainty in the overall normalization of the
        observation (`observation.y_sys_err_bias`) and $\omega$ is the uncertainty in
        the additive bias to the observation (`observation.y_sys_err_offset`).

        Here, also, $y_m(x_i, \alpha)$ is the model prediction for the i-th
        observation.

        Parameters
        ----------
        ym : np.ndarray
            Model prediction for the observation.
        observation : Observation
            The observation object containing the observed data.

        Returns
        -------
        np.ndarray
            Covariance matrix of the observation.
        """
        sigma_sys = systematic_covariance(
            observation.y_sys_err_bias,
            observation.y_sys_err_offset,
            ym,
        )
        sigma_model = uncorrelated_model_covariance(
            self.fractional_uncorrelated_error,
            ym,
        )
        sigma_stat = statistical_covariance(observation.y_stat_err)
        return sigma_sys + sigma_model + sigma_stat


class ParametricLikelihoodModel(LikelihoodModel):
    """
    A class to represent a likelihood model for comparing an Observation
    to a PhysicalModel, in which the LikelihoodModel has it's own parameters
    to calculate the covariance, aside from the parameters of the
    PhysicalModel. This is useful when the covariance is unknown, and one
    would like to calibrate the likelihood parameters to an observation,
    along with the parameters of a PhysicalModel.
    """

    def __init__(self, likelihood_params: list[Parameter]):
        self.params = likelihood_params
        self.n_params = len(likelihood_params)
        super().__init__()

    def chi2(self, observation: Observation, ym: np.ndarray, *likelihood_params):
        """
        Calculate the generalised chi-squared statistic. This is the
        Malahanobis distance between y and ym

        Parameters
        ----------
        observation : Observation
            The observation object containing the observed data.
        ym : np.ndarray
            Model prediction for the observation.
        likelihood_params : tuple
            Additional parameters for the covariance

        Returns
        -------
        float
            Chi-squared statistic.
        """
        assert len(likelihood_params) == self.n_params

        cov = self.covariance(observation, ym, *likelihood_params)
        mahalanobis, _ = mahalanobis_distance_cholesky(observation.y, ym, cov)
        return mahalanobis

    def logpdf(self, observation: Observation, ym: np.ndarray, *likelihood_params):
        """
        Returns the logpdf that ym reproduces y, given the covariance

        Parameters
        ----------
        ym : np.ndarray
            Model prediction for the observation.
        observation : Observation
            The observation object containing the observed data.
        likelihood_params: tuple
            Additional parameters for the covariance

        Returns
        -------
        float
        """
        assert len(likelihood_params) == self.n_params
        cov = self.covariance(observation, ym, *likelihood_params)
        mahalanobis, log_det = mahalanobis_distance_cholesky(observation.y, ym, cov)
        return log_likelihood(mahalanobis, log_det, observation.n_data_pts)

    def covariance(self, observation: Observation, ym: np.ndarray, *likelihood_params):
        """
        Returns the covariance matrix determined by the likelihood model,
        which is dependent on the parameters of the likelihood model.

        Parameters
        ----------
        observation : Observation
            The observation object containing the observed data.
        ym : np.ndarray
            Model prediction for the observation.
        likelihood_params : tuple
            Additional parameters for the covariance.

        Returns
        -------
        np.ndarray
            Covariance matrix of the observation.
        """
        raise NotImplementedError(
            "This method should be implemented in subclasses to return the "
            "covariance matrix."
        )


class UnknownNoiseErrorModel(ParametricLikelihoodModel):
    """
    A ParametricLikelihoodModel in which each data point in the observation
    has the same, unknown, statistical error, which is a parameter, $epsilon$.

    This implies the statistical contribution to the covariance takes the form:

    \[
        \Sigma_{ij}^{stat} = \epsilon^2 \delta_{ij}
    \]
    """

    def __init__(self, fractional_uncorrelated_error: float = 0):
        """
        Initializes the UnknownNoiseErrorModel, optionally with
        a fractional uncorrelated error.

        Parameters
        ----------
        fractional_uncorrelated_error : float, optional
            Fractional uncorrelated error in the model prediction. For example,
            if one expects the model to be correct to 1% in any given data
            point, then this should be set to 0.01. Default is 0.0
        """
        self.fractional_uncorrelated_error = fractional_uncorrelated_error
        likelihood_params = [
            Parameter("noise", float, latex_name=r"\epsilon"),
        ]
        super().__init__(likelihood_params)

    def covariance(self, observation: Observation, ym: np.ndarray, noise: float):
        """
        Returns the following covariance matrix:
            \[
                \Sigma_{ij} = \sigma^2_{i}^{stat} \delta_{ij}
                            + \Sigma_{ij}^{sys}
                            + \gamma^2 y_m^2(x_i, \alpha)
            \]
        where sigma^2_{i}^{stat} is the statistical variance of the i-th
        observation, which is dependent on the parameter $\epsilon$, which
        is the statistical `noise`:

        \[
            \sigma^2_{i}^{stat} = \epsilon^2 \delta_{ij}
        \]

        (note this class ignores `observation.y_stat_err`) and $\gamma$ is the
        fractional uncorrelated error (`self.fractional_uncorrelated_error`).

        Parameters
        ----------
        observation : Observation
            The observation object containing the observed data.
        ym : np.ndarray
            Model prediction for the observation.
        noise : float
            Statistical noise

        Returns
        -------
        np.ndarray
            Covariance matrix of the observation.
        """
        sigma_sys = systematic_covariance(
            observation.y_sys_err_bias, observation.y_sys_err_offset, ym
        )
        sigma_stat = statistical_covariance(np.ones_like(ym) * noise)
        sigma_model = uncorrelated_model_covariance(
            self.fractional_uncorrelated_error,
            ym,
        )
        return sigma_sys + sigma_stat + sigma_model


class UnknownNoiseFractionErrorModel(ParametricLikelihoodModel):
    """
    A ParametricLikelihoodModel in which each data point in the observation
    has the a statistical error corresponding to a fixed fraction of it's value,
    the fraction being a parameter, $epsilon$.

    This implies the statistical contribution to the covariance takes the form:

    \[
        \Sigma_{ij}^{stat} = \epsilon^2 y(x_i)^2 \delta_{ij}
    \]


    """

    def __init__(self, fractional_uncorrelated_error: float = 0):
        self.fractional_uncorrelated_error = fractional_uncorrelated_error
        likelihood_params = [
            Parameter("noise_fraction", float, latex_name=r"\epsilon"),
        ]
        super().__init__(likelihood_params)

    def covariance(
        self, observation: Observation, ym: np.ndarray, noise_fraction: float
    ):
        """
        Returns the following covariance matrix:
            \[
                \Sigma_{ij} = \sigma^2_{i}^{stat} \delta_{ij}
                            + \Sigma_{ij}^{sys}
                            + \gamma^2 y_m^2(x_i, \alpha)
            \]
        where sigma^2_{i}^{stat} is the statistical variance of the i-th
        observation, which is dependent on the parameter $\epsilon$, which
        is the statistical `noise_fraction`:

        \[
            \Sigma_{ij}^{stat} = \epsilon^2 y(x_i)^2 \delta_{ij}
        \]

        (note this class ignores `observation.y_stat_err`) and $\gamma$ is the
        fractional uncorrelated error (`self.fractional_uncorrelated_error`).

        Parameters
        ----------
        observation : Observation
            The observation object containing the observed data.
        ym : np.ndarray
            Model prediction for the observation.
        noise_fraction : float
            Statistical noise as a fraction of `observation.y`

        Returns
        -------
        np.ndarray
            Covariance matrix of the observation.
        """
        sigma_sys = systematic_covariance(
            observation.y_sys_err_bias, observation.y_sys_err_offset, ym
        )
        sigma_stat = statistical_covariance(ym * noise_fraction)
        sigma_model = uncorrelated_model_covariance(
            self.fractional_uncorrelated_error,
            ym,
        )
        return sigma_sys + sigma_stat + sigma_model


class UnknownBiasErrorModel(ParametricLikelihoodModel):
    """
    A ParametricLikelihoodModel in which the systematic bias in normalization
    of the observation is a parameter, $\eta$.
    """

    def __init__(self, fractional_uncorrelated_error: float = 0):
        self.fractional_uncorrelated_error = fractional_uncorrelated_error
        likelihood_params = [
            Parameter("y_sys_err_bias", float, latex_name=r"\eta"),
        ]
        super().__init__(likelihood_params)

    def covariance(
        self, observation: Observation, ym: np.ndarray, y_sys_err_bias: float
    ):
        """
        Returns the following covariance matrix:
            \[
                \Sigma_{ij} = \sigma^2_{i}^{stat} \delta_{ij}
                            + \Sigma_{ij}^{sys}
                            + \gamma^2 y_m^2(x_i, \alpha)
            \]
        where sigma^2_{i}^{stat} is the statistical variance of the i-th
        observation, (`observation.y_stat_err`) and $\gamma$ is the
        fractional uncorrelated error (`self.fractional_uncorrelated_error`).

        Here, Sigma_{ij}^{sys} is the systematic covariance matrix:
            \[
                \Sigma_{ij}^{sys} = \eta**2 y_m(x_i, \alpha) y_m(x_j, \alpha) + \omega,
            \]
        where $\eta$ is the uncertainty in the overall normalization of the
        observation (`y_sys_err_bias` - in this case, this value is a parameter,
        and  corresponding value in `observation` is ignored) and $\omega$ is
        the uncertainty in the additive bias to the observation
        (`observation.y_sys_err_offset`).

        Here, also, $y_m(x_i, \alpha)$ is the model prediction for the i-th
        observation.

        Parameters
        ----------
        observation : Observation
            The observation object containing the observed data.
        ym : np.ndarray
            Model prediction for the observation.
        y_sys_err_bias: float
            Uncertainty in the overall normalization of the observation.

        Returns
        -------
        np.ndarray
            Covariance matrix of the observation.
        """
        sigma_sys = systematic_covariance(
            y_sys_err_bias, observation.y_sys_err_offset, ym
        )
        sigma_stat = statistical_covariance(observation.y_stat_err)
        sigma_model = uncorrelated_model_covariance(
            self.fractional_uncorrelated_error,
            ym,
        )
        return sigma_sys + sigma_stat + sigma_model


def mahalanobis_distance_cholesky(y, ym, cov):
    """
    Calculate the Mahalanobis distance between y and ym, and the
    log determinant of the covariance matrix.

    Parameters:
    y (array-like): The observation vector.
    ym (array-like): The model prediction vector.
    cov (array-like): The covariance matrix.

    Returns:
    tuple: Mahalanobis distance and log determinant of the covariance matrix.
    """
    L = sc.linalg.cholesky(cov, lower=True)
    z = sc.linalg.solve_triangular(L, y - ym, lower=True)
    mahalanobis = np.dot(z, z)
    log_det = 2 * np.sum(np.log(np.diag(L)))

    return mahalanobis, log_det


def log_likelihood(mahalanobis: float, log_det: float, n: int):
    """
    Calculate the log likelihood of a multivariate normal distribution.

    Parameters:
    mahalanobis (float): The Mahalanobis distance.
    log_det (float): The log determinant of the covariance matrix.
    n (int): The dimension of the data.

    Returns:
    float: The log likelihood value.
    """
    return -0.5 * (mahalanobis + log_det + n * np.log(2 * np.pi))


def systematic_covariance(
    y_sys_err_bias: float, y_sys_err_offset: float, ym: np.ndarray
):
    """
    Returns the systematic covariance matrix:
        \[
            \Sigma_{ij}^{sys} = \eta**2 y_m(x_i, \alpha) y_m(x_j, \alpha) + \omega,
        \]
    where $\eta$ is the uncertainty in the overal normalization of the
    observation (`observation.y_sys_err_bias`) and $\omega$ is the uncertainty in
    the additive bias to the observation (`observation.y_sys_err_offset`).

    Here, $y_m(x_i, \alpha)$ is the model prediction for the i-th
    observation.

    See Barlow, R (2021) 'Combining experiments with systematic errors'

    Parameters
    ----------
    y_sys_err_bias : float
        Uncertainty in the overall normalization of the observation.
    y_sys_err_offset : float
        Uncertainty in the additive bias to the observation.
    ym : np.ndarray
        Model prediction for the observation.

    Returns
    -------
    np.ndarray
        Systematic covariance matrix.
    """
    eta = y_sys_err_bias
    omega = y_sys_err_offset
    covariance_bias = eta**2 * np.outer(ym, ym)
    covariance_offset = omega**2 * np.ones_like(covariance_bias)
    return covariance_bias + covariance_offset


def statistical_covariance(y_stat_err: np.ndarray):
    """
    Returns the statistical covariance matrix:
        \[
            \Sigma_{ij}^{stat} = \sigma^2_{i}^{stat} \delta_{ij}
        \]
    where $\sigma^2_{i}^{stat}$ is the statistical variance of the i-th
    observation (`observation.y_stat_err`).

    Parameters
    ----------
    y_stat_err : np.ndarray
        Statistical errors for the observation.

    Returns
    -------
    np.ndarray
        Statistical covariance matrix.
    """
    return np.diag(y_stat_err**2)


def uncorrelated_model_covariance(fractional_uncorrelated_error: float, ym: np.ndarray):
    """
    Returns the uncorrelated model covariance matrix:
        \[
            \Sigma_{ij}^{uncorrelated} = \gamma^2 y_m(x_i, \alpha)^2 \delta_{ij}
        \]
    where $\gamma$ is the fractional uncorrelated error
    (`self.fractional_uncorrelated_error`).

    This is commonly used as a model-error term or unquantified uncertainty
    term. E.g. if one expects the model to be correct to 1% in any given data
    point, then this should be set to 0.01.

    It should be noted that this assumption ignores correlations between data
    points in the model.

    Parameters
    ----------
    fractional_uncorrelated_error : float
        Fractional uncorrelated error in the model prediction.
    ym : np.ndarray
        Model prediction for the observation.

    Returns
    -------
    np.ndarray
        Uncorrelated model error covariance matrix.
    """
    gamma = fractional_uncorrelated_error
    return gamma**2 * np.diag(ym**2)
