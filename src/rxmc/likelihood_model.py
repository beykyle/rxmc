import numpy as np
import scipy as sc

from .observation import FixedCovarianceObservation, Observation
from .params import Parameter


class LikelihoodModel:
    """
    A class to represent a likelihood model for comparing an Observation
    to a PhysicalModel.

    The default behavior uses the following covariance matrix:
        \[
            \Sigma_{ij} = \sigma^2_{i}^{stat} \delta_{ij}
                        + \Sigma_{ij}^{sys}
                        + \gamma^2 y_m^2(x_i, \alpha)
        \]
    where $sigma^2_{i}^{stat}$ is the statistical variance of the i-th
    observation, (`observation.statistical_covariance`) and $\gamma$ is the
    fractional uncorrelated error (`self.frac_err`).

    Here, $Sigma_{ij}^{sys}$ is the systematic covariance matrix:
        \[
            \Sigma_{ij}^{sys} = \eta**2 y_m(x_i, \alpha) y_m(x_j, \alpha) + \omega,
        \]
    where $\eta$ is the uncertainty in the overall normalization of the
    observation (`observation.y_sys_err_normalization`) and $\omega$ is the
    uncertainty in the additive normalization to the observation
    (`observation.y_sys_err_offset`).

    Here, also, $y_m(x_i, \alpha)$ is the model prediction for the i-th
    observation. Thus the covariance matrix is dependent on the values
    of the PhysicalModel and its parameters, as is the case when systematic
    errors are present in the observation, following D'Agostini, G. (1993) 'On
    the use of the covariance matrix to fit correlated data'

    Note that if there is no systematic uncertainty encoded in the `observation`
    and `self.frac_err` takes the default value of 0, the
    covariance matrix becomes diaginal, and the `chi2` function reduces to the
    simple and familiar $\chi^2$ form.

    Note also that this is equivalent to the alternative method to handle systematic
    errors described by Barlow, R (2021) 'Combining experiments with systematic
    errors', in which nuisance parameters are introduced corresponding to the
    normalization and additive offset bias of the observation.

    The advantage of this approach is that it does not require introducing
    nuisance parameters, but instead encodes the correlation between the data
    points in the observation in the covariance matrix directly.
    """

    def __init__(
        self,
        frac_err: float = 0.0,
        divide_by_N: bool = False,
        covariance_scale: float = 1.0,
    ):
        """
        Initializes the LikelihoodModel, optionally with a fractional
        uncorrelated error.

        Parameters
        ----------
        frac_err : float
            Fractional uncorrelated error in the model prediction. For example,
            if one expects the model to be correct to 1% in any given data point,
            then this should be set to 0.01. This is sometimes used to represent
            the uncorrelated uncertainty in the model prediction. Default is 0.0.
        divide_by_N : bool = False
            Divide the covariance matrix by the number of data points in the
            observation?
        scale : float = 1.0
            Arbitrary fixed scale constant for covariance matrix
        """
        self.frac_err = frac_err
        self.params = None
        self.n_params = 0
        self.divide_by_N = divide_by_N
        self.covariance_scale = covariance_scale

    def covariance(self, observation: Observation, ym: np.ndarray):
        """
        Default covariance model. Derived classes of `LikelihoodModel` will
        override this.

        Returns the following covariance matrix:
            \[
                \Sigma_{ij} = \sigma^2_{i}^{stat} \delta_{ij}
                            + \Sigma_{ij}^{sys}
                            + \gamma^2 y_m^2(x_i, \alpha)
            \]
        where $sigma^2_{i}^{stat}$ is the statistical variance of the i-th
        observation, (`observation.statistical_covariance`) and $\gamma$ is the
        fractional uncorrelated error (`self.frac_err`).

        Here, $Sigma_{ij}^{sys}$ is the systematic covariance matrix:
            \[
                \Sigma_{ij}^{sys} = \eta**2 y_m(x_i, \alpha) y_m(x_j, \alpha) + \omega,
            \]
        where $\eta$ is the uncertainty in the overall normalization of the
        observation (`observation.y_sys_err_normalization`) and $\omega$ is the
        uncertainty in the additive normalization to the observation
        (`observation.y_sys_err_offset`).

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
        sigma = observation.covariance(ym)
        sigma_model = uncorrelated_model_covariance(
            self.frac_err,
            ym,
        )
        cov = sigma + sigma_model
        return scale_covariance(
            cov, observation, self.covariance_scale, self.divide_by_N
        )

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

    def chi2(self, observation: Observation, ym: np.ndarray):
        """
        Calculate the generalised chi-squared statistic. This is the
        square of the Mahalanobis distance between y and ym

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
        mahalanobis_sqr, _ = mahalanobis_distance_sqr_cholesky(observation.y, ym, cov)
        return mahalanobis_sqr

    def log_likelihood(self, observation: Observation, ym: np.ndarray):
        """
        Returns the log_likelihood that ym reproduces y, given the covariance

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
        mahalanobis_sqr, log_det = mahalanobis_distance_sqr_cholesky(
            observation.y, ym, cov
        )
        return log_likelihood(mahalanobis_sqr, log_det, observation.n_data_pts)


class FixedCovarianceLikelihood(LikelihoodModel):
    """
    A special LikelihoodModel to handle FixedCovarianceObservation objects,
    where the covariance matrix is fixed and does not depend on the
    parameters of the PhysicalModel.

    This allows for the use of precomputed inverse covariance matrices which
    can speed up the calculation of the chi-squared statistic and log_likelihood.
    """

    def __init__(self):
        super().__init__()

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
        return observation.cov

    def chi2(self, observation: FixedCovarianceObservation, ym: np.ndarray):
        """
        Calculate the generalised chi-squared statistic. This is the
        Mahalanobis distance between y and ym

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

    def log_likelihood(self, observation: FixedCovarianceObservation, ym: np.ndarray):
        """
        Returns the log_likelihood that ym reproduces y, given the fixed
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
        mahalanobis_sqr = self.chi2(observation, ym)
        return log_likelihood(
            mahalanobis_sqr, observation.log_det, observation.n_data_pts
        )


class ParametricLikelihoodModel(LikelihoodModel):
    """
    A class to represent a likelihood model for comparing an `Observation`
    to a `PhysicalModel`, in which the `LikelihoodModel` has it's own parameters
    to calculate the covariance, aside from the parameters of the
    `PhysicalModel`. This is useful when the covariance is unknown, and one
    would like to calibrate the likelihood parameters to an `Observation`,
    along with the parameters of a `PhysicalModel`.
    """

    def __init__(
        self,
        likelihood_params: list[Parameter],
        frac_err: float = 0.0,
        divide_by_N: bool = False,
        covariance_scale: float = 1.0,
    ):
        super().__init__(
            frac_err=frac_err,
            divide_by_N=divide_by_N,
            covariance_scale=covariance_scale,
        )
        self.params = likelihood_params
        self.n_params = len(likelihood_params)

    def chi2(self, observation: Observation, ym: np.ndarray, *likelihood_params):
        """
        Calculate the generalised chi-squared statistic. This is the
        Mahalanobis distance between `Observation.y` and `ym`.

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
        mahalanobis_sqr, _ = mahalanobis_distance_sqr_cholesky(observation.y, ym, cov)
        return mahalanobis_sqr

    def log_likelihood(
        self, observation: Observation, ym: np.ndarray, *likelihood_params
    ):
        """
        Returns the log likelihood that `ym` reproduces `observation.y`

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
        mahalanobis_sqr, log_det = mahalanobis_distance_sqr_cholesky(
            observation.y, ym, cov
        )
        return log_likelihood(mahalanobis_sqr, log_det, observation.n_data_pts)

    def covariance(self, observation: Observation, ym: np.ndarray, *likelihood_params):
        """
        Returns the covariance matrix determined by the likelihood model,
        which is dependent on `likelihood_params`

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

    No matter the Observation, the statistical contribution to the covariance
    thus always takes the form:

    \[
        \Sigma_{ij}^{stat} = \epsilon^2 \delta_{ij}
    \]

    where $\epsilon$ is the statistical `noise` parameter.
    """

    def __init__(
        self,
        frac_err: float = 0,
        divide_by_N: bool = False,
        covariance_scale: float = 1.0,
    ):
        """
        Initializes the UnknownNoiseErrorModel, optionally with
        a fractional uncorrelated error.

        Parameters
        ----------
        frac_err : float, optional
            Fractional uncorrelated error in the model prediction. For example,
            if one expects the model to be correct to 1% in any given data
            point, then this should be set to 0.01. Default is 0.0
        """
        likelihood_params = [
            Parameter("log noise", float, latex_name=r"\log{\epsilon}"),
        ]
        super().__init__(
            likelihood_params,
            frac_err=frac_err,
            divide_by_N=divide_by_N,
            covariance_scale=covariance_scale,
        )

    def covariance(self, observation: Observation, ym: np.ndarray, log_epsilon: float):
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

        (note this class ignores `observation.statistical_covariance`,
        replacing it with $\epsilon$).

        Parameters
        ----------
        observation : Observation
            The observation object containing the observed data.
        ym : np.ndarray
            Model prediction for the observation.
        log_epsilon : float
            natural log of the statistical noise, $\epsilon$

        Returns
        -------
        np.ndarray
            Covariance matrix of the observation.
        """
        sigma_sys = (
            observation.systematic_offset_covariance
            + observation.systematic_normalization_covariance * np.outer(ym, ym)
        )
        sigma_model = uncorrelated_model_covariance(
            self.frac_err,
            ym,
        )
        sigma_stat = statistical_covariance(np.ones_like(ym) * np.exp(log_epsilon))
        cov = sigma_sys + sigma_model + sigma_stat
        return scale_covariance(
            cov, observation, self.covariance_scale, self.divide_by_N
        )


class UnknownNoiseFractionErrorModel(ParametricLikelihoodModel):
    """
    A `ParametricLikelihoodModel` in which each data point in the observation
    has the a statistical error corresponding to a fixed fraction of it's value,
    the fraction being a parameter, $epsilon$.

    This implies the statistical contribution to the covariance takes the form:

    \[
        \Sigma_{ij}^{stat} = \epsilon^2 y(x_i)^2 \delta_{ij}
    \]


    """

    def __init__(
        self,
        frac_err: float = 0,
        divide_by_N: bool = False,
        covariance_scale: float = 1.0,
    ):
        likelihood_params = [
            Parameter(
                "log noise fraction",
                float,
                latex_name=r"\log{\epsilon}",
                unit="dimensionless",
            ),
        ]
        super().__init__(
            likelihood_params,
            frac_err=frac_err,
            divide_by_N=divide_by_N,
            covariance_scale=covariance_scale,
        )

    def covariance(self, observation: Observation, ym: np.ndarray, log_epsilon: float):
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

        (note this class ignores `observation.statistical_covariance`, substituting it with
        the variable `noise_fraction` multiplied by `ym`) and $\gamma$ is the
        fractional uncorrelated error (`self.frac_err`).

        Parameters
        ----------
        observation : Observation
            The observation object containing the observed data.
        ym : np.ndarray
            Model prediction for the observation.
        log_epsilon : float
            natural log of statistical noise as a fraction of `observation.y`

        Returns
        -------
        np.ndarray
            Covariance matrix of the observation.
        """
        sigma_sys = (
            observation.systematic_offset_covariance
            + observation.systematic_normalization_covariance * np.outer(ym, ym)
        )
        sigma_model = uncorrelated_model_covariance(
            self.frac_err,
            ym,
        )
        sigma_stat = statistical_covariance(ym * np.exp(log_epsilon))
        cov = sigma_sys + sigma_model + sigma_stat
        return scale_covariance(
            cov, observation, self.covariance_scale, self.divide_by_N
        )


class UnknownNormalizationErrorModel(ParametricLikelihoodModel):
    """
    A ParametricLikelihoodModel in which the systematic uncertainty
    of the normalization of the observation is a parameter, $\eta$.

    This implies the systematic normalization contribution to the
    covariance takes the form:

        \[
            \Sigma_{ij}^{sys norm} = \eta**2 y_m(x_i, \alpha) y_m(x_j, \alpha)
        \]

    where $\eta$ is a free parameter.
    """

    def __init__(
        self,
        frac_err: float = 0,
        divide_by_N: bool = False,
        covariance_scale: float = 1.0,
    ):
        self.frac_err = frac_err
        likelihood_params = [
            Parameter(
                "log normalization error",
                float,
                latex_name=r"\log{\eta}",
                unit="dimensionless",
            ),
        ]
        super().__init__(
            likelihood_params,
            frac_err=frac_err,
            divide_by_N=divide_by_N,
            covariance_scale=covariance_scale,
        )

    def covariance(self, observation: Observation, ym: np.ndarray, log_eta: float):
        """
        Returns the following covariance matrix:
            \[
                \Sigma_{ij} = \sigma^2_{i}^{stat} \delta_{ij}
                            + \Sigma_{ij}^{sys}
                            + \gamma^2 y_m^2(x_i, \alpha)
            \]
        where $sigma^2_{i}^{stat}$ is the statistical variance of the i-th
        observation, (`observation.statistical_covariance`) and $\gamma$ is the
        fractional uncorrelated error (`self.frac_err`).

        Here, $Sigma_{ij}^{sys}$ is the systematic covariance matrix:
            \[
                \Sigma_{ij}^{sys} = \eta**2 y_m(x_i, \alpha) y_m(x_j, \alpha) + \omega,
            \]
        where $\eta$ is the uncertainty in the overall normalization of the
        observation (`y_sys_err_normalization` - in this case, this value is a parameter,
        and  corresponding value in `observation` is ignored) and $\omega$ is
        the uncertainty in the additive normalization to the observation
        (`observation.y_sys_err_offset`).

        Here, also, $y_m(x_i, \alpha)$ is the model prediction for the i-th
        observation.

        Parameters
        ----------
        observation : Observation
            The observation object containing the observed data.
        ym : np.ndarray
            Model prediction for the observation.
        log_eta: float
            natual log of the uncertainty in the overall normalization
            of the observation.

        Returns
        -------
        np.ndarray
            Covariance matrix of the observation.
        """
        sigma_sys = observation.systematic_offset_covariance + np.exp(
            log_eta
        ) ** 2 * np.outer(ym, ym)
        sigma_model = uncorrelated_model_covariance(
            self.frac_err,
            ym,
        )
        sigma_stat = observation.statistical_covariance
        cov = sigma_sys + sigma_model + sigma_stat
        return scale_covariance(
            cov, observation, self.covariance_scale, self.divide_by_N
        )


class UnknownModelError(ParametricLikelihoodModel):
    """
    A `ParametricLikelihoodModel` in which the `frac_err`
    is a free parameter $\gamma$, such that the covariance due to the
    uncorrelated model error takes the form:

    \[
        \Sigma_{ij}^{uncorrelated} = \gamma^2 y_m(x_i, \alpha)^2 \delta_{ij}
    \]

    where $\gamma$ is a free parameter.

    This is commonly used as a model-error term or unquantified uncertainty.
    """

    def __init__(
        self,
        divide_by_N: bool = False,
        covariance_scale: float = 1.0,
    ):
        likelihood_params = [
            Parameter(
                "frac_err",
                float,
                latex_name=r"\gamma",
                unit="dimensionless",
            ),
        ]
        super().__init__(
            likelihood_params,
            frac_err=0.0,  # this model does not use frac_err
            divide_by_N=divide_by_N,
            covariance_scale=covariance_scale,
        )

    def covariance(
        self,
        observation: Observation,
        ym: np.ndarray,
        frac_err: float,
    ):
        """
        Default covariance model. Derived classes of `LikelihoodModel` will
        override this.

        Returns the following covariance matrix:
            \[
                \Sigma_{ij} = \sigma^2_{i}^{stat} \delta_{ij}
                            + \Sigma_{ij}^{sys}
                            + \gamma^2 y_m^2(x_i, \alpha)
            \]
        where $sigma^2_{i}^{stat}$ is $\gamma$ is the fractional uncorrelated error
        (`frac_err`), treated here as a free parameter, and
        all other definitions are the same as `LikelihoodModel.covariance`


        Parameters
        ----------
        ym : np.ndarray
            Model prediction for the observation.
        observation : Observation
            The observation object containing the observed data.
        frac_err: float
            The fraction of the model prediction at point x_i that
            is treated as the standard deviation of the model prediction
            at that point, such that the model prediction is independent
            at every point.

        Returns
        -------
        np.ndarray
            Covariance matrix of the observation.
        """
        sigma = observation.covariance(ym)
        sigma_model = uncorrelated_model_covariance(
            frac_err,
            ym,
        )
        cov = sigma + sigma_model
        return scale_covariance(
            cov, observation, self.covariance_scale, self.divide_by_N
        )


class CorrelatedDiscrepancyModel(ParametricLikelihoodModel):
    """
    A `ParametricLikelihoodModel` in which the systematic uncertainty
    of the observation is a structured discrepancy, which is a function
    of the model prediction, $y_m(x_i, \alpha)$, and the parameters
    of the model, $\eta$ and $\ell$, such that
    \[
        \Sigma_{ij}^{sys} = K_{ij} + \omega
    \]
    where $K_{ij}$ is the Radial Basis Function (RBF) kernel
    """

    def __init__(self, divide_by_N=False, covariance_scale=1.0):
        likelihood_params = [
            Parameter("discrepancy_amplitude", float, latex_name=r"\eta"),
            Parameter("discrepancy_lengthscale", float, latex_name=r"\ell"),
            Parameter("model_error_fraction", float, latex_name=r"\gamma"),
        ]
        super().__init__(
            likelihood_params,
            divide_by_N=divide_by_N,
            covariance_scale=covariance_scale,
        )

    def covariance(
        self,
        observation: Observation,
        ym: np.ndarray,
        eta,
        lengthscale,
        model_error_fraction,
    ):
        """
        Returns the covariance matrix for the observation, which includes
        a structured discrepancy term based on the RBF kernel.
        Parameters
        ----------
        observation : Observation
            The observation object containing the observed data.
        ym : np.ndarray
            Model prediction for the observation.
        eta : float
            Amplitude of the kernel.
        lengthscale : float
            Length scale of the kernel.
        model_error_fraction : float
            Fractional uncorrelated error in the model prediction.
        Returns
        -------
        np.ndarray
            Covariance matrix of the observation.
        """

        sigma_stat = observation.statistical_covariance
        sigma_sys = (
            observation.systematic_offset_covariance
            + observation.systematic_normalization_covariance * np.outer(ym, ym)
        )
        sigma_model = uncorrelated_model_covariance(model_error_fraction, ym)

        # Structured discrepancy term
        K = rbf_kernel(observation.x, eta, lengthscale)

        cov = sigma_stat + sigma_sys + sigma_model + K
        return scale_covariance(
            cov, observation, self.covariance_scale, self.divide_by_N
        )


class UnknownNormalizationModel(ParametricLikelihoodModel):
    """
    A `ParametricLikelihoodModel` in which the normalization of the model is
    a free parameter, $N$. In this case, the covariance does not include
    systematic errors due to the normalization, as the model predictions
    are explicitly re-normalized by the parameter $N$.
    """

    def __init__(self, divide_by_N=False, covariance_scale=1.0):
        likelihood_params = [
            Parameter(
                "normalization",
                float,
                latex_name=r"N",
                unit="dimensionless",
            ),
        ]
        super().__init__(
            likelihood_params,
            divide_by_N=divide_by_N,
            covariance_scale=covariance_scale,
        )

    def covariance(self, observation: Observation, ym: np.ndarray):
        """
        Returns the covariance matrix for the observation, which discludes
        off-diagonal systematic errors due to the normalization, as the
        model predictions are explicitly re-normalized.

        Parameters
        ----------
        observation : Observation
            The observation object containing the observed data.
        ym : np.ndarray
            Model prediction for the observation, which is expected to be
            normalized by `N`.
        """
        sigma_sys = observation.systematic_offset_covariance
        sigma_model = uncorrelated_model_covariance(
            self.frac_err,
            ym,
        )
        sigma_stat = observation.statistical_covariance
        cov = sigma_sys + sigma_model + sigma_stat
        return scale_covariance(
            cov, observation, self.covariance_scale, self.divide_by_N
        )

    def residual(self, observation: Observation, ym: np.ndarray, normalization: float):
        """
        Returns the residual between the model prediction ym and
        observation.y scaled by the normalization parameter.

        Parameters:
        ----------
        observation : Observation
            The observation object containing the observed data.
        ym : np.ndarray
            Model prediction for the observation.
        normalization : float
            Normalization parameter to scale the model prediction.

        Returns
        -------
        np.ndarray
            Residual vector scaled by the normalization.
        """
        return observation.residual(ym / normalization)

    def log_likelihood(
        self, observation: Observation, ym: np.ndarray, normalization: float
    ):
        """
        Returns the log likelihood that ym reproduces y, given the covariance
        and normalization parameter.

        Parameters
        ----------
        ym : np.ndarray
            Model prediction for the observation.
        observation : Observation
            The observation object containing the observed data.
        normalization : float
            Normalization parameter to scale the model prediction.

        Returns
        -------
            float
        """
        cov = self.covariance(observation, ym / normalization)
        mahalanobis_sqr, log_det = mahalanobis_distance_sqr_cholesky(
            observation.y, ym / normalization, cov
        )
        return log_likelihood(mahalanobis_sqr, log_det, observation.n_data_pts)

    def chi2(self, observation: Observation, ym: np.ndarray, normalization: float):
        """
        Calculate the generalised chi-squared statistic. This is the
        Mahalanobis distance between y and ym scaled by the normalization.

        Parameters
        ----------
        observation : Observation
            The observation object containing the observed data.
        ym : np.ndarray
            Model prediction for the observation.
        normalization : float
            Normalization parameter to scale the model prediction.

        Returns
        -------
        float
            Chi-squared statistic.
        """
        cov = self.covariance(observation, ym / normalization)
        mahalanobis_sqr, _ = mahalanobis_distance_sqr_cholesky(
            observation.y, ym / normalization, cov
        )
        return mahalanobis_sqr


def rbf_kernel(x: np.ndarray, eta: float, length_scale: float) -> np.ndarray:
    """
    Radial Basis Function (RBF) kernel, also known as Gaussian kernel.
    Computes the covariance matrix for a given set of points.
    Parameters
    ----------
    x : np.ndarray
        Input data points, shape (n_samples, n_features).
    eta : float
        Amplitude of the kernel.
    length_scale : float
        Length scale of the kernel.
    Returns
    -------
    np.ndarray
        Covariance matrix computed using the RBF kernel.
    """
    x = np.atleast_2d(x).T
    sqdist = np.sum((x - x.T) ** 2, axis=0)
    return eta**2 * np.exp(-0.5 * sqdist / length_scale**2)


class StudentTLikelihoodModel(LikelihoodModel):
    """
    A `LikelihoodModel` that uses a Student's t-distribution for the likelihood.
    This is useful when the data has heavy tails or outliers, as it is more robust
    to deviations from normality compared to the Gaussian likelihood.
    """

    def __init__(self, nu: float = 1.0, frac_err: float = 0.0):
        """
        Initializes the StudentTLikelihoodModel with a specified degrees of freedom.

        Parameters
        ----------
        nu : float
            Degrees of freedom for the Student's t-distribution.
        frac_err : float
            Fractional uncorrelated error in the model prediction.
        """
        super().__init__(frac_err=frac_err)
        self.nu = nu

    def log_likelihood(self, observation: Observation, ym: np.ndarray):
        """
        Calculate the log likelihood using the Student's t-distribution.

        Parameters
        ----------
        observation : Observation
            The observation object containing the observed data.
        ym : np.ndarray
            Model prediction for the observation.

        Returns
        -------
        float
            Log likelihood value.
        """
        cov = self.covariance(observation, ym)
        mahalanobis_sqr, log_det = mahalanobis_distance_sqr_cholesky(
            observation.y, ym, cov
        )

        # Log likelihood for Student's t-distribution
        n = observation.n_data_pts
        return (
            sc.special.gammaln((n + self.nu) / 2)
            - sc.special.gammaln(self.nu / 2)
            - 0.5 * n * np.log(np.pi * self.nu)
            - 0.5 * log_det
            - (self.nu + n) / 2 * np.log(1 + mahalanobis_sqr / self.nu)
        )


def scale_covariance(
    cov: np.ndarray, observation: Observation, scale: float, divide_by_N: bool
) -> np.ndarray:
    if divide_by_N:
        scale /= observation.n_data_pts
    return scale * cov


def mahalanobis_distance_sqr_cholesky(y, ym, cov):
    """
    Calculate the square of the Mahalanobis distance between
    y and ym, and the log determinant of the covariance matrix.

    Parameters:
    y (array-like): The observation vector.
    ym (array-like): The model prediction vector.
    cov (array-like): The covariance matrix.

    Returns:
    tuple: Mahalanobis distance and log determinant of the covariance matrix.
    """
    L = sc.linalg.cholesky(cov, lower=True)
    z = sc.linalg.solve_triangular(L, y - ym, lower=True)
    mahalanobis_sqr = np.dot(z, z)
    log_det = 2 * np.sum(np.log(np.diag(L)))

    return mahalanobis_sqr, log_det


def log_likelihood(mahalanobis_sqr: float, log_det: float, n: int):
    """
    Calculate the log likelihood of a multivariate normal distribution.

    Parameters:
    mahalanobis_sqr (float): The Mahalanobis distance.
    log_det (float): The log determinant of the covariance matrix.
    n (int): The dimension of the data.

    Returns:
    float: The log likelihood value.
    """
    return -0.5 * (mahalanobis_sqr + log_det + n * np.log(2 * np.pi))


def statistical_covariance(y_stat_err: np.ndarray):
    """
    Returns the statistical covariance matrix:
        \[
            \Sigma_{ij}^{stat} = \sigma^2_{i}^{stat} \delta_{ij}
        \]
    where $\sigma^2_{i}^{stat}$ is the statistical variance of the i-th
    observation (`observation.statistical_covariance`).

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


def uncorrelated_model_covariance(frac_err: float, ym: np.ndarray):
    """
    Returns the uncorrelated model covariance matrix:
        \[
            \Sigma_{ij}^{uncorrelated} = \gamma^2 y_m(x_i, \alpha)^2 \delta_{ij}
        \]
    where $\gamma$ is the fractional uncorrelated error.

    This is commonly used as a model-error term or unquantified uncertainty
    term. E.g. if one expects the model to be correct to 1% in any given data
    point, then this should be set to 0.01.

    It should be noted that this assumption ignores correlations between data
    points in the model.

    Parameters
    ----------
    frac_err : float
        Fractional uncorrelated error in the model prediction.
    ym : np.ndarray
        Model prediction for the observation.

    Returns
    -------
    np.ndarray
        Uncorrelated model error covariance matrix.
    """
    return frac_err**2 * np.diag(ym**2)
