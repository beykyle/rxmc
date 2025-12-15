import numpy as np
import scipy as sc

from .observation import FixedCovarianceObservation, Observation
from .params import Parameter


class LikelihoodModel:
    r"""
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

    def __init__(self):
        r"""
        Initializes the LikelihoodModel, optionally with a fractional
        uncorrelated error.

        """
        self.params = None
        self.n_params = 0

    def covariance(self, observation: Observation, ym: np.ndarray):
        r"""
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
        return observation.covariance(ym)

    def residual(self, observation: Observation, ym: np.ndarray):
        r"""
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
        r"""
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
        r"""
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
    r"""
    A special LikelihoodModel to handle FixedCovarianceObservation objects,
    where the covariance matrix is fixed and does not depend on the
    parameters of the PhysicalModel.

    This allows for the use of precomputed inverse covariance matrices which
    can speed up the calculation of the chi-squared statistic and log_likelihood.
    """

    def __init__(self):
        super().__init__()

    def covariance(self, observation: FixedCovarianceObservation, ym: np.ndarray):
        r"""
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
        r"""
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
        r"""
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


class Chi2LikelihoodModel(LikelihoodModel):
    r"""
    A `LikelihoodModel` that returns the negative half of the chi-squared
    statistic for the log likelihood, ignoring the log determinant term.
    This is useful for doing chi2 minimization without computing the full
    log likelihood.
    """

    def __init__(self):
        super().__init__()

    def log_likelihood(self, observation: Observation, ym: np.ndarray):
        r"""
        Returns -1/2 Chi2  only, ignoring the log determinant term.

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
        chi2 = self.chi2(observation, ym)
        return -0.5 * chi2


class ParametricLikelihoodModel(LikelihoodModel):
    r"""
    A class to represent a likelihood model for comparing an `Observation`
    to a `PhysicalModel`, in which the `LikelihoodModel` has it's own parameters
    to calculate the covariance, aside from the parameters of the
    `PhysicalModel`. This is useful when the covariance is unknown, and one
    would like to calibrate the likelihood parameters to an `Observation`,
    along with the parameters of a `PhysicalModel`.
    """

    def __init__(self, likelihood_params: list[Parameter]):
        super().__init__()
        self.params = likelihood_params
        self.n_params = len(likelihood_params)

    def chi2(self, observation: Observation, ym: np.ndarray, *likelihood_params):
        r"""
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
        r"""
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
        r"""
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
    r"""
    A ParametricLikelihoodModel in which each data point in the observation
    has the same, unknown, statistical error, which is a parameter, $epsilon$.

    No matter the Observation, the statistical contribution to the covariance
    thus always takes the form:

    \[
        \Sigma_{ij}^{stat} = \epsilon^2 \delta_{ij}
    \]

    where $\epsilon$ is the statistical `noise` parameter.
    """

    def __init__(self):
        r"""
        Initializes the UnknownNoiseErrorModel, optionally with
        a fractional uncorrelated error.
        """
        likelihood_params = [
            Parameter("log noise", float, latex_name=r"\log{\epsilon}"),
        ]
        super().__init__(likelihood_params)

    def covariance(self, observation: Observation, ym: np.ndarray, log_epsilon: float):
        r"""
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
        sigma_stat = statistical_covariance(np.ones_like(ym) * np.exp(log_epsilon))
        cov = sigma_sys + sigma_stat
        return cov


class UnknownNoiseFractionErrorModel(ParametricLikelihoodModel):
    r"""
    A `ParametricLikelihoodModel` in which each data point in the observation
    has the a statistical error corresponding to a fixed fraction of it's value,
    the fraction being a parameter, $epsilon$.

    This implies the statistical contribution to the covariance takes the form:

    \[
        \Sigma_{ij}^{stat} = \epsilon^2 y(x_i)^2 \delta_{ij}
    \]


    """

    def __init__(self):
        likelihood_params = [
            Parameter(
                "log noise fraction",
                float,
                latex_name=r"\log{\epsilon}",
                unit="dimensionless",
            )
        ]
        super().__init__(likelihood_params)

    def covariance(self, observation: Observation, ym: np.ndarray, log_epsilon: float):
        r"""
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
        sigma_stat = statistical_covariance(ym * np.exp(log_epsilon))
        cov = sigma_sys + sigma_stat
        return cov


class UnknownNormalizationModel(ParametricLikelihoodModel):
    r"""
    A `ParametricLikelihoodModel` in which the (log of the)
    multiplicative factor of the model output is a free parameter. In
    this case, the covariance does not include systematic errors due to
    the normalization, as the data are explicitly re-normalized.

    This corresponds to a statistical model:
        \[
            y_i  + \epsilon_i = \rho y_m(x_i, \alpha) + \epsilon_i + \dots
        \]

    where $\rho$ is a free parameter corresponding to the multiplicative
    normalization factor. This corresponds to the Kennedy & O'Hagan
    (2001) treatment of normalization of model output in in 'Bayesian
    calibration of computer models'. Here the normalization is not
    associated with the data, but is a latent parameter which scales the
    model output to best fit the data.

    This should not be confused with `UnknownDataNormalizationModel`, in
    which the normalization of the data is a free parameter.

    In fact, the confusion between them is related to Peelle's Pertinent
    Puzzle, or D'Agostin bias.
    """

    def __init__(self):
        likelihood_params = [
            Parameter(
                "log normalization",
                float,
                latex_name=r"\log{\rho}",
                unit="dimensionless",
            ),
        ]
        super().__init__(likelihood_params)

    def covariance(self, observation: Observation, ym: np.ndarray, log_rho: float):
        r"""
        Returns the statistical covariance matrix:
            \[
                \Sigma_{ij}^{stat} = \sigma^2_{i}^{stat} \delta_{ij}
            \]
        where $\sigma^2_{i}^{stat}$ is the statistical variance of the i-th
        observation, (`observation.statistical_covariance`).

        When the normalization is a free parameter, the systematic
        normalization contribution to the covariance is not included,
        as the data are explicitly re-normalized.

        Parameters
        ----------
        observation : Observation
            The observation object containing the observed data.
        ym : np.ndarray
            Model prediction for the observation
        log_rho : float
            natural log of the multiplicative normalization factor.
        """
        return observation.statistical_covariance

    def residual(self, observation: Observation, ym: np.ndarray, log_rho: float):
        r"""
        Returns the residual between the renormalized model prediction ym and
        observation.y

        Parameters:
        ----------
        observation : Observation
            The observation object containing the observed data.
        ym : np.ndarray
            Model prediction for the observation.
        log_rho : float
            natural log of the multiplicative normalization factor.

        Returns
        -------
        np.ndarray
            Residual vector.
        """
        return observation.y - ym * np.exp(log_rho)

    def chi2(self, observation: Observation, ym: np.ndarray, log_rho: float):
        r"""
        Calculate the generalised chi-squared statistic. This is the
        square of the Mahalanobis distance between y and ym

        Parameters
        ----------
        observation : Observation
            The observation object containing the observed data.
        ym : np.ndarray
            Model prediction for the observation.
        log_rho : float
            natural log of the multiplicative normalization factor.

        Returns
        -------
        float
            Chi-squared statistic.
        """
        y_renorm = ym * np.exp(log_rho)
        cov = self.covariance(observation, ym, log_rho)
        mahalanobis_sqr, _ = mahalanobis_distance_sqr_cholesky(
            observation.y, y_renorm, cov
        )
        return mahalanobis_sqr

    def log_likelihood(self, observation: Observation, ym: np.ndarray, log_rho: float):
        r"""
        Returns the log_likelihood that ym reproduces y, given the covariance

        Parameters
        ----------
        ym : np.ndarray
            Model prediction for the observation.
        observation : Observation
            The observation object containing the observed data.
        log_rho : float
            natural log of the multiplicative normalization factor.

        Returns
        -------
        float
        """
        y_renorm = ym * np.exp(log_rho)
        cov = self.covariance(observation, ym, log_rho)
        mahalanobis_sqr, log_det = mahalanobis_distance_sqr_cholesky(
            observation.y, y_renorm, cov
        )
        return log_likelihood(mahalanobis_sqr, log_det, observation.n_data_pts)


class UnknownNormalizationErrorModel(ParametricLikelihoodModel):
    r"""
    A ParametricLikelihoodModel in which the systematic uncertainty
    of the normalization of the observation is a parameter, $\eta$.

    This implies the systematic normalization contribution to the
    covariance takes the form:

        \[
            \Sigma_{ij}^{sys norm} = \eta**2 y_m(x_i, \alpha) y_m(x_j, \alpha)
        \]

    where $\eta$ is a free parameter.
    """

    def __init__(self):
        likelihood_params = [
            Parameter(
                "log normalization error",
                float,
                latex_name=r"\log{\eta}",
                unit="dimensionless",
            )
        ]
        super().__init__(likelihood_params)

    def covariance(self, observation: Observation, ym: np.ndarray, log_eta: float):
        r"""
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
        sigma_stat = observation.statistical_covariance
        cov = sigma_sys + sigma_stat
        return cov


class UnknownModelError(ParametricLikelihoodModel):
    r"""
    A `ParametricLikelihoodModel` in which the `frac_err`
    is a free parameter $\gamma$, such that the covariance due to the
    uncorrelated model error takes the form:

    \[
        \Sigma_{ij}^{uncorrelated} = \gamma^2 y_m(x_i, \alpha)^2 \delta_{ij}
    \]

    where $\gamma$ is a free parameter.

    This is commonly used as a model-error term or unquantified uncertainty.
    """

    def __init__(self, averaging=True):
        """
        Initializes the UnknownModelError instance.

        Parameters
        averaging : bool
            If True, the model error is calculated using the average of
            observation.y and ym, i.e., 0.5 * (observation.y + ym).
            This can help stabilize the optimization when ym is very small
            or zero. If False, the model error is calculated using ym only.
        """
        likelihood_params = [
            Parameter(
                "log fractional err", float, latex_name=r"\gamma", unit="dimensionless"
            )
        ]
        super().__init__(likelihood_params)
        self.averaging = averaging

    def covariance(
        self,
        observation: Observation,
        ym: np.ndarray,
        log_frac_err: float,
    ):
        r"""
        Default covariance model. Derived classes of `LikelihoodModel` will
        override this.

        Returns the following covariance matrix:
            \[
                \Sigma_{ij} = \sigma^2_{i}^{stat} \delta_{ij}
                            + \Sigma_{ij}^{sys}
                            + \gamma^2 y_m^2(x_i, \alpha)
            \]
        where $\gamma$ is the fractional uncorrelated error
        (`frac_err`), treated here as a free parameter, and
        all other definitions are the same as `LikelihoodModel.covariance`


        Parameters
        ----------
        ym : np.ndarray
            Model prediction for the observation.
        observation : Observation
            The observation object containing the observed data.
        log_frac_err: float
            log of fraction of the model prediction at point x_i that
            is treated as the standard deviation of the model prediction
            at that point, such that the model prediction is independent
            at every point (log of $\gamma$).

        Returns
        -------
        np.ndarray
            Covariance matrix of the observation.
        """
        sigma = observation.covariance(ym)
        sigma_model = uncorrelated_model_covariance(
            np.exp(log_frac_err),
            ym if not self.averaging else 0.5 * (observation.y + ym),
        )
        cov = sigma + sigma_model
        return cov


class StudentTLikelihoodModel(LikelihoodModel):
    r"""
    A `LikelihoodModel` that uses a Student's t-distribution for the likelihood.
    This is useful when the data has heavy tails or outliers, as it is more robust
    to deviations from normality compared to the Gaussian likelihood.
    """

    def __init__(self):
        r"""
        Initializes the StudentTLikelihoodModel with a specified degrees of freedom.
        """
        likelihood_params = [
            Parameter("degrees_of_freedom", float, latex_name=r"\nu"),
        ]
        super().__init__(likelihood_params)

    def log_likelihood(self, observation: Observation, ym: np.ndarray, nu: float):
        r"""
        Calculate the log likelihood using the Student's t-distribution.

        Parameters
        ----------
        observation : Observation
            The observation object containing the observed data.
        ym : np.ndarray
            Model prediction for the observation.
        nu : float
            Degrees of freedom for the Student's t-distribution.

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
            sc.special.gammaln((n + nu) / 2)
            - sc.special.gammaln(nu / 2)
            - 0.5 * n * np.log(np.pi * nu)
            - 0.5 * log_det
            - (nu + n) / 2 * np.log(1 + mahalanobis_sqr / nu)
        )


def scale_covariance(
    cov: np.ndarray, observation: Observation, scale: float, divide_by_N: bool
) -> np.ndarray:
    if divide_by_N:
        scale /= observation.n_data_pts
    return scale * cov


def mahalanobis_distance_sqr_cholesky(y, ym, cov):
    r"""
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
    r"""
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
    r"""
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
    r"""
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
