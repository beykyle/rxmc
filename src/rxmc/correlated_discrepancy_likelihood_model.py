"""
Gaussian-process discrepancy likelihood model using sklearn kernels.

:class:`SklearnKernelGPDiscrepancyModel` adds a GP discrepancy term to the
observation covariance, with the kernel hyperparameters sampled alongside the
physical-model parameters via the :class:`~rxmc.likelihood_model.ParametricLikelihoodModel`
interface.
"""

import numpy as np
from sklearn.gaussian_process.kernels import Kernel

from .likelihood_model import ParametricLikelihoodModel
from .observation import Observation
from .params import Parameter


class SklearnKernelGPDiscrepancyModel(ParametricLikelihoodModel):
    """Parametric likelihood with a GP discrepancy covariance term.

    Adds a kernel-based GP discrepancy covariance to the observation covariance:

    .. math::

        \\Sigma = \\Sigma_{\\mathrm{obs}}(y_m) + K_{\\mathrm{disc}}(x, x;\\,\\theta) + \\epsilon I

    The free hyperparameters of the sklearn kernel (those not marked ``fixed``)
    become likelihood parameters that are sampled alongside the physical-model
    parameters.  sklearn stores hyperparameters in log space via ``kernel.theta``,
    so the sampled values are also in log space.

    Parameters
    ----------
    kernel : sklearn.gaussian_process.kernels.Kernel
        Frozen sklearn kernel.  Free hyperparameters (``hp.fixed == False``)
        are registered as likelihood parameters.
    jitter : float, optional
        Small diagonal regularisation added to the GP covariance matrix for
        numerical stability.  Defaults to ``1e-10``.
    param_prefix : str, optional
        Prefix applied to each hyperparameter name when building the
        :class:`~rxmc.params.Parameter` list.  Defaults to ``"discrepancy_"``.
    """

    def __init__(
        self,
        kernel: Kernel,
        jitter: float = 1e-10,
        param_prefix: str = "discrepancy_",
    ):
        self.kernel = kernel
        self.jitter = float(jitter)
        self.param_prefix = param_prefix

        likelihood_params = []
        for hp in kernel.hyperparameters:
            if hp.fixed:
                continue
            likelihood_params.append(
                Parameter(
                    f"{param_prefix}_{hp.name}",
                    float,
                    latex_name=hp.name,
                )
            )

        super().__init__(likelihood_params)

    def _kernel_matrix(
        self, observation: Observation, theta_vec: np.ndarray
    ) -> np.ndarray:
        """Evaluate the GP kernel matrix for the observation input grid.

        Parameters
        ----------
        observation : Observation
            Observation whose ``x`` attribute provides the input locations.
        theta_vec : np.ndarray
            Kernel hyperparameters in sklearn's log space.

        Returns
        -------
        np.ndarray, shape (n, n)
            Kernel matrix evaluated at ``observation.x``.
        """
        X = np.asarray(observation.x)
        if X.ndim == 1:
            X = X[:, None]
        k = self.kernel.clone_with_theta(np.asarray(theta_vec, dtype=float))
        return k(X)

    def covariance(self, observation: Observation, ym: np.ndarray, *kernel_theta):
        """Total covariance: observation covariance plus GP discrepancy.

        Parameters
        ----------
        observation : Observation
            Observation object.
        ym : np.ndarray
            Model prediction for the observation.
        *kernel_theta : float
            Kernel hyperparameter values in sklearn's log space, one per free
            hyperparameter of the kernel.

        Returns
        -------
        np.ndarray, shape (n, n)
            Combined covariance matrix.

        Raises
        ------
        ValueError
            If the number of *kernel_theta* values does not match
            ``self.n_params``.
        """
        if len(kernel_theta) != self.n_params:
            raise ValueError(
                f"Expected {self.n_params} kernel hyperparameters, got {len(kernel_theta)}"
            )

        sigma_obs = observation.covariance(ym)
        K_disc = self._kernel_matrix(observation, np.array(kernel_theta, dtype=float))

        cov = sigma_obs + K_disc
        if self.jitter > 0:
            cov = cov + self.jitter * np.eye(observation.n_data_pts)

        return cov
