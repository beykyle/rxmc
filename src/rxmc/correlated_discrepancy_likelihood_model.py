import numpy as np
from sklearn.gaussian_process.kernels import Kernel

from .likelihood_model import ParametricLikelihoodModel, scale_covariance
from .observation import Observation
from .params import Parameter


class SklearnKernelGPDiscrepancyModel(ParametricLikelihoodModel):
    """
    GP discrepancy via an externally-defined sklearn Kernel object.

    We infer the kernel's free hyperparameters by sampling kernel.theta
    (which sklearn stores in log-space for positive parameters).

    covariance = observation.covariance(ym) + K_disc(x,x; kernel.theta) + jitter*I
    """

    def __init__(
        self,
        kernel: Kernel,
        jitter: float = 1e-10,
        param_prefix: str = "discrepancy_",
        *args,
        **kwargs,
    ):
        self.kernel = kernel
        self.jitter = float(jitter)
        self.param_prefix = param_prefix

        # Build Parameter list from sklearn kernel hyperparameters
        # We only include non-fixed hypers (those that appear in theta).
        # sklearn: kernel.theta is an array of the free hyperparameters (log-space).
        likelihood_params = []
        for hp in kernel.hyperparameters:
            if hp.fixed:
                continue
            # These correspond (in order) to entries in kernel.theta
            # We'll store them as log-values to match sklearn's convention.
            likelihood_params.append(
                Parameter(
                    f"{param_prefix}_{hp.name}",  # e.g. disc_k1__constant_value
                    float,
                    latex_name=hp.name,
                )
            )

        super().__init__(likelihood_params, *args, **kwargs)

    def _kernel_matrix(
        self, observation: Observation, theta_vec: np.ndarray
    ) -> np.ndarray:
        X = np.asarray(observation.x)
        if X.ndim == 1:
            X = X[:, None]
        # clone kernel with the provided theta (log-space)
        k = self.kernel.clone_with_theta(np.asarray(theta_vec, dtype=float))
        return k(X)  # (N,N)

    def covariance(self, observation: Observation, ym: np.ndarray, *kernel_theta):
        if len(kernel_theta) != self.n_params:
            raise ValueError(
                f"Expected {self.n_params} kernel hyperparameters, got {len(kernel_theta)}"
            )

        sigma_obs = observation.covariance(ym)

        K_disc = self._kernel_matrix(observation, np.array(kernel_theta, dtype=float))

        cov = sigma_obs + K_disc
        if self.jitter > 0:
            cov = cov + self.jitter * np.eye(observation.n_data_pts)

        return scale_covariance(
            cov, observation, self.covariance_scale, self.divide_by_N
        )
