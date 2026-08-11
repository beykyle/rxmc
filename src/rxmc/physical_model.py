"""
Abstract physical model and a concrete polynomial model.

A :class:`PhysicalModel` maps a parameter vector to predicted observable values
for a given :class:`~rxmc.observation.Observation`.  Subclasses implement
:meth:`~PhysicalModel.evaluate`; the base class makes the object callable so it
can be used directly as ``model(obs, *params)``.

:class:`Polynomial` is a ready-to-use implementation for polynomial regression.
"""

import numpy as np

from .observation import Observation
from .params import Parameter


class PhysicalModel:
    """Abstract base class for parametric physical models.

    Represents an arbitrary parametric model
    $y_{\\mathrm{model}}(x;\\,\\alpha)$ for comparison to an experimental
    measurement $\\{x_i,\\, y(x_i)\\}$ encapsulated in an
    :class:`~rxmc.observation.Observation`.

    Parameters
    ----------
    params : list of Parameter
        Parameters that define the model.  Each entry should carry a name
        and a data type.
    """

    def __init__(self, params: list[Parameter]):
        self.params = params
        self.n_params = len(self.params)

    def evaluate(self, observation: Observation, *params) -> np.ndarray:
        """Evaluate the model at the given parameter values.

        Must be overridden by subclasses.

        Parameters
        ----------
        observation : Observation
            Observation containing the independent-variable grid.
        *params : float
            Model parameter values.

        Returns
        -------
        np.ndarray
            Predicted observable values on the observation grid.

        Raises
        ------
        NotImplementedError
            Always — subclasses must implement this method.
        """
        raise NotImplementedError("Subclasses must implement the evaluate method.")

    def __call__(self, observation: Observation, *params) -> np.ndarray:
        return self.evaluate(observation, *params)


class Polynomial(PhysicalModel):
    r"""Polynomial model of fixed order.

    Predicts observable values as

    .. math::

        y_{\mathrm{model}}(x;\,a_0,\dots,a_n) = \sum_{i=0}^{n} a_i\, x^i

    Parameters
    ----------
    order : int
        Polynomial order $n$.  The model has $n+1$ free coefficients.
    """

    def __init__(self, order: int):
        params = []
        for i in range(order + 1):
            params.append(Parameter(f"a{i}", latex_name=f"a_{i}", dtype=float))
        self.order = order
        super().__init__(params)

    def evaluate(self, observation: Observation, *params) -> np.ndarray:
        """Evaluate the polynomial at the observation grid.

        Parameters
        ----------
        observation : Observation
            Observation whose ``x`` attribute provides the evaluation grid.
        *params : float
            Polynomial coefficients ``a0, a1, ..., an`` (lowest order first).

        Returns
        -------
        np.ndarray
            Polynomial values at ``observation.x``.

        Raises
        ------
        ValueError
            If the number of supplied coefficients does not match
            ``self.order + 1``.
        """
        if len(params) != self.order + 1:
            raise ValueError(
                f"Expected {len(self.params)} parameters, got {len(params)}"
            )

        x_powers = np.vander(observation.x, self.order + 1, increasing=True)
        y = np.dot(x_powers, np.asarray(params))
        return y


class ScaledModel(PhysicalModel):
    r"""A physical model with a latent multiplicative normalisation.

    Wraps a base :class:`PhysicalModel` and prepends a scale parameter
    :math:`\rho`, returning

    .. math::

        y_{\mathrm{model}}(x;\, \rho, \alpha) = \rho \, y_{\mathrm{base}}(x;\, \alpha)

    This is the Kennedy & O'Hagan latent forward-model scale that the old
    ``UnknownNormalizationModel`` expressed on the likelihood side.  It changes the
    *mean*, not the covariance, so it lives on the model and flows through the
    ordinary model-parameter machinery (priors, ``split_parameters``).

    Parameters
    ----------
    base_model : PhysicalModel
        The model whose prediction is rescaled.
    scale_parameter : Parameter, optional
        The scale parameter.  Defaults to a log-scale ``log rho``; set
        ``log=False`` for a linear scale.
    log : bool, optional
        If ``True`` (default), the sampled value is ``log(rho)`` and the model
        scales by ``exp(value)``; otherwise it scales by ``value`` directly.
    """

    def __init__(
        self, base_model: PhysicalModel, scale_parameter: Parameter = None, log=True
    ):
        self.base_model = base_model
        self.log = log
        if scale_parameter is None:
            scale_parameter = Parameter(
                "log normalization",
                float,
                unit="dimensionless",
                latex_name=r"\log{\rho}" if log else r"\rho",
            )
        self.scale_parameter = scale_parameter
        super().__init__([scale_parameter] + list(base_model.params))

    def evaluate(self, observation: Observation, *params) -> np.ndarray:
        if len(params) != self.n_params:
            raise ValueError(f"Expected {self.n_params} parameters, got {len(params)}")
        scale = np.exp(params[0]) if self.log else params[0]
        return scale * self.base_model.evaluate(observation, *params[1:])


class PerObservationScaledModel(PhysicalModel):
    r"""A physical model with an independent latent normalisation per dataset.

    Wraps a base :class:`PhysicalModel` and assigns one scale parameter
    :math:`\rho_i` to each :class:`~rxmc.observation.Observation` in
    ``observations``, routing **by identity**: when evaluated on observation
    :math:`i` it returns :math:`\rho_i\, y_{\mathrm{base}}`.  The base parameters
    come first, followed by the per-observation scales in ``observations`` order.

    Because every constraint shares one such model instance, the per-dataset
    scales are ordinary *model* parameters (sampled in the model block jointly
    with the physics) rather than per-constraint covariance nuisances — this is
    how the old per-dataset ``UnknownNormalizationModel`` is expressed in v2.  The
    routing reuses the same gather-by-identity idea as
    :class:`~rxmc.covariance.ConstraintCovariance`.

    Parameters
    ----------
    base_model : PhysicalModel
        The model whose prediction is rescaled.
    observations : sequence of Observation
        The datasets, each assigned its own scale parameter (matched by identity
        when :meth:`evaluate` is called).
    scale_parameters : sequence of Parameter, optional
        One scale parameter per observation.  Defaults to ``log_rho_{i}``.
    log : bool, optional
        If ``True`` (default), the sampled value is ``log(rho_i)`` and the model
        scales by ``exp(value)``; otherwise it scales by ``value`` directly.
    prefix : str, optional
        Name prefix for the default scale parameters.
    """

    def __init__(
        self,
        base_model: PhysicalModel,
        observations,
        scale_parameters=None,
        log=True,
        prefix="log_rho",
    ):
        self.base_model = base_model
        self.log = log
        self._n_base = len(base_model.params)
        # hold references so id()-keyed routing can never see a recycled id
        self.observations = list(observations)
        self._index = {id(o): i for i, o in enumerate(self.observations)}
        if len(self._index) != len(self.observations):
            raise ValueError(
                "observations must be distinct objects (routing by identity)"
            )
        if scale_parameters is None:
            scale_parameters = [
                Parameter(
                    f"{prefix}_{i}",
                    float,
                    unit="dimensionless",
                    latex_name=(rf"\log{{\rho_{{{i}}}}}" if log else rf"\rho_{{{i}}}"),
                )
                for i in range(len(self._index))
            ]
        self.scale_parameters = list(scale_parameters)
        super().__init__(list(base_model.params) + self.scale_parameters)

    def evaluate(self, observation: Observation, *params) -> np.ndarray:
        if len(params) != self.n_params:
            raise ValueError(f"Expected {self.n_params} parameters, got {len(params)}")
        if id(observation) not in self._index:
            raise KeyError(
                "observation was not registered with this PerObservationScaledModel"
            )
        base_params = params[: self._n_base]
        value = params[self._n_base + self._index[id(observation)]]
        scale = np.exp(value) if self.log else value
        return scale * self.base_model.evaluate(observation, *base_params)
