"""Physical model interfaces and built-in toy models."""

import numpy as np

from .observation import Observation
from .params import Parameter


class PhysicalModel:
    """
    Represent a parametric model ``y_model(x; params)``.

    The model is compared against measured data held by an
    :class:`~rxmc.observation.Observation`.
    """

    def __init__(self, params: list[Parameter]):
        """
        Initialize the PhysicalModel with a list of parameters.

        Parameters
        ----------
        params : list[Parameter]
            A list of Parameter objects that define the model's parameters.
            Each Parameter should have a name and a dtype.
        """
        self.params = params
        self.n_params = len(self.params)

    def evaluate(self, observation: Observation, *params) -> np.ndarray:
        """
        Evaluate the model at the given parameter values.
        Should be overridden by subclasses.

        Parameters
        ----------
        observation : Observation
            Observation object containing x and y data.
        *params : tuple
            Parameters for the model, in the same order as ``self.params``.

        """
        raise NotImplementedError("Subclasses must implement the evaluate method.")

    def __call__(self, observation: Observation, *params) -> np.ndarray:
        return self.evaluate(observation, *params)


class Polynomial(PhysicalModel):
    """
    Polynomial model for fitting.

    The model form is ``y_model(x; params) = sum(a_i * x**i)``.
    """

    def __init__(self, order: int):
        params = []
        for i in range(order + 1):
            params.append(Parameter(f"a{i}", latex_name=f"a_{i}", dtype=float))
        self.order = order
        super().__init__(params)

    def evaluate(self, observation: Observation, *params) -> np.ndarray:
        """
        Evaluate the polynomial model at the given parameter values.

        Parameters
        ----------
        observation : Observation
            Observation object containing x and y data.
        *params : tuple
            Coefficients for the polynomial.

        Returns
        ----------
        numpy.ndarray
            Evaluated polynomial values at ``observation.x``.

        Raises
        ----------
        ValueError
            If the number of parameters does not match the model order.
        """
        if len(params) != self.order + 1:
            raise ValueError(
                f"Expected {len(self.params)} parameters, got {len(params)}"
            )

        # Create an exponent matrix for the x values
        # alternatively, one could implement a derived class of
        # Observation that precomputes the Vander matrix
        x_powers = np.vander(observation.x, self.order + 1, increasing=True)

        # Compute the dot product to get the result
        y = np.dot(x_powers, np.asarray(params))

        return y
