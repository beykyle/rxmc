import numpy as np

from .observation import Observation
from .params import Parameter


class PhysicalModel:
    """
    Represents an arbitrary parameteric model $y_{model}(x;params)$, for
    comparison to some experimental measurement $\{x_i, y(x_i)\}$ contained
    in an Observation object
    """

    def __init__(self, params: list[Parameter]):
        self.params = params
        self.n_params = len(self.params)

    def evaluate(self, observation: Observation, *params) -> np.ndarray:
        """
        Evaluate the model at the given parameter values.
        Should be overridden by subclasses.

        Parameters:
        ----------
        observation: Observation object containing x and y data.
        params: Parameters for the model, should match the model's parameters.

        """
        raise NotImplementedError("Subclasses must implement the evaluate method.")

    def __call__(self, observation: Observation, *params) -> np.ndarray:
        return self.evaluate(observation, params)


class Polynomial(PhysicalModel):
    """
    Polynomial model for fitting, of the form:
    \[
        y_{model}(x; params) = \sum_{i=0}^{n} a_i x^i
    \]
    where $params = [a_0, a_1, ..., a_n]$.
    """

    def __init__(self, order: int):
        params = []
        for i in range(order + 1):
            params.append(Parameter(f"f{i}", latex_name=f"a_{i}", dtype=float))
        super().__init__(params)

    def evaluate(self, observation: Observation, *params) -> np.ndarray:
        """
        Evaluate the polynomial model at the given parameter values.

        Parameters:
        ----------
            observation: Observation
                Observation object containing x and y data.
            params: tuple
                coefficients for the polynomial

        Returns:
            numpy.ndarray: Evaluated polynomial values at observation.x.

        Raises:
            ValueError: If the number of parameters does not match the model order.
        """
        if len(params) != self.order + 1:
            raise ValueError(
                f"Expected {len(self.params)} parameters, got {len(params)}"
            )

        # Create an exponent matrix for the x values
        # alternatively, one could implement a derived class of
        # Observation that precomputes the Vander matrix
        x_powers = np.vander(self.observation.x, self.order + 1, increasing=True)

        # Compute the dot product to get the result
        y = np.dot(x_powers, np.asarray(params))

        return y
