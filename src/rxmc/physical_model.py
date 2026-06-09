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
