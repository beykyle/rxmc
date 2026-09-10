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
from .transforms import as_transform


class PhysicalModel:
    """Abstract base class for parametric physical models.

    Represents an arbitrary parametric model
    $y_{\\mathrm{model}}(x;\\,\\alpha)$ for comparison to an experimental
    measurement $\\{x_i,\\, y(x_i)\\}$ encapsulated in an
    :class:`~rxmc.observation.Observation`.

    Subclasses implement :meth:`evaluate` in physical space.  An optional
    *parametric* ``transform`` (see :mod:`rxmc.transforms`) is applied on top by
    :meth:`__call__`; its parameters are appended to :attr:`params` so they flow
    through the ordinary model-parameter machinery (priors, ``split_parameters``).
    Typical uses are a latent normalisation :func:`rxmc.transforms.scale` or one
    per dataset via :func:`rxmc.transforms.per_observation_scaling`.  Comparison-
    space transforms (e.g. comparing in log space) are *not* the model's
    business: declare them on the :class:`~rxmc.observation.Observation`.

    Parameters
    ----------
    params : list of Parameter
        Physical parameters of the model.  Each entry should carry a name
        and a data type.
    transform : Transform or callable, optional
        Model-side transform ``y -> transform(y, *values)`` applied after
        :meth:`evaluate`.  Its parameters (if any) are appended to ``params``.
    """

    def __init__(self, params: list[Parameter], transform=None):
        self.base_params = list(params)
        self.transform = as_transform(transform)
        self.params = self.base_params + list(self.transform.params)
        self.n_base_params = len(self.base_params)
        self.n_params = len(self.params)

    def split_params(self, params):
        """Split a full parameter tuple into ``(base_params, transform_values)``."""
        params = tuple(params)
        if len(params) != self.n_params:
            raise ValueError(
                f"{type(self).__name__} expects {self.n_params} parameter(s), "
                f"got {len(params)}"
            )
        return params[: self.n_base_params], params[self.n_base_params :]

    def apply_transform(self, observation, y, transform_values=()):
        """Apply the model-side transform to a physical-space prediction."""
        if self.transform.is_identity:
            return np.asarray(y, dtype=float)
        return self.transform(y, *transform_values, context=observation)

    def evaluate(self, observation: Observation, *params) -> np.ndarray:
        """Evaluate the model at the given parameter values.

        Must be overridden by subclasses.

        Parameters
        ----------
        observation : Observation
            Observation containing the independent-variable grid.
        *params : float
            Physical-model (base) parameter values only; any transform
            parameters are split off by :meth:`__call__` before this is called.

        Returns
        -------
        np.ndarray
            Predicted observable values on the observation grid (physical
            space, before the model transform).

        Raises
        ------
        NotImplementedError
            Always — subclasses must implement this method.
        """
        raise NotImplementedError("Subclasses must implement the evaluate method.")

    def __call__(self, observation: Observation, *params) -> np.ndarray:
        """Physical-space :meth:`evaluate` followed by the model transform."""
        base, values = self.split_params(params)
        return self.apply_transform(
            observation, self.evaluate(observation, *base), values
        )


class Polynomial(PhysicalModel):
    r"""Polynomial model of fixed order.

    Predicts observable values as

    .. math::

        y_{\mathrm{model}}(x;\,a_0,\dots,a_n) = \sum_{i=0}^{n} a_i\, x^i

    Parameters
    ----------
    order : int
        Polynomial order $n$.  The model has $n+1$ free coefficients.
    transform : Transform or callable, optional
        See :class:`PhysicalModel`.
    """

    def __init__(self, order: int, transform=None):
        params = []
        for i in range(order + 1):
            params.append(Parameter(f"a{i}", latex_name=f"a_{i}", dtype=float))
        self.order = order
        super().__init__(params, transform=transform)

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
            raise ValueError(f"Expected {self.order + 1} parameters, got {len(params)}")

        x_powers = np.vander(observation.x, self.order + 1, increasing=True)
        y = np.dot(x_powers, np.asarray(params))
        return y
