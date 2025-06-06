from collections import OrderedDict
from typing import Callable

import numpy as np

from jitr.xs.elastic import DifferentialWorkspace, ElasticXS

from .reaction_workspace import ElasticWorkspace
from .observation import Observation
from .params import Parameter, to_ordered_dict


class PhysicalModel:
    """
    Represents an arbitrary parameteric model for a specific observation:
    y_model(x;params), for comparison to some experimental measurement
    y(x) contained in an Observation object
    """

    # TODO pass in Observation object into __call__ method and evaluate
    # along with params.
    # In the case of ReactionModel, we would need to take in a
    # specialized Observation object, ReactionObservation,
    # which constinas the Workspace objects as well.
    # Then the actual model would just be a callable that takes in
    # the workspace and the parameters, and returns a the corresponding
    # y value.
    # One would define subclasses of ReactionModel for each model type,
    # e.g. KDUQModel
    def __init__(self, params: list[Parameter]):
        self.params = params

    def evaluate(self, parameters: OrderedDict):
        """
        Evaluate the model at the given parameter values.
        Should be overridden by subclasses.

        Parameters:
        ----------
            parameters: An OrderedDict mapping parameter names to values.
        """
        raise NotImplementedError("Subclasses must implement the evaluate method.")

    def __call__(self, params):
        """evaluate y_model(self.x, params)"""
        if isinstance(params, np.ndarray):
            return self.evaluate(to_ordered_dict(params, [p.name for p in self.params]))
        elif isinstance(params, OrderedDict):
            assert set(params.keys()) == set([p.name for p in self.params])
            return self.evaluate(params)
        else:
            raise TypeError(
                "params must be a numpy array (in the same order"
                " as self.params) or an OrderedDict."
            )


class ElasticModel(PhysicalModel):
    """
    Evaluates model for a specific observable quantity in elastic scattering:
    one of differential cross section dXS/dA, differential cross sections as
    a ratio to Rutherford dXS/dRuth, or analyzing power Ay
    """

    def __init__(
        self,
        observation: Observation,
        params: list[Parameter],
        workspace: ElasticWorkspace,
        model: Callable[[DifferentialWorkspace, OrderedDict], ElasticXS],
    ):
        self.workspace = workspace
        self.model = model
        super().__init__(observation, params)

        if observation.x != self.workspace.constraint_workspace.angles:
            raise ValueError(
                "Observation x must match the angles in the workspace's "
                "constraint workspace."
            )

    def __call__(self, params: OrderedDict) -> np.ndarray:
        """
        Params:
            params: Parameters for the model.
        Returns:
            Calculated quantity as a numpy array of shape
            (self.observation.x.shape[0],)
        """
        xs = self.get_model_xs(params)
        return self.workspace.quantity_extractor(
            xs, self.workspace.constraint_workspace
        )

    def get_model_xs(self, params: dict) -> ElasticXS:
        return self.model(self.workspace.constraint_workspace, params)

    def get_model_xs_visualization(self, params: dict) -> ElasticXS:
        return self.model(self.workspace.visualization_workspace, params)
