from collections import OrderedDict
from typing import Callable

import numpy as np

from jitr.xs.elastic import DifferentialWorkspace, ElasticXS

from .reaction_workspace import ElasticWorkspace


class Model:
    """
    Represents an arbitrary parameteric model for a specific observable:
        y_model(x;params), for comparison to some experimental measurement
        y_exp(x)
    """

    def __init__(self, x: np.ndarray):
        self.x = x

    def __call__(self, params: OrderedDict):
        """evaluate y_model(self.x, params)"""
        pass


class ElasticModel(Model):
    """
    Evaluates model for a specific observable quantity in elastic scattering:
    one of differential cross section dXS/dA, differential cross sections as
    a ratio to Rutherford dXS/dRuth, or analyzing power Ay
    """

    def __init__(
        self,
        workspace: ElasticWorkspace,
        model: Callable[[DifferentialWorkspace, OrderedDict], ElasticXS],
    ):
        self.workspace = workspace
        self.model = model
        super().__init__(self.workspace.constraint_workspace.angles)

    def __call__(self, params: OrderedDict) -> np.ndarray:
        """
        Params:
            params: Parameters for the model.
        Returns:
            Calculated quantity as a numpy array (same shape as
                angle_rad_constraint).
        """
        xs = self.get_model_xs(params)
        return self.workspace.quantity_extractor(
            xs, self.workspace.constraint_workspace
        )

    def get_model_xs(self, params: dict) -> ElasticXS:
        return self.model(self.workspace.constraint_workspace, params)

    def get_model_xs_visualization(self, params: dict) -> ElasticXS:
        return self.model(self.workspace.visualization_workspace, params)
