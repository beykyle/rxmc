"""
Physical model for elastic differential cross sections.

:class:`ElasticDifferentialXSModel` wraps a ``jitr`` optical-model solver to
predict elastic differential cross sections (dXS/dΩ, dXS/dRuth, or analysing
power Ay) given a parametric central and spin-orbit interaction.
"""

import numpy as np

from .bfrescox_observation import BfrescoxObservation
from .observation import Observation
from .physical_model import PhysicalModel


class BfrescoxModel(PhysicalModel):
    """
    A model that runs Bfrescox to predict reaction observables
    """

    def __init__(
        self,
        params: list = [],
        model_name: str | None = None,
        channel_name: str | None = None,
    ):
        """
        Parameters
        ----------
        params : list of Parameter, optional
            Parameters of the model.  Defaults to ``[]``.
        model_name : str, optional
            Human-readable model name.  Defaults to ``"ElasticDifferentialXSModel"``.
        """
        self.model_name = model_name or "BfrescoxModel"
        self.channel_name = channel_name or "channel_1"

        super().__init__(params)

    def evaluate(
        self,
        observation: Observation,
        *params: tuple,
    ) -> np.ndarray:
        """ """
        if not isinstance(observation, BfrescoxObservation):
            raise ValueError(
                f"Observation must be a BfrescoxObservation, but got {type(observation)}"
            )

        # convert params tuple to dict
        params_dict = {param.name: val for param, val in zip(self.params, params)}

        # run the Bfrescox calculation and return the predicted observable
        df = observation.run(params_dict)[self.channel_name]

        # extract the predicted observable from the DataFrame and return it
        # as a numpy array
        return df["sigma_mb_sr"].to_numpy()
