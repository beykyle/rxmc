from collections import OrderedDict
from typing import Callable
import numpy as np

from exfor_tools.distribution import AngularDistribution
from jitr.reactions import Reaction

from jitr.xs.elastic import DifferentialWorkspace, ElasticXS

from .corpus import Corpus
from .model import ElasticModel, ElasticWorkspace
from .reaction_constraint import ReactionConstraint


def build_workspaces_from_measurements(
    quantity: str,
    measurements: list[tuple[Reaction, AngularDistribution]],
    angles_vis=None,
    lmax=30,
):
    angles_vis = angles_vis if angles_vis is not None else np.linspace(0.01, 180, 90)
    workspaces = []
    for reaction, measurement in measurements:
        workspace = ElasticWorkspace(
            quantity=quantity,
            reaction=reaction,
            Elab=measurement.Einc,
            angles_rad_constraint=measurement.x * np.pi / 180,
            angles_rad_vis=angles_vis * np.pi / 180,
            lmax=lmax,
        )
        workspaces.append(workspace)
    return workspaces


class ElasticAngularCorpus(Corpus):
    """
    A class to represent a collection of elastic angular constraints.

    Attributes
    ----------
    quantity : str
        The quantity being measured.
    angles_vis : np.ndarray
        Angles for visualization.
    lmax : int
        Maximum angular momentum.
    constraints : list of ReactionConstraint
        A list of reaction constraints.
    """

    def __init__(
        self,
        model: Callable[[DifferentialWorkspace, OrderedDict], ElasticXS],
        params: list,
        model_name: str,
        corpus_name: str,
        quantity: str,
        workspaces: list[ElasticWorkspace],
        measurements: list[tuple[Reaction, AngularDistribution]],
        weights=None,
        **constraint_kwargs,
    ):
        """
        Initialize an `ElasticAngularCorpus` instance from a list of measured
        `AngularDistribution`s and corresponding `Reaction`s, along with the
        corresponding `ElasticWorkspace`s

        Parameters
        ----------
        quantity : str
            The quantity to be measured.
        params : list of str
            The parameter names
        workspaces : list[tuple[AngularDistribution, ElasticWorkspace]]
            A list of tuples containing a reaction and corresponding measured
            angular distribution of given quantity.
        constraint_kwargs : dict, optional
            Additional keyword arguments for constraints. Defaults to `None`.
        """
        constraint_kwargs = constraint_kwargs or {}
        constraints = []

        self.quantity = quantity
        self.measurements = measurements

        for (reaction, measurement), workspace in zip(measurements, workspaces):
            if (
                workspace.kinematics.Elab != measurement.Einc
                or workspace.reaction != reaction
                or not np.allclose(
                    workspace.constraint_workspace.angles, measurement.x * np.pi / 180
                )
            ):
                same_angles = np.allclose(
                    workspace.constraint_workspace.angles,
                    measurement.x * np.pi / 180,
                )
                raise ValueError(
                    f"mismatch between workspace and measurement for subentry "
                    "{measurement.subentry}."
                    "\n          workspace | measuremnt  "
                    "\n ======================================================"
                    f"\n Energy:      {workspace.kinematics.Elab} | {measurement.Einc}"
                    f"\n Reacttion:   {workspace.reaction} |  {reaction}"
                    f"\n Same angles: {same_angles})"
                )
            if self.quantity == "dXS/dRuth" and measurement.quantity == "dXS/dA":
                norm = workspace.constraint_workspace.rutherford / 1000
            elif self.quantity == "dXS/dA" and measurement.quantity == "dXS/dRuth":
                norm = 1000.0 / workspace.constraint_workspace.rutherford
            else:
                norm = None
            try:
                constraints.append(
                    ReactionConstraint(
                        quantity=self.quantity,
                        measurement=measurement,
                        model=ElasticModel(workspace, model),
                        normalize=norm,
                        **constraint_kwargs,
                    )
                )
            except Exception as e:
                raise Exception(
                    "An error occurred while processing measurement "
                    f"from subentry {measurement.subentry}"
                ) from e

        super().__init__(constraints, params, model_name, corpus_name, weights)
