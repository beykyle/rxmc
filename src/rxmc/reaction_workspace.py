from typing import Callable

import numpy as np

import jitr
from jitr.xs.elastic import DifferentialWorkspace, ElasticXS
from jitr import rmatrix


def extract_dXS_dA(xs: ElasticXS, ws: DifferentialWorkspace) -> np.ndarray:
    """Extracts dXS/dA in b/Sr"""
    return xs.dsdo / 1000


def extract_dXS_dRuth(xs: ElasticXS, ws: DifferentialWorkspace) -> np.ndarray:
    """Extracts dXS/dRuth"""
    return xs.dsdo / ws.rutherford


def extract_Ay(xs: ElasticXS, ws: DifferentialWorkspace) -> np.ndarray:
    """Extracts Ay"""
    return xs.Ay


DEFAULT_LMAX = 20


class ElasticWorkspace:
    """
    Encapsulates two `DifferentialWorkspace`s, one on a shared angular grid
    with a measured angular distribution and one on a (typically finer)
    grid for visualization
    """

    def __init__(
        self,
        quantity: str,
        reaction: jitr.reactions.Reaction,
        Elab: float,
        angles_rad_vis: np.ndarray,
        angles_rad_constraint: np.ndarray,
        lmax: int = DEFAULT_LMAX,
    ):
        """
        Params:
            model: A callable that takes in a DifferentialWorkspace and an
                OrderedDict of params and spits out the corresponding ElasticXS
            quantity: The type of quantity to be calculated (e.g., "dXS/dA",
                "dXS/dRuth", "Ay").
            reaction: The reaction object containing details of the reaction.
            Elab: The laboratory energy.
            angles_rad_vis: Array of angles in radians for visualization.
            angles_rad_constraint: Array of angles in radians corresponding to
                experimentally measured constraints.
            lmax: Maximum angular momentum, defaults to 20.
        """
        if Elab <= 0:
            raise ValueError("Elab must be positive.")

        self.quantity = quantity
        self.reaction = reaction

        check_angle_grid(angles_rad_vis, "angles_rad_vis")
        check_angle_grid(angles_rad_constraint, "angles_rad_constraint")

        self.constraint_workspace, self.visualization_workspace, self.kinematics = (
            set_up_solver(
                reaction,
                Elab,
                angles_rad_constraint,
                angles_rad_vis,
                lmax,
            )
        )
        self.Elab = Elab
        self.quantity_extractor = self.get_quantity_extractor(quantity)

    @staticmethod
    def get_quantity_extractor(
        quantity: str,
    ) -> Callable[[ElasticXS, DifferentialWorkspace], np.ndarray]:
        """Returns the appropriate quantity extraction function."""
        if quantity == "dXS/dA":
            return extract_dXS_dA
        elif quantity == "dXS/dRuth":
            return extract_dXS_dRuth
        elif quantity == "Ay":
            return extract_Ay
        else:
            raise ValueError(f"Unknown quantity {quantity}")


def set_up_solver(
    reaction: jitr.reactions.Reaction,
    Elab: float,
    angle_rad_constraint: np.array,
    angle_rad_vis: np.array,
    lmax: int,
):
    """
    Set up the solver for the reaction.

    Parameters
    ----------
    reaction :
        Reaction information.
    Elab : float
        Laboratory energy.
    angle_rad_constraint : np.array
        Angles to compare to experiment (rad).
    angle_rad_vis : np.array
        Angles to visualize on (rad)
    lmax : int
        Maximum angular momentum.

    Returns
    -------
    tuple
        constraint and visualization workspaces.
    """

    # get kinematics and parameters for this experiment
    kinematics = reaction.kinematics(Elab)
    interaction_range_fm = jitr.utils.interaction_range(reaction.target.A)
    a = interaction_range_fm * kinematics.k + 2 * np.pi
    channel_radius_fm = a / kinematics.k
    # Ns = max(30,jitr.utils.suggested_basis_size(a))
    Ns = jitr.utils.suggested_basis_size(a)
    core_solver = rmatrix.Solver(Ns)

    integral_ws = jitr.xs.elastic.IntegralWorkspace(
        reaction=reaction,
        kinematics=kinematics,
        channel_radius_fm=channel_radius_fm,
        solver=core_solver,
        lmax=lmax,
    )

    constraint_ws = jitr.xs.elastic.DifferentialWorkspace(
        integral_workspace=integral_ws, angles=angle_rad_constraint
    )
    visualization_ws = jitr.xs.elastic.DifferentialWorkspace(
        integral_workspace=integral_ws, angles=angle_rad_vis
    )

    return constraint_ws, visualization_ws, kinematics


def check_angle_grid(angles_rad: np.ndarray, name: str):
    if len(angles_rad.shape) > 1:
        raise ValueError(f"{name} must be 1D, is {len(angles_rad.shape)}D")
    if angles_rad[0] < 0 or angles_rad[-1] > np.pi:
        raise ValueError(f"{name} must be on [0,pi)")
