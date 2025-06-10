import numpy as np

import jitr

from .physical_model import PhysicalModel
from .likelihood_model import LikelihoodModel
from .observation import Observation
from .constraint import Constraint

DEFAULT_LMAX = 20


class ElasticDifferentialXSConstraint(Constraint):
    """
    For each Observation, this class also precomputes (the model-independent)
    quantities needed for the calculation of differential cross
    sections (dXS/dA, dXS/dRuth, Ay) for elastic scattering reactions.

    This precomputation is handled by jitr.xs.elastic.DifferentialWorkspace,
    which is initialized with the reaction and laboratory energy.
    """

    def __init__(
        self,
        observations: list[Observation],
        reactions: list[jitr.reactions.Reaction],
        Elab: list[float],
        physical_model: PhysicalModel,
        likelihood_model: LikelihoodModel,
        quantity: str,
        lmax: int = DEFAULT_LMAX,
        angles_rad_vis: np.ndarray = np.linspace(0, np.pi, 100),
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
            lmax: Maximum angular momentum, defaults to 20.
        """
        super().__init__(
            observations=observations,
            physical_model=physical_model,
            likelihood_model=likelihood_model,
        )
        self.quantity = quantity
        self.reactions = reactions
        self.Elab = Elab
        self.lmax = lmax
        self.constraint_workspaces = []
        self.visualization_workspaces = []
        self.kinematics = []

        for i in range(len(observations)):
            if self.Elab[i] <= 0:
                raise ValueError("Elab must be positive.")

            check_angle_grid(self.observations[i].x, f"observation[{i}].x")
            check_angle_grid(angles_rad_vis, "angles_rad_vis")

            constraint_ws, vis_ws, kin = set_up_solver(
                self.reactions[i],
                self.Elab[i],
                self.observations[i].x,
                angles_rad_vis,
                self.lmax,
            )
            self.constraint_workspaces[i] = constraint_ws
            self.visualization_workspaces[i] = vis_ws
            self.kinematics[i] = kin

    def model(self, *params):
        """
        Compute the model output for each observation, given params.

        Parameters:
        ----------
        params : tuple
            The parameters of the physical model

        Returns:
        -------
        float
            The model predictions for the observed data.
        """
        return sum(
            self.physical_model(obs, ws, *params)
            for obs, ws in zip(self.observations, self.constraint_workspaces)
        )

    def logpdf(self, params, likelihood_params=None):
        """
        Calculate the log probability density function (logpdf) that the model
        predictions, given the parameters, reproduce the observed data.

        Parameters:
        ----------
        observation : Observation
            The observed data that the model will attempt to reproduce.
        params : tuple
            The parameters of the physical model
        likelihood_params : tuple, optional
            Additional parameters for the likelihood model, if any.


        Returns:
        -------
        float
            The log probability density of the observation given the
            parameters.
        """
        return sum(
            self.likelihood.logpdf(
                obs, self.physical_model(obs, ws, *params), *likelihood_params
            )
            for obs, ws in zip(self.observations, self.constraint_workspaces)
        )

    def chi2(self, params, likelihood_params=None):
        """
        Calculate the chi-squared statistic (or Mahalanobis distance) between
        the model prediction, given the parameters, and the observed data.

        Parameters:
        ----------
        params : tuple
            The parameters of the physical model
        likelihood_params : tuple, optional
            Additional parameters for the likelihood model, if any.

        Returns:
        -------
        float
            The chi-squared statistic.
        """
        return sum(
            self.likelihood.chi2(
                obs, self.physical_model(obs, ws, *params), *likelihood_params
            )
            for obs, ws in zip(self.observations, self.constraint_workspaces)
        )

    def logpdf_and_ym(self, params, likelihood_params=None):
        """
        Calculate the log probability density function (logpdf) that the model
        predictions, given the parameters, reproduce the observed data, and
        returns it along with the model predictions.

        Parameters:
        ----------
        params : tuple
            The parameters of the physical model
        likelihood_params : tuple, optional
            Additional parameters for the likelihood model, if any.

        Returns:
        -------
        float
            The log probability density of the observation given the
            parameters.
        list
            The model predictions for the observed data.
        """
        ym = []
        logpdf = 0.0
        for obs, ws in zip(self.observations, self.constraint_workspaces):
            y_pred = self.physical_model(obs, ws, *params)
            ym.append(y_pred)
            logpdf += self.likelihood.logpdf(obs, y_pred, *likelihood_params)

        return logpdf, ym


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


def extract_dXS_dA(
    xs: jitr.xs.elastic.ElasticXS, ws: jitr.xs.elastic.DifferentialWorkspace
) -> np.ndarray:
    """Extracts dXS/dA in b/Sr"""
    return xs.dsdo / 1000


def extract_dXS_dRuth(
    xs: jitr.xs.elastic.ElasticXS, ws: jitr.xs.elastic.DifferentialWorkspace
) -> np.ndarray:
    """Extracts dXS/dRuth"""
    return xs.dsdo / ws.rutherford


def extract_Ay(
    xs: jitr.xs.elastic.ElasticXS, ws: jitr.xs.elastic.DifferentialWorkspace
) -> np.ndarray:
    """Extracts Ay"""
    return xs.Ay
