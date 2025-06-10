import numpy as np

import jitr

from .likelihood_model import LikelihoodModel
from .elastic_diffxs_model import ElasticDifferentialXSModel
from .reaction_observation import ReactionObservation
from .constraint import Constraint

DEFAULT_LMAX = 20

# TODO allow for writing and reading the precomputed workspaces to/from disk


class ElasticDifferentialXSConstraint(Constraint):
    """
    A `Constraint` for elastic differential cross sections and analyzing
    powers.

    For each `ReactionObservation`, this class also precomputes (the
    model-independent) quantities needed for the calculation of differential
    cross sections (dXS/dA, dXS/dRuth, Ay) for elastic scattering reactions.

    This precomputation is handled by jitr.xs.elastic.DifferentialWorkspace,
    which is initialized with the reaction and laboratory energy.

    """

    def __init__(
        self,
        observations: list[ReactionObservation],
        physical_model: ElasticDifferentialXSModel,
        likelihood_model: LikelihoodModel,
        quantity: str,
        lmax: int = DEFAULT_LMAX,
        angles_rad_vis: np.ndarray = np.linspace(0, np.pi, 100),
    ):
        """
        Params:
        ----------
        observations: list[ReactionObservation]
            List of ReactionObservations, each containing the reaction,
            laboratory energy, and measurements.
        physical_model: ElasticDifferentialXSModel
            A callable that takes a
            `jitr.xs.elastic.DifferentialWorkspace` and a tuple of params,
            and returns the model predictions for the corresponding
            observation.
        likelihood_model: LikelihoodModel
            The model used to calculate the likelihood of the model parameters
            given the observed data.
        quantity: str
            The type of quantity to be calculated (e.g., "dXS/dA",
            "dXS/dRuth", "Ay").
        lmax: int
            Maximum angular momentum, defaults to 20.
        angles_rad_vis: np.ndarray
            Array of angles in radians for visualization.
        """
        super().__init__(
            observations=observations,
            physical_model=physical_model,
            likelihood_model=likelihood_model,
        )
        self.quantity = quantity
        self.lmax = lmax
        self.constraint_workspaces = []
        self.visualization_workspaces = []
        self.angles_rad_vis = angles_rad_vis

        check_angle_grid(angles_rad_vis, "angles_rad_vis")

        for i in range(len(observations)):
            self.constraint_workspaces.append([])
            self.visualization_workspaces.append([])
            for j in range(self.observations[i].n_measurements):
                angles_rad_constraint = self.observations[i].x[j]
                check_angle_grid(angles_rad_constraint, "angles_rad_constraint")
                constraint_ws, vis_ws, kinematics = set_up_solver(
                    reaction=self.observations[i].reaction,
                    Elab=self.observations[i].Elab,
                    angle_rad_constraint=angles_rad_constraint,
                    angle_rad_vis=self.angles_rad_vis,
                    lmax=self.lmax,
                )
                self.constraint_workspaces[i].append(constraint_ws)
                self.visualization_workspaces[i].append(vis_ws)

    def chi2_observation(self, obs, workspaces, *params):
        """
        Calculate the chi-squared statistic (or Mahalanobis distance) between
        the model prediction, given the parameters, and the observed data for
        a single observation.

        Parameters:
        ----------
        obs : ReactionObservation
            The observed data that the model will attempt to reproduce.
        workspaces : list[DifferentialWorkspace]
            The precomputed workspaces corresponding to the observation.
        params : tuple
            The parameters of the physical model

        Returns:
        -------
        float
            The chi-squared statistic.
        """

    def logpdf_observation(self, obs, workspaces, *params):
        """
        Calculate the log probability density function (logpdf) that the model
        predictions, given the parameters, reproduce the observed data for a
        single observation.

        Parameters:
        ----------
        obs : ReactionObservation
            The observed data that the model will attempt to reproduce.
        workspaces : list[DifferentialWorkspace]
            The precomputed workspaces corresponding to the observation.
        params : tuple
            The parameters of the physical model

        Returns:
        -------
        float
            The log probability density of the observation given the
            parameters.
        """
        # TODO concatenate model predictions into 1d array and pass into
        # LikelihoodModel
        ym = np.concat([self.model_observation(obs, ws, *params) for ws in workspaces])

    def model_observation(self, obs, workspaces, *params):
        """
        Compute the model output for a single observation, given params.

        Parameters:
        ----------
        obs : ReactionObservation
            The observed data that the model will attempt to reproduce.
        workspaces : list[DifferentialWorkspace]
            The precomputed workspaces corresponding to the observation.
        params : tuple
            The parameters of the physical model

        Returns:
        -------
        float
            The model prediction for the observed data.
        """
        return [self.physical_model(obs, ws, *params) for ws in workspaces]

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
        return [
            self.model_observation(obs, workspaces)
            for obs, workspaces in zip(self.observations, self.constraint_workspaces)
        ]

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
            self.logpdf_observation(obs, ws, *params)
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
