from typing import Type

import jitr
import numpy as np
from exfor_tools.distribution import Distribution
from pint import UnitRegistry

from .observation import FixedCovarianceObservation, Observation
from .observation_from_measurement import check_angle_grid, set_up_observation

DEFAULT_LMAX = 20


class IsobaricAnalogPNObservation:
    """
    Observation for (p,n) isobaric analog state (IAS) reactions.

    This class dynamically inherits from `Observation` or any other
    derived class of `Observation` based on the `ObservationClass`
    parameter in the initializer. The default behavior is to inherit
    from `Observation`, but users can specify a different subclass, such as
    `FixedCovarianceObservation`, to precompute the covariance matrix inverse
    in cases where the covariance is fixed.

    It is designed to handle (p,n) IAS reaction measurements in differential cross
    section form.

    Internally, this involves initializing a jitr.xs.quasielastic_pn.Workspace
    which precomputes things like boundary conditions to speed up computation of
    observables for a given set of interaction parameters.
    """

    def __init__(
        self,
        measurement: Distribution,
        reaction: jitr.reactions.Reaction,
        lmax: int = DEFAULT_LMAX,
        angles_vis: np.ndarray = np.linspace(0.01, 180, 100),
        ObservationClass: Type[Observation] = Observation,
        error_kwargs: dict = None,
    ):
        """
        Initialize a Observation instance for the (p,n) IAS reaction.

        Parameters:
        ----------
        measurements : list[Distribution]
            List of measurements, each containing x, y, and associated errors.
        reactions : list[jitr.reactions.Reaction]
            List of reactions associated with the measurements.
        lmax: int
            Maximum angular momentum, defaults to 20.
        angles_vis: np.ndarray
            Array of angles in degrees for visualization.
        ObservationClass: Type[Observation]
            The base class Type that this instance will inherit from;
            must be a subclass of `Observation`. Defaults to the base
            class `Observation`, but the user can supply any other subclass.
            For example, if one wants the covariance to be precomputed one
            can supply `FixedCovarianceObservation` instead here.
        error_kwargs: dict
            Additional keyword arguments for error handling.
        """
        if not issubclass(ObservationClass, Observation):
            raise ValueError("ObservationClass must be a subclass of Observation")
        self.reaction = reaction
        self.lmax = lmax
        self.subentry = measurement.subentry
        self.angle_units = ureg.radian

        self.angles_vis = angles_vis
        angles_rad_vis = np.deg2rad(angles_vis)
        check_angle_grid(angles_rad_vis, "angles_rad_vis")

        angles_rad_constraint = np.deg2rad(measurement.x)
        check_angle_grid(
            angles_rad_constraint,
            f"x values for subentry: {measurement.subentry}",
        )

        # set up workspaces to precompute things for the solver
        # for quick evaluation of observables
        constraint_ws, vis_ws, kinematics = set_up_solver(
            reaction=self.reaction,
            Elab=measurement.Einc,
            angle_rad_constraint=angles_rad_constraint,
            angle_rad_vis=angles_rad_vis,
            lmax=self.lmax,
        )
        self.constraint_workspace = constraint_ws
        self.visualization_workspace = vis_ws

        # TODO handle units
        self.y_units = y_units

        # initialize the observation instance
        args, kwargs, y_stat_err = set_up_observation(
            ObservationClass,
            measurement=measurement,
            normalization=norm,
            x=angles_rad_constraint,
            **error_kwargs if error_kwargs is not None else {},
        )

        # Create an instance of the chosen ObservationClass
        self._obs = ObservationClass(*args, **kwargs)

        self.x = self._obs.x
        self.y = self._obs.y
        self.y_stat_err = y_stat_err
        self.n_data_pts = self._obs.n_data_pts

    def covariance(self, y):
        return self._obs.covariance(y)

    def residual(self, ym):
        return self._obs.residual(ym)

    def num_pts_within_interval(self, interval):
        return self._obs.num_pts_within_interval(interval)


def set_up_solver(
    reaction: jitr.reactions.Reaction,
    Elab: float,
    ExIAS: float,
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
    kinematics_entrance = reaction.kinematics(Elab=Elab)
    kinematics_exit = reaction.kinematics_exit(
        kinematics_entrance, residual_excitation_energy=ExIAS
    )

    interaction_range_fm = jitr.utils.interaction_range(reaction.target.A)
    a = interaction_range_fm * kinematics.k + 2 * np.pi
    channel_radius_fm = a / kinematics.k
    Ns = jitr.utils.suggested_basis_size(a)
    core_solver = jitr.rmatrix.Solver(Ns)

    constraint_workspace = jitr.xs.quasielastic_pn.Workspace(
        reaction,
        kinematics_entrance,
        kinematics_exit,
        core_solver,
        angle_rad_constraint,
        lmax,
        channel_radius_fm,
        tmatrix_abs_tol: float = 1e-8,
    )

    visualization_workspace = jitr.xs.quasielastic_pn.Workspace(
        reaction,
        kinematics_entrance,
        kinematics_exit,
        core_solver,
        angle_rad_vis,
        lmax,
        channel_radius_fm,
        tmatrix_abs_tol: float = 1e-8,
    )

    return constraint_workspace, visualization_workspace, kinematics_entrance, kinematics_exit


def check_angle_grid(angles_rad: np.ndarray, name: str):
    if len(angles_rad.shape) > 1:
        raise ValueError(f"{name} must be 1D, is {len(angles_rad.shape)}D")
    if angles_rad[0] < 0 or angles_rad[-1] > np.pi:
        raise ValueError(f"{name} must be on [0,pi)")
