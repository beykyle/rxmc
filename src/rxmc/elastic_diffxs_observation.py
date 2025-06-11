import numpy as np

from exfor_tools.distribution import Distribution
import jitr

from .observation import Observation

DEFAULT_LMAX = 20


class ElasticDifferentialXSObservation(Observation):
    """
    A `ReactionObservation` represents a single experiment, and may contain
    multiple data sets corresponding to different reactions or energies, which
    may be correlated in the `LikelihoodModel`. An example could be cross
    sections from an EXFOR entry, which contains multiple subentries for
    different reactions but has a COMMON field that indicates `ERR-SYS`
    for all subentries - this would imply that the `LikelihoodModel`
    should account for the correlation (e.g. by using a
    `LikelihoodWithSystematicError` in which the attribute
    `y_sys_err_normalization` corresponds to the COMMON `ERR-SYS` value in the
    EXFOR entry).

    The order of `reactions` must match the order of `measurements`.

    Also included in `ReactionObservation` are the precomputed workspaces
    for the differential cross section calculations, which are set up
    using `jitr.xs.elastic.DifferentialWorkspace`. This is done do that,
    for any physical model, the compute time to get a log likelihood for a
    given set of model parameters is minimized.
    """

    def __init__(
        self,
        measurements: list[Distribution],
        reactions: list[jitr.reactions.Reaction],
        quantity: str,
        y_sys_err_normalization: float = 0,
        y_sys_err_offset: np.ndarray = 0,
        lmax: int = DEFAULT_LMAX,
        angles_vis: np.ndarray = np.linspace(0, np.pi, 100),
    ):
        """
        Initialize a ReactionObservation.

        Parameters:
        ----------
        measurements : list[Distribution]
            List of measurements, each containing x, y, and associated errors.
        reactions : list[jitr.reactions.Reaction]
            List of reactions associated with the measurements.
        quantity: str
            The type of quantity to be calculated (e.g., "dXS/dA",
            "dXS/dRuth", "Ay").
        y_sys_err_normalization : float
            Systematic error normalization for the y values, default is 0.
        y_sys_err_offset : float, optional
            Global systematic error offset for the y values, default is 0
        lmax: int
            Maximum angular momentum, defaults to 20.
        angles_vis: np.ndarray
            Array of angles in degrees for visualization.
        """
        self.measurements = measurements
        size = sum(len(m.x) for m in measurements)
        self.n_measurements = len(measurements)
        self.reactions = reactions
        self.quantity = quantity
        self.lmax = lmax
        self.angles_vis = angles_vis
        angles_rad_vis = angles_vis * np.pi / 180

        check_angle_grid(angles_rad_vis, "angles_rad_vis")

        if len(reactions) != self.n_measurements:
            raise ValueError("Number of reactions must match number of measurements.")

        measurement_quantity = measurements[0].quantity
        if any(m.quantity != measurement_quantity for m in measurements):
            raise ValueError(
                "All measurements must have the same quantity, "
                f"but got {[m.quantity for m in measurements]}"
            )

        self.visualization_workspaces = []
        self.constraint_workspaces = []
        x = []
        y = []
        y_stat_err = []
        meas_err_normalization = []
        meas_err_offset = []
        for rxn, m in zip(self.reactions, self.measurements):
            angles_rad_constraint = m.x * np.pi / 180
            check_angle_grid(
                angles_rad_constraint, f"x values for subentry: {m.subentry}"
            )
            constraint_ws, vis_ws, kinematics = set_up_solver(
                reaction=rxn,
                Elab=m.Einc,
                angle_rad_constraint=angles_rad_constraint,
                angle_rad_vis=angles_rad_vis,
                lmax=self.lmax,
            )
            self.constraint_workspaces.append(constraint_ws)
            self.visualization_workspaces.append(vis_ws)

            # convert measurement to correct quantity and normalize to `b/sr`
            if self.quantity == "dXS/dRuth" and m.quantity == "dXS/dA":
                if m.y_units == "b/sr":
                    norm = self.constraint_workspaces[-1].rutherford / 1000
                else:
                    raise ValueError(
                        f"Measurement units mismatch: {m.y_units} != b/sr for subentry {m.subentry}"
                    )
            elif self.quantity == "dXS/dA" and m.quantity == "dXS/dRuth":
                norm = 1000.0 / self.constraint_workspaces[-1].rutherford
            else:
                norm = 1.0
                if self.quantity != m.quantity:
                    raise ValueError(
                        f"Quantity mismatch: {self.quantity} != {m.quantity}"
                    )

            x.append(angles_rad_constraint)
            y.append(m.y / norm)
            y_stat_err.append(m.y_stat_err / norm)
            meas_err_offset.append(m.y_sys_err_offset / norm)
            #  fractional error in normalization not normalized
            meas_err_normalization.append(m.y_sys_err_normalization)

        super().__init__(
            x=np.concatenate(x),
            y=np.concatenate(y),
            y_stat_err=np.concatenate(y_stat_err),
            y_sys_err_offset=y_sys_err_offset,
        )


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
    core_solver = jitr.rmatrix.Solver(Ns)

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
