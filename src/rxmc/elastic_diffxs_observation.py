from typing import Type

import numpy as np

from exfor_tools.distribution import Distribution
import jitr

from .observation import Observation, FixedCovarianceObservation

DEFAULT_LMAX = 20


class ElasticDifferentialXSObservation:
    """
    Observation for elastic differential cross sections.

    This class dynamically inherits from `Observation` or any other
    derived class of `Observation` based on the `ObservationClass`
    parameter in the initializer. The default behavior is to inherit
    from `Observation`, but users can specify a different subclass, such as
    `FixedCovarianceObservation`, to precompute the covariance matrix inverse
    in cases where the covariance is fixed.

    It is designed to handle elastic differential cross section
    measurements, specifically absolute differential cross sections,
    Rutherford normalized differential cross sections, and analyzing
    powers (Ay).

    Internally, this involves initializing a
    `jitr.xs.elastic.DifferentialWorkspace` which precomputes
    things like boundary conditions to speed up computation of
    observables for a given set of interaction parameter.
    """

    def __init__(
        self,
        measurement: Distribution,
        reaction: jitr.reactions.Reaction,
        quantity: str,
        lmax: int = DEFAULT_LMAX,
        angles_vis: np.ndarray = np.linspace(0.01, 180, 100),
        ObservationClass: Type[Observation] = Observation,
        error_kwargs: dict = None,
    ):
        """
        Initialize a ReactionObservation instance.

        Parameters:
        ----------
        measurements : list[Distribution]
            List of measurements, each containing x, y, and associated errors.
        reactions : list[jitr.reactions.Reaction]
            List of reactions associated with the measurements.
        quantity: str
            The type of quantity to be calculated (e.g., "dXS/dA",
            "dXS/dRuth", "Ay").
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
        self.quantity = quantity
        self.lmax = lmax

        self.angles_vis = angles_vis
        angles_rad_vis = angles_vis * np.pi / 180
        check_angle_grid(angles_rad_vis, "angles_rad_vis")

        angles_rad_constraint = measurement.x * np.pi / 180
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

        # Convert measurement to correct quantity and normalize to `b/sr`
        norm = self.compute_normalization(measurement)

        # initialize the observation instance
        args, kwargs = set_up_observation(
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
        self.n_data_pts = self._obs.n_data_pts

    def covariance(self, y):
        return self._obs.covariance(y)

    def residual(self, ym):
        return self._obs.residual(ym)

    def num_pts_within_interval(self, interval):
        return self._obs.num_pts_within_interval(interval)

    def compute_normalization(self, measurement):
        if self.quantity == "dXS/dRuth" and measurement.quantity == "dXS/dA":
            if measurement.y_units != "b/Sr":
                raise ValueError(
                    f"Expected y_units to be 'b/Sr', got {measurement.y_units}"
                )
            return self.constraint_workspace.rutherford / 1000
        elif self.quantity == "dXS/dA" and measurement.quantity == "dXS/dRuth":
            return 1 / self.constraint_workspace.rutherford
        elif self.quantity == "dXS/dA" and measurement.quantity == "dXS/dA":
            if measurement.y_units != "b/Sr":
                raise ValueError(
                    f"Expected y_units to be 'b/Sr', got {measurement.y_units}"
                )
            return 1.0
        elif (
            self.quantity in {"dXS/dRuth", "Ay"}
            and self.quantity == measurement.quantity
        ):
            if measurement.y_units != "no-dim":
                raise ValueError(
                    f"Expected y_units to be 'no-dim', got {measurement.y_units}"
                )
            return 1.0
        else:
            if self.quantity != measurement.quantity:
                raise ValueError(
                    f"Quantity mismatch: {self.quantity} != {measurement.quantity}"
                )
            return 1.0


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
    kinematics = reaction.kinematics(Elab)
    interaction_range_fm = jitr.utils.interaction_range(reaction.target.A)
    a = interaction_range_fm * kinematics.k + 2 * np.pi
    channel_radius_fm = a / kinematics.k
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


def set_up_observation(
    ObservationClass: Type[Observation],
    measurement: Distribution,
    normalization: np.ndarray,
    x=None,
    include_sys_norm_err=True,
    include_sys_offset_err=True,
    include_statistical_err=True,
):
    r"""
    Set up an `Observation` from a `Distribution`.

    This function converts a `Distribution` into an `Observation` object,
    normalizing the y-values and handling systematic and statistical errors.

    Parameters
    ----------
    ObservationClass : Type[Observation]
        The class type of the `Observation` to be created. It must be a
        subclass of `Observation`, such as `FixedCovarianceObservation`.
    measurement : Distribution
        The measurement data containing x, y, and associated errors.
    normalization : np.ndarray
        Normalization factor which the y-values and all dimensionfull
        errors (e.g. all others than normalization errors) will be divided by.
    include_sys_norm_err : bool, optional
        Whether to include systematic normalization errors, by default True.
    include_sys_offset_err : bool, optional
        Whether to include systematic offset errors, by default True.
    include_statistical_err : bool, optional
        Whether to include statistical errors, by default True.
    x : np.ndarray, optional
        Custom x-values to use instead of the measurement's x-values.
    Returns
    -------
        args: tuple
            args for the ObservationClass initializer
        kwargs: dict
            kwargs for the ObservationClass initializer
    """

    x = x if x is not None else measurement.x
    y = measurement.y / normalization
    y_stat_err = (
        measurement.statistical_err / normalization
        if include_statistical_err
        else np.zeros_like(y)
    )

    y_sys_err_offset = None
    y_sys_err_offset_mask = None
    if include_sys_offset_err:
        y_sys_err_offset = measurement.systematic_offset_err / normalization
        # check if systematic errors are common to all angles
        if not np.isscalar(y_sys_err_offset) or not np.all(y_sys_err_offset == y_sys_err_offset[0]):
        # Get unique elements in the array
            unique_elements, inverse_indices = np.unique(
                y_sys_err_offset, return_inverse=True
            )

            # Generate a list of boolean masks
            y_sys_err_offset_mask = [
                inverse_indices == i for i in np.arange(len(unique_elements))
            ]
            y_sys_err_offset = unique_elements

        else:
            y_sys_err_offset = y_sys_err_offset if np.isscalar(y_sys_err_offset) else y_sys_err_offset[0]
            y_sys_err_offset_mask = None

    y_sys_err_normalization = None
    y_sys_err_normalization_mask = None
    if include_sys_norm_err:
        y_sys_err_normalization = measurement.systematic_norm_err
        # check if systematic errors are common to all angles
        ratio = y_sys_err_normalization
        if not np.isscalar(ratio) or np.allclose(ratio, ratio[0]):
            # Get unique elements in the array
            unique_elements, inverse_indices = np.unique(
                y_sys_err_normalization, return_inverse=True
            )

            # Generate a list of boolean masks
            y_sys_err_normalization_mask = [
                inverse_indices == i for i in np.arange(len(unique_elements))
            ]
            y_sys_err_normalization = unique_elements
        else:
            y_sys_err_normalization_mask = None
            y_sys_err_normalization = ratio if np.isscalar(ratio) else ratio[0]

    if ObservationClass is Observation:
        # If the base class is Observation, we can directly return it
        args = (x, y)
        kwargs = {
            "y_stat_err": y_stat_err,
            "y_sys_err_offset": y_sys_err_offset,
            "y_sys_err_offset_mask": y_sys_err_offset_mask,
            "y_sys_err_normalization": y_sys_err_normalization,
            "y_sys_err_normalization_mask": y_sys_err_normalization_mask,
        }
        return args, kwargs
    elif ObservationClass is FixedCovarianceObservation:
        if include_sys_norm_err:
            raise ValueError(
                "FixedCovarianceObservation does not support systematic normalization errors."
            )
        covariance = np.diag(y_stat_err**2)
        if y_sys_err_offset is not None and include_sys_offset_err:
            covariance += np.outer(y_sys_err_offset, y_sys_err_offset)
        args = (x, y, covariance)
        return args, {}
    else:
        # if a new ObservationClass is written, a case for it must be added here
        raise NotImplementedError(
            f"ObservationClass {ObservationClass} is not implemented."
        )


def check_angle_grid(angles_rad: np.ndarray, name: str):
    if len(angles_rad.shape) > 1:
        raise ValueError(f"{name} must be 1D, is {len(angles_rad.shape)}D")
    if angles_rad[0] < 0 or angles_rad[-1] > np.pi:
        raise ValueError(f"{name} must be on [0,pi)")
