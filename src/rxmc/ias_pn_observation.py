from typing import Type

import jitr
import numpy as np
from exfor_tools.distribution import Distribution
from pint import UnitRegistry

from .observation import Observation
from .observation_from_measurement import check_angle_grid, set_up_observation

# Create a unit registry
ureg = UnitRegistry()

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
        x: np.ndarray,
        y: np.ndarray,
        Elab: float,
        reaction: jitr.reactions.Reaction,
        ExIAS: float,
        y_units: str,
        y_stat_err=None,
        y_sys_err_normalization=None,
        y_sys_err_offset=None,
        dataset_label: str | None = None,
        lmax: int = DEFAULT_LMAX,
        angles_vis: np.ndarray = np.linspace(0.01, 180, 100),
        ObservationClass: Type[Observation] = Observation,
        error_kwargs: dict = None,
        wavelengths_beyond_range: float = 2.0,
        zeros_per_node: int = 5,
    ):
        """
        Initialize a Observation instance for the (p,n) IAS reaction.

        Parameters:
        ----------
        x : np.ndarray
            Measured angle grid in degrees.
        y : np.ndarray
            Measured differential cross section data.
        Elab : float
            Laboratory energy of the incoming proton (MeV).
        reaction : jitr.reactions.Reaction
            Reaction information.
        ExIAS : float
            Excitation energy of the IAS in the residual nucleus (MeV).
        y_units : str
            Units of the supplied `y` values.
        y_stat_err : np.ndarray, optional
            Statistical errors associated with `y`.
        y_sys_err_normalization : float or array-like, optional
            Systematic normalization error(s) associated with `y`.
        y_sys_err_offset : float or array-like, optional
            Systematic offset error(s) associated with `y`.
        dataset_label : str, optional
            Human-readable dataset identifier used in error messages.
        lmax: int
            Maximum angular momentum
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
        wavelengths_beyond_range: float
            Number of wavelengths beyond the interaction range to set the channel radius.
        zeros_per_node: int
            Number of zeros of the basis functions per node in the R-matrix solver.
        """
        if not issubclass(ObservationClass, Observation):
            raise ValueError("ObservationClass must be a subclass of Observation")
        self.reaction = reaction
        self.lmax = lmax
        self.subentry = dataset_label
        self.angle_units = ureg.radian
        self.quantity = "dXS/dA"

        self.angles_vis = angles_vis
        angles_rad_vis = np.deg2rad(angles_vis)
        check_angle_grid(angles_rad_vis, "angles_rad_vis")

        angles_rad_constraint = np.deg2rad(x)
        label = dataset_label or "dataset"
        check_angle_grid(
            angles_rad_constraint,
            f"x values for {label}",
        )

        # set up workspaces to precompute things for the solver
        # for quick evaluation of observables
        constraint_ws, vis_ws, kinematics_entrance, kinematics_exit = set_up_solver(
            reaction=self.reaction,
            Elab=Elab,
            ExIAS=ExIAS,
            angle_rad_constraint=angles_rad_constraint,
            angle_rad_vis=angles_rad_vis,
            lmax=self.lmax,
        )
        self.constraint_workspace = constraint_ws
        self.visualization_workspace = vis_ws

        self.y_units = ureg.barn / ureg.steradian
        measurement_unit = 1 * ureg(y_units)
        if not measurement_unit.check(self.y_units):
            raise ValueError(
                f"Expected measurement_unit to be dimensionally "
                f"compatible with 'b/sr', got {y_units}"
            )

        norm = 1.0 / measurement_unit.to(self.y_units).magnitude

        # initialize the observation instance
        args, kwargs, y_stat_err = set_up_observation(
            ObservationClass,
            x=angles_rad_constraint,
            y=y,
            y_stat_err=y_stat_err,
            y_sys_err_normalization=y_sys_err_normalization,
            y_sys_err_offset=y_sys_err_offset,
            dataset_label=dataset_label,
            normalization=norm,
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

    @classmethod
    def from_measurement(
        cls,
        measurement: Distribution,
        reaction: jitr.reactions.Reaction,
        ExIAS: float,
        lmax: int = DEFAULT_LMAX,
        angles_vis: np.ndarray = np.linspace(0.01, 180, 100),
        ObservationClass: Type[Observation] = Observation,
        error_kwargs: dict = None,
        wavelengths_beyond_range: float = 2.0,
        zeros_per_node: int = 5,
    ):
        return cls(
            x=measurement.x,
            y=measurement.y,
            Elab=measurement.Einc,
            reaction=reaction,
            ExIAS=ExIAS,
            y_units=measurement.y_units,
            y_stat_err=measurement.statistical_err,
            y_sys_err_normalization=measurement.systematic_norm_err,
            y_sys_err_offset=measurement.systematic_offset_err,
            dataset_label=getattr(measurement, "subentry", None),
            lmax=lmax,
            angles_vis=angles_vis,
            ObservationClass=ObservationClass,
            error_kwargs=error_kwargs,
            wavelengths_beyond_range=wavelengths_beyond_range,
            zeros_per_node=zeros_per_node,
        )


def set_up_solver(
    reaction: jitr.reactions.Reaction,
    Elab: float,
    ExIAS: float,
    angle_rad_constraint: np.array,
    angle_rad_vis: np.array,
    lmax: int,
    wavelengths_beyond_range: float = 2.0,
    zeros_per_node: int = 5,
):
    """
    Set up the solver for the reaction.

    Parameters
    ----------
    reaction :
        Reaction information.
    Elab : float
        Laboratory energy of the incoming proton (MeV).
    ExIAS : float
        Excitation energy of the IAS in the residual nucleus (MeV).
    angle_rad_constraint : np.array
        Angles to compare to experiment (rad).
    angle_rad_vis : np.array
        Angles to visualize on (rad)
    lmax : int
        Maximum angular momentum.
    wavelengths_beyond_range : float
        Number of wavelengths beyond the interaction
        range to set the channel radius.
    zeros_per_node : int
        Number of zeros of the basis functions per
        node in the R-matrix solver.

    Returns
    -------
    tuple
        constraint and visualization workspaces.
    """
    kinematics_entrance = reaction.kinematics(Elab=Elab)
    kinematics_exit = reaction.kinematics_exit(
        kinematics_entrance, residual_excitation_energy=ExIAS
    )

    k = kinematics_entrance.k
    interaction_range_fm = jitr.utils.interaction_range(reaction.target.A) + 2
    a = k * interaction_range_fm + wavelengths_beyond_range * 2 * np.pi
    channel_radius_fm = a / k
    N = jitr.utils.suggested_basis_size(a, zeros_per_node)
    core_solver = jitr.rmatrix.Solver(N)

    constraint_workspace = jitr.xs.quasielastic_pn.Workspace(
        reaction,
        kinematics_entrance,
        kinematics_exit,
        core_solver,
        angle_rad_constraint,
        lmax,
        channel_radius_fm,
        tmatrix_abs_tol=1e-8,
    )

    visualization_workspace = jitr.xs.quasielastic_pn.Workspace(
        reaction,
        kinematics_entrance,
        kinematics_exit,
        core_solver,
        angle_rad_vis,
        lmax,
        channel_radius_fm,
        tmatrix_abs_tol=1e-8,
    )

    return (
        constraint_workspace,
        visualization_workspace,
        kinematics_entrance,
        kinematics_exit,
    )
