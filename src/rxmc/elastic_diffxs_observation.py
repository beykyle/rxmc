from typing import Type

import jitr
import numpy as np
from exfor_tools.distribution import Distribution
from pint import UnitRegistry

from .observation import Observation
from .observation_from_measurement import (
    check_angle_grid,
    set_up_observation,
)

# Create a unit registry
ureg = UnitRegistry()


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
        x: np.ndarray,
        y: np.ndarray,
        Elab: float,
        reaction: jitr.reactions.Reaction,
        quantity: str,
        measurement_quantity: str,
        y_units: str,
        y_stat_err=None,
        y_sys_err_normalization=None,
        y_sys_err_offset=None,
        dataset_label: str | None = None,
        lmax: int = DEFAULT_LMAX,
        wavelengths_beyond_range=2.0,
        zeros_per_node=5,
        angles_vis: np.ndarray = np.linspace(0.01, 180, 100),
        ObservationClass: Type[Observation] = Observation,
        error_kwargs: dict = None,
        compound_correction: np.ndarray = None,
    ):
        """
        Initialize a ReactionObservation instance.

        Parameters:
        ----------
        x : np.ndarray
            Measured angle grid in degrees.
        y : np.ndarray
            Measured observable values.
        Elab : float
            Laboratory energy of the measurement in MeV.
        quantity: str
            The type of quantity to be calculated (e.g., "dXS/dA",
            "dXS/dRuth", "Ay").
        measurement_quantity: str
            The quantity represented by the supplied `y` values.
        y_units: str
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
            Maximum angular momentum, defaults to 20.
        wavelengths_beyond_range: float
            Number of wavelengths beyond the interaction range to set the channel radius.
        zeros_per_node: int
            Number of zeros of the basis functions per node in the R-matrix solver.
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
        compound_correction: np.ndarray
            Optional array of the compound contribution to the differential
            cross section in mb/sr to be added to the calculated cross section
            before comparing to data.
        """
        if not issubclass(ObservationClass, Observation):
            raise ValueError("ObservationClass must be a subclass of Observation")
        self.reaction = reaction
        self.quantity = quantity
        self.lmax = lmax
        self.subentry = dataset_label
        self.angle_units = ureg.radian
        self.compound_correction = compound_correction

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
        constraint_ws, vis_ws, kinematics = set_up_solver(
            reaction=self.reaction,
            Elab=Elab,
            angle_rad_constraint=angles_rad_constraint,
            angle_rad_vis=angles_rad_vis,
            lmax=self.lmax,
            wavelengths_beyond_range=wavelengths_beyond_range,
            zeros_per_node=zeros_per_node,
        )
        self.constraint_workspace = constraint_ws
        self.visualization_workspace = vis_ws

        # Convert measurement to correct quantity and normalize to `b/sr`
        norm, normalized_y_units = self.calculate_normalization(
            measurement_quantity, y_units
        )
        self.y_units = normalized_y_units

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
        quantity: str,
        lmax: int = DEFAULT_LMAX,
        wavelengths_beyond_range=2.0,
        zeros_per_node=5,
        angles_vis: np.ndarray = np.linspace(0.01, 180, 100),
        ObservationClass: Type[Observation] = Observation,
        error_kwargs: dict = None,
        compound_correction: np.ndarray = None,
    ):
        return cls(
            x=measurement.x,
            y=measurement.y,
            Elab=measurement.Einc,
            reaction=reaction,
            quantity=quantity,
            measurement_quantity=measurement.quantity,
            y_units=measurement.y_units,
            y_stat_err=measurement.statistical_err,
            y_sys_err_normalization=measurement.systematic_norm_err,
            y_sys_err_offset=measurement.systematic_offset_err,
            dataset_label=getattr(measurement, "subentry", None),
            lmax=lmax,
            wavelengths_beyond_range=wavelengths_beyond_range,
            zeros_per_node=zeros_per_node,
            angles_vis=angles_vis,
            ObservationClass=ObservationClass,
            error_kwargs=error_kwargs,
            compound_correction=compound_correction,
        )

    def calculate_normalization(
        self, measurement_quantity: str, measurement_y_units: str
    ):
        # Determine the xs_unit based on self.quantity
        xs_unit = ureg.barn / ureg.steradian
        rutherford_unit = ureg.millibarn / ureg.steradian
        if self.quantity == "dXS/dA":
            y_unit = xs_unit
        elif self.quantity in {"dXS/dRuth", "Ay"}:
            y_unit = ureg.dimensionless
        else:
            raise ValueError(f"Unrecognized quantity: {self.quantity}")

        # Process different cases based on the quantity types
        if self.quantity == "dXS/dRuth" and measurement_quantity == "dXS/dA":
            measurement_unit = 1 * ureg(measurement_y_units)
            if not measurement_unit.check(xs_unit):
                raise ValueError(
                    "Expected measurement_unit to be dimensionally compatible "
                    f"with 'b/Sr', got {measurement_y_units}"
                )

            conversion_factor = 1.0 / measurement_unit.to(rutherford_unit).magnitude
            return self.constraint_workspace.rutherford * conversion_factor, y_unit

        elif self.quantity == "dXS/dA" and measurement_quantity == "dXS/dRuth":
            conversion_factor = 1.0 / rutherford_unit.to(y_unit).magnitude
            return conversion_factor / self.constraint_workspace.rutherford, y_unit

        elif self.quantity == "dXS/dA" and measurement_quantity == "dXS/dA":
            measurement_unit = 1 * ureg(measurement_y_units)
            if not measurement_unit.check(y_unit):
                raise ValueError(
                    "Expected measurement_unit to be dimensionally compatible "
                    f"with 'b/Sr', got {measurement_y_units}"
                )

            return 1.0 / measurement_unit.to(y_unit).magnitude, y_unit

        elif (
            self.quantity in {"dXS/dRuth", "Ay"}
            and self.quantity == measurement_quantity
        ):
            if measurement_y_units != "no-dim":
                raise ValueError(
                    f"Expected measurement_unit to be 'no-dim', got {measurement_y_units}"
                )
            return 1.0, y_unit

        else:
            if self.quantity != measurement_quantity:
                raise ValueError(
                    f"Quantity mismatch: {self.quantity} != {measurement_quantity}"
                )


def set_up_solver(
    reaction: jitr.reactions.Reaction,
    Elab: float,
    angle_rad_constraint: np.ndarray,
    angle_rad_vis: np.ndarray,
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
        Laboratory energy.
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
    kinematics = reaction.kinematics(Elab)
    k = kinematics.k
    interaction_range_fm = jitr.utils.interaction_range(reaction.target.A) + 2
    a = k * interaction_range_fm + wavelengths_beyond_range * 2 * np.pi
    channel_radius_fm = a / k
    N = jitr.utils.suggested_basis_size(a, zeros_per_node)
    core_solver = jitr.rmatrix.Solver(N)

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
