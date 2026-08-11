"""
Observation class for elastic differential cross sections.

:class:`ElasticDifferentialXSObservation` is an :class:`~rxmc.observation.Observation`
that sets up a ``jitr`` :class:`jitr.xs.elastic.DifferentialWorkspace` to pre-compute
boundary conditions and Rutherford cross sections.  It carries statistical error
only; correlated systematics are composed as :class:`~rxmc.covariance.Term` s in the
:class:`~rxmc.constraint.Constraint`.
"""

import jitr
import numpy as np
from exfor_tools.distribution import Distribution
from pint import UnitRegistry

from .observation import Observation
from .observation_from_measurement import check_angle_grid, normalized_error_kwargs

# Create a unit registry
ureg = UnitRegistry()


DEFAULT_LMAX = 20


class ElasticDifferentialXSObservation(Observation):
    """
    Observation for elastic differential cross sections.

    This is an :class:`~rxmc.observation.Observation` (statistical error only): it
    inherits ``statistical_term`` and ``num_pts_within_interval``.  Any correlated
    systematic — the dataset's reported normalisation/offset, or a fixed covariance
    (:class:`~rxmc.covariance.DenseTerm`) — is composed explicitly as an
    ``extra_terms`` entry in the :class:`~rxmc.constraint.Constraint`.

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
        compound_correction: np.ndarray = None,
    ):
        """
        Parameters
        ----------
        x : np.ndarray
            Measured angle grid in degrees.
        y : np.ndarray
            Measured observable values.
        Elab : float
            Laboratory energy in MeV.
        reaction : jitr.reactions.Reaction
            Reaction system definition.
        quantity : str
            Observable to compute: ``"dXS/dA"``, ``"dXS/dRuth"``, or ``"Ay"``.
        measurement_quantity : str
            Observable represented by the supplied *y* values.
        y_units : str
            Units of the supplied *y* values (e.g. ``"mb/sr"``).
        y_stat_err : np.ndarray, optional
            Statistical errors associated with *y*.
        y_sys_err_normalization : float or np.ndarray, optional
            Reported *fractional* (dimensionless) normalisation uncertainty.
            Retained as inert metadata (see
            :meth:`rxmc.observation.Observation.systematic_terms`); not divided
            by the unit normalisation.
        y_sys_err_offset : float or np.ndarray, optional
            Reported *absolute* offset uncertainty in the same units as *y*.
            Retained as inert metadata, converted to internal units (divided by
            the unit normalisation, per-angle where applicable).
        dataset_label : str, optional
            Human-readable dataset identifier used in error messages.
        lmax : int, optional
            Maximum angular momentum.  Defaults to ``20``.
        wavelengths_beyond_range : float, optional
            Number of wavelengths beyond the interaction range used to set
            the channel radius.  Defaults to ``2.0``.
        zeros_per_node : int, optional
            Number of basis-function zeros per node in the R-matrix solver.
            Defaults to ``5``.
        angles_vis : np.ndarray, optional
            Angle grid in degrees for visualisation.  Defaults to
            ``np.linspace(0.01, 180, 100)``.
        compound_correction : np.ndarray, optional
            Compound-nuclear contribution to dXS/dΩ in mb/sr, added to the
            calculated cross section before comparing to data.
        """
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
        # retained for provenance / manual term recomposition; a scalar, or a
        # per-angle array in the Rutherford-conversion cases
        self.norm = norm

        super().__init__(
            angles_rad_constraint,
            np.asarray(y) / norm,
            label=dataset_label,
            **normalized_error_kwargs(
                norm, y_stat_err, y_sys_err_normalization, y_sys_err_offset
            ),
        )

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
            # rutherford is stored in mb/sr; convert one unit of it to b/sr
            conversion_factor = 1.0 / (1 * rutherford_unit).to(y_unit).magnitude
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
            raise ValueError(
                f"Cannot convert measurement quantity '{measurement_quantity}' "
                f"(units '{measurement_y_units}') to '{self.quantity}'"
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
    Set up ``jitr`` workspaces for a reaction at a given energy.

    Parameters
    ----------
    reaction : jitr.reactions.Reaction
        Reaction system definition.
    Elab : float
        Laboratory energy in MeV.
    angle_rad_constraint : np.ndarray
        Angles in radians for comparison to experiment.
    angle_rad_vis : np.ndarray
        Angles in radians for visualisation.
    lmax : int
        Maximum angular momentum.
    wavelengths_beyond_range : float, optional
        Number of wavelengths beyond the interaction range used to set the
        channel radius.  Defaults to ``2.0``.
    zeros_per_node : int, optional
        Number of basis-function zeros per node in the R-matrix solver.
        Defaults to ``5``.

    Returns
    -------
    constraint_ws : jitr.xs.elastic.DifferentialWorkspace
        Workspace on the constraint angle grid.
    visualization_ws : jitr.xs.elastic.DifferentialWorkspace
        Workspace on the visualisation angle grid.
    kinematics : jitr.reactions.Kinematics
        Kinematic quantities for the reaction.
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
