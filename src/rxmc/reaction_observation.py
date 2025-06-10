import numpy as np

from exfor_tools.distribution import Distribution
import jitr

from .observation import Observation


class ReactionObservation(Observation):
    """
    A `ReactionObservation` represents a single experiment, and may contain
    multiple data sets corresponding to different reactions or energies, which
    may be correlated in the `LikelihoodModel`. An example could be cross
    sections from an EXFOR entry, which contains multiple subentries for
    different reactions but has a COMMON field that indicates `ERR-SYS`
    for all subentries - this would imply that the `LikelihoodModel`
    should account for the correlation (e.g. by using a
    `LikelihoodWithSystematicError` in which the attribute
    `y_sys_err_bias` corresponds to the COMMON `ERR-SYS` value in the
    EXFOR entry).

    The order of `reactions` and `Elab` must match the order of `measurements`.
    """

    def __init__(
        self,
        reactions: list[jitr.reaction.Reaction],
        Elab: list[float],
        measurements: list[Distribution],
        y_sys_err_bias: float = 0,
        y_sys_err_offset: float = 0,
    ):
        """
        Initialize a ReactionObservation with measurements and reactions.

        Parameters:
        ----------
        measurements : list[Distribution]
            List of measurements, each containing x, y, and associated errors.
        reactions : list[jitr.reaction.Reaction]
            List of reactions associated with the measurements.
        Elab : list[float]
            Laboratory energies for the reactions.
        y_sys_err_bias : float
            Systematic error bias for the y values, default is 0.
        y_sys_err_offset : float, optional
            Systematic error offset for the y values, default is 0.0.
        """
        super().__init__(
            x=np.array([m.x for m in measurements]),
            y=np.array([m.y for m in measurements]),
            y_stat_err=np.array([m.stat_err for m in measurements]),
            y_sys_err_bias=y_sys_err_bias,
            y_sys_err_offset=y_sys_err_offset,
        )
        self.n_measurements = len(measurements)
        self.reactions = reactions
        self.Elab = Elab
        self.measurements = measurements

        if len(reactions) != self.n_measurements:
            raise ValueError("Number of reactions must match number of measurements.")
        if len(Elab) != self.n_measurements:
            raise ValueError(
                "Number of laboratory energies must match number of measurements."
            )
