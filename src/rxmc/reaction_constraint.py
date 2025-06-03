import numpy as np

from .constraint import FixedCovarianceConstraint
from .model import Model
from exfor_tools.distribution import Distribution


def covariance_from_distribution(
    measurement: Distribution,
    normalize=None,
    include_sys_norm_err=True,
    include_sys_offset_err=True,
    include_sys_gen_err=True,
):
    x = np.copy(measurement.x)
    y = np.copy(measurement.y)
    stat_err_y = np.copy(measurement.statistical_err)
    sys_err_norm = np.copy(measurement.systematic_norm_err)
    sys_err_offset = np.copy(measurement.systematic_offset_err)
    sys_err_general = np.copy(measurement.general_systematic_err)

    if normalize is not None:
        y /= normalize
        stat_err_y /= normalize
        sys_err_general /= normalize
        if sys_err_offset > 0:
            sys_err_general += sys_err_offset * np.ones_like(x) / normalize
            sys_err_offset = 0

    covariance = np.diag(stat_err_y**2)
    if include_sys_norm_err:
        covariance += np.outer(y, y) * sys_err_norm**2
    if include_sys_offset_err:
        n = y.shape[0]
        covariance += np.ones((n, n)) * sys_err_offset
    if include_sys_gen_err:
        covariance += np.outer(sys_err_general, sys_err_general)

    return y, covariance


class ReactionConstraint(FixedCovarianceConstraint):
    """
    Represents the constraint determined by a AngularDistribution,
    with the appropriate covariance matrix given statistical
    and systematic errors.
    """

    def __init__(
        self,
        quantity: str,
        measurement: Distribution,
        model: Model,
        normalize=None,
        include_sys_norm_err=True,
        include_sys_offset_err=True,
        include_sys_gen_err=True,
    ):
        """
        Params:
        quantity : str
            The name of the measured quantity
        measurement : Distribution
            An object containing the measured values along
            with their statistical and systematic errors.
        model : ElasticModel
            The model to predict the quantity
        normalize : float, optional
            A value to normalize the measurements and errors. If None, no
            normalization is applied. Default is None.
        include_sys_norm_err : bool, optional
            If True, includes systematic normalization error in the covariance
            matrix. Default is True.
        include_sys_offset_err : bool, optional
            If True, includes systematic offset error in the covariance matrix.
            Default is True.
        include_sys_gen_err : bool, optional
            If True, includes general systematic error in the covariance
            matrix. Default is True.

        """
        self.quantity = quantity
        self.subentry = measurement.subentry
        y, covariance = covariance_from_distribution(
            measurement,
            normalize,
            include_sys_norm_err,
            include_sys_offset_err,
            include_sys_gen_err,
        )

        super().__init__(y, covariance, model)
