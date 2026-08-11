"""
Observation: a leaf of experimental data.

An :class:`Observation` is *pure data* — an independent variable ``x``, a
dependent variable ``y``, and the statistical error ``y_stat_err`` on ``y``.  It
emits **only** its statistical diagonal, via :meth:`Observation.statistical_term`.

Every *correlated* mode — a dataset's own normalisation/offset systematic, an
unknown-noise term, a cross-dataset coupling — is an **explicit**
:class:`~rxmc.covariance.Term` added at constraint-assembly time (see
:mod:`rxmc.covariance`).  Nothing correlated is hidden in a default.  This is a
deliberate change from the old behaviour, which folded normalisation/offset into
``Observation.covariance`` silently; there is no compatibility path that
re-folds them.

An observation may still *carry* its reported systematic magnitudes
(``y_sys_err_normalization``, ``y_sys_err_offset``) as **inert metadata** —
provenance from the measurement.  :meth:`Observation.systematic_terms` turns them
into fixed-magnitude rank-one terms, but only when the caller asks: pass its
result via ``Constraint(extra_terms=...)``.
"""

import numpy as np

from .covariance import DenseTerm, normalization_term, offset_term, statistical_term


def _store_error_spec(value, n, name):
    """Validate/normalize a systematic-error spec: None, scalar, or shape (n,)."""
    if value is None:
        return None
    if np.ndim(value) == 0:
        return float(value)
    v = np.asarray(value, dtype=float)
    if v.shape != (n,):
        raise ValueError(
            f"{name} must be a scalar or have shape ({n},), got shape {v.shape}"
        )
    return v


class Observation:
    """Experimental data: ``x``, ``y``, and the statistical error on ``y``.

    Parameters
    ----------
    x : np.ndarray
        Independent-variable data.
    y : np.ndarray
        Dependent-variable data, same shape as ``x``.
    y_stat_err : np.ndarray, optional
        Statistical (uncorrelated) error on ``y``.  Defaults to zeros.
    y_sys_err_normalization : float or np.ndarray, optional
        Reported *fractional* (dimensionless) normalisation uncertainty —
        inert metadata; see :meth:`systematic_terms`.
    y_sys_err_offset : float or np.ndarray, optional
        Reported *absolute* offset uncertainty, in the same units as ``y`` —
        inert metadata; see :meth:`systematic_terms`.
    label : str, optional
        Human-readable dataset identifier used in error messages.

    Attributes
    ----------
    x, y : np.ndarray
        The data.
    y_stat_err : np.ndarray
        Statistical error on ``y`` (raw, not squared).
    y_sys_err_normalization : float or np.ndarray or None
        Fractional normalisation uncertainty (dimensionless).
    y_sys_err_offset : float or np.ndarray or None
        Absolute offset uncertainty (units of ``y``).
    label : str or None
        Human-readable dataset identifier.
    n_data_pts : int
        Number of data points.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        y_stat_err=None,
        y_sys_err_normalization=None,
        y_sys_err_offset=None,
        label=None,
    ):
        self.label = label
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        if self.x.shape != self.y.shape:
            raise ValueError(
                "x and y must have the same shape, they have shapes "
                f"{self.x.shape} and {self.y.shape}"
            )
        self.n_data_pts = self.x.shape[0]

        y_stat_err = y_stat_err if y_stat_err is not None else np.zeros_like(self.y)
        y_stat_err = np.asarray(y_stat_err, dtype=float)
        if y_stat_err.shape != self.y.shape:
            raise ValueError(
                "y_stat_err must have the same shape as y, "
                f"it has shape {y_stat_err.shape} and y has shape {self.y.shape}"
            )
        self.y_stat_err = y_stat_err

        self.y_sys_err_normalization = _store_error_spec(
            y_sys_err_normalization, self.n_data_pts, "y_sys_err_normalization"
        )
        self.y_sys_err_offset = _store_error_spec(
            y_sys_err_offset, self.n_data_pts, "y_sys_err_offset"
        )

    def statistical_term(self, support) -> DenseTerm:
        """The always-on, genuinely uncorrelated statistical diagonal.

        Parameters
        ----------
        support : np.ndarray
            Indices of this observation's block in the stacked vector.

        Returns
        -------
        DenseTerm
            ``diag(y_stat_err**2)`` on ``support``.
        """
        return statistical_term(support, self.y_stat_err)

    def systematic_terms(self, support) -> list:
        """This dataset's reported correlated systematics as fixed rank-one terms.

        Opt-in — **not** added to any covariance automatically.  Pass the result
        via ``Constraint(extra_terms=[*obs.systematic_terms(support), ...])``.
        Zero magnitudes are skipped, so an observation without reported
        systematics yields an empty list.

        Parameters
        ----------
        support : np.ndarray
            Indices of this observation's block in the stacked vector.

        Returns
        -------
        list of Term
            The absolute offset mode (``outer(omega, omega)``) first, then the
            fractional, prediction-scaled normalisation mode
            (``eta**2 * outer(ym, ym)``).
        """
        terms = []
        if self.y_sys_err_offset is not None and np.any(
            np.asarray(self.y_sys_err_offset) != 0.0
        ):
            terms.append(offset_term(support, magnitude=self.y_sys_err_offset))
        if self.y_sys_err_normalization is not None and np.any(
            np.asarray(self.y_sys_err_normalization) != 0.0
        ):
            terms.append(
                normalization_term(support, magnitude=self.y_sys_err_normalization)
            )
        return terms

    def num_pts_within_interval(
        self,
        ylow: np.ndarray,
        yhigh: np.ndarray,
        xlim=None,
    ):
        """Number of points of ``y`` that fall within ``[ylow, yhigh)``.

        Useful for empirical-coverage diagnostics.

        Parameters
        ----------
        ylow, yhigh : np.ndarray
            Interval bounds, same shape as ``y``.
        xlim : tuple, optional
            ``(x_min, x_max)`` range to restrict the count.
        """
        mask = np.ones_like(self.y, dtype=bool)
        if xlim is not None:
            xlow, xhigh = xlim
            mask = np.logical_and(self.x >= xlow, self.x < xhigh)
        return int(
            np.sum(
                np.logical_and(
                    self.y[mask] >= ylow[mask],
                    self.y[mask] < yhigh[mask],
                )
            )
        )
