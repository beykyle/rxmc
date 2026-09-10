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

import copy

import numpy as np

from .covariance import offset_term, statistical_term, systematic_term
from .transforms import as_transform


def _as_point_mask(mask, n) -> np.ndarray:
    """Coerce a point mask to a boolean array of shape ``(n,)``."""
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (n,):
        raise ValueError(f"mask must have shape ({n},), got {mask.shape}")
    return mask


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
    transform : Transform or callable, optional
        Parameter-free *comparison-space* transform (see :mod:`rxmc.transforms`).
        Pass **raw** ``y``: the observation stores ``y = transform(y_raw)`` and
        propagates ``y_stat_err`` by the delta method, and the
        :class:`~rxmc.constraint.Constraint` applies the same transform to the
        model prediction — so ``transform=rxmc.transforms.log`` compares in log
        space with the model written once, in physical space.
    mask : array_like of bool, optional
        Which points are *active* in a likelihood (default all).  Inactive
        points stay in the block (supports/terms are authored over all points)
        but are excluded from the residual; use :meth:`masked` /
        :meth:`masked_where` to derive fit/held-out views.

    Attributes
    ----------
    x, y : np.ndarray
        The data, ``y`` in comparison space.
    y_raw, y_stat_err_raw : np.ndarray
        ``y`` and its statistical error as given (physical space).
    y_stat_err : np.ndarray
        Statistical error on ``y`` in comparison space (raw, not squared).
    transform : Transform
        The comparison-space transform (identity by default).
    mask : np.ndarray of bool
        Active points.
    identity : Observation
        The root observation this one is a view of.  Views made by
        :meth:`masked` share it, so anything routing by observation (e.g.
        :func:`rxmc.transforms.per_observation_scaling`) treats a masked view
        and its root as the same dataset.
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
        transform=None,
        mask=None,
    ):
        self.label = label
        self.identity = self
        self.x = np.asarray(x)
        y_raw = np.asarray(y, dtype=float)
        if self.x.shape != y_raw.shape:
            raise ValueError(
                "x and y must have the same shape, they have shapes "
                f"{self.x.shape} and {y_raw.shape}"
            )
        self.n_data_pts = self.x.shape[0]

        y_stat_err = y_stat_err if y_stat_err is not None else np.zeros_like(y_raw)
        y_stat_err = np.asarray(y_stat_err, dtype=float)
        if y_stat_err.shape != y_raw.shape:
            raise ValueError(
                "y_stat_err must have the same shape as y, "
                f"it has shape {y_stat_err.shape} and y has shape {y_raw.shape}"
            )

        self.transform = as_transform(transform)
        if self.transform.params:
            raise ValueError(
                "an Observation's comparison-space transform must be parameter-free"
            )
        self.y_raw = y_raw
        self.y_stat_err_raw = y_stat_err
        self.mask = (
            np.ones(self.n_data_pts, dtype=bool)
            if mask is None
            else _as_point_mask(mask, self.n_data_pts)
        )
        if self.transform.is_identity:
            self.y = y_raw
            self.y_stat_err = y_stat_err
            self._abs_jacobian = None
        else:
            # |t'(y_raw)|: the delta-method factor, reused by log_jacobian and
            # systematic_terms
            # inactive points may be non-finite (inf * 0 -> nan); the guard
            # below only inspects the active ones
            with np.errstate(invalid="ignore", divide="ignore"):
                self._abs_jacobian = np.abs(self.transform.derivative(y_raw))
                self.y = self.transform(y_raw)
                self.y_stat_err = self._abs_jacobian * y_stat_err
        self._check_finite()

        self.y_sys_err_normalization = _store_error_spec(
            y_sys_err_normalization, self.n_data_pts, "y_sys_err_normalization"
        )
        self.y_sys_err_offset = _store_error_spec(
            y_sys_err_offset, self.n_data_pts, "y_sys_err_offset"
        )

    def _check_finite(self):
        """Reject non-finite comparison-space values at the active points."""
        if self.transform.is_identity:
            return
        bad = self.mask & ~(np.isfinite(self.y) & np.isfinite(self.y_stat_err))
        if np.any(bad):
            raise ValueError(
                f"transform {self.transform.name!r} is not finite at "
                f"{int(bad.sum())} active data point(s) of dataset "
                f"{self.label or 'observation'!r} (e.g. non-positive y under a "
                "log transform); mask or drop those points"
            )

    # ------------------------------------------------------------------
    # Masks (active points)
    # ------------------------------------------------------------------

    @property
    def n_active(self) -> int:
        """Number of active (unmasked) points."""
        return int(self.mask.sum())

    def masked(self, mask, label=None):
        """A shallow copy of this observation with a new point mask.

        No data or pre-computed workspaces are rebuilt: the copy shares them and
        only changes which points enter a likelihood.
        """
        new = copy.copy(self)
        new.mask = _as_point_mask(mask, self.n_data_pts)
        if label is not None:
            new.label = label
        new._check_finite()
        return new

    def masked_where(self, predicate, label=None):
        """:meth:`masked` with ``mask = predicate(x)`` (points where it is True)."""
        return self.masked(np.asarray(predicate(self.x), dtype=bool), label=label)

    # ------------------------------------------------------------------
    # Comparison-space bookkeeping
    # ------------------------------------------------------------------

    @property
    def log_jacobian(self) -> float:
        r"""``sum(log |t'(y_raw)|)`` over the active points.

        The log-Jacobian of the comparison-space transform: a constant in the
        parameters, needed only to compare marginal likelihoods (log Z) across
        different comparison spaces (``log Z_raw = log Z_transformed +
        log_jacobian``).  Zero for the identity.
        """
        if self.transform.is_identity:
            return 0.0
        return float(np.sum(np.log(self._abs_jacobian[self.mask])))

    def _raw_prediction(self, ym):
        """Invert the comparison-space transform on a prediction."""
        if self.transform.is_identity:
            return np.asarray(ym, dtype=float)
        inv = self.transform.inverse
        if inv is None:
            raise ValueError(
                f"transform {self.transform.name!r} has no inverse; cannot map "
                "predictions back to physical space"
            )
        return inv(ym)

    # ------------------------------------------------------------------
    # Covariance terms
    # ------------------------------------------------------------------

    def statistical_term(self, support=None):
        """The always-on, genuinely uncorrelated statistical diagonal.

        Parameters
        ----------
        support : np.ndarray, optional
            Indices of this observation's block in the stacked vector
            (``None`` for a single-observation constraint).

        Returns
        -------
        Term
            ``diag(y_stat_err**2)`` on ``support`` (comparison space).
        """
        return statistical_term(self.y_stat_err, support=support)

    def systematic_terms(self, support=None) -> list:
        """This dataset's reported correlated systematics as fixed rank-one terms.

        Opt-in — **not** added to any covariance automatically.  Pass the result
        via ``Constraint(extra_terms=[*obs.systematic_terms(), ...])``.
        Zero magnitudes are skipped, so an observation without reported
        systematics yields an empty list.  Magnitudes are reported in physical
        space and propagated to the comparison space by the delta method
        (``|t'| * omega`` for an offset, ``|t'(ym_raw)| * eta * ym_raw`` for a
        normalisation).

        Parameters
        ----------
        support : np.ndarray, optional
            Indices of this observation's block in the stacked vector.  ``None``
            binds the terms to the whole constraint, which is right only for a
            single-observation constraint; in a multi-observation constraint
            they then fail loudly with a shape error, so pass the block's
            support there (see :func:`rxmc.covariance.stacked_supports`).

        Returns
        -------
        list of Term
            The absolute offset mode (``outer(omega, omega)``) first, then the
            fractional, prediction-scaled normalisation mode
            (``eta**2 * outer(ym, ym)``).
        """

        def reported(spec):
            if spec is None or not np.any(np.asarray(spec) != 0.0):
                return None
            return np.broadcast_to(np.asarray(spec, dtype=float), (self.n_data_pts,))

        # the identity transform has unit Jacobian and trivial inverse, so the
        # delta-method expressions below reduce to the plain magnitudes.  The
        # two modes are linearised at different points on purpose: the offset
        # is an error on the *data*, so it is propagated at y_raw; the
        # normalisation multiplies the *prediction*, so its mode eta * ym_raw is
        # propagated at ym_raw.
        t = self.transform
        terms = []
        omega = reported(self.y_sys_err_offset)
        if omega is not None:
            if not t.is_identity:
                omega = self._abs_jacobian * omega
            terms.append(offset_term(magnitude=omega, support=support))
        eta = reported(self.y_sys_err_normalization)
        if eta is not None:
            if not t.is_identity and t.inverse is None:
                raise ValueError(
                    f"transform {t.name!r} has no inverse; the normalisation "
                    "systematic needs the physical-space prediction"
                )

            def basis(c):
                ym_raw = self._raw_prediction(c.ym)
                return eta * ym_raw * np.abs(t.derivative(ym_raw))

            terms.append(systematic_term(None, basis, support=support))
        return terms

    def num_pts_within_interval(
        self,
        ylow: np.ndarray,
        yhigh: np.ndarray,
        xlim=None,
    ):
        """Number of active points of ``y`` that fall within ``[ylow, yhigh)``.

        Useful for empirical-coverage diagnostics.  ``ylow``/``yhigh`` are in
        comparison space and indexed over *all* points of the block.

        Parameters
        ----------
        ylow, yhigh : np.ndarray
            Interval bounds, same shape as ``y``.
        xlim : tuple, optional
            ``(x_min, x_max)`` range to restrict the count.
        """
        mask = self.mask.copy()
        if xlim is not None:
            xlow, xhigh = xlim
            mask &= np.logical_and(self.x >= xlow, self.x < xhigh)
        return int(
            np.sum(
                np.logical_and(
                    self.y[mask] >= ylow[mask],
                    self.y[mask] < yhigh[mask],
                )
            )
        )
