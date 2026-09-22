"""
Comparisons and constraints: the declarations a likelihood is built from.

A :class:`Comparison` is the unit a residual is formed on: one dataset, one
model bound to its grid, and one parameter-free comparison ``space`` (e.g.
``log``) applied to both.  A :class:`Constraint` is a tuple of comparisons
plus the covariance terms, the likelihood functional, the tempering weight
and the active-point masks; it is the maximal block of mutually correlated
data, and constraints are independent of each other.  Each comparison is one
block of the constraint's stacked covariance.

Masks live on the constraint, not on the comparison or the data, so
:meth:`Constraint.masked`, :meth:`Constraint.masked_where` and
:meth:`Constraint.complement` return constraints sharing every
``Comparison``, ``Term`` and ``Parameter`` object with the original.  A
held-out problem built from ``complement()`` therefore has the same
parameter columns as the fit.

Nothing here walks the parameter graph; :class:`~rxmc.problem.Problem` does
that once.  The checks here need only the constraint itself: comparisons are
distinct, every term's ``on`` resolves inside the constraint, and an
array-valued term has the right shape for its support.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable

import numpy as np

from .data import Dataset
from .likelihood import Gaussian, Likelihood
from .model import Model, Predictor
from .terms import Term
from .transforms import as_transform, identity

__all__ = ["Comparison", "Constraint"]


@dataclass(eq=False, frozen=True)
class Comparison:
    """One dataset compared with one model in one space.

    Parameters
    ----------
    data : Dataset
    model : Model
        Bound to ``data.x`` with ``data.meta`` at construction.
    space : Transform or callable, optional
        Parameter-free comparison transform applied to the data once and to
        every prediction.  ``y = space(data.y)``, ``y_err`` by the delta
        method, and the log-Jacobian are constants.  Non-finite values are
        allowed here (the point may be masked); the compile step checks the
        active points and names the comparison.
    """

    data: Dataset
    model: Model
    space: Any = identity
    predictor: Predictor = field(init=False, repr=False)
    y: np.ndarray = field(init=False, repr=False)
    y_err: np.ndarray = field(init=False, repr=False)
    log_jac: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.data, Dataset):
            raise TypeError(f"data must be a Dataset, got {type(self.data).__name__}")
        if not isinstance(self.model, Model):
            raise TypeError(f"model must be a Model, got {type(self.model).__name__}")
        held = (self.data.meta or {}).get("quantity")
        predicted = getattr(self.model, "quantity", None)
        if held is not None and predicted is not None and held != predicted:
            raise ValueError(
                f"dataset {self.data.label or 'dataset'!r} holds {held!r} but the "
                f"model predicts {predicted!r}: convert the data "
                "(from_measurement(..., quantity=)) or use a matching model"
            )
        space = as_transform(self.space)
        if space.params:
            raise ValueError(
                "a comparison space must be parameter-free; put parametric "
                "transforms on the model (model | transform)"
            )
        set_ = object.__setattr__
        set_(self, "space", space)
        set_(self, "predictor", self.model.bind(self.data.x, self.data.meta))
        if space.is_identity:
            set_(self, "y", self.data.y)
            set_(self, "y_err", self.data.y_err)
            set_(self, "log_jac", np.zeros(self.data.n))
            return
        with np.errstate(all="ignore"):
            # a derivative may come back as a scalar; the Jacobian is per point
            jac = np.broadcast_to(np.abs(space.derivative(self.data.y)), (self.data.n,))
            set_(self, "y", space(self.data.y))
            set_(self, "y_err", jac * self.data.y_err)
            set_(self, "log_jac", np.log(jac))

    @property
    def n(self) -> int:
        return self.data.n

    def predict(self, *values) -> np.ndarray:
        """The prediction on the data grid, in comparison space."""
        ym = self.predictor(*values)
        return ym if self.space.is_identity else self.space(ym)

    def log_jacobian(self, mask=None) -> float:
        r"""``sum(log |space'(data.y)|)`` over the active points.

        A constant in the parameters, needed only to compare marginal
        likelihoods across comparison spaces (``log Z_raw = log Z_transformed
        + log_jacobian``).  Zero for the identity.
        """
        lj = self.log_jac if mask is None else self.log_jac[np.asarray(mask, bool)]
        return float(np.sum(lj))

    def reported_terms(self) -> list[Term]:
        """The dataset's reported systematics as fixed rank-one modes, ``on=self``.

        Opt-in: nothing is folded into a covariance automatically.  The
        absolute offset mode comes first, then the fractional normalisation
        mode, each skipped when its magnitude is zero.  Both are propagated
        to the comparison space by the delta method: the offset is an error on
        the *data* and is linearised at the data; the normalisation multiplies
        the *prediction* and is linearised at the prediction, which needs the
        space's inverse.  A scalar normalisation error gives a mode that is a
        function of the prediction alone, so it also has a value on a new grid
        (:func:`~rxmc.predictive.grid_draws`); an offset, or a per-point
        normalisation, is defined only at the measured points.
        """
        d, t = self.data, self.space
        terms = []
        omega = _reported(d.offset_err, d.n)
        if omega is not None:
            if not t.is_identity:
                omega = np.abs(t.derivative(d.y)) * omega
            terms.append(Term(omega, kind="mode", on=self))
        eta = _reported(d.norm_err, d.n)
        if eta is not None and np.ndim(d.norm_err) == 0:
            # a scalar keeps the mode a function of the prediction alone, so
            # it can be evaluated on any grid (grid_draws), like
            # T.normalization(magnitude=)
            eta = float(d.norm_err)
        if eta is not None:
            if t.is_identity:
                terms.append(Term(lambda c: eta * c.ym, kind="mode", on=self))
            else:
                inv = t.inverse
                if inv is None:
                    raise ValueError(
                        f"comparison space {t.name!r} has no inverse; the "
                        "normalisation systematic needs the physical-space prediction"
                    )

                def basis(c, eta=eta, t=t, inv=inv):
                    ym_raw = inv(c.ym)
                    return eta * ym_raw * np.abs(t.derivative(ym_raw))

                terms.append(Term(basis, kind="mode", on=self))
        return terms

    def __repr__(self):
        label = self.data.label or "dataset"
        return f"Comparison({label!r}, {self.model!r}, space={self.space.name})"


def _reported(spec, n):
    """A reported magnitude as a length-``n`` array, or ``None`` when absent/zero."""
    if spec is None or not np.any(np.asarray(spec) != 0.0):
        return None
    return np.broadcast_to(np.asarray(spec, dtype=float), (n,)).copy()


def _as_mask(mask, n) -> np.ndarray:
    # a copy, so the caller reusing its array can't move the mask
    mask = np.array(mask, dtype=bool)
    if mask.shape != (n,):
        raise ValueError(f"mask must have shape ({n},), got {mask.shape}")
    return mask


@dataclass(eq=False, frozen=True)
class Constraint:
    """The maximal block of mutually correlated data: one likelihood.

    Parameters
    ----------
    comparisons : iterable of Comparison
        Distinct comparisons; each is one block of the stacked covariance.
    terms : iterable of Term, optional
        Covariance contributions, authored in comparison space.
    likelihood : Likelihood, optional
        Functional of ``(d2, logdet, n)``; :class:`~rxmc.likelihood.Gaussian`
        by default.
    weight : float, optional
        Tempering: multiplies this constraint's log-likelihood only.
    statistical : bool, optional
        Add each comparison's ``y_err`` diagonal (default).  ``False`` composes
        the whole covariance from ``terms``.
    masks : sequence of bool arrays, optional
        Active points, one array per comparison; ``None`` means all active.
    """

    comparisons: Any
    terms: Any = ()
    likelihood: Likelihood = field(default_factory=Gaussian)
    weight: float = 1.0
    statistical: bool = True
    masks: Any = None
    offsets: tuple = field(init=False, repr=False)
    active: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        set_ = object.__setattr__
        comps = tuple(self.comparisons)
        for c in comps:
            if not isinstance(c, Comparison):
                raise TypeError(f"comparisons must be Comparison objects, got {c!r}")
        if len({id(c) for c in comps}) != len(comps):
            raise ValueError("comparisons must be distinct objects")
        if not comps:
            raise ValueError("a constraint needs at least one comparison")
        terms = tuple(self.terms)
        for t in terms:
            if not isinstance(t, Term):
                raise TypeError(f"terms must be Term objects, got {type(t).__name__}")
        if not isinstance(self.likelihood, Likelihood):
            raise TypeError("likelihood must be a Likelihood")
        weight = float(self.weight)
        if not (np.isfinite(weight) and weight >= 0):
            raise ValueError(f"weight must be finite and non-negative, got {weight}")
        ns = [c.n for c in comps]
        starts = np.concatenate([[0], np.cumsum(ns)[:-1]]).astype(int)
        offsets = tuple(slice(int(s), int(s + n)) for s, n in zip(starts, ns))
        if self.masks is None:
            masks = None
            active = np.arange(int(sum(ns)))
        else:
            masks = tuple(self.masks)
            if len(masks) != len(comps):
                raise ValueError(
                    f"masks must have one entry per comparison ({len(comps)})"
                )
            masks = tuple(_as_mask(m, n) for m, n in zip(masks, ns))
            active = np.concatenate(
                [np.arange(o.start, o.stop)[m] for o, m in zip(offsets, masks)]
            ).astype(int)
        set_(self, "comparisons", comps)
        set_(self, "terms", terms)
        set_(self, "weight", weight)
        set_(self, "masks", masks)
        set_(self, "offsets", offsets)
        set_(self, "active", active)
        for t in terms:  # eager: every on= resolves here, arrays have the right shape
            rows = self.support(t.on)
            if not callable(t.fn) and t.fn.shape != t.expected_shape(len(rows)):
                raise ValueError(
                    f"{t.kind} term expects shape {t.expected_shape(len(rows))} on "
                    f"its support, got {t.fn.shape}"
                )

    # -- structure ----------------------------------------------------------

    @property
    def n_total(self) -> int:
        return int(sum(c.n for c in self.comparisons))

    @property
    def n_active(self) -> int:
        return int(self.active.size)

    @property
    def log_jacobian(self) -> float:
        """Sum of the comparisons' log-Jacobians over the active points."""
        masks = self.masks or (None,) * len(self.comparisons)
        return float(sum(c.log_jacobian(m) for c, m in zip(self.comparisons, masks)))

    def support(self, on) -> np.ndarray:
        """The stacked rows a term's ``on`` resolves to, in constraint order."""
        if on is None:
            return np.arange(self.n_total)
        if isinstance(on, (Comparison, Dataset)):
            targets = [on]
        else:
            try:
                targets = list(on)
            except TypeError:
                raise TypeError(
                    f"on= must reference comparisons or datasets, got {on!r}"
                ) from None
        keep = np.zeros(len(self.comparisons), dtype=bool)
        for target in targets:
            if isinstance(target, Comparison):
                hits = [i for i, c in enumerate(self.comparisons) if c is target]
            elif isinstance(target, Dataset):
                hits = [i for i, c in enumerate(self.comparisons) if c.data is target]
            else:
                raise TypeError(
                    f"on= must reference comparisons or datasets, got {target!r}"
                )
            if not hits:
                raise ValueError(
                    f"term on={target!r} does not reference a comparison of this "
                    "constraint"
                )
            keep[hits] = True
        return np.concatenate(
            [np.arange(o.start, o.stop) for o, k in zip(self.offsets, keep) if k]
        ).astype(int)

    # -- masked views ---------------------------------------------------------

    def masked(self, masks) -> "Constraint":
        """The same constraint with new active-point masks."""
        return replace(self, masks=masks)

    def masked_where(self, predicate: Callable) -> "Constraint":
        """Active where ``predicate(comparison.data.x)`` is true, per comparison."""
        return self.masked([predicate(c.data.x) for c in self.comparisons])

    def complement(self) -> "Constraint":
        """Every inactive point active, and vice versa."""
        if self.masks is None:
            return self.masked([np.zeros(c.n, dtype=bool) for c in self.comparisons])
        return self.masked([~m for m in self.masks])

    def __repr__(self):
        return (
            f"Constraint({len(self.comparisons)} comparison(s), {len(self.terms)} "
            f"term(s), {type(self.likelihood).__name__}, n_active={self.n_active})"
        )
