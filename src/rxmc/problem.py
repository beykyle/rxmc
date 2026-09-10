"""
The compile step: from declarations to the flat interface a sampler wants.

:class:`Problem` is the only place in the package that walks the parameter
graph.  It assigns every distinct :class:`~rxmc.params.Parameter` a slot in
first-seen order (each constraint's predictors, then its terms, then its
likelihood), checks that names are unique, resolves every term's ``on`` to
rows, factors the constant parts of every covariance, and assembles the
prior so that every slot is covered exactly once.  The result exposes what
emcee, dynesty and ``black-box-bayes`` need: ``ndim``, ``names``,
``log_prior``, ``log_likelihood``, ``log_posterior``, ``prior_transform`` and
``sample_prior``.

Prior rules, per slot (see :class:`~rxmc.params.Parameter`):

* ``Parameter(prior=dist)``: the marginal, truncated to ``bounds``;
* finite ``bounds`` and no ``prior``: uniform on the bounds;
* neither: the parameter must appear in exactly one joint block passed as
  ``priors=[(params, joint), ...]``, where ``joint`` exposes
  ``logpdf(values)`` over ``params`` in that order and, optionally,
  ``prior_transform(u)`` and ``rvs(n)``.  A frozen
  ``scipy.stats.multivariate_normal`` gets a whitening unit-cube map for
  free.  A hyperprior is a joint block that includes its hyperparameter.

Nothing user-facing is mutated by compiling.  Compile the same declarations
twice and you get two independent problems.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from scipy import stats

from .constraint import Constraint
from .covariance import StructuredCovariance
from .params import Parameter
from .terms import statistical

__all__ = ["Problem", "ParameterIndex", "CompiledConstraint", "clip_unit_cube"]


def clip_unit_cube(u) -> np.ndarray:
    """``u`` as a float array clipped into the open unit cube.

    Exact ``0.0`` / ``1.0`` map to ``±inf`` under an unbounded marginal's
    ``ppf``; clipping to ``[eps, 1 - eps]`` keeps every ``prior_transform``
    finite.
    """
    u = np.asarray(u, dtype=float)
    eps = np.finfo(float).eps
    return np.clip(u, eps, 1.0 - eps)


# ----------------------------------------------------------------------------
# The index
# ----------------------------------------------------------------------------


class ParameterIndex:
    """Unique parameters in first-seen order, each with a slot."""

    def __init__(self):
        self._slot: dict[Parameter, int] = {}
        self._params: list[Parameter] = []

    def add_all(self, params: Iterable[Parameter]) -> np.ndarray:
        """Register ``params`` and return their gather array (slots in order)."""
        out = []
        for p in params:
            if not isinstance(p, Parameter):
                raise TypeError(f"expected a Parameter, got {p!r}")
            if p not in self._slot:
                self._slot[p] = len(self._params)
                self._params.append(p)
            out.append(self._slot[p])
        return np.asarray(out, dtype=int)

    def slot(self, p: Parameter) -> int:
        try:
            return self._slot[p]
        except KeyError:
            raise KeyError(f"{p!r} is not a parameter of this problem") from None

    def slots(self, params) -> np.ndarray:
        if isinstance(params, Parameter):
            return np.asarray([self.slot(params)], dtype=int)
        return np.asarray([self.slot(p) for p in params], dtype=int)

    @property
    def params(self) -> tuple[Parameter, ...]:
        return tuple(self._params)

    @property
    def names(self) -> list[str]:
        return [p.name for p in self._params]

    @property
    def bounds(self) -> np.ndarray:
        return np.asarray([p.bounds for p in self._params], dtype=float).reshape(-1, 2)

    @property
    def ndim(self) -> int:
        return len(self._params)

    def check_names_unique(self) -> None:
        seen, dup = {}, []
        for p in self._params:
            if p.name in seen:
                dup.append(p.name)
            seen[p.name] = p
        if dup:
            raise ValueError(
                f"duplicate parameter name(s) {sorted(set(dup))}: distinct Parameter "
                "objects with one name would become two chain columns with the same "
                "label.  Pass the SAME object everywhere the value is shared, or "
                "name them apart (prefix= for kernel terms, nu= for StudentT)."
            )


# ----------------------------------------------------------------------------
# The prior
# ----------------------------------------------------------------------------


def _is_frozen_mvn(joint) -> bool:
    return type(joint).__name__ == "multivariate_normal_frozen"


class _Marginal:
    """One slot: a marginal (truncated to bounds) or the uniform on bounds."""

    def __init__(self, p: Parameter, slot: int):
        self.p, self.slot = p, slot
        lo, hi = p.bounds
        self.lo, self.hi = lo, hi
        self.dist = p.prior
        self.bounded = np.isfinite(lo) and np.isfinite(hi)
        if self.dist is None:
            if not self.bounded:
                raise ValueError(
                    f"parameter {p.name!r} has no prior: give it prior=, finite "
                    "bounds, or cover it with a joint block in Problem(priors=)"
                )
            self.c_lo, self.c_hi = 0.0, 1.0
            self.log_norm = np.log(hi - lo)
        else:
            self.c_lo = float(self.dist.cdf(lo)) if np.isfinite(lo) else 0.0
            self.c_hi = float(self.dist.cdf(hi)) if np.isfinite(hi) else 1.0
            mass = self.c_hi - self.c_lo
            if not mass > 0:
                raise ValueError(
                    f"the prior of {p.name!r} has no mass inside its bounds"
                )
            self.log_norm = float(np.log(mass))

    def logpdf(self, v) -> float:
        if v < self.lo or v > self.hi:
            return -np.inf
        if self.dist is None:
            return -self.log_norm
        return float(self.dist.logpdf(v)) - self.log_norm

    def transform(self, u):
        if self.dist is None:
            return self.lo + u * (self.hi - self.lo)
        return self.dist.ppf(self.c_lo + u * (self.c_hi - self.c_lo))


class _Joint:
    """A joint block over several slots."""

    def __init__(self, params: Sequence[Parameter], joint, slots: np.ndarray):
        self.params, self.joint, self.slots = tuple(params), joint, slots
        self.bounds = np.asarray([p.bounds for p in self.params], dtype=float)
        self.bounded = bool(np.any(np.isfinite(self.bounds)))
        self.names = [p.name for p in self.params]
        if not hasattr(joint, "logpdf"):
            raise TypeError(f"joint prior over {self.names} must have logpdf(values)")
        if _is_frozen_mvn(joint):
            self._L = np.linalg.cholesky(np.atleast_2d(joint.cov))
            self._mean = np.atleast_1d(joint.mean)
        else:
            self._L = None

    @property
    def has_transform(self) -> bool:
        return not self.bounded and (
            self._L is not None or hasattr(self.joint, "prior_transform")
        )

    def logpdf(self, values) -> float:
        if self.bounded and (
            np.any(values < self.bounds[:, 0]) or np.any(values > self.bounds[:, 1])
        ):
            return -np.inf
        return float(self.joint.logpdf(values))

    def transform(self, u):
        if self.bounded:
            raise NotImplementedError(
                f"the joint prior over {self.names} is truncated by bounds and has no "
                "unit-cube map; drop the bounds or use a sampler that needs only "
                "log_posterior"
            )
        if self._L is not None:
            return self._mean + self._L @ stats.norm.ppf(u)
        if hasattr(self.joint, "prior_transform"):
            return np.asarray(self.joint.prior_transform(u), dtype=float)
        raise NotImplementedError(
            f"the joint prior over {self.names} has no prior_transform(u); give it "
            "one, or use a sampler that needs only log_posterior"
        )

    def sample(self, n, rng):
        if self.has_transform:
            return np.asarray(
                [self.transform(rng.uniform(size=len(self.params))) for _ in range(n)]
            )
        if not hasattr(self.joint, "rvs"):
            raise NotImplementedError(
                f"the joint prior over {self.names} has neither prior_transform nor rvs"
            )
        draws = np.empty((0, len(self.params)))
        for _ in range(1000):
            d = np.atleast_2d(
                np.asarray(self.joint.rvs(size=n, random_state=rng), dtype=float)
            )
            if d.shape[1] != len(self.params):
                d = d.reshape(-1, len(self.params))
            if self.bounded:
                ok = np.all((d >= self.bounds[:, 0]) & (d <= self.bounds[:, 1]), axis=1)
                d = d[ok]
            draws = np.vstack([draws, d])
            if len(draws) >= n:
                return draws[:n]
        raise RuntimeError(
            f"could not draw from the joint prior over {self.names} inside its bounds"
        )


class _Prior:
    def __init__(self, index: ParameterIndex, priors):
        self.ndim = index.ndim
        self.joints: list[_Joint] = []
        covered: dict[Parameter, str] = {}
        for entry in priors:
            try:
                params, joint = entry
            except (TypeError, ValueError):
                raise TypeError(
                    "priors= must be a sequence of (params, joint) pairs"
                ) from None
            params = [params] if isinstance(params, Parameter) else list(params)
            for p in params:
                if p.prior is not None:
                    raise ValueError(
                        f"parameter {p.name!r} has its own prior= and is also covered "
                        "by a joint block: cover it once"
                    )
                if p in covered:
                    raise ValueError(
                        f"parameter {p.name!r} appears in two joint blocks"
                    )
                covered[p] = "joint"
            self.joints.append(_Joint(params, joint, index.slots(params)))
        self.marginals = [
            _Marginal(p, i) for i, p in enumerate(index.params) if p not in covered
        ]

    def logpdf(self, theta) -> float:
        total = 0.0
        for m in self.marginals:
            total += m.logpdf(theta[m.slot])
            if total == -np.inf:
                return -np.inf
        for j in self.joints:
            total += j.logpdf(theta[j.slots])
            if total == -np.inf:
                return -np.inf
        return float(total)

    def transform(self, u) -> np.ndarray:
        u = clip_unit_cube(u)
        theta = np.empty(self.ndim)
        for m in self.marginals:
            theta[m.slot] = m.transform(u[m.slot])
        for j in self.joints:
            theta[j.slots] = j.transform(u[j.slots])
        return theta

    def sample(self, n, rng) -> np.ndarray:
        out = np.empty((n, self.ndim))
        u = rng.uniform(size=(n, self.ndim))
        for m in self.marginals:
            out[:, m.slot] = m.transform(clip_unit_cube(u[:, m.slot]))
        for j in self.joints:
            out[:, j.slots] = j.sample(n, rng)
        return out


# ----------------------------------------------------------------------------
# The compiled constraint
# ----------------------------------------------------------------------------


def _stack_meta(constraint: Constraint) -> dict | None:
    keys = set()
    for c in constraint.comparisons:
        keys |= set(c.data.meta)
    if not keys:
        return None
    meta = {}
    for key in keys:
        parts = []
        for c in constraint.comparisons:
            v = c.data.meta.get(key, None)
            arr = np.asarray(v) if v is not None else None
            if arr is not None and arr.ndim >= 1 and arr.shape[0] == c.n:
                parts.append(arr)
            else:
                parts.append(
                    np.full(c.n, v, dtype=object)
                    if not np.isscalar(v)
                    else np.full(c.n, v)
                )
        try:
            stacked = np.concatenate(parts)
        except (TypeError, ValueError):
            stacked = np.concatenate([np.asarray(p, dtype=object) for p in parts])
        meta[key] = stacked
    return meta


class CompiledConstraint:
    """A :class:`~rxmc.constraint.Constraint` with slots resolved and its
    covariance factored.  Built by :class:`Problem`; not constructed by users."""

    def __init__(self, constraint: Constraint, index: ParameterIndex):
        self.source = constraint
        comps = constraint.comparisons
        self.comparisons = comps
        self.labels = [c.data.label or f"comparison {i}" for i, c in enumerate(comps)]
        self.offsets = constraint.offsets
        self.active = constraint.active
        self.n_active = constraint.n_active
        self.weight = constraint.weight
        self.likelihood = constraint.likelihood
        self.log_jacobian = constraint.log_jacobian
        try:
            self.x = np.concatenate([np.asarray(c.data.x) for c in comps])
        except ValueError as err:
            raise ValueError(
                f"the comparisons {self.labels} have x grids that cannot be stacked "
                f"({err}); put them in separate constraints"
            ) from None
        self.y = np.concatenate([c.y for c in comps])
        self.y_err = np.concatenate([c.y_err for c in comps])
        for i, (c, o) in enumerate(zip(comps, self.offsets)):
            rows = self.active[(self.active >= o.start) & (self.active < o.stop)]
            bad = ~(np.isfinite(self.y[rows]) & np.isfinite(self.y_err[rows]))
            if np.any(bad):
                raise ValueError(
                    f"comparison {self.labels[i]!r}: space {c.space.name!r} is not "
                    f"finite at {int(bad.sum())} active data point(s) (e.g. "
                    "non-positive y under a log transform); mask or drop those points"
                )
        self.meta = _stack_meta(constraint)
        self.predictors = [
            (o, index.add_all(c.predictor.params), c)
            for o, c in zip(self.offsets, comps)
        ]
        terms = list(constraint.terms)
        if constraint.statistical:
            terms = [statistical(c.y_err, on=c) for c in comps] + terms
        entries = [
            (t, constraint.support(t.on), index.add_all(t.params)) for t in terms
        ]
        self.like_gather = index.add_all(self.likelihood.params)
        self.covariance = StructuredCovariance(
            entries,
            self.x,
            self.y,
            self.offsets,
            self.active,
            meta=self.meta,
            labels=self.labels,
        )

    def ym(self, theta) -> np.ndarray:
        """The stacked prediction in comparison space, all points."""
        return np.concatenate([c.predict(*theta[g]) for _, g, c in self.predictors])

    def predict_physical(self, theta) -> list[np.ndarray]:
        return [c.predictor(*theta[g]) for _, g, c in self.predictors]

    def _stats(self, theta):
        ym = self.ym(theta)
        if not np.all(np.isfinite(ym[self.active])):
            return None
        d2, logdet = self.covariance.distance(ym, theta)
        return d2, logdet

    def log_likelihood(self, theta) -> float:
        s = self._stats(theta)
        if s is None:
            return -np.inf
        return float(
            self.likelihood.log_likelihood(*s, self.n_active, *theta[self.like_gather])
        )

    def chi2(self, theta) -> float:
        s = self._stats(theta)
        if s is None:
            return np.inf
        return float(self.likelihood.chi2(*s, self.n_active, *theta[self.like_gather]))

    def matrix(self, theta) -> np.ndarray:
        """The dense covariance on the active points at ``theta``."""
        return self.covariance.matrix(self.ym(theta), theta)

    def __repr__(self):
        return f"CompiledConstraint({self.labels}, n_active={self.n_active})"


# ----------------------------------------------------------------------------
# The problem
# ----------------------------------------------------------------------------


class Problem:
    """The compiled calibration problem: the flat interface a sampler wants.

    Parameters
    ----------
    constraints : iterable of Constraint
    priors : sequence of (params, joint), optional
        Joint prior blocks; see the module docstring.
    """

    def __init__(self, constraints, priors=()):
        constraints = tuple(constraints)
        if not constraints:
            raise ValueError("a problem needs at least one constraint")
        for c in constraints:
            if not isinstance(c, Constraint):
                raise TypeError(f"constraints must be Constraint objects, got {c!r}")
        self.index = ParameterIndex()
        self.constraints = tuple(CompiledConstraint(c, self.index) for c in constraints)
        self.index.check_names_unique()
        self.priors = tuple(priors)
        self._prior = _Prior(self.index, self.priors)

    # -- structure ----------------------------------------------------------

    @property
    def params(self) -> tuple[Parameter, ...]:
        return self.index.params

    @property
    def names(self) -> list[str]:
        return self.index.names

    @property
    def ndim(self) -> int:
        return self.index.ndim

    @property
    def bounds(self) -> np.ndarray:
        return self.index.bounds

    def columns(self, params) -> np.ndarray:
        """Chain columns of one parameter or of a sequence of them."""
        return self.index.slots(params)

    def _theta(self, theta) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (self.ndim,):
            raise ValueError(f"theta must have shape ({self.ndim},), got {theta.shape}")
        return theta

    # -- densities ------------------------------------------------------------

    def log_prior(self, theta) -> float:
        return self._prior.logpdf(self._theta(theta))

    def log_likelihood(self, theta) -> float:
        theta = self._theta(theta)
        total = 0.0
        for c in self.constraints:
            if c.weight == 0.0:
                continue
            ll = c.log_likelihood(theta)
            if ll == -np.inf:
                return -np.inf
            total += c.weight * ll
        return float(total)

    def log_posterior(self, theta) -> float:
        theta = self._theta(theta)
        lp = self._prior.logpdf(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta)

    def chi2(self, theta) -> float:
        theta = self._theta(theta)
        return float(sum(c.chi2(theta) for c in self.constraints))

    def log_jacobian(self) -> float:
        """Sum of the constraints' comparison-space log-Jacobians."""
        return float(sum(c.log_jacobian for c in self.constraints))

    def prior_transform(self, u) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        if u.shape != (self.ndim,):
            raise ValueError(f"u must have shape ({self.ndim},), got {u.shape}")
        return self._prior.transform(u)

    def sample_prior(self, n: int, rng=None) -> np.ndarray:
        """``(n, ndim)`` draws from the prior."""
        rng = np.random.default_rng(rng)
        return self._prior.sample(int(n), rng)

    def predict(self, theta, physical: bool = False) -> list[list[np.ndarray]]:
        """Per constraint, per comparison: the prediction on all points."""
        theta = self._theta(theta)
        out = []
        for c in self.constraints:
            if physical:
                out.append(c.predict_physical(theta))
            else:
                out.append([cmp.predict(*theta[g]) for _, g, cmp in c.predictors])
        return out

    # -- black-box-bayes spellings ------------------------------------------------

    @property
    def NDIM(self) -> int:  # noqa: N802 - the bbb name
        return self.ndim

    @property
    def parameter_names(self) -> list[str]:
        return self.names

    def starting_location(self, n: int) -> np.ndarray:
        return self.sample_prior(n)

    def log_posterior_batch(self, thetas) -> np.ndarray:
        thetas = np.asarray(thetas, dtype=float)
        return np.asarray([self.log_posterior(t) for t in thetas])

    def __repr__(self):
        return f"Problem(ndim={self.ndim}, constraints={len(self.constraints)})"
