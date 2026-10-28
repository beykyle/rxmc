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
``sample_prior``, and what a gradient-based sampler or optimiser needs:
``grad_log_posterior``, ``value_and_grad_log_posterior`` and
``hessian_log_posterior``.  ``log_posterior_unconstrained`` is the same
density on ``R^ndim`` (bounded parameters mapped through ``log`` or
``logit``; ``to_theta`` and ``to_unconstrained`` convert), for HMC/NUTS,
whose trajectories cannot cross the ``-inf`` wall at a bound.

The densities are written in ``jax.numpy`` and compiled with ``jax.jit`` once
per problem, so ``jax.grad``, ``jax.hessian`` and ``jax.vmap`` work through
``log_posterior`` and everything under it: the models, the covariance terms,
the likelihood and the prior.  Called on a concrete ``theta`` the densities
return a Python ``float`` (what emcee, dynesty and scipy expect); under a JAX
transformation they return the traced value.  ``prior_transform`` and
``sample_prior`` are not differentiated and stay numpy/scipy.

Prior rules, per slot (see :class:`~rxmc.params.Parameter`):

* ``Parameter(prior=dist)``: the marginal, truncated to ``bounds``;
* finite ``bounds`` and no ``prior``: uniform on the bounds;
* neither: the parameter must appear in exactly one joint block passed as
  ``priors=[(params, joint), ...]``, where ``joint`` exposes
  ``logpdf(values)`` over ``params`` in that order and, optionally,
  ``prior_transform(u)`` and ``rvs(size=n, random_state=rng)`` (scipy's
  spelling).  A joint that declares its dimension (``dim``, as scipy's
  multivariate distributions do) must match its parameters.  A frozen
  ``scipy.stats.multivariate_normal`` gets a whitening unit-cube map for
  free, and a one-parameter block holding a scipy univariate distribution is
  that parameter's marginal.  Finite bounds truncate a joint: a frozen MVN is
  renormalised by its mass inside them, while a custom joint's ``logpdf`` must
  already be normalised on its truncated support.  A hyperprior is a joint
  block that includes its hyperparameter.

The log density of a prior is evaluated in ``jax.numpy``.  A frozen scipy
marginal is translated to an equivalent JAX density for the families in
:data:`JAX_LOGPDFS`; any other marginal needs its density written in
``jax.numpy`` as ``Parameter(logpdf=...)``.  A custom joint's ``logpdf`` must
be written in ``jax.numpy``; a frozen multivariate normal is translated.

Nothing user-facing is mutated by compiling.  Compile the same declarations
twice and you get two independent problems.
"""

from __future__ import annotations

import warnings
from typing import Any, Iterable, Sequence

import jax
import jax.numpy as jnp
import jax.scipy.stats as jst
import numpy as np
from scipy import stats
from scipy.stats.distributions import rv_frozen

from .constraint import Constraint
from .covariance import StructuredCovariance
from .params import Parameter
from .terms import statistical

__all__ = [
    "Problem",
    "ParameterIndex",
    "CompiledConstraint",
    "clip_unit_cube",
    "JAX_LOGPDFS",
]


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
                "name them apart (prefix= of kernel_params, nu= for StudentT)."
            )


# ----------------------------------------------------------------------------
# The prior
# ----------------------------------------------------------------------------


def _host(x):
    """``x`` itself under a JAX transformation; otherwise numpy (a ``float`` when
    0-d), so samplers and user code get plain values."""
    if isinstance(x, jax.core.Tracer):
        return x
    x = np.asarray(x)
    return float(x) if x.ndim == 0 else x


def _std_uniform(z):
    return jnp.where((z >= 0) & (z <= 1), 0.0, -jnp.inf)


def _std_lognorm(z, s):
    pos = z > 0
    zs = jnp.where(pos, z, 1.0)
    out = -jnp.log(zs * s * jnp.sqrt(2 * jnp.pi)) - jnp.log(zs) ** 2 / (2 * s**2)
    return jnp.where(pos, out, -jnp.inf)


def _std_halfnorm(z):
    return jnp.where(z >= 0, jnp.log(2.0) + jst.norm.logpdf(z), -jnp.inf)


def _std_halfcauchy(z):
    return jnp.where(z >= 0, jnp.log(2.0) + jst.cauchy.logpdf(z), -jnp.inf)


def _std_truncnorm(z, a, b):
    log_mass = jnp.log(jax.scipy.special.ndtr(b) - jax.scipy.special.ndtr(a))
    return jnp.where((z >= a) & (z <= b), jst.norm.logpdf(z) - log_mass, -jnp.inf)


def _std_invgamma(z, a):
    pos = z > 0
    zs = jnp.where(pos, z, 1.0)
    out = -jax.scipy.special.gammaln(a) - (a + 1) * jnp.log(zs) - 1.0 / zs
    return jnp.where(pos, out, -jnp.inf)


#: scipy family name -> standardised JAX log density ``f(z, *shapes)`` with
#: ``z = (x - loc) / scale``.
JAX_LOGPDFS = {
    "norm": jst.norm.logpdf,
    "uniform": _std_uniform,
    "gamma": jst.gamma.logpdf,
    "beta": jst.beta.logpdf,
    "lognorm": _std_lognorm,
    "halfnorm": _std_halfnorm,
    "cauchy": jst.cauchy.logpdf,
    "halfcauchy": _std_halfcauchy,
    "t": jst.t.logpdf,
    "expon": jst.expon.logpdf,
    "laplace": jst.laplace.logpdf,
    "logistic": jst.logistic.logpdf,
    "invgamma": _std_invgamma,
    "truncnorm": _std_truncnorm,
}


def _jax_logpdf(dist, name):
    """A JAX log density equal to a frozen scipy univariate's ``logpdf``."""
    family = getattr(getattr(dist, "dist", None), "name", None)
    if family not in JAX_LOGPDFS:
        raise TypeError(
            f"the prior of {name!r} ({family or type(dist).__name__}) has no JAX "
            f"translation; supported scipy families are {sorted(JAX_LOGPDFS)}.  "
            "Give the parameter its density in jax.numpy as "
            "Parameter(..., logpdf=f), or cover it with a joint block whose "
            "logpdf is written in jax.numpy"
        )
    shapes, loc, scale = dist.dist._parse_args(*dist.args, **dist.kwds)
    f, shapes = JAX_LOGPDFS[family], tuple(float(a) for a in shapes)
    loc, scale = float(loc), float(scale)
    log_scale = np.log(scale)
    return lambda x: f((x - loc) / scale, *shapes) - log_scale


def _is_frozen_mvn(joint) -> bool:
    return type(joint).__name__ == "multivariate_normal_frozen"


def _joint_dim(joint) -> int | None:
    """The dimension a joint declares, or ``None`` when it declares none."""
    if isinstance(joint, rv_frozen):
        return 1
    dim = getattr(joint, "dim", None)
    return None if dim is None else int(dim)


class _Marginal:
    """One slot: a marginal (truncated to bounds) or the uniform on bounds.

    ``dist`` overrides ``p.prior``: a one-parameter prior block's distribution.
    """

    def __init__(self, p: Parameter, slot: int, dist=None):
        self.p, self.slot = p, slot
        lo, hi = p.bounds
        self.lo, self.hi = lo, hi
        self.dist = p.prior if dist is None else dist
        self.bounded = np.isfinite(lo) and np.isfinite(hi)
        self.upper = False
        if self.dist is None:
            if not self.bounded:
                raise ValueError(
                    f"parameter {p.name!r} has no prior: give it prior=, finite "
                    "bounds, or cover it with a joint block in Problem(priors=)"
                )
            self.c_lo, self.c_hi = 0.0, 1.0
            self.log_norm = np.log(hi - lo)
        else:
            # far in the upper tail cdf(lo) rounds to 1: work in the survival
            # function there, so the mass and the unit-cube map keep their precision
            self.upper = bool(np.isfinite(lo) and self.dist.cdf(lo) > 0.5)
            cdf, at_inf = (self.dist.sf, 0.0) if self.upper else (self.dist.cdf, 1.0)
            self.c_lo = float(cdf(lo)) if np.isfinite(lo) else 0.0
            self.c_hi = float(cdf(hi)) if np.isfinite(hi) else at_inf
            mass = abs(self.c_hi - self.c_lo)
            if not mass > 0:
                raise ValueError(
                    f"the prior of {p.name!r} has no mass inside its bounds"
                )
            self.log_norm = float(np.log(mass))
        # the JAX density, and a point inside the support at which it is
        # evaluated when v is outside (so the discarded branch of the where
        # has a finite gradient)
        if self.dist is None:
            self._logpdf = None
        elif p.logpdf is not None:
            self._logpdf = p.logpdf
        else:
            self._logpdf = _jax_logpdf(self.dist, p.name)
        self.safe = float(self.transform(0.5))

    def logpdf(self, v):
        inside = (v >= self.lo) & (v <= self.hi)
        if self._logpdf is None:
            return jnp.where(inside, -self.log_norm, -jnp.inf)
        lp = self._logpdf(jnp.where(inside, v, self.safe)) - self.log_norm
        return jnp.where(inside, lp, -jnp.inf)

    def transform(self, u):
        if self.dist is None:
            return self.lo + u * (self.hi - self.lo)
        inverse = self.dist.isf if self.upper else self.dist.ppf
        # clipped: the inverse can round a hair outside the bounds at u ~ 0 or 1
        return np.clip(
            inverse(self.c_lo + u * (self.c_hi - self.c_lo)), self.lo, self.hi
        )


class _Joint:
    """A joint block over several slots."""

    def __init__(self, params: Sequence[Parameter], joint, slots: np.ndarray):
        self.params, self.joint, self.slots = tuple(params), joint, slots
        self.bounds = np.asarray([p.bounds for p in self.params], dtype=float)
        self.bounded = bool(np.any(np.isfinite(self.bounds)))
        # a point inside the bounds, at which the density is evaluated when the
        # values are outside (so the discarded branch has a finite gradient)
        lo, hi = self.bounds.T
        self.safe = np.clip(0.0, lo + 1.0, hi - 1.0)
        both = np.isfinite(lo) & np.isfinite(hi)
        self.safe[both] = 0.5 * (lo[both] + hi[both])
        self.names = [p.name for p in self.params]
        if not hasattr(joint, "logpdf"):
            raise TypeError(f"joint prior over {self.names} must have logpdf(values)")
        dim = _joint_dim(joint)
        if dim is not None and dim != len(self.params):
            raise ValueError(
                f"the joint prior over {self.names} is {dim}-dimensional; it must "
                f"cover exactly those {len(self.params)} parameter(s), in order"
            )
        if _is_frozen_mvn(joint):
            self._L = np.linalg.cholesky(np.atleast_2d(joint.cov))
            self._mean = np.atleast_1d(joint.mean)
            mean, cov = jnp.asarray(self._mean), jnp.asarray(np.atleast_2d(joint.cov))
            self._logpdf = lambda v: jst.multivariate_normal.logpdf(v, mean, cov)
        else:
            self._L = None
            self._logpdf = joint.logpdf
            _check_traceable(self._logpdf, len(self.params), self.names)
        # a truncated MVN is renormalised by its mass inside the bounds; the box
        # probability is quasi-Monte Carlo from 3 dimensions, so it is seeded
        self.log_mass = 0.0
        if self.bounded and _is_frozen_mvn(joint):
            box = stats.multivariate_normal(joint.mean, joint.cov, seed=0)
            mass = float(box.cdf(self.bounds[:, 1], lower_limit=self.bounds[:, 0]))
            if not mass > 0:
                raise ValueError(
                    f"the joint prior over {self.names} has no mass inside its bounds"
                )
            self.log_mass = float(np.log(mass))

    @property
    def has_transform(self) -> bool:
        return not self.bounded and (
            self._L is not None or hasattr(self.joint, "prior_transform")
        )

    def logpdf(self, values):
        inside = jnp.all((values >= self.bounds[:, 0]) & (values <= self.bounds[:, 1]))
        # reshape(()): a length-1 result is fine, a longer one (a mis-sized
        # joint) is not
        lp = self._logpdf(jnp.where(inside, values, self.safe))
        lp = jnp.reshape(lp, ()) - self.log_mass
        return jnp.where(inside, lp, -jnp.inf) if self.bounded else lp

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


def _check_traceable(logpdf, n, names):
    """Fail at compile, by name, when a joint's ``logpdf`` is not jax.numpy."""
    try:
        jax.eval_shape(logpdf, jax.ShapeDtypeStruct((n,), jnp.float64))
    except Exception as err:  # any tracing failure: a numpy/scipy logpdf
        raise TypeError(
            f"the joint prior over {names} has a logpdf that JAX cannot trace "
            f"({type(err).__name__}: {err}); write it with jax.numpy and "
            "jax.scipy.stats so the posterior can be differentiated"
        ) from None


class _Prior:
    def __init__(self, index: ParameterIndex, priors):
        self.ndim = index.ndim
        self.joints: list[_Joint] = []
        covered: set[Parameter] = set()
        univariate: dict[Parameter, Any] = {}
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
                covered.add(p)
            if len(params) == 1 and isinstance(joint, rv_frozen):
                # a scipy univariate over one parameter is that parameter's
                # marginal: truncated to its bounds, with a ppf unit-cube map
                univariate[params[0]] = joint
            else:
                self.joints.append(_Joint(params, joint, index.slots(params)))
        self.marginals = [
            _Marginal(p, i, univariate.get(p))
            for i, p in enumerate(index.params)
            if p not in covered or p in univariate
        ]

    def logpdf(self, theta):
        total = 0.0
        for m in self.marginals:
            total = total + m.logpdf(theta[m.slot])
        for j in self.joints:
            total = total + j.logpdf(theta[j.slots])
        return jnp.asarray(total)

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


def _per_point(v, n) -> np.ndarray:
    """A metadata value as one entry per point: a length-``n`` sequence as is,
    anything else (a scalar, a 0-d array, a provenance tuple, an object)
    repeated ``n`` times."""
    if isinstance(v, (np.ndarray, np.generic)) and v.ndim == 0:
        v = v.item()
    if np.isscalar(v):
        return np.full(n, v)
    try:
        arr = np.asarray(v)
    except ValueError:  # ragged: not per-point
        arr = None
    if arr is not None and arr.ndim >= 1 and arr.shape[0] == n:
        return arr
    out = np.empty(n, dtype=object)
    for i in range(n):  # element by element: numpy would broadcast a sequence
        out[i] = v
    return out


def _stack_meta(constraint: Constraint) -> dict | None:
    keys = set()
    for c in constraint.comparisons:
        keys |= set(c.data.meta)
    if not keys:
        return None
    meta = {}
    for key in keys:
        parts = [_per_point(c.data.meta.get(key), c.n) for c in constraint.comparisons]
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
        # the reported statistical diagonals are built here, not by the user, so
        # a term selection names them with statistical=True rather than by object
        self.statistical_terms = (
            [statistical(c.y_err, on=c) for c in comps]
            if constraint.statistical
            else []
        )
        terms = self.statistical_terms + list(constraint.terms)
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

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_jit_cache", None)
        return state

    def _jitted(self, key, fn):
        """``jax.jit(fn)``, built once per key (the diagnostics call ``ym`` and
        ``matrix`` per posterior sample)."""
        cache = self.__dict__.setdefault("_jit_cache", {})
        if key not in cache:
            cache[key] = jax.jit(fn)
        return cache[key]

    def ym(self, theta):
        """The stacked prediction in comparison space, all points.

        A numpy array on a concrete ``theta``.
        """
        return _host(self._jitted("ym", self._ym)(jnp.asarray(theta, dtype=float)))

    def _ym(self, theta):
        parts = []
        for (_, g, c), label in zip(self.predictors, self.labels):
            y = c.predict(*theta[g])
            if y.shape != (c.n,):
                raise ValueError(
                    f"comparison {label!r}: the model returned shape {y.shape} on a "
                    f"grid of {c.n} point(s)"
                )
            parts.append(y)
        return jnp.concatenate(parts)

    def predict_physical(self, theta) -> list:
        return [c.predictor(*theta[g]) for _, g, c in self.predictors]

    def _stats(self, theta):
        """``(d2, logdet, ok)``: ``ok`` is false where the prediction is not
        finite on the active points or a parametric covariance is singular."""
        ym = self._ym(theta)
        d2, logdet = self.covariance.distance(ym, theta)
        ok = (
            jnp.all(jnp.isfinite(ym[self.active]))
            & jnp.isfinite(d2)
            & jnp.isfinite(logdet)
        )
        return d2, logdet, ok

    def log_likelihood(self, theta):
        d2, logdet, ok = self._stats(theta)
        ll = self.likelihood.log_likelihood(
            d2, logdet, self.n_active, *theta[self.like_gather]
        )
        return jnp.where(ok, ll, -jnp.inf)

    def chi2(self, theta):
        d2, logdet, ok = self._stats(theta)
        c2 = self.likelihood.chi2(d2, logdet, self.n_active, *theta[self.like_gather])
        return jnp.where(ok, c2, jnp.inf)

    def entries_for(self, terms=None, statistical: bool = True):
        """Covariance entries of a term selection, or ``None`` for all of them.

        ``terms`` are :class:`~rxmc.terms.Term` objects declared on this
        constraint, matched *by identity*; ``None`` means every declared term.
        ``statistical`` says whether the reported statistical diagonals join
        them.  ``None`` is returned for the full selection so the caller takes
        the cached path.
        """
        if terms is None and statistical:
            return None
        chosen = list(self.source.terms) if terms is None else list(terms)
        for t in chosen:
            if not any(t is u for u in self.source.terms):
                raise ValueError(
                    f"{t!r} is not a term of this constraint; terms= selects among "
                    "the terms it was declared with (the reported statistical "
                    "errors are selected with statistical=True/False instead)"
                )
        if statistical:
            chosen = self.statistical_terms + chosen
        return [e for e in self.covariance.entries if any(e.term is t for t in chosen)]

    def matrix(self, theta, *, terms=None, statistical: bool = True):
        """The dense covariance on the active points at ``theta``.

        ``terms`` and ``statistical`` select part of the error model; see
        :meth:`entries_for`.  The default is the covariance the likelihood uses.
        A numpy array on a concrete ``theta``.
        """
        entries = self.entries_for(terms, statistical)
        key = ("matrix", None if entries is None else tuple(map(id, entries)))
        fn = self._jitted(
            key, lambda t: self.covariance.matrix(self._ym(t), t, entries=entries)
        )
        return _host(fn(jnp.asarray(theta, dtype=float)))

    def __repr__(self):
        return f"CompiledConstraint({self.labels}, n_active={self.n_active})"


def _warn_on_shared_rows(constraints) -> None:
    """Warn when one dataset's rows are active in two weighted constraints.

    Its likelihood would count those rows twice.  Disjoint masks (a fit and its
    complement) and weight-0 monitor constraints are fine.
    """
    seen: dict[int, list] = {}
    for k, c in enumerate(constraints):
        if c.weight == 0.0:
            continue
        for comp, o in zip(c.comparisons, c.offsets):
            rows = c.active[(c.active >= o.start) & (c.active < o.stop)] - o.start
            for j, prev in seen.get(id(comp.data), []):
                if np.intersect1d(rows, prev).size:
                    warnings.warn(
                        f"dataset {comp.data.label or 'dataset'!r} has rows active in "
                        f"constraints {j} and {k}: the likelihood counts them twice "
                        "(mask them apart, or give one constraint weight 0)",
                        UserWarning,
                        stacklevel=3,
                    )
            seen.setdefault(id(comp.data), []).append((k, rows))


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
        _warn_on_shared_rows(self.constraints)
        self.priors = tuple(priors)
        # a hyperprior block may introduce a parameter no model or term uses (its
        # hyperparameter); it gets a slot after every constraint's parameters
        for entry in self.priors:
            params = entry[0] if not isinstance(entry, Parameter) else entry
            params = [params] if isinstance(params, Parameter) else list(params)
            self.index.add_all(params)
        self.index.check_names_unique()
        self._prior = _Prior(self.index, self.priors)
        self._weighted = [(c.weight, c) for c in self.constraints if c.weight != 0.0]
        # which slots the unconstrained map sends through exp (one finite bound)
        # or a scaled sigmoid (two); static, so the map traces to fixed ops
        lo, hi = self.bounds.T
        self._lo_only = np.flatnonzero(np.isfinite(lo) & ~np.isfinite(hi))
        self._hi_only = np.flatnonzero(~np.isfinite(lo) & np.isfinite(hi))
        self._both = np.flatnonzero(np.isfinite(lo) & np.isfinite(hi))
        self._jit()

    def _jit(self):
        """Compile the densities once per problem (and again after unpickling)."""
        self._log_prior_j = jax.jit(self._log_prior_impl)
        self._log_likelihood_j = jax.jit(self._log_likelihood_impl)
        self._log_posterior_j = jax.jit(self._log_posterior_impl)
        self._chi2_j = jax.jit(self._chi2_impl)
        self._vg_j = jax.jit(jax.value_and_grad(self._log_posterior_impl))
        self._hess_j = jax.jit(jax.hessian(self._log_posterior_impl))
        self._batch_j = jax.jit(jax.vmap(self._log_posterior_impl))
        self._to_theta_j = jax.jit(lambda z: self._to_theta_impl(z)[0])
        self._to_z_j = jax.jit(self._to_unconstrained_impl)
        self._lpu_j = jax.jit(self._log_posterior_unconstrained_impl)

    def __getstate__(self):
        state = self.__dict__.copy()
        for k in [k for k in state if k.endswith("_j")]:
            del state[k]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._jit()

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

    def _theta(self, theta):
        theta = jnp.asarray(theta, dtype=float)
        if theta.shape != (self.ndim,):
            raise ValueError(f"theta must have shape ({self.ndim},), got {theta.shape}")
        return theta

    # -- densities: pure jax.numpy, compiled in _jit ---------------------------

    def _log_prior_impl(self, theta):
        return self._prior.logpdf(theta)

    def _log_likelihood_impl(self, theta):
        total = 0.0
        for w, c in self._weighted:
            total = total + w * c.log_likelihood(theta)
        return jnp.asarray(total)

    def _log_posterior_impl(self, theta):
        # prior first: outside its support the forward models are not run (a
        # cond executes one branch; under vmap it evaluates both and selects)
        lp = self._prior.logpdf(theta)
        return jax.lax.cond(
            jnp.isfinite(lp),
            lambda t: lp + self._log_likelihood_impl(t),
            lambda t: jnp.asarray(-jnp.inf),
            theta,
        )

    def _chi2_impl(self, theta):
        total = 0.0
        for c in self.constraints:
            total = total + c.chi2(theta)
        return jnp.asarray(total)

    # -- densities ------------------------------------------------------------

    def log_prior(self, theta) -> float:
        """Log prior density at ``theta``; ``-inf`` outside its support."""
        return _host(self._log_prior_j(self._theta(theta)))

    def log_likelihood(self, theta) -> float:
        """Weighted sum of the constraints' log likelihoods at ``theta``."""
        return _host(self._log_likelihood_j(self._theta(theta)))

    def log_posterior(self, theta) -> float:
        """Log prior plus log likelihood at ``theta``.

        A float on a concrete ``theta``, the traced value under ``jax.grad``
        or ``jax.vmap``.  ``-inf`` outside the prior's support (the forward
        models do not run, and the gradient is zero) or where a covariance is
        singular (the gradient stays finite).
        """
        return _host(self._log_posterior_j(self._theta(theta)))

    def chi2(self, theta) -> float:
        """Sum of the constraints' chi-square statistics at ``theta``."""
        return _host(self._chi2_j(self._theta(theta)))

    # -- derivatives ------------------------------------------------------------

    def grad_log_posterior(self, theta) -> np.ndarray:
        """The gradient of :meth:`log_posterior` (a numpy array)."""
        return np.asarray(self._vg_j(self._theta(theta))[1])

    def value_and_grad_log_posterior(self, theta) -> tuple[float, np.ndarray]:
        """``(log_posterior(theta), grad_log_posterior(theta))`` in one pass."""
        v, g = self._vg_j(self._theta(theta))
        return float(v), np.asarray(g)

    def hessian_log_posterior(self, theta) -> np.ndarray:
        """The ``(ndim, ndim)`` Hessian of :meth:`log_posterior` (e.g. Laplace)."""
        return np.asarray(self._hess_j(self._theta(theta)))

    # -- the unconstrained space ----------------------------------------------

    def _to_theta_impl(self, z):
        """``(theta, log|d theta / d z|)``: ``exp`` away from one finite bound,
        a scaled sigmoid between two, the identity otherwise."""
        lo, hi = self.bounds.T
        a, b, c = self._lo_only, self._hi_only, self._both
        z = jnp.asarray(z, dtype=float)
        theta = z.at[a].set(lo[a] + jnp.exp(z[a]))
        theta = theta.at[b].set(hi[b] - jnp.exp(z[b]))
        width = hi[c] - lo[c]
        theta = theta.at[c].set(lo[c] + width * jax.nn.sigmoid(z[c]))
        log_jac = (
            jnp.sum(z[a])
            + jnp.sum(z[b])
            + jnp.sum(
                np.log(width) + jax.nn.log_sigmoid(z[c]) + jax.nn.log_sigmoid(-z[c])
            )
        )
        return theta, log_jac

    def _to_unconstrained_impl(self, theta):
        lo, hi = self.bounds.T
        a, b, c = self._lo_only, self._hi_only, self._both
        theta = jnp.asarray(theta, dtype=float)
        z = theta.at[a].set(jnp.log(theta[a] - lo[a]))
        z = z.at[b].set(jnp.log(hi[b] - theta[b]))
        return z.at[c].set(
            jax.scipy.special.logit((theta[c] - lo[c]) / (hi[c] - lo[c]))
        )

    def _log_posterior_unconstrained_impl(self, z):
        theta, log_jac = self._to_theta_impl(z)
        return self._log_posterior_impl(theta) + log_jac

    def to_theta(self, z):
        """Map a point of ``R^ndim`` into the bounds (see
        :meth:`log_posterior_unconstrained`)."""
        return _host(self._to_theta_j(self._theta(z)))

    def to_unconstrained(self, theta):
        """The inverse of :meth:`to_theta`, for ``theta`` strictly inside the bounds."""
        return _host(self._to_z_j(self._theta(theta)))

    def log_posterior_unconstrained(self, z) -> float:
        """:meth:`log_posterior` at ``to_theta(z)`` plus its log-Jacobian.

        The density of ``z = to_unconstrained(theta)``, finite on all of
        ``R^ndim`` where the posterior is: a parameter with one finite bound
        is sampled as the log of its distance from it, one with two as the
        logit of its position between them.  Gradient samplers (HMC/NUTS,
        recipe 41) need this, because their trajectories cannot cross the
        ``-inf`` wall at a bound.  A float on a concrete ``z``, the traced
        value under JAX.
        """
        return _host(self._lpu_j(self._theta(z)))

    def log_jacobian(self) -> float:
        """Sum of the constraints' comparison-space log-Jacobians, each times its
        ``weight``.

        A tempered constraint enters the likelihood as ``weight * log L``, so
        its Jacobian enters ``log Z_raw = log Z + log_jacobian()`` with the same
        weight, and a weight-0 constraint not at all.
        """
        return float(
            sum(c.weight * c.log_jacobian for c in self.constraints if c.weight)
        )

    def prior_transform(self, u) -> np.ndarray:
        """Map a point of the unit cube to ``theta`` (nested sampling)."""
        u = np.asarray(u, dtype=float)
        if u.shape != (self.ndim,):
            raise ValueError(f"u must have shape ({self.ndim},), got {u.shape}")
        return self._prior.transform(u)

    def sample_prior(self, n: int, rng=None) -> np.ndarray:
        """``(n, ndim)`` draws from the prior."""
        rng = np.random.default_rng(rng)
        return self._prior.sample(int(n), rng)

    def predict(self, theta, physical: bool = False) -> list[list[np.ndarray]]:
        """Per constraint, per comparison: the prediction on all points.

        numpy arrays on a concrete ``theta``.
        """
        theta = self._theta(theta)
        out = []
        for c in self.constraints:
            if physical:
                out.append([_host(y) for y in c.predict_physical(theta)])
            else:
                ym = c.ym(theta)
                out.append([ym[o] for o in c.offsets])
        return out

    # -- black-box-bayes spellings ------------------------------------------------

    @property
    def NDIM(self) -> int:  # noqa: N802 - the bbb name
        return self.ndim

    @property
    def parameter_names(self) -> list[str]:
        return self.names

    def starting_location(self, n: int) -> np.ndarray:
        """``(n, ndim)`` draws from the prior (:meth:`sample_prior`)."""
        return self.sample_prior(n)

    def log_posterior_batch(self, thetas) -> np.ndarray:
        """:meth:`log_posterior` over the rows of ``thetas``.

        One ``jax.vmap`` call; with emcee, pass it with ``vectorize=True``
        instead of a process pool.
        """
        thetas = jnp.asarray(thetas, dtype=float)
        if thetas.ndim != 2 or thetas.shape[1] != self.ndim:
            raise ValueError(
                f"thetas must have shape (n, {self.ndim}), got {thetas.shape}"
            )
        return np.asarray(self._batch_j(thetas))

    def __repr__(self):
        return f"Problem(ndim={self.ndim}, constraints={len(self.constraints)})"
