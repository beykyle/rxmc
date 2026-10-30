"""Reference posteriors the sampler benchmark scores bias against.

Three problems are linear-Gaussian once a few parameters are fixed: their
log posterior is quadratic in the remaining (linear) columns, so it can be
integrated over those exactly and over the rest on a grid.  That gives the
posterior moments and the evidence to quadrature accuracy::

    linear      no grid: (m, b) is Gaussian
    gp          (log variance, log length scale) on a grid; (m, b) Gaussian
    hierarchy   tau on a grid; (m, mu, eta_j) Gaussian

The others have no closed form, and their reference is one long blackjax
NUTS run from a ball about the MAP, with a seed of its own, so the benchmarked NUTS row is not scored
against itself.  ``python bench_samplers.py --reference`` writes them all to
``results/references.json``.
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy.special import logsumexp

#: per problem: the grid columns and their nodes (the rest are integrated
#: exactly), or None for a long NUTS run; the nodes are cell midpoints, so
#: a grid with a closed end (tau = 0) never evaluates on the boundary
GRIDS = {
    "linear": {},
    "gp": {2: (-12.0, 2.0, 100), 3: (-3.0, 1.0, 100)},
    "hierarchy": {2: (0.0, 6.0, 4000)},
}
#: smaller steps than the benchmark's NUTS (target acceptance 0.8), to stay
#: clear of divergences
NUTS = dict(chains=4, warmup=1000, draws=3000, seed=1000, target_acceptance_rate=0.95)


def _midpoints(lo, hi, n):
    edges = np.linspace(lo, hi, n + 1)
    return 0.5 * (edges[1:] + edges[:-1]), (hi - lo) / n


def conditional_gaussian(problem, grid, x0=None):
    """Exact moments and log Z of a posterior quadratic in every column not in
    ``grid`` (``{column: (lo, hi, n)}``) given the grid columns.

    At each grid node, one Newton step over the linear block from ``x0`` lands
    on the conditional mode, and the Hessian there is the conditional
    precision, both exact for a quadratic.  The conditional normalisers weight
    the nodes; moments mix over them.
    """
    cols = sorted(grid)
    lin = np.array([k for k in range(problem.ndim) if k not in grid])
    x0 = np.zeros(lin.size) if x0 is None else np.asarray(x0, float)
    axes = [_midpoints(*grid[k]) for k in cols]
    cell = np.prod([h for _, h in axes]) if axes else 1.0
    nodes = list(itertools.product(*[v for v, _ in axes])) or [()]
    log_z, means, variances = [], [], []
    theta = np.empty(problem.ndim)
    for node in nodes:
        theta[cols] = node
        theta[lin] = x0
        f, g = problem.value_and_grad_log_posterior(theta)
        precision = -problem.hessian_log_posterior(theta)[np.ix_(lin, lin)]
        step = np.linalg.solve(precision, g[lin])
        _, logdet = np.linalg.slogdet(precision)
        log_z.append(
            f + 0.5 * g[lin] @ step + 0.5 * lin.size * np.log(2 * np.pi) - 0.5 * logdet
        )
        means.append(x0 + step)
        variances.append(np.diag(np.linalg.inv(precision)))
    log_z, means, variances = map(np.asarray, (log_z, means, variances))
    w = np.exp(log_z - log_z.max())
    w /= w.sum()
    mean, sd = np.empty(problem.ndim), np.empty(problem.ndim)
    mean[lin] = w @ means
    sd[lin] = np.sqrt(w @ (variances + (means - mean[lin]) ** 2))
    pts = np.array(nodes, float).reshape(len(nodes), len(cols))
    mean[cols] = w @ pts
    sd[cols] = np.sqrt(w @ (pts - mean[cols]) ** 2)
    return dict(
        method="quadrature",
        mean=mean.tolist(),
        sd=sd.tolist(),
        logz=float(logsumexp(log_z) + np.log(cell)),
    )


def long_nuts(problem):
    """Window-adapted NUTS chains, much longer than the benchmark's; accepted
    only if they converged without divergences.  They start about the MAP: a
    chain started from a prior draw can be trapped in a minor mode (on
    ``modes``, one with a log posterior 40 below the main one)."""
    import blackjax
    import jax
    from bench_samplers import RHAT_MAX, _ball, _blackjax_chain, _summary

    import rxmc as rx

    b = NUTS
    keys = jax.random.split(jax.random.key(b["seed"]), b["chains"])
    mode = rx.optimize.find_map(problem, rng=b["seed"])
    starts = _ball(problem, mode.x, b["chains"], b["seed"])
    chains, divergent = [], 0
    for key, theta0 in zip(keys, starts):
        zs, div, _, _ = _blackjax_chain(
            problem,
            blackjax.nuts,
            key,
            problem.to_unconstrained(theta0),
            b["warmup"],
            b["draws"],
            target_acceptance_rate=b["target_acceptance_rate"],
        )
        chains.append(np.asarray(jax.vmap(problem.to_theta)(zs)))
        divergent += div
    s = _summary(np.stack(chains), 0.0, {})
    if s["max_rhat"] > RHAT_MAX or divergent:
        raise RuntimeError(
            f"reference NUTS: R-hat {s['max_rhat']:.3f}, {divergent} divergences"
        )
    return dict(
        method=f"NUTS, {b['chains']} x {b['draws']} draws",
        mean=s["mean"],
        sd=s["sd"],
        logz=None,
        min_ess=s["min_ess"],
        max_rhat=s["max_rhat"],
    )


def reference(name, problem):
    """The reference posterior of benchmark problem ``name``."""
    if name in GRIDS:
        return conditional_gaussian(problem, GRIDS[name])
    return long_nuts(problem)
