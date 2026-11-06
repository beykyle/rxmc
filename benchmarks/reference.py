"""Reference posteriors the sampler benchmark scores bias against.

Three problems are linear-Gaussian once a few parameters are fixed: their
log posterior is quadratic in the remaining (linear) columns, so it can be
integrated over those exactly and over the rest on a grid.  That gives the
posterior moments and the evidence to quadrature accuracy::

    linear      no grid: (m, b) is Gaussian
    gp          (log variance, log length scale) on a grid; (m, b) Gaussian
    hierarchy   tau on a grid; (m, mu, eta_j) Gaussian

The others have no closed form.  ``modes`` and ``elastic`` have one mode,
and their reference is one long blackjax NUTS run from a ball about the MAP,
with a seed of its own, so the benchmarked NUTS row is not scored against
itself.  ``omp`` has two separated islands, and a NUTS chain stays in the
island it starts in, so R-hat cannot see an island no chain visited; its
reference pools several pocomc runs at four times the benchmark's
resolution, each with a seed of its own, accepted only if they agree.  ``python bench_samplers.py --reference`` writes them all to
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
#: per problem: pocomc runs pooled into the reference, and how closely each
#: must agree with the pool (bias and width, in posterior standard
#: deviations); then dynesty runs, pooled the same way, that check the pool
#: from an independent algorithm.  One dynesty run is too noisy for that
#: (0.12 to 0.37 sd on omp), so three are pooled; rslice is the setting of the
#: sweep that does best on the funnels.  The check is recorded, not enforced
#: (``CORROBORATION`` has its bars).
POPULATION = {
    name: dict(
        n_effective=8192,
        n_active=4096,
        seeds=(2000, 2001, 2002),
        agree=0.05,
        corroborate=dict(nlive=1000, sample="rslice", seeds=(3000, 3001, 3002)),
    )
    for name in ("omp", "alpha_gp", "n40ca_gp")
}
# alpha_gp's single dynesty runs scatter in the share of the deeper family
# (0.55 to 0.91; the bar is 0.03), so it pools nine
POPULATION["alpha_gp"]["corroborate"]["seeds"] = tuple(range(3000, 3009))
#: the pooled dynesty runs corroborate the pooled pocomc ones if their log Z
#: agree within twice the combined error, their mode shares (where a problem
#: has two modes) within ``share``, and their moments within ``moments`` sd
CORROBORATION = dict(share=0.03, moments=0.2)
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


def long_pocomc(problem, seeds, agree, corroborate=None, name=None, **settings):
    """pocomc runs at ``settings``, one per seed, pooled with equal weight;
    accepted only if every run is within ``agree`` of the pool.  The
    evidence is the runs' mean, its error their half-spread or the largest
    quoted error, whichever is larger.

    With ``corroborate`` (dynesty settings and ``seeds``), dynesty runs are
    pooled the same way and compared with the pocomc pool (``CORROBORATION``).
    The comparison is recorded either way, ``corroborated`` true or false, so
    both answers survive for a person to judge.

    On a cluster every run is a separate task (``pocomc_part``,
    ``dynesty_part``) and ``pool_parts`` assembles them."""
    from problems import mode_share

    summary = mode_share(name, problem) if name else None
    runs = [pocomc_part(problem, seed, settings, summary) for seed in seeds]
    dynesty = []
    if corroborate:
        d_settings = {k: v for k, v in corroborate.items() if k != "seeds"}
        dynesty = [
            dynesty_part(problem, seed, d_settings, summary)
            for seed in corroborate["seeds"]
        ]
    return pool_parts(runs, agree, settings, dynesty, names=problem.names)


def pocomc_part(problem, seed, settings, summary=None):
    """One pocomc run of a population reference; logged as it ends."""
    from bench_samplers import run_pocomc

    r = run_pocomc(problem, {}, seed, summary=summary, **settings)
    r["seed"] = seed
    print(
        f"  pocomc seed {seed}: {r['wall_s']:.0f} s, log Z {r['logz']:.2f}, "
        f"mean {np.round(r['mean'], 3).tolist()}, sd {np.round(r['sd'], 3).tolist()}",
        flush=True,
    )
    return r


def dynesty_part(problem, seed, settings, summary=None):
    """One of the dynesty runs that check a population reference."""
    from bench_samplers import run_dynesty

    d = run_dynesty(problem, {}, seed, summary=summary, **settings)
    d.update(seed=seed, method=f"dynesty {settings}")
    print(
        f"  dynesty seed {seed}: {d['wall_s']:.0f} s, log Z {d['logz']:.2f}", flush=True
    )
    return d


def _pool(runs):
    """Equal-weight pool of runs: means, sds (within-run variances plus the
    spread of the means), log Z and its error (half-spread or the largest
    quoted error), and the mean mode share where the runs have one."""
    mean = np.mean([r["mean"] for r in runs], axis=0)
    var = np.mean(
        [np.square(r["sd"]) + np.square(np.subtract(r["mean"], mean)) for r in runs],
        axis=0,
    )
    logz = np.array([r["logz"] for r in runs])
    pool = dict(
        mean=mean.tolist(),
        sd=np.sqrt(var).tolist(),
        logz=float(logz.mean()),
        logz_err=float(max(np.ptp(logz) / 2, max(r["logz_err"] for r in runs))),
    )
    if all("mode_share" in r for r in runs):
        pool["mode_share"] = float(np.mean([r["mode_share"] for r in runs]))
        pool["mode_share_label"] = runs[0]["mode_share_label"]
    return pool


def _summarised(r):
    keep = ("seed", "mean", "sd", "logz", "logz_err", "wall_s", "mode_share")
    return {k: r[k] for k in keep if k in r}


def pool_parts(runs, agree, settings, dynesty=(), names=None, strict=True):
    """The reference from its parts: the pocomc ``runs`` pooled and checked
    against ``agree``, and the ``dynesty`` runs pooled and compared with it.
    A run outside ``agree`` raises if ``strict``; otherwise (assembling cluster
    results) the reference is returned with ``accepted: False``."""
    from bench_samplers import errors

    pool = _pool(runs)
    worst = max(max(errors(r, pool)) for r in runs)
    accepted = worst <= agree
    if not accepted and strict:
        raise RuntimeError(f"reference pocomc: a run is {worst:.3f} sd from the pool")
    out = dict(
        method=f"pocomc {settings['n_effective']}/{settings['n_active']}, {len(runs)} runs",
        **pool,
        max_disagreement=float(worst),
        accepted=bool(accepted),
        runs=[_summarised(r) for r in runs],
    )
    if names is not None:
        out["names"] = list(names)
    if dynesty:
        d = _pool(dynesty)
        bias, width = errors(d, pool)
        combined = float(np.hypot(pool["logz_err"], d["logz_err"]))
        checks = {
            "log Z": abs(d["logz"] - pool["logz"]) <= 2 * combined,
            "moments": max(bias, width) <= CORROBORATION["moments"],
        }
        if "mode_share" in d and "mode_share" in pool:
            share_gap = abs(d["mode_share"] - pool["mode_share"])
            checks["mode share"] = share_gap <= CORROBORATION["share"]
        out["corroboration"] = dict(
            method=f"{dynesty[0]['method']}, {len(dynesty)} runs pooled",
            **d,
            bias=bias,
            width=width,
            logz_combined_err=combined,
            checks={k: bool(v) for k, v in checks.items()},
            runs=[_summarised(r) for r in dynesty],
        )
        out["corroborated"] = all(checks.values())
        failed = [k for k, v in checks.items() if not v]
        print(
            f"  dynesty pool: bias {bias:.3f}, width {width:.3f}, log Z "
            f"{d['logz']:.2f} against {pool['logz']:.2f}"
            + (f", mode share {d['mode_share']:.3f} against {pool['mode_share']:.3f}"
               if "mode_share" in d and "mode_share" in pool else "")
            + (": agrees" if not failed else f": WARNING: disagrees on {failed}"),
            flush=True,
        )  # fmt: skip
    return out


def reference(name, problem):
    """The reference posterior of benchmark problem ``name``."""
    if name in GRIDS:
        return conditional_gaussian(problem, GRIDS[name])
    if name in POPULATION:
        return long_pocomc(problem, name=name, **POPULATION[name])
    return long_nuts(problem)
