"""Compare samplers on the benchmark problems, per effective sample.

JAX rxmc only (the gradient samplers need the gradient)::

    python benchmarks/bench_samplers.py --out benchmarks/results/samplers.json
    python benchmarks/bench_samplers.py --out ... --problems omp --only ptemcee pocomc
    python benchmarks/bench_samplers.py --report benchmarks/results/samplers.json --markdown
    python benchmarks/bench_samplers.py --reference --problems omp

emcee, dynesty, pocomc and blackjax NUTS run on every problem; the optical
model ``omp`` also runs ptemcee, blackjax HMC and PyMC's NUTS and HMC (the
``samplers`` extra).  docs/benchmarks.md explains the set-up and the scoring.

Each sampler starts the way it is usually used, and pays for the search:

* ensemble samplers (emcee, ptemcee) and HMC and NUTS (blackjax, PyMC) from
  a small ball about the posterior mode, ``rx.optimize.find_map`` (L-BFGS-B
  with the exact gradient), the gradient samplers on
  ``log_posterior_unconstrained``.  Not the maximum of the likelihood: on
  the hierarchy that is a ridge (the likelihood sees only ``mu + tau *
  eta_j``), not a point.  The exception is the hierarchy, whose MAP is far
  from the bulk of its posterior: the non-centred mode shrinks the
  ``eta_j`` toward their prior and inflates ``tau`` to compensate (1.27 at
  the MAP, against a posterior of 0.31 +/- 0.04), so emcee and NUTS start
  there from prior draws;
* nested sampling (dynesty) and preconditioned Monte Carlo (pocomc) from the
  prior, as their algorithms do; pocomc samples the unit cube with
  ``log_likelihood(prior_transform(u))``, the prior exactly.

A ball is ``to_theta(to_unconstrained(start) + 0.01 N(0, I))``, inside the
support.  A search's wall time and calls are charged to every sampler
that starts from it.

A call is one evaluation of the log density and, for a gradient sampler
(and the MAP search), its gradient.  Calls are recorded by phase: the start
search, warm-up (emcee's and ptemcee's discarded burn-in, NUTS's and HMC's
adaptation, pocomc's tempering from the prior), sampling, and pocomc's
evidence step.  ESS per thousand calls does not depend on the machine; the
wall time per call shows what a call costs (a gradient, PyMC's overhead,
pocomc's flow training).

Recorded per sampler: wall time, calls, the smallest bulk and tail ESS and
the largest rank-normalised split R-hat over the parameters (Vehtari et al.
2021: ``blackjax.diagnostics.ess_bulk``, ``ess_tail`` and ``rhat``) for every
chain-based sampler, walkers taken as chains for the ensembles (which
flatters them), the ensembles' integrated autocorrelation time tau (flagged under the
50 tau emcee asks for), the evidence where the algorithm gives one, and the
posterior means and standard deviations.  A chain-based run with R-hat above
1.01, or an ensemble run shorter than 50 tau, has not converged.

Every run is scored against a reference posterior (``reference.py``, written
by ``--reference`` to ``results/references.json``): exact quadrature where the
problem is linear-Gaussian given a few parameters, pooled high-resolution
pocomc runs on the two-island ``omp``, a long NUTS run otherwise.  Its bias is the largest error of a posterior mean, and its width
error the largest of ``|log(sd / sd_ref)|``, both over the parameters and in
reference standard deviations.  A run is correct if it converged and both
are at most ``TOLERANCE``; ESS rates are reported only for correct runs.

The ESS of the weighted draws of dynesty and pocomc is Kish's, from the
importance weights alone.  It ignores the correlation between particles that
share an ancestor (pocomc's resampling, dynesty's random walks from live
points), so it overstates their ESS, and pocomc stops once Kish's ESS reaches
its target, so its ESS is near that target by construction.  The bias and
width columns are the check that matters for them.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import blackjax
import dynesty
import emcee
import jax
import numpy as np
from problems import PROBLEMS

import rxmc as rx

REFERENCES = Path(__file__).parent / "results" / "references.json"

#: per problem: the samplers, and their budgets
COMMON = ("emcee", "dynesty", "pocomc", "NUTS")
EXTRA = ("ptemcee", "HMC", "PyMC NUTS", "PyMC HMC")
BUDGETS = {
    # 12000 steps: rank-normalised R-hat was 1.019 at 3000 and 1.010 at 6000
    "linear": dict(samplers=COMMON, emcee=12000, warmup=1000, draws=1000, chains=4),
    "gp": dict(samplers=COMMON, emcee=3000, warmup=1000, draws=1000, chains=4),
    "modes": dict(samplers=COMMON, emcee=3000, warmup=1000, draws=1000, chains=4),
    "elastic": dict(samplers=COMMON, emcee=1500, warmup=500, draws=500, chains=4),
    "hierarchy": dict(
        samplers=COMMON,
        starts=dict(NUTS="prior", emcee="prior"),  # the MAP is far from the bulk
        emcee=6000,
        warmup=1000,
        draws=3000,  # at 1000 the rank-normalised R-hat was 1.021
        chains=4,
    ),
    "omp": dict(
        samplers=COMMON + EXTRA,
        emcee=3000,
        ptemcee=2000,
        temps=8,
        warmup=300,
        draws=1000,
        chains=2,
        hmc_steps=20,
        # two islands: at 1024/512 pocomc's split between them swings from
        # seed to seed (docs/benchmarks.md)
        pocomc=dict(n_effective=4096, n_active=2048),
    ),
}
# the two GP optical-model problems of the notebooks, at omp's budgets
BUDGETS["alpha_gp"] = dict(BUDGETS["omp"])
BUDGETS["n40ca_gp"] = dict(BUDGETS["omp"])
RHAT_MAX = 1.01
#: largest bias and width error, in reference standard deviations
TOLERANCE = 0.1
#: the settings of the samplers that start from the prior: the cheapest that
#: are correct on ``elastic`` over three seeds (the defaults are not, and no
#: setting tried is correct on the funnels gp and modes; docs/benchmarks.md).
#: ``omp`` sets its own pocomc resolution in ``BUDGETS``
DYNESTY = dict(nlive=500, sample="rwalk")
POCOMC = dict(n_effective=1024, n_active=512)


# -- shared pieces -------------------------------------------------------------


def _calls(warmup=0, sampling=0, evidence=0):
    """A run's calls by phase.  A call is one evaluation of the log density
    and, for a gradient sampler, its gradient: emcee's burn-in, NUTS's
    warm-up and pocomc's tempering are ``warmup``; pocomc's importance
    sampling for the evidence is ``evidence``; ``search`` is filled by
    ``_charge``."""
    return dict(
        search=0, warmup=int(warmup), sampling=int(sampling), evidence=int(evidence)
    )


def _summary(chains, wall, calls):
    """``chains``: (n_chains, n_draws, ndim)."""
    flat = chains.reshape(-1, chains.shape[-1])
    ess = np.asarray(blackjax.diagnostics.ess_bulk(chains))
    ess_tail = np.asarray(blackjax.diagnostics.ess_tail(chains))
    rhat = np.asarray(blackjax.diagnostics.rhat(chains))
    return dict(
        wall_s=wall,
        calls=calls,
        min_ess=float(ess.min()),
        min_ess_tail=float(ess_tail.min()),
        max_rhat=float(rhat.max()),
        mean=flat.mean(0).tolist(),
        sd=flat.std(0).tolist(),
        mcse=(flat.std(0) / np.sqrt(ess)).tolist(),
    )


def _weighted(samples, weights, wall, calls, logz=None, logz_err=None):
    """Summary of importance-weighted draws; ESS is Kish's."""
    w = weights / weights.sum()
    mean = w @ samples
    sd = np.sqrt(w @ (samples - mean) ** 2)
    ess = 1.0 / np.sum(w**2)
    return dict(
        wall_s=wall,
        calls=calls,
        min_ess=float(ess),
        logz=logz,
        logz_err=logz_err,
        mean=mean.tolist(),
        sd=sd.tolist(),
        mcse=(sd / np.sqrt(ess)).tolist(),
    )


def _ball(problem, start, n, seed):
    z = problem.to_unconstrained(start)
    dz = 0.01 * np.random.default_rng(seed).standard_normal((n, problem.ndim))
    return np.array([problem.to_theta(z + d) for d in dz])


def _charge(out, search):
    """Charge a start search's wall time and calls to a sampler (none for
    prior draws).  An L-BFGS-B step evaluates the density and its gradient,
    one call."""
    if search is None:
        return out
    out["wall_s"] += search.wall
    out["start_search_s"] = search.wall
    out["calls"]["search"] = int(search.nfev)
    return out


def _tau(kept):
    """emcee's integrated autocorrelation time of ``kept`` (steps, walkers, dim)."""
    tau = float(np.max(emcee.autocorr.integrated_time(kept, quiet=True)))
    return dict(tau=tau, tau_reliable=len(kept) > 50 * tau)


# -- ensemble samplers, from the MAP or the prior -------------------------------


def _walkers(problem, start, n, seed):
    """A ball about ``start``, or prior draws if it is None."""
    if start is None:
        return problem.sample_prior(n, rng=seed)
    return _ball(problem, start.x, n, seed)


def run_emcee(problem, b, start, seed=0):
    n_walkers = max(32, 2 * problem.ndim + 2)
    steps = b["emcee"]
    t0 = time.perf_counter()
    sampler = emcee.EnsembleSampler(
        n_walkers, problem.ndim, problem.log_posterior_batch, vectorize=True
    )
    sampler.random_state = np.random.RandomState(seed).get_state()
    sampler.run_mcmc(_walkers(problem, start, n_walkers, seed), steps, progress=False)
    wall = time.perf_counter() - t0
    kept = sampler.get_chain(discard=steps // 3)
    burn = n_walkers * (steps // 3)
    calls = _calls(warmup=burn, sampling=n_walkers * steps - burn)
    out = _summary(np.swapaxes(kept, 0, 1), wall, calls)
    out.update(walkers=n_walkers, steps=steps, **_tau(kept))
    return _charge(out, start)


def run_ptemcee(problem, b, start, seed=0):
    import ptemcee

    n_walkers = max(32, 2 * problem.ndim + 2)
    steps, temps = b["ptemcee"], b["temps"]
    betas = ptemcee.make_ladder(problem.ndim, temps, Tmax=np.inf)  # to beta = 0

    def mapper(_, points):
        """Every proposal, at every temperature, in one vmapped call each for
        the prior and the likelihood; ptemcee's own rule: log L = 0 where the
        prior is -inf."""
        points = np.asarray(points)
        lp = problem.log_prior_batch(points)
        ll = np.where(np.isfinite(lp), problem.log_likelihood_batch(points), 0.0)
        return zip(ll, lp)

    t0 = time.perf_counter()
    sampler = ptemcee.Sampler(
        n_walkers,
        problem.ndim,
        problem.log_likelihood,  # what mapper's ignored evaluator would call
        problem.log_prior,
        betas=betas,
        adaptive=True,
        mapper=mapper,
    )
    x0 = _walkers(problem, start, temps * n_walkers, seed)
    x0 = x0.reshape(temps, n_walkers, -1)
    chain = sampler.chain(x0, random=np.random.RandomState(seed))
    chain.run(steps)
    wall = time.perf_counter() - t0
    kept = chain.x[steps // 3 :, 0]  # the beta = 1 walkers, (steps, walkers, dim)
    burn = temps * n_walkers * (steps // 3)  # every rung counts
    calls = _calls(warmup=burn, sampling=temps * n_walkers * steps - burn)
    out = _summary(np.swapaxes(kept, 0, 1), wall, calls)
    logz, logz_err = _ti_log_evidence(chain, fburnin=1 / 3)
    out.update(
        walkers=n_walkers,
        temps=temps,
        steps=steps,
        logz=float(logz),
        logz_err=float(logz_err),
        **_tau(kept),
    )
    return _charge(out, start)


def _ti_log_evidence(chain, fburnin):
    """``chain.log_evidence_estimate`` of ptemcee, which calls ``np.trapz``
    (removed in NumPy 2): thermodynamic integration of the mean log
    likelihood over the final ladder, its error the difference from every
    other rung."""
    start = int(chain.logl.shape[0] * fburnin + 0.5)
    logls = chain.logl[start:].mean(axis=(0, 2))
    betas = chain.betas[-1]
    order = np.argsort(betas)[::-1]
    betas, logls = betas[order], logls[order]
    if betas[-1] != 0:  # the hottest rung stands in at beta = 0
        betas2 = np.concatenate((betas[::2], [0]))
        logls2 = np.concatenate((logls[::2], [logls[-1]]))
        betas, logls = np.concatenate((betas, [0])), np.append(logls, logls[-1])
    else:
        betas2 = np.concatenate((betas[:-1:2], [0]))
        logls2 = np.concatenate((logls[:-1:2], [logls[-1]]))
    logz = -np.trapezoid(logls, betas)
    return logz, abs(logz + np.trapezoid(logls2, betas2))


# -- nested and preconditioned Monte Carlo, from the prior -----------------------


def run_dynesty(problem, b, seed=0, summary=None, **settings):
    t0 = time.perf_counter()
    sampler = dynesty.NestedSampler(
        problem.log_likelihood,
        problem.prior_transform,
        problem.ndim,
        rstate=np.random.default_rng(seed),
        **(DYNESTY | settings),
    )
    sampler.run_nested(dlogz=0.5, print_progress=False)
    wall = time.perf_counter() - t0
    res = sampler.results
    weights = np.exp(res.logwt - res.logz[-1])
    out = _weighted(
        res.samples,
        weights,
        wall,
        _calls(sampling=np.sum(res.ncall)),
        float(res.logz[-1]),
        float(res.logzerr[-1]),
    )
    if summary is not None:
        out.update(summary(np.asarray(res.samples), weights))
    return out


def run_pocomc(problem, b, seed=0, summary=None, **settings):
    """``summary(theta, weights) -> dict``, if given, adds to the result
    anything the moments cannot show (a mode's share, say).  ``n_total`` and
    ``n_evidence`` among ``settings`` go to ``run`` (pocomc's defaults
    otherwise, 4096 each); the rest configure the sampler."""
    import pocomc
    from scipy import stats

    def log_likelihood(u):  # every particle in one vmapped call
        return problem.log_likelihood_batch(problem.prior_transform_batch(u))

    prior = pocomc.Prior([stats.uniform(0.0, 1.0)] * problem.ndim)
    settings = POCOMC | b.get("pocomc", {}) | settings
    lengths = {k: settings.pop(k) for k in ("n_total", "n_evidence") if k in settings}
    t0 = time.perf_counter()
    sampler = pocomc.Sampler(
        prior, log_likelihood, vectorize=True, random_state=seed, **settings
    )
    sampler.run(progress=False, **lengths)
    wall = time.perf_counter() - t0
    u, weights, _, _ = sampler.posterior()
    logz, logz_err = sampler.evidence()
    theta = problem.prior_transform_batch(u)
    # the cumulative calls of each iteration: tempering until beta first
    # reaches 1, then sampling at beta = 1, then the evidence's draws
    beta, cumulative = (sampler.particles.get(k) for k in ("beta", "calls"))
    tempered = int(cumulative[np.argmax(np.asarray(beta) >= 1.0)])
    calls = _calls(
        warmup=tempered,
        sampling=cumulative[-1] - tempered,
        evidence=sampler.calls - cumulative[-1],
    )
    out = _weighted(theta, weights, wall, calls, float(logz), float(logz_err))
    if summary is not None:
        out.update(summary(np.asarray(theta), np.asarray(weights)))
    return out


# -- HMC and NUTS, from the MAP ---------------------------------------------------


def _blackjax_chain(problem, kernel, key, z0, warmup, draws, **fixed):
    logdensity = problem.log_posterior_unconstrained
    k_warm, k_draw = jax.random.split(key)
    adapt = blackjax.window_adaptation(kernel, logdensity, **fixed)
    (state, tuned), warm_info = adapt.run(k_warm, z0, num_steps=warmup)
    step = jax.jit(kernel(logdensity, **tuned).step)

    def one_step(state, key):
        state, info = step(key, state)
        return state, (state.position, info.is_divergent, info.num_integration_steps)

    _, (zs, divergent, n_leapfrog) = jax.lax.scan(
        one_step, state, jax.random.split(k_draw, draws)
    )
    warm = int(warm_info.info.num_integration_steps.sum())
    return zs, int(divergent.sum()), warm, int(n_leapfrog.sum())


def _run_blackjax(problem, b, mode, kernel, seed=0, **fixed):
    """From a ball about ``mode``, or from prior draws if it is None."""
    if mode is None:
        starts = problem.sample_prior(b["chains"], rng=seed)
    else:
        starts = _ball(problem, mode.x, b["chains"], seed)
    keys = jax.random.split(jax.random.key(seed), b["chains"])
    t0 = time.perf_counter()
    chains, divergent, warm, draw = [], 0, 0, 0
    for key, theta0 in zip(keys, starts):
        zs, div, n_warm, n_draw = _blackjax_chain(
            problem,
            kernel,
            key,
            problem.to_unconstrained(theta0),
            b["warmup"],
            b["draws"],
            **fixed,
        )
        chains.append(np.asarray(jax.vmap(problem.to_theta)(zs)))
        divergent, warm, draw = divergent + div, warm + n_warm, draw + n_draw
    wall = time.perf_counter() - t0
    out = _summary(np.stack(chains), wall, _calls(warmup=warm, sampling=draw))
    out.update(divergences=divergent, warmup=b["warmup"], draws=b["draws"], **fixed)
    return _charge(out, mode)


def run_nuts(problem, b, mode, seed=0):
    return _run_blackjax(problem, b, mode, blackjax.nuts, seed)


def run_hmc(problem, b, mode, seed=0):
    return _run_blackjax(
        problem, b, mode, blackjax.hmc, seed, num_integration_steps=b["hmc_steps"]
    )


def _pymc_logdensity(problem):
    """A PyTensor Op for ``log_posterior_unconstrained``, its gradient a second
    Op on ``jax.grad`` (PyMC's black-box likelihood pattern).  PyMC runs both
    at every point; the pair is one call, counted by the gradient Op."""
    import pytensor.tensor as pt
    from pytensor.graph.basic import Apply
    from pytensor.graph.op import Op

    value = jax.jit(problem.log_posterior_unconstrained)
    grad = jax.jit(jax.grad(problem.log_posterior_unconstrained))

    class Grad(Op):
        calls = 0

        def make_node(self, z):
            z = pt.as_tensor_variable(z)
            return Apply(self, [z], [z.type()])

        def perform(self, node, inputs, outputs):
            Grad.calls += 1
            outputs[0][0] = np.asarray(grad(inputs[0]), dtype=float)

    class LogDensity(Op):
        def make_node(self, z):
            z = pt.as_tensor_variable(z)
            return Apply(self, [z], [pt.dscalar()])

        def perform(self, node, inputs, outputs):
            outputs[0][0] = np.asarray(value(inputs[0]), dtype=float)

        def grad(self, inputs, output_grads):
            return [output_grads[0] * Grad()(inputs[0])]

    return LogDensity(), Grad


def _run_pymc(problem, b, mode, step, seed=0):
    import pymc as pm

    op, grads = _pymc_logdensity(problem)
    starts = _ball(problem, mode.x, b["chains"], seed)
    t0 = time.perf_counter()
    with pm.Model():
        z = pm.Flat("z", shape=problem.ndim)
        pm.Potential("logp", op(z))
        idata = pm.sample(
            draws=b["draws"],
            tune=b["warmup"],
            chains=b["chains"],
            cores=1,  # a forked process under JAX can deadlock
            step=step(),
            initvals=[{"z": problem.to_unconstrained(s)} for s in starts],
            random_seed=seed,
            progressbar=False,
            compute_convergence_checks=False,
        )
    wall = time.perf_counter() - t0
    zs = idata.posterior["z"].values
    chains = np.array([[problem.to_theta(z) for z in chain] for chain in zs])
    stats = idata.sample_stats
    # leapfrog steps of the kept draws; the rest went to tuning
    drawn = int(stats["tree_size" if "tree_size" in stats else "n_steps"].sum())
    out = _summary(chains, wall, _calls(warmup=grads.calls - drawn, sampling=drawn))
    out.update(
        divergences=int(stats["diverging"].sum()) if "diverging" in stats else None,
        warmup=b["warmup"],
        draws=b["draws"],
    )
    return _charge(out, mode)


def run_pymc_nuts(problem, b, mode, seed=0):
    import pymc as pm

    return _run_pymc(problem, b, mode, pm.NUTS, seed)


def run_pymc_hmc(problem, b, mode, seed=0):
    import pymc as pm

    return _run_pymc(problem, b, mode, lambda: pm.HamiltonianMC(path_length=2.0), seed)


# -- the table --------------------------------------------------------------------


def converged(r):
    """R-hat at most ``RHAT_MAX`` and, for an ensemble, a run longer than
    50 tau: walkers that swap or stretch toward one another can agree with
    each other long before they have explored the posterior."""
    return r.get("max_rhat", 1.0) <= RHAT_MAX and r.get("tau_reliable", True)


def errors(r, ref):
    """``(bias, width error)`` of run ``r`` against ``ref``: the largest
    ``|mean - mean_ref|`` and ``|log(sd / sd_ref)|`` over the parameters,
    in reference standard deviations."""
    sd = np.asarray(ref["sd"])
    bias = np.abs(np.subtract(r["mean"], ref["mean"])) / sd
    return float(bias.max()), float(np.abs(np.log(np.divide(r["sd"], sd))).max())


def correct(r, ref):
    """Converged, and (given a reference) within ``TOLERANCE`` of it."""
    return converged(r) and (ref is None or max(errors(r, ref)) <= TOLERANCE)


PHASES = ("search", "warmup", "sampling", "evidence")
HEAD = (
    "wall time",
    "calls",
    "search",
    "warm-up",
    "sampling",
    "evidence",
    "ms/call",
) + ("bias", "width", "min ESS", "ESS/s", "ESS/1k calls", "R-hat", "τ (steps)", "log Z")


def _row(v, ref):
    """One sampler's cells under ``HEAD``; ESS only when correct."""
    ok = correct(v, ref)
    bias = width = "—"
    if ref is not None:
        b, w = errors(v, ref)
        bias = f"{b:.2f}" + ("" if b <= TOLERANCE else " ✗")
        width = f"{w:.2f}" + ("" if w <= TOLERANCE else " ✗")
    total = sum(v["calls"].values())
    share = [f"{100 * v['calls'][k] / total:.0f}%" for k in PHASES]
    rhat = v.get("max_rhat")
    rhat = "—" if rhat is None else f"{rhat:.3f}" + ("" if rhat <= RHAT_MAX else " ✗")
    tau = ""
    if "tau" in v:
        tau = f"{v['tau']:.0f}" + ("" if v["tau_reliable"] else " (< 50 τ)")
    logz = "" if v.get("logz") is None else f"{v['logz']:.1f} ± {v['logz_err']:.2f}"
    rates = ("—", "—", "—")
    if ok:
        ess = v["min_ess"]
        rates = (f"{ess:.0f}", f"{ess / v['wall_s']:.2f}", f"{1e3 * ess / total:.2f}")
    cost = (f"{v['wall_s']:.0f} s", str(total), *share)
    per_call = f"{1e3 * v['wall_s'] / total:.2f}"
    return (*cost, per_call, bias, width, *rates, rhat, tau, logz)


def _reference_cells(ref):
    logz = ""
    if ref.get("logz") is not None and ref["method"] == "quadrature":
        logz = f"{ref['logz']:.2f} (exact)"
    elif ref.get("logz") is not None:
        logz = f"{ref['logz']:.2f} ± {ref['logz_err']:.2f}"
    return ("",) * (len(HEAD) - 1) + (logz,)


def usable_reference(name, names, ref):
    """``ref`` if it may score runs of problem ``name`` with parameters
    ``names``: it must name the same parameters (a reference of a redefined
    problem is refused), and a rejected or uncorroborated reference is
    flagged.  None if there is no reference."""
    if ref is None:
        return None
    if ref.get("names") is None:
        raise ValueError(
            f"{name}: the reference does not name its parameters, so it cannot be "
            "matched to the runs; rerun it (bench_samplers.py --reference)"
        )
    if names is not None and ref["names"] != list(names):
        raise ValueError(
            f"{name}: the reference's parameters {ref['names']} are not the "
            f"run's {list(names)}; rerun the reference for this problem"
        )
    for flag, meaning in (
        ("accepted", "its pocomc runs disagree"),
        ("corroborated", "the dynesty check disagrees"),
    ):
        if ref.get(flag) is False:
            print(f"WARNING: {name}'s reference is not {flag}: {meaning}")
    return ref


def _reference_flags(ref):
    flags = [f for f in ("accepted", "corroborated") if ref.get(f) is False]
    return "".join(f" (NOT {f.upper()})" for f in flags)


def report(results, markdown=False, refs=None):
    """Print the comparison table of a run, scored against ``refs``."""
    refs = refs or {}
    if markdown:
        print("| problem | sampler | " + " | ".join(HEAD) + " |")
        print("|" + "---|" * (len(HEAD) + 2))
    else:
        print(f"{'':>12s}  " + "  ".join(f"{h:>10s}" for h in HEAD))
    for name, d in results.items():
        ref = usable_reference(name, d.get("names"), refs.get(name))
        if not markdown:
            print(f"{name} ({d['ndim']} parameters)")
        for sampler, v in d["samplers"].items():
            if markdown:
                cells = (f"{name} ({d['ndim']})", sampler, *_row(v, ref))
                print("| " + " | ".join(cells) + " |")
            else:
                print(
                    f"{sampler:>12s}  " + "  ".join(f"{c:>10s}" for c in _row(v, ref)),
                    flush=True,
                )
        if ref is None:
            print("  (no reference: bias not scored)")
        elif markdown:
            cells = (
                f"{name} ({d['ndim']})",
                f"reference: {ref['method']}{_reference_flags(ref)}",
            )
            print("| " + " | ".join(cells + _reference_cells(ref)) + " |")
        else:
            logz = "" if ref.get("logz") is None else f", log Z = {ref['logz']:.2f}"
            print(f"  reference: {ref['method']}{_reference_flags(ref)}{logz}")


RUNNERS = {
    "emcee": (run_emcee, "map"),
    "ptemcee": (run_ptemcee, "map"),
    "dynesty": (run_dynesty, None),
    "pocomc": (run_pocomc, None),
    "NUTS": (run_nuts, "map"),
    "HMC": (run_hmc, "map"),
    "PyMC NUTS": (run_pymc_nuts, "map"),
    "PyMC HMC": (run_pymc_hmc, "map"),
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--problems", nargs="*", default=list(BUDGETS))
    ap.add_argument("--report", type=Path, help="print a saved run and exit")
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument(
        "--reference",
        action="store_true",
        help="compute the problems' reference posteriors into --references",
    )
    ap.add_argument(
        "--references",
        type=Path,
        default=REFERENCES,
        help=f"the references file to read and, with --reference, to write "
        f"(default {REFERENCES.name}; give each parallel job its own and merge "
        "them with merge_results.py)",
    )
    ap.add_argument(
        "--only",
        nargs="*",
        choices=list(RUNNERS),
        help="re-run these samplers into the saved --out file, keeping the rest",
    )
    args = ap.parse_args(argv)
    refs = json.loads(args.references.read_text()) if args.references.exists() else {}
    if args.report:
        problems = json.loads(args.report.read_text())["problems"]
        return report(problems, args.markdown, refs)
    if args.reference:
        from reference import reference

        for name in args.problems:
            t0 = time.perf_counter()
            problem = PROBLEMS[name]()[0]
            refs[name] = reference(name, problem)
            refs[name]["names"] = problem.names
            refs[name]["wall_s"] = time.perf_counter() - t0
            print(f"{name}: {refs[name]['method']}, {refs[name]['wall_s']:.0f} s")
            args.references.write_text(json.dumps(refs, indent=1))
        return None
    saved = {}
    if args.out and args.out.exists():
        saved = json.loads(args.out.read_text())["problems"]
    results = dict(saved)
    for name in args.problems:
        problem, _ = PROBLEMS[name]()
        b = BUDGETS[name]
        todo = [s for s in b["samplers"] if not args.only or s in args.only]
        start_of = {s: b.get("starts", {}).get(s, RUNNERS[s][1]) for s in todo}
        starts = {"prior": None}
        if "map" in start_of.values():
            starts["map"] = rx.optimize.find_map(problem, rng=0)
        r = dict(saved.get(name, {}).get("samplers", {})) if args.only else {}
        for sampler in todo:
            run, start = RUNNERS[sampler][0], start_of[sampler]
            r[sampler] = run(problem, b, starts[start]) if start else run(problem, b)
            r[sampler]["start"] = start or "prior"
            one = {name: dict(ndim=problem.ndim, samplers={sampler: r[sampler]})}
            report(one, refs=refs)
        r = {k: r[k] for k in b["samplers"] if k in r}
        results[name] = dict(ndim=problem.ndim, names=problem.names, samplers=r)
        report({name: results[name]}, refs=refs)
        meta = dict(jax=jax.__version__, blackjax=blackjax.__version__)
        if args.out:
            args.out.write_text(json.dumps(dict(meta=meta, problems=results), indent=1))


if __name__ == "__main__":
    main()
