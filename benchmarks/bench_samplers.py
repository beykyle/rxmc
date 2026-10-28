"""Compare samplers on the benchmark problems, per effective sample.

JAX rxmc only (the gradient samplers need the gradient)::

    python benchmarks/bench_samplers.py --out benchmarks/results/samplers.json
    python benchmarks/bench_samplers.py --out ... --problems omp --only ptemcee pocomc
    python benchmarks/bench_samplers.py --report benchmarks/results/samplers.json --markdown

emcee, dynesty and blackjax NUTS run on every problem; the optical model
``omp`` also runs ptemcee, pocomc, blackjax HMC and PyMC's NUTS and HMC.  The
extra samplers are the ``samplers`` extra (install torch from
``https://download.pytorch.org/whl/cpu`` first, for pocomc, to avoid the CUDA
wheels).

Each sampler starts the way it is usually used, and pays for the search:

* ensemble samplers (emcee, ptemcee) from a small ball about the maximum of
  the likelihood, ``rx.optimize.find_mle`` (Nelder-Mead, no gradient);
* HMC and NUTS (blackjax, PyMC) from a ball about the posterior mode,
  ``rx.optimize.find_map`` (L-BFGS-B with the exact gradient), on
  ``log_posterior_unconstrained``;
* nested sampling (dynesty) and preconditioned Monte Carlo (pocomc) from the
  prior, as their algorithms do; pocomc samples the unit cube with
  ``log_likelihood(prior_transform(u))``, the prior exactly.

A ball is ``to_theta(to_unconstrained(start) + 0.01 N(0, I))``, inside the
support.  A search's wall time and evaluations are added to every sampler
that starts from it.

Recorded per sampler: wall time, density or gradient evaluations, the
smallest bulk ESS over the parameters (``blackjax.diagnostics`` for every
chain-based sampler, walkers taken as chains for the ensembles, which
flatters them; Kish's ESS of the importance weights for dynesty and pocomc),
ESS per second, the largest split R-hat, the ensembles' integrated
autocorrelation time tau (flagged under the 50 tau emcee asks for), the
evidence where the algorithm gives one, and the posterior means and standard
deviations.  A chain-based run with R-hat above 1.01 has not converged and
its ESS is not reported; the converged runs' means are checked against each
other in Monte Carlo standard errors.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import blackjax
import dynesty
import emcee
import jax
import numpy as np
from problems import PROBLEMS

import rxmc as rx

# PyMC's graphs here hold one Op that calls JAX, so PyTensor's C backend
# (which needs a compiler and the Python headers) would buy nothing
os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")

#: per problem: the samplers, and their budgets
COMMON = ("emcee", "dynesty", "NUTS")
EXTRA = ("ptemcee", "pocomc", "HMC", "PyMC NUTS", "PyMC HMC")
BUDGETS = {
    "linear": dict(samplers=COMMON, emcee=3000, warmup=1000, draws=1000, chains=4),
    "gp": dict(samplers=COMMON, emcee=3000, warmup=1000, draws=1000, chains=4),
    "elastic": dict(samplers=COMMON, emcee=1500, warmup=500, draws=500, chains=4),
    "hierarchy": dict(samplers=COMMON, emcee=6000, warmup=1000, draws=1000, chains=4),
    "omp": dict(
        samplers=COMMON + EXTRA,
        emcee=3000,
        ptemcee=2000,
        temps=8,
        warmup=300,
        draws=1000,
        chains=2,
        hmc_steps=20,
    ),
}
RHAT_MAX = 1.01


# -- shared pieces -------------------------------------------------------------


def _summary(chains, wall, evals):
    """``chains``: (n_chains, n_draws, ndim)."""
    flat = chains.reshape(-1, chains.shape[-1])
    ess = np.asarray(blackjax.diagnostics.effective_sample_size(chains))
    rhat = np.asarray(blackjax.diagnostics.potential_scale_reduction(chains))
    return dict(
        wall_s=wall,
        evaluations=int(evals),
        min_ess=float(ess.min()),
        ess_per_s=float(ess.min() / wall),
        max_rhat=float(rhat.max()),
        mean=flat.mean(0).tolist(),
        sd=flat.std(0).tolist(),
        mcse=(flat.std(0) / np.sqrt(ess)).tolist(),
    )


def _weighted(samples, weights, wall, evals, logz=None, logz_err=None):
    """Summary of importance-weighted draws; ESS is Kish's."""
    w = weights / weights.sum()
    mean = w @ samples
    sd = np.sqrt(w @ (samples - mean) ** 2)
    ess = 1.0 / np.sum(w**2)
    return dict(
        wall_s=wall,
        evaluations=int(evals),
        min_ess=float(ess),
        ess_per_s=float(ess / wall),
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
    """Add a start search's cost to a sampler's."""
    out["wall_s"] += search.wall
    out["evaluations"] += search.nfev
    out["ess_per_s"] = out["min_ess"] / out["wall_s"]
    out["start_search_s"] = search.wall
    return out


def _tau(kept):
    """emcee's integrated autocorrelation time of ``kept`` (steps, walkers, dim)."""
    tau = float(np.max(emcee.autocorr.integrated_time(kept, quiet=True)))
    return dict(tau=tau, tau_reliable=len(kept) > 50 * tau)


# -- ensemble samplers, from the MLE --------------------------------------------


def run_emcee(problem, b, mle, seed=0):
    n_walkers = max(32, 2 * problem.ndim + 2)
    steps = b["emcee"]
    t0 = time.perf_counter()
    sampler = emcee.EnsembleSampler(
        n_walkers, problem.ndim, problem.log_posterior_batch, vectorize=True
    )
    sampler.random_state = np.random.RandomState(seed).get_state()
    sampler.run_mcmc(_ball(problem, mle.x, n_walkers, seed), steps, progress=False)
    wall = time.perf_counter() - t0
    kept = sampler.get_chain(discard=steps // 3)
    out = _summary(np.swapaxes(kept, 0, 1), wall, n_walkers * steps)
    out.update(walkers=n_walkers, steps=steps, **_tau(kept))
    return _charge(out, mle)


def run_ptemcee(problem, b, mle, seed=0):
    import ptemcee

    n_walkers = max(32, 2 * problem.ndim + 2)
    steps, temps = b["ptemcee"], b["temps"]
    betas = ptemcee.make_ladder(problem.ndim, temps, Tmax=np.inf)  # to beta = 0
    t0 = time.perf_counter()
    sampler = ptemcee.Sampler(
        n_walkers,
        problem.ndim,
        problem.log_likelihood,
        problem.log_prior,
        betas=betas,
        adaptive=True,
    )
    x0 = _ball(problem, mle.x, temps * n_walkers, seed).reshape(temps, n_walkers, -1)
    chain = sampler.chain(x0, random=np.random.RandomState(seed))
    chain.run(steps)
    wall = time.perf_counter() - t0
    kept = chain.x[steps // 3 :, 0]  # the beta = 1 walkers, (steps, walkers, dim)
    out = _summary(np.swapaxes(kept, 0, 1), wall, temps * n_walkers * steps)
    logz, logz_err = _ti_log_evidence(chain, fburnin=1 / 3)
    out.update(
        walkers=n_walkers,
        temps=temps,
        steps=steps,
        logz=float(logz),
        logz_err=float(logz_err),
        **_tau(kept),
    )
    return _charge(out, mle)


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


def run_dynesty(problem, b, seed=0, nlive=200):
    t0 = time.perf_counter()
    sampler = dynesty.NestedSampler(
        problem.log_likelihood,
        problem.prior_transform,
        problem.ndim,
        nlive=nlive,
        sample="rwalk",
        rstate=np.random.default_rng(seed),
    )
    sampler.run_nested(dlogz=0.5, print_progress=False)
    wall = time.perf_counter() - t0
    res = sampler.results
    return _weighted(
        res.samples,
        np.exp(res.logwt - res.logz[-1]),
        wall,
        np.sum(res.ncall),
        float(res.logz[-1]),
        float(res.logzerr[-1]),
    )


def run_pocomc(problem, b, seed=0):
    import pocomc
    from scipy import stats

    def log_likelihood(u):
        return np.array([problem.log_likelihood(problem.prior_transform(x)) for x in u])

    prior = pocomc.Prior([stats.uniform(0.0, 1.0)] * problem.ndim)
    t0 = time.perf_counter()
    sampler = pocomc.Sampler(prior, log_likelihood, vectorize=True, random_state=seed)
    sampler.run(progress=False)
    wall = time.perf_counter() - t0
    u, weights, _, _ = sampler.posterior()
    logz, logz_err = sampler.evidence()
    theta = np.array([problem.prior_transform(x) for x in u])
    return _weighted(theta, weights, wall, sampler.calls, float(logz), float(logz_err))


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
    evals = int(warm_info.info.num_integration_steps.sum() + n_leapfrog.sum())
    return zs, int(divergent.sum()), evals


def _run_blackjax(problem, b, mode, kernel, seed=0, **fixed):
    starts = _ball(problem, mode.x, b["chains"], seed)
    keys = jax.random.split(jax.random.key(seed), b["chains"])
    t0 = time.perf_counter()
    chains, divergent, evals = [], 0, 0
    for key, theta0 in zip(keys, starts):
        zs, div, n = _blackjax_chain(
            problem,
            kernel,
            key,
            problem.to_unconstrained(theta0),
            b["warmup"],
            b["draws"],
            **fixed,
        )
        chains.append(np.asarray(jax.vmap(problem.to_theta)(zs)))
        divergent, evals = divergent + div, evals + n
    wall = time.perf_counter() - t0
    out = _summary(np.stack(chains), wall, evals)
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
    Op on ``jax.grad`` (PyMC's black-box likelihood pattern)."""
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

    op, counter = _pymc_logdensity(problem)
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
    out = _summary(chains, wall, counter.calls)
    stats = idata.sample_stats
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
    return r.get("max_rhat", 1.0) <= RHAT_MAX


def agreement(results):
    """Largest standardised difference of the posterior means between the
    converged samplers."""
    names = [k for k, r in results.items() if converged(r)]
    worst = 0.0
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            ra, rb = results[a], results[b]
            d = np.abs(np.subtract(ra["mean"], rb["mean"]))
            se = np.sqrt(np.square(ra["mcse"]) + np.square(rb["mcse"]))
            worst = max(worst, float(np.max(d / se)))
    return worst


def _row(v):
    """``(min ESS, ESS/s, R-hat, tau, log Z)`` as table text; ESS only when
    converged."""
    ok = converged(v)
    rhat = v.get("max_rhat")
    rhat = "—" if rhat is None else f"{rhat:.3f}" + ("" if ok else " ✗")
    tau = ""
    if "tau" in v:
        tau = f"{v['tau']:.0f}" + ("" if v["tau_reliable"] else " (< 50 τ)")
    logz = "" if v.get("logz") is None else f"{v['logz']:.1f} ± {v['logz_err']:.1f}"
    ess = f"{v['min_ess']:.0f}" if ok else "—"
    rate = f"{v['ess_per_s']:.2f}" if ok else "—"
    return ess, rate, rhat, tau, logz


def report(results, markdown=False):
    """Print the comparison table of a run."""
    head = ("problem", "sampler", "wall time", "evaluations", "min ESS", "ESS/s")
    head += ("R-hat", "τ (steps)", "log Z")
    if markdown:
        print("| " + " | ".join(head) + " |")
        print("|" + "---|" * len(head))
    for name, d in results.items():
        if not markdown:
            print(f"{name} ({d['ndim']} parameters)")
        for sampler, v in d["samplers"].items():
            cells = (f"{name} ({d['ndim']})", sampler, f"{v['wall_s']:.0f} s")
            cells += (str(v["evaluations"]), *_row(v))
            if markdown:
                print("| " + " | ".join(cells) + " |")
            else:
                print("  " + "  ".join(f"{c:>12s}" for c in cells[1:]), flush=True)
        if not markdown:
            print(f"  converged means agree to {agreement(d['samplers']):.1f} MCSE")


RUNNERS = {
    "emcee": (run_emcee, "mle"),
    "ptemcee": (run_ptemcee, "mle"),
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
        "--only",
        nargs="*",
        choices=list(RUNNERS),
        help="re-run these samplers into the saved --out file, keeping the rest",
    )
    args = ap.parse_args(argv)
    if args.report:
        return report(json.loads(args.report.read_text())["problems"], args.markdown)
    saved = {}
    if args.out and args.out.exists():
        saved = json.loads(args.out.read_text())["problems"]
    results = dict(saved)
    for name in args.problems:
        problem, _ = PROBLEMS[name]()
        b = BUDGETS[name]
        todo = [s for s in b["samplers"] if not args.only or s in args.only]
        starts = {}
        if any(RUNNERS[s][1] == "mle" for s in todo):
            starts["mle"] = rx.optimize.find_mle(problem, rng=0)
        if any(RUNNERS[s][1] == "map" for s in todo):
            starts["map"] = rx.optimize.find_map(problem, rng=0)
        r = dict(saved.get(name, {}).get("samplers", {})) if args.only else {}
        for sampler in todo:
            run, start = RUNNERS[sampler]
            r[sampler] = run(problem, b, starts[start]) if start else run(problem, b)
            report({name: dict(ndim=problem.ndim, samplers={sampler: r[sampler]})})
        r = {k: r[k] for k in b["samplers"] if k in r}
        results[name] = dict(ndim=problem.ndim, names=problem.names, samplers=r)
        results[name]["max_mean_difference_in_mcse"] = agreement(r)
        report({name: results[name]})
        meta = dict(jax=jax.__version__, blackjax=blackjax.__version__)
        if args.out:
            args.out.write_text(json.dumps(dict(meta=meta, problems=results), indent=1))


if __name__ == "__main__":
    main()
