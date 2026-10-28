"""Compare emcee, dynesty and NUTS on the benchmark problems, per effective sample.

JAX rxmc only (NUTS needs the gradient)::

    python benchmarks/bench_samplers.py --out benchmarks/results/samplers.json

Each sampler runs a fixed budget from the prior on the same problem:

* emcee: ``max(32, 2 ndim + 2)`` walkers on ``log_posterior_batch`` with
  ``vectorize=True``, the first third discarded, walkers taken as chains;
* dynesty: ``rwalk`` with 200 live points to ``dlogz = 0.5``, its effective
  sample size Kish's ``(sum w)^2 / sum w^2`` of the importance weights;
* NUTS: blackjax's window adaptation, then draws, on
  ``log_posterior_unconstrained``, four chains one after another (under
  ``vmap`` every chain would wait for the longest trajectory).

Recorded per sampler: wall time (compilation included, as a user sees it),
density or gradient evaluations, the smallest bulk ESS over the parameters
(``blackjax.diagnostics``, the same estimator for every chain-based sampler),
ESS per second, the largest split R-hat, and the posterior means and
standard deviations.  A chain-based run with R-hat above 1.01 has not
converged and its ESS means nothing; the means of the converged runs are
checked against each other in Monte Carlo standard errors.

``--report results.json`` prints the table again from a saved run (as
markdown with ``--markdown``).
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

#: per problem: emcee steps, NUTS warm-up steps and draws per chain
BUDGETS = {
    "linear": dict(emcee=3000, warmup=1000, draws=1000),
    "gp": dict(emcee=3000, warmup=1000, draws=1000),
    "elastic": dict(emcee=1500, warmup=500, draws=500),
    "hierarchy": dict(emcee=6000, warmup=1000, draws=1000),
}


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


def run_emcee(problem, steps, seed=0):
    n_walkers = max(32, 2 * problem.ndim + 2)
    t0 = time.perf_counter()
    sampler = emcee.EnsembleSampler(
        n_walkers, problem.ndim, problem.log_posterior_batch, vectorize=True
    )
    sampler.random_state = np.random.RandomState(seed).get_state()
    sampler.run_mcmc(problem.sample_prior(n_walkers, rng=seed), steps, progress=False)
    wall = time.perf_counter() - t0
    chains = np.swapaxes(sampler.get_chain(discard=steps // 3), 0, 1)
    out = _summary(chains, wall, n_walkers * steps)
    out.update(walkers=n_walkers, steps=steps)
    return out


def run_dynesty(problem, seed=0, nlive=200):
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
    w = np.exp(res.logwt - res.logz[-1])
    ess = w.sum() ** 2 / np.sum(w**2)
    samples = res.samples_equal(rstate=np.random.default_rng(seed))
    sd = samples.std(0)
    return dict(
        wall_s=wall,
        evaluations=int(np.sum(res.ncall)),
        min_ess=float(ess),
        ess_per_s=float(ess / wall),
        logz=float(res.logz[-1]),
        logz_err=float(res.logzerr[-1]),
        mean=samples.mean(0).tolist(),
        sd=sd.tolist(),
        mcse=(sd / np.sqrt(ess)).tolist(),
    )


def _nuts_chain(problem, key, z0, warmup, draws):
    logdensity = problem.log_posterior_unconstrained
    k_warm, k_draw = jax.random.split(key)
    adapt = blackjax.window_adaptation(blackjax.nuts, logdensity)
    (state, tuned), warm_info = adapt.run(k_warm, z0, num_steps=warmup)
    step = jax.jit(blackjax.nuts(logdensity, **tuned).step)

    def one_step(state, key):
        state, info = step(key, state)
        return state, (state.position, info.is_divergent, info.num_integration_steps)

    _, (zs, divergent, n_leapfrog) = jax.lax.scan(
        one_step, state, jax.random.split(k_draw, draws)
    )
    evals = int(warm_info.info.num_integration_steps.sum() + n_leapfrog.sum())
    return zs, int(divergent.sum()), evals, float(tuned["step_size"])


def run_nuts(problem, warmup, draws, seed=0, n_chains=4):
    starts = problem.sample_prior(n_chains, rng=seed)
    keys = jax.random.split(jax.random.key(seed), n_chains)
    t0 = time.perf_counter()
    chains, divergent, evals, steps = [], 0, 0, []
    for key, theta0 in zip(keys, starts):
        zs, div, n, eps = _nuts_chain(
            problem, key, problem.to_unconstrained(theta0), warmup, draws
        )
        chains.append(np.asarray(jax.vmap(problem.to_theta)(zs)))
        divergent, evals = divergent + div, evals + n
        steps.append(eps)
    wall = time.perf_counter() - t0
    out = _summary(np.stack(chains), wall, evals)
    out.update(divergences=divergent, step_sizes=steps, warmup=warmup, draws=draws)
    return out


RHAT_MAX = 1.01


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


def report(results, markdown=False):
    """Print the comparison table of a run."""
    if markdown:
        print(
            "| problem | sampler | wall time | evaluations | min ESS | ESS/s | R-hat |"
        )
        print("|---|---|---|---|---|---|---|")
    for name, d in results.items():
        if not markdown:
            print(f"{name} ({d['ndim']} parameters)")
        for sampler, v in d["samplers"].items():
            rhat = v.get("max_rhat")
            flag = "" if converged(v) else "  not converged"
            if markdown:
                ess = f"{v['min_ess']:.0f}" if converged(v) else "—"
                rate = f"{v['ess_per_s']:.2f}" if converged(v) else "—"
                rh = (
                    "—"
                    if rhat is None
                    else f"{rhat:.3f}" + ("" if converged(v) else " ✗")
                )
                print(
                    f"| {name} ({d['ndim']}) | {sampler} | {v['wall_s']:.0f} s | "
                    f"{v['evaluations']} | {ess} | {rate} | {rh} |"
                )
            else:
                print(
                    f"  {sampler:8s} {v['wall_s']:8.1f} s  {v['evaluations']:9d} evals  "
                    f"min ESS {v['min_ess']:8.0f}  ESS/s {v['ess_per_s']:8.2f}  "
                    f"R-hat {rhat if rhat is not None else float('nan'):.3f}{flag}"
                )
        if not markdown:
            print(f"  converged means agree to {agreement(d['samplers']):.1f} MCSE")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--problems", nargs="*", default=list(BUDGETS))
    ap.add_argument("--report", type=Path, help="print a saved run and exit")
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args(argv)
    if args.report:
        return report(json.loads(args.report.read_text())["problems"], args.markdown)
    results = {}
    for name in args.problems:
        problem, _ = PROBLEMS[name]()
        b = BUDGETS[name]
        print(f"{name} ({problem.ndim} parameters)", flush=True)
        r = {
            "emcee": run_emcee(problem, b["emcee"]),
            "dynesty": run_dynesty(problem),
            "NUTS": run_nuts(problem, b["warmup"], b["draws"]),
        }
        results[name] = dict(ndim=problem.ndim, names=problem.names, samplers=r)
        results[name]["max_mean_difference_in_mcse"] = agreement(r)
        report({name: results[name]})
    meta = dict(jax=jax.__version__, blackjax=blackjax.__version__)
    args.out.write_text(json.dumps(dict(meta=meta, problems=results), indent=1))


if __name__ == "__main__":
    main()
