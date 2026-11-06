"""Time and record ``Problem.log_posterior`` on the benchmark problems.

Runs against whichever rxmc is importable (the numpy baseline or the JAX
rewrite) and writes one JSON file::

    python benchmarks/bench_log_posterior.py --out benchmarks/results/numpy.json
    python benchmarks/bench_log_posterior.py --out benchmarks/results/jax.json
    python benchmarks/compare.py benchmarks/results/numpy.json benchmarks/results/jax.json

Recorded per problem: the values at the fixed parameter points (the value
regression), the first-call latency, the steady-state per-call time, the
gradient cost (``value_and_grad`` in JAX, a central finite difference of
``2 * ndim`` calls in the baseline), batch throughput at several batch sizes
and, with ``--emcee``, the wall time of a fixed emcee budget.  Timings are the
median and interquartile range over repeated calls after warm-up.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from problems import JAX, PROBLEMS, rxmc  # noqa: E402

BATCHES = (1, 32, 256)


def _block(v):
    """Wait for an asynchronous JAX result; a no-op on numpy values."""
    return v.block_until_ready() if hasattr(v, "block_until_ready") else v


def _timeit(fn, n_calls, min_time=0.2):
    times = []
    t_start = time.perf_counter()
    while len(times) < n_calls or (time.perf_counter() - t_start) < min_time:
        t0 = time.perf_counter()
        _block(fn())
        times.append(time.perf_counter() - t0)
        if len(times) >= 20 * n_calls:
            break
    q1, med, q3 = np.percentile(times, [25, 50, 75])
    return {"median": med, "iqr": q3 - q1, "n": len(times)}


def _fd_grad(f, theta, h=1e-6):
    g = np.empty_like(theta)
    for i in range(len(theta)):
        e = np.zeros_like(theta)
        e[i] = h * max(1.0, abs(theta[i]))
        g[i] = (float(f(theta + e)) - float(f(theta - e))) / (2 * e[i])
    return g


def bench(name, n_calls, emcee_steps):
    t0 = time.perf_counter()
    problem, thetas = PROBLEMS[name]()
    build = time.perf_counter() - t0
    out = {"ndim": problem.ndim, "names": problem.names, "build_s": build}

    t0 = time.perf_counter()
    _block(problem.log_posterior(thetas[0]))
    out["first_call_s"] = time.perf_counter() - t0
    out["values"] = [float(problem.log_posterior(t)) for t in thetas]
    out["thetas"] = np.asarray(thetas).tolist()

    rotation = iter(np.tile(np.arange(len(thetas)), 10**6))
    out["log_posterior"] = _timeit(
        lambda: problem.log_posterior(thetas[next(rotation)]), n_calls
    )

    if JAX:
        vg = problem.value_and_grad_log_posterior
        t0 = time.perf_counter()
        _block(vg(thetas[0])[1])
        out["grad_first_call_s"] = time.perf_counter() - t0
        out["grad"] = _timeit(lambda: vg(thetas[next(rotation)])[1], n_calls)
        out["grad_method"] = "value_and_grad"
    else:
        out["grad"] = _timeit(
            lambda: _fd_grad(problem.log_posterior, thetas[next(rotation)]),
            max(5, n_calls // (2 * problem.ndim)),
        )
        out["grad_method"] = "central finite difference"

    out["batch"] = {}
    rng = np.random.default_rng(1)
    for b in BATCHES:
        batch = thetas[rng.integers(0, len(thetas), b)]
        _block(problem.log_posterior_batch(batch))  # warm-up / compile
        t = _timeit(lambda: problem.log_posterior_batch(batch), max(3, n_calls // b))
        out["batch"][str(b)] = {**t, "theta_per_s": b / t["median"]}

    if emcee_steps:
        import emcee

        nwalkers = max(32, 2 * problem.ndim + 2)
        p0 = problem.sample_prior(nwalkers, rng=3)
        sampler = emcee.EnsembleSampler(
            nwalkers, problem.ndim, lambda t: float(problem.log_posterior(t))
        )
        t0 = time.perf_counter()
        sampler.run_mcmc(p0, emcee_steps, progress=False)
        out["emcee"] = {
            "nwalkers": nwalkers,
            "steps": emcee_steps,
            "wall_s": time.perf_counter() - t0,
        }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--problems", nargs="*", default=list(PROBLEMS))
    ap.add_argument("--calls", type=int, default=200)
    ap.add_argument("--emcee", type=int, default=0, help="emcee steps (0: skip)")
    args = ap.parse_args(argv)

    results = {
        "backend": "jax" if JAX else "numpy",
        "rxmc": getattr(rxmc, "__version__", "?"),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "problems": {},
    }
    try:
        import jitr

        results["jitr"] = getattr(jitr, "__version__", "?")
    except ImportError:
        pass
    for name in args.problems:
        print(f"[{results['backend']}] {name} ...", flush=True)
        emcee_steps = args.emcee if name in ("linear", "gp", "elastic") else 0
        results["problems"][name] = bench(name, args.calls, emcee_steps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
