"""
Point estimates of a compiled problem: the posterior mode and the maximum of
the likelihood.

Both search from several prior draws and keep the best end, because reaction
posteriors have local modes; the log density at every end is returned too,
so the modes are visible.  The two share one signature and one return
convention:

* :func:`find_map` maximises ``log_posterior`` with L-BFGS-B and its exact
  gradient: where a gradient sampler (HMC/NUTS, recipe 41) or a Laplace
  approximation (recipe 33) starts;
* :func:`find_mle` maximises ``log_likelihood`` with Nelder-Mead, using no
  gradient: where an ensemble sampler such as emcee is usually started.

Both stay inside :attr:`~rxmc.problem.Problem.support`, the bounds narrowed by
the priors' supports, where the densities are defined.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import OptimizeResult, minimize

__all__ = ["find_map", "find_mle"]


def find_map(problem, n_starts: int = 8, rng=None) -> OptimizeResult:
    """The posterior mode, by L-BFGS-B with the exact gradient.

    Returns the best search's ``OptimizeResult`` with ``x`` the mode,
    ``fun`` the log posterior there, ``nfev`` the evaluations of all
    ``n_starts`` searches, ``wall`` their time in seconds and ``ends`` the
    log posterior at every search's end, best first.
    """
    return _search(
        problem,
        problem.log_posterior,
        dict(method="L-BFGS-B", jac=lambda t: -problem.grad_log_posterior(t)),
        n_starts,
        rng,
    )


def find_mle(problem, n_starts: int = 8, rng=None) -> OptimizeResult:
    """The maximum of the likelihood, by Nelder-Mead (no gradient).

    The prior is ignored except for its support.  Returns as
    :func:`find_map`, with ``fun`` and ``ends`` the log likelihood.
    """
    return _search(
        problem, problem.log_likelihood, dict(method="Nelder-Mead"), n_starts, rng
    )


def _search(problem, logdensity, options, n_starts, rng):
    support = problem.support
    # an infinite end is no bound to scipy; a finite one is kept, nudged
    # inside so the search never evaluates exactly on it
    width = np.where(np.isfinite(support).all(axis=1), np.diff(support).ravel(), 1.0)
    lo = np.where(np.isfinite(support[:, 0]), support[:, 0] + 1e-9 * width, None)
    hi = np.where(np.isfinite(support[:, 1]), support[:, 1] - 1e-9 * width, None)
    bounds = list(zip(lo, hi))
    t0 = time.perf_counter()
    fits = [
        minimize(lambda t: -logdensity(t), start, bounds=bounds, **options)
        for start in problem.sample_prior(n_starts, rng=rng)
    ]
    ends = np.sort([-f.fun for f in fits])[::-1]
    best = min(fits, key=lambda f: f.fun)
    best.update(
        fun=-best.fun,
        nfev=int(sum(f.nfev for f in fits)),
        wall=time.perf_counter() - t0,
        ends=ends,
    )
    return best
