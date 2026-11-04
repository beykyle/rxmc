"""
Point estimates of a compiled problem: the posterior mode and the maximum of
the likelihood.

Both search from several prior draws and keep the best end, because reaction
posteriors have local modes; the log density at every end is returned too,
so the modes are visible.  The two share one signature and one return
convention, and both use L-BFGS-B with the exact gradient:

* :func:`find_map` maximises ``log_posterior``: where a gradient sampler
  (HMC/NUTS, recipe 41) or a Laplace approximation (recipe 33) starts;
* :func:`find_mle` maximises ``log_likelihood``.

Both stay inside :attr:`~rxmc.problem.Problem.support`, the bounds narrowed by
the priors' supports, where the densities are defined.  A search that does
not converge is not silent: the result counts the searches that succeeded
(``n_success``), and a warning names scipy's message when even the best one
did not.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
from scipy.optimize import OptimizeResult, minimize

__all__ = ["find_map", "find_mle"]


def find_map(problem, n_starts: int = 8, rng=None, options=None) -> OptimizeResult:
    """The posterior mode, by L-BFGS-B with the exact gradient.

    Returns the best search's ``OptimizeResult`` with ``x`` the mode,
    ``fun`` the log posterior there, ``nfev`` the evaluations of all
    ``n_starts`` searches, ``wall`` their time in seconds, ``ends`` the log
    posterior at every search's end, best first, and ``n_success`` the number
    of searches scipy reports as converged on a finite density.  ``options`` is passed to scipy
    (e.g. ``{"maxiter": 500}``).  Warns when the best search did not converge.
    """
    return _search(
        problem, problem.value_and_grad_log_posterior, n_starts, rng, options
    )


def find_mle(problem, n_starts: int = 8, rng=None, options=None) -> OptimizeResult:
    """The maximum of the likelihood, by L-BFGS-B with the exact gradient.

    The prior is ignored except for its support.  Returns as
    :func:`find_map`, with ``fun`` and ``ends`` the log likelihood.  Where the
    likelihood has no single maximum the result is one point of many: a
    hierarchy's likelihood depends on ``mu + tau * eta_j`` only, so its
    maximum is a ridge, and the search stops wherever it meets it.
    """
    return _search(
        problem, problem.value_and_grad_log_likelihood, n_starts, rng, options
    )


def _search(problem, value_and_grad, n_starts, rng, options):
    support = problem.support
    # an infinite end is no bound to scipy; a finite one is kept, nudged
    # inside so the search never evaluates exactly on it
    width = np.where(np.isfinite(support).all(axis=1), np.diff(support).ravel(), 1.0)
    lo = np.where(np.isfinite(support[:, 0]), support[:, 0] + 1e-9 * width, None)
    hi = np.where(np.isfinite(support[:, 1]), support[:, 1] - 1e-9 * width, None)
    bounds = list(zip(lo, hi))

    def negative(t):
        v, g = value_and_grad(t)
        return -v, -g

    t0 = time.perf_counter()
    fits = [
        minimize(
            negative,
            start,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options=options,
        )
        for start in problem.sample_prior(n_starts, rng=rng)
    ]
    # a search that stops on a -inf plateau (zero gradient) reports success,
    # so success counts only where it ended on a finite density
    value = np.array([-f.fun if np.isfinite(f.fun) else -np.inf for f in fits])
    ends = np.sort(value)[::-1]
    best = fits[int(np.argmax(value))]
    if not np.isfinite(value.max()):
        warnings.warn(
            f"none of {len(fits)} searches ended where the density is finite",
            RuntimeWarning,
            stacklevel=3,
        )
    elif not best.success:
        warnings.warn(
            f"the best of {len(fits)} searches did not converge: {best.message}",
            RuntimeWarning,
            stacklevel=3,
        )
    best.update(
        fun=float(value.max()),
        nfev=int(sum(f.nfev for f in fits)),
        wall=time.perf_counter() - t0,
        ends=ends,
        n_success=int(sum(bool(f.success) and np.isfinite(f.fun) for f in fits)),
    )
    return best
