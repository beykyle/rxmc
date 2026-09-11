"""Recipe 31: simulation-based calibration of the sampler.

Before trusting a chain, I want to check that the sampler recovers
parameters drawn from the prior when the data are simulated from the model.
"""

import dataclasses

import emcee
import numpy as np
import pytest
from scipy import stats

from common import line, line_data, linear_posterior
from rxmc import Comparison, Constraint, Problem
from rxmc.diagnostics import predictive_draws


def simulate(problem, comp, theta0, rng):
    """A dataset drawn from the model at ``theta0``, back in physical units."""
    y_sim = predictive_draws(problem, theta0[None], n_rep=1, rng=rng)[0]
    d_sim = dataclasses.replace(comp.data, y=comp.space.inverse(y_sim))
    return Problem([Constraint([Comparison(d_sim, comp.model)])])


def ranks_from(posterior, n_sims=300, L=20, seed=0):
    """``posterior(p_sim, rng) -> (L, ndim)`` draws; one rank per column."""
    rng = np.random.default_rng(seed)
    d = line_data()
    comp = Comparison(d, line())
    p = Problem([Constraint([comp])])
    ranks = []
    for theta0 in p.sample_prior(n_sims, rng):
        p_sim = simulate(p, comp, theta0, rng)
        s = posterior(p_sim, rng)
        ranks.append((s < theta0).sum(0))
    return np.array(ranks), L


def uniform_pvalues(ranks, L):
    return np.array(
        [
            stats.chisquare(np.bincount(ranks[:, j], minlength=L + 1)).pvalue
            for j in range(2)
        ]
    )


def exact(p_sim, rng, L=20, inflate=1.0):
    mean, cov, _ = linear_posterior(p_sim)
    return rng.multivariate_normal(mean, inflate * cov, L)


def test_exact_posterior_gives_uniform_ranks_and_a_too_wide_one_does_not():
    ranks, L = ranks_from(exact)
    assert ranks.shape == (300, 2) and ranks.min() >= 0 and ranks.max() <= L
    assert np.all(uniform_pvalues(ranks, L) > 0.01)
    # a posterior twice too wide piles the ranks in the middle (inverted U)
    wide, _ = ranks_from(lambda p, rng: exact(p, rng, inflate=4.0))
    assert np.all(uniform_pvalues(wide, L) < 0.01)
    hist = np.bincount(wide[:, 0], minlength=L + 1)
    assert hist[L // 2 - 2 : L // 2 + 3].sum() > hist[:3].sum() + hist[-3:].sum()


@pytest.mark.slow
def test_emcee_passes_simulation_based_calibration():
    def chain(p_sim, rng, L=20):
        p0 = p_sim.sample_prior(16, rng=rng)
        sampler = emcee.EnsembleSampler(16, p_sim.ndim, p_sim.log_posterior)
        sampler.random_state = np.random.RandomState(
            int(rng.integers(2**31))
        ).get_state()
        sampler.run_mcmc(p0, 600, progress=False)
        flat = sampler.get_chain(discard=300, thin=15, flat=True)
        return flat[rng.choice(len(flat), L, replace=False)]

    ranks, L = ranks_from(chain, n_sims=60, seed=1)
    assert np.all(uniform_pvalues(ranks, L) > 0.005), uniform_pvalues(ranks, L)
