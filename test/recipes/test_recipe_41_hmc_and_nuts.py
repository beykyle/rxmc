"""Recipe 41: sample with gradients (HMC/NUTS).

My problem has many parameters, or a sampler that walks without gradients
mixes slowly on it.  I want the No-U-Turn sampler, driven by the exact
gradient of the posterior.
"""

import blackjax
import jax
import numpy as np
from scipy import stats

import rxmc as rx
from common import line_data, line_problem, linear_posterior
from rxmc import Comparison, Constraint, Model, Parameter, Problem


def nuts(problem, seed, n_warmup=300, n_draws=1000):
    """The recipe's spelling: one adapted NUTS chain, mapped back to theta."""
    logdensity = problem.log_posterior_unconstrained
    z0 = problem.to_unconstrained(rx.optimize.find_map(problem, rng=seed).x)
    k_warm, k_draw = jax.random.split(jax.random.key(seed))
    warmup = blackjax.window_adaptation(blackjax.nuts, logdensity)
    (state, tuned), _ = warmup.run(k_warm, z0, num_steps=n_warmup)
    step = jax.jit(blackjax.nuts(logdensity, **tuned).step)

    def one_step(state, key):
        state, info = step(key, state)
        return state, (state.position, info.is_divergent)

    keys = jax.random.split(k_draw, n_draws)
    _, (zs, divergent) = jax.lax.scan(one_step, state, keys)
    return np.asarray(jax.vmap(problem.to_theta)(zs)), int(divergent.sum())


def test_nuts_recovers_the_linear_gaussian_posterior():
    problem, _ = line_problem()
    mean, cov, _ = linear_posterior(problem)
    chains = [nuts(problem, seed) for seed in (0, 1)]
    samples = np.concatenate([c[0] for c in chains])
    assert sum(c[1] for c in chains) == 0
    ess = blackjax.diagnostics.effective_sample_size(np.stack([c[0] for c in chains]))
    mcse = np.sqrt(np.diag(cov) / ess)
    assert np.all(np.abs(samples.mean(0) - mean) < 4 * mcse)
    np.testing.assert_allclose(np.cov(samples.T), cov, rtol=0.2)


def test_bounded_parameters_are_sampled_inside_their_bounds():
    # the intercept's posterior (1.0 +/- 0.07) presses on its upper bound
    m = Parameter("m", prior=stats.norm(0, 5))
    b = Parameter("b", prior=stats.norm(0, 5), bounds=(0.0, 1.05))
    model = Model(lambda x, m, b: m * x + b, [m, b])
    problem = Problem([Constraint([Comparison(line_data(), model)])])
    samples, divergent = nuts(problem, seed=2)
    col = samples[:, problem.columns(b)]
    assert divergent == 0
    assert np.all((col > 0.0) & (col < 1.05))
    assert col.mean() < 1.05 and np.mean(col > 1.0) > 0.1  # mass right up to the bound
