"""Recipe 16: drive the calibration with an external sampler.

I want to use emcee, dynesty, or the black-box-bayes CLI, and read the chain
back without positional arithmetic.
"""

import dill
import dynesty
import emcee
import numpy as np
import pytest

from common import TRUE, line_data, line_problem
from rxmc import Comparison, Constraint, Model, Parameter, Problem


def test_emcee_uses_sample_prior_and_log_posterior():
    p, _ = line_problem()
    p0 = 0.05 * p.sample_prior(16, rng=0) + np.array(TRUE)
    sampler = emcee.EnsembleSampler(16, p.ndim, p.log_posterior)
    sampler.random_state = np.random.RandomState(1).get_state()
    sampler.run_mcmc(p0, 100, progress=False)
    samples = sampler.get_chain(discard=40, flat=True)
    assert samples.shape == (16 * 60, p.ndim)
    # the chain is in problem.names order, so columns() is the only bookkeeping
    m = p.columns(p.params[0])
    assert abs(samples[:, m].mean() - TRUE[0]) < 3 * samples[:, m].std() + 0.05


def test_dynesty_uses_log_likelihood_and_prior_transform():
    p, _ = line_problem()
    ns = dynesty.NestedSampler(
        p.log_likelihood,
        p.prior_transform,
        p.ndim,
        nlive=40,
        rstate=np.random.default_rng(0),
    )
    ns.run_nested(dlogz=1.0, print_progress=False)
    res = ns.results
    assert np.isfinite(res.logz[-1]) and res.samples_equal().shape[1] == p.ndim


def test_black_box_bayes_contract_and_dill():
    p, _ = line_problem()
    blob = dill.dumps(p)
    q = dill.loads(blob)
    theta = np.array(TRUE)
    assert q.NDIM == p.ndim and q.parameter_names == p.names
    assert q.starting_location(3).shape == (3, p.ndim)
    assert q.log_posterior(theta) == pytest.approx(p.log_posterior(theta))
    assert q.log_likelihood(theta) == pytest.approx(p.log_likelihood(theta))
    np.testing.assert_allclose(
        q.prior_transform([0.3, 0.7]), p.prior_transform([0.3, 0.7])
    )
    np.testing.assert_allclose(
        q.log_posterior_batch([theta, theta]), [p.log_posterior(theta)] * 2
    )


def test_log_posterior_evaluates_the_prior_first():
    calls = []
    m, b = Parameter("m", bounds=(0.0, 4.0)), Parameter("b", bounds=(0.0, 4.0))

    def fn(x, m, b):
        calls.append(1)
        return m * x + b

    p = Problem([Constraint([Comparison(line_data(), Model(fn, [m, b]))])])
    assert p.log_posterior([5.0, 1.0]) == -np.inf and calls == []
