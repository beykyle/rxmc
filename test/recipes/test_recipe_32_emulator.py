"""Recipe 32: emulator as the model, emulator variance as a term.

My model is too expensive to run in the chain.  I have a GP or PCA emulator
trained on a design of runs, and I want its predictive variance in the
likelihood.
"""

import numpy as np
import pytest
from scipy import stats

from common import TRUE, line_data
from rxmc import Comparison, Constraint, Model, Parameter, Problem, Term
from rxmc import terms as T


class Emulator:
    """A stand-in: the line itself plus a parameter-dependent variance."""

    def __init__(self, x):
        self.x = x

    def mean(self, theta):
        return theta[0] * self.x + theta[1]

    def var(self, theta):
        return 0.01 * (1 + theta[0] ** 2) * np.ones_like(self.x)


def test_the_term_declares_the_models_parameters_and_sees_the_same_values():
    d = line_data()
    emu = Emulator(d.x)
    m, b = Parameter("m", prior=stats.norm(0, 5)), Parameter(
        "b", prior=stats.norm(0, 5)
    )
    params = [m, b]
    seen = {}

    def emu_std(c, *theta):
        seen["theta"] = theta
        return np.sqrt(emu.var(theta))

    model = Model(lambda x, *theta: emu.mean(theta), params)
    emu_var = Term(emu_std, params, kind="diag")
    log_eps = Parameter("log_eps", prior=stats.norm(-2, 1))
    p = Problem([Constraint([Comparison(d, model)], terms=[emu_var, T.noise(log_eps)])])
    assert p.names == ["m", "b", "log_eps"]  # the term adds no columns of its own
    theta = np.array([*TRUE, np.log(0.1)])
    S = p.constraints[0].matrix(theta)
    assert seen["theta"] == tuple(TRUE)
    np.testing.assert_allclose(np.diag(S), d.y_err**2 + emu.var(TRUE) + 0.01)
    assert np.isfinite(p.log_posterior(theta))
    assert p.log_likelihood(theta) != pytest.approx(
        Problem(
            [Constraint([Comparison(d, model)], terms=[T.noise(log_eps)])]
        ).log_likelihood(theta)
    )
