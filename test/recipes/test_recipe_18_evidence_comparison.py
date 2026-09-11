"""Recipe 18: compare error models by evidence.

The alpha + Ca study: several error models for data without reported
errors.  I want the evidence for each, comparable across comparison spaces.

The error-model labels below are those of recipe 18's table in
docs/recipes.md; all are covariances of the residual in log space unless
stated:

    L0   constant noise: sigma = err on every point
    E0   fractional noise in linear space: sigma_i = err * ym_i
    L2y  L0 plus a free correlated normalisation mode, sys * ym
    Lgp  L0 plus a Matern(5/2) Gaussian process in u = theta / pi with a
         constant amplitude
    L0t  L0 under a Student-t likelihood
"""

import dynesty
import numpy as np
import pytest
from scipy import stats
from sklearn.gaussian_process.kernels import Matern

from common import TRUE, line, line_data
from rxmc import Comparison, Constraint, Model, Parameter, Problem
from rxmc import terms as T
from rxmc import transforms as tf
from rxmc.diagnostics import compare_logz, logz_summary
from rxmc.likelihood import StudentT


def error_models(d, model):
    comp_log = Comparison(d, model, space=tf.log)
    comp_lin = Comparison(d, model)
    log_eps = Parameter("log_eps", prior=stats.norm(-2, 1))
    log_sys = Parameter("log_sys", prior=stats.norm(-3, 1))
    log_A = Parameter("log_A", prior=stats.norm(-2, 1))
    gp = T.kernel(
        Matern(0.3, nu=2.5),
        on=comp_log,
        coords=lambda x: x / np.pi,
        amplitude=T.constant_amplitude,
        amplitude_params=(log_A,),
    )
    nu = Parameter("nu", prior=stats.uniform(1, 30))
    return {
        "L0": Constraint([comp_log], terms=[T.noise(log_eps)], statistical=False),
        "E0": Constraint(
            [comp_lin], terms=[T.noise_fraction(log_eps)], statistical=False
        ),
        "L2y": Constraint(
            [comp_log],
            terms=[T.noise(log_eps), T.normalization(log_sys)],
            statistical=False,
        ),
        "Lgp": Constraint([comp_log], terms=[T.noise(log_eps), gp], statistical=False),
        "L0t": Constraint(
            [comp_log],
            terms=[T.noise(log_eps)],
            statistical=False,
            likelihood=StudentT(nu),
        ),
    }


def test_each_problem_compiles_independently_with_shared_parameter_objects():
    d = line_data()
    models = error_models(d, line())
    problems = {name: Problem([c]) for name, c in models.items()}
    assert problems["L0"].names == ["m", "b", "log_eps"]
    assert problems["E0"].names == ["m", "b", "log_eps"]
    assert problems["L2y"].names == ["m", "b", "log_eps", "log_sys"]
    assert problems["Lgp"].names == [
        "m",
        "b",
        "log_eps",
        "discrepancy_length_scale",
        "log_A",
    ]
    assert problems["L0t"].names == ["m", "b", "log_eps", "nu"]
    # the same log_eps object is one column in each, at the same slot
    assert all(p.names.index("log_eps") == 2 for p in problems.values())
    theta = np.array([*TRUE, np.log(0.1)])
    assert np.isfinite(problems["L0"].log_posterior(theta))
    assert np.isfinite(problems["E0"].log_posterior(theta))


def test_log_jacobian_makes_spaces_comparable():
    d = line_data()
    models = error_models(d, line())
    p_log, p_lin = Problem([models["L0"]]), Problem([models["E0"]])
    assert p_log.log_jacobian() == pytest.approx(-np.sum(np.log(d.y)))
    assert p_lin.log_jacobian() == 0.0
    # a cut applies to the Jacobian too: only active points count
    cut = Problem([models["L0"].masked_where(lambda x: x < 1.5)])
    assert cut.log_jacobian() == pytest.approx(-np.sum(np.log(d.y[d.x < 1.5])))
    # the recipe's bookkeeping: add the Jacobian before summarising
    fake_logz, fake_err = -12.0, 0.2
    a = logz_summary(fake_logz + p_log.log_jacobian(), fake_err)
    b = logz_summary(fake_logz + p_lin.log_jacobian(), fake_err)
    assert compare_logz(a, b)["dlogZ"] == pytest.approx(p_log.log_jacobian())


def run_dynesty(p, seed, nlive=100):
    ns = dynesty.NestedSampler(
        p.log_likelihood,
        p.prior_transform,
        p.ndim,
        nlive=nlive,
        rstate=np.random.default_rng(seed),
    )
    ns.run_nested(dlogz=0.1, print_progress=False)
    return ns.results


@pytest.mark.slow
def test_a_normalisation_defect_is_preferred_by_the_model_that_has_one():
    # Data scaled by 30 %.  A line with a free intercept would absorb a scaling
    # into its parameters, so the model here has a known intercept: only the
    # normalisation mode of L2y can explain the defect.
    from dataclasses import replace

    d = replace(line_data(err=0.05), y=1.3 * line_data(err=0.05).y)
    m = Parameter("m", prior=stats.norm(0, 5))
    slope_only = Model(lambda x, m: m * x + 1.0, [m])
    models = error_models(d, slope_only)
    logz = {}
    for name in ("L0", "L2y"):
        p = Problem([models[name]])
        res = [run_dynesty(p, seed) for seed in (0, 1)]
        logz[name] = logz_summary(
            [r.logz[-1] + p.log_jacobian() for r in res], [r.logzerr[-1] for r in res]
        )
    verdict = compare_logz(logz["L2y"], logz["L0"])
    assert verdict["verdict"] == "a" and verdict["dlogZ"] > 1.5, (logz, verdict)
