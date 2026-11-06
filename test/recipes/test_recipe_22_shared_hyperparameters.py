"""Recipe 22: share hyperparameters across datasets, with values that depend
on the dataset.

I have elastic data at several energies.  I want one GP discrepancy per
dataset with a common length scale and an amplitude that runs with energy,
so that two shared parameters describe every dataset.
"""

import jax.numpy as jnp
import jax.scipy.stats as jst
import numpy as np
import pytest
import tinygp
from scipy import stats

# sklearn's Matern serves only as the independent dense reference
from sklearn.gaussian_process.kernels import Matern

from common import TRUE, line
from rxmc import Comparison, Constraint, Dataset, Parameter, Problem
from rxmc import terms as T


def matern52(log_ell):
    return tinygp.kernels.Matern52(scale=jnp.exp(log_ell))


def datasets():
    out = []
    for i, E in enumerate((10.0, 30.0, 50.0)):
        x = np.linspace(0.5, 2.5, 5)
        out.append(
            Dataset(
                x,
                TRUE[0] * x + TRUE[1],
                0.1 * np.ones(5),
                label=f"E{E:.0f}",
                meta={"Elab": E},
            )
        )
    return out


def test_two_shared_parameters_describe_every_dataset():
    log_A0, p_ = (
        Parameter("log_A0", prior=stats.norm(0, 2)),
        Parameter("p", prior=stats.norm(0, 1)),
    )
    ell = Parameter("gp_length", prior=stats.norm(0, 1))
    amp = lambda c, lA, p: jnp.exp(lA) * (c.meta("Elab") / 50.0) ** p  # noqa: E731
    model = line()
    comps = [Comparison(d, model) for d in datasets()]
    terms = [
        T.kernel(
            matern52,
            [ell],
            on=c,
            amplitude=amp,
            amplitude_params=(log_A0, p_),
            jitter=0.0,
        )
        for c in comps
    ]
    prob = Problem([Constraint(comps, terms=terms)])
    assert prob.names == ["m", "b", "gp_length", "log_A0", "p"]
    theta = np.array([*TRUE, np.log(0.4), np.log(0.3), 1.0])
    S = prob.constraints[0].matrix(theta)
    # each block's amplitude is (E / 50)^p times 0.3
    for i, E in enumerate((10.0, 30.0, 50.0)):
        block = S[5 * i : 5 * i + 5, 5 * i : 5 * i + 5] - 0.01 * np.eye(5)
        a = 0.3 * (E / 50.0)
        u = np.linspace(0.5, 2.5, 5)
        np.testing.assert_allclose(block, a**2 * Matern(0.4, nu=2.5)(u[:, None]))
    assert not prob.constraints[0].covariance.dense


class Standard:
    """A joint block giving every covered parameter an independent N(0, 1)."""

    def logpdf(self, v):
        return jst.norm.logpdf(v).sum()


def test_distinct_prefixes_give_each_kernel_its_own_length_scale():
    model = line()
    comps = [Comparison(d, model) for d in datasets()[:2]]
    terms = [
        T.kernel(matern52, T.kernel_params(["length_scale"], prefix=f"gp{i}"), on=c)
        for i, c in enumerate(comps)
    ]
    derived = [p for t in terms for p in t.params]
    prob = Problem([Constraint(comps, terms=terms)], priors=[(derived, Standard())])
    assert [n for n in prob.names if "length" in n] == [
        "gp0_length_scale",
        "gp1_length_scale",
    ]
    same = [
        T.kernel(matern52, T.kernel_params(["length_scale"]), on=c) for c in comps
    ]  # both declare "discrepancy_length_scale"
    with pytest.raises(ValueError, match="duplicate"):
        Problem(
            [Constraint(comps, terms=same)],
            priors=[([p for t in same for p in t.params], Standard())],
        )
