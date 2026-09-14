"""Recipe 3: use the reported systematic errors.

The measurement reports a fractional normalisation error and an absolute
offset error.  I want them in the likelihood as correlated modes.
"""

from types import SimpleNamespace

import jitr
import numpy as np
from scipy import stats

from common import line
from helpers import assemble_dense
from rxmc import Comparison, Constraint, Parameter, Problem, from_measurement
from rxmc import terms as T
from rxmc import transforms as tf

P_CA = jitr.reactions.ElasticReaction(target=(40, 20), projectile=(1, 1))


def measurement(**kw):
    fields = dict(
        x=np.array([20.0, 40.0, 60.0]),
        y=np.array([3.0, 2.0, 1.0]),
        Einc=8.0,
        quantity="dXS/dA",
        y_units="barns/ster",
        statistical_err=np.array([0.1, 0.1, 0.1]),
        systematic_norm_err=0.04,
        systematic_offset_err=0.05,
        subentry="E1234-002",
    )
    fields.update(kw)
    return SimpleNamespace(**fields)


def test_nothing_is_folded_in_silently_and_reported_terms_recover_the_old_covariance():
    d = from_measurement(measurement(), reaction=P_CA)
    comp = Comparison(d, line())
    bare = Problem([Constraint([comp])])
    theta = np.array([-0.05, 4.0])
    np.testing.assert_allclose(bare.constraints[0].matrix(theta), np.diag(d.y_err**2))
    terms = comp.reported_terms()
    assert [t.kind for t in terms] == ["mode", "mode"] and all(
        t.on is comp for t in terms
    )
    with_sys = Problem([Constraint([comp], terms=terms)])
    ym = comp.predict(*theta)
    old = np.diag(d.y_err**2) + 0.05**2 * np.ones((3, 3)) + 0.04**2 * np.outer(ym, ym)
    np.testing.assert_allclose(with_sys.constraints[0].matrix(theta), old)
    np.testing.assert_allclose(
        assemble_dense([T.statistical(comp.y_err), *terms], d.x, comp.y, ym), old
    )


def test_zero_magnitudes_yield_no_terms():
    d = from_measurement(
        measurement(systematic_norm_err=0.0, systematic_offset_err=None), reaction=P_CA
    )
    assert Comparison(d, line()).reported_terms() == []


def test_delta_method_under_log():
    d = from_measurement(measurement(), reaction=P_CA)
    comp = Comparison(d, line(), space=tf.log)
    offset, norm = comp.reported_terms()
    ym = comp.predict(-0.05, 4.0)
    np.testing.assert_allclose(offset.value(d.x, comp.y, ym), 0.05 / d.y)  # at the data
    np.testing.assert_allclose(
        norm.value(d.x, comp.y, ym), 0.04
    )  # at the prediction: constant
    log_eps = Parameter("log_eps", prior=stats.norm(-3, 1))
    p = Problem([Constraint([comp], terms=[offset, norm, T.noise(log_eps)])])
    assert np.isfinite(p.log_posterior(np.array([-0.05, 4.0, -3.0])))
