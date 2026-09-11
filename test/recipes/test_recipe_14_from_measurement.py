"""Recipe 14: from an EXFOR measurement to a dataset.

I have an exfor_tools distribution in mb/sr, or as a ratio to Rutherford,
with its reported errors.  I want a dataset in the library's units with
nothing lost.
"""

from types import SimpleNamespace

import jitr
import numpy as np
import pytest

from rxmc import from_measurement
from rxmc.reactions import rutherford

P_CA = jitr.reactions.ElasticReaction(target=(40, 20), projectile=(1, 1))
N_CA = jitr.reactions.ElasticReaction(target=(40, 20), projectile=(1, 0))


def measurement(**kw):
    fields = dict(
        x=np.array([30.0, 60.0]),
        y=np.array([500.0, 50.0]),
        Einc=12.0,
        quantity="dXS/dA",
        y_units="mb/sr",
        statistical_err=np.array([10.0, 2.0]),
        systematic_norm_err=0.05,
        systematic_offset_err=1.0,
        subentry="E0001-003",
    )
    fields.update(kw)
    return SimpleNamespace(**fields)


def test_units_errors_and_meta():
    d = from_measurement(measurement(), reaction=P_CA)
    np.testing.assert_allclose(d.x, np.deg2rad([30.0, 60.0]))
    np.testing.assert_allclose(d.y, [0.5, 0.05])  # b/sr
    np.testing.assert_allclose(d.y_err, [0.01, 0.002])
    assert d.offset_err == pytest.approx(0.001)  # dimensionful: converted
    assert d.norm_err == 0.05  # fractional: untouched
    assert d.label == "E0001-003"
    for key in ("reaction", "Elab", "quantity", "k", "eta"):
        assert key in d.meta


def test_rutherford_conversion_is_a_per_angle_factor():
    m = measurement()
    d = from_measurement(m, reaction=P_CA, quantity="dXS/dRuth")
    ruth_b = rutherford(P_CA.kinematics(12.0), np.deg2rad(m.x)) / 1000.0
    np.testing.assert_allclose(d.y, (m.y / 1000.0) / ruth_b)
    np.testing.assert_allclose(d.offset_err, (1.0 / 1000.0) / ruth_b)  # now an array
    back = from_measurement(
        measurement(
            y=d.y,
            quantity="dXS/dRuth",
            y_units="no-dim",
            statistical_err=d.y_err,
            systematic_offset_err=None,
        ),
        reaction=P_CA,
        quantity="dXS/dA",
    )
    np.testing.assert_allclose(back.y, m.y / 1000.0)


def test_ias_channel_and_failures():
    pn = jitr.reactions.Reaction(
        target=(48, 20), projectile=(1, 1), product=(1, 0), residual=(48, 21)
    )
    d = from_measurement(measurement(), reaction=pn, ExIAS=6.7)
    assert d.meta["ExIAS"] == 6.7
    with pytest.raises(ValueError, match="unknown unit label"):
        from_measurement(measurement(y_units="fm^2"))
    with pytest.raises(ValueError, match="cannot convert"):
        from_measurement(measurement(), reaction=P_CA, quantity="Ay")
    with pytest.raises(ValueError, match="charged projectile"):
        from_measurement(measurement(), reaction=N_CA, quantity="dXS/dRuth")
