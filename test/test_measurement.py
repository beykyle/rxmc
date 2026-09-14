"""from_measurement: EXFOR measurements to datasets in internal units.

No solver is touched: the Rutherford conversion is a closed form of the
kinematics, so a real ``jitr`` reaction is cheap here.
"""

from types import SimpleNamespace

import jitr
import numpy as np
import pytest
from scipy import stats

from helpers import assemble_dense
from rxmc import Comparison, Model, Parameter, from_measurement
from rxmc.reactions import rutherford
from rxmc.terms import statistical
from rxmc.transforms import log

P_CA = jitr.reactions.ElasticReaction(target=(40, 20), projectile=(1, 1))
N_CA = jitr.reactions.ElasticReaction(target=(40, 20), projectile=(1, 0))


def measurement(**overrides):
    """A minimal ``exfor_tools``-like Distribution stub."""
    fields = dict(
        x=np.array([20.0, 40.0]),
        y=np.array([2.0, 1.0]),
        Einc=8.0,
        quantity="dXS/dA",
        y_units="barns/ster",
        statistical_err=np.array([0.2, 0.1]),
        systematic_norm_err=0.03,
        systematic_offset_err=0.02,
        subentry="subentry",
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_construction_in_internal_units():
    m = measurement(subentry="E1234-002")
    d = from_measurement(m, reaction=P_CA)
    np.testing.assert_allclose(d.x, np.deg2rad(m.x))
    np.testing.assert_allclose(d.y, m.y)
    np.testing.assert_allclose(d.y_err, m.statistical_err)
    assert d.label == "E1234-002"
    assert d.norm_err == 0.03 and d.offset_err == pytest.approx(0.02)
    kin = P_CA.kinematics(8.0)
    assert d.meta["reaction"] is P_CA and d.meta["Elab"] == 8.0
    assert d.meta["quantity"] == "dXS/dA" and d.meta["subentry"] == "E1234-002"
    assert d.meta["k"] == pytest.approx(kin.k) and d.meta["eta"] == pytest.approx(
        kin.eta
    )


def test_lab_frame_angles_are_refused():
    with pytest.raises(ValueError, match="LAB frame"):
        from_measurement(measurement(x_units="LAB-degrees"), reaction=P_CA)
    d = from_measurement(measurement(x_units="CM-degrees"), reaction=P_CA)
    np.testing.assert_allclose(d.x, np.deg2rad([20.0, 40.0]))


def test_exfor_tools_labels_and_millibarns():
    for label in ("barns/ster", "b/Sr", "MB/SR", "mb/sr"):
        d = from_measurement(measurement(y_units=label), reaction=P_CA)
        expected = (
            1.0 if "b" in label.lower()[:1] and "m" not in label.lower() else 1e-3
        )
        np.testing.assert_allclose(d.y, expected * measurement().y)


def test_ratio_from_absolute_uses_the_per_angle_rutherford_norm():
    m = measurement(
        y=np.array([1800.0, 300.0]),
        y_units="mb/sr",
        statistical_err=np.array([20.0, 10.0]),
        systematic_offset_err=5.0,  # mb/sr
    )
    d = from_measurement(m, reaction=P_CA, quantity="dXS/dRuth")
    ruth_b = rutherford(P_CA.kinematics(8.0), np.deg2rad(m.x)) / 1000.0
    norm = 1e-3 / ruth_b
    np.testing.assert_allclose(d.y, m.y * norm)
    np.testing.assert_allclose(d.y_err, m.statistical_err * norm)
    np.testing.assert_allclose(d.offset_err, 5.0 * norm)  # a per-angle array now
    assert d.norm_err == 0.03  # fractional: untouched
    assert d.meta["quantity"] == "dXS/dRuth"
    # regression: the reported terms plus the statistical diagonal recover the
    # old auto-folded covariance, in internal (normalised) units
    comp = Comparison(d, Model(lambda x, c: c * np.ones_like(x), [Parameter("c")]))
    ym = np.array([0.9, 0.6])
    S = assemble_dense(
        [statistical(comp.y_err), *comp.reported_terms()], d.x, comp.y, ym
    )
    omega = 5.0 * norm
    old = (
        np.diag((m.statistical_err * norm) ** 2)
        + np.outer(omega, omega)
        + 0.03**2 * np.outer(ym, ym)
    )
    np.testing.assert_allclose(S, old)


def test_absolute_from_ratio():
    m = measurement(y=np.array([0.9, 0.6]), quantity="dXS/dRuth", y_units="no-dim")
    d = from_measurement(m, reaction=P_CA, quantity="dXS/dA")
    ruth_b = rutherford(P_CA.kinematics(8.0), np.deg2rad(m.x)) / 1000.0
    np.testing.assert_allclose(d.y, m.y * ruth_b)
    np.testing.assert_allclose(d.y_err, m.statistical_err * ruth_b)


def test_errors_are_named():
    with pytest.raises(ValueError, match="unknown unit label 'MeV'"):
        from_measurement(measurement(y_units="MeV"))
    with pytest.raises(ValueError, match="needs differential units"):
        from_measurement(measurement(y_units="no-dim"))
    with pytest.raises(ValueError, match="needs dimensionless units"):
        from_measurement(measurement(quantity="Ay", y_units="mb/sr"))
    with pytest.raises(ValueError, match="cannot convert"):
        from_measurement(measurement(), quantity="Ay")
    with pytest.raises(ValueError, match="pass reaction="):
        from_measurement(measurement(), quantity="dXS/dRuth")
    with pytest.raises(ValueError, match="charged projectile"):
        from_measurement(measurement(), reaction=N_CA, quantity="dXS/dRuth")
    with pytest.raises(ValueError, match="unknown quantity"):
        from_measurement(measurement(quantity="sigma"))


def test_no_reaction_and_no_errors():
    m = measurement(
        statistical_err=None,
        systematic_norm_err=None,
        systematic_offset_err=None,
        subentry=None,
    )
    d = from_measurement(m)
    assert "reaction" not in d.meta and d.label == ""
    np.testing.assert_allclose(d.y_err, 0.0)
    assert d.norm_err is None and d.offset_err is None


def test_log_space_and_ias_channel():
    d = from_measurement(measurement(), reaction=P_CA)
    comp = Comparison(
        d, Model(lambda x, c: c * np.ones_like(x), [Parameter("c")]), space=log
    )
    np.testing.assert_allclose(comp.y, np.log(d.y))
    pn = jitr.reactions.Reaction(
        target=(48, 20), projectile=(1, 1), product=(1, 0), residual=(48, 21)
    )
    m = measurement(
        x=np.array([5.0, 15.0]),
        y=np.array([900.0, 700.0]),
        Einc=18.0,
        y_units="mb/sr",
        statistical_err=np.array([80.0, 70.0]),
        systematic_norm_err=0.02,
        systematic_offset_err=10.0,
    )
    d = from_measurement(m, reaction=pn, ExIAS=4.5)
    np.testing.assert_allclose(d.y, [0.9, 0.7])
    np.testing.assert_allclose(d.y_err, [0.08, 0.07])
    assert d.offset_err == pytest.approx(0.01) and d.norm_err == 0.02
    assert d.meta["ExIAS"] == 4.5 and d.meta["Elab"] == 18.0
    assert (
        stats.norm(0, 1).cdf(0) == 0.5
    )  # keep scipy imported for the recipe-style spelling
