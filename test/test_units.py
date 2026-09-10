"""The unit contract: one registry, consistent constants."""

import numpy as np
import pytest

from rxmc.units import (
    DEFAULT_LMAX,
    MB_PER_B,
    RUTHERFORD_UNIT,
    XS_UNIT,
    check_angle_grid,
    ureg,
)


def test_unit_constants_agree():
    assert MB_PER_B == 1000.0
    assert (1 * RUTHERFORD_UNIT).to(XS_UNIT).magnitude == pytest.approx(1e-3)
    assert (1 * XS_UNIT).to(RUTHERFORD_UNIT).magnitude == pytest.approx(MB_PER_B)
    assert DEFAULT_LMAX == 20


def test_registry_is_shared():
    # quantities built from the constants combine without a registry error
    q = (2 * ureg.millibarn / ureg.steradian) + 1 * XS_UNIT
    assert q.to(XS_UNIT).magnitude == pytest.approx(1.002)


def test_check_angle_grid():
    check_angle_grid(np.linspace(0.0, np.pi, 5), "angles")
    with pytest.raises(ValueError, match="1D"):
        check_angle_grid(np.zeros((2, 2)), "angles")
    with pytest.raises(ValueError, match="radians"):
        check_angle_grid(np.array([0.0, 4.0]), "angles")
    with pytest.raises(ValueError, match="radians"):
        check_angle_grid(np.array([-0.1, 1.0]), "angles")
