"""The unit contract: a fixed vocabulary of labels, consistent constants."""

import numpy as np
import pytest

from rxmc.units import (
    DEFAULT_LMAX,
    MB_PER_B,
    RUTHERFORD_UNIT,
    XS_UNIT,
    check_angle_grid,
    parse_unit,
)


def test_unit_constants_agree():
    assert MB_PER_B == 1000.0
    assert parse_unit(RUTHERFORD_UNIT)[0] == pytest.approx(1.0 / MB_PER_B)
    assert parse_unit(XS_UNIT) == (1.0, "differential")
    assert DEFAULT_LMAX == 20


@pytest.mark.parametrize(
    "label, factor, kind",
    [
        ("barns/ster", 1.0, "differential"),  # what exfor_tools emits
        ("b/Sr", 1.0, "differential"),
        ("MB/SR", 1e-3, "differential"),  # raw EXFOR spelling
        ("barn / steradian", 1.0, "differential"),
        ("millibarn / steradian", 1e-3, "differential"),
        ("MICRO-B/SR", 1e-6, "differential"),
        ("barns", 1.0, "integral"),
        ("mb", 1e-3, "integral"),
        ("no-dim", 1.0, "dimensionless"),
        ("unitless", 1.0, "dimensionless"),
        ("NO-DIM", 1.0, "dimensionless"),
    ],
)
def test_parse_unit_vocabulary(label, factor, kind):
    f, k = parse_unit(label)
    assert f == pytest.approx(factor)
    assert k == kind


def test_parse_unit_rejects_unknown_label():
    with pytest.raises(ValueError, match="unknown unit label 'fm\\^2'"):
        parse_unit("fm^2")
    with pytest.raises(ValueError, match="accepted labels"):
        parse_unit("MeV")


def test_check_angle_grid():
    check_angle_grid(np.linspace(0.0, np.pi, 5), "angles")
    with pytest.raises(ValueError, match="1D"):
        check_angle_grid(np.zeros((2, 2)), "angles")
    with pytest.raises(ValueError, match="radians"):
        check_angle_grid(np.array([0.0, 4.0]), "angles")
    with pytest.raises(ValueError, match="radians"):
        check_angle_grid(np.array([-0.1, 1.0]), "angles")
    with pytest.raises(ValueError, match="finite"):
        check_angle_grid(np.array([0.3, np.nan, 1.0]), "angles")
