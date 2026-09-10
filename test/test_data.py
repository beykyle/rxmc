"""Validation and identity semantics of :class:`rxmc.Dataset`."""

import numpy as np
import pytest

from rxmc import Dataset


def test_arrays_coerced_and_validated():
    d = Dataset([0, 1, 2], [1, 2, 3], [0.1, 0.2, 0.3])
    assert d.n == 3
    assert d.y.dtype == float and d.y_err.dtype == float
    assert d.x.shape == (3,)
    with pytest.raises(ValueError, match="y_err must have shape"):
        Dataset([0, 1, 2], [1, 2, 3], [0.1, 0.2])
    with pytest.raises(ValueError, match="first dimension"):
        Dataset([0, 1], [1, 2, 3], [0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="1-D"):
        Dataset([0, 1], [[1, 2]], [0.1])
    with pytest.raises(ValueError, match="non-negative"):
        Dataset([0, 1], [1, 2], [0.1, -0.2])


def test_x_is_opaque():
    x2d = np.array([[0.0, 5.0], [1.0, 5.0], [2.0, 5.0]])  # (theta, E) pairs
    d = Dataset(x2d, [1, 2, 3], [0.1, 0.1, 0.1])
    assert d.x.shape == (3, 2)
    d = Dataset(np.arange(3), [1, 2, 3], [0.1, 0.1, 0.1])  # an index is fine too
    assert d.x.dtype.kind == "i"


def test_error_specs():
    d = Dataset([0, 1, 2], [1, 2, 3], [0.1, 0.1, 0.1])
    assert d.norm_err is None and d.offset_err is None
    d = Dataset([0, 1, 2], [1, 2, 3], [0.1, 0.1, 0.1], norm_err=np.array(0.05))
    assert d.norm_err == 0.05 and isinstance(d.norm_err, float)
    d = Dataset([0, 1, 2], [1, 2, 3], [0.1, 0.1, 0.1], offset_err=[0.01, 0.02, 0.03])
    np.testing.assert_allclose(d.offset_err, [0.01, 0.02, 0.03])
    with pytest.raises(ValueError, match="norm_err"):
        Dataset([0, 1, 2], [1, 2, 3], [0.1, 0.1, 0.1], norm_err=[0.05, 0.05])


def test_meta_is_copied():
    meta = {"Elab": 10.0}
    d = Dataset([0], [1], [0.1], meta=meta)
    meta["Elab"] = 99.0
    assert d.meta["Elab"] == 10.0


def test_identity_and_repr():
    a = Dataset([0], [1], [0.1], label="A")
    b = Dataset([0], [1], [0.1], label="A")
    assert a != b and len({a, b}) == 2
    assert "A" in repr(a) and "n=1" in repr(a)
