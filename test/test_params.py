"""Identity semantics and validation of :class:`rxmc.Parameter`."""

import numpy as np
import pytest
from scipy import stats

from rxmc import Parameter


def test_identity_is_equality():
    p, q = Parameter("a"), Parameter("a")
    assert p == p
    assert p != q  # same name, distinct objects: two parameters
    assert len({p, q}) == 2
    assert {p: 1, q: 2}[p] == 1


def test_bounds_coerced_and_validated():
    p = Parameter("a", bounds=(1, 3))
    assert p.bounds == (1.0, 3.0)
    assert all(isinstance(b, float) for b in p.bounds)
    with pytest.raises(ValueError, match="lower < upper"):
        Parameter("a", bounds=(1.0, 0.0))
    with pytest.raises(ValueError, match="lower, upper"):
        Parameter("a", bounds=(1.0,))


def test_default_bounds_infinite():
    p = Parameter("a")
    assert p.bounds == (-np.inf, np.inf)


def test_name_required():
    with pytest.raises(ValueError):
        Parameter("")


def test_prior_and_labels():
    prior = stats.norm(0, 1)
    p = Parameter("V", prior=prior, unit="MeV", latex=r"V_0")
    assert p.prior is prior
    assert p.unit == "MeV"
    assert p.label == r"V_0"
    assert Parameter("W").label == "W"


def test_repr_names_the_parameter():
    assert "V" in repr(Parameter("V"))
    assert "prior=set" in repr(Parameter("V", prior=stats.norm()))
