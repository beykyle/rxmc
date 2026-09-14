"""Recipe 35: energy-dependent parameters and per-comparison model instances.

A potential depth depends on energy through a few coefficients I want to
share across datasets at different energies.
"""

import numpy as np
from scipy import stats

from rxmc import Comparison, Constraint, Dataset, Model, Parameter, Problem


def test_one_model_instance_per_comparison_shares_the_coefficients():
    V0, V1 = Parameter("V0", prior=stats.norm(2, 1)), Parameter(
        "V1", prior=stats.norm(0, 0.1)
    )
    b = Parameter("b", prior=stats.norm(0, 1))

    def depth_model(E):
        return Model(lambda x, v0, v1, b: (v0 + v1 * E) * x + b, [V0, V1, b])

    x = np.linspace(0.5, 2.5, 5)
    datasets = [
        Dataset(x, x, 0.1 * np.ones(5), label=f"E{E}", meta={"Elab": E})
        for E in (10.0, 30.0)
    ]
    comps = [Comparison(d, depth_model(d.meta["Elab"])) for d in datasets]
    p = Problem([Constraint(comps)])
    assert p.names == ["V0", "V1", "b"]
    theta = np.array([2.0, -0.02, 1.0])
    pred = p.predict(theta)[0]
    np.testing.assert_allclose(pred[0], (2.0 - 0.02 * 10.0) * x + 1.0)
    np.testing.assert_allclose(pred[1], (2.0 - 0.02 * 30.0) * x + 1.0)
