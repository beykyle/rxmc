"""Benchmark problems, buildable against both the numpy and the JAX rxmc.

Each builder returns ``(problem, thetas)``: a compiled :class:`rxmc.Problem`
and a fixed ``(k, ndim)`` array of parameter points (seeded prior draws) at
which values are recorded and timings taken.  The problems span the cost
regimes of the performance comparison in the JAX rewrite plan:

* ``linear``      -- recipe 01: a line with reported errors (overhead-bound)
* ``modes``       -- recipes 04/37: several comparisons, free normalisations
                     and an offset per comparison, one shared mode (Woodbury)
* ``gp``          -- recipe 07: a dense GP discrepancy over N points
* ``elastic``     -- recipes 15/26: ElasticXS at several energies with a
                     proportional model error (solver-bound)
* ``ias``         -- an IsobaricAnalogPN prediction at one energy
* ``omp``         -- the log-space p + 40Ca optical model of the notebooks
                     ``local_optical_model_calibration`` and
                     ``gradient_based_sampling``: EXFOR O1199007 at 35 MeV,
                     seven potential parameters and two inferred relative
                     errors (9 parameters; JAX rxmc only)
* ``hierarchy``   -- recipe 38 at scale: 30 lines sharing a slope, their
                     intercepts non-centred about a common mean (33
                     parameters; the sampler benchmark's high-dimensional case)

The only API difference the builders branch on is the GP kernel (sklearn in
the numpy rxmc, tinygp in the JAX one).
"""

from __future__ import annotations

import numpy as np
from scipy import stats

import rxmc
from rxmc import Comparison, Constraint, Dataset, Model, Parameter, Problem
from rxmc import terms as T

JAX = hasattr(Problem, "grad_log_posterior")
N_THETA = 8


def _thetas(problem, seed=0):
    return problem.sample_prior(N_THETA, rng=seed)


def _line(prefix=""):
    m = Parameter(f"{prefix}m", prior=stats.norm(0, 5))
    b = Parameter(f"{prefix}b", prior=stats.norm(0, 5))
    return Model(lambda x, m, b: m * x + b, [m, b])


def _line_data(rng, n, label, x=None):
    x = np.linspace(0.5, 2.5, n) if x is None else x
    y = 2.0 * x + 1.0 + rng.normal(0, 0.1, n)
    return Dataset(x, y, 0.1 * np.ones(n), label=label)


def linear():
    d = _line_data(np.random.default_rng(0), 8, "d")
    p = Problem([Constraint([Comparison(d, _line())])])
    return p, _thetas(p)


def modes(n_comparisons=6, n=20):
    rng = np.random.default_rng(1)
    model = _line()
    comps = [
        Comparison(_line_data(rng, n, f"d{i}"), model) for i in range(n_comparisons)
    ]
    terms = []
    for i, c in enumerate(comps):
        eta = Parameter(f"log_eta{i}", bounds=(-6.0, 0.0))
        terms.append(T.normalization(eta, on=c))
        terms.append(T.offset(magnitude=0.02, on=c))
    shared = Parameter("log_s", bounds=(-6.0, 0.0))
    terms.append(T.systematic(shared, T.x_basis(2.5)))
    p = Problem([Constraint(comps, terms=terms)])
    return p, _thetas(p)


def gp(n=100):
    rng = np.random.default_rng(2)
    d = _line_data(rng, n, "d")
    if JAX:
        import jax.numpy as jnp
        import tinygp

        # the same parameterisation the sklearn kernel derives: log variance
        # and log length scale, sklearn's bounds in log space
        params = [
            Parameter("discrepancy_k1__constant_value", bounds=(-12.0, 2.0)),
            Parameter("discrepancy_k2__length_scale", bounds=(-3.0, 1.0)),
        ]
        term = T.kernel(
            lambda log_var, log_ell: (
                jnp.exp(log_var) * tinygp.kernels.ExpSquared(scale=jnp.exp(log_ell))
            ),
            params,
        )
    else:
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel

        term = T.kernel(
            ConstantKernel(0.1, (np.exp(-12.0), np.exp(2.0)))
            * RBF(1.0, (np.exp(-3.0), np.exp(1.0))),
        )
    p = Problem([Constraint([Comparison(d, _line())], terms=[term])])
    return p, _thetas(p)


def _ca40():
    import jitr
    from jitr.optical_potentials.potential_forms import thomas_safe, woods_saxon_safe

    A, Z = 40, 20
    R = 1.2 * A ** (1 / 3)
    mso = 1.0 / jitr.utils.constants.WAVENUMBER_PION

    def central(r, Vv, Wv, Rv, av):
        return -(Vv + 1j * Wv) * woods_saxon_safe(r, Rv, av)

    def spin_orbit(r, Vso, Rso, aso):
        return Vso * mso**2 * thomas_safe(r, Rso, aso)

    return jitr, A, Z, R, central, spin_orbit


def _omp_params(R):
    return [
        Parameter("Vv", bounds=(40.0, 56.0)),
        Parameter("Wv", bounds=(1.0, 8.0)),
        Parameter("Rv", bounds=(0.9 * R, 1.1 * R)),
        Parameter("av", bounds=(0.5, 0.8)),
    ]


def elastic(energies=(10.0, 14.1, 20.0, 26.0), n=30):
    from rxmc.reactions import ElasticXS

    jitr, A, Z, R, central, spin_orbit = _ca40()
    rxn = jitr.reactions.ElasticReaction(target=(A, Z), projectile=(1, 0))
    model = ElasticXS(
        "dXS/dA",
        central,
        spin_orbit,
        lambda ws, *x: (tuple(x), (6.0, R, 0.45)),
        _omp_params(R),
        lmax=20,
    )
    angles = np.linspace(0.2, 2.8, n)
    truth = (48.0, 3.5, R, 0.7)
    comps = []
    for E in energies:
        meta = {"reaction": rxn, "Elab": E}
        y = model.bind(angles, meta)(*truth)
        y = np.asarray(y) * (1 + 0.05 * np.random.default_rng(int(E)).normal(size=n))
        comps.append(
            Comparison(Dataset(angles, y, 0.03 * y, label=f"E{E}", meta=meta), model)
        )
    log_c = Parameter("log_c", bounds=(-6.0, 0.0))
    constraints = [
        Constraint([c], terms=[T.proportional_error(log_c, on=c)]) for c in comps
    ]
    p = Problem(constraints)
    return p, _thetas(p)


def ias(n=20):
    from jitr.optical_potentials.potential_forms import coulomb_charged_sphere

    from rxmc.reactions import IsobaricAnalogPN

    jitr, _, _, _, central, spin_orbit = _ca40()
    A, Z = 48, 20
    R = 1.2 * A ** (1 / 3)
    rxn = jitr.reactions.Reaction(
        target=(A, Z), projectile=(1, 1), product=(1, 0), residual=(A, Z + 1)
    )
    model = IsobaricAnalogPN(
        coulomb_charged_sphere,
        central,
        spin_orbit,
        central,
        spin_orbit,
        lambda ws, Vv, Wv, Rv, av: (
            (Z, R),
            (Vv + 4.0, Wv, Rv, av),
            (6.0, R, 0.45),
            (Vv - 4.0, Wv, Rv, av),
            (6.0, R, 0.45),
        ),
        _omp_params(R),
        lmax=20,
    )
    angles = np.linspace(0.2, 2.6, n)
    meta = {"reaction": rxn, "Elab": 25.0, "ExIAS": 6.7}
    y = np.asarray(model.bind(angles, meta)(48.0, 3.5, R, 0.7))
    d = Dataset(angles, y, 0.05 * y, label="ias", meta=meta)
    p = Problem([Constraint([Comparison(d, model)])])
    return p, _thetas(p)


def hierarchy(n_groups=30, n=5):
    """``y_j = m x + mu + tau eta_j``, ``eta_j ~ N(0, 1)``, ``tau`` half-normal."""
    rng = np.random.default_rng(5)
    m = Parameter("m", prior=stats.norm(0, 5))
    mu = Parameter("mu", prior=stats.norm(0, 5))
    tau = Parameter("tau", prior=stats.halfnorm(scale=1.0), bounds=(0.0, np.inf))
    x = np.linspace(0.0, 1.0, n)
    comps = []
    for j in range(n_groups):
        eta = Parameter(f"eta_{j}", prior=stats.norm(0, 1))
        y = 0.8 * x + 1.0 + 0.3 * rng.normal() + rng.normal(0, 0.1, n)
        model = Model(
            lambda x, m, mu, tau, eta: m * x + mu + tau * eta, [m, mu, tau, eta]
        )
        comps.append(Comparison(Dataset(x, y, 0.1 * np.ones(n), label=f"g{j}"), model))
    p = Problem([Constraint(comps)])
    return p, _thetas(p)


def omp():
    import jax.numpy as jnp
    import jitr
    import pandas as pd
    from jitr.optical_potentials.potential_forms import (
        coulomb_charged_sphere,
        thomas_safe,
        woods_saxon_prime_safe,
        woods_saxon_safe,
    )

    from rxmc import transforms as tf

    here = __import__("pathlib").Path(__file__).resolve().parents[1]
    df = pd.read_csv(here / "examples" / "data" / "p40ca_elastic_35mev.csv")
    measurement = {
        "x": df["angle_cm_deg"].to_numpy(),
        "y": df["ratio_to_rutherford"].to_numpy(),
        "statistical_err": df["stat_err"].to_numpy(),
        "Einc": 35.0,
        "quantity": "dXS/dRuth",
        "y_units": "no-dim",
        "x_units": "CM-degrees",
        "systematic_norm_err": 0.0,
        "systematic_offset_err": 0.0,
        "subentry": "O1199007",
    }
    reaction = jitr.reactions.ElasticReaction(target=(40, 20), projectile=(1, 1))
    data = rxmc.from_measurement(measurement, reaction=reaction)
    A13, mso = 40 ** (1 / 3), 1.0 / jitr.utils.constants.WAVENUMBER_PION

    def central(r, Vv, Wv, Rv, av, Wd, Rd, ad):
        surface = -4 * ad * woods_saxon_prime_safe(r, Rd, ad)
        return -(Vv + 1j * Wv) * woods_saxon_safe(r, Rv, av) - 1j * Wd * surface

    def spin_orbit(r, Vso, Rso, aso):
        return Vso * mso**2 * thomas_safe(r, Rso, aso)

    def coulomb(r):
        return coulomb_charged_sphere(r, 20, 1.3 * A13)

    names = ["Vv", "Wv", "Rv", "av", "Wd", "Rd", "ad"]
    start = [45.0, 3.0, 1.18 * A13, 0.68, 6.0, 1.28 * A13, 0.55]
    sd = [8.0, 3.0, 0.25, 0.15, 5.0, 0.25, 0.15]
    lower = [0.0, 0.0, 1.0, 0.2, 0.0, 1.0, 0.2]
    params = [
        Parameter(n, prior=stats.norm(m, s), bounds=(lo, np.inf))
        for n, m, s, lo in zip(names, start, sd, lower)
    ]
    model = rxmc.reactions.ElasticXS(
        "dXS/dRuth",
        central,
        spin_orbit,
        lambda ws, *x: (tuple(x), (5.5, 1.0 * A13, 0.6)),
        params,
        coulomb=coulomb,
        lmax=30,
    )
    log_uniform = stats.uniform(np.log(0.05), np.log(2.0) - np.log(0.05))
    log_s = Parameter("log_s", prior=log_uniform)
    log_b = Parameter("log_b", prior=log_uniform)

    def point_to_point_sd(c, log_s, log_b):
        s2, b2 = jnp.exp(2 * log_s), jnp.exp(2 * log_b)
        return jnp.full(len(c), jnp.sqrt(jnp.log1p(s2 / (1 + b2))))

    def common_sd(c, log_b):
        return jnp.full(len(c), jnp.sqrt(jnp.log1p(jnp.exp(2 * log_b))))

    errors = [
        T.Term(point_to_point_sd, (log_s, log_b), kind="diag"),
        T.Term(common_sd, (log_b,), kind="mode"),
    ]
    p = Problem([Constraint([Comparison(data, model, space=tf.log)], terms=errors)])
    return p, _thetas(p)


PROBLEMS = {
    "linear": linear,
    "modes": modes,
    "gp": gp,
    "gp500": lambda: gp(500),
    "elastic": elastic,
    "ias": ias,
    "hierarchy": hierarchy,
    "omp": omp,
}

__all__ = ["PROBLEMS", "JAX", "rxmc"]
