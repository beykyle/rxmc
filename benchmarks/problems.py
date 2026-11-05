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
* ``omp``         -- exactly the log-space p + 40Ca problem of the notebook
                     ``local_optical_model_calibration``: EXFOR C0076002
                     (Gross et al. 1967) at 35.8 MeV, seven potential
                     parameters and two median-one log-space errors (9
                     parameters; JAX rxmc only)
* ``alpha_gp``    -- exactly the power-of-q problem of
                     ``alpha_ca_error_model_comparison``: two log-space errors
                     and a Matern GP whose amplitude grows with q (9
                     parameters, two families of potentials; JAX only)
* ``n40ca_gp``    -- exactly the "GP, growing amplitude" rung of
                     ``gp_discrepancy``'s cross-section ladder (7 parameters;
                     JAX only)
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
    df = pd.read_csv(
        here / "examples" / "data" / "p40ca_elastic_35p8mev_gross1967.csv",
        comment="#",
    )
    measurement = {
        "x": df["angle_cm_deg"].to_numpy(),
        "y": df["dxs_mb_sr"].to_numpy(),
        "statistical_err": df["stat_err_mb_sr"].to_numpy(),
        "Einc": 35.8,
        "quantity": "dXS/dA",
        "y_units": "mb/sr",
        "x_units": "CM-degrees",
        "systematic_norm_err": 0.0,
        "systematic_offset_err": 0.0,
        "subentry": "C0076002",
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
        "dXS/dA",
        central,
        spin_orbit,
        lambda ws, *x: (tuple(x), (5.5, 1.0 * A13, 0.6)),
        params,
        coulomb=coulomb,
        lmax=30,
    )
    # median-one point-to-point and common errors, as log-space sds
    log_sigma_A = Parameter("log_sigma_A", prior=_log_uniform(0.05, 2.0))
    log_sigma_B = Parameter("log_sigma_B", prior=_log_uniform(0.05, 2.0))
    errors = [T.noise(log_sigma_A), T.offset(log_sigma_B)]
    p = Problem([Constraint([Comparison(data, model, space=tf.log)], terms=errors)])
    return p, _thetas(p)


def _log_uniform(lo, hi):
    return stats.uniform(np.log(lo), np.log(hi) - np.log(lo))


def alpha_gp():
    """alpha_ca_error_model_comparison's power-of-q model, exactly as the
    notebook builds it: 44Ca(a,a) at 29 MeV, every fourth angle, in log
    space, with point-to-point and common log-space errors and a Matern GP
    in u = theta / pi whose amplitude grows as sin(theta / 2) ** p."""
    import jax.numpy as jnp
    import jitr
    import pandas as pd
    import tinygp
    from jitr.optical_potentials.potential_forms import (
        coulomb_charged_sphere,
        woods_saxon_safe,
    )

    from rxmc import transforms as tf

    here = __import__("pathlib").Path(__file__).resolve().parents[1]
    df = pd.read_csv(here / "examples" / "data" / "alpha_ca_ratio_ruth.csv")
    df = df[df["target"] == "Ca44"].iloc[::4]
    reaction = jitr.reactions.ElasticReaction(target=(44, 20), projectile=(4, 2))
    meta = {"reaction": reaction, "Elab": 29.0, "quantity": "dXS/dRuth"}
    angles = np.deg2rad(df["angle_cm_deg"].to_numpy())
    ratio = df["ratio_to_rutherford"].to_numpy()
    data = Dataset(
        angles, ratio, np.zeros(ratio.size), label="44Ca(a,a) 29 MeV", meta=meta
    )
    A13 = 44 ** (1 / 3)

    def central(r, V, W, R, a):
        return -(V + 1j * W) * woods_saxon_safe(r, R * A13, a)

    def coulomb(r):
        return coulomb_charged_sphere(
            r, reaction.target.Z * reaction.projectile.Z, 1.3 * A13
        )

    params = [
        Parameter("V", prior=stats.norm(150.0, 20.0), bounds=(0.0, np.inf)),
        Parameter("W", prior=stats.norm(20.0, 10.0), bounds=(0.0, np.inf)),
        Parameter("r", prior=stats.norm(1.4, 0.3), bounds=(0.8, 2.0)),
        Parameter("a", prior=stats.norm(0.5, 0.2), bounds=(0.2, 1.0)),
    ]
    model = rxmc.reactions.ElasticXS(
        "dXS/dRuth",
        central,
        None,
        lambda ws, *x: (tuple(x), (), ()),
        params,
        coulomb=coulomb,
        lmax=30,
    )
    log_sigma_A = Parameter("log_sigma_A", prior=_log_uniform(0.05, 2.0))
    log_sigma_B = Parameter("log_sigma_B", prior=_log_uniform(0.05, 2.0))

    def by_angle(x):
        return np.asarray(x, dtype=float) / np.pi

    u = np.sort(by_angle(angles))
    log_ell = Parameter(
        "log_ell", prior=_log_uniform(np.median(np.diff(u)), u[-1] - u[0])
    )
    log_A_q = Parameter("log_A_q", prior=_log_uniform(0.05, 2.0))
    power = Parameter("p", prior=stats.uniform(0.0, 16.0))

    def power_of_q(c, log_amplitude, p):
        return jnp.exp(log_amplitude) * jnp.sin(0.5 * jnp.pi * c.x) ** p

    gp = T.kernel(
        lambda log_ell: tinygp.kernels.Matern52(scale=jnp.exp(log_ell)),
        coords=by_angle,
        amplitude=power_of_q,
        amplitude_params=(log_A_q, power),
        params=[log_ell],
    )
    comp = Comparison(data, model, space=tf.log)
    terms = [T.noise(log_sigma_A), T.offset(log_sigma_B), gp]
    p = Problem([Constraint([comp], terms=terms, statistical=False)])
    return p, _thetas(p)


RUNGS = ("bare", "proportional error", "GP, growing amplitude")


def n40ca_ladder(rungs=RUNGS):
    """gp_discrepancy's cross-section ladder, exactly as the notebook builds
    it: mock n + 40Ca data at 14.1 MeV from a potential with volume and
    surface absorption (seed 2024), fitted in log space with a volume-only
    potential under three error models.  Returns ``{rung: problem}`` for the
    ``rungs`` asked for (each compiled only if asked for)."""
    import jax.numpy as jnp
    import jitr
    import tinygp
    from jitr.optical_potentials.potential_forms import (
        thomas_safe,
        woods_saxon_prime_safe,
        woods_saxon_safe,
    )

    from rxmc import transforms as tf

    rng = np.random.default_rng(2024)
    reaction = jitr.reactions.ElasticReaction(target=(40, 20), projectile=(1, 0))
    R40 = 40 ** (1 / 3)
    mso = 1.0 / jitr.utils.constants.WAVENUMBER_PION

    def full_central(r, Vv, Wv, Rv, av, Wd, Rd, ad):
        return -(Vv + 1j * Wv) * woods_saxon_safe(r, Rv, av) - 1j * Wd * (
            -4 * ad
        ) * woods_saxon_prime_safe(r, Rd, ad)

    def volume_central(r, Vv, Wv, Rv, av):
        return -(Vv + 1j * Wv) * woods_saxon_safe(r, Rv, av)

    def spin_orbit(r, Vso, Rso, aso):
        return Vso * mso**2 * thomas_safe(r, Rso, aso)

    so_args = (6.0, 1.1 * R40, 0.45)
    full_truth = np.array([48.0, 3.5, 1.1 * R40, 0.7, 21.0, 1.2 * R40, 0.5])
    omp_full = rxmc.reactions.ElasticXS(
        "dXS/dA",
        full_central,
        spin_orbit,
        lambda ws, *x: (tuple(x), so_args),
        [
            Parameter(n, prior=stats.norm(0, 1))
            for n in ("Vv", "Wv", "Rv", "av", "Wd", "Rd", "ad")
        ],
        lmax=10,
    )
    volume_params = [
        Parameter("Vv", prior=stats.norm(48.0, 8.0)),
        Parameter("Wv", prior=stats.norm(4.0, 4.0), bounds=(0.0, 30.0)),
        Parameter("Rv", prior=stats.norm(1.15 * R40, 0.2)),
        Parameter("av", prior=stats.norm(0.65, 0.1), bounds=(0.3, 1.2)),
    ]
    omp_vol = rxmc.reactions.ElasticXS(
        "dXS/dA",
        volume_central,
        spin_orbit,
        lambda ws, *x: (tuple(x), so_args),
        volume_params,
        lmax=10,
    )
    angles = np.deg2rad(np.linspace(5.0, 160.0, 25))
    meta = {"reaction": reaction, "Elab": 14.1}
    y_full = omp_full.bind(angles, meta)(*full_truth)
    y = y_full * (1 + rng.normal(0.0, 0.04, angles.size))
    d_xs = Dataset(angles, y, 0.04 * y, label="n+40Ca 14.1 MeV", meta=meta)
    comp = Comparison(d_xs, omp_vol, space=tf.log)
    gamma = Parameter("log_gamma_xs", prior=stats.norm(np.log(0.05), 1.0))
    amp = Parameter("log_A_xs", prior=stats.norm(np.log(0.05), 1.5))
    slope = Parameter("amp_slope_xs", prior=stats.norm(0.0, 1.5))
    spacing, span = np.diff(angles).min(), angles.max() - angles.min()
    ell = Parameter(
        "log_ell_xs", prior=stats.uniform(np.log(spacing), np.log(span / spacing))
    )
    gp = T.kernel(
        lambda log_ell: tinygp.kernels.Matern52(scale=jnp.exp(log_ell)),
        params=[ell],
        jitter=1e-6,
        amplitude=T.exp_growth_amplitude(np.pi),
        amplitude_params=(amp, slope),
    )
    terms = {
        "bare": [],
        "proportional error": [T.proportional_error(gamma, averaging=True)],
        "GP, growing amplitude": [gp],
    }
    return {r: Problem([Constraint([comp], terms=terms[r])]) for r in rungs}


def n40ca_gp():
    """The ladder's "GP, growing amplitude" rung (7 parameters)."""
    p = n40ca_ladder(["GP, growing amplitude"])["GP, growing amplitude"]
    return p, _thetas(p)


def mode_share(name, problem):
    """``summary(theta, weights)`` for the problems with two modes: the
    weighted share of draws away from the main one, counted as the notebooks
    count it; None for the other problems."""
    cols = {n: i for i, n in enumerate(problem.names)}
    if name == "omp":  # the island away from the MAP, as local_optical counts it

        def away(t):
            return (t[:, cols["Rd"]] < 4.37) & (t[:, cols["ad"]] > 0.34)

        label = "share away from the MAP island"
    elif name == "alpha_gp":  # the deeper family, as alpha_ca counts it

        def away(t):
            return t[:, cols["V"]] > 145.0

        label = "share in the deeper family"
    else:
        return None

    def summary(theta, weights):
        w = weights / weights.sum()
        return {"mode_share": float(w @ away(theta)), "mode_share_label": label}

    return summary


PROBLEMS = {
    "linear": linear,
    "modes": modes,
    "gp": gp,
    "gp500": lambda: gp(500),
    "elastic": elastic,
    "ias": ias,
    "hierarchy": hierarchy,
    "omp": omp,
    "alpha_gp": alpha_gp,
    "n40ca_gp": n40ca_gp,
}

__all__ = ["PROBLEMS", "JAX", "rxmc"]
