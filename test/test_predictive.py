"""GP conditioning, and posterior-predictive draws on a new grid."""

import numpy as np
import pytest
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from rxmc import (
    Comparison,
    Constraint,
    Dataset,
    Model,
    Parameter,
    Problem,
    StudentT,
    Term,
)
from rxmc.diagnostics import predictive_draws
from rxmc.predictive import (
    gp_posterior_predictive,
    gp_predictive_draws,
    grid_draws,
    predictive_band,
)
from rxmc.terms import (
    constant_amplitude,
    exp_growth_amplitude,
    kernel,
    model_error,
    noise,
    noise_fraction,
    normalization,
    systematic,
    x_basis,
)
from rxmc.transforms import log


def make_kernel():
    return ConstantKernel(1.0) * RBF(length_scale=0.7) + WhiteKernel(1e-6)


class TestGPPosteriorPredictive:
    def setup_method(self):
        rng = np.random.default_rng(0)
        self.X_train = np.sort(rng.uniform(-2, 2, 12))
        self.residuals = np.sin(self.X_train) + 0.05 * rng.standard_normal(12)
        self.X_pred = np.linspace(-2.5, 2.5, 25)
        self.kernel = make_kernel()
        self.theta = self.kernel.theta

    def test_matches_sklearn_gpr(self):
        gpr = GaussianProcessRegressor(
            kernel=self.kernel.clone_with_theta(self.theta),
            optimizer=None,
            alpha=0.01,
            normalize_y=False,
        )
        gpr.fit(self.X_train[:, None], self.residuals)
        mean_sk, cov_sk = gpr.predict(self.X_pred[:, None], return_cov=True)
        mean, cov = gp_posterior_predictive(
            self.kernel,
            self.theta,
            self.X_train,
            self.residuals,
            self.X_pred,
            train_noise_var=0.01,
        )
        np.testing.assert_allclose(mean, mean_sk, atol=1e-6)
        np.testing.assert_allclose(cov, cov_sk, atol=1e-6)

    def test_noiseless_interpolation_and_2d_inputs(self):
        k = ConstantKernel(1.0) * RBF(length_scale=0.7)
        mean, cov = gp_posterior_predictive(
            k, k.theta, self.X_train, self.residuals, self.X_train
        )
        # the nugget keeps a near-singular K invertible; interpolation to 1e-2
        np.testing.assert_allclose(mean, self.residuals, atol=1e-2)
        assert np.all(np.diag(cov) < 1e-2)
        X2 = np.column_stack([self.X_train, self.X_train**2])
        mean2, cov2 = gp_posterior_predictive(
            k, k.theta, X2, self.residuals, X2[:5], train_noise_var=0.01
        )
        assert mean2.shape == (5,) and cov2.shape == (5, 5)

    def test_predictive_band_percentiles(self):
        draws = np.tile(np.arange(101.0)[:, None], (1, 3))
        band = predictive_band(draws)
        np.testing.assert_allclose(band, [[16] * 3, [50] * 3, [84] * 3])


# ----------------------------------------------------------------------------
# gp_predictive_draws on a Problem
# ----------------------------------------------------------------------------


def line():
    m = Parameter("m", prior=stats.norm(0, 10))
    b = Parameter("b", prior=stats.norm(0, 10))
    return Model(lambda x, m, b: m * x + b, [m, b])


X = np.linspace(0.0, 1.0, 10)
Y = 0.5 * X + 0.2 + 0.3 * np.sin(3 * X)  # a smooth defect on a line
X_PRED = np.linspace(-0.2, 1.2, 30)


def problem(terms_first, space=None, err=0.05, fixed=True):
    """A line with a GP (hyperparameters fixed or free) and a noise nuisance."""
    d = Dataset(X, Y, np.full(10, err), label="d")
    model = line()
    comp = Comparison(d, model) if space is None else Comparison(d, model, space=space)
    if fixed:
        gp = kernel(RBF(0.3, "fixed"))
    else:
        gp = kernel(RBF(0.3), params=[Parameter("log_ell", prior=stats.norm(-1, 1))])
    eps = noise(Parameter("log_eps", prior=stats.norm(-3, 1)))
    terms = [eps, gp] if terms_first == "noise" else [gp, eps]
    p = Problem([Constraint([comp], terms=terms)])
    return p, gp, model, comp


class TestGPPredictiveDraws:
    def chain(self, p, n=40, seed=2):
        rng = np.random.default_rng(seed)
        rows = np.zeros((n, p.ndim))
        rows[:, p.names.index("m")] = 0.5 + 0.05 * rng.standard_normal(n)
        rows[:, p.names.index("b")] = 0.2 + 0.05 * rng.standard_normal(n)
        rows[:, p.names.index("log_eps")] = np.log(0.05)
        if "log_ell" in p.names:
            rows[:, p.names.index("log_ell")] = np.log(0.3)
        return rows

    def test_shape_finite_and_columns_from_the_problem(self):
        p1, gp1, model1, _ = problem("noise", fixed=False)
        p2, gp2, model2, _ = problem("kernel", fixed=False)
        assert p1.names != p2.names  # the nuisance and kernel columns swapped
        bands = []
        for p, gp, model in ((p1, gp1, model1), (p2, gp2, model2)):
            band = gp_predictive_draws(
                p, gp, model.bind(X_PRED, {}), X_PRED, self.chain(p),
                terms=list(p.constraints[0].source.terms), levels=(16, 84), rng=0,
            )  # fmt: skip
            assert band.shape == (2, 30) and np.all(np.isfinite(band))
            assert np.all(band[1] >= band[0])
            bands.append(band)
        np.testing.assert_allclose(bands[0], bands[1])  # no column arithmetic

    def test_conditioned_band_passes_through_the_data(self):
        p, gp, model, comp = problem("noise", err=1e-4)
        chain = np.tile([0.5, 0.2, np.log(1e-4)], (5, 1))
        band = gp_predictive_draws(
            p, gp, model.bind(X, {}), X, chain, terms=[gp], levels=(50,), rng=1,
            conditioned=True,
        )  # fmt: skip
        np.testing.assert_allclose(band[0], Y, atol=2e-3)

    def test_joint_draws_carry_the_kernels_correlation(self):
        """A draw is a whole curve, not 30 independent points."""
        p, gp, model, _ = problem("noise")
        chain = np.tile([0.5, 0.2, np.log(0.05)], (400, 1))  # the model is fixed
        kw = dict(terms=[gp], return_draws=True, rng=7)
        pred = model.bind(X_PRED, {})
        joint = gp_predictive_draws(p, gp, pred, X_PRED, chain, **kw)
        indep = gp_predictive_draws(p, gp, pred, X_PRED, chain, joint=False, **kw)
        assert joint.shape == (400, 30)
        cj, ci = np.corrcoef(joint.T), np.corrcoef(indep.T)
        # RBF(0.3) over a grid of pitch 0.048: neighbours move together
        assert cj[0, 1] > 0.9
        assert abs(cj[0, -1]) < 0.2  # 1.4 apart: five length scales, nothing left
        assert abs(ci[0, 1]) < 0.2  # joint=False has no correlation anywhere

    def test_the_default_draws_are_mean_zero_about_the_model(self):
        """Zero mean says where the model fails; the amplitude says where."""
        d = Dataset(X, Y, np.full(10, 0.05), label="d")
        m1, m2 = line(), line()
        lA1, lA2 = (Parameter("log_A", prior=stats.norm(0, 1)) for _ in range(2))
        slope = Parameter("slope", prior=stats.norm(0, 1))
        gp_c = kernel(
            RBF(0.3, "fixed"), amplitude=constant_amplitude, amplitude_params=(lA1,)
        )
        gp_g = kernel(
            RBF(0.3, "fixed"),
            amplitude=exp_growth_amplitude(1.0),
            amplitude_params=(lA2, slope),
        )
        p1 = Problem([Constraint([Comparison(d, m1)], terms=[gp_c])])
        p2 = Problem([Constraint([Comparison(d, m2)], terms=[gp_g])])
        n = 800
        b1 = gp_predictive_draws(
            p1, gp_c, m1.bind(X_PRED, {}), X_PRED,
            np.tile([0.5, 0.2, np.log(0.2)], (n, 1)),
            terms=[gp_c], levels=(16, 84), rng=8,
        )  # fmt: skip
        b2 = gp_predictive_draws(
            p2, gp_g, m2.bind(X_PRED, {}), X_PRED,
            np.tile([0.5, 0.2, np.log(0.2), 2.0], (n, 1)),
            terms=[gp_g], levels=(16, 84), rng=8,
        )  # fmt: skip
        # the band straddles the model's own prediction, not the data
        mu = 0.5 * X_PRED + 0.2
        assert np.all((b1[0] < mu) & (mu < b1[1]))
        w1, w2 = b1[1] - b1[0], b2[1] - b2[0]
        assert w1.max() / w1.min() < 1.25  # a constant amplitude is flat in x
        assert w2[-1] > 3 * w2[0]  # exp(2x) over [-0.2, 1.2] grows by 16

    def test_terms_choose_what_a_draw_carries(self):
        p, gp, model, _ = problem("noise")
        eps = next(t for t in p.constraints[0].source.terms if t is not gp)
        chain = np.tile([0.5, 0.2, 0.0], (400, 1))  # noise of 1, kernel variance 1
        pred = model.bind(X_PRED, {})
        kw = dict(levels=(16, 84), rng=9)
        narrow = gp_predictive_draws(p, gp, pred, X_PRED, chain, terms=[gp], **kw)
        wide = gp_predictive_draws(p, gp, pred, X_PRED, chain, terms=[gp, eps], **kw)
        assert np.all((wide[1] - wide[0]) > (narrow[1] - narrow[0]))
        # the reported errors are one number per measured point: never on a grid
        with pytest.raises(ValueError, match=r"reported statistical.*'d'"):
            gp_predictive_draws(p, gp, pred, X_PRED, chain, **kw)
        with pytest.raises(ValueError, match="must include the kernel term"):
            gp_predictive_draws(p, gp, pred, X_PRED, chain, terms=[eps])
        with pytest.raises(ValueError, match="not a term of this constraint"):
            gp_predictive_draws(
                p, gp, pred, X_PRED, chain, terms=[gp, noise(Parameter("q"))]
            )

    def test_physical_band_under_log_space(self):
        p, gp, model, comp = problem("noise", space=log)
        # 26 rows put the 16th and 84th percentiles on order statistics, so the
        # monotone exp commutes with the percentile
        chain = self.chain(p, n=26)
        pred = model.bind(X_PRED, {})
        kw = dict(terms=[gp], levels=(16, 84), rng=3)
        band_log = gp_predictive_draws(p, gp, pred, X_PRED, chain, **kw)
        band_phys = gp_predictive_draws(p, gp, pred, X_PRED, chain, physical=True, **kw)
        np.testing.assert_allclose(band_phys, np.exp(band_log))

    def test_amplitude_matches_a_scaled_kernel(self):
        d = Dataset(X, Y, np.full(10, 0.05), label="d")
        A = 0.7
        m1, m2 = line(), line()
        lA = Parameter("log_A", prior=stats.norm(0, 1))
        gp_amp = kernel(
            RBF(0.3, "fixed"), amplitude=constant_amplitude, amplitude_params=(lA,)
        )
        gp_fix = kernel(ConstantKernel(A**2, "fixed") * RBF(0.3, "fixed"))
        p1 = Problem([Constraint([Comparison(d, m1)], terms=[gp_amp])])
        p2 = Problem([Constraint([Comparison(d, m2)], terms=[gp_fix])])
        chain1 = np.tile([0.5, 0.2, np.log(A)], (8, 1))
        chain2 = np.tile([0.5, 0.2], (8, 1))
        b1 = gp_predictive_draws(
            p1, gp_amp, m1.bind(X_PRED, {}), X_PRED, chain1, terms=[gp_amp], rng=4
        )
        b2 = gp_predictive_draws(
            p2, gp_fix, m2.bind(X_PRED, {}), X_PRED, chain2, terms=[gp_fix], rng=4
        )
        np.testing.assert_allclose(b1, b2, atol=1e-8)

    def test_amplitude_reads_the_predictor_meta(self):
        # recipe 22: an amplitude keyed on the dataset's energy is, at 25 MeV, a
        # constant amplitude of 0.5 at the data and at the prediction points
        d = Dataset(X, Y, np.full(10, 0.05), label="d", meta={"Elab": 25.0})
        m1, m2 = line(), line()
        lA1, lA2 = (Parameter("log_A", prior=stats.norm(0, 1)) for _ in range(2))
        gp_meta = kernel(
            RBF(0.3, "fixed"),
            amplitude=lambda c, lA: np.exp(lA) * c.meta("Elab") / 50.0,
            amplitude_params=(lA1,),
        )
        gp_const = kernel(
            RBF(0.3, "fixed"), amplitude=constant_amplitude, amplitude_params=(lA2,)
        )
        p1 = Problem([Constraint([Comparison(d, m1)], terms=[gp_meta])])
        p2 = Problem([Constraint([Comparison(d, m2)], terms=[gp_const])])
        chain1 = np.tile([0.5, 0.2, 0.0], (8, 1))
        chain2 = np.tile([0.5, 0.2, np.log(0.5)], (8, 1))
        b1 = gp_predictive_draws(
            p1, gp_meta, m1.bind(X_PRED, d.meta), X_PRED, chain1, terms=[gp_meta], rng=4
        )
        b2 = gp_predictive_draws(
            p2, gp_const, m2.bind(X_PRED, {}), X_PRED, chain2, terms=[gp_const], rng=4
        )
        np.testing.assert_allclose(b1, b2, atol=1e-8)

    def test_one_bare_callable_is_one_space(self):
        # space=np.log wraps into a new Transform on each comparison
        model = line()
        ds = [Dataset(X, Y + 1.0 + k, np.full(10, 0.05), label=f"d{k}") for k in (0, 1)]
        comps = [Comparison(d, model, space=np.log) for d in ds]
        gp = kernel(RBF(0.3, "fixed"), on=comps)
        p = Problem([Constraint(comps, terms=[gp])])
        chain = np.tile([0.5, 1.2], (4, 1))
        band = gp_predictive_draws(
            p, gp, model.bind(X_PRED, {}), X_PRED, chain, terms=[gp]
        )
        assert band.shape == (3, 30) and np.all(np.isfinite(band))

    def test_fully_masked_kernel_support_says_so(self):
        model = line()
        c1, c2 = (
            Comparison(Dataset(X, Y, np.full(10, 0.05), label=lab), model)
            for lab in "ab"
        )
        gp = kernel(RBF(0.3, "fixed"), on=c2)
        c = Constraint([c1, c2], terms=[gp]).masked(
            [np.ones(10, bool), np.zeros(10, bool)]
        )
        chain = np.tile([0.5, 0.2], (4, 1))
        with pytest.raises(ValueError, match="fully masked"):
            gp_predictive_draws(Problem([c]), gp, model.bind(X_PRED, {}), X_PRED, chain)

    def test_explicit_noise_and_errors(self):
        p, gp, model, comp = problem("noise")
        chain = self.chain(p, n=6)
        pred = model.bind(X_PRED, {})
        band = gp_predictive_draws(
            p, gp, pred, X_PRED, chain, terms=[gp], train_noise_var=0.05**2,
            noise_std=0.1, rng=5, conditioned=True,
        )  # fmt: skip
        assert band.shape == (3, 30)
        with pytest.raises(ValueError, match="conditioned=True"):
            gp_predictive_draws(p, gp, pred, X_PRED, chain, train_noise_var=0.01)
        with pytest.raises(TypeError, match="KernelTerm"):
            gp_predictive_draws(p, Term(np.ones(10), kind="diag"), pred, X_PRED, chain)
        with pytest.raises(ValueError, match="not part of any constraint"):
            gp_predictive_draws(p, kernel(RBF(1.0, "fixed")), pred, X_PRED, chain)
        with pytest.raises(ValueError, match=r"\(n, 3\)"):
            gp_predictive_draws(p, gp, pred, X_PRED, chain[:, :2])


# ----------------------------------------------------------------------------
# grid_draws: any error model whose terms are functions of x
# ----------------------------------------------------------------------------

X_GRID = np.linspace(-1.0, 2.0, 16)  # well past the data on both sides


def one(term_or_terms, err=0.05, statistical=False, likelihood=None, space=None):
    """A line on (X, Y) with the given terms, and the model it compares."""
    terms = term_or_terms if isinstance(term_or_terms, list) else [term_or_terms]
    model = line()
    d = Dataset(X, Y, np.full(10, err), label="d")
    comp = Comparison(d, model) if space is None else Comparison(d, model, space=space)
    kw = {} if likelihood is None else {"likelihood": likelihood}
    c = Constraint([comp], terms=terms, statistical=statistical, **kw)
    return Problem([c]), model, comp


def cov_of(p, model, theta, x=X_GRID, n_rep=40000, rng=0, **kw):
    draws = grid_draws(
        p, model.bind(x, {}), x, theta, n_rep=n_rep, rng=rng, return_draws=True, **kw
    )
    return draws.mean(axis=0), np.cov(draws.T)


class TestGridDraws:
    def test_the_band_is_the_percentiles_of_the_draws(self):
        """One return convention for all three draw functions."""
        eps = noise(Parameter("log_eps", prior=stats.norm(-2, 1)))
        p, model, _ = one(eps)
        rows = np.tile([0.5, 0.2, np.log(0.1)], (50, 1))
        pred = model.bind(X_GRID, {})
        draws = grid_draws(p, pred, X_GRID, rows, rng=1, return_draws=True)
        np.testing.assert_allclose(
            grid_draws(p, pred, X_GRID, rows, rng=1), predictive_band(draws)
        )
        at_data = predictive_draws(p, rows, rng=1, return_draws=True)
        np.testing.assert_allclose(
            predictive_draws(p, rows, rng=1, levels=(5, 95)),
            predictive_band(at_data, (5, 95)),
        )
        gp = kernel(RBF(0.3, "fixed"))
        pk, mk, _ = one([gp])
        rk = np.tile([0.5, 0.2], (50, 1))
        pk_pred = mk.bind(X_GRID, {})
        gdraws = gp_predictive_draws(
            pk, gp, pk_pred, X_GRID, rk, rng=2, return_draws=True
        )
        np.testing.assert_allclose(
            gp_predictive_draws(pk, gp, pk_pred, X_GRID, rk, rng=2),
            predictive_band(gdraws),
        )

    def test_model_only_is_the_hand_push_from_posterior_or_prior_rows(self):
        eps = noise(Parameter("log_eps", prior=stats.norm(-2, 1)))
        p, model, _ = one(eps)
        pred = model.bind(X_GRID, {})
        for rows in (
            np.random.default_rng(0).normal(size=(7, 3)),
            p.sample_prior(7, rng=1),
        ):
            by_hand = np.array([pred(*r[p.columns(model.params)]) for r in rows])
            got = grid_draws(p, pred, X_GRID, rows, model_only=True, return_draws=True)
            np.testing.assert_allclose(got, by_hand)

    def test_at_the_measured_points_it_is_the_likelihoods_own_predictive(self):
        """Every term a function: the grid draw and the data draw agree."""
        eps = noise(Parameter("log_eps", prior=stats.norm(-2, 1)))
        eta = normalization(Parameter("log_eta", prior=stats.norm(-2, 1)))
        gp = kernel(ConstantKernel(0.04, "fixed") * RBF(0.3, "fixed"))
        p, model, _ = one([eps, eta, gp])
        theta = np.array([0.5, 0.2, np.log(0.1), np.log(0.2)])
        mean, cov = cov_of(p, model, theta, x=X)
        np.testing.assert_allclose(cov, p.constraints[0].matrix(theta), atol=2e-3)
        at_data = predictive_draws(p, theta, n_rep=40000, rng=1, return_draws=True)
        np.testing.assert_allclose(np.cov(at_data.T), cov, atol=3e-3)
        np.testing.assert_allclose(mean, 0.5 * X + 0.2, atol=0.01)

    def test_constant_noise_extrapolates_with_the_model(self):
        eps = noise(Parameter("log_eps", prior=stats.norm(-2, 1)))
        p, model, _ = one(eps)
        mean, cov = cov_of(p, model, np.array([0.5, 0.2, np.log(0.1)]))
        np.testing.assert_allclose(mean, 0.5 * X_GRID + 0.2, atol=0.005)
        np.testing.assert_allclose(cov, 0.01 * np.eye(16), atol=5e-4)

    def test_fractional_noise_and_model_error_scale_with_the_prediction(self):
        ym = 0.5 * X_GRID + 1.2
        for make in (noise_fraction, model_error):
            p, model, _ = one(make(Parameter("log_f", prior=stats.norm(-2, 1))))
            _, cov = cov_of(p, model, np.array([0.5, 1.2, np.log(0.1)]))
            # model_error averages y and ym; on a grid y *is* the model
            np.testing.assert_allclose(np.sqrt(np.diag(cov)), 0.1 * ym, rtol=0.03)
            # a diagonal term: no correlation between grid points
            np.testing.assert_allclose(cov - np.diag(np.diag(cov)), 0.0, atol=5e-4)

    def test_a_normalisation_mode_is_one_curve_scaled_by_the_prediction(self):
        ym = 0.5 * X_GRID + 1.2
        free = normalization(Parameter("log_eta", prior=stats.norm(-2, 1)))
        fixed = normalization(magnitude=0.1)
        for term, theta in ((free, [0.5, 1.2, np.log(0.1)]), (fixed, [0.5, 1.2])):
            p, model, _ = one([noise(Parameter("e", prior=stats.norm(-9, 1))), term])
            theta = np.array([*theta[:2], -9.0, *theta[2:]])
            _, cov = cov_of(p, model, theta)
            np.testing.assert_allclose(np.sqrt(np.diag(cov)), 0.1 * ym, rtol=0.03)
            corr = cov / np.sqrt(np.outer(np.diag(cov), np.diag(cov)))
            assert corr[0, 1] > 0.99 and corr[0, -1] > 0.99

    def test_a_systematic_mode_with_an_x_basis_grows_along_x(self):
        s = systematic(Parameter("log_s", prior=stats.norm(-2, 1)), basis=x_basis())
        eps = noise(Parameter("e", prior=stats.norm(-9, 1)))
        p, model, _ = one([eps, s])
        _, cov = cov_of(p, model, np.array([0.5, 0.2, -9.0, np.log(0.1)]))
        np.testing.assert_allclose(
            np.sqrt(np.diag(cov)), 0.1 * np.abs(X_GRID), rtol=0.03, atol=2e-3
        )

    def test_a_parametric_matrix_term_is_evaluated_from_its_definition(self):
        """Any Term(fn, params, kind="matrix") of c.x travels, not only kernels."""
        log_l = Parameter("log_l", prior=stats.norm(-1, 1))

        def sq_exp(c, ll):
            dx = np.subtract.outer(c.x, c.x)
            return 0.04 * np.exp(-0.5 * dx**2 / np.exp(ll) ** 2)

        p, model, _ = one([Term(sq_exp, (log_l,), kind="matrix")])
        theta = np.array([0.5, 0.2, np.log(0.5)])
        _, cov = cov_of(p, model, theta)
        dx = np.subtract.outer(X_GRID, X_GRID)
        np.testing.assert_allclose(cov, 0.04 * np.exp(-0.5 * dx**2 / 0.25), atol=2e-3)

    def test_a_kernel_through_grid_draws_is_the_gp_function_unconditioned(self):
        gp = kernel(RBF(0.3), params=[Parameter("log_ell", prior=stats.norm(-1, 1))])
        eps = noise(Parameter("log_eps", prior=stats.norm(-3, 1)))
        p, model, _ = one([eps, gp])
        rows = np.column_stack(
            [np.full(30, 0.5), np.full(30, 0.2), np.full(30, np.log(0.05)),
             np.log(np.linspace(0.2, 0.4, 30))]
        )  # fmt: skip
        pred = model.bind(X_GRID, {})
        kw = dict(terms=[eps, gp], n_rep=3, rng=11, return_draws=True)
        np.testing.assert_allclose(
            grid_draws(p, pred, X_GRID, rows, **kw),
            gp_predictive_draws(p, gp, pred, X_GRID, rows, **kw),
        )

    def test_joint_draws_are_curves_and_independent_draws_are_not(self):
        gp = kernel(RBF(0.5, "fixed"))
        p, model, _ = one([gp])
        pred = model.bind(X_GRID, {})
        rows = np.array([0.5, 0.2])
        kw = dict(n_rep=4000, rng=3, return_draws=True)
        joint = grid_draws(p, pred, X_GRID, rows, **kw)
        indep = grid_draws(p, pred, X_GRID, rows, joint=False, **kw)
        assert np.corrcoef(joint.T)[0, 1] > 0.9
        assert abs(np.corrcoef(indep.T)[0, 1]) < 0.1
        np.testing.assert_allclose(joint.var(0), indep.var(0), rtol=0.1)

    def test_a_term_reads_the_predictor_meta_on_the_grid(self):
        d = Dataset(X, Y, np.full(10, 0.05), label="d", meta={"Elab": 25.0})
        model = line()
        log_a = Parameter("log_a", prior=stats.norm(-2, 1))
        by_energy = Term(
            lambda c, la: np.exp(la) * c.meta("Elab") / 50.0 * np.ones(len(c)),
            (log_a,),
            kind="diag",
        )
        p = Problem(
            [Constraint([Comparison(d, model)], terms=[by_energy], statistical=False)]
        )
        theta = np.array([0.5, 0.2, 0.0])
        draws = grid_draws(
            p, model.bind(X_GRID, {"Elab": 100.0}), X_GRID, theta,
            n_rep=40000, rng=4, return_draws=True,
        )  # fmt: skip
        np.testing.assert_allclose(draws.std(0), 2.0, rtol=0.03)  # 100 / 50

    def test_physical_and_a_student_t_likelihood(self):
        eps = noise(Parameter("log_eps", prior=stats.norm(-2, 1)))
        p, model, _ = one(eps, space=log)
        rows = np.tile([0.5, 1.2, np.log(0.1)], (26, 1))
        pred = model.bind(X_GRID, {})
        kw = dict(levels=(16, 84), rng=5)
        np.testing.assert_allclose(
            grid_draws(p, pred, X_GRID, rows, physical=True, **kw),
            np.exp(grid_draws(p, pred, X_GRID, rows, **kw)),
        )
        eps_t = noise(Parameter("log_eps", prior=stats.norm(-2, 1)))
        pt, mt, _ = one(eps_t, likelihood=StudentT())
        theta = np.zeros(pt.ndim)
        theta[pt.names.index("m")], theta[pt.names.index("b")] = 0.5, 0.2
        theta[pt.names.index("log_eps")] = np.log(0.1)
        theta[pt.names.index("nu")] = 3.0
        t_draws = grid_draws(pt, mt.bind(X_GRID, {}), X_GRID, theta, n_rep=40000,
                             rng=6, return_draws=True)  # fmt: skip
        # a multivariate t with nu = 3 has variance nu / (nu - 2) = 3 times the scale
        np.testing.assert_allclose(t_draws.var(0).mean(), 3 * 0.01, rtol=0.2)

    def test_several_comparisons_need_to_say_which_experiment_the_grid_is(self):
        model = line()
        da = Dataset(X, Y, np.full(10, 0.05), label="a")
        db = Dataset(X, Y + 0.1, np.full(10, 0.05), label="b")
        ca, cb = Comparison(da, model), Comparison(db, model)
        na = normalization(magnitude=0.1, on=ca)
        nb = normalization(magnitude=0.3, on=cb)
        eps = noise(Parameter("log_eps", prior=stats.norm(-9, 1)))
        p = Problem([Constraint([ca, cb], terms=[eps, na, nb], statistical=False)])
        theta = np.array([0.5, 1.2, -9.0])
        pred = model.bind(X_GRID, {})
        with pytest.raises(ValueError, match="comparison="):
            grid_draws(p, pred, X_GRID, theta)
        draws = grid_draws(
            p, pred, X_GRID, theta, comparison=ca, n_rep=40000, rng=7, return_draws=True
        )
        # experiment a's normalisation only, not b's as well
        np.testing.assert_allclose(draws.std(0), 0.1 * (0.5 * X_GRID + 1.2), rtol=0.03)
        with pytest.raises(ValueError, match="not a comparison"):
            grid_draws(p, pred, X_GRID, theta, comparison=Comparison(da, model))
        # explicit terms need no comparison when the space is shared
        assert grid_draws(p, pred, X_GRID, theta, terms=[nb]).shape == (3, 16)
        cl = Comparison(
            Dataset(X, Y + 1.0, np.full(10, 0.05), label="l"), model, space=log
        )
        pl = Problem([Constraint([ca, cl], terms=[eps], statistical=False)])
        with pytest.raises(ValueError, match="one comparison space"):
            grid_draws(pl, pred, X_GRID, theta, terms=[eps])
        assert grid_draws(pl, pred, X_GRID, theta, comparison=cl).shape == (3, 16)

    def test_point_by_point_terms_have_no_value_at_a_new_x(self):
        model = line()
        d = Dataset(X, Y, np.full(10, 0.05), label="d")
        comp = Comparison(d, model)
        theta = np.array([0.5, 0.2])
        pred = model.bind(X_GRID, {})
        reported = Problem([Constraint([comp])])
        with pytest.raises(ValueError, match=r"reported statistical errors.*'d'"):
            grid_draws(reported, pred, X_GRID, theta)
        # the choice made explicit is allowed: the model alone
        assert grid_draws(reported, pred, X_GRID, theta, terms=[]).shape == (3, 16)
        fixed = Term(np.full(10, 0.1), kind="diag")
        p = Problem([Constraint([comp], terms=[fixed])])
        with pytest.raises(ValueError, match="array-valued term"):
            grid_draws(p, pred, X_GRID, theta, terms=[fixed])
        per_point = normalization(magnitude=np.full(10, 0.1))
        eps = noise(Parameter("log_eps", prior=stats.norm(-2, 1)))
        p = Problem([Constraint([comp], terms=[eps, per_point], statistical=False)])
        theta = np.array([0.5, 0.2, np.log(0.1)])
        with pytest.raises(ValueError, match="could not be evaluated"):
            grid_draws(p, pred, X_GRID, theta)
        with pytest.raises(ValueError, match="not a term of this constraint"):
            grid_draws(p, pred, X_GRID, theta, terms=[normalization(magnitude=0.1)])
        with pytest.raises(ValueError, match=r"\(n, 3\)"):
            grid_draws(p, pred, X_GRID, np.zeros((4, 2)))
