"""The compile step: index, priors, compiled constraints, the flat interface."""

import dill
import dynesty
import emcee
import numpy as np
import pytest
from scipy import stats

from helpers import manual_mvn_loglike
from rxmc import Comparison, Constraint, Dataset, Model, Parameter, Problem
from rxmc.likelihood import StudentT
from rxmc.problem import ParameterIndex, clip_unit_cube
from rxmc.terms import Term, kernel, noise, normalization, offset, statistical
from rxmc.transforms import log, scale

X = np.linspace(0.0, 2.0, 6)
TRUE = (2.0, 1.0)


def line_model(prior=True):
    m = Parameter("m", prior=stats.norm(0, 5) if prior else None)
    b = Parameter("b", prior=stats.norm(0, 5) if prior else None)
    return Model(lambda x, m, b: m * x + b, [m, b])


def dataset(seed=0, n=6, label="d"):
    rng = np.random.default_rng(seed)
    y = TRUE[0] * X[:n] + TRUE[1] + rng.normal(0, 0.1, n)
    return Dataset(X[:n], y, 0.1 * np.ones(n), label=label)


def problem(**kw):
    d = dataset()
    return Problem([Constraint([Comparison(d, line_model())], **kw)])


# ----------------------------------------------------------------------------
# The index
# ----------------------------------------------------------------------------


class TestIndex:
    def test_first_seen_order_predictors_terms_likelihood(self):
        model = line_model()
        eps, nu = Parameter("log_eps", prior=stats.norm()), Parameter(
            "nu", bounds=(1, 50)
        )
        c = Constraint(
            [Comparison(dataset(), model)], terms=[noise(eps)], likelihood=StudentT(nu)
        )
        p = Problem([c])
        assert p.names == ["m", "b", "log_eps", "nu"]
        assert p.params == (*model.params, eps, nu)
        assert p.ndim == 4 and p.bounds.shape == (4, 2)

    def test_shared_object_is_one_slot_across_constraints(self):
        eps = Parameter("log_eps", prior=stats.norm())
        model = line_model()
        cs = [
            Constraint([Comparison(dataset(0, label="a"), model)], terms=[noise(eps)]),
            Constraint([Comparison(dataset(1, label="b"), model)], terms=[noise(eps)]),
        ]
        p = Problem(cs)
        assert p.names == ["m", "b", "log_eps"]
        assert np.array_equal(p.columns(eps), [2])
        assert np.array_equal(p.columns([eps, model.params[0]]), [2, 0])

    def test_duplicate_names_raise_with_hint(self):
        a1, a2 = Parameter("log_eps", prior=stats.norm()), Parameter(
            "log_eps", prior=stats.norm()
        )
        c = Constraint(
            [Comparison(dataset(), line_model())], terms=[noise(a1), noise(a2)]
        )
        with pytest.raises(ValueError, match="duplicate parameter name.*SAME object"):
            Problem([c])
        # two default StudentT likelihoods derive two "nu"
        cs = [
            Constraint(
                [Comparison(dataset(0, label="a"), line_model())], likelihood=StudentT()
            ),
            Constraint(
                [Comparison(dataset(1, label="b"), line_model(prior=False))],
                likelihood=StudentT(),
            ),
        ]
        with pytest.raises(ValueError, match="nu="):
            Problem(cs)

    def test_index_api(self):
        ix = ParameterIndex()
        a, b = Parameter("a"), Parameter("b")
        assert np.array_equal(ix.add_all([a, b, a]), [0, 1, 0])
        assert ix.slot(b) == 1 and ix.names == ["a", "b"]
        with pytest.raises(KeyError):
            ix.slot(Parameter("c"))
        with pytest.raises(TypeError):
            ix.add_all(["a"])


# ----------------------------------------------------------------------------
# Priors
# ----------------------------------------------------------------------------


class TestPriors:
    def test_marginals_bounded_and_uniform(self):
        m = Parameter("m", prior=stats.norm(0, 2), bounds=(-1.0, 3.0))
        b = Parameter("b", bounds=(0.0, 4.0))
        p = Problem(
            [
                Constraint(
                    [Comparison(dataset(), Model(lambda x, m, b: m * x + b, [m, b]))]
                )
            ]
        )
        tn = stats.truncnorm(-0.5, 1.5, loc=0, scale=2)
        assert p.log_prior([0.5, 1.0]) == pytest.approx(tn.logpdf(0.5) - np.log(4.0))
        assert p.log_prior([3.5, 1.0]) == -np.inf and p.log_prior([0.5, 5.0]) == -np.inf
        theta = p.prior_transform([0.3, 0.25])
        assert theta[0] == pytest.approx(tn.ppf(0.3)) and theta[1] == pytest.approx(1.0)
        assert np.all(np.isfinite(p.prior_transform([0.0, 1.0])))

    def test_unbounded_marginal(self):
        p = problem()
        assert p.log_prior([1.0, 2.0]) == pytest.approx(
            stats.norm(0, 5).logpdf([1.0, 2.0]).sum()
        )
        u = np.array([0.1, 0.9])
        np.testing.assert_allclose(stats.norm(0, 5).cdf(p.prior_transform(u)), u)

    def test_joint_mvn_whitening(self):
        model = line_model(prior=False)
        mu, cov = np.array([2.0, 1.0]), np.array([[0.5, 0.2], [0.2, 0.3]])
        mvn = stats.multivariate_normal(mu, cov)
        p = Problem(
            [Constraint([Comparison(dataset(), model)])], priors=[(model.params, mvn)]
        )
        assert p.log_prior([1.0, 1.0]) == pytest.approx(mvn.logpdf([1.0, 1.0]))
        rng = np.random.default_rng(0)
        draws = np.array([p.prior_transform(u) for u in rng.uniform(size=(4000, 2))])
        np.testing.assert_allclose(draws.mean(0), mu, atol=0.05)
        np.testing.assert_allclose(np.cov(draws.T), cov, rtol=0.1, atol=0.03)
        assert np.all(np.isfinite(p.prior_transform([0.0, 1.0])))
        s = p.sample_prior(500, rng=1)
        assert s.shape == (500, 2) and np.allclose(s.mean(0), mu, atol=0.15)

    def test_coverage_errors_name_the_parameter(self):
        model = line_model(prior=False)
        with pytest.raises(ValueError, match="'m' has no prior"):
            Problem([Constraint([Comparison(dataset(), model)])])
        m, b = model.params
        with pytest.raises(ValueError, match="'m' appears in two joint blocks"):
            Problem(
                [Constraint([Comparison(dataset(), model)])],
                priors=[
                    ([m], stats.norm()),
                    ([m, b], stats.multivariate_normal(np.zeros(2))),
                ],
            )
        model2 = line_model()  # marginals set
        with pytest.raises(ValueError, match="'m' has its own prior= and is also"):
            Problem(
                [Constraint([Comparison(dataset(), model2)])],
                priors=[(model2.params, stats.multivariate_normal(np.zeros(2)))],
            )

    def test_bounded_joint_has_no_transform_but_truncates(self):
        m, b = Parameter("m", bounds=(0.0, 10.0)), Parameter("b")
        model = Model(lambda x, m, b: m * x + b, [m, b])
        mvn = stats.multivariate_normal(np.zeros(2), np.eye(2))
        p = Problem(
            [Constraint([Comparison(dataset(), model)])], priors=[([m, b], mvn)]
        )
        assert p.log_prior([-1.0, 0.0]) == -np.inf
        assert p.log_prior([1.0, 0.0]) == pytest.approx(mvn.logpdf([1.0, 0.0]))
        with pytest.raises(NotImplementedError, match="truncated"):
            p.prior_transform([0.5, 0.5])
        s = p.sample_prior(50, rng=0)  # rejection sampling inside the bounds
        assert np.all(s[:, 0] >= 0.0)

    def test_custom_joint_with_prior_transform(self):
        class Hier:
            def logpdf(self, v):
                *r, lt = v
                return stats.norm(0, np.exp(lt)).logpdf(r).sum() + stats.norm(
                    0, 1
                ).logpdf(lt)

            def prior_transform(self, u):
                lt = stats.norm(0, 1).ppf(u[-1])
                return np.append(stats.norm(0, np.exp(lt)).ppf(u[:-1]), lt)

        rhos = [Parameter(f"log_rho_{i}") for i in range(2)]
        log_tau = Parameter("log_tau")
        model = Model(lambda x, r0, r1, lt: np.ones_like(x), rhos + [log_tau])
        p = Problem(
            [Constraint([Comparison(dataset(), model)])],
            priors=[(rhos + [log_tau], Hier())],
        )
        theta = p.prior_transform([0.5, 0.5, 0.5])
        np.testing.assert_allclose(theta, [0.0, 0.0, 0.0], atol=1e-12)
        assert np.isfinite(p.log_prior(theta))

    def test_clip_unit_cube(self):
        u = clip_unit_cube([0.0, 0.5, 1.0])
        assert 0 < u[0] < 1e-10 and u[1] == 0.5 and 1 - 1e-10 < u[2] < 1


# ----------------------------------------------------------------------------
# Compiled constraints and the flat interface
# ----------------------------------------------------------------------------


class TestCompile:
    def test_log_likelihood_matches_manual_mvn(self):
        d = dataset()
        p = problem()
        theta = np.array(TRUE)
        ym = TRUE[0] * d.x + TRUE[1]
        assert p.log_likelihood(theta) == pytest.approx(
            manual_mvn_loglike(d.y, ym, np.diag(d.y_err**2))
        )
        assert p.chi2(theta) == pytest.approx(np.sum(((d.y - ym) / d.y_err) ** 2))
        assert p.log_posterior(theta) == pytest.approx(
            p.log_prior(theta) + p.log_likelihood(theta)
        )

    def test_weights_and_two_constraints(self):
        model = line_model()
        c1 = Constraint([Comparison(dataset(0, label="a"), model)], weight=0.5)
        c2 = Constraint([Comparison(dataset(1, label="b"), model)], weight=2.0)
        p = Problem([c1, c2])
        theta = np.array(TRUE)
        assert p.log_likelihood(theta) == pytest.approx(
            0.5 * p.constraints[0].log_likelihood(theta)
            + 2.0 * p.constraints[1].log_likelihood(theta)
        )

    def test_prior_first_skips_the_forward_model(self):
        calls = []
        m = Parameter("m", bounds=(0.0, 5.0))
        b = Parameter("b", bounds=(0.0, 5.0))

        def fn(x, m, b):
            calls.append(1)
            return m * x + b

        p = Problem([Constraint([Comparison(dataset(), Model(fn, [m, b]))])])
        assert p.log_posterior([-1.0, 1.0]) == -np.inf
        assert calls == []
        p.log_posterior([1.0, 1.0])
        assert calls == [1]

    def test_non_finite_prediction(self):
        d = dataset()
        p = Problem([Constraint([Comparison(d, line_model(), space=log)])])
        theta = np.array([-5.0, 0.0])  # negative prediction under log
        assert p.log_likelihood(theta) == -np.inf and p.chi2(theta) == np.inf

    def test_non_positive_data_under_log_named_unless_masked(self):
        d = Dataset(X[:3], [1.0, -1.0, 2.0], [0.1, 0.1, 0.1], label="neg")
        c = Constraint([Comparison(d, line_model(), space=log)])
        with pytest.raises(ValueError, match="'neg'.*not finite"):
            Problem([c])
        Problem([c.masked([np.array([True, False, True])])])  # fine

    def test_meta_reaches_terms(self):
        seen = {}

        def fn(c):
            seen["E"], seen["w"], seen["r"] = (
                c.meta("Elab"),
                c.meta("w"),
                c.meta("reaction"),
            )
            return np.ones(len(c))

        model = line_model()
        d1 = Dataset(
            X[:2],
            [1.0, 2.0],
            [0.1, 0.1],
            meta={"Elab": 10.0, "w": [1.0, 2.0], "reaction": "n+Ca"},
        )
        d2 = Dataset(
            X[:3],
            [1.0, 2.0, 3.0],
            [0.1, 0.1, 0.1],
            meta={"Elab": 20.0, "w": [3.0, 4.0, 5.0]},
        )
        c = Constraint(
            [Comparison(d1, model), Comparison(d2, model)],
            terms=[Term(fn, kind="diag", constant=True)],
        )
        Problem([c])
        np.testing.assert_allclose(seen["E"], [10, 10, 20, 20, 20])
        np.testing.assert_allclose(seen["w"], [1, 2, 3, 4, 5])
        assert list(seen["r"]) == ["n+Ca", "n+Ca", None, None, None]

    def test_singular_covariance_names_the_comparison(self):
        d = Dataset(X[:3], [1.0, 2.0, 3.0], np.zeros(3), label="E1234-002")
        with pytest.raises(ValueError, match="E1234-002.*reported_terms"):
            Problem([Constraint([Comparison(d, line_model())])])
        eps = Parameter("log_eps", prior=stats.norm())
        Problem(
            [Constraint([Comparison(d, line_model())], terms=[noise(eps)])]
        )  # parametric: fine

    def test_predict_and_matrix(self):
        d = dataset()
        p = Problem(
            [
                Constraint(
                    [Comparison(d, line_model(), space=log)],
                    terms=[normalization(magnitude=0.05)],
                )
            ]
        )
        theta = np.array(TRUE)
        ((pred,),) = p.predict(theta)
        np.testing.assert_allclose(pred, np.log(TRUE[0] * d.x + TRUE[1]))
        ((phys,),) = p.predict(theta, physical=True)
        np.testing.assert_allclose(phys, TRUE[0] * d.x + TRUE[1])
        S = p.constraints[0].matrix(theta)
        assert S.shape == (6, 6) and np.all(np.linalg.eigvalsh(S) > 0)

    def test_theta_shape_checked(self):
        with pytest.raises(ValueError, match="shape"):
            problem().log_posterior([1.0])


class TestViews:
    def test_masked_views_share_columns_and_partition(self):
        d = dataset()
        eps = Parameter("log_eps", prior=stats.norm())
        c = Constraint([Comparison(d, line_model())], terms=[noise(eps)])
        fit, held = (
            c.masked_where(lambda x: x < 1.0),
            c.masked_where(lambda x: x < 1.0).complement(),
        )
        pf, ph, pa = Problem([fit]), Problem([held]), Problem([c])
        assert pf.names == ph.names == pa.names
        theta = np.array([*TRUE, np.log(0.2)])
        assert pf.log_likelihood(theta) + ph.log_likelihood(theta) == pytest.approx(
            pa.log_likelihood(theta)
        )

    def test_parameter_on_fully_masked_comparison_keeps_its_slot(self):
        model = line_model()
        rho1, rho2 = Parameter("log_rho_1", prior=stats.norm(0, 0.1)), Parameter(
            "log_rho_2", prior=stats.norm(0, 0.1)
        )
        c1, c2 = Comparison(dataset(0, label="a"), model | scale(rho1)), Comparison(
            dataset(1, label="b"), model | scale(rho2)
        )
        c = Constraint([c1, c2]).masked([np.ones(6, bool), np.zeros(6, bool)])
        p = Problem([c])
        assert p.names == ["m", "b", "log_rho_1", "log_rho_2"]
        assert np.isfinite(p.log_prior([*TRUE, 0.0, 0.0]))
        s = p.sample_prior(10, rng=0)
        assert s.shape == (10, 4) and np.std(s[:, 3]) > 0


# ----------------------------------------------------------------------------
# Drivers
# ----------------------------------------------------------------------------


class TestDrivers:
    def test_emcee_recovers_the_line(self):
        p = problem()
        rng = np.random.default_rng(0)
        p0 = p.sample_prior(16, rng=rng) * 0.05 + np.array(TRUE)
        sampler = emcee.EnsembleSampler(16, p.ndim, p.log_posterior)
        sampler.random_state = np.random.RandomState(0).get_state()
        sampler.run_mcmc(p0, 150, progress=False)
        chain = sampler.get_chain(discard=50, flat=True)
        assert abs(chain[:, 0].mean() - TRUE[0]) < 3 * chain[:, 0].std() + 0.05

    def test_dynesty_runs(self):
        p = problem()
        ns = dynesty.NestedSampler(
            p.log_likelihood,
            p.prior_transform,
            p.ndim,
            nlive=50,
            rstate=np.random.default_rng(0),
        )
        ns.run_nested(dlogz=1.0, print_progress=False)
        assert np.isfinite(ns.results.logz[-1])

    def test_dill_round_trip(self):
        p = problem()
        q = dill.loads(dill.dumps(p))
        theta = np.array(TRUE)
        assert q.names == p.names
        assert q.log_posterior(theta) == pytest.approx(p.log_posterior(theta))
        assert p.NDIM == p.ndim and p.parameter_names == p.names
        assert p.starting_location(3).shape == (3, 2)
        np.testing.assert_allclose(
            p.log_posterior_batch([theta, theta]), [p.log_posterior(theta)] * 2
        )

    def test_kernel_and_offset_terms_pickle(self):
        eps = Parameter("log_A", prior=stats.norm())
        from sklearn.gaussian_process.kernels import RBF

        c = Constraint(
            [Comparison(dataset(), line_model())],
            terms=[
                kernel(RBF(1.0), params=[Parameter("ell", prior=stats.norm())]),
                offset(parameter=eps),
                statistical(np.ones(6)),
            ],
        )
        p = Problem([c])
        q = dill.loads(dill.dumps(p))
        theta = np.array([*TRUE, 0.0, -1.0])
        assert q.log_posterior(theta) == pytest.approx(p.log_posterior(theta))
