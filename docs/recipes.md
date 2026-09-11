# Recipes

Short user stories for the ground-up `rxmc` described in `groundup_design.md`.
Each recipe says what a user wants, how it is spelled, and what behaviour
they should expect.  Together they are the minimal set of use cases the
rewrite must support; every one of them is backed by a notebook or a test
in the current repository, or by a design decision recorded in the design
document.  Every recipe is also a test: `test/recipes/test_recipe_NN_<slug>.py`
asserts its *Expected behaviour* bullets, and the richest recipes are
tutorial notebooks (design document §7 and §9).  Snippets assume:

```python
import numpy as np, rxmc as rx
from rxmc import terms as T, transforms as tf
from scipy import stats
from sklearn.gaussian_process.kernels import Matern, RBF
```

---

## 1. Fit a model to data with reported statistical errors

*I have `x`, `y`, and a statistical error per point, and a model with a few
parameters.  I want the posterior.*

```python
m, b = rx.Parameter("m", prior=stats.norm(0, 5)), rx.Parameter("b", prior=stats.norm(0, 5))
line = rx.Model(lambda x, m, b: m * x + b, [m, b])
d = rx.Dataset(x, y, y_err)
problem = rx.Problem([rx.Constraint([rx.Comparison(d, line)])])
```

Expected behaviour:

- The covariance is `diag(y_err**2)` and nothing else.  `problem.chi2(theta)`
  equals `sum(((y - ym) / y_err)**2)` exactly.
- `problem.ndim == 2`, `problem.names == ["m", "b"]`, in declaration order.
- `problem.log_posterior`, `problem.sample_prior(n)`, `problem.prior_transform`
  are all that emcee or dynesty need (recipe 16).

## 2. Infer an unknown noise level

*My data have no usable error bars, or I do not trust them.  I want to
infer the noise magnitude alongside the model.*

```python
log_eps = rx.Parameter("log_eps", prior=stats.norm(-2, 2))
c = rx.Constraint([rx.Comparison(d, line)], terms=[T.noise(log_eps)], statistical=False)
# or fractional noise:  T.noise_fraction(log_eps)
# or model error on the average of data and prediction:  T.model_error(log_gamma)
# or noise growing along x:  T.noise(log_eps, basis=T.exp_growth(np.pi), basis_params=(slope,))
```

Expected behaviour:

- With `statistical=False` the inferred noise *replaces* the reported
  errors; with the default `statistical=True` it is *added* to them.
- The posterior of `log_eps` reflects the residual scatter.  In the
  `sampling_algos` scenario its truth is recovered.
- `noise_fraction` and `model_error` scale with the prediction, so the
  covariance changes with the model parameters.  That is allowed and costs
  nothing extra.

## 3. Use the reported systematic errors

*The measurement reports a fractional normalisation error and an absolute
offset error.  I want them in the likelihood as correlated modes.*

```python
d = rx.from_measurement(m, reaction=reaction, quantity="dXS/dA")   # norm_err, offset_err filled
comp = rx.Comparison(d, omp)
c = rx.Constraint([comp], terms=comp.reported_terms())
```

Expected behaviour:

- `reported_terms()` returns the offset mode first, then the
  prediction-scaled normalisation mode.  Zero magnitudes yield no term.
- Without the call, neither systematic enters the covariance.  Nothing is
  folded in silently.
- Re-adding both terms reproduces the pre-0.1 auto-folded covariance to
  the last digit (`test_regression`, log-likelihood `1.195784087817536`).
- Under `space=tf.log` the modes are propagated by the delta method: the
  offset at the data, the normalisation at the prediction.

## 4. Infer a normalisation or offset the experiment did not report

*I suspect an unreported normalisation (or background offset) and want its
magnitude as a nuisance parameter.*

```python
log_eta = rx.Parameter("log_eta", prior=stats.norm(-3, 1))
c = rx.Constraint([comp], terms=[T.normalization(parameter=log_eta)])
# absolute offset instead:  T.offset(parameter=log_omega)
# a mode with any shape:     T.systematic(log_s, basis=T.x_basis(np.pi))
```

Expected behaviour:

- One rank-one mode `exp(log_eta)**2 * outer(ym, ym)` is added.
- The model parameters decorrelate from the overall scale of the data; the
  data's normalisation pull moves into `log_eta`.

## 5. Share an error model between datasets, or couple them

*Two datasets.  Case B: each has its own independent normalisation
measurement, but I believe the two magnitudes are the same.  Case A: both
were normalised against the same uncertain flux, so their errors are
correlated.*

```python
comp1, comp2 = rx.Comparison(d1, omp), rx.Comparison(d2, omp)
log_eta = rx.Parameter("log_eta", prior=stats.norm(-3, 1))

# case B: one parameter, two comparison-local modes; the comparisons stay independent
cB = rx.Constraint([comp1, comp2], terms=[T.normalization(log_eta, on=comp1), T.normalization(log_eta, on=comp2)])
# equivalently, two constraints sharing the parameter object
cB1, cB2 = rx.Constraint([comp1], terms=[T.normalization(log_eta)]), rx.Constraint([comp2], terms=[T.normalization(log_eta)])

# case A: one mode spanning both comps; the comps are correlated
cA = rx.Constraint([comp1, comp2], terms=[T.normalization(log_eta, on=[comp1, comp2])])
```

Expected behaviour:

- All three spellings have exactly one nuisance parameter.
- Case B's covariance is block diagonal; case A's has a non-zero
  off-diagonal block.  The two likelihoods differ, and treating case A
  data as case B is overconfident (`correlated_observations`).
- Sharing is by object: two `Parameter("log_eta")` objects would be two
  parameters and a compile error for the duplicate name.
- Case A costs no more than case B: the cross-comparison mode goes through the
  low-rank path, never a dense factorisation.

## 6. One latent scale per dataset

*Each dataset has its own unknown normalisation.  I want a Kennedy–O'Hagan
scale factor on the model prediction, one per dataset, sampled with the
model parameters.*

```python
rhos = [rx.Parameter(f"log_rho_{i}", prior=stats.norm(0, np.log1p(s))) for i, s in enumerate(sys_errs)]
comps = [rx.Comparison(d_i, omp | tf.scale(rho_i)) for d_i, rho_i in zip(datasets, rhos)]
# one global scale instead:  omp | tf.scale(rho) on every comparison
```

Expected behaviour:

- The scale multiplies the *mean*, so it is not a covariance term.  Its
  parameters follow the model's in `problem.names`.
- A masked or held-out view of the comparison keeps the same `rho_i` because
  the comparison, not the data, carries the model.
- Fitting the ρᵢ recovers the true normalisations by MAP
  (`normalization_inference`); folding the reported σ_sys in as fixed
  normalisation modes (recipe 3) gives statistically the same model
  posterior.

## 7. Absorb model deficiency with a Gaussian process

*My model is missing physics.  I want a smooth correlated discrepancy, in
angle or in momentum transfer, learned from the residuals.*

```python
log_A = rx.Parameter("log_A", prior=stats.norm(0, 2))
gp = T.kernel(Matern(1.0, nu=2.5), on=comp, coords=lambda x: x / np.pi,
              amplitude=T.constant_amplitude, amplitude_params=(log_A,))
# in momentum transfer with amplitude A q^(r/2):
q = lambda x: rx.reactions.momentum_transfer(x, d.meta["k"])
gp_q = T.kernel(RBF(1.0), on=comp, coords=q, amplitude=lambda c, lA, r: np.exp(lA) * c.x ** (r / 2),
                amplitude_params=(log_A, r))
c = rx.Constraint([comp], terms=[gp])
band = rx.predictive.total_predictive_band(problem, gp, omp.bind(x_fine, d.meta), x_fine, samples)
```

Expected behaviour:

- One parameter per free kernel hyperparameter element, named
  `discrepancy_<hyperparameter>`, sampled in sklearn's log-theta space and
  bounded by the log of the kernel's bounds, so it compiles with a uniform
  prior there (`params=` for any other prior).
- `kernel()` returns a `KernelTerm`, a `Term` that also carries the kernel,
  so the predictive band can condition the discrepancy from the term alone.
- The model parameters relax from their biased values toward the truth
  (`gp_discrepancy`); the learned discrepancy tracks the true defect.
- `total_predictive_band` finds the kernel's columns from the term itself;
  no column arithmetic.  It conditions on the residuals of the training rows
  with everything *else* in the constraint's covariance (statistical errors,
  noise, modes) as the regression noise, and predicts in the comparison
  space of the term's comparisons (`physical=True` maps back).
- Every error-model form of the α+Ca study reproduces a hand-built dense
  matrix (`TestStudyForms`).

## 8. Sample a mean discrepancy explicitly

*Instead of marginalising the discrepancy, I want to sample an additive
correction with a parametric shape.*

```python
phi = [rx.Parameter(f"c{i}", prior=stats.norm(0, 1)) for i in range(3)]
delta = rx.Model(lambda x, c0, c1, c2: c0 + c1 * x + c2 * x**2, phi)
comp = rx.Comparison(d, omp + delta)
```

Expected behaviour:

- The prediction is `omp(x) + delta(x)` in physical space;
  `problem.names` lists `omp.params` then `phi`.
- `(omp + delta) | tf.scale(rho)` scales the sum; `(omp | tf.scale(rho)) + delta`
  scales only the model.
- A multiplicative, `x`-dependent correction is `omp * Model(g_fn, phi)`;
  an additive discrepancy in log space is exactly that.  `scale(rho)` is
  its constant special case.
- The GP of recipe 7 may be used alongside it.

## 9. Heavy tails

*A few points are gross outliers.  I do not want them to drag the fit; I
want a likelihood that widens instead of breaking.*

```python
nu = rx.Parameter("nu", bounds=(1.0, 100.0))          # uniform on the bounds
c = rx.Constraint([comp], likelihood=rx.StudentT(nu=nu))
```

Expected behaviour:

- Same covariance, different functional.  The Gaussian version puts the
  truth at several σ; the Student-t version covers it
  (`robust_likelihoods`), and the posterior of `nu` is small.
- The multivariate t applies one radial tail to the whole stacked residual
  of the constraint.  A comparison that needs Gaussian tails goes in its own
  constraint.
- `rx.Chi2()` drops the log-determinant for a pure chi-squared objective.
- The Student-t value is pinned to its closed form (`test_likelihood`).

## 10. Compare in log space

*Cross sections span orders of magnitude.  I want the Gaussian to live in
log space, with the model still written in physical units.*

```python
comp = rx.Comparison(d, omp, space=tf.log)
c = rx.Constraint([comp], terms=[T.noise(log_eps)], statistical=False)   # constant noise in log space
```

Expected behaviour:

- `comp.y == log(d.y)` and `comp.y_err == d.y_err / d.y` (delta method).  The
  prediction is transformed the same way at every evaluation, so it can
  never be double-transformed.
- A non-positive prediction gives `log_likelihood == -inf` and
  `chi2 == +inf`.  A non-positive datum on an active point is a compile
  error naming the dataset.
- `problem.log_jacobian()` is `-sum(log y)` over active points; add it to
  the log-evidence of the log-space fit before comparing with a
  linear-space fit of the same data.
- Constant noise in log space and fractional noise in linear space are
  different models with different evidences.  That is the point.

## 11. Hold out data and score it

*I want to fit below an angular cut and score the prediction above it, or
compare error models by their held-out predictive density.*

```python
fit  = c.masked_where(lambda x: x < cut)
held = fit.complement()
problem = rx.Problem([fit], priors=priors)
heldout = rx.Problem([held], priors=problem.priors)
lp = rx.diagnostics.heldout_log_predictive(heldout, samples)
score = rx.diagnostics.log_posterior_predictive(lp, logw=res.logwt)   # weighted for nested sampling
```

Expected behaviour:

- `fit` and `held` share every `Comparison`, `Term`, and `Parameter`; the two
  problems have identical `names`, so a chain from one scores the other.
- Active sets are disjoint, their union is every point, and
  `ll(fit) + ll(held) == ll(full)` for a block-local covariance.  When a term
  spans the split (a GP over several experiments), the held-out problem's
  own likelihood is the *marginal* of its rows; pass the fitted problem as
  `given=` to `heldout_log_predictive` / `predictive_draws` for the
  conditional `p(y_held | y_fit, theta)` under the full covariance (recipes
  28 and 30).
- Terms are authored once over all points; masking selects rows, it never
  rebuilds anything.

## 12. Temper the likelihood

*I have many points and worry the posterior is overconfident, or I want
a power posterior.*

```python
c = rx.Constraint([comp], weight=2 / n)
```

Expected behaviour:

- The log-likelihood of that constraint is multiplied by `weight`; the
  prior is never touched.  One knob, honoured by every driver.
- Empirical coverage of the posterior predictive (recipe 17) moves toward
  the nominal level as the weight decreases (`overconfidence`).

## 13. Declare priors

*I want each nuisance parameter to carry its own prior, the optical
potential to have a correlated multivariate normal prior, and everything
to work with nested sampling.*

```python
log_eps = rx.Parameter("log_eps", prior=stats.halfnorm(scale=1))            # marginal
nu      = rx.Parameter("nu", bounds=(1, 100))                               # uniform on bounds
V       = rx.Parameter("V", prior=stats.norm(50, 5), bounds=(30, 70))       # truncated marginal
problem = rx.Problem([c], priors=[(omp.params, stats.multivariate_normal(mu, cov))])   # joint block
```

Expected behaviour:

- Every slot is covered exactly once, or `Problem` raises naming the
  parameter (uncovered) or the pair of priors (double covered).
- `problem.prior_transform(u)` exists for all of the above: rescaled `ppf`
  for marginals, whitening for the multivariate normal.  It is finite at
  `u = 0` and `u = 1`.
- `problem.sample_prior(n)` gives an `(n, ndim)` array suitable as emcee's
  starting positions.

## 14. From an EXFOR measurement to a dataset

*I have an `exfor_tools` distribution in mb/sr, or as a ratio to
Rutherford, with its reported errors.  I want a dataset in the library's
units with nothing lost.*

```python
d = rx.from_measurement(m, reaction=reaction, quantity="dXS/dA")      # or "dXS/dRuth", "Ay"
d_ias = rx.from_measurement(m, reaction=reaction, ExIAS=Ex)           # (p,n) IAS channel
```

Expected behaviour:

- `y`, `y_err` and `offset_err` are divided by the unit conversion
  factor; `norm_err` (fractional) passes through untouched.  Angles are
  stored in radians.
- Requesting `dXS/dRuth` from a `dXS/dA` measurement (or the reverse)
  uses the Rutherford cross section on the data angles, a closed form of
  the kinematics; the factor is then a per-angle array.
- `d.meta` carries `reaction`, `Elab`, `quantity`, `k` (and `ExIAS`), which
  is everything a reaction model needs to bind.
- Incompatible units, or a quantity the measurement cannot be converted
  to, raise at conversion time.

## 15. Evaluate a reaction model on any grid

*I want the model on the data angles for the likelihood and on a fine
grid for plotting, with the solver set up once per grid.*

```python
omp = rx.reactions.ElasticXS("dXS/dA", central, spin_orbit, args_from_params, params,
                             lmax=20, wavelengths_beyond_range=2.0, zeros_per_node=5)
comp = rx.Comparison(d, omp)                                   # bound to d.x, d.meta
fine = omp.bind(np.deg2rad(np.linspace(0.5, 179.5, 200)), d.meta)
ys = [fine(*s[problem.columns(omp.params)]) for s in samples[::50]]
band = rx.predictive.predictive_band(ys, levels=(5, 50, 95))
```

Expected behaviour:

- The model owns the jitr workspace; the dataset never does.  Solver
  settings on the model reach jitr (`TestSolverSettingsForwarding`).
- The predictor is a pure function of the potential.  A compound-elastic
  contribution is subtracted from `d.y` before the dataset is built
  (recipe 20); there is no correction hook on the model.
- Cross sections are in b/sr; `dXS/dRuth` and `Ay` are dimensionless.
- `problem.columns(omp.params)` is the only chain bookkeeping a user does.

## 16. Drive the calibration with an external sampler

*I want to use emcee, dynesty, or the `black-box-bayes` CLI, and read the
chain back without positional arithmetic.*

```python
# emcee
p0 = problem.sample_prior(32)
sampler = emcee.EnsembleSampler(32, problem.ndim, problem.log_posterior)
samples = sampler.run_mcmc(p0, 5000) and sampler.get_chain(discard=1000, flat=True)

# dynesty
ns = dynesty.NestedSampler(problem.log_likelihood, problem.prior_transform, problem.ndim)
ns.run_nested(); samples = ns.results.samples_equal()
logz = rx.diagnostics.logz_summary(ns.results.logz[-1], ns.results.logzerr[-1])

# black-box-bayes
dill.dump(problem, open("problem.pkl", "wb"))        # posterior.py delegates six names to it
```

Expected behaviour:

- Chains from all three are `(n, ndim)` arrays in `problem.names` order.
  Every analysis function takes `(problem, samples)`.
- `log_posterior` evaluates the prior first and never calls the forward
  model when the prior is `-inf`.
- `dill.loads(dill.dumps(problem)).log_posterior(theta)` equals
  `problem.log_posterior(theta)`, workspaces included.
- `problem.NDIM`, `problem.parameter_names`, `problem.starting_location`,
  `problem.log_posterior_batch` are the bbb spellings of the same things.

## 17. Check the posterior predictive

*I want to know whether my error model is calibrated: do 68 % intervals
contain 68 % of the points, and how wide are they?*

```python
draws = rx.diagnostics.predictive_draws(problem, samples, constraint=0, n_rep=4)
cov = rx.diagnostics.coverage_curve(draws, problem.constraints[0].y[problem.constraints[0].active])
err = rx.diagnostics.coverage_error(draws, y_active)
width = rx.diagnostics.sharpness(draws, transform=np.exp)      # widths in physical space for a log fit
```

Expected behaviour:

- Draws are `ym(theta) + L z` on the active points in comparison space;
  `model_only=True` returns `ym(theta)` and assembles no covariance.
- Coverage is near nominal for a correct error model and clearly below
  it for an overconfident one.
- `logz_summary` reports the max of the replicate half-range and the
  sampler's own error; `compare_logz` returns `"tie"` unless the
  difference exceeds `sigma * hypot(err_a, err_b)`.

## 18. Compare error models by evidence

*The α+Ca study: several error models for data without reported errors.
I want the evidence for each, comparable across comparison spaces.*

```python
models = {
    "L0": rx.Constraint([comp_log], terms=[T.noise(log_eps)], statistical=False),
    "E0": rx.Constraint([comp_lin], terms=[T.noise_fraction(log_eps)], statistical=False),
    "L2y": rx.Constraint([comp_log], terms=[T.noise(log_eps), T.normalization(log_sys)], statistical=False),
    "Lgp": rx.Constraint([comp_log], terms=[T.noise(log_eps), gp], statistical=False),
    "L0t": rx.Constraint([comp_log], terms=[T.noise(log_eps)], statistical=False, likelihood=rx.StudentT()),
}
# the labels are defined in the table below this block
logz = {}
for name, c in models.items():
    p = rx.Problem([c.masked_where(lambda x: x < cut)], priors=priors)
    res = run_dynesty(p)
    logz[name] = rx.diagnostics.logz_summary(res.logz[-1] + p.log_jacobian(), res.logzerr[-1])
verdict = rx.diagnostics.compare_logz(logz["Lgp"], logz["L0"])
```

The error-model ladder of the study, in the words a reader needs.  All
forms are covariances of the residual in log space unless stated; `theta`
is the scattering angle in radians and `u = theta / pi`.

| label | error model |
|---|---|
| `L0` | constant noise: `sigma = err` on every point |
| `E0` | fractional noise in linear space: `sigma_i = err * ym_i` |
| `L1` | noise growing with angle: `sigma(theta) = err * exp(slope * u)` |
| `L2` | `L0` plus one correlated mode proportional to angle, `sys * u` |
| `L2n` | `L0` plus a free correlated offset mode, `sys * 1` |
| `L2y` | `L0` plus a free correlated normalisation mode, `sys * ym` |
| `L12` | `L1` plus the angle mode of `L2` |
| `Lgp` | `L0` plus a Matérn(5/2) Gaussian process in `u` with constant amplitude |
| `Lgpn` | `L0` plus the Gaussian process with an angle-growing amplitude |
| `LKp` | noise, an offset mode, and an RBF Gaussian process in momentum transfer `q = 2 k sin(theta/2)` with amplitude `A q^(r/2)` |
| `L0t` | `L0` under a Student-t likelihood |

The test suite builds every covariance row of this table against a
hand-built dense matrix, and a test compares the table above with the
legend the tests carry, so the two cannot drift.

Expected behaviour:

- The same `log_eps` object is reused across models without conflict:
  each `Problem` is compiled independently.
- Adding `log_jacobian` makes the log-space and linear-space evidences
  comparable.
- Every term list in the table reproduces the corresponding hand-built
  covariance in `TestStudyForms`.

## 19. Bring my own covariance or term

*I have a full covariance matrix from a correlated measurement, or a noise
model no helper expresses.*

```python
fixed = rx.Term(C, on=comp)                                       # any symmetric PD matrix
stat  = rx.Term(sig, kind="diag", on=comp)                        # a fixed std-dev vector
custom = rx.Term(lambda c, e, l: np.exp(e) * np.exp(l * c.x / np.pi), (log_e, slope), kind="diag")
```

Expected behaviour:

- A plain array is a fixed contribution, factored once.  Shape and
  symmetry are checked at construction against the term's `on`.
- A callable sees a `TermContext` with `x` (through `coords`), `y`, `ym`,
  `len(c)`, per-point `c.meta(key)`, and, for a term spanning several
  comparisons, `c.segments`/`c.labels`/`c.split(a)` giving the rows of
  each comparison in the gathered stack; it returns a vector for
  `diag`/`mode` or a matrix.
- Fitting correlated data with the correct `Term(C)` instead of its
  diagonal is the difference between an honest and an overconfident
  posterior (`normalization_inference` gallery).

## 20. Preprocess instead of asking for a feature

*I want to mean-subtract, standardise, or project both data and model onto
principal components of a prior predictive ensemble.*

```python
mu = ens.mean(0); A = np.linalg.svd(ens - mu, full_matrices=False)[2][:k]
d_pc = rx.Dataset(np.arange(k), A @ (d.y - mu), np.zeros(k), label=f"{d.label} PC")
native = omp.bind(d.x, d.meta)
proj = rx.Model(lambda x, *theta: A @ (native(*theta) - mu), omp.params)
stat = rx.Term(A @ np.diag(d.y_err**2) @ A.T, on=d_pc)
c = rx.Constraint([rx.Comparison(d_pc, proj)], terms=[stat, T.noise(log_eps, on=d_pc)], statistical=False)
```

Expected behaviour:

- `Dataset.x` is opaque to the library; a model may ignore its `x` and
  close over a predictor on another grid; `y_err` may be zero when
  `statistical=False`.  None of these is "fixed" later.
- Subtracting a known contribution from the data, such as a compound-
  elastic cross section, is the same pattern: `replace(d, y=d.y - cn)`.
  A reported *fractional* normalisation error refers to the measured
  value, so build that mode from the unsubtracted prediction if it
  matters; the offset error is unaffected.
- Pointwise maps (mean subtraction, standardisation) can equally be a
  `space=` transform built from closed-over arrays, which keeps the delta
  method and `log_jacobian` correct.
- Terms that scale with the native prediction see the projected `ym`;
  they must call the native predictor themselves if they need it.
- A projected fit's evidence is not comparable to the unprojected one.

## 21. Get a useful error, not a `LinAlgError`

*My measurement reports no statistical error and I forgot to add any
term.*

```python
d = rx.from_measurement(m_without_errors, reaction=reaction, quantity="dXS/dA")
rx.Problem([rx.Constraint([rx.Comparison(d, omp)])])
# ValueError: covariance of constraint 0 is singular on comparison 'E1234-002': the dataset
# reports zero statistical error and no term covers its points.  Add
# comp.reported_terms(), a noise term, or a fixed Term; or set statistical=False and
# compose the covariance explicitly.
```

Expected behaviour:

- Raised by `Problem`, before any sampler runs, naming the comparison by its
  label.  A parametric covariance is not checked eagerly (it may be fine
  at some parameter values).
- The same compile step reports duplicate parameter names, an uncovered
  prior slot, an `on=` outside its constraint, and a non-finite
  comparison-space value on an active point.

## 22. Share hyperparameters across datasets, with values that depend on the dataset

*I have elastic data at several energies.  I want one GP discrepancy per
dataset with a common length scale and an amplitude that runs with energy,
so that two shared parameters describe every dataset.*

```python
log_A0 = rx.Parameter("log_A0", prior=stats.norm(0, 2))
p      = rx.Parameter("p", prior=stats.norm(0, 1))
ell    = rx.Parameter("gp_length", prior=stats.norm(0, 1))

amp = lambda c, lA, p: np.exp(lA) * (c.meta("Elab") / 50.0) ** p
terms = [T.kernel(Matern(1.0, nu=2.5), on=comp, params=[ell],
                  amplitude=amp, amplitude_params=(log_A0, p)) for comp in comps]
c = rx.Constraint(comps, terms=terms)
```

Expected behaviour:

- `problem.names` contains one `log_A0`, one `p`, one `gp_length`, however
  many comparisons there are.  Sharing is by object; the loop reuses the same
  three.
- `c.meta("Elab")` is the comparison's energy broadcast to every point of the
  support, so the amplitude function is written once and works on any
  comparison.  On a term spanning comparisons it is the per-point concatenation.
- Without `params=[ell]`, each `kernel` call would derive its own length
  scale, one per dataset.  That is also a legitimate model; it is just not
  this one.
- The comparisons stay uncorrelated, so the constraint is on the structured
  path.

## 23. A discrepancy correlated across energies and angles

*I believe the model's defect varies smoothly in both energy and angle.  I
want one GP over (E, θ) that correlates the datasets at different
energies.*

```python
kE, kθ = RBF(10.0), Matern(0.3, nu=2.5)
lE, lθ, log_A = (rx.Parameter(n, prior=stats.norm(0, 1)) for n in ("log_lE", "log_ltheta", "log_A"))

def fn(c, lE, lθ, lA):
    E, θ = c.meta("Elab")[:, None], c.x[:, None]
    K = kE.clone_with_theta([lE])(E) * kθ.clone_with_theta([lθ])(θ)     # separable product
    return np.exp(2 * lA) * K + 1e-10 * np.eye(len(c))

md = rx.Term(fn, (lE, lθ, log_A), kind="matrix", on=comps)
c = rx.Constraint(comps, terms=[md])
```

Expected behaviour:

- The term spans comparisons, so the constraint's covariance has non-zero
  off-diagonal blocks between energies and takes the dense path.  Cost
  grows as the cube of the total number of points; per-comparison GPs with a
  linked amplitude (recipe 22) are the cheap alternative when
  cross-energy correlation is not needed.
- Any kernel built from `c.meta` and `c.x` is allowed; the product of two
  one-dimensional kernels is the factorised form.  There is no Kronecker
  speed-up because datasets share no angle grid.
- The dense matrix equals the hand-built product kernel on the stacked
  `(E, θ)` inputs.

## 24. Per-dataset parameters drawn from a sampled hyperprior

*Each dataset has its own normalisation, and I want to learn how spread
out those normalisations are, rather than fix the spread.*

```python
rhos = [rx.Parameter(f"log_rho_{i}") for i in range(len(comps))]    # no marginal: the joint covers them
log_tau = rx.Parameter("log_tau")

class RhoHierarchy:
    def logpdf(self, v):                       # v ordered as (rhos..., log_tau)
        *r, lt = v
        return (stats.norm(0, np.exp(lt)).logpdf(r).sum()
                + stats.halfnorm(scale=0.3).logpdf(np.exp(lt)) + lt)    # hyperprior on tau, with Jacobian
    def prior_transform(self, u):              # hyperparameter first, children given it
        lt = np.log(stats.halfnorm(scale=0.3).ppf(u[-1]))
        return np.append(stats.norm(0, np.exp(lt)).ppf(u[:-1]), lt)

comps = [rx.Comparison(d_i, omp | tf.scale(rho_i)) for d_i, rho_i in zip(datasets, rhos)]
problem = rx.Problem([rx.Constraint(comps)], priors=[(rhos + [log_tau], RhoHierarchy())])
```

Expected behaviour:

- `log_tau` is an ordinary column of the chain.  Its posterior is the
  learned spread of the normalisations; the `rho_i` shrink toward zero
  when `tau` is small.
- Forgetting the joint block is a compile error: the `rho_i` carry no
  marginal and have infinite bounds, so no prior covers them.
- Nested sampling works because the comparison supplies its own unit-cube map,
  drawing the hyperparameter before the children.
- The same mechanism gives per-dataset GP amplitudes with a shared spread,
  or any other partially pooled parameter.

## 25. SafeBayes: learn the tempering exponent

*I suspect my model is misspecified and want the tempering exponent η
chosen by the data rather than by hand, following Grünwald's SafeBayes
(arXiv:1412.3730): minimise the prequential log-loss of the η-generalised
posterior over prefixes of the data.*

```python
from dataclasses import replace
order = rng.permutation(n)                                   # one ordering; average a few
prefix = lambda i: [np.isin(np.arange(n), order[:i])]
etas = [1.0, 0.5, 0.25, 0.125]
loss = dict.fromkeys(etas, 0.0)

for i in range(n0, n):
    before = rx.Problem([c.masked(prefix(i))], priors)       # untempered, for scoring
    after  = rx.Problem([c.masked(prefix(i + 1))], priors)
    for eta in etas:
        post = rx.Problem([replace(c.masked(prefix(i)), weight=eta)], priors)
        samples = run(post)                                   # emcee or dynesty
        ll_i = np.array([after.log_likelihood(s) - before.log_likelihood(s) for s in samples])
        loss[eta] -= ll_i.mean()                              # R-log-loss

eta_hat = min(loss, key=loss.get)
final = rx.Problem([replace(c, weight=eta_hat)], priors)
```

Expected behaviour:

- `replace(c, weight=eta)` and `c.masked(...)` share every comparison, term and
  parameter with `c`, so no workspace is rebuilt across the `n × |η|`
  problems and every problem has the same `names`.
- `after.log_likelihood(θ) − before.log_likelihood(θ)` is
  `log p(y_i | y_<i, θ)` exactly, also under a correlated covariance.  For
  a diagonal covariance it reduces to the single-point marginal, which is
  the paper's setting.
- With dynesty one nested run per prefix serves every η: the η-posterior
  weights over the same samples are `logvol + η·logl`.  The cost is then
  `n` runs, not `n × |η|`.
- `weight` is a float by type.  Putting a prior on η and sampling it is
  not SafeBayes and is not coherent, since `p(D|θ)^η` is not a normalised
  likelihood in η; the design refuses it.
- The same loop at fixed η with the held-out score of recipe 11 is the
  cheaper "choose η by held-out predictive"; SafeBayes is its principled
  form.

---

The recipes below come from a survey of the calibration literature, general
and nuclear.  Each is a pure user-level pattern over the skeleton as it
stands; none needs a library addition.  They are ordered by how often the
pattern appears in nuclear-physics calibration practice.  A closing section
lists what the API does not express.

## 26. Unaccounted-for model error per data type (KDUQ)

*I am calibrating a global optical potential to many datasets of several
observable types.  I want one fractional "unaccounted-for" uncertainty per
type, sampled with the potential, added in quadrature to the reported
errors and scaled with the average of datum and prediction.*

```python
delta = {t: rx.Parameter(f"delta_{t}", prior=stats.halfnorm(scale=s0[t])) for t in ("dxs", "ay", "sig_tot")}
comps = [rx.Comparison(d, omp_for(d)) for d in datasets]
terms = [T.model_error(delta[d.meta["type"]], averaging=True, log=False, on=comp)
         for d, comp in zip(datasets, comps)]   # log=False: delta is the fraction itself
c = rx.Constraint(comps, terms=terms)          # statistical=True: reported errors are a floor

# KDUQ additionally scales the whole log-likelihood by k/N ("democratic"), or
# each data type's share by k/(n_types N_t) ("federal").  That is tempering:
c_dem = rx.Constraint(comps, terms=terms, weight=n_params / n_data)
c_fed = [rx.Constraint([comp for comp in comps if comp.data.meta["type"] == t],
                       terms=[tt for tt in terms if tt.on.data.meta["type"] == t],
                       weight=n_params / (len(types) * n_pts[t]))
         for t in types]                                     # one constraint per type, one delta_t each
```

Expected behaviour:

- One `delta_t` column per type in `problem.names`, shared by every comparison of
  that type through the same `Parameter` object.
- `Σ_ii = y_err_i² + (δ_T (y_i + ym_i) / 2)²`.  Because the term scales
  with the prediction it changes with the model parameters; that is
  allowed.
- The democratic and federal variants are `Constraint(weight=)`.  The
  published text describes them as a rescaling of the covariance, which
  would also change `logdet`; the authors have confirmed that what was
  actually done is a scaling of the whole log-likelihood by `k/N`, which
  is tempering.  The federal form is one constraint per data type, holding
  every comparison of that type, with its own weight; each `delta_t` lives in
  its type's constraint, and only the potential parameters are common to
  all constraints, as in any multi-constraint problem.
- Reported uncertainties are respected, then augmented; a dataset's known
  normalisation systematic can still be added as a mode (recipe 3).

Reference: Pruitt, Escher, Rahman, *Uncertainty-quantified phenomenological
optical potentials for single-nucleon scattering*, Phys. Rev. C 107, 014602
(2023), arXiv:2211.07741.

## 27. Peelle's Pertinent Puzzle: normalise the prediction, never the data

*My datasets carry a common fractional normalisation error and I want the
fit not to be biased low.*

```python
# right: the mode is built from the prediction (this is what normalization() does)
c_ok = rx.Constraint([comp], terms=[T.normalization(magnitude=d.norm_err, on=comp)])

# wrong on purpose, to see the bias: the mode built from the data
c_bad = rx.Constraint([comp], terms=[rx.Term(lambda c: d.norm_err * c.y, kind="mode", on=comp, constant=True)])

# t0 variant: a fixed mode from a reference prediction, refit once
t0 = comp.predictor(*theta_hat)
c_t0 = rx.Constraint([comp], terms=[rx.Term(d.norm_err * comp.space(t0), kind="mode", on=comp)])
```

Expected behaviour:

- With the data-built mode, a fit of a constant `t` to `n` points with
  statistical error `σ` and fractional normalisation error `s` has the
  exact closed form `t = ȳ / (1 + (s/σ)² Σ(yᵢ − ȳ)²)`: the fluctuations
  feed back into the covariance and pull the estimate low, by
  `1 / (1 + (n − 1) s²)` in leading-order expectation, independent of `σ`
  (two points at 1.5 and 1.0 with `s = 0.2` fit *below both*).  This is
  D'Agostini's bias and the origin of Peelle's Pertinent Puzzle.  The
  prediction-built mode removes that bias; what remains is a smaller pull
  from the log-determinant, which grows with the fitted value, and the
  `t0` refit removes that too.  An additive offset mode has no such bias
  either way.
- `normalization()` reads `c.ym`, so the default spelling is the safe one.
  A free `log_eta` (recipe 4) also multiplies the prediction.
- The `t0` mode makes the covariance constant, so it is factored once;
  one refit after a zeroth-order fit suffices in practice.

References: D'Agostini, *On the use of the covariance matrix to fit
correlated data*, Nucl. Instrum. Meth. A 346, 306 (1994); Frühwirth,
Neudecker, Leeb, *Peelle's Pertinent Puzzle and its solution*, EPJ Web Conf.
27, 00008 (2012); Ball et al. (NNPDF), *Fitting parton distribution data
with multiplicative normalization uncertainties*, JHEP 05 (2010) 075.

## 28. Stacking by leave-one-dataset-out

*Evidence weights assume the true model is among my candidates.  I would
rather weight models by how well they predict each dataset when it is left
out.*

```python
def loo_scores(model):
    out = []
    for i in range(len(comps)):
        fit_c = c.masked([np.full(comp.data.y.shape, j != i) for j, comp in enumerate(comps)])
        fit, held = rx.Problem([fit_c], priors), rx.Problem([fit_c.complement()], priors)
        s = run(fit)
        out.append(rx.diagnostics.log_posterior_predictive(rx.diagnostics.heldout_log_predictive(held, s)))
    return np.array(out)                                   # one log score per held-out dataset

S = np.stack([loo_scores(m) for m in models])              # (n_models, n_datasets)
w = maximise(lambda w: np.sum(logsumexp(np.log(w)[:, None] + S, axis=0)), simplex)
```

Expected behaviour:

- Held-out log densities are joint over the held-out comparison and exact
  under a correlated covariance (`given=fit` when a term spans the fitted
  and the held-out comparisons); PSIS-LOO per point is not available
  without per-point likelihood factors (closing section).
- Stacking weights need not sum to the evidence weights; in the M-open
  setting they are the ones to prefer.
- The cost is one fit per dataset per model; for a diagonal covariance and
  many datasets the importance-sampling shortcut of Vehtari et al. is the
  cheaper path, implemented from `problem.predict` and `y_err` by the user.

References: Yao, Vehtari, Simpson, Gelman, *Using stacking to average
Bayesian predictive distributions*, Bayesian Analysis 13, 917 (2018);
Vehtari, Gelman, Gabry, *Practical Bayesian model evaluation using
leave-one-out cross-validation and WAIC*, Stat. Comput. 27, 1413 (2017).

## 29. Cut (modular) posterior by multiple imputation

*One module of my model, say a systematic-error parameter or a GP
hyperparameter, should be learned from its own data only and not be
contaminated by the primary data, which I trust less.*

```python
stage1 = rx.Problem([rx.Constraint([rx.Comparison(d_aux, aux_model)])], priors_aux)     # phi from Z only
phis = run(stage1)[::thin][:T]

pooled = []
for phi in phis:
    model_t = rx.Model(functools.partial(f, phi=phi), theta_params)                  # phi fixed
    stage2 = rx.Problem([rx.Constraint([rx.Comparison(d, model_t)], terms=terms_at(phi))], priors_theta)
    pooled.append(run(stage2))
samples = np.concatenate(pooled)

# partial feedback instead of a cut: per-module powers
c_aux = rx.Constraint([rx.Comparison(d_aux, aux_model)], weight=gamma1)
c_pri = rx.Constraint([rx.Comparison(d, full_model)], weight=gamma2)
```

Expected behaviour:

- The marginal for `phi` equals its stage 1 posterior; the primary data
  never update it.  Uncertainty in `phi` still propagates into `theta`.
- The cut distribution is not a single posterior, so it cannot be handed to
  emcee or dynesty as one `log_posterior`; the two-stage loop is the
  definition, not an approximation.  A single alternating chain converges to
  updater-dependent limits and is not a substitute.
- The power-weighted alternative is one `Problem` with two weights and is
  a legitimate posterior; it gives partial feedback control.
- Bayarri's "modular MLE" is the same shape with T = 1: fix GP
  hyperparameters at their stage 1 estimates and pass them as constants.

References: Plummer, *Cuts in Bayesian graphical models*, Stat. Comput. 25,
37 (2015); Jacob, Murray, Holmes, Robert, *Better together? Statistical
learning in models made of modules*, arXiv:1708.08719; Bayarri et al., *A
framework for validation of computer models*, Technometrics 49, 138 (2007).

## 30. Leave-one-experiment-out prediction

*I want to know whether the calibrated model, with its discrepancy,
predicts an experiment it was not fit to, and with what tolerance.*

```python
for i, comp in enumerate(comps):
    fit  = c.masked([np.full(other.data.y.shape, j != i) for j, other in enumerate(comps)])
    p    = rx.Problem([fit], priors)
    s    = run(p)
    held = rx.Problem([fit.complement()], priors)
    draws = rx.diagnostics.predictive_draws(held, s, n_rep=4, given=p)   # conditional on the fit
    tol = np.percentile(np.abs(draws - draws.mean(0)), 90, axis=0)      # tolerance bound per point
    cov = rx.diagnostics.coverage_curve(draws, held.constraints[0].y[held.constraints[0].active])
```

Expected behaviour:

- The held-out comparison's covariance terms are the same objects as in the fit;
  a GP discrepancy conditioned on the other experiments carries into the
  prediction through `predictive_draws(..., given=p)`, which draws from
  `p(y_held | y_fit, theta)` under the full covariance.  Without `given=` the
  draws use the marginal block, which forgets what the fit taught the GP.
- Coverage on the held-out experiment is the honest check; in-sample
  coverage is not.
- Tolerance bounds are empirical percentiles of the draws, componentwise.

References: Higdon, Gattiker, Williams, Rightley, *Computer model
calibration using high-dimensional output*, J. Am. Stat. Assoc. 103, 570
(2008); Bayarri et al., Technometrics 49, 138 (2007).

## 31. Simulation-based calibration of the sampler

*Before trusting a chain, I want to check that the sampler recovers
parameters drawn from the prior when the data are simulated from the
model.*

```python
ranks = []
for theta0 in problem.sample_prior(n_sims, rng):
    y_sim = rx.diagnostics.predictive_draws(problem, theta0[None], n_rep=1)[0]
    d_sim = dataclasses.replace(d, y=comp.space.inverse(y_sim))            # back to physical units
    p_sim = rx.Problem([rx.Constraint([rx.Comparison(d_sim, model)], terms=terms)], priors)
    s = run(p_sim)[::thin][:L]
    ranks.append((s < theta0).sum(0))                                 # one rank per column
ranks = np.array(ranks)                                               # uniform on 0..L if correct
```

Expected behaviour:

- Every rank histogram is uniform when the sampler is correct.  An inverted
  U means the computed posterior is too wide, a U too narrow, a skew a
  bias.
- Chains must be thinned to roughly independent draws first, or spurious
  boundary spikes appear.
- `comp.space.inverse` is defined for every built-in parameter-free space
  (`identity`, `log`, `exp`), so the simulated data go back to physical
  units regardless of the comparison space.
- SBC validates the computation under the assumed model; it says nothing
  about whether the model fits real data; that is the posterior predictive
  coverage check of recipe 17.

Reference: Talts, Betancourt, Simpson, Vehtari, Gelman, *Validating
Bayesian inference algorithms with simulation-based calibration*,
arXiv:1804.06788.

## 32. Emulator as the model, emulator variance as a term

*My model is too expensive to run in the chain.  I have a GP or PCA
emulator trained on a design of runs, and I want its predictive variance
in the likelihood.*

```python
emu = rx.Model(lambda x, *theta: emulator.mean(theta), params)             # x ignored: the grid is the design's
emu_var = rx.Term(lambda c, *theta: np.sqrt(emulator.var(theta)), params, kind="diag", on=comp)
c = rx.Constraint([rx.Comparison(d, emu)], terms=[emu_var, T.noise(log_eps)])
```

Expected behaviour:

- The term declares the *same* `Parameter` objects as the model, so it
  receives the current `theta` and can evaluate the emulator variance
  there.  No special mechanism.
- Bayarri et al. recommend fixing emulator hyperparameters at their
  estimates from the design runs (recipe 29 with T = 1) because emulator
  uncertainty is usually dominated by calibration and bias uncertainty.
- The only full-posterior nuclear EDF calibration to 2015 replaced the
  code by a GP response surface exactly this way.

References: Higdon et al., J. Am. Stat. Assoc. 103, 570 (2008); McDonnell,
Schunck, Higdon, Sarich, Wild, Nazarewicz, Phys. Rev. Lett. 114, 122501
(2015); Schunck et al., *Uncertainty quantification and propagation in
nuclear density functional theory*, Eur. Phys. J. A 51, 169 (2015); Bayarri
et al., Technometrics 49, 138 (2007).

## 33. MAP and Laplace approximation

*I want a quick Gaussian approximation to the posterior, and to know when
it is good enough.*

```python
from scipy.optimize import minimize
res = minimize(lambda t: -problem.log_posterior(t), problem.sample_prior(1)[0], method="L-BFGS-B",
               bounds=problem.bounds)
H = numerical_hessian(lambda t: -problem.log_posterior(t), res.x)
cov = np.linalg.inv(H)                                   # Laplace covariance
```

Expected behaviour:

- For a near-Gaussian posterior (a five-parameter optical potential with a
  flat prior) the Laplace covariance reproduces the emcee uncertainties.
  For a six-parameter potential the posterior is banana-shaped and it does
  not; compare against a chain before reporting.
- This is the dominant uncertainty-propagation practice in nuclear DFT
  (inverse Hessian at the optimum, propagated linearly).
- `problem.bounds` feeds the optimiser; the prior handles the rest.

References: Pruitt, Lovell, Hebborn, Nunes, *The role of the likelihood for
elastic scattering uncertainty quantification*, arXiv:2403.00753; Schunck
et al., Eur. Phys. J. A 51, 169 (2015).

## 34. Global error scale factor and unrecognised sources of uncertainty

*Repeated measurements scatter more than their stated errors.  I want a
global scale on the reported errors, or a fully correlated unknown
component per experimental technique.*

```python
log_s = rx.Parameter("log_s", prior=stats.norm(0, 0.5))
scaled = rx.Term(lambda c, ls: np.exp(ls) * comp.y_err, (log_s,), kind="diag", on=comp)   # closes over the comparison's errors
c = rx.Constraint([comp], terms=[scaled], statistical=False)               # s multiplies every stated error

# USU: one unknown, fully correlated component per technique, shared across its datasets
log_delta = {tech: rx.Parameter(f"log_usu_{tech}") for tech in techniques}
usu = [T.offset(parameter=log_delta[tech], on=[comp for comp in comps if comp.data.meta["technique"] == tech])
       for tech in techniques]
```

Expected behaviour:

- Under a Student-t likelihood only a modest scale is needed where least
  squares needs a large one, because the tail absorbs the outliers.
- A global Birge-type rescaling inflates every point equally and is judged
  poor evaluation practice; the targeted USU component is preferred, added
  only when other explanations are exhausted.
- A USU component inside the fit shifts the evaluated means, not only the
  widths, whenever more than one quantity is evaluated.  It must be in the
  covariance, not added afterwards, which is why it is a sampled term here.
- With `statistical=False` the scaled term *is* the diagonal; it closes
  over `comp.y_err`, the comparison-space errors, so it is right under a
  `space` transform too.

References: Hanson, *Lessons about likelihood functions from nuclear
physics*, AIP Conf. Proc. 954 (2007), arXiv:0712.0021; Capote et al.,
*Unrecognized sources of uncertainties (USU) in experimental nuclear data*,
Nucl. Data Sheets 163, 191 (2020), arXiv:1911.01825.

## 35. Energy-dependent parameters and per-comparison model instances

*A potential depth depends on energy through a few coefficients I want to
share across datasets at different energies.*

```python
V0, V1 = rx.Parameter("V0", prior=stats.norm(50, 5)), rx.Parameter("V1", prior=stats.norm(-0.3, 0.1))

def depth_model(E):
    return rx.Model(lambda x, v0, v1, *rest: potential(x, V=v0 + v1 * E, *rest), [V0, V1, *rest_params])

comps = [rx.Comparison(d, depth_model(d.meta["Elab"])) for d in datasets]
```

Expected behaviour:

- One model instance per comparison, all sharing the coefficient objects, so
  `problem.names` has one `V0` and one `V1`.
- Energy enters by closure at construction; nothing in the library reads
  it.  The same pattern gives energy-dependent systematic errors in a term
  through `c.meta("Elab")` (recipe 22).
- A smooth energy dependence with more freedom is a discrepancy on a basis
  (recipe 36) or a GP over energy (recipe 23).

References: Schnabel, Capote, Koning, Brown, *Nuclear data evaluation with
Bayesian networks*, arXiv:2110.10322; Pruitt, Escher, Rahman, Phys. Rev. C
107, 014602 (2023).

## 36. Discrepancy on a physically constrained basis

*I know the shape the model defect can take, say a few Legendre modes in
angle, and want the discrepancy restricted to that basis.*

```python
from scipy.special import eval_legendre
modes = [T.systematic(rx.Parameter(f"log_s{k}", prior=stats.norm(-3, 1)),
                      basis=lambda c, k=k: eval_legendre(k, np.cos(c.x)), on=comp) for k in range(1, 4)]
c_marg = rx.Constraint([comp], terms=modes)                        # marginalised: rank-3 covariance

coeffs = [rx.Parameter(f"c{k}", prior=stats.norm(0, 0.1)) for k in range(1, 4)]
delta = rx.Model(lambda x, *cs: sum(ck * eval_legendre(k, np.cos(x)) for k, ck in enumerate(cs, 1)), coeffs)
c_samp = rx.Constraint([rx.Comparison(d, omp + delta)])              # sampled: mean correction
```

Expected behaviour:

- Each mode is one rank-one term; the covariance is low rank and stays on
  the structured path.
- The sampled form gives the coefficients' posterior directly; the
  marginalised form gives their scales.  Both confound with the model
  parameters if the basis overlaps the model's own response, so put a real
  prior on the amplitudes.
- Process-convolution and kernel bases are the same pattern with a
  different `basis`.

Reference: Higdon, Gattiker, Williams, Rightley, J. Am. Stat. Assoc. 103,
570 (2008).

## 37. Correlated normalisations between quantities of one experiment

*One experiment reports several physical quantities, each measured one or
more times, all multiplied by normalisations that were themselves measured
with correlated uncertainties.  I want the covariance across the quantities
built so that it does not bias the evaluation.*

This is the two-and-more-dimensional Peelle's Pertinent Puzzle of Neudecker,
Frühwirth, Kawano and Leeb (reference below).  Quantity `i` is
`rho_i = alpha_i * eta_i`; `alpha_i` is measured as `q_i` (once or several
times, independent errors `sigma_i`) and `eta_i` as `N_i`, the `N_i` sharing
a covariance `B` with correlation `c`.  The reported data are the products
`r_i = q_i N_i`.

```python
# one comparison per quantity; the model is the quantity itself
rhos = [rx.Parameter(f"rho_{i}", prior=stats.norm(r_i.mean(), 10.0)) for i in range(n)]
comps = [rx.Comparison(rx.Dataset(np.full(len(r_i), i), r_i, N_i * sigma_i, label=f"q{i}"),
                       rx.Model(lambda x, rho: np.full(len(x), rho), [rho_i]))
         for i, (r_i, rho_i) in enumerate(zip(products, rhos))]
frac = sigma_N / N                                  # fractional normalisation errors
corr = np.array([[1, c], [c, 1]])                   # the correlation matrix of the N_i

def normalisations(c):                              # C_I: built from the prediction
    u = np.concatenate([f * ym for f, ym in zip(frac, c.split(c.ym))])
    which = np.concatenate([np.full(s.stop - s.start, k) for k, s in enumerate(c.segments)])
    return np.outer(u, u) * corr[np.ix_(which, which)]

c_I = rx.Constraint(comps, terms=[rx.Term(normalisations, kind="matrix", on=comps)])
c_F = rx.Constraint(comps, terms=[rx.Term(normalisations_from(y), kind="matrix", on=comps)])  # Peelle: from the data
```

Expected behaviour:

- One `Constraint`, one `matrix` term spanning every comparison.  A
  spanning term sees the gathered stack, so the term reads its per-quantity
  pieces through `c.segments` / `c.split` and pairs them with the
  normalisation correlation matrix.
- With the covariance built from the *data* (`C_F`, eq. 11 of the
  reference) the posterior mean under a flat prior is the generalised
  least-squares solution and is biased low: `<rho_1>_F = qbar_1 N_1 / (1 +
  xi)` with `xi = (q_1 - q_1')^2 sigma_N1^2 var(alpha_1) / (N_1^2 sigma_1^2
  sigma_1'^2)`, and `<rho_2>_F` is pulled down through `c` even though
  `alpha_2` was measured once; the variances and the covariance are
  deflated in their normalisation parts (eqs. 13-17).  The fast tier pins
  these closed forms exactly.
- With the covariance built from the *estimate* (`C_I`, eq. 12: the
  weighted means, in rxmc a constant term built from a first estimate and
  refit, recipe 27) the means are `qbar_i N_i` and the variances
  `var(alpha_i) N_i^2 + sigma_Ni^2 qbar_i^2`, with covariance
  `c qbar_1 qbar_2 sigma_N1 sigma_N2` (eqs. 18-22): no puzzle.
- The *live* term reading `c.ym` is the generative model's marginal
  likelihood, not `C_I`: its covariance grows with the prediction, so the
  log-determinant pulls the mode below the exact values (5 % in the
  two-quantity case, 9 % in the five-quantity one, against 23 % and about
  30 % for `C_F`), and under a flat prior the `1 / rho` tail pulls the mean
  above them.  A proper prior on the quantities, or the refit, removes the pull.
- The five-quantity numerical study of the reference (its Table I: `q_i`
  = {1.0, 1.5}, {1.8}, {2.2, 2.4}, {1.9, 1.5}, {1.4, 1.2}; `N_i` = 1, 1.1,
  1.25, 1.15, 1.05; `sigma_i = 0.1 q_i`, `sigma_Ni = 0.2 N_i`, `c = 0.8`)
  reproduces its Fig. 1: `C_F` gives lower means and smaller standard
  deviations on every lattice point, `C_I` agrees with the exact values
  (both exact in the fast tier, being generalised least squares).  The
  notebook `correlated_observations` recreates the figure.
- Analysing powers are *not* an instance of this recipe: a ratio of cross
  sections has a fixed normalisation, so nothing correlated can be inferred
  for it.  The real-data case of the reference (`237Np(n,f)` measured
  relative to `235U(n,f)` by three experiments, converted with the standard
  and its covariance) has the same structure with the standard's covariance
  as `B`.

Reference: Neudecker, Frühwirth, Kawano, Leeb, *Adequate treatment of
correlated experimental data in nuclear data evaluations avoiding Peelle's
Pertinent Puzzle*, Nucl. Data Sheets 118, 364 (2014).

## 38. The classic normal hierarchical model (eight schools)

*Several groups each report an estimate `y_j` with a known standard error
`σ_j`.  I believe the group effects `θ_j` are drawn from a common
distribution `N(μ, τ²)` and want to learn `μ`, `τ`, and the shrunken
`θ_j`.*

```python
d = rx.Dataset(x=np.arange(J), y=estimates, y_err=std_errors)          # x is the group index
mu      = rx.Parameter("mu", prior=stats.norm(0, 25))
log_tau = rx.Parameter("log_tau", prior=stats.norm(1, 1))

# 1. marginalised: integrate the theta_j out; y_j ~ N(mu, sigma_j^2 + tau^2)
c_marg = rx.Constraint([rx.Comparison(d, rx.Model(lambda x, mu: np.full(len(x), mu), [mu]))],
                       terms=[T.noise(log_tau)])
# theta_j afterwards, from the conditional normal at each draw (mu, tau):
#   mean = (y_j / s_j^2 + mu / tau^2) / (1 / s_j^2 + 1 / tau^2), var = 1 / (1 / s_j^2 + 1 / tau^2)

# 2. non-centred: theta_j = mu + tau * eta_j, eta_j ~ N(0, 1); marginal priors only
etas = [rx.Parameter(f"eta_{j}", prior=stats.norm(0, 1)) for j in range(J)]
school = rx.Model(lambda x, mu, lt, *eta: mu + np.exp(lt) * np.asarray(eta), [mu, log_tau, *etas])
c_nc = rx.Constraint([rx.Comparison(d, school)])

# 3. centred: theta_j as parameters, hyperprior as a joint block including mu, tau (recipe 24)
thetas = [rx.Parameter(f"theta_{j}") for j in range(J)]
c_c = rx.Constraint([rx.Comparison(d, rx.Model(lambda x, *th: np.asarray(th), thetas))])
problem_c = rx.Problem([c_c], priors=[(thetas + [mu, log_tau], SchoolHierarchy())])
```

Expected behaviour:

- All three spellings give the same posterior for `mu` and `tau`.  The
  posterior of `tau` piles up near zero, as in the book, and the `theta_j`
  shrink toward `mu` as `tau` falls.
- The marginalised form has two columns and is the design's own
  philosophy: a Gaussian latent belongs in the covariance.  The
  non-centred form has `J + 2` columns, all with marginal priors, so
  `prior_transform` is available with no joint block.  The centred form
  has the funnel geometry the book warns about and samples worst with
  emcee or dynesty.
- A parameter attached only to a fully masked comparison is sampled from its
  prior.  Holding out one school therefore leaves its `eta_j` in the
  chain, driven by `tau` alone, so `complement()` and `predictive_draws`
  give the predictive for a new group with no extra code.
- Meta-analysis on log-odds with per-study standard errors is the same
  structure.  The beta-binomial rat-tumour example is not expressible: a
  binomial likelihood is not elliptical (see the closing section).  A
  normal approximation with `y_err = sqrt(n_j p_j (1 - p_j))` as a
  prediction-scaled `diag` term is expressible but is not the book's
  model.

The notebook `hierarchical_calibration` (design document §7) is the
parameter-space version of this hierarchy: per-dataset deviation vectors on
the physics parameters with a hyperprior on their spread, used to repair a
misspecified energy dependence.

Reference: Gelman, Carlin, Stern, Dunson, Vehtari, Rubin, *Bayesian Data
Analysis*, 3rd ed., CRC Press (2013), Chapter 5.

---

## What this API does not express

Each item names the assumption that breaks, the nearest workaround, and
the size of the addition that would lift it.

- **Non-elliptical likelihoods.**  Poisson counts, censored points and upper
  limits, and two-component good/bad mixtures `(1 − β) N + β t` (Hanson
  2007) are not functionals of `(d2, logdet, n)`.  Workaround: none that is
  faithful.  Addition: a `Likelihood` that receives the residual and the
  factor, plus a second evaluation path in `CompiledConstraint`; about 100
  lines.
- **Chain-dependent masks.**  KDUQ's iterative rejection of points more
  than 3σ from the current model, updated during the walk, needs a mask
  that depends on chain state.  Masks are compiled.  Workaround: an outer
  loop of problems with the mask refit between runs.  Addition refused by
  design: it is mutable state inside a spec.
- **Per-point latent variables.**  Errors-in-variables in `x` (Berkson),
  explicit latent function values on a mesh (Schnabel et al. 2021), or a
  sampled per-point scale in a scale mixture.  Expressible in principle as
  one `Parameter` per point, but the dimension defeats the drivers.
  Addition: latent nodes with analytic marginalisation; large.
- **Per-point log-likelihood factors** for PSIS-LOO and WAIC under a
  correlated covariance.  Workaround: the prefix-difference of recipe 25,
  one problem per point.  Addition:
  `CompiledConstraint.pointwise_log_likelihood` from the factor; about 30
  lines.
- **Input-dependent mixture likelihoods.**  Bayesian model mixing with a
  per-point two-component Gaussian likelihood (Semposki et al. 2022).  Mean
  mixing is a `Model` (`w(x; θ) f1 + (1 − w) f2`); the likelihood form is the first bullet.
- **Correlation across constraints.**  By construction.  Merge the
  constraints.
- **A sampled tempering exponent.**  `weight` is a float by type; a
  sampled `η` is not a coherent posterior.  SafeBayes chooses it outside the
  sampler (recipe 25).
- **Nonlinear, non-pointwise comparison maps.**  `space` is pointwise so
  the delta method and the log-Jacobian stay exact.  A linear projection is
  preprocessing (recipe 20); a nonlinear feature of the whole vector would
  need a full Jacobian.  Not planned.
- **Kronecker-structured covariances.**  A separable GP over energy and
  angle on a common grid could factor as a Kronecker product; real data
  share no grid, so the dense path is used (recipe 23).  Addition: a fourth
  term kind; not planned until a case needs it.
- **Joint integration over emulator hyperparameters** alongside
  calibration.  Bayarri et al. recommend fixing them; not a gap worth
  filling.
