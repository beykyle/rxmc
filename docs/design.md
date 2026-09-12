# Design: declare, then compile

`rxmc` calibrates reaction models to experimental data by Bayesian
inference, with the error model declared as part of the problem.  Every
statistical choice a study makes, which errors are statistical and which
are correlated, whether a normalisation is inferred or marginalised,
whether the model is allowed a discrepancy, which points are held out,
appears in the declaration, so that a reviewer can read the declaration and
write down the likelihood.  The library owns no sampler: a compiled
`Problem` exposes the densities and the prior transform that emcee, dynesty
or `black-box-bayes` need.

This document is the maintainer's description of the library as it is.
Read it with `recipes.md`, which states every supported use case with its
spelling and expected behaviour, one test per recipe, and with the
notebooks in `examples/`, which are the tutorials for the recipes.  The
plan the rewrite was executed from, including what was harvested from 0.x
and the milestone history, is `groundup_design.md`.

The mechanics rest on one rule:

> **Everything the user constructs is an immutable declaration.
> `Problem` is the only compile step.**

## 1. Rules for the maintainer

The review checklist for every change.

1. **Specs hold no caches and no solver state.**  `Parameter`, `Dataset`,
   `Term`, `Comparison` and `Constraint` are frozen dataclasses with
   `eq=False`; identity is the equality, for every spec.  `Model` is a
   plain class with the same contract.  Anything expensive or grid-dependent
   lives in the objects `Problem` builds, or in a cache a reaction model
   drops on pickling.
2. **Exactly one function walks the parameter graph:** `Problem.__init__`.
   Nothing else assigns slots, checks names, resolves supports or decides
   which prior covers which slot.
3. **Gather, never split.**  Every callable node receives one integer
   gather array at compile time and is evaluated as `node(*theta[gather])`.
   There is no parameter count to carry around and no chain slicing by
   position; `problem.columns(params)` is how a caller finds a column.
4. **Every user-supplied callable has one shape:** `fn(context, *values)`,
   where `context` is the grid `x` (models, transforms) or a `TermContext`
   (terms, bases, amplitudes) and `values` are the sampled values of the
   parameters the node declares, in declaration order.
5. **Sharing is spelled by passing the same `Parameter` object.**  Inside a
   term, between terms, between a model and a term, across constraints.
   There is no second identity notion and no value equality anywhere.
6. **One factorisation path.**  `StructuredCovariance` is the only way a
   covariance is factored.  The dense matrix exists as a display method and
   as the reference in tests.
7. **Fail at compile, and name the dataset.**  A singular constant
   covariance, duplicate parameter names, a slot no prior covers, an `on=`
   that references a comparison outside its constraint, a non-finite value
   in comparison space: all raised by `Problem`, never mid-chain.
8. **Nothing is folded into a covariance silently.**  A dataset's reported
   systematics become terms only when asked, through
   `comparison.reported_terms()`.  The 0.x default that folded them in is
   pinned as a *difference* in `test/test_regression.py`.
9. **A term on several comparisons sees the gathered stack.**  Its
   `TermContext` carries `segments`, `labels` and `split()` so that a basis
   which differentiates or smooths along the grid stays within one
   comparison.  Nothing is ever split for the term.
10. **A covariance that reads the prediction is the generative model's
    marginal likelihood.**  Its log-determinant pulls the posterior mode
    toward smaller predictions, and a flat prior on a prediction-scaled
    covariance has a `1/rho` tail.  That is a property of the model, not a
    bug; the Peelle-safe *evaluation* is the estimate-built refit (recipes
    27 and 37 state both, with numbers).
11. **A `Problem` pickles with `dill`.**  `black-box-bayes` ships it to
    every MPI rank by path.  A round trip of a reaction problem is in the
    suite.
12. **Correctness lives in tests that pin numbers**, not in defensive
    branches: the closed-form Student-t, delta-method errors under `log`,
    the regression log-likelihood `1.195784087817536`, dense-versus-
    structured equality on every error-model form, and the closed forms of
    the reference papers.  Every recipe in `recipes.md` is a test under
    `test/recipes/` (`test/test_recipes_index.py` enforces the mapping),
    and every notebook names the recipes it teaches
    (`test/test_notebooks_index.py`).

## 2. The API, module by module

Modules in dependency order.  Signatures are the ones in the source; the
docstrings carry the details.

### 2.1 `params.py`

```python
@dataclass(eq=False, frozen=True)
class Parameter:
    name: str
    bounds: tuple[float, float] = (-inf, inf)
    prior: object | None = None      # frozen scipy univariate: logpdf, cdf, ppf, rvs
    unit: str = ""
    latex: str | None = None
    label -> str                     # latex, falling back to name
```

A parameter *is* its object; it keys dictionaries and is matched by
identity everywhere.  It may carry its own marginal prior.  The rules,
enforced once at compile:

| declared | prior used |
|---|---|
| `prior=dist` | `dist` truncated to `bounds` (log-density `-inf` outside; `ppf` rescaled between `cdf(lo)` and `cdf(hi)`) |
| finite `bounds`, no `prior` | uniform on `bounds` |
| neither | must be covered by a joint prior given to `Problem`, else a compile error naming the parameter |

### 2.2 `transforms.py`

```python
class Transform:                       # fn(a, *values) -> array; params; derivative; inverse
    def __call__(self, a, *values)
    def derivative(self, a, *values)
    def __or__(self, other)            # (f | g)(a) = g(f(a)); params f + g
    n_params, is_identity, inverse

identity, log, exp                     # parameter-free; identity.inverse is identity, log/exp are inverses
def scale(parameter=None, log=True, name=None) -> Transform   # rho * a; default Parameter("log_rho")
def as_transform(t) -> Transform       # None -> identity; callable -> parameter-free Transform
```

One type, three roles: the comparison space of a `Comparison`, a mean
transform composed onto a `Model`, and the coordinates a `Term` is
evaluated in.  `is_identity` is an object-identity test on the singleton,
the only fast path.

### 2.3 `units.py` and `data.py`

```python
# units.py
XS_UNIT = "b/sr"; RUTHERFORD_UNIT = "mb/sr"; MB_PER_B = 1000.0; DEFAULT_LMAX = 20
def parse_unit(label) -> (factor, kind)   # the exfor_tools / x4i3 label vocabulary; kind is
                                          # "differential" or "dimensionless"; anything else raises
def check_angle_grid(angles_rad, name)
```

```python
# data.py
@dataclass(eq=False, frozen=True)
class Dataset:
    x: ndarray                                  # radians for reaction data
    y: ndarray                                  # physical units
    y_err: ndarray                              # statistical, physical units; zeros allowed
    norm_err: float | ndarray | None = None     # reported fractional normalisation, inert
    offset_err: float | ndarray | None = None   # reported absolute offset, physical units, inert
    label: str = ""
    meta: Mapping = {}                          # reaction, Elab, ExIAS, quantity, k, eta, ...
    n -> int

def from_measurement(measurement, *, reaction=None, quantity=None, ExIAS=None) -> Dataset
```

`Dataset` is pure data: no comparison transform, no mask, no solver
workspace.  `from_measurement` is the one EXFOR adapter.  It reads the
`exfor_tools.Distribution` fields (`x, y, Einc, quantity, y_units,
statistical_err, systematic_norm_err, systematic_offset_err, subentry`) from
any object shaped like one, converts angles to radians and the cross
section through `parse_unit`, scales every dimensionful error with the data
and leaves the fractional normalisation error alone, and fills `meta` with
the kinematics.  `dXS/dA` and `dXS/dRuth` convert into each other through
the closed-form Rutherford cross section of the kinematics, which makes the
conversion a per-angle factor; a neutron projectile or a missing reaction
is a named error.  `rxmc` never imports `exfor_tools`.

### 2.4 `model.py`

```python
class Model:                                  # fn(x, *values) -> y in physical space
    params: tuple[Parameter, ...]
    def bind(self, x, meta=None) -> Predictor  # generic: closes over x; reaction models override
    def __or__(self, transform) -> Model       # mean transform; its params appended
    def __add__(self, other) -> Model          # additive mean discrepancy; params left then right
    def __mul__(self, other) -> Model          # multiplicative correction; scale() is its constant case

class Predictor:                              # a model bound to a grid
    params, x
    def __call__(self, *values) -> ndarray

def polynomial(order) -> Model                # a_0 + a_1 x + ...; params a0..an, no priors
```

- `omp | scale(rho_1)` is a model with parameters `omp.params + (rho_1,)`;
  per-dataset Kennedy and O'Hagan scales are distinct `rho_i` objects on
  distinct comparisons.
- `omp + delta` with `delta = Model(fn, phi)` is the explicit, sampled mean
  discrepancy; `omp * g` the multiplicative one (an additive discrepancy in
  log space).  Both bind each side to the same grid and combine the
  predictors, so a reaction model and a plain function combine without
  special cases.  Composition is left to right: `(omp + delta) | scale(rho)`
  scales the sum.
- Plotting on a fine grid is `model.bind(x_fine, d.meta)(*row[problem.columns(model.params)])`.

### 2.5 `terms.py`

```python
@dataclass(eq=False, frozen=True)
class TermContext:
    x: ndarray            # coords(x) on the support
    y: ndarray            # data on the support, comparison space
    ym: ndarray | None    # prediction on the support; None while a constant term is evaluated
    __len__
    meta(key) -> ndarray  # the owning dataset's meta[key], one value per point
    segments -> tuple[slice, ...]   # rows of each spanned comparison within the gathered support
    labels -> tuple[str, ...]       # their labels, in the same order
    split(a) -> list                # a[s] for s in segments

@dataclass(eq=False, frozen=True)
class Term:
    fn: Callable | ndarray                 # fn(c: TermContext, *values) -> vector | matrix
    params: tuple[Parameter, ...] = ()
    kind: "diag" | "mode" | "matrix" = "matrix"
    on: Comparison | Dataset | sequence | None = None   # None = whole constraint
    coords: Transform = identity           # applied to x before fn sees it; its params appended
    constant: bool = False                 # fn reads neither ym nor parameters; evaluated once

@dataclass(eq=False, frozen=True)
class KernelTerm(Term):                    # what kernel() returns
    kernel, n_kernel, amplitude, jitter
```

| `kind` | `fn` returns | contribution |
|---|---|---|
| `"diag"` | standard-deviation vector `v` | `Σ_ii += v_i²` |
| `"mode"` | vector `v` | `Σ += v vᵀ` |
| `"matrix"` | symmetric block `M` | `Σ_block += M` |

Support is a reference, not integer indices: `on=comp` places the term on
that comparison, `on=[c1, c2]` spans both (case A), `on=None` is the whole
constraint; a `Dataset` also resolves.  A term is stateless and may sit in
two constraints.  An array-valued `fn` is a fixed contribution checked for
shape and symmetry at construction.

The factories:

```python
statistical(y_err, on=None)
offset(parameter=None, magnitude=None, mask=None, log=True, on=None)
normalization(parameter=None, magnitude=None, mask=None, log=True, on=None)
noise(parameter, log=True, basis=None, basis_params=(), on=None, coords=None)
noise_fraction(parameter, log=True, on=None)
model_error(parameter, averaging=True, log=True, on=None)
systematic(parameter, basis, log=True, basis_params=(), on=None, coords=None)
kernel(kernel, coords=None, amplitude=None, amplitude_params=(), jitter=1e-10,
       prefix="discrepancy", params=None, on=None) -> KernelTerm
# bases and amplitudes: ones, ym, averaging, x_basis(scale), exp_growth(scale, base=ones),
#                       constant_amplitude, exp_growth_amplitude(scale)
```

`noise` and `noise_fraction` are additive on top of the reported diagonal;
`Constraint(statistical=False)` makes them replace it.  `normalization`
reads `c.ym`, never `c.y`.  `kernel` derives one `Parameter` per free
hyperparameter element in sklearn's log-theta space, bounded by the log of
the kernel's bounds so it compiles with a uniform prior there; `params=`
passes the objects instead, which is also how hyperparameters are shared
between per-comparison kernels.  Two kernel terms with derived names and
one prefix fail compile on the duplicate name.

Hierarchy is sharing plus `meta`: a hyperparameter shared by several
datasets is one object placed in one term per comparison, and anything
dataset-specific the term needs comes from `c.meta(key)`.  A discrepancy
correlated *across* datasets is one `matrix` term spanning them that builds
its inputs from `c.meta("Elab")` and `c.x`; it takes the dense path.

### 2.6 `likelihood.py`

```python
class Likelihood:                  # functional of (d2, logdet, n, *values); params are ordinary nodes
    def log_likelihood(self, d2, logdet, n, *values)
    def chi2(self, d2, logdet, n, *values)          # d2
class Gaussian(Likelihood)
class StudentT(Likelihood)         # StudentT(nu=None) -> Parameter("nu", prior=gamma(a=2, scale=10), bounds=(1, inf)); pass nu= to share or rename
class Chi2(Likelihood)             # -d2/2, no log-determinant
```

All three are exported from the package (`rx.Gaussian`, `rx.StudentT`,
`rx.Chi2`).

### 2.7 `constraint.py`

```python
@dataclass(eq=False, frozen=True)
class Comparison:                    # one dataset, one model, one comparison space
    data: Dataset
    model: Model                     # bound at construction: model.bind(data.x, data.meta)
    space: Transform = identity      # parameter-free
    predictor, y, y_err, log_jac     # derived once: space(data.y), delta-method errors, per-point Jacobian
    n; predict(*values); log_jacobian(mask=None)
    def reported_terms(self) -> list[Term]   # offset then normalisation modes from the dataset's
                                             # reported errors, delta-method propagated, on=self

@dataclass(eq=False, frozen=True)
class Constraint:                    # the maximal block of mutually correlated data: one likelihood
    comparisons: tuple[Comparison, ...]
    terms: tuple[Term, ...] = ()
    likelihood: Likelihood = Gaussian()
    weight: float = 1.0              # tempering; multiplies this constraint's log-likelihood only
    statistical: bool = True         # add each comparison's y_err diagonal
    masks: tuple[ndarray, ...] | None = None
    offsets, active, n_total, n_active, log_jacobian, support(on)
    def masked(self, masks); def masked_where(self, predicate); def complement(self)
```

The comparison space lives on the comparison because it is a modelling
choice.  Masks live on the constraint: `masked`, `masked_where` and
`complement` return a `Constraint` with the same `Comparison`, `Term` and
`Parameter` objects and new masks, so a held-out problem built from
`complement()` shares every parameter with the fit and a posterior sample
scores it directly.  `weight` is the one tempering knob.  Eager checks
here need nothing from the parameter graph: distinct comparisons, an
array-valued term of the right shape for its `on`, a non-finite comparison
space caught at compile with the comparison's label.

### 2.8 How the pieces thread

There are exactly two parametric entry points; everything else is fixed
when the comparison is built.

| stage | space | what enters | parametric |
|---|---|---|---|
| 1. `Predictor` | physical | `f(x; θ)` on the data grid | model parameters |
| 2. mean modifications, composed on the `Model` | physical | `\| scale(ρ)`, `* g(x; φ)`, `+ δ(x; φ)` | ρ, φ |
| 3. `space` | physical → comparison | `y = space(data.y)`, `ym = space(step 2)`, `y_err = \|space'\| · data.y_err`, `log_jacobian` | none |
| 4. `Term`s, seeing `TermContext(x, y, ym)` | comparison | statistical diagonal; experimental terms (noise, reported modes, USU); model-discrepancy terms (kernel, bases) | term parameters, and `ym` |
| 5. `Likelihood` | comparison | functional of `y − ym` and Σ | Student-t ν only |

Experimental and model-discrepancy terms are one mechanism; the difference
is what the user means.  A term that reads `ym` sees the prediction after
the mean modifications and after `space`, so a reported normalisation
error applies to the measured scale and `reported_terms()` gets that for
free.  Mean-side discrepancy is physical-space and `x`-aware; covariance-
side discrepancy is comparison-space and `ym`-aware.  The normalisation
stays on the mean rather than dividing the data: dividing would make
`space` parametric, and in linear space a data-side normalisation is what
Peelle's Pertinent Puzzle warns about (recipe 27).

### 2.9 `problem.py`: the compile step

```python
class Problem:
    def __init__(self, constraints, priors=()):   # priors: [(params, joint), ...]
    index: ParameterIndex; constraints: tuple[CompiledConstraint, ...]; priors
    ndim, names, bounds, params
    def columns(self, params) -> ndarray
    def log_prior / log_likelihood / log_posterior / chi2 (theta)
    def prior_transform(self, u); def sample_prior(self, n, rng=None)
    def predict(self, theta, physical=False) -> list[list[ndarray]]   # per constraint, per comparison
    def log_jacobian(self) -> float
    NDIM, parameter_names, starting_location(n), log_posterior_batch(thetas)   # black-box-bayes spellings
```

`Problem.__init__` is the only place that walks the graph.  For each
constraint it stacks the comparisons (comparison space, one slice each),
registers every predictor's, term's and likelihood's parameters in
first-seen order through `index.add_all`, which returns the gather array
for that node, resolves each term's `on` to rows, and builds the
`StructuredCovariance`, whose constant parts are factored eagerly so a
singular block is reported with its label.  Parameters that only a joint
prior block mentions, a hyperprior's hyperparameter, get slots after every
constraint.  Names are then checked unique and the prior assembled.
Compile the same declarations twice and you get two independent problems.

**Prior assembly.**  Each slot is covered by its parameter's marginal or by
exactly one joint block `(params, joint)`, where `joint` has
`logpdf(values)` over those parameters in that order and optionally
`prior_transform(u)` and `rvs(size=n, random_state=rng)` (scipy's
spelling); `scipy.stats.multivariate_normal` qualifies and is whitened for
the unit-cube map, and renormalised by its mass inside any bounds (a custom
joint's `logpdf` must already be normalised on its truncated support).  A
joint that declares its dimension (`dim`) must match
its parameters, and a one-parameter block holding a scipy univariate
distribution is that parameter's marginal.  A hyperprior is a joint
block that includes its hyperparameter, whose `logpdf` is
`sum log p(child | hyper) + log p(hyper)` and whose `prior_transform` draws
the hyperparameter first (recipe 24).  A slot no prior covers, or covered
twice, is a compile error naming it.

**`CompiledConstraint`** holds the stacked `x`, `y`, `y_err`, the offsets
and active rows, the predictors with their gathers, the covariance, the
likelihood and its gather, the weight and the Jacobian; it exposes
`ym(theta)`, `log_likelihood`, `chi2` and `matrix(theta)`, the dense
covariance on the active rows for display and tests.

### 2.10 `covariance.py`: the structured covariance

The three term kinds are a decomposition:

```
Σ = diag(D) + blockdiag(M_b) + U Uᵀ        (+ dense fallback)
```

`D` is the sum of squares of every `diag` term; `M_b` one dense block per
comparison from the `matrix` terms inside it; `U` has one column per
`mode` term scattered onto its support, a mode spanning comparisons being
a column with entries in several blocks.  A `matrix` term that crosses
comparisons forces the dense path for that constraint.  With
`B = blockdiag(M_b + diag(D_b))` and per-block Cholesky factors,

```
z = L⁻¹ r,  W = L⁻¹ U,  S = I + WᵀW
d2 = zᵀz − (Wᵀz)ᵀ S⁻¹ (Wᵀz),   logdet Σ = Σ_b logdet B_b + logdet S
```

at cost `O(Σ_b n_b³ + N r² + r³)`.  Masking slices rows before assembly.
Constant parts are evaluated once at compile; a constraint whose
covariance is entirely constant caches its factors.  `B` must be positive
definite: a comparison covered only by modes is singular in `B` even when
`Σ` is not, and compile says so with the label and the remedies.  A
constraint boundary is about the likelihood functional and the weight, not
about cost.

```python
class StructuredCovariance:
    def distance(self, ym, theta=()) -> (d2, logdet)      # active rows
    def matrix(self, ym, theta=()) -> ndarray             # dense, active rows
    dense: bool                                           # whether a cross-comparison matrix term forced it
```

### 2.11 `diagnostics.py` and `predictive.py`

Everything takes `(problem, samples)` with `samples` of shape
`(n, problem.ndim)` in `problem.names` order, which is what emcee's
`get_chain(flat=True)`, dynesty's `samples_equal()` and `black-box-bayes`
give.

```python
predictive_draws(problem, samples, constraint=0, *, n_rep=1, rng=None, model_only=False, given=None)
coverage_curve(draws, y, levels=None); coverage_error(draws, y, levels=None)
sharpness(draws, percentiles=(16, 84), transform=None)
heldout_log_predictive(heldout_problem, samples, *, given=None)
log_posterior_predictive(logp_samples, logw=None)
logz_summary(logz, logzerr); compare_logz(a, b, sigma=2.0)

gp_posterior_predictive(kernel, theta, X_train, residuals, X_pred, *, train_noise_var=None, jitter=1e-10)
predictive_band(draws, levels=(16, 50, 84))
total_predictive_band(problem, term, predictor, x_pred, samples, *, noise_std=0.0,
                      train_noise_var=None, levels=(16, 84), n_draws=400, rng=None, physical=False)
```

A held-out problem built from `complement()` has the *marginal*
covariance of its rows, which is the right density when no term spans the
split and `ll(fit) + ll(held) = ll(full)` then holds.  When a term does
span it (a GP over several experiments), `given=fit_problem` makes both
functions compute the Gaussian conditional `p(y_held | y_fit, θ)` under
the full covariance; without a spanning term it equals the marginal.

`total_predictive_band` locates the `KernelTerm` in the problem, takes its
training rows, comparison space, kernel and amplitude columns from there,
conditions the discrepancy on the residuals with every *other* term of the
constraint as the regression noise, evaluates the amplitude at `x_pred`
through a `TermContext`, and returns percentiles in the comparison space
of the term's comparisons, or in physical units with `physical=True`.

### 2.12 `reactions/`

```python
class ElasticXS(Model):
    def __init__(self, quantity, central, spin_orbit, args_from_params, params, coulomb=None,
                 *, lmax=DEFAULT_LMAX, wavelengths_beyond_range=2.0, zeros_per_node=5)
    def bind(self, x, meta) -> Predictor      # reads meta["reaction"], meta["Elab"]
class IsobaricAnalogPN(Model):
    def __init__(self, U_p_coulomb, U_p_central, U_p_spin_orbit, U_n_central, U_n_spin_orbit,
                 args_from_params, params, *, lmax=..., wavelengths_beyond_range=..., zeros_per_node=...)
    def bind(self, x, meta) -> Predictor      # reads reaction, Elab, ExIAS
def rutherford(kinematics, angles_rad) -> ndarray     # mb/sr, closed form
def momentum_transfer(angles_rad, k) -> ndarray        # 2 k sin(theta/2), for coords=
```

`ElasticXS` evaluates `central(r, *args)`, `spin_orbit(r, *args)` and,
when given, `coulomb(r, *args)` on the workspace's radial grid, where
`args_from_params(ws, *values)` returns two or three argument tuples,
solves with jitr and extracts `dXS/dA` (b/sr), `dXS/dRuth` or `Ay`.  The
angular basis is cached per model instance and kinematics, so a plotting
grid reuses the data grid's basis; the cache is dropped on pickling.  The
model owns its solver and the data does not, so a masked view or an
unpickled problem cannot lose a workspace.  There is no compound-elastic
hook: that contribution is subtracted from `data.y` as preprocessing
(recipe 20).

## 3. Worked example: an error-model comparison

The shape of `examples/alpha_ca_error_model_comparison.ipynb`: real data
without reported errors, one potential, a ladder of covariances compared by
evidence and by held-out prediction.

```python
import numpy as np, dynesty, rxmc as rx
from rxmc import terms as T, transforms as tf
from scipy import stats
from sklearn.gaussian_process.kernels import Matern

data = rx.Dataset(angles_rad, ratio_to_rutherford, np.zeros(n), label="44Ca(a,a) 29 MeV",
                  meta={"reaction": reaction, "Elab": 29.0})
omp = rx.reactions.ElasticXS("dXS/dRuth", central, spin_orbit, args_from_params, params,
                             coulomb=coulomb, lmax=30)
comp_log = rx.Comparison(data, omp, space=tf.log)
comp_lin = rx.Comparison(data, omp)
log_eps = rx.Parameter("log_eps", prior=stats.uniform(np.log(0.05), np.log(40)))
log_eta = rx.Parameter("log_eta", prior=stats.uniform(np.log(0.05), np.log(40)))
gp = T.kernel(Matern(0.1, nu=2.5), on=comp_log, coords=lambda x: x / np.pi,
              amplitude=T.constant_amplitude, amplitude_params=(log_A,), params=[log_ell])
ladder = {
    "L0":  rx.Constraint([comp_log], terms=[T.noise(log_eps)], statistical=False),
    "E0":  rx.Constraint([comp_lin], terms=[T.noise_fraction(log_eps)], statistical=False),
    "L2y": rx.Constraint([comp_log], terms=[T.noise(log_eps), T.normalization(log_eta)], statistical=False),
    "Lgp": rx.Constraint([comp_log], terms=[T.noise(log_eps), gp], statistical=False),
}
logz = {}
for name, c in ladder.items():
    p = rx.Problem([c])
    ns = dynesty.NestedSampler(p.log_likelihood, p.prior_transform, p.ndim, nlive=80, sample="rwalk")
    ns.run_nested(dlogz=1.5, print_progress=False)
    res = ns.results
    logz[name] = rx.diagnostics.logz_summary(res.logz[-1] + p.log_jacobian(), res.logzerr[-1])
verdict = rx.diagnostics.compare_logz(logz["Lgp"], logz["L2y"])

fit = ladder["Lgp"].masked_where(lambda x: x < np.deg2rad(90))
p_fit, p_held = rx.Problem([fit]), rx.Problem([fit.complement()])
samples = run(p_fit)                                  # rows in p_fit.names order
# the GP spans the cut, so score and draw from p(y_held | y_fit, theta): given=
lp = rx.diagnostics.heldout_log_predictive(p_held, samples, given=p_fit)
score = rx.diagnostics.log_posterior_predictive(lp)
draws = rx.diagnostics.predictive_draws(p_held, samples, n_rep=4, given=p_fit)
```

The labels `L0`, `E0`, `L2y`, `Lgp` are the error-model ladder of recipe
18, whose table defines every label.  The same problem runs under emcee
from `p.sample_prior` and `p.log_posterior`, and under `black-box-bayes`
from a `dill` pickle and a six-line module that forwards
`starting_location`, `log_posterior`, `log_likelihood` and
`prior_transform`.  Reaction problems are driven by dynesty by preference:
affine-invariant ensembles mix poorly on optical-model posteriors.

## 4. Capability map

Every statistical capability the library supports, its spelling, and the
test that pins it.  Recipe numbers refer to `recipes.md`.

| capability | spelling | pinned by |
|---|---|---|
| user-defined model `y = m x + b` | `Model(lambda x, m, b: m*x + b, [m, b])` | test_model |
| `polynomial(order)` | `polynomial(order)` | test_model |
| statistical diagonal only; `chi2` | default `Constraint`; `problem.chi2(theta)` | test_problem, recipe 1 |
| unknown fractional / constant noise | `noise_fraction(log_eps)`, `noise(log_eps)` | test_terms, recipe 2 |
| inferred noise replacing reported statistics | `Constraint(statistical=False, terms=[noise(...)])` | test_constraint, recipe 2 |
| reported normalisation / offset as modes | `comparison.reported_terms()`; `normalization(magnitude=)`, `offset(magnitude=)` | test_constraint, test_regression, recipe 3 |
| free normalisation / offset nuisance | `normalization(log_eta)`, `offset(log_omega)` | test_terms, recipe 4 |
| fixed dense covariance; fixed diagonal | `Term(C, on=comp)`, `Term(sig, kind="diag", on=comp)` | test_terms, recipe 19 |
| case B: one parameter, two block-local terms | `noise(log_eps, on=c1), noise(log_eps, on=c2)`; or two constraints sharing `log_eps` | test_problem, recipe 5 |
| case A: one mode across comparisons | `normalization(log_eta, on=[c1, c2])` | test_covariance, test_regression, recipe 5 |
| per-dataset Kennedy and O'Hagan scale | `Comparison(d_i, omp \| scale(rho_i))` | test_model, recipe 6 |
| sampled mean discrepancy | `omp + Model(delta_fn, phi)`; alone or with `kernel` | test_model, recipe 8 |
| multiplicative `x`-dependent correction | `omp * Model(g_fn, phi)` | test_model |
| GP discrepancy in `x` or momentum transfer, with amplitude | `kernel(k, on=comp, coords=..., amplitude=..., amplitude_params=...)` | test_terms::TestStudyForms, recipe 7 |
| total predictive band from the problem alone | `total_predictive_band(problem, term, predictor, x_pred, samples)` | test_predictive, recipe 7 |
| hyperparameters shared across datasets, values from `meta` | same objects in one term per comparison; `c.meta("Elab")`; `kernel(params=)` | test_terms, test_problem, recipe 22 |
| discrepancy correlated across energies | one `matrix` term `on=comps` from `c.meta` and `c.x`; dense path | test_covariance, recipe 23 |
| term spanning comparisons reading its pieces | `c.segments`, `c.labels`, `c.split(a)` | test_terms, test_covariance, recipe 37 |
| correlated normalisations between quantities | one comparison per quantity, a spanning `matrix` term from `c.split(c.ym)`; the reference's closed forms | test_recipe_37 |
| Peelle's Pertinent Puzzle | `normalization()` reads `c.ym`; the estimate-built refit; the log-determinant pull of the live term | recipes 27, 37 |
| unaccounted-for model error per data type (KDUQ) | `model_error(delta_T, averaging=True, on=comp)`; scalings as `Constraint(weight=)` | test_terms, recipe 26 |
| tempering | `Constraint(weight=)` | test_problem, recipe 12 |
| Student-t with bounded ν; `Chi2` | `StudentT(nu=Parameter("nu", bounds=(1, 100)))`; `Chi2()` | test_likelihood, recipe 9 |
| log-space comparison, delta-method errors, Jacobian | `Comparison(d, m, space=log)`; `problem.log_jacobian()` | test_constraint, recipe 10 |
| masks, hold-out, complement | `masked_where`, `complement()`, `heldout_log_predictive` | test_constraint, test_diagnostics, recipe 11 |
| held-out scoring under a spanning term | `heldout_log_predictive(held, s, given=fit)`, `predictive_draws(..., given=fit)` | test_diagnostics, recipe 30 |
| stacking by leave-one-dataset-out | `Constraint.masked` dropping a block; `log_posterior_predictive` | recipe 28 |
| cut posterior by multiple imputation | stage-1 and per-draw stage-2 problems | recipe 29 |
| simulation-based calibration | `sample_prior`, `predictive_draws`, `dataclasses.replace(d, y=)` | recipe 31 |
| emulator as `Model`, its variance as a `diag` term sharing parameters | recipe 32 | test_terms |
| MAP and Laplace | `scipy.optimize` on `log_posterior`, `problem.bounds` | recipe 33 |
| global error scale and USU modes | `diag` term closing over `comp.y_err` with `statistical=False`; `offset(parameter=, on=technique)` | recipe 34 |
| energy-dependent parameters | per-comparison `Model` instances closing over `meta`, shared objects | test_model, recipe 35 |
| discrepancy on a physical basis | `systematic` modes or `omp + Model(basis_sum)` | recipe 36 |
| hierarchy: joint block with a sampled hyperparameter | `Problem(priors=[(children + [hyper], obj)])` | test_problem, recipe 24 |
| classic normal hierarchical model | marginalised, non-centred, centred | recipe 38 |
| SafeBayes | `replace(c, weight=η)` and prefix masks | recipe 25 |
| joint MVN prior; truncated marginals; `prior_transform` | `Problem(priors=[(omp.params, mvn)])`; `Parameter(prior=, bounds=)` | test_problem, recipe 13 |
| emcee, dynesty, black-box-bayes drivers | the flat interface; `dill` round trip | test_problem, recipe 16 |
| EXFOR to dataset with unit conversion | `from_measurement(m, reaction=, quantity=)` | test_measurement, recipe 14 |
| elastic `dXS/dA`, `dXS/dRuth`, `Ay`; (p,n) IAS | `ElasticXS`, `IsobaricAnalogPN` | test_reactions, recipe 15 |
| singular-covariance guard naming the dataset | compile-time in `Problem` | test_covariance, recipe 21 |
| covariance heat maps; fine-grid plotting | `problem.constraints[i].matrix(theta)`; `model.bind(x_fine, meta)` | notebooks |

## 5. Testing

Two tiers, chosen per assertion.

- **Fast tier**, `pytest`: every unit test and every recipe test's
  sampler-free assertions.  Most expected-behaviour bullets are structural
  or analytic: names and columns, a covariance equal to a hand-built
  matrix, `chi2` identities, `ll(fit) + ll(held) = ll(full)`, compile-time
  errors.  Where a recipe says "run a sampler", the fast tier uses
  `test/recipes/oracle.py`, the closed-form posterior of a linear-Gaussian
  problem, through `common.linear_posterior` and `common.oracle_samples`;
  exact rows stand in for a chain, so coverage, held-out scores, stacking
  weights and calibration ranks are pinned exactly.  Seeded short chains
  appear only for ordering claims with a wide margin.  Budget: a few
  minutes, zero tolerated flakiness; a flaky assertion is demoted, never
  loosened.
- **Converged tier**, `pytest -m slow`, then `pytest -n 4 --nbmake
  --nbmake-timeout=3600 examples`: the numeric claims that need a
  converged sampler, and the notebooks.  It is the "Converged tier"
  workflow, required on pushes and pull requests into `main`, also run by
  hand with `workflow_dispatch`; there is no scheduled run.

Three index tests keep the documents honest: `test_recipes_index.py`
(one file per `## NN.` heading, each quoting its recipe; the recipe-18
legend equal to the tests' legend), `test_notebooks_index.py` (each
notebook cites the recipes of the design's table, all nine exist), and
`test_regression.py` (the 0.x pins).  `test/helpers.py` holds the dense
references and `STUDY_LEGEND`, the labelled error-model forms of recipe
18 built against hand-written matrices.

## 6. Notebooks

The notebooks in `examples/` each name the recipes they teach in their
first cell.  Runtimes are wall times on an eight-core laptop, one kernel at
a time unless noted.

| notebook | recipes | driver | content | runtime |
|---|---|---|---|---|
| `linear_calibration` | 1, 17 | emcee | the whole workflow on a line; prior and posterior predictive; the coverage curve | 23 s |
| `error_models` | 2, 4, 19 | emcee | the covariance ladder on one comparison, the Peelle matrix as a fixed term, offsets known, free and ignored | 89 s |
| `sharing_error_models` | 5 | emcee | two experiments with opposite normalisation defects: sharing a parameter, a mode per dataset, one mode spanning both, and the assembled covariance seen directly | 78 s |
| `normalization_and_covariance_structure` | 3, 4, 6, 27 | emcee | six treatments of five experiments' normalisations, one of them badly mis-quoted and alone in its range; Peelle's puzzle in the two-point case it was found in; a gallery of covariance structures from `matrix(theta)` | 353 s |
| `gp_discrepancy` | 7 | emcee, dynesty | mean-zero discrepancies with amplitudes growing in x: four rungs on a toy line, three on n+⁴⁰Ca missing its surface absorption; the total predictive band | 616 s |
| `robust_likelihoods` | 9, 39 | emcee | Student-t versus Gaussian on three gross outliers; the iterative rejection loop, including the round that over-rejects and recovers | 54 s |
| `error_scale_and_usu` | 34 | emcee | a global scale on the reported errors under both likelihoods; a USU offset on the technique we suspect | 90 s |
| `local_optical_model_calibration` | 4, 12, 14, 15, 16, 21, 26 | dynesty | EXFOR O1199007, p + ⁴⁰Ca at 35 MeV, which quotes no systematics: the unit contract, three error models against a potential wrong at the 30 % level, the singular guard, tempering and its coverage, other drivers | 1565 s |
| `alpha_ca_error_model_comparison` | 10, 11, 13, 17, 18 | dynesty | real ⁴⁴Ca(α,α) data, a four-parameter potential, the ladder by evidence with the Jacobian, predictive draws carrying the covariance and their coverage, held-out backward angles scored conditionally | 1394 s |
| `hierarchical_calibration` | 22, 24, 30, 35, 38 | dynesty | eight schools; a hierarchy on the physics parameters repairing a misspecified energy dependence, in sample and at a held-out energy | 937 s (alongside another notebook) |

**`hierarchical_calibration` in detail.**  The truth is
`y = a0(E) + a1(E) x + a2(E) x²`, measured by seven synthetic datasets at
known energies plus one held-out dataset at an energy bracketed by two of
them; the true `a_k(E)` are a smooth trend plus non-monotonic bumps.  Three
fits of the same data: the correct mapping with global `φ`; a misspecified
linear mapping with global `φ`; the linear mapping plus a per-dataset
deviation vector `δ_j = τ ⊙ η_j`, non-centred, `η_jk ~ N(0, 1)`, `τ_k`
half-normal, as marginal priors.  One `Model` per comparison closes over
`E_j`; the held-out comparison is fully masked, so its `η_new` is sampled
from the prior, driven by `τ`, and `complement()`, `predictive_draws` and
`heldout_log_predictive` score the new energy with no extra code.  The
correct mapping covers in and out of sample; the misspecified one
under-covers both and leaves structured per-dataset residuals; the
hierarchy recovers coverage with wider, longer-tailed bands at the new
energy, a `τ` posterior away from zero, and the better held-out score of
the misspecified pair.  A hierarchy learns the spread of deviations it has
seen: holding out an unmodelled peak between its datasets is the
few-datasets caveat, not a prediction it can make.

## 7. Layout and dependencies

```
src/rxmc/
  __init__.py       re-exports
  params.py  transforms.py  units.py  data.py  model.py  terms.py
  likelihood.py  constraint.py  covariance.py  problem.py
  diagnostics.py  predictive.py
  reactions/  elastic.py  ias.py
test/               one file per module, the index tests, test_regression.py, helpers.py
test/recipes/       one file per recipe; common.py, oracle.py
examples/           the notebooks and plotstyle.py; data/ (committed measurements)
docs/               this document, recipes.md, examples.rst, api.rst, groundup_design.md (history)
```

Runtime dependencies: `numpy`, `scipy`, `jitr>=3.0`, `exfor-tools`.
Extras: `examples` (emcee, dynesty, corner, matplotlib, scikit-learn,
dill, jupyter, ipykernel, tqdm), `validation` (examples plus pytest,
nbmake, nbqa, ruff, black, isort, build), `docs` (sphinx, the pydata
theme, myst-nb).  Python 3.12 or later.  Kernels are duck-typed on the
scikit-learn interface, so scikit-learn is not a runtime dependency.

## 8. Open questions and non-goals

- **Term-level partial masks.**  The factories accept `mask=`; a first-class
  `on=(comparison, point_mask)` would be tidier.  Not needed yet.
- **Workspace caching across models.**  The angular basis is cached per
  model instance; two models on one kinematics still build two.  A factor
  of a few in setup time, not in solve time.
- **Cross-constraint modes.**  `U` is per constraint so that constraints
  stay independent and weights stay meaningful.  A mode that couples two
  constraints is a reason to merge them.
- **The log-determinant pull.**  A prediction-scaled covariance biases the
  mode down by an amount that grows with the normalisation error (5 to 9 %
  at 20 %).  Whether to offer the estimate-built refit as a helper rather
  than a pattern is open.
- **Known non-goals.**  The closing section of `recipes.md` lists the
  calibration classes the design rules out (non-elliptical likelihoods,
  chain-dependent masks, per-point latents, per-point likelihood factors,
  mixture likelihoods) with the size of the addition each would need.

## 9. Release path and history

The repository, its pull-request history and its Pages site are kept.  0.x
was closed out with tag `v0.1.0` and branch `legacy/0.x`; the rewrite
happened on `rewrite`, cut from that `main`, in nine milestones (bootstrap;
parameters, transforms, units, likelihood and terms; data, model and
constraint; the structured covariance; the problem with the first
twenty-eight recipe tests; reactions and `from_measurement`; diagnostics
and predictive; the recipe index; the notebooks) followed by this
document.  Pre-release tags `v1.0.0a1`, `b1`, `rc1` publish to PyPI
through trusted publishing on tag push; they were deferred by decision
and remain available.  The release is a pull request of `rewrite` into
`main`, the tag `v1.0.0`, a GitHub Release and the Pages rebuild.
