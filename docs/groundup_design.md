---
orphan: true
---

# A ground-up rxmc: declare, then compile

This document guides a rewrite of `rxmc` from a blank repository.  It is the
plan of record for that rewrite, not a comparison with the current code: the
current `api_generalisation` branch stays where it is as the reference
implementation, and `design.md` / `bugs_found.md` remain the record of how
it was designed and what was found wrong with it.  `recipes.md` is the
companion: one short user story per use case the rewrite must support,
each with the spelling below and the behaviour a user should expect.

The *concepts* of `design.md` survive unchanged: evidence as a weighted sum
over independent constraints; a constraint as the maximal block of mutually
correlated data; one `Term` type with three kinds; the likelihood as a
functional of `(d2, logdet, n)`; the comparison space owned by the data side;
masks as part of support; and the case A (couple the data) / case B (share
a parameter) distinction.  The *mechanics* are rebuilt around one rule:

> **Everything the user constructs is an immutable declaration.
> `Problem` is the only compile step.**

Three goals drive every choice below, in this order:

1. **The maintainer.**  The smallest implementation that covers every
   capability the current notebooks and tests demonstrate, with no
   lifecycle rules to remember.  Budget: about 2,300 lines of package code
   in 14 modules, down from 6,199 in 23.
2. **The user.**  Declaring a problem, including each statistical modelling
   choice in it, reads as a statement of the model.  A reviewer should be
   able to read the declaration and write down the likelihood.
3. **External samplers only.**  emcee, dynesty and the `black-box-bayes`
   CLI are the calibration drivers.  The in-package Gibbs walker and its
   samplers are not carried over.

## 1. Rules for the maintainer

These are the review checklist for every change in the new repository.

1. **Specs hold no caches and no solver state.**  `Parameter`, `Dataset`,
   `Model`, `Term`, `Comparison`, `Constraint` are frozen dataclasses with
   `eq=False`: identity is the equality, for every spec, not only
   `Parameter`.  (A generated `__eq__` would try to compare the arrays
   they hold and raise.)  Anything expensive or grid-dependent lives in
   the objects `Problem` builds.
2. **Exactly one function walks the parameter graph:** `Problem.__init__`.
   Nothing else assigns slots, checks names, resolves supports, or decides
   which prior covers which slot.
3. **Gather, never split.**  Every callable node receives one integer
   gather array at compile time and is evaluated as `node(*theta[gather])`.
   There is no `n_model_params`, no `indices[:-1]`, no `split_params`.
4. **Every user-supplied callable has one shape:** `fn(context, *values)`,
   where `context` is the grid `x` (models, transforms) or a `TermContext`
   (terms, bases, amplitudes) and `values` are the sampled values of the
   parameters the node declares, in declaration order.
5. **Sharing is spelled by passing the same `Parameter` object.**  Inside a
   term, between terms, between a model and a term, across constraints.
   There is no second identity notion and no value-equality anywhere.
6. **One factorisation path.**  `StructuredCovariance` is the only way a
   covariance is factored.  The dense matrix exists as a display method
   and as the reference in tests.
7. **Fail at compile, and name the dataset.**  Singular constant
   covariance, duplicate parameter names, a slot no prior covers, an `on=`
   that references a block outside its constraint, a non-finite value in
   comparison space: all raised by `Problem`, never mid-chain.
8. **Nothing is folded into a covariance silently.**  A dataset's reported
   systematics become terms only when the user asks
   (`block.reported_terms()`).
9. **A `Problem` pickles with `dill`.**  `black-box-bayes` ships it to every
   MPI rank by path.  A round-trip test on a reaction problem is part of
   the suite.
10. **Correctness lives in tests that pin numbers**, not in defensive
    branches: the closed-form Student-t, delta-method errors under `log`,
    the regression log-likelihood `1.195784087817536`, dense-versus-
    structured equality on every error-model form.  **Every recipe in
    `recipes.md` is a test** under `test/recipes/`, one file per recipe
    named `test_recipe_NN_<slug>.py`, whose docstring quotes the recipe's
    intent and whose assertions are its *Expected behaviour* bullets.  A
    CI check fails when a `## NN.` heading in `recipes.md` has no test
    file, or a test file has no heading.

## 2. The API skeleton

Read top to bottom.  Module names are the package layout of §6.  Type hints
are the documentation; the prose says only what the signature cannot.

### 2.1 `params.py`

```python
@dataclass(eq=False, frozen=True)
class Parameter:
    name: str
    bounds: tuple[float, float] = (-inf, inf)
    prior: object | None = None      # frozen scipy univariate: logpdf, cdf, ppf, rvs
    unit: str = ""
    latex: str | None = None
```

`eq=False` keeps identity hashing: a parameter *is* its object, and it can
key a dict.  A parameter may carry its own marginal prior.  The rules, enforced
once at compile:

| declared | prior used |
|---|---|
| `prior=dist` | `dist` truncated to `bounds` (log-density `-inf` outside; `ppf` rescaled between `cdf(lo)` and `cdf(hi)`) |
| finite `bounds`, no `prior` | uniform on `bounds` |
| neither | must be covered by a joint prior given to `Problem`, else a compile error naming the parameter |

This replaces `IndependentPrior`, `TruncatedNormalPrior`, `as_prior` and the
list-of-scipy-distributions form with no classes at all.

### 2.2 `transforms.py`

Harvested from the current module minus `per_observation_scaling` and the
`contextual` flag.

```python
class Transform:
    fn: Callable                       # fn(a, *values) -> array
    params: tuple[Parameter, ...]
    derivative: Callable | None        # d fn / d a, elementwise; finite difference fallback
    inverse: Transform | None          # parameter-free transforms only
    def __call__(self, a, *values) -> ndarray
    def __or__(self, other) -> Transform      # (f | g)(a) = g(f(a)); params f + g

identity, log, exp                     # parameter-free singletons
def scale(parameter=None, log=True) -> Transform   # rho * a; default Parameter("log_rho")
def as_transform(t) -> Transform       # None -> identity; callable -> parameter-free Transform
```

One type, three roles: the comparison space of a `Comparison`, a mean transform
composed onto a `Model`, and the coordinates a `Term` is evaluated in.

### 2.3 `units.py` and `data.py`

```python
# units.py — the unit contract, without a unit library
XS_UNIT = "b/sr"                              # every cross section stored in b/sr
RUTHERFORD_UNIT = "mb/sr"                     # what jitr reports
MB_PER_B = 1000.0
def parse_unit(label) -> tuple[float, str]    # (factor into the internal unit, kind)
# x4i3 converts every EXFOR cross section to barns while parsing and exfor_tools
# labels the result "barns/ster", "b" or "unitless"; parse_unit maps that fixed
# vocabulary (plus the obvious spellings) and rejects anything else loudly.
MB_PER_B = 1000.0
DEFAULT_LMAX = 20
def check_angle_grid(angles_rad, name) -> None
```

```python
# data.py
@dataclass(frozen=True)
class Dataset:
    x: ndarray                                  # angles in radians for reaction data
    y: ndarray                                  # physical units (b/sr, dimensionless, ...)
    y_err: ndarray                              # statistical, physical units
    norm_err: float | ndarray | None = None     # reported fractional normalisation
    offset_err: float | ndarray | None = None   # reported absolute offset, physical units
    label: str = ""
    meta: Mapping = field(default_factory=dict) # kinematics: reaction, Elab, ExIAS, quantity, k, ...

def from_measurement(measurement, *, reaction=None, quantity=None, ExIAS=None) -> Dataset
```

`Dataset` is pure data.  No comparison transform, no mask, no solver
workspace, no identity key.

`from_measurement` is the single EXFOR adapter.  It reads the
`exfor_tools.Distribution` fields (`x, y, Einc, quantity, y_units,
statistical_err, systematic_norm_err, systematic_offset_err, subentry`),
converts units once through `parse_unit`, divides every dimensionful error by the
conversion factor `norm`, passes the fractional normalisation error through
untouched, converts angles to radians, and fills `meta` with `reaction`,
`Elab`, `quantity`, `k` (and `ExIAS` for the (p,n) channel).  The
`dXS/dA` ↔ `dXS/dRuth` conversions need the Rutherford cross section on
the data grid.  In jitr that is a closed form of the kinematics,
`10 * eta**2 / (4 * k**2 * sin(theta/2)**4)` mb/sr
(`DifferentialWorkspace.rutherford_xs`), so `from_measurement` computes it
from `reaction.kinematics(Elab)` with no workspace.  A per-angle `norm`
array is the result in those cases, exactly as today.

### 2.4 `model.py`

A model is a spec of "observables from parameters".  A predictor is that
model bound to a grid.

```python
class Model:
    params: tuple[Parameter, ...]
    def __init__(self, fn, params): ...             # fn(x, *values) -> y, physical space
    def bind(self, x, meta=None) -> Predictor: ...  # generic: closes over x
    def __or__(self, transform) -> Model: ...       # mean transform; its params appended
    def __add__(self, other) -> Model: ...          # additive mean discrepancy; params concatenated
    def __mul__(self, other) -> Model: ...          # multiplicative x-dependent correction; scale() is its constant case

class Predictor:
    params: tuple[Parameter, ...]
    x: ndarray
    def __call__(self, *values) -> ndarray          # physical space, on x

def polynomial(order) -> Model                      # a_0 + a_1 x + ... ; params a0..an
```

- `Model.__or__` composes at the model level, so `omp | scale(rho_1)` is a
  model whose parameters are `omp.params + (rho_1,)`; `bind` composes the
  bound predictor with the transform.  Per-dataset Kennedy–O'Hagan scales
  are distinct `rho_i` objects on distinct comparisons.  No routing table.
- `Model.__add__` is the explicit mean discrepancy: `omp + delta` with
  `delta = Model(lambda x, *phi: ..., phi_params)` predicts
  `omp(x) + delta(x)` in physical space, with parameters
  `omp.params + delta.params` (a parameter object in both is shared, as
  everywhere).  `bind` binds both sides to the same grid and sums.  This is
  the Kennedy–O'Hagan *sampled* discrepancy, in contrast to `kernel`, which
  marginalises it into the covariance; the two may be used together.  A
  reaction model and a plain-function model add without special cases
  because addition happens on the bound predictors.  Composition is
  left-to-right: `(omp + delta) | scale(rho)` scales the sum,
  `(omp | scale(rho)) + delta` scales only the model.
- `Model.__mul__` is the multiplicative counterpart: `omp * g` with
  `g = Model(lambda x, *phi: ..., phi)` predicts `omp(x) · g(x)`.  An
  additive discrepancy in log space is exactly this, and `scale(rho)` is
  the constant case `omp * Model(lambda x, r: np.exp(r), [rho])`, kept as
  a helper.  `+` and `*` share one binary-composition implementation; the
  parametric use of `|` remains for non-separable transforms only.
- Plotting on a fine grid is `omp.bind(x_fine, d.meta)(*values)`.  This
  replaces `visualization_workspace` and `visualizable_model_prediction`.
- Reaction models subclass `Model` and override only `bind`.

```python
# reactions/elastic.py
class ElasticXS(Model):
    def __init__(self, quantity, central, spin_orbit, args_from_params, params,
                 coulomb=None, *, lmax=DEFAULT_LMAX, wavelengths_beyond_range=2.0,
                 zeros_per_node=5): ...
    def bind(self, x, meta) -> Predictor
        # reads meta["reaction"], meta["Elab"];
        # builds the jitr IntegralWorkspace + DifferentialWorkspace on x
        # (set_up_solver, harvested); returns a Predictor that evaluates the
        # potentials on ws.radial_grid(), solves, and extracts dXS/dA | dXS/dRuth | Ay

def momentum_transfer(x, k) -> ndarray             # q = 2 k sin(x/2), for coords=

# reactions/ias.py
class IsobaricAnalogPN(Model):
    def __init__(self, U_p_coulomb, U_p_central, U_p_spin_orbit, U_n_central,
                 U_n_spin_orbit, args_from_params, params, *, lmax=..., ...): ...
    def bind(self, x, meta) -> Predictor          # reads reaction, Elab, ExIAS
```

The model owns its solver; the data does not.  A masked view, a copy, or an
unpickled problem cannot lose a workspace, because nothing routes on the
identity of a data object.  The predictor is a pure function of the
potential: there is no `compound_correction` hook.  A compound-elastic
contribution is subtracted from `data.y` as preprocessing (recipe 20),
which keeps the model free of data-side state and works on any grid.  Optional, not in v0: cache the
`IntegralWorkspace` on the model instance keyed on
`(Elab, lmax, wavelengths_beyond_range, zeros_per_node)` so two datasets at
one energy solve the basis once.

### 2.5 `terms.py`

```python
@dataclass(frozen=True)
class TermContext:
    x: ndarray            # coords(x) on the support
    y: ndarray            # data on the support, comparison space
    ym: ndarray | None    # prediction on the support; None while factoring constant parts
    def __len__(self) -> int
    def meta(self, key) -> ndarray   # the owning block's data.meta[key], one value per point;
                                     # for a term spanning blocks, the per-point concatenation
    segments: tuple[slice, ...]      # rows of each spanned comparison within the gathered support
    labels: tuple[str, ...]          # their comparison labels, in the same order
    def split(self, a) -> list       # a[s] for s in segments

@dataclass(frozen=True)
class Term:
    fn: Callable | ndarray                 # fn(c: TermContext, *values) -> vector | matrix
    params: tuple[Parameter, ...] = ()
    kind: Literal["diag", "mode", "matrix"] = "matrix"
    on: Comparison | Dataset | Sequence[Comparison | Dataset] | None = None   # None = whole constraint
    coords: Transform = identity           # applied to x before fn sees it; its params appended
    constant: bool = False                 # fn reads neither ym nor parameters
```

| `kind` | `fn` returns | contribution |
|---|---|---|
| `"diag"` | standard-deviation vector `v` | `Σ_ii += v_i²` |
| `"mode"` | vector `v` | `Σ += v vᵀ` |
| `"matrix"` | symmetric block `M` | `Σ_block += M` |

- **Support is a reference, not integer indices.**  `on=b1` places the term
  on that block, `on=[b1, b2]` spans both (case A), `on=None` is the whole
  constraint.  `on=` also accepts a block's `Dataset`, and compile resolves
  `block is target or block.data is target`, so a term written against a
  block still resolves after `masked`/`complement` rebuild the constraint.
  `stacked_supports` and `support=np.arange(...)` are gone.
- **A term is stateless.**  No `bind`, no cache.  The same object may be
  placed in two constraints.

The factory helpers keep their bodies and signatures with `support=` renamed
`on=`:

```python
statistical(y_err, on=None)
offset(magnitude=None, parameter=None, mask=None, log=True, on=None)
normalization(magnitude=None, parameter=None, mask=None, log=True, on=None)
noise(parameter, log=True, basis=None, basis_params=(), on=None, coords=None)
noise_fraction(parameter, log=True, on=None)
model_error(parameter, averaging=True, log=True, on=None)
systematic(parameter, basis, log=True, basis_params=(), on=None, coords=None)
kernel(kernel, coords=None, amplitude=None, amplitude_params=(), jitter=1e-10,
       prefix="discrepancy", params=None, on=None) -> KernelTerm
# KernelTerm(Term) adds kernel, n_kernel, amplitude, jitter so predictive.gp_predictive_draws
#          can condition the discrepancy from the term alone.  A derived hyperparameter is
#          bounded by the log of the kernel's bounds (a uniform prior in log-theta).
# params=: the hyperparameter Parameter objects, one per free element in kernel.theta order;
#          None derives fresh ones named f"{prefix}_{name}".  Pass the same objects to share
#          hyperparameters between per-block kernels.  Two kernel terms with derived
#          names and the same prefix fail compile on the duplicate name; the error says
#          to pass prefix= (distinct kernels) or params= (one shared kernel).
# bases and amplitudes: ones, ym, averaging, x_basis(scale), exp_growth(scale, base=ones),
#                       constant_amplitude, exp_growth_amplitude(scale)
```

**Hierarchy is sharing plus `meta`.**  A hyperparameter shared by several
datasets is one `Parameter` object placed in one term per block; anything
dataset-specific the term needs (energy, excitation energy, a flag) comes
from `c.meta(key)`.  A discrepancy correlated *across* datasets, such as a
GP over energy and angle, is one `matrix` term spanning the blocks whose
`fn` builds its inputs from `c.meta("Elab")` and `c.x`.  It takes the dense
path; there is no Kronecker structure because real data share no angle
grid.  Recipes 22–24 spell all three out.

### 2.6 `likelihood.py`

Harvested.  A likelihood is a functional of `(d2, logdet, n, *values)` with
optional parameters that are ordinary nodes in the index.

```python
class Likelihood:
    params: tuple[Parameter, ...] = ()
    def log_likelihood(self, d2, logdet, n, *values) -> float
    def chi2(self, d2, logdet, n, *values) -> float      # d2

class Gaussian(Likelihood): ...
class StudentT(Likelihood):        # StudentT(nu=None) -> Parameter("nu", bounds=(1, inf))
                                   # two constraints with the default each derive a "nu";
                                   # compile fails on the duplicate name and says to pass nu=
class Chi2(Likelihood): ...        # -0.5 d2, no log-determinant
```

### 2.7 `constraint.py`

A block is the unit the residual is formed on: one dataset, one model, one
comparison space.  A constraint is a tuple of comparisons plus the terms,
the likelihood functional, the tempering weight, and the active-point
masks.  Each comparison is one block of the stacked covariance (§2.9), so
"block" below and in §2.9 means that.

```python
@dataclass(frozen=True)
class Comparison:
    data: Dataset
    model: Model                         # bound at construction: model.bind(data.x, data.meta)
    space: Transform = identity          # parameter-free comparison transform
    # derived, computed once in __post_init__ / cached_property:
    predictor: Predictor
    y: ndarray                           # space(data.y)
    y_err: ndarray                       # |space'(data.y)| * data.y_err   (delta method)
    log_jacobian_all: ndarray            # log |space'(data.y)| per point
    def reported_terms(self) -> list[Term]   # offset then normalisation modes from data.norm_err /
                                             # data.offset_err, delta-method propagated, on=self

@dataclass(frozen=True)
class Constraint:
    comparisons: tuple[Comparison, ...]
    terms: tuple[Term, ...] = ()
    likelihood: Likelihood = Gaussian()
    weight: float = 1.0                  # tempering; multiplies the log-likelihood only
    statistical: bool = True             # add each block's y_err diagonal
    masks: tuple[ndarray, ...] | None = None   # active points per block; None = all active
    def masked(self, masks) -> Constraint
    def masked_where(self, predicate) -> Constraint     # masks = [predicate(comp.data.x) for comp in comparisons]
    def complement(self) -> Constraint                  # every inactive point active, and vice versa
```

- The comparison transform lives on the block because it is a modelling
  choice: a Gaussian in log space is a different distribution from one in
  linear space.  Terms are authored in comparison space, as today.
- **Masks live on the constraint, not on the block or the data.**
  `masked`, `masked_where`, `complement` return a `Constraint` with the same
  `Comparison`, `Term`, and `Parameter` objects and new masks.  No `copy.copy`,
  no identity key.  A held-out problem built from `complement()` shares
  every parameter with the fit, so a posterior sample scores it directly.
- `weight` subsumes both `Evidence(weights=)` and
  `CalibrationConfig.likelihood_scaling`: one knob, honoured by every driver.
- Eager checks here need nothing from the parameter graph: comparisons are
  distinct, an array-valued term has the right shape for its `on`, the
  comparison space is finite on every active point (named block).

### 2.7b How the pieces thread

There are exactly two parametric entry points.  Everything else is a
constant fixed when the comparison is built.

| stage | space | what enters | parametric |
|---|---|---|---|
| 1. `Predictor` | physical | `f(x; θ)` on the data grid | model parameters |
| 2. mean modifications, composed on the `Model` | physical | `\| scale(ρ)`, `* g(x; φ)`, `+ δ(x; φ)` | ρ, φ |
| 3. `space` | physical → comparison | `y = space(data.y)`, `ym = space(step 2)`, `y_err = \|space'\| · data.y_err`, `log_jacobian` | none |
| 4. `Term`s, seeing `TermContext(x, y, ym)` | comparison | statistical diagonal; experimental terms (noise, reported modes, USU); model-discrepancy terms (kernel, EFT truncation) | term parameters, and `ym` |
| 5. `Likelihood` | comparison | functional of `y − ym` and Σ | Student-t ν only |

Two rules follow.  Experimental and model-discrepancy covariance terms
are one mechanism; the difference is what the user means, not what the
code does.  A term that reads `ym` sees the prediction *after* the mean
modifications and *after* `space`, which is right: a reported
normalisation error applies to the measured scale, so its mode is
`η · ρ f`, and `Comparison.reported_terms()` gets that for free.  And:
mean-side discrepancy is physical-space and `x`-aware; covariance-side
discrepancy is comparison-space and `ym`-aware; neither sees the other.
That is why the normalisation stays on the mean rather than dividing the
data: dividing the data would make `space` parametric, with a
ρ-dependent Jacobian and ρ-dependent error propagation, and in linear
space it is the data-side normalisation that Peelle's Pertinent Puzzle
warns about (recipe 27).

### 2.8 `problem.py` — the compile step

```python
class Problem:
    def __init__(self, constraints, priors=()):    # priors: [(params, joint), ...]
    index: ParameterIndex                # slot(p), slots(ps), names, bounds, ndim
    constraints: tuple[CompiledConstraint, ...]
    priors: tuple                        # the (params, joint) pairs as given; reuse for a held-out Problem
    ndim: int
    names: list[str]
    bounds: ndarray                      # (ndim, 2)

    def log_prior(self, theta) -> float
    def log_likelihood(self, theta) -> float          # sum_c c.weight * c.log_likelihood(theta)
    def log_posterior(self, theta) -> float           # prior first; -inf short-circuits the forward model
    def prior_transform(self, u) -> ndarray           # unit cube -> theta (dynesty, bbb)
    def sample_prior(self, n, rng=None) -> ndarray    # (n, ndim)
    def predict(self, theta, physical=False) -> list[list[ndarray]]   # per constraint, per block
    def chi2(self, theta) -> float
    def log_jacobian(self) -> float                   # sum over constraints, active points
    def columns(self, params) -> ndarray              # chain columns of these parameters

    # black-box-bayes spellings of the same six things
    NDIM, parameter_names, starting_location(n), log_posterior_batch(thetas)
```

`Problem.__init__` is the only place that walks the graph.  It produces
`index`, the compiled constraints, and the assembled prior.  Compile the
same declarations twice and you get two independent problems; nothing
user-facing is mutated.

```python
def compile(constraints, priors):
    index = ParameterIndex()                         # ordered: Parameter -> slot, first seen
    compiled = []
    for c in constraints:
        y, y_err, offsets = stack(c.comparisons)     # comparison space, one slice per comparison
        active = concatenate(offsets[i][mask_i] for each block)
        preds = [(offsets[i], index.add_all(b.predictor.params), b.predictor) for i, b in ...]
        terms = ([statistical(comp.y_err, on=comp) for comp in c.comparisons] if c.statistical else []) + list(c.terms)
        resolved = [(resolve(t.on, c.comparisons, offsets), index.add_all(t.params), t) for t in terms]
        like_gather = index.add_all(c.likelihood.params)
        cov = StructuredCovariance(resolved, offsets, active, blocks=c.comparisons)
        cov.factor_constant_parts()                  # eager; names the block on failure
        compiled.append(CompiledConstraint(...))
    index.check_names_unique()
    prior = assemble_prior(index, priors)            # every slot covered exactly once
    return index, tuple(compiled), prior
```

`index.add_all(params)` returns the gather array for a node, adding unseen
parameters in first-seen order (block predictors, then terms, then the
likelihood, constraint by constraint).  That is the whole routing story.

**Prior assembly.**  Each slot is covered by its parameter's marginal
(§2.1) or by exactly one joint block from `priors`.  A joint block is
`(params, joint)` where `joint` is any object with `logpdf(values)` over
those parameters in that order, optionally `prior_transform(u)` and
`rvs(n)`; `scipy.stats.multivariate_normal` qualifies.  A hyperprior is a
joint block that *includes its hyperparameter*: the children carry no
marginal, the block's `logpdf` is `sum_i log p(child_i | hyper) + log p(hyper)`,
and its `prior_transform` draws the hyperparameter first (recipe 24).  `log_prior` sums
marginal log-densities and joint `logpdf`s.  `prior_transform` maps each
marginal slot through its rescaled `ppf`; a joint `scipy.stats.multivariate_normal`
block is whitened, `theta = mu + L Φ⁻¹(u)` with `L Lᵀ = cov`; any other joint
must expose `prior_transform(u)` or `Problem.prior_transform` raises a clear
error naming it.  A parameter with finite `bounds` inside a joint block is
truncated in `log_prior` only (`-inf` outside the bounds); a truncated joint
has no unit-cube map, so `prior_transform` raises for that problem and names
the parameter.  `sample_prior` draws marginals and joints and scatters them
into columns.

**`CompiledConstraint`** is today's `Constraint` plus `ConstraintCovariance`
minus the parameter splitting:

```python
class CompiledConstraint:
    y, y_err, x: ndarray                 # stacked, comparison space, all points
    offsets: tuple[slice, ...]
    active: ndarray
    predictors: list[(slice, gather, Predictor)]
    covariance: StructuredCovariance
    likelihood: Likelihood; like_gather: ndarray
    weight: float
    log_jacobian: float
    def ym(self, theta) -> ndarray
    def log_likelihood(self, theta) -> float
    def chi2(self, theta) -> float
    def matrix(self, theta) -> ndarray   # dense, active rows; display and tests
```

### 2.9 `covariance.py` — the structured covariance

The three term kinds already describe a decomposition:

```
Σ = diag(D) + blockdiag(M_b) + U Uᵀ        (+ dense fallback)
```

- `D`: length-`N` vector, the sum of squares of every `"diag"` term.
- `M_b`: one dense block per observation block, the sum of `"matrix"`
  terms whose support lies inside block `b`.
- `U`: `N × r`, one column per `"mode"` term, scattered onto its support
  and zero elsewhere.  A mode spanning several blocks (case A) is a column
  with entries in several blocks.
- A `"matrix"` term that crosses block boundaries (a GP kernel over the
  union of two datasets) forces the dense path for that constraint.

With `B = blockdiag(M_b + diag(D_b))` and per-block Cholesky `L_b`:

```
z  = L⁻¹ r                  (block by block)
W  = L⁻¹ U                  (block by block, N × r)
S  = I_r + Wᵀ W             (r × r)
d2 = zᵀz − (Wᵀz)ᵀ S⁻¹ (Wᵀz)
logdet Σ = Σ_b logdet B_b + logdet S
```

Cost `O(Σ_b n_b³ + N r² + r³)` instead of `O(N³)`.  Consequences:

- All three "known limitations" of `design.md` are gone: no dense assembly
  for block-diagonal covariances, a low-rank path exists, and masking is
  slicing rows out of `D`, `U`, and `M_b` before assembly.
- The block path is the only path.  `uses_block_path`, `block_diagonal`,
  `cholesky` versus `block_cholesky`, and the dispatch in
  `stacked_distance` do not exist.
- Each of `D`, `U`, `M_b` is a constant part (evaluated once at compile)
  plus a parametric part.  When everything is constant the factors are
  cached.
- A constraint boundary is now about the likelihood functional and the
  weight, not about cost: a coupling mode across blocks is essentially free.
- `B` must be positive definite.  A block covered only by modes is singular
  in `B` even when `Σ` is not; compile reports it with the comparison's label
  and the remedies (reported terms, a noise term, `statistical=True`).

```python
class StructuredCovariance:
    def __init__(self, resolved_terms, offsets, active, blocks): ...
    def factor_constant_parts(self) -> None
    def distance(self, y, ym, theta) -> tuple[float, float]     # (d2, logdet) on active rows
    def matrix(self, y, ym, theta) -> ndarray                    # dense, active rows
```

### 2.10 `diagnostics.py` and `predictive.py`

Harvested statistics with the entry points retargeted to `(problem,
samples)`, where `samples` has shape `(n, ndim)` in `problem.index` order.
That is what emcee's `get_chain(flat=True)`, dynesty's `samples_equal()`,
and bbb's `InferenceData["theta"]` all give.  Column selection goes through
`problem.columns(...)`; `split_samples`, `n_model_params`, `theta_cols` are
gone.

```python
# diagnostics.py
predictive_draws(problem, samples, constraint=0, *, terms=None, statistical=True, n_rep=1, rng=None,
                 model_only=False, given=None, levels=(16, 50, 84), return_draws=False)
coverage_curve(draws, y, levels=None); coverage_error(draws, y, levels=None)
sharpness(draws, percentiles=(16, 84), transform=None)
heldout_log_predictive(heldout_problem, samples, *, given=None)   # Problem([fit.complement()])
# given=: the fitted problem.  A held-out problem's own likelihood is the marginal of its
#         rows, wrong when a term spans fit and held-out rows (a GP over experiments);
#         given= computes the Gaussian conditional p(y_held | y_fit, theta) under the full
#         covariance, which equals the marginal when nothing spans.
log_posterior_predictive(logp_samples, logw=None)
logz_summary(logz, logzerr); compare_logz(a, b, sigma=2.0)

# predictive.py
gp_posterior_predictive(kernel, theta, X_train, residuals, X_pred, *, train_noise_var=None, jitter=1e-10)
predictive_band(draws, levels=(16, 50, 84))
grid_draws(problem, predictor, x_pred, samples, constraint=0, *, comparison=None, terms=None,
           model_only=False, joint=True, physical=False, n_rep=1, rng=None,
           levels=(16, 50, 84), return_draws=False)
gp_predictive_draws(problem, term, predictor, x_pred, samples, *, terms=None, conditioned=False,
                    joint=True, noise_std=0.0, train_noise_var=None, physical=False,
                    n_rep=1, rng=None, levels=(16, 50, 84), return_draws=False)
# term is the KernelTerm; its columns and the predictor's come from problem.columns.  The
# conditioning noise defaults to the constraint's covariance minus the kernel block; the
# band is in the comparison space of the term's comparisons unless physical=True.
```

## 3. Worked example: the α+Ca study shape

Two datasets without reported errors, compared in log space, one noise
magnitude shared between them (case B), one normalisation mode coupling them
(case A), one latent scale per dataset, one held out below an angular cut,
nested sampling, then held-out scoring.

```python
import numpy as np, dynesty, rxmc as rx
from rxmc import terms as T, transforms as tf
from scipy import stats

d1 = rx.from_measurement(m1, reaction=reaction, quantity="dXS/dA")
d2 = rx.from_measurement(m2, reaction=reaction, quantity="dXS/dA")

omp = rx.reactions.ElasticXS("dXS/dA", central, spin_orbit, args_from_params, params=omp_params)
rho1 = rx.Parameter("log_rho_1", prior=stats.norm(0, 0.1))
rho2 = rx.Parameter("log_rho_2", prior=stats.norm(0, 0.1))
comp1 = rx.Comparison(d1, omp | tf.scale(rho1), space=tf.log)
comp2 = rx.Comparison(d2, omp | tf.scale(rho2), space=tf.log)

log_eps = rx.Parameter("log_eps", prior=stats.norm(-2, 1))
log_eta = rx.Parameter("log_eta", prior=stats.norm(-2, 1))
c = rx.Constraint(
    comparisons=[comp1, comp2],
    terms=[
        T.noise(log_eps, on=comp1),              # case B: one magnitude, two comparisons
        T.noise(log_eps, on=comp2),
        T.normalization(log_eta, on=[comp1, comp2]),   # case A: one mode across both
    ],
    likelihood=rx.StudentT(),
    statistical=False,                           # no reported errors: the noise term is the diagonal
)
fit = c.masked_where(lambda x: x < cut)
held = fit.complement()

problem = rx.Problem([fit], priors=[(omp.params, stats.multivariate_normal(mu, cov))])
sampler = dynesty.NestedSampler(problem.log_likelihood, problem.prior_transform, problem.ndim)
sampler.run_nested()
res = sampler.results
samples = res.samples_equal()

heldout = rx.Problem([held], priors=problem.priors)          # same Parameter objects
lp = rx.diagnostics.heldout_log_predictive(heldout, samples)
score = rx.diagnostics.log_posterior_predictive(lp)
logz = rx.diagnostics.logz_summary(res.logz[-1], res.logzerr[-1])
# to compare with a linear-space fit of the same data: logz_raw = logz + problem.log_jacobian()
```

The same problem under emcee:

```python
import emcee
p0 = problem.sample_prior(32, rng=np.random.default_rng(1))
sampler = emcee.EnsembleSampler(32, problem.ndim, problem.log_posterior)
sampler.run_mcmc(p0, 5000)
samples = sampler.get_chain(discard=1000, thin=10, flat=True)
band = rx.predictive.predictive_band(
    [omp.bind(x_fine, d1.meta)(*s[problem.columns(omp.params)]) for s in samples[::20]]
)
```

And under `black-box-bayes`, which needs a pickle and a six-line module:

```python
# make_problem.py
import dill
with open("problem.pkl", "wb") as f:
    dill.dump(problem, f)

# posterior.py
import dill
def init_posterior(path):
    global P, NDIM, parameter_names
    P = dill.load(open(path, "rb")); NDIM = P.ndim; parameter_names = P.names
def starting_location(n):  return P.sample_prior(n)
def log_posterior(theta):   return P.log_posterior(theta)
def log_likelihood(theta):  return P.log_likelihood(theta)
def prior_transform(u):     return P.prior_transform(u)
```

Compare with the current spelling of the same study: `Observation(...,
transform=log)` per dataset, `stacked_supports` to place the terms,
`per_observation_scaling([obs1, obs2])` with `obs` passed to three places,
`Constraint(...)`, `Evidence([...])`, `ParameterConfig` twice with a
`TruncatedNormalPrior` because the MVN prior has no unit-cube map,
`CalibrationConfig`, and `split_samples` to read the chain back.

## 4. Capability map

Every statistical capability a current notebook or test demonstrates, and
its spelling in the new API.  This table is the acceptance list for the
rewrite: a capability is done when its row has a test.

| capability (where it is demonstrated today) | new spelling | pinned by |
|---|---|---|
| user-defined model `y = m x + b` (linear_calibration_demo) | `Model(lambda x, m, b: m*x + b, [m, b])` | test_model |
| `Polynomial(order)` (normalization_inference) | `polynomial(order)` | test_model |
| statistical diagonal only; `chi2 / n` (linear_calibration_demo) | default `Constraint`; `problem.chi2(theta)` | test_problem |
| unknown fractional / constant noise (systematic_err_demo, sampling_algos) | `noise_fraction(log_eps)`, `noise(log_eps)` | test_terms::TestFactories |
| inferred noise *replacing* reported statistics (prose today) | `Constraint(statistical=False, terms=[noise(...)])` | test_constraint |
| reported normalisation / offset as fixed modes (measurement_to_calibration) | `block.reported_terms()`; `normalization(magnitude=)`, `offset(magnitude=)` | test_constraint::reported_terms, regression number |
| free normalisation / offset nuisance (systematic_err_demo) | `normalization(parameter=log_eta)`, `offset(parameter=log_omega)` | test_terms |
| fixed dense covariance; fixed diagonal (systematic_err_demo, normalization_inference gallery) | `Term(C, on=b)`, `Term(sig, kind="diag", on=b)` | test_terms::TestTermKinds |
| case B: one parameter, two block-local terms (systematic_err_demo, correlated_observations) | `noise_fraction(log_eps, on=b1), noise_fraction(log_eps, on=b2)`; or two constraints sharing `log_eps` | test_problem::sharing |
| case A: one mode across blocks (correlated_observations) | `normalization(log_eta, on=[b1, b2])` | test_covariance::case_a, regression |
| per-dataset Kennedy–O'Hagan scale ρᵢ (normalization_inference) | `Comparison(d_i, omp \| scale(rho_i))` | test_model, test_constraint |
| single global ρ (test only) | `omp \| scale(rho)` on every block | test_model |
| sampled mean discrepancy `y = f(x; θ) + δ(x; φ)` (new) | `omp + Model(delta_fn, phi)`; alone or alongside `kernel` | test_model (params order, shared parameter, `\|` precedence) |
| multiplicative `x`-dependent correction, i.e. an additive discrepancy in log space (new) | `omp * Model(g_fn, phi)`; `scale(rho)` is the constant case | test_model (`omp * const(rho)` equals `omp \| scale(rho)`) |
| hyperparameters shared across datasets, per-dataset values from `meta` (new) | one term per block with the same `Parameter` objects; `c.meta("Elab")`; `kernel(params=)` | test_terms (`meta` on one block and on a union), test_problem (one slot per shared object) |
| discrepancy correlated across energies: GP over (E, θ) (new) | one `matrix` `Term` with `on=comps` building inputs from `c.meta` and `c.x`; dense path | test_covariance (dense fallback equals hand-built product kernel) |
| unaccounted-for model error per data type, KDUQ (new, reference) | `model_error(delta_T, averaging=True, on=b)` with one `delta_T` per type; the `k/N` democratic and per-type federal scalings are `Constraint(weight=)`; recipe 26 | test_terms (shared object gives one column per type) |
| Peelle's Pertinent Puzzle avoidance (new, reference) | `normalization()` reads `c.ym`; the `t0` variant as a constant `mode`; recipe 27 | test_terms (data-built mode reproduces the `1/(1+n s²)` bias; prediction-built does not) |
| stacking by leave-one-dataset-out (new, reference) | `Constraint.masked` dropping a block, `heldout_log_predictive`; recipe 28 | test_diagnostics |
| cut / modular posterior by multiple imputation (new, reference) | stage-1 `Problem`, per-draw stage-2 `Problem` with the module fixed by closure; per-module `weight`; recipe 29 | test_problem (stage-1 marginal unchanged) |
| leave-one-experiment-out prediction (new, reference) | block masks, `complement`, `predictive_draws`, `coverage_curve`; recipe 30 | test_diagnostics |
| simulation-based calibration of the sampler (new, reference) | `sample_prior`, `predictive_draws`, `dataclasses.replace(d, y=)`; recipe 31 | test_problem (rank uniformity on the linear problem) |
| emulator as `Model`, emulator variance as a `diag` term sharing the model's parameters (new, reference) | recipe 32 | test_terms (a term declaring model parameters receives them) |
| MAP + Laplace (new, reference) | `scipy.optimize` on `log_posterior`, `problem.bounds`; recipe 33 | test_problem |
| global error scale and USU modes (new, reference) | `diag` term scaling `c.meta("y_err")` with `statistical=False`; `offset(parameter=, on=blocks_of_technique)`; recipe 34 | test_terms |
| energy-dependent parameters (new, reference) | per-block `Model` instances closing over `meta`, shared coefficient objects; recipe 35 | test_model |
| discrepancy on a physical basis, Legendre (new, reference) | `systematic` modes or `omp + Model(basis_sum)`; recipe 36 | test_terms |
| correlated normalisations between quantities of one experiment, Peelle's puzzle in more than one dimension (new, reference) | one comparison per quantity, one constraint, a spanning `matrix` term built from `c.split(c.ym)`; recipe 37 | test_covariance |
| classic normal hierarchical model, BDA3 ch. 5 (new, reference) | marginalised as `noise(log_tau)`, non-centred as a `Model` over `[mu, log_tau, *etas]`, centred as a joint block; recipe 38 | test_problem (marginalised and non-centred agree on `mu, tau`; a parameter on a fully masked block is sampled from its prior) |
| SafeBayes: learn the tempering exponent (new, reference) | driver loop over `replace(c, weight=η)` and `c.masked(prefix)`; next-point density as a log-likelihood difference; recipe 25 | test_problem (`replace` keeps names; `ll(prefix i+1) − ll(prefix i)` equals the Gaussian conditional) |
| hyperprior: per-dataset parameters with a sampled spread (new) | joint block `(children + [hyper], obj)` with `logpdf` and `prior_transform` | test_problem (children uncovered without the block; `prior_transform` round trip) |
| GP discrepancy in x / in momentum transfer with amplitude (gp_discrepancy, TestStudyForms) | `kernel(k, on=b, coords=lambda x: momentum_transfer(x, k), amplitude=..., amplitude_params=...)` | test_terms::TestStudyForms |
| angle-growing noise, mode ∝ θ, `exp_growth`, `x_basis` (TestStudyForms) | same helpers with `on=` | test_terms::TestStudyForms |
| direct two-parameter `Term` escape hatch (TestStudyForms) | `Term(lambda c, e, l: ..., (e, l), kind="diag")` | test_terms |
| Student-t with bounded ν; `Chi2` (robust_likelihoods) | `StudentT(nu=Parameter("nu", bounds=(1, 100)))`; `Chi2()` | test_likelihood closed forms |
| log-space comparison, delta-method errors, `log_jacobian` (tests) | `Comparison(d, m, space=log)`; `problem.log_jacobian()` | test_constraint::TestComparisonSpace |
| masks, `masked_where`, `complement`, held-out scoring (tests) | `Constraint.masked_where`, `.complement()`; `heldout_log_predictive` | test_constraint::TestMask (partition, `ll(fit)+ll(held)==ll(full)`) |
| tempering, two spellings (overconfidence) | `Constraint(weight=)` only | test_problem |
| joint MVN prior over the optical set (most notebooks) | `Problem(..., priors=[(omp.params, mvn)])` | test_problem::priors |
| truncated-normal / bounded independent priors (calibration_config_emcee_dynesty) | `Parameter(prior=stats.norm(...), bounds=(lo, hi))` | test_problem::priors |
| nested-sampling `prior_transform`, now including joint MVN | `problem.prior_transform` | test_problem (round trip, finite at 0 and 1) |
| emcee, dynesty, black-box-bayes drivers | flat interface on `Problem`; dill round trip | test_problem::drivers |
| EXFOR → dataset with unit conversion; per-angle Rutherford `norm` (measurement_to_calibration, tests) | `from_measurement(m, reaction=, quantity=)` | test_data (both conversion directions) |
| elastic `dXS/dA`, `dXS/dRuth`, `Ay` | `ElasticXS(quantity, ...)`; a compound-elastic contribution is subtracted from the data as preprocessing | test_reactions |
| (p,n) IAS channel (tests) | `IsobaricAnalogPN(...)` | test_reactions (Lane term) |
| solver settings reach jitr | `ElasticXS(..., lmax=, wavelengths_beyond_range=, zeros_per_node=)` | test_reactions |
| singular-covariance guard naming the dataset (measurement_to_calibration) | compile-time in `Problem` | test_problem |
| duplicate-name / same-object validation | `ParameterIndex.check_names_unique`; sharing across constraints is legal | test_problem |
| covariance heat-maps (normalization_inference, correlated_observations) | `problem.constraints[i].matrix(theta)` | test_covariance dense equality |
| predictive draws, coverage, sharpness, evidence bookkeeping (tests) | `diagnostics.*` | test_diagnostics |
| GP conditioning; total predictive band (gp_discrepancy) | `predictive.*` | test_predictive vs sklearn |
| fine-grid plotting (every reaction notebook) | `omp.bind(x_fine, d.meta)(*s[problem.columns(omp.params)])` | test_model |

Four rows are new rather than ported.  The sampled mean discrepancy: today a
model-side correction that depends on `x` cannot be written, because the
model transform sees only the prediction.  The three hierarchical rows:
today a term cannot read its dataset's metadata, `kernel_term` cannot share
hyperparameters between calls, and a prior cannot depend on a sampled
hyperparameter.

**Dropped, deliberately:** `Walker`, `MetropolisHastingsSampler`,
`AdaptiveMetropolisSampler`, `BatchedAdaptiveMetropolisSampler`, `proposal`,
`marginal_log_likelihood`, `conditional_posterior`, `predict_parametric`,
`weighted_marginal_log_likelihood`, `CalibrationConfig`, `ParameterConfig`,
`Evidence`, `per_observation_scaling`, `identity` keys, `stacked_supports`,
`split_samples`, `likelihood_scaling`, `IndependentPrior`,
`TruncatedNormalPrior`, the two `Observation` subclasses, `n_dof`.

## 5. Harvest table

What comes across from `src/rxmc` on `api_generalisation`, by file.

### Lift verbatim (about 1,100 lines)

| current file | pieces | destination |
|---|---|---|
| `transforms.py` | `Transform` (minus `contextual`, `_unpack`), `as_transform`, `identity`, `log`, `exp`, `_safe_log`, `_reciprocal`, `scale` | `transforms.py` |
| `likelihood_model.py` | `Likelihood`, `GaussianLikelihood`→`Gaussian`, `StudentT`, `Chi2`, `log_likelihood` | `likelihood.py` |
| `covariance.py` | `TermContext`, `chol_logdet`, `as_2d`, bases `ones`, `ym`, `averaging`, `x_basis`, `exp_growth`, `constant_amplitude`, `exp_growth_amplitude`; helpers `_masked`, `_full`, `_coefficient`, `_scaled_term`, `_kernel_params`; factories `statistical_term`, `offset_term`, `normalization_term`, `noise_term`, `noise_fraction_term`, `model_error_term`, `systematic_term`, `kernel_term` (drop the `_term` suffix, `support=`→`on=`) | `terms.py` |
| `observation_from_measurement.py` | `XS_UNIT`, `RUTHERFORD_UNIT`, `MB_PER_B`, `DEFAULT_LMAX`, `check_angle_grid`, `measurement_kwargs`; the pint registry is replaced by a fixed label table | `units.py`, `data.py` |
| `elastic_diffxs_observation.py` | `set_up_solver`, the `calculate_normalization` conversion table, `momentum_transfer` | `reactions/elastic.py`, `data.py` |
| `ias_pn_observation.py` | `set_up_solver` | `reactions/ias.py` |
| `elastic_diffxs_model.py` | `_xs` body, `extract_dXS_dA`, `extract_dXS_dRuth`, `extract_Ay` | `reactions/elastic.py` |
| `ias_pn_model.py` | `_xs` body | `reactions/ias.py` |
| `predictive.py` | `gp_posterior_predictive`, `_gp_condition`, `_gp_posterior_mean_var`, `_train_noise_matrix`, `predictive_band` | `predictive.py` |
| `model_comparison.py` | `_psd_factor`, `_rows`, `coverage_curve`, `coverage_error`, `sharpness`, `log_posterior_predictive`, `logz_summary`, `compare_logz` | `diagnostics.py` |
| `priors.py` | `clip_unit_cube` | `problem.py` |

### Lift with edits

| current | change |
|---|---|
| `Observation.systematic_terms` | → `Comparison.reported_terms`; same delta-method logic (offset at `y_raw`, normalisation at `ym_raw`), `on=self` |
| `Observation.__init__` transform handling and `_check_finite` | → `Comparison.__post_init__`; error names `data.label` |
| `Constraint._validate_constant_covariance` message | → compile error raised by `StructuredCovariance.factor_constant_parts`, remedies updated to the new spellings |
| `model_comparison.predictive_draws`, `heldout_log_predictive` | take `(problem, samples)`; read `CompiledConstraint.ym`, `.matrix`, `.log_likelihood` |
| `predictive.total_predictive_band` (now `grid_draws` and `gp_predictive_draws`) | take `(problem, term, predictor, ...)`; columns from `problem.columns` |
| `ParameterConfig.prior_transform` cursor | → per-slot map in `assemble_prior` |
| `ElasticDifferentialXSObservation.from_measurement`, `IsobaricAnalogPNObservation.from_measurement` | one free `from_measurement`; Rutherford from kinematics |
| `PhysicalModel.Polynomial` | → `polynomial(order)` factory |

### Rewrite

`params.py` (identity semantics, `prior`), the `Term` class body (state
removed), `ConstraintCovariance` → `StructuredCovariance`, `constraint.py`,
`evidence.py` + `config.py` → `problem.py`, `physical_model.py` →
`model.py`, the two observation classes → `from_measurement` + `Model.bind`.

### Drop

`walker.py`, `param_sampling.py`, `metropolis_hastings.py`,
`adaptive_metropolis.py`, `proposal.py`, `IndependentPrior`,
`TruncatedNormalPrior`, `as_prior`.

### Tests to port by body

The bodies below encode behaviour, not API, and port with renamed calls:

- `test_covariance.py`: `TestTermKinds`, `TestTermCoords`, `TestFactories`
  (including `test_old_observation_covariance_equivalence`),
  `TestKernelTerm`, `TestStudyForms` (every form of the α+Ca error-model ladder, whose legend is the table in recipe 18, against
  a hand-built dense matrix), `test_custom_term_direct`.
- `test_likelihood_model.py`: the closed-form Student-t and `Chi2` values.
- `test_constraint.py`: `TestComparisonSpaceTransform` (delta method,
  `log_jacobian == -Σ log y`, `-inf` likelihood and `+inf` chi2 for a
  non-positive prediction), `TestMask` (complement partition,
  `ll(fit) + ll(held) == ll(full)`, shared parameters), `TestSingularCovarianceGuard`,
  `TestSharedParameterCaseB`, `TestStackedConstraint` (case A differs from
  the independent spelling).
- `test_observation.py`: `test_systematic_terms_propagated_by_delta_method`,
  offset-then-normalisation order, zero magnitudes skipped.
- `test_regression.py`: all three pins, including `1.195784087817536`.
- `test_model_comparison.py`, `test_predictive.py` (GP matches sklearn's
  `GaussianProcessRegressor`).
- `test_reaction_observation.py`: both Rutherford conversion directions,
  `TestSolverSettingsForwarding`, `TestSharedUnits`.
- `test_reaction_models.py`: finite non-negative cross sections through
  real jitr solves; the Lane-term note for the IAS test.
- `test_config.py::test_black_box_bayes_interface`,
  `test_priors.py::TestUnitCubeClipping`.

New tests the old suite could not have: dense-versus-structured equality on
every `TestStudyForms` form, with masks, with a cross-block kernel (dense
fallback), and with a mode-only block (singular `B`); a parameter shared
across two constraints; a `dill` round trip of an elastic `Problem` with
equal `log_posterior`; MVN `prior_transform` round trip and finiteness at
exactly 0 and 1; `Constraint.weight` scaling the likelihood only.

The ported bodies are unit tests of modules.  The recipe tests under
`test/recipes/` are the acceptance suite, and §9 lists which recipes each
milestone unlocks.  Recipe tests use synthetic data and the generic `Model`
wherever the recipe does not require a reaction model, so they stay fast;
reaction recipes patch `set_up_solver` as the current
`test_reaction_observation.py` does, except for one real-solve smoke test.

## 6. Repository layout and budget

```
rxmc/
  __init__.py            re-exports; nothing else
  params.py       ~40    transforms.py   ~230   units.py         ~60
  data.py        ~180    model.py        ~120   terms.py        ~380
  likelihood.py   ~90    constraint.py   ~170   covariance.py   ~260
  problem.py     ~320    diagnostics.py  ~250   predictive.py   ~200
  reactions/
    __init__.py
    elastic.py   ~170    ias.py          ~120
test/            unit tests, one file per module, plus test_regression.py
test/recipes/    one file per recipe in docs/recipes.md (~38), the acceptance suite;
                 oracle.py holds the closed-form linear-Gaussian posterior used by the fast tier
examples/        9 notebooks (§7)
docs/            design.md rewritten from this document once the code lands
```

Runtime dependencies: `numpy`, `scipy`, `jitr>=3.0`,
`exfor-tools`.  `pandas` and `scikit-learn` leave `requirements.txt`
(neither is imported; kernels stay duck-typed and sklearn moves to the
`examples` extra).  Extras: `examples` (emcee, dynesty, corner, matplotlib,
scikit-learn, dill, jupyter), `validation` (examples + pytest, nbmake,
ruff, black, isort).  Python ≥ 3.12.  Validation is the current contract:
ruff/black/isort on `src test`, nbqa on `examples`, `pytest test`, then
`pytest --nbmake examples`.  `pyproject` registers the `slow` marker and
deselects it by default (`addopts = -m "not slow"`, §9).

## 7. Notebooks

Nine notebooks, each naming the current one it inherits.  Every notebook
is driven by emcee or dynesty.

| notebook | inherits | driver | new content | runtime |
|---|---|---|---|---|
| `linear_calibration` | linear_calibration_demo | emcee | prior predictive, posterior, predictive band with `problem.columns`, the coverage curve | 23 s |
| `error_models` | systematic_err_demo | emcee | the ladder on one comparison, the Peelle matrix as a fixed `Term`, offsets known and free; two-constraint section with case B via a shared `Parameter` | 153 s |
| `normalization_and_covariance_structure` | normalization_inference | emcee | ρᵢ as `quartic \| tf.scale(rho_i)` against `reported_terms()`; the four-case gallery via `matrix(theta)` | 328 s |
| `correlated_observations` | correlated_observations | emcee | case A vs B on the toy; Neudecker et al. (2014) §II.A and §II.B recreated: the multi-quantity Peelle puzzle with a spanning `matrix` term built through `c.split` | 141 s |
| `gp_discrepancy` | gp_discrepancy | emcee (toy), dynesty (reaction) | `kernel` term; `gp_predictive_draws(problem, term, ...)`; the same defect fit with a sampled Legendre mean correction for contrast; n+⁴⁰Ca with the surface absorption missing | 617 s |
| `robust_likelihoods` | robust_likelihoods | emcee | Student-t vs Gaussian; ν bounded on the `Parameter`; a global error scale and a USU offset per technique | 154 s |
| `measurement_to_calibration` | measurement_to_calibration + 30s_optical_potential_calibration + the tempering/coverage section of overconfidence | dynesty | `from_measurement`, `reported_terms`, the singular-covariance error, `Constraint(weight=)`, `coverage_curve`, emcee and `dill` as other drivers, the KDUQ `model_error` spelling | 182 s |
| `alpha_ca_error_model_comparison` | **new** (the `jitr` quickstart's α+⁴⁴Ca data, EXFOR F0567) | dynesty | real data without errors, a four-parameter potential, log space with `log_jacobian`, the `L0`/`E0`/`L2y`/`Lgp` ladder by evidence, `masked_where`/`complement` with `heldout_log_predictive` and held-out coverage | 1197 s (alongside another notebook) |
| `hierarchical_calibration` | **new** (recipes 24, 35, 38) | dynesty | eight schools non-centred; the hierarchy on the physics parameters; see below | 937 s (alongside another notebook) |

Runtimes are single-process wall times on an eight-core laptop with the
kernels run one at a time; the converged-tier workflow runs four at once
with a 40-minute timeout each.  The reaction notebooks are driven by
dynesty because emcee mixes poorly on optical-model posteriors.

**`hierarchical_calibration` in detail.**  The truth is
`y = a0(E) + a1(E) x + a2(E) x²`, measured by J = 7 synthetic datasets at
known energies `E_j` (in `meta`) plus one held-out dataset at a new energy
bracketed by two fitted ones (a hierarchy learns the spread of deviations
it has seen; an unmodelled peak between its datasets is the few-datasets
caveat below, not a prediction it can make).  The
true coefficient mappings `a_k(E)` are a smooth trend plus non-monotonic
bumps.  Three fits of the same data:

1. the correct mapping form with global `φ`;
2. a misspecified smooth mapping `a_k(E; φ) = φ_k0 + φ_k1 E` with global `φ`;
3. the same smooth mapping plus a per-dataset deviation vector in parameter
   space, non-centred `δ_j = τ ⊙ η_j`, `η_jk ~ N(0, 1)`,
   `τ_k ~ HalfNormal`, so the inter-dataset covariance is diagonal with a
   learned spread (a full covariance through an LKJ-style joint block is
   the named extension).

Mechanics: one `Model` per block closing over `E_j` (recipe 35).  The
held-out block is fully masked, so its `η_new` is sampled from the prior
only, driven by `τ`; `complement()`, `predictive_draws` and
`heldout_log_predictive` then score the new energy with no extra code.  The
notebook states this design property explicitly.  Plots: the coefficient
mappings against `E` with the truth; per-dataset residuals for case 2;
empirical coverage curves in-sample and on the held-out energy for all
three cases; sharpness; the posterior of `τ`; the held-out log predictive
per case.  Expected: case 1 covers in and out of sample; case 2
under-covers both and shows structured per-dataset residuals; case 3
recovers coverage with wider, longer-tailed bands at the new energy (a
scale mixture over `τ`), a `τ` posterior away from zero, and the best
held-out score of the misspecified pair.  Caveat stated in the notebook:
with few datasets `φ` and `δ_j` trade off, and the hyperprior on `τ` is
what resolves it.

Dropped: `sampling_algos` (in-package samplers), `calibration_config_emcee_dynesty`
(every notebook is now this), `overconfidence` as a standalone.

## 8. Gaps: what exists nowhere today

- **G1 `ParameterIndex` and compile** (`problem.py`): first-seen slot
  assignment, `on=` resolution against comparisons and datasets, `masks` →
  `active`, name uniqueness, prior coverage.  About 120 lines.
- **G2 Prior assembly**: marginal truncation to `bounds`, uniform default,
  the joint-block protocol including hyperprior blocks,
  joint blocks, `log_prior`, `prior_transform` (rescaled `ppf`; MVN
  whitening; other joints must supply their own), `sample_prior`.  About
  100 lines.
- **G3 `StructuredCovariance`**: new linear algebra.  Verified against the
  dense matrix on every `TestStudyForms` case, with masks, with a
  cross-block kernel, and with a mode-only block.  About 260 lines.
- **G4 Stateless `Term`, `Comparison`, `Constraint` masks**: including
  `reported_terms` under a comparison transform and the complement
  partition property.
- **G5 Rutherford in `from_measurement`**: resolved, it is a closed form of
  `reaction.kinematics(Elab)`; no workspace needed.
- **G6 `dill` picklability**: a test round-trips an elastic `Problem` and
  compares `log_posterior`.  If a jitr workspace does not pickle,
  `Predictor.__reduce__` rebuilds it from `(model, x, meta)`; the factory
  closures in `terms.py` pickle under `dill` as they are.
- **G7 Analysis on `(problem, samples)`**: `predictive_draws`,
  `heldout_log_predictive`, `grid_draws`/`gp_predictive_draws` selecting columns via
  `problem.columns`.
- **G8 The α+Ca and hierarchical notebooks** and the bbb `posterior.py` shim.
- **G9 Docs**: `design.md` rewritten from this document; API reference
  regenerated; README quickstart in the new spelling.

## 9. Milestones

Each step is testable on its own in the new repository.  Every milestone
lists the modules it delivers, the unit tests it ports, and the recipe
tests it unlocks: a recipe test lands in the first milestone where every
object it uses exists.  A milestone is done when its unit tests and every
recipe test it unlocks pass.

0. **Bootstrap.**  A branch `rewrite` of *this* repository, cut from
   `main` after the 0.x close-out (§11), developed in a git worktree at
   `~/el/rxmc-ng` with its own virtual environment (`jitr>=3.0` from PyPI,
   Python ≥ 3.12).  Its first commit removes `src/`, `test/` and
   `examples/` and keeps `docs/` (this document and `recipes.md` are the
   plan of record) and the configuration.  Files that move mostly intact
   (`transforms`, `likelihood`, `units`, the term factories) come over by
   `git mv` in a commit of their own before any edit, so blame survives
   the harvest.  Then: a `pyproject` in the current shape with the `slow`
   marker registered and deselected by default; ruff, black and isort
   configuration carried over; the CI workflow (fast tier on pull
   requests, `pytest -m slow` and nbmake in the converged workflow that
   gates `main`); a trusted-
   publishing workflow that uploads to PyPI on tag push.
1. **`params`, `transforms`, `units`, `likelihood`, `terms`** (verbatim
   harvest plus the stateless `Term`).  Ported: `TestTermKinds`,
   `TestTermCoords`, `TestFactories`, `TestKernelTerm`, `TestStudyForms`
   (term values against hand-built matrices), the closed-form Student-t
   and `Chi2`, `TestUnitCubeClipping`, `TestSharedUnits`.  Recipe tests
   unlocked: none end to end (there is no `Problem` yet).  The term-level
   halves of recipes 19 (fixed-array shape and symmetry checks, a custom
   callable) and 27 (a `normalization` mode evaluates from `c.ym`, a
   data-built mode from `c.y`) are written here as unit tests and reused
   by those recipe tests later.
2. **`data`** (without `from_measurement`), **`model`** (generic and
   `polynomial`), **`constraint`** (`Comparison`, `Constraint`, masks,
   `reported_terms`).  Ported: `TestComparisonSpaceTransform` at block
   level, delta-method `reported_terms`, `TestMask` construction and the
   `complement` partition of active sets, name and shape validation.
   Recipe tests unlocked: none (no likelihood without the covariance).
3. **`covariance`.**  New: dense-versus-structured equality on every
   `TestStudyForms` form, with masks, with a cross-block kernel (dense
   fallback), and with a mode-only block (singular `B` error).  No recipe
   tests yet.
4. **`problem`**: index, compile, priors, flat interface; emcee and
   dynesty smoke tests on the linear problem; `dill` round trip.  The
   acceptance suite starts here, all on synthetic data with generic
   models.  Recipe tests unlocked:
   - core: 1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 16, 19, 20, 21;
   - hierarchy and sharing: 22, 23, 24, 38;
   - driver loops needing only `log_likelihood`, `log_prior`,
     `sample_prior`: 25 (SafeBayes), 26 (KDUQ weights), 27 (Peelle), 29
     (cut posterior), 32 (an emulator term sees the model's parameters),
     33 (MAP and Laplace), 34, 35, 36.

   Regression pins carried here: `1.195784087817536`, the old-covariance
   equivalence, `ll(fit) + ll(held) == ll(full)`, and "a parameter on a
   fully masked block is sampled from its prior", which recipe 38 and the
   hierarchical notebook rely on.
5. **`reactions/elastic`, `reactions/ias`, `from_measurement`.**  Ported:
   both Rutherford conversion directions, `TestSolverSettingsForwarding`,
   real-solve smoke tests, the Lane-term IAS check.  Recipe tests
   unlocked: 3, 14, 15, 37; the reaction variants of 7 and 22
   (momentum-transfer coordinates, an energy-running amplitude) run
   against a patched solver.
6. **`diagnostics`, `predictive`.**  Ported: GP-versus-sklearn,
   `predictive_draws` covariance recovery, `heldout_log_predictive`,
   `logz_summary` and `compare_logz`.  Recipe tests unlocked: 7
   (`gp_predictive_draws` finds the kernel columns itself), 17, 18, 28,
   30, 31.
7. **CI wiring.**  The heading-to-file check between `recipes.md` and
   `test/recipes/` is itself a test (`test/test_recipes_index.py`: one
   file per heading and vice versa, each file's docstring quoting its
   recipe, the recipe-18 legend equal to the tests' legend), so bare
   `pytest` runs it with both suites; the fast tier must finish in a few
   minutes on a laptop (patched solvers, small `J` and `n`), and CI lists
   its ten slowest tests.
8. **Notebooks 1–9.**  Each notebook names the recipes it is the tutorial
   for: `linear_calibration` (1, 17); `error_models` (2, 4, 5, 19);
   `normalization_and_covariance_structure` (3, 6, 27);
   `correlated_observations` (5, 37); `gp_discrepancy` (7, 8, 36);
   `robust_likelihoods` (9, 34); `measurement_to_calibration` (12, 14,
   15, 16, 21, 26); `alpha_ca_error_model_comparison` (10, 11, 13, 18,
   25); `hierarchical_calibration` (22, 24, 35, 38, and 30 for the
   held-out energy).  A recipe without a notebook is fine; a notebook must
   cite at least one recipe.
9. **`design.md` and README** rewritten from this document.

### Fast and converged tiers

Recipe tests must be fast in CI, but some expected behaviours only hold
for a converged sampler.  The resolution is two tiers with a strong
preference for assertions that need no sampler at all.

- **Prefer sampler-free assertions.**  Most expected-behaviour bullets are
  structural or analytic: names and columns, a covariance equal to a
  hand-built matrix, `chi2` identities, `ll(fit) + ll(held) == ll(full)`,
  a mode built from `c.ym`, `prior_transform` round trips, compile-time
  errors.  These are exact and form the core of every recipe test.
- **Linear-Gaussian oracle.**  `test/recipes/oracle.py` computes the
  closed-form posterior and predictive of any recipe instantiated with a
  linear model and Gaussian terms.  Tests compare `log_posterior` and
  `predictive_draws` statistics against it without sampling.  This covers
  the coverage-type claims of recipes 1, 6, 12, 17, 26 and the
  marginalised form of 38 exactly.
- **Seeded short chains, qualitative assertions.**  Ordering claims ("case
  2 under-covers and case 3 recovers"; "the Student-t covers the truth
  and the Gaussian does not") use a seeded 16-walker, few-hundred-step
  emcee run on a toy problem and assert only the ordering, with a wide
  margin.
- **Converged tier.**  Tests marked `slow` hold the numeric claims
  (coverage within 0.05 of nominal, `τ` recovered within its interval,
  evidence differences), record the seed, R-hat and effective sample
  size so a failure is diagnosable, and run the notebooks through nbmake.
  They are deselected by default; every push runs the fast tier, and a
  separate "Converged tier" workflow runs `pytest -m slow` and the
  notebooks on pushes and pull requests into `main` (and by hand with
  `workflow_dispatch`).  There is no scheduled run.
- **Rules.**  The fast tier has a budget of a few minutes and zero
  tolerated flakiness.  A flaky fast assertion is demoted to `slow`, never
  loosened until it passes.  Every recipe test file has at least one fast
  test; a slow test is optional and holds the numbers.  Which tier pins
  each bullet is decided per recipe as the implementation lands.

## 10. Open questions

- **Term-level partial masks.**  The factories accept `mask=` for a partial
  support today.  `on=(block, point_mask)` would make it a first-class
  support form; whether that is worth the extra spelling is a matter of
  taste.  v0 keeps `mask=` on the factories.
- **Workspace caching across datasets at one energy.**  Correct without
  it; a factor of a few in setup time with it.  Not in v0.
- **Cross-constraint modes.**  `U` is per constraint so that constraints
  stay independent and weights stay meaningful.  A mode that couples two
  constraints is, as today, a reason to merge them.
- **Known non-goals.**  The closing section of `recipes.md`, "What this API
  does not express", lists the calibration classes the skeleton rules out
  (non-elliptical likelihoods, chain-dependent masks, per-point latents,
  per-point likelihood factors, mixture likelihoods) with the size of the
  addition each would need.  Revisit when a study needs one.

## 11. Release path: from 0.x to 1.0 in the same repository

The GitHub repository, its pull-request history, collaborator branches and
Pages documentation are kept.  `rxmc` is not on PyPI, so the name is free
and version history there starts at 1.0.

1. **Close out 0.x.**  Merge `api_generalisation` into `main` by pull
   request.  Tag the merge `v0.1.0` and add a branch `legacy/0.x` at the
   same commit: the last state of the old design, with this plan in its
   tree, reachable by name forever.
2. **Rewrite on a branch.**  `rewrite` is cut from that `main` (§9,
   milestone 0).  History stays linear; no orphan branch and no force
   push.  The README on `main` carries a one-line banner pointing at the
   branch while the rewrite is in progress.
3. **Pre-releases by tag.**  setuptools_scm reads the version from tags.
   Tag `v1.0.0a1` after milestone 4, `v1.0.0b1` after milestone 6,
   `v1.0.0rc1` after milestone 8; each tag push publishes to PyPI as a
   pre-release (installable with `pip install --pre rxmc`).  Publishing
   `a1` early claims the PyPI name; try TestPyPI once first.
4. **Release.**  Pull request `rewrite` into `main`, ordinary merge, tag
   `v1.0.0`, GitHub Release, PyPI publish, Pages rebuild from `main`.  The
   README then notes that 0.x lives at `v0.1.0` and `legacy/0.x`.
5. **Old branches.**  Collaborators' branches are left alone; superseded
   ones may be deleted after 1.0.
