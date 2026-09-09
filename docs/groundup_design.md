# A ground-up rxmc: declare, then compile

This document sketches what `rxmc` would look like if rewritten from scratch
around one principle, compares it with the current `api_generalisation`
design (`design.md`), and lays out an incremental path from one to the other.
It is a design proposal, not a plan of record.

The conclusion first: the *concepts* in `design.md` survive intact.  Evidence
over independent constraints, constraint as the maximal correlated block, one
`Term` with three kinds, likelihood as a functional of `(d2, logdet, n)`,
comparison space owned by the data side, masks as part of support, and the
case A / case B distinction are all things this design keeps.  What changes
is the *mechanics*: how parameters are routed, how the covariance is
represented, where solver state lives, and when validation happens.

## 1. The principle

**Everything the user constructs is an immutable declaration.  One compile
step turns the declaration into evaluators.**

Today the same objects do both jobs.  A `Term` is authored by the user and
then binds its support, caches coordinate transforms, and refuses to be reused
in a second constraint.  An `Observation` is "pure data" and also carries a
jitr workspace, a comparison transform, a mask, and an identity key that
`per_observation_scaling` routes on.  `Constraint.__init__` is the compile
step for one constraint, `Evidence.__init__` re-validates across constraints,
and `CalibrationConfig` / `Walker` each compile the flat parameter vector
again.  Every lifecycle rule in the current docs ("one Term belongs to one
constraint", "masked views share terms", "parameters are constraint-scoped",
"the same Parameter object across constraints is an error") is a consequence
of declaration and evaluation being fused.

Separating them gives:

- one place where parameters get slots, names get checked, supports get
  resolved, constant pieces get factored, and singular covariances get
  reported;
- specs that are reusable, comparable, and serialisable because they hold no
  caches;
- a single flat parameter vector that both samplers, model comparison, and
  plotting read from through one index instead of ten positional splits.

## 2. Components

The declarative layer, bottom up.

### 2.1 `Parameter`

```python
@dataclass(eq=False, frozen=True)
class Parameter:
    name: str
    bounds: tuple[float, float] = (-inf, inf)
    unit: str = ""
    latex: str | None = None
    prior: Distribution | None = None   # optional 1-D marginal
```

- Identity **is** the object.  `eq=False` keeps the default identity hash, so
  parameters are hashable and can key dicts and sets.  This replaces the
  current `__eq__`-without-`__hash__` and every `id(p)` table.
- Sharing a parameter anywhere in the graph means passing the same object.
  There is one rule and it holds across constraints too.  (Today: identity
  inside a constraint, value-equality for model parameters across
  constraints, and a hard error for covariance parameters across
  constraints.)
- A parameter may carry its own marginal prior.  Joint priors over several
  parameters (e.g. a multivariate normal over the optical-potential set) are
  attached at the `Problem` level (§2.8).

### 2.2 `Dataset`

Pure data, nothing else.

```python
@dataclass(frozen=True)
class Dataset:
    x: ndarray
    y: ndarray
    y_err: ndarray                  # statistical, physical units
    meta: Mapping = field(default_factory=dict)
    # meta holds: label, units, kinematics (reaction, Elab, ...),
    # reported systematics (norm fraction, offset), provenance (subentry)
```

- No comparison transform, no mask, no solver workspace, no identity key.
- `from_measurement` becomes a free function that reads an EXFOR-style
  measurement, converts units once (one shared `UnitRegistry`), and returns a
  `Dataset` with `meta` filled.  The current `ElasticDifferentialXSObservation`
  and `IsobaricAnalogPNObservation` classes disappear; what they store beyond
  data moves into `meta`, and the solver they build moves into `bind` (§2.3).

### 2.3 `Model` and `Predictor`

A model is a spec of "how to compute observables from parameters".  A
predictor is that model **bound to a grid**.

```python
class Model:
    params: tuple[Parameter, ...]
    def bind(self, grid_or_dataset) -> Predictor: ...

class Predictor:
    params: tuple[Parameter, ...]       # model params (+ transform params)
    grid: ndarray
    def __call__(self, *values) -> ndarray: ...   # physical space, on grid
    def __or__(self, transform) -> Predictor: ... # compose a mean transform
```

- `bind` is where expensive, grid-dependent state is built.  For a plain
  function model it closes over `x`.  For a reaction model it builds the
  jitr workspace from `dataset.meta["kinematics"]` and the grid, and caches
  it keyed on `(reaction, Elab, lmax, grid)` so that two datasets at the same
  energy solve the basis once.  The model owns its solver; the data does not.
- Binding to a bare array gives the plotting predictor every notebook
  currently hand-writes as `model.y(x, ...)`, and replaces
  `visualization_workspace` / `visualizable_model_prediction`.
- A Kennedy–O'Hagan scale is a transform composed onto a predictor:
  `model.bind(d) | scale(rho)`.  Because a predictor is per block, per-dataset
  scales are just distinct `rho_i` objects on distinct predictors.  The
  contextual transform, `_root`, and `per_observation_scaling`'s id table are
  gone.
- The generic model is `Model(params, fn)` with `fn(grid, *values)`, matching
  the "one class + callables" style of `Term` and `Transform`.  Reaction
  models subclass only to override `bind`.

### 2.4 `Block`

A block is the unit the residual is formed on: one dataset, one predictor,
one comparison space, one point mask.

```python
@dataclass(frozen=True)
class Block:
    data: Dataset
    predictor: Predictor
    space: Transform = identity       # parameter-free comparison transform
    mask: ndarray | None = None       # active points

    n: int; n_active: int
    y: ndarray                        # space(data.y)
    y_err: ndarray                    # |space'(data.y)| * data.y_err
    log_jacobian: float
    def predict(self, *values) -> ndarray      # space(predictor(*values))
    def masked(self, mask) / masked_where(pred) -> Block   # shares data+predictor
    def reported_terms(self) -> list[Term]     # from data.meta, delta-method propagated
```

- The comparison transform lives here, not on `Dataset`, because it is a
  modelling choice (a Gaussian in log space is a different distribution from
  a Gaussian in linear space, not merely a different covariance).  Terms are
  still authored in comparison space, exactly as today.
- `masked` returns a new `Block` sharing the dataset and predictor.  No
  `identity` attribute is needed: anything that wants "the same dataset"
  compares `block.data`.

### 2.5 `Term`

Unchanged in spirit; changed in what it holds.

```python
@dataclass(frozen=True)
class Term:
    fn: Callable | ndarray             # fn(c, *values) -> vector or matrix
    params: tuple[Parameter, ...] = ()
    kind: Literal["diag", "mode", "matrix"] = "matrix"
    on: Block | Sequence[Block] | None = None   # None = all blocks of the constraint
    coords: Transform = identity
    constant: bool = False
```

- **Support is a block reference, not stacked integer indices.**  `on=obs1`
  places the term on that block, `on=[obs1, obs2]` spans both (case A), and
  `on=None` means the whole constraint.  Compile resolves these to indices.
  `stacked_supports` and the `support=np.arange(...)` ceremony in the
  notebooks disappear.
- A term holds no state.  No `bind`, no `_x_cache`, no `_bound_N`.  The
  same term object can be placed in two constraints; each compile resolves it
  independently.
- The factory helpers (`noise_term`, `normalization_term`, `kernel_term`, …)
  keep their signatures with `support=` renamed `on=`.

### 2.6 `Constraint`

A container of blocks and terms plus the likelihood functional and weight.

```python
@dataclass(frozen=True)
class Constraint:
    blocks: tuple[Block, ...]
    terms: tuple[Term, ...] = ()
    likelihood: Likelihood = Gaussian()
    weight: float = 1.0
    statistical: bool = True          # add each block's y_err diagonal
    def complement(self) -> Constraint
```

- Eager checks only on things that do not need the parameter graph: blocks
  are distinct, every `on=` references a block in this constraint, an
  array-valued term has the right shape.
- `weight` moves here from `Evidence(weights=)` and subsumes
  `CalibrationConfig.likelihood_scaling`: one tempering knob, applied to the
  likelihood only, honoured by every driver.

### 2.7 `Evidence`

A tuple of independent constraints.  It no longer validates anything; it
exists so that "the calibration problem" has one name.  (It could be a plain
list; keeping the class gives a place for the `compile` entry point.)

### 2.8 `Problem` — the compile step

```python
problem = Problem(evidence, priors=[(omp.params, mvn), (log_eps, halfnormal)])
```

`Problem.__init__` is the only place in the package that walks the graph.
It produces:

- `problem.index: ParameterIndex` — the unique `Parameter` objects in
  first-seen order (blocks' predictors, then terms, then likelihoods,
  constraint by constraint), each with a slot.  `index.slot(p)`,
  `index.slots(ps)`, `index.names`, `index.bounds`, `index.ndim`.  Name
  uniqueness is checked once, here.
- `problem.constraints: tuple[CompiledConstraint, ...]` (§3).
- `problem.prior` — assembled from per-parameter marginals and the joint
  priors passed in; every slot must be covered exactly once, checked here.
- The flat interface external samplers want, which is what
  `CalibrationConfig` exposes today:
  `ndim`, `names`, `log_likelihood(theta)`, `log_prior(theta)`,
  `log_posterior(theta)`, `prior_transform(u)`, `starting_location(n)`.
- `problem.groups` — named slot groups for Gibbs drivers: `"model"` (all
  predictor slots) and one group per constraint's nuisance slots.  Any other
  partition is a list of slot arrays.
- `problem.predict(theta) -> list[ndarray]` per block, in comparison or
  physical space.

Nothing user-facing is mutated by compiling.  Compile the same evidence twice
and you get two independent problems.

### 2.9 Likelihood

Unchanged.  A `Likelihood` is a functional of `(d2, logdet, n, *values)` with
optional parameters (`StudentT.nu`).  Those parameters are ordinary nodes in
the index like every other.

## 3. The compiled constraint

`CompiledConstraint` is what today's `Constraint` + `ConstraintCovariance`
are, minus the parameter splitting.

```python
class CompiledConstraint:
    x, y, y_err: ndarray               # stacked, comparison space
    offsets: tuple[slice, ...]         # one per block
    active: ndarray                    # stacked indices of active points
    predictors: list[(slice, gather, Predictor)]
    covariance: StructuredCovariance
    likelihood: Likelihood; like_gather: ndarray
    weight: float

    def ym(self, theta) -> ndarray           # memoised on theta[predictor slots]
    def log_likelihood(self, theta) -> float
    def matrix(self, theta) -> ndarray       # dense, for display only
```

Two things are worth spelling out.

**Gather, never split.**  Every callable node received a gather array at
compile time.  Evaluation is `node(*theta[gather])`.  The ten positional
splits in the current code (`PhysicalModel.split_params`,
`Constraint._split`, the term's fn/coords split, `_scaled_term`'s
coefficient/basis split, `kernel_term`'s kernel/amplitude split,
`Transform.__or__`, `CalibrationConfig.split_parameters` and its
`prior_transform` cursor, `model_comparison.split_samples`, and
`predictive.total_predictive_band`'s `n_model_params`) all become reads of
`problem.index`.  Composite nodes (`f | g`, coefficient times basis) still
concatenate their children's parameters, but they do so once at construction
and the compile step assigns one gather for the composite.

**Memoised forward model.**  `ym(theta)` caches the last prediction keyed on
the values of the predictor slots.  A Gibbs sweep over the nuisance group
changes no predictor slot, so it never re-solves the reaction model.  This one
mechanism replaces `Constraint.marginal_log_likelihood`,
`Constraint.predict`-then-closure in `Walker.run_likelihood_batches`,
`Evidence.weighted_marginal_log_likelihood`, `CalibrationConfig.predict_parametric`
and `CalibrationConfig.conditional_posterior`.

## 4. The structured covariance

This is the one change that is a dramatic simplification *and* a speed-up,
and it needs no API change.

The three term kinds already describe a decomposition:

```
Sigma = D + sum_b M_b + U U^T (+ dense fallback)
```

- `D` is a length-`N` vector: the sum of squares of every `"diag"` term.
- `M_b` is one dense block per observation block: the sum of `"matrix"` terms
  whose support lies inside block `b`.
- `U` is `N x r`: one column per `"mode"` term, the mode vector scattered onto
  its support and zero elsewhere.  A mode spanning several blocks (case A) is
  just a column with entries in several blocks.
- A `"matrix"` term that crosses block boundaries (a GP kernel over the union
  of two datasets) forces the dense path for that constraint.  Everything
  else stays structured.

With `B = blockdiag(M_b + diag(D_b))` and per-block Cholesky `L_b`:

```
z   = L^{-1} r                     (block by block)
W   = L^{-1} U                     (block by block, N x r)
S   = I_r + W^T W                  (r x r)
d2  = z^T z - (W^T z)^T S^{-1} (W^T z)
logdet Sigma = sum_b logdet B_b + logdet S
```

Cost is `O(sum_b n_b^3 + N r^2 + r^3)` instead of `O(N^3)`.  For the
motivating cases (two or three datasets sharing a normalisation mode,
`r = 1`) the cross-block coupling is essentially free.

Consequences:

- **All three "known limitations" in `design.md` go away.**  There is no
  dense `N x N` assembly for non-constant block-diagonal covariances, the
  low-rank path exists, and masking is slicing rows out of `D`, `U`, and
  `M_b` rather than assembling and then restricting.
- **The block path is the only path.**  `uses_block_path`, `block_diagonal`,
  `cholesky` versus `block_cholesky`, and the dispatch in `stacked_distance`
  collapse into one routine.
- **Constant pieces are still cached.**  Each of `D`, `U`, `M_b` is the sum of
  a constant part (evaluated once) and a parametric part.  When everything is
  constant the factors are cached exactly as now.
- **Constraint boundaries carry less weight.**  Today a constraint is the
  unit of dense factorisation, so the design must forbid sharing across
  constraints and push cross-dataset systematics into one constraint.  With
  Woodbury the cost of a coupling mode is independent of `N`, so the choice
  of where to draw constraint boundaries becomes about the likelihood
  functional and the weight, not about cost.
- `B` must be positive definite.  A block covered by modes alone (no
  statistical diagonal, no noise term) is singular in `B` even if `Sigma` is
  not.  Compile catches this as today's eager singular check does, and the
  remedy list is the same; a dense fallback for that constraint is the
  escape hatch.

## 5. Drivers

### 5.1 External samplers

`Problem` *is* the interface `black-box-bayes`, `emcee`, and `dynesty` want.
`CalibrationConfig` and `ParameterConfig` are not needed.  The prior list
form, `IndependentPrior`, and `TruncatedNormalPrior` collapse into
"a `Parameter` carries a marginal, or a group of parameters carries a joint".

### 5.2 In-package Gibbs

```python
walker = Walker(problem, groups=problem.groups, samplers={...}, rng=rng)
walker.walk(n_steps, burnin, batch_size)
walker.chain                 # (n, ndim), columns named by problem.index.names
```

- A sector is a slot group.  The walker alternates over groups, holding the
  others fixed, calling `problem.log_posterior` each time.  The forward-model
  memo makes the nuisance sweeps cheap without special methods.
- There is one chain.  The two-sector `model_sampler.chain` /
  `likelihood_samplers[i].chain` and the `np.hstack` in every notebook go
  away, as does the duplicated validation between `Walker` and
  `CalibrationConfig`.

### 5.3 Model comparison and prediction

`model_comparison` and `predictive` take `(problem, chain)` and select
columns through `problem.index`.  `split_samples` is not needed;
`total_predictive_band` gets its kernel columns from
`index.slots(term.params)` instead of a caller-supplied integer.
`Constraint.complement()` works on blocks exactly as it does on observations
today, and the held-out problem shares `Parameter` objects with the fit, so a
posterior sample scores it directly.

## 6. Worked example

The α+Ca study shape: two datasets without reported errors, compared in log
space, a noise level shared between them (case B), a normalisation mode
coupling them (case A), one latent scale per dataset, one held out below a
cut, nested sampling.

```python
import numpy as np, rxmc as rx
from rxmc import terms as T, transforms as tf

d1 = rx.from_measurement(m1)             # Dataset, meta filled, units converted
d2 = rx.from_measurement(m2)

omp = MyOpticalModel(params=[...])       # Model; bind() builds jitr workspaces
rho1, rho2 = rx.Parameter("log_rho_1"), rx.Parameter("log_rho_2")
b1 = rx.Block(d1, omp.bind(d1) | tf.scale(rho1), space=tf.log)
b2 = rx.Block(d2, omp.bind(d2) | tf.scale(rho2), space=tf.log)

log_eps = rx.Parameter("log_eps", prior=halfnormal)
log_eta = rx.Parameter("log_eta", prior=halfnormal)
c = rx.Constraint(
    blocks=[b1, b2],
    terms=[
        T.noise(log_eps, on=b1),           # case B: one magnitude, two blocks
        T.noise(log_eps, on=b2),
        T.normalization(log_eta, on=[b1, b2]),   # case A: one mode across both
    ],
    likelihood=rx.StudentT(),
)
fit  = c.masked_where(lambda x: x < cut)
held = fit.complement()

problem = rx.Problem([fit], priors=[(omp.params, mvn_prior)])
sampler = dynesty.NestedSampler(problem.log_likelihood, problem.prior_transform, problem.ndim)
...
heldout = rx.Problem([held], priors=problem.prior)     # same Parameter objects
lp = rx.model_comparison.heldout_log_predictive(heldout, chain)
```

Compare with the current version of the same study: `Observation(...,
transform=log)` per dataset, `stacked_supports` to place the terms,
`per_observation_scaling([obs1, obs2])` on the model with `obs` passed to
three places, `Constraint(...)`, `Evidence([...])`, `ParameterConfig` twice,
`CalibrationConfig`, and `split_samples` to read the chain back.

## 7. Compile, in pseudo-code

```python
def compile(evidence, priors):
    index = ParameterIndex()                     # ordered dict Parameter -> slot
    compiled = []
    for c in evidence:
        offsets, x, y, err = stack(c.blocks)     # comparison space
        active = concatenate(offset[b.mask] for each block)
        preds = [(offsets[i], index.add_all(b.predictor.params), b.predictor)
                 for i, b in enumerate(c.blocks)]
        terms = list(c.terms)
        if c.statistical:
            terms = [T.statistical(b.y_err, on=b) for b in c.blocks] + terms
        resolved = [(resolve(t.on, c.blocks, offsets), index.add_all(t.params), t)
                    for t in terms]
        like_gather = index.add_all(c.likelihood.params)
        cov = StructuredCovariance(resolved, N=len(y), offsets, active)
        cov.factor_constant_parts()              # eager; names the block on failure
        compiled.append(CompiledConstraint(...))
    index.check_names_unique()
    prior = assemble_prior(index, priors)        # every slot covered exactly once
    return Problem(index, compiled, prior)
```

`index.add_all(params)` returns the gather array for that node, adding new
parameters in first-seen order.  That is the whole routing story.

## 8. Mapping from the current code

| today | ground-up | note |
|---|---|---|
| `Parameter` (`__eq__`, no hash) | `Parameter` (identity, hashable, optional prior) | one identity notion |
| `Observation` | `Dataset` + `Block` | data vs. modelling choices |
| `ElasticDifferentialXSObservation`, `IsobaricAnalogPNObservation` | `from_measurement` → `Dataset`; solver in `Model.bind` | classes removed |
| `PhysicalModel(params, transform)` | `Model.bind(grid) -> Predictor`; `predictor \| transform` | per-block mean transforms |
| `per_observation_scaling`, `_root`, `identity` | `omp.bind(d_i) \| scale(rho_i)` | no routing table |
| `Term(support=indices)` + `bind` + caches | `Term(on=blocks)`, stateless | resolved at compile |
| `stacked_supports` | not needed | |
| `Constraint.__init__` + `ConstraintCovariance` | `Constraint` (spec) + `CompiledConstraint` | |
| `ConstraintCovariance.matrix/cholesky/block_cholesky/stacked_distance` | `StructuredCovariance` | Woodbury, one path |
| `Evidence(weights=)` + `likelihood_scaling` | `Constraint.weight` | one knob |
| `Evidence._validate_constraint_params`, `Constraint._validate_parameter_names` | `ParameterIndex.check_names_unique` | once |
| `ParameterConfig`, `CalibrationConfig` | `Problem` | |
| `IndependentPrior`, `TruncatedNormalPrior`, list priors | `Parameter.prior` + joint priors on `Problem` | |
| `Walker(model_sampler, likelihood_samplers)` | `Walker(problem, groups)` | one chain |
| `marginal_log_likelihood`, `predict_parametric`, `conditional_posterior`, `weighted_marginal_log_likelihood` | `CompiledConstraint.ym` memo | |
| `split_samples`, `n_model_params`, `theta_cols` | `problem.index.slots(...)` | |

## 9. What stays the same

- The hierarchy: evidence → constraint → block → point.
- `Term` with `kind in {"diag", "mode", "matrix"}`, the factory helpers, bases
  and amplitudes as callables of a local context.
- `Transform` as one type with derivative, inverse, and `|` composition,
  serving comparison space, mean transforms, and term coordinates.
- Terms authored in comparison space; delta-method propagation of reported
  errors; `log_jacobian` for cross-space evidence comparison.
- Likelihood as a functional of `(d2, logdet, n)`; `Gaussian`, `StudentT`,
  `Chi2`.
- Masks as part of support; `complement()` sharing parameters with the fit.
- Fail fast on singular constant covariances with a named dataset.
- Tempering applies to the likelihood only.

## 10. Incremental path

None of this requires a rewrite.  In order of payoff per unit of risk:

1. **`StructuredCovariance` inside `ConstraintCovariance`.**  Replace the
   internals of `matrix`, `cholesky`, `block_cholesky`, and
   `stacked_distance`; keep `matrix()` as the dense view.  No public API
   change.  The existing dense-versus-block equivalence tests are the
   acceptance tests.  This alone removes every "known limitation".
2. **Make `Parameter` hashable** (drop `__eq__` or add `__hash__`).
3. **`support=` accepts observation objects** and is resolved in
   `Constraint`.  Keep integer supports working.
4. **`ParameterIndex` built by `Evidence`**, exposing
   `log_likelihood(theta_flat)` alongside the nested form.  Point
   `CalibrationConfig.split_parameters`, `model_comparison.split_samples`,
   `predictive`, and `Walker`'s validation at it.  Lift the cross-constraint
   sharing ban.  This is the step that unifies the two drivers and fixes the
   inconsistencies in `bugs_found.md` §5–7.
5. **`Problem`** as the single compile entry point; `CalibrationConfig`
   becomes a thin alias, `Walker` takes a `Problem` and slot groups, one
   chain.
6. **Bind-time predictors.**  Move workspaces off the reaction observations
   into `Model.bind`, fold the two observation subclasses into
   `from_measurement`.  Do this after the α+Ca study lands, since it
   touches the classes that study uses.

Steps 1–3 are local and safe.  Step 4 is the structural one.  Steps 5–6 are
cleanup that the earlier steps make small.

## 11. Open questions

- **Joint priors and `prior_transform`.**  Nested sampling needs a
  unit-cube map.  Independent marginals have one; a multivariate normal
  needs a whitening transform.  Same situation as today; `Problem` should
  reject `prior_transform` for joints without a `ppf`-like method rather
  than guess.
- **Cross-constraint modes.**  `U` is per constraint so that constraints
  stay independent and weights stay meaningful.  A mode that couples two
  constraints is, as today, a reason to merge them.
- **`n_dof`.**  With sharing across constraints allowed, count unique
  slots, not the sum of per-constraint counts.
- **Term-level masks.**  Today the factories accept `mask=` for partial
  support.  With `on=(block, point_mask)` that becomes a first-class support
  form; whether it is worth the extra spelling is a matter of taste.
