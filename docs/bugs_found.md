# Bugs and inconsistencies found during the architecture review

Found while reading the `api_generalisation` branch (head `b8bc7d8`) for the
ground-up design comparison in `groundup_design.md`.  Nothing here has been
fixed; every item was verified by reading the code at the cited lines.  Items
are grouped by how sure I am that they are wrong rather than merely fragile.

## Confirmed bugs

### 1. `IsobaricAnalogPNObservation` silently drops two solver arguments

- **Where:** `src/rxmc/ias_pn_observation.py:52-53` (constructor signature)
  and `:113-120` (the `set_up_solver` call).
- **What:** the constructor accepts `wavelengths_beyond_range` and
  `zeros_per_node`, documents them (`:92-94`), and `set_up_solver` takes them
  (`:209-210`), but the call inside `__init__` never forwards them.  The
  defaults are always used.  The elastic observation forwards both
  (`src/rxmc/elastic_diffxs_observation.py:149-150`).
- **Why it matters:** a user tuning the Lagrange basis size for the (p,n) IAS
  channel gets no effect and no error.
- **Fix:** add the two keyword arguments to the `set_up_solver` call.

### 2. `BatchedAdaptiveMetropolisSampler` adapts its proposal during burn-in

- **Where:** `src/rxmc/param_sampling.py:376-388`.
- **What:** the class docstring (`:304`), the `sample` docstring (`:354`) and
  the `burn` parameter docstring (`:366-368`) all say the proposal covariance
  is replaced only after *non-burn* batches.  The code recomputes
  `self.proposal_cov` and `self.args` unconditionally; only `record_batch` is
  guarded by `if not burn`.
- **Why it matters:** burn-in batches feed the adaptation, so the behaviour
  differs from what the docstring promises and from `AdaptiveMetropolisSampler`.
  Either the doc or the code is wrong.  Adapting during burn-in is arguably
  the *better* behaviour, so the likely fix is to the docstrings.
- **Fix:** decide, then make the docstrings and the `if not burn:` guard agree.

### 3. `Parameter` defines `__eq__` without `__hash__`

- **Where:** `src/rxmc/params.py:39-48`.
- **What:** defining `__eq__` sets `__hash__ = None`, so `Parameter` objects
  are unhashable.  Every uniqueness check in the package keys on `id(p)` or
  `p.name` for this reason (`covariance.py`, `constraint.py`, `evidence.py`),
  and nothing documents it.
- **Why it matters:** `set(model.params)` or `{p: value}` raises `TypeError`
  at runtime.  It is a trap for anyone extending the package, and it is the
  root reason the by-identity routing needs `id()` bookkeeping.
- **Fix:** either drop `__eq__` (identity is the sharing semantics anyway) or
  add `__hash__ = object.__hash__`.  See `groundup_design.md` §2.1.

### 4. Two independent pint `UnitRegistry` instances

- **Where:** `src/rxmc/elastic_diffxs_observation.py:24` and
  `src/rxmc/ias_pn_observation.py:12`.
- **What:** each module builds its own registry.  pint refuses to combine
  quantities from different registries.
- **Why it matters:** latent today because no code path mixes the two, but any
  helper that takes a quantity from one module into the other will raise.
  `DEFAULT_LMAX = 20` is likewise duplicated (`:27` and `:14`).
- **Fix:** one `ureg` in a shared module (`observation_from_measurement.py`
  is the natural home; its docstring already says it holds what the two share).

## Inconsistencies between the two sampler front ends

`CalibrationConfig` and `Walker` are meant to be two drivers over the same
posterior.  They are not.

### 5. `Walker.log_posterior` evaluates the likelihood when the prior is `-inf`

- **Where:** `src/rxmc/walker.py:140-143` versus
  `src/rxmc/config.py:405-411`.
- **What:** the config path short-circuits on a non-finite prior and skips
  the forward model.  The walker path always calls `Evidence.log_likelihood`
  first.  The Gibbs conditional has the same split:
  `config.conditional_posterior` (`config.py:579-583`) short-circuits,
  the inline closure at `walker.py:130-132` does not.
- **Why it matters:** out-of-bounds proposals cost a full reaction-model
  solve in the walker.  With bounds also enforced inside the kernels this is
  a performance bug, not a correctness bug, but it is a silent divergence.

### 6. Tempering exists only on the config path

- **Where:** `config.py:245-252, 384, 583`; no counterpart in `walker.py`.
- **What:** `likelihood_scaling` scales the likelihood (and the Gibbs
  conditionals) in `CalibrationConfig`.  `Walker` has no such knob; the only
  way to temper is `Evidence(weights=...)`.  `examples/overconfidence.ipynb`
  demonstrates both and prints a check that they agree.
- **Why it matters:** two names for one concept, with one of them reachable
  from only one driver.

### 7. List-of-scipy priors are accepted by one driver and rejected by the other

- **Where:** `config.py:133-135, 153-155, 184-185` (list branches in
  `ParameterConfig`) versus `param_sampling.py:66-70` (`_validate_object`
  requires `prior.logpdf`) and `walker.py:131, 160` (calls
  `sampler.prior.logpdf` directly).
- **What:** `ParameterConfig` special-cases a plain list of frozen scipy
  distributions in three places.  A `Sampler` built with the same list fails
  at construction because a list has no `logpdf`.
- **Why it matters:** the prior protocol is documented as one thing and
  implemented as two.  `IndependentPrior` already exists to wrap a list;
  `ParameterConfig` could wrap in `__init__` and delete all three branches.

### 8. `ParameterConfig._infer_dim` misreads priors whose `mean` is a method

- **Where:** `src/rxmc/config.py:92-98`.
- **What:** if the prior has no integer `dim`, the fallback is
  `int(np.size(dist.mean))`.  For any object whose `mean` is a *method* this
  is `1`, regardless of the true dimension.
- **Why it matters:** a custom multi-dimensional prior exposing `mean()` is
  reported as one-dimensional and rejected at `config.py:109-113` with a
  misleading message.  Frozen scipy multivariate distributions happen to
  expose `mean` as an array, which is why the tests pass.
- **Fix:** call `mean` if callable, or require `dim` and drop the guess.

## Fragile, not wrong

These are not bugs today but each is one refactor away from becoming one.

### 9. Unit conventions split across model and observation with no shared constant

- `elastic_diffxs_model.py:166` and `ias_pn_model.py:124, 170` divide by a
  bare `1000` (mb/sr to b/sr).  The matching assumption lives in the
  observation as `ureg.millibarn / ureg.steradian`
  (`elastic_diffxs_observation.py:204`).  Nothing ties them together.

### 10. Model and observation compatibility is checked by string, or not at all

- `elastic_diffxs_model.py:137, 174` compare `observation.quantity` to
  `self.quantity`.  `ias_pn_model.py:135, 159` reach straight for
  `observation.constraint_workspace` with no check.  Pairing an elastic model
  with an IAS observation fails inside jitr with a shape error.

### 11. Masked views share solver workspaces by reference, routed by `id()`

- `observation.py:204` uses `copy.copy`, so a masked view of a reaction
  observation shares both jitr workspaces and every array with its root.
  `transforms.py:281, 301` route `per_observation_scaling` by
  `id(obs.identity)`, and `test_holds_observation_references` exists only to
  stop id recycling.  Any deep copy, pickle, or reconstruction of an
  observation breaks the routing with a `KeyError` at evaluation time.

### 12. The burn-in loop in `Walker.walk` duplicates the main loop

- `walker.py:207-221` versus `:229-241`: identical bodies apart from
  `burn=True` and the progress string.  Any change to one must be mirrored.

### 13. `prior_transform` clips the unit cube only at the top level

- `config.py:492-506` clips `u` to `[eps, 1-eps]`; `ParameterConfig.prior_transform`
  and `IndependentPrior.prior_transform` (`priors.py:273-277`) do not.
  Calling either directly with an exact `0.0` or `1.0` returns `±inf`.

## Already fixed on this branch

- The `_rows` row-count check in `model_comparison.py` was reported by an
  earlier read as unable to fire.  At `b8bc7d8` it is called without `n` for
  the model samples and with `n` for the covariance samples (`:138, 143`),
  which is correct.
