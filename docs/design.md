# Design: the stacked covariance model

This page records the architecture of `rxmc`'s covariance layer — the design
that replaced the pre-0.1 "likelihood model zoo" — and the decisions locked in
during that refactor.

## The two mechanisms

Two different things hide under "share a covariance," and they live on
different axes:

- **(A) Correlating observations** is a statement about **covariance
  structure** — off-diagonal blocks coupling observation *i* and *j*.
- **(B) Two covariance terms sharing a parameter** is a statement about
  **parameter wiring** — the observations stay independent; one θ component
  feeds two different terms.

**A couples the data; B couples the parameters.** They get distinct
mechanisms:

- **A — the covariance owns the stacked block.** A
  {class}`~rxmc.constraint.Constraint` is the maximal block of
  mutually-correlated data: it owns one multivariate likelihood over the
  stacked vector `y = [y1; y2; ...]` of its observations. A *coupling* term is
  simply one whose `support` spans more than one observation block.
- **B — parameter routing by identity.**
  {class}`~rxmc.covariance.ConstraintCovariance` deduplicates the
  {class}`~rxmc.params.Parameter` objects its terms reference **by identity**
  (gather, not slice): referencing the *same* object in two terms yields one
  entry in the sampled vector, gathered into both.

The normalization example shows why both are needed: two datasets with
*independent* flux measurements of the same *magnitude* are case B (two
block-local `normalization_term`s sharing one `Parameter`); two datasets
normalized by the *same* uncertain flux are case A (one `normalization_term`
whose support spans both blocks). This is the D'Agostini / Barlow
correlated-systematics distinction.

## Terms and the assembled covariance

There is exactly one term type. A {class}`~rxmc.covariance.Term` is a
numpy-style callable `fn(c, *values) -> array` of a
{class}`~rxmc.covariance.TermContext` `c` — the term's local view of the
stacked `x`, `y` and `ym` on its `support` — and the sampled values of the
`Parameter`s it declares, plus a `kind` saying how the array enters the
covariance:

| `kind`     | `fn` returns                  | contribution                    |
|------------|-------------------------------|---------------------------------|
| `"diag"`   | standard-deviation vector `v` | `Σ_ii += v_i²`                  |
| `"mode"`   | mode vector `v`               | `Σ += v vᵀ` (one correlated mode) |
| `"matrix"` | symmetric block `M`           | `Σ_block += M`                  |

A plain array instead of `fn` is a fixed contribution (factored once and
cached). `support=None` (the default) means *the whole constraint* and is
bound when the term is added to a `ConstraintCovariance`; an explicit support
places a term on a subset of a multi-observation constraint. A `coords`
transform (see below) is applied to `x[support]` before `fn` sees it, so a
kernel can live in momentum transfer rather than angle without the term
knowing.

The factory helpers are one-line conveniences over this single type:
`statistical_term`, `normalization_term`, `offset_term`, `noise_term`,
`noise_fraction_term`, `model_error_term`, `systematic_term` (a mode with a
user basis), and `kernel_term` (sklearn kernels; one parameter per free
hyperparameter *element*, plus an optional parametric `amplitude` so that
`Σ += a aᵀ ∘ K`). Bases are ordinary callables of the `TermContext`
(`ones`, `ym`, `averaging`, `x_basis(scale)`, or parametric ones like
`exp_growth(scale)` whose extra parameters are passed as `basis_params`).
Anything the helpers cannot say is a direct `Term`:

```python
# noise growing with angle: sigma(theta) = eps * exp(l * theta / pi)
Term(lambda c, e, l: np.exp(e) * np.exp(l * c.x / np.pi), (log_eps, slope), kind="diag")
# the same thing through the helper
noise_term(log_eps, basis=exp_growth(np.pi), basis_params=(slope,))
```

{class}`~rxmc.covariance.ConstraintCovariance` assembles the terms. It is
constructed with the true observation block boundaries
(`blocks=stacked_supports(observations)`), from which two structural facts are
decided **once, conservatively**:

- `block_diagonal` — true only if every off-diagonal-capable term
  (`couples_offdiagonal`, i.e. `kind != "diag"`) provably sits inside a single
  block. With no blocks supplied, any coupling-capable term forces the dense
  path; there is no guessing from support shape.
- `is_constant` — true when no term depends on parameters or context; the
  Cholesky factors (dense and per-block) are then computed once and cached
  read-only.

`ConstraintCovariance.stacked_distance(ctx, params)` owns the dispatch between
the block-diagonal fast path (factor each block separately, `O(Σ nᵢ³)`) and a
single dense Cholesky — and is the seam where a future low-rank (Woodbury)
path would slot in.

## Transforms are one low-level type

{class}`~rxmc.transforms.Transform` is a numpy-style callable
`fn(a, *values)` with an optional tuple of `Parameter`s, an optional analytic
derivative and inverse, and composition via `|`. Anything callable is accepted
wherever a transform is expected. The same type serves three roles, so there
is no wrapper class per use:

- **Comparison space, on the observation.** `Observation(x, y,
  transform=log)` takes *raw* `y`, stores `y_raw`, `y = log(y_raw)`, and the
  delta-method statistical error `|t′(y_raw)|·σ`; the `Constraint` applies the
  same transform to the model prediction, so the model is written once, in
  physical space, and can never be double-transformed. `obs.log_jacobian`
  (`Σ log|t′|`) is the constant needed to compare evidences across comparison
  spaces. Parametric transforms are rejected here.
- **Parametric model transforms, on the model.** `PhysicalModel(params,
  transform=scale())` appends the transform's parameters to the model's and
  applies it after `evaluate` (unlike the former `ScaledModel`, which prepended a
  `log normalization` parameter, the scale parameters come *last* and default to
  `log_rho` / `log_rho_i`). This is the Kennedy–O'Hagan latent scale ρ (it
  changes the *mean*, so it is not a covariance term);
  `per_observation_scaling(observations)` gives one ρᵢ per dataset, routed by
  observation identity.
- **Coordinates, on a term.** `Term(..., coords=q)` / `kernel_term(kernel,
  coords=q)` evaluate the term in transformed coordinates; a parametric
  `coords` contributes its parameters to the term.

## Masks: hold-out as part of support

Which points *enter* a likelihood is a property of the support machinery, not
of the data: `Observation(..., mask=)` (or `obs.masked(mask)`,
`obs.masked_where(lambda x: x < cut)`) marks points active at the point level
without rebuilding anything — a reaction observation keeps its solver
workspace — and `Constraint(..., mask=)` selects observations at the
constraint level. The two combine into `constraint.active`, the stacked
indices the residual and the factorisation are restricted to; `n_data_pts` is
the active count. Terms are always authored over the full stack, so the same
term list describes the fit and the held-out views: `constraint.complement()`
is the held-out counterpart (every inactive point becomes active), sharing the
`Term`/`Parameter` objects so a posterior sample of the fit scores it directly
(see {mod}`rxmc.model_comparison`).

## Observations are leaves

An {class}`~rxmc.observation.Observation` is pure data — `x`, `y`,
`y_stat_err` — plus the measurement's reported systematic magnitudes retained
as **inert metadata** (`y_sys_err_normalization` fractional,
`y_sys_err_offset` absolute in internal units). It emits only its statistical
diagonal automatically. Every correlated mode is an explicit term:
`obs.systematic_terms()` converts the metadata on request (propagated to the
comparison space by the delta method when the observation has a transform), and
**nothing is ever folded into a covariance silently** — a deliberate behavior
change from pre-0.1 versions, pinned by regression tests.

The reaction observation classes' `from_measurement` constructors keep this
contract across unit conversion: dimensionful errors (statistical, offset) are
divided by the unit normalization (retained as `obs.norm`; a per-angle array
in the Rutherford-conversion cases), the fractional normalization error passes
through untouched.

## Constraints, likelihood functionals, and parameters

`Constraint(observations, physical_model, likelihood=GaussianLikelihood(),
extra_terms=(), include_statistical_term=True, mask=None)` builds the stacked covariance
from each observation's statistical term plus the explicit `extra_terms`
(`include_statistical_term=False` composes the entire covariance from
`extra_terms`, e.g. to let a `noise_term` *replace* reported statistics).

A likelihood ({class}`~rxmc.likelihood_model.Likelihood`:
`GaussianLikelihood`, `StudentT`, `Chi2`) is a thin functional of the
pre-computed `(d2, logdet, n)` statistics. The constraint's parameter vector
is the **full tuple** — covariance parameters followed by likelihood
parameters (e.g. Student-t `nu`) — and every method (`log_likelihood`, `chi2`,
`covariance_matrix`, `marginal_log_likelihood`) takes it in that order,
validating the count.

Mean renormalization (a Kennedy–O'Hagan latent scale ρ) is **not** a
covariance term: it changes the mean, so it lives on the model side as a
parametric transform (`PhysicalModel(..., transform=rxmc.transforms.scale())`
or `per_observation_scaling(observations)`), flowing through the ordinary
model-parameter machinery.

## Model comparison lives outside the sampler

{mod}`rxmc.model_comparison` consumes a constraint plus posterior *samples* and
never touches a sampler: posterior-predictive draws from `N(ym(θ), Σ(θ))` on
the active points (or the model-only predictive), empirical coverage curves and
sharpness, held-out log predictive scores on `constraint.complement()`, and
nested-sampling evidence bookkeeping (`logz_summary` with replicate-based
errors — the sampler's own error is a lower bound — and `compare_logz` with a
conservative tie verdict). `log_jacobian` supplies the comparison-space
constant for comparing evidences of, say, a log-space and a linear-space fit
of the same data.

## Error-model recipes

The motivating study — comparing error models for α+Ca elastic scattering data
without reported uncertainties — becomes one term list per model:

| error model                                              | `extra_terms`                                                                                                              |
|----------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------|
| constant noise (log space)                               | `[noise_term(log_err)]` with `Observation(..., transform=log)`                                                             |
| fractional noise (linear space)                          | `[noise_fraction_term(log_err)]`                                                                                           |
| noise growing with angle                                 | `[noise_term(log_err, basis=exp_growth(np.pi), basis_params=(slope,))]`                                                    |
| + rank-one mode ∝ θ                                      | `[..., systematic_term(log_sys, basis=x_basis(np.pi))]`                                                                    |
| + rank-one offset / normalisation                        | `[..., offset_term(parameter=log_sys)]` / `[..., normalization_term(parameter=log_sys)]`                                   |
| GP discrepancy in angle, constant amplitude              | `[..., kernel_term(Matern(1.0, nu=2.5), coords=lambda x: x/np.pi, amplitude=constant_amplitude, amplitude_params=(log_A,))]` |
| GP with angle-growing amplitude                          | `[..., kernel_term(..., amplitude=exp_growth_amplitude(1.0), amplitude_params=(log_A, slope))]`                            |
| GP in momentum transfer with amplitude `A q^{r/2}`       | `[..., kernel_term(RBF(1.0), coords=lambda x: 2*k*np.sin(x/2), amplitude=lambda c, lA, r: np.exp(lA)*c.x**(r/2), amplitude_params=(log_A, r))]` |
| heavy tails                                              | any of the above with `likelihood=StudentT()`                                                                              |

Fit/held-out splits are `obs.masked_where(lambda x: x < cut)` and
`constraint.complement()`; evidences are compared with
`compare_logz(logz_summary(...), logz_summary(...))` after adding
`log_jacobian` to the log-space fits.

## Scope decisions (locked)

- **Constraint = maximal correlated block.** {class}`~rxmc.evidence.Evidence`
  stays a weighted sum over *independent* constraints, so factorization cost
  is bounded at the block level.
- **Covariance/likelihood parameters are constraint-scoped.** Case-A and
  case-B sharing both happen *within* a constraint. Sharing a `Parameter`
  object across constraints, or duplicating a parameter name anywhere in an
  `Evidence`, is a hard error — the sanctioned model for a systematic shared
  between datasets is one constraint with a cross-block coupling term.
- **Tempering is consistent**: `Evidence` weights and
  `CalibrationConfig.likelihood_scaling` apply to the likelihood only (never
  the prior), including inside the Gibbs conditionals.
- **Fail fast**: constant covariances are factored eagerly at `Constraint`
  construction, so a singular covariance (e.g. an EXFOR subentry with no
  statistical error and no covering term) raises a named, actionable error
  instead of a `LinAlgError` mid-chain.

## Known limitations

- **No low-rank fast path yet.** Cross-block couplings are typically low rank,
  and the design anticipates a Woodbury / matrix-determinant-lemma update on
  top of the block-diagonal base; today they take the dense `O(N³)` path.
  `ConstraintCovariance.stacked_distance` is the seam.
- **Non-constant block-diagonal covariances still assemble the dense `N×N`
  matrix** before factoring its blocks (per-term `add_to` writes into the full
  matrix by design).
- **Masked rows are still assembled.** The full `N×N` covariance is built and
  then restricted to the active rows; a held-out view pays for the inactive
  rows' terms (cheap next to the forward model, but not free for large dense
  kernels).
- Multi-mode systematics on `Observation` are deferred; the factory helpers
  accept a `mask=` argument directly for masked (partial-support) terms.
