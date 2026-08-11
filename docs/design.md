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

A {class}`~rxmc.covariance.Term` is one additive contribution to the stacked
covariance: it carries a `support` (indices into the stacked vector), the
`Parameter`s it consumes, and writes its block via `add_to(Sigma, ctx, theta)`.
Context-dependent terms receive a `StackContext` bundling the stacked
`x`/`y`/`ym` and the block supports, so modes can be prediction-scaled
(`ctx.ym[support]`) or coordinate-dependent (kernels).

The primitives — {class}`~rxmc.covariance.DenseTerm` (fixed block, validated
for shape and symmetry), {class}`~rxmc.covariance.DiagonalTerm`,
{class}`~rxmc.covariance.RankOneTerm`, and
{class}`~rxmc.covariance.KernelTerm` (sklearn kernels; one parameter per free
hyperparameter *element*, so anisotropic kernels contribute
`len(kernel.theta)` parameters) — are wrapped by the factory helpers that form
the primary authoring API: `statistical_term`, `normalization_term`,
`offset_term`, `noise_term`, `noise_fraction_term`, `model_error_term`, and
`discrepancy_term`.

{class}`~rxmc.covariance.ConstraintCovariance` assembles the terms. It is
constructed with the true observation block boundaries
(`blocks=stacked_supports(observations)`), from which two structural facts are
decided **once, conservatively**:

- `block_diagonal` — true only if every off-diagonal-capable term
  (`couples_offdiagonal`) provably sits inside a single block. With no blocks
  supplied, any coupling-capable term forces the dense path; there is no
  guessing from support shape.
- `is_constant` — true when no term depends on parameters or context; the
  Cholesky factors (dense and per-block) are then computed once and cached
  read-only.

`ConstraintCovariance.stacked_distance(ctx, params)` owns the dispatch between
the block-diagonal fast path (factor each block separately, `O(Σ nᵢ³)`) and a
single dense Cholesky — and is the seam where a future low-rank (Woodbury)
path would slot in.

## Observations are leaves

An {class}`~rxmc.observation.Observation` is pure data — `x`, `y`,
`y_stat_err` — plus the measurement's reported systematic magnitudes retained
as **inert metadata** (`y_sys_err_normalization` fractional,
`y_sys_err_offset` absolute in internal units). It emits only its statistical
diagonal automatically. Every correlated mode is an explicit term:
`obs.systematic_terms(support)` converts the metadata on request, and
**nothing is ever folded into a covariance silently** — a deliberate behavior
change from pre-0.1 versions, pinned by regression tests.

The reaction observation classes' `from_measurement` constructors keep this
contract across unit conversion: dimensionful errors (statistical, offset) are
divided by the unit normalization (retained as `obs.norm`; a per-angle array
in the Rutherford-conversion cases), the fractional normalization error passes
through untouched.

## Constraints, likelihood functionals, and parameters

`Constraint(observations, physical_model, likelihood=GaussianLikelihood(),
extra_terms=(), include_statistical_term=True)` builds the stacked covariance
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
covariance term: it changes the mean, so it lives on the model side as
{class}`~rxmc.physical_model.ScaledModel` /
{class}`~rxmc.physical_model.PerObservationScaledModel`, flowing through the
ordinary model-parameter machinery.

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
- Masked/multi-mode systematics on `Observation` are deferred; the factory
  helpers accept a `mask=` argument directly for masked terms.
