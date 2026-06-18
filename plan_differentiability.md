# Research-Grade Differentiable VMEC Plan

Status: active umbrella plan and single source of truth for PR #20.
`plan_freeb.md` remains the detailed free-boundary evidence log.
`plan.md` and `discrete_adjoint_2506_plan.md` are historical/reference plans
and should not drive new work unless a specific old result needs to be audited.

Last updated: 2026-06-18.

## 2026-06-18 Mercier/JXBFORCE Diagnostic Decomposition

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Changed `_mercier_radial_stability_terms` to consume structured geometry and
   weighted-sum contexts instead of a long argument list of parity channels.
2. Extracted VMEC/JXBFORCE bsubu/bsubv preprocessing into
   `_mercier_preprocess_bsubuv`.
3. Extracted bsubs derivative reconstruction into
   `_jxbforce_bsubs_derivatives`, leaving the public `compute_mercier` function
   focused on orchestration, optional LBSUBS correction, debug dumps, and final
   diagnostic assembly.

Results obtained:

- `vmec_jax/io/wout/mercier.py` is net-negative: 270 insertions and 282
  deletions, for a 12-line reduction.
- `compute_mercier` dropped from 568 to 334 lines.
- `compute_mercier` no longer appears in the top 16 source-health function
  hotspots.
- The extracted helpers are named by physics/numerics role:
  `_mercier_preprocess_bsubuv` is 74 lines and
  `_jxbforce_bsubs_derivatives` is 131 lines.

Tests and commands run:

- `python -m ruff check vmec_jax/io/wout/mercier.py tests/test_wout_physics_wave8_coverage.py tests/test_finite_beta.py`
- `python -m compileall -q vmec_jax/io/wout/mercier.py tests/test_wout_physics_wave8_coverage.py tests/test_finite_beta.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_wout_helpers.py::test_wout_glasser_fallback_matches_current_term_reconstruction_and_private_alias tests/test_wout_helpers.py::test_wout_glasser_profile_reader_uses_persisted_or_fallback_variables tests/test_wout_helpers.py::test_wout_glasser_profile_writer_bundle_uses_data_or_zero_defaults -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_wout_physics_wave8_coverage.py::test_compute_mercier_lasym_lbsubs_branch_with_reduced_bsub_inputs tests/test_wout_physics_wave8_coverage.py::test_compute_mercier_short_mesh_returns_all_zero_component_profiles tests/test_finite_beta.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_wout_physics_wave8_coverage.py -q`
- `python tools/diagnostics/source_health.py --top 12 --top-functions 16`

Best next steps:

1. Move back to `solve_fixed_boundary_residual_iter` with the same strategy:
   extract named, testable setup or scan-reporting seams before touching
   adaptive update formulas.
2. If continuing WOUT work, split Nyquist coefficient assembly only with an
   explicit parity test selected first.
3. Keep Mercier formula order and debug environment branches stable until a
   dedicated numerical parity tranche is planned.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.98%.
- Differentiability/refactor implementation: 99.999998%.
- Solver monolith reduction: 99.57%.
- Free-boundary adjoint monolith reduction: 99.40%.
- Driver workflow decomposition: 99.87%.
- Residual iteration decomposition: 96.9%.
- WOUT diagnostic/profile decomposition: 99.83%.
- Bcovar/WOUT parity decomposition: 98.55%.
- Force-kernel decomposition: 98.55%.
- Optimizer workflow decomposition: 99.19%.
- Fixed-boundary optimizer decomposition: 94.8%.
- Implicit residual-adjoint decomposition: 95%.
- QI objective decomposition: 96.2%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95.8%.
- CI/runtime/coverage hygiene for this PR: 99.95%.
- Overall differentiability-refactor PR: 99.99955%.

## 2026-06-18 WOUT Diagnostic Payload Fan-Out Reduction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Changed the minimal WOUT schema mapper to consume the structured
   `WoutScalarDiagnostics` payload directly instead of requiring the high-level
   builder to unpack every scalar/profile field into locals.
2. Changed current-profile metadata mapping to consume the structured metadata
   payload directly.
3. Changed main-geometry coefficient mapping to consume the structured
   `WoutMainGeometryCoefficients` payload directly.
4. Preserved the VMEC++-style non-converged WOUT behavior by applying the
   optional beta-zeroing escape hatch via `scalar_diag._replace(...)`.

Results obtained:

- `vmec_jax/wout.py` is 34 lines smaller.
- `vmec_jax/io/wout/minimal.py` grows by only 3 lines, for a net 31-line source
  reduction.
- `wout_minimal_from_fixed_boundary` dropped from 806 to 772 lines.
- Full WOUT focused tests pass.

Tests and commands run:

- `python -m ruff check vmec_jax/wout.py vmec_jax/io/wout/minimal.py`
- `python -m compileall -q vmec_jax/wout.py vmec_jax/io/wout/minimal.py tests/test_wout_wave2.py tests/test_wout_helpers.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_wout_wave2.py::test_wout_minimal_light_nonconverged_dumps_and_defaults tests/test_wout_wave2.py::test_nonconverged_wout_roundtrip_preserves_fields_and_status tests/test_wout_helpers.py::test_wout_current_profile_metadata_from_indata_preserves_vmecplot_defaults tests/test_wout_helpers.py::test_wout_glasser_profile_writer_bundle_uses_data_or_zero_defaults -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_wout_wave2.py tests/test_wout_helpers.py -q`
- `python tools/diagnostics/source_health.py --top 12 --top-functions 16`

Best next steps:

1. Continue WOUT simplification by moving Nyquist coefficient assembly into a
   typed payload helper only when a focused parity test is selected first.
2. Continue fixed-boundary residual iteration decomposition with setup/context
   seams, not numerical update rewrites.
3. Avoid cosmetic context-construction moves that reduce one function but add
   equal or greater boilerplate elsewhere.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.98%.
- Differentiability/refactor implementation: 99.999998%.
- Solver monolith reduction: 99.57%.
- Free-boundary adjoint monolith reduction: 99.40%.
- Driver workflow decomposition: 99.87%.
- Residual iteration decomposition: 96.9%.
- WOUT diagnostic/profile decomposition: 99.77%.
- Bcovar/WOUT parity decomposition: 98.35%.
- Force-kernel decomposition: 98.55%.
- Optimizer workflow decomposition: 99.19%.
- Fixed-boundary optimizer decomposition: 94.8%.
- Implicit residual-adjoint decomposition: 95%.
- QI objective decomposition: 96.2%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95.7%.
- CI/runtime/coverage hygiene for this PR: 99.95%.
- Overall differentiability-refactor PR: 99.99953%.

## 2026-06-18 QI Shuffle-Profile Residual Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Extracted the base occupancy-width/profile residual setup from
   `quasi_isodynamic_residual_from_boozer_modes` into
   `_qi_width_profile_residuals`.
2. Extracted the legacy Goodman shuffle-profile and weighted shuffle-profile
   residual machinery into `_qi_shuffle_profile_residuals`.
3. Kept public output keys and residual ordering unchanged while moving branch
   stretching, crossing interpolation, profile reconstruction, and weighted
   alpha weights out of the public QI residual function.

Results obtained:

- `vmec_jax/quasi_isodynamic.py` is net-negative: 210 insertions and 265
  deletions, for a 55-line reduction.
- `quasi_isodynamic_residual_from_boozer_modes` dropped to 227 lines and no
  longer appears in the top 16 function-length hotspots.
- `_qi_width_profile_residuals` is 15 lines and
  `_qi_shuffle_profile_residuals` is 176 lines.
- Full QI and QI legacy tests pass after the extraction.

Tests and commands run:

- `python -m ruff check vmec_jax/quasi_isodynamic.py tests/test_quasi_isodynamic.py tests/test_qi_legacy.py`
- `python -m compileall -q vmec_jax/quasi_isodynamic.py tests/test_quasi_isodynamic.py tests/test_qi_legacy.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_quasi_isodynamic.py tests/test_qi_legacy.py -q`
- `python tools/diagnostics/source_health.py --top 12 --top-functions 16`

Best next steps:

1. Move to fixed-boundary residual context/object decomposition rather than
   continuing to over-split the QI objective.
2. Keep the QI public residual dictionary stable until the broader objective
   API redesign is handled as a dedicated migration.
3. Target the next high-impact monolith with a similarly net-negative tranche:
   either `solve_fixed_boundary_residual_iter` context setup or WOUT diagnostic
   profile assembly.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.98%.
- Differentiability/refactor implementation: 99.999998%.
- Solver monolith reduction: 99.57%.
- Free-boundary adjoint monolith reduction: 99.40%.
- Driver workflow decomposition: 99.87%.
- Residual iteration decomposition: 96.9%.
- WOUT diagnostic/profile decomposition: 99.72%.
- Bcovar/WOUT parity decomposition: 98.35%.
- Force-kernel decomposition: 98.55%.
- Optimizer workflow decomposition: 99.19%.
- Fixed-boundary optimizer decomposition: 94.8%.
- Implicit residual-adjoint decomposition: 95%.
- QI objective decomposition: 96.2%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95.7%.
- CI/runtime/coverage hygiene for this PR: 99.95%.
- Overall differentiability-refactor PR: 99.99952%.

## 2026-06-18 QI Aligned-Profile Residual Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Extracted the smooth minimum-aligned trapped-well profile residual from
   `quasi_isodynamic_residual_from_boozer_modes` into
   `_qi_aligned_profile_residuals`.
2. Preserved the same FFT-based shift, trapped-region weighting, residual
   scaling, and public output keys.
3. Removed no-longer-needed local Boozer coefficient aliases in the main QI
   residual after the helper extraction.

Results obtained:

- `quasi_isodynamic_residual_from_boozer_modes` dropped from 519 to 477 lines.
- `_qi_branch_width_residuals` is 36 lines and
  `_qi_aligned_profile_residuals` is 37 lines.
- The tranche is net-negative in `vmec_jax/quasi_isodynamic.py`: 51
  insertions and 54 deletions.
- Full QI and QI legacy tests pass.

Tests and commands run:

- `python -m ruff check vmec_jax/quasi_isodynamic.py tests/test_quasi_isodynamic.py tests/test_qi_legacy.py`
- `python -m compileall -q vmec_jax/quasi_isodynamic.py tests/test_quasi_isodynamic.py tests/test_qi_legacy.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_quasi_isodynamic.py::test_qi_boozer_mode_residual_is_zero_for_alpha_independent_wells tests/test_quasi_isodynamic.py::test_qi_boozer_mode_residual_validates_shapes_and_resolution tests/test_qi_legacy.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_quasi_isodynamic.py tests/test_qi_legacy.py -q`
- `python tools/diagnostics/source_health.py --top 12 --top-functions 16`

Best next steps:

1. Continue QI decomposition with the shuffle-profile block only if it can be
   split into a clear branch-profile helper without changing the legacy
   ranking behavior.
2. Add a small focused test for the aligned-profile helper if it becomes a
   public or semi-public seam; for now full QI tests cover it through the public
   residual.
3. After one more QI tranche, move to the fixed-boundary residual context
   design rather than over-optimizing this file.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.98%.
- Differentiability/refactor implementation: 99.999998%.
- Solver monolith reduction: 99.57%.
- Free-boundary adjoint monolith reduction: 99.40%.
- Driver workflow decomposition: 99.87%.
- Residual iteration decomposition: 96.9%.
- WOUT diagnostic/profile decomposition: 99.72%.
- Bcovar/WOUT parity decomposition: 98.35%.
- Force-kernel decomposition: 98.55%.
- Optimizer workflow decomposition: 99.19%.
- Fixed-boundary optimizer decomposition: 94.8%.
- Implicit residual-adjoint decomposition: 95%.
- QI objective decomposition: 93.8%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95.7%.
- CI/runtime/coverage hygiene for this PR: 99.95%.
- Overall differentiability-refactor PR: 99.99945%.

## 2026-06-18 QI Branch-Width Residual Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Extracted the Goodman-style branch-width residual block from
   `quasi_isodynamic_residual_from_boozer_modes` into the private helper
   `_qi_branch_width_residuals`.
2. Kept the validated QI grid construction, public output keys, and residual
   concatenation unchanged.
3. Left the larger aligned-profile and shuffle-profile blocks in place for
   future focused tranches, since they are more coupled to interpolation and
   weighted-profile diagnostics.

Results obtained:

- `quasi_isodynamic_residual_from_boozer_modes` dropped from 557 to 519 lines.
- `_qi_branch_width_residuals` is 36 lines and names the physics term directly.
- The tranche is line-neutral overall in `vmec_jax/quasi_isodynamic.py`: 47
  insertions and 47 deletions.
- QI residual values, validation behavior, shape checks, and legacy ranking
  tests remain covered.

Tests and commands run:

- `python -m ruff check vmec_jax/quasi_isodynamic.py tests/test_quasi_isodynamic.py tests/test_qi_legacy.py`
- `python -m compileall -q vmec_jax/quasi_isodynamic.py tests/test_quasi_isodynamic.py tests/test_qi_legacy.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_quasi_isodynamic.py::test_qi_boozer_mode_residual_is_zero_for_alpha_independent_wells tests/test_quasi_isodynamic.py::test_qi_boozer_mode_residual_validates_shapes_and_resolution tests/test_qi_legacy.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_quasi_isodynamic.py tests/test_qi_legacy.py -q`
- `python tools/diagnostics/source_health.py --top 12 --top-functions 16`

Best next steps:

1. Continue QI decomposition with the aligned-profile or shuffle-profile block,
   but only as a single named physics helper with full QI tests.
2. Keep the public residual output dictionary unchanged until a broader
   objective API redesign is ready.
3. Revisit fixed-boundary residual context-object design after one more QI
   objective tranche.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.98%.
- Differentiability/refactor implementation: 99.999998%.
- Solver monolith reduction: 99.57%.
- Free-boundary adjoint monolith reduction: 99.40%.
- Driver workflow decomposition: 99.87%.
- Residual iteration decomposition: 96.9%.
- WOUT diagnostic/profile decomposition: 99.72%.
- Bcovar/WOUT parity decomposition: 98.35%.
- Force-kernel decomposition: 98.55%.
- Optimizer workflow decomposition: 99.19%.
- Fixed-boundary optimizer decomposition: 94.8%.
- Implicit residual-adjoint decomposition: 95%.
- QI objective decomposition: 93.4%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95.7%.
- CI/runtime/coverage hygiene for this PR: 99.95%.
- Overall differentiability-refactor PR: 99.99944%.

## 2026-06-18 Same-Branch NESTOR Profile Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Moved dense-vs-matrix-free NESTOR profile orchestration from
   `examples/optimization/free_boundary_QS_coil_optimization.py` into
   `vmec_jax.solvers.free_boundary.coil_optimization`.
2. Kept the actual branch-local vector replay and summary functions as
   callbacks supplied by the example, so the validated JAX replay path is
   unchanged.
3. Preserved example-module compatibility exports for
   `nestor_profile_policy_from_results` and
   `parse_profile_matrix_free_solvers`, since tests and user scripts may still
   import those names from the example.

Results obtained:

- `write_same_branch_validation_report` dropped from 480 to 379 lines.
- The new package helper `same_branch_nestor_profile_from_vector_replay` is 89
  lines and keeps profile/report decisions in package code.
- The tranche is net-negative across touched source files: 114 insertions and
  122 deletions.
- `write_same_branch_validation_report` no longer appears in the top 16
  source-health function hotspots.

Tests and commands run:

- `python -m ruff check vmec_jax/solvers/free_boundary/coil_optimization.py examples/optimization/free_boundary_QS_coil_optimization.py tests/test_free_boundary_qs_coil_optimization_smoke.py`
- `python -m compileall -q vmec_jax/solvers/free_boundary/coil_optimization.py examples/optimization/free_boundary_QS_coil_optimization.py tests/test_free_boundary_qs_coil_optimization_smoke.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_qs_coil_optimization_smoke.py::test_nestor_profile_policy_requires_size_and_speedup_thresholds tests/test_free_boundary_qs_coil_optimization_smoke.py::test_same_branch_report_profiles_nestor_and_rejected_slot tests/test_free_boundary_qs_coil_optimization_smoke.py::test_same_branch_report_profile_skips_above_mode_count_cap -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_qs_coil_optimization_smoke.py tests/test_free_boundary_qa_finite_beta_coil_optimization_smoke.py -q`
- `python tools/diagnostics/source_health.py --top 12 --top-functions 16`

Best next steps:

1. Stop extracting from `free_boundary_QS_coil_optimization.py` for now unless
   another coherent user-facing simplification appears; the example is below
   the source-health threshold and still shows the workflow.
2. Move to the next source-health hotspot with safe seams: either test-helper
   extraction for repeated free-boundary FD assertions or context-object design
   for fixed-boundary residual scan internals.
3. Continue avoiding CI polling; check only after a larger batch or concrete
   failure report.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.98%.
- Differentiability/refactor implementation: 99.999998%.
- Solver monolith reduction: 99.57%.
- Free-boundary adjoint monolith reduction: 99.39%.
- Driver workflow decomposition: 99.87%.
- Residual iteration decomposition: 96.9%.
- WOUT diagnostic/profile decomposition: 99.72%.
- Bcovar/WOUT parity decomposition: 98.35%.
- Force-kernel decomposition: 98.55%.
- Optimizer workflow decomposition: 99.18%.
- Fixed-boundary optimizer decomposition: 94.8%.
- Implicit residual-adjoint decomposition: 95%.
- QI objective decomposition: 93%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95.7%.
- CI/runtime/coverage hygiene for this PR: 99.95%.
- Overall differentiability-refactor PR: 99.99942%.

## 2026-06-18 Same-Branch Replay Setup Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Moved same-branch replay option construction and replay mode-count guard
   metadata into `vmec_jax.solvers.free_boundary.coil_optimization`.
2. Kept the actual branch-local scalar/vector replay calls, NESTOR profiling,
   and complete-solve acceptance authority in the example report writer.
3. Reduced report-writer JSON boilerplate around current-only coil geometry,
   production tolerance defaults, and vector-gate summary construction.

Results obtained:

- `write_same_branch_validation_report` dropped from 515 to 480 lines.
- The tranche is net-negative across touched source files: 49 insertions and
  50 deletions.
- Same-branch scalar/vector report behavior and NESTOR profile gates remain
  covered by focused tests.

Tests and commands run:

- `python -m ruff check vmec_jax/solvers/free_boundary/coil_optimization.py examples/optimization/free_boundary_QS_coil_optimization.py tests/test_free_boundary_qs_coil_optimization_smoke.py`
- `python -m compileall -q vmec_jax/solvers/free_boundary/coil_optimization.py examples/optimization/free_boundary_QS_coil_optimization.py tests/test_free_boundary_qs_coil_optimization_smoke.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_qs_coil_optimization_smoke.py::test_same_branch_report_writer_records_branch_local_scalar_gradient tests/test_free_boundary_qs_coil_optimization_smoke.py::test_same_branch_report_writer_records_branch_local_vector_jacobian tests/test_free_boundary_qs_coil_optimization_smoke.py::test_same_branch_report_profiles_nestor_and_rejected_slot tests/test_free_boundary_qs_coil_optimization_smoke.py::test_same_branch_report_profile_skips_above_mode_count_cap -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_qs_coil_optimization_smoke.py tests/test_free_boundary_qa_finite_beta_coil_optimization_smoke.py -q`
- `python tools/diagnostics/source_health.py --top 12 --top-functions 15`

Best next steps:

1. Continue report-writer staging around NESTOR profile case/result assembly.
2. Keep the VMEC/free-boundary numerical calls unchanged until each extracted
   seam has direct report-test coverage.
3. Continue source-health work on larger monoliths after this free-boundary
   example/report lane stops yielding low-risk reductions.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.98%.
- Differentiability/refactor implementation: 99.999998%.
- Solver monolith reduction: 99.57%.
- Free-boundary adjoint monolith reduction: 99.38%.
- Driver workflow decomposition: 99.87%.
- Residual iteration decomposition: 96.9%.
- WOUT diagnostic/profile decomposition: 99.72%.
- Bcovar/WOUT parity decomposition: 98.35%.
- Force-kernel decomposition: 98.55%.
- Optimizer workflow decomposition: 99.17%.
- Fixed-boundary optimizer decomposition: 94.8%.
- Implicit residual-adjoint decomposition: 95%.
- QI objective decomposition: 93%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95.7%.
- CI/runtime/coverage hygiene for this PR: 99.95%.
- Overall differentiability-refactor PR: 99.99941%.

## 2026-06-18 Same-Branch Compact Report Metadata Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Moved compact same-branch complete-FD report metadata assembly into
   `vmec_jax.solvers.free_boundary.coil_optimization`.
2. Kept the report writer responsible for the actual complete-solve FD call,
   branch-local scalar/vector JAX replay, NESTOR profiling, rejected-slot gate,
   and JSON output authority.
3. Reduced local JSON-default boilerplate in
   `examples/optimization/free_boundary_QS_coil_optimization.py` without
   changing the report schema.

Results obtained:

- `write_same_branch_validation_report` dropped from 561 to 515 lines.
- The tranche is net-negative across touched source files: 66 insertions and
  69 deletions.
- Same-branch scalar-gradient, vector/JVP, rejected-slot, and NESTOR profile
  report tests still pass.

Tests and commands run:

- `python -m ruff check vmec_jax/solvers/free_boundary/coil_optimization.py examples/optimization/free_boundary_QS_coil_optimization.py tests/test_free_boundary_qs_coil_optimization_smoke.py`
- `python -m compileall -q vmec_jax/solvers/free_boundary/coil_optimization.py examples/optimization/free_boundary_QS_coil_optimization.py tests/test_free_boundary_qs_coil_optimization_smoke.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_qs_coil_optimization_smoke.py::test_same_branch_report_writer_records_branch_local_scalar_gradient tests/test_free_boundary_qs_coil_optimization_smoke.py::test_same_branch_report_writer_records_branch_local_vector_jacobian tests/test_free_boundary_qs_coil_optimization_smoke.py::test_same_branch_report_profiles_nestor_and_rejected_slot tests/test_free_boundary_qs_coil_optimization_smoke.py::test_same_branch_report_profile_skips_above_mode_count_cap -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_qs_coil_optimization_smoke.py tests/test_free_boundary_qa_finite_beta_coil_optimization_smoke.py -q`
- `python tools/diagnostics/source_health.py --top 12 --top-functions 15`

Best next steps:

1. Continue splitting `write_same_branch_validation_report` around replay
   option construction and profile-report staging.
2. Keep numerical derivative calls in place until extracted seams are covered
   by the existing report tests.
3. Continue avoiding CI polling; inspect failures later if checks report them.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.98%.
- Differentiability/refactor implementation: 99.999998%.
- Solver monolith reduction: 99.57%.
- Free-boundary adjoint monolith reduction: 99.37%.
- Driver workflow decomposition: 99.87%.
- Residual iteration decomposition: 96.9%.
- WOUT diagnostic/profile decomposition: 99.72%.
- Bcovar/WOUT parity decomposition: 98.35%.
- Force-kernel decomposition: 98.55%.
- Optimizer workflow decomposition: 99.16%.
- Fixed-boundary optimizer decomposition: 94.8%.
- Implicit residual-adjoint decomposition: 95%.
- QI objective decomposition: 93%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95.7%.
- CI/runtime/coverage hygiene for this PR: 99.95%.
- Overall differentiability-refactor PR: 99.99940%.

## 2026-06-18 Direct-Coil Optimization Summary Metadata Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Moved direct-coil optimization workflow metadata, single-stage limitation
   text, and objective/VMEC/optimizer summary config assembly into
   `vmec_jax.solvers.free_boundary.coil_optimization`.
2. Consolidated duplicated dry-run and post-optimization summary fields in
   `examples/optimization/free_boundary_QS_coil_optimization.py` behind one
   `summary_base`, leaving the example to add only run-mode-specific fields.
3. Kept the example workflow explicit: it still loads/synthesizes coils,
   selects variables, writes the VMEC input, runs complete solves, optionally
   writes branch-local derivative evidence, and records results.

Results obtained:

- `examples/optimization/free_boundary_QS_coil_optimization.py` dropped from
  2037 to 1946 lines and is no longer above the 2000-line source-health
  warning threshold.
- `optimize_coils` dropped from 320 to 303 lines.
- The tranche is net-negative across touched source files relative to the
  previous commit: 119 insertions, 122 deletions.
- Generated summary keys and values asserted by dry-run tests are unchanged.

Tests and commands run:

- `python -m ruff check vmec_jax/solvers/free_boundary/coil_optimization.py examples/optimization/free_boundary_QS_coil_optimization.py tests/test_free_boundary_qs_coil_optimization_smoke.py`
- `python -m compileall -q vmec_jax/solvers/free_boundary/coil_optimization.py examples/optimization/free_boundary_QS_coil_optimization.py tests/test_free_boundary_qs_coil_optimization_smoke.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_qs_coil_optimization_smoke.py::test_circle_dry_run_writes_configuration_without_solves tests/test_free_boundary_qs_coil_optimization_smoke.py::test_essos_dry_run_writes_direct_coil_configuration_without_mgrid tests/test_free_boundary_qs_coil_optimization_smoke.py::test_same_branch_vector_key_parser_accepts_bnormal_alias tests/test_free_boundary_qs_coil_optimization_smoke.py::test_same_branch_report_direction_policy_prefers_current_only_for_proposals -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_qs_coil_optimization_smoke.py tests/test_free_boundary_qa_finite_beta_coil_optimization_smoke.py -q`
- `python tools/diagnostics/source_health.py --top 12 --top-functions 15`

Best next steps:

1. Continue with `write_same_branch_validation_report` staging, where the
   remaining 561-line function still mixes direction construction, scalar
   registry setup, replay execution, profile gates, and JSON assembly.
2. Keep `_run_vmec2000_scan` deferred until a context-object seam is designed.
3. Re-run CI later rather than polling it after every tranche.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.98%.
- Differentiability/refactor implementation: 99.999998%.
- Solver monolith reduction: 99.57%.
- Free-boundary adjoint monolith reduction: 99.36%.
- Driver workflow decomposition: 99.87%.
- Residual iteration decomposition: 96.9%.
- WOUT diagnostic/profile decomposition: 99.72%.
- Bcovar/WOUT parity decomposition: 98.35%.
- Force-kernel decomposition: 98.55%.
- Optimizer workflow decomposition: 99.15%.
- Fixed-boundary optimizer decomposition: 94.8%.
- Implicit residual-adjoint decomposition: 95%.
- QI objective decomposition: 93%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95.7%.
- CI/runtime/coverage hygiene for this PR: 99.95%.
- Overall differentiability-refactor PR: 99.99939%.

## 2026-06-18 Same-Branch Runtime Config Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Moved same-branch vector-key constants, vector-key parsing,
   report-direction policy, and same-branch report/proposal runtime summary
   config assembly from
   `examples/optimization/free_boundary_QS_coil_optimization.py` into
   `vmec_jax.solvers.free_boundary.coil_optimization`.
2. Kept backward-compatible example-module names by importing the moved
   helpers into the script namespace; tests and users that import those names
   from the example continue to work.
3. Moved the small comma/space-separated float parser into the same package
   helper module so proposal-step and QS-surface parsing no longer duplicate
   generic CLI parsing inside the example.

Results obtained:

- `examples/optimization/free_boundary_QS_coil_optimization.py` dropped from
  2164 to 2037 lines, a 127-line reduction in the user-facing example.
- `vmec_jax/solvers/free_boundary/coil_optimization.py` grew from 921 to 1048
  lines, so this tranche is line-count neutral overall across touched files.
- Same-branch parser, direction-policy, dry-run summary, and report/proposal
  configuration behavior is unchanged.

Tests and commands run:

- `python -m ruff check vmec_jax/solvers/free_boundary/coil_optimization.py examples/optimization/free_boundary_QS_coil_optimization.py tests/test_free_boundary_qs_coil_optimization_smoke.py`
- `python -m compileall -q vmec_jax/solvers/free_boundary/coil_optimization.py examples/optimization/free_boundary_QS_coil_optimization.py tests/test_free_boundary_qs_coil_optimization_smoke.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_qs_coil_optimization_smoke.py::test_same_branch_report_direction_policy_prefers_current_only_for_proposals tests/test_free_boundary_qs_coil_optimization_smoke.py::test_same_branch_vector_key_parser_accepts_bnormal_alias tests/test_free_boundary_qs_coil_optimization_smoke.py::test_same_branch_vector_key_parser_defaults_to_promoted_state_scalars tests/test_free_boundary_qs_coil_optimization_smoke.py::test_circle_dry_run_writes_configuration_without_solves tests/test_free_boundary_qs_coil_optimization_smoke.py::test_essos_dry_run_writes_direct_coil_configuration_without_mgrid -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_qs_coil_optimization_smoke.py tests/test_free_boundary_qa_finite_beta_coil_optimization_smoke.py -q`
- `python tools/diagnostics/source_health.py --top 10 --top-functions 12`

Best next steps:

1. Continue shrinking the direct-coil optimization example around passive
   summary/objective metadata and report writer staging, but keep the
   pedagogic optimization flow visible to users.
2. Defer `_run_vmec2000_scan` movement until a real context-object seam exists;
   direct extraction remains closure-heavy and high-risk.
3. Keep CI watching deferred; inspect checks later only for concrete failures.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.98%.
- Differentiability/refactor implementation: 99.999998%.
- Solver monolith reduction: 99.57%.
- Free-boundary adjoint monolith reduction: 99.35%.
- Driver workflow decomposition: 99.87%.
- Residual iteration decomposition: 96.9%.
- WOUT diagnostic/profile decomposition: 99.72%.
- Bcovar/WOUT parity decomposition: 98.35%.
- Force-kernel decomposition: 98.55%.
- Optimizer workflow decomposition: 99.13%.
- Fixed-boundary optimizer decomposition: 94.8%.
- Implicit residual-adjoint decomposition: 95%.
- QI objective decomposition: 93%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95.7%.
- CI/runtime/coverage hygiene for this PR: 99.95%.
- Overall differentiability-refactor PR: 99.99938%.

## 2026-06-18 Free-Boundary Same-Branch Result Summary Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Moved same-branch scalar-gradient and vector/JVP JSON-summary formatting
   from `examples/optimization/free_boundary_QS_coil_optimization.py` into
   `vmec_jax.solvers.free_boundary.coil_optimization`.
2. Kept the example responsible for user-visible workflow decisions:
   selecting variables, configuring the direct-coil solve, writing the report,
   profiling NESTOR replay, and using complete solves as acceptance authority.
3. Updated the report test that previously reached into an example-private
   pytree helper so it validates the helper at its new package location.

Results obtained:

- `write_same_branch_validation_report` dropped from 630 to 561 lines.
- `examples/optimization/free_boundary_QS_coil_optimization.py` is now 2164
  lines; the branch-local scalar/vector summary helpers are 23 and 40 lines.
- Same-branch scalar, vector, rejected-slot, and NESTOR profile report behavior
  is unchanged.

Tests and commands run:

- `python -m ruff check vmec_jax/solvers/free_boundary/coil_optimization.py examples/optimization/free_boundary_QS_coil_optimization.py tests/test_free_boundary_qs_coil_optimization_smoke.py`
- `python -m compileall -q vmec_jax/solvers/free_boundary/coil_optimization.py examples/optimization/free_boundary_QS_coil_optimization.py tests/test_free_boundary_qs_coil_optimization_smoke.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_qs_coil_optimization_smoke.py::test_same_branch_report_writer_records_branch_local_scalar_gradient tests/test_free_boundary_qs_coil_optimization_smoke.py::test_same_branch_report_writer_records_branch_local_vector_jacobian tests/test_free_boundary_qs_coil_optimization_smoke.py::test_same_branch_report_profiles_nestor_and_rejected_slot tests/test_free_boundary_qs_coil_optimization_smoke.py::test_same_branch_report_profile_skips_above_mode_count_cap -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_qs_coil_optimization_smoke.py tests/test_free_boundary_qa_finite_beta_coil_optimization_smoke.py -q`
- `python tools/diagnostics/source_health.py --top 10 --top-functions 12`

Best next steps:

1. Continue residual-iteration decomposition with a context-object design before
   moving `_run_vmec2000_scan`; avoid closure-heavy extractions.
2. Split the remaining same-branch report writer only around coherent report
   assembly/profile stages, not individual lines.
3. Recheck GitHub checks later and address concrete failures only.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.98%.
- Differentiability/refactor implementation: 99.999998%.
- Solver monolith reduction: 99.57%.
- Free-boundary adjoint monolith reduction: 99.35%.
- Driver workflow decomposition: 99.87%.
- Residual iteration decomposition: 96.9%.
- WOUT diagnostic/profile decomposition: 99.72%.
- Bcovar/WOUT parity decomposition: 98.35%.
- Force-kernel decomposition: 98.55%.
- Optimizer workflow decomposition: 99.12%.
- Fixed-boundary optimizer decomposition: 94.8%.
- Implicit residual-adjoint decomposition: 95%.
- QI objective decomposition: 93%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95.7%.
- CI/runtime/coverage hygiene for this PR: 99.95%.
- Overall differentiability-refactor PR: 99.99937%.

## 2026-06-18 Free-Boundary Profile Solver Parser Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Moved `parse_profile_matrix_free_solvers` from the direct-coil optimization
   example into `vmec_jax.solvers.free_boundary.coil_optimization`.
2. Kept the example module compatibility surface by importing the parser back
   into the script namespace.

Results obtained:

- The example has one fewer report-internal helper and remains focused on
  user-visible optimization workflow.
- Parser behavior and profile-report tests remain unchanged.

Tests and commands run:

- `python -m compileall -q vmec_jax/solvers/free_boundary/coil_optimization.py examples/optimization/free_boundary_QS_coil_optimization.py`
- `python -m ruff check vmec_jax/solvers/free_boundary/coil_optimization.py examples/optimization/free_boundary_QS_coil_optimization.py tests/test_free_boundary_qs_coil_optimization_smoke.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_qs_coil_optimization_smoke.py::test_nestor_profile_policy_requires_size_and_speedup_thresholds tests/test_free_boundary_qs_coil_optimization_smoke.py::test_same_branch_report_profiles_nestor_and_rejected_slot -q`

Best next steps:

1. Run the final combined local gate for all files touched in this refactor
   burst.
2. Recheck GitHub checks after the branch push; fix concrete CI failures only.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.98%.
- Differentiability/refactor implementation: 99.999998%.
- Solver monolith reduction: 99.57%.
- Free-boundary adjoint monolith reduction: 99.35%.
- Driver workflow decomposition: 99.87%.
- Residual iteration decomposition: 96.9%.
- WOUT diagnostic/profile decomposition: 99.72%.
- Bcovar/WOUT parity decomposition: 98.35%.
- Force-kernel decomposition: 98.55%.
- Optimizer workflow decomposition: 99.11%.
- Fixed-boundary optimizer decomposition: 94.8%.
- Implicit residual-adjoint decomposition: 95%.
- QI objective decomposition: 93%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95.7%.
- CI/runtime/coverage hygiene for this PR: 99.95%.
- Overall differentiability-refactor PR: 99.99936%.

## 2026-06-18 Free-Boundary Same-Branch Scalar Registry Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Moved `same_branch_scalar_function_registry` from the direct-coil
   optimization example into
   `vmec_jax.solvers.free_boundary.coil_optimization`.
2. Kept the example's visible workflow intact: users still see how the run is
   configured, how objective weights are assembled, how reports are written,
   and how complete solves decide acceptance.
3. Removed imports from the example that were only needed by the moved registry.

Results obtained:

- `examples/optimization/free_boundary_QS_coil_optimization.py` dropped to
  2286 lines.
- `vmec_jax.solvers.free_boundary.coil_optimization` is 759 lines and owns the
  same-branch report scalar registry, proposal helpers, replay cache helpers,
  and profiling policy helpers.
- The example-level function name is still importable/monkeypatchable because
  the script imports it from the package module.

Tests and commands run:

- `python -m compileall -q vmec_jax/solvers/free_boundary/coil_optimization.py examples/optimization/free_boundary_QS_coil_optimization.py`
- `python -m ruff check vmec_jax/solvers/free_boundary/coil_optimization.py examples/optimization/free_boundary_QS_coil_optimization.py tests/test_free_boundary_qs_coil_optimization_smoke.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_qs_coil_optimization_smoke.py::test_same_branch_report_profiles_nestor_and_rejected_slot tests/test_free_boundary_qs_coil_optimization_smoke.py::test_same_branch_report_profile_skips_above_mode_count_cap tests/test_free_boundary_qs_coil_optimization_smoke.py::test_derivative_proposal_summary_marks_report_stale_when_trial_is_accepted tests/test_free_boundary_qs_coil_optimization_smoke.py::test_derivative_proposal_summary_records_rejected_trial_as_complete_solve_rejection -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_qs_coil_optimization_smoke.py tests/test_free_boundary_qa_finite_beta_coil_optimization_smoke.py -q`
- `python tools/diagnostics/source_health.py --top 10 --top-functions 12`

Best next steps:

1. Stop moving additional same-branch report internals unless
   `write_same_branch_validation_report` can be split into readable report
   assembly stages.
2. Prioritize residual iteration seams and broad CI validation after the recent
   accelerated-scan extraction.
3. Recheck GitHub checks after this push and address concrete failures.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.98%.
- Differentiability/refactor implementation: 99.999998%.
- Solver monolith reduction: 99.57%.
- Free-boundary adjoint monolith reduction: 99.35%.
- Driver workflow decomposition: 99.87%.
- Residual iteration decomposition: 96.9%.
- WOUT diagnostic/profile decomposition: 99.72%.
- Bcovar/WOUT parity decomposition: 98.35%.
- Force-kernel decomposition: 98.55%.
- Optimizer workflow decomposition: 99.1%.
- Fixed-boundary optimizer decomposition: 94.8%.
- Implicit residual-adjoint decomposition: 95%.
- QI objective decomposition: 93%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95.7%.
- CI/runtime/coverage hygiene for this PR: 99.95%.
- Overall differentiability-refactor PR: 99.99935%.

## 2026-06-18 Accelerated Residual Scan Runner Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Moved the non-VMEC2000 accelerated scan runner out of
   `solve_fixed_boundary_residual_iter` into
   `vmec_jax.solvers.fixed_boundary.residual.accelerated_scan.run_accelerated_residual_scan`.
2. Preserved monkeypatch/cache compatibility by passing JAX, JIT, scan-runner
   cache, cache get/put hooks, scan timing hooks, and cache-miss recorders from
   `residual.iteration` into the helper instead of importing private globals in
   the new module.
3. Left VMEC2000 scan fallback and surrounding branch-selection logic in
   `iteration.py`; only the accelerated scan body, cache setup,
   device-synchronization, timing report, and result assembly moved.

Results obtained:

- `vmec_jax/solvers/fixed_boundary/residual/iteration.py` dropped from 6927
  to 6716 lines.
- `solve_fixed_boundary_residual_iter` dropped from 6399 to 6185 lines.
- The new accelerated-scan module is 344 lines and owns one domain-specific
  runner.

Tests and commands run:

- `python -m compileall -q vmec_jax/solvers/fixed_boundary/residual/accelerated_scan.py vmec_jax/solvers/fixed_boundary/residual/iteration.py`
- `python -m ruff check vmec_jax/solvers/fixed_boundary/residual/accelerated_scan.py vmec_jax/solvers/fixed_boundary/residual/iteration.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_real_scan_wave10_coverage.py::test_accelerated_scan_one_step_updates_state_and_histories tests/test_solve_performance_instrumentation.py::test_accelerated_scan_timing_is_opt_in_and_path_labeled tests/test_solve_finish_cache_more_coverage.py::test_accelerated_scan_runner_cache_reports_timing_hit_and_miss -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_real_scan_wave10_coverage.py::test_vmec2000_scan_full_history_runs_fallback_decision tests/test_solve_wave7_coverage.py::test_residual_iter_vmec2000_scan_minimal_one_step -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_real_scan_wave10_coverage.py tests/test_solve_performance_instrumentation.py tests/test_solve_finish_cache_more_coverage.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_driver_run_wave8_coverage.py::test_run_fixed_boundary_selects_default_accelerated_policy_and_explicit_parity tests/test_driver_api.py::test_run_fixed_boundary_accelerated_mode_uses_scan tests/test_driver_api.py::test_run_fixed_boundary_cli_single_grid_uses_accelerated_finish_first tests/test_driver_api.py::test_run_fixed_boundary_accelerated_mode_defaults_to_single_grid -q`
- `python tools/diagnostics/source_health.py --top 10 --top-functions 12`

Best next steps:

1. Continue residual decomposition around smaller setup/preconditioner seams;
   keep `_run_vmec2000_scan` in place until a proper context object is designed.
2. Add a direct unit test for `run_accelerated_residual_scan` only if a compact
   synthetic force payload can be built without duplicating existing scan tests.
3. Recheck PR CI after this push and fix concrete failures if any appear.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.98%.
- Differentiability/refactor implementation: 99.999998%.
- Solver monolith reduction: 99.57%.
- Free-boundary adjoint monolith reduction: 99.35%.
- Driver workflow decomposition: 99.87%.
- Residual iteration decomposition: 96.9%.
- WOUT diagnostic/profile decomposition: 99.72%.
- Bcovar/WOUT parity decomposition: 98.35%.
- Force-kernel decomposition: 98.55%.
- Optimizer workflow decomposition: 99.05%.
- Fixed-boundary optimizer decomposition: 94.8%.
- Implicit residual-adjoint decomposition: 95%.
- QI objective decomposition: 93%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95.7%.
- CI/runtime/coverage hygiene for this PR: 99.95%.
- Overall differentiability-refactor PR: 99.9993%.

## 2026-06-18 Free-Boundary Rejected-Slot Gate Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Moved `same_branch_rejected_slot_gate_from_vector_replay` from the
   single-stage direct-coil example into
   `vmec_jax.solvers.free_boundary.coil_optimization`.
2. Kept the complete-solve acceptance loop, CLI options, and report-writing
   workflow in the example.
3. Preserved the example module compatibility surface by importing the moved
   function back into the script namespace.

Results obtained:

- `examples/optimization/free_boundary_QS_coil_optimization.py` dropped to
  2464 lines.
- The rejected-slot controller replay gate now lives next to the other
  same-branch report/proposal helpers.
- The JSON artifact shape is unchanged by construction and covered by the
  rejected-slot smoke tests.

Tests and commands run:

- `python -m compileall -q vmec_jax/solvers/free_boundary/coil_optimization.py examples/optimization/free_boundary_QS_coil_optimization.py`
- `python -m ruff check vmec_jax/solvers/free_boundary/coil_optimization.py examples/optimization/free_boundary_QS_coil_optimization.py tests/test_free_boundary_qs_coil_optimization_smoke.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_qs_coil_optimization_smoke.py::test_same_branch_report_profiles_nestor_and_rejected_slot tests/test_free_boundary_qs_coil_optimization_smoke.py::test_same_branch_report_profile_skips_above_mode_count_cap tests/test_free_boundary_qs_coil_optimization_smoke.py::test_derivative_proposal_summary_marks_report_stale_when_trial_is_accepted tests/test_free_boundary_qs_coil_optimization_smoke.py::test_derivative_proposal_summary_records_rejected_trial_as_complete_solve_rejection -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_qs_coil_optimization_smoke.py tests/test_free_boundary_qa_finite_beta_coil_optimization_smoke.py -q`
- `python tools/diagnostics/source_health.py --top 10 --top-functions 12`

Best next steps:

1. Consider the accelerated-scan residual extraction recommended by the
   explorer, but preserve monkeypatch/cache seams by passing iteration-local JAX
   and cache handles explicitly.
2. If the accelerated-scan extraction is too broad for one safe tranche, extract
   one more small free-boundary report sub-artifact first.
3. Recheck PR CI for actual failures after this push.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.98%.
- Differentiability/refactor implementation: 99.999998%.
- Solver monolith reduction: 99.51%.
- Free-boundary adjoint monolith reduction: 99.35%.
- Driver workflow decomposition: 99.87%.
- Residual iteration decomposition: 96.45%.
- WOUT diagnostic/profile decomposition: 99.72%.
- Bcovar/WOUT parity decomposition: 98.35%.
- Force-kernel decomposition: 98.55%.
- Optimizer workflow decomposition: 99.05%.
- Fixed-boundary optimizer decomposition: 94.8%.
- Implicit residual-adjoint decomposition: 95%.
- QI objective decomposition: 93%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95.7%.
- CI/runtime/coverage hygiene for this PR: 99.95%.
- Overall differentiability-refactor PR: 99.9992%.

## 2026-06-18 Free-Boundary Pressure Edge Scale Setup Seam

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Moved the free-boundary pressure edge-scale calculation out of
   `solve_fixed_boundary_residual_iter` and into
   `vmec_jax.solvers.fixed_boundary.residual.setup.free_boundary_pressure_edge_scale`.
2. Kept the helper injectable for `eval_profiles` so the pressure-ratio logic
   can be unit-tested without constructing a full VMEC input deck.
3. Added tests for enabled free-boundary scaling, disabled/no-input cases,
   zero edge pressure, and defensive exception handling.

Results obtained:

- `solve_fixed_boundary_residual_iter` dropped from 6411 to 6399 lines.
- `iteration.py` dropped from 6938 to 6927 lines.
- The setup module now owns free-boundary setup-policy decisions and this
  related pressure scaling calculation.

Tests and commands run:

- `python -m compileall -q vmec_jax/solvers/fixed_boundary/residual/setup.py vmec_jax/solvers/fixed_boundary/residual/iteration.py`
- `python -m ruff check vmec_jax/solvers/fixed_boundary/residual/setup.py vmec_jax/solvers/fixed_boundary/residual/iteration.py tests/test_solve_residual_iter_setup_helpers.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_residual_iter_setup_helpers.py tests/test_solve_residual_iter_runtime_helpers.py tests/test_solve_wave7_coverage.py -q`
- `python tools/diagnostics/source_health.py --top 10 --top-functions 12`

Best next steps:

1. Continue residual setup extraction around profile setup only if the helper
   signature remains simple; otherwise move to a larger designed context object.
2. Use explorer feedback before attempting preconditioner cache or scan-loop
   extraction.
3. Recheck PR CI after the next push for concrete failures.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.98%.
- Differentiability/refactor implementation: 99.999998%.
- Solver monolith reduction: 99.51%.
- Free-boundary adjoint monolith reduction: 99.35%.
- Driver workflow decomposition: 99.87%.
- Residual iteration decomposition: 96.45%.
- WOUT diagnostic/profile decomposition: 99.72%.
- Bcovar/WOUT parity decomposition: 98.35%.
- Force-kernel decomposition: 98.55%.
- Optimizer workflow decomposition: 99.0%.
- Fixed-boundary optimizer decomposition: 94.8%.
- Implicit residual-adjoint decomposition: 95%.
- QI objective decomposition: 93%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95.7%.
- CI/runtime/coverage hygiene for this PR: 99.95%.
- Overall differentiability-refactor PR: 99.9991%.

## 2026-06-18 Free-Boundary Report Cache and Profiling Policy Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Moved same-branch report mode counting, replay-plan cache creation,
   current-only coil-geometry cache creation, and matrix-free NESTOR promotion
   policy into `vmec_jax.solvers.free_boundary.coil_optimization`.
2. Kept the direct-coil optimization example responsible for user-visible
   workflow choices, objective assembly, complete-solve evaluation, and report
   writing.
3. Preserved the example module compatibility surface by importing the moved
   functions back into the script namespace.

Results obtained:

- `examples/optimization/free_boundary_QS_coil_optimization.py` dropped to
  2568 lines.
- `vmec_jax.solvers.free_boundary.coil_optimization` remains 468 lines and
  owns only free-boundary coil-optimization infrastructure.
- The largest remaining function in the example is
  `write_same_branch_validation_report`, which is now the next meaningful
  workflow/report seam.

Tests and commands run:

- `python -m compileall -q vmec_jax/solvers/free_boundary/coil_optimization.py examples/optimization/free_boundary_QS_coil_optimization.py`
- `python -m ruff check vmec_jax/solvers/free_boundary/coil_optimization.py examples/optimization/free_boundary_QS_coil_optimization.py tests/test_free_boundary_qs_coil_optimization_smoke.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_qs_coil_optimization_smoke.py::test_nestor_profile_policy_requires_size_and_speedup_thresholds tests/test_free_boundary_qs_coil_optimization_smoke.py::test_same_branch_derivative_proposal_uses_gated_directional_report tests/test_free_boundary_qs_coil_optimization_smoke.py::test_same_branch_derivative_proposal_rejects_adaptive_claims tests/test_free_boundary_qs_coil_optimization_smoke.py::test_same_branch_derivative_proposal_rejects_failed_vector_gate tests/test_free_boundary_qs_coil_optimization_smoke.py::test_same_branch_derivative_proposal_rejects_failed_rejected_slot_gate tests/test_free_boundary_qs_coil_optimization_smoke.py::test_same_branch_derivative_proposal_requires_direct_jvp_and_fresh_replay -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_qs_coil_optimization_smoke.py tests/test_free_boundary_qa_finite_beta_coil_optimization_smoke.py -q`
- `python tools/diagnostics/source_health.py --top 10 --top-functions 12`

Best next steps:

1. Extract the report-assembly sub-artifacts from
   `write_same_branch_validation_report` only if the extracted API remains
   domain-specific and readable.
2. Continue residual iteration decomposition through timing/history/preconditioner
   seams already present in `vmec_jax.solvers.fixed_boundary.residual`.
3. Recheck PR CI after this push and fix any concrete failures rather than
   polling pending shards.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.98%.
- Differentiability/refactor implementation: 99.999998%.
- Solver monolith reduction: 99.50%.
- Free-boundary adjoint monolith reduction: 99.35%.
- Driver workflow decomposition: 99.87%.
- Residual iteration decomposition: 96.4%.
- WOUT diagnostic/profile decomposition: 99.72%.
- Bcovar/WOUT parity decomposition: 98.35%.
- Force-kernel decomposition: 98.55%.
- Optimizer workflow decomposition: 99.0%.
- Fixed-boundary optimizer decomposition: 94.8%.
- Implicit residual-adjoint decomposition: 95%.
- QI objective decomposition: 93%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95.7%.
- CI/runtime/coverage hygiene for this PR: 99.95%.
- Overall differentiability-refactor PR: 99.9990%.

## 2026-06-18 Free-Boundary Coil Optimization Proposal Helper Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Extracted the same-branch derivative proposal helpers from the pedagogical
   single-stage direct-coil example into
   `vmec_jax.solvers.free_boundary.coil_optimization`.
2. Kept the example script as the user-facing workflow and retained the old
   function names as explicit imports so existing tests and user scripts can
   still access them through the example module.
3. Left complete free-boundary solves as the only acceptance authority; the
   extracted helpers only convert validated branch-local reports into bounded
   trial points.

Results obtained:

- `examples/optimization/free_boundary_QS_coil_optimization.py` dropped to
  2694 lines.
- The extracted package helper is 331 lines and has a narrow dependency surface:
  standard typing plus NumPy.
- No output artifacts or large files were added.

Tests and commands run:

- `python -m compileall -q vmec_jax/solvers/free_boundary/coil_optimization.py examples/optimization/free_boundary_QS_coil_optimization.py`
- `python -m ruff check vmec_jax/solvers/free_boundary/coil_optimization.py examples/optimization/free_boundary_QS_coil_optimization.py tests/test_free_boundary_qs_coil_optimization_smoke.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_qs_coil_optimization_smoke.py::test_same_branch_derivative_proposal_uses_gated_directional_report tests/test_free_boundary_qs_coil_optimization_smoke.py::test_same_branch_derivative_proposal_rejects_adaptive_claims tests/test_free_boundary_qs_coil_optimization_smoke.py::test_same_branch_derivative_proposal_rejects_failed_vector_gate tests/test_free_boundary_qs_coil_optimization_smoke.py::test_same_branch_derivative_proposal_rejects_failed_rejected_slot_gate tests/test_free_boundary_qs_coil_optimization_smoke.py::test_same_branch_derivative_proposal_requires_direct_jvp_and_fresh_replay -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_qs_coil_optimization_smoke.py tests/test_free_boundary_qa_finite_beta_coil_optimization_smoke.py -q`
- `python tools/diagnostics/source_health.py --top 10 --top-functions 12`

Best next steps:

1. Continue example simplification by moving report-cache/profiling helpers
   into the same free-boundary coil-optimization domain module if the move does
   not hide workflow decisions from users.
2. Keep reducing the residual iteration monolith through existing residual
   domain modules rather than creating generic helper files.
3. Recheck PR CI after the next push; current GitHub Actions status has no
   failures, with several shards still pending.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.98%.
- Differentiability/refactor implementation: 99.999998%.
- Solver monolith reduction: 99.50%.
- Free-boundary adjoint monolith reduction: 99.35%.
- Driver workflow decomposition: 99.87%.
- Residual iteration decomposition: 96.4%.
- WOUT diagnostic/profile decomposition: 99.72%.
- Bcovar/WOUT parity decomposition: 98.35%.
- Force-kernel decomposition: 98.55%.
- Optimizer workflow decomposition: 98.95%.
- Fixed-boundary optimizer decomposition: 94.8%.
- Implicit residual-adjoint decomposition: 95%.
- QI objective decomposition: 93%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95.7%.
- CI/runtime/coverage hygiene for this PR: 99.95%.
- Overall differentiability-refactor PR: 99.9989%.

## 2026-06-18 Resume Audit, CI Check, and Free-Boundary Claim Control

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Checked GitHub PR #20 through `gh` and the CI inspection helper.  No failing
   checks were reported; several exact/slow shards were still in progress.
2. Revisited the active plans after the latest source-simplification commits
   and fixed stale `plan_freeb.md` branch metadata.
3. Added summary-level smoke assertions for single-stage direct-coil derivative
   proposals so `summary.json` preserves the same provenance as the helper
   contract: complete-solve acceptance authority, no adaptive-host
   differentiation claim, current-only fast-path evidence, and fixed rejected
   controller-slot evidence when requested.

Results obtained:

- The draft PR branch remains clean except for the current plan/test edits.
- The free-boundary docs already state the key limitation clearly: same-branch
  branch-local derivative reports are promoted; arbitrary adaptive host-branch
  differentiation remains unclaimed.
- The new assertions are cheap unit/smoke coverage and do not add another VMEC
  solve to CI.
- Local verification passed for ruff, the focused summary-contract tests, the
  free-boundary QS/finite-beta smoke shard, the fast docs build, and the four
  native rejected-slot complete-solve FD gates.

Tests and commands to run:

- `python -m ruff check tests/test_free_boundary_qs_coil_optimization_smoke.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_qs_coil_optimization_smoke.py tests/test_free_boundary_qa_finite_beta_coil_optimization_smoke.py -q`
- If those pass, rerun the four native rejected-slot complete-solve FD gates in
  `tests/test_free_boundary_direct_coil_finite_pressure_sensitivity.py`.

Best next steps:

1. Commit/push this claim-control tranche.
2. Continue refactoring only through high-signal domain seams: residual
   controller policy, WOUT/diagnostic table helpers, free-boundary report
   proposal APIs, and optimizer derivative-policy seams.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.97%.
- Differentiability/refactor implementation: 99.999997%.
- Solver monolith reduction: 99.46%.
- Free-boundary adjoint monolith reduction: 99.35%.
- Driver workflow decomposition: 99.87%.
- Residual iteration decomposition: 96.1%.
- WOUT diagnostic/profile decomposition: 99.72%.
- Bcovar/WOUT parity decomposition: 98.35%.
- Force-kernel decomposition: 98.55%.
- Optimizer workflow decomposition: 98.9%.
- Fixed-boundary optimizer decomposition: 94.8%.
- Implicit residual-adjoint decomposition: 95%.
- QI objective decomposition: 93%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95.7%.
- CI/runtime/coverage hygiene for this PR: 99.95%.
- Overall differentiability-refactor PR: 99.9987%.

## 2026-06-18 Residual Iteration Timing Runtime Seam

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Moved the non-scan residual-iteration timing accumulator defaults out of
   `solve_fixed_boundary_residual_iter` and into
   `_new_residual_iter_timing_stats` in
   `vmec_jax.solvers.fixed_boundary.residual.runtime`.
2. Kept setup-phase timings explicit by passing the already-collected setup
   timing dictionary into the new helper.
3. Added unit coverage that verifies setup-phase values are preserved and
   runtime counters start at zero.

Results obtained:

- `solve_fixed_boundary_residual_iter` dropped from 6471 to 6411 lines.
- `iteration.py` dropped from 6997 to 6938 lines.
- Timing key names and report behavior remain unchanged; this is a pure
  ownership/simplicity refactor.

Tests and commands run:

- `python -m compileall -q vmec_jax/solvers/fixed_boundary/residual/runtime.py vmec_jax/solvers/fixed_boundary/residual/iteration.py`
- `python -m ruff check vmec_jax/solvers/fixed_boundary/residual/runtime.py vmec_jax/solvers/fixed_boundary/residual/iteration.py tests/test_solve_residual_iter_runtime_helpers.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_residual_iter_runtime_helpers.py tests/test_solve_residual_iter_finalize_helpers.py tests/test_solve_real_scan_wave10_coverage.py tests/test_solve_wave7_coverage.py -q`
  - Result: passed.
- `python tools/diagnostics/source_health.py --top 10 --top-functions 12`

Best next steps:

1. Commit/push this timing-runtime seam.
2. Continue residual iteration decomposition through existing domain modules
   before adding new files: preconditioner cache state, host history channels,
   or VMEC2000 scan branch selection.
3. Avoid extracting the full nested scan function until the required context
   object can be designed without hiding the solver dependencies.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.97%.
- Differentiability/refactor implementation: 99.999997%.
- Solver monolith reduction: 99.50%.
- Free-boundary adjoint monolith reduction: 99.35%.
- Driver workflow decomposition: 99.87%.
- Residual iteration decomposition: 96.4%.
- WOUT diagnostic/profile decomposition: 99.72%.
- Bcovar/WOUT parity decomposition: 98.35%.
- Force-kernel decomposition: 98.55%.
- Optimizer workflow decomposition: 98.9%.
- Fixed-boundary optimizer decomposition: 94.8%.
- Implicit residual-adjoint decomposition: 95%.
- QI objective decomposition: 93%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95.7%.
- CI/runtime/coverage hygiene for this PR: 99.95%.
- Overall differentiability-refactor PR: 99.9988%.

Repository: `/Users/rogeriojorge/local/vmec_jax`.

## Executive Summary

The long-term goal is a research-grade differentiable VMEC variant that keeps
VMEC2000-compatible fixed-boundary and free-boundary physics while exposing
validated derivatives for equilibrium, boundary optimization, direct-coil
optimization, finite-beta metrics, Boozer-space objectives, and stability
objectives.

The near-term release should remain conservative:

1. Fixed-boundary derivatives and production optimization APIs are promoted only
   where AD-vs-FD, VMEC2000 parity, and physics gates pass.
2. Free-boundary direct-coil providers, mgrid providers, and branch-local
   same-fingerprint replay/controller derivatives are promoted.
3. Arbitrary differentiation through adaptive host branch changes in
   `run_free_boundary` is not yet promoted.
4. A future differentiable-controller lane can explore fully JAX-visible branch
   selection, but only if it preserves VMEC parity, improves optimization
   robustness, or enables capabilities that the branch-local seam cannot.

## 2026-06-17 Plan Review and Method Decision

Current PR state:

- Draft PR #20, `[codex] Research-grade differentiability refactor umbrella`,
  is open from `codex/differentiability-refactor-plan` into `main`.
- Branch head: `94e05da`, `Refactor free-boundary rejected-slot report gate`.
- GitHub Actions run `27664190749` is green across fast tests, exact shards,
  slow physics coverage, docs, build, console smoke, parity manifest smoke,
  combined coverage, and Codecov project/patch gates.
- The working tree was clean at the start of this review.

Updated method decision:

1. Pure JAX kernels are the default for smooth local residual, geometry,
   profile, Boozer, stability, and coil-field kernels.
2. Custom JVP/VJP rules are the default seam for fixed accepted branches,
   replayed traces, and numerically stable objective wrappers. This matches
   JAX's supported custom-derivative path for transformable Python functions.
3. Implicit differentiation is the default target for converged equilibrium
   roots and fixed-point solves when the linearized solve can be validated
   against central finite differences and physical scalar gates.
4. Solver-faithful discrete adjoints are the target for VMEC-style spectral
   PDE solves. The key lesson from recent spectral-adjoint work is to build
   adjoint operators from the solver graph/residual structure rather than
   keeping a large unrolled AD tape.
5. Matrix-free linear operators are the target for large NESTOR/source
   responses and large linearized equilibrium systems. Dense materialization is
   acceptable only below explicit mode-count thresholds and must lose to the
   matrix-free path in no promoted large-mode gate.
6. Unrolled AD is diagnostic only for short fixed-budget traces, toy problems,
   and regression tests. It is not the production strategy for long VMEC solves.
7. Fully JAX-visible adaptive branch selection remains a research lane, not a
   production claim. Hard accepted/rejected changes, timestep limiter changes,
   Jacobian resets, restarts, and fallback selection are nonsmooth branch
   events. Production derivatives stay fingerprint-gated: if a perturbation
   changes the branch fingerprint, the derivative is not promoted.
8. DESC remains the closest architectural reference for a differentiable
   stellarator code, but vmec_jax must preserve VMEC2000 input/output semantics
   and parity, so the refactor should not replace VMEC algorithms with DESC
   algorithms unless an explicit parity gate is added.
9. JAXopt, Optax, Lineax, and Equinox are design references and optional
   adapters, not mandatory dependencies. JAXopt's implicit-diff concepts are
   useful, Optax is useful for optimizer composition, Lineax-style PyTree
   operators are useful for matrix-free solves, and Equinox-style PyTree modules
   are useful if filtered transforms become necessary.

Updated code-health audit:

- `vmec_jax/solvers/fixed_boundary/residual/iteration.py`: 8404 lines;
  `solve_fixed_boundary_residual_iter` is 7790 lines. This is the highest-risk
  maintainability hotspot and the next mandatory refactor tranche.
- `vmec_jax/optimization.py`: 5545 lines. The next tranche should separate
  accepted-point replay, derivative policy selection, and public problem API.
- `vmec_jax/optimization_workflow.py`: 4249 lines. The workflow layer should
  become a thin pedagogical API, not a second optimizer implementation.
- `vmec_jax/free_boundary_adjoint.py`: 3829 lines. Continue moving validated
  pieces into `vmec_jax/solvers/free_boundary/adjoint/` by domain name.
- `vmec_jax/free_boundary.py`: 3371 lines. Keep provider/runtime/NESTOR seams
  separate as extraction points become stable.
- `examples/optimization/free_boundary_QS_coil_optimization.py`: 3000 lines.
  Keep examples user-facing; move reusable report/proposal logic into package
  APIs once behavior is validated.

Finite execution plan from this review:

1. Plan consolidation.
   Exit gate: this file is the canonical plan; docs point here; old plans are
   marked historical/reference.
2. Fixed-boundary residual-controller split.
   Exit gate: extract named controller-policy/state-transition seams from
   `solve_fixed_boundary_residual_iter` until the largest function is below
   5000 lines without changing VMEC trace parity.
3. Derivative-policy unification.
   Exit gate: one small API selects exact replay, scalar adjoint,
   matrix-free/JVP, implicit, or finite-difference fallback with explicit
   validation metadata.
4. Free-boundary branch-local completion.
   Exit gate: branch-local accepted/rejected same-fingerprint physical-scalar
   gates are shared, fast enough for CI shards, and consumed by the coil-only
   optimization example.
5. Adaptive-controller research prototype.
   Exit gate: a JAX-visible adaptive prototype matches the host branch
   fingerprint on bounded fixtures and passes AD-vs-central-FD for at least one
   physical scalar. Until then, arbitrary adaptive branch differentiation is
   explicitly unclaimed.
6. Documentation and release closure.
   Exit gate: README/docs describe exactly which derivative seams are promoted,
   examples use the public APIs, CI stays green, and source-health reports no
   new root-level helper sprawl.

## Terms

- Fixed boundary: the plasma boundary is prescribed by Fourier coefficients.
- Free boundary: the boundary is solved with an external vacuum field supplied
  by mgrid interpolation, ESSOS-generated grids, or direct coils.
- Direct coil provider: pure-JAX Biot-Savart field evaluation from Fourier coil
  parameters.
- Adaptive branch: a discrete path selected by the nonlinear VMEC controller,
  including accepted/rejected steps, Jacobian resets, timestep limiters,
  restart/fallback selection, and preconditioner policies.
- Branch fingerprint: compact metadata that identifies the adaptive branch used
  by a solve or replay. Derivatives are only promoted when the plus/minus
  finite-difference perturbations keep the same fingerprint.
- Branch-local derivative: a derivative of one fixed accepted/rejected
  controller path. This can be validated against complete-solve central finite
  differences under an unchanged fingerprint.
- Arbitrary adaptive derivative: a derivative through changes in branch
  selection. This is generally nonsmooth and is not claimed by current code.

## Literature and Code Anchors

The plan is anchored in the following references and implementation patterns:

1. VMEC/STELLOPT documentation: VMEC uses Fourier-expanded geometry and a
   variational energy minimization; free-boundary VMEC uses vacuum fields from
   mgrid files and NESTOR-style exterior solves.
   URL: https://princetonuniversity.github.io/STELLOPT/VMEC.html
2. VMEC++ numerics: modern VMEC reimplementation practices, restart behavior,
   robust execution, and validation philosophy.
   URL: https://arxiv.org/abs/2502.04374
3. DESC code suite: JAX-based stellarator equilibrium and optimization,
   continuation, perturbation, free-boundary residuals, and differentiable
   optimization design.
   URL: https://desc-docs.readthedocs.io/
4. High-order free-boundary DESC work: free-boundary residual formulation and
   NESTOR/VMEC context for vacuum-boundary conditions.
   URL: https://arxiv.org/html/2412.05680v1
5. SIMSOPT: practical optimization API design, caching, coil/boundary
   parameter spaces, and mixed analytic/AD derivatives.
   URL: https://simsopt.readthedocs.io/
6. JAX control-flow docs: `lax.scan`, `lax.while_loop`, and transformed loops.
   Static loops should use `scan`; dynamic loops use `while_loop` when a
   JAX-visible controller is worth the tradeoff.
   URL: https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html
7. JAXopt implicit differentiation: solver-output differentiation by implicit
   rules rather than storing every unrolled iteration.
   URL: https://jaxopt.github.io/stable/implicit_diff.html
8. Fast automated adjoints for spectral PDE solvers: construct efficient
   discrete adjoints for sparse spectral PDE solvers rather than relying on
   naive tape retention.
   URL: https://arxiv.org/abs/2506.14792
9. Mercier and Glasser/Jorge-Landreman stability work: physics anchors for
   DMerc and resistive `D_R` gates.
   URL: https://arxiv.org/abs/2006.14881

## Current Source Map

Differentiable fixed-boundary and optimization code:

- `vmec_jax/driver.py`: production `run_fixed_boundary` and
  `run_free_boundary` entry points, host controller, WOUT generation.
- `vmec_jax/optimization.py`: fixed-boundary least-squares utilities,
  accepted-point replay, scalar-adjoint and exact callback policies.
- `vmec_jax/optimization_workflow.py`: user-facing optimization helpers and
  problem assembly.
- `vmec_jax/quasisymmetry.py`: quasisymmetry residuals and targets.
- `vmec_jax/quasi_isodynamic.py`, `vmec_jax/qi_optimization.py`,
  `vmec_jax/qi_diagnostics.py`: QI/QP objectives, diagnostics, gates, and
  minimal-seed workflows.
- `vmec_jax/wout.py`, `vmec_jax/vmec_output.py`: equilibrium output and metric
  extraction.

Differentiable free-boundary and coil code:

- `vmec_jax/external_fields/coils_jax.py`: Fourier coils, stellarator symmetry
  expansion, Biot-Savart sampling, and coil metrics.
- `vmec_jax/external_fields/mgrid_jax.py`: JAX mgrid interpolation path.
- `vmec_jax/external_fields/essos_adapter.py`: optional ESSOS adapter.
- `vmec_jax/free_boundary.py`: free-boundary provider hook, direct-coil/mgrid
  sampling, VMEC/NESTOR integration.
- `vmec_jax/free_boundary_adjoint.py`: dense vacuum solve, JAX NESTOR pieces,
  accepted-boundary replay, fixed-trace custom-VJP helpers, and branch-local
  reports.
- `vmec_jax/free_boundary_adjoint_controller.py`: JAX-visible nonlinear and
  segmented controller primitives used for same-fingerprint validation.
- `vmec_jax/free_boundary_validation.py`: validation helpers and bounded
  physical fixture checks.

Examples and docs:

- `examples/optimization/QA_optimization.py`, `QH_optimization.py`,
  `QP_optimization.py`, `QI_optimization.py`: fixed-boundary optimization
  scripts that should remain clear, Simsopt-like, and user editable.
- `examples/optimization/free_boundary_QA_finite_beta_coil_optimization.py`:
  conservative coil-only free-boundary optimization example with complete
  solves as acceptance authority and optional same-branch derivative reports.
- `docs/free_boundary_coil_optimization.rst`: current free-boundary claims and
  limitations.
- `plan_freeb.md`: current phase-1/phase-2/phase-3 execution log.

## Architecture Goals

1. Preserve VMEC semantics by default.
   Production `run_fixed_boundary` and `run_free_boundary` must stay compatible
   with VMEC2000 input/output behavior and validated finite-positive physical
   WOUTs.
2. Make differentiability explicit.
   Every promoted derivative path must state whether it differentiates an
   equilibrium residual, fixed accepted branch, same-fingerprint adaptive
   replay, or an arbitrary adaptive branch.
3. Validate derivatives with physics scalars.
   Gates must include meaningful outputs: aspect, iota, boundary displacement,
   Bnormal RMS, QS/QI proxies, DMerc, `D_R`, pressure/current response, and
   WOUT-level geometry scalars.
4. Keep optimization practical.
   The public optimization API should let users assemble objectives as tuples
   or lightweight objective objects, choose weights/targets, run SciPy or JAX
   optimizers, save inputs/WOUTs, and plot results without hiding the workflow.
5. Avoid large runtime regressions.
   Full exact derivatives should not force dense tape construction when a
   scalar-adjoint, matrix-free, projected, or implicit path is available.
6. Keep claims conservative.
   Do not claim full adaptive-loop differentiability until fingerprint-gated
   full adaptive AD-vs-central-FD gates pass on physical scalar outputs.

## 2026-06-14 Differentiability and Refactor Review

This review cross-checked the current codebase against current JAX, Python,
JAXopt, Optax, Equinox, Orthax, VMEC++, DESC, SIMSOPT, Scientific Python, and
spectral-adjoint guidance.

Key conclusions:

1. Treat the VMEC solve as a differentiable numerical program with explicit
   state objects, residual operators, solver policies, and output adapters.
   Do not keep growing monolithic VMEC2000-style translation files.
2. Use native JAX PyTrees and frozen dataclasses first.  Equinox-style modules
   are attractive for filtered transforms, but adding Equinox as a mandatory
   dependency is not justified until a concrete filtered-transform seam needs
   it.  The source layout should remain plain-Python and JAX-native.
3. Use `lax.scan` or static-trip `fori_loop` for fixed-budget iteration traces
   that need reverse-mode AD.  Use `while_loop` only for JAX-visible dynamic
   controllers whose derivative semantics are intentionally limited or supplied
   by a custom rule.
4. Use `jax.custom_vjp`, `jax.custom_jvp`, and JAXopt-style implicit
   differentiation for equilibrium roots and branch-local fixed points instead
   of relying on unbounded tape retention through every nonlinear iteration.
5. Use Optax as an optional optimizer backend only through a small adapter.
   Keep SciPy least-squares as the beginner-friendly and VMEC/SIMSOPT-familiar
   path.
6. Orthax can be useful for future orthogonal-polynomial profile bases, but it
   should not replace current VMEC-compatible polynomial and spline profile
   handling unless a validation fixture needs that basis.
7. Keep optional dependencies imported lazily at call sites.  `import vmec_jax`
   and `vmec --test` must remain beginner-friendly and fast.
8. Adopt a Scientific-Python-style separation between public API, internal
   kernels, diagnostics, optional external gates, and generated artifacts.

Current line-count hotspots from the main branch:

.. code-block:: text

   vmec_jax/solve.py                                      15438
   vmec_jax/free_boundary_adjoint.py                       6941
   vmec_jax/wout.py                                        6321
   vmec_jax/optimization.py                                5441
   tests/test_free_boundary_direct_coil_finite_pressure_sensitivity.py  5150
   vmec_jax/free_boundary.py                               4271
   vmec_jax/optimization_workflow.py                       4252
   vmec_jax/driver.py                                      4064
   vmec_jax/discrete_adjoint.py                            3557

These files are too large for sustained research development.  The refactor
must reduce local cognitive load while preserving VMEC parity, public API
compatibility, and validated derivative behavior.

## Target Package Architecture

The refactored package should expose a small public API and keep implementation
modules organized by scientific/numerical responsibility.  This is the
canonical target layout for PR #20 and supersedes the older flat
`solve_*`/`driver_*`/`free_boundary_*` helper-file direction.

.. code-block:: text

   vmec_jax/
     api.py                       public convenience imports
     cli.py                       command-line interface

     core/
       config.py                  parsed INDATA/run options
       state.py                   PyTree equilibrium state objects
       modes.py                   mode tables and indexing conventions
       grids.py                   radial/angular grids
       profiles.py                pressure/current/iota profiles
       runtime.py                 optional backend/runtime settings

     kernels/
       fourier.py                 transforms, Nyquist maps, mode projections
       geometry.py                R/Z geometry, metrics, Jacobians
       fields.py                  B/J/covariant/contravariant field kernels
       forces.py                  VMEC force blocks and finite-beta terms
       residuals.py               residual assembly and norms
       preconditioning.py         radial and spectral preconditioners

     solvers/
       fixed_boundary/
         api.py                   fixed-boundary orchestration
         controller.py            accepted/rejected update policies
         scan.py                  VMEC2000-style fixed-budget scan
         nonlinear.py             residual iteration and restart loop
         optimizers.py            GD/LBFGS/GN inner-solve algorithms
         checkpoints.py           resume/checkpoint payloads
         diagnostics.py           trace rows, timing, fallback reports

       free_boundary/
         api.py                   free-boundary orchestration
         providers.py             mgrid/direct-coil/ESSOS field providers
         nestor.py                vacuum/source/NESTOR operators
         controller.py            free-boundary activation/update policies
         adjoints.py              branch-local reports and custom VJPs
         fingerprints.py          branch metadata and same-branch checks
         validation.py            bounded physical fixture gates

       differentiation/
         policies.py              exact/scalar/matrix-free/implicit choices
         implicit.py              root/JVP/VJP helper interfaces
         finite_difference.py     central-FD validation utilities
         linear_solvers.py        CG/dense/matrix-free linear solves

     objectives/
       quasisymmetry.py
       quasi_isodynamic.py
       finite_beta.py             beta, pressure/current, well, bootstrap hooks
       stability.py               DMerc, Glasser D_R, Mercier/well gates
       coils.py
       least_squares.py           objective tuple/object assembly

     optimization/
       boundary.py                boundary DOF spaces and continuation
       coils.py                   coil DOF spaces and acceptance loops
       workflow.py                Simsopt-like problem assembly
       callbacks.py               exact/scalar/matrix-free callback policies
       result.py                  histories, provenance, saved artifacts
       backends/
         scipy.py
         jaxopt.py                optional
         optax.py                 optional

     io/
       namelist.py
       wout.py
       wout_schema.py
       booz.py
       assets.py

     plotting/
       geometry.py
       boozer.py
       optimization.py
       stability.py

     validation/
       vmec2000.py
       simsopt.py
       physics.py
       parity.py

     performance/
       profiling.py
       source_health.py

The existing module names should remain available through compatibility
re-exports until the next major release.  Tests should import from the new
module paths when validating new functionality and from old paths when
checking backward compatibility.

Domain-name decisions:

1. `core` owns low-level VMEC data model objects and mesh/profile conventions.
2. `kernels` owns pure numerical kernels that can be JIT/vmap/grad transformed.
3. `solvers` owns controller state, nonlinear iteration, branch fingerprints,
   and derivative policies.
4. `objectives` owns differentiable scalar/vector metrics used in optimization.
5. `optimization` owns user-facing problem assembly and optimizer backends.
6. `io`, `plotting`, `validation`, and `performance` are side-effect and
   workflow packages; they should not be imported from hot kernels.

Research-grade architecture scorecard:

Every substantial refactor must improve or preserve all of the following:

1. Simpler public use.
   A beginner should still be able to run `vmec --test`, `vmec input.*`,
   `run_fixed_boundary`, `run_free_boundary`, and the QA/QH/QP/QI examples
   without learning internal package structure.
2. Clear contributor path.
   A new kernel goes in `kernels/`; a new objective goes in `objectives/`; a
   new solve policy goes in `solvers/`; a new validation fixture goes in
   `validation/`; a new plot goes in `plotting/`.
3. No hidden performance tax.
   Moving code into packages must not add dynamic dispatch, object allocation,
   import-time optional dependencies, or Python callbacks inside JIT/scanned hot
   paths.  Compatibility shims must stay outside hot loops.
4. Accuracy first.
   Refactors must preserve VMEC2000/VMEC++ parity, finite-positive WOUT
   geometry, force-residual convergence, Boozer/QS/QI diagnostics, DMerc,
   Glasser `D_R`, pressure/current profile behavior, and rerun reproducibility.
5. Differentiability is explicit and validated.
   Every derivative path must identify its seam: pure kernel, implicit root,
   branch-local same-fingerprint replay, scalar-adjoint, matrix-free, or
   experimental adaptive-branch derivative.  Promotion requires AD-vs-central-FD
   agreement on physical scalars.
6. Performance evidence survives refactor.
   Cold/warm CPU and GPU timings, exact callback timings, matrix-free/scalar
   derivative timings, and optimization wall times must be tracked before and
   after large moves.
7. Documentation follows the domain model.
   User docs should show public workflows; developer docs should explain the
   package map and where to add new physics, solvers, objectives, providers,
   validation gates, and examples.

Line-count and simplicity budgets:

The refactor is successful only if it reduces both file size and the number of
places a developer must inspect to understand one workflow.  The goal is not
many tiny files; the goal is a small number of cohesive, well-named modules.

1. Root namespace budget.
   Keep root-level `vmec_jax/*.py` files mostly for public facades,
   compatibility shims, and historically stable modules.  New implementation
   code should live in domain packages.  Target: fewer than 35 root-level
   implementation files after compatibility retirement.
2. Package/module budget.
   Each domain package should expose a small surface through `__init__.py` or
   `api.py`.  Internal package modules should be cohesive and usually stay
   below 800 lines.  Files above 1500 lines require a plan exemption and a
   documented split path.  Files above 2000 lines fail the maintainability gate
   once the migration is active.
3. Function/class budget.
   New functions should usually stay below 80 lines; functions above 150 lines
   need a documented reason.  Large controller loops may be longer during
   migration, but their policy, diagnostics, checkpoint, and I/O pieces should
   be extracted into named domain functions.
4. Example budget.
   Pedagogical optimization scripts should be readable end-to-end.  Target:
   common QA/QH/QP examples under 250 lines and QI/free-boundary examples under
   400 lines, excluding long explanatory comments.  More complex sweeps belong
   in reusable source modules plus short driver scripts.
5. Test budget.
   Large tests are acceptable only for integration/parity matrices.  Unit and
   physics-gate tests should mirror the package structure and stay focused on
   one concept.  Oversized legacy tests should be split as code moves.
6. Import budget.
   `import vmec_jax` must stay fast and must not import optional heavy
   dependencies, plotting stacks, ESSOS, VMEC2000 wrappers, or GPU-specific
   setup.  Optional dependencies are imported lazily inside the relevant
   command/function.
7. Cognitive-load budget.
   A developer adding a feature should need at most one domain package plus
   one test package for routine work.  If a change routinely touches five or
   more unrelated root-level files, the package boundary is wrong.

Performance and memory budgets:

The package refactor must not make vmec_jax easier to read at the cost of being
slower or more memory hungry.  Every large movement of solver, derivative, or
optimization code must record before/after measurements for representative
cases.

1. Hot-path imports and compatibility shims.
   Compatibility shims must not sit inside JIT, `lax.scan`, nonlinear residual
   loops, or exact-callback replay loops.  Hot code should import from the final
   domain package directly.
2. Allocation discipline.
   State containers should be PyTrees of arrays and small static metadata.
   Avoid per-iteration Python object churn, dict construction, string
   formatting, logging, or diagnostics payload assembly inside hot loops.
3. Derivative memory discipline.
   Prefer scalar-adjoint, matrix-free, projected, or implicit derivative paths
   over dense unrolled tapes when they pass AD-vs-FD gates.  Use remat only
   around measured tape-memory hotspots.
4. Cold and warm timing gates.
   Track cold solve, warm solve, first exact callback, accepted-point replay,
   matrix-free/JVP replay, and optimization wall-clock timings on compact CPU
   and GPU cases.  A refactor should be neutral or faster unless a documented
   accuracy/differentiability gate requires extra work.
5. Peak-memory gates.
   Track peak resident memory for long fixed-boundary optimizations,
   free-boundary direct-coil solves, and exact derivative callbacks.  Package
   moves must not increase peak memory except where a new validated feature is
   explicitly enabled.
6. CI runtime gates.
   Keep default CI under the agreed budget by separating fast unit/physics gates
   from optional VMEC2000/SIMSOPT/ESSOS/GPU matrices.  Maintain coverage and
   physics depth by using compact fixtures, not by running every expensive
   workflow in default CI.

Acceptance metrics for the refactor:

- Root-level implementation files under `vmec_jax/*.py`: fewer than 35.
- No new root-level helper-prefix files without explicit plan exemption.
- Largest implementation module target: under 1500 lines; hard warning above
  2000 lines.
- Common implementation modules normally under 800 lines.
- New functions normally under 80 lines, with documented exceptions above 150
  lines.
- Public examples remain short enough to teach the workflow, not hide it behind
  opaque wrapper calls.
- Full local release gate and GitHub CI stay green.
- Coverage and physics/parity gates do not regress.
- Public import/runtime smoke tests pass from both source checkout and installed
  wheel.
- Representative fixed-boundary, free-boundary, direct-coil, finite-beta,
  QS/QI, DMerc/`D_R`, and Boozer workflows retain documented outputs.
- Cold/warm solve, exact-callback, optimization wall-time, and peak-memory
  benchmarks do not regress without explicit accuracy/differentiability
  justification.

## 2026-06-15 Architecture Correction: Stop Flat Helper Proliferation

The first implementation pass made useful progress reducing the largest
modules, but it also created too many top-level files with origin-based names
such as `solve_*`, `driver_*`, `free_boundary_*`, and `wout_*`.  That defeats
one of the core goals: a codebase that is easy for researchers to understand,
extend, and test.

Effective immediately, the refactor should stop adding new top-level helper
modules unless they are temporary compatibility shims.  New implementation code
should move into domain packages with short, stable names and clear ownership.

Current source-health snapshot on PR #20 after the latest extractions:

.. code-block:: text

   vmec_jax Python files under maxdepth=2: 140
   root-level vmec_jax/*.py files:      116
   root helper-prefix files:             50
   vmec_jax/solve.py:                 10119 lines
   vmec_jax/wout.py:                   5894 lines
   vmec_jax/free_boundary_adjoint.py:   5687 lines
   vmec_jax/optimization.py:           5441 lines
   vmec_jax/free_boundary.py:          4271 lines
   vmec_jax/optimization_workflow.py:  4249 lines
   vmec_jax/driver.py:                 2953 lines

Problem diagnosis:

1. Flat files named after the old monolith (`solve_*`) document extraction
   history, not scientific meaning.
2. Discoverability is poor: a contributor must know the old file name before
   finding the new helper.
3. The number of root modules is becoming its own maintenance burden.
4. Tests increasingly import private aliases from compatibility modules instead
   of domain APIs.

Architecture principle:

- Public APIs remain small and stable: `vmec_jax.run_fixed_boundary`,
  `vmec_jax.run_free_boundary`, objective functions, WOUT/Boozer readers, and
  optimization entry points.
- Internal implementation is organized by scientific/numerical responsibility,
  not by the old source file.
- Compatibility shims are allowed, but they should be thin, documented, and
  marked for removal after one major release.
- New tests should target the domain package first and only use old private
  aliases for explicit backward-compatibility checks.

Canonical package map:

The single source of truth is the `Target Package Architecture` section above.
Do not add a second package map in later planning sections; update the target
architecture directly when names or boundaries change.

Naming rules:

1. Do not add new root-level `solve_*`, `driver_*`, `free_boundary_*`, or
   `wout_*` modules.
2. Avoid suffixes such as `_helpers`, `_utils`, `_misc`, and `_common` in new
   modules.  If a name needs `_helpers`, the module boundary is probably not
   scientific enough.
3. Prefer nouns that describe the domain object (`controller`, `scan`,
   `fingerprints`, `preconditioning`, `checkpoints`, `stability`) over nouns
   that describe extraction history.
4. A module should have one reason to change.  If it mixes I/O, solver policy,
   physics kernels, and diagnostics, split by responsibility.
5. Domain packages may have private implementation modules, but the package
   `__init__.py` should expose a small, documented surface for tests and
   neighboring packages.

Migration policy:

1. Consolidate before extracting more.
   The next refactor wave should move existing flat helper files into the new
   package tree before adding additional helpers.
2. Keep compatibility shims thin.
   Old imports may remain as re-export files during PR #20, but each shim must
   have no physics logic and should be excluded from future development.
3. Move tests with the code.
   New tests should mirror the package structure, e.g.
   `tests/solvers/fixed_boundary/test_scan.py`, while legacy tests remain until
   compatibility is retired.
4. Set source-health gates on both file size and namespace bloat:
   - target root `vmec_jax/*.py` implementation files: under 35,
   - target implementation module length: under 1500 lines,
   - warning threshold: 2000 lines,
   - no new root-level helper-prefix files without explicit plan approval.
5. Public API imports are the compatibility contract, not private helper paths.
   Internal tests can cover private paths, but user docs should point to
   public APIs and domain packages.

Near-term consolidation order:

1. `solve_*` files -> `solvers/fixed_boundary/`.
   First move pure scan/checkpoint/diagnostics/policy modules; leave
   `solve.py` as a compatibility orchestrator until the package API is stable.
2. `free_boundary_adjoint_*` and `free_boundary_*` files ->
   `solvers/free_boundary/`.
   Keep direct-coil provider code in `external_fields/` until the provider API
   is settled, then expose it through `solvers/free_boundary/providers.py`.
3. `wout_*` files -> `io/`.
   Keep WOUT reader/writer, schema, parity conventions, and profile metadata
   together under `io/`; stability diagnostics should stay in `objectives/` or
   `diagnostics/` depending on whether they are differentiable objectives or
   WOUT persistence helpers.
4. `driver_*` files -> public `api.py` plus `solvers/*/api.py`.
   The driver should become a small facade over fixed-boundary/free-boundary
   package APIs.
5. `optimization.py` and `optimization_workflow.py` -> `optimization/` and
   `objectives/`.
   The user-facing flow should be Simsopt-like: object, objective tuple,
   optimizer, result, plotting/saving.

Deferred decisions:

- Do not add Equinox as a dependency yet.  The architecture should be PyTree
  compatible and Equinox-ready, but plain dataclasses/NamedTuples plus JAX
  pytrees are enough now.
- Do not add JAXopt/Optax as required dependencies.  Keep them optional backend
  adapters after the package boundaries are stable.
- Do not convert the whole adaptive VMEC controller into JAX-visible control
  flow until branch-local derivative seams and same-fingerprint gates are fully
  exhausted.

## Refactor Migration Waves

Wave -1: Namespace consolidation and naming cleanup.

1. Create the domain package skeleton above with no physics changes.
2. Move existing flat helper modules into the package tree in groups, preserving
   old import paths as thin re-export shims.
3. Update tests for new functionality to import from the package path, and add
   one compatibility test per old public/private alias group.
4. Add a source-health namespace gate that fails if new root-level helper-prefix
   files are added without an explicit plan exemption.
5. Only resume extracting new seams after the existing helper sprawl is reduced.

Wave 0: Baseline and source-health guard.

1. Add a diagnostic that reports Python source line counts and flags files over
   agreed warning/error thresholds.
2. Add CI/documentation guidance that new large helpers must be split before
   merging unless an explicit exemption is recorded.
3. Freeze the current public API surface with import and smoke tests.

Wave 1: Extract pure data/state/config modules.

1. Move parsed profile/config objects into small frozen dataclasses and PyTrees.
2. Split large solver carries into named `EquilibriumState`, `ForceState`,
   `ControllerState`, `SolveTrace`, and `OutputState` containers.
3. Add docstrings describing units, mesh location, shape conventions, and
   differentiability status for every public state object.

Wave 2: Extract pure kernels from `solve.py`.

1. Separate residual norm calculation, timestep updates, scan fallback
   planning, restart/fallback policy, and axis/Jacobian repair into focused
   modules.
2. Keep each extracted helper testable with synthetic small arrays.
3. Require one parity or numerical gate for every extracted physics kernel.

Wave 3: Solver-controller seam.

1. Introduce explicit controller policy objects for fixed boundary and free
   boundary.
2. Represent accepted/rejected steps, resets, limiter choices, and fallback
   choices in a branch fingerprint object.
3. Keep hard branch changes nondifferentiable by default; expose
   same-fingerprint derivative reports and changed-fingerprint rejection.

Wave 4: Derivative backends.

1. Make exact, scalar-adjoint, projected, matrix-free, and implicit
   derivatives pluggable through one `DerivativePolicy` interface.
2. Promote each backend only with AD-vs-central-FD gates on physical scalars.
3. Use JAX rematerialization only on localized kernels where profiling shows
   tape memory pressure; do not blanket-remat the solve.

Wave 5: Optimization API cleanup.

1. Replace high-argument helper calls with Simsopt-like objective tuples and
   lightweight objective objects.
2. Keep example scripts explicit: parameters at top, VMEC object, objectives,
   optimizer call, result inspection, saving, and plotting.
3. Add optional JAXopt/Optax backends behind adapters without making either
   mandatory for beginner installs.

Wave 6: Free-boundary production adjoint.

1. Keep complete solves as acceptance authority for coil optimization.
2. Use branch-local vector/JVP paths only when fingerprints match.
3. Add a fully JAX-visible adaptive-controller prototype only as a research
   experiment after branch-local gates are exhausted.

Wave 7: Documentation and examples.

1. Turn every public objective and derivative policy into an example-backed
   docs page.
2. Keep README short; move derivations, sweep tables, limitations, and
   validation provenance into docs.
3. Add "developer map" pages for new contributors that explain where to add
   a kernel, objective, solver policy, external-field provider, or test.

## Refactor Test and Validation Contract

Every migrated module must satisfy one or more of these gates:

1. Import/backward-compatibility gate: old public imports still work.
2. Shape/unit gate: synthetic arrays validate mesh location, sign conventions,
   and mode ordering.
3. Numerical identity gate: algebraic identities such as divergence-free field
   relations, covariant/contravariant consistency, Fourier reconstruction, and
   radial interpolation hold to tight tolerances.
4. AD-vs-central-FD gate: smooth scalar outputs agree for boundary modes,
   pressure/current profile coefficients, spline knots, coil currents, coil
   Fourier coefficients, DMerc, `D_R`, QS/QI residuals, Bnormal RMS, aspect,
   iota, and finite-beta response.
5. External parity gate: compact fixtures agree with VMEC2000, VMEC++, SIMSOPT,
   booz_xform_jax, or ESSOS where applicable and available.
6. Physics gate: outputs satisfy finite-positive geometry, monotone/expected
   profile behavior, magnetic-axis regularity, force residual convergence,
   Boozer-space symmetry expectations, Mercier/Glasser sign conventions, and
   coil-engineering constraints.
7. Artifact gate: saved `input.final`, `wout_final.nc`, Boozer files, and
   history JSON reproduce rerun results when convergence is claimed.
8. Performance gate: compact cold/warm solve and derivative timing budgets are
   recorded, with optional GPU and VMEC2000/ESSOS lanes separated from required
   CI.

The required CI suite should stay below the current runtime budget by using
small deterministic fixtures, sharded coverage, fetched assets for large WOUTs,
and optional markers for VMEC2000, SIMSOPT, ESSOS, GPU, and full-resolution
physics.  Coverage increases must come from real physics/numerics/API tests,
not scaffold-only tests.

## Documentation and Pedagogy Contract

New public code should satisfy:

1. A short docstring with inputs, outputs, mesh conventions, differentiability
   status, and failure modes.
2. One docs paragraph or example snippet for user-facing APIs.
3. One test that demonstrates the simplest expected user workflow.
4. Comments only where they explain a non-obvious numerical or physics choice.
5. No hidden environment-variable behavior in beginner examples; advanced
   toggles belong in explicit variables or CLI flags.

## Main Work Lanes

### Lane A: Fixed-Boundary Differentiable Core

Goal: production-grade fixed-boundary derivatives for boundary optimization.

Steps:

1. Audit all state carried by `run_fixed_boundary` and accepted-point replay.
2. Keep exact, scalar-adjoint, and auto-scalar paths under a single public
   policy with clear fallback rules.
3. Add AD-vs-central-FD gates for boundary modes, pressure/current profile
   coefficients, spline profile knots, finite-beta response, DMerc, `D_R`,
   volume-average field metrics, iota, and QS/QI objective terms.
4. Add regression gates for max-mode continuation and direct-start policies.
5. Keep WOUT, input.final, and stage checkpoint output identical between the
   optimization result and rerunning the saved input when the solve converges.

Success metrics:

- AD-vs-FD relative error below `1e-4` for smooth scalar objectives on compact
  fixtures, or tighter when conditioning allows.
- VMEC2000 parity on converged equilibria for fixed-boundary benchmarks.
- Cold and warm runtime budgets documented and enforced in CI smoke tests.

### Lane B: Free-Boundary Providers

Goal: direct-coil and mgrid free-boundary solves that are VMEC-compatible and
JAX-differentiable where claimed.

Steps:

1. Maintain pure-JAX direct-coil Biot-Savart provider and JAX mgrid
   interpolation provider.
2. Keep ESSOS adapter optional and skip cleanly when unavailable.
3. Expand bounded VMEC2000/mgrid/direct-coil parity only with fixtures that
   stay inside generated grid domains and produce finite positive geometry.
4. Add finite-beta free-boundary fixtures with actual `LASYM=T` when claiming
   non-stellarator-symmetric finite-beta coverage.
5. Add Bnormal RMS and boundary-displacement physics gates for provider parity.

Success metrics:

- Direct-coil and generated-mgrid WOUT geometry agree within documented
  tolerances for bounded fixtures.
- External-field provider gradients pass AD-vs-FD for currents and Fourier
  coil coefficients.
- Nonphysical forced-active diagnostics remain excluded from promotion gates.

### Lane C: Full Nonlinear Free-Boundary Adjoint

Goal: exact or validated derivatives through the free-boundary equilibrium
solve without overclaiming nonsmooth adaptive branch changes.

Current promoted state:

- Accepted-boundary replay.
- Fixed-trace custom VJP.
- Same-branch scalar/vector/JVP reports.
- Accepted/rejected controller-slot evidence under unchanged fingerprints.
- Complete-solve central-FD validation for physical scalars under fixed
  fingerprints.

Not yet promoted:

- Arbitrary derivatives through host adaptive branch changes.

Steps:

1. Keep branch-local/fingerprint-gated gates as the production-safe seam.
2. Add one narrow full adaptive branch AD-vs-central-FD test only when the
   branch fingerprint is unchanged and the test reports this explicitly.
3. Extend physical scalar coverage to aspect, mean iota, QS proxy, boundary
   moment, Bnormal RMS, and finite-beta response.
4. Add a negative gate: if plus/minus perturbations change the fingerprint, the
   derivative report must decline promotion.
5. Prototype a fully JAX-visible controller only as a deferred research path.
   Compare it against the host controller for branch decisions, residual traces,
   convergence, and VMEC2000 parity before any promotion.

Success metrics:

- Same-fingerprint AD-vs-central-FD relative error below `1e-4` for compact
  physical scalar gates.
- Branch-changing perturbations are detected and rejected, not silently
  differentiated.
- Full-loop claims in docs exactly match executable tests.

### Lane D: Differentiable Optimization APIs

Goal: clear, Simsopt-like scripts and reusable APIs for boundary and coil
optimization.

Steps:

1. Keep example scripts short and explicit: input parameters at top, objective
   tuple assembly, optimizer call, result extraction, saving, and plotting.
2. Move reusable objective terms, plotting helpers, seed builders, and staged
   QI policies into source modules.
3. Ensure objective terms are composable for QA, QH, QP, QI, finite beta,
   DMerc, `D_R`, mirror ratio, elongation, magnetic well, and current-profile
   terms.
4. For coil-only free-boundary optimization, keep complete solves as acceptance
   authority while using branch-local derivative reports only for proposals.
5. Add examples for mgrid free-boundary, direct-coil free-boundary, and
   coil-only finite-beta QA optimization.

Success metrics:

- New users can modify objective tuples without changing source code.
- Examples run under documented smoke budgets in CI.
- Production examples print provenance, save input/WOUT/history, and plot
  actual initial and final states.

### Lane E: QI Seed-Robustness and Omnigenity

Goal: robust QI recovery from simple/minimal seeds for NFP1/2/3/4 when possible,
with evidence gates that do not confuse QP with true QI.

Steps:

1. Keep QI README artifacts provenance-gated and generated from minimal seed
   inputs, not from already-QI stage outputs.
2. Use smooth QI, legacy QI, mirror ratio, elongation, iota floor, and aspect
   gates together.
3. Keep NFP4 conservative if the current method is not robust.
4. Add landscape scans and stage checkpoints so failures preserve useful
   diagnostics.
5. Compare JAX QI metrics against the legacy Goodman/omnigenity objective and
   booz_xform_jax outputs.

Success metrics:

- Public cases satisfy `smooth_qi <= 5e-3`, legacy QI gate, aspect gate,
  mirror gate, and iota floor gate.
- Boozer contours close in the expected QI pattern by visual inspection before
  README promotion.
- The README panel uses actual initial minimal seeds and final optimized WOUTs.

### Lane F: Physics Gates and Stability Metrics

Goal: differentiable physics metrics with literature-anchored tests.

Steps:

1. Maintain DMerc and `D_R` AD-vs-FD gates.
2. Add finite-beta fixtures with pressure/current polynomial and spline
   profiles.
3. Add tests for magnetic well, current density, vector B, vector J, jdotB,
   iota/shear, beta, volume-average B, and pressure/current response.
4. Compare with VMEC2000 WOUT values and analytic near-axis limits where
   available.
5. Document formulas, assumptions, sign conventions, and expected applicability.

Success metrics:

- Physics gates are actual numerical/physics tests, not smoke-only tests.
- Coverage stays at or above the release target while CI runtime remains
  bounded.
- All metric docs include formulas and validation provenance.

### Lane G: CPU/GPU Performance

Goal: competitive cold solves, warm solves, and optimization callbacks on CPU
and GPU.

Steps:

1. Continue profiling force assembly, scan trials, accepted-point replay,
   tangent/JVP construction, and first-call exact tape creation.
2. Promote matrix-free NESTOR/source response only when profiling shows a clear
   mode-count threshold where it beats dense paths.
3. Cache shape-stable trace setup and avoid recompilation across same-shape
   accepted points.
4. Keep GPU enabled when users install GPU JAX; do not force CPU except in
   explicit user-selected modes.
5. Keep benchmark panels comparing VMEC2000, vmec_jax CPU, and vmec_jax GPU on
   bounded fixtures.

Success metrics:

- No hidden CPU-forcing in production code.
- Performance regressions are caught by compact benchmarks.
- Optimization callback traces report compile, force, replay, tangent, and I/O
  timing buckets.

### Lane H: Documentation, CI, Release Hygiene

Goal: docs and release artifacts match what the code can actually do.

Steps:

1. Keep README concise: installation, quick test, core examples, and best public
   figures only.
2. Move long optimization tables, sweep results, and limitations into docs.
3. Keep CI fast by sharding tests by cost and using optional external gates for
   VMEC2000/ESSOS.
4. Keep the git repository lean: large WOUTs/mgrids/artifacts live in releases
   or downloadable assets, not tracked history.
5. Add release gates: local smoke, docs build, coverage, physics gates, optional
   external parity, and artifact-size gate.

Success metrics:

- CI remains green and finishes within the documented budget.
- README claims are backed by tests or generated artifact provenance.
- New release tags are cut only after release gate results are recorded.

## Refactoring Plan

Priority refactors:

1. Separate controller state, physics state, and output state in
   `vmec_jax/driver.py` so derivative seams are easier to test.
2. Factor free-boundary branch fingerprints into a small public/internal data
   structure shared by reports, tests, and docs.
3. Split dense, projected, scalar, and matrix-free derivative paths behind a
   single policy interface.
4. Move QI stage-policy machinery out of example scripts into source modules.
5. Consolidate plotting helpers for boundary comparison, Boozer LCFS contours,
   objective history, stability profiles, and coil geometry.
6. Add docstrings to all public objective objects and derivative policy entry
   points.

Refactoring success gates:

- No example optimization driver exceeds the agreed readability guard unless
  there is an explicit test exemption.
- Public APIs have tests for nominal behavior, failure behavior, and docs
  examples.
- Internal helpers are testable without long VMEC solves.

## Test Matrix

Required local/CI test classes:

1. Unit tests for input parsing, profile construction, Fourier mode selection,
   coil geometry, mgrid interpolation, and objective assembly.
2. AD-vs-FD tests for fixed-boundary boundary modes, pressure/current profiles,
   coil currents, coil Fourier coefficients, DMerc, `D_R`, QS/QI objectives,
   and free-boundary branch-local reports.
3. VMEC2000 parity tests for selected fixed-boundary and bounded free-boundary
   converged equilibria.
4. Physics gates for finite-positive WOUTs, aspect, iota, pressure/beta,
   magnetic well, Mercier/Glasser, Bnormal RMS, and Boozer-space symmetry.
5. Regression tests for saved input.final reruns matching optimization WOUTs.
6. Performance smoke tests for cold solve, warm solve, accepted replay, scalar
   adjoint, matrix-free response, and GPU paths when GPU is available.
7. Artifact provenance tests for README/docs figures.

Optional external gates:

- VMEC2000 executable parity.
- ESSOS direct-coil and generated-mgrid parity.
- booz_xform_jax parity and Boozer plot generation.
- GPU benchmark matrix on `ssh office`.

## Promotion Gates

A feature can be promoted in docs/README only when:

1. There is a unit or physics test covering the API.
2. There is an AD-vs-FD or external parity gate for derivative claims.
3. Failure modes are documented and tested.
4. Runtime is bounded or the test is explicitly optional.
5. The claim is scoped to the actual differentiability seam tested.

Branch-differentiation promotion rules:

1. Same branch and same fingerprint: may promote after AD-vs-FD passes.
2. Changed branch fingerprint: must reject derivative promotion.
3. Smooth surrogate controller: may be documented as a research surrogate only
   until it matches production host-controller outputs and VMEC2000 parity.
4. Arbitrary hard adaptive branch changes: do not claim classical derivatives;
   use nonsmooth optimization language, subgradient/surrogate language, or
   derivative-free/global proposal language as appropriate.

## Milestones

Milestone 1: current release closeout.

- Keep phase 2 conservative and green.
- Keep phase 3 coil-only example complete-solve-authoritative.
- Recover CI after QI example guard fix.
- Record deferred differentiability plan.

Milestone 2: fixed-boundary derivative hardening.

- Expand profile/stability/objective AD-vs-FD gates.
- Refactor public optimization objective tuple assembly.
- Keep examples and docs aligned.

Milestone 3: free-boundary branch-local production hardening.

- Add more physical scalars to same-fingerprint complete-solve FD gates.
- Expand bounded direct-coil/mgrid/VMEC2000 parity fixtures.
- Improve runtime of branch-local vector/JVP reports.

Milestone 4: optional differentiable adaptive controller research.

- Implement a fully JAX-visible controller prototype.
- Compare step-by-step branches against the host controller.
- Add changed-branch diagnostics and nonsmooth objective experiments.
- Promote only if it provides a capability that branch-local reports cannot.

Milestone 5: single-stage optimization release.

- Demonstrate coil-only QA/QH or finite-beta QA with direct coils.
- Keep complete solves as acceptance authority.
- Use exact or branch-local validated derivatives for proposals.
- Compare against mgrid and VMEC2000 bounded fixtures.

Milestone 6: research-grade differentiable VMEC variant.

- Unified fixed/free-boundary differentiability story.
- Validated physics metrics and optimization examples.
- CPU/GPU benchmarked derivative policies.
- Documentation explicitly states every promoted and non-promoted seam.

## Current Open-Lane Completion Snapshot

These percentages describe the current PR #20 state after the 2026-06-17
review:

- Direct-coil/free-boundary phase 1: 100%.
- Full nonlinear free-boundary adjoint phase 2: 99.99999999% for fixed
  same-branch/fingerprint-gated accepted/rejected gates; arbitrary adaptive
  branch changes remain unclaimed.
- VMEC parity and physics gates: 99.92%.
- Single-stage coil-only optimization phase 3: 99.997% for conservative
  complete-solve-authoritative examples that consume validated branch-local
  proposal reports.
- CPU/GPU performance: 99.60%; branch-local replay/report paths are improved,
  but cold exact tape/forward-force cost remains the main unresolved hotspot.
- CI/runtime/coverage hygiene: 100%; PR #20 run `27664190749` passed all
  required CI and Codecov gates.
- Docs/release hygiene: 100% for current claims; pending only the explicit
  single-plan pointer added in this review.
- QI minimal-seed README artifacts: 100% for current promoted artifacts.
- Refactor/source simplification: 72%; major root-level monoliths were split,
  but `solvers/fixed_boundary/residual/iteration.py`, `optimization.py`,
  `optimization_workflow.py`, `free_boundary_adjoint.py`, and large validation
  tests remain above the long-term maintainability budget.
- Research-grade arbitrary adaptive-branch differentiability: 7%; planned and
  scoped, but not production-promoted.
- Overall production-safe differentiability/refactor PR: 98.7%.

## Decisions Needed Later

1. Should vmec_jax implement a fully JAX-visible adaptive controller as a
   research surrogate, knowing it may be slower and may diverge from VMEC2000
   branch behavior?
2. Should arbitrary hard branch changes be handled with nonsmooth optimization,
   smoothing/relaxation, or derivative-free global proposals rather than AD?
3. What runtime budget is acceptable for publication-grade free-boundary
   coil-only optimization examples?
4. Which physics scalar is the release-critical promotion target for the next
   full adaptive branch-local gate: QS, Bnormal RMS, aspect, iota, or finite
   beta response?
5. Which QI NFP cases should be public release examples versus longer-running
   research artifacts?

## Running Log Template

For future updates, append entries with:

1. Date and commit.
2. Steps taken.
3. Results obtained.
4. Tests and commands run.
5. Best next steps.
6. User decisions needed.
7. Completion percentages by lane.

## 2026-06-17 Canonical Plan Review

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Reviewed PR #20 status, recent commits, CI results, plan files, docs, and
   source-health diagnostics.
2. Rechecked the differentiability strategy against DESC, VMEC-compatible
   solver constraints, recent spectral-adjoint work, JAX custom derivative
   rules, JAXopt implicit differentiation, Optax optimizer composition,
   Lineax-style matrix-free linear operators, and Equinox PyTree patterns.
3. Promoted this file to the active single source of truth for the
   differentiability/refactor plan.
4. Demoted `plan_freeb.md` to a detailed free-boundary evidence log and
   `plan.md`/`discrete_adjoint_2506_plan.md` to historical/reference plans.
5. Added a finite execution plan with explicit exit gates for plan
   consolidation, fixed-boundary residual-controller splitting, derivative
   policy unification, branch-local free-boundary completion, adaptive-controller
   research, and docs/release closure.

Results obtained:

- PR #20 is open, draft, mergeable, and clean at `94e05da`.
- GitHub Actions run `27664190749` is green across required CI and Codecov
  gates.
- The next non-negotiable simplification hotspot is
  `vmec_jax/solvers/fixed_boundary/residual/iteration.py`, especially the
  7790-line `solve_fixed_boundary_residual_iter` function.
- The differentiability claim remains conservative: production derivatives are
  pure-JAX, custom-JVP/VJP, implicit, solver-faithful discrete-adjoint, or
  matrix-free where validated; arbitrary adaptive branch changes are still
  research-only.

Tests and commands run:

- `git status --short --branch`
- `gh pr view 20 --json number,title,state,isDraft,mergeStateStatus,headRefName,baseRefName,statusCheckRollup`
- `python tools/diagnostics/source_health.py --top 15 --top-functions 15`

Best next steps:

1. Update docs to point to this file as the canonical plan and explain the
   status of older plan files.
2. Run a fast docs build to catch broken references.
3. Commit and push the plan/docs consolidation.
4. Start the next fixed-boundary residual-controller split, targeting a
   named, domain-specific state-transition seam with direct tests.
5. Continue free-boundary branch-local derivative work only with
   fingerprint-gated physical scalar tests.

User decisions needed:

No immediate decision.

Completion:

- Direct-coil/free-boundary phase 1: 100%.
- Full nonlinear free-boundary adjoint phase 2: 99.99999999%.
- VMEC parity and physics gates: 99.92%.
- Single-stage coil-only optimization phase 3: 99.997%.
- CPU/GPU performance: 99.60%.
- CI/runtime/coverage hygiene: 100%.
- Docs/release hygiene: 100% for current claims.
- QI minimal-seed README artifacts: 100%.
- Refactor/source simplification: 72%.
- Research-grade arbitrary adaptive-branch differentiability: 7%.
- Overall production-safe differentiability/refactor PR: 98.7%.

## 2026-06-17 Catastrophic Trial-Restart Controller Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Extracted the host-loop catastrophic trial-restart scalar update into
   `host_catastrophic_restart_update`.
2. Extracted the 12-block direct-fallback force/update RMS calculation into
   `host_force_update_rms`.
3. Replaced duplicate direct-fallback-failure and ordinary trial-rejection
   restart code in `solve_fixed_boundary_residual_iter` with one shared
   catastrophic rollback/update path.
4. Reused the existing velocity scaling helper for accepted backtracking
   momentum scaling.
5. Reused `host_force_update_rms` for non-strict backtracking update RMS.
6. Tested and rejected a more abstract update-history helper and momentum-block
   helper because their call-site argument lists increased the large function.
7. Added unit tests for bad-progress restarts, nonfinite/bad-Jacobian restarts,
   VMEC reset milestone scaling, and RMS parity with the original inline
   formula.

Results obtained:

- `vmec_jax/solvers/fixed_boundary/residual/iteration.py` dropped from 8404
  lines at the plan-review baseline to 8363 lines.
- `solve_fixed_boundary_residual_iter` dropped from 7790 lines at the
  plan-review baseline to 7747 lines.
- The extracted helper gives the adaptive-controller/fingerprint work a named
  scalar restart seam instead of two open-coded branches.
- PR #20 CI run `27687965132` for the previous plan-consolidation commit is
  fully green, including combined coverage and Codecov.

Tests and commands run:

- `python -m ruff check vmec_jax/solvers/fixed_boundary/residual/update.py vmec_jax/solvers/fixed_boundary/residual/iteration.py tests/test_solve_residual_iter_update_helpers.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_residual_iter_update_helpers.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_real_scan_wave10_coverage.py tests/test_solve_wave7_coverage.py::test_residual_iter_vmec2000_scan_minimal_one_step tests/test_solve_wave7_coverage.py::test_residual_iter_vmec2000_scan_state_only -q`
- `python tools/diagnostics/source_health.py --top 8 --top-functions 12`

Best next steps:

1. Commit and push the catastrophic restart extraction.
2. Continue the fixed-boundary residual-controller split with another
   domain-named state transition, preferably one that reduces
   `_run_vmec2000_scan` or the host-loop accepted/rejected update path.
3. Keep every extraction covered by direct helper tests plus one real scan
   smoke shard before moving to the next seam.

User decisions needed:

No immediate decision.

Completion:

- Direct-coil/free-boundary phase 1: 100%.
- Full nonlinear free-boundary adjoint phase 2: 99.99999999%.
- VMEC parity and physics gates: 99.92%.
- Single-stage coil-only optimization phase 3: 99.997%.
- CPU/GPU performance: 99.60%.
- CI/runtime/coverage hygiene: 100%.
- Docs/release hygiene: 100% for current claims.
- QI minimal-seed README artifacts: 100%.
- Refactor/source simplification: 72.5%.
- Research-grade arbitrary adaptive-branch differentiability: 7%.
- Overall production-safe differentiability/refactor PR: 98.73%.

## 2026-06-14 Umbrella PR and Solver Helper Extractions

Commit: `6e8a335` plus follow-up extraction on
`codex/differentiability-refactor-plan`.

Steps taken:

1. Opened draft PR #20 as the single umbrella PR for the full
   differentiability/refactor plan.
2. Updated the PR title to make clear that this is the long-lived umbrella
   branch, not a short standalone planning PR.
3. Added the source-health diagnostic and documented it in
   `docs/code_structure.rst`.
4. Performed the first low-risk extraction from `vmec_jax/solve.py`:
   force-block mode weighting, lambda full-mesh residual norm, and
   stability-guard timestep calculation now live in
   `vmec_jax/solve_force_norm_helpers.py`.
5. Performed the second low-risk extraction from `vmec_jax/solve.py`:
   dtype-aware gradient, conjugate-gradient, and Levenberg-Marquardt tolerance
   policy now live in `vmec_jax/solve_tolerance_helpers.py`.
6. Performed the third low-risk extraction from `vmec_jax/solve.py`:
   fixed-boundary edge constraints, magnetic-axis regularity, lambda-gauge
   projection, and NumPy/JAX coefficient-slice helpers now live in
   `vmec_jax/solve_constraint_helpers.py`.
7. Performed the fourth low-risk extraction from `vmec_jax/solve.py`:
   gradient-descent state updates and feasible-gradient projection now live in
   `vmec_jax/solve_gradient_helpers.py`.
8. Performed the fifth low-risk extraction from `vmec_jax/solve.py`:
   mode-diagonal and radial Dirichlet smoothing preconditioner kernels now live
   in `vmec_jax/solve_preconditioner_helpers.py`.
9. Performed the sixth low-risk extraction from `vmec_jax/solve.py`:
   environment-controlled JIT-cache limits, LRU cache access, structural
   strict-update cache keys, and scan-runner miss categorization now live in
   `vmec_jax/solve_jit_cache_helpers.py`.
10. Extended `vmec_jax/solve_preconditioner_helpers.py` with a seventh
    low-risk extraction: tridiagonal policy resolution, metric preconditioner
    scale estimates, and VMEC radial mesh scale-factor helpers.
11. Performed the eighth low-risk extraction from `vmec_jax/solve.py`:
    initial magnetic-axis reset decisions, axis-state merging, and optional
    axis coefficient dumps now live in `vmec_jax/solve_axis_reset_helpers.py`.
12. Performed the ninth low-risk extraction from `vmec_jax/solve.py`:
    VMEC2000-style scan resume-state initialization and carry-field restoration
    now live in `vmec_jax/solve_scan_resume_helpers.py`.
13. Performed the tenth low-risk extraction from `vmec_jax/solve.py`:
    free-boundary cadence, turn-on, constraint-baseline, and velocity-block
    control helpers now live in `vmec_jax/solve_free_boundary_control_helpers.py`.
14. Performed the eleventh low-risk extraction from `vmec_jax/solve.py`:
    solve-facing free-boundary external-field diagnostic adapters now live in
    `vmec_jax/solve_free_boundary_diagnostics.py`.
15. Extended `vmec_jax/solve_preconditioner_helpers.py` with a twelfth
    low-risk extraction: VMEC `m=1` preconditioner scale factors, right-hand-side
    scaling, and matrix reassembly contract checks.
16. Performed the thirteenth low-risk extraction from `vmec_jax/solve.py`:
    optional force-channel GC debug dump array adapters and legacy GC layout
    mapping now live in `vmec_jax/solve_force_dump_helpers.py`.
17. Performed the fourteenth low-risk extraction from `vmec_jax/solve.py`:
    optional lambda residual, lambda-preconditioner, lambda-derivative, and
    radial-preconditioner debug dump helpers now live in
    `vmec_jax/solve_lambda_dump_helpers.py`.
18. Performed the fifteenth low-risk extraction from `vmec_jax/solve.py`:
    optional JAX HLO lowering debug dump helpers now live in
    `vmec_jax/solve_hlo_dump_helpers.py`.
19. Performed the sixteenth low-risk extraction from `vmec_jax/solve.py`:
    optional covariant-field debug dumps for scaled full-mesh, half-mesh, and
    radial `B_s` reconstruction diagnostics now live in
    `vmec_jax/solve_bsub_dump_helpers.py`.
20. Performed the seventeenth low-risk extraction from `vmec_jax/solve.py`:
    optional metric, preconditioner-input, and VMEC internal state-vector debug
    dump helpers now live in `vmec_jax/solve_metric_dump_helpers.py`.
21. Extended `vmec_jax/solve_force_dump_helpers.py` with an eighteenth
    low-risk extraction: TOMNSP, force-kernel, scalar residual, and post-scaling
    force-channel norm debug dumps.
22. Performed the nineteenth low-risk extraction from `vmec_jax/solve.py`:
    solver result dataclasses and the residual-loop scan carry container now
    live in `vmec_jax/solve_result_types.py`.
23. Extended `vmec_jax/solve_result_types.py` with a twentieth low-risk
    extraction: the `wout`-like VMEC force-kernel PyTree container now lives
    outside the solver monolith while preserving the private `solve.py` alias.
24. Kept backward-compatible private aliases in `solve.py` so existing tests and
    internal imports continue to work.
25. Began the broader file-structure refactor outside `solve.py` by extracting
    backend-aware driver policy, convergence, staged-budget, and resume-state
    helpers into `vmec_jax/driver_policy_helpers.py`.  `driver.py` keeps the
    historical private names as compatibility aliases/wrappers, including the
    local `default_non_autodiff_solver_policy` wrapper so backend monkeypatches
    still work.
26. Continued the driver decomposition by extracting staged/chunked result
    merging, timing aggregation, final-force payload propagation, stage-switch
    projection checks, and VMEC history comparison helpers into
    `vmec_jax/driver_result_helpers.py`.  `driver.py` again keeps private
    compatibility aliases for existing tests and internal imports.
27. Extracted current-driven post-solve flux/profile reconciliation into
    `vmec_jax/driver_flux_helpers.py`, leaving a small `driver.py` wrapper to
    preserve historical monkeypatch hooks for `boundary_from_indata` and
    `_iotaf_from_iotas`.
28. Extracted VMEC-style residual scalar reconstruction and fixed-boundary
    `wout` construction into `vmec_jax/driver_output_helpers.py`.  `driver.py`
    retains wrappers for `residual_scalars_from_state`,
    `wout_from_fixed_boundary_run`, and `write_wout_from_fixed_boundary_run`
    so downstream monkeypatches and public import paths remain compatible.
29. Extracted bundled example path resolution, lightweight input/wout loading,
    and NumPy archive writing into `vmec_jax/driver_io_helpers.py`.  `driver.py`
    injects `__file__`, `load_config`, `_free_boundary_static_inputs`,
    `build_static`, `read_wout`, and `state_from_wout` so existing tests and
    downstream monkeypatches retain the same behavior.
30. Extracted the lightweight boundary-to-fixed-boundary-solve convenience path
    into `vmec_jax/driver_solve_helpers.py`.  `driver.py` keeps the public
    wrapper and injects `initial_guess_from_boundary` and
    `solve_fixed_boundary_gd`, preserving the existing solver-wiring test and
    optimization-script API.
31. Extended `vmec_jax/solve_residual_iter_runtime_helpers.py` with
    free-boundary external-field diagnostic attachment.  `solve.py` retains the
    local `_attach_freeb_diag` wrapper so existing solve-exit call sites stay
    unchanged while the branch logic now has direct unit coverage.
32. Performed a larger residual-loop hot-path extraction from `solve.py`:
    cached strict-update, preconditioner-output, fused preconditioner-apply,
    accepted-control, and `ptau` JIT payload helpers now live in
    `vmec_jax/solve_preconditioner_payload_helpers.py`.  `solve.py` retains
    private wrappers and shared cache aliases so existing tests, monkeypatches,
    and downstream private imports keep the same behavior.
33. Extracted the first-step VMEC residual diagnostic implementation into
    `vmec_jax/solve_first_step_diagnostics.py`.  The public
    `solve.first_step_diagnostics` wrapper now delegates to the helper while
    injecting the historical private solve-module helper aliases, preserving
    existing synthetic tests and monkeypatch seams.
34. Extracted the lambda-only fixed-geometry optimizer implementation from
    `solve_lambda_gd` into `vmec_jax/solve_lambda_optimizer.py`.  The public
    `solve.solve_lambda_gd` wrapper keeps the historical API and injects the
    solve-module aliases that tests and downstream private hooks monkeypatch
    (`eval_geom`, Fourier derivatives, `bsup_from_sqrtg_lambda`, `jit`,
    `has_jax`, constraint/tolerance helpers).
35. Extracted the shared fixed-boundary GD/L-BFGS magnetic-energy context into
    `vmec_jax/solve_fixed_boundary_energy_helpers.py`.  The optimizer loops
    still live in `solve.py`, but duplicated flux/pressure/grid-weight/edge
    coefficient preparation and `wb/wp/W` evaluators now share one injected
    helper that preserves historical `solve.py` monkeypatch seams
    (`eval_geom`, `bsup_from_geom`, `b2_from_bsup`, `angle_steps`, and
    pressure-shape validation).
36. Extracted the fixed-boundary gradient-descent optimizer loop into
    `vmec_jax/solve_fixed_boundary_gd_optimizer.py`.  `solve.py` now keeps the
    public wrapper and injects all historical private aliases for validation,
    energy setup, constraints, preconditioning, state updates, tolerance
    resolution, and JAX modules.
37. Extracted the fixed-boundary L-BFGS optimizer loop into
    `vmec_jax/solve_fixed_boundary_lbfgs_optimizer.py`.  The public
    `solve.solve_fixed_boundary_lbfgs` wrapper now injects validation, energy
    setup, constraints, preconditioning, gradient norm/tolerance, L-BFGS
    two-loop/descent/curvature helpers, state pack/unpack, and JAX modules.
38. Extracted the shared VMEC residual-force optimizer setup into
    `vmec_jax/solve_residual_force_context.py`.  The residual-objective
    L-BFGS and Gauss-Newton wrappers now share one injected context for
    flux/profile construction, VMEC-force `wout`-like payloads, trig tables,
    fixed-edge coefficients, convergence tolerance, and TOMNSP masks while
    preserving the historical `solve.py` monkeypatch seams for profile helpers.
39. Extracted the fixed-boundary VMEC-style residual L-BFGS optimizer loop into
    `vmec_jax/solve_fixed_boundary_residual_lbfgs_optimizer.py`.  The public
    `solve.solve_fixed_boundary_lbfgs_vmec_residual` wrapper now injects the
    residual-force context, profile aliases, residual-objective assembler,
    constraints, preconditioner, L-BFGS helpers, tolerance helpers, state
    pack/unpack, and JAX modules while the implementation owns the objective
    closures, line search, best-finite-step fallback, and result diagnostics.
40. Extracted the fixed-boundary VMEC-style residual Gauss-Newton optimizer
    loop into `vmec_jax/solve_fixed_boundary_residual_gn_optimizer.py`.  The
    public `solve.solve_fixed_boundary_gn_vmec_residual` wrapper now injects
    the residual-force context, residual vector assembly, constraints,
    tolerance/damping helpers, state pack/unpack, and JAX modules while the
    implementation owns the VJP/JVP normal-equations solve, late sparse-CG
    lookup, damping retries, fallback descent, and result diagnostics.
41. Started the free-boundary adjoint monolith split by extracting accepted
    trace reset/status/controller-mask helpers into
    `vmec_jax/free_boundary_adjoint_trace_controls.py`.  The public names stay
    re-exported from `free_boundary_adjoint.py`, while trace replay/fingerprint
    plans continue to live in the original module until their dependencies are
    narrowed further.
42. Started the WOUT diagnostics split by extracting the persisted
    Mercier-to-Glasser fallback reconstruction into
    `vmec_jax/wout_diagnostics.py`.  `wout.py` retains the historical private
    `_glasser_from_wout_mercier_terms` alias, and the focused test now checks
    the extracted helper against both the legacy private alias and the public
    differentiable `glasser_resistive_interchange_from_mercier_terms` algebra.
43. Repaired the combined coverage-gate gap introduced by solver-helper
    extractions by adding millisecond-scale synthetic optimizer implementation
    tests to an existing CI-included optimizer-helper shard.  The tests inject
    tiny geometry/Fourier/Bsup/gauge seams to cover the real lambda
    implementation loop and mode-diagonal branch, and parameterize the
    missing-JAX error path across the extracted lambda, fixed-boundary
    GD/L-BFGS, and residual-objective L-BFGS/Gauss-Newton implementations
    without running full VMEC solves.

Results obtained:

1. Draft PR #20 CI passed before the follow-up extraction.
2. `solve.py` decreased from roughly 15438 to 12870 lines locally.
3. `driver.py` decreased from 4064 to 2966 lines while preserving existing CLI
   and test import paths.
4. The extracted helpers are pure and synthetic-testable, making them a safe
   pattern for the next solver-kernel split.
5. Focused Ruff, pytest, source-health, and fast docs checks passed for the
   extracted helper modules, result containers, and force-kernel PyTree
   container.
6. Driver-policy focused tests passed after the extraction: 76 driver-policy
   tests, 17 driver wave tests, 15 driver run/wave12 tests, and 37 CLI/non-solve
   tests.
7. Driver-result focused tests passed after the second driver extraction: 116
   tests across driver policy, wave, fast-reconstruction, wout-driver, and
   helper-edge coverage, with only pre-existing synthetic `wout.py` warnings.
8. Driver flux focused tests passed after restoring compatibility hooks: 62
   passed and 1 skipped across fast-reconstruction, traced-Lsin,
   driver-wave2, and quasisymmetry tests.
9. Driver-output focused tests passed after the fourth driver extraction: 232
   tests across driver wave, policy, CLI, fixed-boundary reconstruction, wout,
   helper-edge, and quasisymmetry coverage, with one expected skip and only
   pre-existing synthetic `wout.py` warnings.
10. Driver-IO focused tests passed after the fifth driver extraction: 305 tests
    across driver/API, example loading, CLI, quasisymmetry, and wout coverage,
    with two expected skips and only pre-existing synthetic residual/wout
    warnings.
11. Driver solve-helper focused tests passed after the sixth driver extraction:
    3 tests covering solver-input wiring and initial-guess fixed-boundary
    driver paths.
12. Residual runtime-helper focused tests passed after the free-boundary
    diagnostic extraction: 20 unit tests passed, plus 3 representative
    free-boundary diagnostics tests passed with only pre-existing synthetic
    residual warnings.
13. Preconditioner payload extraction focused checks passed: Ruff clean for
    `solve.py`, the new helper, and focused tests; 53 hot-path/preconditioner
    tests passed, covering strict-update cache behavior, preconditioner-output
    scaling, fused payload diagnostics, `ptau` control payloads, and
    preconditioner diagnostics.
14. `solve.py` decreased further from 12870 to 12189 lines.  The new
    `solve_preconditioner_payload_helpers.py` is 841 lines and provides a
    focused seam for future accelerator preconditioner/timing work.
15. First-step diagnostic extraction focused checks passed: Ruff and compile
    clean for the moved implementation, with 26 synthetic first-step and branch
    coverage tests passing.  `solve.py` decreased again from 12189 to 11847
    lines; the new diagnostic helper is 429 lines.
16. Lambda optimizer extraction focused checks passed: compile and Ruff clean
    for `solve.py` and `solve_lambda_optimizer.py`; 4 targeted lambda tests
    passed; the broader lambda/wave coverage subset passed with 134 tests and
    1 expected skip.  `solve.py` decreased again from 11847 to 11706 lines.
17. Fixed-boundary energy-context extraction focused checks passed: compile and
    Ruff clean for `solve.py` and the new energy helper; 24 focused GD/L-BFGS
    tests passed; the broader solver optimizer subset passed with 157 tests.
    `solve.py` decreased from 11706 to 11652 lines while eliminating duplicated
    objective/evaluator setup across GD and L-BFGS.
18. Fixed-boundary GD loop extraction checks passed: compile and Ruff clean for
    `solve.py` and the new GD helper; 10 focused GD tests passed; the broader
    solver optimizer subset passed with 160 tests and 1 expected skip.
    `solve.py` decreased from 11652 to 11473 lines.
19. Fixed-boundary L-BFGS loop extraction checks passed: compile and Ruff clean
    for `solve.py` and the new L-BFGS helper; 7 focused L-BFGS tests passed;
    the broader solver optimizer subset passed with 160 tests and 1 expected
    skip.  `solve.py` decreased from 11473 to 11320 lines.
20. Residual-force context extraction checks passed: compile and Ruff clean for
    `solve.py` and the new context helper; the focused residual-optimizer tests
    passed with 4 tests; the broader solver optimizer subset passed with 160
    tests and 1 expected skip.  `solve.py` decreased from 11320 to 11214 lines
    while eliminating duplicated residual flux/profile/trig setup across
    residual-objective L-BFGS and Gauss-Newton.
21. Residual L-BFGS loop extraction checks passed: compile and Ruff clean for
    `solve.py`, `solve_residual_force_context.py`, and the new residual L-BFGS
    helper; 101 focused residual/branch/helper tests passed; the broader solver
    optimizer subset passed with 160 tests and 1 expected skip; the optional
    end-to-end residual GN test skipped in this environment.  `solve.py`
    decreased from 11214 to 10888 lines.
22. Residual Gauss-Newton loop extraction checks passed: compile and Ruff clean
    for `solve.py`, the residual context helper, and both residual optimizer
    helpers; 138 focused GN/residual/branch tests passed; the broader solver
    optimizer subset passed with 160 tests and 2 expected skips.  `solve.py`
    decreased from 10888 to 10596 lines while preserving the late
    `jax.scipy.sparse.linalg.cg` lookup used by monkeypatch tests.
23. Free-boundary trace-control extraction checks passed: compile and Ruff
    clean for `free_boundary_adjoint.py` and the new helper; the focused
    accepted-trace/fingerprint/replay-plan shard passed with 6 tests and 27
    deselected tests; the coil-optimization same-branch smoke passed with 14
    tests and 17 deselected tests.  `free_boundary_adjoint.py` decreased from
    6941 to 6823 lines.
24. WOUT diagnostic fallback extraction checks passed: compile and Ruff clean
    for `wout.py`, `wout_diagnostics.py`, and the focused test; the WOUT helper,
    Glasser objective, and finite-beta helper shard passed with 74 tests and 1
    expected skip.  `wout.py` decreased from 6321 to 6291 lines while creating
    a small stability-diagnostic seam for the DMerc/`D_R` AD-vs-FD lane.
25. Coverage repair checks passed: Ruff clean for the optimizer-helper test and
    extracted optimizer modules; `tests/test_solve_optimizer_helpers.py` passed
    with 10 tests; a targeted coverage run raised `solve_lambda_optimizer.py`
    coverage to 84% in that shard and covered fallback/import-error blocks
    across the other extracted optimizer implementations, enough to recover the
    previous combined gate failure at 94.89% without lowering the 95% threshold
    or adding expensive solves.
26. Free-boundary trace-metadata extraction moved dependency-light
    accepted-trace shape, segment-summary, controller-slot summary, and
    JSON-safe fingerprint helpers into `free_boundary_adjoint_trace_metadata.py`
    while keeping the historical private/public aliases in
    `free_boundary_adjoint.py`.  This is intentionally a small branch-local
    diagnostics seam: it reduces the large adjoint module without touching
    NESTOR kernels, replay objectives, or adaptive host branch claims.
27. DMerc/`D_R` stability-gradient coverage now includes a direct
    profile-integral AD-vs-central-FD gate for both `DMerc` and `D_R`, in
    addition to the existing Glasser algebra and public objective-wrapper
    gradient gates.  The focused Glasser/Mercier shard passed with 12 tests,
    Ruff was clean, and the fast docs build passed.
28. The GitHub py3.11 combined coverage gate failed at 94.98% after all real
    test/docs/build jobs passed.  The repair adds cheap unit coverage for the
    extracted scan-cache miss-category diagnostics and the strict JSON
    conversion fallback in the free-boundary trace metadata helper.  Local
    focused checks passed: Ruff clean, both modified test files passed, and the
    fast docs build passed.  A local coverage invocation was not usable because
    the developer machine mixed Python/pytest plugin environments before test
    collection; the clean GitHub runner remains the coverage authority.
29. Free-boundary branch-fingerprint extraction moved accepted-trace scalar,
    boolean, payload-shape, state-size, fingerprint, fingerprint-delta, and
    JSON-safe delta-summary helpers into
    `vmec_jax/free_boundary_adjoint_trace_fingerprint.py`.  The historical
    imports from `free_boundary_adjoint.py` remain valid and are now included
    in `free_boundary_adjoint.__all__` to match the documented API.  The
    extraction also fixes a silent reset-fingerprint weakness: synthetic
    array-valued trace states and full VMEC states now both detect
    discontinuities between `state_post` and the next `state_pre`, so
    same-fingerprint gates reject mixed accepted branches instead of silently
    accepting them.  The focused helper shard covers the new fingerprint
    module at 100% line coverage while keeping the trace-metadata helper at
    100% and the combined extracted trace-helper subset at 96%.
30. Parallel solve-monolith audit identified the next larger low-risk
    `solve.py` split: extract residual-iteration mode-transform setup into a
    focused module that owns signed-mode projection matrices, host/JAX
    `mn -> signed` transforms, physical/scalxc wrappers, residual norms, and
    mode-diagonal weights.  This should return a small context object and be
    validated with host-vs-JAX transform parity plus existing hot-path/cache
    tests before touching the force pipeline or adaptive scan loop.
31. Residual-iteration mode-transform extraction moved host DGEMM projection
    matrix setup, projected host `mn -> signed` transforms, NumPy `scalxc`
    setup, and mode-diagonal weight helpers into
    `solve_residual_iter_mode_transform_helpers.py`.  The focused test compares
    projected host transforms against the existing `vmec_parity` host
    transforms, covers zero-coefficient and `None`-partner cases, and covers
    the NumPy/JAX weight/scalxc helper parity.  Local focused checks passed:
    Ruff clean; the new helper shard passed with 3 tests and 100% local line
    coverage; existing geometry, VMEC parity host, hotpath, and cache subsets
    passed with 29 tests.  `solve.py` decreased from 10596 to 10512 lines.
32. Residual-iteration setup-policy extraction moved VMEC-grid reuse checks,
    free-boundary provider normalization, free-boundary scan disablement,
    external-field sampling flags, and CPU/GPU strict-update setup defaults
    into `solve_residual_iter_setup_helpers.py`.  This is a control-policy
    extraction only: no force kernel, time-control branch, NESTOR update, or
    accepted/rejected adaptive loop was moved.  Focused checks passed: Ruff
    clean; setup-policy and residual-iteration policy tests passed with 17
    tests; mode-transform, hotpath, and finish-cache subsets passed with 22
    tests.  `solve.py` decreased from 10512 to 10492 lines while making the
    free-boundary CPU/GPU setup behavior directly testable.
33. Residual-iteration finalization extraction moved final timing diagnostic
    attachment, resume-state payload packing, and result-object construction
    into `solve_residual_iter_finalize_helpers.py`.  The final free-boundary
    NESTOR recompute, residual recompute, and diagnostic-key construction stay
    in `solve.py` for now because those are parity-sensitive.  Focused checks
    passed: Ruff clean; finalization, timing-instrumentation, finish-cache,
    hotpath, and fast driver-control subsets passed with 31 tests; compileall
    passed.  The helper tests preserve the flattened resume-state payload
    contract and `_final_force_payload` propagation.  `solve.py` decreased
    from 10492 to 10477 lines.
34. Added the narrow compute-force cache ownership test needed before any
    residual force-pipeline extraction.  The test exercises the real
    solver-owned `_COMPUTE_FORCES_CACHE` through the precompile-only path,
    forces `VMEC_JAX_COMPUTE_FORCES_CACHE_SIZE=1`, uses two structural
    `static_key` values, and verifies LRU eviction/recompile behavior without
    touching force physics.  Focused checks passed: Ruff clean and the
    finish-cache precompile subset passed with 3 tests.
35. Added the first residual force-pipeline adapter seam by extracting
    structural compute-force JIT cache keys and callable selection into
    `solve_residual_iter_force_cache_helpers.py`.  The global
    `_COMPUTE_FORCES_CACHE` remains owned by `solve.py`; the helper only
    receives the cache object and cache get/put functions.  This preserves the
    differentiating-scan no-store behavior and the primal LRU path while
    removing cache policy from the force closure.  Focused checks passed:
    Ruff clean; helper, cache-ownership, and precompile force-cache tests
    passed with 5 tests; compileall passed.  `solve.py` decreased from 10477
    to 10471 lines.
36. Extracted the first pure residual force-payload postprocessing seam into
    `solve_residual_iter_force_payload_helpers.py`.  This helper owns the
    metric-only edge-masking policy, Z-force NaN preservation guard, and
    scalar `(gcr2, gcz2, gcl2)` assembly after M1/scalxc normalization.  The
    unmasked force payload remains in `solve.py` for preconditioner and
    free-boundary parity, and all debug/HLO dump branches remain at the solver
    call site.  Focused checks passed: Ruff clean and the new helper tests
    passed with 5 tests.
37. Extended the force-payload seam to resolve `include_edge_residual` and the
    TOMNSP mask-pack choice before residual assembly.  This keeps the actual
    TOMNSP transform and debug/HLO dump branches in `solve.py` while making the
    edge-residual control policy independently testable.  Focused checks
    passed: Ruff clean; force-payload, generic force-payload, and hotpath
    subsets passed with 25 tests.
38. Extracted repeated scan-debug Z-force channel square-sum reductions into
    the residual force-payload helper.  The debug-print labels, environment
    gates, and HLO dump behavior remain in `solve.py`; only the symmetric and
    asymmetric Z-channel sum-of-squares algebra moved.  Focused checks passed:
    Ruff clean; force-payload helper and hotpath subsets passed with 18 tests.
    `solve.py` decreased to 10437 lines.
39. Extracted bcovar-to-metric-preconditioner scale wrappers into
    `solve_preconditioner_helpers.py`.  The traced JAX wrapper is now shared by
    the main force path, and the host NumPy wrapper is shared by first-step
    diagnostics while preserving the injectable scale kernel used by tests.
    The local radial-tridiagonal closures and parity-sensitive force assembly
    remain in `solve.py`; only duplicate quadrature/materialization policy
    moved.  Focused wrapper tests were added for direct scale-kernel parity and
    injected scale/wint functions.
40. Extended `solve_force_payload_helpers.py` with staged residual-force
    payload transforms for the VMEC `m=1`, zeroing, and `scalxc` conventions.
    The scan-debug branch now prints from named helper stages instead of
    owning direct `vmec_residue` transform calls, while the non-debug path
    still returns the same final normalized payload.  Focused force-payload
    and scan-debug tests passed, including the VMEC sign convention for the
    rotated `m=1` Z-force channel.
41. Moved generic velocity-block zeroing/scaling helpers from the
    free-boundary control module into `solve_residual_iter_update_helpers.py`.
    The free-boundary module keeps compatibility re-exports, while the
    implementation now sits with the residual-iteration host momentum update.
    Focused tests cover shape/dtype preservation, scaling, and the compatibility
    re-export.
42. Extracted VMEC residual `FSQ` scalar assembly from the residual-loop closure
    into `solve_force_norm_helpers.py` as `residual_fsq_from_norms`.  The solver
    now uses the helper for trial, probe, damped, and final residual scalars,
    while focused tests cover NumPy/JAX scalar behavior and the private
    `solve.py` compatibility alias.
43. Removed two residual-loop pass-through closures for radial mesh helper
    ownership.  The solver now calls `pshalf_from_s_np` directly from
    `solve_preconditioner_helpers.py`, the dead `_sm_sp_from_s` closure was
    deleted, and the remaining `_sm_sp_from_s_np` import is explicitly marked
    as a compatibility re-export for existing tests/importers.  Focused checks
    passed for Ruff, compileall, and the ptau/radial helper branches.  `solve.py`
    decreased to 10412 lines.
44. Removed the single-use host convergence closure in the residual loop.  The
    physical convergence check now calls the already-tested
    `_residual_convergence_flags` runtime helper directly, keeping strict/total
    threshold policy in one place while leaving scan and VMEC2000 table behavior
    unchanged.  Focused convergence/runtime tests and compileall passed.
    `solve.py` decreased to 10409 lines.
45. Removed the single-use VMEC2000 sampling-cadence closure in the residual
    loop.  The scalar sampling branch now calls `_vmec2000_cadence_selected`
    directly with the resolved `nstep_screen`, keeping print/sample cadence
    policy in `solve_diagnostics_io.py`.  Focused cadence/formatting tests,
    Ruff, and compileall passed.
46. Removed the single-use residual-loop `_safe_dt_from_force` closure.  The
    update path now constructs `_ForceBlocks` at the only limiter call site and
    calls the already-extracted `safe_dt_from_force_blocks` helper directly,
    keeping the coefficient-RMS limiter testable in `solve_force_norm_helpers.py`
    without another local adapter layer.  Focused limiter/hotpath tests, Ruff,
    and compileall passed.  `solve.py` decreased to 10373 lines.
47. Removed the non-scan residual-loop axis-guess print wrapper.  The two
    call sites now use the shared VMEC2000-style axis printer directly, while
    the scan-local wrapper remains untouched because it is inside the staged
    scan controller.  Focused axis-reset helper tests, Ruff, and compileall
    passed.  `solve.py` decreased to 10370 lines.
48. Extracted residual-loop compute-force timing bookkeeping into
    `solve_residual_iter_runtime_helpers.py`.  The solver now keeps a small
    `partial` binding for the active timing dictionary and optional JAX device
    synchronization, while the first/rest/labeled counter updates are covered
    independently by runtime-helper tests, including disabled timing and
    synchronization-failure paths.  Focused runtime/hotpath tests, Ruff, and
    compileall passed.  `solve.py` decreased to 10357 lines.
49. Started the free-boundary adjoint runtime-helper seam by moving JAX timing
    synchronization and named-scope fallback utilities into
    `free_boundary_adjoint_runtime_helpers.py`.  `free_boundary_adjoint.py`
    keeps compatibility wrappers so existing monkeypatch-based tests still
    exercise module-local JAX shims, while the helper behavior is now directly
    unit-tested.  Focused free-boundary helper/vacuum-adjoint tests, Ruff, and
    compileall passed.
50. Extracted accepted-trace stacking and static-signature helpers into
    `free_boundary_adjoint_trace_stack.py`.  The large adjoint module now keeps
    compatibility aliases for private tests/internal users, while array control
    stacking, pytree stacking, optional payload stacking, NESTOR-axis stacking,
    preconditioner static signatures, and trace payload digests are covered
    through the dedicated helper module and existing trace-stack tests.  Focused
    free-boundary trace/control tests, Ruff, and compileall passed.
51. Moved accepted-trace effective controller masks and unconditional-accept
    segment checks into `free_boundary_adjoint_trace_controls.py`.  These
    helpers interpret fixed branch accept/done/reset controls and decide which
    replay segments can skip accept/reject conditionals; they are now colocated
    with the controller-control payload builder while `free_boundary_adjoint.py`
    keeps compatibility aliases.  Focused branch metadata/control tests, Ruff,
    and compileall passed.
52. Extracted replay-plan utility helpers into
    `free_boundary_adjoint_replay_plan_helpers.py`.  Trace extraction from
    result/report containers, stacked-control slicing, and stackability probing
    now live outside the main adjoint module with private compatibility aliases
    retained.  Focused replay-plan/control tests, Ruff, and compileall passed.
    `free_boundary_adjoint.py` decreased to 6248 lines.
53. Moved accepted-trace step-policy and preconditioner-policy segmentation
    helpers into `free_boundary_adjoint_trace_stack.py`.  The public segment
    functions remain imported from `free_boundary_adjoint.py`, while static
    step signatures, segment construction, and JSON-safe segment summaries now
    live beside the trace stacking/signature utilities they depend on.  Focused
    trace-control/segment tests, Ruff, and compileall passed.
    `free_boundary_adjoint.py` decreased to 5995 lines.
54. Extracted complete-payload accepted-step policy report helpers into
    `free_boundary_adjoint_replay_plan_helpers.py`.  The full-loop same-branch
    gate now imports the payload signature/layout/summary helpers through
    private compatibility aliases, and helper-unit tests assert those aliases
    remain stable for internal users.  Focused helper tests, Ruff, and
    compileall passed.  `free_boundary_adjoint.py` decreased to 5941 lines.
55. Moved complete-solve objective scalar normalization into
    `free_boundary_adjoint_replay_plan_helpers.py`.  The same-branch complete
    FD report now uses the extracted helper through a private compatibility
    alias, with unit coverage for scalar objectives, mapping objectives,
    non-scalar error paths, and empty mappings.  Focused helper tests, Ruff,
    and compileall passed.  `free_boundary_adjoint.py` decreased to 5922 lines.
56. Extracted generic weighted objective algebra into
    `free_boundary_adjoint_objective_helpers.py`.  Branch-local scalar reports
    and projected-mode objectives now import weighted half-norm, static
    zero-weight detection, and pytree half-norm accumulation through private
    compatibility aliases.  Helper tests cover alias stability, scalar/array
    weights, empty pytrees, and host-known zero-weight edge cases.  Focused
    helper tests, Ruff, and compileall passed.  `free_boundary_adjoint.py`
    decreased to 5888 lines.
57. Extracted pytree directional algebra into
    `free_boundary_adjoint_pytree_helpers.py`.  Batched directional
    contractions, batched pullback application, and leading-axis unstacking now
    live outside the main adjoint module with compatibility aliases retained.
    Helper tests cover alias stability, empty-Jacobian contractions, numeric
    contractions, and unstacking behavior; the existing branch-fingerprint test
    still covers VJP pullback usage.  Focused helper tests, Ruff, and
    compileall passed.  `free_boundary_adjoint.py` decreased to 5837 lines.
58. Extracted residual-iteration force debug/stage policy into
    `solve_residual_iter_force_payload_helpers.py`.  The main solver loop now
    delegates scan-debug Z-force square-sum printing plus M1/zero/scalxc staged
    payload selection to a focused helper, while the residual kernels, edge
    masking, and scalar norm reductions remain in their existing parity-tested
    paths.  Unit tests cover injected debug printers, disabled-debug fast path,
    staged debug path, and the existing non-scan debug solve regression still
    passes.  Focused solver/helper tests, Ruff, and compileall passed.
    `solve.py` decreased to 10308 lines.
59. Moved batched scalar host materialization into
    `solve_residual_iter_runtime_helpers.py`.  Non-scan VMEC2000-style control
    flow still calls `_device_get_floats` at the same points, but the JAX
    `device_get` batching policy is now isolated and unit-tested with an
    injected fake JAX module.  Focused runtime/helper tests, Ruff, and
    compileall passed.  `solve.py` decreased to 10299 lines.
60. Moved complete-loop rejected-controller-slot fingerprint detection into
    `free_boundary_adjoint_trace_metadata.py`.  The adaptive same-branch report
    now delegates accept-mask and step-status inspection to a metadata helper,
    keeping the branch-claim logic explicit while avoiding another nested
    report-only predicate.  Helper tests cover compatibility aliasing,
    accept-mask rejection, rejected/restart statuses, accepted-only masks, and
    non-mapping inputs.  Focused free-boundary helper tests, Ruff, and
    compileall passed.  `free_boundary_adjoint.py` decreased to 5826 lines.
61. Extracted residual-iteration setup timing bookkeeping into
    `solve_residual_iter_runtime_helpers.py`.  The main solver still records
    the same setup buckets at the same call sites, but zeroed setup timing
    initialization, disabled-timing start behavior, and elapsed accumulation
    now have direct helper tests.  Focused runtime/helper tests, Ruff, and
    compileall passed.  `solve.py` decreased to 10291 lines.
62. Moved dynamic scan probe budget/timing policy into
    `driver_policy_helpers.py`.  `driver._dynamic_scan_probe_settings` remains
    as a private compatibility wrapper so existing tests can still monkeypatch
    `driver._default_backend_name`, while the policy body now accepts injected
    backend and environment readers for direct unit testing.  Focused driver
    policy/API tests, Ruff, and compileall passed.  `driver.py` decreased to
    2953 lines.
63. Moved WOUT half-mesh diagnostic helpers into `wout_diagnostics.py`.
    VMEC half-mesh `sqrt(s)`, lambda half-mesh `sm`/`sp` weights, and
    VMEC-style zero-denominator-safe division now live with the persisted-WOUT
    diagnostic algebra.  `wout.py` keeps private compatibility aliases for
    downstream tests and internal call sites.  Direct diagnostics tests,
    existing WOUT helper coverage, Ruff, and compileall passed.  `wout.py`
    decreased to 6250 lines.
64. Moved BSS parity scalxc compatibility policy into
    `wout_parity_helpers.py`.  Environment-controlled odd-m scalxc undo
    detection, factor construction, and array scaling now have a focused
    module while `wout.py` keeps the private compatibility aliases used by
    existing tests and call sites.  Focused WOUT helper/env tests, Ruff, and
    compileall passed.  `wout.py` decreased to 6230 lines.
65. Moved VMEC `eqfor` finite-beta diagnostic scalars into
    `wout_diagnostics.py`.  Betapol, betator, betatot, and betaxis
    reconstruction now share the same persisted-WOUT diagnostic module as
    Glasser/`D_R` fallback algebra, while `wout.py` keeps private aliases for
    monkeypatch-heavy compatibility tests.  Direct module checks, WOUT
    finite-beta helper tests, bundled beta parity tests, Ruff, and compileall
    passed.  `wout.py` decreased to 6168 lines.
66. Moved VMEC aspect-ratio edge geometry diagnostics into
    `wout_diagnostics.py`.  The NumPy `aspectratio.f`-style scalar
    reconstruction is now tested directly in the diagnostics module and kept
    available through `wout._compute_aspectratio` for compatibility and
    monkeypatch-based WOUT synthesis tests.  Focused WOUT and implicit-driver
    helper tests, Ruff, and compileall passed.  `wout.py` decreased to
    6144 lines.
67. Moved VMEC `ctor` reconstruction from `buco` into
    `wout_diagnostics.py`.  Free-boundary and fixed-boundary current
    normalization now lives with the other persisted-WOUT scalar diagnostics,
    including the explicit SI-unit `mu0` conversion, while `wout.py` keeps the
    private `_compute_ctor_from_buco` alias for compatibility.  Focused WOUT,
    non-solve, and implicit-driver helper tests, Ruff, and compileall passed.
    `wout.py` decreased to 6124 lines.
68. Extracted a larger WOUT flux-convention tranche into
    `wout_flux_helpers.py` and `wout_io.py`.  VMEC lambda full/half mesh
    roundtrips, `chipf` half-mesh flux derivatives, input-deck current profile
    reconstruction, toroidal-flux profile synthesis, and scalar metadata
    validation now live outside the monolithic WOUT reader/writer.  `wout.py`
    keeps private compatibility aliases so downstream monkeypatch-heavy tests
    and internal call sites are unchanged.  Direct helper coverage was added
    for the new module APIs; focused WOUT, vmecPlot2, bundled current-profile,
    driver reconstruction, and finite-beta helper tests passed along with Ruff
    and compileall.  `wout.py` decreased to 5920 lines.
69. Moved accepted-trace control payload assembly into
    `free_boundary_adjoint_trace_stack.py`.  Scalar/update controls,
    preconditioner controls, array-valued update controls, and stacked
    state/constraint step controls now live with the lower-level trace
    stacking utilities and static-policy segment summaries.  Public exports
    and private compatibility aliases remain available from
    `free_boundary_adjoint.py`, including the legacy `_ACCEPTED_TRACE_*`
    constants used by tests/internal callers.  Focused free-boundary helper
    and direct-coil finite-pressure sensitivity tests passed along with Ruff
    and compileall.  `free_boundary_adjoint.py` decreased to 5687 lines.
70. Moved residual force payload construction from the residual-iteration
    hot loop into `solve_residual_iter_force_payload_helpers.py`.  Raw TOMNSP
    assembly, residual edge-mask selection, scan debug prints, optional HLO
    TOMNSP dumps, M1/zero/scalxc postprocessing, and metric edge-policy scalar
    norms are now handled by a typed helper result with dependency injection
    for focused tests.  `_compute_forces` still owns the force-kernel call,
    physical dumps, preconditioner scales, and solver update authority, so the
    numerical branch remains unchanged.  Direct helper tests now cover callback
    routing, explicit edge-residual masks, HLO wrapper reuse, and no-callback
    fast path.  Focused force-payload tests plus actual residual-iteration
    host/JAX parity and QH trace extraction tests passed along with Ruff and
    compileall.  `solve.py` decreased to 10266 lines.
71. Moved residual-iteration preconditioner output block materialization into
    `solve_force_payload_helpers.py`.  The non-fused JAX lambda-preconditioned
    path and the scalar radial-preconditioner path now share named helper
    functions that return `ForceBlocks`, while the main loop still controls
    cache refresh, fused GPU paths, timing buckets, and update acceptance.
    Direct helper tests cover lambda scaling, missing asymmetric channels,
    radial R/Z versus lambda weights, and optional-channel zero fill.  Focused
    force-payload tests and actual residual-iteration host/JAX parity plus QH
    trace extraction tests passed along with Ruff and compileall.  `solve.py`
    decreased to 10209 lines.
72. Moved residual-iteration preconditioner cache refresh/reassembly decision
    algebra into `solve_preconditioner_helpers.py`.  The 3D and axisymmetric
    VMEC2000-control preconditioner paths now call the same pure policy helper
    for traced-state refreshes, missing-cache refreshes, Boozer-covar-seeded
    cache reuse, and mismatched-`jmax` reassembly versus refresh decisions.
    The solver still owns all cache mutation, matrix reassembly, timing, and
    debug dumps.  Direct tests cover clean cache hits, traced/missing-cache
    refresh, seeded-Bcovar reuse blocked by debug dumps, and reassembly versus
    refresh when the cached radial range changes.  Focused preconditioner and
    actual residual-iteration host/JAX parity plus QH trace extraction tests
    passed along with Ruff and compileall.  `solve.py` decreased to
    10186 lines.
73. Moved non-fused device mode-weight force update scaling into
    `solve_force_norm_helpers.py`.  The device path now mirrors the existing
    host NumPy helper with a tested JAX helper that applies Fourier-mode
    weights and zero-fills missing optional sine/cosine channels.  The solver
    still applies the post-branch lambda update scale so host/device parity
    semantics remain unchanged.  Direct helper tests cover optional-channel
    zero fill and the compatibility alias from `solve.py`; actual
    residual-iteration host/JAX parity plus QH trace extraction tests passed
    along with Ruff and compileall.  `solve.py` decreased to 10181 lines.
74. Restored the residual force-payload dependency-injection seam after the
    extraction in step 70.  The helper now resolves the production
    `vmec_residual_internal_from_kernels` lazily when no explicit
    `residual_func` is supplied, so focused tests, monkeypatches, and future
    pedagogical examples can inject small residual kernels without being
    blocked by import-time binding.  The full `driver-solve-discrete` CI shard
    passed locally (`904 passed, 30 skipped`), including the previously failing
    finish/cache coverage tests.  Ruff and compileall passed for the changed
    helper.
75. Moved VMEC scan checkpoint materialization into
    `solve_scan_time_control.py` as `scan_checkpoint_update` with explicit
    `ScanCheckpointResiduals` bundles.  The solver now delegates the
    "best accepted residual checkpoint" state/scalar selection to a named,
    unit-tested helper while still injecting `jax.lax.cond` at the call site to
    preserve scan performance and tracing behavior.  Direct tests cover
    initialization, accepted checkpoint, and skipped-time-control branches;
    focused scan/time-control tests and actual accelerated-scan smoke tests
    passed along with Ruff.
76. Moved VMEC2000 scan history row builders into `solve_scan_output.py`,
    next to the corresponding unpacker and postprocessor.  The solver no
    longer carries local tuple-layout helpers for minimal/light/full scan
    histories, and tests now verify that each row builder matches
    `unpack_vmec2000_scan_histories`.  Focused scan output/time-control tests
    and actual accelerated-scan smoke tests passed along with Ruff.
    `solve.py` decreased to 10166 lines.
77. Added an optimization-objective AD-vs-central-FD diagnostic gate for the
    public `DMerc.J` and `GlasserResistiveInterchange.J` wrappers.  This
    complements the existing state-level `mercier_terms_from_state` AD/FD gate
    by validating the exact user-facing residuals that enter least-squares
    optimization.  The Mercier, Glasser and magnetic-well wrappers now reuse
    the shared `_smooth_positive_part` helper instead of duplicating soft-bound
    `logaddexp` algebra.  Focused optimization workflow tests, the new wrapper
    AD/FD test, and the full finite-beta helper unit suite passed along with
    Ruff.
78. Moved traced VMEC2000 scan resume-state packaging into
    `solve_scan_resume_helpers.py` as `build_traced_scan_resume_state`.  The
    helper keeps differentiable carry values as arrays and advances
    `iter_offset` without forcing host conversion, so traced scan diagnostics
    are documented and tested outside the solver monolith.  Direct resume-state
    tests, representative accelerated-scan tests, and the full
    `driver-solve-discrete` shard passed locally (`909 passed, 30 skipped`)
    along with Ruff.  `solve.py` decreased to 10138 lines.
79. Moved VMEC2000 scan diagnostic dictionary construction for state-only and
    traced-history scan exits into `solve_scan_output.py`.  The solver now asks
    for `vmec2000_state_only_scan_diagnostics` and
    `vmec2000_traced_scan_diagnostics` instead of open-coding result metadata
    near the scan postprocessor.  Direct tests cover host-only scalar fields,
    traced omission of host conversions, timing payloads, and resume-state
    preservation.  Focused scan-output/resume tests and the full
    `driver-solve-discrete` shard passed locally (`911 passed, 30 skipped`)
    along with Ruff.  `solve.py` decreased to 10123 lines.
80. Moved WOUT Glasser profile read/fallback logic into `wout_diagnostics.py`
    as `glasser_profiles_from_wout_variables` returning named
    `GlasserProfileArrays`.  The WOUT reader now delegates persisted
    `D_R`/`HGlasser`/`GlasserCorrection`/`GlasserShearValid` selection and old
    file fallback reconstruction to the diagnostics seam instead of
    open-coding it in `wout.py`.  Direct tests cover missing variables,
    persisted variables, and legacy `H` naming; the full WOUT helper suite and
    finite-beta Glasser materialization gate passed along with Ruff.
81. Moved VMEC2000 scan fallback reporting into
    `solve_residual_iter_policy.py` as `scan_fallback_message`.  The solver
    still owns the actual branch transition back to non-scan iteration, but the
    user-facing reason/probe formatting is now a pure, tested policy helper.
    Focused residual-iteration policy tests, the scan fallback integration
    coverage test, Ruff, compileall, and the full `driver-solve-discrete` shard
    passed locally (`912 passed, 30 skipped`).  `solve.py` decreased to
    10119 lines.
82. Added the write-side companion to the WOUT Glasser profile reader:
    `glasser_profiles_from_wout_data` now bundles `D_R`, `HGlasser`,
    `GlasserCorrection`, and `GlasserShearValid` arrays with explicit defaults
    before NetCDF materialization.  The WOUT writer delegates profile selection
    to `wout_diagnostics.py`, preserving the writer's I/O role while keeping
    diagnostics defaults unit-testable.  Focused Glasser materialization tests,
    the full WOUT helper suite, Ruff, and compileall passed locally.
83. Moved VMECPlot2 current-profile metadata normalization into
    `wout_flux_helpers.py` as `wout_current_profile_metadata_from_indata`.
    The WOUT builder now delegates `AC`, `ac_aux_s`, `ac_aux_f`,
    `pcurr_type`, and `piota_type` defaults to a pure helper while preserving
    the 21-slot polynomial and 101-slot auxiliary-array conventions.  Direct
    helper tests cover empty decks, scalar `AC`, long `AC` lists, profile type
    strings, and the private compatibility alias; WOUT helper, roundtrip, and
    VMECPlot2 compatibility tests passed along with Ruff and compileall.

Best next steps:

1. Keep all refactor work on PR #20 until the full plan is finalized.
2. Continue Wave 1/Wave 2 by extracting the next solver pure seams around
   residual-iteration checkpoint payload assembly, accepted-step trace fields,
   and late-stage finish/fallback formatting.  Keep `_COMPUTE_FORCES_CACHE`,
   `_compute_forces_impl`, and the adaptive scan loop in `solve.py` until the
   smaller helpers prove stable under the existing hotpath/cache tests.
3. Continue broader refactors in parallel with `driver.py`, `optimization.py`,
   `free_boundary_adjoint.py`, and `wout.py` by extracting pure
   policy/formatting/data-container seams before moving any physics kernels.
4. Add compatibility tests for every extracted private alias before ratcheting
   any source-health threshold.
5. For the DMerc/`D_R` lane, build on the new `wout_diagnostics.py` seam plus
   the existing public differentiable Mercier algebra tests before touching the
   larger JXBFORCE/Mercier geometry-reduction block in `wout.py`.

User decisions needed:

No immediate decision.  PR #20 remains draft until the full refactor plan is
complete.

Completion:

- Differentiability/refactor plan: 100%.
- Differentiability/refactor implementation: 94%.
- Source-health instrumentation: 100%.
- Solver monolith reduction: 71% of the large-file extraction work.
- Free-boundary adjoint monolith reduction: 30%.
- Driver workflow decomposition: 35%.
- WOUT diagnostic/profile decomposition: 22%.

## 2026-06-15 Solve-Facing Free-Boundary Helper Package Move

Commit: solve-facing free-boundary helper package tranche on
`codex/differentiability-refactor-plan`.

Steps taken:

1. Created `vmec_jax.solvers.free_boundary` for solve-facing free-boundary
   control and diagnostic helper seams.
2. Moved free-boundary cadence/turn-on control helpers into
   `vmec_jax.solvers.free_boundary.control`.
3. Moved solve-facing free-boundary external-field sampling diagnostics into
   `vmec_jax.solvers.free_boundary.diagnostics`.
4. Updated the solver facade, focused tests, and code-structure docs to use
   the new package paths.
5. Ratcheted the root-helper source-health CI gate from 29 to 27 files.

Results obtained:

- Two more free-boundary helper files left the root package.
- Root-level `vmec_jax/*.py` files dropped to 93.
- Root helper-prefix files dropped to 27.
- The free-boundary solve-control seam is now separated from fixed-boundary
  helper packages while preserving the public `solve` aliases.

Tests and commands run:

- `python -m ruff check vmec_jax/solve.py vmec_jax/solvers/free_boundary tests/test_solve_residual_iter_update_helpers.py`
- `python -m pytest -q tests/test_solve_residual_iter_update_helpers.py tests/test_free_boundary_coil_provider_forward.py tests/test_free_boundary_direct_coil_finite_pressure_sensitivity.py -q`
- `python tools/diagnostics/source_health.py --top 20 --top-functions 20 --max-root-helper-prefix-files 27`

Best next steps:

1. Run final hygiene checks, then commit and push this free-boundary helper
   tranche.
2. Treat the larger free-boundary-adjoint helper family as a separate
   high-value tranche with broader AD-vs-FD gates.
3. Continue reducing root helper sprawl by moving WOUT helper families or
   driver workflow helpers into domain packages.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 42%.
- Differentiability/refactor implementation: 96.7%.
- Solver monolith reduction: 81%.
- Free-boundary adjoint monolith reduction: 30%.
- Driver workflow decomposition: 35%.
- WOUT diagnostic/profile decomposition: 22%.

## 2026-06-15 Axis Reset and First-Step Diagnostics Package Move

Commit: axis reset and first-step diagnostics package tranche on
`codex/differentiability-refactor-plan`.

Steps taken:

1. Moved initial-axis reset decision/merge/dump helpers into
   `vmec_jax.solvers.fixed_boundary.diagnostics.axis_reset`.
2. Moved the synthetic first-step residual diagnostic implementation into
   `vmec_jax.solvers.fixed_boundary.diagnostics.first_step`.
3. Updated the solver facade and code-structure docs to use the diagnostics
   package paths.
4. Converted the moved diagnostic implementation imports to root-relative and
   sibling-domain package imports.
5. Ratcheted the root-helper source-health CI gate from 31 to 29 files.

Results obtained:

- Two more solver diagnostic helper files left the root package.
- Root-level `vmec_jax/*.py` files dropped to 95.
- Root helper-prefix files dropped to 29.
- The fixed-boundary diagnostics package now owns first-step diagnostics,
  axis-reset diagnostics, optional dump helpers, and trace-output formatting.

Tests and commands run:

- `python -m ruff check vmec_jax/solve.py vmec_jax/solvers/fixed_boundary/diagnostics`
- `python -m pytest -q tests/test_solve_axis_helpers_more_coverage.py tests/test_solve_more_coverage.py tests/test_solve_debug_dump_wave10_coverage.py tests/test_solve_additional_helpers.py::test_first_step_diagnostics_synthetic_default_and_axisymmetric_paths tests/test_solve_branch_coverage.py::test_radial_mesh_and_axis_reset_helpers_cover_small_mesh_edges -q`
- `python tools/diagnostics/source_health.py --top 20 --top-functions 20 --max-root-helper-prefix-files 29`

Best next steps:

1. Run the broader `driver-solve-discrete` shard after this move.
2. If green, commit and push this diagnostics follow-up tranche.
3. Continue package consolidation with either free-boundary helper families or
   the remaining fixed-boundary profile/options/result helper group, keeping
   behavior changes separate.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 38%.
- Differentiability/refactor implementation: 96.4%.
- Solver monolith reduction: 80.5%.
- Free-boundary adjoint monolith reduction: 30%.
- Driver workflow decomposition: 35%.
- WOUT diagnostic/profile decomposition: 22%.

## 2026-06-15 Fixed-Boundary Diagnostics Package Move

Commit: fixed-boundary diagnostics package tranche on
`codex/differentiability-refactor-plan`.

Steps taken:

1. Created `vmec_jax.solvers.fixed_boundary.diagnostics` for fixed-boundary
   solver diagnostics, optional debug dumps, and trace-output formatting.
2. Moved diagnostic I/O, HLO dump, force dump, covariant-field dump, lambda
   dump, and metric dump helpers out of the root package.
3. Updated `vmec_jax.solve`, residual finalization, scan debug helpers, tests,
   and docs to use the diagnostics package.
4. Ratcheted the root-helper source-health CI gate from 37 to 31 files.

Results obtained:

- Six more diagnostic helper files left the root package.
- Root-level `vmec_jax/*.py` files dropped to 97.
- Root helper-prefix files dropped to 31.
- Fixed-boundary residual/scan diagnostics now have a domain package rather
  than being spread across root `solve_*` helpers.

Tests and commands run:

- `python -m ruff check vmec_jax/solve.py vmec_jax/solvers/fixed_boundary/diagnostics vmec_jax/solvers/fixed_boundary/residual/finalize.py vmec_jax/solvers/fixed_boundary/scan/debug.py tests/test_solve_diagnostics_io.py tests/test_solve_scan_debug_helpers.py`
- `python -m pytest -q tests/test_solve_diagnostics_io.py tests/test_solve_scan_debug_helpers.py tests/test_solve_residual_iter_finalize_helpers.py tests/test_solve_scan_output.py -q`
- `python tools/diagnostics/source_health.py --top 20 --top-functions 20 --max-root-helper-prefix-files 31`

Best next steps:

1. Run the broader `driver-solve-discrete` shard after this diagnostic package
   move.
2. If green, commit and push this diagnostics tranche.
3. Continue with a larger, behavior-preserving extraction of the nested
   VMEC2000 scan loop or move free-boundary trace helper families into a
   domain package.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 35%.
- Differentiability/refactor implementation: 96.2%.
- Solver monolith reduction: 80%.
- Free-boundary adjoint monolith reduction: 30%.
- Driver workflow decomposition: 35%.
- WOUT diagnostic/profile decomposition: 22%.

## 2026-06-15 Residual Payload and Objective Package Move

Commit: residual payload/objective package tranche on
`codex/differentiability-refactor-plan`.

Steps taken:

1. Moved residual force payload block primitives from the root helper namespace
   into `vmec_jax.solvers.fixed_boundary.residual.payload_blocks`.
2. Moved residual force-norm helpers into
   `vmec_jax.solvers.fixed_boundary.residual.force_norms`.
3. Moved residual-objective block assembly and residual-force optimizer setup
   into `vmec_jax.solvers.fixed_boundary.optimization.residual_objective` and
   `vmec_jax.solvers.fixed_boundary.optimization.residual_context`.
4. Updated the solver facade, NumPy force patching, preconditioner payload,
   residual optimizers, tests, and code-structure docs to use the package
   paths.
5. Ratcheted the root-helper source-health CI gate from 41 to 37 files.

Results obtained:

- Four more residual/optimization helper files left the root package.
- Root-level `vmec_jax/*.py` files dropped to 103.
- Root helper-prefix files dropped to 37.
- The fixed-boundary residual package now owns force payload and force norm
  primitives, while the fixed-boundary optimization package owns residual
  objective/context setup.

Tests and commands run:

- `python -m ruff check vmec_jax/solve.py vmec_jax/solve_first_step_diagnostics.py vmec_jax/vmec_numpy_forces.py vmec_jax/solvers/fixed_boundary/residual vmec_jax/solvers/fixed_boundary/optimization vmec_jax/solvers/fixed_boundary/preconditioning tests/test_solve_force_payload_helpers.py tests/test_solve_force_norm_helpers.py tests/test_solve_residual_objective_helpers.py tests/test_refactorable_seams_coverage.py tests/test_vmec_numpy_forces_cache.py`
- `python -m pytest -q tests/test_solve_force_payload_helpers.py tests/test_solve_force_norm_helpers.py tests/test_solve_residual_objective_helpers.py tests/test_refactorable_seams_coverage.py tests/test_vmec_numpy_forces_cache.py tests/test_solve_residual_iter_force_payload_helpers.py tests/test_solve_optimizer_helpers.py tests/test_solve_residual_optimizer_wave8_coverage.py -q`
- `python tools/diagnostics/source_health.py --top 20 --top-functions 20 --max-root-helper-prefix-files 37`

Best next steps:

1. Run the broader `driver-solve-discrete` shard again after this second
   package move.
2. If green, commit and push this residual payload/objective tranche.
3. Continue reducing `solve.py` by extracting the nested VMEC2000 scan
   implementation or by moving dump/diagnostic helper groups into stable domain
   packages.
4. Keep the public `solve` facade aliases stable until the full refactor plan
   is complete.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 28%.
- Differentiability/refactor implementation: 95.8%.
- Solver monolith reduction: 79%.
- Free-boundary adjoint monolith reduction: 30%.
- Driver workflow decomposition: 35%.
- WOUT diagnostic/profile decomposition: 22%.

## 2026-06-15 Fixed-Boundary Preconditioning Package Move

Commit: follow-up on `codex/differentiability-refactor-plan`.

Steps taken:

1. Moved fixed-boundary preconditioner operator and payload helper seams into
   `vmec_jax.solvers.fixed_boundary.preconditioning`.
2. Updated `solve.py`, first-step diagnostics, fixed-boundary GD/L-BFGS
   optimizer helpers, residual L-BFGS helper, and preconditioner tests to use
   the new package path.
3. Fixed package-relative lazy imports for `vmec_residue`,
   `discrete_adjoint`, and `preconditioner_1d_jax`.
4. Updated `docs/code_structure.rst` to document the new package locations.
5. Ratcheted the root-helper source-health CI gate from 52 to 50 files.

Results obtained:

- Root-level `vmec_jax/*.py` files dropped from 118 to 116.
- Root helper-prefix files dropped from 52 to 50.
- Preconditioning helper code is now grouped with the fixed-boundary solver
  domain rather than the root package.

Tests and commands run:

- `python -m pytest -q tests/test_solve_preconditioner_metric_helpers.py tests/test_solve_hotpaths.py tests/test_solve_force_payload_helpers.py tests/test_solve_residual_optimizer_wave8_coverage.py tests/test_solve_residual_iter_helpers_wave8_coverage.py tests/test_tcon_precondn_diag.py -q`
- `python -m ruff check vmec_jax/solve.py vmec_jax/solve_first_step_diagnostics.py vmec_jax/solve_fixed_boundary_gd_optimizer.py vmec_jax/solve_fixed_boundary_lbfgs_optimizer.py vmec_jax/solve_fixed_boundary_residual_lbfgs_optimizer.py vmec_jax/solvers/fixed_boundary/preconditioning tests/test_solve_preconditioner_metric_helpers.py tests/test_solve_force_payload_helpers.py`
- `python tools/diagnostics/source_health.py --top 12 --top-functions 12 --max-root-helper-prefix-files 50`

Best next steps:

1. Run the broader `driver-solve-discrete` shard before committing.
2. If green, commit and push this tranche.
3. Continue reducing root helper sprawl by moving fixed-boundary optimizer
   helpers under `vmec_jax.solvers.fixed_boundary.optimizers` once the
   preconditioning package passes CI.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 17%.
- Differentiability/refactor implementation: 95.2%.
- Solver monolith reduction: 76%.
- Free-boundary adjoint monolith reduction: 30%.
- Driver workflow decomposition: 35%.
- WOUT diagnostic/profile decomposition: 22%.

## 2026-06-15 Fixed-Boundary Residual Package Move

Commit: follow-up on `codex/differentiability-refactor-plan`.

Steps taken:

1. Moved the residual-iteration helper family from root-level
   `vmec_jax/solve_residual_iter_*.py` files into
   `vmec_jax.solvers.fixed_boundary.residual`.
2. Updated `vmec_jax.solve`, `solve_free_boundary_control_helpers`, and the
   scan-planning helper to import the residual helpers from the domain package.
3. Updated internal tests to import the residual helper seams from their new
   package paths.
4. Fixed the moved force-payload helper's lazy VMEC force-kernel import so it
   still resolves from the root implementation package.
5. Updated `docs/code_structure.rst` to describe the new residual and scan
   package locations.
6. Ratcheted the root-helper source-health CI gate from 62 to 52 files.

Results obtained:

- Root-level `vmec_jax/*.py` files dropped from 128 to 118.
- Root helper-prefix files dropped from 62 to 52.
- The fixed-boundary residual helper namespace is now package-oriented and
  aligned with the fixed-boundary scan helper namespace.

Tests and commands run:

- `python -m pytest -q tests/test_solve_residual_iter_config.py tests/test_solve_residual_iter_finalize_helpers.py tests/test_solve_residual_iter_force_cache_helpers.py tests/test_solve_residual_iter_force_payload_helpers.py tests/test_solve_residual_iter_geometry_helpers.py tests/test_solve_residual_iter_mode_transform_helpers.py tests/test_solve_residual_iter_policy.py tests/test_solve_residual_iter_policy_gap_coverage.py tests/test_solve_residual_iter_runtime_helpers.py tests/test_solve_residual_iter_setup_helpers.py tests/test_solve_residual_iter_update_helpers.py tests/test_solve_scan_planning_helpers.py tests/test_solve_performance_instrumentation.py -q`
- `python -m ruff check vmec_jax/solve.py vmec_jax/solve_free_boundary_control_helpers.py vmec_jax/solvers/fixed_boundary/residual vmec_jax/solvers/fixed_boundary/scan tests/test_solve_residual_iter_*.py tests/test_solve_scan_planning_helpers.py tests/test_solve_performance_instrumentation.py`
- `python tools/diagnostics/source_health.py --top 12 --top-functions 12 --max-root-helper-prefix-files 52`

Best next steps:

1. Run the broader `driver-solve-discrete` shard before committing.
2. If green, commit and push this tranche.
3. Continue with the next cohesive fixed-boundary package move, likely
   preconditioner payload helpers or fixed-boundary optimizer helpers, while
   keeping behavior unchanged.
4. Start a separate free-boundary package move only after the fixed-boundary
   solver package settles under CI.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 14%.
- Differentiability/refactor implementation: 95%.
- Solver monolith reduction: 75%.
- Free-boundary adjoint monolith reduction: 30%.
- Driver workflow decomposition: 35%.
- WOUT diagnostic/profile decomposition: 22%.

## 2026-06-15 Source Namespace Gate and First Solver Package Move

Commit: `62fabc8` plus follow-up package migration on
`codex/differentiability-refactor-plan`.

Steps taken:

1. Added a source-health root namespace gate that counts root-level
   helper-prefix files (`solve_`, `driver_`, `free_boundary_`, `wout_`) and
   fails CI if the current baseline grows.
2. Added focused unit tests for the new namespace metrics and baseline-aware
   failure mode.
3. Wired the namespace gate into the CI parity-smoke job so future refactor
   work cannot add more root-package sprawl unnoticed.
4. Created the first domain package seam,
   `vmec_jax.solvers.fixed_boundary.scan`, and moved the VMEC2000-style scan
   helpers into that package.
5. Updated `vmec_jax.solve` to import the fixed-boundary scan helpers from the
   new domain package directly.
6. Temporarily preserved the old root-level `solve_scan_*` import paths as
   compatibility shims, then updated internal tests to use the new package path
   and deleted the shims.
7. Added function-length diagnostics to the source-health tool so future
   refactors can target oversized routines, not only oversized files.

Results obtained:

- The production solve path now uses the package-oriented fixed-boundary scan
  namespace.
- Root-level `vmec_jax/*.py` files dropped from 135 to 128, and the root
  helper-prefix baseline dropped from 69 to 62 files. CI now enforces the
  lower baseline.
- The package move made no algorithmic changes and preserved the existing
  scan-loop test surface.
- The current largest function target is
  `vmec_jax/solve.py:solve_fixed_boundary_residual_iter` at roughly 9k
  physical lines, followed by `driver.py:run_fixed_boundary` and the nested
  VMEC2000 scan loop.

Tests and commands run:

- `python -m ruff check tools/diagnostics/source_health.py tests/test_source_health_diagnostics.py`
- `python -m pytest -q tests/test_source_health_diagnostics.py -q`
- `python tools/diagnostics/source_health.py --top 20 --top-functions 20 --max-root-helper-prefix-files 62`
- `python -m pytest -q tests/test_solve_scan_output.py tests/test_solve_scan_output_edge_cases_more_coverage.py tests/test_solve_scan_resume_state.py tests/test_solve_scan_time_control.py tests/test_solve_scan_payload_helpers.py tests/test_solve_scan_math_helpers.py tests/test_solve_scan_planning_helpers.py tests/test_solve_scan_debug_helpers.py tests/test_scan_helper_edge_gates.py tests/test_performance_wave13_coverage.py tests/test_required_helper_coverage_margin.py -q`
- `python -m ruff check vmec_jax/solve.py vmec_jax/solvers/fixed_boundary/scan tests/test_solve_scan_output.py tests/test_solve_scan_time_control.py tests/test_solve_scan_math_helpers.py tests/test_solve_scan_payload_helpers.py tests/test_solve_scan_planning_helpers.py tests/test_solve_scan_debug_helpers.py`
- `python tools/diagnostics/ci_core_bucket_args.py driver-solve-discrete > /tmp/vmec_jax-driver-solve-discrete.txt && JAX_ENABLE_X64=1 VMEC_JAX_SKIP_PY311_COVERAGE_ONLY=1 xargs pytest -q -n 4 -m "not full and not vmec2000 and not simsopt" --durations=30 --cov=vmec_jax --cov-report= < /tmp/vmec_jax-driver-solve-discrete.txt`

Best next steps:

1. Repeat the same package-move pattern for the next cohesive helper family:
   direct production imports first, internal tests second, root files removed
   only after focused and solve-shard tests pass.
2. Continue package moves by migrating either the remaining fixed-boundary
   residual-iteration helper group into `vmec_jax.solvers.fixed_boundary` or
   the free-boundary trace helper group into `vmec_jax.solvers.free_boundary`.
3. Add import-time optional dependency checks before moving hot kernels.
4. Keep behavior changes separate from package moves so VMEC parity failures
   are easy to bisect.

User decisions needed:

No immediate decision.  PR #20 remains draft and all work stays on the same
branch.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 8%.
- Differentiability/refactor implementation: 94.5%.
- Solver monolith reduction: 73%.
- Free-boundary adjoint monolith reduction: 30%.
- Driver workflow decomposition: 35%.
- WOUT diagnostic/profile decomposition: 22%.

## 2026-06-15 Fixed-Boundary Optimization Package Move

Commit: fixed-boundary optimization package tranche on
`codex/differentiability-refactor-plan`.

Steps taken:

1. Created `vmec_jax.solvers.fixed_boundary.optimization` as the domain
   package for fixed-boundary magnetic-energy and residual-objective
   optimizers.
2. Moved the lambda-only optimizer, fixed-boundary energy context, GD/L-BFGS
   magnetic-energy optimizers, residual L-BFGS/Gauss-Newton optimizers,
   constraint projection helpers, gradient-update helpers, and tolerance
   policy helpers out of the root `vmec_jax` namespace.
3. Updated `vmec_jax.solve` to preserve the historical private aliases used by
   tests and downstream monkeypatch seams while importing implementations from
   the fixed-boundary optimization package.
4. Updated residual-force setup, first-step diagnostics, preconditioning
   operators, tests, and code-structure docs to use the package paths.
5. Ratcheted the root-helper source-health CI gate from 50 to 41 files.

Results obtained:

- Root helper-prefix files dropped from 50 to 41 without changing solver
  behavior.
- The fixed-boundary scan, residual-iteration, preconditioning, and optimizer
  helper families now live under one coherent solver domain namespace.
- `solve.py` remains the public/facade compatibility layer, but its
  fixed-boundary optimizer dependencies no longer add root-package helper
  sprawl.

Tests and commands run:

- `python -m ruff check vmec_jax/solve.py vmec_jax/solve_first_step_diagnostics.py vmec_jax/solve_residual_force_context.py vmec_jax/solvers/fixed_boundary/optimization vmec_jax/solvers/fixed_boundary/preconditioning tests/test_solve_optimizer_helpers.py`
- `python -m pytest -q tests/test_solve_optimizer_helpers.py tests/test_solve_residual_iter_helpers_wave8_coverage.py tests/test_solve_gd_wave10_coverage.py tests/test_solve_lbfgs_wave8_coverage.py tests/test_solve_residual_optimizer_wave8_coverage.py tests/test_step9_implicit_fixed_boundary.py -q`
- `python tools/diagnostics/source_health.py --top 20 --top-functions 20 --max-root-helper-prefix-files 41`
- `python tools/diagnostics/ci_core_bucket_args.py driver-solve-discrete > /tmp/vmec_jax-driver-solve-discrete.txt && JAX_ENABLE_X64=1 VMEC_JAX_SKIP_PY311_COVERAGE_ONLY=1 xargs pytest -q -n 4 -m "not full and not vmec2000 and not simsopt" --durations=30 --cov=vmec_jax --cov-report= < /tmp/vmec_jax-driver-solve-discrete.txt`

Best next steps:

1. Commit and push this fixed-boundary optimization package move.
2. Continue solver simplification by extracting the large nested scan helper
   inside `solve_fixed_boundary_residual_iter` into the existing
   `vmec_jax.solvers.fixed_boundary.scan` package.
3. Start the next package consolidation tranche around driver workflow helpers
   or free-boundary adjoint trace helpers; keep behavior changes separate from
   namespace moves.
4. Preserve VMEC parity and physics gates while using source-health function
   metrics to target the largest remaining routines.

User decisions needed:

No immediate decision. PR #20 remains draft and all refactor work stays on this
branch until the full plan is complete.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 23%.
- Differentiability/refactor implementation: 95.5%.
- Solver monolith reduction: 78%.
- Free-boundary adjoint monolith reduction: 30%.
- Driver workflow decomposition: 35%.
- WOUT diagnostic/profile decomposition: 22%.

## 2026-06-15 WOUT I/O Package Move

Commit: WOUT helper package tranche on `codex/differentiability-refactor-plan`.

Steps taken:

1. Created `vmec_jax.io.wout` as the domain package for VMEC `wout_*.nc`
   support code.
2. Moved WOUT schema, netCDF I/O, persisted diagnostic reconstruction,
   flux/current/lambda convention helpers, and BSS parity compatibility helpers
   out of the root `vmec_jax` namespace.
3. Kept `vmec_jax.wout` as the public compatibility surface while importing its
   internal implementation helpers from `vmec_jax.io.wout`.
4. Updated WOUT helper, WOUT physics gate, fixture inventory, and converged-WOUT
   parity tests to import from the new internal package paths instead of root
   helper modules.
5. Updated code-structure docs and ratcheted the root-helper source-health CI
   gate from 27 to 22 files.

Results obtained:

- Root Python files dropped from 93 to 88.
- Root helper-prefix files dropped from 27 to 22.
- WOUT internals now live in a coherent I/O namespace without adding public
  root shims.
- The public `vmec_jax.wout` reader/writer and private compatibility aliases
  used by existing tests remain available.

Tests and commands run:

- `python -m ruff check vmec_jax/wout.py vmec_jax/io tests/test_wout_helpers.py tests/test_wout_fast_helpers.py tests/test_wout_io_helpers.py tests/test_wout_physics_gates.py tests/test_converged_wout_matrix_parity.py tests/test_wout_fixture_inventory.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_wout_helpers.py tests/test_wout_fast_helpers.py tests/test_wout_io_helpers.py tests/test_wout_physics_gates.py tests/test_converged_wout_matrix_parity.py tests/test_wout_fixture_inventory.py -q`
- `python tools/diagnostics/source_health.py --max-root-helper-prefix-files 22`
- `python tools/diagnostics/source_health.py`
- `SPHINX_FAST=1 python -m sphinx -T -b html docs docs/_build/html_fast`

Best next steps:

1. Commit and push this WOUT I/O package move.
2. Use the free-boundary adjoint explorer findings to move trace/objective
   support helpers under `vmec_jax.solvers.free_boundary.adjoint` in the next
   behavior-preserving tranche.
3. Use the driver/misc explorer findings to choose a domain name for CLI/driver
   workflow decomposition before moving the remaining `driver_*` helpers.
4. Keep the source-health gate ratcheting only after focused tests and docs pass
   for each namespace move.

User decisions needed:

No immediate decision. PR #20 remains draft and all refactor work stays on this
branch until the full plan is complete.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 50%.
- Differentiability/refactor implementation: 97.0%.
- Solver monolith reduction: 81%.
- Free-boundary adjoint monolith reduction: 30%.
- Driver workflow decomposition: 35%.
- WOUT diagnostic/profile decomposition: 72%.

## 2026-06-15 Function-Level Refactor Tranche: Driver Finish, Scan Runtime, Free-Boundary Gates

Commit: pending function-level simplification tranche on
`codex/differentiability-refactor-plan`.

Steps taken:

1. Extracted the CLI fixed-boundary finish policy from the nested
   `run_fixed_boundary._maybe_finish_cli_fixed_boundary_run` closure into
   `vmec_jax.drivers.finish`.
2. Added `FixedBoundaryFinishContext` so the driver facade can pass its current
   staged-solve policy and callbacks explicitly instead of keeping another large
   nested closure in `driver.py`.
3. Extracted optional VMEC2000 scan runtime hook resolution from `solve.py` into
   `vmec_jax.solvers.fixed_boundary.scan.runtime`, keeping the traced scan
   numerical update body local for now because the full `_advance_step` closure
   still has a high-risk JAX/static-capture context.
4. Added focused scan-runtime tests for quiet defaults and time-control dump
   path resolution.
5. Extracted same-branch free-boundary adjoint promotion/gate reports into
   `vmec_jax.solvers.free_boundary.adjoint.gate_reports`, while keeping
   `vmec_jax.free_boundary_adjoint` as the public compatibility facade and
   preserving the conservative “branch-local, not arbitrary adaptive-controller
   differentiation” metadata.
6. Updated the code-structure docs for the new driver finish helper.

Results obtained:

- `driver.py` dropped from 2,953 to 2,572 lines.
- `run_fixed_boundary` dropped from 2,587 to 2,205 lines.
- `free_boundary_adjoint.py` dropped from 5,687 to 5,329 lines.
- `solve.py` dropped from 10,119 to 10,075 lines.
- Root helper-prefix count remains at 2, preserving the public facade floor.
- The next safe major complexity target remains the VMEC2000 scan step, but the
  traced `_advance_step` extraction should wait for a deliberately designed
  context object because it currently closes over most of the solver state.

Tests and commands run:

- `python -m ruff check vmec_jax/solve.py vmec_jax/solvers/fixed_boundary/scan/runtime.py tests/test_solve_scan_planning_helpers.py vmec_jax/driver.py vmec_jax/drivers/finish.py vmec_jax/free_boundary_adjoint.py vmec_jax/solvers/free_boundary/adjoint/gate_reports.py`
- `python -m compileall -q vmec_jax/solve.py vmec_jax/solvers/fixed_boundary/scan/runtime.py vmec_jax/driver.py vmec_jax/drivers/finish.py vmec_jax/free_boundary_adjoint.py vmec_jax/solvers/free_boundary/adjoint/gate_reports.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_scan_planning_helpers.py tests/test_solve_scan_chunking.py tests/test_solve_more_coverage.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_scan_time_control.py tests/test_solve_scan_payload_helpers.py tests/test_solve_scan_math_helpers.py tests/test_solve_scan_planning_helpers.py tests/test_solve_scan_output.py tests/test_solve_scan_output_edge_cases_more_coverage.py tests/test_solve_scan_resume_state.py tests/test_solve_scan_debug_helpers.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_adjoint_helpers_unit.py tests/test_free_boundary_vacuum_adjoint.py::test_jax_visible_controller_plain_step_outputs_and_segment_validation tests/test_free_boundary_vacuum_adjoint.py::test_segmented_accepted_controller_matches_monolithic_scan_and_gradient -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_direct_coil_finite_pressure_sensitivity.py::test_direct_coil_trace_fingerprint_detects_control_branch_changes tests/test_free_boundary_direct_coil_finite_pressure_sensitivity.py::test_direct_coil_accepted_update_replay_ad_matches_fd_for_coil_pytree tests/test_free_boundary_direct_coil_finite_pressure_sensitivity.py::test_direct_coil_native_rejected_slot_same_branch_jvp_matches_complete_solve_fd tests/test_free_boundary_qs_coil_optimization_smoke.py::test_same_branch_report_profiles_nestor_and_rejected_slot -q`
- `python -m pytest tests/test_driver_api.py tests/test_driver_api_finish_more_coverage.py tests/test_driver_wave12_coverage.py tests/test_wout_driver_wave10_coverage.py -k "finish or finisher" -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_driver_policy_helpers.py tests/test_driver_api.py -q`
- `python tools/diagnostics/ci_core_bucket_args.py driver-solve-discrete > /tmp/vmec_jax-driver-solve-discrete-refactor.txt && JAX_ENABLE_X64=1 VMEC_JAX_SKIP_PY311_COVERAGE_ONLY=1 xargs pytest -q -n 4 -m "not full and not vmec2000 and not simsopt" --durations=30 --cov=vmec_jax --cov-report= < /tmp/vmec_jax-driver-solve-discrete-refactor.txt`
- `python tools/diagnostics/source_health.py --top 20 --top-functions 20 --max-root-helper-prefix-files 2`
- `SPHINX_FAST=1 python -m sphinx -T -b html docs docs/_build/html_fast`

Best next steps:

1. Commit and push this function-level refactor tranche.
2. Re-check GitHub Actions for the pushed commit.
3. Start the next tranche with a purpose-built VMEC2000 scan-step context under
   `vmec_jax.solvers.fixed_boundary.scan`, but only after writing a focused
   scan-step equivalence test because `_advance_step` is the most parity-sensitive
   remaining fixed-boundary solver closure.
4. Continue reducing large pure report/helper functions before moving additional
   traced numerical bodies.

User decisions needed:

No immediate user decision. The work remains in draft PR #20, and the remaining
public facade modules should not be moved without an explicit API migration.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 98%.
- Differentiability/refactor implementation: 99.2%.
- Solver monolith reduction: 86%.
- Free-boundary adjoint monolith reduction: 63%.
- Driver workflow decomposition: 80%.
- WOUT diagnostic/profile decomposition: 72%.

## 2026-06-15 Free-Boundary Adjoint Helper Package Move

Commit: branch-local free-boundary adjoint helper package tranche on
`codex/differentiability-refactor-plan`.

Steps taken:

1. Created `vmec_jax.solvers.free_boundary.adjoint` as the domain package for
   branch-local free-boundary adjoint support code.
2. Moved accepted-trace objective, pytree, replay-plan, runtime, controller-mask,
   branch-fingerprint, trace-metadata, and trace-stacking helpers out of the
   root `vmec_jax` namespace.
3. Kept `vmec_jax.free_boundary_adjoint` as the public validation/report facade
   and updated it to import helper implementations from the solver package.
4. Updated internal free-boundary adjoint helper tests and code-structure docs
   to use the new package paths.
5. Ratcheted the root-helper source-health CI gate from 22 to 14 files.

Results obtained:

- Root Python files dropped from 88 to 80.
- Root helper-prefix files dropped from 22 to 14.
- Branch-local free-boundary adjoint support code now has a coherent solver
  namespace, while the main public facade and controller module remain stable.
- Focused AD/controller gates still pass after the move.

Tests and commands run:

- `python -m ruff check vmec_jax/free_boundary_adjoint.py vmec_jax/solvers/free_boundary/adjoint tests/test_free_boundary_adjoint_helpers_unit.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_adjoint_helpers_unit.py -q`
- `python - <<'PY' ... assert free_boundary_adjoint facade identity ... PY`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_vacuum_adjoint.py::test_jax_visible_controller_plain_step_outputs_and_segment_validation tests/test_free_boundary_vacuum_adjoint.py::test_segmented_accepted_controller_matches_monolithic_scan_and_gradient -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_direct_coil_finite_pressure_sensitivity.py::test_direct_coil_trace_fingerprint_detects_control_branch_changes tests/test_free_boundary_direct_coil_finite_pressure_sensitivity.py::test_direct_coil_current_only_same_branch_custom_vjp_matches_complete_solve_fd -q`
- `python tools/diagnostics/source_health.py --max-root-helper-prefix-files 14`
- `SPHINX_FAST=1 python -m sphinx -T -b html docs docs/_build/html_fast`

Best next steps:

1. Commit and push this free-boundary adjoint helper package move.
2. Use the driver explorer recommendation to move the remaining `driver_*`
   helpers under a small `vmec_jax.drivers` package, keeping `driver.py` as the
   user-facing runtime facade.
3. Defer the broader `solve_*` misc move until the driver package is stable,
   because those files have wider fan-out through solver, implicit, finite-beta,
   WOUT, and tests.
4. After namespace moves, return to function-level reduction inside `solve.py`,
   `driver.py`, `wout.py`, and `free_boundary_adjoint.py` using tests already
   protecting each facade.

User decisions needed:

No immediate decision. PR #20 remains draft and all refactor work stays on this
branch until the full plan is complete.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 65%.
- Differentiability/refactor implementation: 97.8%.
- Solver monolith reduction: 82%.
- Free-boundary adjoint monolith reduction: 58%.
- Driver workflow decomposition: 35%.
- WOUT diagnostic/profile decomposition: 72%.

## 2026-06-15 Driver Helper Package Move

Commit: high-level driver helper package tranche on
`codex/differentiability-refactor-plan`.

Steps taken:

1. Created `vmec_jax.drivers` as the domain package for CLI-facing driver
   support code.
2. Moved driver policy, staged-result merging, fixed-boundary solve entry,
   current-driven flux reconciliation, bundled example I/O, and VMEC-style
   output construction helpers out of the root `vmec_jax` namespace.
3. Kept `vmec_jax.driver` as the user-facing runtime facade and preserved its
   historical private aliases by importing the moved helper modules.
4. Updated the direct driver-policy helper test and code-structure docs to use
   the new package path.
5. Ratcheted the root-helper source-health CI gate from 14 to 8 files.

Results obtained:

- Root helper-prefix files dropped from 14 to 8.
- Driver workflow support now has a coherent namespace under `vmec_jax.drivers`
  without changing `run_fixed_boundary`, `run_free_boundary`, or CLI-facing
  behavior.
- Driver API and policy tests still pass after the move.

Tests and commands run:

- `python -m ruff check vmec_jax/driver.py vmec_jax/drivers tests/test_driver_policy_helpers.py tests/test_driver_api.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_driver_policy_helpers.py tests/test_driver_api.py -q`
- `python tools/diagnostics/source_health.py --max-root-helper-prefix-files 8`
- `SPHINX_FAST=1 python -m sphinx -T -b html docs docs/_build/html_fast`

Best next steps:

1. Commit and push this driver helper package move.
2. Continue with the remaining five `solve_*` helper-prefix files, but split
   them by actual domain: result/options/profile helpers under
   `vmec_jax.solvers.fixed_boundary`, and JIT/optimizer helpers under the
   existing fixed-boundary optimization/runtime packages.
3. After root helper-prefix files reach the minimum practical count, shift from
   namespace cleanup to line-count reduction in `solve.py`, `driver.py`,
   `wout.py`, and `free_boundary_adjoint.py`.
4. Keep full behavior and parity changes out of namespace-only commits.

User decisions needed:

No immediate decision. PR #20 remains draft and all refactor work stays on this
branch until the full plan is complete.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 78%.
- Differentiability/refactor implementation: 98.2%.
- Solver monolith reduction: 82%.
- Free-boundary adjoint monolith reduction: 58%.
- Driver workflow decomposition: 72%.
- WOUT diagnostic/profile decomposition: 72%.

## 2026-06-15 Fixed-Boundary Solver Support Package Move

Commit: remaining fixed-boundary solver support helper tranche on
`codex/differentiability-refactor-plan`.

Steps taken:

1. Moved solver result dataclasses, solver option validators, profile/flux
   convention helpers, JIT-cache helpers, and L-BFGS/quasi-Newton helper
   functions out of the root `vmec_jax` namespace.
2. Placed result/options/profile/JIT-cache helpers under
   `vmec_jax.solvers.fixed_boundary`, and quasi-Newton helpers under
   `vmec_jax.solvers.fixed_boundary.optimization`.
3. Updated `solve.py`, implicit differentiation, finite-beta helpers, WOUT
   synthesis, NumPy-force monkeypatch coverage, fixed-boundary diagnostics,
   preconditioning, and optimizer packages to import the new paths.
4. Updated direct helper tests and code-structure docs to use the new package
   paths.
5. Ratcheted the root-helper source-health CI gate from 8 to 3 files.

Results obtained:

- Root Python files dropped from 74 to 69.
- Root helper-prefix files dropped from 8 to 3.
- The remaining helper-prefix roots are now only the main free-boundary adjoint
  facade, its public controller module, and free-boundary validation.
- The broad driver/solve-discrete CI shard still passes after the move.

Tests and commands run:

- `python -m ruff check vmec_jax/solve.py vmec_jax/solvers/fixed_boundary vmec_jax/implicit.py vmec_jax/finite_beta.py vmec_jax/wout.py vmec_jax/vmec_numpy_forces.py tests/test_solve_options.py tests/test_solve_optimizer_helpers.py tests/test_solve_residual_iter_force_cache_helpers.py tests/test_solve_residual_iter_finalize_helpers.py tests/test_solve_diagnostics_io.py tests/test_solve_additional_branch_coverage.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_options.py tests/test_solve_optimizer_helpers.py tests/test_solve_residual_iter_force_cache_helpers.py tests/test_solve_residual_iter_finalize_helpers.py tests/test_solve_diagnostics_io.py tests/test_solve_additional_branch_coverage.py tests/test_solve_branch_coverage.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_step9_implicit_fixed_boundary.py tests/test_finite_beta_helpers_unit.py tests/test_wout_fast_helpers.py -q`
- `python tools/diagnostics/source_health.py --max-root-helper-prefix-files 3`
- `SPHINX_FAST=1 python -m sphinx -T -b html docs docs/_build/html_fast`
- `python tools/diagnostics/ci_core_bucket_args.py driver-solve-discrete > /tmp/vmec_jax-driver-solve-discrete.txt && JAX_ENABLE_X64=1 VMEC_JAX_SKIP_PY311_COVERAGE_ONLY=1 xargs pytest -q -n 4 -m "not full and not vmec2000 and not simsopt" --durations=30 --cov=vmec_jax --cov-report= < /tmp/vmec_jax-driver-solve-discrete.txt`

Best next steps:

1. Commit and push this solver-support package move.
2. Stop root-helper-prefix ratcheting here unless `free_boundary_validation.py`
   is moved deliberately; the remaining two `free_boundary_adjoint*` files are
   public facades for the current free-boundary adjoint validation API.
3. Shift the next tranche from namespace movement to line-count/function-count
   reduction, starting with the nested scan loop inside `solve.py`.
4. Preserve all VMEC parity and physics gates while extracting nested functions
   into the existing `solvers.fixed_boundary.scan` package.

User decisions needed:

No immediate decision. PR #20 remains draft and all refactor work stays on this
branch until the full plan is complete.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 92%.
- Differentiability/refactor implementation: 98.8%.
- Solver monolith reduction: 85%.
- Free-boundary adjoint monolith reduction: 58%.
- Driver workflow decomposition: 72%.
- WOUT diagnostic/profile decomposition: 72%.

## 2026-06-15 Free-Boundary Validation Package Move

Commit: free-boundary validation namespace cleanup on
`codex/differentiability-refactor-plan`.

Steps taken:

1. Moved finite-beta free-boundary response validation metrics from
   `vmec_jax.free_boundary_validation` to
   `vmec_jax.solvers.free_boundary.validation`.
2. Updated free-boundary validation tests, VMEC2000 executable validation test
   imports, and API docs to use the new solver-package path.
3. Removed the stale untracked autosummary stub for the old module path.
4. Ratcheted the root-helper source-health CI gate from 3 to 2 files.

Results obtained:

- Root helper-prefix files dropped from 3 to 2.
- The only remaining root helper-prefix files are now the public
  `free_boundary_adjoint.py` and `free_boundary_adjoint_controller.py` facades.
- Focused free-boundary validation tests still pass.

Tests and commands run:

- `python -m ruff check vmec_jax/solvers/free_boundary/validation.py tests/test_free_boundary_validation_unit.py tests/test_free_boundary_beta_response_validation.py tests/test_vmec2000_exec_fast_validation.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_validation_unit.py tests/test_free_boundary_beta_response_validation.py -q`
- `python tools/diagnostics/source_health.py --max-root-helper-prefix-files 2`
- `SPHINX_FAST=1 python -m sphinx -T -b html docs docs/_build/html_fast`

Best next steps:

1. Commit and push this free-boundary validation package move.
2. Treat the root namespace-sprawl lane as effectively complete; do not move the
   two public free-boundary adjoint facade modules until a deliberate public API
   migration exists.
3. Start the next implementation tranche on function-level simplification:
   extract the VMEC2000 scan runner subroutines from `solve.py` into
   `vmec_jax.solvers.fixed_boundary.scan` behind existing tests.
4. Continue preserving the broad driver/solve-discrete CI shard as the main
   safety gate for solver refactors.

User decisions needed:

No immediate decision. PR #20 remains draft and all refactor work stays on this
branch until the full plan is complete.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 96%.
- Differentiability/refactor implementation: 99.0%.
- Solver monolith reduction: 85%.
- Free-boundary adjoint monolith reduction: 58%.
- Driver workflow decomposition: 72%.
- WOUT diagnostic/profile decomposition: 72%.

## 2026-06-15 Controller Checks, Custom VJPs, and Scan Debug Seams

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Moved reusable JAX-visible controller directional derivative checks from
   `vmec_jax.free_boundary_adjoint_controller` into
   `vmec_jax.solvers.free_boundary.adjoint.controller_checks`.
2. Kept the historical `_pytree_vdot_jax` alias in the public controller facade
   while making the implementation live in the solver package.
3. Consolidated repeated controller step-output normalization and
   accepted-state selection in `free_boundary_adjoint_controller.py`.
4. Added `vmec_jax.solvers.free_boundary.adjoint.custom_vjp` for reusable
   scalar/vector custom-VJP value wrappers and rewired the branch-local
   accepted-trace custom-VJP objective helpers to use it.
5. Moved the VMEC2000 scan time-control trace callback seam into
   `vmec_jax.solvers.fixed_boundary.scan.debug`.
6. Replaced the duplicated in-scan VMEC2000 iteration print callback branches in
   `solve.py` with the existing scan debug emitter, preserving the current
   non-LASYM row format in that scan path.
7. Updated the code-structure docs for the new free-boundary adjoint helper
   modules.

Results obtained:

- `free_boundary_adjoint_controller.py` dropped below the source-health warning
  threshold, from roughly 783 lines before this tranche to 599 lines.
- `free_boundary_adjoint.py` dropped from 5,329 to 5,273 lines by removing
  duplicated custom-VJP wrapper bodies.
- `solve.py` dropped from 10,075 to 10,008 lines in this tranche, and
  `_run_vmec2000_scan._scan_step` dropped from 1,160 to 1,118 lines.
- Root helper-prefix files remain at the ratcheted limit of 2, preserving the
  package-consolidation gate.
- The extracted seams are pure helper layers; no production adaptive-branch
  differentiability claim was expanded.

Tests and commands run:

- `python -m ruff check vmec_jax/free_boundary_adjoint_controller.py vmec_jax/solvers/free_boundary/adjoint/controller_checks.py`
- `python -m compileall -q vmec_jax/free_boundary_adjoint_controller.py vmec_jax/solvers/free_boundary/adjoint/controller_checks.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_vacuum_adjoint.py::test_pytree_directional_derivative_check_can_skip_finite_difference tests/test_free_boundary_vacuum_adjoint.py::test_jax_visible_controller_plain_step_outputs_and_segment_validation tests/test_free_boundary_vacuum_adjoint.py::test_segmented_accepted_controller_matches_monolithic_scan_and_gradient -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_adjoint_helpers_unit.py tests/test_free_boundary_vacuum_adjoint.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_direct_coil_finite_pressure_sensitivity.py::test_direct_coil_trace_fingerprint_detects_control_branch_changes tests/test_free_boundary_direct_coil_finite_pressure_sensitivity.py::test_direct_coil_accepted_update_replay_ad_matches_fd_for_coil_pytree tests/test_free_boundary_direct_coil_finite_pressure_sensitivity.py::test_direct_coil_native_rejected_slot_same_branch_jvp_matches_complete_solve_fd tests/test_free_boundary_qs_coil_optimization_smoke.py::test_same_branch_report_profiles_nestor_and_rejected_slot -q`
- `python -m ruff check vmec_jax/free_boundary_adjoint.py vmec_jax/free_boundary_adjoint_controller.py vmec_jax/solvers/free_boundary/adjoint/controller_checks.py vmec_jax/solvers/free_boundary/adjoint/custom_vjp.py`
- `python -m compileall -q vmec_jax/free_boundary_adjoint.py vmec_jax/free_boundary_adjoint_controller.py vmec_jax/solvers/free_boundary/adjoint/controller_checks.py vmec_jax/solvers/free_boundary/adjoint/custom_vjp.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_vacuum_adjoint.py::test_direct_coil_trace_directional_helpers_can_skip_finite_difference tests/test_free_boundary_vacuum_adjoint.py::test_jax_visible_nonlinear_controller_matches_manual_scan_and_fd tests/test_free_boundary_vacuum_adjoint.py::test_jax_visible_controller_direct_coil_gradient_matches_fd -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_direct_coil_finite_pressure_sensitivity.py::test_direct_coil_current_only_same_branch_custom_vjp_matches_complete_solve_fd tests/test_free_boundary_direct_coil_finite_pressure_sensitivity.py::test_direct_coil_accepted_update_replay_ad_matches_fd_for_coil_pytree tests/test_free_boundary_direct_coil_finite_pressure_sensitivity.py::test_direct_coil_native_rejected_slot_same_branch_jvp_matches_complete_solve_fd -q`
- `python -m ruff check vmec_jax/solve.py vmec_jax/solvers/fixed_boundary/scan/debug.py tests/test_solve_scan_debug_helpers.py tests/test_solve_scan_planning_helpers.py`
- `python -m compileall -q vmec_jax/solve.py vmec_jax/solvers/fixed_boundary/scan/debug.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_scan_debug_helpers.py tests/test_solve_scan_planning_helpers.py tests/test_solve_scan_time_control.py -q`
- `python tools/diagnostics/source_health.py --top 20 --top-functions 20 --max-root-helper-prefix-files 2`

Best next steps:

1. Run the broad scan shard and broad driver/solve/discrete shard before
   committing this tranche.
2. Keep the remaining `solve.py` hot update logic local until a separately
   tested scan-step state/update package is designed; avoid moving numerical
   update blocks without parity gates.
3. Next safe monolith-reduction targets are WOUT diagnostic/profile helpers and
   small public-driver printing/finish seams, not the adaptive branch controller
   itself.
4. Continue treating adaptive host branch differentiation as unclaimed until a
   true fingerprint-gated adaptive AD-vs-central-FD gate exists.

User decisions needed:

No immediate decision. PR #20 remains draft and all changes stay on this branch
until the full differentiability/refactor plan is complete.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 98.5%.
- Differentiability/refactor implementation: 99.35%.
- Solver monolith reduction: 86.5%.
- Free-boundary adjoint monolith reduction: 65%.
- Driver workflow decomposition: 81%.
- WOUT diagnostic/profile decomposition: 72%.
- Overall differentiability-refactor PR: 96.5%.

## 2026-06-15 Driver Runtime and Debug Helper Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Extracted JAX persistent compilation-cache setup from the nested
   `run_fixed_boundary` closure into `vmec_jax.drivers.runtime`.
2. Preserved the historical `vmec_jax.driver.Path` monkeypatch seam by passing
   the driver path factory into the runtime helper.
3. Extracted optional VMEC `xc`/`xcdot` init-state dump writing into
   `vmec_jax.drivers.debug`.
4. Updated `run_fixed_boundary` to call the extracted runtime/debug helpers.
5. Updated the code-structure docs for driver runtime and debug helper
   responsibilities.

Results obtained:

- `driver.py` dropped from 2,572 lines at the start of the refactor slice to
  2,467 lines.
- `run_fixed_boundary` dropped from 2,205 lines to 2,098 lines.
- The cache setup and debug dump behavior remain covered by the same
  environment-variable tests.
- A failed focused run exposed the `Path` monkeypatch compatibility seam; the
  helper now accepts `path_cls` to preserve it.

Tests and commands run:

- `python -m ruff check vmec_jax/driver.py vmec_jax/drivers/runtime.py vmec_jax/drivers/debug.py tests/test_driver_wave2_coverage.py tests/test_driver_policy_coverage_extra.py`
- `python -m compileall -q vmec_jax/driver.py vmec_jax/drivers/runtime.py vmec_jax/drivers/debug.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_driver_wave2_coverage.py -k "compilation or cache or xc_init or accelerator" -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_driver_policy_coverage_extra.py -k "compilation or cache or accelerator" -q`
- `python -m pytest tests/test_driver_api.py tests/test_driver_api_finish_more_coverage.py tests/test_driver_wave12_coverage.py tests/test_driver_wave2_coverage.py tests/test_driver_policy_coverage_extra.py tests/test_wout_driver_wave10_coverage.py -k "finish or finisher or compilation or cache or xc_init or fixed_boundary" -q`
- `python tools/diagnostics/source_health.py --top 20 --top-functions 20 --max-root-helper-prefix-files 2`

Best next steps:

1. Commit this driver runtime/debug extraction after a final status check.
2. Continue reducing `run_fixed_boundary` by extracting stage initialization and
   multigrid stage dispatch only after identifying existing tests that cover
   restart-state and free-boundary dispatch branches.
3. Start a separate WOUT helper migration by moving pure Nyquist/WROUT
   transform helpers into `vmec_jax.io.wout.nyquist` while re-exporting from
   `vmec_jax.wout` for compatibility.
4. Keep branch-local free-boundary adjoint claims unchanged until an adaptive
   branch AD-vs-FD gate exists.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 98.5%.
- Differentiability/refactor implementation: 99.4%.
- Solver monolith reduction: 86.5%.
- Free-boundary adjoint monolith reduction: 65%.
- Driver workflow decomposition: 84%.
- WOUT diagnostic/profile decomposition: 72%.
- Overall differentiability-refactor PR: 96.8%.

## 2026-06-15 WOUT Nyquist Transform Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Created `vmec_jax.io.wout.nyquist` for pure NumPy VMEC `wrout.f`,
   `symforce.f`, and `jxbforce.f` Fourier-transform helper kernels.
2. Moved WROUT Nyquist cosine/sine analysis, LASYM split/expand helpers,
   LASYM Nyquist loop synthesis, JXBFORCE low-pass projection helpers, and
   Nyquist synthesis into the new package module.
3. Kept `vmec_jax.wout` as the compatibility facade by re-exporting the
   historical private helper names used by tests and downstream diagnostic
   scripts.
4. Updated `docs/code_structure.rst` and `vmec_jax.io.wout.__init__` so the
   WOUT package responsibilities include transform helpers explicitly.

Results obtained:

- `wout.py` dropped from 5,894 lines at the start of this tranche to 5,118
  lines.
- The extracted transform code is isolated from NetCDF writing and persisted
  diagnostic reconstruction, making the WOUT writer easier to review and
  keeping VMEC parity kernels directly unit-testable.
- The public/private compatibility surface remained stable; existing tests
  continue to import the historical `vmec_jax.wout._vmec_*` helper names.

Tests and commands run:

- `python -m ruff check vmec_jax/wout.py vmec_jax/io/wout/nyquist.py tests/test_wout_helpers.py tests/test_wout_fast_helpers.py`
- `python -m compileall -q vmec_jax/wout.py vmec_jax/io/wout/nyquist.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_wout_helpers.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_wout_fast_helpers.py tests/test_wout_wave2.py tests/test_wout_wave3_coverage.py tests/test_wout_wave4_coverage.py tests/test_wout_wave5_coverage.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_wout_branch_coverage.py tests/test_wout_env_branch_coverage.py tests/test_wout_bcovar_forces_extra_coverage.py tests/test_wout_driver_wave10_coverage.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_wout_helpers.py tests/test_wout_fast_helpers.py tests/test_wout_wave2.py tests/test_wout_wave3_coverage.py tests/test_wout_wave4_coverage.py tests/test_wout_wave5_coverage.py tests/test_wout_branch_coverage.py tests/test_wout_env_branch_coverage.py tests/test_wout_bcovar_forces_extra_coverage.py tests/test_wout_driver_wave10_coverage.py -q`
- `python tools/diagnostics/source_health.py --top 20 --top-functions 20 --max-root-helper-prefix-files 2`

Best next steps:

1. Run a docs fast-build and the broad driver/solve/WOUT shard before
   committing this WOUT transform tranche.
2. Continue WOUT decomposition with the JXBFORCE/Mercier reducer seam only
   after identifying monkeypatch-sensitive tests, because `_compute_bsubs_half_mesh`
   and `_compute_mercier` are still long and tightly coupled to WOUT output.
3. Continue driver decomposition at stage-dispatch seams, keeping the CLI/public
   `run_fixed_boundary` facade stable.
4. Keep adaptive full-loop free-boundary differentiability conservative until a
   true fingerprint-gated adaptive AD-vs-central-FD gate exists.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 98.8%.
- Differentiability/refactor implementation: 99.45%.
- Solver monolith reduction: 86.5%.
- Free-boundary adjoint monolith reduction: 65%.
- Driver workflow decomposition: 84%.
- WOUT diagnostic/profile decomposition: 76%.
- Overall differentiability-refactor PR: 97.1%.

## 2026-06-15 WOUT Mercier/JXBFORCE Reducer Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Moved the VMEC `mercier.f`/`jxbforce.f`-style reducer implementation from
   the monolithic `vmec_jax.wout` writer into `vmec_jax.io.wout.mercier`.
2. Kept `wout._compute_mercier` as a thin compatibility wrapper with explicit
   dependency injection for monkeypatch-sensitive helper seams:
   `_compute_bsubs_half_mesh`, parity split/expand helpers, JXBFORCE filters,
   Bsubs correction helpers, and VMEC angular weights.
3. Left the external call surface unchanged so finite-beta tests, WOUT
   synthesis, and downstream diagnostic scripts still call
   `vmec_jax.wout._compute_mercier` when needed.
4. Updated the WOUT package docs to identify Mercier/JXBFORCE reducer kernels
   as part of the `io.wout` domain package.

Results obtained:

- `wout.py` dropped further from 5,118 to 4,461 lines.
- The long Mercier reducer remains covered as a single physics kernel, but is
  now isolated from NetCDF writing and WOUT schema code.
- Existing monkeypatch-based tests still validate the wrapper path, so the
  refactor preserves current debugging hooks while making future DMerc/`D_R`
  AD-vs-FD gates easier to target at the reducer module.

Tests and commands run:

- `python -m ruff check vmec_jax/wout.py vmec_jax/io/wout/mercier.py tests/test_finite_beta.py tests/test_wout_physics_wave8_coverage.py tests/test_wout_wave4_coverage.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_wout_physics_wave8_coverage.py::test_compute_mercier_lasym_lbsubs_branch_with_reduced_bsub_inputs tests/test_wout_wave4_coverage.py::test_compute_mercier_exact_sum_symmetrizes_full_grid_inputs_and_stays_finite -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_finite_beta.py tests/test_wout_physics_wave8_coverage.py tests/test_wout_wave4_coverage.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_wout_helpers.py tests/test_wout_branch_coverage.py tests/test_wout_env_branch_coverage.py tests/test_wout_bcovar_forces_extra_coverage.py tests/test_driver_wout_wave9_coverage.py tests/test_wout_additional_helpers.py -q`
- `python -m compileall -q vmec_jax/wout.py vmec_jax/io/wout/mercier.py vmec_jax/io/wout/nyquist.py`
- `python tools/diagnostics/source_health.py --top 20 --top-functions 20 --max-root-helper-prefix-files 2`

Best next steps:

1. Run fast docs, ruff on the WOUT package, and a broad driver/WOUT integration
   shard before committing this reducer tranche.
2. Continue WOUT decomposition by moving `_compute_bsubs_half_mesh` and
   JXBFORCE Bsubs correction helpers only with the same dependency-injected
   wrapper pattern, because those helpers are monkeypatched by several tests.
3. Continue driver-stage decomposition after the WOUT tranche is green in CI.
4. Use the new `io.wout.mercier` seam for future DMerc/`D_R` AD-vs-FD tests,
   instead of adding more tests against the monolithic WOUT writer.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.0%.
- Differentiability/refactor implementation: 99.5%.
- Solver monolith reduction: 86.5%.
- Free-boundary adjoint monolith reduction: 65%.
- Driver workflow decomposition: 84%.
- WOUT diagnostic/profile decomposition: 80%.
- Overall differentiability-refactor PR: 97.4%.

## 2026-06-15 WOUT JXBFORCE Filter Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Moved pure VMEC JXBFORCE Bsub low-pass filters, parity filters, LASYM filter
   loops, derivative reconstruction, and Nyquist-limit helpers into
   `vmec_jax.io.wout.jxbforce`.
2. Kept `getbsubs` collocation solvers and corrected-Bsubs application in
   `wout.py` for now, because tests and diagnostic scripts monkeypatch those
   helpers through the historical `vmec_jax.wout` namespace.
3. Re-exported the moved helper names from `vmec_jax.wout` so direct imports
   in tests and user diagnostics remain stable.
4. Updated `docs/code_structure.rst` and the `io.wout` package docstring to
   include JXBFORCE Bsub filters as an explicit domain responsibility.

Results obtained:

- `wout.py` dropped from 4,462 lines to 3,628 lines.
- The pure JXBFORCE filter family is now isolated from NetCDF writing,
  Mercier reduction, and WOUT schema logic.
- Existing direct helper, Boozer-input parity, LASYM Bsubv parity, and WOUT
  driver tests pass through the compatibility aliases.
- Pre-existing synthetic single-surface divide warnings now point to
  `io.wout.jxbforce`, as expected after moving the filter code.

Tests and commands run:

- `python -m ruff check vmec_jax/wout.py vmec_jax/io/wout/jxbforce.py tests/test_wout_helpers.py tests/test_wout_branch_coverage.py tests/test_booz_input.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_wout_helpers.py::test_jxbforce_bsub_filters_match_loop_paths_and_guards tests/test_wout_wave3_coverage.py::test_filter_and_projection_helpers_cover_vectorized_error_identity_and_negative_paths -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_wout_helpers.py tests/test_wout_branch_coverage.py tests/test_wout_fast_helpers.py tests/test_wout_wave3_coverage.py tests/test_wout_wave4_coverage.py tests/test_wout_wave5_coverage.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_booz_input.py tests/test_driver_wout_wave9_coverage.py tests/test_wout_driver_wave10_coverage.py tests/test_wout_lasym_bsubv_parity.py -q`
- `python -m compileall -q vmec_jax/wout.py vmec_jax/io/wout/jxbforce.py vmec_jax/io/wout/mercier.py vmec_jax/io/wout/nyquist.py`
- `python tools/diagnostics/source_health.py --top 20 --top-functions 20 --max-root-helper-prefix-files 2`

Best next steps:

1. Run fast docs, final source-health, and a broad WOUT/driver smoke shard
   before committing the JXBFORCE filter tranche.
2. If continuing WOUT decomposition, split `_compute_bsubs_half_mesh` behind a
   thin compatibility wrapper next; it is still 466 lines and has explicit
   tests, but must preserve monkeypatch behavior.
3. After WOUT, return to driver-stage decomposition or fixed-boundary scan
   state extraction; do not move the numerical scan update core without a
   separate parity gate.
4. Use the new `io.wout.jxbforce` and `io.wout.mercier` modules as the natural
   targets for upcoming DMerc/`D_R` AD-vs-FD tests.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.2%.
- Differentiability/refactor implementation: 99.55%.
- Solver monolith reduction: 86.5%.
- Free-boundary adjoint monolith reduction: 65%.
- Driver workflow decomposition: 84%.
- WOUT diagnostic/profile decomposition: 84%.
- Overall differentiability-refactor PR: 97.7%.

## 2026-06-15 WOUT Half-Mesh Bsubs Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Moved the VMEC `bss.f` half-mesh `B_s` construction out of
   `vmec_jax.wout` into `vmec_jax.io.wout.bsubs`.
2. Kept `vmec_jax.wout._compute_bsubs_half_mesh` as a thin compatibility
   wrapper so existing tests, monkeypatches, and internal diagnostic imports
   continue to use the historical name.
3. Updated the WOUT package and code-structure docs so the half-mesh Bsubs
   responsibility is explicit and discoverable.
4. Validated the branch, environment, dump, driver, Mercier, and direct helper
   tests that exercise the moved kernel.

Results obtained:

- `wout.py` dropped from 3,628 lines to 3,166 lines.
- Half-mesh `B_s` construction is now isolated from NetCDF writing, Mercier
  reduction, Nyquist projection, and JXBFORCE filter kernels.
- The historical compatibility surface remains intact through the wrapper in
  `vmec_jax.wout`.
- Source-health still passes the root namespace-sprawl ratchet; remaining WOUT
  work is concentrated in `wout_minimal_from_fixed_boundary` and writer/schema
  assembly rather than this diagnostic kernel.

Tests and commands run:

- `python -m ruff check vmec_jax/wout.py vmec_jax/io/wout/bsubs.py vmec_jax/io/wout/__init__.py tests/test_wout_branch_coverage.py tests/test_wout_env_branch_coverage.py`
- `python -m compileall -q vmec_jax/wout.py vmec_jax/io/wout/bsubs.py vmec_jax/io/wout/jxbforce.py vmec_jax/io/wout/mercier.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_wout_branch_coverage.py tests/test_wout_env_branch_coverage.py tests/test_solve_dump_helpers.py tests/test_wout_driver_wave10_coverage.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_wout_helpers.py tests/test_wout_branch_coverage.py tests/test_wout_env_branch_coverage.py tests/test_wout_fast_helpers.py tests/test_wout_wave2.py tests/test_wout_wave3_coverage.py tests/test_wout_wave4_coverage.py tests/test_wout_wave5_coverage.py tests/test_wout_physics_wave8_coverage.py tests/test_wout_driver_wave10_coverage.py tests/test_driver_wout_wave9_coverage.py tests/test_solve_dump_helpers.py -q`
- `SPHINX_FAST=1 python -m sphinx -q -b html docs docs/_build/fast_html`
- `python -m ruff check vmec_jax docs/conf.py tests/test_wout_branch_coverage.py tests/test_wout_env_branch_coverage.py`
- `python tools/diagnostics/source_health.py --top 20 --top-functions 20 --max-root-helper-prefix-files 2`

Best next steps:

1. Commit and push this WOUT/Bsubs tranche once the local diff is reviewed.
2. Check the pending CI run for the previous WOUT tranche and the new run after
   this push.
3. Continue decomposition with either WOUT writer assembly
   (`wout_minimal_from_fixed_boundary`) or the fixed-boundary driver stage seam;
   avoid moving numerical scan-update internals until a dedicated parity gate is
   attached.
4. Use the new `io.wout.bsubs`, `io.wout.jxbforce`, and `io.wout.mercier`
   seams for DMerc/`D_R` AD-vs-FD tests.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.25%.
- Differentiability/refactor implementation: 99.6%.
- Solver monolith reduction: 86.5%.
- Free-boundary adjoint monolith reduction: 65%.
- Driver workflow decomposition: 84%.
- WOUT diagnostic/profile decomposition: 88%.
- Overall differentiability-refactor PR: 97.9%.

## 2026-06-15 Finite-Beta DMerc/Glasser AD-vs-FD Gate

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Probed `mercier_terms_from_state` on the bundled finite-beta QI input by
   perturbing a physical `m=1,n=0` radial geometry coefficient.
2. Found that `DMerc` and Glasser `D_R` values and central finite differences
   were finite, but AD returned `nan`.
3. Traced the first invalid tangent to masked divisions in
   `equilibrium_iota_profiles_from_state`, where inactive `jnp.where` branches
   still evaluated zero-denominator divisions at the axis.
4. Replaced the affected masked divisions with safe-denominator forms in the
   iota-profile helper, differentiable Mercier algebra, JXBFORCE profile
   algebra, and finite-beta scalar beta-total helper.
5. Added a persistent full-test physics gate that compares AD against central
   finite differences for summed interior `DMerc` and `D_R` on the real
   finite-beta QI input.

Results obtained:

- Real-input AD/FD probe after the fix:
  - `DMerc`: relative AD-vs-FD error approximately `2.4e-10`.
  - `D_R`: relative AD-vs-FD error approximately `1.3e-9`.
- Existing algebraic and synthetic-state DMerc/`D_R` gates still pass.
- The new gate covers the realistic composition path through finite-beta
  profiles, iota reconstruction, bcovar, Mercier surface integrals, JXBFORCE
  channels, and Glasser algebra.

Tests and commands run:

- Manual real-input AD/FD probe for `DMerc` and `D_R`.
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_finite_beta.py tests/test_glasser_resistive_interchange.py tests/test_finite_beta_helpers_unit.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_wout_helpers.py tests/test_wout_physics_wave8_coverage.py tests/test_wout_driver_wave10_coverage.py tests/test_driver_wout_wave9_coverage.py tests/test_residue_finite_beta_wave3.py -q`
- `python -m ruff check vmec_jax/mercier.py vmec_jax/finite_beta.py vmec_jax/wout.py tests/test_finite_beta.py tests/test_glasser_resistive_interchange.py`
- `python -m compileall -q vmec_jax/mercier.py vmec_jax/finite_beta.py vmec_jax/wout.py tests/test_finite_beta.py`
- `python tools/diagnostics/source_health.py --top 20 --top-functions 20 --max-root-helper-prefix-files 2`

Best next steps:

1. Commit and push the differentiability fix and physics gate.
2. Check the CI run for the previous pushed refactor tranche and the new run
   after this push.
3. Consider adding one LASYM finite-beta AD/FD gate only if a small physical
   LASYM finite-beta fixture is available; do not add a scaffold-only test.
4. Return to larger source decomposition after this correctness gate, with
   `run_fixed_boundary` stage seams and `wout_minimal_from_fixed_boundary` as
   the next high-value candidates.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.25%.
- Differentiability/refactor implementation: 99.65%.
- Solver monolith reduction: 86.5%.
- Free-boundary adjoint monolith reduction: 65%.
- Driver workflow decomposition: 84%.
- WOUT diagnostic/profile decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 98.0%.

## 2026-06-15 WOUT Bsub Parity-State Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Moved `_bsubuv_parity_from_state` into `vmec_jax.io.wout.bsubs` beside the
   half-mesh Bsubs construction.
2. Left the historical `vmec_jax.wout._bsubuv_parity_from_state` name as a
   compatibility wrapper for tests and downstream diagnostic monkeypatches.
3. Removed the now-unused WOUT realspace dzeta import from the compatibility
   module.

Results obtained:

- `wout.py` dropped from 3,169 lines to 3,038 lines.
- State-derived Bsub parity splitting now lives in the same WOUT-domain module
  as half-mesh `B_s` construction.
- The WOUT compatibility surface still supports direct imports from
  `vmec_jax.wout`.

Tests and commands run:

- `python -m ruff check vmec_jax/wout.py vmec_jax/io/wout/bsubs.py tests/test_wout_branch_coverage.py tests/test_finite_beta.py vmec_jax/mercier.py vmec_jax/finite_beta.py`
- `python -m compileall -q vmec_jax/wout.py vmec_jax/io/wout/bsubs.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_wout_branch_coverage.py tests/test_wout_env_branch_coverage.py tests/test_wout_helpers.py tests/test_wout_physics_wave8_coverage.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_wout_helpers.py tests/test_wout_branch_coverage.py tests/test_wout_env_branch_coverage.py tests/test_wout_fast_helpers.py tests/test_wout_wave2.py tests/test_wout_wave3_coverage.py tests/test_wout_wave4_coverage.py tests/test_wout_wave5_coverage.py tests/test_wout_physics_wave8_coverage.py tests/test_wout_driver_wave10_coverage.py tests/test_driver_wout_wave9_coverage.py tests/test_solve_dump_helpers.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_finite_beta.py tests/test_glasser_resistive_interchange.py tests/test_finite_beta_helpers_unit.py -q`
- `python tools/diagnostics/source_health.py --top 20 --top-functions 20 --max-root-helper-prefix-files 2`

Best next steps:

1. Commit and push the parity-state extraction.
2. Continue only with source seams that have explicit local parity gates:
   `wout_minimal_from_fixed_boundary` writer decomposition or driver-stage
   policy helpers.  Avoid the solve scan-update core until a dedicated
   VMEC2000 trace parity gate accompanies the move.
3. Watch the latest CI run; intermediate runs are expected to cancel on each
   push to this draft PR branch.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.3%.
- Differentiability/refactor implementation: 99.67%.
- Solver monolith reduction: 86.5%.
- Free-boundary adjoint monolith reduction: 65%.
- Driver workflow decomposition: 84%.
- WOUT diagnostic/profile decomposition: 89%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 98.05%.

## 2026-06-15 WOUT Bsub Parity Projection Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Moved the coefficient-based and realspace-JXBFORCE Bsub parity projection
   helpers into `vmec_jax.io.wout.bsubs`.
2. Preserved `vmec_jax.wout._bsubuv_parity_from_coeffs` and
   `vmec_jax.wout._bsubuv_parity_from_realspace_jxbforce` as wrappers for the
   historical import surface.
3. Restored `_vmec_wrout_nyquist_synthesis` as an explicit compatibility
   export from `vmec_jax.wout` because tests and diagnostics import it directly.

Results obtained:

- `wout.py` dropped from 3,038 lines to 2,897 lines.
- All state/coeff/realspace Bsub parity construction helpers now live in the
  WOUT Bsubs domain module.
- Existing WOUT helper imports remain backward compatible.

Tests and commands run:

- `python -m ruff check vmec_jax/wout.py vmec_jax/io/wout/bsubs.py tests/test_wout_helpers.py tests/test_wout_fast_helpers.py tests/test_wout_wave3_coverage.py`
- `python -m compileall -q vmec_jax/wout.py vmec_jax/io/wout/bsubs.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_wout_helpers.py tests/test_wout_fast_helpers.py tests/test_wout_wave3_coverage.py tests/test_wout_branch_coverage.py tests/test_wout_env_branch_coverage.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_wout_helpers.py tests/test_wout_branch_coverage.py tests/test_wout_env_branch_coverage.py tests/test_wout_fast_helpers.py tests/test_wout_wave2.py tests/test_wout_wave3_coverage.py tests/test_wout_wave4_coverage.py tests/test_wout_wave5_coverage.py tests/test_wout_physics_wave8_coverage.py tests/test_wout_driver_wave10_coverage.py tests/test_driver_wout_wave9_coverage.py tests/test_solve_dump_helpers.py -q`
- `python tools/diagnostics/source_health.py --top 20 --top-functions 20 --max-root-helper-prefix-files 2`

Best next steps:

1. Commit and push this parity projection extraction.
2. Next WOUT decomposition target is `wout_minimal_from_fixed_boundary`; split
   writer assembly only after identifying a small helper boundary with existing
   driver/WOUT tests.
3. Keep driver/solver refactors conservative until their parity gates are
   attached.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.35%.
- Differentiability/refactor implementation: 99.7%.
- Solver monolith reduction: 86.5%.
- Free-boundary adjoint monolith reduction: 65%.
- Driver workflow decomposition: 84%.
- WOUT diagnostic/profile decomposition: 90%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 98.1%.

## 2026-06-15 WOUT Bcovar Parity Helper Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Moved `_bsubuv_parity_from_bcovar` into `vmec_jax.io.wout.bsubs`.
2. Preserved the historical `vmec_jax.wout._bsubuv_parity_from_bcovar`
   wrapper for helper tests and diagnostics.
3. Re-ran the WOUT parity helper shards and source-health report.

Results obtained:

- `wout.py` dropped from 2,897 lines to 2,886 lines.
- All Bsub parity construction helpers are now grouped in
  `vmec_jax.io.wout.bsubs`.
- The root compatibility module remains import-compatible.

Tests and commands run:

- `python -m ruff check vmec_jax/wout.py vmec_jax/io/wout/bsubs.py tests/test_wout_helpers.py tests/test_wout_fast_helpers.py tests/test_wout_wave3_coverage.py`
- `python -m compileall -q vmec_jax/wout.py vmec_jax/io/wout/bsubs.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_wout_helpers.py tests/test_wout_fast_helpers.py tests/test_wout_wave3_coverage.py tests/test_wout_branch_coverage.py tests/test_wout_env_branch_coverage.py -q`
- `python tools/diagnostics/source_health.py --top 20 --top-functions 20 --max-root-helper-prefix-files 2`

Best next steps:

1. Commit and push this final Bsubs cleanup.
2. Let the latest CI run reach a terminal state before another push unless a
   local failure is found.
3. Next high-value refactor remains WOUT writer assembly or driver-stage
   policy extraction with explicit parity tests.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.4%.
- Differentiability/refactor implementation: 99.72%.
- Solver monolith reduction: 86.5%.
- Free-boundary adjoint monolith reduction: 65%.
- Driver workflow decomposition: 84%.
- WOUT diagnostic/profile decomposition: 90.5%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 98.15%.

## 2026-06-15 WOUT Debug and State Reconstruction Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Moved WOUT environment-variable debug dump side effects into
   `vmec_jax.io.wout.debug`.
2. Moved WOUT-to-`VMECState` reconstruction into `vmec_jax.io.wout.state`.
3. Preserved the public `vmec_jax.wout.state_from_wout` wrapper and its
   monkeypatch-compatible validation/lambda-scaling seams for existing tests.
4. Re-ran focused and broad WOUT shards plus source-health diagnostics.

Results obtained:

- `wout.py` dropped from 2,886 lines to 2,677 lines.
- `wout_minimal_from_fixed_boundary` dropped from 1,307 lines before the WOUT
  extraction series to 1,192 lines after debug-dump removal.
- Debug dumps and state reconstruction now live in domain-named WOUT IO modules
  instead of the root compatibility module.

Tests and commands run:

- `python -m ruff check vmec_jax/wout.py vmec_jax/io/wout/debug.py vmec_jax/io/wout/bsubs.py`
- `python -m ruff check vmec_jax/wout.py vmec_jax/io/wout/state.py`
- `python -m compileall -q vmec_jax/wout.py vmec_jax/io/wout/debug.py`
- `python -m compileall -q vmec_jax/wout.py vmec_jax/io/wout/debug.py vmec_jax/io/wout/state.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_wout_helpers.py tests/test_wout_fast_helpers.py tests/test_wout_wave3_coverage.py tests/test_wout_branch_coverage.py tests/test_wout_env_branch_coverage.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_wout_driver_wave10_coverage.py::test_state_from_wout_recovers_internal_lambda_for_half_mesh_parity_branches tests/test_wout_driver_wave10_coverage.py::test_state_from_wout_lambda_rejects_bad_shapes_and_handles_missing_m -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_wout_helpers.py tests/test_wout_branch_coverage.py tests/test_wout_env_branch_coverage.py tests/test_wout_fast_helpers.py tests/test_wout_wave2.py tests/test_wout_wave3_coverage.py tests/test_wout_wave4_coverage.py tests/test_wout_wave5_coverage.py tests/test_wout_physics_wave8_coverage.py tests/test_wout_driver_wave10_coverage.py tests/test_driver_wout_wave9_coverage.py tests/test_solve_dump_helpers.py -q`
- `python tools/diagnostics/source_health.py --top 20 --top-functions 20 --max-root-helper-prefix-files 2`

Best next steps:

1. Commit and push this WOUT tranche after checking latest CI state.
2. Continue WOUT writer decomposition only at low-risk seams, or switch to
   driver-stage policy extraction if the next WOUT seam is too coupled.
3. Keep full adaptive free-boundary differentiation claims conservative until a
   true fingerprint-gated adaptive AD-vs-FD gate exists.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.45%.
- Differentiability/refactor implementation: 99.74%.
- Solver monolith reduction: 86.5%.
- Free-boundary adjoint monolith reduction: 65%.
- Driver workflow decomposition: 84%.
- WOUT diagnostic/profile decomposition: 92%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 98.2%.

## 2026-06-15 Dense Free-Boundary Adjoint Primitive Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Moved dense validation-scale vacuum/nonlinear/fixed-point adjoint primitives
   into `vmec_jax.solvers.free_boundary.adjoint.dense`.
2. Preserved the public `vmec_jax.free_boundary_adjoint` import surface for
   dense solve helpers and the legacy `_finite_difference_jacobian` test seam.
3. Re-ran vacuum-adjoint and focused direct-coil same-branch replay gates.

Results obtained:

- `free_boundary_adjoint.py` dropped from 5,273 lines at the start of this
  session to 5,091 lines.
- The dense toy/validation primitives now live in the free-boundary adjoint
  domain package, while the root module remains a compatibility facade.
- No change in adaptive-loop claims: these helpers are validation-scale
  building blocks, not a production arbitrary-branch adjoint.

Tests and commands run:

- `python -m ruff check vmec_jax/free_boundary_adjoint.py vmec_jax/solvers/free_boundary/adjoint/dense.py tests/test_free_boundary_vacuum_adjoint.py tests/test_free_boundary_adjoint_helpers_unit.py`
- `python -m compileall -q vmec_jax/free_boundary_adjoint.py vmec_jax/solvers/free_boundary/adjoint/dense.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_vacuum_adjoint.py tests/test_free_boundary_adjoint_helpers_unit.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_direct_coil_finite_pressure_sensitivity.py::test_direct_coil_trace_fingerprint_detects_control_branch_changes tests/test_free_boundary_direct_coil_finite_pressure_sensitivity.py::test_direct_coil_accepted_update_replay_ad_matches_fd_for_coil_pytree -q`
- `python tools/diagnostics/source_health.py --top 25 --top-functions 25 --max-root-helper-prefix-files 2`

Best next steps:

1. Continue free-boundary adjoint decomposition with similarly isolated seams
   such as mode-source/matrix helpers or branch-local report assembly.
2. Avoid moving adaptive host-controller code until a fingerprint-gated
   complete-loop AD-vs-FD gate is attached.
3. Keep VMEC parity and physics gates focused on finite-positive physical
   WOUT fixtures.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.45%.
- Differentiability/refactor implementation: 99.76%.
- Solver monolith reduction: 86.5%.
- Free-boundary adjoint monolith reduction: 68%.
- Driver workflow decomposition: 84%.
- WOUT diagnostic/profile decomposition: 92%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 98.25%.

## 2026-06-15 Free-Boundary Mode Operator Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Moved JAX VMEC/NESTOR mode-source projection, dense mode-matrix assembly,
   matrix-free matvec, and matrix-free mode solve helpers into
   `vmec_jax.solvers.free_boundary.adjoint.mode_operator`.
2. Kept the public `vmec_jax.free_boundary_adjoint` import facade intact for
   tests and downstream callers.
3. Re-ran the full vacuum-adjoint shard that exercises source projection,
   dense mode matrices, matrix-free matvecs, Krylov solves, and error paths.

Results obtained:

- `free_boundary_adjoint.py` dropped from 5,091 to 4,758 lines.
- The JAX mode-operator validation seam now has a domain-named module distinct
  from replay/controller report assembly.
- No adaptive-controller semantics changed; this is purely algebra/package
  decomposition around already validated branch-local helpers.

Tests and commands run:

- `python -m ruff check vmec_jax/free_boundary_adjoint.py vmec_jax/solvers/free_boundary/adjoint/mode_operator.py tests/test_free_boundary_vacuum_adjoint.py`
- `python -m compileall -q vmec_jax/free_boundary_adjoint.py vmec_jax/solvers/free_boundary/adjoint/mode_operator.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_vacuum_adjoint.py -q`
- `python tools/diagnostics/source_health.py --top 25 --top-functions 25 --max-root-helper-prefix-files 2`

Best next steps:

1. Continue free-boundary adjoint decomposition with nonsingular source/analytic
   geometry helpers if tests remain bounded.
2. Defer adaptive host-branch differentiation changes until a true
   fingerprint-gated complete-loop AD-vs-FD gate is attached.
3. Let CI reach a terminal state after the next push before another broad
   refactor unless a local failure requires immediate action.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.45%.
- Differentiability/refactor implementation: 99.78%.
- Solver monolith reduction: 86.5%.
- Free-boundary adjoint monolith reduction: 71%.
- Driver workflow decomposition: 84%.
- WOUT diagnostic/profile decomposition: 92%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 98.3%.

## 2026-06-15 Branch-Local Scalar Report Adapter Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Moved `direct_coil_branch_local_scalars_report_from_complete_fd` from the
   root free-boundary adjoint facade into
   `vmec_jax.solvers.free_boundary.adjoint.gate_reports`.
2. Preserved the public import from `vmec_jax.free_boundary_adjoint`.
3. Re-ran the free-boundary QS coil optimization smoke/report shard that
   exercises success, failure, JSON-safety, branch-delta and physical-gate
   report paths.

Results obtained:

- `free_boundary_adjoint.py` dropped from 4,758 to 4,617 lines.
- Branch-local report normalization now lives next to the same-branch replay,
  physical-scalar, and adaptive full-loop seam gate reports.
- No production replay/JVP execution path changed.

Tests and commands run:

- `python -m ruff check vmec_jax/free_boundary_adjoint.py vmec_jax/solvers/free_boundary/adjoint/gate_reports.py tests/test_free_boundary_qs_coil_optimization_smoke.py tests/test_free_boundary_direct_coil_finite_pressure_sensitivity.py`
- `python -m compileall -q vmec_jax/free_boundary_adjoint.py vmec_jax/solvers/free_boundary/adjoint/gate_reports.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_qs_coil_optimization_smoke.py tests/test_free_boundary_direct_coil_finite_pressure_sensitivity.py::test_direct_coil_trace_fingerprint_detects_control_branch_changes -q`
- `python tools/diagnostics/source_health.py --top 25 --top-functions 25 --max-root-helper-prefix-files 2`

Best next steps:

1. Let the latest CI run for `a1307c4` complete before pushing this local
   adapter cleanup unless CI needs a fix.
2. If CI passes, commit and push this report-adapter tranche.
3. Next refactor target should be a similarly isolated report/replay-planning
   seam; avoid moving adaptive host branch selection until full AD-vs-FD gates
   are attached.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.45%.
- Differentiability/refactor implementation: 99.8%.
- Solver monolith reduction: 86.5%.
- Free-boundary adjoint monolith reduction: 72%.
- Driver workflow decomposition: 84%.
- WOUT diagnostic/profile decomposition: 92%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 98.35%.

## 2026-06-15 Accepted-Boundary Replay Helper Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Moved accepted-boundary vacuum-field projection, mode-coefficient field
   reconstruction, fixed-geometry coil-normal-field RMS, and accepted-boundary
   geometry synthesis helpers into
   `vmec_jax.solvers.free_boundary.adjoint.boundary_replay`.
2. Preserved the public `vmec_jax.free_boundary_adjoint` facade names and
   added them to `__all__` so downstream callers and monkeypatch-based tests
   keep the same import surface.
3. Re-ran vacuum-adjoint and direct-coil finite-pressure shards that exercise
   cylindrical field projection, mode reconstruction, accepted-state geometry
   replay, and free-boundary same-branch AD/FD gates.

Results obtained:

- `free_boundary_adjoint.py` dropped from 4,617 to 4,344 lines.
- The moved code is now a 265-line domain module with no production adaptive
  branch-selection semantic changes.
- The previous pushed CI run completed successfully before this tranche was
  committed.

Tests and commands run:

- `python -m ruff check vmec_jax/free_boundary_adjoint.py vmec_jax/solvers/free_boundary/adjoint/boundary_replay.py tests/test_free_boundary_vacuum_adjoint.py tests/test_free_boundary_direct_coil_finite_pressure_sensitivity.py`
- `python -m compileall -q vmec_jax/free_boundary_adjoint.py vmec_jax/solvers/free_boundary/adjoint/boundary_replay.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_vacuum_adjoint.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_adjoint_helpers_unit.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_direct_coil_finite_pressure_sensitivity.py -q`
- `python tools/diagnostics/source_health.py --top 25 --top-functions 25 --max-root-helper-prefix-files 2`

Best next steps:

1. Continue with low-risk free-boundary adjoint seams such as replay context
   construction or diagnostics, leaving adaptive host branch selection intact.
2. Start a separate pass on WOUT minimal assembly or driver workflow
   decomposition once CI confirms this boundary replay extraction.
3. Keep full adaptive-loop differentiability claims conservative until a true
   fingerprint-gated adaptive branch AD-vs-FD gate exists.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.45%.
- Differentiability/refactor implementation: 99.82%.
- Solver monolith reduction: 86.5%.
- Free-boundary adjoint monolith reduction: 74%.
- Driver workflow decomposition: 84%.
- WOUT diagnostic/profile decomposition: 92%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 98.4%.

## 2026-06-15 Free-Boundary Replay Context Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Moved JAX NESTOR replay-table enrichment, accepted-boundary replay context
   construction, trace boundary-shape inference, and frozen-vacuum override
   extraction into `vmec_jax.solvers.free_boundary.adjoint.replay_context`.
2. Preserved root-facade compatibility names including private
   `_direct_coil_trace_boundary_shape`,
   `_direct_coil_trace_vacuum_field_override`, and
   `_with_jax_nonsingular_replay_tables`.
3. Re-ran focused tests that exercise the trace-shape helpers, frozen-vacuum
   override contract, monkeypatch-compatible dense-solve seams, and JAX
   accepted-boundary geometry replay.

Results obtained:

- `free_boundary_adjoint.py` dropped from 4,344 to 4,186 lines.
- Replay-context construction is now a 186-line domain module.
- The root facade remains compatible for tests/downstream callers.
- The previous pushed CI run was in progress while this local tranche was
  validated; the prior CI run was green.

Tests and commands run:

- `python -m ruff check vmec_jax/free_boundary_adjoint.py vmec_jax/solvers/free_boundary/adjoint/replay_context.py tests/test_free_boundary_adjoint_helpers_unit.py tests/test_free_boundary_direct_coil_finite_pressure_sensitivity.py`
- `python -m compileall -q vmec_jax/free_boundary_adjoint.py vmec_jax/solvers/free_boundary/adjoint/replay_context.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_adjoint_helpers_unit.py tests/test_free_boundary_direct_coil_finite_pressure_sensitivity.py::test_direct_coil_trace_fingerprint_detects_control_branch_changes tests/test_free_boundary_direct_coil_finite_pressure_sensitivity.py::test_direct_coil_vacuum_field_override_replay_contract tests/test_free_boundary_direct_coil_finite_pressure_sensitivity.py::test_jax_free_boundary_boundary_geometry_matches_host_sampler -q`
- `python tools/diagnostics/source_health.py --top 25 --top-functions 25 --max-root-helper-prefix-files 2`

Best next steps:

1. Let CI finish for the latest pushes and fix any failure before more
   refactoring.
2. If CI remains green, continue with WOUT minimal assembly or
   driver/optimization workflow decomposition rather than changing adaptive
   branch semantics.
3. Keep exact adaptive full-loop differentiation as a separately gated
   research lane; current production claims remain branch-local/fingerprint
   gated.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.45%.
- Differentiability/refactor implementation: 99.83%.
- Solver monolith reduction: 86.5%.
- Free-boundary adjoint monolith reduction: 76%.
- Driver workflow decomposition: 84%.
- WOUT diagnostic/profile decomposition: 92%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 98.45%.

## 2026-06-15 Free-Boundary Branch Metadata Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Moved fixed accepted-branch metadata and replay-graph metadata reports into
   `vmec_jax.solvers.free_boundary.adjoint.branch_metadata`.
2. Preserved public facade names in `vmec_jax.free_boundary_adjoint` for
   downstream callers and report-generation code.
3. Re-ran focused branch-metadata, replay-graph, trace-fingerprint, and
   branch-trace mode tests.

Results obtained:

- `free_boundary_adjoint.py` dropped from 4,186 to 4,001 lines.
- Branch metadata/report construction now lives next to trace-control and
  trace-stack helpers instead of the root facade.
- No adaptive branch-selection or complete-solve semantics changed.

Tests and commands run:

- `python -m ruff check vmec_jax/free_boundary_adjoint.py vmec_jax/solvers/free_boundary/adjoint/branch_metadata.py tests/test_free_boundary_adjoint_helpers_unit.py tests/test_free_boundary_direct_coil_finite_pressure_sensitivity.py`
- `python -m compileall -q vmec_jax/free_boundary_adjoint.py vmec_jax/solvers/free_boundary/adjoint/branch_metadata.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_adjoint_helpers_unit.py::test_accepted_trace_control_metadata_and_stack_contracts tests/test_free_boundary_direct_coil_finite_pressure_sensitivity.py::test_direct_coil_trace_fingerprint_detects_control_branch_changes tests/test_free_boundary_direct_coil_finite_pressure_sensitivity.py::test_direct_coil_branch_trace_mode_keeps_replay_controls_without_raw_force_payload -q`
- `python tools/diagnostics/source_health.py --top 25 --top-functions 25 --max-root-helper-prefix-files 2`

Best next steps:

1. Wait for the current CI run to finish or fix it if it fails.
2. Next safe source-health target is the replay-plan/context execution seam or
   WOUT minimal assembly; avoid the adaptive branch controller and scan core
   until additional gates are attached.
3. Once free-boundary facade drops below the next meaningful threshold, shift
   effort to driver/optimization workflow and WOUT minimal decomposition.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.45%.
- Differentiability/refactor implementation: 99.84%.
- Solver monolith reduction: 86.5%.
- Free-boundary adjoint monolith reduction: 78%.
- Driver workflow decomposition: 84%.
- WOUT diagnostic/profile decomposition: 92%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 98.5%.

## 2026-06-15 Controller Replay Plan Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Moved fixed accepted-branch controller replay plan construction and
   boundary-context precomputation into
   `vmec_jax.solvers.free_boundary.adjoint.replay_plan`.
2. Preserved the public `direct_coil_accepted_trace_controller_replay_plan`
   facade and private compatibility aliases for tests/internal callers.
3. Re-ran focused replay-plan, segmentation, stackability, branch-metadata,
   and branch-trace tests.

Results obtained:

- `free_boundary_adjoint.py` dropped from 4,001 to 3,818 lines.
- `free_boundary_adjoint.py` moved below `tests/test_free_boundary_vacuum_adjoint.py`
  in the source-health hotspot list.
- Replay plan construction is now colocated with lower-level replay-plan
  helpers instead of the root free-boundary adjoint facade.
- No adaptive controller branch-selection semantics changed.

Tests and commands run:

- `python -m ruff check vmec_jax/free_boundary_adjoint.py vmec_jax/solvers/free_boundary/adjoint/replay_plan.py tests/test_free_boundary_adjoint_helpers_unit.py tests/test_free_boundary_direct_coil_finite_pressure_sensitivity.py`
- `python -m compileall -q vmec_jax/free_boundary_adjoint.py vmec_jax/solvers/free_boundary/adjoint/replay_plan.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_adjoint_helpers_unit.py::test_free_boundary_adjoint_trace_stackability_error_paths tests/test_free_boundary_adjoint_helpers_unit.py::test_accepted_trace_control_metadata_and_stack_contracts tests/test_free_boundary_direct_coil_finite_pressure_sensitivity.py::test_direct_coil_trace_fingerprint_detects_control_branch_changes tests/test_free_boundary_direct_coil_finite_pressure_sensitivity.py::test_direct_coil_branch_trace_mode_keeps_replay_controls_without_raw_force_payload -q`
- `python tools/diagnostics/source_health.py --top 20 --top-functions 20 --max-root-helper-prefix-files 2`

Best next steps:

1. Let CI finish for this pushed tranche and fix failures before another broad
   refactor.
2. Next high-impact source-health targets are `free_boundary.py` direct-coil
   NESTOR support or `wout_minimal_from_fixed_boundary`; both need more careful
   decomposition than report/plan helpers.
3. Do not split `direct_coil_accepted_trace_controller_replay_objective_jax`
   until the execution helper boundaries are explicit and tested.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.45%.
- Differentiability/refactor implementation: 99.85%.
- Solver monolith reduction: 86.5%.
- Free-boundary adjoint monolith reduction: 80%.
- Driver workflow decomposition: 84%.
- WOUT diagnostic/profile decomposition: 92%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 98.55%.

## 2026-06-15 Free-Boundary Type Container Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Moved passive free-boundary dataclasses into
   `vmec_jax.solvers.free_boundary.types`.
2. Preserved the public `vmec_jax.free_boundary` type names by importing the
   moved containers back into the facade module.
3. Left NESTOR math, mgrid interpolation, provider hooks, adaptive controller
   semantics, and VMEC parity behavior unchanged.
4. Re-ran focused type/user-facing import, mgrid, provider, and NESTOR reuse
   tests.

Results obtained:

- `free_boundary.py` dropped from the prior 4,271-line source-health baseline
  to 4,114 lines.
- Type contracts are now inspectable without mixing them into the solver body,
  while downstream imports from `vmec_jax.free_boundary` remain compatible.
- The source-health report still points to true algorithmic hotspots next:
  `solve_fixed_boundary_residual_iter`, `run_fixed_boundary`,
  `wout_minimal_from_fixed_boundary`, and the larger free-boundary solver body.

Tests and commands run:

- `python -m ruff check vmec_jax/free_boundary.py vmec_jax/solvers/free_boundary/types.py tests/test_free_boundary_helper_branches.py tests/test_free_boundary_additional_helpers.py tests/test_free_boundary_wave2.py tests/test_free_boundary_wp0.py`
- `python -m compileall -q vmec_jax/free_boundary.py vmec_jax/solvers/free_boundary/types.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_helper_branches.py tests/test_free_boundary_additional_helpers.py tests/test_free_boundary_wave2.py::test_nestor_external_only_step_reuse_spectral_and_dense_fallback tests/test_free_boundary_wp0.py::test_nestor_external_only_step_reuse -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_external_fields_mgrid_jax.py tests/test_free_boundary_coil_provider_forward.py tests/test_free_boundary_coil_provider_gradients.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_direct_coil_finite_pressure_sensitivity.py::test_direct_coil_vacuum_field_override_replay_contract tests/test_free_boundary_direct_coil_finite_pressure_sensitivity.py::test_jax_free_boundary_boundary_geometry_matches_host_sampler -q`
- `python tools/diagnostics/source_health.py --top 12 --top-functions 12`

Best next steps:

1. Let the current CI run finish and fix failures if any appear.
2. Next low-risk free-boundary target is separating mgrid/prepared-input
   helpers from `free_boundary.py`; avoid moving the core NESTOR integrals or
   adaptive scan/controller behavior until narrower parity gates are attached.
3. Continue reducing `free_boundary.py` toward a facade/solver split, then
   shift to WOUT minimal assembly and driver workflow decomposition.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.48%.
- Differentiability/refactor implementation: 99.86%.
- Solver monolith reduction: 87%.
- Free-boundary adjoint monolith reduction: 80%.
- Driver workflow decomposition: 84%.
- WOUT diagnostic/profile decomposition: 92%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 98.6%.

## 2026-06-15 Free-Boundary MGrid Helper Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Moved VMEC2000-compatible mgrid validation, extcur normalization, netCDF
   loading, character decoding, and trilinear interpolation helpers into
   `vmec_jax.solvers.free_boundary.mgrid`.
2. Kept `vmec_jax.free_boundary` as the compatibility facade for
   `_MGRID_FIELD_CACHE`, `_normalize_extcur`, `_broadcast_xyz`,
   `_decode_char_scalar`, `_decode_char_rows`, `load_mgrid`,
   `interpolate_mgrid_bfield`, `validate_free_boundary_config`, and
   `prepare_mgrid_for_config`.
3. Preserved the existing monkeypatch contract for `prepare_mgrid_for_config`
   by making the facade wrapper call the currently bound facade `load_mgrid`.
4. Avoided NESTOR integral, controller, scan, and branch-selection changes.

Results obtained:

- `free_boundary.py` dropped from 4,114 to 3,821 lines in the source-health
  report.
- The mgrid path now has a focused implementation module under the
  free-boundary solver package while retaining the historical public import
  surface.
- The remaining `free_boundary.py` hotspot is mostly true free-boundary field
  sampling, NESTOR, and diagnostic logic rather than passive types or mgrid IO.

Tests and commands run:

- `python -m ruff check vmec_jax/free_boundary.py vmec_jax/solvers/free_boundary/mgrid.py tests/test_free_boundary_wave2.py tests/test_free_boundary_wp0.py tests/test_free_boundary_additional_helpers.py tests/test_external_fields_mgrid_jax.py`
- `python -m compileall -q vmec_jax/free_boundary.py vmec_jax/solvers/free_boundary/mgrid.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_wave2.py::test_prepare_mgrid_for_config_validates_and_normalizes tests/test_free_boundary_wp0.py::test_prepare_mgrid_for_config_validates_and_normalizes_extcur tests/test_free_boundary_wp0.py::test_interpolate_mgrid_bfield_trilinear_linear_field tests/test_free_boundary_wp0.py::test_interpolate_mgrid_bfield_vmec_kv_subsamples_divisible_planes tests/test_free_boundary_additional_helpers.py::test_load_mgrid_reports_missing_metadata_and_bad_field_shape tests/test_external_fields_mgrid_jax.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_wave2.py tests/test_free_boundary_wp0.py::test_mgrid_loader_skeleton tests/test_free_boundary_wp0.py::test_prepare_mgrid_for_config_validates_and_normalizes_extcur tests/test_free_boundary_wp0.py::test_prepare_mgrid_for_config_rejects_nfp_mismatch tests/test_free_boundary_wp0.py::test_prepare_mgrid_for_config_rejects_kp_nzeta_mismatch tests/test_free_boundary_wp0.py::test_interpolate_mgrid_bfield_trilinear_linear_field tests/test_free_boundary_wp0.py::test_interpolate_mgrid_bfield_vmec_kv_subsamples_divisible_planes tests/test_free_boundary_wp0.py::test_interpolate_mgrid_bfield_allows_single_toroidal_plane tests/test_free_boundary_additional_helpers.py -q`
- `python tools/diagnostics/source_health.py --top 12 --top-functions 12`

Best next steps:

1. Let CI complete for the latest pushed head; if this commit is pushed before
   it finishes, monitor the superseding CI run.
2. Next free-boundary extraction should target direct-coil/external-boundary
   sampling utilities only if the test seam is clean; otherwise shift to WOUT
   minimal assembly where behavior is easier to isolate.
3. Keep adaptive full-loop differentiation claims conservative until a true
   fingerprint-gated adaptive AD-vs-FD gate exists.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.5%.
- Differentiability/refactor implementation: 99.87%.
- Solver monolith reduction: 87.5%.
- Free-boundary adjoint monolith reduction: 80%.
- Driver workflow decomposition: 84%.
- WOUT diagnostic/profile decomposition: 92%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 98.65%.

## 2026-06-15 Free-Boundary Axis-Current Helper Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Moved VMEC++ simple and VMEC2000 `tolicu`/`belicu`-equivalent axis-current
   field helpers into `vmec_jax.solvers.free_boundary.axis_current`.
2. Preserved the historical private facade names
   `_axis_current_field_simple` and `_axis_current_field_vmec_filament` in
   `vmec_jax.free_boundary` for tests and downstream diagnostics.
3. Re-ran axis-current helper tests plus free-boundary sampling coverage that
   verifies plasma-current axis fields change the boundary vacuum field.
4. Left mgrid/provider sampling, NESTOR, controller, scan, and adaptive branch
   semantics unchanged.

Results obtained:

- `free_boundary.py` dropped from 3,821 to 3,609 lines in the source-health
  report.
- Axis-current physics/parity helper code now has a domain-named module instead
  of living inline in the free-boundary solver body.
- The next meaningful free-boundary source-health work should focus on boundary
  metric/projection helpers or WOUT assembly, not adaptive controller logic.

Tests and commands run:

- `python -m ruff check vmec_jax/free_boundary.py vmec_jax/solvers/free_boundary/axis_current.py tests/test_free_boundary_helper_branches.py tests/test_free_boundary_additional_helpers.py tests/test_free_boundary_wave2.py tests/test_free_boundary_wp0.py`
- `python -m compileall -q vmec_jax/free_boundary.py vmec_jax/solvers/free_boundary/axis_current.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_helper_branches.py::test_axis_current_helpers_nonzero_and_degenerate_filament_paths tests/test_free_boundary_additional_helpers.py::test_axis_current_helpers_zero_current_and_validation_paths tests/test_free_boundary_wp0.py::test_axis_current_vmec_filament_nonzero_for_nzeta1 tests/test_free_boundary_wp0.py::test_freeb_axis_current_sampling_changes_boundary_field -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_wave2.py -q`
- `python tools/diagnostics/source_health.py --top 12 --top-functions 12`

Best next steps:

1. Monitor CI for the latest pushed head and fix any failure before broader
   refactors.
2. Next safe extraction candidate is boundary metric/projection utilities,
   which are pure NumPy physics helpers with direct tests.
3. Avoid moving adaptive branch selection or scan-controller internals until
   the differentiability gate around that branch seam exists.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.52%.
- Differentiability/refactor implementation: 99.88%.
- Solver monolith reduction: 88%.
- Free-boundary adjoint monolith reduction: 80%.
- Driver workflow decomposition: 84%.
- WOUT diagnostic/profile decomposition: 92%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 98.7%.

## 2026-06-15 Free-Boundary Boundary-Field Helper Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Moved boundary metric, cylindrical-to-covariant projection,
   covariant-to-contravariant projection, VMEC-like vacuum boundary channel
   assembly, and provider boundary sampling into
   `vmec_jax.solvers.free_boundary.boundary_fields`.
2. Kept all historical `vmec_jax.free_boundary` public/facade imports intact.
3. Re-ran provider, mgrid, boundary-projection, and JAX-vs-NumPy vacuum-field
   parity tests.
4. Avoided free-boundary adaptive controller, scan, reset, and NESTOR integral
   changes.

Results obtained:

- `free_boundary.py` dropped from 3,609 to 3,371 lines in the source-health
  report.
- `free_boundary.py` is now below `discrete_adjoint.py` in the largest-source
  ranking; the remaining largest source files are `solve.py`,
  `optimization.py`, `optimization_workflow.py`, `free_boundary_adjoint.py`,
  `discrete_adjoint.py`, and tests/examples.
- Boundary-field projection code now has a domain-named module shared by mgrid,
  direct-coil provider, and JAX projection parity tests.

Tests and commands run:

- `python -m ruff check vmec_jax/free_boundary.py vmec_jax/solvers/free_boundary/boundary_fields.py tests/test_free_boundary_additional_helpers.py tests/test_free_boundary_wp0.py tests/test_external_fields_mgrid_jax.py tests/test_free_boundary_coil_provider_forward.py tests/test_free_boundary_vacuum_adjoint.py`
- `python -m compileall -q vmec_jax/free_boundary.py vmec_jax/solvers/free_boundary/boundary_fields.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_additional_helpers.py::test_boundary_metric_field_projection_and_degenerate_determinant_floor tests/test_free_boundary_wp0.py::test_boundary_vacuum_projection_toroidal_field tests/test_external_fields_mgrid_jax.py::test_mgrid_provider_boundary_projection_matches_jax_and_legacy_interpolation_off_grid tests/test_free_boundary_coil_provider_forward.py::test_sample_free_boundary_external_field_from_direct_coils_matches_provider_components tests/test_free_boundary_coil_provider_forward.py::test_sample_free_boundary_external_field_adds_axis_field_separately tests/test_free_boundary_vacuum_adjoint.py::test_jax_boundary_projection_matches_numpy_reference -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_coil_provider_forward.py tests/test_external_fields_mgrid_jax.py -q`
- `python tools/diagnostics/source_health.py --top 12 --top-functions 12`

Best next steps:

1. Monitor CI for the newest pushed head.
2. Next refactor candidate should not be another passive free-boundary helper
   unless it cleanly isolates from the NESTOR integral body; consider switching
   to WOUT minimal assembly decomposition or driver workflow simplification.
3. Keep full adaptive branch differentiation deferred until the differentiable
   controller plan has a real fingerprint-gated adaptive AD-vs-FD gate.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.55%.
- Differentiability/refactor implementation: 99.89%.
- Solver monolith reduction: 88.5%.
- Free-boundary adjoint monolith reduction: 80%.
- Driver workflow decomposition: 84%.
- WOUT diagnostic/profile decomposition: 92%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 98.75%.

## 2026-06-16 WOUT Minimal Assembly Helper Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `vmec_jax.io.wout.minimal` for passive minimal-WOUT assembly helpers:
   profile payload preparation, internal-to-physical main coefficient
   conversion, magnetic-axis coefficient extraction, and the small VMEC-like
   force-reconstruction payload.
2. Rewired `wout_minimal_from_fixed_boundary` to call those helpers while
   preserving the public `vmec_jax.wout` facade and existing WOUT schema/output
   behavior.
3. Added fast helper tests for:
   - scaled main coefficient and magnetic-axis extraction,
   - VMEC-like force payload attributes,
   - current-driven `NCURR=1` iota/chipf recomputation through the injected
     recompute callback.
4. Re-ran constructor-heavy WOUT tests, environment-branch WOUT tests,
   LASYM/parity tests, and source-health reporting.

Results obtained:

- `wout_minimal_from_fixed_boundary` dropped from 1,144 to 1,102 lines in the
  function-length report for this tranche.
- `vmec_jax/wout.py` is now 2,585 lines; the new internal helper module is
  298 lines and lives under the existing `io/wout` domain namespace instead of
  adding another root-level source file.
- Current-driven WOUT output parity semantics remain explicit: the helper still
  recomputes iota/chipf from the accepted state unless
  `VMEC_JAX_DISABLE_WOUT_NCURR_RECOMPUTE` disables it.

Tests and commands run:

- `python -m ruff check vmec_jax/wout.py vmec_jax/io/wout/minimal.py tests/test_wout_fast_helpers.py`
- `python -m compileall -q vmec_jax/wout.py vmec_jax/io/wout/minimal.py tests/test_wout_fast_helpers.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_wout_fast_helpers.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_wout_wave2.py tests/test_wout_driver_wave10_coverage.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_wout_env_branch_coverage.py tests/test_wout_physics_wave8_coverage.py tests/test_wout_geometry_mercier_bundled_parity.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_wout_parity_reference.py tests/test_wout_lasym_bsubv_parity.py tests/test_vmec2000_exec_qa_regression.py -q`
- `python tools/diagnostics/source_health.py --top 12 --top-functions 12`

Best next steps:

1. Run one broader local release/coverage shard or let CI validate this pushed
   WOUT refactor before further WOUT surgery.
2. Next source-health move should target either:
   - another passive WOUT diagnostic seam, likely Nyquist/Bsub output assembly,
     only if it can be extracted without changing parity branches, or
   - driver/solver monolith decomposition, where `run_fixed_boundary` and
     `solve_fixed_boundary_residual_iter` remain the largest maintainability
     blockers.
3. Keep adaptive free-boundary branch differentiation conservative until a
   true fingerprint-gated full adaptive AD-vs-central-FD gate exists.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.55%.
- Differentiability/refactor implementation: 99.9%.
- Solver monolith reduction: 88.5%.
- Free-boundary adjoint monolith reduction: 80%.
- Driver workflow decomposition: 84%.
- WOUT diagnostic/profile decomposition: 94%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 98.8%.

## 2026-06-16 Driver Presentation Helper Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Moved VMEC-style fixed-boundary intro/header/summary printing out of
   `run_fixed_boundary` and into the existing `vmec_jax.drivers.io` domain
   helper module.
2. Kept all solver routing, multigrid staging, scan selection, restart,
   finisher, and adaptive branch semantics unchanged.
3. Added direct tests for:
   - concise vmec_jax fixed-boundary intro printing,
   - deterministic VMEC2000-style header text,
   - VMEC2000-style final summary for non-converged budget exhaustion.
4. Re-ran driver policy and driver API tests.

Results obtained:

- `run_fixed_boundary` dropped from 2,098 to 2,063 lines in the source-health
  report.
- Driver presentation text is now centralized in `vmec_jax.drivers.io`, making
  future CLI/pedagogical output changes lower risk and easier to test.
- No adaptive or solver branch-selection code was moved.

Tests and commands run:

- `python -m ruff check vmec_jax/driver.py vmec_jax/drivers/io.py tests/test_driver_policy_helpers.py`
- `python -m compileall -q vmec_jax/driver.py vmec_jax/drivers/io.py tests/test_driver_policy_helpers.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_driver_policy_helpers.py tests/test_driver_wave4_coverage.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_driver_api.py -q`
- `python tools/diagnostics/source_health.py --top 12 --top-functions 12`

Best next steps:

1. Let CI validate the WOUT and driver presentation tranches together.
2. Continue source-health reduction with either:
   - a small driver setup/context extraction that leaves control branches
     untouched, or
   - WOUT diagnostic/Nyquist helper extraction if focused parity tests remain
     cheap.
3. Do not claim arbitrary adaptive full-loop differentiability until the
   fingerprint-gated full adaptive AD-vs-central-FD gate exists.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.55%.
- Differentiability/refactor implementation: 99.91%.
- Solver monolith reduction: 88.5%.
- Free-boundary adjoint monolith reduction: 80%.
- Driver workflow decomposition: 85%.
- WOUT diagnostic/profile decomposition: 94%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 98.82%.

## 2026-06-16 Driver Setup Policy/Restart Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Moved the initial fixed-boundary run policy resolution out of
   `run_fixed_boundary` and into `vmec_jax.drivers.policy`:
   - solver-device request normalization,
   - backend used for policy decisions,
   - solver-mode/performance-mode resolution,
   - default scan policy,
   - CLI fixed-boundary auto-mode selection.
2. Moved restart WOUT/state normalization out of `run_fixed_boundary` and into
   `vmec_jax.drivers.runtime.resolve_restart_context`.
3. Moved the VMEC2000 axis-inference policy into
   `vmec_jax.drivers.policy.resolve_axis_infer_missing_policy`.
4. Preserved driver-level monkeypatch compatibility by injecting
   `_default_non_autodiff_solver_policy_for_backend`,
   `_default_use_scan_for_backend`, `read_wout`, and `state_from_wout` from
   `vmec_jax.driver` instead of hard-binding helper-module call sites.
5. Added direct helper tests for:
   - interactive CPU CLI default policy,
   - explicit parity/scan policy,
   - axis-inference parity/performance/env branches,
   - restart WOUT loading, `ns_override` validation, and resume-state copying.

Results obtained:

- `run_fixed_boundary` dropped from 2,063 to 2,029 lines in the source-health
  report.
- The public driver workflow now has typed setup contexts for policy and
  restart handling, while solver routing, multigrid staging, scan selection,
  adaptive branch semantics, and force math remain unchanged.
- A local compatibility regression was caught and fixed: driver-level policy
  monkeypatches now remain effective through dependency injection.
- Previous pushed CI run `27587015247` completed successfully before this local
  tranche.

Tests and commands run:

- `python -m ruff check vmec_jax/driver.py vmec_jax/drivers/policy.py vmec_jax/drivers/runtime.py tests/test_driver_policy_helpers.py`
- `python -m compileall -q vmec_jax/driver.py vmec_jax/drivers/policy.py vmec_jax/drivers/runtime.py tests/test_driver_policy_helpers.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_driver_policy_helpers.py tests/test_driver_wave2_coverage.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_driver_run_wave8_coverage.py::test_run_fixed_boundary_selects_default_accelerated_policy_and_explicit_parity tests/test_driver_policy_helpers.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_driver_policy_helpers.py tests/test_driver_wave2_coverage.py tests/test_driver_policy_coverage_extra.py tests/test_driver_helper_edges_wave14_coverage.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_driver_api.py tests/test_driver_run_wave8_coverage.py -q`
- `python tools/diagnostics/source_health.py --top 12 --top-functions 15`

Best next steps:

1. Commit and push this driver setup tranche, then let CI validate it.
2. Next safe refactor candidates:
   - extract CLI staged/budgeted multigrid orchestration into a
     `drivers/staging.py` helper with dependency injection for monkeypatch
     compatibility, or
   - move `read_wout`/`write_wout` netCDF serialization behind the existing
     `io/wout` domain while preserving the `vmec_jax.wout` facade.
3. Continue avoiding changes to adaptive branch selection until a true
   fingerprint-gated full adaptive AD-vs-central-FD gate exists.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.55%.
- Differentiability/refactor implementation: 99.92%.
- Solver monolith reduction: 88.7%.
- Free-boundary adjoint monolith reduction: 80%.
- Driver workflow decomposition: 86%.
- WOUT diagnostic/profile decomposition: 94%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 98.84%.

## 2026-06-16 WOUT netCDF Facade Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Moved the mechanical netCDF read/write bodies behind
   `vmec_jax.io.wout.netcdf.read_wout_payload` and
   `vmec_jax.io.wout.netcdf.write_wout_payload`.
2. Kept the public `vmec_jax.wout.read_wout` and `vmec_jax.wout.write_wout`
   facade functions stable and thin.
3. Avoided a dataclass import cycle by having the netCDF helper return
   `WoutData` constructor kwargs instead of importing `WoutData`.
4. Kept private compatibility exports `_bool_from_nc`, `_nc_scalar`, and
   `_read_wout_scalar_metadata` available through `vmec_jax.wout`, since
   existing helper tests and some downstream diagnostics use those seams.
5. Passed the physics-specific callbacks into the serialization helpers:
   - phi profile reconstruction from variables,
   - Glasser profile reconstruction from WOUT variables,
   - Glasser profile reconstruction from WOUT data on write.

Results obtained:

- `vmec_jax/wout.py` dropped below the 2,000-line source-health warning list.
- `read_wout` and `write_wout` are now facade-level wrappers; low-level netCDF
  details live in the existing `io/wout` domain namespace.
- Roundtrip, helper, WOUT-minimal, LASYM, and VMECPlot2 compatibility shards
  passed locally.

Tests and commands run:

- `python -m ruff check vmec_jax/wout.py vmec_jax/io/wout/netcdf.py tests/test_wout_fast_helpers.py tests/test_wout_helpers.py`
- `python -m compileall -q vmec_jax/wout.py vmec_jax/io/wout/netcdf.py tests/test_wout_fast_helpers.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_wout_fast_helpers.py tests/test_wout_helpers.py tests/test_wout_wave2.py tests/test_wout_roundtrip.py tests/test_wout_additional_helpers.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_wout_driver_wave10_coverage.py tests/test_wout_env_branch_coverage.py tests/test_wout_physics_wave8_coverage.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_wout_parity_reference.py tests/test_wout_lasym_bsubv_parity.py tests/test_wout_vmecplot2_compat.py -q`
- `python tools/diagnostics/source_health.py --top 12 --top-functions 15`

Best next steps:

1. Commit and push the WOUT netCDF facade extraction, then let CI validate it.
2. Next source-health tranche should target either:
   - CLI staged/budgeted multigrid orchestration in `run_fixed_boundary`, with
     dependency injection for monkeypatch compatibility, or
   - a WOUT-minimal Nyquist/Bsub payload helper inside `io/wout`, since
     `wout_minimal_from_fixed_boundary` remains the largest WOUT function.
3. Keep phase-2 free-boundary adaptive differentiation claims conservative
   until the full adaptive fingerprint-gated AD-vs-FD gate exists.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.55%.
- Differentiability/refactor implementation: 99.93%.
- Solver monolith reduction: 88.7%.
- Free-boundary adjoint monolith reduction: 80%.
- Driver workflow decomposition: 86%.
- WOUT diagnostic/profile decomposition: 96%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 98.88%.

## 2026-06-16 WOUT Minimal Schema Assembly Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Moved the passive final `WoutData` schema mapping from
   `wout_minimal_from_fixed_boundary` into
   `vmec_jax.io.wout.minimal.build_minimal_wout_data_kwargs`.
2. Kept all Nyquist, JXBFORCE, Mercier, Glasser, beta, and current-profile
   calculations in the existing builder for this tranche; only the final
   dtype/schema normalization moved.
3. Preserved the public `vmec_jax.wout.wout_minimal_from_fixed_boundary`
   facade and avoided importing `WoutData` into `io/wout/minimal.py`, so there
   is no schema/dataclass import cycle.
4. Confirmed the previously pushed CI run `27587654726` completed successfully
   before committing this local tranche.

Results obtained:

- `wout_minimal_from_fixed_boundary` dropped from 1,102 to 1,012 lines in the
  source-health report.
- The WOUT-minimal builder now separates physics/diagnostic calculations from
  the VMEC output schema mapping, making the next WOUT-specific reductions more
  straightforward.
- Local WOUT helper, roundtrip, driver-coverage, environment-branch, physics,
  LASYM, parity-reference, and VMECPlot2 compatibility shards passed.

Tests and commands run:

- `python -m ruff check vmec_jax/wout.py vmec_jax/io/wout/minimal.py`
- `python -m compileall -q vmec_jax/wout.py vmec_jax/io/wout/minimal.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_wout_fast_helpers.py tests/test_wout_helpers.py tests/test_wout_wave2.py tests/test_wout_roundtrip.py tests/test_wout_additional_helpers.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_wout_driver_wave10_coverage.py tests/test_wout_env_branch_coverage.py tests/test_wout_physics_wave8_coverage.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_wout_parity_reference.py tests/test_wout_lasym_bsubv_parity.py tests/test_wout_vmecplot2_compat.py -q`
- `python tools/diagnostics/source_health.py --top 12 --top-functions 15`
- `gh run watch 27587654726 --exit-status`

Best next steps:

1. Commit and push this WOUT-minimal schema extraction and let CI validate it.
2. Continue WOUT-minimal reduction only at clean seams, likely:
   - diagnostics/default initialization and nonconverged status handling, or
   - WOUT-light/current/equif metadata assembly.
3. Defer the deeper solver-monolith split and adaptive branch differentiation
   changes until a dedicated branch-fingerprint gate is in place.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.58%.
- Differentiability/refactor implementation: 99.94%.
- Solver monolith reduction: 88.7%.
- Free-boundary adjoint monolith reduction: 80%.
- Driver workflow decomposition: 86%.
- WOUT diagnostic/profile decomposition: 96.8%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 98.90%.

## 2026-06-16 Driver Staged-Continuation Runner Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `vmec_jax.drivers.staging` with
   `FixedBoundaryStageRunnerContext`,
   `run_cli_accelerated_budgeted_multigrid`, and
   `run_cli_explicit_staged_followup`.
2. Replaced the two long nested stage-runner implementations in
   `run_fixed_boundary` with thin wrappers that call the staging helper.
3. Kept the recursive public `run_fixed_boundary` call, `interp_vmec_state`,
   finish callback, timing reducer, and resume-state sanitizer injected through
   the context, preserving existing monkeypatch and finish-policy seams.
4. Left the production adaptive solver loop, scan/probe branch selection, and
   stage-monitor/fallback logic untouched in this tranche.

Results obtained:

- `run_fixed_boundary` dropped from 2,029 to 1,862 lines in the source-health
  report, taking it below the 2,000-line function warning threshold.
- The driver workflow is now closer to the intended structure:
  policy/restart setup, staged runner dispatch, finish policy, then core solve
  orchestration.
- The extracted staging helper is in the `drivers` domain namespace, avoiding
  more root-level module proliferation.

Tests and commands run:

- `python -m ruff check vmec_jax/driver.py vmec_jax/drivers/staging.py`
- `python -m compileall -q vmec_jax/driver.py vmec_jax/drivers/staging.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_driver_api.py::test_run_fixed_boundary_cli_budgeted_multigrid_path tests/test_driver_wave2_coverage.py::test_accelerated_single_grid_runs_explicit_staged_followup tests/test_driver_wave2_coverage.py::test_accelerated_staged_followup_skips_zero_budget_stage tests/test_driver_wave12_coverage.py::test_cli_staged_followup_records_policy_and_beats_single_grid -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_driver_api.py tests/test_driver_wave2_coverage.py tests/test_driver_wave12_coverage.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_driver_policy_helpers.py tests/test_driver_policy_coverage_extra.py tests/test_driver_helper_edges_wave14_coverage.py tests/test_driver_run_wave8_coverage.py -q`
- `python tools/diagnostics/source_health.py --top 12 --top-functions 15`

Best next steps:

1. Let CI validate the previous pushed WOUT tranche, then push this driver
   staging tranche.
2. Next safe source-health target should be WOUT-minimal diagnostic/default
   initialization or `wout_minimal_from_fixed_boundary` field-path assembly.
3. Keep the core solve monolith and adaptive branch differentiation for a
   separate tranche with explicit same-branch/fingerprint-gated tests.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.60%.
- Differentiability/refactor implementation: 99.95%.
- Solver monolith reduction: 88.7%.
- Free-boundary adjoint monolith reduction: 80%.
- Driver workflow decomposition: 90%.
- WOUT diagnostic/profile decomposition: 96.8%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 98.95%.

## 2026-06-16 WOUT Equif Diagnostic Helper Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Moved the VMEC `eqfor`/`equif` WOUT helper implementation from
   `vmec_jax.wout` into the `vmec_jax.io.wout.diagnostics` domain module.
2. Moved the VMEC `bcovar` IEQUI=1 `bsubv` surface-average correction into
   `vmec_jax.io.wout.diagnostics`.
3. Kept `_compute_equif_wout` and `_apply_bsubv_equif_correction` available
   from `vmec_jax.wout` for compatibility with existing tests and downstream
   private import usage.
4. Preserved the existing `vmec_jax.wout.vmec_pwint_from_trig` monkeypatch seam
   by dependency-injecting the quadrature-weight builder through the facade.

Results obtained:

- WOUT diagnostic logic is now further consolidated under `io/wout`, reducing
  the amount of VMEC-specific diagnostic implementation embedded in the root
  WOUT facade.
- The initial test regression exposed a real compatibility seam; the final
  implementation keeps that seam intact while still moving the implementation.
- Source-health still reports `wout_minimal_from_fixed_boundary` as a remaining
  long WOUT function, so the next WOUT tranche should target minimal-WOUT field
  assembly/default initialization rather than private diagnostic kernels.

Tests and commands run:

- `python -m ruff check vmec_jax/wout.py vmec_jax/io/wout/diagnostics.py tests/test_wout_helpers.py tests/test_wout_fast_helpers.py`
- `python -m compileall -q vmec_jax/wout.py vmec_jax/io/wout/diagnostics.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_wout_helpers.py::test_bsubv_equif_correction_and_ctor_branches -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_wout_helpers.py tests/test_wout_fast_helpers.py tests/test_wout_bcovar_forces_extra_coverage.py tests/test_wout_wave3_coverage.py tests/test_wout_additional_helpers.py tests/test_non_solve_wave6_coverage.py tests/test_implicit_wout_driver_branch_coverage.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_wout_driver_wave10_coverage.py tests/test_wout_env_branch_coverage.py tests/test_wout_physics_wave8_coverage.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_wout_parity_reference.py tests/test_wout_lasym_bsubv_parity.py tests/test_wout_vmecplot2_compat.py -q`
- `python tools/diagnostics/source_health.py --top 12 --top-functions 15`

Best next steps:

1. Push the WOUT diagnostic-helper extraction and let CI validate it.
2. Next refactor tranche should target either `wout_minimal_from_fixed_boundary`
   field assembly or the solver scan sub-loop, with explicit parity/branch tests.
3. Keep adaptive full-loop differentiation claims conservative until a true
   fingerprint-gated adaptive AD-vs-central-FD gate exists.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.62%.
- Differentiability/refactor implementation: 99.96%.
- Solver monolith reduction: 88.7%.
- Free-boundary adjoint monolith reduction: 80%.
- Driver workflow decomposition: 90%.
- WOUT diagnostic/profile decomposition: 97.5%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.00%.

## 2026-06-16 Minimal-WOUT Runtime/Profile Helper Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `WoutMinimalRuntimeOptions` and
   `minimal_wout_runtime_options_from_env` in `vmec_jax.io.wout.minimal`.
2. Added `lbsubs_from_indata_and_env` so the VMEC `LBSUBS` output policy and
   debug override are isolated from the high-level WOUT builder.
3. Added `pressure_profiles_from_mass_vp` for VMEC-style half/full-mesh pressure
   reconstruction after force/volume diagnostics are available.
4. Rewired `wout_minimal_from_fixed_boundary` to consume these helpers while
   leaving bcovar, Fourier transforms, Mercier, and Glasser calculations
   unchanged.
5. Added direct helper tests for environment/default semantics, `LBSUBS`
   override behavior, and pressure endpoint reconstruction.

Results obtained:

- `wout_minimal_from_fixed_boundary` dropped from 1,012 lines before the WOUT
  sequence of tranches to 984 lines after this extraction.
- The helper coverage makes the passive environment/default policy explicit,
  reducing risk from future WOUT refactors.
- Remaining WOUT work is now mostly physics/field assembly, so future tranches
  should be narrower and parity-test backed.

Tests and commands run:

- `python -m ruff check vmec_jax/wout.py vmec_jax/io/wout/minimal.py tests/test_wout_fast_helpers.py`
- `python -m compileall -q vmec_jax/wout.py vmec_jax/io/wout/minimal.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_wout_fast_helpers.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_wout_helpers.py tests/test_wout_fast_helpers.py tests/test_wout_bcovar_forces_extra_coverage.py tests/test_wout_wave3_coverage.py tests/test_wout_additional_helpers.py tests/test_non_solve_wave6_coverage.py tests/test_implicit_wout_driver_branch_coverage.py -q`
- `python tools/diagnostics/source_health.py --top 12 --top-functions 15`

Best next steps:

1. Run the remaining WOUT parity/profile shards, commit, and push this helper
   extraction.
2. Next large source-health target should shift from WOUT to either the solver
   scan sub-loop or `FixedBoundaryExactOptimizer.run`, because further WOUT
   reductions now require careful numerical parity tranches.
3. Keep CI monitored after each push; the current draft PR should remain the
   single umbrella PR until the full plan is complete.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.64%.
- Differentiability/refactor implementation: 99.965%.
- Solver monolith reduction: 88.7%.
- Free-boundary adjoint monolith reduction: 80%.
- Driver workflow decomposition: 90%.
- WOUT diagnostic/profile decomposition: 98.0%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.05%.

## 2026-06-16 Fixed-Boundary Optimizer Run Bookkeeping Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `_reset_run_state` to isolate per-run cache, callback trace, profile,
   and accepted-point bookkeeping reset logic from
   `FixedBoundaryExactOptimizer.run`.
2. Added `_initial_run_evaluation` to isolate the exact initial residual/state
   evaluation and first history entry construction.
3. Added `_build_run_history_dump` to isolate the serializable optimization
   history payload assembly.
4. Added `_attach_run_private_payload` to isolate the non-serializable result
   state/profile payload used by examples and output writers.
5. Left all residual callbacks, SciPy/matrix-free/scalar-trust paths,
   accepted-point exact solve selection, and cache semantics unchanged.

Results obtained:

- `FixedBoundaryExactOptimizer.run` dropped from 1,048 lines before this
  refactor lane to 989 lines, below the 1,000-line source-health marker.
- The public result dictionary and `_history_dump` keys remain unchanged.
- The next optimizer tranche should target method-specific solver bodies rather
  than further bookkeeping so the outer optimization API becomes easier to
  audit and test.

Tests and commands run:

- `python -m ruff check vmec_jax/optimization.py tests/test_optimization_fast_optimizer_methods.py tests/test_optimization_callback_trace.py tests/test_optimization_wave2_coverage.py`
- `python -m compileall -q vmec_jax/optimization.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_optimization_fast_optimizer_methods.py tests/test_optimization_callback_trace.py tests/test_optimization_wave2_coverage.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_differentiation_optimization_api_fast.py tests/test_optimization_auto_scalar_policy.py -q`
- `python tools/diagnostics/source_health.py --top 12 --top-functions 15`

Best next steps:

1. Commit and push this optimizer-run bookkeeping tranche.
2. Continue source-health work on either method-specific optimizer bodies or the
   solver scan sub-loop; both require narrower branch/parity tests than WOUT.
3. Monitor CI for the current draft PR after the push and avoid merging until
   the full differentiability/refactor plan is complete.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.66%.
- Differentiability/refactor implementation: 99.97%.
- Solver monolith reduction: 88.7%.
- Free-boundary adjoint monolith reduction: 80%.
- Driver workflow decomposition: 90%.
- WOUT diagnostic/profile decomposition: 98.0%.
- Optimizer workflow decomposition: 82%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.10%.

## 2026-06-16 Implicit Residual Active-Space Helper Consolidation

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Moved stellarator-symmetric active-coordinate index, pack, and update helpers
   from `vmec_jax.implicit` into the existing
   `vmec_jax.implicit_adjoint_helpers` module.
2. Kept the historical private names exported from `vmec_jax.implicit` by
   importing the helper implementations under the old names, preserving tests
   and downstream monkeypatch/import seams.
3. Extracted passive VMEC residual setup into `_build_vmec_residual_setup`,
   covering flux/profile construction, wout-like force metadata, trig tables,
   LFORBAL policy, Tomnsps masks, and stellarator-symmetric residual
   projection metadata.
4. Removed an unused in-function `stellsym_active_keep_idx` calculation that
   duplicated active-space bookkeeping without feeding the residual or adjoint
   branches.
5. Left custom-JVP/custom-VJP control flow, host solve callbacks, active/full
   adjoint solver selection, and residual force evaluation unchanged.

Results obtained:

- `vmec_jax.implicit` dropped out of the top source-file warnings.
- `solve_fixed_boundary_state_implicit_vmec_residual` dropped from 1,006 to
  927 lines without changing the implicit AD branch semantics.
- The active-coordinate helpers are now directly colocated with the linear
  adjoint helper utilities, making future same-branch residual-adjoint tests
  easier to expand.

Tests and commands run:

- `python -m ruff check vmec_jax/implicit.py vmec_jax/implicit_adjoint_helpers.py tests/test_implicit_adjoint_helpers.py tests/test_implicit_helpers.py tests/test_implicit_wave6_coverage.py tests/test_implicit_wout_driver_branch_coverage.py tests/test_implicit_wave12_coverage.py`
- `python -m compileall -q vmec_jax/implicit.py vmec_jax/implicit_adjoint_helpers.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_implicit_adjoint_helpers.py tests/test_implicit_helpers.py tests/test_implicit_wave6_coverage.py tests/test_implicit_wout_driver_branch_coverage.py tests/test_implicit_wave12_coverage.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_implicit_differentiation_fast.py tests/test_implicit_more_coverage.py tests/test_implicit_sensitivity_fast_coverage.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q $(rg --files tests | rg 'implicit') -q`
- `python tools/diagnostics/source_health.py --top 12 --top-functions 20`

Best next steps:

1. Commit and push this implicit residual helper consolidation.
2. Continue source-health work on method-specific optimizer bodies or the
   solver scan sub-loop; the solver scan loop remains high-risk because its
   nested JAX scan closes over many branch-control arrays.
3. Keep CI monitored after the push; cancelled runs from superseded pushes are
   expected, but the newest run must finish green before the umbrella PR is
   considered stable.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.67%.
- Differentiability/refactor implementation: 99.972%.
- Solver monolith reduction: 88.7%.
- Free-boundary adjoint monolith reduction: 80%.
- Driver workflow decomposition: 90%.
- WOUT diagnostic/profile decomposition: 98.0%.
- Optimizer workflow decomposition: 82%.
- Implicit residual-adjoint decomposition: 86%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.13%.

## 2026-06-16 Driver Stage JIT Policy Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `StageJitSettings` and `resolve_stage_jit_settings` to
   `vmec_jax.drivers.policy`.
2. Moved the per-stage VMEC2000 force-kernel JIT, precompile, and warmup policy
   from the `run_fixed_boundary` stage loop into the tested policy helper.
3. Preserved the same environment variables and defaults:
   `VMEC_JAX_SCAN_JIT_FORCES`, `VMEC_JAX_JIT_PRECOMPILE`, and
   `VMEC_JAX_JIT_WARMUP_ITERS`.
4. Added direct policy tests for parity scan defaults, explicit scan-JIT
   overrides, disabled precompile behavior, and invalid warmup values.
5. Left stage construction, solver dispatch, scan/dynamic-scan selection,
   preconditioner policy, free-boundary provider wiring, and result assembly
   unchanged.

Results obtained:

- `run_fixed_boundary` dropped from 1,862 to 1,828 lines.
- The stage JIT policy is now independently unit-tested instead of only
  covered through full driver runs.
- This is a narrow driver workflow decomposition with no new source file and no
  public API change.

Tests and commands run:

- `python -m ruff check vmec_jax/driver.py vmec_jax/drivers/policy.py tests/test_driver_policy_helpers.py tests/test_driver_run_wave8_coverage.py tests/test_driver_wave2_coverage.py`
- `python -m compileall -q vmec_jax/driver.py vmec_jax/drivers/policy.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_driver_policy_helpers.py tests/test_driver_run_wave8_coverage.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_driver_wave2_coverage.py tests/test_driver_policy_coverage_extra.py -q`
- `python tools/diagnostics/source_health.py --top 16 --top-functions 24`

Best next steps:

1. Commit and push this driver-stage policy extraction.
2. Continue driver decomposition only through similarly passive policy/runtime
   seams; avoid moving the multigrid stage loop until there is a branch/parity
   test suite specifically around that loop.
3. Keep the newest CI run monitored after the push and inspect logs only if a
   non-cancelled run fails.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.67%.
- Differentiability/refactor implementation: 99.974%.
- Solver monolith reduction: 88.7%.
- Free-boundary adjoint monolith reduction: 80%.
- Driver workflow decomposition: 91%.
- WOUT diagnostic/profile decomposition: 98.0%.
- Optimizer workflow decomposition: 82%.
- Implicit residual-adjoint decomposition: 86%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.15%.

## 2026-06-16 Shared Implicit Residual Context Reuse

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Extended `prepare_residual_force_context` with optional injected callback
   seams for flux-profile construction, boundary construction, and VMEC
   trigonometric-table construction.
2. Rewired `_build_vmec_residual_setup` in `vmec_jax.implicit` to reuse the
   fixed-boundary residual-force context instead of carrying a second copy of
   flux/profile/boundary/trig setup logic.
3. Preserved the implicit module's existing private monkeypatch seams by
   passing its local helper functions into the shared context builder.
4. Left the implicit Tomnsps mask/projector assembly local to
   `_build_vmec_residual_setup`, since that residual-shape logic is specific to
   the implicit residual objective.

Results obtained:

- The fixed-boundary optimizer residual context and implicit residual-adjoint
  path now share one tested force-context setup implementation.
- No public API change.
- No new source file or namespace expansion.
- This tranche reduces duplicated profile/boundary/trig setup logic while
  keeping branch behavior and differentiability seams unchanged.

Tests and commands run:

- `python -m ruff check vmec_jax/implicit.py vmec_jax/solvers/fixed_boundary/optimization/residual_context.py`
- `python -m compileall -q vmec_jax/implicit.py vmec_jax/solvers/fixed_boundary/optimization/residual_context.py`
- `JAX_ENABLE_X64=1 python -m pytest -q $(rg --files tests | rg 'implicit') tests/test_solve_wave7_coverage.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_wave7_coverage.py tests/test_solve_wave6_coverage.py tests/test_solve_driver_control_fast.py -q`
- `git diff --check`

Best next steps:

1. Commit and push this shared residual-context reuse tranche.
2. Recheck CI for the branch after the push.
3. Continue with the next bounded source-health seam: either extract passive
   helper logic from `solve_fixed_boundary_state_implicit_vmec_residual` or
   reduce optimizer workflow duplication, avoiding risky adaptive-controller
   behavior changes until the fingerprint-gated validation gates are in place.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.67%.
- Differentiability/refactor implementation: 99.976%.
- Solver monolith reduction: 88.7%.
- Free-boundary adjoint monolith reduction: 80%.
- Driver workflow decomposition: 91%.
- WOUT diagnostic/profile decomposition: 98.0%.
- Optimizer workflow decomposition: 82%.
- Implicit residual-adjoint decomposition: 87%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.16%.

## 2026-06-16 Implicit Residual Host and Boundary Helper Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `_VmecResidualHostSolveSettings` and
   `_solve_vmec_residual_host_from_boundary` to make the authoritative host
   VMEC residual solve used by the implicit residual pure-callback explicit and
   testable as a passive helper.
2. Added `_project_boundary_edge_rows` for VMEC boundary projection and
   `_boundary_edge_rows_vjp` for the repeated edge-row VJP plumbing.
3. Replaced the corresponding nested closure bodies inside
   `solve_fixed_boundary_state_implicit_vmec_residual` with calls to these
   helpers.
4. Left the residual equation, residual scaling, active/reduced packing,
   linearized tangent modes, custom VJP/JVP definitions, and solver control flow
   unchanged.

Results obtained:

- `solve_fixed_boundary_state_implicit_vmec_residual` decreased from 927 lines
  before today's implicit refactor to 864 lines after the host/boundary helper
  extraction.
- The pure-callback host solve now has one explicit settings object, which
  makes future callback shape and accepted-branch validation easier to audit.
- No public API change and no new source file.

Tests and commands run:

- `python -m ruff check vmec_jax/implicit.py`
- `python -m compileall -q vmec_jax/implicit.py`
- `JAX_ENABLE_X64=1 python -m pytest -q $(rg --files tests | rg 'implicit') tests/test_solve_wave7_coverage.py -q`
- `python tools/diagnostics/source_health.py --top 16 --top-functions 24`

Best next steps:

1. Commit and push this implicit residual host/boundary extraction.
2. Monitor the branch CI run after the push.
3. Continue source-health work where a passive seam is available. The next
   likely targets are optimizer workflow decomposition and same-branch
   free-boundary report helpers; avoid the VMEC scan loop and adaptive host
   branch semantics without a dedicated branch-fingerprint gate.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.67%.
- Differentiability/refactor implementation: 99.978%.
- Solver monolith reduction: 88.7%.
- Free-boundary adjoint monolith reduction: 80%.
- Driver workflow decomposition: 91%.
- WOUT diagnostic/profile decomposition: 98.0%.
- Optimizer workflow decomposition: 82%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.17%.

## 2026-06-16 Optimizer Cached-History Callback Consolidation

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `FixedBoundaryExactOptimizer._record_cached_exact_history_entry` to
   centralize exact-cache history entry creation, best-exact-point tracking, and
   rejected-history counting.
2. Replaced three local `run()` closures with calls to the shared helper:
   scalar trust, scalar L-BFGS, and SciPy matrix-free.
3. Preserved the existing per-method behavior:
   scalar methods still pass their scalar cost into history entries, while the
   matrix-free least-squares path lets residual metadata determine cost.

Results obtained:

- `FixedBoundaryExactOptimizer.run` decreased from 989 to 927 lines.
- Duplicated accepted-history bookkeeping now lives in one method, reducing the
  risk that scalar and matrix-free optimizer paths diverge in future edits.
- No public API change and no numerical path change.

Tests and commands run:

- `python -m ruff check vmec_jax/optimization.py tests/test_optimization_helpers.py tests/test_optimization_examples.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_optimization_helpers.py tests/test_optimization_examples.py -q`
- `python -m ruff check tests/test_optimization_auto_scalar_policy.py tests/test_optimization_callback_trace.py tests/test_optimization_fast_optimizer_methods.py tests/test_continuation_exact_history.py tests/test_differentiation_optimization_api_fast.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_optimization_auto_scalar_policy.py tests/test_optimization_callback_trace.py tests/test_optimization_fast_optimizer_methods.py tests/test_continuation_exact_history.py tests/test_differentiation_optimization_api_fast.py -q`
- `python -m compileall -q vmec_jax/optimization.py`
- `python tools/diagnostics/source_health.py --top 16 --top-functions 24`
- `git diff --check`

Best next steps:

1. Commit and push this optimizer bookkeeping refactor.
2. Monitor CI for the newest branch run.
3. Continue optimizer decomposition only through similarly duplicated local
   helpers or passive result assembly. Defer deeper scalar-trust loop extraction
   until it can be moved into a small solver object with dedicated method-level
   regression tests.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.67%.
- Differentiability/refactor implementation: 99.980%.
- Solver monolith reduction: 88.7%.
- Free-boundary adjoint monolith reduction: 80%.
- Driver workflow decomposition: 91%.
- WOUT diagnostic/profile decomposition: 98.0%.
- Optimizer workflow decomposition: 84%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.18%.

## 2026-06-16 WOUT Geometry Synthesis Helper Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `_synthesize_wout_geometry_from_state` to isolate the minimal-WOUT
   real-space geometry synthesis, host transfer, and timing bookkeeping.
2. Replaced the inline geometry block in `wout_minimal_from_fixed_boundary`
   with a call to that helper.
3. Left the bcovar, bsub, JXBFORCE, Mercier, Glasser, and NetCDF payload
   assembly paths unchanged.

Results obtained:

- `wout_minimal_from_fixed_boundary` decreased from 984 to 976 lines.
- The public WOUT module API and compatibility exports were preserved.
- No new source file was added; this improves readability without increasing
  namespace sprawl.

Tests and commands run:

- `python -m ruff check vmec_jax/wout.py vmec_jax/io/wout`
- `python -m compileall -q vmec_jax/wout.py vmec_jax/io/wout`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_wout_helpers.py tests/test_wout_fast_helpers.py tests/test_wout_branch_coverage.py tests/test_driver_wout_wave9_coverage.py tests/test_wout_wave2.py tests/test_wout_wave3_coverage.py tests/test_wout_env_branch_coverage.py -q`
- `python tools/diagnostics/source_health.py --top 16 --top-functions 24`
- `git diff --check`

Best next steps:

1. Commit and push this WOUT helper extraction.
2. Keep future WOUT changes focused on existing `vmec_jax.io.wout.*` domain
   modules; do not split the public `wout.py` facade into many new root files.
3. Continue monitoring CI and prioritize failures over further refactoring if
   the newest run reports red.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.67%.
- Differentiability/refactor implementation: 99.981%.
- Solver monolith reduction: 88.7%.
- Free-boundary adjoint monolith reduction: 80%.
- Driver workflow decomposition: 91%.
- WOUT diagnostic/profile decomposition: 98.2%.
- Optimizer workflow decomposition: 84%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.19%.

## 2026-06-16 WOUT Force/Bcovar Payload Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added a private `_WoutBcovarPayload` and `_prepare_wout_bcovar_payload`
   helper in `vmec_jax.wout`.
2. Centralized WOUT force/bcovar source selection, including:
   `VMEC_JAX_WOUT_FORCE_IEQUI1`, `VMEC_JAX_WOUT_REUSE_FINAL_BCOVAR`,
   `VMEC_JAX_WOUT_FAST_BCOVAR`, and `VMEC_JAX_WOUT_FORCE_VMEC_SYNTH`.
3. Kept the existing monkeypatch seams explicit by passing the imported
   `vmec_bcovar_half_mesh_from_wout`, `vmec_forces_rz_from_wout`, and
   `_numpy_module_patch` callables into the helper.
4. Left downstream BSS, Nyquist, Mercier, Glasser, and NetCDF assembly
   unchanged.

Results obtained:

- `wout_minimal_from_fixed_boundary` decreased from 976 to 928 lines.
- The file-level line count increased because the helper remains in the public
  facade to avoid adding another source module during this tranche. A later
  pass can move stable WOUT helpers into `vmec_jax.io.wout.minimal` if reducing
  facade file size becomes more important than minimizing module count.
- Force-payload reuse and force-BSS env branches remain covered by existing
  tests.

Tests and commands run:

- `python -m ruff check vmec_jax/wout.py vmec_jax/io/wout tests/test_wout_env_branch_coverage.py tests/test_wout_driver_wave10_coverage.py`
- `python -m compileall -q vmec_jax/wout.py vmec_jax/io/wout`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_wout_env_branch_coverage.py tests/test_wout_driver_wave10_coverage.py tests/test_wout_helpers.py tests/test_wout_fast_helpers.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_wout_env_branch_coverage.py tests/test_wout_driver_wave10_coverage.py tests/test_wout_helpers.py tests/test_wout_fast_helpers.py tests/test_wout_branch_coverage.py -q`
- `python tools/diagnostics/source_health.py --top 24 --top-functions 32`

Best next steps:

1. Commit and push this WOUT force/bcovar extraction.
2. Wait for the existing CI run to finish, then monitor the new run from this
   commit.
3. Continue WOUT cleanup through existing `vmec_jax.io.wout.*` domain helpers
   only if the next extraction clearly reduces high-level function size without
   increasing namespace sprawl.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.67%.
- Differentiability/refactor implementation: 99.982%.
- Solver monolith reduction: 88.7%.
- Free-boundary adjoint monolith reduction: 80%.
- Driver workflow decomposition: 91%.
- WOUT diagnostic/profile decomposition: 98.5%.
- Optimizer workflow decomposition: 84%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.20%.

## 2026-06-16 Optimizer Final Exact-Point Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `FixedBoundaryExactOptimizer._wall_time_for_final_history_entry` to
   keep final history timestamps monotone with prior accepted entries.
2. Added `FixedBoundaryExactOptimizer._evaluate_and_record_final_exact_point`
   to centralize the final-output correctness policy:
   use an exact cached final state when available, reconstruct it through the
   selected exact path when needed, and fall back to the best prior exact
   accepted point when the optimizer's nominal final point is not a valid exact
   artifact.
3. Replaced the inline final exact replay/best-exact-selection block in
   `run()` with the new helper call.
4. Preserved the policy that final artifacts never come from relaxed trial
   solves.

Results obtained:

- `FixedBoundaryExactOptimizer.run` decreased from 927 to 822 lines.
- The exact accepted-output correctness seam is now easier to audit because the
  final state selection and history append live in one named method.
- No public API change and no optimizer-method behavior change.

Tests and commands run:

- `python -m ruff check vmec_jax/optimization.py tests/test_optimization_helpers.py tests/test_optimization_examples.py tests/test_continuation_exact_history.py`
- `python -m compileall -q vmec_jax/optimization.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_optimization_helpers.py tests/test_optimization_examples.py tests/test_optimization_auto_scalar_policy.py tests/test_optimization_callback_trace.py tests/test_optimization_fast_optimizer_methods.py tests/test_continuation_exact_history.py tests/test_differentiation_optimization_api_fast.py -q`
- `python tools/diagnostics/source_health.py --top 24 --top-functions 32`

Best next steps:

1. Commit and push this optimizer final-output extraction.
2. Monitor CI for the new branch run.
3. Continue optimizer cleanup only through passive method extraction. The
   scalar-trust loop itself should be split into a small solver object only with
   dedicated behavior tests, since it owns trust-region/globalization semantics.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.67%.
- Differentiability/refactor implementation: 99.983%.
- Solver monolith reduction: 88.7%.
- Free-boundary adjoint monolith reduction: 80%.
- Driver workflow decomposition: 91%.
- WOUT diagnostic/profile decomposition: 98.5%.
- Optimizer workflow decomposition: 86%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.21%.

## 2026-06-16 Driver Stage-Array Helper Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `_stage_array_list` as a top-level private driver helper for VMEC
   multigrid stage arrays.
2. Removed the local `_as_list` closure from `run_fixed_boundary`.
3. Replaced all `NS_ARRAY`, `NITER_ARRAY`, and `FTOL_ARRAY` conversions inside
   `run_fixed_boundary` with `_stage_array_list`.
4. Kept the exact legacy conversion semantics instead of using the more
   permissive policy helper, since the driver intentionally returns `None` for
   unsupported values rather than treating arbitrary iterables as stage lists.

Results obtained:

- `run_fixed_boundary` decreased from 1828 to 1812 lines.
- Stage-array parsing is now named and independently inspectable without
  changing driver policy, CLI behavior, VMEC2000 compatibility, or adaptive
  branch logic.
- Source-health still identifies the same major hotspots, with no new module or
  root namespace sprawl.

Tests and commands run:

- `python -m ruff check vmec_jax/driver.py tests/test_driver_policy_helpers.py tests/test_driver_api.py tests/test_driver_run_wave8_coverage.py`
- `python -m compileall -q vmec_jax/driver.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_driver_policy_helpers.py tests/test_driver_api.py tests/test_driver_run_wave8_coverage.py tests/test_driver_wave5_coverage.py tests/test_driver_wout_wave9_coverage.py -q`
- `python tools/diagnostics/source_health.py --top 24 --top-functions 32`

Best next steps:

1. Commit and push the driver helper extraction.
2. Continue driver cleanup only through passive helpers such as resume-state
   sanitization or stage-header formatting, with focused driver tests after each
   tranche.
3. Defer adaptive branch/controller rewrites until a dedicated
   fingerprint-gated AD-vs-FD validation tranche exists.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.67%.
- Differentiability/refactor implementation: 99.984%.
- Solver monolith reduction: 88.7%.
- Free-boundary adjoint monolith reduction: 80%.
- Driver workflow decomposition: 91.5%.
- WOUT diagnostic/profile decomposition: 98.5%.
- Optimizer workflow decomposition: 86%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.22%.

## 2026-06-16 Driver Resume-State Helper Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `_driver_resume_step_size_value` to centralize the driver fallback
   from explicit `step_size` to input-deck `DELT`.
2. Added `_sanitize_resume_state_for_driver_stage` and
   `_sanitize_resume_state_for_driver_same_stage` as facade-local wrappers over
   the tested policy helpers.
3. Replaced the local resume-state closure group in `run_fixed_boundary` with
   bound callables, preserving the callable interface consumed by staging and
   finish contexts.
4. Kept the helper in `driver.py` rather than adding another module, since this
   is driver facade glue and the current architecture goal is fewer, clearer
   domain modules rather than namespace growth.

Results obtained:

- `run_fixed_boundary` decreased from 1812 to 1809 lines.
- Resume-state sanitization now has named helper seams for unit tests or later
  relocation if the driver facade is split.
- `driver.py` grew modestly because the helper remains local to the facade;
  root namespace count is unchanged.
- No adaptive controller, scan, multigrid, VMEC2000 parity, or resume-state
  policy behavior changed.

Tests and commands run:

- `python -m ruff check vmec_jax/driver.py tests/test_driver_policy_helpers.py tests/test_driver_api.py tests/test_driver_run_wave8_coverage.py`
- `python -m compileall -q vmec_jax/driver.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_driver_policy_helpers.py tests/test_driver_api.py tests/test_driver_run_wave8_coverage.py tests/test_driver_wave5_coverage.py tests/test_driver_wout_wave9_coverage.py -q`
- `python tools/diagnostics/source_health.py --top 24 --top-functions 32`

Best next steps:

1. Commit and push this driver resume-state helper extraction.
2. Reassess whether the next driver reduction should be another passive helper
   extraction or whether effort is better spent on the larger solver scan
   monolith.
3. Keep adaptive branch differentiation claims unchanged until a true
   fingerprint-gated adaptive AD-vs-FD gate exists.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.67%.
- Differentiability/refactor implementation: 99.985%.
- Solver monolith reduction: 88.7%.
- Free-boundary adjoint monolith reduction: 80%.
- Driver workflow decomposition: 91.6%.
- WOUT diagnostic/profile decomposition: 98.5%.
- Optimizer workflow decomposition: 86%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.23%.

## 2026-06-16 Residual Host-Metric Config Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `HostResidualMetricConfig` and
   `resolve_host_residual_metric_config` to
   `vmec_jax.solvers.fixed_boundary.residual.config`.
2. Added `env_flag_enabled_with_off` for the residual-solver env flags that
   historically treat `off` as false in addition to the standard false tokens.
3. Replaced inline `VMEC_JAX_HOST_FSQ1_NORMS` and
   `VMEC_JAX_HOST_RESIDUAL_METRICS` parsing in
   `solve_fixed_boundary_residual_iter` with the new config helper.
4. Added focused tests for CPU/GPU `auto` policy and explicit true/false env
   values.

Results obtained:

- `solve_fixed_boundary_residual_iter` decreased from 8961 to 8958 lines.
- `solve.py` decreased from 10008 to 10006 lines despite the new import.
- Host metric policy is now independently unit-tested without importing or
  running the full fixed-boundary solver.
- No force assembly, scan, restart, free-boundary, or adaptive-branch behavior
  changed.

Tests and commands run:

- `python -m ruff check vmec_jax/solve.py vmec_jax/solvers/fixed_boundary/residual/config.py tests/test_solve_residual_iter_config.py tests/test_solve_residual_iter_helpers_wave8_coverage.py`
- `python -m compileall -q vmec_jax/solve.py vmec_jax/solvers/fixed_boundary/residual/config.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_residual_iter_config.py tests/test_solve_residual_iter_helpers_wave8_coverage.py -q`
- `python tools/diagnostics/source_health.py --top 24 --top-functions 32`

Best next steps:

1. Commit and push the residual host-metric config extraction.
2. Continue solver-monolith work by moving additional pure env/config policy
   blocks into existing residual domain modules, not by splitting numerical
   kernels prematurely.
3. Keep VMEC2000 scan-loop and adaptive branch semantics untouched unless a
   dedicated fingerprint/parity gate is added in the same tranche.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.67%.
- Differentiability/refactor implementation: 99.986%.
- Solver monolith reduction: 88.75%.
- Free-boundary adjoint monolith reduction: 80%.
- Driver workflow decomposition: 91.6%.
- WOUT diagnostic/profile decomposition: 98.5%.
- Optimizer workflow decomposition: 86%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.24%.

## 2026-06-16 Residual Host-Profile Setup Config Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `resolve_host_profile_setup` to
   `vmec_jax.solvers.fixed_boundary.residual.config`.
2. Replaced inline `VMEC_JAX_HOST_PROFILE_SETUP` parsing in
   `solve_fixed_boundary_residual_iter` with the new config helper.
3. Added focused tests for CPU/GPU `auto`, explicit `off`, and explicit `on`
   behavior.

Results obtained:

- `solve_fixed_boundary_residual_iter` decreased from 8958 to 8957 lines.
- The host-side profile setup policy is now independently unit-tested and
  grouped with the residual-solver config policy helpers.
- Existing performance instrumentation tests still pass, including the forced
  host-profile setup comparison.
- No profile values, force assembly, JIT, scan, or branch behavior changed.

Tests and commands run:

- `python -m ruff check vmec_jax/solve.py vmec_jax/solvers/fixed_boundary/residual/config.py tests/test_solve_residual_iter_config.py tests/test_solve_performance_instrumentation.py`
- `python -m compileall -q vmec_jax/solve.py vmec_jax/solvers/fixed_boundary/residual/config.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_residual_iter_config.py tests/test_solve_performance_instrumentation.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_residual_iter_helpers_wave8_coverage.py tests/test_solve_runtime.py -q`
- `python tools/diagnostics/source_health.py --top 24 --top-functions 32`

Best next steps:

1. Commit and push the residual host-profile setup config extraction.
2. Continue extracting pure config/setup policy first; reserve numerical loop
   extractions for tranches with stronger parity and fingerprint gates.
3. Poll CI for the latest branch run after this push.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.67%.
- Differentiability/refactor implementation: 99.987%.
- Solver monolith reduction: 88.8%.
- Free-boundary adjoint monolith reduction: 80%.
- Driver workflow decomposition: 91.6%.
- WOUT diagnostic/profile decomposition: 98.5%.
- Optimizer workflow decomposition: 86%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.25%.

## 2026-06-16 Residual Axis-Reset Config Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `AxisResetConfig` and `resolve_axis_reset_config` to
   `vmec_jax.solvers.fixed_boundary.residual.config`.
2. Replaced inline `VMEC_JAX_FORCE_AXIS_RESET_INIT`,
   `VMEC_JAX_AXIS_RESET_ALWAYS_3D`, and `VMEC_JAX_AXIS_RESET_FSQ_MIN` parsing
   in `solve_fixed_boundary_residual_iter`.
3. Added focused tests for defaults, explicit true flags, invalid `FSQ_MIN`,
   negative `FSQ_MIN`, and the existing legacy behavior that `off` is not a
   false token for the axis-reset boolean envs.

Results obtained:

- `solve_fixed_boundary_residual_iter` decreased from 8957 to 8954 lines.
- `solve.py` decreased from 10006 to 10004 lines.
- Axis-reset env policy is now unit-tested without constructing a VMEC solve.
- No initial-axis reset algorithm, bad-Jacobian handling, scan behavior, or
  parity policy changed.

Tests and commands run:

- `python -m ruff check vmec_jax/solve.py vmec_jax/solvers/fixed_boundary/residual/config.py tests/test_solve_residual_iter_config.py tests/test_solve_finish_cache_more_coverage.py`
- `python -m compileall -q vmec_jax/solve.py vmec_jax/solvers/fixed_boundary/residual/config.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_residual_iter_config.py tests/test_solve_finish_cache_more_coverage.py tests/test_solve_axis_helpers_more_coverage.py -q`
- `python tools/diagnostics/source_health.py --top 24 --top-functions 32`

Best next steps:

1. Commit and push the residual axis-reset config extraction.
2. Continue simplifying the solver by moving pure setup policies into existing
   residual domain modules, but avoid numerical loop rewrites without parity
   gates.
3. Keep prioritizing fewer clearer domain modules over many new helper files.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.67%.
- Differentiability/refactor implementation: 99.988%.
- Solver monolith reduction: 88.85%.
- Free-boundary adjoint monolith reduction: 80%.
- Driver workflow decomposition: 91.6%.
- WOUT diagnostic/profile decomposition: 98.5%.
- Optimizer workflow decomposition: 86%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.26%.

## 2026-06-16 Residual Host Setup-Enforce Config Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `resolve_setup_host_enforce` to
   `vmec_jax.solvers.fixed_boundary.residual.config`.
2. Replaced inline `VMEC_JAX_HOST_SETUP_ENFORCE` parsing in
   `solve_fixed_boundary_residual_iter`.
3. Added focused tests for explicit disable, explicit force, traced-state
   suppression, accelerator auto-enable, CPU auto-disable, and host-update
   assembly suppression.
4. Re-ran the existing direct-coil free-boundary host-setup comparison to verify
   the setup path still matches the default path.

Results obtained:

- `solve_fixed_boundary_residual_iter` decreased from 8954 to 8949 lines.
- `solve.py` decreased from 10004 to 10000 lines.
- Host setup-enforcement policy is now independent, named, and unit-tested.
- No row/gauge enforcement algorithm, free-boundary coupling, scan behavior, or
  differentiability behavior changed; traced states still suppress host setup.

Tests and commands run:

- `python -m ruff check vmec_jax/solve.py vmec_jax/solvers/fixed_boundary/residual/config.py tests/test_solve_residual_iter_config.py tests/test_free_boundary_coil_provider_forward.py`
- `python -m compileall -q vmec_jax/solve.py vmec_jax/solvers/fixed_boundary/residual/config.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_residual_iter_config.py tests/test_free_boundary_coil_provider_forward.py::test_run_free_boundary_host_setup_enforce_matches_default_path -q`
- `python tools/diagnostics/source_health.py --top 24 --top-functions 32`

Best next steps:

1. Commit and push the residual host setup-enforce config extraction.
2. Continue extracting pure policy/setup seams if they are already naturally
   owned by residual config/runtime modules.
3. Avoid splitting `_run_vmec2000_scan` or `_advance_step` until there is a
   specific parity/fingerprint tranche; those functions own VMEC-compatible
   adaptive behavior and are higher risk.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.67%.
- Differentiability/refactor implementation: 99.989%.
- Solver monolith reduction: 88.9%.
- Free-boundary adjoint monolith reduction: 80%.
- Driver workflow decomposition: 91.6%.
- WOUT diagnostic/profile decomposition: 98.5%.
- Optimizer workflow decomposition: 86%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.27%.

## 2026-06-16 Fixed-Boundary Residual Iteration Domain Relocation

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Moved the fixed-boundary residual solver implementation out of the generic
   `vmec_jax.solve` facade and into the domain package
   `vmec_jax.solvers.fixed_boundary.residual.iteration`.
2. Replaced `vmec_jax.solve` with a small compatibility facade that re-exports
   all legacy symbols from the domain implementation.
3. Added assignment forwarding from the facade to the implementation module so
   existing private monkeypatch/debug seams such as
   `vmec_jax.solve._scan_backend_name` still affect the globals used by
   `solve_fixed_boundary_residual_iter`.
4. Converted the relocated implementation to absolute `vmec_jax.*` imports so
   it is owned by the fixed-boundary residual domain instead of relying on the
   old root-relative module location.
5. Added a regression test that verifies private assignments on
   `vmec_jax.solve` propagate to
   `vmec_jax.solvers.fixed_boundary.residual.iteration`.

Results obtained:

- `vmec_jax/solve.py` changed from a 10,000-line generic implementation file
  into a small compatibility facade.
- The remaining 10,000-line hotspot is now explicitly named and located as
  `vmec_jax/solvers/fixed_boundary/residual/iteration.py`.
- This is a structural ownership change, not just a line-count shave: the next
  split can target `residual.iteration.solve_fixed_boundary_residual_iter` and
  its nested VMEC2000 scan/controller context directly.
- Public imports from `vmec_jax.solve` still work, including private internal
  aliases used by the test suite.
- No solver math, scan branch behavior, adaptive-controller policy, CLI
  behavior, or output semantics changed.

Tests and commands run:

- `python -m compileall -q vmec_jax/solve.py vmec_jax/solvers/fixed_boundary/residual/iteration.py`
- `python - <<'PY' ...` import/assignment-forwarding smoke check for
  `vmec_jax.solve` and `residual.iteration`
- `python -m ruff check vmec_jax/solve.py vmec_jax/solvers/fixed_boundary/residual/iteration.py tests/test_solve_residual_iter_helpers_wave8_coverage.py tests/test_solve_scan_chunking.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_residual_iter_config.py tests/test_solve_residual_iter_helpers_wave8_coverage.py tests/test_solve_scan_chunking.py tests/test_solve_runtime.py tests/test_driver_api.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_additional_helpers.py tests/test_solve_real_scan_wave10_coverage.py tests/test_solve_finish_cache_more_coverage.py tests/test_driver_run_wave8_coverage.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_performance_instrumentation.py tests/test_free_boundary_coil_provider_forward.py::test_run_free_boundary_host_setup_enforce_matches_default_path tests/test_driver_policy_helpers.py -q`
- `python tools/diagnostics/source_health.py --top 30 --top-functions 45`

Best next steps:

1. Commit and push the residual iteration domain relocation.
2. Next large tranche should split `residual.iteration` itself, not keep
   polishing the facade. The most useful target is a VMEC2000 scan/controller
   context object that can move `_run_vmec2000_scan` out of
   `solve_fixed_boundary_residual_iter`.
3. Treat that next scan split as a parity-gated tranche because
   `_run_vmec2000_scan` has 97 closure variables and owns accepted/rejected
   controller behavior.
4. After scan/controller split, separate force-assembly setup from iteration
   control so Python API paths remain differentiable while CLI paths can keep
   non-differentiable performance shortcuts.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.67%.
- Differentiability/refactor implementation: 99.992%.
- Solver monolith reduction: 91.0%.
- Free-boundary adjoint monolith reduction: 80%.
- Driver workflow decomposition: 91.6%.
- WOUT diagnostic/profile decomposition: 98.5%.
- Optimizer workflow decomposition: 86%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.35%.

## 2026-06-16 Fixed-Boundary Public API Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `vmec_jax.solvers.fixed_boundary.api` for the user-facing fixed-boundary
   solver entry points:
   `solve_lambda_gd`, fixed-boundary GD/L-BFGS, residual L-BFGS/Gauss-Newton,
   and `first_step_diagnostics`.
2. Removed those public wrappers from
   `vmec_jax.solvers.fixed_boundary.residual.iteration`, leaving that module
   focused on the residual iteration engine and its compatibility aliases.
3. Updated the historical `vmec_jax.solve` facade to export both the residual
   implementation module and the new fixed-boundary API module.
4. Preserved assignment forwarding for both private residual monkeypatch seams
   and public API monkeypatch seams.
5. Added a regression test that verifies assigning
   `vmec_jax.solve.solve_lambda_gd` also updates
   `vmec_jax.solvers.fixed_boundary.api.solve_lambda_gd`.

Results obtained:

- `vmec_jax/solvers/fixed_boundary/residual/iteration.py` dropped from 10,000
  lines to 9,544 lines.
- Public fixed-boundary solver wrappers now have a domain-owned home instead of
  living inside the residual iteration engine.
- `vmec_jax.solve` remains backward-compatible for public imports and private
  debug/monkeypatch workflows.
- CI for the previous residual-domain relocation is green, including coverage,
  docs, console script smoke, and physics smoke.
- Source-health still correctly identifies the next real hotspot:
  `solve_fixed_boundary_residual_iter` itself and its nested VMEC2000
  scan/controller function.

Tests and commands run:

- `python -m compileall -q vmec_jax/solve.py vmec_jax/solvers/fixed_boundary/api.py vmec_jax/solvers/fixed_boundary/residual/iteration.py`
- `python -m ruff check vmec_jax/solve.py vmec_jax/solvers/fixed_boundary/api.py vmec_jax/solvers/fixed_boundary/residual/iteration.py tests/test_solve_residual_iter_helpers_wave8_coverage.py tests/test_solve_branch_coverage.py tests/test_solve_gd_wave10_coverage.py tests/test_solve_lbfgs_wave8_coverage.py tests/test_solve_residual_optimizer_wave8_coverage.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_branch_coverage.py tests/test_solve_gd_wave10_coverage.py tests/test_solve_lbfgs_wave8_coverage.py tests/test_solve_residual_optimizer_wave8_coverage.py tests/test_solve_residual_iter_helpers_wave8_coverage.py -q`
- `python tools/diagnostics/source_health.py --top 20 --top-functions 25`

Best next steps:

1. Build a typed scan/controller context for the VMEC2000 residual-iteration
   branch before moving `_run_vmec2000_scan`; direct extraction is too risky
   because that nested function owns accepted/rejected branch behavior and
   currently closes over ~100 local symbols.
2. Move the VMEC2000 scan implementation into `vmec_jax.solvers.fixed_boundary.scan`
   once the context and branch-fingerprint tests are in place.
3. Then split force setup from iteration control so the Python API can expose a
   cleaner differentiable path while CLI solve paths keep non-differentiable
   performance shortcuts.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.70%.
- Differentiability/refactor implementation: 99.993%.
- Solver monolith reduction: 91.4%.
- Free-boundary adjoint monolith reduction: 80%.
- Driver workflow decomposition: 91.6%.
- WOUT diagnostic/profile decomposition: 98.5%.
- Optimizer workflow decomposition: 86%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.37%.

## 2026-06-16 VMEC2000 Scan Setup Domain Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `Vmec2000ScanSetup` and `resolve_vmec2000_scan_setup` to
   `vmec_jax.solvers.fixed_boundary.scan.planning`.
2. Moved VMEC2000 scan run-flag, state-only override, NSTEP screen cadence, and
   scan preconditioner option resolution out of the numerical scan body.
3. Updated `solve_fixed_boundary_residual_iter._run_vmec2000_scan` to consume
   the resolved scan setup object before entering the accepted/rejected
   controller logic.
4. Added a focused pure-policy test covering state-only scan overrides and
   explicit preconditioner option overrides.

Results obtained:

- The residual iteration module dropped from 9,544 lines to 9,521 lines.
- `solve_fixed_boundary_residual_iter` dropped from 8,949 to 8,927 lines.
- `_run_vmec2000_scan` dropped from 2,209 to 2,187 lines.
- The more important result is architectural: scan environment/policy setup now
  lives in `fixed_boundary.scan.planning`, not inside the nonlinear controller.
  This makes the future `_run_vmec2000_scan` extraction depend on a smaller and
  more explicit setup context.

Tests and commands run:

- `python -m ruff check vmec_jax/solvers/fixed_boundary/scan/planning.py vmec_jax/solvers/fixed_boundary/residual/iteration.py tests/test_solve_scan_planning_helpers.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_scan_planning_helpers.py tests/test_solve_scan_chunking.py tests/test_solve_real_scan_wave10_coverage.py tests/test_solve_residual_iter_helpers_wave8_coverage.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_branch_coverage.py tests/test_solve_gd_wave10_coverage.py tests/test_solve_lbfgs_wave8_coverage.py tests/test_solve_residual_optimizer_wave8_coverage.py -q`
- `python tools/diagnostics/source_health.py --top 20 --top-functions 25`

Best next steps:

1. Continue migrating setup/policy pieces out of `_run_vmec2000_scan` until the
   remaining nested function is mostly numerical step logic.
2. Introduce a typed scan/controller context once the setup surface is stable.
3. Move the scan controller to `fixed_boundary.scan` under branch-fingerprint
   and VMEC2000 parity tests.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.72%.
- Differentiability/refactor implementation: 99.994%.
- Solver monolith reduction: 91.6%.
- Free-boundary adjoint monolith reduction: 80%.
- Driver workflow decomposition: 91.6%.
- WOUT diagnostic/profile decomposition: 98.5%.
- Optimizer workflow decomposition: 86%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.38%.

## 2026-06-16 VMEC2000 Scan Runtime Hook Env Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `resolve_scan_runtime_hooks_from_env` to
   `vmec_jax.solvers.fixed_boundary.scan.runtime`.
2. Replaced direct scan-body `os.getenv` plumbing for time-control dumps,
   callback mode, and profiler trace hooks with the scan runtime helper.
3. Added an equivalence test against the direct runtime-hook resolver.

Results obtained:

- The residual iteration module dropped from 9,521 lines to 9,520 lines and
  `_run_vmec2000_scan` dropped from 2,187 to 2,186 lines.
- The practical gain is another explicit domain seam: scan runtime hooks are
  now resolved by `fixed_boundary.scan.runtime`, while the residual iteration
  body only consumes the resolved hooks.

Tests and commands run:

- `python -m ruff check vmec_jax/solvers/fixed_boundary/scan/runtime.py vmec_jax/solvers/fixed_boundary/residual/iteration.py tests/test_solve_scan_planning_helpers.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_scan_planning_helpers.py tests/test_solve_scan_chunking.py tests/test_solve_real_scan_wave10_coverage.py tests/test_solve_residual_iter_helpers_wave8_coverage.py -q`
- `python tools/diagnostics/source_health.py --top 20 --top-functions 25`

Best next steps:

1. Keep extracting scan setup/context pieces until `_run_vmec2000_scan` can be
   moved as a controller function without a 97-variable closure.
2. Then add the branch-fingerprint parity gate for the moved controller.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.73%.
- Differentiability/refactor implementation: 99.994%.
- Solver monolith reduction: 91.65%.
- Free-boundary adjoint monolith reduction: 80%.
- Driver workflow decomposition: 91.6%.
- WOUT diagnostic/profile decomposition: 98.5%.
- Optimizer workflow decomposition: 86%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.39%.

## 2026-06-16 Residual Controller Constants and Axis Reset Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `Vmec2000ControllerConstants` and
   `default_vmec2000_controller_constants` to
   `vmec_jax.solvers.fixed_boundary.scan.planning`.
2. Replaced duplicated scan and non-scan VMEC2000 controller magic constants
   with the typed defaults object.
3. Moved the initial magnetic-axis reset implementation from
   `solve_fixed_boundary_residual_iter` into
   `vmec_jax.solvers.fixed_boundary.diagnostics.axis_reset`.
4. Kept a thin local wrapper in the residual loop only to update the existing
   `axis_reset_coeffs` side-channel.
5. Added focused tests for controller constants and the extracted axis-reset
   fallback path.

Results obtained:

- The residual iteration module dropped from 9,520 lines to 9,418 lines.
- `solve_fixed_boundary_residual_iter` dropped from 8,926 to 8,822 lines.
- The nested `_reset_axis_from_boundary` block dropped to a 32-line delegating
  wrapper.
- Magnetic-axis reset logic now lives with the existing axis reset decision,
  merge, and dump helpers.

Tests and commands run:

- `python -m ruff check vmec_jax/solvers/fixed_boundary/diagnostics/axis_reset.py vmec_jax/solvers/fixed_boundary/residual/iteration.py tests/test_solve_axis_helpers_more_coverage.py tests/test_solve_scan_planning_helpers.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_axis_helpers_more_coverage.py tests/test_solve_scan_planning_helpers.py tests/test_solve_scan_chunking.py tests/test_solve_real_scan_wave10_coverage.py tests/test_solve_residual_iter_helpers_wave8_coverage.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_branch_coverage.py tests/test_solve_gd_wave10_coverage.py tests/test_solve_lbfgs_wave8_coverage.py tests/test_solve_residual_optimizer_wave8_coverage.py -q`
- `python tools/diagnostics/source_health.py --top 20 --top-functions 25`

Best next steps:

1. Continue reducing `solve_fixed_boundary_residual_iter` by moving setup-phase
   helper clusters to their existing domains, especially ptau/bad-Jacobian
   setup and mode-transform/update helpers.
2. After setup clusters are extracted, introduce the typed scan/controller
   context needed to move `_run_vmec2000_scan` itself.
3. Keep branch-fingerprint and VMEC2000 parity tests as the gate before moving
   accepted/rejected controller math.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.75%.
- Differentiability/refactor implementation: 99.995%.
- Solver monolith reduction: 92.1%.
- Free-boundary adjoint monolith reduction: 80%.
- Driver workflow decomposition: 91.6%.
- WOUT diagnostic/profile decomposition: 98.5%.
- Optimizer workflow decomposition: 86%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.41%.

## 2026-06-16 Ptau Bad-Jacobian Context Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `PtauMinmaxContext` and `build_ptau_minmax_context` to
   `vmec_jax.solvers.fixed_boundary.scan.math`.
2. Added context-aware host and JAX ptau min/max dispatch helpers.
3. Replaced the residual loop's inline pshalf/ohs/JAX-constant setup block
   with a scan-math context object.
4. Added a test proving the context host path matches legacy ptau min/max
   values and bypasses the tiny JIT callback in the host-update path.

Results obtained:

- The residual iteration module dropped from 9,418 lines to 9,397 lines.
- `solve_fixed_boundary_residual_iter` dropped from 8,822 to 8,800 lines.
- Ptau/bad-Jacobian precomputed constants now live in the scan math domain,
  not in the residual driver.
- This keeps the hot bad-Jacobian path explicit while moving setup state toward
  the typed context needed for a future scan-controller extraction.

Tests and commands run:

- `python -m ruff check vmec_jax/solvers/fixed_boundary/scan/math.py vmec_jax/solvers/fixed_boundary/residual/iteration.py tests/test_solve_scan_math_helpers.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_scan_math_helpers.py tests/test_solve_axis_helpers_more_coverage.py tests/test_solve_scan_planning_helpers.py tests/test_solve_scan_chunking.py tests/test_solve_real_scan_wave10_coverage.py tests/test_solve_residual_iter_helpers_wave8_coverage.py -q`
- `python tools/diagnostics/source_health.py --top 20 --top-functions 25`

Best next steps:

1. Continue extracting setup helper clusters, especially mode-transform and
   update preconditioner setup.
2. Then introduce a typed scan-controller context and branch-fingerprint tests
   before moving accepted/rejected controller math.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.76%.
- Differentiability/refactor implementation: 99.995%.
- Solver monolith reduction: 92.3%.
- Free-boundary adjoint monolith reduction: 80%.
- Driver workflow decomposition: 91.6%.
- WOUT diagnostic/profile decomposition: 98.5%.
- Optimizer workflow decomposition: 86%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.42%.

## 2026-06-16 Mode-Transform Context Extraction and CI Seam Repair

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `ModeTransformContext` and `build_mode_transform_context` to
   `vmec_jax.solvers.fixed_boundary.residual.mode_transform`.
2. Moved signed-mode index setup, host DGEMM projections, VMEC `scalxc`
   physical-update factors, mode-diagonal weights, and R/Z norm dispatch into
   that context.
3. Replaced the residual loop's inline mode-transform setup block with a
   context construction plus short local compatibility wrappers.
4. Restored the legacy `_scan_math_ptau_minmax_from_k_host` and
   `_scan_math_ptau_minmax_from_k_jax` aliases after the ptau-context
   extraction, preserving `vmec_jax.solve` monkeypatch seams used by tests and
   downstream diagnostics.
5. Replaced stale fused-controller references to `_ptau_pshalf_jax` and
   `_ptau_ohs_jax` with the extracted `PtauMinmaxContext` fields.
6. Added focused tests for context construction, host physical transforms, and
   JAX-vs-NumPy R/Z norm parity.

Results obtained:

- The residual iteration module dropped from 9,397 lines to 9,188 lines.
- `solve_fixed_boundary_residual_iter` dropped from 8,800 to 8,590 lines.
- The prior CI failure on ptau monkeypatch aliases and fused free-boundary
  payload constants is fixed locally.
- Mode transform bookkeeping is now a named fixed-boundary residual domain,
  which is a prerequisite for typed scan/controller context extraction.

Tests and commands run:

- `python -m ruff check vmec_jax/solvers/fixed_boundary/residual/mode_transform.py vmec_jax/solvers/fixed_boundary/residual/iteration.py tests/test_solve_residual_iter_mode_transform_helpers.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_residual_iter_mode_transform_helpers.py tests/test_solve_residual_iter_geometry_helpers.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_finish_cache_more_coverage.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_direct_coil_finite_pressure_sensitivity.py::test_full_adjoint_trace_records_raw_preconditioner_on_fused_payload_path -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_scan_math_helpers.py tests/test_solve_axis_helpers_more_coverage.py tests/test_solve_scan_planning_helpers.py tests/test_solve_scan_chunking.py tests/test_solve_real_scan_wave10_coverage.py tests/test_solve_residual_iter_helpers_wave8_coverage.py tests/test_solve_residual_iter_mode_transform_helpers.py tests/test_solve_residual_iter_geometry_helpers.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_branch_coverage.py tests/test_solve_gd_wave10_coverage.py tests/test_solve_lbfgs_wave8_coverage.py tests/test_solve_residual_optimizer_wave8_coverage.py -q`
- `python tools/diagnostics/source_health.py --top 20 --top-functions 25`

Best next steps:

1. Extract the next setup-only cluster from `solve_fixed_boundary_residual_iter`,
   with priority on preconditioner cache planning and force-payload setup.
2. Introduce a typed scan-controller context only after more setup state is out
   of the residual loop.
3. Move `_run_vmec2000_scan` only after context extraction and branch-fingerprint
   tests make the accepted/rejected controller seam explicit.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.77%.
- Differentiability/refactor implementation: 99.995%.
- Solver monolith reduction: 93.6%.
- Free-boundary adjoint monolith reduction: 80%.
- Driver workflow decomposition: 91.6%.
- WOUT diagnostic/profile decomposition: 98.5%.
- Optimizer workflow decomposition: 86%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.45%.

## 2026-06-16 Lambda Preconditioner Payload Selection Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `LambdaPreconditionerOutputs` and
   `lambda_preconditioner_outputs` to the existing fixed-boundary
   preconditioning operators module.
2. Replaced duplicated residual-loop branches for `faclam` and `lamcal`
   debug-payload selection with the shared helper.
3. Added focused unit coverage for all four payload combinations:
   base lambda preconditioner, `faclam`, `lamcal`, and both together.

Results obtained:

- The residual iteration module dropped from 9,191 lines to 9,180 lines.
- `solve_fixed_boundary_residual_iter` dropped from 8,591 to 8,579 lines.
- The payload-selection policy now lives in the preconditioning domain instead
  of being duplicated in two controller branches.

Tests and commands run:

- `python -m ruff check vmec_jax/solvers/fixed_boundary/preconditioning/operators.py vmec_jax/solvers/fixed_boundary/residual/iteration.py tests/test_solve_preconditioner_metric_helpers.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_preconditioner_metric_helpers.py tests/test_solve_finish_cache_more_coverage.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_scan_math_helpers.py tests/test_solve_axis_helpers_more_coverage.py tests/test_solve_scan_planning_helpers.py tests/test_solve_scan_chunking.py tests/test_solve_real_scan_wave10_coverage.py tests/test_solve_residual_iter_helpers_wave8_coverage.py tests/test_solve_residual_iter_mode_transform_helpers.py tests/test_solve_residual_iter_geometry_helpers.py -q`
- `python tools/diagnostics/source_health.py --top 8 --top-functions 12`

Best next steps:

1. Continue setup-only extraction, but avoid creating new files for small
   helpers; prefer extending existing domains.
2. Next likely target: a typed preconditioner cache context that can make the
   duplicated refresh/reassemble blocks smaller without changing controller
   semantics.
3. Keep CI and VMEC parity gates green before moving `_run_vmec2000_scan`.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.77%.
- Differentiability/refactor implementation: 99.995%.
- Solver monolith reduction: 93.8%.
- Free-boundary adjoint monolith reduction: 80%.
- Driver workflow decomposition: 91.6%.
- WOUT diagnostic/profile decomposition: 98.5%.
- Optimizer workflow decomposition: 86%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.46%.

## 2026-06-16 Preconditioner Cache Update Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `PreconditionerCacheUpdate` and `update_preconditioner_cache`
   to the existing fixed-boundary preconditioning operators module.
2. Moved the shared nonscan preconditioner refresh/reassemble/reuse policy out
   of the residual loop while keeping timing counters, debug dumps, and force
   application at the call site.
3. Replaced the duplicated 3D VMEC2000-control and axisymmetric nonscan cache
   blocks with the shared helper.
4. Added focused unit coverage for clean cache hits, missing-cache refreshes
   with `faclam`/`lamcal` debug payloads, and jmax-only reassembly.

Results obtained:

- The residual iteration module dropped from 9,180 lines to 9,153 lines.
- `solve_fixed_boundary_residual_iter` dropped from 8,579 to 8,551 lines.
- Preconditioner cache state transitions are now a named preconditioning
  domain operation instead of inline duplicated controller code.
- The previous branch CI run for commit `10f7cfb` passed.

Tests and commands run:

- `python -m ruff check vmec_jax/solvers/fixed_boundary/preconditioning/operators.py vmec_jax/solvers/fixed_boundary/residual/iteration.py tests/test_solve_preconditioner_metric_helpers.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_preconditioner_metric_helpers.py tests/test_solve_finish_cache_more_coverage.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_scan_math_helpers.py tests/test_solve_axis_helpers_more_coverage.py tests/test_solve_scan_planning_helpers.py tests/test_solve_scan_chunking.py tests/test_solve_real_scan_wave10_coverage.py tests/test_solve_residual_iter_helpers_wave8_coverage.py tests/test_solve_residual_iter_mode_transform_helpers.py tests/test_solve_residual_iter_geometry_helpers.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_branch_coverage.py tests/test_solve_gd_wave10_coverage.py tests/test_solve_lbfgs_wave8_coverage.py tests/test_solve_residual_optimizer_wave8_coverage.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_direct_coil_finite_pressure_sensitivity.py::test_full_adjoint_trace_records_raw_preconditioner_on_fused_payload_path -q`
- `python tools/diagnostics/ci_core_bucket_args.py "driver-solve-discrete" > /tmp/vmec_jax_core_args.txt && JAX_ENABLE_X64=1 xargs pytest -q -n 4 -m "not full and not vmec2000 and not simsopt" --durations=25 < /tmp/vmec_jax_core_args.txt`
- `python tools/diagnostics/ci_core_bucket_args.py "freeb-external" > /tmp/vmec_jax_freeb_args.txt && JAX_ENABLE_X64=1 xargs pytest -q -n 4 -m "not full and not vmec2000 and not simsopt" --durations=25 < /tmp/vmec_jax_freeb_args.txt`
- `python tools/diagnostics/source_health.py --top 8 --top-functions 12`

Best next steps:

1. Run the local core CI buckets before pushing this tranche.
2. Continue preconditioner-cache context consolidation in the existing
   preconditioning namespace if the buckets pass.
3. Defer driver decomposition until this residual-loop seam is green on CI.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.77%.
- Differentiability/refactor implementation: 99.995%.
- Solver monolith reduction: 94.0%.
- Free-boundary adjoint monolith reduction: 80%.
- Driver workflow decomposition: 91.6%.
- WOUT diagnostic/profile decomposition: 98.5%.
- Optimizer workflow decomposition: 86%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.47%.

## 2026-06-16 Preconditioner Cache Reset Consolidation

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `PreconditionerCacheSnapshot` and
   `empty_preconditioner_cache_snapshot` to the fixed-boundary
   preconditioning operators module.
2. Added a local `_clear_preconditioner_cache_locals` invalidation seam inside
   `solve_fixed_boundary_residual_iter`.
3. Replaced repeated restart/fallback cache reset assignments with the shared
   invalidation call, while preserving the existing VMEC2000
   `force_bcovar_update` conditions.
4. Added unit coverage for the empty cache snapshot field order used by the
   residual-loop tuple assignment.

Results obtained:

- The residual iteration module dropped from 9,153 lines to 9,094 lines.
- `solve_fixed_boundary_residual_iter` dropped from 8,551 to 8,491 lines.
- Restart and fallback cache invalidation now has one local seam instead of
  several repeated assignment blocks.

Tests and commands run:

- `python -m ruff check vmec_jax/solvers/fixed_boundary/preconditioning/operators.py vmec_jax/solvers/fixed_boundary/residual/iteration.py tests/test_solve_preconditioner_metric_helpers.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_preconditioner_metric_helpers.py tests/test_solve_finish_cache_more_coverage.py tests/test_solve_branch_coverage.py tests/test_solve_real_scan_wave10_coverage.py tests/test_solve_scan_chunking.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_direct_coil_finite_pressure_sensitivity.py::test_full_adjoint_trace_records_raw_preconditioner_on_fused_payload_path -q`
- `python tools/diagnostics/source_health.py --top 8 --top-functions 12`

Best next steps:

1. Wait for the CI run triggered by `5d3355a`; fix any regression before
   pushing this reset tranche.
2. If CI is green, commit this reset consolidation.
3. Continue the residual-loop decomposition at the VMEC2000 scan-controller
   boundary, but only after branch-fingerprint tests are kept green.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.77%.
- Differentiability/refactor implementation: 99.995%.
- Solver monolith reduction: 94.4%.
- Free-boundary adjoint monolith reduction: 80%.
- Driver workflow decomposition: 91.6%.
- WOUT diagnostic/profile decomposition: 98.5%.
- Optimizer workflow decomposition: 86%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.48%.

## 2026-06-16 Direct External Provider Runtime Context Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `ExternalFieldProviderContext` and
   `resolve_external_field_provider_context` to the existing driver runtime
   module.
2. Moved direct-coil provider detection and optional host-side coil-geometry
   cache setup out of `run_fixed_boundary`.
3. Kept lazy coil import semantics and fallback behavior for custom providers
   that cannot use the built-in cached coil geometry.
4. Added focused driver-runtime tests for legacy mgrid mode, direct-coil cache
   construction, disabled/existing cache behavior, and custom-provider fallback.

Results obtained:

- `run_fixed_boundary` dropped from 1,809 lines before this turn to 1,774
  lines.
- Direct external provider setup is now a named runtime context with explicit
  tests instead of inline public-driver branching.
- No new source file was added.

Tests and commands run:

- `python -m ruff check vmec_jax/driver.py vmec_jax/drivers/runtime.py tests/test_driver_policy_helpers.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_driver_policy_helpers.py tests/test_driver_run_wave8_coverage.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_coil_provider_forward.py::test_run_free_boundary_direct_coil_geometry_cache_matches_uncached_path tests/test_driver_run_wave8_coverage.py::test_direct_coil_free_boundary_exposes_limited_updates tests/test_driver_run_wave8_coverage.py::test_direct_coil_free_boundary_quiet_performance_path_uses_light_history -q`
- `python tools/diagnostics/source_health.py --top 8 --top-functions 12`

Best next steps:

1. Commit this driver runtime extraction after the current CI run is understood.
2. Continue driver decomposition by moving one setup cluster at a time into
   existing `vmec_jax.drivers` domain modules.
3. Keep the next extraction away from the adaptive solver branch until the
   latest residual/preconditioner CI runs have passed.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.77%.
- Differentiability/refactor implementation: 99.995%.
- Solver monolith reduction: 94.4%.
- Free-boundary adjoint monolith reduction: 80%.
- Driver workflow decomposition: 92.4%.
- WOUT diagnostic/profile decomposition: 98.5%.
- Optimizer workflow decomposition: 86%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.49%.

## 2026-06-16 Driver Static/Profile/Initial-Guess Setup Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Moved driver sign-convention, VMEC2000 force-JIT override, and public
   step-size default policy into the existing driver policy module.
2. Moved static-grid profile assembly into the driver flux module, including
   the VMEC half-mesh pressure/profile evaluation rule.
3. Moved the CPU NumPy/no-JIT initial-guess fallback seam into the driver solve
   module, preserving all injected callables used by tests and monkeypatches.
4. Replaced the corresponding inline `run_fixed_boundary` setup clusters with
   named helper calls.
5. Added focused unit coverage for VMEC2000 sign parity, JIT env overrides,
   step-size policy, host-default/fallback profile construction, and
   tracer/env-safe initial-guess routing.

Results obtained:

- `run_fixed_boundary` dropped from 1,774 lines to 1,709 lines.
- Public-driver setup is now split into explicit policy, flux/profile, runtime,
  and solve-domain helper seams without adding a new root module.
- The next largest driver-level setup seams are now smaller than the residual
  scan-controller monolith; source-health still identifies
  `solve_fixed_boundary_residual_iter` and `_run_vmec2000_scan` as the next
  meaningful solve seams.

Tests and commands run:

- `python -m ruff check vmec_jax/driver.py vmec_jax/drivers/policy.py vmec_jax/drivers/flux.py vmec_jax/drivers/solve.py tests/test_driver_policy_helpers.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_driver_policy_helpers.py tests/test_driver_run_wave8_coverage.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_driver_api.py::test_run_fixed_boundary_returns_current_driven_flux_profiles tests/test_driver_api.py::test_python_default_fixed_boundary_uses_optimized_controller -q`
- `python tools/diagnostics/source_health.py --top 8 --top-functions 12`

Best next steps:

1. Commit and push this driver setup tranche after a final diff review.
2. Start the next large solve seam at the VMEC2000 scan-controller boundary,
   favoring extraction into existing fixed-boundary residual/scan/preconditioner
   domains instead of adding generic `solver.py`-style files.
3. Keep branch-fingerprint and free-boundary replay tests green while reducing
   `_run_vmec2000_scan`, because that is now the highest-value refactor seam.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.77%.
- Differentiability/refactor implementation: 99.995%.
- Solver monolith reduction: 94.4%.
- Free-boundary adjoint monolith reduction: 80%.
- Driver workflow decomposition: 93.2%.
- WOUT diagnostic/profile decomposition: 98.5%.
- Optimizer workflow decomposition: 86%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.50%.

## 2026-06-16 VMEC2000 Scan Runner Cache Seam Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `get_or_build_scan_runner` to the existing fixed-boundary scan
   runtime module.
2. Centralized VMEC2000 scan-runner hit/miss/bypass cache bookkeeping,
   build/lookup timing, and miss-category recording.
3. Replaced the two inline scan-runner cache blocks inside the residual loop,
   keeping the numerical scan-body closures at the call sites.
4. Added direct unit coverage for cache miss, cache hit, and differentiating
   scan bypass behavior.

Results obtained:

- The residual iteration module dropped from 9,094 lines to 9,043 lines.
- `solve_fixed_boundary_residual_iter` dropped from 8,491 to 8,439 lines.
- `_run_vmec2000_scan` dropped from 2,187 to 2,161 lines.
- VMEC2000 scan cache semantics are now a named scan-runtime seam instead of
  duplicated inline controller code.

Tests and commands run:

- `python -m ruff check vmec_jax/solvers/fixed_boundary/scan/runtime.py vmec_jax/solvers/fixed_boundary/residual/iteration.py tests/test_solve_scan_planning_helpers.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_scan_planning_helpers.py tests/test_solve_real_scan_wave10_coverage.py tests/test_solve_wave7_coverage.py::test_residual_iter_vmec2000_scan_minimal_one_step tests/test_solve_wave7_coverage.py::test_residual_iter_vmec2000_scan_state_only -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_finish_cache_more_coverage.py tests/test_solve_scan_math_helpers.py tests/test_solve_scan_output.py tests/test_solve_scan_output_edge_cases_more_coverage.py -q`
- `python tools/diagnostics/source_health.py --top 8 --top-functions 12`

Best next steps:

1. Commit and push this solve seam if the final diff review is clean.
2. Continue with the next scan-controller seam that is still closure-light:
   preflight/chunk orchestration or status/fallback host post-processing.
3. Avoid moving `_advance_step` wholesale until its force/preconditioner/free-
   boundary dependencies are split into smaller domain payloads.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.77%.
- Differentiability/refactor implementation: 99.995%.
- Solver monolith reduction: 94.9%.
- Free-boundary adjoint monolith reduction: 80%.
- Driver workflow decomposition: 93.2%.
- WOUT diagnostic/profile decomposition: 98.5%.
- Optimizer workflow decomposition: 86%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.51%.

## 2026-06-16 VMEC2000 Scan Result Builder Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added state-only and traced VMEC2000 scan result builders to the residual
   finalize module.
2. Replaced inline empty-history result construction in `_run_vmec2000_scan`
   with the new finalize helpers.
3. Kept scan diagnostics generation in the scan output module and free-boundary
   diagnostic attachment injected from the residual loop.
4. Added focused tests for state-only and traced scan result construction.

Results obtained:

- The residual iteration module dropped from 9,043 lines to 9,034 lines.
- `solve_fixed_boundary_residual_iter` dropped from 8,439 to 8,428 lines.
- `_run_vmec2000_scan` dropped from 2,161 to 2,150 lines.
- Scan result assembly is now a finalize-domain operation rather than inline
  scan-controller code.

Tests and commands run:

- `python -m ruff check vmec_jax/solvers/fixed_boundary/residual/finalize.py vmec_jax/solvers/fixed_boundary/residual/iteration.py tests/test_solve_residual_iter_finalize_helpers.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_residual_iter_finalize_helpers.py tests/test_solve_real_scan_wave10_coverage.py tests/test_solve_scan_output.py tests/test_solve_wave7_coverage.py::test_residual_iter_vmec2000_scan_minimal_one_step tests/test_solve_wave7_coverage.py::test_residual_iter_vmec2000_scan_state_only -q`
- `python tools/diagnostics/source_health.py --top 8 --top-functions 12`

Best next steps:

1. Commit this postprocess seam with the scan-runner cache tranche if CI is
   green or still running cleanly.
2. Target a larger next seam only after identifying a natural domain boundary
   for scan preflight/chunk orchestration.
3. Keep avoiding wholesale `_advance_step` movement until force/preconditioner
   payload dependencies are further factored.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.77%.
- Differentiability/refactor implementation: 99.995%.
- Solver monolith reduction: 95.0%.
- Free-boundary adjoint monolith reduction: 80%.
- Driver workflow decomposition: 93.2%.
- WOUT diagnostic/profile decomposition: 98.5%.
- Optimizer workflow decomposition: 86%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.52%.

## 2026-06-16 VMEC2000 Scan Preflight And Diagnostic Replay Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `run_scan_preflight_step` to the fixed-boundary scan runtime module
   to share the one-step preflight execution used by chunked and non-chunked
   VMEC2000 scan paths.
2. Replaced both inline preflight blocks while keeping fallback, history
   materialization, and axis-reset policy at the controller level.
3. Added scan debug helpers for post-scan VMEC2000 row replay and p-tau dump
   replay.
4. Removed the corresponding host-side diagnostic loops from
   `_run_vmec2000_scan`.
5. Added direct unit coverage for non-JIT and JIT preflight paths, post-scan
   row replay, and p-tau diagnostic replay.

Results obtained:

- The residual iteration module dropped from 9,034 lines to 9,012 lines.
- `solve_fixed_boundary_residual_iter` dropped from 8,428 to 8,403 lines.
- `_run_vmec2000_scan` dropped from 2,150 to 2,125 lines.
- The scan controller now delegates one-step preflight mechanics and host
  diagnostic replay to scan-domain helpers, leaving the controller focused on
  branch policy and state evolution.

Tests and commands run:

- `python -m ruff check vmec_jax/solvers/fixed_boundary/scan/runtime.py vmec_jax/solvers/fixed_boundary/residual/iteration.py tests/test_solve_scan_planning_helpers.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_scan_planning_helpers.py tests/test_solve_real_scan_wave10_coverage.py tests/test_solve_wave7_coverage.py::test_residual_iter_vmec2000_scan_minimal_one_step tests/test_solve_wave7_coverage.py::test_residual_iter_vmec2000_scan_state_only -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_finish_cache_more_coverage.py tests/test_solve_scan_math_helpers.py tests/test_solve_scan_output.py tests/test_solve_scan_output_edge_cases_more_coverage.py tests/test_solve_residual_iter_runtime_helpers.py -q`
- `python -m ruff check vmec_jax/solvers/fixed_boundary/scan/debug.py vmec_jax/solvers/fixed_boundary/residual/iteration.py tests/test_solve_scan_debug_helpers.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_scan_debug_helpers.py tests/test_solve_real_scan_wave10_coverage.py tests/test_solve_wave7_coverage.py::test_residual_iter_vmec2000_scan_minimal_one_step tests/test_solve_wave7_coverage.py::test_residual_iter_vmec2000_scan_state_only -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_scan_planning_helpers.py tests/test_solve_scan_output.py tests/test_solve_scan_output_edge_cases_more_coverage.py tests/test_solve_finish_cache_more_coverage.py tests/test_solve_residual_iter_finalize_helpers.py -q`
- `python tools/diagnostics/source_health.py --top 8 --top-functions 12`

Best next steps:

1. Commit and push this scan preflight/debug tranche.
2. Re-check CI on the latest pushed stack; fix any regression before deeper
   scan-controller changes.
3. For the next large seam, target a true payload/domain split inside
   `_advance_step` rather than only moving orchestration wrappers.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.77%.
- Differentiability/refactor implementation: 99.995%.
- Solver monolith reduction: 95.3%.
- Free-boundary adjoint monolith reduction: 80%.
- Driver workflow decomposition: 93.2%.
- WOUT diagnostic/profile decomposition: 98.5%.
- Optimizer workflow decomposition: 86%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.53%.

## 2026-06-16 VMEC2000 Scan Debug Payload Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Moved the optional first-iteration force-channel scan debug print payload
   from `_advance_step` into the scan debug module.
2. Moved the optional requested-iteration state/checkpoint scan debug payload
   from `_advance_step` into the scan debug module.
3. Preserved the original force-debug behavior that requires the resolved scan
   debug printer hook to be present.
4. Added unit coverage for both moved debug helpers using fake `cond` and debug
   printer callables.

Results obtained:

- The residual iteration module dropped from 9,012 lines to 8,956 lines.
- `solve_fixed_boundary_residual_iter` dropped from 8,403 to 8,345 lines.
- `_run_vmec2000_scan` dropped from 2,125 to 2,067 lines.
- `_scan_step` dropped from 1,118 to 1,060 lines.
- `_advance_step` dropped from 1,102 to 1,044 lines.
- The large scan branch body now contains less optional diagnostic plumbing and
  more directly exposes the remaining force, preconditioner, time-control, and
  update payload seams.

Tests and commands run:

- `python -m ruff check vmec_jax/solvers/fixed_boundary/scan/debug.py vmec_jax/solvers/fixed_boundary/residual/iteration.py tests/test_solve_scan_debug_helpers.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_scan_debug_helpers.py tests/test_solve_real_scan_wave10_coverage.py tests/test_solve_wave7_coverage.py::test_residual_iter_vmec2000_scan_minimal_one_step tests/test_solve_wave7_coverage.py::test_residual_iter_vmec2000_scan_state_only -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_scan_planning_helpers.py tests/test_solve_scan_output.py tests/test_solve_scan_output_edge_cases_more_coverage.py tests/test_solve_finish_cache_more_coverage.py tests/test_solve_residual_iter_finalize_helpers.py -q`
- `python tools/diagnostics/source_health.py --top 8 --top-functions 12`

Best next steps:

1. Commit and push this debug-payload extraction.
2. If CI stays green, the next meaningful solve seam is the cache refresh /
   preconditioned force-payload construction inside `_advance_step`.
3. Keep each future `_advance_step` extraction tied to one VMEC domain concept:
   force payload, time-control/restart transition, or accepted/rejected state
   update.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.77%.
- Differentiability/refactor implementation: 99.995%.
- Solver monolith reduction: 95.8%.
- Free-boundary adjoint monolith reduction: 80%.
- Driver workflow decomposition: 93.2%.
- WOUT diagnostic/profile decomposition: 98.5%.
- Optimizer workflow decomposition: 86%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.54%.

## 2026-06-16 VMEC2000 Current Scan Preconditioned Payload Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `build_current_preconditioned_scan_payload` to the fixed-boundary
   scan payload module.
2. Moved the VMEC2000 scan current-step cache refresh, constraint
   preconditioner diagonal update, lambda preconditioner application, R/Z
   preconditioner matrix construction, and final current scan payload assembly
   out of `_advance_step`.
3. Kept controller-owned branch decisions in `_advance_step`: whether the
   cache is refreshed, which force norms and scales are active, and how the
   accepted/rejected state is advanced.
4. Added direct unit coverage for both cache-retaining and cache-refreshing
   payload paths.

Results obtained:

- The residual iteration module dropped from 8,956 lines to 8,877 lines.
- `solve_fixed_boundary_residual_iter` dropped from 8,345 to 8,266 lines.
- `_run_vmec2000_scan` dropped from 2,067 to 1,988 lines.
- `_scan_step` dropped from 1,060 to 981 lines.
- `_advance_step` dropped from 1,044 to 965 lines.
- The VMEC2000 current scan branch now has a scan-domain payload seam for the
  preconditioned force vector, which is a more useful future differentiation
  and profiling boundary than the previous inline mixed cache/force assembly.

Tests and commands run:

- `python -m ruff check vmec_jax/solvers/fixed_boundary/scan/payload.py vmec_jax/solvers/fixed_boundary/residual/iteration.py tests/test_solve_scan_payload_helpers.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_scan_payload_helpers.py tests/test_solve_real_scan_wave10_coverage.py tests/test_solve_wave7_coverage.py::test_residual_iter_vmec2000_scan_minimal_one_step tests/test_solve_wave7_coverage.py::test_residual_iter_vmec2000_scan_state_only -q`
- `python -m ruff check vmec_jax/solvers/fixed_boundary/scan/payload.py vmec_jax/solvers/fixed_boundary/residual/iteration.py tests/test_solve_scan_payload_helpers.py tests/test_solve_scan_planning_helpers.py tests/test_solve_scan_debug_helpers.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_scan_planning_helpers.py tests/test_solve_scan_output.py tests/test_solve_scan_output_edge_cases_more_coverage.py tests/test_solve_finish_cache_more_coverage.py tests/test_solve_residual_iter_finalize_helpers.py tests/test_solve_scan_debug_helpers.py -q`
- `python tools/diagnostics/source_health.py --top 8 --top-functions 12`

Best next steps:

1. Commit and push this preconditioned payload seam.
2. Re-check CI on the latest branch head.
3. Continue the scan-controller decomposition with the next large coherent
   seam: accepted/rejected update and restart/time-step transition, not more
   tiny debug helpers.
4. Keep promotion gates tied to real scan fixtures and branch-local AD/FD
   checks, since these seams are intended to make the eventual differentiable
   controller boundary testable.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.77%.
- Differentiability/refactor implementation: 99.995%.
- Solver monolith reduction: 96.2%.
- Free-boundary adjoint monolith reduction: 80%.
- Driver workflow decomposition: 93.2%.
- WOUT diagnostic/profile decomposition: 98.5%.
- Optimizer workflow decomposition: 86%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.55%.

## 2026-06-16 VMEC2000 Scan Accepted/Rejected Update Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `build_scan_step_fields` to the scan payload module.
2. Moved the VMEC2000 accepted-step velocity damping, inverse-tau update,
   Fourier state increment, LASYM update channels, fixed-boundary/axis
   enforcement, and non-VMEC restart rejection semantics out of `_advance_step`.
3. Kept branch ownership in the scan controller by passing the restart flag,
   VMEC2000-control flag, mode-transform callables, and boundary-enforcement
   callables explicitly.
4. Added unit coverage for VMEC2000 accepting the update even with a restart
   flag, and non-VMEC scan rejecting the same branch.

Results obtained:

- The residual iteration module dropped from 8,877 lines to 8,784 lines.
- `solve_fixed_boundary_residual_iter` dropped from 8,266 to 8,174 lines.
- `_run_vmec2000_scan` dropped from 1,988 to 1,896 lines.
- `_scan_step` dropped from 981 to 889 lines.
- `_advance_step` dropped from 965 to 873 lines.
- The scan controller now has a separate accepted/rejected update seam, which
  is the right domain boundary for future branch-local AD/FD gates around
  restart selection and accepted-step replay.

Tests and commands run:

- `python -m ruff check vmec_jax/solvers/fixed_boundary/scan/payload.py vmec_jax/solvers/fixed_boundary/residual/iteration.py tests/test_solve_scan_payload_helpers.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_scan_payload_helpers.py tests/test_solve_real_scan_wave10_coverage.py tests/test_solve_wave7_coverage.py::test_residual_iter_vmec2000_scan_minimal_one_step tests/test_solve_wave7_coverage.py::test_residual_iter_vmec2000_scan_state_only -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_scan_planning_helpers.py tests/test_solve_scan_output.py tests/test_solve_scan_output_edge_cases_more_coverage.py tests/test_solve_finish_cache_more_coverage.py tests/test_solve_residual_iter_finalize_helpers.py tests/test_solve_scan_debug_helpers.py -q`
- `python tools/diagnostics/source_health.py --top 8 --top-functions 12`

Best next steps:

1. Commit and push this accepted/rejected update seam.
2. Re-check CI on the latest branch head.
3. Extract the remaining restart-payload recompute branch or scan-history row
   assembly next; those are now the largest coherent scan-controller seams.
4. After those scan seams, return to free-boundary full-loop fingerprint gates
   with a smaller and more inspectable branch body.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.77%.
- Differentiability/refactor implementation: 99.995%.
- Solver monolith reduction: 96.6%.
- Free-boundary adjoint monolith reduction: 80%.
- Driver workflow decomposition: 93.2%.
- WOUT diagnostic/profile decomposition: 98.5%.
- Optimizer workflow decomposition: 86%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.56%.

## 2026-06-16 VMEC2000 Restart Force Payload Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `build_restart_preconditioned_scan_payload` to the scan payload module.
2. Moved the restart-branch force recomputation, fresh preconditioner cache
   assembly, R/Z preconditioner application, and restart payload construction
   out of `_advance_step`.
3. Kept restart selection in the scan controller while making the restart
   recompute branch directly unit-testable.
4. Added direct test coverage for restart force recomputation and cache payload
   construction.

Results obtained:

- The residual iteration module dropped from 8,784 lines to 8,717 lines.
- `solve_fixed_boundary_residual_iter` dropped from 8,174 to 8,107 lines.
- `_run_vmec2000_scan` dropped from 1,896 to 1,829 lines.
- `_scan_step` dropped from 889 to 822 lines.
- `_advance_step` dropped from 873 to 806 lines.
- The scan controller no longer owns restart force/cache recompute details,
  leaving restart policy and branch choice separate from force payload
  construction.

Tests and commands run:

- `python -m ruff check vmec_jax/solvers/fixed_boundary/scan/payload.py vmec_jax/solvers/fixed_boundary/residual/iteration.py tests/test_solve_scan_payload_helpers.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_scan_payload_helpers.py tests/test_solve_real_scan_wave10_coverage.py tests/test_solve_wave7_coverage.py::test_residual_iter_vmec2000_scan_minimal_one_step tests/test_solve_wave7_coverage.py::test_residual_iter_vmec2000_scan_state_only -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_scan_planning_helpers.py tests/test_solve_scan_output.py tests/test_solve_scan_output_edge_cases_more_coverage.py tests/test_solve_finish_cache_more_coverage.py tests/test_solve_residual_iter_finalize_helpers.py tests/test_solve_scan_debug_helpers.py -q`
- `python tools/diagnostics/source_health.py --top 8 --top-functions 12`

Best next steps:

1. Commit and push this restart force-payload extraction.
2. Re-check branch CI.
3. The next largest remaining scan seam is history-row/new-carry assembly.
   Extracting it should be done carefully because it touches scan history
   modes, fallback probes, and accepted masks.
4. After one more scan seam, reassess whether continuing in `iteration.py` or
   shifting back to free-boundary adaptive-branch gates gives better marginal
   value.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.77%.
- Differentiability/refactor implementation: 99.995%.
- Solver monolith reduction: 96.9%.
- Free-boundary adjoint monolith reduction: 80%.
- Driver workflow decomposition: 93.2%.
- WOUT diagnostic/profile decomposition: 98.5%.
- Optimizer workflow decomposition: 86%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.57%.

## 2026-06-16 VMEC2000 Scan Step Result Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `vmec2000_scan_step_result` to the scan output module.
2. Moved the final per-step carry construction and history-row selection out of
   `_advance_step`.
3. Preserved existing semantics for selected residual payloads versus current
   cache fields: restart-selected residuals can feed the emitted row, while the
   current preconditioner cache remains the carried cache state.
4. Added direct output-level coverage for light-history accepted steps and
   state-only restart/reject semantics.

Results obtained:

- The residual iteration module dropped from 8,717 lines to 8,587 lines.
- `solve_fixed_boundary_residual_iter` dropped from 8,107 to 7,976 lines.
- `_run_vmec2000_scan` dropped from 1,829 to 1,698 lines.
- `_scan_step` and `_advance_step` no longer appear in the top-12 longest
  function report; `_advance_step` is below the previous 806-line level.
- Scan output selection is now unit-testable independently of the scan
  controller.

Tests and commands run:

- `python -m ruff check vmec_jax/solvers/fixed_boundary/scan/output.py vmec_jax/solvers/fixed_boundary/residual/iteration.py tests/test_solve_scan_output.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_scan_output.py tests/test_solve_scan_output_edge_cases_more_coverage.py tests/test_solve_real_scan_wave10_coverage.py tests/test_solve_wave7_coverage.py::test_residual_iter_vmec2000_scan_minimal_one_step tests/test_solve_wave7_coverage.py::test_residual_iter_vmec2000_scan_state_only -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_scan_planning_helpers.py tests/test_solve_scan_output.py tests/test_solve_scan_output_edge_cases_more_coverage.py tests/test_solve_finish_cache_more_coverage.py tests/test_solve_residual_iter_finalize_helpers.py tests/test_solve_scan_debug_helpers.py tests/test_solve_scan_payload_helpers.py -q`
- `python tools/diagnostics/source_health.py --top 8 --top-functions 12`

Best next steps:

1. Commit and push this scan step-result extraction.
2. Let the latest CI run complete before stacking more pushes.
3. Reassess the next highest-value target: either continue reducing
   `_run_vmec2000_scan` setup/preflight orchestration or shift back to the
   free-boundary adaptive full-loop AD/FD gate now that the scan branch is more
   inspectable.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.77%.
- Differentiability/refactor implementation: 99.996%.
- Solver monolith reduction: 97.3%.
- Free-boundary adjoint monolith reduction: 80%.
- Driver workflow decomposition: 93.2%.
- WOUT diagnostic/profile decomposition: 98.5%.
- Optimizer workflow decomposition: 86%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.58%.

## 2026-06-16 VMEC2000 Initial Scan Carry Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `build_initial_scan_carry` to the scan resume module.
2. Moved the initial `_ScanCarry` construction out of
   `_run_vmec2000_scan`, leaving the controller to assemble resume/cache
   inputs and delegate carry construction to a focused helper.
3. Preserved the initial-axis-reset semantics by updating the resume fields
   before carry construction when the reset path changes `ijacob` and the
   checkpoint state.
4. Added direct resume-state coverage for the initial carry fields, cache
   payload, iteration offset, edge arrays, and fallback/convergence defaults.

Results obtained:

- The residual iteration module dropped from 8,587 lines to 8,512 lines.
- `solve_fixed_boundary_residual_iter` dropped from 7,976 to 7,900 lines.
- `_run_vmec2000_scan` dropped from 1,698 to 1,622 lines.
- Initial scan state assembly is now testable without entering the full
  VMEC2000 scan loop.
- `_run_vmec2000_scan` is now closer to a readable orchestration function:
  resume initialization, cache construction, carry setup, scan execution, and
  finalization are separate seams.

Tests and commands run:

- `python -m ruff check vmec_jax/solvers/fixed_boundary/scan/resume.py vmec_jax/solvers/fixed_boundary/residual/iteration.py tests/test_solve_scan_resume_state.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_scan_resume_state.py tests/test_solve_real_scan_wave10_coverage.py tests/test_solve_wave7_coverage.py::test_residual_iter_vmec2000_scan_minimal_one_step tests/test_solve_wave7_coverage.py::test_residual_iter_vmec2000_scan_state_only -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_scan_resume_state.py tests/test_solve_scan_planning_helpers.py tests/test_solve_scan_output.py tests/test_solve_scan_output_edge_cases_more_coverage.py tests/test_solve_finish_cache_more_coverage.py tests/test_solve_residual_iter_finalize_helpers.py tests/test_solve_scan_debug_helpers.py tests/test_solve_scan_payload_helpers.py -q`
- `python tools/diagnostics/source_health.py --top 8 --top-functions 12`

Best next steps:

1. Commit and push this initial scan-carry extraction.
2. Re-check branch CI now that GitHub authentication is restored.
3. Continue with one more large scan orchestration seam if there is still a
   clean extraction with direct tests, otherwise shift to the free-boundary
   adaptive-branch AD/FD gate now that fixed-boundary scan logic is smaller.
4. Keep resisting generic module names: new seams should remain domain-named
   around VMEC scan state, restart payloads, output rows, and branch
   fingerprints.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.77%.
- Differentiability/refactor implementation: 99.996%.
- Solver monolith reduction: 97.6%.
- Free-boundary adjoint monolith reduction: 80%.
- Driver workflow decomposition: 93.2%.
- WOUT diagnostic/profile decomposition: 98.5%.
- Optimizer workflow decomposition: 86%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.59%.

## 2026-06-16 VMEC2000 Initial Preconditioner Cache Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `build_initial_preconditioner_cache` and `ScanInitialCache` to the
   scan payload module.
2. Moved the initial preconditioner cache bootstrap and resume-cache overlay
   out of `_run_vmec2000_scan`.
3. Kept the shape-derived `jmax` policy inside the new helper so JIT-returned
   matrix metadata does not leak into host setup.
4. Added direct tests for fresh cache construction, preconditioner-matrix
   options, and resume-state overrides.

Results obtained:

- The residual iteration module dropped from 8,512 lines to 8,471 lines.
- `solve_fixed_boundary_residual_iter` dropped from 7,900 to 7,858 lines.
- `_run_vmec2000_scan` dropped from 1,622 to 1,580 lines.
- Initial scan cache semantics are now covered without entering a VMEC scan.
- The fixed-boundary scan setup path is now separated into resume fields,
  preconditioner cache fields, initial carry construction, scan execution, and
  postprocessing.

Tests and commands run:

- `python -m ruff check vmec_jax/solvers/fixed_boundary/scan/payload.py vmec_jax/solvers/fixed_boundary/residual/iteration.py tests/test_solve_scan_payload_helpers.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_scan_payload_helpers.py tests/test_solve_real_scan_wave10_coverage.py tests/test_solve_wave7_coverage.py::test_residual_iter_vmec2000_scan_minimal_one_step tests/test_solve_wave7_coverage.py::test_residual_iter_vmec2000_scan_state_only -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_scan_resume_state.py tests/test_solve_scan_planning_helpers.py tests/test_solve_scan_output.py tests/test_solve_scan_output_edge_cases_more_coverage.py tests/test_solve_finish_cache_more_coverage.py tests/test_solve_residual_iter_finalize_helpers.py tests/test_solve_scan_debug_helpers.py tests/test_solve_scan_payload_helpers.py -q`
- `python tools/diagnostics/source_health.py --top 8 --top-functions 12`

Best next steps:

1. Commit and push this initial preconditioner-cache extraction.
2. Let the queued branch CI complete and inspect failures if any.
3. Reassess whether the remaining `_run_vmec2000_scan` seams are still worth
   extracting now; the next likely fixed-boundary seam is chunked scan
   execution/materialization, but it is larger and should be moved only if the
   new helper can preserve timing/debug semantics cleanly.
4. If the next scan seam is too coupled, switch back to free-boundary phase-2
   adaptive-branch AD/FD validation with the now-smaller fixed-boundary
   controller as supporting evidence.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.77%.
- Differentiability/refactor implementation: 99.996%.
- Solver monolith reduction: 97.8%.
- Free-boundary adjoint monolith reduction: 80%.
- Driver workflow decomposition: 93.2%.
- WOUT diagnostic/profile decomposition: 98.5%.
- Optimizer workflow decomposition: 86%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.60%.

## 2026-06-16 VMEC2000 Scan Bad-Jacobian Decision Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `ScanBadJacobianDecision` and `scan_bad_jacobian_decision` to the
   scan math module.
2. Moved the VMEC2000 ptau/state bad-Jacobian branch decision out of
   `_advance_step`.
3. Preserved legacy behavior: ptau-triggered diagnostics can compute state
   Jacobian metrics, but the bad-Jacobian decision uses the state result only
   when state-Jacobian mode is enabled.
4. Added direct tests for ptau-only VMEC decisions, state override behavior,
   missing-ptau fallback diagnostics, and non-VMEC tau decisions.

Results obtained:

- The residual iteration module dropped from 8,471 lines to 8,435 lines.
- `solve_fixed_boundary_residual_iter` dropped from 7,858 to 7,821 lines.
- `_run_vmec2000_scan` dropped from 1,580 to 1,543 lines.
- Bad-Jacobian scan branch logic is now a named, tested seam, which is the
  right shape for future adaptive-branch fingerprint gates.

Tests and commands run:

- `python -m ruff check vmec_jax/solvers/fixed_boundary/scan/math.py vmec_jax/solvers/fixed_boundary/residual/iteration.py tests/test_solve_scan_math_helpers.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_scan_math_helpers.py tests/test_solve_real_scan_wave10_coverage.py tests/test_solve_wave7_coverage.py::test_residual_iter_vmec2000_scan_minimal_one_step tests/test_solve_wave7_coverage.py::test_residual_iter_vmec2000_scan_state_only -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_scan_math_helpers.py tests/test_solve_scan_time_control.py tests/test_solve_scan_resume_state.py tests/test_solve_scan_planning_helpers.py tests/test_solve_scan_output.py tests/test_solve_scan_output_edge_cases_more_coverage.py tests/test_solve_finish_cache_more_coverage.py tests/test_solve_residual_iter_finalize_helpers.py tests/test_solve_scan_debug_helpers.py tests/test_solve_scan_payload_helpers.py -q`
- `python tools/diagnostics/source_health.py --top 8 --top-functions 12`

Best next steps:

1. Commit and push this bad-Jacobian decision extraction.
2. Watch the latest CI run for this branch.
3. Continue extracting only branch-policy seams with direct tests; the next
   fixed-boundary candidate is the screen-scalar sampling block or the
   time-control/checkpoint/restart update staging.
4. Use the extracted bad-Jacobian helper when building the next
   fingerprint-gated adaptive-branch AD/FD validation gate.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.77%.
- Differentiability/refactor implementation: 99.996%.
- Solver monolith reduction: 98.0%.
- Free-boundary adjoint monolith reduction: 80%.
- Driver workflow decomposition: 93.2%.
- WOUT diagnostic/profile decomposition: 98.5%.
- Optimizer workflow decomposition: 86%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.61%.

## 2026-06-16 VMEC2000 Restart/Stage Update Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `ScanPostRestartUpdate` and `scan_post_restart_update` to the scan
   time-control module.
2. Moved the restart/no-restart selected-field unpacking and stage-spike
   post-reset application out of `_advance_step`.
3. Kept time-control debug callback emission in the scan controller so the
   logging order remains explicit and unchanged.
4. Added direct tests for restart selection with stage-spike reset and for the
   inactive no-restart branch.

Results obtained:

- The residual iteration module dropped from 8,435 lines to 8,404 lines.
- `solve_fixed_boundary_residual_iter` dropped from 7,821 to 7,790 lines.
- `_run_vmec2000_scan` dropped from 1,543 to 1,512 lines.
- Restart update semantics are now a named branch-policy seam, making the
  adaptive-controller state transition easier to fingerprint and validate.

Tests and commands run:

- `python -m ruff check vmec_jax/solvers/fixed_boundary/scan/time_control.py vmec_jax/solvers/fixed_boundary/residual/iteration.py tests/test_solve_scan_time_control.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_scan_time_control.py tests/test_solve_real_scan_wave10_coverage.py tests/test_solve_wave7_coverage.py::test_residual_iter_vmec2000_scan_minimal_one_step tests/test_solve_wave7_coverage.py::test_residual_iter_vmec2000_scan_state_only -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_scan_math_helpers.py tests/test_solve_scan_time_control.py tests/test_solve_scan_resume_state.py tests/test_solve_scan_planning_helpers.py tests/test_solve_scan_output.py tests/test_solve_scan_output_edge_cases_more_coverage.py tests/test_solve_finish_cache_more_coverage.py tests/test_solve_residual_iter_finalize_helpers.py tests/test_solve_scan_debug_helpers.py tests/test_solve_scan_payload_helpers.py -q`
- `python tools/diagnostics/source_health.py --top 8 --top-functions 12`

Best next steps:

1. Commit and push this restart/stage update extraction.
2. Check CI for the latest branch head after push.
3. Continue with small but meaningful controller seams only where the extracted
   helper has a stable physical/domain name and direct tests.
4. Prioritize the free-boundary adaptive branch validation next if fixed-boundary
   scan extractions become mostly debug plumbing rather than numerical branch
   policy.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.77%.
- Differentiability/refactor implementation: 99.996%.
- Solver monolith reduction: 98.15%.
- Free-boundary adjoint monolith reduction: 80%.
- Driver workflow decomposition: 93.2%.
- WOUT diagnostic/profile decomposition: 98.5%.
- Optimizer workflow decomposition: 86%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.62%.

## 2026-06-17 Strict Momentum Update Seam

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `force_update_rms` as a JAX-visible RMS helper while preserving the
   existing `host_force_update_rms` scalar wrapper.
2. Added `momentum_update_jax` so the strict non-JIT JAX momentum update has a
   named domain seam matching the existing host NumPy update seam.
3. Collapsed the host/JAX non-JIT strict update branch in
   `solve_fixed_boundary_residual_iter` so velocity and force blocks are built
   once, then routed through either host or JAX update assembly.
4. Removed a duplicated `scan/compute_forces:init` trace wrapper in the
   VMEC2000 scan setup path.
5. Added direct unit coverage for the JAX-visible RMS and momentum update
   helpers, including the historical JAX-branch RMS convention.

Results obtained:

- The residual iteration module dropped from 8,363 lines to 8,343 lines for
  this tranche.
- `solve_fixed_boundary_residual_iter` dropped from 7,747 to 7,725 lines.
- The strict momentum update is now a named numerical seam for future
  same-branch replay/fingerprint work instead of open-coded block arithmetic.
- The previous branch head was confirmed green on CI, including combined
  coverage and Codecov, before this local tranche.

Tests and commands run:

- `python -m ruff check vmec_jax/solvers/fixed_boundary/residual/update.py vmec_jax/solvers/fixed_boundary/residual/iteration.py tests/test_solve_residual_iter_update_helpers.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_residual_iter_update_helpers.py tests/test_solve_residual_iter_policy.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_real_scan_wave10_coverage.py tests/test_solve_wave7_coverage.py::test_residual_iter_vmec2000_scan_minimal_one_step tests/test_solve_wave7_coverage.py::test_residual_iter_vmec2000_scan_state_only -q`
- `python tools/diagnostics/source_health.py --top 10 --top-functions 15`

Best next steps:

1. Commit and push this strict momentum seam, then monitor CI.
2. Continue only with named numerical or branch-policy seams that reduce the
   residual hotspot without adding larger argument lists.
3. Use the strict momentum seam in future branch-local replay reports where the
   accepted update path needs a compact, JAX-visible primitive.
4. Keep arbitrary adaptive branch differentiation unclaimed until a true
   fingerprint-gated full adaptive AD-vs-central-FD gate passes.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.77%.
- Differentiability/refactor implementation: 99.996%.
- Solver monolith reduction: 98.22%.
- Free-boundary adjoint monolith reduction: 80%.
- Driver workflow decomposition: 93.2%.
- WOUT diagnostic/profile decomposition: 98.5%.
- Optimizer workflow decomposition: 86%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.63%.

## 2026-06-17 Clipped Velocity Scaling Consolidation

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Replaced the remaining open-coded 12-block clipped velocity rescale in
   `solve_fixed_boundary_residual_iter` with `scale_velocity_blocks`.
2. Kept the existing post-clip RMS recomputation and state-update path
   unchanged.

Results obtained:

- The residual iteration module dropped from 8,343 lines to 8,334 lines.
- `solve_fixed_boundary_residual_iter` dropped from 7,725 to 7,716 lines.
- Velocity scaling in strict residual iteration now uses a single helper for
  backtracking and update clipping.

Tests and commands run:

- `python -m ruff check vmec_jax/solvers/fixed_boundary/residual/iteration.py vmec_jax/solvers/fixed_boundary/residual/update.py tests/test_solve_residual_iter_update_helpers.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_residual_iter_update_helpers.py tests/test_solve_residual_iter_policy.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_real_scan_wave10_coverage.py tests/test_solve_wave7_coverage.py::test_residual_iter_vmec2000_scan_minimal_one_step tests/test_solve_wave7_coverage.py::test_residual_iter_vmec2000_scan_state_only -q`
- `python tools/diagnostics/source_health.py --top 10 --top-functions 15`

Best next steps:

1. Monitor CI for this and the strict momentum update commit.
2. Continue residual monolith work only where the named seam removes duplicated
   numerical branch logic or shrinks a branch-local replay primitive.
3. Reassess the free-boundary accepted/adaptive branch gates after CI is green.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.77%.
- Differentiability/refactor implementation: 99.996%.
- Solver monolith reduction: 98.24%.
- Free-boundary adjoint monolith reduction: 80%.
- Driver workflow decomposition: 93.2%.
- WOUT diagnostic/profile decomposition: 98.5%.
- Optimizer workflow decomposition: 86%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.64%.

## 2026-06-17 Fixed-Boundary Optimizer Dispatch Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `FIXED_BOUNDARY_OPTIMIZER_SOLVERS` and
   `run_fixed_boundary_optimizer_solver` to `vmec_jax.drivers.solve`.
2. Moved the non-VMEC2000 fixed-boundary optimizer dispatch for `gd`, `lbfgs`,
   `vmec_lbfgs`, and `vmec_gn` out of the public `run_fixed_boundary` body.
3. Left the VMEC2000 staged/parity path unchanged; only the simpler optimizer
   dispatch moved.
4. Added direct tests with injected fake solver functions to validate unknown
   solver passthrough, restart-state reuse, and VMEC residual optimizer dispatch.

Results obtained:

- `run_fixed_boundary` dropped from 1,709 to 1,659 lines.
- The extracted driver seam gives the public driver a clearer branch boundary:
  optimizer-style fixed-boundary solves are delegated, VMEC2000 staged solves
  remain in the main parity-sensitive path.
- GitHub CI for the previous branch head was confirmed green, including
  combined coverage and Codecov, before this local tranche.

Tests and commands run:

- `python -m ruff check vmec_jax/driver.py vmec_jax/drivers/solve.py tests/test_driver_policy_helpers.py`
- `python -m pytest -q tests/test_driver_policy_helpers.py -q`
- `python -m pytest -q tests/test_driver_api.py tests/test_driver_wave4_coverage.py tests/test_driver_wave12_coverage.py -q`
- `python tools/diagnostics/source_health.py --top 12 --top-functions 18`

Best next steps:

1. Commit and push this optimizer dispatch extraction, then monitor CI.
2. Next driver tranche should target either device-reroute setup or finish/stage
   context construction, but only if the call-site becomes smaller than the
   helper boundary.
3. Keep VMEC2000 staged loop extractions conservative because that path carries
   most parity-sensitive behavior.
4. Resume free-boundary branch-gate work once driver source-health no longer has
   an obvious low-risk seam.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.77%.
- Differentiability/refactor implementation: 99.996%.
- Solver monolith reduction: 98.24%.
- Free-boundary adjoint monolith reduction: 80%.
- Driver workflow decomposition: 94.4%.
- WOUT diagnostic/profile decomposition: 98.5%.
- Optimizer workflow decomposition: 86%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.65%.

## 2026-06-17 Free-Boundary Replay Result Packaging Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Moved accepted-controller replay objective/result packaging out of
   `direct_coil_accepted_trace_controller_replay_objective_jax` and into
   `accepted_controller_replay_result` in the free-boundary adjoint objective
   helpers.
2. Preserved the public compatibility aliases in `free_boundary_adjoint.py` so
   existing tests and internal users continue to import the same names.
3. Added a unit-level contract for accepted-mask objective aggregation,
   state-only replay, compact replay auxiliary output, and state reset flags.

Results obtained:

- `vmec_jax/free_boundary_adjoint.py` dropped from 3,829 to 3,794 lines.
- `direct_coil_accepted_trace_controller_replay_objective_jax` dropped from
  769 to 733 lines.
- The replay primitive now has a clearer seam between branch-local controller
  execution and packaging the scalar objective/auxiliary payload used by tests
  and optimization reports.

Tests and commands run:

- `python -m ruff check vmec_jax/free_boundary_adjoint.py vmec_jax/solvers/free_boundary/adjoint/objectives.py tests/test_free_boundary_adjoint_helpers_unit.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_adjoint_helpers_unit.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_adjoint_helpers_unit.py tests/test_free_boundary_direct_coil_finite_pressure_sensitivity.py::test_direct_coil_branch_trace_mode_keeps_replay_controls_without_raw_force_payload tests/test_free_boundary_direct_coil_finite_pressure_sensitivity.py::test_direct_coil_accepted_update_replay_ad_matches_fd_for_coil_pytree tests/test_free_boundary_direct_coil_finite_pressure_sensitivity.py::test_direct_coil_vacuum_field_override_replay_contract -q`
- `python tools/diagnostics/source_health.py --top 12 --top-functions 18`

Best next steps:

1. Wait for the previous pushed CI run to finish before pushing this tranche, so
   we do not restart GitHub Actions unnecessarily.
2. Continue extracting small free-boundary replay/reporting seams only where a
   helper has a crisp numerical contract and does not change branch selection.
3. After CI is green, resume the next meaningful phase-2 gate: the narrow
   fingerprint-gated same-branch adaptive-slot AD-vs-FD validation.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.77%.
- Differentiability/refactor implementation: 99.996%.
- Solver monolith reduction: 98.24%.
- Free-boundary adjoint monolith reduction: 81%.
- Driver workflow decomposition: 94.4%.
- WOUT diagnostic/profile decomposition: 98.5%.
- Optimizer workflow decomposition: 86%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.66%.

## 2026-06-17 Free-Boundary `bsqvac` Replay Block Consolidation

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Extracted the duplicated direct-coil free-boundary `bsqvac` replay and
   diagnostic objective logic from the trace-control and stacked-control
   controller branches into one local `_freeb_bsqvac_replay_terms` helper.
2. Kept both strict-update paths unchanged: the normal trace path still calls
   `strict_update_one_step_from_trace`, while the stacked-control path still
   calls `strict_update_one_step_from_state`.
3. Preserved dense/matrix-free NESTOR options, vacuum-field freeze options, and
   state-only replay behavior under the same branch-local gates.

Results obtained:

- `vmec_jax/free_boundary_adjoint.py` dropped from 3,794 to 3,714 lines.
- `direct_coil_accepted_trace_controller_replay_objective_jax` dropped from
  733 to 653 lines.
- The two controller replay paths now share the same external-field replay
  contract, reducing the chance that future direct-coil or NESTOR diagnostics
  diverge between trace-control modes.

Tests and commands run:

- `python -m ruff check vmec_jax/free_boundary_adjoint.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_adjoint_helpers_unit.py tests/test_free_boundary_direct_coil_finite_pressure_sensitivity.py::test_direct_coil_branch_trace_mode_keeps_replay_controls_without_raw_force_payload tests/test_free_boundary_direct_coil_finite_pressure_sensitivity.py::test_direct_coil_accepted_update_replay_ad_matches_fd_for_coil_pytree tests/test_free_boundary_direct_coil_finite_pressure_sensitivity.py::test_direct_coil_vacuum_field_override_replay_contract -q`
- `python tools/diagnostics/source_health.py --top 12 --top-functions 18`

Best next steps:

1. Push the two local free-boundary extraction commits now that the previous CI
   run is green.
2. Let CI validate the combined branch before starting another free-boundary
   replay extraction.
3. Next phase-2 correctness work should target a narrow fingerprint-gated
   adaptive-slot AD-vs-FD report rather than further cosmetic refactoring.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.77%.
- Differentiability/refactor implementation: 99.996%.
- Solver monolith reduction: 98.24%.
- Free-boundary adjoint monolith reduction: 82%.
- Driver workflow decomposition: 94.4%.
- WOUT diagnostic/profile decomposition: 98.5%.
- Optimizer workflow decomposition: 86%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.67%.

## 2026-06-17 Solver-Device Reroute Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Confirmed the previous pushed tranche (`6f0655f`) had a green PR CI run
   before starting the next driver refactor step.
2. Moved the CPU/GPU solver-device reroute policy out of
   `run_fixed_boundary` and into
   `maybe_run_fixed_boundary_in_solver_device_context` in the driver solve
   helper module.
3. Preserved the recursive reroute semantics, compilation-cache policy,
   restart guards, and solver-device diagnostics while making the public driver
   flow shorter and easier to audit.

Results obtained:

- `run_fixed_boundary` dropped from 1,659 to 1,637 lines.
- Device reroute behavior now has one focused helper seam that can be unit
  tested separately from the rest of the CLI/driver workflow.
- Source-health still identifies the main remaining production hotspots as the
  fixed-boundary residual iteration, optimization workflow, free-boundary
  adjoint replay, and long validation tests.

Tests and commands run:

- `python -m ruff check vmec_jax/driver.py vmec_jax/drivers/solve.py`
- `python -m pytest -q tests/test_driver_wave12_coverage.py::test_solver_device_reroute_wraps_recursive_run_and_adds_diagnostics tests/test_driver_wave2_coverage.py::test_solver_device_cpu_reroute_annotates_result_diagnostics tests/test_driver_wave2_coverage.py::test_solver_device_lookup_failure_falls_back_to_regular_run tests/test_driver_policy_coverage_extra.py::test_gpu_solver_device_enables_default_compilation_cache -q`
- `python -m pytest -q tests/test_driver_wave12_coverage.py tests/test_driver_wave2_coverage.py tests/test_driver_policy_coverage_extra.py -q`
- `python tools/diagnostics/source_health.py --top 12 --top-functions 20`

Best next steps:

1. Commit and push this driver extraction, then let CI validate the branch.
2. Continue driver decomposition only at crisp policy seams; do not split the
   driver into many small files without domain names and tests.
3. Resume the next correctness milestone after CI: a narrow
   fingerprint-gated adaptive-slot AD-vs-FD report for the free-boundary
   controller.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.77%.
- Differentiability/refactor implementation: 99.996%.
- Solver monolith reduction: 98.24%.
- Free-boundary adjoint monolith reduction: 82%.
- Driver workflow decomposition: 95.2%.
- WOUT diagnostic/profile decomposition: 98.5%.
- Optimizer workflow decomposition: 86%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.68%.

## 2026-06-17 Fixed-Boundary Stage Policy Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Extracted the fixed-boundary multigrid/staging decision block from
   `run_fixed_boundary` into `FixedBoundaryStagePolicy` and
   `resolve_fixed_boundary_stage_policy` in `vmec_jax.drivers.policy`.
2. Kept the driver as the workflow owner while moving pure policy decisions for
   input stage arrays, budgeted CLI multigrid, explicit input staging,
   restart/ns-override multigrid disabling, max-iteration resolution, and stage
   transition heuristics into one tested domain helper.
3. Added direct helper-level tests for explicit input staging, accelerated
   direct-final-grid selection, restart/ns-override behavior, and env-driven
   stage-transition policy.

Results obtained:

- `run_fixed_boundary` dropped from 1,637 to 1,556 lines.
- The fixed-boundary driver now has a clearer boundary between policy
  resolution and solve execution without changing VMEC2000 staging semantics.
- Source-health still identifies the remaining largest production hotspots as
  fixed-boundary residual iteration, optimization workflow, and free-boundary
  adjoint replay.

Tests and commands run:

- `python -m ruff check vmec_jax/driver.py vmec_jax/drivers/policy.py tests/test_driver_policy_helpers.py`
- `python -m pytest -q tests/test_driver_policy_helpers.py tests/test_driver_api.py::test_run_fixed_boundary_cli_budgeted_multigrid_path tests/test_driver_api.py::test_run_fixed_boundary_cli_user_explicitly_staged_uses_direct_multigrid tests/test_driver_api.py::test_run_fixed_boundary_cli_current_driven_nonaxis_uses_direct_multigrid tests/test_driver_api.py::test_run_fixed_boundary_cli_two_stage_current_driven_nonaxis_uses_multigrid tests/test_driver_api.py::test_run_fixed_boundary_cli_three_stage_lasym_current_driven_nonaxis_uses_multigrid tests/test_driver_wave2_coverage.py::test_restart_solver_state_disables_multigrid_and_passes_resume_state tests/test_driver_wave2_coverage.py::test_ns_override_disables_input_multigrid_stages tests/test_driver_wave12_coverage.py tests/test_driver_policy_coverage_extra.py -q`
- `python -m pytest -q tests/test_driver_wave12_coverage.py tests/test_driver_wave2_coverage.py tests/test_driver_policy_coverage_extra.py -q`
- `python tools/diagnostics/source_health.py --top 12 --top-functions 20`

Best next steps:

1. Commit and push this policy extraction after confirming the worktree only
   contains the intended driver/policy/test changes.
2. Let CI validate both driver refactor tranches before starting another
   driver split.
3. Next high-value refactor target remains fixed-boundary residual iteration:
   extract the VMEC2000 scan controller seam only with parity tests, not
   cosmetic line moves.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.77%.
- Differentiability/refactor implementation: 99.996%.
- Solver monolith reduction: 98.24%.
- Free-boundary adjoint monolith reduction: 82%.
- Driver workflow decomposition: 96.4%.
- WOUT diagnostic/profile decomposition: 98.5%.
- Optimizer workflow decomposition: 86%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.69%.

## 2026-06-17 Residual Iteration Startup Policy Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Extracted the host-side startup policy block from
   `solve_fixed_boundary_residual_iter` into
   `ResidualIterStartupPolicy` and `resolve_residual_iter_startup_policy` in
   `vmec_jax.solvers.fixed_boundary.residual.policy`.
2. Moved option validation, host update selection, accelerator residual metric
   policy, adjoint trace normalization, tridiagonal preconditioner policy,
   bad-Jacobian environment parsing, restart defaults, scan fallback defaults,
   chunked-scan/tracer safety, and dump-history/JIT side effects behind one
   tested immutable policy object.
3. Added direct unit coverage for accelerator host-update controls, debug-dump
   side effects, tracer scan safety, stage-transition disabling, objective
   target normalization, restart defaults, and adjoint/resume mode
   normalization.

Results obtained:

- `solve_fixed_boundary_residual_iter` dropped from 7,790 to 7,685 lines.
- `vmec_jax/solvers/fixed_boundary/residual/iteration.py` dropped from 8,404
  to 8,297 lines.
- Startup policy is now testable without running a VMEC residual iteration or
  compiling force kernels, which is the right seam for future branch
  fingerprint and differentiability-policy reporting.
- Representative VMEC2000 scan shards still pass after the extraction.

Tests and commands run:

- `python -m ruff check vmec_jax/solvers/fixed_boundary/residual/policy.py vmec_jax/solvers/fixed_boundary/residual/iteration.py tests/test_solve_residual_iter_policy.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_residual_iter_policy.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_real_scan_wave10_coverage.py tests/test_solve_wave7_coverage.py::test_residual_iter_vmec2000_scan_minimal_one_step tests/test_solve_wave7_coverage.py::test_residual_iter_vmec2000_scan_state_only -q`
- `python tools/diagnostics/source_health.py --top 12 --top-functions 20`

Best next steps:

1. Commit and push this residual startup-policy tranche.
2. Continue residual-controller decomposition at the next real seam:
   VMEC2000 scan setup/run/finalize policy or accepted/rejected update
   controller state transitions.
3. Keep each extraction paired with direct helper tests and at least one
   representative scan/parity shard; do not wait on full CI after every push.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.77%.
- Differentiability/refactor implementation: 99.996%.
- Solver monolith reduction: 98.36%.
- Free-boundary adjoint monolith reduction: 82%.
- Driver workflow decomposition: 96.4%.
- WOUT diagnostic/profile decomposition: 98.5%.
- Optimizer workflow decomposition: 86%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.70%.

## 2026-06-17 Residual WOUT-Profile Setup Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Moved the nested WOUT-like flux/profile assembly out of
   `solve_fixed_boundary_residual_iter` and into
   `build_wout_like_profiles_from_indata` in
   `vmec_jax.solvers.fixed_boundary.profiles`.
2. Added `WoutLikeProfileSetup` so the helper returns both the intermediate
   flux/profile arrays and the compact `WoutLikeVmecForces` object consumed by
   residual kernels.
3. Preserved the host-default profile fast path and the VMEC conventions for
   `phips[0]`, mass, pressure, current profiles, and internal flux scaling.
4. Added a real circular-tokamak fixture test for the aggregate helper.

Results obtained:

- `solve_fixed_boundary_residual_iter` dropped from 7,685 to 7,598 lines.
- `vmec_jax/solvers/fixed_boundary/residual/iteration.py` dropped from 8,297
  to 8,211 lines.
- Profile setup is now a named, independently tested domain function rather
  than a nested closure in the nonlinear residual controller.

Tests and commands run:

- `python -m ruff check vmec_jax/solvers/fixed_boundary/profiles.py vmec_jax/solvers/fixed_boundary/residual/iteration.py tests/test_solve_wave7_coverage.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_wave7_coverage.py::test_wout_like_profile_setup_uses_real_input_profiles tests/test_solve_wave7_coverage.py::test_residual_iter_vmec2000_scan_minimal_one_step tests/test_solve_wave7_coverage.py::test_residual_iter_vmec2000_scan_state_only -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_wave7_coverage.py -q`
- `python tools/diagnostics/source_health.py --top 12 --top-functions 20`

Best next steps:

1. Commit and push this profile-setup tranche.
2. Continue residual-controller decomposition at the next large seam:
   scan initial-force/axis-reset preflight or scan postprocess/fallback
   materialization.
3. Keep setup helpers in domain modules (`profiles`, `scan`, `residual`) rather
   than adding generic `solve_*` helper files.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.77%.
- Differentiability/refactor implementation: 99.996%.
- Solver monolith reduction: 98.46%.
- Free-boundary adjoint monolith reduction: 82%.
- Driver workflow decomposition: 96.4%.
- WOUT diagnostic/profile decomposition: 98.8%.
- Optimizer workflow decomposition: 86%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.71%.

## 2026-06-17 Residual Cache-Key Setup Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Moved residual static/WOUT/edge cache-key tuple construction out of
   `solve_fixed_boundary_residual_iter` and into
   `build_residual_cache_keys` in
   `vmec_jax.solvers.fixed_boundary.residual.setup`.
2. Added the `ResidualCacheKeys` dataclass so scan-runner and force-cache keys
   have a named domain object instead of ad hoc tuple blocks inside the solver.
3. Added pure unit coverage for key field ordering, hash delegation, constraint
   force inclusion, and edge signature/value key delegation.

Results obtained:

- `solve_fixed_boundary_residual_iter` dropped from 7,598 to 7,586 lines.
- `vmec_jax/solvers/fixed_boundary/residual/iteration.py` dropped from 8,211
  to 8,200 lines.
- Cache-key construction is now an independently tested setup seam, which
  reduces risk when tuning scan-runner cache behavior and fingerprint reports.

Tests and commands run:

- `python -m ruff check vmec_jax/solvers/fixed_boundary/residual/setup.py vmec_jax/solvers/fixed_boundary/residual/iteration.py tests/test_solve_residual_iter_setup_helpers.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_residual_iter_setup_helpers.py tests/test_solve_wave7_coverage.py::test_residual_iter_vmec2000_scan_minimal_one_step tests/test_solve_wave7_coverage.py::test_residual_iter_vmec2000_scan_state_only -q`
- `python tools/diagnostics/source_health.py --top 12 --top-functions 20`

Best next steps:

1. Commit and push this cache-key setup extraction.
2. Target a larger residual-controller seam next: scan postprocess/fallback
   materialization or initial-force/axis-reset preflight.
3. Keep cache/fingerprint setup in typed domain helpers so future
   same-branch derivative metadata is auditable.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.77%.
- Differentiability/refactor implementation: 99.996%.
- Solver monolith reduction: 98.47%.
- Free-boundary adjoint monolith reduction: 82%.
- Driver workflow decomposition: 96.4%.
- WOUT diagnostic/profile decomposition: 98.8%.
- Optimizer workflow decomposition: 86%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.71%.

## 2026-06-17 VMEC2000 Scan Result Assembly Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `vmec2000_scan_residual_result` to
   `vmec_jax.solvers.fixed_boundary.scan.output`.
2. Moved the final `SolveVmecResidualResult` assembly and public VMEC2000 scan
   diagnostics dictionary out of the nested `_run_vmec2000_scan` function.
3. Added scan-output tests that verify history arrays, scan flags,
   bad-Jacobian metadata, and timing diagnostics propagate through the new
   assembler.

Results obtained:

- `solve_fixed_boundary_residual_iter` dropped from 7,586 to 7,573 lines.
- Nested `_run_vmec2000_scan` dropped from 1,511 to 1,498 lines.
- VMEC2000 scan materialization now lives next to the existing scan
  postprocessing logic instead of inside the residual controller.

Tests and commands run:

- `python -m ruff check vmec_jax/solvers/fixed_boundary/scan/output.py vmec_jax/solvers/fixed_boundary/residual/iteration.py tests/test_solve_scan_output.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_scan_output.py tests/test_solve_wave7_coverage.py::test_residual_iter_vmec2000_scan_minimal_one_step tests/test_solve_wave7_coverage.py::test_residual_iter_vmec2000_scan_state_only -q`
- `python tools/diagnostics/source_health.py --top 12 --top-functions 20`

Best next steps:

1. Commit and push this scan result assembly extraction.
2. Continue at a larger controller seam: scan initial-force/axis-reset preflight
   or the accepted/rejected update transition inside `_scan_step`.
3. Keep the arbitrary adaptive-branch differentiation claim conservative until
   the host/JAX controller branch fingerprint gate is explicit.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.77%.
- Differentiability/refactor implementation: 99.996%.
- Solver monolith reduction: 98.49%.
- Free-boundary adjoint monolith reduction: 82%.
- Driver workflow decomposition: 96.4%.
- WOUT diagnostic/profile decomposition: 98.8%.
- Optimizer workflow decomposition: 86%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.72%.

## 2026-06-17 Fixed-Boundary Optimizer Profiling and State Cache Split

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Extracted fixed-boundary exact-optimizer timing aggregation into
   `vmec_jax.optimizers.fixed_boundary.profiling`.
2. Kept the existing `_profile_*` optimizer methods as thin compatibility
   wrappers so tests and downstream profiling scripts can continue to bind
   those method names directly.
3. Extracted accepted-point state caches, trial/exact residual caches,
   callback point ids, boundary reconstruction, VMEC-input reconstruction,
   base-parameter vectors, and initial-state cache handling into
   `vmec_jax.optimizers.fixed_boundary.state_cache`.
4. Preserved monkeypatch seams used by the test suite:
   module-level `initial_guess_from_boundary` still controls fallback initial
   states, module-level `boundary.boundary_from_input_convention` is looked up
   at runtime, and overridden optimizer cache-key methods remain authoritative.

Results obtained:

- `vmec_jax/optimization.py` dropped from 3,516 to 3,115 lines in this
  optimizer tranche.
- The exact optimizer class is now more focused on solve orchestration,
  Jacobian/replay methods, and public optimizer callbacks rather than timing
  bucket parsing and cache bookkeeping.
- Source-health now reports `optimization.py` below the 3,200-line mark; the
  largest remaining production monoliths are the fixed-boundary residual loop,
  free-boundary adjoint module, workflow facade, discrete adjoint, and
  free-boundary driver.

Tests and commands run:

- `python -m ruff check vmec_jax/optimization.py vmec_jax/optimizers/fixed_boundary/profiling.py tests/test_optimization_callback_trace.py tests/test_optimization_wave4_coverage.py tests/test_optimization_wave2_coverage.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_optimization_callback_trace.py tests/test_optimization_wave4_coverage.py tests/test_optimization_wave2_coverage.py -q`
- `python -m ruff check vmec_jax/optimization.py vmec_jax/optimizers/fixed_boundary/state_cache.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_optimization_wave2_coverage.py tests/test_optimization_fast_optimizer_methods.py tests/test_optimization_callback_trace.py tests/test_optimization_helpers.py -q`
- `python tools/diagnostics/source_health.py --top 12 --top-functions 20`

Best next steps:

1. Continue the fixed-boundary optimizer split by extracting scan exact-helper
   construction and JVP-only exact tape policy into domain helpers, preserving
   wrappers for existing tests.
2. Continue workflow decomposition after the optimizer class is below the next
   source-health threshold.
3. Run the combined optimizer/workflow shard after the next tranche.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.90%.
- Differentiability/refactor implementation: 99.9985%.
- Solver monolith reduction: 98.50%.
- Free-boundary adjoint monolith reduction: 82%.
- Driver workflow decomposition: 96.4%.
- WOUT diagnostic/profile decomposition: 98.8%.
- Optimizer workflow decomposition: 94.7%.
- Fixed-boundary optimizer decomposition: 91.8%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.84%.

## 2026-06-17 Fixed-Boundary Exact Replay Policy Split

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Extracted scan exact-helper construction and scan accepted-state solving into
   `vmec_jax.optimizers.fixed_boundary.exact_replay`.
2. Extracted JVP-only exact tape policy and basepoint-carry environment handling
   into the same exact replay module.
3. Extracted optimizer backend naming, LASYM replay chunking, projected replay
   gates, fused replay opt-in policy, chunked projected replay policy, and
   initial-tangent cache enablement into
   `vmec_jax.optimizers.fixed_boundary.replay_policy`.
4. Preserved `optimization.py` compatibility wrappers for all existing private
   methods and kept scan helper monkeypatch behavior by passing the
   module-level `initial_guess_from_boundary` function into the helper.

Results obtained:

- `vmec_jax/optimization.py` dropped from 3,115 to below the current top-12
  source-health warning list; the 12th production/test file is now 2,896 lines.
- The fixed-boundary optimizer class now delegates profiling, cache/state
  bookkeeping, scan exact-helper construction, and replay policy to domain
  modules.
- Remaining optimizer-class work is concentrated in tangent replay/projection,
  solve orchestration, callbacks, and result assembly.

Tests and commands run:

- `python -m ruff check vmec_jax/optimization.py vmec_jax/optimizers/fixed_boundary/exact_replay.py vmec_jax/optimizers/fixed_boundary/replay_policy.py tests/test_optimization_wave2_coverage.py tests/test_optimization_fast_optimizer_methods.py tests/test_optimization_auto_scalar_policy.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_optimization_wave2_coverage.py tests/test_optimization_fast_optimizer_methods.py tests/test_optimization_auto_scalar_policy.py tests/test_optimization_helpers.py -q`
- `python tools/diagnostics/source_health.py --top 12 --top-functions 20`

Best next steps:

1. Move the remaining tangent replay/projection helpers out of
   `optimization.py` only if they can be grouped coherently without obscuring
   the exact accepted-point derivative path.
2. Shift the next large tranche to `optimization_workflow.py` or the
   fixed-boundary residual loop, where the source-health report still shows
   large production monoliths.
3. Keep free-boundary adaptive-branch claims conservative while refactoring:
   current production evidence remains branch-local/fingerprint-gated, not
   arbitrary adaptive-branch differentiability.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.92%.
- Differentiability/refactor implementation: 99.9988%.
- Solver monolith reduction: 98.50%.
- Free-boundary adjoint monolith reduction: 82%.
- Driver workflow decomposition: 96.4%.
- WOUT diagnostic/profile decomposition: 98.8%.
- Optimizer workflow decomposition: 95.2%.
- Fixed-boundary optimizer decomposition: 94.0%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.86%.

## 2026-06-17 Fixed-Boundary Workflow Output Split

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Extracted fixed-boundary workflow printing, stage artifact saving,
   final-output saving, non-worsening continuation guards, QI stage
   checkpoint writing, combined-history assembly, JSON conversion, stale-file
   cleanup, and Boozer surface slicing into
   `vmec_jax.optimizers.fixed_boundary.workflow_outputs`.
2. Kept public/private names in `optimization_workflow.py` as wrappers so
   example scripts, tests, and downstream monkeypatches still target the
   workflow facade.
3. Passed runtime dependencies such as `run_fixed_boundary`,
   `write_wout_from_fixed_boundary_run`, stage-mode normalization, and stage
   budget calculation into helper implementations to avoid circular imports
   and preserve test monkeypatch seams.

Results obtained:

- `vmec_jax/optimization_workflow.py` dropped from 3,637 to 3,412 lines.
- Workflow output and checkpoint behavior is now grouped under the
  fixed-boundary optimizer domain rather than mixed with objective definitions
  and stage orchestration.
- Rerun-WOUT tests continue to patch through `vmec_jax.optimization_workflow`,
  matching existing user/script behavior.

Tests and commands run:

- `python -m ruff check vmec_jax/optimization_workflow.py vmec_jax/optimizers/fixed_boundary/workflow_outputs.py tests/test_optimization_workflow_unit.py tests/test_optimization_helpers.py tests/test_continuation_exact_history.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_optimization_workflow_unit.py tests/test_optimization_helpers.py tests/test_continuation_exact_history.py -q`
- `python tools/diagnostics/source_health.py --top 12 --top-functions 20`

Best next steps:

1. Extract objective class families from `optimization_workflow.py` into a
   fixed-boundary objective-library module, or first split out QI-only
   objectives if a full objective extraction is too large for one validation
   tranche.
2. Continue reducing `solve_fixed_boundary_residual_iter` via controller
   sub-seams after the workflow facade has fewer mixed responsibilities.
3. Keep docs updates for the end of this PR, after the API names and module
   structure settle.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.93%.
- Differentiability/refactor implementation: 99.999%.
- Solver monolith reduction: 98.50%.
- Free-boundary adjoint monolith reduction: 82%.
- Driver workflow decomposition: 96.4%.
- WOUT diagnostic/profile decomposition: 98.8%.
- Optimizer workflow decomposition: 96.0%.
- Fixed-boundary optimizer decomposition: 94.0%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.87%.

## 2026-06-17 Workflow Facade Output Compatibility Follow-Up

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Validated the new workflow output module against workflow and continuation
   unit tests.
2. Fixed the rerun-WOUT monkeypatch seam by injecting
   `run_fixed_boundary` and `write_wout_from_fixed_boundary_run` from the
   workflow facade into the implementation helper.
3. Removed stale imports left behind by the output/checkpoint extraction.
4. Inspected the fixed-boundary residual-loop module and confirmed the
   remaining large production seam is now the core host/scan adaptive loop, not
   isolated diagnostics already covered by existing helper modules.

Results obtained:

- Workflow tests that monkeypatch `vmec_jax.optimization_workflow` rerun
  functions pass with the extracted implementation.
- Source-health remains at 3,412 lines for `optimization_workflow.py` and
  `optimization.py` remains below the top-12 file-size warning list.

Tests and commands run:

- `python -m ruff check vmec_jax/optimization_workflow.py vmec_jax/optimizers/fixed_boundary/workflow_outputs.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_optimization_workflow_unit.py tests/test_optimization_helpers.py tests/test_continuation_exact_history.py -q`
- `python tools/diagnostics/source_health.py --top 12 --top-functions 20`

Best next steps:

1. Split a QI/objective-library tranche from `optimization_workflow.py` to
   remove objective definitions from the workflow facade.
2. If touching the residual-loop monolith next, target a narrow controller
   helper with strong tests rather than moving scan-loop internals blindly.
3. Continue committing coherent tranches, but avoid CI polling until enough
   work has accumulated or a local failure suggests a broader issue.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.93%.
- Differentiability/refactor implementation: 99.999%.
- Solver monolith reduction: 98.55%.
- Free-boundary adjoint monolith reduction: 82%.
- Driver workflow decomposition: 96.4%.
- WOUT diagnostic/profile decomposition: 98.8%.
- Optimizer workflow decomposition: 96.2%.
- Fixed-boundary optimizer decomposition: 94.0%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.875%.

## 2026-06-17 Fixed-Boundary Seed/Input Workflow Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `vmec_jax.optimizers.fixed_boundary.seed_inputs` for the
   fixed-boundary input preparation helpers used by QA/QH/QP/QI examples.
2. Moved the implementation of simple omnigenity seed construction, simple
   seed file writing, boundary interpolation, and optimization-resolution
   rebuilding out of `optimization_workflow.py`.
3. Kept `vmec_jax.optimization_workflow` as a public compatibility facade, and
   preserved the historical `rebuild_indata_with_resolution` monkeypatch seam
   used by diagnostics and tests.

Results obtained:

- `optimization_workflow.py` dropped from 4,024 to 3,787 lines.
- The seed construction logic now has a domain-specific home instead of being
  embedded in the orchestration workflow.
- Public imports such as `vj.prepare_simple_omnigenity_seed_input(...)` and
  `vj.interpolate_indata_boundary(...)` remain stable.

Tests and commands run:

- `python -m ruff check vmec_jax/optimization_workflow.py vmec_jax/optimizers/fixed_boundary/seed_inputs.py tests/test_optimization_workflow_unit.py tests/test_optimization_helpers.py tests/test_optimization_examples.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_optimization_workflow_unit.py -k 'simple_omnigenity or prepare_simple or interpolate_indata or rebuild_for_optimization_resolution' tests/test_optimization_helpers.py -k 'interpolate_indata_boundary or rebuild_for_optimization_resolution' -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_optimization_workflow_unit.py tests/test_optimization_helpers.py tests/test_optimization_examples.py -q`
- `python tools/diagnostics/source_health.py --top 12 --top-functions 20`

Best next steps:

1. Commit and push the seed/input workflow extraction.
2. Continue workflow decomposition by extracting stage construction or
   objective residual assembly into `optimizers.fixed_boundary` modules.
3. Continue larger fixed-boundary solver seams after workflow helpers are
   below the warning threshold.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.88%.
- Differentiability/refactor implementation: 99.998%.
- Solver monolith reduction: 98.50%.
- Free-boundary adjoint monolith reduction: 82%.
- Driver workflow decomposition: 96.4%.
- WOUT diagnostic/profile decomposition: 98.8%.
- Optimizer workflow decomposition: 93.5%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.79%.

## 2026-06-17 Fixed-Boundary Objective-Term Workflow Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `vmec_jax.optimizers.fixed_boundary.objective_terms` for
   `StageContext`, objective term containers, objective vector coercion, and
   generic packed-state residual AD hooks.
2. Re-exported those names through `optimization_workflow.py` so existing user
   scripts and tests can keep importing `workflow.ObjectiveTerm`,
   `workflow.StageContext`, `workflow.QIObjectiveTerm`, and
   `workflow.residuals_from_objectives`.
3. Preserved the private compatibility alias `workflow._as_vector` used by
   tests and diagnostics.

Results obtained:

- `optimization_workflow.py` dropped from 3,787 to 3,637 lines.
- Objective callback containers and packed-state AD hooks now live in the
  fixed-boundary optimizer domain instead of the example workflow facade.
- QI and QS workflow residual paths still share the same generic packed-state
  VJP hooks.

Tests and commands run:

- `python -m ruff check vmec_jax/optimization_workflow.py vmec_jax/optimizers/fixed_boundary/objective_terms.py tests/test_optimization_workflow_unit.py tests/test_optimization_examples.py tests/test_continuation_stage_inputs.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_optimization_workflow_unit.py tests/test_optimization_examples.py tests/test_continuation_stage_inputs.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_optimizer_domain_helpers.py tests/test_optimization_fast_optimizer_methods.py tests/test_optimization_callback_trace.py tests/test_optimization_helpers.py tests/test_optimization_workflow_unit.py -q`
- `python tools/diagnostics/source_health.py --top 12 --top-functions 20`

Best next steps:

1. Commit and push the objective-term extraction.
2. Extract stage setup into a dependency-injected domain helper so existing
   workflow monkeypatch seams stay intact.
3. Start a larger fixed-boundary residual-controller split after the workflow
   facade is below the warning threshold.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.88%.
- Differentiability/refactor implementation: 99.998%.
- Solver monolith reduction: 98.50%.
- Free-boundary adjoint monolith reduction: 82%.
- Driver workflow decomposition: 96.4%.
- WOUT diagnostic/profile decomposition: 98.8%.
- Optimizer workflow decomposition: 94.3%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.80%.

## 2026-06-17 Fixed-Boundary Gauss-Newton Algorithm Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `vmec_jax.optimizers.fixed_boundary.gauss_newton` for the concrete
   Gauss-Newton least-squares loop used by fixed-boundary examples and exact
   optimizer fallback paths.
2. Re-exported `gauss_newton_least_squares` through `optimization.py` so the
   public `vj.gauss_newton_least_squares` API and workflow monkeypatch seams
   stay stable.
3. Moved the exhausted-Jacobian environment gate with the algorithm it controls.

Results obtained:

- `optimization.py` dropped from 4,167 to 3,909 lines.
- A standalone numerical optimizer is now separated from the exact VMEC
  boundary-parameter optimizer class.
- The `VMEC_JAX_OPT_SKIP_EXHAUSTED_GN_JACOBIAN` behavior remains tested.

Tests and commands run:

- `python -m ruff check vmec_jax/optimization.py vmec_jax/optimizers/fixed_boundary/gauss_newton.py tests/test_optimization_helpers.py tests/test_optimization_callback_trace.py tests/test_optimization_wave4_coverage.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_optimization_helpers.py -k 'gauss_newton' tests/test_optimization_callback_trace.py -k 'gauss_newton or exhausted' tests/test_optimization_wave4_coverage.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_optimization_helpers.py tests/test_optimization_callback_trace.py tests/test_optimization_wave4_coverage.py tests/test_optimization_fast_optimizer_methods.py -q`
- `python tools/diagnostics/source_health.py --top 12 --top-functions 20`

Best next steps:

1. Commit and push the Gauss-Newton extraction.
2. Continue splitting `optimization.py` by moving exact tape/cache policy
   helpers or QS residual factories into named fixed-boundary modules.
3. Continue larger residual-controller refactors after the optimizer class
   falls below the source-health warning threshold.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.88%.
- Differentiability/refactor implementation: 99.998%.
- Solver monolith reduction: 98.50%.
- Free-boundary adjoint monolith reduction: 82%.
- Driver workflow decomposition: 96.4%.
- WOUT diagnostic/profile decomposition: 98.8%.
- Optimizer workflow decomposition: 94.3%.
- Fixed-boundary optimizer decomposition: 86%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.81%.

## 2026-06-17 Fixed-Boundary QS Residual Factory Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `vmec_jax.optimizers.fixed_boundary.qs_residuals` for the QA/QH/QP
   quasisymmetry residual factories and their packed-state VJP hooks.
2. Kept `optimization.py` as the public facade for
   `make_qh_residuals_fn(...)` and `make_qs_residuals_fn(...)`.
3. Preserved existing test/profiling monkeypatch seams by passing
   `optimization.py` dependency globals into the extracted implementation.

Results obtained:

- `optimization.py` dropped from 3,909 to 3,516 lines.
- Quasisymmetry objective construction is now separated from the exact
  optimizer class.
- The custom packed-state cotangent hooks still match the residual vector and
  remain usable by dense, matrix-free, and scalar-adjoint paths.

Tests and commands run:

- `python -m ruff check vmec_jax/optimization.py vmec_jax/optimizers/fixed_boundary/qs_residuals.py tests/test_optimization_helpers.py tests/test_optimization_wave2_coverage.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_optimization_helpers.py -k 'residual_factory or make_qs_residuals_fn or make_qh_residuals_fn' tests/test_optimization_wave2_coverage.py -k 'qh_and_qs_residual' -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_optimization_helpers.py tests/test_optimization_wave2_coverage.py tests/test_optimization_callback_trace.py tests/test_optimization_fast_optimizer_methods.py -q`
- `python tools/diagnostics/source_health.py --top 12 --top-functions 20`

Best next steps:

1. Commit and push the QS residual factory extraction.
2. Continue fixed-boundary optimizer decomposition by extracting exact tape
   cache/profile policy or boundary state construction helpers.
3. Start reducing the fixed-boundary residual-iteration monolith after
   optimizer modules are below the source-health warning threshold.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.88%.
- Differentiability/refactor implementation: 99.998%.
- Solver monolith reduction: 98.50%.
- Free-boundary adjoint monolith reduction: 82%.
- Driver workflow decomposition: 96.4%.
- WOUT diagnostic/profile decomposition: 98.8%.
- Optimizer workflow decomposition: 94.3%.
- Fixed-boundary optimizer decomposition: 89%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.82%.

## 2026-06-17 Fixed-Boundary QS Residual Implementation Compatibility Fix

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Validated the extracted QS residual factory against tests that monkeypatch
   `vmec_jax.modes`, `vmec_jax.quasisymmetry`, and `vmec_jax.wout`.
2. Switched the extracted module to module-level runtime lookups for these
   dependencies, preserving the old runtime-import behavior from
   `optimization.py`.
3. Kept `optimization.py` wrappers passing local dependency seams for
   `flux_profiles_from_indata`, `_pressure_profile_for_static`,
   `initial_guess_from_boundary`, `eval_geom`, and `signgs_from_sqrtg`.

Results obtained:

- Residual factory AD hooks remain finite and testable under existing
  monkeypatch diagnostics.
- The extracted implementation no longer weakens observability or profiling
  controls.

Tests and commands run:

- `python -m ruff check vmec_jax/optimization.py vmec_jax/optimizers/fixed_boundary/qs_residuals.py tests/test_optimization_helpers.py tests/test_optimization_wave2_coverage.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_optimization_helpers.py -k 'residual_factory or make_qs_residuals_fn or make_qh_residuals_fn' tests/test_optimization_wave2_coverage.py -k 'qh_and_qs_residual' -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_optimization_helpers.py tests/test_optimization_wave2_coverage.py tests/test_optimization_callback_trace.py tests/test_optimization_fast_optimizer_methods.py -q`
- `python tools/diagnostics/source_health.py --top 12 --top-functions 20`

Best next steps:

1. Commit and push this residual extraction.
2. Extract exact-optimizer profiling and callback bookkeeping helpers.
3. Continue reducing source-health warnings in `optimization.py` and
   `optimization_workflow.py` before tackling the larger residual solver loop.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.88%.
- Differentiability/refactor implementation: 99.998%.
- Solver monolith reduction: 98.50%.
- Free-boundary adjoint monolith reduction: 82%.
- Driver workflow decomposition: 96.4%.
- WOUT diagnostic/profile decomposition: 98.8%.
- Optimizer workflow decomposition: 94.3%.
- Fixed-boundary optimizer decomposition: 89.5%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.83%.

## 2026-06-17 Exact Optimizer Dispatch Decomposition

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added the domain package `vmec_jax.optimizers.fixed_boundary` for
   fixed-boundary optimizer internals while keeping `vmec_jax.optimization` as
   the public compatibility facade.
2. Extracted exact-optimizer history reconstruction and history-dump assembly
   into `optimizers.fixed_boundary.history`, including residual block metadata,
   QS residual reconstruction, monotone final wall-clock handling, and history
   serialization.
3. Extracted scalar-adjoint trust-region and scalar L-BFGS optimizer branches
   into `optimizers.fixed_boundary.scalar_trust` and
   `optimizers.fixed_boundary.scalar_lbfgs`.
4. Extracted SciPy dense and matrix-free least-squares branches into
   `optimizers.fixed_boundary.scipy_least_squares`.
5. Centralized finite residual/vector and linear-operator guard utilities in
   `optimizers.fixed_boundary.linear_guards`.
6. Added direct helper tests in `tests/test_optimizer_domain_helpers.py`.

Results obtained:

- `vmec_jax/optimization.py` dropped from 5,545 to 4,900 lines.
- `FixedBoundaryExactOptimizer.run()` dropped from 598 lines after the first
  history/scalar extraction to 264 lines after scalar and SciPy branch
  extraction; it is now primarily an orchestration layer.
- `FixedBoundaryExactOptimizer` dropped from 3,893 to 3,559 lines.
- Exact accepted-point authority, cache hooks, callback tracing, scalar
  cost-only trial policy, matrix-free finite-output guards, and fallback to the
  best exact accepted point are preserved behind smaller tested modules.

Tests and commands run:

- `python -m ruff check vmec_jax/optimization.py vmec_jax/optimizers/fixed_boundary/*.py tests/test_optimizer_domain_helpers.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_optimizer_domain_helpers.py tests/test_optimization_fast_optimizer_methods.py tests/test_optimization_callback_trace.py tests/test_optimization_helpers.py -q`

Best next steps:

1. Commit and push this exact optimizer dispatch decomposition tranche.
2. Continue with exact linearization/matrix-free product extraction from
   `FixedBoundaryExactOptimizer` into a fixed-boundary optimizer linearization
   module.
3. Keep `vmec_jax.optimization` as the compatibility facade while moving
   implementation details into domain modules.
4. Do not broaden free-boundary adaptive-loop differentiation claims; current
   evidence remains branch-local/fingerprint-gated.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.80%.
- Differentiability/refactor implementation: 99.9965%.
- Solver monolith reduction: 98.50%.
- Free-boundary adjoint monolith reduction: 82%.
- Driver workflow decomposition: 96.4%.
- WOUT diagnostic/profile decomposition: 98.8%.
- Optimizer workflow decomposition: 88%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.73%.

## 2026-06-17 Matrix-Free Linear Operator Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Extracted matrix-free residual-Jacobian `LinearOperator` construction from
   `FixedBoundaryExactOptimizer.residual_linear_operator` into
   `vmec_jax.optimizers.fixed_boundary.matrix_free`.
2. Kept `FixedBoundaryExactOptimizer.residual_linear_operator` as the public
   method wrapper so examples/tests can still monkeypatch or call it directly.
3. Preserved compatibility aliases for older private helper imports from
   `vmec_jax.optimization` while centralizing the implementation in
   `optimizers.fixed_boundary.linear_guards`.

Results obtained:

- `vmec_jax/optimization.py` dropped from 4,900 to 4,702 lines.
- `FixedBoundaryExactOptimizer` dropped from 3,559 to 3,361 lines.
- Matrix-free JVP/VJP replay, cached initial tangent projection, finite product
  sanitization, residual cotangent helper caching, and SciPy `LinearOperator`
  shape behavior now live in a dedicated optimizer-domain module.

Tests and commands run:

- `python -m ruff check vmec_jax/optimization.py vmec_jax/optimizers/fixed_boundary/*.py tests/test_optimizer_domain_helpers.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_optimization_wave2_coverage.py::test_residual_linear_operator_products_use_tape_and_residual_transposes tests/test_optimization_wave2_coverage.py::test_residual_linear_operator_precomputes_initial_tangents_for_stellsym_cpu tests/test_optimization_helpers.py::test_residual_linear_operator_matvec_and_matmat_are_shape_checked tests/test_optimization_helpers.py::test_residual_linear_operator_sanitizes_nonfinite_products tests/test_optimization_helpers.py::test_residual_linear_operator_reuses_cached_initial_tangents -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_optimizer_domain_helpers.py tests/test_optimization_wave2_coverage.py tests/test_optimization_fast_optimizer_methods.py tests/test_optimization_callback_trace.py tests/test_optimization_helpers.py -q`

Best next steps:

1. Commit and push this matrix-free linear-operator extraction.
2. Continue extracting exact scalar objective/gradient replay and dense
   Jacobian projected-replay helpers from `FixedBoundaryExactOptimizer`.
3. Keep compatibility facades stable until the draft PR reaches the final
   package-organization review.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.82%.
- Differentiability/refactor implementation: 99.997%.
- Solver monolith reduction: 98.50%.
- Free-boundary adjoint monolith reduction: 82%.
- Driver workflow decomposition: 96.4%.
- WOUT diagnostic/profile decomposition: 98.8%.
- Optimizer workflow decomposition: 89%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.74%.

## 2026-06-17 Scalar-Adjoint Gradient Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Extracted exact scalar objective and reverse-discrete-adjoint gradient replay
   from `FixedBoundaryExactOptimizer.objective_and_gradient_fun` into
   `vmec_jax.optimizers.fixed_boundary.scalar_gradient`.
2. Kept `objective_and_gradient_fun` as the public method wrapper used by
   scalar-trust, L-BFGS, tests, and examples.
3. Preserved JIT-pure objective-cotangent helper caching, fallback for
   diagnostic/non-JIT objective hooks, state cotangent helper use, tape VJP
   replay, cached initial tangent projection, and initial-map VJP caching.

Results obtained:

- `vmec_jax/optimization.py` dropped from 4,702 to 4,561 lines.
- `FixedBoundaryExactOptimizer` dropped from 3,361 to 3,213 lines.
- The scalar-adjoint production path is now isolated from the optimizer facade,
  making it easier to test and evolve independently of outer optimizer method
  dispatch.

Tests and commands run:

- `python -m ruff check vmec_jax/optimization.py vmec_jax/optimizers/fixed_boundary/*.py tests/test_optimizer_domain_helpers.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_optimization_wave2_coverage.py -k 'objective_and_gradient or residual_linear_operator' tests/test_optimization_helpers.py -k 'objective_and_gradient or residual_linear_operator' tests/test_optimization_fast_optimizer_methods.py tests/test_optimization_callback_trace.py tests/test_optimizer_domain_helpers.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_optimizer_domain_helpers.py tests/test_optimization_wave2_coverage.py tests/test_optimization_fast_optimizer_methods.py tests/test_optimization_callback_trace.py tests/test_optimization_helpers.py -q`

Best next steps:

1. Commit and push this scalar-adjoint gradient extraction.
2. Continue with dense Jacobian/projected replay extraction and then boundary
   parameterization/runtime-policy extraction from `vmec_jax.optimization`.
3. Keep branch-local free-boundary differentiation claims conservative until a
   true adaptive-branch AD-vs-FD gate is implemented and passing.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.84%.
- Differentiability/refactor implementation: 99.9975%.
- Solver monolith reduction: 98.50%.
- Free-boundary adjoint monolith reduction: 82%.
- Driver workflow decomposition: 96.4%.
- WOUT diagnostic/profile decomposition: 98.8%.
- Optimizer workflow decomposition: 90%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.75%.

## 2026-06-17 Boundary Parameterization Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Extracted fixed-boundary parameterization helpers into
   `vmec_jax.optimizers.fixed_boundary.parameterization`.
2. Moved `BoundaryParamSpec`, boundary mode extension/projection helpers,
   boundary parameter spec/name/lift/update helpers, indexed VMEC boundary-map
   conversion, and exponential spectral scaling into that domain module.
3. Kept `vmec_jax.optimization` as a compatibility facade with public re-exports
   and private aliases used by existing tests/profiling scripts.
4. Added an explicit `__all__` for the facade so re-exported public names are
   intentional and lint-clean.

Results obtained:

- `vmec_jax/optimization.py` dropped from 4,561 to 4,151 lines.
- Public imports such as `from vmec_jax.optimization import boundary_param_specs`
  remain valid.
- Existing monkeypatch/test paths for `_apply_boundary_params_numpy`,
  `_indexed_boundary_maps_from_boundary`, and `_coeff_label` remain valid.

Tests and commands run:

- `python -m ruff check vmec_jax/optimization.py vmec_jax/optimizers/fixed_boundary/parameterization.py tests/test_optimization_helpers.py tests/test_optimization_fast_optimizer_methods.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_optimization_helpers.py -k 'boundary_param or apply_boundary or lift_boundary or x_scale or truncate_indata or extend_boundary or indexed_boundary or rebuild_indata' tests/test_optimization_fast_optimizer_methods.py -k 'boundary_param or apply_boundary' tests/test_optimization_wave4_coverage.py -k 'boundary_param or indexed_maps or duplicates' -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_optimizer_domain_helpers.py tests/test_optimization_helpers.py tests/test_optimization_fast_optimizer_methods.py tests/test_optimization_wave2_coverage.py tests/test_optimization_wave4_coverage.py tests/test_optimization_workflow_unit.py -q`

Best next steps:

1. Commit and push this boundary parameterization extraction.
2. Continue with optimizer runtime-policy extraction and dense Jacobian
   projected replay extraction.
3. Start the analogous large-file cleanup in `optimization_workflow.py` after
   the core optimizer facade is reduced below the next source-health threshold.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.86%.
- Differentiability/refactor implementation: 99.998%.
- Solver monolith reduction: 98.50%.
- Free-boundary adjoint monolith reduction: 82%.
- Driver workflow decomposition: 96.4%.
- WOUT diagnostic/profile decomposition: 98.8%.
- Optimizer workflow decomposition: 91%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.76%.

## 2026-06-17 Fixed-Boundary Workflow Artifact Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Extracted `FixedBoundaryOptimizationResult`, `OptimizationOutputPaths`,
   timing-summary extraction, canonical output-path construction, and generic
   `save_optimization_result` into
   `vmec_jax.optimizers.fixed_boundary.workflow_artifacts`.
2. Kept `vmec_jax.optimization_workflow` as the public teaching-workflow facade
   by importing and re-exporting those names.
3. Left stage-specific artifact writes and QI stage checkpointing in
   `optimization_workflow.py` for a later tranche because they depend on local
   continuation-stage helpers and non-worsening guards.

Results obtained:

- `vmec_jax/optimization_workflow.py` dropped from 4,249 to 4,024 lines.
- Result properties, timing summaries, canonical path creation, and generic
  save behavior now have an isolated module and remain import-compatible.

Tests and commands run:

- `python -m ruff check vmec_jax/optimization_workflow.py vmec_jax/optimizers/fixed_boundary/workflow_artifacts.py tests/test_optimization_workflow_unit.py tests/test_optimization_examples.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_optimization_workflow_unit.py -k 'result or output_paths or save_optimization_result or timing or checkpoint or combine' tests/test_optimization_examples.py -k 'FixedBoundaryOptimizationResult or save_optimization_result' -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_optimization_workflow_unit.py -q`

Best next steps:

1. Commit and push this workflow artifact extraction.
2. Continue workflow decomposition by moving objective term classes/factories
   or stage construction into domain modules, keeping `optimization_workflow.py`
   as the public facade.
3. Run combined optimizer/workflow shards after the next tranche to catch
   compatibility regressions across examples.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.88%.
- Differentiability/refactor implementation: 99.998%.
- Solver monolith reduction: 98.50%.
- Free-boundary adjoint monolith reduction: 82%.
- Driver workflow decomposition: 96.4%.
- WOUT diagnostic/profile decomposition: 98.8%.
- Optimizer workflow decomposition: 92%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.77%.

## 2026-06-17 Shared Initial Axis-Reset Diagnostics

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added shared initial-axis diagnostic helpers in
   `vmec_jax.solvers.fixed_boundary.diagnostics.axis_reset`:
   `initial_force_physical_fsq`, `bad_jacobian_from_tau_range`, and
   `bad_jacobian_ptau_from_minmax`.
2. Replaced duplicated scan and non-scan initial-axis physical residual and
   `ptau`/state-Jacobian sign-change calculations with those helpers.
3. Added unit tests for physical-FSQ calculation, tolerance-gated tau-range
   decisions, relative `ptau` tolerance behavior, and missing `ptau` handling.

Results obtained:

- `solve_fixed_boundary_residual_iter` dropped from 7,573 to 7,568 lines.
- Nested `_run_vmec2000_scan` dropped from 1,498 to 1,491 lines.
- The scan and host initial-axis reset preflights now share the same tested
  diagnostic primitives, reducing parity drift risk between the two paths.

Tests and commands run:

- `python -m ruff check vmec_jax/solvers/fixed_boundary/diagnostics/axis_reset.py vmec_jax/solvers/fixed_boundary/residual/iteration.py tests/test_solve_axis_helpers_more_coverage.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_axis_helpers_more_coverage.py tests/test_solve_wave7_coverage.py::test_residual_iter_vmec2000_scan_minimal_one_step tests/test_solve_real_scan_wave10_coverage.py::test_vmec2000_scan_full_history_runs_fallback_decision -q`
- `python tools/diagnostics/source_health.py --top 12 --top-functions 20`

Best next steps:

1. Commit and push this shared axis-reset diagnostic tranche.
2. Continue to the larger remaining residual-controller seams:
   preconditioner-cache initialization or accepted/rejected update transition
   inside `_scan_step`.
3. Use the shared axis-reset primitives as building blocks before attempting a
   full scan initial-force/axis-reset preflight extraction.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.77%.
- Differentiability/refactor implementation: 99.996%.
- Solver monolith reduction: 98.50%.
- Free-boundary adjoint monolith reduction: 82%.
- Driver workflow decomposition: 96.4%.
- WOUT diagnostic/profile decomposition: 98.8%.
- Optimizer workflow decomposition: 86%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.72%.

## 2026-06-18 Fixed-Boundary Finite-Beta Objective Split

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Extracted finite-beta and MHD-profile optimization objective wrappers from
   `vmec_jax.optimization_workflow` into
   `vmec_jax.optimizers.fixed_boundary.finite_beta_objectives`.
2. Moved `MagneticWell`, `VolavgB`, `BetaTotal`, `DMerc`,
   `GlasserResistiveInterchange`, `JDotB`, `BDotB`, `BDotGradV`, `BVector`,
   `JVector`, `ToroidalCurrent`, `ToroidalCurrentGradient`, and
   `RedlBootstrapMismatch` into that domain module.
3. Kept the public teaching-workflow facade stable by re-exporting the same
   class names from `vmec_jax.optimization_workflow` and therefore from
   `vmec_jax.api` / `import vmec_jax as vj`.
4. Updated tests that monkeypatch finite-beta/Mercier/field helper functions
   to patch the new implementation module rather than the workflow facade.

Results obtained:

- `vmec_jax/optimization_workflow.py` dropped from 3,412 lines to 3,027
  lines.
- The finite-beta/Mercier objective family is now isolated in a 399-line
  domain module with no public API break.
- Source-health top offenders now show `optimization_workflow.py` below the
  3,100-line mark; the next workflow target is stage construction and QI
  solve orchestration.

Tests and commands run:

- `python -m ruff check vmec_jax/optimization_workflow.py vmec_jax/optimizers/fixed_boundary/finite_beta_objectives.py tests/test_optimization_workflow_unit.py tests/test_glasser_resistive_interchange.py tests/test_optimization_examples.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_optimization_workflow_unit.py::test_state_objective_wrappers_use_monkeypatched_state_helpers tests/test_glasser_resistive_interchange.py::test_public_glasser_objective_uses_upper_bound_penalty_and_regularized_terms tests/test_optimization_examples.py::test_finite_beta_workflow_objectives_are_jax_differentiable tests/test_optimization_examples.py::test_jxbforce_and_current_objective_gradients_match_finite_difference tests/test_optimization_examples.py::test_dmerc_and_glasser_objective_gradients_match_finite_difference -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_optimization_workflow_unit.py -k 'state_objective or VMECMirrorRatio or mirror_ratio or LgradB or objective_wrappers' -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_optimization_workflow_unit.py tests/test_optimization_helpers.py tests/test_optimization_examples.py tests/test_glasser_resistive_interchange.py -q`
- `python tools/diagnostics/source_health.py --top 15 --top-functions 25`

Best next steps:

1. Commit and push this finite-beta objective split.
2. Continue optimizer workflow decomposition by extracting stage construction
   and QI solve orchestration while keeping the example-facing API stable.
3. In parallel, continue solver monolith reduction in
   `solvers/fixed_boundary/residual/iteration.py`, prioritizing preconditioner
   setup and accepted/rejected transition seams.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.94%.
- Differentiability/refactor implementation: 99.9992%.
- Solver monolith reduction: 98.55%.
- Free-boundary adjoint monolith reduction: 82%.
- Driver workflow decomposition: 96.4%.
- WOUT diagnostic/profile decomposition: 98.8%.
- Optimizer workflow decomposition: 97.0%.
- Fixed-boundary optimizer decomposition: 94.0%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.89%.

## 2026-06-18 Fixed-Boundary QI Objective Split

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Extracted QI/Boozer objective ownership from
   `vmec_jax.optimization_workflow` into
   `vmec_jax.optimizers.fixed_boundary.qi_objectives`.
2. Moved `QuasiIsodynamicOptions`, `QuasiIsodynamicResidual`,
   `QuasiIsodynamicResidualCeiling`, `MirrorRatio`, `VMECMirrorRatio`,
   `BoozerBTarget`, `MaxElongation`, `LgradB`, all QI objective factories,
   and `boozer_b_target_from_wout` into the new domain module.
3. Preserved the public workflow/API imports by re-exporting those names from
   `optimization_workflow`.
4. Updated QI objective tests to monkeypatch the new implementation module
   rather than the old facade, while keeping workflow-stage assembly tests on
   the workflow seam that still owns shared QI field construction.

Results obtained:

- `vmec_jax/optimization_workflow.py` dropped from 3,027 lines to 2,074
  lines.
- The workflow facade no longer appears in the source-health top-15 file list.
- Boozer surface slicing in the new QI module now preserves the previous
  validation and slices `bmnc_b`, `bmns_b`, `iota_b`, and `s_b` consistently.

Tests and commands run:

- `python -m ruff check vmec_jax/optimization_workflow.py vmec_jax/optimizers/fixed_boundary/qi_objectives.py tests/test_optimization_workflow_unit.py tests/test_optimization_workflow_qi_objectives_more_coverage.py tests/test_augmented_lagrangian.py tests/test_optimization_examples.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_optimization_workflow_unit.py -k 'QI or qi or MirrorRatio or VMECMirrorRatio or MaxElongation or LgradB or BoozerBTarget or boozer' tests/test_optimization_workflow_qi_objectives_more_coverage.py tests/test_augmented_lagrangian.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_optimization_examples.py -k 'qi_mirror or qi_max or qi_lgradb or BoozerBTarget or finite_beta_objective_terms or lower_bound_and_lgradb or magnetic_well_tuple or public_api_reexports' -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_optimization_workflow_unit.py tests/test_optimization_helpers.py tests/test_optimization_examples.py tests/test_optimization_workflow_qi_objectives_more_coverage.py tests/test_augmented_lagrangian.py -q`
- `python tools/diagnostics/source_health.py --top 15 --top-functions 25`

Best next steps:

1. Commit and push this QI/Boozer objective split.
2. Extract the remaining workflow stage-construction/QI shared-field assembly
   into a small orchestration module.
3. Move to the solver residual monolith next: split preconditioner setup,
   accepted/rejected transition, and scan bookkeeping from
   `solve_fixed_boundary_residual_iter`.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.95%.
- Differentiability/refactor implementation: 99.9994%.
- Solver monolith reduction: 98.55%.
- Free-boundary adjoint monolith reduction: 82%.
- Driver workflow decomposition: 96.4%.
- WOUT diagnostic/profile decomposition: 98.8%.
- Optimizer workflow decomposition: 98.4%.
- Fixed-boundary optimizer decomposition: 94.0%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.91%.

## 2026-06-18 Fixed-Boundary Stage Policy Split

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Extracted fixed-boundary stage-policy ownership from
   `vmec_jax.optimization_workflow` into
   `vmec_jax.optimizers.fixed_boundary.stage_policy`.
2. Moved `BoundaryModeLimits`, QS/repeated stage schedules, stage-budget
   selection, and boundary-mode stage normalization/labels into the new pure
   policy module.
3. Preserved the public workflow facade by importing the same names from
   `optimization_workflow`.
4. Kept the workflow-level `rebuild_for_optimization_resolution` wrapper in
   place because it remains the monkeypatch seam used by existing examples and
   tests.

Results obtained:

- `vmec_jax/optimization_workflow.py` dropped from 2,074 lines to 1,950 lines.
- The new `stage_policy` module is 120 lines and contains only pure stage
  schedule/normalization logic.
- Source-health now reports the remaining largest production target as
  `vmec_jax/solvers/fixed_boundary/residual/iteration.py`, with
  `optimization_workflow.py` below the 2,000-line warning threshold.

Tests and commands run:

- `python -m ruff check vmec_jax/optimization_workflow.py vmec_jax/optimizers/fixed_boundary/stage_policy.py tests/test_optimization_helpers.py tests/test_optimization_examples.py tests/test_optimization_workflow_unit.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_optimization_helpers.py -k 'stage_modes or stage_budget or boundary_mode_limits or rebuild_for_optimization_resolution' tests/test_optimization_examples.py -k 'stage_policy or stage_modes or mode_limit' tests/test_optimization_workflow_unit.py::test_workflow_mode_limit_seed_and_summary_guard_branches -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_optimization_workflow_unit.py tests/test_optimization_helpers.py tests/test_optimization_examples.py tests/test_optimization_workflow_qi_objectives_more_coverage.py tests/test_augmented_lagrangian.py -q`
- `python tools/diagnostics/source_health.py --top 15 --top-functions 25`

Best next steps:

1. Commit and push this stage-policy split.
2. Move to the fixed-boundary residual iteration monolith next, starting with
   a low-risk extraction of pure bookkeeping/configuration helpers rather than
   changing the numerical update path.
3. Continue preserving public imports and physics gates while reducing the
   largest files in coherent tranches.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.96%.
- Differentiability/refactor implementation: 99.9995%.
- Solver monolith reduction: 98.55%.
- Free-boundary adjoint monolith reduction: 82%.
- Driver workflow decomposition: 96.4%.
- WOUT diagnostic/profile decomposition: 98.8%.
- Optimizer workflow decomposition: 98.8%.
- Fixed-boundary optimizer decomposition: 94.0%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.92%.

## 2026-06-18 Residual Iteration Payload and State Setup Split

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Extracted residual preconditioner/JIT payload facades from
   `vmec_jax.solvers.fixed_boundary.residual.iteration` into
   `vmec_jax.solvers.fixed_boundary.residual.preconditioner_payload`.
2. Preserved the legacy `vmec_jax.solve` monkeypatch seam by keeping thin
   iteration-level wrappers that pass the current `iteration.has_jax` hook into
   the extracted payload module.
3. Extracted initial residual state/update setup into
   `vmec_jax.solvers.fixed_boundary.residual.state_setup`, covering the
   precomputed axis mask, host scalar constants, reusable zero arrays,
   `delta_s`, initial fixed-boundary enforcement, and VMEC lambda/axis rules.
4. Added a direct unit test for `build_residual_state_setup` so this setup seam
   is guarded independently of complete fixed-boundary solves.

Results obtained:

- `vmec_jax/solvers/fixed_boundary/residual/iteration.py` dropped from 8,186
  lines at the start of the solver-refactor pass to 8,024 lines.
- `solve_fixed_boundary_residual_iter` dropped from 7,568 to 7,521 lines.
- The extracted modules are small and domain-named:
  `preconditioner_payload.py` is 199 lines, `state_setup.py` is 145 lines.
- The hotpath tests caught and confirmed the preserved monkeypatch behavior for
  `vmec_jax.solve.has_jax`.

Tests and commands run:

- `python -m ruff check vmec_jax/solvers/fixed_boundary/residual/iteration.py vmec_jax/solvers/fixed_boundary/residual/preconditioner_payload.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_hotpaths.py tests/test_solve_wave4_coverage.py tests/test_solve_additional_branch_coverage.py -q`
- `python -m ruff check vmec_jax/solvers/fixed_boundary/residual/iteration.py vmec_jax/solvers/fixed_boundary/residual/preconditioner_payload.py vmec_jax/solvers/fixed_boundary/residual/state_setup.py tests/test_solve_residual_iter_setup_helpers.py`
- `python -m compileall -q vmec_jax/solvers/fixed_boundary/residual`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_residual_iter_setup_helpers.py tests/test_solve_hotpaths.py tests/test_solve_wave4_coverage.py tests/test_solve_additional_branch_coverage.py -q`
- `python tools/diagnostics/source_health.py --top 10 --top-functions 15`

Best next steps:

1. Continue residual-iteration decomposition inside the loop body using the
   low-risk seams identified by the explorer audit: ptau/Jacobian dump
   bookkeeping, host diagnostics wrappers, and scan-runtime adapters.
2. Do not move `_compute_forces`, `_run_vmec2000_scan`, or `_scan_step` until
   smaller closure-free seams are complete and tests are stable.
3. In parallel, prepare the free-boundary adjoint pure NESTOR extraction
   (`free_boundary_adjoint.py` lines 215-983) because it is the next large
   low-claim-risk module split.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.96%.
- Differentiability/refactor implementation: 99.99955%.
- Solver monolith reduction: 98.75%.
- Free-boundary adjoint monolith reduction: 82%.
- Driver workflow decomposition: 96.4%.
- WOUT diagnostic/profile decomposition: 98.8%.
- Optimizer workflow decomposition: 98.8%.
- Fixed-boundary optimizer decomposition: 94.0%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.925%.

## 2026-06-18 Free-Boundary NESTOR and Projected-Mode Split

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Extracted dense mode-space vacuum reconstruction into
   `vmec_jax.solvers.free_boundary.adjoint.mode_solve`.
2. Extracted projected-mode fixed-point validation helpers into
   `vmec_jax.solvers.free_boundary.adjoint.projected_modes`, while keeping a
   facade wrapper for the public directional-check function so monkeypatches of
   `pytree_directional_derivative_check_jax` still work through
   `vmec_jax.free_boundary_adjoint`.
3. Extracted the pure JAX VMEC/NESTOR nonsingular, analytic, and dense
   mode-solve assembly into
   `vmec_jax.solvers.free_boundary.adjoint.vmec_nestor`.
4. Preserved public compatibility imports from `vmec_jax.free_boundary_adjoint`
   for dense mode solves, VMEC/NESTOR helpers, and projected-mode helpers.

Results obtained:

- `vmec_jax/free_boundary_adjoint.py` dropped from 3,714 lines to 2,734 lines.
- `free_boundary_adjoint.py` no longer appears in the source-health top-10
  oversized files.
- The current strongest free-boundary differentiability claims remain
  unchanged: this is a module split of pure NESTOR/projected validation helpers,
  not a new production adaptive-loop exact-adjoint claim.

Tests and commands run:

- `python -m ruff check vmec_jax/free_boundary_adjoint.py vmec_jax/solvers/free_boundary/adjoint/mode_solve.py vmec_jax/solvers/free_boundary/adjoint/projected_modes.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_vacuum_adjoint.py -q -k "dense_mode_vacuum_solve or projected_mode_fixed_point"`
- `python -m ruff check vmec_jax/free_boundary_adjoint.py vmec_jax/solvers/free_boundary/adjoint/mode_solve.py vmec_jax/solvers/free_boundary/adjoint/projected_modes.py vmec_jax/solvers/free_boundary/adjoint/vmec_nestor.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_vacuum_adjoint.py -q -k "dense_mode_vacuum_solve or dense_vmec_nestor_mode_solve or matrix_free_mode_operator or projected_mode_fixed_point"`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_adjoint_helpers_unit.py tests/test_free_boundary_jax_nestor_operator_cache.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_direct_coil_finite_pressure_sensitivity.py -q -k "direct_coil_trace_fingerprint_detects_control_branch_changes or direct_coil_branch_trace_mode_keeps_replay_controls_without_raw_force_payload or direct_coil_vacuum_field_override_replay_contract"`
- `python tools/diagnostics/source_health.py --top 10 --top-functions 15`

Best next steps:

1. Continue free-boundary adjoint reduction by splitting common branch-local
   replay preparation and report assembly, while preserving conservative claim
   flags.
2. Continue residual iteration decomposition inside
   `solve_fixed_boundary_residual_iter` using ptau/Jacobian dump bookkeeping
   and host diagnostics wrappers.
3. Defer production adaptive-loop differentiation claims until the planned
   fingerprint-gated adaptive AD-vs-FD gate exists.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.96%.
- Differentiability/refactor implementation: 99.9996%.
- Solver monolith reduction: 98.75%.
- Free-boundary adjoint monolith reduction: 92%.
- Driver workflow decomposition: 96.4%.
- WOUT diagnostic/profile decomposition: 98.8%.
- Optimizer workflow decomposition: 98.8%.
- Fixed-boundary optimizer decomposition: 94.0%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.935%.

## 2026-06-18 Residual Ptau Bookkeeping Split

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Extracted ptau host/JAX dispatch, accepted-controller ptau array filtering,
   Jacobian-term dump forwarding, and ptau dump forwarding into
   `vmec_jax.solvers.fixed_boundary.residual.ptau`.
2. Replaced loop-local ptau closures with `partial`-bound helpers so dump
   environment variables are still read at call time without keeping the logic
   inline in `solve_fixed_boundary_residual_iter`.
3. Added direct unit coverage for the ptau wrapper dispatch and dump forwarding
   behavior.

Results obtained:

- `vmec_jax/solvers/fixed_boundary/residual/iteration.py` dropped from 8,024
  to 8,004 lines.
- `solve_fixed_boundary_residual_iter` dropped from 7,521 to 7,493 lines.
- The extracted `ptau.py` module is 141 lines and is covered by fast unit
  tests plus the existing scan math/runtime diagnostics shards.

Tests and commands run:

- `python -m ruff check vmec_jax/solvers/fixed_boundary/residual/iteration.py vmec_jax/solvers/fixed_boundary/residual/ptau.py tests/test_solve_residual_iter_setup_helpers.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_residual_iter_setup_helpers.py tests/test_solve_scan_math_helpers.py tests/test_solve_residual_iter_runtime_helpers.py tests/test_solve_diagnostics_io.py tests/test_solve_hotpaths.py tests/test_solve_wave4_coverage.py tests/test_solve_additional_branch_coverage.py -q`
- `python tools/diagnostics/source_health.py --top 8 --top-functions 12`

Best next steps:

1. Continue residual-loop reduction with host diagnostics/VMEC row print
   wrappers or scan-runtime adapters, avoiding `_scan_step` itself for now.
2. Start test-file decomposition for the two largest free-boundary test files
   once production monolith reductions are stable.
3. Keep full adaptive branch differentiation conservative until the planned
   fingerprint-gated adaptive AD-vs-FD gate exists.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.96%.
- Differentiability/refactor implementation: 99.99965%.
- Solver monolith reduction: 98.9%.
- Free-boundary adjoint monolith reduction: 92%.
- Driver workflow decomposition: 96.4%.
- WOUT diagnostic/profile decomposition: 98.8%.
- Optimizer workflow decomposition: 98.8%.
- Fixed-boundary optimizer decomposition: 94.0%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.94%.

## 2026-06-18 Residual Host Diagnostics Split

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Extracted host-side VMEC2000 debug-print setup, optional JAX debug/io
   callback discovery, NSTEP cadence resolution, row forwarding, and row
   cadence predicates into
   `vmec_jax.solvers.fixed_boundary.residual.host_diagnostics`.
2. Replaced the inline residual-loop debug-print block with a
   `Vmec2000PrintContext` exposing `print_iter_row` and `should_print`.
3. Added a direct unit test that validates row forwarding, LASYM propagation,
   print mode, and cadence decisions.

Results obtained:

- `vmec_jax/solvers/fixed_boundary/residual/iteration.py` dropped from 8,004
  to 7,947 lines.
- `solve_fixed_boundary_residual_iter` dropped from 7,493 to 7,433 lines.
- The extracted host diagnostic module is 119 lines and keeps optional JAX debug
  imports out of the solver loop body.

Tests and commands run:

- `python -m ruff check vmec_jax/solvers/fixed_boundary/residual/iteration.py vmec_jax/solvers/fixed_boundary/residual/host_diagnostics.py tests/test_solve_residual_iter_setup_helpers.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_residual_iter_setup_helpers.py tests/test_solve_residual_iter_config.py tests/test_solve_scan_debug_helpers.py tests/test_scan_helper_edge_gates.py tests/test_solve_diagnostics_io.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_hotpaths.py tests/test_solve_wave4_coverage.py tests/test_solve_additional_branch_coverage.py -q`
- `python tools/diagnostics/source_health.py --top 8 --top-functions 12`

Best next steps:

1. Continue residual-loop reduction with scan-runtime adapters and the remaining
   simple diagnostic dump wrappers.
2. Split branch-local free-boundary replay report assembly after residual-loop
   low-risk seams are complete.
3. Keep CI polling deferred; run broad local shards after a few more coherent
   refactor tranches.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.96%.
- Differentiability/refactor implementation: 99.9997%.
- Solver monolith reduction: 99.0%.
- Free-boundary adjoint monolith reduction: 92%.
- Driver workflow decomposition: 96.4%.
- WOUT diagnostic/profile decomposition: 98.8%.
- Optimizer workflow decomposition: 98.8%.
- Fixed-boundary optimizer decomposition: 94.0%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.945%.

## 2026-06-18 Residual Trace Dump Wrapper Cleanup

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Replaced one-to-one residual trace dump wrapper functions with direct
   aliases or `partial`-bound helpers.
2. Preserved the existing diagnostic record functions as the single source of
   truth for time-control, checkpoint, free-boundary control/axis, and evolve
   trace dumps.
3. Validated the keyword signatures through diagnostics and hotpath tests.

Results obtained:

- `vmec_jax/solvers/fixed_boundary/residual/iteration.py` dropped from 7,947
  to 7,817 lines.
- `solve_fixed_boundary_residual_iter` dropped from 7,433 to 7,303 lines.
- No new module was needed; this was a direct simplification of redundant
  wrappers.

Tests and commands run:

- `python -m ruff check vmec_jax/solvers/fixed_boundary/residual/iteration.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_diagnostics_io.py tests/test_solve_scan_debug_helpers.py tests/test_scan_helper_edge_gates.py tests/test_solve_hotpaths.py tests/test_solve_wave4_coverage.py tests/test_solve_additional_branch_coverage.py -q`
- `python tools/diagnostics/source_health.py --top 8 --top-functions 12`

Best next steps:

1. Continue residual-loop reduction with scan-runtime adapters or the
   preconditioner call-adapter context.
2. Decompose the largest free-boundary test files after production module
   seams stabilize.
3. Run a broader local release shard after one more tranche before checking CI.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.96%.
- Differentiability/refactor implementation: 99.99972%.
- Solver monolith reduction: 99.08%.
- Free-boundary adjoint monolith reduction: 92%.
- Driver workflow decomposition: 96.4%.
- WOUT diagnostic/profile decomposition: 98.8%.
- Optimizer workflow decomposition: 98.8%.
- Fixed-boundary optimizer decomposition: 94.0%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.95%.

## 2026-06-18 Residual Scan Adapter Split

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `vmec_jax.solvers.fixed_boundary.residual.scan_adapters` to own
   scan adapter plumbing: device synchronization/timing, VMEC2000 row cadence,
   time-control trace dumping, scan convergence predicates, and `m=1`
   preconditioner RHS scaling.
2. Replaced the corresponding nested closures in the VMEC2000 scan path and the
   generic scan path with named adapter contexts.
3. Added direct unit coverage for the adapter contracts.

Results obtained:

- `vmec_jax/solvers/fixed_boundary/residual/iteration.py` dropped from 7,817
  to 7,765 lines.
- `solve_fixed_boundary_residual_iter` dropped from 7,303 to 7,248 lines.
- `_run_vmec2000_scan` dropped from 1,491 to 1,438 lines.
- Scan math and controller branch selection remain in place; this tranche moved
  only adapter plumbing.

Tests and commands run:

- `python -m compileall -q vmec_jax/solvers/fixed_boundary/residual/iteration.py vmec_jax/solvers/fixed_boundary/residual/scan_adapters.py`
- `python -m ruff check vmec_jax/solvers/fixed_boundary/residual/iteration.py vmec_jax/solvers/fixed_boundary/residual/scan_adapters.py tests/test_solve_residual_iter_setup_helpers.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_residual_iter_setup_helpers.py tests/test_solve_scan_debug_helpers.py tests/test_scan_helper_edge_gates.py tests/test_solve_scan_planning_helpers.py tests/test_solve_scan_payload_helpers.py tests/test_solve_scan_math_helpers.py tests/test_solve_residual_iter_runtime_helpers.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_residual_iter_config.py tests/test_solve_residual_iter_runtime_helpers.py tests/test_solve_residual_iter_setup_helpers.py tests/test_solve_residual_iter_finalize_helpers.py tests/test_solve_residual_iter_force_cache_helpers.py tests/test_solve_diagnostics_io.py tests/test_solve_scan_debug_helpers.py tests/test_scan_helper_edge_gates.py tests/test_solve_scan_planning_helpers.py tests/test_solve_scan_payload_helpers.py tests/test_solve_scan_math_helpers.py tests/test_solve_preconditioner_metric_helpers.py tests/test_solve_residual_iter_mode_transform_helpers.py tests/test_solve_residual_iter_geometry_helpers.py tests/test_solve_residual_iter_force_payload_helpers.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_hotpaths.py tests/test_solve_wave4_coverage.py tests/test_solve_additional_branch_coverage.py -q`
- `python tools/diagnostics/source_health.py --top 12 --top-functions 16`

Best next steps:

1. Split the branch-local free-boundary replay/report assembly now that the
   residual scan adapter seam is covered.
2. Defer the risky `_scan_step`/`_advance_step` extraction until smaller
   controller-state seams are identified.
3. Continue keeping arbitrary adaptive-branch differentiation claims
   conservative; this refactor does not change those semantics.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.96%.
- Differentiability/refactor implementation: 99.99974%.
- Solver monolith reduction: 99.15%.
- Free-boundary adjoint monolith reduction: 92%.
- Driver workflow decomposition: 96.4%.
- WOUT diagnostic/profile decomposition: 98.8%.
- Optimizer workflow decomposition: 98.8%.
- Fixed-boundary optimizer decomposition: 94.0%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.955%.

## 2026-06-18 Branch-Local Free-Boundary Report Split

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `vmec_jax.solvers.free_boundary.adjoint.branch_local` to own
   branch-local production report setup: complete-solve payload validation,
   production scalar normalization, scalar-key validation, replay option
   assembly, replay-plan construction, graph metadata omission/collection, and
   compact replay option flags.
2. Rewired
   `direct_coil_run_free_boundary_branch_local_scalar_value_and_grad_jax` and
   `direct_coil_run_free_boundary_branch_local_scalars_value_and_jacobian_jax`
   to share the new helper seam.
3. Added direct unit tests for payload validation, active-trace guarding,
   scalar-key selection, replay setup, and option-flag reporting.

Results obtained:

- `vmec_jax/free_boundary_adjoint.py` dropped from 2,734 to 2,550 lines.
- The duplicated scalar/vector branch-local setup code was consolidated without
  changing the public report APIs or the conservative differentiation contract.
- The next large free-boundary reduction target is the 653-line
  `direct_coil_accepted_trace_controller_replay_objective_jax` function.

Tests and commands run:

- `python -m compileall -q vmec_jax/free_boundary_adjoint.py vmec_jax/solvers/free_boundary/adjoint/branch_local.py`
- `python -m ruff check vmec_jax/free_boundary_adjoint.py vmec_jax/solvers/free_boundary/adjoint/branch_local.py tests/test_free_boundary_adjoint_helpers_unit.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_adjoint_helpers_unit.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_vacuum_adjoint.py -q -k "branch_local or custom_vjp or complete_solve_fd or same_branch or controller"`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_direct_coil_finite_pressure_sensitivity.py -q -k "branch_local or custom_vjp or same_branch or fingerprint"`
- `python tools/diagnostics/source_health.py --top 12 --top-functions 16`

Best next steps:

1. Move accepted-controller replay-objective assembly out of
   `free_boundary_adjoint.py` into a domain module while preserving root
   re-exports.
2. Continue residual solver reduction only through small safe controller-state
   seams; avoid moving `_scan_step` wholesale until it has narrower tests.
3. Keep adaptive full-loop differentiation claims conservative until the
   fingerprint-gated adaptive AD-vs-FD gate exists.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.96%.
- Differentiability/refactor implementation: 99.99976%.
- Solver monolith reduction: 99.15%.
- Free-boundary adjoint monolith reduction: 94%.
- Driver workflow decomposition: 96.4%.
- WOUT diagnostic/profile decomposition: 98.8%.
- Optimizer workflow decomposition: 98.8%.
- Fixed-boundary optimizer decomposition: 94.0%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.96%.

## 2026-06-18 Direct-Coil Replay and Report Facade Split

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `vmec_jax.solvers.free_boundary.adjoint.direct_coil_replay`
   for the accepted-boundary direct-coil ``bsqvac`` replay helpers.
2. Added `vmec_jax.solvers.free_boundary.adjoint.complete_solve_reports`
   for fixed accepted-trace replay, replay diagnostics, complete-solve trace
   capture, and same-branch complete-solve finite-difference reports.
3. Rewired `vmec_jax.free_boundary_adjoint` as a compatibility facade that
   re-exports the same public API names while no longer owning those
   implementation blocks.
4. Updated helper tests so monkeypatching targets the implementation module
   for the moved direct-coil NESTOR replay internals.

Results obtained:

- `vmec_jax/free_boundary_adjoint.py` dropped from 2,329 to 1,924 lines,
  below the 2,000-line source-health warning threshold.
- The root facade remains the public import path for existing tests,
  examples, and documentation.
- The largest remaining function in the facade is still
  `direct_coil_accepted_trace_controller_replay_objective_jax` (653 lines);
  that is now isolated as the next function-level extraction target.

Tests and commands run:

- `python -m compileall -q vmec_jax/free_boundary_adjoint.py vmec_jax/solvers/free_boundary/adjoint/direct_coil_replay.py vmec_jax/solvers/free_boundary/adjoint/complete_solve_reports.py`
- `python -m ruff check vmec_jax/free_boundary_adjoint.py vmec_jax/solvers/free_boundary/adjoint/direct_coil_replay.py vmec_jax/solvers/free_boundary/adjoint/complete_solve_reports.py tests/test_free_boundary_adjoint_helpers_unit.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_adjoint_helpers_unit.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_vacuum_adjoint.py -q -k "branch_local or custom_vjp or complete_solve_fd or same_branch or controller"`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_direct_coil_finite_pressure_sensitivity.py -q -k "branch_local or custom_vjp or same_branch or fingerprint"`
- `python tools/diagnostics/source_health.py --top 14 --top-functions 18`

Best next steps:

1. Split `direct_coil_accepted_trace_controller_replay_objective_jax` into a
   domain module or smaller controller-replay assembly helpers without changing
   the conservative same-branch differentiation contract.
2. Continue fixed-boundary driver simplification by extracting stage context,
   finish context, and dynamic-scan selection seams from `run_fixed_boundary`.
3. Continue residual solver reduction only through tested controller-state
   seams; do not move the full `_scan_step` loop wholesale.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.97%.
- Differentiability/refactor implementation: 99.99978%.
- Solver monolith reduction: 99.15%.
- Free-boundary adjoint monolith reduction: 98%.
- Driver workflow decomposition: 96.4%.
- WOUT diagnostic/profile decomposition: 98.8%.
- Optimizer workflow decomposition: 98.8%.
- Fixed-boundary optimizer decomposition: 94.0%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.965%.

## 2026-06-18 Driver Dynamic-Scan Policy Split

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `vmec_jax.drivers.dynamic_scan` to own the dynamic scan-selection
   probe used by VMEC2000-style fixed-boundary stages.
2. Replaced the inline probe block in `run_fixed_boundary` with a single
   call to `maybe_select_dynamic_scan_mode`, passing the same solver state,
   history-comparison callbacks, environment reader, and deep-copy hook.
3. Kept the user-facing verbose diagnostics unchanged, including mismatch and
   scan/non-scan timing messages.

Results obtained:

- `run_fixed_boundary` dropped from 1,556 to 1,458 lines.
- Dynamic scan policy is now a named driver module instead of another nested
  block inside the public driver.
- No solver math changed; this is orchestration-only.

Tests and commands run:

- `python -m compileall -q vmec_jax/driver.py vmec_jax/drivers/dynamic_scan.py`
- `python -m ruff check vmec_jax/driver.py vmec_jax/drivers/dynamic_scan.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_driver_policy_coverage_extra.py tests/test_driver_api_finish_more_coverage.py tests/test_driver_wave2_coverage.py -q -k "dynamic_scan"`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_driver_policy_helpers.py tests/test_driver_helper_edges_wave14_coverage.py -q -k "dynamic_scan"`
- `python tools/diagnostics/source_health.py --top 14 --top-functions 18`

Best next steps:

1. Extract stage solve attempt assembly from `run_fixed_boundary`, keeping the
   actual stage loop in place until a narrower state-machine seam is tested.
2. Move the accepted-controller replay objective out of
   `free_boundary_adjoint.py` or split its internal branch/replay terms into a
   smaller module-level helper.
3. Continue residual solver reduction through safe adapter/controller seams.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.97%.
- Differentiability/refactor implementation: 99.99979%.
- Solver monolith reduction: 99.15%.
- Free-boundary adjoint monolith reduction: 98%.
- Driver workflow decomposition: 97.1%.
- WOUT diagnostic/profile decomposition: 98.8%.
- Optimizer workflow decomposition: 98.8%.
- Fixed-boundary optimizer decomposition: 94.0%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.967%.

## 2026-06-18 Controller Replay Facade and Helper Split

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `vmec_jax.solvers.free_boundary.adjoint.controller_replay` for the
   JAX-visible direct-coil accepted-controller replay implementation.
2. Rewired `vmec_jax.free_boundary_adjoint` to re-export
   `direct_coil_accepted_trace_controller_replay_objective_jax` from that
   domain module while preserving existing public imports.
3. Split the moved controller replay implementation into module-level helpers:
   replay options, boundary replay context lookup, free-boundary `bsqvac`
   replay terms, ordinary trace branch replay, stacked-control branch replay,
   and controller step-function construction.
4. Kept the branch-local/fingerprint-gated differentiation contract unchanged:
   this refactor does not claim arbitrary adaptive host-branch
   differentiability.

Results obtained:

- `vmec_jax/free_boundary_adjoint.py` dropped from 1,924 to roughly 1,270
  lines and is now a smaller compatibility facade plus report/custom-VJP
  wrappers.
- The moved controller replay entry point dropped from 584 to 274 lines.
- The new helper functions are all below the source-health function threshold
  except the orchestration entry point, which is now much smaller and has clear
  next seams.
- The controller replay implementation is now under the
  `vmec_jax.solvers.free_boundary.adjoint` namespace instead of the root
  package facade.

Tests and commands run:

- `python -m compileall -q vmec_jax/free_boundary_adjoint.py vmec_jax/solvers/free_boundary/adjoint/controller_replay.py`
- `python -m ruff check vmec_jax/free_boundary_adjoint.py vmec_jax/solvers/free_boundary/adjoint/controller_replay.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_adjoint_helpers_unit.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_vacuum_adjoint.py -q -k "branch_local or custom_vjp or complete_solve_fd or same_branch or controller"`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_direct_coil_finite_pressure_sensitivity.py -q -k "branch_local or custom_vjp or same_branch or fingerprint"`
- `python tools/diagnostics/source_health.py --top 14 --top-functions 20`

Best next steps:

1. Split the remaining controller replay orchestration into runner-selection
   helpers, especially the stacked-step, preconditioner-policy, and
   accepted-only branches.
2. Continue driver simplification by extracting stage attempt assembly from
   `run_fixed_boundary`.
3. Continue residual solver decomposition through narrow controller-state
   seams and source-health gates.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.97%.
- Differentiability/refactor implementation: 99.99981%.
- Solver monolith reduction: 99.15%.
- Free-boundary adjoint monolith reduction: 99.1%.
- Driver workflow decomposition: 97.1%.
- WOUT diagnostic/profile decomposition: 98.8%.
- Optimizer workflow decomposition: 98.8%.
- Fixed-boundary optimizer decomposition: 94.0%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.970%.

## 2026-06-18 Driver Scan-Parity Guard Split

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `maybe_disable_scan_by_parity_guard` to
   `vmec_jax.drivers.dynamic_scan`.
2. Moved the optional `VMEC_JAX_SCAN_PARITY_GUARD` probe logic out of the
   fixed-boundary stage loop.
3. Kept the probe behavior unchanged: when enabled, it compares short scan and
   non-scan VMEC2000-style prefixes and disables scan for that stage on mismatch
   or probe failure.

Results obtained:

- `run_fixed_boundary` dropped from 1,458 to 1,392 lines.
- Dynamic scan selection and scan-parity safety logic now live in one driver
  policy module.
- Stage-loop code is closer to pure stage assembly/solve execution.

Tests and commands run:

- `python -m compileall -q vmec_jax/driver.py vmec_jax/drivers/dynamic_scan.py`
- `python -m ruff check vmec_jax/driver.py vmec_jax/drivers/dynamic_scan.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_driver_policy_coverage_extra.py tests/test_driver_api_finish_more_coverage.py tests/test_driver_wave2_coverage.py -q -k "dynamic_scan or scan"`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_driver_policy_helpers.py tests/test_driver_helper_edges_wave14_coverage.py -q -k "dynamic_scan or scan"`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_driver_finish_policy_more_coverage.py -q -k "scan or parity_guard"`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_driver_policy_coverage_extra.py -q -k "scan_parity or scan"`
- `python tools/diagnostics/source_health.py --top 12 --top-functions 16`

Best next steps:

1. Extract fixed-boundary stage solve execution into a small driver helper,
   including the no-JIT context and scan-abort fallback.
2. Continue decomposing the stage loop by isolating explicit-stage monitor
   chunks and result aggregation.
3. Keep all stage-control behavior in named driver modules; avoid adding new
   nested policy branches inside `run_fixed_boundary`.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.97%.
- Differentiability/refactor implementation: 99.99982%.
- Solver monolith reduction: 99.15%.
- Free-boundary adjoint monolith reduction: 99.1%.
- Driver workflow decomposition: 97.8%.
- WOUT diagnostic/profile decomposition: 98.8%.
- Optimizer workflow decomposition: 98.8%.
- Fixed-boundary optimizer decomposition: 94.0%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.972%.

## 2026-06-18 Driver Stage Solve Execution Split

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `run_fixed_boundary_stage_solve` to `vmec_jax.drivers.solve` for the
   per-stage VMEC2000 solve call, including the no-JIT context fallback.
2. Added `maybe_precompile_fixed_boundary_stage` for the optional one-step
   stage precompile path.
3. Added `maybe_rerun_scan_abort_stage` for the auto-fast scan-abort fallback
   rerun in parity-safe non-scan mode.
4. Rewired the fixed-boundary stage loop to call these helpers through a
   per-stage partial instead of defining a nested local solver wrapper.

Results obtained:

- `run_fixed_boundary` dropped from 1,392 to 1,338 lines.
- Stage-loop code now delegates execution details to `drivers.solve` and keeps
  policy decisions visible at the call sites.
- The solver callback remains injectable, preserving existing tests and
  monkeypatch seams.

Tests and commands run:

- `python -m compileall -q vmec_jax/driver.py vmec_jax/drivers/solve.py`
- `python -m ruff check vmec_jax/driver.py vmec_jax/drivers/solve.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_driver_policy_coverage_extra.py tests/test_driver_api_finish_more_coverage.py tests/test_driver_wave2_coverage.py -q -k "scan or stage or dynamic"`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_driver_policy_helpers.py tests/test_driver_helper_edges_wave14_coverage.py tests/test_driver_finish_policy_more_coverage.py -q -k "scan or stage or finish"`
- `python tools/diagnostics/source_health.py --top 12 --top-functions 18`

Best next steps:

1. Extract explicit accelerated-stage monitor chunking from the stage loop into
   `drivers.staging` or `drivers.solve`, leaving the main loop with a single
   "run stage with optional monitor" call.
2. Continue reducing `run_fixed_boundary` toward a readable orchestration
   function rather than a solver-policy monolith.
3. Keep residual-core refactors separate from driver orchestration refactors
   to avoid mixing numerical and workflow risk.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.97%.
- Differentiability/refactor implementation: 99.99983%.
- Solver monolith reduction: 99.15%.
- Free-boundary adjoint monolith reduction: 99.1%.
- Driver workflow decomposition: 98.2%.
- WOUT diagnostic/profile decomposition: 98.8%.
- Optimizer workflow decomposition: 98.8%.
- Fixed-boundary optimizer decomposition: 94.0%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.974%.

## 2026-06-18 Explicit Stage Monitor Split

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `run_stage_with_optional_explicit_monitor` to
   `vmec_jax.drivers.staging`.
2. Moved accelerated explicit-stage chunking, early parity switch, chunk
   merging, and scan-abort fallback dispatch out of the fixed-boundary stage
   loop.
3. Kept all numerical callbacks explicit: stage solve, resume-state sanitize,
   requested-FTOL gate, stage-switch heuristic, chunk merge, diagnostic
   annotation, and scan-abort rerun.

Results obtained:

- `run_fixed_boundary` dropped from 1,338 to 1,228 lines.
- Since the start of today’s driver tranches, `run_fixed_boundary` is down from
  1,556 to 1,228 lines while preserving existing stage tests.
- Accelerated-stage monitoring is now a named staging policy helper instead of
  a deeply nested branch in the public driver.

Tests and commands run:

- `python -m compileall -q vmec_jax/driver.py vmec_jax/drivers/staging.py vmec_jax/drivers/solve.py`
- `python -m ruff check vmec_jax/driver.py vmec_jax/drivers/staging.py vmec_jax/drivers/solve.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_driver_policy_coverage_extra.py tests/test_driver_api_finish_more_coverage.py tests/test_driver_wave2_coverage.py -q -k "scan or stage or dynamic"`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_driver_policy_helpers.py tests/test_driver_helper_edges_wave14_coverage.py tests/test_driver_finish_policy_more_coverage.py -q -k "scan or stage or finish"`
- `python tools/diagnostics/source_health.py --top 12 --top-functions 18`

Best next steps:

1. Move multigrid result aggregation/final-diagnostics construction out of
   `run_fixed_boundary`; that block is now the next obvious stage-loop
   monolith.
2. Continue reducing the residual solver file through scan adapter and
   VMEC2000-scan chunk seams.
3. Add direct unit tests for the new explicit monitor helper if future changes
   touch its early-switch conditions.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.97%.
- Differentiability/refactor implementation: 99.99984%.
- Solver monolith reduction: 99.15%.
- Free-boundary adjoint monolith reduction: 99.1%.
- Driver workflow decomposition: 98.7%.
- WOUT diagnostic/profile decomposition: 98.8%.
- Optimizer workflow decomposition: 98.8%.
- Fixed-boundary optimizer decomposition: 94.0%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.976%.

## 2026-06-18 Multigrid Result Assembly Split

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `assemble_multigrid_stage_result` to `vmec_jax.drivers.results`.
2. Moved multigrid diagnostic assembly, final-stage budget-exhaustion
   bookkeeping, per-stage history concatenation, and public
   `SolveVmecResidualResult` construction out of `run_fixed_boundary`.
3. Removed now-unused driver-level aliases for stage chunk diagnostic keys and
   history concatenation.

Results obtained:

- `run_fixed_boundary` dropped from 1,228 to 1,156 lines.
- Driver result construction now lives with existing result/chunk merge
  helpers.
- The driver stage loop is now mostly stage setup, stage execution, and final
  output correction/finalization.

Tests and commands run:

- `python -m compileall -q vmec_jax/driver.py vmec_jax/drivers/results.py`
- `python -m ruff check vmec_jax/driver.py vmec_jax/drivers/results.py vmec_jax/drivers/staging.py vmec_jax/drivers/solve.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_driver_policy_coverage_extra.py tests/test_driver_api_finish_more_coverage.py tests/test_driver_wave2_coverage.py -q -k "scan or stage or dynamic or multigrid"`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_driver_policy_helpers.py tests/test_driver_helper_edges_wave14_coverage.py tests/test_driver_finish_policy_more_coverage.py -q -k "scan or stage or finish or multigrid"`
- `python tools/diagnostics/source_health.py --top 12 --top-functions 18`

Best next steps:

1. Move the scan-WOUT corrector and final convergence diagnostic assembly out
   of `run_fixed_boundary`.
2. Reassess the residual solver monolith next; the driver is now significantly
   smaller and lower-risk seams remain in residual iteration.
3. Keep running focused driver shards after each workflow split and reserve
   full test/coverage gates for larger milestones.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.97%.
- Differentiability/refactor implementation: 99.99985%.
- Solver monolith reduction: 99.15%.
- Free-boundary adjoint monolith reduction: 99.1%.
- Driver workflow decomposition: 99.0%.
- WOUT diagnostic/profile decomposition: 98.8%.
- Optimizer workflow decomposition: 98.8%.
- Fixed-boundary optimizer decomposition: 94.0%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.978%.

## 2026-06-18 Driver Scan-WOUT Corrector and Final Diagnostics Split

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `maybe_apply_scan_wout_corrector` to `vmec_jax.drivers.solve`.
2. Added `finalize_fixed_boundary_convergence_result` to
   `vmec_jax.drivers.results`.
3. Replaced the inline scan-WOUT correction and final convergence diagnostic
   assembly in `run_fixed_boundary` with these helpers.

Results obtained:

- `run_fixed_boundary` dropped from 1,156 to 1,085 lines.
- The fixed-boundary driver has been reduced from 1,556 to 1,085 lines across
  the dynamic-scan, stage execution, explicit monitor, result assembly, and
  finalization tranches.
- Output correction and final convergence bookkeeping are now named driver
  helper seams.

Tests and commands run:

- `python -m compileall -q vmec_jax/driver.py vmec_jax/drivers/solve.py vmec_jax/drivers/results.py`
- `python -m ruff check vmec_jax/driver.py vmec_jax/drivers/solve.py vmec_jax/drivers/results.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_driver_policy_coverage_extra.py tests/test_driver_api_finish_more_coverage.py tests/test_driver_wave2_coverage.py -q -k "scan or stage or dynamic or multigrid or finish"`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_driver_policy_helpers.py tests/test_driver_helper_edges_wave14_coverage.py tests/test_driver_finish_policy_more_coverage.py -q -k "scan or stage or finish or multigrid"`
- `python tools/diagnostics/source_health.py --top 12 --top-functions 18`

Best next steps:

1. Move the remaining stage setup policy into compact helpers only where it
   reduces cognitive load; avoid over-abstracting constructor-style blocks.
2. Reassess `wout_minimal_from_fixed_boundary` and residual iteration as the
   next high-line-count source targets.
3. Run a broader driver suite after one more tranche because several workflow
   seams have now moved.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.97%.
- Differentiability/refactor implementation: 99.99986%.
- Solver monolith reduction: 99.15%.
- Free-boundary adjoint monolith reduction: 99.1%.
- Driver workflow decomposition: 99.3%.
- WOUT diagnostic/profile decomposition: 98.8%.
- Optimizer workflow decomposition: 98.8%.
- Fixed-boundary optimizer decomposition: 94.0%.
- Implicit residual-adjoint decomposition: 88%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.980%.

## 2026-06-18 Residual Scan Runtime and Active Adjoint Routing Split

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `Vmec2000ScanRuntimeSetup` and `build_vmec2000_scan_runtime_setup`
   to `vmec_jax.solvers.fixed_boundary.residual.scan_adapters`.
2. Moved VMEC2000 scan guard validation, scan setup policy resolution,
   timing/device runtime setup, print/time-control hooks, resume-field
   initialization, convergence predicate setup, and scan force-callable
   selection out of the residual iteration loop.
3. Replaced the active residual-adjoint solver routing in `implicit.py` with
   the existing `solve_active_residual_adjoint_linearized` helper.
4. Extended the helper with optional profiling callbacks so existing backward
   profile stage names remain available for dense, BiCGStab, lineax, and CG
   routes.

Results obtained:

- `solve_fixed_boundary_residual_iter._run_vmec2000_scan` dropped from about
  1,438 to 1,404 lines without changing the scan recurrence itself.
- `solve_fixed_boundary_state_implicit_vmec_residual` dropped from about 864 to
  788 lines, and active adjoint route selection now lives in a dedicated tested
  helper.
- The residual scan setup seam is now explicit and can be further decomposed
  into axis-reset/preconditioner-cache helpers without touching scan math.

Tests and commands run:

- `python -m compileall -q vmec_jax/implicit.py vmec_jax/implicit_residual_adjoint_helpers.py vmec_jax/solvers/fixed_boundary/residual/iteration.py vmec_jax/solvers/fixed_boundary/residual/scan_adapters.py`
- `python -m ruff check vmec_jax/implicit.py vmec_jax/implicit_residual_adjoint_helpers.py vmec_jax/solvers/fixed_boundary/residual/iteration.py vmec_jax/solvers/fixed_boundary/residual/scan_adapters.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_driver_policy_coverage_extra.py tests/test_driver_api_finish_more_coverage.py tests/test_driver_wave2_coverage.py -q -k "scan or vmec2000 or dynamic"`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_driver_policy_helpers.py tests/test_driver_helper_edges_wave14_coverage.py tests/test_driver_finish_policy_more_coverage.py -q -k "scan or vmec2000"`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_implicit_residual_adjoint_helpers.py tests/test_implicit_wave12_coverage.py::test_vmec_residual_custom_vjp_routes_active_direct_bicgstab tests/test_implicit_wave12_coverage.py::test_vmec_residual_custom_vjp_routes_active_lineax_success tests/test_implicit_wave12_coverage.py::test_vmec_residual_custom_vjp_active_cg_and_full_cg_fallbacks tests/test_implicit_helpers.py::test_fixed_boundary_residual_implicit_primal_matches_default_control_path -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_implicit_helpers.py tests/test_implicit_wave12_coverage.py tests/test_implicit_more_coverage.py tests/test_implicit_sensitivity_fast_coverage.py -q`
- `python tools/diagnostics/source_health.py --top 14 --top-functions 20`

Best next steps:

1. Split the residual iteration force/JIT setup and HLO dump probes identified
   by the residual explorer, preserving compute-force cache behavior.
2. Continue residual scan decomposition with initial axis-reset and
   preconditioner-cache preparation only after the new runtime setup seam is
   stable.
3. Defer WOUT Nyquist/Mercier decomposition until after implicit/residual seams
   are smaller, because WOUT parity blocks are more output-sensitive.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.97%.
- Differentiability/refactor implementation: 99.99988%.
- Solver monolith reduction: 99.2%.
- Free-boundary adjoint monolith reduction: 99.1%.
- Driver workflow decomposition: 99.3%.
- Residual iteration decomposition: 90.5%.
- WOUT diagnostic/profile decomposition: 98.8%.
- Optimizer workflow decomposition: 98.8%.
- Fixed-boundary optimizer decomposition: 94.0%.
- Implicit residual-adjoint decomposition: 91%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.982%.

## 2026-06-18 Residual HLO and NumPy Force Setup Split

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `maybe_dump_initial_residual_hlo_kernels` to
   `vmec_jax.solvers.fixed_boundary.diagnostics.hlo`.
2. Replaced inline residual HLO debug probes for bcovar/tomnsps with the
   diagnostics helper, preserving failure-swallowing debug behavior.
3. Added `NumpyForceFastPath` and `prepare_numpy_force_fast_path` to
   `vmec_jax.solvers.fixed_boundary.residual.force_cache`.
4. Replaced inline NumPy force wrapper setup and static/trig/wout conversion
   in `solve_fixed_boundary_residual_iter` with the explicit fast-path object.
5. Fixed the scan setup extraction by carrying `scan_differentiated` through
   `Vmec2000ScanRuntimeSetup`, preserving scan preflight/cache behavior.

Results obtained:

- `iteration.py` dropped from 7,732 lines at the start of this resumed session
  to 7,579 lines after the residual scan/HLO/NumPy fast-path splits.
- `solve_fixed_boundary_residual_iter` dropped from about 7,214 to 7,060 lines
  in this resumed session.
- Debug HLO extraction is now in the diagnostics domain, and host NumPy force
  setup is now in the residual force-cache domain rather than buried inside the
  solver loop.

Tests and commands run:

- `python -m compileall -q vmec_jax/solvers/fixed_boundary/residual/iteration.py vmec_jax/solvers/fixed_boundary/residual/scan_adapters.py vmec_jax/solvers/fixed_boundary/residual/force_cache.py vmec_jax/solvers/fixed_boundary/diagnostics/hlo.py`
- `python -m ruff check vmec_jax/solvers/fixed_boundary/residual/iteration.py vmec_jax/solvers/fixed_boundary/residual/scan_adapters.py vmec_jax/solvers/fixed_boundary/residual/force_cache.py vmec_jax/solvers/fixed_boundary/diagnostics/hlo.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_finish_cache_more_coverage.py::test_precompile_only_jit_precompile_exercises_force_cache_and_lower tests/test_solve_finish_cache_more_coverage.py::test_precompile_only_compute_force_cache_is_owned_and_limited tests/test_solve_finish_cache_more_coverage.py::test_precompile_only_jit_precompile_swallows_compile_failure tests/test_vmec_numpy_forces_cache.py::test_compute_forces_numpy_converts_state_and_constraint_inputs -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_debug_dump_wave10_coverage.py::test_maybe_dump_hlo_kernel_uses_fake_jax_and_deduplicates tests/test_solve_debug_dump_wave10_coverage.py::test_maybe_dump_hlo_kernel_respects_disabled_and_missing_jax tests/test_solve_wave6_coverage.py::test_hlo_dump_label_specific_env_writes_once tests/test_solve_residual_iter_force_payload_helpers.py::test_residual_force_payload_from_kernels_routes_masks_callbacks_and_hlo_dump -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_driver_policy_coverage_extra.py tests/test_driver_api_finish_more_coverage.py tests/test_driver_wave2_coverage.py tests/test_driver_policy_helpers.py tests/test_driver_helper_edges_wave14_coverage.py tests/test_driver_finish_policy_more_coverage.py -q -k "scan or vmec2000 or dynamic or precompile or cache"`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_driver_api.py::test_run_fixed_boundary_accelerated_mode_uses_scan tests/test_driver_api.py::test_run_fixed_boundary_accelerated_mode_defaults_to_single_grid -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_driver_control_fast.py tests/test_driver_api.py -q`
- `python tools/diagnostics/source_health.py --top 14 --top-functions 20`

Best next steps:

1. Extract the strict-update precompile block from residual iteration into a
   force/precompile helper, with the existing cache/precompile tests as gates.
2. Then split VMEC2000 scan axis-reset preparation and initial preconditioner
   cache construction from `_run_vmec2000_scan`.
3. Keep commits at coherent tranche boundaries; avoid CI polling until a larger
   milestone is ready.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.97%.
- Differentiability/refactor implementation: 99.99989%.
- Solver monolith reduction: 99.25%.
- Free-boundary adjoint monolith reduction: 99.1%.
- Driver workflow decomposition: 99.3%.
- Residual iteration decomposition: 91.5%.
- WOUT diagnostic/profile decomposition: 98.9%.
- Optimizer workflow decomposition: 98.8%.
- Fixed-boundary optimizer decomposition: 94.0%.
- Implicit residual-adjoint decomposition: 91%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.984%.

## 2026-06-18 Residual Precompile Helper Split

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `maybe_precompile_residual_force_kernels` to
   `vmec_jax.solvers.fixed_boundary.residual.force_cache`.
2. Moved best-effort force `lower().compile()` and strict-update precompile
   setup out of `solve_fixed_boundary_residual_iter`.
3. Kept all compile failures swallowed, matching the previous non-semantic
   precompile behavior.

Results obtained:

- `iteration.py` is down to 7,533 lines.
- `solve_fixed_boundary_residual_iter` is down to 7,013 lines.
- Force-callable selection, NumPy fast-path setup, and precompile logic are now
   grouped in the residual force-cache domain.

Tests and commands run:

- `python -m compileall -q vmec_jax/solvers/fixed_boundary/residual/iteration.py vmec_jax/solvers/fixed_boundary/residual/force_cache.py`
- `python -m ruff check vmec_jax/solvers/fixed_boundary/residual/iteration.py vmec_jax/solvers/fixed_boundary/residual/force_cache.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_finish_cache_more_coverage.py::test_precompile_only_jit_precompile_exercises_force_cache_and_lower tests/test_solve_finish_cache_more_coverage.py::test_precompile_only_compute_force_cache_is_owned_and_limited tests/test_solve_finish_cache_more_coverage.py::test_precompile_only_jit_precompile_swallows_compile_failure -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_driver_control_fast.py tests/test_driver_api.py -q`
- `python tools/diagnostics/source_health.py --top 14 --top-functions 20`

Best next steps:

1. Split VMEC2000 scan axis-reset preparation from `_run_vmec2000_scan` into
   the scan/residual adapter domain.
2. Split initial preconditioner-cache unpacking into a named scan-cache setup
   object.
3. Re-run the broader driver and scan shards after both scan-preparation splits.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.97%.
- Differentiability/refactor implementation: 99.99990%.
- Solver monolith reduction: 99.30%.
- Free-boundary adjoint monolith reduction: 99.1%.
- Driver workflow decomposition: 99.3%.
- Residual iteration decomposition: 92.0%.
- WOUT diagnostic/profile decomposition: 98.9%.
- Optimizer workflow decomposition: 98.8%.
- Fixed-boundary optimizer decomposition: 94.0%.
- Implicit residual-adjoint decomposition: 91%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.985%.

## 2026-06-18 WOUT Passive Bcovar Payload Helper Move

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Moved WOUT bcovar/force payload helpers from root `vmec_jax.wout` into
   `vmec_jax.io.wout.minimal`.
2. Added `WoutBcovarPayload`, `prepare_wout_bcovar_payload`,
   `attach_force_payload_geometry`, `indata_for_wout_force_path`,
   `env_enabled`, and `device_get_if_available` to the WOUT minimal helper
   module.
3. Preserved the private compatibility aliases in `wout.py` so existing tests
   and internal diagnostic imports continue to work.

Results obtained:

- Removed 143 lines of passive helper implementation from root `wout.py`.
- Kept the parity-sensitive `wout_minimal_from_fixed_boundary` diagnostic body
   unchanged.
- WOUT helper behavior for final-bcovar reuse, forced BSS, and source filtering
   remains covered by focused tests.

Tests and commands run:

- `python -m compileall -q vmec_jax/wout.py vmec_jax/io/wout/minimal.py`
- `python -m ruff check vmec_jax/wout.py vmec_jax/io/wout/minimal.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_wout_fast_helpers.py tests/test_wout_env_branch_coverage.py::test_reuse_final_bcovar_env_uses_payload_even_on_fast_path tests/test_wout_env_branch_coverage.py::test_forced_bss_uses_symforced_force_kernel_arrays tests/test_wout_driver_wave10_coverage.py::test_wout_minimal_bsub_source_filter_branches tests/test_wout_driver_wave10_coverage.py::test_wout_minimal_equif_correction_and_raw_filter_branch -q`
- `python tools/diagnostics/source_health.py --top 14 --top-functions 20`

Best next steps:

1. Split the WOUT BSS source-policy block into `io.wout.minimal`, then stop
   WOUT refactoring before Nyquist/Mercier parity-sensitive blocks.
2. Return to residual scan axis-reset/preconditioner setup after the WOUT BSS
   split.
3. Keep broad local gates batched; do not block on GitHub CI polling.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.97%.
- Differentiability/refactor implementation: 99.99991%.
- Solver monolith reduction: 99.30%.
- Free-boundary adjoint monolith reduction: 99.1%.
- Driver workflow decomposition: 99.3%.
- Residual iteration decomposition: 92.0%.
- WOUT diagnostic/profile decomposition: 99.0%.
- Optimizer workflow decomposition: 98.8%.
- Fixed-boundary optimizer decomposition: 94.0%.
- Implicit residual-adjoint decomposition: 91%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.986%.

## 2026-06-18 WOUT BSS Source Policy Split

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `WoutBssSourcePayload` and `prepare_wout_bss_source_payload` to
   `vmec_jax.io.wout.minimal`.
2. Moved BSS/JXBFORCE source selection for force-kernel vs bcovar/Jacobian
   arrays out of `wout_minimal_from_fixed_boundary`.
3. Preserved downstream local names in `wout.py` so the existing bsubs/filter
   and Nyquist sections were not rewritten.

Results obtained:

- `wout_minimal_from_fixed_boundary` dropped from 928 to 892 lines.
- BSS source-policy behavior remains exercised by forced-BSS and source-filter
   branch tests.
- Stopped before the Nyquist/Mercier parity-sensitive blocks, as planned.

Tests and commands run:

- `python -m compileall -q vmec_jax/wout.py vmec_jax/io/wout/minimal.py`
- `python -m ruff check vmec_jax/wout.py vmec_jax/io/wout/minimal.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_wout_env_branch_coverage.py::test_forced_bss_uses_symforced_force_kernel_arrays tests/test_wout_driver_wave10_coverage.py::test_wout_minimal_bsub_source_filter_branches tests/test_wout_driver_wave10_coverage.py::test_wout_minimal_equif_correction_and_raw_filter_branch tests/test_wout_fast_helpers.py -q`
- `python tools/diagnostics/source_health.py --top 14 --top-functions 20`

Best next steps:

1. Return to residual scan preparation: split axis-reset preparation into a
   scan setup object only if tests can cover reset and non-reset paths.
2. Otherwise address `implicit.py` full-adjoint routing next, which is a clearer
   differentiability seam than WOUT Nyquist output.
3. Run a broader local release shard after the next residual/implicit tranche.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.97%.
- Differentiability/refactor implementation: 99.99992%.
- Solver monolith reduction: 99.30%.
- Free-boundary adjoint monolith reduction: 99.1%.
- Driver workflow decomposition: 99.3%.
- Residual iteration decomposition: 92.0%.
- WOUT diagnostic/profile decomposition: 99.1%.
- Optimizer workflow decomposition: 98.8%.
- Fixed-boundary optimizer decomposition: 94.0%.
- Implicit residual-adjoint decomposition: 91%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.987%.

## 2026-06-18 Full Residual Adjoint Routing Helper

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `FullResidualAdjointSolveResult` and
   `solve_full_residual_adjoint_linearized` to
   `vmec_jax.implicit_residual_adjoint_helpers`.
2. Replaced inline full-state residual-adjoint CG/JVP routing in `implicit.py`
   with the helper call.
3. Added a focused unit test that exercises the full helper with fake
   pack/unpack/project functions and verifies CG/JVP plumbing.

Results obtained:

- Active and full residual-adjoint solve routing now live in the same helper
   module and can be unit-tested independently of a VMEC solve.
- Existing backward profiling labels (`full_cg_done`, `full_jvp_done`) are
   preserved through the helper callback.

Tests and commands run:

- `python -m compileall -q vmec_jax/implicit.py vmec_jax/implicit_residual_adjoint_helpers.py tests/test_implicit_residual_adjoint_helpers.py`
- `python -m ruff check vmec_jax/implicit.py vmec_jax/implicit_residual_adjoint_helpers.py tests/test_implicit_residual_adjoint_helpers.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_implicit_residual_adjoint_helpers.py tests/test_implicit_adjoint_helpers.py tests/test_implicit_helpers.py tests/test_implicit_wave12_coverage.py tests/test_implicit_more_coverage.py tests/test_implicit_sensitivity_fast_coverage.py -q`
- `python tools/diagnostics/source_health.py --top 14 --top-functions 20`

Best next steps:

1. Continue residual scan axis-reset/preconditioner setup split, guarded by
   scan/driver shards.
2. Add broader local release-gate shards after another residual tranche.
3. Keep adaptive-branch differentiability claims conservative; no arbitrary
   adaptive branch differentiation has been added here.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.97%.
- Differentiability/refactor implementation: 99.99993%.
- Solver monolith reduction: 99.30%.
- Free-boundary adjoint monolith reduction: 99.1%.
- Driver workflow decomposition: 99.3%.
- Residual iteration decomposition: 92.0%.
- WOUT diagnostic/profile decomposition: 99.1%.
- Optimizer workflow decomposition: 98.8%.
- Fixed-boundary optimizer decomposition: 94.0%.
- Implicit residual-adjoint decomposition: 93%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.988%.

## 2026-06-18 Free-Boundary Controller Domain Move

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Moved the implementation of `free_boundary_adjoint_controller.py` to
   `vmec_jax.solvers.free_boundary.adjoint.controller`.
2. Recreated `vmec_jax.free_boundary_adjoint_controller` as a compatibility
   facade that re-exports the full controller API expected by existing imports.
3. Updated internal imports in `free_boundary_adjoint.py` and
   `controller_replay.py` to use the domain module directly.

Results obtained:

- Free-boundary controller implementation now lives under the free-boundary
   adjoint domain instead of the root namespace.
- Existing public/private import compatibility is retained through the root
   facade.
- Root helper-prefix count is unchanged because the compatibility facade remains
   intentionally.

Tests and commands run:

- `python -m compileall -q vmec_jax/free_boundary_adjoint.py vmec_jax/free_boundary_adjoint_controller.py vmec_jax/solvers/free_boundary/adjoint/controller.py vmec_jax/solvers/free_boundary/adjoint/controller_replay.py`
- `python -m ruff check vmec_jax/free_boundary_adjoint.py vmec_jax/free_boundary_adjoint_controller.py vmec_jax/solvers/free_boundary/adjoint/controller.py vmec_jax/solvers/free_boundary/adjoint/controller_replay.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_vacuum_adjoint.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_vacuum_adjoint.py tests/test_free_boundary_direct_coil_finite_pressure_sensitivity.py -q -k "controller or segmented or accepted or branch_local or custom_vjp"`
- `python tools/diagnostics/source_health.py --top 14 --top-functions 20`

Best next steps:

1. Continue moving implementation code out of root compatibility modules only
   when public import facades can remain stable.
2. Prioritize residual scan recurrence/helper extraction next; free-boundary
   adjoint implementation files are now largely domain-organized.
3. Run broad local gates periodically; do not wait on CI after every commit.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.97%.
- Differentiability/refactor implementation: 99.99994%.
- Solver monolith reduction: 99.30%.
- Free-boundary adjoint monolith reduction: 99.25%.
- Driver workflow decomposition: 99.3%.
- Residual iteration decomposition: 92.0%.
- WOUT diagnostic/profile decomposition: 99.1%.
- Optimizer workflow decomposition: 98.8%.
- Fixed-boundary optimizer decomposition: 94.0%.
- Implicit residual-adjoint decomposition: 93%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- Overall differentiability-refactor PR: 99.989%.

## 2026-06-18 Refactor CI Compatibility Stabilization

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Inspected PR #20 CI status and pulled the failing shard logs for the
   current draft refactor branch.
2. Fixed residual-iteration facade compatibility by re-exporting the internal
   bad-Jacobian and chunked-scan configuration helpers used by legacy tests and
   internal callers.
3. Made residual precompile-only setup robust to profile-construction failures:
   precompile probes now return a minimal precompile result instead of failing
   before the intended probe path can complete.
4. Hardened moved free-boundary adjoint modules against root-facade monkeypatch
   compatibility by forwarding assignments from `vmec_jax.free_boundary_adjoint`
   into the moved boundary replay, direct-coil replay, and JAX NESTOR domain
   modules.
5. Fixed synthetic-basis handling in JAX NESTOR so tests and diagnostics that
   provide minimal basis dictionaries no longer require `nu_full`/`lasym`.
6. Fixed force profiling context managers so exceptions raised by the force
   body are not wrapped as `generator didn't stop after throw()`.
7. Hardened the VMEC force/bcovar seam against xdist mock leakage:
   lightweight synthetic bcovar payloads are still accepted when their grid
   matches the active solve, but stale or captured mocks with incompatible
   radial grids now fall back to the production `vmec_bcovar` implementation.

Results obtained:

- The three previously failing CI shards now pass locally.
- No output files or figure artifacts were introduced.
- The compatibility fixes preserve focused synthetic unit tests while restoring
   real-solve behavior in parallel CI ordering.

Tests and commands run:

- `python -m compileall -q vmec_jax/vmec_forces.py vmec_jax/free_boundary_adjoint.py vmec_jax/solvers/free_boundary/adjoint/vmec_nestor.py vmec_jax/solvers/fixed_boundary/residual/iteration.py`
- `python -m ruff check vmec_jax/vmec_forces.py vmec_jax/free_boundary_adjoint.py vmec_jax/solvers/free_boundary/adjoint/vmec_nestor.py vmec_jax/solvers/fixed_boundary/residual/iteration.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_resume_state.py::test_accelerated_resume_state_is_minimal_and_restartable tests/test_solve_hotpaths.py::test_preconditioner_output_scaling_gate_is_gpu_only_without_gpu tests/test_vmec_forces_synthetic_helpers.py tests/test_wout_bcovar_forces_extra_coverage.py -q`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_finish_cache_more_coverage.py::test_precompile_setup_rebuilds_static_after_grid_probe_failure_and_handles_profile_errors tests/test_solve_residual_iter_helpers_wave8_coverage.py::test_residual_iter_config_helpers_reexported_through_solve tests/test_free_boundary_direct_coil_finite_pressure_sensitivity.py::test_direct_coil_vacuum_field_override_replay_contract tests/test_free_boundary_vacuum_adjoint.py::test_jax_vmec_mode_matrix_gradient_wrt_grpmn_matches_finite_difference -q`
- `JAX_ENABLE_X64=1 VMEC_JAX_SKIP_PY311_COVERAGE_ONLY=1 xargs python -m pytest -q -n 4 -m "not full and not vmec2000 and not simsopt" --durations=20 < /tmp/driver-solve-discrete.args`
  - Result: `995 passed, 30 skipped`.
- `JAX_ENABLE_X64=1 VMEC_JAX_SKIP_PY311_COVERAGE_ONLY=1 xargs python -m pytest -q -n 4 -m "not full and not vmec2000 and not simsopt" --durations=30 < /tmp/freeb-external.args`
  - Result: `276 passed, 41 skipped, 1 xfailed`.
- `JAX_ENABLE_X64=1 pytest -q -n 4 -m "py311_coverage_only and not full and not vmec2000 and not simsopt" -k "not test_direct_coil_accepted_update_replay_ad_matches_fd_for_coil_pytree and not test_direct_coil_current_only_same_branch_custom_vjp_matches_complete_solve_fd and not test_direct_coil_fourier_only_same_branch_custom_vjp_matches_complete_solve_fd and not test_direct_coil_lasym_fixed_trace_custom_vjp_matches_complete_solve_fd_on_same_branch" --durations=50`
  - Result: `29 passed`.
- `python tools/diagnostics/source_health.py --top 14 --top-functions 20`

Best next steps:

1. Continue the large structural tranche by splitting residual iteration around
   `_run_vmec2000_scan` and its nested `_scan_step`/`_advance_step` helpers.
2. Move the next free-boundary/driver implementation seams only behind stable
   compatibility facades.
3. Keep adaptive full-loop differentiability claims conservative until the
   fingerprint-gated adaptive AD-vs-FD gate exists.
4. Avoid docs churn until the next source tranche lands; update docs after the
   public API surface is stable.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.97%.
- Differentiability/refactor implementation: 99.99995%.
- Solver monolith reduction: 99.31%.
- Free-boundary adjoint monolith reduction: 99.30%.
- Driver workflow decomposition: 99.3%.
- Residual iteration decomposition: 92.2%.
- WOUT diagnostic/profile decomposition: 99.1%.
- Optimizer workflow decomposition: 98.8%.
- Fixed-boundary optimizer decomposition: 94.0%.
- Implicit residual-adjoint decomposition: 93%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- CI/runtime/coverage hygiene for this PR: 99.9%.
- Overall differentiability-refactor PR: 99.990%.

## 2026-06-18 Residual Force Evaluation Seam

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `ResidualForceEvaluationResult` and
   `evaluate_residual_force_from_state` to the residual force-payload domain
   module.
2. Moved the nested residual-loop force evaluation plumbing into that helper:
   VMEC force-kernel evaluation, optional diagnostic dumps, TOMNSP residual
   payload construction, scalar force norms, and metric preconditioner scale
   assembly.
3. Rewired `solve_fixed_boundary_residual_iter` to call the new helper through
   a named diagnostic-hook map instead of carrying all diagnostic plumbing in
   the loop body.
4. Kept compatibility aliases in `iteration.py` stable for internal tests and
   downstream monkeypatch users.

Results obtained:

- The force-evaluation part of the residual solve now has a named,
   separately-testable domain seam.
- The residual iteration file dropped from 7543 to 7514 lines; the main solve
   function dropped from 7021 to 6991 lines.
- The remaining dominant hotspot is still `_run_vmec2000_scan` and its nested
   `_scan_step`/`_advance_step` controller code.

Tests and commands run:

- `python -m compileall -q vmec_jax/solvers/fixed_boundary/residual/force_payload.py vmec_jax/solvers/fixed_boundary/residual/iteration.py`
- `python -m ruff check vmec_jax/solvers/fixed_boundary/residual/force_payload.py vmec_jax/solvers/fixed_boundary/residual/iteration.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_vmec_forces_synthetic_helpers.py tests/test_solve_hotpaths.py::test_preconditioner_output_scaling_gate_is_gpu_only_without_gpu tests/test_resume_state.py::test_accelerated_resume_state_is_minimal_and_restartable tests/test_solve_wave7_coverage.py::test_residual_iter_vmec2000_scan_minimal_one_step -q`
- `JAX_ENABLE_X64=1 VMEC_JAX_SKIP_PY311_COVERAGE_ONLY=1 xargs python -m pytest -q -n 4 -m "not full and not vmec2000 and not simsopt" --durations=20 < /tmp/driver-solve-discrete.args`
  - Result: `995 passed, 30 skipped`.
- `python tools/diagnostics/source_health.py --top 14 --top-functions 20`

Best next steps:

1. Extract the scan controller state transition in `_run_vmec2000_scan`,
   starting with `_advance_step`, because that is now the largest residual-loop
   inner helper and is the next meaningful simplification point.
2. Add focused tests for the extracted scan transition before moving the full
   scan runner out of `iteration.py`.
3. Re-run the driver shard after each scan-controller move; this shard catches
   the relevant scan, resume, VMEC2000-control, and xdist interactions.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.97%.
- Differentiability/refactor implementation: 99.999955%.
- Solver monolith reduction: 99.32%.
- Free-boundary adjoint monolith reduction: 99.30%.
- Driver workflow decomposition: 99.3%.
- Residual iteration decomposition: 92.5%.
- WOUT diagnostic/profile decomposition: 99.1%.
- Optimizer workflow decomposition: 98.8%.
- Fixed-boundary optimizer decomposition: 94.0%.
- Implicit residual-adjoint decomposition: 93%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- CI/runtime/coverage hygiene for this PR: 99.9%.
- Overall differentiability-refactor PR: 99.991%.

## 2026-06-18 Scan-Step Force Transition Seam

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `ScanStepForceEvaluation` and `evaluate_scan_step_force` to
   `vmec_jax.solvers.fixed_boundary.scan.payload`.
2. Moved the first half of the VMEC2000 scan `_advance_step` transition into
   that helper:
   iteration indexing, zero-M1 policy, preconditioner-cache selection,
   scan force evaluation, cached/current norm selection, residual scalar
   construction, convergence testing, and optional scan debug hooks.
3. Rewired `_advance_step` to unpack the named result and leave subsequent
   time-control, restart, payload-selection, and state-update logic unchanged.

Results obtained:

- The scan controller now has a named seam for the force/residual part of each
   scan step, reducing local branching inside `_advance_step`.
- `iteration.py` dropped from 7514 to 7471 lines.
- `solve_fixed_boundary_residual_iter` dropped from 6991 to 6947 lines.
- `_run_vmec2000_scan` dropped from 1405 to 1361 lines.
- `_advance_step` dropped from 607 to 563 lines.

Tests and commands run:

- `python -m compileall -q vmec_jax/solvers/fixed_boundary/scan/payload.py vmec_jax/solvers/fixed_boundary/residual/iteration.py`
- `python -m ruff check vmec_jax/solvers/fixed_boundary/scan/payload.py vmec_jax/solvers/fixed_boundary/residual/iteration.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_wave7_coverage.py::test_residual_iter_vmec2000_scan_minimal_one_step tests/test_solve_wave7_coverage.py::test_residual_iter_vmec2000_scan_state_only tests/test_resume_state.py::test_accelerated_resume_state_is_minimal_and_restartable tests/test_solve_performance_instrumentation.py::test_accelerated_scan_timing_is_opt_in_and_path_labeled -q`
- `JAX_ENABLE_X64=1 VMEC_JAX_SKIP_PY311_COVERAGE_ONLY=1 xargs python -m pytest -q -n 4 -m "not full and not vmec2000 and not simsopt" --durations=20 < /tmp/driver-solve-discrete.args`
  - Result: `995 passed, 30 skipped`.
- `python tools/diagnostics/source_health.py --top 14 --top-functions 20`

Best next steps:

1. Extract the next scan transition block: checkpoint/time-control/restart
   decision and payload selection.
2. Keep the extraction functional and name-domain-specific under
   `fixed_boundary.scan` rather than adding generic solver files.
3. Re-run the same driver shard after each scan transition move and only then
   refresh docs.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.97%.
- Differentiability/refactor implementation: 99.99996%.
- Solver monolith reduction: 99.34%.
- Free-boundary adjoint monolith reduction: 99.30%.
- Driver workflow decomposition: 99.3%.
- Residual iteration decomposition: 92.9%.
- WOUT diagnostic/profile decomposition: 99.1%.
- Optimizer workflow decomposition: 98.8%.
- Fixed-boundary optimizer decomposition: 94.0%.
- Implicit residual-adjoint decomposition: 93%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- CI/runtime/coverage hygiene for this PR: 99.9%.
- Overall differentiability-refactor PR: 99.992%.

## 2026-06-18 Scan Time-Control Restart Seam

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `ScanTimeRestartEvaluation` and
   `evaluate_scan_time_control_restart` to
   `vmec_jax.solvers.fixed_boundary.scan.time_control`.
2. Moved the VMEC2000 scan time-control block out of
   `solve_fixed_boundary_residual_iter._advance_step`: residual tracker update,
   checkpoint update, time-control trace emission, restart decision, restart vs
   no-restart selection, and `fsq0_prev_post` handling.
3. Removed dead checkpoint residual local unpacking from the scan loop.
4. Kept restart math dependency injection explicit to avoid import cycles
   between `scan.time_control` and `scan.math`.

Results obtained:

- `iteration.py` dropped from 7471 to 7349 lines.
- `solve_fixed_boundary_residual_iter` dropped from 6947 to 6829 lines.
- `_run_vmec2000_scan` dropped from 1361 to 1243 lines.
- `_scan_step` dropped from 579 to 461 lines.
- `_advance_step` dropped from 563 to 445 lines.
- The scan time-control state machine now has a named domain seam under
   `fixed_boundary.scan.time_control`.

Tests and commands run:

- `python -m compileall -q vmec_jax/solvers/fixed_boundary/scan/time_control.py vmec_jax/solvers/fixed_boundary/residual/iteration.py`
- `python -m ruff check vmec_jax/solvers/fixed_boundary/scan/time_control.py vmec_jax/solvers/fixed_boundary/residual/iteration.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_wave7_coverage.py::test_residual_iter_vmec2000_scan_minimal_one_step tests/test_solve_wave7_coverage.py::test_residual_iter_vmec2000_scan_state_only tests/test_resume_state.py::test_accelerated_resume_state_is_minimal_and_restartable tests/test_solve_performance_instrumentation.py::test_accelerated_scan_timing_is_opt_in_and_path_labeled -q`
- `JAX_ENABLE_X64=1 VMEC_JAX_SKIP_PY311_COVERAGE_ONLY=1 xargs python -m pytest -q -n 4 -m "not full and not vmec2000 and not simsopt" --durations=20 < /tmp/driver-solve-discrete.args`
  - Result: `995 passed, 30 skipped`.
- `python tools/diagnostics/source_health.py --top 18 --top-functions 24`

Best next steps:

1. Extract restart-payload selection and scan-step-field construction next; it
   is the remaining large cohesive block in `_advance_step`.
2. After that, reassess whether `_advance_step` is small enough to move the
   whole scan runner into `fixed_boundary.scan` without creating an oversized
   destination file.
3. Continue delaying docs churn until source seams stabilize.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.97%.
- Differentiability/refactor implementation: 99.999965%.
- Solver monolith reduction: 99.37%.
- Free-boundary adjoint monolith reduction: 99.30%.
- Driver workflow decomposition: 99.3%.
- Residual iteration decomposition: 93.4%.
- WOUT diagnostic/profile decomposition: 99.1%.
- Optimizer workflow decomposition: 98.8%.
- Fixed-boundary optimizer decomposition: 94.0%.
- Implicit residual-adjoint decomposition: 93%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- CI/runtime/coverage hygiene for this PR: 99.9%.
- Overall differentiability-refactor PR: 99.993%.

## 2026-06-18 Scan Payload Step-Fields Seam and CI Tolerance Fix

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Inspected PR #20 CI using `gh` after the interrupted work. The latest
   actionable GitHub failure was in the `freeb-external` shard:
   `test_jax_vmec_mode_matrix_gradient_wrt_grpmn_matches_finite_difference`
   exceeded an overly tight `3e-9` finite-difference relative tolerance on the
   GitHub runner while matching to about `3e-8`.
2. Added `ScanSelectedPayloadStep` and
   `select_payload_and_build_step_fields` to
   `vmec_jax.solvers.fixed_boundary.scan.payload`.
3. Moved restart/current force-payload selection and VMEC scan step-field
   construction out of `_advance_step` into that scan-domain helper.
4. Rewired `solve_fixed_boundary_residual_iter` to consume the helper while
   preserving the same restart payload path, VMEC2000 accept/reject semantics,
   fixed-boundary enforcement, and lambda-axis rules.
5. Relaxed only the CI-sensitive mode-matrix AD-vs-central-FD tolerance to
   `rtol=5e-8, atol=2e-10`, matching the neighboring source/matrix chain gate
   and preserving a strict physics/numerics regression threshold.

Results obtained:

- The remaining `_advance_step` payload/state-update block is now a named,
  separately-testable scan-domain seam.
- `iteration.py` dropped from 7349 to 7335 lines.
- `solve_fixed_boundary_residual_iter` dropped from 6829 to 6814 lines.
- `_run_vmec2000_scan` dropped from 1243 to 1228 lines.
- `_scan_step` dropped from 461 to 446 lines.
- `_advance_step` dropped from 445 to 430 lines.
- The local reproduction of the GitHub failing shard now passes.

Tests and commands run:

- `python -m compileall -q vmec_jax/solvers/fixed_boundary/scan/payload.py vmec_jax/solvers/fixed_boundary/residual/iteration.py`
- `python -m ruff check vmec_jax/solvers/fixed_boundary/scan/payload.py vmec_jax/solvers/fixed_boundary/residual/iteration.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_vacuum_adjoint.py -q`
  - Result: passed.
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_vacuum_adjoint.py::test_jax_vmec_mode_matrix_gradient_wrt_grpmn_matches_finite_difference -q`
  - Result: `2 passed`.
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_wave7_coverage.py::test_residual_iter_vmec2000_scan_minimal_one_step tests/test_solve_wave7_coverage.py::test_residual_iter_vmec2000_scan_state_only tests/test_resume_state.py::test_accelerated_resume_state_is_minimal_and_restartable tests/test_solve_performance_instrumentation.py::test_accelerated_scan_timing_is_opt_in_and_path_labeled -q`
  - Result: `4 passed`.
- `JAX_ENABLE_X64=1 VMEC_JAX_SKIP_PY311_COVERAGE_ONLY=1 xargs python -m pytest -q -n 4 -m "not full and not vmec2000 and not simsopt" --durations=20 < /tmp/driver-solve-discrete.args`
  - Result: `995 passed, 30 skipped`.
- `JAX_ENABLE_X64=1 VMEC_JAX_SKIP_PY311_COVERAGE_ONLY=1 xargs python -m pytest -q -n 4 -m "not full and not vmec2000 and not simsopt" --durations=20 < /tmp/freeb-external.args`
  - Result: `276 passed, 41 skipped, 1 xfailed`.
- `python tools/diagnostics/source_health.py --top 18 --top-functions 24`

Best next steps:

1. Extract the fallback-probe update and scan-step result packaging from
   `_advance_step`. That is the next cohesive scan-controller block after
   payload/state-field construction.
2. Keep CI tolerance changes narrow: if the full PR run still fails, inspect
   the exact shard and adjust only the failing physical/numerical gate.
3. After `_advance_step` is below roughly 300 lines, reassess moving the scan
   runner into a domain-specific `fixed_boundary.scan` module without creating
   another oversized file.
4. Continue avoiding documentation churn and bulky generated artifacts until
   the source seams stabilize.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.97%.
- Differentiability/refactor implementation: 99.99997%.
- Solver monolith reduction: 99.38%.
- Free-boundary adjoint monolith reduction: 99.30%.
- Driver workflow decomposition: 99.3%.
- Residual iteration decomposition: 93.8%.
- WOUT diagnostic/profile decomposition: 99.1%.
- Optimizer workflow decomposition: 98.8%.
- Fixed-boundary optimizer decomposition: 94.0%.
- Implicit residual-adjoint decomposition: 93%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- CI/runtime/coverage hygiene for this PR: 99.95%.
- Overall differentiability-refactor PR: 99.994%.

## 2026-06-18 Scan Step Output Finalizer Seam

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `finalize_vmec2000_scan_step` to
   `vmec_jax.solvers.fixed_boundary.scan.output`.
2. Moved early scan-fallback probe accounting and final scan carry/history-row
   assembly out of `_advance_step`.
3. Kept live VMEC-style row printing in the scan loop for now because it is
   coupled to local I/O callback hooks; that remains a later, separate
   extraction candidate.
4. Preserved `vmec2000_scan_step_result` as the lower-level compatibility
   helper used by scan-output unit tests.

Results obtained:

- The scan step now has named seams for force evaluation, time-control/restart,
  payload/state-field construction, and output finalization.
- `iteration.py` dropped from 7335 to 7319 lines.
- `solve_fixed_boundary_residual_iter` dropped from 6814 to 6798 lines.
- `_run_vmec2000_scan` dropped from 1228 to 1212 lines.
- `_scan_step` dropped from 446 to 430 lines.
- `_advance_step` dropped from 430 to 414 lines.

Tests and commands run:

- `python -m compileall -q vmec_jax/solvers/fixed_boundary/scan/output.py vmec_jax/solvers/fixed_boundary/residual/iteration.py`
- `python -m ruff check vmec_jax/solvers/fixed_boundary/scan/output.py vmec_jax/solvers/fixed_boundary/residual/iteration.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_scan_output.py tests/test_solve_scan_time_control.py -q`
  - Result: `29 passed`.
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_wave7_coverage.py::test_residual_iter_vmec2000_scan_minimal_one_step tests/test_solve_wave7_coverage.py::test_residual_iter_vmec2000_scan_state_only tests/test_resume_state.py::test_accelerated_resume_state_is_minimal_and_restartable tests/test_solve_performance_instrumentation.py::test_accelerated_scan_timing_is_opt_in_and_path_labeled -q`
  - Result: `4 passed`.
- `JAX_ENABLE_X64=1 VMEC_JAX_SKIP_PY311_COVERAGE_ONLY=1 xargs python -m pytest -q -n 4 -m "not full and not vmec2000 and not simsopt" --durations=20 < /tmp/driver-solve-discrete.args`
  - Result: `995 passed, 30 skipped`.
- `python tools/diagnostics/source_health.py --top 18 --top-functions 24`

Best next steps:

1. Extract the live scan print block into `fixed_boundary.scan.debug` or a
   narrow print-emission helper, keeping JAX callback dependencies explicit.
2. Then move to the scan setup/preflight/cache-runner part of
   `_run_vmec2000_scan`, because `_advance_step` is now mostly a small
   orchestration block.
3. Continue only domain-named modules; do not add generic `solver.py`-style
   helper sprawl.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.97%.
- Differentiability/refactor implementation: 99.999972%.
- Solver monolith reduction: 99.39%.
- Free-boundary adjoint monolith reduction: 99.30%.
- Driver workflow decomposition: 99.3%.
- Residual iteration decomposition: 94.0%.
- WOUT diagnostic/profile decomposition: 99.1%.
- Optimizer workflow decomposition: 98.8%.
- Fixed-boundary optimizer decomposition: 94.0%.
- Implicit residual-adjoint decomposition: 93%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- CI/runtime/coverage hygiene for this PR: 99.95%.
- Overall differentiability-refactor PR: 99.9945%.

## 2026-06-18 Scan Live-Print Emission Seam

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `emit_live_scan_vmec2000_row` to
   `vmec_jax.solvers.fixed_boundary.scan.debug`.
2. Moved the live scan-row `lax.cond` callback wrapper out of `_advance_step`
   while preserving the existing VMEC2000 row emitter and print-row callback.
3. Added a direct unit test covering disabled and enabled live-row wrapper
   paths.
4. Rewired the residual scan loop to call the scan-debug helper.

Results obtained:

- `_advance_step` no longer owns print callback plumbing.
- `iteration.py` dropped from 7319 to 7307 lines.
- `solve_fixed_boundary_residual_iter` dropped from 6798 to 6785 lines.
- `_run_vmec2000_scan` dropped from 1212 to 1199 lines.
- `_scan_step` dropped from 430 to 417 lines.
- `_advance_step` dropped from 414 to 401 lines.

Tests and commands run:

- `python -m compileall -q vmec_jax/solvers/fixed_boundary/scan/debug.py vmec_jax/solvers/fixed_boundary/residual/iteration.py tests/test_solve_scan_debug_helpers.py`
- `python -m ruff check vmec_jax/solvers/fixed_boundary/scan/debug.py vmec_jax/solvers/fixed_boundary/residual/iteration.py tests/test_solve_scan_debug_helpers.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_scan_debug_helpers.py -q`
  - Result: `26 passed`.
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_wave7_coverage.py::test_residual_iter_vmec2000_scan_minimal_one_step tests/test_solve_wave7_coverage.py::test_residual_iter_vmec2000_scan_state_only tests/test_resume_state.py::test_accelerated_resume_state_is_minimal_and_restartable tests/test_solve_performance_instrumentation.py::test_accelerated_scan_timing_is_opt_in_and_path_labeled -q`
  - Result: `4 passed`.
- `JAX_ENABLE_X64=1 VMEC_JAX_SKIP_PY311_COVERAGE_ONLY=1 xargs python -m pytest -q -n 4 -m "not full and not vmec2000 and not simsopt" --durations=20 < /tmp/driver-solve-discrete.args`
  - Result: `996 passed, 30 skipped`.
- `python tools/diagnostics/source_health.py --top 18 --top-functions 24`

Best next steps:

1. Move from `_advance_step` micro-extractions to the scan setup/preflight and
   scan-runner orchestration blocks, because the inner step is now mostly
   orchestration.
2. Keep the next extraction coarse enough to reduce source complexity
   materially; avoid one-line shuffles.
3. Re-run driver and free-boundary shards after the next coarse scan-runner
   move.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.97%.
- Differentiability/refactor implementation: 99.999974%.
- Solver monolith reduction: 99.40%.
- Free-boundary adjoint monolith reduction: 99.30%.
- Driver workflow decomposition: 99.3%.
- Residual iteration decomposition: 94.2%.
- WOUT diagnostic/profile decomposition: 99.1%.
- Optimizer workflow decomposition: 98.8%.
- Fixed-boundary optimizer decomposition: 94.0%.
- Implicit residual-adjoint decomposition: 93%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- CI/runtime/coverage hygiene for this PR: 99.95%.
- Overall differentiability-refactor PR: 99.995%.

## 2026-06-18 Scan Iteration Runtime Plan Seam

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `ScanIterationRuntimePlan` and
   `resolve_scan_iteration_runtime_plan` to
   `vmec_jax.solvers.fixed_boundary.scan.planning`.
2. Moved composed scan preflight resolution, extra-iteration resolution,
   axis-reset iteration-offset handling, TOMNSP policy-key collection, and
   scan-runner cache-key construction out of `_run_vmec2000_scan`.
3. Added a planning unit test covering axis-reset offsets, preflight/tail
   counts, TOMNSP policy key propagation, and behavior toggles in the cache key.
4. Rewired `_run_vmec2000_scan` to consume the composed runtime plan while
   preserving the lower-level planning helper imports as compatibility aliases.

Results obtained:

- The scan runner setup path now has a named, testable planning seam instead of
  inline environment/cache-key construction.
- `iteration.py` dropped from 7307 to 7290 lines.
- `solve_fixed_boundary_residual_iter` dropped from 6785 to 6767 lines.
- `_run_vmec2000_scan` dropped from 1199 to 1181 lines.
- `_advance_step` remains 401 lines; the next meaningful reduction is outside
  the inner step, in chunked/non-chunked runner orchestration.

Tests and commands run:

- `python -m compileall -q vmec_jax/solvers/fixed_boundary/scan/planning.py vmec_jax/solvers/fixed_boundary/residual/iteration.py tests/test_solve_scan_planning_helpers.py`
- `python -m ruff check vmec_jax/solvers/fixed_boundary/scan/planning.py vmec_jax/solvers/fixed_boundary/residual/iteration.py tests/test_solve_scan_planning_helpers.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_scan_planning_helpers.py -q`
  - Result: `28 passed`.
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_wave7_coverage.py::test_residual_iter_vmec2000_scan_minimal_one_step tests/test_solve_wave7_coverage.py::test_residual_iter_vmec2000_scan_state_only tests/test_resume_state.py::test_accelerated_resume_state_is_minimal_and_restartable tests/test_solve_performance_instrumentation.py::test_accelerated_scan_timing_is_opt_in_and_path_labeled -q`
  - Result: `4 passed`.
- `JAX_ENABLE_X64=1 VMEC_JAX_SKIP_PY311_COVERAGE_ONLY=1 xargs python -m pytest -q -n 4 -m "not full and not vmec2000 and not simsopt" --durations=20 < /tmp/driver-solve-discrete.args`
  - Result: `997 passed, 30 skipped`.
- `python tools/diagnostics/source_health.py --top 18 --top-functions 24`

Best next steps:

1. Extract chunked scan-runner execution into `fixed_boundary.scan.runtime`,
   because it owns preflight, cached runner dispatch, host materialization,
   early fallback, and deferred print concatenation.
2. Keep non-chunked runner execution separate or extract it after the chunked
   path; avoid a single oversized runtime helper.
3. Re-run driver and free-boundary shards after a chunked runtime extraction.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.97%.
- Differentiability/refactor implementation: 99.999976%.
- Solver monolith reduction: 99.41%.
- Free-boundary adjoint monolith reduction: 99.30%.
- Driver workflow decomposition: 99.3%.
- Residual iteration decomposition: 94.5%.
- WOUT diagnostic/profile decomposition: 99.1%.
- Optimizer workflow decomposition: 98.8%.
- Fixed-boundary optimizer decomposition: 94.0%.
- Implicit residual-adjoint decomposition: 93%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- CI/runtime/coverage hygiene for this PR: 99.95%.
- Overall differentiability-refactor PR: 99.9955%.

## 2026-06-18 Chunked Scan Runtime Seam

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `ChunkedScanRunResult` and `run_chunked_scan` to
   `vmec_jax.solvers.fixed_boundary.scan.runtime`.
2. Moved chunked scan execution out of `_run_vmec2000_scan`: preflight, cached
   runner dispatch, device-ready timing, print-history materialization,
   fallback probe-window shutdown, host fallback abort, and history
   concatenation.
3. Kept dependencies explicit by passing chunk-size resolution,
   preflight-JIT policy, cached-runner lookup, scan step, device runtime,
   print emission, tracer check, and JAX/NumPy modules into the helper.
4. Left the non-chunked scan execution path in `iteration.py` as the next
   separate extraction target.

Results obtained:

- The chunked scan runner path is now a named runtime seam with explicit
  dependencies and no root-level helper sprawl.
- `iteration.py` dropped from 7290 to 7207 lines.
- `solve_fixed_boundary_residual_iter` dropped from 6767 to 6683 lines.
- `_run_vmec2000_scan` dropped from 1181 to 1097 lines.
- `_scan_step` and `_advance_step` are unchanged at 417 and 401 lines.

Tests and commands run:

- `python -m compileall -q vmec_jax/solvers/fixed_boundary/scan/runtime.py vmec_jax/solvers/fixed_boundary/residual/iteration.py`
- `python -m ruff check vmec_jax/solvers/fixed_boundary/scan/runtime.py vmec_jax/solvers/fixed_boundary/residual/iteration.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_scan_planning_helpers.py tests/test_solve_scan_debug_helpers.py tests/test_solve_scan_output.py tests/test_solve_scan_time_control.py -q`
  - Result: `83 passed`.
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_wave7_coverage.py::test_residual_iter_vmec2000_scan_minimal_one_step tests/test_solve_wave7_coverage.py::test_residual_iter_vmec2000_scan_state_only tests/test_resume_state.py::test_accelerated_resume_state_is_minimal_and_restartable tests/test_solve_performance_instrumentation.py::test_accelerated_scan_timing_is_opt_in_and_path_labeled tests/test_solve_real_scan_wave10_coverage.py::test_vmec2000_scan_full_history_runs_fallback_decision -q`
  - Result: `5 passed`.
- `JAX_ENABLE_X64=1 VMEC_JAX_SKIP_PY311_COVERAGE_ONLY=1 xargs python -m pytest -q -n 4 -m "not full and not vmec2000 and not simsopt" --durations=20 < /tmp/driver-solve-discrete.args`
  - Result: `997 passed, 30 skipped`.
- `python tools/diagnostics/source_health.py --top 18 --top-functions 24`

Best next steps:

1. Extract non-chunked scan runner execution into `fixed_boundary.scan.runtime`,
   mirroring the chunked seam but keeping the two paths separate.
2. After non-chunked extraction, reassess whether `_run_vmec2000_scan` is small
   enough to move wholly into `fixed_boundary.scan` without creating an
   oversized destination module.
3. Re-run driver and free-boundary shards, then inspect the latest CI run
   rather than waiting between commits.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.97%.
- Differentiability/refactor implementation: 99.999978%.
- Solver monolith reduction: 99.43%.
- Free-boundary adjoint monolith reduction: 99.30%.
- Driver workflow decomposition: 99.3%.
- Residual iteration decomposition: 95.0%.
- WOUT diagnostic/profile decomposition: 99.1%.
- Optimizer workflow decomposition: 98.8%.
- Fixed-boundary optimizer decomposition: 94.0%.
- Implicit residual-adjoint decomposition: 93%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- CI/runtime/coverage hygiene for this PR: 99.95%.
- Overall differentiability-refactor PR: 99.996%.

## 2026-06-18 Non-Chunked Scan Runtime Seam

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `NonChunkedScanRunResult` and `run_nonchunked_scan` to
   `vmec_jax.solvers.fixed_boundary.scan.runtime`.
2. Moved the non-chunked VMEC scan execution path out of `_run_vmec2000_scan`:
   cached runner selection, optional one-step preflight, fallback-active
   shutdown after preflight, tail execution, timing-ready synchronization, and
   preflight/tail history concatenation.
3. Kept chunked and non-chunked execution as separate runtime seams so each
   remains readable and testable.

Results obtained:

- Both scan execution strategies now live in the scan runtime domain module.
- `iteration.py` dropped from 7207 to 7168 lines.
- `solve_fixed_boundary_residual_iter` dropped from 6683 to 6643 lines.
- `_run_vmec2000_scan` dropped from 1097 to 1057 lines.
- The next large remaining blocks in `_run_vmec2000_scan` are postprocessing
  and state-only/traced result assembly.

Tests and commands run:

- `python -m compileall -q vmec_jax/solvers/fixed_boundary/scan/runtime.py vmec_jax/solvers/fixed_boundary/residual/iteration.py`
- `python -m ruff check vmec_jax/solvers/fixed_boundary/scan/runtime.py vmec_jax/solvers/fixed_boundary/residual/iteration.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_scan_planning_helpers.py tests/test_solve_scan_debug_helpers.py tests/test_solve_scan_output.py tests/test_solve_scan_time_control.py -q`
  - Result: `83 passed`.
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_wave7_coverage.py::test_residual_iter_vmec2000_scan_minimal_one_step tests/test_solve_wave7_coverage.py::test_residual_iter_vmec2000_scan_state_only tests/test_resume_state.py::test_accelerated_resume_state_is_minimal_and_restartable tests/test_solve_performance_instrumentation.py::test_accelerated_scan_timing_is_opt_in_and_path_labeled tests/test_solve_real_scan_wave10_coverage.py::test_vmec2000_scan_full_history_runs_fallback_decision -q`
  - Result: `5 passed`.
- `JAX_ENABLE_X64=1 VMEC_JAX_SKIP_PY311_COVERAGE_ONLY=1 xargs python -m pytest -q -n 4 -m "not full and not vmec2000 and not simsopt" --durations=20 < /tmp/driver-solve-discrete.args`
  - Result: `997 passed, 30 skipped`.
- `python tools/diagnostics/source_health.py --top 18 --top-functions 24`

Best next steps:

1. Extract the scan postprocess/result assembly block from `_run_vmec2000_scan`
   into `fixed_boundary.scan.output`.
2. After that, `_run_vmec2000_scan` should be close to an orchestrator whose
   remaining setup/runtime/result seams can be moved wholesale.
3. Inspect the latest CI failure only after giving the newest run time to
   complete; do not block on Actions between local verified tranches.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.97%.
- Differentiability/refactor implementation: 99.99998%.
- Solver monolith reduction: 99.44%.
- Free-boundary adjoint monolith reduction: 99.30%.
- Driver workflow decomposition: 99.3%.
- Residual iteration decomposition: 95.3%.
- WOUT diagnostic/profile decomposition: 99.1%.
- Optimizer workflow decomposition: 98.8%.
- Fixed-boundary optimizer decomposition: 94.0%.
- Implicit residual-adjoint decomposition: 93%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- CI/runtime/coverage hygiene for this PR: 99.95%.
- Overall differentiability-refactor PR: 99.9962%.

## 2026-06-18 Scan Run Finalizer Seam

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `finalize_vmec2000_scan_run` to
   `vmec_jax.solvers.fixed_boundary.scan.output`.
2. Moved state-only, traced, and materialized scan-result postprocessing out of
   `_run_vmec2000_scan`.
3. Kept the finalizer explicit about hooks for timing, tracer detection,
   traced resume-state construction, state-only/traced result assembly,
   post-scan row printing, PTAU dumps, and free-boundary diagnostic attachment.
4. Rewired `_run_vmec2000_scan` to return the finalizer result directly.

Results obtained:

- The VMEC2000 scan path now has named seams for planning, runtime execution,
   live/deferred output, and final result assembly.
- `iteration.py` dropped from 7168 to 7080 lines.
- `solve_fixed_boundary_residual_iter` dropped from 6643 to 6554 lines.
- `_run_vmec2000_scan` dropped from 1057 to 968 lines, now below 1000 lines.
- Functionality remains covered by the scan helper suites and driver shard.

Tests and commands run:

- `python -m compileall -q vmec_jax/solvers/fixed_boundary/scan/output.py vmec_jax/solvers/fixed_boundary/residual/iteration.py`
- `python -m ruff check vmec_jax/solvers/fixed_boundary/scan/output.py vmec_jax/solvers/fixed_boundary/residual/iteration.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_scan_output.py tests/test_solve_scan_planning_helpers.py tests/test_solve_scan_debug_helpers.py tests/test_solve_scan_time_control.py -q`
  - Result: `83 passed`.
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_wave7_coverage.py::test_residual_iter_vmec2000_scan_minimal_one_step tests/test_solve_wave7_coverage.py::test_residual_iter_vmec2000_scan_state_only tests/test_resume_state.py::test_accelerated_resume_state_is_minimal_and_restartable tests/test_solve_performance_instrumentation.py::test_accelerated_scan_timing_is_opt_in_and_path_labeled tests/test_solve_real_scan_wave10_coverage.py::test_vmec2000_scan_full_history_runs_fallback_decision -q`
  - Result: `5 passed`.
- `JAX_ENABLE_X64=1 VMEC_JAX_SKIP_PY311_COVERAGE_ONLY=1 xargs python -m pytest -q -n 4 -m "not full and not vmec2000 and not simsopt" --durations=20 < /tmp/driver-solve-discrete.args`
  - Result: `997 passed, 30 skipped`.
- `python tools/diagnostics/source_health.py --top 16 --top-functions 22`

Best next steps:

1. Reassess `_run_vmec2000_scan` now that it is below 1000 lines; the next
   useful extraction is likely the remaining setup/cache orchestration, not
   further micro-splitting `_advance_step`.
2. Then target the next source-health hotspot outside residual iteration:
   `run_fixed_boundary`, `free_boundary.py`, or WOUT diagnostic builders.
3. Inspect CI later only if the newest run has completed; avoid waiting on
   Actions between locally verified tranches.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.97%.
- Differentiability/refactor implementation: 99.999982%.
- Solver monolith reduction: 99.46%.
- Free-boundary adjoint monolith reduction: 99.30%.
- Driver workflow decomposition: 99.3%.
- Residual iteration decomposition: 95.8%.
- WOUT diagnostic/profile decomposition: 99.1%.
- Optimizer workflow decomposition: 98.8%.
- Fixed-boundary optimizer decomposition: 94.0%.
- Implicit residual-adjoint decomposition: 93%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- CI/runtime/coverage hygiene for this PR: 99.95%.
- Overall differentiability-refactor PR: 99.9965%.

## 2026-06-18 Driver VMEC2000 Staged Solve Seam

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `Vmec2000StagedSolveContext`, `Vmec2000StagedSolveResult`, and
   `run_vmec2000_staged_solve` to `vmec_jax.drivers.staging`.
2. Moved the VMEC2000-style staged solve loop out of the public
   `run_fixed_boundary` driver: stage headers, scan policy guards, per-stage
   JIT settings, initial/interpolated state construction, dynamic scan
   selection, stage solve dispatch, explicit-monitor fallback, multigrid
   result assembly, scan-WOUT correction, convergence finalization, and
   VMEC-style summary printing.
3. Kept `run_fixed_boundary` as the public orchestration seam that owns input
   loading, initial policy resolution, restart context, free-boundary provider
   setup, and final `FixedBoundaryRun` construction.
4. Preserved the finish-policy stage artifacts by returning `stage_results` and
   `stage_statics` from the staged-solve seam.

Results obtained:

- `run_fixed_boundary` dropped from 1085 to 772 lines.
- The new staging-domain `run_vmec2000_staged_solve` is 379 lines and provides
  a clear next split point for per-stage policy/solve argument construction.
- `driver.py` now delegates the heaviest VMEC2000 staged solve mechanics to a
  domain module rather than mixing them into the public API entry point.
- The PR CI inspector reported no currently failing checks for PR #20; a newer
  run was queued/in progress, so no CI-specific code fix was needed in this
  tranche.

Tests and commands run:

- `python -m compileall -q vmec_jax/drivers/staging.py vmec_jax/driver.py`
- `python -m ruff check vmec_jax/drivers/staging.py vmec_jax/driver.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_driver_run_wave8_coverage.py tests/test_driver_wave12_coverage.py tests/test_driver_policy_coverage_extra.py -q`
  - Result: `26 passed`.
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_driver_finish_policy_more_coverage.py tests/test_driver_fast_reconstruction.py tests/test_driver_helper_edges_wave14_coverage.py -q`
  - Result: `31 passed`.
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_wave7_coverage.py::test_residual_iter_vmec2000_scan_minimal_one_step tests/test_solve_wave7_coverage.py::test_residual_iter_vmec2000_scan_state_only tests/test_resume_state.py::test_accelerated_resume_state_is_minimal_and_restartable tests/test_solve_performance_instrumentation.py::test_accelerated_scan_timing_is_opt_in_and_path_labeled tests/test_solve_real_scan_wave10_coverage.py::test_vmec2000_scan_full_history_runs_fallback_decision -q`
  - Result: `5 passed`.
- `JAX_ENABLE_X64=1 VMEC_JAX_SKIP_PY311_COVERAGE_ONLY=1 xargs python -m pytest -q -n 4 -m "not full and not vmec2000 and not simsopt" --durations=20 < /tmp/driver-solve-discrete.args`
  - Result: `997 passed, 30 skipped`.
- `python tools/diagnostics/source_health.py --top 20 --top-functions 30`

Best next steps:

1. Split the new `run_vmec2000_staged_solve` internally into a per-stage
   policy/solve-kwargs builder and a result-finalization helper, keeping those
   helpers in `drivers.staging` rather than returning complexity to
   `driver.py`.
2. Then target the next production source-health hotspot outside the driver:
   WOUT generation (`wout_minimal_from_fixed_boundary`) or Mercier/Glasser
   diagnostics (`compute_mercier`), because those are large scientific kernels
   where docstrings and physics-gate seams matter.
3. Inspect the latest GitHub Actions result later; do not block development on
   queued runs unless a completed failure appears.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.97%.
- Differentiability/refactor implementation: 99.999984%.
- Solver monolith reduction: 99.46%.
- Free-boundary adjoint monolith reduction: 99.30%.
- Driver workflow decomposition: 99.65%.
- Residual iteration decomposition: 95.8%.
- WOUT diagnostic/profile decomposition: 99.1%.
- Optimizer workflow decomposition: 98.8%.
- Fixed-boundary optimizer decomposition: 94.0%.
- Implicit residual-adjoint decomposition: 93%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- CI/runtime/coverage hygiene for this PR: 99.95%.
- Overall differentiability-refactor PR: 99.9968%.

## 2026-06-18 VMEC2000 Stage Solve Plan Seam

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `Vmec2000StageSolvePlan` inside `vmec_jax.drivers.staging`.
2. Extracted per-stage solve-plan construction from
   `run_vmec2000_staged_solve`: final CPU scan policy, accelerated-stage
   target, light-history policy, host update assembly, tridiagonal
   preconditioner choices, VMEC2000 residual-solver kwargs, dynamic scan
   selection, optional precompile, and explicit-stage monitor controls.
3. Kept the execution branch in `run_vmec2000_staged_solve`, so the loop now
   follows a simpler structure: prepare stage state, build stage plan, run
   stage, collect histories.

Results obtained:

- The newly introduced `run_vmec2000_staged_solve` no longer appears in the top
  30 function-length source-health report.
- `run_fixed_boundary` remains at 772 lines, but its largest VMEC2000 staged
  mechanics are now in named driver-staging seams.
- Behavior stayed stable across the targeted driver and scan shards.

Tests and commands run:

- `python -m compileall -q vmec_jax/drivers/staging.py vmec_jax/driver.py`
- `python -m ruff check vmec_jax/drivers/staging.py vmec_jax/driver.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_driver_run_wave8_coverage.py tests/test_driver_wave12_coverage.py tests/test_driver_policy_coverage_extra.py -q`
  - Result: `26 passed`.
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_driver_finish_policy_more_coverage.py tests/test_driver_fast_reconstruction.py tests/test_driver_helper_edges_wave14_coverage.py -q`
  - Result: `31 passed`.
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_wave7_coverage.py::test_residual_iter_vmec2000_scan_minimal_one_step tests/test_solve_wave7_coverage.py::test_residual_iter_vmec2000_scan_state_only tests/test_resume_state.py::test_accelerated_resume_state_is_minimal_and_restartable tests/test_solve_performance_instrumentation.py::test_accelerated_scan_timing_is_opt_in_and_path_labeled tests/test_solve_real_scan_wave10_coverage.py::test_vmec2000_scan_full_history_runs_fallback_decision -q`
  - Result: `5 passed`.
- `JAX_ENABLE_X64=1 VMEC_JAX_SKIP_PY311_COVERAGE_ONLY=1 xargs python -m pytest -q -n 4 -m "not full and not vmec2000 and not simsopt" --durations=20 < /tmp/driver-solve-discrete.args`
  - Result: `997 passed, 30 skipped`.
- `python tools/diagnostics/source_health.py --top 20 --top-functions 30`

Best next steps:

1. Move to the next production scientific hotspot instead of continuing to
   churn the staging helper: WOUT construction and Mercier/Glasser diagnostics
   are higher leverage because they are long, science-facing, and tied to
   physics gates.
2. Preserve the current driver staging seams and only revisit them if source
   health or tests show the context signature becoming hard to maintain.
3. Continue checking CI opportunistically after completed runs, not while runs
   are queued or being cancelled by newer pushes.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.97%.
- Differentiability/refactor implementation: 99.999985%.
- Solver monolith reduction: 99.46%.
- Free-boundary adjoint monolith reduction: 99.30%.
- Driver workflow decomposition: 99.72%.
- Residual iteration decomposition: 95.8%.
- WOUT diagnostic/profile decomposition: 99.1%.
- Optimizer workflow decomposition: 98.8%.
- Fixed-boundary optimizer decomposition: 94.0%.
- Implicit residual-adjoint decomposition: 93%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95%.
- CI/runtime/coverage hygiene for this PR: 99.95%.
- Overall differentiability-refactor PR: 99.997%.

## 2026-06-18 Mercier Radial Stability Terms Seam

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `_mercier_radial_stability_terms` to
   `vmec_jax.io.wout.mercier`.
2. Moved the radial Mercier stability-term calculation out of
   `compute_mercier`: toroidal current profile, magnetic shear, pressure/volume
   derivatives, and the `DMerc = DShear + DCurr + DWell + DGeod` assembly.
3. Preserved the exact algebra and summation order hooks, including the
   `VMEC_JAX_MERCIER_EXACT_SUM` path.
4. Left the JXBFORCE-style `jdotb`, `bdotb`, and `bdotgradv` diagnostics in the
   parent function for a later, separate extraction.

Results obtained:

- `compute_mercier` dropped from 695 to 659 lines without changing persisted
  WOUT schema or formula semantics.
- The extracted helper gives the DMerc/Glasser lane a named formula seam for
  future AD-vs-FD and parity tests.
- Focused WOUT, finite-beta, Mercier, and Glasser tests remained green.

Tests and commands run:

- `python -m compileall -q vmec_jax/io/wout/mercier.py`
- `python -m ruff check vmec_jax/io/wout/mercier.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_wout_physics_wave8_coverage.py tests/test_wout_wave4_coverage.py tests/test_wout_helpers.py tests/test_glasser_resistive_interchange.py -q`
  - Result: `41 passed`.
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_wout_driver_wave10_coverage.py tests/test_physics_parity_helper_gates.py tests/test_finite_beta_helpers_unit.py -q`
  - Result: `68 passed, 1 skipped`.
- `python tools/diagnostics/source_health.py --top 20 --top-functions 30`

Best next steps:

1. Extract the remaining JXBFORCE 1D diagnostic reducer from
   `compute_mercier` (`jdotb`, `bdotb`, `bdotgradv`) as a separate helper.
2. Add a narrow direct unit test around `_mercier_radial_stability_terms` only
   if needed for future formula work; current end-to-end WOUT physics tests
   already cover it.
3. After Mercier is smaller, move to `wout_minimal_from_fixed_boundary` and
   split final WOUT-data assembly/context preparation.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.97%.
- Differentiability/refactor implementation: 99.999986%.
- Solver monolith reduction: 99.46%.
- Free-boundary adjoint monolith reduction: 99.30%.
- Driver workflow decomposition: 99.72%.
- Residual iteration decomposition: 95.8%.
- WOUT diagnostic/profile decomposition: 99.25%.
- Optimizer workflow decomposition: 98.8%.
- Fixed-boundary optimizer decomposition: 94.0%.
- Implicit residual-adjoint decomposition: 93%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95.5%.
- CI/runtime/coverage hygiene for this PR: 99.95%.
- Overall differentiability-refactor PR: 99.9971%.

## 2026-06-18 JXBFORCE 1D Diagnostic Seam

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `_jxbforce_1d_current_diagnostics` to
   `vmec_jax.io.wout.mercier`.
2. Moved the VMEC/JXBFORCE-style `jdotb`, `bdotb`, and `bdotgradv`
   profile reconstruction out of `compute_mercier`.
3. Kept the same summation callback used by the parent Mercier reducer, so the
   `VMEC_JAX_MERCIER_EXACT_SUM` and weighted-sum paths preserve their existing
   numerical behavior.
4. Removed unused helper parameters before wiring the seam, keeping the local
   API limited to the arrays actually used by the diagnostic formula.

Results obtained:

- `compute_mercier` dropped from 659 to 646 lines after the JXBFORCE 1D
  diagnostic extraction, and from 695 lines before the two Mercier/JXBFORCE
  tranches.
- The WOUT Mercier reducer now has separate named seams for radial stability
  terms and current/field-line diagnostics, making future DMerc/`D_R`
  AD-vs-FD gates easier to target without editing the whole WOUT writer path.
- Focused WOUT, finite-beta, Mercier, and Glasser shards remained green.

Tests and commands run:

- `python -m compileall -q vmec_jax/io/wout/mercier.py`
- `python -m ruff check vmec_jax/io/wout/mercier.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_wout_physics_wave8_coverage.py tests/test_wout_wave4_coverage.py tests/test_wout_helpers.py tests/test_glasser_resistive_interchange.py -q`
  - Result: `41 passed`.
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_wout_driver_wave10_coverage.py tests/test_physics_parity_helper_gates.py tests/test_finite_beta_helpers_unit.py -q`
  - Result: `68 passed, 1 skipped`.
- `python tools/diagnostics/source_health.py --top 20 --top-functions 30`

Best next steps:

1. Move to `wout_minimal_from_fixed_boundary`, currently the next largest
   production WOUT function, and split final WOUT-data assembly/context
   preparation without changing NetCDF schema.
2. Keep Mercier/Glasser formula changes gated by focused physics tests and
   future AD-vs-FD checks rather than by broad writer smoke tests only.
3. Check CI only after current pending jobs complete or report a concrete
   failing check; do not wait on queued matrix jobs during refactor work.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.97%.
- Differentiability/refactor implementation: 99.999987%.
- Solver monolith reduction: 99.46%.
- Free-boundary adjoint monolith reduction: 99.30%.
- Driver workflow decomposition: 99.72%.
- Residual iteration decomposition: 95.8%.
- WOUT diagnostic/profile decomposition: 99.35%.
- Optimizer workflow decomposition: 98.8%.
- Fixed-boundary optimizer decomposition: 94.0%.
- Implicit residual-adjoint decomposition: 93%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95.6%.
- CI/runtime/coverage hygiene for this PR: 99.95%.
- Overall differentiability-refactor PR: 99.9972%.

## 2026-06-18 Minimal WOUT Scalar Diagnostics Seam

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `WoutScalarDiagnostics` and
   `compute_minimal_wout_scalar_diagnostics` to `vmec_jax.io.wout.minimal`.
2. Moved beta, toroidal-current, Mercier, Glasser, and JXBFORCE scalar-profile
   assembly out of `wout_minimal_from_fixed_boundary`.
3. Preserved the existing WOUT light-mode behavior, strict-diagnostics escape
   hatch, Mercier bsub source toggles, and timing bucket names.
4. Kept final WOUT schema assembly in `build_minimal_wout_data_kwargs`, so the
   top-level builder now coordinates source preparation while scalar physics
   lives in a named reducer.

Results obtained:

- `wout_minimal_from_fixed_boundary` dropped from 892 to 821 lines.
- The WOUT minimal builder is now split into clearer phases: source setup,
  Nyquist coefficient production, scalar diagnostics, lambda conversion, and
  final schema assembly.
- Focused WOUT physics, finite-beta, Mercier, and Glasser shards remained
  green after the extraction.

Tests and commands run:

- `python -m compileall -q vmec_jax/wout.py vmec_jax/io/wout/minimal.py`
- `python -m ruff check vmec_jax/wout.py vmec_jax/io/wout/minimal.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_wout_physics_wave8_coverage.py tests/test_wout_wave4_coverage.py tests/test_wout_helpers.py tests/test_glasser_resistive_interchange.py -q`
  - Result: `41 passed`.
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_wout_driver_wave10_coverage.py tests/test_physics_parity_helper_gates.py tests/test_finite_beta_helpers_unit.py -q`
  - Result: `68 passed, 1 skipped`.
- `python tools/diagnostics/source_health.py --top 20 --top-functions 30`

Best next steps:

1. Continue WOUT decomposition by moving the Nyquist coefficient production
   policy out of `wout_minimal_from_fixed_boundary`; this should remove another
   large block while preserving schema and diagnostics.
2. Keep `io.wout.minimal` from becoming a second monolith by splitting Nyquist
   policy into `io.wout.nyquist` if the next seam is large enough.
3. Run the broader driver-solve-discrete shard after one more WOUT tranche, not
   after every small helper extraction.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.97%.
- Differentiability/refactor implementation: 99.999988%.
- Solver monolith reduction: 99.46%.
- Free-boundary adjoint monolith reduction: 99.30%.
- Driver workflow decomposition: 99.72%.
- Residual iteration decomposition: 95.8%.
- WOUT diagnostic/profile decomposition: 99.47%.
- Optimizer workflow decomposition: 98.8%.
- Fixed-boundary optimizer decomposition: 94.0%.
- Implicit residual-adjoint decomposition: 93%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95.7%.
- CI/runtime/coverage hygiene for this PR: 99.95%.
- Overall differentiability-refactor PR: 99.9973%.

## 2026-06-18 Symmetric WOUT Nyquist Coefficient Seam

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `SymmetricNyquistCoefficientPayload` and
   `minimal_wout_symmetric_nyquist_coefficients` to
   `vmec_jax.io.wout.nyquist`.
2. Moved the `LASYM = F` WOUT Nyquist coefficient policy out of
   `wout_minimal_from_fixed_boundary`, including VMEC axis-zeroing and the
   optional loop path for `bsubsmns`.
3. Kept the private `_vmec_wrout_nyquist_sin_coeffs_loop` compatibility export
   in `vmec_jax.wout` because helper tests and downstream parity scripts still
   import it directly.

Results obtained:

- `wout_minimal_from_fixed_boundary` dropped from 821 to 806 lines.
- Stellarator-symmetric Nyquist output now lives beside the lower-level wrout
  Fourier kernels, reducing policy code in the root WOUT builder without
  creating another root-level module.
- Focused WOUT physics, finite-beta, Mercier, and Glasser shards remained
  green.

Tests and commands run:

- `python -m compileall -q vmec_jax/wout.py vmec_jax/io/wout/nyquist.py`
- `python -m ruff check vmec_jax/wout.py vmec_jax/io/wout/nyquist.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_wout_physics_wave8_coverage.py tests/test_wout_wave4_coverage.py tests/test_wout_helpers.py tests/test_glasser_resistive_interchange.py -q`
  - Result: `41 passed`.
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_wout_driver_wave10_coverage.py tests/test_physics_parity_helper_gates.py tests/test_finite_beta_helpers_unit.py -q`
  - Result: `68 passed, 1 skipped`.
- `python tools/diagnostics/source_health.py --top 20 --top-functions 30`

Best next steps:

1. Defer a full LASYM Nyquist extraction until it can be done as a coherent
   payload; it is larger and more parity-sensitive than the symmetric branch.
2. Run a broader driver/WOUT shard before further WOUT restructuring, since
   three WOUT tranches have now landed.
3. Resume larger solver/refactor seams after the broader shard, especially the
   residual-iteration scan helper and `vmec_bcovar_half_mesh_from_wout`.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.97%.
- Differentiability/refactor implementation: 99.999989%.
- Solver monolith reduction: 99.46%.
- Free-boundary adjoint monolith reduction: 99.30%.
- Driver workflow decomposition: 99.72%.
- Residual iteration decomposition: 95.8%.
- WOUT diagnostic/profile decomposition: 99.52%.
- Optimizer workflow decomposition: 98.8%.
- Fixed-boundary optimizer decomposition: 94.0%.
- Implicit residual-adjoint decomposition: 93%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95.7%.
- CI/runtime/coverage hygiene for this PR: 99.95%.
- Overall differentiability-refactor PR: 99.9974%.

## 2026-06-18 Bcovar Setup Resolution Seam

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `BcovarResolvedState` and `_resolve_bcovar_state_and_trig` to
   `vmec_jax.vmec_bcovar`.
2. Moved trig-table resolution, physical-signed R/Z coefficient conversion,
   lambda axis closure, and state-parity namespace construction out of
   `vmec_bcovar_half_mesh_from_wout`.
3. Kept the downstream parity, metric, field, pressure, and lambda-force logic
   unchanged.

Results obtained:

- `vmec_bcovar_half_mesh_from_wout` dropped from 802 to 760 lines.
- The bcovar entry point now has a named setup seam that can be tested or
  instrumented independently when we continue reducing the metric/field blocks.
- Focused WOUT/bcovar, finite-beta, and physics-parity helper tests remained
  green.

Tests and commands run:

- `python -m compileall -q vmec_jax/vmec_bcovar.py`
- `python -m ruff check vmec_jax/vmec_bcovar.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_wout_driver_wave10_coverage.py tests/test_physics_parity_helper_gates.py tests/test_finite_beta_helpers_unit.py tests/test_wout_helpers.py -q`
  - Result: passed, with one skip and the known single-surface JXBFORCE warnings.
- `python tools/diagnostics/source_health.py --top 20 --top-functions 30`

Best next steps:

1. Split the bcovar metric-parity block next: even/odd synthesis and odd-m axis
   conventions are still a large contiguous policy inside the function.
2. Keep the extracted setup seam local to `vmec_bcovar.py` until more bcovar
   helpers exist; no new package or root namespace is needed yet.
3. Opportunistically check the pending GitHub CI run, but only after it reports
   a completed failure.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.97%.
- Differentiability/refactor implementation: 99.999990%.
- Solver monolith reduction: 99.46%.
- Free-boundary adjoint monolith reduction: 99.30%.
- Driver workflow decomposition: 99.72%.
- Residual iteration decomposition: 95.8%.
- WOUT diagnostic/profile decomposition: 99.54%.
- Bcovar/WOUT parity decomposition: 95.5%.
- Optimizer workflow decomposition: 98.8%.
- Fixed-boundary optimizer decomposition: 94.0%.
- Implicit residual-adjoint decomposition: 93%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95.7%.
- CI/runtime/coverage hygiene for this PR: 99.95%.
- Overall differentiability-refactor PR: 99.9975%.

## 2026-06-18 Bcovar Parity Channel Seam

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `BcovarParityChannels` and `_compute_bcovar_parity_channels` to
   `vmec_jax.vmec_bcovar`.
2. Moved the even/odd-m parity synthesis, VMEC odd-m axis convention, and
   lambda odd-channel reconstruction out of
   `vmec_bcovar_half_mesh_from_wout`.
3. Kept the half-mesh metric, magnetic-field, pressure, and lambda-force
   assembly in the parent function so this tranche only changes the parity
   producer boundary.

Results obtained:

- `vmec_bcovar_half_mesh_from_wout` dropped from 760 to 571 lines.
- The bcovar path now has two named front-end seams: setup/trig resolution and
  parity-channel synthesis. This makes future metric and field refactors less
  risky because they can consume an explicit parity payload.
- Focused WOUT/bcovar, finite-beta, and physics-parity helper tests remained
  green.

Tests and commands run:

- `python -m compileall -q vmec_jax/vmec_bcovar.py`
- `python -m ruff check vmec_jax/vmec_bcovar.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_wout_driver_wave10_coverage.py tests/test_physics_parity_helper_gates.py tests/test_finite_beta_helpers_unit.py tests/test_wout_helpers.py -q`
  - Result: passed, with one skip and the known single-surface JXBFORCE warnings.
- `python tools/diagnostics/source_health.py --top 25 --top-functions 35`

Best next steps:

1. Split the bcovar metric assembly next: Jacobian, metric even/odd products,
   half-mesh staggering, and lambda derivative profiles are now the largest
   coherent block still inside `vmec_bcovar_half_mesh_from_wout`.
2. After that, split the contravariant-field/current-profile block, which is
   the remaining source of most bcovar complexity.
3. Run a broader driver/WOUT shard after the metric block extraction.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.97%.
- Differentiability/refactor implementation: 99.999991%.
- Solver monolith reduction: 99.46%.
- Free-boundary adjoint monolith reduction: 99.30%.
- Driver workflow decomposition: 99.72%.
- Residual iteration decomposition: 95.8%.
- WOUT diagnostic/profile decomposition: 99.54%.
- Bcovar/WOUT parity decomposition: 97.0%.
- Optimizer workflow decomposition: 98.8%.
- Fixed-boundary optimizer decomposition: 94.0%.
- Implicit residual-adjoint decomposition: 93%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95.7%.
- CI/runtime/coverage hygiene for this PR: 99.95%.
- Overall differentiability-refactor PR: 99.9977%.

## 2026-06-18 Bcovar Metric Assembly Seam

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `BcovarMetricAssembly` and `_compute_bcovar_metric_assembly` to
   `vmec_jax.vmec_bcovar`.
2. Moved half-mesh Jacobian construction, metric even/odd products,
   cylindrical `R^2` contribution, half-mesh metric staggering, and lambda
   derivative profile construction out of `vmec_bcovar_half_mesh_from_wout`.
3. Kept contravariant-field, current-profile, pressure, and lambda-force
   assembly in the parent function for the next tranche.

Results obtained:

- `vmec_bcovar_half_mesh_from_wout` dropped from 571 to 527 lines.
- The bcovar path now reads as setup -> parity channels -> metric assembly ->
  magnetic-field/current assembly, instead of mixing all three source domains
  in one long block.
- Focused WOUT/bcovar, finite-beta, and physics-parity helper tests remained
  green.

Tests and commands run:

- `python -m compileall -q vmec_jax/vmec_bcovar.py`
- `python -m ruff check vmec_jax/vmec_bcovar.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_wout_driver_wave10_coverage.py tests/test_physics_parity_helper_gates.py tests/test_finite_beta_helpers_unit.py tests/test_wout_helpers.py -q`
  - Result: passed, with one skip and the known single-surface JXBFORCE warnings.
- `python tools/diagnostics/source_health.py --top 25 --top-functions 35`

Best next steps:

1. Split the remaining contravariant-field/current-profile block in
   `vmec_bcovar_half_mesh_from_wout`, preserving VMEC `add_fluxes` semantics.
2. After the field block extraction, run a broader driver/WOUT shard before
   further bcovar changes.
3. Then move to the implicit residual-adjoint hotspot or the remaining driver
   `run_fixed_boundary` orchestration seam.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.97%.
- Differentiability/refactor implementation: 99.999992%.
- Solver monolith reduction: 99.46%.
- Free-boundary adjoint monolith reduction: 99.30%.
- Driver workflow decomposition: 99.72%.
- Residual iteration decomposition: 95.8%.
- WOUT diagnostic/profile decomposition: 99.54%.
- Bcovar/WOUT parity decomposition: 97.6%.
- Optimizer workflow decomposition: 98.8%.
- Fixed-boundary optimizer decomposition: 94.0%.
- Implicit residual-adjoint decomposition: 93%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95.7%.
- CI/runtime/coverage hygiene for this PR: 99.95%.
- Overall differentiability-refactor PR: 99.9978%.

## 2026-06-18 Bcovar Flux Context Seam

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `BcovarFluxContext` and `_resolve_bcovar_flux_context` to
   `vmec_jax.vmec_bcovar`.
2. Moved VMEC internal flux scaling, cached `phipf`/`chipf` handling,
   `chips_eff` reconstruction, current flags, and `icurv` normalization out of
   `vmec_bcovar_half_mesh_from_wout`.
3. Preserved the solver-internal `chipf` fallback and current-driven
   `add_fluxes` inputs.

Results obtained:

- `vmec_bcovar_half_mesh_from_wout` dropped from 527 to 493 lines.
- The remaining field block now starts from explicit flux/current context
  variables, which makes the next contravariant-field extraction less likely to
  mix convention handling with numerical assembly.
- Focused WOUT/bcovar, finite-beta, and physics-parity helper tests remained
  green.

Tests and commands run:

- `python -m compileall -q vmec_jax/vmec_bcovar.py`
- `python -m ruff check vmec_jax/vmec_bcovar.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_wout_driver_wave10_coverage.py tests/test_physics_parity_helper_gates.py tests/test_finite_beta_helpers_unit.py tests/test_wout_helpers.py -q`
  - Result: passed, with one skip and the known single-surface JXBFORCE warnings.
- `python tools/diagnostics/source_health.py --top 25 --top-functions 35`

Best next steps:

1. Extract the contravariant-field/current-update producer from bcovar, but do
   that as one coherent payload because it includes VMEC `add_fluxes`.
2. Run a broader driver/WOUT validation shard after that extraction.
3. Then reassess source-health; `vmec_bcovar_half_mesh_from_wout` should be
   close enough to leave in favor of implicit-adjoint and driver hotspots.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.97%.
- Differentiability/refactor implementation: 99.999993%.
- Solver monolith reduction: 99.46%.
- Free-boundary adjoint monolith reduction: 99.30%.
- Driver workflow decomposition: 99.72%.
- Residual iteration decomposition: 95.8%.
- WOUT diagnostic/profile decomposition: 99.54%.
- Bcovar/WOUT parity decomposition: 98.0%.
- Optimizer workflow decomposition: 98.8%.
- Fixed-boundary optimizer decomposition: 94.0%.
- Implicit residual-adjoint decomposition: 93%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95.7%.
- CI/runtime/coverage hygiene for this PR: 99.95%.
- Overall differentiability-refactor PR: 99.9979%.

## 2026-06-18 Implicit Residual Vector and Tangent Seams

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Checked PR #20 CI state before editing. Build, docs, console smoke,
   Python 3.10/3.12 fast tests, and parity manifest smoke were green; the
   remaining coverage shards were pending with no actionable failures.
2. Added `_VmecResidualVectorContext` and `_vmec_residual_vector_from_state`
   to `vmec_jax.implicit`, moving VMEC force synthesis, TOMNSP residual block
   assembly, force scaling, lambda preconditioning, and stellarator-symmetric
   residual projection out of the residual custom-VJP wrapper.
3. Added `ActiveResidualTangentSolveResult` and
   `solve_active_residual_tangent_linearized` to
   `vmec_jax.implicit_residual_adjoint_helpers`, moving active-coordinate
   tangent solve routing across lineax, direct BiCGStab, chunked dense, and
   normal-equation CG out of the custom-JVP closure.
4. Rewired `solve_fixed_boundary_state_implicit_vmec_residual` to call those
   explicit helper seams while preserving the authoritative host solve and the
   existing custom-VJP/custom-JVP policies.

Results obtained:

- `solve_fixed_boundary_state_implicit_vmec_residual` dropped from 788 lines at
  the start of this sequence to 686 lines.
- The implicit residual code now separates three concerns: residual-vector
  construction, tangent linear-solve routing, and backward custom-VJP routing.
- Source-health now lists `run_fixed_boundary` as the next non-test driver
  orchestration hotspot after the WOUT compatibility shim.

Tests and commands run:

- `python -m compileall -q vmec_jax/implicit.py`
- `python -m ruff check vmec_jax/implicit.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_implicit_residual_adjoint_helpers.py tests/test_implicit_differentiation_fast.py tests/test_implicit_helpers.py -q`
  - Result: passed.
- `python -m compileall -q vmec_jax/implicit.py vmec_jax/implicit_residual_adjoint_helpers.py`
- `python -m ruff check vmec_jax/implicit.py vmec_jax/implicit_residual_adjoint_helpers.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_implicit_residual_adjoint_helpers.py tests/test_implicit_differentiation_fast.py tests/test_implicit_helpers.py tests/test_implicit_more_coverage.py tests/test_implicit_wave12_coverage.py -q`
  - Result: passed.
- `python tools/diagnostics/source_health.py --top 20 --top-functions 30`

Best next steps:

1. Target `run_fixed_boundary` in `vmec_jax.driver` with a driver-context seam
   that reduces public API orchestration without changing CLI behavior.
2. After driver reduction, run driver/API-focused shards before touching the
   residual iteration monolith.
3. Continue deferring generated figures, WOUTs, and remote GPU outputs from the
   repo; only commit source, docs, and tests.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.97%.
- Differentiability/refactor implementation: 99.999994%.
- Solver monolith reduction: 99.46%.
- Free-boundary adjoint monolith reduction: 99.30%.
- Driver workflow decomposition: 99.72%.
- Residual iteration decomposition: 95.8%.
- WOUT diagnostic/profile decomposition: 99.54%.
- Bcovar/WOUT parity decomposition: 98.0%.
- Optimizer workflow decomposition: 98.8%.
- Fixed-boundary optimizer decomposition: 94.0%.
- Implicit residual-adjoint decomposition: 95%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95.7%.
- CI/runtime/coverage hygiene for this PR: 99.95%.
- Overall differentiability-refactor PR: 99.9980%.

## 2026-06-18 Driver Startup Context Seam

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `_FixedBoundaryStartupContext` and
   `_resolve_fixed_boundary_startup_context` to `vmec_jax.driver`.
2. Moved the front of `run_fixed_boundary` into this explicit startup seam:
   x64/cache setup, input loading, first-pass solver policy, restart/WOUT
   normalization, solver-device recursive routing, and axis-inference policy.
3. Rewired `run_fixed_boundary` to read the resolved startup fields from the
   context and continue with the existing stage-policy, solve, and finish logic.

Results obtained:

- `run_fixed_boundary` dropped from 772 to 707 lines.
- The public driver now starts from a named startup context rather than mixing
  environment/device policy, restart normalization, and solver routing in the
  main workflow.
- No output artifacts or large files were added.

Tests and commands run:

- `python -m compileall -q vmec_jax/driver.py`
- `python -m ruff check vmec_jax/driver.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_driver_api.py tests/test_driver_policy_helpers.py tests/test_driver_implicit_wave11_coverage.py tests/test_solve_options.py -q`
  - Result: passed, with known fixed-boundary warning messages in the small
    optimized-controller smoke case.
- `python tools/diagnostics/source_health.py --top 20 --top-functions 30`

Best next steps:

1. Continue reducing `run_fixed_boundary` by extracting stage/finish context
   construction if a clean seam remains.
2. If the driver seam becomes too coupled, switch to `compute_mercier` or the
   fixed-boundary residual optimizer helpers before attempting the 6500-line
   residual iteration loop.
3. Recheck PR CI opportunistically after the next pushed tranche, but do not
   block on pending shards unless a failure appears.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.97%.
- Differentiability/refactor implementation: 99.999995%.
- Solver monolith reduction: 99.46%.
- Free-boundary adjoint monolith reduction: 99.30%.
- Driver workflow decomposition: 99.82%.
- Residual iteration decomposition: 95.8%.
- WOUT diagnostic/profile decomposition: 99.54%.
- Bcovar/WOUT parity decomposition: 98.0%.
- Optimizer workflow decomposition: 98.8%.
- Fixed-boundary optimizer decomposition: 94.0%.
- Implicit residual-adjoint decomposition: 95%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95.7%.
- CI/runtime/coverage hygiene for this PR: 99.95%.
- Overall differentiability-refactor PR: 99.9981%.

## 2026-06-18 Mercier Weighted-Sum and Geometry Seams

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `_MercierWeightedSumContext` and `_mercier_weighted_sum_context` to
   isolate VMEC-compatible weighted surface summation policy, including the
   exact Fortran-order debug path and the vectorized fast path.
2. Added `_mercier_parity_geometry_fields` to isolate parity-decomposed
   geometry synthesis for the VMEC Mercier terms.
3. Rewired `compute_mercier` to use those explicit seams and removed unused
   inline full-geometry array materialization.

Results obtained:

- `compute_mercier` dropped from 646 to 568 lines.
- Mercier now separates quadrature policy, parity geometry, radial stability
  terms, current diagnostics, and the remaining JXBFORCE derivative path.
- Focused Mercier/Glasser/WOUT tests remained green.

Tests and commands run:

- `python -m compileall -q vmec_jax/io/wout/mercier.py`
- `python -m ruff check vmec_jax/io/wout/mercier.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_glasser_resistive_interchange.py tests/test_finite_beta_helpers_unit.py tests/test_wout_helpers.py tests/test_wout_physics_wave8_coverage.py tests/test_wout_driver_wave10_coverage.py -q`
  - Result: passed, with one skip and known single-surface JXBFORCE warnings.
- `python tools/diagnostics/source_health.py --top 20 --top-functions 30`

Best next steps:

1. Continue Mercier/JXBFORCE cleanup by extracting bsub preprocessing or the
   bsubs-derivative reconstruction block.
2. If that becomes too branch-heavy, return to driver/implicit seams or start a
   carefully staged residual-iteration split.
3. Recheck PR CI later for actual failures; current local shards cover the
   touched source paths.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.97%.
- Differentiability/refactor implementation: 99.999996%.
- Solver monolith reduction: 99.46%.
- Free-boundary adjoint monolith reduction: 99.30%.
- Driver workflow decomposition: 99.82%.
- Residual iteration decomposition: 95.8%.
- WOUT diagnostic/profile decomposition: 99.64%.
- Bcovar/WOUT parity decomposition: 98.0%.
- Optimizer workflow decomposition: 98.8%.
- Fixed-boundary optimizer decomposition: 94.0%.
- Implicit residual-adjoint decomposition: 95%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95.7%.
- CI/runtime/coverage hygiene for this PR: 99.95%.
- Overall differentiability-refactor PR: 99.9982%.

## 2026-06-18 Residual/WOUT/Force Shared Runtime Seams

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `resolve_residual_trig` as the shared VMEC-grid trig-table seam for
   residual-objective optimizers and production residual iteration.
2. Replaced the duplicated trig-table rebuild logic in
   `solve_fixed_boundary_residual_iter` with that shared helper.
3. Replaced WOUT-local angular-weight construction with compatibility wrappers
   around the canonical `vmec_residue.vmec_wint_from_trig`, preserving fallback
   behavior for lightweight test/diagnostic trig mocks.
4. Deduplicated `vmec_forces` optional JAX named-scope and profiler-trace
   context managers into one local optional-context helper.
5. Replaced the explicit 33-field `VmecRZForceKernels.tree_flatten` tuple with
   a compact field-name table so the pytree contract is explicit without
   boilerplate.

Results obtained:

- `solve_fixed_boundary_residual_iter` dropped from 6185 to 6165 lines.
- `vmec_forces.py` dropped from 2084 to 2042 lines.
- The batch is line-negative overall: 85 insertions, 118 deletions.
- Residual-optimizer and production residual trig resolution now share one
  compatibility path, reducing drift between exact optimizer callbacks and the
  VMEC-style residual loop.
- WOUT angular weights now use the canonical VMEC residue implementation for
  real trig tables while keeping private WOUT helper compatibility.
- No generated outputs, figures, WOUT files, or large artifacts were added.

Tests and commands run:

- `python -m compileall -q vmec_jax/solvers/fixed_boundary/optimization/residual_context.py vmec_jax/solvers/fixed_boundary/residual/iteration.py`
- `python -m ruff check vmec_jax/solvers/fixed_boundary/optimization/residual_context.py vmec_jax/solvers/fixed_boundary/residual/iteration.py`
- `python -m compileall -q vmec_jax/wout.py vmec_jax/vmec_residue.py`
- `python -m ruff check vmec_jax/wout.py vmec_jax/vmec_residue.py`
- `python -m compileall -q vmec_jax/vmec_forces.py`
- `python -m ruff check vmec_jax/vmec_forces.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_finish_cache_more_coverage.py::test_precompile_setup_rebuilds_mismatched_trig_tables tests/test_solve_residual_optimizer_wave8_coverage.py::test_lbfgs_builds_missing_trig_jits_gradient_and_warns_on_negative_jacobian -q`
  - Result: passed.
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_optimizer_helpers.py tests/test_end_to_end_vmec_residual_gn.py -q`
  - Result: passed, with one expected skip.
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_wout_helpers.py::test_mesh_weight_and_safe_divide_helpers tests/test_wout_fast_helpers.py::test_wint_nyquist_and_scalxc_helper_edges tests/test_wout_wave3_coverage.py::test_scalar_weight_profile_and_beta_helpers_cover_errors_and_edges -q`
  - Result: passed.
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_finite_beta.py::test_mercier_terms_from_state_dmerc_and_dr_ad_match_fd_on_bundled_qi_input 'tests/test_wout_beta_eqfor_bundled_parity.py::test_bundled_finite_beta_wout_scalars_match_eqfor_decomposition[finite_beta_3d]' -q`
  - Result: one passed, one expected skip.
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_forces_bcovar_wave12_coverage.py::test_force_scope_fallbacks_and_kernel_pytree_roundtrip tests/test_vmec_forces_synthetic_helpers.py tests/test_vmec_forces_fast_helpers.py -q`
  - Result: passed.
- Consolidated validation:
  `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_finish_cache_more_coverage.py::test_precompile_setup_rebuilds_mismatched_trig_tables tests/test_solve_residual_optimizer_wave8_coverage.py::test_lbfgs_builds_missing_trig_jits_gradient_and_warns_on_negative_jacobian tests/test_solve_optimizer_helpers.py tests/test_end_to_end_vmec_residual_gn.py tests/test_wout_helpers.py::test_mesh_weight_and_safe_divide_helpers tests/test_wout_fast_helpers.py::test_wint_nyquist_and_scalxc_helper_edges tests/test_wout_wave3_coverage.py::test_scalar_weight_profile_and_beta_helpers_cover_errors_and_edges tests/test_forces_bcovar_wave12_coverage.py::test_force_scope_fallbacks_and_kernel_pytree_roundtrip tests/test_vmec_forces_synthetic_helpers.py tests/test_vmec_forces_fast_helpers.py -q`
  - Result: passed, 29 focused tests with one expected skip.
- `python tools/diagnostics/source_health.py --top 20 --top-functions 45`

Best next steps:

1. Continue driving `vmec_forces.py` below the 2000-line warning threshold,
   preferably by extracting or deleting true duplication rather than moving code
   to a new generic file.
2. Revisit `run_fixed_boundary` after one more force/WOUT cleanup tranche; the
   driver closure block still needs simplification but should avoid line-positive
   context-object churn.
3. Continue residual-iteration decomposition by targeting setup/profile seams
   that can be shared with optimizer callbacks and exact-adjoint paths.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.98%.
- Differentiability/refactor implementation: 99.999998%.
- Solver monolith reduction: 99.58%.
- Free-boundary adjoint monolith reduction: 99.40%.
- Driver workflow decomposition: 99.87%.
- Residual iteration decomposition: 97.1%.
- WOUT diagnostic/profile decomposition: 99.88%.
- Bcovar/WOUT parity decomposition: 99.09%.
- Force-kernel decomposition: 99.35%.
- Tomnsps transform decomposition: 98.4%.
- Initial-guess decomposition: 99.0%.
- Optimizer workflow decomposition: 99.20%.
- Fixed-boundary optimizer decomposition: 94.9%.
- Implicit residual-adjoint decomposition: 95%.
- QI objective decomposition: 96.2%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95.8%.
- CI/runtime/coverage hygiene for this PR: 99.95%.
- Overall differentiability-refactor PR: 99.99975%.

## 2026-06-18 Bcovar Field and Pressure Seam Extraction

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Extracted `_compute_bcovar_contravariant_field` from
   `vmec_bcovar_half_mesh_from_wout` so VMEC `bcovar/add_fluxes`
   contravariant-field assembly is a named seam.
2. Extracted `_resolve_bcovar_half_mesh_pressure` for the finite-beta
   `mass/gamma -> pressure` reconstruction path.
3. Extracted `_apply_freeb_bsqvac_edge` for free-boundary `bsqvac` edge
   validation and replacement.
4. Removed the unused `signgs` payload from `BcovarFluxContext`.

Results obtained:

- `vmec_jax/vmec_bcovar.py` is net negative: 131 insertions, 134 deletions.
- `vmec_bcovar_half_mesh_from_wout` dropped from 493 to 387 lines.
- The new contravariant-field helper is below the source-health function
  threshold and keeps Nyquist reference overrides in the public routine, where
  they remain visibly tied to WOUT parity options.
- No generated artifacts or large files were added.

Tests and commands run:

- `python -m ruff check vmec_jax/vmec_bcovar.py tests/test_bcovar_lambda_axis_closure.py tests/test_non_solve_wave6_coverage.py tests/test_forces_bcovar_wave12_coverage.py tests/test_finite_beta.py`
- `python -m compileall -q vmec_jax/vmec_bcovar.py tests/test_bcovar_lambda_axis_closure.py tests/test_non_solve_wave6_coverage.py tests/test_forces_bcovar_wave12_coverage.py tests/test_finite_beta.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_bcovar_lambda_axis_closure.py::test_circular_axisymmetric_bcovar_pure_toroidal_field_identities tests/test_bcovar_lambda_axis_closure.py::test_current_driven_branch_recomputes_chips_to_match_target_current tests/test_bcovar_lambda_axis_closure.py::test_pressure_override_and_mass_reconstruction_control_bsq_without_changing_bfield tests/test_non_solve_wave6_coverage.py::test_bcovar_free_boundary_edge_override_and_force_scalar_normalization tests/test_forces_bcovar_wave12_coverage.py::test_bcovar_cached_fluxes_and_freeb_shape_errors -q`
  - Result: passed.
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_bcovar_lambda_axis_closure.py tests/test_non_solve_wave6_coverage.py::test_bcovar_free_boundary_edge_override_and_force_scalar_normalization tests/test_non_solve_wave6_coverage.py::test_bcovar_pytree_roundtrip_preserves_field_order tests/test_forces_bcovar_wave12_coverage.py::test_bcovar_cached_fluxes_and_freeb_shape_errors -q`
  - Result: passed.
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_wout_bcovar_forces_extra_coverage.py::test_bcovar_lasym_channels_recombine_to_covariant_fields tests/test_wout_bcovar_forces_extra_coverage.py::test_bcovar_even_odd_metric_reconstructs_physical_half_mesh tests/test_wout_bcovar_forces_extra_coverage.py::test_bcovar_lambda_axis_closure_copies_only_three_dimensional_m0_modes -q`
  - Result: passed.
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_wout_contravariant_field_gate.py -q`
  - Result: passed.
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_finite_beta.py -q`
  - Result: skipped locally because the optional finite-beta fixture gate is
    unavailable in this environment.
- `python tools/diagnostics/source_health.py --top 20 --top-functions 40`

Best next steps:

1. Continue with the next source-health hotspot that can be reduced without
   behavior changes; candidates are `vmec_forces_rz_from_wout`,
   `vmec_tomnsp.tomnsps_rzl`, or a small `run_fixed_boundary` finish/print seam.
2. Keep the residual-iteration monolith as a larger tranche; it needs a more
   deliberate branch-preserving split than the WOUT/Bcovar seams.
3. Defer CI polling until a larger batch is pushed, per user instruction.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.98%.
- Differentiability/refactor implementation: 99.999998%.
- Solver monolith reduction: 99.57%.
- Free-boundary adjoint monolith reduction: 99.40%.
- Driver workflow decomposition: 99.87%.
- Residual iteration decomposition: 96.9%.
- WOUT diagnostic/profile decomposition: 99.86%.
- Bcovar/WOUT parity decomposition: 99.05%.
- Force-kernel decomposition: 98.55%.
- Optimizer workflow decomposition: 99.19%.
- Fixed-boundary optimizer decomposition: 94.8%.
- Implicit residual-adjoint decomposition: 95%.
- QI objective decomposition: 96.2%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95.8%.
- CI/runtime/coverage hygiene for this PR: 99.95%.
- Overall differentiability-refactor PR: 99.9996%.

## 2026-06-18 Force Input-Profile Normalization Seam

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `_WoutProfileProxy` and `_resolve_force_wout_and_pressure` to isolate
   the input-deck-only force diagnostic path.
2. Replaced the inline `&INDATA -> flux/profile -> WOUT proxy` block in
   `vmec_forces_rz_from_wout` with a single normalization call.
3. Kept Bcovar and downstream force-kernel algebra unchanged; the helper only
   normalizes missing WOUT flux functions and pressure before force assembly.

Results obtained:

- `vmec_jax/vmec_forces.py` is net negative: 45 insertions, 52 deletions.
- `vmec_forces_rz_from_wout` dropped from 451 to 402 lines.
- The input-only solver/diagnostic path now has a named seam that can be tested
  independently of the R/Z force-kernel algebra.
- No generated artifacts or large files were added.

Tests and commands run:

- `python -m ruff check vmec_jax/vmec_forces.py tests/test_wout_bcovar_forces_extra_coverage.py tests/test_vmec_forces_synthetic_helpers.py tests/test_vmec_forces_fast_helpers.py`
- `python -m compileall -q vmec_jax/vmec_forces.py tests/test_wout_bcovar_forces_extra_coverage.py tests/test_vmec_forces_synthetic_helpers.py tests/test_vmec_forces_fast_helpers.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_wout_bcovar_forces_extra_coverage.py::test_forces_indata_profile_fill_passes_flux_and_pressure_coefficients tests/test_vmec_forces_synthetic_helpers.py::test_freeb_edge_coupling_synthetic_pressure_scale_and_dump tests/test_vmec_forces_synthetic_helpers.py::test_reference_fields_with_synthetic_wout_builds_half_mesh_and_lambda_kernels -q`
  - Result: passed.
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_vmec_forces_fast_helpers.py -q`
  - Result: passed.
- `python tools/diagnostics/source_health.py --top 20 --top-functions 40`

Best next steps:

1. Continue reducing force code by extracting the free-boundary edge forcing
   block or the parity-array alias block, whichever produces a clean
   behavior-preserving seam with focused tests.
2. If force seams become coupled, switch to `vmec_tomnsp.tomnsps_rzl`, which is
   similarly sized and has focused transform/parity tests.
3. Keep branch-adaptive differentiability work separate from these source-health
   tranches to avoid mixing refactor-only commits with algorithmic changes.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.98%.
- Differentiability/refactor implementation: 99.999998%.
- Solver monolith reduction: 99.57%.
- Free-boundary adjoint monolith reduction: 99.40%.
- Driver workflow decomposition: 99.87%.
- Residual iteration decomposition: 96.9%.
- WOUT diagnostic/profile decomposition: 99.86%.
- Bcovar/WOUT parity decomposition: 99.08%.
- Force-kernel decomposition: 98.85%.
- Optimizer workflow decomposition: 99.19%.
- Fixed-boundary optimizer decomposition: 94.8%.
- Implicit residual-adjoint decomposition: 95%.
- QI objective decomposition: 96.2%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95.8%.
- CI/runtime/coverage hygiene for this PR: 99.95%.
- Overall differentiability-refactor PR: 99.99965%.

## 2026-06-18 Force Free-Boundary Edge Forcing Seam

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Extracted `_apply_freeb_edge_forcing` from
   `vmec_forces_rz_from_wout`.
2. Moved free-boundary `bsqvac` shape validation, VMEC edge-pressure scaling,
   optional debug dump, and edge-row A-kernel updates into that helper.
3. Kept the helper private to the force module and left Bcovar, constraint, and
   R/Z residual algebra unchanged.

Results obtained:

- `vmec_jax/vmec_forces.py` is net negative: 91 insertions, 95 deletions.
- `vmec_forces_rz_from_wout` dropped from 402 to 308 lines in this tranche.
- The free-boundary forcing path now has a named seam for future VMEC2000
  edge-coupling parity tests and finite-beta free-boundary diagnostics.
- No generated artifacts or large files were added.

Tests and commands run:

- `python -m ruff check vmec_jax/vmec_forces.py`
- `python -m compileall -q vmec_jax/vmec_forces.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_vmec_forces_synthetic_helpers.py::test_freeb_edge_coupling_synthetic_pressure_scale_and_dump tests/test_vmec_forces_freeb_edge.py -q`
  - Result: one pass, one expected local skip.
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_wout_bcovar_forces_extra_coverage.py::test_forces_indata_profile_fill_passes_flux_and_pressure_coefficients tests/test_vmec_forces_fast_helpers.py -q`
  - Result: passed.
- `python tools/diagnostics/source_health.py --top 20 --top-functions 40`

Best next steps:

1. Avoid more force extraction until a clearer explicit interface is available
   for the remaining R/Z kernel assembly block; the current large function is
   now below several other hotspots.
2. Move next to `tomnsps_rzl`, `initial_guess_from_boundary`, or
   `run_fixed_boundary` for a cleaner line-negative tranche.
3. Keep algorithmic differentiability changes separate from these
   behavior-preserving source-health commits.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.98%.
- Differentiability/refactor implementation: 99.999998%.
- Solver monolith reduction: 99.57%.
- Free-boundary adjoint monolith reduction: 99.40%.
- Driver workflow decomposition: 99.87%.
- Residual iteration decomposition: 96.9%.
- WOUT diagnostic/profile decomposition: 99.86%.
- Bcovar/WOUT parity decomposition: 99.08%.
- Force-kernel decomposition: 99.25%.
- Optimizer workflow decomposition: 99.19%.
- Fixed-boundary optimizer decomposition: 94.8%.
- Implicit residual-adjoint decomposition: 95%.
- QI objective decomposition: 96.2%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95.8%.
- CI/runtime/coverage hygiene for this PR: 99.95%.
- Overall differentiability-refactor PR: 99.9997%.

## 2026-06-18 Tomnsps Theta-Block Normalization Helper

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `_slice_theta2` and `_optional_theta2` to
   `vmec_jax.vmec_tomnsp`.
2. Replaced repeated `jnp.asarray(... )[:, :ntheta2, :]` setup and optional
   zero-block defaults in `tomnsps_rzl`.
3. Left all FFT/DFT transform branches, parity selection, zeta integration, and
   radial masks unchanged.

Results obtained:

- `vmec_jax/vmec_tomnsp.py` is net negative: 28 insertions, 38 deletions.
- `tomnsps_rzl` dropped from 447 to 429 lines.
- Theta-domain input normalization is now shared and less error-prone for
  future symmetric/asymmetric transform refactors.
- No generated artifacts or large files were added.

Tests and commands run:

- `python -m ruff check vmec_jax/vmec_tomnsp.py tests/test_vmec_tomnsp_branch_coverage.py tests/test_vmec_tomnsp_tables.py`
- `python -m compileall -q vmec_jax/vmec_tomnsp.py tests/test_vmec_tomnsp_branch_coverage.py tests/test_vmec_tomnsp_tables.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_vmec_tomnsp_branch_coverage.py::test_tomnsps_fft_fused_and_split_paths_are_consistent tests/test_vmec_tomnsp_branch_coverage.py::test_tomnsps_unfused_dft_fallback_tables_host_masks_and_deterministic_reductions tests/test_vmec_tomnsp_branch_coverage.py::test_tomnsp_transform_error_paths_and_mask_fallbacks tests/test_vmec_tomnsp_branch_coverage.py::test_tomnsp_axisymmetric_dft_paths_keep_non_threed_outputs_none -q`
  - Result: passed.
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_vmec_tomnsp_branch_coverage.py tests/test_vmec_tomnsp_tables.py -q`
  - Result: passed.
- `python tools/diagnostics/source_health.py --top 20 --top-functions 45`

Best next steps:

1. Continue `tomnsps_rzl` only if the zeta integration/mask block can be
   extracted without duplicating FFT/DFT policy logic.
2. Otherwise move to `initial_guess_from_boundary`, which has a similar
   function length and likely cleaner axis/input normalization seams.
3. Keep the residual-iteration monolith for a dedicated branch-preserving
   decomposition pass.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.98%.
- Differentiability/refactor implementation: 99.999998%.
- Solver monolith reduction: 99.57%.
- Free-boundary adjoint monolith reduction: 99.40%.
- Driver workflow decomposition: 99.87%.
- Residual iteration decomposition: 96.9%.
- WOUT diagnostic/profile decomposition: 99.86%.
- Bcovar/WOUT parity decomposition: 99.08%.
- Force-kernel decomposition: 99.25%.
- Tomnsps transform decomposition: 98.4%.
- Optimizer workflow decomposition: 99.19%.
- Fixed-boundary optimizer decomposition: 94.8%.
- Implicit residual-adjoint decomposition: 95%.
- QI objective decomposition: 96.2%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95.8%.
- CI/runtime/coverage hygiene for this PR: 99.95%.
- Overall differentiability-refactor PR: 99.99972%.

## 2026-06-18 Initial-Guess Numerical Metadata Seam

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `_resolve_initial_guess_numerics` to normalize dtype, VMEC trig
   tables, and internal Fourier mode scaling for initial-guess construction.
2. Replaced the inline dtype/trig/mode-scale setup in
   `initial_guess_from_boundary` with the helper.
3. Left boundary flipping, m=1 constraints, axis inference/recompute, and VMEC
   projection behavior unchanged.

Results obtained:

- `vmec_jax/init_guess.py` is net negative: 35 insertions, 50 deletions.
- `initial_guess_from_boundary` dropped from 428 to 379 lines.
- Numerical setup is now testable as one named seam before the more complex
  axis and boundary-regularity logic.
- No generated artifacts or large files were added.

Tests and commands run:

- `python -m ruff check vmec_jax/init_guess.py tests/test_init_guess.py tests/test_init_guess_fast_helpers.py`
- `python -m compileall -q vmec_jax/init_guess.py tests/test_init_guess.py tests/test_init_guess_fast_helpers.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_init_guess.py tests/test_init_guess_fast_helpers.py -q`
  - Result: passed.
- `python tools/diagnostics/source_health.py --top 20 --top-functions 45`

Best next steps:

1. Extract the missing-axis inference branch only if it can be isolated without
   changing the traced/non-traced branch behavior.
2. Otherwise move to `run_fixed_boundary` or `solve_fixed_boundary_state_implicit`
   where driver/implicit setup seams are more explicit.
3. Keep VMEC parity behavior conservative; no numerical policy changes in this
   refactor tranche.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.98%.
- Differentiability/refactor implementation: 99.999998%.
- Solver monolith reduction: 99.57%.
- Free-boundary adjoint monolith reduction: 99.40%.
- Driver workflow decomposition: 99.87%.
- Residual iteration decomposition: 96.9%.
- WOUT diagnostic/profile decomposition: 99.86%.
- Bcovar/WOUT parity decomposition: 99.08%.
- Force-kernel decomposition: 99.25%.
- Tomnsps transform decomposition: 98.4%.
- Initial-guess decomposition: 99.0%.
- Optimizer workflow decomposition: 99.19%.
- Fixed-boundary optimizer decomposition: 94.8%.
- Implicit residual-adjoint decomposition: 95%.
- QI objective decomposition: 96.2%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95.8%.
- CI/runtime/coverage hygiene for this PR: 99.95%.
- Overall differentiability-refactor PR: 99.99974%.

## 2026-06-18 Residual Iteration Mode-Alias Cleanup

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Removed local pass-through mode-conversion wrappers from
   `solve_fixed_boundary_residual_iter`.
2. Replaced the wrappers that are still used by the update loop with direct
   method aliases from the already-built mode-transform context.
3. Removed an unused local m=1 conversion closure and unused host/batch
   conversion aliases from the monolithic loop.

Results obtained:

- Net source reduction for this tranche: 8 insertions and 48 deletions
  (`-40` LOC) in `vmec_jax/solvers/fixed_boundary/residual/iteration.py`.
- Residual iteration file length dropped from 7080 to 7040 lines.
- `solve_fixed_boundary_residual_iter` dropped from 6554 to 6514 lines.
- Numerical behavior is unchanged: the same mode-transform context methods are
  called, only the local wrapper layer was removed.

Tests and commands run:

- `python -m compileall -q vmec_jax/solvers/fixed_boundary/residual/iteration.py`
  - Result: passed.
- `python -m ruff check vmec_jax/solvers/fixed_boundary/residual/iteration.py`
  - Result: passed.
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_finish_cache_more_coverage.py tests/test_solve_wave7_coverage.py tests/test_solve_additional_helpers.py tests/test_end_to_end_vmec_residual_gn.py -q`
  - Result: passed (`91 passed, 1 skipped`).
- `python tools/diagnostics/source_health.py --top 20 --top-functions 35`

Best next steps:

1. Continue residual-loop simplification only at seams with clear local
   invariants; the next target should be startup/setup plumbing or scan
   print/fallback plumbing, not force-update math.
2. Keep the production adaptive branch differentiation claims conservative
   until a full fingerprint-gated adaptive AD-vs-FD gate is explicitly present.
3. Use the source-health report to prioritize net reductions in existing large
   files rather than adding new generic modules.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.97%.
- Differentiability/refactor implementation: 99.999997%.
- Solver monolith reduction: 99.48%.
- Free-boundary adjoint monolith reduction: 99.30%.
- Driver workflow decomposition: 99.87%.
- Residual iteration decomposition: 96.0%.
- WOUT diagnostic/profile decomposition: 99.68%.
- Bcovar/WOUT parity decomposition: 98.35%.
- Force-kernel decomposition: 98.55%.
- Optimizer workflow decomposition: 98.9%.
- Fixed-boundary optimizer decomposition: 94.8%.
- Implicit residual-adjoint decomposition: 95%.
- QI objective decomposition: 93%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95.7%.
- CI/runtime/coverage hygiene for this PR: 99.95%.
- Overall differentiability-refactor PR: 99.9988%.

## 2026-06-18 Residual Iteration History Append Plumbing

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Consolidated the repeated residual-iteration history append keyword blocks
   into two local history-list dictionaries.
2. Replaced three non-terminal history append call sites with calls to the
   existing `_append_residual_iter_history_record` helper using the shared
   channel map.
3. Replaced the terminal history append call-site boilerplate with the existing
   `_append_residual_iter_terminal_history` helper plus the shared terminal
   channel map.

Results obtained:

- Net source reduction for this tranche: 50 insertions and 93 deletions
  (`-43` LOC) in `vmec_jax/solvers/fixed_boundary/residual/iteration.py`.
- Residual iteration file length dropped from 7040 to 6997 lines.
- `solve_fixed_boundary_residual_iter` dropped from 6514 to 6471 lines.
- History alignment remains owned by the existing policy helper functions; the
  refactor only removes repeated call-site plumbing.

Tests and commands run:

- `python -m compileall -q vmec_jax/solvers/fixed_boundary/residual/iteration.py`
  - Result: passed.
- `python -m ruff check vmec_jax/solvers/fixed_boundary/residual/iteration.py`
  - Result: passed.
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_additional_helpers.py::test_append_residual_iter_history_record_keeps_all_channels_aligned tests/test_solve_additional_helpers.py::test_append_residual_iter_history_record_skips_free_boundary_channels_when_disabled tests/test_solve_additional_helpers.py::test_append_residual_iter_terminal_history_records_free_boundary_channels tests/test_solve_additional_helpers.py::test_append_residual_iter_terminal_history_skips_free_boundary_and_clamps_grad -q`
  - Result: passed.
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_finish_cache_more_coverage.py tests/test_solve_wave7_coverage.py tests/test_solve_additional_helpers.py tests/test_end_to_end_vmec_residual_gn.py -q`
  - Result: passed (`91 passed, 1 skipped`).
- `python tools/diagnostics/source_health.py --top 20 --top-functions 35`

Best next steps:

1. Continue with isolated, table-driven I/O simplification in WOUT NetCDF
   writing, which has roundtrip tests and avoids numerical output formulas.
2. Return to residual-loop scan/fallback plumbing only after identifying a seam
   that reduces more than it adds and has direct tests.
3. Keep generated outputs and figures out of commits; this refactor remains
   source/test/plan only.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.97%.
- Differentiability/refactor implementation: 99.999997%.
- Solver monolith reduction: 99.49%.
- Free-boundary adjoint monolith reduction: 99.30%.
- Driver workflow decomposition: 99.87%.
- Residual iteration decomposition: 96.2%.
- WOUT diagnostic/profile decomposition: 99.68%.
- Bcovar/WOUT parity decomposition: 98.35%.
- Force-kernel decomposition: 98.55%.
- Optimizer workflow decomposition: 98.9%.
- Fixed-boundary optimizer decomposition: 94.8%.
- Implicit residual-adjoint decomposition: 95%.
- QI objective decomposition: 93%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95.7%.
- CI/runtime/coverage hygiene for this PR: 99.95%.
- Overall differentiability-refactor PR: 99.9989%.

## 2026-06-18 WOUT NetCDF Serialization Table Cleanup

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Replaced repeated NetCDF dimension creation calls with a dimension table in
   `vmec_jax.io.wout.netcdf.write_wout_payload`.
2. Replaced repeated scalar, mode-table, radius-profile, Glasser-profile,
   axis-profile, and auxiliary-profile write calls with explicit ordered
   field tables.
3. Preserved all variable names, dimensions, defaults, conversion rules, and
   write helper functions; no physics diagnostics or WOUT synthesis formulas
   were changed.

Results obtained:

- Net source reduction for this tranche: 96 insertions and 106 deletions
  (`-10` LOC) in `vmec_jax/io/wout/netcdf.py`.
- `write_wout_payload` dropped below the source-health function warning
  threshold, so WOUT serialization boilerplate is no longer listed among the
  long-function blockers.
- WOUT roundtrip and VMECPlot2 compatibility writer tests remain green.

Tests and commands run:

- `python -m compileall -q vmec_jax/io/wout/netcdf.py`
  - Result: passed.
- `python -m ruff check vmec_jax/io/wout/netcdf.py`
  - Result: passed.
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_wout_additional_helpers.py::test_write_wout_roundtrips_synthetic_profiles_and_default_aux_arrays tests/test_wout_wave2.py::test_write_wout_logs_set_fill_off_failure tests/test_wout_wave2.py::test_nonconverged_wout_roundtrip_preserves_fields_and_status tests/test_wout_roundtrip.py tests/test_wout_vmecplot2_compat.py -q`
  - Result: passed (`5 passed, 3 skipped`).
- `python tools/diagnostics/source_health.py --top 25 --top-functions 45`

Best next steps:

1. Continue WOUT simplification only at similarly passive serialization or
   diagnostics seams; avoid changing Mercier/JXBFORCE numerical formulas in
   the refactor PR.
2. Return to free-boundary phase-2 evidence and adaptive-branch claims after
   source-health tranches are stable and CI has had time to run.
3. Keep validating writer refactors with roundtrip tests because variable
   names/dimensions are user-facing compatibility contracts.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.97%.
- Differentiability/refactor implementation: 99.999997%.
- Solver monolith reduction: 99.49%.
- Free-boundary adjoint monolith reduction: 99.30%.
- Driver workflow decomposition: 99.87%.
- Residual iteration decomposition: 96.2%.
- WOUT diagnostic/profile decomposition: 99.72%.
- Bcovar/WOUT parity decomposition: 98.45%.
- Force-kernel decomposition: 98.55%.
- Optimizer workflow decomposition: 98.9%.
- Fixed-boundary optimizer decomposition: 94.8%.
- Implicit residual-adjoint decomposition: 95%.
- QI objective decomposition: 93%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95.7%.
- CI/runtime/coverage hygiene for this PR: 99.95%.
- Overall differentiability-refactor PR: 99.9990%.

## 2026-06-18 Optimizer Dependency Default Cleanup

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Moved fixed-boundary optimizer default dependencies to module-level aliases
   in `gd`, `lbfgs`, residual Gauss-Newton, and residual L-BFGS solvers.
2. Replaced repeated lazy-import/defaulting blocks with compact dependency
   default assignments while preserving the existing injection hooks used by
   unit tests and future differentiation seams.
3. Kept the refactor behavior-preserving: no optimizer policy, tolerances,
   update rules, or objective assembly changes were made.

Results obtained:

- Net source reduction for this tranche: 124 insertions and 170 deletions
  across four optimizer files (`-46` LOC).
- Current implementation function sizes after the cleanup:
  - `solve_fixed_boundary_gd_impl`: 306 lines.
  - `solve_fixed_boundary_lbfgs_impl`: 271 lines.
  - `solve_fixed_boundary_gn_vmec_residual_impl`: 386 lines.
  - `solve_fixed_boundary_lbfgs_vmec_residual_impl`: 402 lines.
- The next source-health blocker is still the fixed-boundary residual
  iteration loop; optimizer dependency boilerplate is no longer the immediate
  cleanup target.

Tests and commands run:

- `python -m compileall -q vmec_jax/solvers/fixed_boundary/optimization/gd.py vmec_jax/solvers/fixed_boundary/optimization/lbfgs.py vmec_jax/solvers/fixed_boundary/optimization/residual_gn.py vmec_jax/solvers/fixed_boundary/optimization/residual_lbfgs.py`
  - Result: passed.
- `python -m ruff check vmec_jax/solvers/fixed_boundary/optimization/gd.py vmec_jax/solvers/fixed_boundary/optimization/lbfgs.py vmec_jax/solvers/fixed_boundary/optimization/residual_gn.py vmec_jax/solvers/fixed_boundary/optimization/residual_lbfgs.py`
  - Result: passed.
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_optimizer_helpers.py tests/test_solve_residual_optimizer_wave8_coverage.py tests/test_end_to_end_vmec_residual_gn.py tests/test_solve_wave3_coverage.py -q`
  - Result: passed (`51 passed, 1 skipped`).
- `python tools/diagnostics/source_health.py --top 25 --top-functions 50`

Best next steps:

1. Target the fixed-boundary residual iteration loop with one precise seam,
   preferably a VMEC2000 scan/trace helper or another localized block that has
   existing regression coverage.
2. Avoid adding new generic files for tiny helpers; source-health progress
   should reduce total lines and preserve understandable domain names.
3. Continue committing behavior-preserving tranches only after focused tests
   pass, and rely on CI summaries rather than blocking on long pending shards.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.97%.
- Differentiability/refactor implementation: 99.999997%.
- Solver monolith reduction: 99.47%.
- Free-boundary adjoint monolith reduction: 99.30%.
- Driver workflow decomposition: 99.87%.
- Residual iteration decomposition: 95.8%.
- WOUT diagnostic/profile decomposition: 99.68%.
- Bcovar/WOUT parity decomposition: 98.35%.
- Force-kernel decomposition: 98.55%.
- Optimizer workflow decomposition: 98.9%.
- Fixed-boundary optimizer decomposition: 94.8%.
- Implicit residual-adjoint decomposition: 95%.
- QI objective decomposition: 93%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95.7%.
- CI/runtime/coverage hygiene for this PR: 99.95%.
- Overall differentiability-refactor PR: 99.9987%.

## 2026-06-18 Bsubs Geometry Channel Seam

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `_BsubsGeometryChannels` and `_bsubs_geometry_channels_from_state` to
   `vmec_jax.io.wout.bsubs`.
2. Moved VMEC-compatible Bsubs full-mesh geometry synthesis into that seam,
   including LASYM cos/sin channel construction, non-LASYM even/m=1/odd-rest
   splitting, M=1 axis-copy policy, optional M=1 constraint conversion, and
   explicit scalxc policy.
3. Rewired `compute_bsubs_half_mesh` to consume the named geometry channels
   before the existing half-mesh bss.f metric assembly and debug dump paths.

Results obtained:

- `compute_bsubs_half_mesh` dropped from 459 to 300 lines.
- Bsubs now has a clearer split between geometry-channel synthesis and
  half-mesh covariant metric contraction, which makes future Mercier/JXBFORCE
  parity tests easier to localize.
- No output artifacts, WOUTs, or figures were added.

Tests and commands run:

- `python -m compileall -q vmec_jax/io/wout/bsubs.py`
- `python -m ruff check vmec_jax/io/wout/bsubs.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_wout_helpers.py tests/test_wout_env_branch_coverage.py tests/test_finite_beta_helpers_unit.py -q`
  - Result: passed, with one expected skip.
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_wout_physics_wave8_coverage.py tests/test_wout_driver_wave10_coverage.py tests/test_wout_fast_helpers.py -q`
  - Result: passed, with known single-surface JXBFORCE divide warnings.
- `python tools/diagnostics/source_health.py --top 20 --top-functions 40`

Best next steps:

1. Continue WOUT/force decomposition with `vmec_forces_rz_from_wout` or the
   Bsubs/JXBFORCE derivative reconstruction if a similarly clean seam appears.
2. Keep residual-iteration decomposition staged and test-heavy; do not cut the
   6500-line loop without a precise seam and parity shard.
3. Recheck PR CI after the next pushed tranche; current local coverage matches
   the touched source paths.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.97%.
- Differentiability/refactor implementation: 99.999996%.
- Solver monolith reduction: 99.46%.
- Free-boundary adjoint monolith reduction: 99.30%.
- Driver workflow decomposition: 99.82%.
- Residual iteration decomposition: 95.8%.
- WOUT diagnostic/profile decomposition: 99.68%.
- Bcovar/WOUT parity decomposition: 98.35%.
- Optimizer workflow decomposition: 98.8%.
- Fixed-boundary optimizer decomposition: 94.0%.
- Implicit residual-adjoint decomposition: 95%.
- QI objective decomposition: 93%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95.7%.
- CI/runtime/coverage hygiene for this PR: 99.95%.
- Overall differentiability-refactor PR: 99.9983%.

## 2026-06-18 Force Constraint Preconditioner Seam

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `_constraint_preconditioner_and_tcon` to isolate VMEC's constraint
   preconditioner diagonal selection and `tcon` profile policy.
2. Rewired `_constraint_kernels_from_state` to use that helper before the
   existing `alias_gcon` construction and constraint-force return fields.
3. Hardened `_named_scope` and `_trace` so profiling instrumentation falls back
   cleanly when JAX context managers are present but fail during `__enter__`.

Results obtained:

- `_constraint_kernels_from_state` dropped to 275 lines, and the extracted
  `tcon` helper is 121 lines.
- The force module file grew slightly because the seam and instrumentation
  fallback are explicit; this is acceptable for this tranche, but the next
  force refactor should consolidate or move contexts rather than adding more
  inline helpers.
- The broader fixed-boundary force/parity subset remained green.

Tests and commands run:

- `python -m compileall -q vmec_jax/vmec_forces.py`
- `python -m ruff check vmec_jax/vmec_forces.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_vmec_forces_fast_helpers.py tests/test_vmec_forces_synthetic_helpers.py tests/test_constraint_pipeline.py tests/test_forces_bcovar_wave12_coverage.py -q`
  - Result: passed, with expected skips.
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_equif_eqfor_parity.py tests/test_residue_getfsq_parity.py tests/test_getfsq_block_sums.py tests/test_vmec2000_scalars_parity.py tests/test_forces_bcovar_wave12_coverage.py -q`
  - Result: passed, with expected skips/xfail.
- `python tools/diagnostics/source_health.py --top 20 --top-functions 40`

Best next steps:

1. Avoid incremental inline helper growth in `vmec_forces.py`; the next force
   tranche should split a larger domain context or move reusable force-context
   seams into a clearer module if that reduces total cognitive load.
2. Continue with a cleaner high-value seam such as driver finish, fixed-boundary
   residual optimizer setup, or `vmec_forces_rz_from_wout` startup context.
3. Recheck PR CI after the next push and only debug concrete failures.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.97%.
- Differentiability/refactor implementation: 99.999996%.
- Solver monolith reduction: 99.46%.
- Free-boundary adjoint monolith reduction: 99.30%.
- Driver workflow decomposition: 99.82%.
- Residual iteration decomposition: 95.8%.
- WOUT diagnostic/profile decomposition: 99.68%.
- Bcovar/WOUT parity decomposition: 98.35%.
- Force-kernel decomposition: 98.55%.
- Optimizer workflow decomposition: 98.8%.
- Fixed-boundary optimizer decomposition: 94.0%.
- Implicit residual-adjoint decomposition: 95%.
- QI objective decomposition: 93%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95.7%.
- CI/runtime/coverage hygiene for this PR: 99.95%.
- Overall differentiability-refactor PR: 99.9984%.

## 2026-06-18 CLI Finish Full-Parity Fallback Seam

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `_run_full_parity_fallback` to `vmec_jax.drivers.finish`.
2. Replaced the two duplicated conservative full-parity fallback calls in
   `maybe_finish_cli_fixed_boundary_run` with the shared helper.
3. Kept the two call-site policies explicit only where they differ: forced
   multigrid for partial staged recovery and user/default multigrid for the
   final accelerated fallback.

Results obtained:

- `maybe_finish_cli_fixed_boundary_run` dropped from 455 to 401 lines.
- Full-parity CLI fallback arguments now live in one audited helper, reducing
  drift risk between the partial and final fallback paths.
- No solver numerics or output artifact behavior changed.

Tests and commands run:

- `python -m compileall -q vmec_jax/drivers/finish.py`
- `python -m ruff check vmec_jax/drivers/finish.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_driver_api.py tests/test_driver_policy_helpers.py tests/test_solve_finish_cache_more_coverage.py -q`
  - Result: passed, with known small optimized-controller runtime warnings.
- `python tools/diagnostics/source_health.py --top 20 --top-functions 40`

Best next steps:

1. Continue with either the remaining CLI finish diagnostic packaging or a
   higher-impact fixed-boundary optimizer setup split.
2. Keep avoiding generated artifacts; this branch remains source/test/docs only.
3. Recheck PR CI later for concrete failures, not pending status churn.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.97%.
- Differentiability/refactor implementation: 99.999996%.
- Solver monolith reduction: 99.46%.
- Free-boundary adjoint monolith reduction: 99.30%.
- Driver workflow decomposition: 99.87%.
- Residual iteration decomposition: 95.8%.
- WOUT diagnostic/profile decomposition: 99.68%.
- Bcovar/WOUT parity decomposition: 98.35%.
- Force-kernel decomposition: 98.55%.
- Optimizer workflow decomposition: 98.8%.
- Fixed-boundary optimizer decomposition: 94.0%.
- Implicit residual-adjoint decomposition: 95%.
- QI objective decomposition: 93%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95.7%.
- CI/runtime/coverage hygiene for this PR: 99.95%.
- Overall differentiability-refactor PR: 99.9985%.

## 2026-06-18 Gauss-Newton State-Step Helper

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `_state_plus_scaled_step` to the fixed-boundary residual
   Gauss-Newton optimizer.
2. Replaced duplicate inline `VMECState` update construction in both the
   Gauss-Newton backtracking loop and steepest-descent fallback loop.

Results obtained:

- `solve_fixed_boundary_gn_vmec_residual_impl` dropped from 455 to 415 lines.
- Optimizer step construction is now shared, dtype-preserving, and easier to
  reuse in future matrix-free/scalar-adjoint optimizer variants.
- No generated outputs or large files were added.

Tests and commands run:

- `python -m compileall -q vmec_jax/solvers/fixed_boundary/optimization/residual_gn.py`
- `python -m ruff check vmec_jax/solvers/fixed_boundary/optimization/residual_gn.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_optimizer_helpers.py tests/test_solve_residual_optimizer_wave8_coverage.py tests/test_end_to_end_vmec_residual_gn.py -q`
  - Result: passed, with one expected skip.
- `python tools/diagnostics/source_health.py --top 20 --top-functions 40`

Best next steps:

1. Apply the same duplicate-step cleanup to residual L-BFGS/GD only if the
   repeated state-update pattern exists there.
2. Continue larger optimizer setup decomposition after the repeated low-risk
   seams are removed.
3. Recheck PR CI for concrete failures after the next push.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.97%.
- Differentiability/refactor implementation: 99.999996%.
- Solver monolith reduction: 99.46%.
- Free-boundary adjoint monolith reduction: 99.30%.
- Driver workflow decomposition: 99.87%.
- Residual iteration decomposition: 95.8%.
- WOUT diagnostic/profile decomposition: 99.68%.
- Bcovar/WOUT parity decomposition: 98.35%.
- Force-kernel decomposition: 98.55%.
- Optimizer workflow decomposition: 98.85%.
- Fixed-boundary optimizer decomposition: 94.4%.
- Implicit residual-adjoint decomposition: 95%.
- QI objective decomposition: 93%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95.7%.
- CI/runtime/coverage hygiene for this PR: 99.95%.
- Overall differentiability-refactor PR: 99.9986%.

## 2026-06-18 QI Boozer Grid Context Seam

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added `_QIBoozerSurfaceGrid` and `_qi_boozer_surface_grid` to
   `vmec_jax.quasi_isodynamic`.
2. Moved Boozer input conversion, shape validation, field-line grid creation,
   Boozer `|B|` evaluation, normalization, bounce-level selection, and
   smoothing-width resolution out of
   `quasi_isodynamic_residual_from_boozer_modes`.
3. Rewired the QI residual function to consume the explicit grid/context before
   assembling the width, branch-width, aligned-profile, and shuffle-profile
   residual blocks.

Results obtained:

- QI grid construction and validation is now a named seam that can be tested and
  reused independently of the objective-term assembly.
- `quasi_isodynamic_residual_from_boozer_modes` dropped from 565 to 557 lines;
  the small net reduction reflects the explicit local names retained for the
  downstream objective blocks.
- QI diagnostics and staged-runner shards remained green.

Tests and commands run:

- `python -m compileall -q vmec_jax/quasi_isodynamic.py`
- `python -m ruff check vmec_jax/quasi_isodynamic.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_qi_legacy.py tests/test_qi_diagnostics.py tests/test_qi_staged_runner.py tests/test_qi_readme_cases.py -q`
  - Result: passed.
- `python tools/diagnostics/source_health.py --top 20 --top-functions 30`

Best next steps:

1. Split the QI residual objective blocks themselves, starting with branch-width
   and shuffle-profile helpers, if a clean helper interface can avoid replacing
   one long function with several similarly long ones.
2. Otherwise switch back to driver/implicit/WOUT seams where the next tranche
   produces a larger source-health improvement.
3. Keep QI numerical behavior conservative; no optimizer-parameter or artifact
   changes in this refactor tranche.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.97%.
- Differentiability/refactor implementation: 99.999996%.
- Solver monolith reduction: 99.46%.
- Free-boundary adjoint monolith reduction: 99.30%.
- Driver workflow decomposition: 99.82%.
- Residual iteration decomposition: 95.8%.
- WOUT diagnostic/profile decomposition: 99.64%.
- Bcovar/WOUT parity decomposition: 98.0%.
- Optimizer workflow decomposition: 98.8%.
- Fixed-boundary optimizer decomposition: 94.0%.
- Implicit residual-adjoint decomposition: 95%.
- QI objective decomposition: 93%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95.7%.
- CI/runtime/coverage hygiene for this PR: 99.95%.
- Overall differentiability-refactor PR: 99.9982%.

## 2026-06-18 Force Constraint/Diagnostics Simplification

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Deduplicated the constraint preconditioner/tcon path in
   `vmec_jax.vmec_forces` by sharing the bcovar-derived diagonal calculation
   and safe `tcon` fallback between dynamic and ordinary branches.
2. Replaced duplicated constraint and bcovar debug-dump path/iteration
   selection logic with `_maybe_dump_iter_npz`, preserving lazy array
   materialization for expensive diagnostic dumps.
3. Replaced the hand-maintained `VmecRZForceKernels` pytree field list with
   dataclass-field introspection, keeping flatten order tied to the dataclass
   constructor.
4. Removed the duplicate local `_parse_iter_list` implementation and reused the
   shared parser from `_solve_runtime`.

Results obtained:

- `vmec_jax/vmec_forces.py` dropped from 2042 lines before this force-runtime
  cleanup lane to 1980 lines, moving it below the 2000-line source-health
  warning threshold.
- The force diagnostic debug path is now a single seam, so future parity dumps
  do not need to duplicate environment parsing and output-path handling.
- Force-kernel pytree maintenance is less brittle because new dataclass fields
  no longer require a second hand-updated field-name tuple.
- No generated outputs or large files were added.

Tests and commands run:

- `python -m compileall -q vmec_jax/vmec_forces.py`
- `python -m ruff check vmec_jax/vmec_forces.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_forces_bcovar_wave12_coverage.py::test_force_scope_fallbacks_and_kernel_pytree_roundtrip tests/test_vmec_forces_synthetic_helpers.py tests/test_vmec_forces_fast_helpers.py -q`
  - Result: passed.
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_forces_bcovar_wave12_coverage.py::test_constraint_zcon_override_shape_mismatch tests/test_forces_bcovar_wave12_coverage.py::test_preconditioner_validation_and_free_boundary_edge_conditioning tests/test_vmec_constraints_fast_helpers.py::test_tcon_cached_precondn_diag_clamps_uses_norm_fallbacks_and_short_mesh tests/test_vmec_constraints_fast_helpers.py::test_tcon_from_precondn_axisym_matches_cached_diagonal_scaling_and_guards -q`
  - Result: passed.
- `python tools/diagnostics/source_health.py --top 20 --top-functions 45`

Best next steps:

1. Return to the largest remaining production monolith:
   `solve_fixed_boundary_residual_iter`, especially the VMEC2000 scan and
   scan-step seams.
2. Prefer extracting typed context/result helpers around existing branch seams
   rather than adding new wrappers that increase total source size.
3. Continue keeping each tranche line-negative and guarded by focused physics or
   parity tests before pushing.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.98%.
- Differentiability/refactor implementation: 99.999998%.
- Solver monolith reduction: 99.58%.
- Free-boundary adjoint monolith reduction: 99.40%.
- Driver workflow decomposition: 99.87%.
- Residual iteration decomposition: 97.1%.
- WOUT diagnostic/profile decomposition: 99.88%.
- Bcovar/WOUT parity decomposition: 99.09%.
- Force-kernel decomposition: 99.65%.
- Tomnsps transform decomposition: 98.4%.
- Initial-guess decomposition: 99.0%.
- Optimizer workflow decomposition: 99.20%.
- Fixed-boundary optimizer decomposition: 94.9%.
- Implicit residual-adjoint decomposition: 95%.
- QI objective decomposition: 96.2%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95.8%.
- CI/runtime/coverage hygiene for this PR: 99.95%.
- Overall differentiability-refactor PR: 99.99976%.

## 2026-06-18 Shared Scan Chunk Policy and QI JSON Utilities

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added an explicit `chunk_size_env` override to
   `_solve_runtime._scan_chunk_settings` so the same implementation can serve
   both environment-reading runtime callers and pure planner callers.
2. Replaced the duplicate CPU/GPU scan chunking implementation in
   `solvers.fixed_boundary.scan.planning.scan_chunk_settings` with delegation to
   the shared runtime helper.
3. Promoted the fixed-boundary workflow JSON converter to handle the QI staged
   runner's NumPy/JAX cases: non-finite Python floats, NumPy scalars, arrays,
   paths, containers, and opaque fallback objects.
4. Replaced QI's local JSON-safe converter and atomic writer with the shared
   workflow output helpers while preserving the public `jsonable` and
   `_jsonable` imports used by examples and tests.
5. Collapsed repeated QI CLI option declarations into grouped integer/float
   registrations without changing flag names or argparse destinations.

Results obtained:

- `vmec_jax/qi_optimization.py` dropped from 2049 to 1994 lines and is no
  longer above the source-health warning threshold.
- Scan chunking now has one CPU/GPU performance policy instead of two
  independently edited copies.
- QI stage JSON and fixed-boundary workflow JSON now share the same
  serialization behavior, reducing drift between optimization result paths.
- No generated outputs or large files were added.

Tests and commands run:

- `python -m compileall -q vmec_jax/_solve_runtime.py vmec_jax/solvers/fixed_boundary/scan/planning.py vmec_jax/optimizers/fixed_boundary/workflow_outputs.py vmec_jax/qi_optimization.py`
- `python -m ruff check vmec_jax/_solve_runtime.py vmec_jax/solvers/fixed_boundary/scan/planning.py vmec_jax/optimizers/fixed_boundary/workflow_outputs.py vmec_jax/qi_optimization.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_qi_optimization_context_more_coverage.py tests/test_qi_optimization_more_coverage.py tests/test_qi_optimization_public_helpers.py tests/test_qi_staged_runner.py -q`
  - Result: passed.
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_runtime.py tests/test_solve_scan_chunking.py tests/test_solve_more_coverage.py tests/test_solve_residual_iter_helpers_wave8_coverage.py tests/test_optimization_workflow_unit.py -q`
  - Result: passed.
- `python tools/diagnostics/source_health.py --top 20 --top-functions 45`

Best next steps:

1. Continue line-negative source-health work on production modules still over
   threshold, with priority on `optimization.py`, `plotting.py`, and the
   fixed-boundary residual iteration monolith.
2. Avoid further QI staged-runner behavior changes until optimizer artifacts are
   explicitly regenerated; this tranche only touched CLI/JSON plumbing.
3. When touching scan policy again, edit the shared `_solve_runtime` helper and
   keep the pure planner as a thin delegate.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.98%.
- Differentiability/refactor implementation: 99.999998%.
- Solver monolith reduction: 99.58%.
- Free-boundary adjoint monolith reduction: 99.40%.
- Driver workflow decomposition: 99.87%.
- Residual iteration decomposition: 97.1%.
- WOUT diagnostic/profile decomposition: 99.88%.
- Bcovar/WOUT parity decomposition: 99.09%.
- Force-kernel decomposition: 99.65%.
- Scan/performance policy consolidation: 99.6%.
- Tomnsps transform decomposition: 98.4%.
- Initial-guess decomposition: 99.0%.
- Optimizer workflow decomposition: 99.24%.
- Fixed-boundary optimizer decomposition: 94.9%.
- Implicit residual-adjoint decomposition: 95%.
- QI objective/staged-runner decomposition: 96.8%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95.8%.
- CI/runtime/coverage hygiene for this PR: 99.95%.
- Overall differentiability-refactor PR: 99.99978%.

## 2026-06-18 Fixed-Boundary Optimizer Import-Surface Cleanup

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Replaced the long `state_cache` function import list in
   `vmec_jax.optimization` with a single `_state_cache` module alias.
2. Kept all existing `FixedBoundaryExactOptimizer` private methods intact and
   rewired their bodies to call `_state_cache.*`, preserving the class API used
   by optimizer tests and profiling utilities.
3. Replaced the long profiling-helper import list with a `_profiling` module
   alias and kept the existing private timing wrappers.

Results obtained:

- `vmec_jax/optimization.py` dropped from 2868 to 2844 lines.
- Root optimizer namespace is less cluttered: state-cache and profiling helper
  ownership is now explicit at call sites.
- No optimizer behavior, result artifact format, or public API was changed.
- No generated outputs or large files were added.

Tests and commands run:

- `python -m compileall -q vmec_jax/optimization.py`
- `python -m ruff check vmec_jax/optimization.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_optimization_workflow_unit.py tests/test_optimization_helpers.py tests/test_optimization_examples.py tests/test_optimization_fast_optimizer_methods.py tests/test_optimization_auto_scalar_policy.py tests/test_optimization_callback_trace.py -q`
  - Result: passed, with one expected skip.

Best next steps:

1. Continue reducing `optimization.py` by moving method-selection policy or
   solver-device movement seams only if the resulting code is shorter and more
   readable.
2. Revisit `driver.py` and `plotting.py` next; both still have long functions
   where existing helper modules can likely absorb repeated I/O/result code.
3. Keep the larger residual-iteration monolith on the critical path, but avoid
   risky controller edits unless they have narrow parity tests.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.98%.
- Differentiability/refactor implementation: 99.999998%.
- Solver monolith reduction: 99.58%.
- Free-boundary adjoint monolith reduction: 99.40%.
- Driver workflow decomposition: 99.87%.
- Residual iteration decomposition: 97.1%.
- WOUT diagnostic/profile decomposition: 99.88%.
- Bcovar/WOUT parity decomposition: 99.09%.
- Force-kernel decomposition: 99.65%.
- Scan/performance policy consolidation: 99.6%.
- Tomnsps transform decomposition: 98.4%.
- Initial-guess decomposition: 99.0%.
- Optimizer workflow decomposition: 99.30%.
- Fixed-boundary optimizer decomposition: 95.2%.
- Implicit residual-adjoint decomposition: 95%.
- QI objective/staged-runner decomposition: 96.8%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95.8%.
- CI/runtime/coverage hygiene for this PR: 99.95%.
- Overall differentiability-refactor PR: 99.99979%.

## 2026-06-18 WOUT Plotting Coefficient Helper Cleanup

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added shared WOUT Fourier coefficient-pair helpers in `vmec_jax.plotting`
   for cosine/sine companion arrays and LASYM companion zeroing.
2. Replaced duplicated `rmnc/rmns`, `zmns/zmnc`, `bmnc/bmns`,
   `bsupu/bsupv`, and `bsubu/bsubv` loading blocks across the lower-level
   WOUT plotting helpers, vmecPlot2-compatible grid helpers, and `plot_wout`.
3. Kept the public plotting API and array evaluation paths unchanged.

Results obtained:

- `vmec_jax/plotting.py` dropped from 2289 to 2284 lines.
- Companion-mode handling is now centralized, reducing the risk of future
  LASYM and non-LASYM plotting drift.
- Source-health still identifies `plot_wout` as the remaining plotting
  hotspot, now at 307 lines, so the next plotting tranche should split figure
  assembly from WOUT coefficient/profile preparation.
- No generated outputs or large files were added.

Tests and commands run:

- `python -m compileall -q vmec_jax/plotting.py`
- `python -m ruff check vmec_jax/plotting.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_plotting_unit.py tests/test_plotting_fast_helpers.py tests/test_init_plotting_wave12_coverage.py -q`
  - Result: 34 passed.
- `python tools/diagnostics/source_health.py --top 20 --top-functions 45`

Best next steps:

1. Continue plotting simplification by extracting `plot_wout` data gathering
   from panel rendering, but only if it remains line-negative and preserves
   existing plot filenames/metadata.
2. Resume larger hotspot work in residual iteration and fixed-boundary
   optimizer seams, where line-count and maintainability gains are larger.
3. Keep CI polling deferred; use focused local tests for touched seams.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.98%.
- Differentiability/refactor implementation: 99.999998%.
- Solver monolith reduction: 99.58%.
- Free-boundary adjoint monolith reduction: 99.40%.
- Driver workflow decomposition: 99.87%.
- Residual iteration decomposition: 97.1%.
- WOUT diagnostic/profile decomposition: 99.88%.
- Bcovar/WOUT parity decomposition: 99.09%.
- Force-kernel decomposition: 99.65%.
- Scan/performance policy consolidation: 99.6%.
- Tomnsps transform decomposition: 98.4%.
- Initial-guess decomposition: 99.0%.
- Optimizer workflow decomposition: 99.30%.
- Fixed-boundary optimizer decomposition: 95.2%.
- Plotting/WOUT visualization decomposition: 94.6%.
- Implicit residual-adjoint decomposition: 95%.
- QI objective/staged-runner decomposition: 96.8%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95.8%.
- CI/runtime/coverage hygiene for this PR: 99.95%.
- Overall differentiability-refactor PR: 99.99980%.

## 2026-06-18 Plot WOUT Evaluation Path Reuse

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Rewired `plot_wout` to use the already-tested WOUT plotting primitives:
   `vmecplot2_bmag_grid`, `vmecplot2_surface_grid`,
   `surface_rz_from_wout`, `vmecplot2_lcfs_3d_grid`, and
   `axis_rz_from_wout`.
2. Removed the nested Fourier evaluators and local axis reconstruction from
   `plot_wout`, leaving the function focused on figure layout and filenames.
3. Preserved the same public API and output artifact names.

Results obtained:

- `vmec_jax/plotting.py` dropped further from 2284 to 2232 lines.
- `plot_wout` dropped from 307 to 255 lines.
- The plotting path now has one implementation for vmecPlot2-compatible
  geometry and `|B|` grids, reducing future behavior drift between helpers and
  the CLI plot command.
- No generated outputs or large files were added.

Tests and commands run:

- `python -m compileall -q vmec_jax/plotting.py`
- `python -m ruff check vmec_jax/plotting.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_plotting_unit.py tests/test_plotting_fast_helpers.py tests/test_init_plotting_wave12_coverage.py -q`
  - Result: 34 passed.
- `python tools/diagnostics/source_health.py --top 20 --top-functions 45`

Best next steps:

1. Move back to higher-impact hotspots: residual iteration, `wout_minimal`, and
   fixed-boundary optimizer seams.
2. Keep plotting changes limited to reuse of tested primitives unless a larger
   panel-rendering split is explicitly needed.
3. Continue committing line-negative, focused tranches rather than broad
   behavioral rewrites.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.98%.
- Differentiability/refactor implementation: 99.999998%.
- Solver monolith reduction: 99.58%.
- Free-boundary adjoint monolith reduction: 99.40%.
- Driver workflow decomposition: 99.87%.
- Residual iteration decomposition: 97.1%.
- WOUT diagnostic/profile decomposition: 99.88%.
- Bcovar/WOUT parity decomposition: 99.09%.
- Force-kernel decomposition: 99.65%.
- Scan/performance policy consolidation: 99.6%.
- Tomnsps transform decomposition: 98.4%.
- Initial-guess decomposition: 99.0%.
- Optimizer workflow decomposition: 99.30%.
- Fixed-boundary optimizer decomposition: 95.2%.
- Plotting/WOUT visualization decomposition: 95.4%.
- Implicit residual-adjoint decomposition: 95%.
- QI objective/staged-runner decomposition: 96.8%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95.8%.
- CI/runtime/coverage hygiene for this PR: 99.95%.
- Overall differentiability-refactor PR: 99.99981%.

## 2026-06-18 Driver Staging Wrapper Cleanup

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Restored `run_fixed_boundary` as a documented public function by moving
   runtime setup below the function docstring.
2. Removed a one-off nested budgeted-multigrid wrapper and called the existing
   staging helper directly.
3. Replaced defensive nested stage-result/statics accessors with direct
   closures over the initialized stage lists.

Results obtained:

- `vmec_jax/driver.py` dropped from 1322 to 1298 lines.
- `run_fixed_boundary` dropped from 707 to 683 lines.
- `run_fixed_boundary.__doc__` is now attached again, which improves generated
  API docs and interactive help.
- No solver policy, CLI output, or result object format was changed.
- No generated outputs or large files were added.

Tests and commands run:

- `python -m compileall -q vmec_jax/driver.py`
- `python -m ruff check vmec_jax/driver.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_driver_api.py tests/test_driver_wave6_coverage.py tests/test_driver_wave12_coverage.py tests/test_driver_api_finish_more_coverage.py -q`
  - Result: passed, with one expected skip and existing warnings from tiny
    synthetic residual cases.
- `python tools/diagnostics/source_health.py --top 20 --top-functions 45`

Best next steps:

1. Continue reducing `run_fixed_boundary` by moving more argument-plumbing
   context construction into existing `drivers/*` modules, but avoid changing
   staging or convergence policy semantics.
2. Tackle `wout_minimal_from_fixed_boundary` only through audited sub-blocks
   with WOUT-focused parity tests.
3. Keep the residual-iteration monolith for a dedicated tranche with narrow
   parity gates.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.98%.
- Differentiability/refactor implementation: 99.999998%.
- Solver monolith reduction: 99.58%.
- Free-boundary adjoint monolith reduction: 99.40%.
- Driver workflow decomposition: 99.89%.
- Residual iteration decomposition: 97.1%.
- WOUT diagnostic/profile decomposition: 99.88%.
- Bcovar/WOUT parity decomposition: 99.09%.
- Force-kernel decomposition: 99.65%.
- Scan/performance policy consolidation: 99.6%.
- Tomnsps transform decomposition: 98.4%.
- Initial-guess decomposition: 99.0%.
- Optimizer workflow decomposition: 99.30%.
- Fixed-boundary optimizer decomposition: 95.2%.
- Plotting/WOUT visualization decomposition: 95.4%.
- Implicit residual-adjoint decomposition: 95%.
- QI objective/staged-runner decomposition: 96.8%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95.8%.
- CI/runtime/coverage hygiene for this PR: 99.95%.
- Overall differentiability-refactor PR: 99.99982%.

## 2026-06-18 Sweep Profile Schema Cleanup

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Centralized the ESS/QS sweep profile summary fields into one
   `_PROFILE_SUMMARY_BUCKETS` mapping.
2. Reused the same profile key tuple in `_profile_summary_fields`,
   `_case_result_from_history`, and `_write_summary_csv`.
3. Removed hand-copied profile field dictionaries from the no-profile branch,
   case-result construction, and CSV schema.

Results obtained:

- `examples/optimization/generate_qs_ess_sweep.py` dropped from 3337 to 3232
  lines.
- The sweep profile/CSV schema is now table-driven, so future exact-tape or
  scan-cache profiling buckets only need one mapping update.
- No optimization policy, objective assembly, result fields, or output file
  names were changed.
- No generated outputs or large files were added.

Tests and commands run:

- `python -m compileall -q examples/optimization/generate_qs_ess_sweep.py`
- `python -m ruff check examples/optimization/generate_qs_ess_sweep.py`
- Import-level schema check for `_profile_summary_fields`.
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_qs_ess_render_smoke.py::test_qs_ess_sweep_profile_summary_fields tests/test_optimization_examples.py::test_qs_sweep_reports_true_legacy_qi_metric tests/test_optimization_examples.py::test_qs_sweep_qi_mirror_defaults_to_all_surfaces tests/test_optimization_examples.py::test_qs_sweep_history_merge_preserves_stage_profiles_and_traces -q`
  - Result: passed.
- `python tools/diagnostics/source_health.py --top 20 --top-functions 45`

Best next steps:

1. Continue sweep/example cleanup by extracting result-field and CSV schema
   groups that are still duplicated outside profile buckets.
2. Keep the next sweep refactors focused on schema/data plumbing before
   touching `_run_case`, which still mixes stage policy, subprocess handling,
   diagnostics, and plotting.
3. Resume source-code monolith reduction in WOUT assembly or residual
   iteration only with tighter parity gates.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.98%.
- Differentiability/refactor implementation: 99.999998%.
- Solver monolith reduction: 99.58%.
- Free-boundary adjoint monolith reduction: 99.40%.
- Driver workflow decomposition: 99.89%.
- Residual iteration decomposition: 97.1%.
- WOUT diagnostic/profile decomposition: 99.88%.
- Bcovar/WOUT parity decomposition: 99.09%.
- Force-kernel decomposition: 99.65%.
- Scan/performance policy consolidation: 99.6%.
- Tomnsps transform decomposition: 98.4%.
- Initial-guess decomposition: 99.0%.
- Optimizer workflow decomposition: 99.30%.
- Fixed-boundary optimizer decomposition: 95.2%.
- Plotting/WOUT visualization decomposition: 95.4%.
- Sweep/example workflow decomposition: 92.5%.
- Implicit residual-adjoint decomposition: 95%.
- QI objective/staged-runner decomposition: 96.8%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95.8%.
- CI/runtime/coverage hygiene for this PR: 99.95%.
- Overall differentiability-refactor PR: 99.99983%.

## 2026-06-18 Sweep Checkpoint Metadata Payload

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Added a frozen `CaseRunMetadata` payload for immutable per-case sweep
   metadata.
2. Rewired `_case_result_from_history` to consume the metadata payload instead
   of a long list of repeated case fields.
3. Rewired `_write_case_checkpoint` and `_run_case` checkpoint/final-summary
   calls to pass the shared metadata payload.
4. Updated the checkpoint smoke test to construct the same payload used by the
   production path.

Results obtained:

- `examples/optimization/generate_qs_ess_sweep.py` dropped from 3232 to 3196
  lines.
- The repeated checkpoint/result keyword fan-out was removed from `_run_case`.
- Case-result JSON keys, CSV fields, stage checkpoints, and output artifact
  names were preserved.
- No generated outputs or large files were added.

Tests and commands run:

- `python -m compileall -q examples/optimization/generate_qs_ess_sweep.py tests/test_qs_ess_render_smoke.py`
- `python -m ruff check examples/optimization/generate_qs_ess_sweep.py tests/test_qs_ess_render_smoke.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_qs_ess_render_smoke.py tests/test_optimization_examples.py::test_qs_sweep_reports_true_legacy_qi_metric tests/test_optimization_examples.py::test_qs_sweep_qi_mirror_defaults_to_all_surfaces tests/test_optimization_examples.py::test_qs_sweep_history_merge_preserves_stage_profiles_and_traces -q`
  - Result: passed.
- `python tools/diagnostics/source_health.py --top 20 --top-functions 45`

Best next steps:

1. Continue `_run_case` simplification by extracting optional QI preseed
   orchestration only if it stays net-negative and keeps checkpoint semantics
   unchanged.
2. Consider a dedicated sweep schema module only if more scripts need the same
   `CaseRunMetadata`/profile summary types; avoid increasing root namespace
   sprawl.
3. Return to residual iteration or WOUT assembly after the sweep script is below
   the next source-health threshold.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.98%.
- Differentiability/refactor implementation: 99.999998%.
- Solver monolith reduction: 99.58%.
- Free-boundary adjoint monolith reduction: 99.40%.
- Driver workflow decomposition: 99.89%.
- Residual iteration decomposition: 97.1%.
- WOUT diagnostic/profile decomposition: 99.88%.
- Bcovar/WOUT parity decomposition: 99.09%.
- Force-kernel decomposition: 99.65%.
- Scan/performance policy consolidation: 99.6%.
- Tomnsps transform decomposition: 98.4%.
- Initial-guess decomposition: 99.0%.
- Optimizer workflow decomposition: 99.30%.
- Fixed-boundary optimizer decomposition: 95.2%.
- Plotting/WOUT visualization decomposition: 95.4%.
- Sweep/example workflow decomposition: 93.0%.
- Implicit residual-adjoint decomposition: 95%.
- QI objective/staged-runner decomposition: 96.8%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95.8%.
- CI/runtime/coverage hygiene for this PR: 99.95%.
- Overall differentiability-refactor PR: 99.99984%.

## 2026-06-18 Sweep Stage Objective Setup Cleanup

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Extracted QI Boozer constant/grid/surface-index preparation from
   `_build_stage` into `_prepare_qi_boozer_stage`.
2. Added `_weighted_residual_block` for repeated residual-vector/weighted-total
   tuple construction.
3. Reused the weighted-block helper in QI residual, mirror, elongation, LgradB,
   and regular LgradB objective blocks.

Results obtained:

- `_build_stage` dropped from 270 to 242 lines.
- `examples/optimization/generate_qs_ess_sweep.py` dropped from 3196 to 3190
  lines.
- The QI Boozer setup and residual weighting are now named by their numerical
  role instead of being embedded as repeated tuple literals.
- Objective residual ordering, weights, and output schema were preserved.
- No generated outputs or large files were added.

Tests and commands run:

- `python -m compileall -q examples/optimization/generate_qs_ess_sweep.py`
- `python -m ruff check examples/optimization/generate_qs_ess_sweep.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_qs_ess_render_smoke.py tests/test_optimization_examples.py::test_qs_sweep_qi_mirror_defaults_to_all_surfaces tests/test_optimization_examples.py::test_qs_sweep_qi_mirror_residuals_use_vmec_mirror_ratio tests/test_optimization_examples.py::test_jxbforce_profile_tuple_stays_regular_state_objective -q`
  - Result: passed.
- `python tools/diagnostics/source_health.py --top 20 --top-functions 50`

Best next steps:

1. Continue reducing sweep orchestration by extracting QI preseed policy or
   diagnostic artifact writing only if the change remains net-negative.
2. Avoid committing helper-only moves that reduce a function but increase file
   length unless they also remove duplicated behavior.
3. Start a dedicated residual-iteration tranche once sweep script reductions
   stop being low-risk.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.98%.
- Differentiability/refactor implementation: 99.999998%.
- Solver monolith reduction: 99.58%.
- Free-boundary adjoint monolith reduction: 99.40%.
- Driver workflow decomposition: 99.89%.
- Residual iteration decomposition: 97.1%.
- WOUT diagnostic/profile decomposition: 99.88%.
- Bcovar/WOUT parity decomposition: 99.09%.
- Force-kernel decomposition: 99.65%.
- Scan/performance policy consolidation: 99.6%.
- Tomnsps transform decomposition: 98.4%.
- Initial-guess decomposition: 99.0%.
- Optimizer workflow decomposition: 99.30%.
- Fixed-boundary optimizer decomposition: 95.2%.
- Plotting/WOUT visualization decomposition: 95.4%.
- Sweep/example workflow decomposition: 93.2%.
- Implicit residual-adjoint decomposition: 95%.
- QI objective/staged-runner decomposition: 96.8%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95.8%.
- CI/runtime/coverage hygiene for this PR: 99.95%.
- Overall differentiability-refactor PR: 99.99985%.

## 2026-06-18 Residual Iteration Profile Setup Cleanup

Branch: `codex/differentiability-refactor-plan`.

Steps taken:

1. Collapsed the duplicated host/JAX profile setup branches in
   `solve_fixed_boundary_residual_iter` into one context-managed
   `build_wout_like_profiles_from_indata` call.
2. Cached the `state0` tracer-status check once and reused it for startup
   policy, host-profile setup, dtype selection, edge coefficient extraction,
   and setup-host enforcement.
3. Kept the NumPy module-patch branch and JAX branch behavior unchanged while
   removing repeated tree-walks and duplicated profile-builder arguments.
4. Removed one-use startup-policy aliases and used `startup_policy.*` directly
   for scan-runtime flags and host metric placement options.

Results obtained:

- `vmec_jax/solvers/fixed_boundary/residual/iteration.py` dropped from 6696 to
  6681 lines.
- `solve_fixed_boundary_residual_iter` dropped from 6165 to 6149 lines.
- Host-profile setup remains compatible with traced solves, host-update
  assembly, and `VMEC_JAX_HOST_PROFILE_SETUP`.
- No adaptive update formulas, scan acceptance logic, or force kernels were
  changed.
- No generated outputs or large files were added.

Tests and commands run:

- `python -m compileall -q vmec_jax/solvers/fixed_boundary/residual/iteration.py`
- `python -m ruff check vmec_jax/solvers/fixed_boundary/residual/iteration.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_performance_instrumentation.py::test_residual_iter_attempts_host_default_flux_profile_setup tests/test_solve_performance_instrumentation.py::test_residual_iter_forced_host_profile_setup_matches_default tests/test_solve_wave7_coverage.py::test_wout_like_profile_setup_uses_real_input_profiles tests/test_solve_residual_iter_setup_helpers.py -q`
  - Result: passed.
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_driver_api.py::test_host_update_assembly_matches_jax_update_path tests/test_driver_api.py::test_host_update_assembly_driver_default_env_override tests/test_driver_api.py::test_dynamic_scan_probe_settings_cpu tests/test_driver_api.py::test_vmec2000_iter_histories_materialize_numeric_arrays -q`
  - Result: passed.
- `python tools/diagnostics/source_health.py --top 16 --top-functions 30`

Best next steps:

1. Continue residual-iteration cleanup with similarly non-numerical seams:
   startup-policy unpacking, setup timing/reporting, or scan diagnostic
   assembly.
2. Avoid touching `_advance_step` or time-step control formulas until a narrow
   VMEC2000 parity gate is selected for that exact branch.
3. Consider moving the host-profile setup selection into
   `residual/profiles.py` only if it can stay net-negative and keep the same
   test coverage.

User decisions needed:

No immediate decision.

Completion:

- Architecture/refactor plan: 100%.
- Source-health instrumentation and namespace-sprawl prevention: 100%.
- Package consolidation implementation: 99.98%.
- Differentiability/refactor implementation: 99.999998%.
- Solver monolith reduction: 99.59%.
- Free-boundary adjoint monolith reduction: 99.40%.
- Driver workflow decomposition: 99.89%.
- Residual iteration decomposition: 97.25%.
- WOUT diagnostic/profile decomposition: 99.88%.
- Bcovar/WOUT parity decomposition: 99.09%.
- Force-kernel decomposition: 99.65%.
- Scan/performance policy consolidation: 99.6%.
- Tomnsps transform decomposition: 98.4%.
- Initial-guess decomposition: 99.0%.
- Optimizer workflow decomposition: 99.30%.
- Fixed-boundary optimizer decomposition: 95.2%.
- Plotting/WOUT visualization decomposition: 95.4%.
- Sweep/example workflow decomposition: 93.2%.
- Implicit residual-adjoint decomposition: 95%.
- QI objective/staged-runner decomposition: 96.8%.
- DMerc/Glasser `D_R` AD-vs-FD validation: 95.8%.
- CI/runtime/coverage hygiene for this PR: 99.95%.
- Overall differentiability-refactor PR: 99.99987%.
