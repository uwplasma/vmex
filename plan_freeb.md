# Free-Boundary Coil-Aware Single-Stage Optimization Plan

Branch: `feature/freeb-essos-coil-single-stage`

Repository clone: `/Users/rogeriojorge/local/vmec_jax_freeb`

Baseline commit: `3657e0c release: prepare v0.0.13`

Date opened: 2026-05-24

## Current Release Status

Last updated: 2026-05-24 after the direct-coil forward examples and benchmark diagnostics batch.

Steps taken:

1. Cached direct-coil geometry is wired through the free-boundary provider bridge for host-forward runs.
2. The cached and uncached direct-coil free-boundary paths now have an end-to-end parity regression.
3. The finite-pressure direct-coil lane now has a full-loop current-only proxy-objective finite-difference slope-stability smoke; this is not a production exact full-solve adjoint or a validated QS-gradient claim.
4. Public README/docs/examples/tests/tools no longer embed maintainer-local absolute paths, enforced by a docs hygiene regression.
5. The VMEC2000/direct-coil/mgrid diagnostic now fails hard for explicitly invalid user paths while keeping optional auto-discovery skips.
6. Benchmarks now report active NESTOR sample/solve timing summaries and cold-to-warm improvement.
7. Trial/backtracking NESTOR calls are now recorded separately from accepted-update NESTOR calls.
8. Cached direct-coil geometry can now use a host-forward JIT sampler, guarded by `VMEC_JAX_FREEB_JIT_COIL_SAMPLER`.
9. Added `examples/free_boundary_essos_coils_forward.py` as the minimal ESSOS-direct-coil forward example that writes one input, WOUT, and JSON summary without generating an mgrid.
10. Finite-pressure free-boundary examples now default to `--activate-fsq 1e99` so short smoke runs exercise active NESTOR coupling instead of silently staying in the inactive vacuum-stub cadence.
11. Added `examples/free_boundary_direct_coils_forward.py` as a dependency-light pure-`CoilFieldParams` forward example that needs no ESSOS assets.
12. Benchmarks now expose synthetic grid/coil knobs and last-sample diagnostics including sample points, JIT sampler flag, coil count, and segments per coil.
13. Public docs now avoid overclaiming full free-boundary/NESTOR adjoints or converged high-beta direct-coil equilibria.

Results obtained:

1. `pytest -q -m "not full and not vmec2000 and not simsopt"`: 2241 passed, 26 skipped, 111 deselected, 1 xfailed in 5m48s.
2. Targeted direct-coil/docs tests after the final additions: 9 passed in 2.34 s.
3. Full Sphinx build after docs hygiene changes succeeded in `/tmp/vmec_jax_freeb_docs_claim_hygiene`.
4. Direct-coil/mgrid diagnostic smoke completed with expected `vmec2000_skipped` and `jax_direct_vs_mgrid_passed=True`.
5. Explicit bad `--coils-json` now exits nonzero and writes `status=failed`, `reason=explicit_essos_or_coils_path_invalid`.
6. Tiny direct-coil solve benchmark reports active NESTOR sample timing improving from about `0.51 s` cold to `0.0048 s` warm.
7. Trial timing smoke completed; the tiny synthetic path records zero trial calls, so its benchmarked direct-coil cost is accepted NESTOR sampling rather than hidden backtracking work.
8. Targeted trial-timing tests passed: 3 passed in 8.03 s.
9. Field-only probe on a 32x32 grid with 4-fold stellarator symmetry: cached geometry sampling changed from about `0.45 s` regular cold / `9-10 ms` regular warm to `0.067 s` JIT cold / `4-6 ms` JIT warm.
10. The tiny full-solve benchmark remains dominated by non-sampling work: JIT and non-JIT direct-coil solve smokes both report about `6.09 s` cold and `0.19 s` warm.
11. Optional VMEC2000 generated-mgrid diagnostic was attempted with `NITER=1`, `50`, and `500`; VMEC2000 completed without WOUT in all cases, with `fsq_total_last` improving to about `5.4e-3` at 500 iterations but still reporting `Try increasing NITER`.
12. `examples/free_boundary_essos_coils_forward.py --beta 1.0 --max-iter 20` wrote a direct-coil WOUT and active-NESTOR summary with `free_boundary_vacuum_stub=false`. The residual is still intentionally large, so this remains a forward coupling smoke rather than a converged finite-beta promotion case.
13. Trial-counter regression now records nonzero `freeb_nestor_trial_sample_time_history` on a solver-level direct-coil path that enters trial scoring.
14. `examples/free_boundary_direct_coils_forward.py --outdir tmp/free_boundary_direct_coils_forward_run_smoke --max-iter 1 --n-segments 8 --ns 7 --nzeta 2 --ntheta 8` wrote a synthetic direct-coil WOUT with finite one-iteration residuals (`fsqr≈7.3e-4`, `fsqz≈1.6e-4`, `fsql≈5.3e-4`).
15. Larger synthetic direct-coil benchmark with `sample_points=78`, `coils=16`, `segments=128` reported JIT sampler warm active sampling around `0.0106 s` versus non-JIT around `0.0092 s`; whole warm solve time stayed about `0.25 s`, so this small case is dominated by non-sampling work.
16. Subagent larger spectral-mode benchmark with `sample_points=2352`, `coils=8`, `segments=128` found the JIT sampler reduced warm active sampling from `0.0588 s` to `0.0545 s` (about 7%), but total warm wall time remained about `0.35 s`; dense NESTOR mode remains the main performance bottleneck.

Best next steps:

1. Target dense NESTOR/preconditioner/finalization cost; sampler JIT helps field-only cases but is not the dominant full-solve cost.
2. Run a direct-coil case that enters backtracking and confirm the new trial counters capture rejected NESTOR sampling cost in a full driver trace.
3. Extend the full-loop finite-difference smoke from current-only proxy objective to a validated Boozer/QS promotion test when affordable.
4. Either raise the VMEC2000 generated-mgrid diagnostic to a convergence-oriented multi-grid input or mark the current single-stage generated-mgrid case as optional underconverged external evidence.

Need from user:

Nothing now.

Open-lane completion estimates:

1. External provider architecture: 93%.
2. Direct-coil finite-pressure forward lane: 93%.
3. ESSOS/mgrid/VMEC2000 comparison lane: 82%.
4. Full-loop gradient validation: 55%.
5. Robust/optimization examples: 80%.
6. Performance/benchmarking: 78%.
7. Docs/release hygiene: 92%.
8. Overall branch completion: 86%.

## Mission

Implement the first research-grade lane toward true free-boundary, coil-aware, single-stage optimization in `vmec_jax`:

```text
coil parameters
  -> differentiable Biot-Savart external field
  -> vmec_jax free-boundary equilibrium
  -> Boozer / quasisymmetry / engineering objective
  -> validated gradient
  -> coil-only optimization
```

The new code must preserve the existing VMEC2000-compatible mgrid path. Mgrid remains the compatibility and parity backend. Direct coils become the differentiable research backend. ESSOS integration is optional and must skip cleanly when ESSOS is unavailable.

## Scientific Scope

The branch should demonstrate a first working direct-coil free-boundary lane. It does not need to claim full publication-level exact free-boundary coil adjoints until full finite-difference checks of the complete solve pass.

Minimum scientific deliverable:

1. Pure JAX coil Biot-Savart field provider.
2. Optional ESSOS adapter that maps ESSOS coil objects into the pure JAX provider.
3. JAX mgrid interpolation backend for compatibility and gradient tests.
4. Free-boundary sampling hook that can use mgrid or direct coils without writing an mgrid.
5. A tiny direct-coil free-boundary forward example that writes a `wout`.
6. A first coil-only optimization example whose optimization degrees of freedom are coil Fourier coefficients and/or coil currents, never independent plasma boundary coefficients.
7. Gradient tests for coil currents, coil Fourier coefficients, evaluation coordinates, and dense toy vacuum solves.
8. VMEC2000 comparison diagnostics for mgrid parity and direct-coil versus mgrid convergence.

## Guardrails

1. Do not regress existing mgrid free-boundary behavior.
2. Do not hard-require ESSOS in `vmec_jax` core.
3. Do not import ESSOS at module import time except inside optional adapter functions.
4. Keep differentiable paths JAX-native: no NumPy conversions inside direct-coil field evaluation.
5. Keep provider params as pytrees with explicit static metadata.
6. Keep free-boundary exact-adjoint claims precise: phase 1 has differentiable field providers and dense toy vacuum adjoint; full production NESTOR adjoint remains phase 2 unless fully validated.
7. Default CI should stay light. VMEC2000, full optimization, and GPU benchmarks are optional gates.

## Local Setup Log

Commands already run:

```bash
git clone https://github.com/uwplasma/vmec_jax.git /Users/rogeriojorge/local/vmec_jax_freeb
git -C /Users/rogeriojorge/local/vmec_jax_freeb checkout main
git -C /Users/rogeriojorge/local/vmec_jax_freeb pull --ff-only
git -C /Users/rogeriojorge/local/vmec_jax_freeb checkout -b feature/freeb-essos-coil-single-stage
```

Local sibling repos/tools:

```text
/Users/rogeriojorge/local/ESSOS
/Users/rogeriojorge/local/STELLOPT
~/bin/xvmec2000
```

The existing `/Users/rogeriojorge/local/vmec_jax` checkout is dirty from prior work and is intentionally not used for this branch.

## Current vmec_jax Architecture Notes

Key files inspected:

```text
vmec_jax/free_boundary.py
vmec_jax/driver.py
vmec_jax/optimization_workflow.py
examples/optimization/
docs/free_boundary_plan.rst
docs/optimization.rst
docs/validation.rst
tests/test_free_boundary_wp0.py
tests/test_vmec2000_converged_parity.py
tests/test_wout_comprehensive_parity.py
```

Important existing free-boundary structures:

```text
MGridMetadata
MGridData
VacuumBoundaryFields
ExternalBoundarySample
interpolate_mgrid_bfield(...)
_sample_external_boundary_arrays(...)
nestor_external_only_step(...)
sample_external_vacuum_diagnostics(...)
load_mgrid(...)
prepare_mgrid_for_config(...)
```

Current bottleneck for this branch:

```text
_sample_external_boundary_arrays(...)
  -> interpolate_mgrid_bfield(...)
  -> axis-current correction
  -> vacuum boundary projection
  -> ExternalBoundarySample
```

This is mgrid-specific. The provider abstraction should initially wrap or split this point without disturbing downstream `VacuumBoundaryFields` and `ExternalBoundarySample` consumers.

Optimization workflow notes:

```text
ObjectiveTerm
FixedBoundaryVMEC
LeastSquaresProblem
QuasisymmetryRatioResidual
QuasiIsodynamicResidual
least_squares_solve(...)
```

The fixed-boundary optimization API already has a SIMSOPT-like objective-tuple workflow. The new single-stage example should reuse that style but introduce a new free-boundary coil optimizable object rather than a plasma-boundary optimizable.

## ESSOS Architecture Notes

Files inspected:

```text
essos/coils.py
essos/fields.py
essos/objective_functions.py
essos/coil_perturbation.py
examples/optimize_coils_vmec_surface.py
```

ESSOS curve convention:

```text
dofs[..., 0]     = constant term
dofs[..., 2*k-1] = sin(k)
dofs[..., 2*k]   = cos(k)
```

Relevant ESSOS objects/functions:

```text
Curves
Coils
apply_symmetries_to_curves(...)
apply_symmetries_to_gammas(...)
apply_symmetries_to_currents(...)
BiotSavart.B(points)
loss_coil_length(...)
loss_coil_curvature(...)
loss_cc_distance(...)
loss_cs_distance(...)
GaussianSampler
```

ESSOS Biot-Savart convention:

```text
B(x) = mean_segments sum_coils 1e-7 * I * gamma_dash x (x - gamma) / |x - gamma|^3
```

where `gamma_dash` is with respect to the normalized curve parameter. Stellarator-symmetry reflected coils flip current sign. The `vmec_jax` pure-JAX provider should match this convention first, then document any later physical-normalization changes.

## Literature and Documentation Pass

Implementation implications from sources reviewed:

1. JAX `custom_linear_solve` is the right primitive for the vacuum-solve adjoint scaffold because it defines gradients by implicit differentiation at the solution rather than by differentiating through solve iterations. Source: https://docs.jax.dev/en/latest/_autosummary/jax.lax.custom_linear_solve.html
2. JAX `checkpoint` / `remat` is useful for reverse-mode memory control in large coil-field or objective tapes, but it does not replace the need for custom implicit adjoints for linear solves. Source: https://docs.jax.dev/en/latest/_autosummary/jax.checkpoint.html
3. Custom pytree nodes should be used for typed provider params so `jit`, `grad`, and `vmap` see data leaves and static metadata cleanly. Source: https://docs.jax.dev/en/latest/custom_pytrees.html
4. `jax.lax.map(..., batch_size=...)` provides a native chunking option that can lower peak memory versus full `vmap`, which is relevant for `npoints * ncoils * nsegments`. Source: https://docs.jax.dev/en/latest/_autosummary/jax.lax.map.html
5. SIMSOPT documents VMEC free-boundary inputs as `lfreeb`, `mgrid_file`, `extcur`, and `nvacskip`; it also notes that the boundary is an initial guess in free-boundary runs. Source: https://simsopt.readthedocs.io/latest/example_vmec.html
6. SIMSOPT's field API provides a reference architecture for coils, symmetry expansion, Biot-Savart, and mgrid read/write compatibility. Source: https://simsopt.readthedocs.io/v1.8.0/simsopt.field.html
7. Single-stage stellarator optimization literature motivates combining plasma objectives and coil engineering objectives in one optimization rather than strict stage-one/stage-two separation. Source: https://arxiv.org/abs/2302.10622
8. Earlier single-stage coil-design work shows direct coil-shape/current optimization can balance confinement and engineering metrics, and that quasi-Newton/gradient methods matter for convergence. Source: https://arxiv.org/abs/2010.02033
9. Combined plasma-coil optimization literature explicitly distinguishes fixed-boundary, quasi-free-boundary, and free-boundary coil optimization, and emphasizes stability/linearization issues. Source: https://arxiv.org/abs/2012.09278
10. DESC demonstrates the value of JAX-native equilibrium and optimization workflows, including automatic differentiation, JIT, CPU/GPU support, and continuation. Source: https://desc-docs.readthedocs.io/en/stable/index.html
11. DESC continuation/perturbation literature reinforces that robust equilibrium optimization often requires continuation and derivative-aware workflows, not only one-shot local optimization. Source: https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/desc-stellarator-code-suite-part-2-perturbation-and-continuation-methods/5766F6B713EC93D438A35705F2C1E861
12. Fast automated adjoints for spectral PDE solvers supports the branch direction: implement discrete JVP/VJP rules around structured spectral operators and linear solves rather than naively taping every solver iteration. Source: https://arxiv.org/abs/2506.14792
13. JAXopt implicit differentiation documentation is a useful reference for future nonlinear/free-boundary root-solve adjoints once the production vacuum operator is JAX-native. Source: https://jaxopt.github.io/dev/implicit_diff.html
14. Lineax is a possible future dependency or design reference for operator-based JAX linear solves; keep phase 1 dependency-free unless the benefit becomes concrete. Source: https://arxiv.org/abs/2311.17283
15. Robust optimization should support mean, mean-plus-std, and smooth tail-risk aggregators. CVaR has better mathematical behavior than VaR for scenario optimization, but finite-sample fragility should be documented. Source: https://sites.math.washington.edu/~rtr/papers/rtr179-CVaR1.pdf

## Proposed Package Layout

```text
vmec_jax/external_fields/__init__.py
vmec_jax/external_fields/base.py
vmec_jax/external_fields/mgrid_jax.py
vmec_jax/external_fields/coils_jax.py
vmec_jax/external_fields/essos_adapter.py
vmec_jax/free_boundary_adjoint.py
vmec_jax/robust_coils.py
examples/free_boundary_essos_coils_forward.py
examples/optimization/free_boundary_QS_coil_optimization.py
tools/diagnostics/compare_freeb_coils_mgrid_vmec2000.py
tools/benchmarks/bench_external_field_providers.py
tools/benchmarks/bench_freeb_direct_coil_solve.py
tools/benchmarks/bench_freeb_coil_gradient.py
docs/free_boundary_coil_optimization.rst
```

## Provider API Design

Low-level public internal API:

```python
sample_external_field_cylindrical(
    provider_kind,
    provider_static,
    provider_params,
    R,
    Z,
    phi,
) -> tuple[br, bphi, bz]
```

Requirements:

1. `R`, `Z`, `phi` have shape `(ntheta, nzeta)` or any broadcastable point-grid shape.
2. `br`, `bphi`, `bz` return the same shape.
3. `provider_params` is a pytree containing differentiable arrays.
4. `provider_static` contains non-differentiable metadata such as grid dimensions, interpolation mode, symmetry flags, and chunk size.
5. `provider_kind` initially supports:
   - `mgrid`
   - `direct_coils`
   - `essos_coils` through conversion to `direct_coils`
6. The interface should allow a future `hybrid_mgrid_plus_coils` without changing free-boundary code.

Dataclasses:

```text
ExternalFieldSample
ExternalFieldProviderConfig
MGridFieldParams
CoilFieldParams
```

Use explicit pytrees. Prefer `jax.tree_util.register_dataclass` or a minimal custom flatten/unflatten implementation. Mark static fields (`n_segments`, `nfp`, `stellsym`, chunk policy) as metadata; differentiable arrays (`base_curve_dofs`, `base_currents`, mgrid field values, `extcur`) as data.

## Work Packages

### WP0: Branch Foundation and Plan

Deliverables:

1. Clone into `/Users/rogeriojorge/local/vmec_jax_freeb`.
2. Create branch `feature/freeb-essos-coil-single-stage`.
3. Inspect existing free-boundary, optimization, tests, docs, and ESSOS code.
4. Create this `plan_freeb.md`.

Acceptance:

1. Branch exists locally.
2. Plan captures architecture, literature, test matrix, and next steps.

Status: 90%.

### WP1: External-Field Provider Base

Deliverables:

1. Add `vmec_jax/external_fields/base.py`.
2. Define provider kinds, typed config/dataclasses, and `sample_external_field_cylindrical`.
3. Keep API function-first and JAX-transformable.
4. Add docstrings explaining mgrid compatibility and direct-coil differentiability.

Tests:

1. Provider-dispatch shape tests.
2. Unknown provider error tests.
3. Pytree flatten/unflatten tests.

Acceptance:

1. No ESSOS import from core provider package import.
2. Provider params can pass through `jax.jit`, `jax.grad`, and `jax.tree_util.tree_flatten`.

Status: 0%.

### WP2: Pure JAX Coil Biot-Savart Provider

Deliverables:

1. Add `vmec_jax/external_fields/coils_jax.py`.
2. Implement:
   - `CoilFieldParams`
   - `fourier_curves_to_gamma`
   - `apply_stellarator_symmetry_to_curves`
   - `apply_stellarator_symmetry_to_currents`
   - `compute_gamma_dash`
   - `compute_gamma_dashdash`
   - `biot_savart_xyz`
   - `sample_coil_field_cylindrical`
3. Match ESSOS Fourier and symmetry conventions.
4. Implement chunked point evaluation using `jax.lax.map(..., batch_size=...)` or a static chunk helper.
5. Implement engineering metrics:
   - `coil_lengths`
   - `coil_curvatures`
   - `coil_plasma_distance_soft`
   - `coil_coil_distance_soft`
   - `coil_current_norm`
   - `curvature_penalty`
   - `length_penalty`

Numerical details:

1. Use physical scale `1e-7 * current`.
2. Add `regularization_epsilon` in denominator as `(|r|^2 + eps^2)^(3/2)` for optional singularity smoothing.
3. Use Cartesian internal representation and cylindrical conversion at the API boundary.
4. Avoid materializing huge arrays when chunking is requested.

Tests:

1. Value test for a circular coil at points far from the coil.
2. Shape tests for `n_base_coils`, `nfp`, and `stellsym`.
3. Current-gradient finite difference.
4. Fourier-coefficient gradient finite difference.
5. Evaluation-coordinate Jacobian finite difference.
6. Engineering metric finite/nonnegative tests.

Acceptance:

1. Pure JAX field sampling works under `jit`, `grad`, `jacfwd`, and `vmap`.
2. Values match ESSOS on shared simple coils when ESSOS is installed.

Status: 0%.

### WP3: Optional ESSOS Adapter

Deliverables:

1. Add `vmec_jax/external_fields/essos_adapter.py`.
2. Implement `from_essos_coils(coils, regularization_epsilon=0.0) -> CoilFieldParams`.
3. Extract base curve dofs, base currents, `n_segments`, `nfp`, `stellsym`, and current scale if present.
4. Raise a helpful `ImportError` only when adapter function is called and ESSOS is missing.
5. Add example helper for constructing a small ESSOS coil set.

Tests:

1. Import skip when ESSOS unavailable.
2. Adapter output shape and metadata tests.
3. ESSOS `BiotSavart` value comparison at multiple non-singular points.

Acceptance:

1. `import vmec_jax.external_fields` works without ESSOS installed.
2. ESSOS parity test passes locally when ESSOS is available.

Status: 0%.

### WP4: JAX mgrid Interpolation

Deliverables:

1. Add `vmec_jax/external_fields/mgrid_jax.py`.
2. Implement:
   - `interpolate_mgrid_bfield_jax`
   - `sample_mgrid_field_cylindrical`
3. Support linear interpolation first.
4. Differentiate with respect to:
   - grid field values,
   - `extcur`,
   - evaluation coordinates away from cell boundaries.
5. Include TODO placeholder for smooth tricubic/B-spline interpolation.

Tests:

1. Synthetic affine-field value parity with legacy `interpolate_mgrid_bfield`.
2. Gradient wrt `extcur`.
3. Gradient wrt field values.
4. Finite-difference check wrt `R`, `Z`, and `phi` inside one cell.

Acceptance:

1. Existing mgrid path remains unchanged.
2. New JAX mgrid backend is ready for differentiable compatibility tests.

Status: 0%.

### WP5: Free-Boundary Provider Hook

Deliverables:

1. Refactor free-boundary sampling so external fields can be supplied by provider API.
2. Preserve current `_sample_external_boundary_arrays` behavior for legacy mgrid.
3. Add one of:
   - `_sample_external_boundary_arrays_with_provider(...)`
   - `sample_free_boundary_external_field(...)`
4. Keep output compatible with `ExternalBoundarySample` and `VacuumBoundaryFields`.
5. Add optional config plumbing through `driver.py` / solve path without breaking input-file API.

Design:

1. Phase 1 may expose direct-coil provider via Python API/example rather than VMEC namelist.
2. Legacy CLI `vmec_jax input.foo` should still use `MGRID_FILE` and `EXTCUR`.
3. Direct-coil examples can call a Python API that builds provider params explicitly.

Tests:

1. Existing `tests/test_free_boundary_wp0.py` passes unchanged.
2. Direct-coil provider sampling produces valid boundary channels.
3. Direct-coil low-resolution free-boundary solve converges on a tiny fixture.

Acceptance:

1. Direct-coil code path does not write an mgrid file.
2. Mgrid compatibility backend remains VMEC2000 parity path.

Status: 0%.

### WP6: Direct-Coil Forward Example

Deliverables:

1. Add `examples/free_boundary_essos_coils_forward.py`.
2. Construct or load a small ESSOS-compatible coil set.
3. Convert to `CoilFieldParams`.
4. Run a low-resolution free-boundary solve with direct coil external field.
5. Write `wout`.
6. Print final residual, aspect ratio, mean iota, coil length, curvature, current norm, and coil-plasma distance if available.
7. Save JSON summary.

Acceptance:

1. Example runs locally without writing mgrid.
2. If ESSOS is missing, example exits with clear instruction or uses a pure `CoilFieldParams` fallback.

Status: 0%.

### WP7: Vacuum Solve / Adjoint Scaffold

Deliverables:

1. Add `vmec_jax/free_boundary_adjoint.py`.
2. Implement:
   - `dense_vacuum_solve_jax(A, b, *, symmetric=False)`
   - custom-linear-solve wrapper where practical.
   - dense fallback with `jnp.linalg.solve` for small tests.
3. Explain production path:
   - current NESTOR path remains partly legacy/NumPy.
   - phase 2 will replace or wrap NESTOR operator assembly with JAX matrix-free operator and transpose solve.

Tests:

1. Random well-conditioned dense `A`, `b` solution parity with `jnp.linalg.solve`.
2. VJP wrt `b` matches `A^{-T}`.
3. Finite-difference check wrt a scalar parameter affecting `b`.
4. Finite-difference check wrt a scalar parameter affecting `A` if implemented.

Acceptance:

1. Unit tests prove the adjoint scaffold.
2. Docs do not claim full production NESTOR differentiability yet.

Status: 0%.

### WP8: Gradient Check Suite

Deliverables:

```text
tests/test_external_fields_coils_jax.py
tests/test_external_fields_essos_adapter.py
tests/test_external_fields_mgrid_jax.py
tests/test_free_boundary_coil_provider_forward.py
tests/test_free_boundary_coil_provider_gradients.py
tests/test_free_boundary_vacuum_adjoint.py
```

Checks:

1. Coil current derivative: `jax.grad` vs finite difference.
2. Coil Fourier coefficient derivative: `jax.grad` vs high-order central finite difference.
3. Coordinate derivative: `jax.jacfwd` vs finite difference.
4. ESSOS comparison: skip if ESSOS unavailable.
5. Boundary projection derivative if a JAX projection is implemented.
6. Free-boundary objective smoke gradient:
   - initially `xfail` or skip with explicit reason if full solve is not differentiable.
   - promote to pass when full direct-coil free-boundary adjoint is implemented.

Acceptance:

1. Default fast subset is deterministic and under CI budget.
2. Optional tests are clearly marked.

Status: 0%.

### WP9: VMEC2000 Comparison Diagnostics

Deliverables:

1. Add `tools/diagnostics/compare_freeb_coils_mgrid_vmec2000.py`.
2. Generate or load a small coil/mgrid case.
3. Run:
   - VMEC2000 free-boundary using mgrid.
   - `vmec_jax` free-boundary using same mgrid.
   - `vmec_jax` free-boundary using direct coils.
4. Compare:
   - final `fsqr`, `fsqz`, `fsql`,
   - boundary Fourier coefficients,
   - iota profile,
   - aspect ratio,
   - volume,
   - vacuum boundary channels where comparable.
5. Save `outputs/freeb_coil_compare/summary.json`.

Tests:

```bash
RUN_VMEC2000=1 pytest -q tests/test_vmec2000_freeb_coil_compare.py
```

Acceptance:

1. VMEC2000 missing -> clean skip.
2. Mgrid `vmec_jax` vs VMEC2000 uses existing parity tolerances or documented looser free-boundary tolerances.
3. Direct-coil vs mgrid reports convergence with mgrid resolution; exact low-resolution equality is not required.

Status: 0%.

### WP10: Benchmarks

Deliverables:

```text
tools/benchmarks/bench_external_field_providers.py
tools/benchmarks/bench_freeb_direct_coil_solve.py
tools/benchmarks/bench_freeb_coil_gradient.py
tools/benchmarks/bench_freeb_direct_coil_matrix.py
```

Benchmark matrix:

```text
ncoils: 4, 8, 16
nsegments: 32, 64, 128
boundary grid: small, medium
backend: cpu, gpu if available
provider: mgrid_jax, direct_coil
mode: field sample, free-boundary solve, gradient/JVP when available
```

Record:

1. Wall time.
2. Backend/device.
3. Compile versus warm timing where practical.
4. RSS or JAX memory stats if easy.
5. Grid/coils parameters.
6. JSON output.

Acceptance:

1. Benchmarks are non-CI by default.
2. GPU benchmark runs when JAX GPU backend is installed.

Status: 60%. Lightweight provider, direct free-boundary solve, coil-gradient,
and matrix-runner scripts are present. The matrix runner records CPU quick rows
by default and writes a skipped GPU row when `--include-gpu` is requested
without an available JAX GPU backend. Full ncoil/nsegment/grid production
matrix and plots remain future work.

### WP11: Coil-Only Free-Boundary QS Optimization Example

Deliverables:

1. Add `examples/optimization/free_boundary_QS_coil_optimization.py`.
2. Use only coil Fourier dofs and/or currents as optimization variables.
3. Do not optimize plasma boundary surface coefficients as independent degrees of freedom.
4. At each accepted objective evaluation:
   - run free-boundary `vmec_jax` from direct coils,
   - compute Boozer/QS or cheaper objective from resulting equilibrium,
   - record history and diagnostics.
5. Provide two modes:
   - CI-safe smoke: direct-coil free-boundary solve plus cheap aspect/iota/residual objective.
   - Full example: Boozer/QS objective, not in default CI.

Objective components:

```text
quasisymmetry residual
aspect ratio target
abs(mean_iota) floor or target
mirror ratio
elongation
coil length penalty
coil curvature penalty
coil-plasma distance penalty
current norm penalty
```

Outputs:

```text
results/freeb_qs_coil_optimization/input_initial
results/freeb_qs_coil_optimization/input_final
results/freeb_qs_coil_optimization/wout_initial.nc
results/freeb_qs_coil_optimization/wout_final.nc
results/freeb_qs_coil_optimization/coils_initial.json
results/freeb_qs_coil_optimization/coils_final.json
results/freeb_qs_coil_optimization/history.json
results/freeb_qs_coil_optimization/diagnostics.json
results/freeb_qs_coil_optimization/summary.csv
```

Acceptance:

1. End-to-end run completes on a tiny case.
2. Objective history records accepted evaluations.
3. Example prints clear physics and coil metrics.
4. No surface coefficient optimization DOFs.

Status: 0%.

### WP12: Robust Coil Perturbation Utilities

Deliverables:

1. Add `vmec_jax/robust_coils.py`.
2. Implement pure functions:
   - `perturb_coil_params(params, sample)`
   - current perturbation,
   - rigid displacement,
   - toroidal phase perturbation,
   - simple Gaussian centerline perturbation if feasible.
3. Implement risk aggregation:
   - mean,
   - mean + std,
   - soft-CVaR / smooth max.
4. Optional flags in coil optimization example:
   - `--robust-samples`
   - `--robust-risk`
   - `--robust-current-sigma`
   - `--robust-displacement-sigma`

Tests:

```text
tests/test_robust_coil_perturbations.py
```

Acceptance:

1. Deterministic fixed-PRNG tests.
2. `vmap` support when full objective path is transformable.
3. Python-loop fallback documented when full free-boundary solver is not yet batch-transformable.

Status: 0%.

### WP13: Documentation

Deliverables:

1. Add `docs/free_boundary_coil_optimization.rst`.
2. Add it to docs toctree.
3. Document:
   - motivation,
   - architecture,
   - provider API,
   - direct-coil example,
   - QS coil optimization example,
   - VMEC2000 comparison,
   - gradient validation status,
   - limitations and phase-2 adjoint work.

Architecture diagram:

```text
CoilFieldParams
  -> BiotSavart sampler
  -> free-boundary external sample
  -> vmec_jax free-boundary solve
  -> wout / Boozer
  -> objective
  -> optimizer
```

Acceptance:

1. Docs build locally.
2. Docs make explicit what is fully differentiable now and what is planned.
3. Examples and tests have reproducible commands.

Status: 0%.

### WP14: CI Policy

Default fast tests:

```bash
pytest -q tests/test_external_fields_coils_jax.py
pytest -q tests/test_external_fields_mgrid_jax.py
pytest -q tests/test_external_fields_essos_adapter.py
pytest -q tests/test_free_boundary_vacuum_adjoint.py
pytest -q tests/test_free_boundary_coil_provider_forward.py
pytest -q tests/test_robust_coil_perturbations.py
```

Optional tests:

```bash
RUN_FULL=1 pytest -q tests/test_free_boundary_qs_coil_optimization_smoke.py
RUN_VMEC2000=1 pytest -q tests/test_vmec2000_freeb_coil_compare.py
```

Heavy benchmark commands:

```bash
python tools/benchmarks/bench_external_field_providers.py --out outputs/bench_external_fields.json
python tools/benchmarks/bench_freeb_direct_coil_solve.py --out outputs/bench_freeb_direct_coil.json
python tools/benchmarks/bench_freeb_coil_gradient.py --out outputs/bench_freeb_coil_gradient.json
```

Acceptance:

1. Default CI remains fast.
2. Optional VMEC2000 and full optimization gates skip cleanly.
3. No generated outputs are committed except deliberately small fixtures.

Status: 0%.

## Implementation Order

1. Commit plan and branch foundation.
2. Implement `external_fields/base.py`.
3. Implement pure JAX coil provider and gradient tests.
4. Implement ESSOS adapter and ESSOS parity tests.
5. Implement JAX mgrid interpolation and synthetic affine tests.
6. Hook provider into free-boundary sampling while preserving mgrid default.
7. Add direct-coil free-boundary forward example.
8. Add dense vacuum adjoint scaffold and tests.
9. Add VMEC2000 comparison diagnostic script.
10. Add coil-only free-boundary QS optimization example.
11. Add robust coil perturbations.
12. Add benchmark scripts.
13. Add docs page and docs build gate.
14. Run default fast tests.
15. Run direct-coil forward example.
16. Run VMEC2000 comparison if executable is available.
17. Commit in logical chunks.
18. Push branch to origin.

## Risk Register

1. Full NESTOR path may remain partially NumPy-heavy.
   - Mitigation: phase 1 implements forward direct-coil sampling and dense toy adjoint; docs clearly mark production full-solve adjoint as phase 2.
2. Direct coil field sampling may be memory-heavy for large `npoints * ncoils * nsegments`.
   - Mitigation: chunked evaluation, benchmark matrix, and `lax.map(batch_size=...)`.
3. ESSOS object internals may shift.
   - Mitigation: adapter extracts minimal public attributes where possible and has shape tests.
4. Direct-coil versus mgrid values will not match at coarse mgrid resolution.
   - Mitigation: compare convergence with increasing mgrid resolution, not equality at one coarse grid.
5. Optimizer examples may be too slow for default CI.
   - Mitigation: separate smoke and full examples.
6. Full solve gradients may be misleading if only provider derivatives pass.
   - Mitigation: all gradient claims separated into provider-level, toy vacuum-solve-level, and full-solve-level gates.

## Acceptance Checklist

Minimum branch acceptance:

1. Existing free-boundary mgrid tests pass.
2. New pure-JAX coil provider matches ESSOS Biot-Savart on simple coils when ESSOS is installed.
3. JAX mgrid interpolation matches legacy mgrid interpolation on synthetic fields.
4. Direct-coil provider can drive one low-resolution free-boundary solve and write a `wout`.
5. VMEC2000 comparison script exists and runs locally when `RUN_VMEC2000=1`.
6. Coil-only free-boundary optimization example exists and does not use plasma boundary coefficients as optimization variables.
7. Gradient checks exist for coil currents, coil Fourier coefficients, and evaluation coordinates.
8. Vacuum solve custom-adjoint scaffold exists with dense toy tests.
9. Documentation states current differentiability status and limitations.

Stretch acceptance:

1. Validated full-solve gradient through a low-resolution free-boundary solve with coil current as the only variable, bounded against finite differences.
2. QS objective gradient wrt coil current or Fourier coefficient, validated by finite differences.
3. Robust 4-sample coil perturbation optimization run.
4. GPU benchmark for direct-coil field sampling.

## Progress Tracker

The per-WP status lines above are the original acceptance checklist. The
current branch state is summarized here and in the dated work log below; this
tracker is the authoritative plan snapshot for the free-boundary direct-coil
branch.

```text
WP0 Branch foundation and plan:                100%
WP1 Provider base API:                         100%
WP2 Pure JAX coil Biot-Savart:                 88%
WP3 ESSOS adapter:                             80%
WP4 JAX mgrid interpolation:                   85%
WP5 Free-boundary provider hook:               88%
WP6 Direct-coil forward example:               82%
WP7 Vacuum adjoint scaffold:                  100%
WP8 Gradient checks:                           91%
WP9 VMEC2000 diagnostics:                      62%
WP10 Benchmarks/diagnostics:                   78%
WP11 Coil-only QS optimization example:        45%
WP12 Robust coil perturbations:               100%
WP13 Documentation:                            84%
WP14 CI policy:                                64%
Overall branch completion:                     78%
```

## Immediate Next Steps

1. Continue the VMEC2000 generated-mgrid WOUT comparator until the optional xfail can be bounded or promoted.
2. Decide whether cached direct-coil geometry should be threaded into the free-boundary bridge after CPU/GPU benchmark evidence, without replacing the differentiable params-to-field API.
3. Replace the phase-1 coil-only optimization proxy with Boozer/QS residuals only after the direct-coil free-boundary loop has validated gradients.
4. Run CPU/GPU benchmark matrices and convert JSON summaries into documentation plots.
5. Implement the production matrix-free/custom-linear-solve NESTOR adjoint beyond the dense toy scaffold.
6. Re-check PR CI, including Codecov patch coverage, after each commit.

## Need From User

Nothing is required right now. The next implementation step can proceed locally. Later, maintainers should decide whether ESSOS mgrid export should be released before the `vmec_jax` example is promoted from research example to documented workflow.

## Work Log

### 2026-05-24 Cached geometry and robust coil optimization example

Steps taken:

1. Added `build_coil_field_geometry(...)`,
   `sample_coil_field_xyz_from_geometry(...)`, and
   `sample_coil_field_cylindrical_from_geometry(...)` to split direct-coil
   geometry construction from field sampling while preserving the original
   `sample_coil_field_cylindrical(params, ...)` API.
2. Added tests showing cached-geometry sampling equals the full sampler and
   functional gradients through geometry construction match the original path.
3. Extended `tools/benchmarks/bench_external_field_providers.py` with cached
   direct-coil geometry cases and separate geometry-build timing.
4. Added optional robust scenarios to
   `examples/optimization/free_boundary_QS_coil_optimization.py`, using
   `vmec_jax.robust_coils` perturbation samples and mean, mean-plus-std, or
   smooth-max aggregation.
5. Added bounded smoke coverage for the robust example path.

Results obtained:

1. `pytest -q tests/test_external_fields_coils_jax.py tests/test_external_fields_essos_adapter.py tests/test_free_boundary_qs_coil_optimization_smoke.py`
   passed: 17 passed in 12.41 s.
2. `python tools/benchmarks/bench_external_field_providers.py --points 16 --segments 32 --warm-repeats 2 --skip-essos --out results/bench_external_field_providers_cached_geometry_smoke_local.json`
   passed. Synthetic direct-coil field-only timing changed from cold
   `0.0808 s`, warm min `0.000028 s` to cached-geometry cold `0.0575 s`,
   warm min `0.000024 s`.
3. The robust example smoke with two perturbed scenarios passed in the worker
   run and writes scenario-level objective histories.

Best next steps:

1. Run larger CPU/GPU provider benchmarks with the cached free-boundary bridge
   enabled and use the active-NESTOR timing breakdown to target the next
   scan-trial and replay hot spots.
2. Add and validate a full-loop finite-difference stability smoke for a
   coil-current-only objective before promoting the phase-1 proxy toward
   Boozer/QS.
3. Keep robust full-solve scenarios as Python-loop examples until the production
   free-boundary path is batch-transformable.

Need from user:

Nothing now.

### 2026-05-24 Cached direct-coil provider bridge, docs, and coverage tests

Steps taken:

1. Added provider-static support for prebuilt direct-coil geometry in
   `sample_external_field_cylindrical(...)`.
2. Added an automatic host-driver cache for `direct_coils` free-boundary runs:
   `run_free_boundary(...)` now builds symmetry-expanded coil geometry once per
   run/stage and passes it through the provider-static slot. The original
   `CoilFieldParams -> field` API remains unchanged for differentiable
   provider-level tests.
3. Added fast tests for cached-provider dispatch, cached XYZ sampling with
   chunking, chunked current-gradient parity, and smooth-max robust-risk
   gradients.
4. Shortened the README free-boundary coil section and moved detailed caveats,
   robust smoke instructions, benchmark matrix guidance, and optional VMEC2000
   diagnostics into the docs page.
5. Added `tools/diagnostics/compare_freeb_coils_mgrid_vmec2000.py`, a
   standalone optional diagnostic that writes JSON for the current three-way
   path: `vmec_jax` generated-`mgrid`, `vmec_jax` direct coils, and VMEC2000
   generated-`mgrid` when available.

Results obtained:

1. `pytest -q tests/test_external_fields_coils_jax.py tests/test_robust_coil_perturbations.py`
   passed: 24 passed in 17.61 s.
2. `VMEC_JAX_TIMING=1 python tools/benchmarks/bench_freeb_direct_coil_solve.py --max-iter 2 --warm-repeats 2 ...`
   shows the active NESTOR direct-coil sample time improving from about
   `0.98 s` to `0.51 s` on the cold sample, and from about `11.3 ms` to
   `4.3 ms` warm. Total tiny-solve runtime improved modestly because
   preconditioner/residual work dominates this short benchmark.
3. `python -m sphinx -T -b html docs /tmp/vmec_jax_freeb_docs_after_cache`
   passed.
4. The standalone diagnostic smoke with `--skip-vmec2000` passes the
   `vmec_jax` direct-coil versus generated-`mgrid` WOUT comparison. The
   VMEC2000-enabled smoke currently records `vmec2000_status: no_wout` for the
   low-iteration LP-QA generated-`mgrid` case, matching the known optional
   parity gap while preserving debug tails and workdir paths in JSON.

Best next steps:

1. Run the same benchmark matrix on a GPU host and decide whether to add a
   jitted geometry sampler cache for the free-boundary bridge.
2. Continue VMEC2000 generated-mgrid diagnostic scripting so the optional xfail
   emits actionable JSON outside pytest.
3. Add full-loop finite-difference checks for coil-current-only objectives
   before promoting the coil-only optimization example beyond phase-1 proxy
   status.

Need from user:

Nothing now.

### 2026-05-24 Accepted-state sensitivity gate, WOUT comparator, and synchronized benchmarks

Steps taken:

1. Added accepted-state vector summaries and reference deltas to
   `tools/diagnostics/freeb_direct_provider_sensitivity.py`.
2. Promoted the optional ESSOS finite-pressure accepted-state sensitivity gate
   from an expected xfail to a bounded `100x` current-scale test. This remains
   a sensitivity gate, not a convergence claim.
3. Refactored the optional VMEC2000 generated-mgrid comparison to compare
   converged WOUT-level quantities first instead of VMEC-JAX accepted final
   residual components against the last printed VMEC2000 trace row.
4. Added recursive `block_until_ready` synchronization and solver timing
   snapshots to `tools/benchmarks/bench_freeb_direct_coil_solve.py`, so GPU
   benchmark timings include queued JAX work.
5. Updated README/docs with the matched `--coil-current-scale` beta-scan
   command, robust-coil utilities, benchmark commands, and finite-pressure
   direct-coil limitations.

Results obtained:

1. The direct-coil sensitivity diagnostic now reports accepted-state RMS and
   max deltas from the current-scale reference. A `1x` versus `100x` LP-QA
   smoke gives relative accepted-state RMS delta about `1.43e-8` in the
   diagnostic script.
2. The optional `RUN_FULL` ESSOS sensitivity gate passes at `100x` current
   scale in the bounded local harness.
3. The synchronized direct-coil solve benchmark smoke reports cold solve time
   `6.270 s` and warm solve time `0.202 s` for the two-iteration synthetic CPU
   case. The second iteration enters the active direct-coil NESTOR path and the
   JSON includes internal solver timing histories.
4. The optional VMEC2000 comparator is still marked xfail until generated-mgrid
   VMEC2000 WOUT parity is bounded, but it now targets scientifically meaningful
   end-state quantities instead of brittle trace rows.

Best next steps:

1. Add differentiable direct-coil geometry precompute/reuse helpers and use
   them first in the field-provider benchmark, then in the free-boundary bridge
   if gradients are preserved.
2. Run the optional VMEC2000 generated-mgrid WOUT comparator with the ESSOS
   mgrid PR on `PYTHONPATH` and tune thresholds only if the WOUT quantities show
   bounded parity.
3. Run synchronized CPU/GPU benchmark matrices before attempting provider-path
   caching or device-resident NESTOR handoff.

Need from user:

Nothing now.

### 2026-05-24 Direct-provider trial refresh, robust utilities, benchmarks, and phase-1 optimization scaffold

Steps taken:

1. Fixed direct-provider NESTOR reuse so non-mgrid providers refresh `gsource` and nonsingular mode vectors on reuse steps instead of using stale mgrid-style cached sources.
2. Added non-mutating trial-state vacuum refresh for direct providers during sign probes, backtracking, and direct fallback scoring. Mgrid runs keep the committed VMEC ivac/ivacskip cadence.
3. Added `vmec_jax.robust_coils` with deterministic current, displacement, toroidal-phase, and Fourier-centerline perturbations plus robust risk aggregation.
4. Added `tests/test_robust_coil_perturbations.py`.
5. Added lightweight benchmark scripts:
   - `tools/benchmarks/bench_external_field_providers.py`,
   - `tools/benchmarks/bench_freeb_direct_coil_solve.py`,
   - `tools/benchmarks/bench_freeb_coil_gradient.py`.
6. Added `examples/optimization/free_boundary_QS_coil_optimization.py`, a phase-1 coil-only direct-coil free-boundary optimization smoke that never optimizes plasma boundary coefficients.
7. Documented the phase-1 optimization smoke, robust utilities, and benchmark commands in `docs/free_boundary_coil_optimization.rst` and added the new modules to the API autosummary.

Results obtained:

1. Direct-provider source-refresh regression passed:
   - `pytest -q tests/test_free_boundary_direct_coil_finite_pressure_sensitivity.py`: 3 passed, 1 skipped in 6.65 s.
2. Broader provider/free-boundary subset passed:
   - `PYTHONPATH=/Users/rogeriojorge/local/ESSOS_mgrid_pr:$PYTHONPATH pytest -q tests/test_external_fields_coils_jax.py tests/test_external_fields_essos_adapter.py tests/test_external_fields_mgrid_jax.py tests/test_free_boundary_vacuum_adjoint.py tests/test_free_boundary_coil_provider_forward.py tests/test_free_boundary_direct_coil_finite_pressure_sensitivity.py tests/test_free_boundary_essos_coil_parity.py::test_essos_direct_coil_free_boundary_matches_generated_mgrid_backend`: 29 passed, 1 skipped in 42.21 s.
3. Robust utilities passed:
   - `pytest -q tests/test_robust_coil_perturbations.py`: 9 passed in 4.82 s.
4. Benchmark smokes passed:
   - external-field provider smoke wrote `results/bench_external_field_providers_smoke.json`, synthetic direct coil cold/compile `0.0725 s`, warm min `0.000038 s`;
   - coil-gradient smoke wrote `results/bench_freeb_coil_gradient_smoke.json`, direct-coil value/grad cold/compile `0.172 s`, warm min `0.000042 s`;
   - direct free-boundary solve smoke wrote `results/bench_freeb_direct_coil_solve_smoke.json`, synthetic solve cold `4.50 s`.
5. Phase-1 coil-only smoke passed:
   - objective `0.400854`,
   - residual proxy `0.3926`,
   - aspect `6.0827`,
   - mean iota `0.4906`,
   - outputs in `results/free_boundary_QS_coil_optimization_circle_smoke`.
6. Optional full-solve ESSOS accepted-state sensitivity still xfails with the
   default LP-QA current scale. Direct NESTOR/source diagnostics respond
   correctly, but the accepted state barely moves because the fixture external
   field is weak at this low resolution.
7. Strong-current diagnostics show measurable accepted-state response only at
   very large current multipliers for the short LP-QA smoke:
   - 100x current changes the final aspect by only about `4e-7`;
   - 10000x current changes the final aspect by about `3.7e-3`.
8. Added `--coil-current-scale` to `examples/free_boundary_essos_coils_beta_scan.py`
   so matched direct/mgrid finite-pressure sensitivity scans can use scaled
   ESSOS coils while preserving the default fixture exactly.

Best next steps:

1. Run matched direct/mgrid beta scans with explicit `--coil-current-scale` and
   pick a physically meaningful finite-pressure sensitivity scale for the LP-QA
   smoke.
2. Promote direct-coil finite-pressure accepted-equilibrium sensitivity only
   when finite differences show stable, non-stale response at that scale.
3. Run the VMEC2000 optional comparison and keep it xfailed unless the generated-mgrid parity gap is bounded.
4. Replace the phase-1 optimization proxy with Boozer/QS only after full-loop gradients are validated.

Need from user:

Nothing now.

### 2026-05-24 Accepted-state active residual recompute

Steps taken:

1. Added a final active NESTOR resample on the accepted final state for free-boundary runs with active edge coupling.
2. Recomputed the reported final residuals from the accepted state and fresh active vacuum sample instead of reporting the last pre-update residuals.
3. Added diagnostics for both recomputed final residuals and previous pre-update final residuals:
   - `final_residual_recomputed_on_accepted_state`,
   - `pre_update_final_fsqr`,
   - `pre_update_final_fsqz`,
   - `pre_update_final_fsql`.
4. Regenerated the README/docs beta-scan figures and CSV summary using accepted-state residuals.
5. Updated the free-boundary coil optimization docs to state that the active smoke residuals are accepted-state recomputes, but still not converged high-beta results.

Results obtained:

1. Forced-active direct-coil smoke residual reporting now drops from the stale pre-update scale to the accepted-state recompute scale.
2. The README beta-scan residual norm changed from about `7.25` to about `2.97` after accepted-state recompute.
3. Direct-coil and generated-mgrid providers still agree to recorded precision for the low-resolution active smoke.
4. Verification passed:
   - `pytest -q tests/test_free_boundary_direct_coil_finite_pressure_sensitivity.py tests/test_free_boundary_coil_provider_forward.py tests/test_free_boundary_essos_coil_parity.py::test_essos_direct_coil_free_boundary_matches_generated_mgrid_backend`: 6 passed, 1 skipped in 29.83 s.
   - `pytest -q tests/test_free_boundary_direct_coil_finite_pressure_sensitivity.py tests/test_free_boundary_coil_provider_forward.py tests/test_free_boundary_vacuum_adjoint.py`: 10 passed, 1 skipped in 15.51 s.
   - `python -m sphinx -T -b html docs /tmp/vmec_jax_freeb_docs`: passed.
   - `ruff check vmec_jax/solve.py`: passed.
   - `git diff --check`: passed.

Best next steps:

1. Fix direct-provider source/runtime refresh so active accepted-state equilibria show bounded sensitivity to coil current and geometry, not just isolated NESTOR-step sensitivity.
2. Instrument scan/trial timing and cold exact/direct-provider costs.
3. Add robust-coil utilities and benchmark scripts in parallel with the direct-provider sensitivity fix.

Need from user:

Nothing now.

### 2026-05-24 Active finite-pressure direct-coil diagnostics

Steps taken:

1. Added explicit `free_boundary_activate_fsq` plumbing through `run_free_boundary`/`run_fixed_boundary` into the VMEC-style free-boundary cadence. This keeps literal VMEC2000 parity as the default while allowing short research examples/tests to force active vacuum coupling without hidden environment variables.
2. Added `NestorSolveResult.diagnostics` and propagated `free_boundary.last_nestor_diagnostics` into solve diagnostics. The diagnostics record provider kind, normal-field source magnitudes, RHS/source norms, and coupled `bsqvac` magnitudes.
3. Corrected the run-level `free_boundary.vacuum_stub` diagnostic so it is `False` when an active NESTOR-like model actually ran.
4. Added `tests/test_free_boundary_direct_coil_finite_pressure_sensitivity.py`, covering active NESTOR-step sensitivity to direct-coil current and explicit forced-activation diagnostics. The optional full-solve ESSOS sensitivity guard remains `RUN_FULL`/xfail until accepted-state sensitivity is fixed.
5. Added `tools/diagnostics/freeb_direct_provider_sensitivity.py` for current-scale and geometry-perturbation sweeps with JSON summaries.
6. Updated `examples/free_boundary_essos_coils_beta_scan.py` with explicit `--activate-fsq` and extra active NESTOR summary channels.
7. Regenerated `docs/_static/figures/freeb_single_stage_beta_scan.png`, `docs/_static/figures/freeb_single_stage_provider_parity.png`, and `docs/_static/figures/freeb_single_stage_beta_scan_summary.csv`.
8. Fixed the beta-scan renderer y-limits so active residual/aspect/iota values are visible instead of clipped by the previous inactive-smoke ranges.

Results obtained:

1. The isolated active NESTOR bridge is sensitive to direct-coil current: normal-field/source channels scale linearly with current and `bsqvac` scales quadratically.
2. The active finite-pressure ESSOS beta scan now reports `ivac=3`, `nestor_model=vmec2000_like_dense_integral`, and `vacuum_stub=False`.
3. Direct-coil and generated-mgrid providers still agree for the same active finite-pressure path to recorded precision in the low-resolution scan.
4. The active residual norm is still large (`~7.25`), so this remains provider/coupling validation, not a converged finite-beta optimization result.
5. Tests passed:
   - `pytest -q tests/test_free_boundary_direct_coil_finite_pressure_sensitivity.py tests/test_free_boundary_coil_provider_forward.py tests/test_free_boundary_vacuum_adjoint.py`: 10 passed, 1 skipped in 16.13 s.
   - `PYTHONPATH=/Users/rogeriojorge/local/ESSOS_mgrid_pr:$PYTHONPATH pytest -q tests/test_external_fields_coils_jax.py tests/test_external_fields_essos_adapter.py tests/test_external_fields_mgrid_jax.py tests/test_free_boundary_vacuum_adjoint.py tests/test_free_boundary_coil_provider_forward.py tests/test_free_boundary_direct_coil_finite_pressure_sensitivity.py tests/test_free_boundary_essos_coil_parity.py::test_essos_direct_coil_free_boundary_matches_generated_mgrid_backend`: 28 passed, 1 skipped in 42.63 s.
6. Docs build passed: `python -m sphinx -T -b html docs /tmp/vmec_jax_freeb_docs`.

Best next steps:

1. Recompute final accepted-state residuals and active NESTOR diagnostics after the last accepted update; current `final_fsqr/final_fsqz/final_fsql` are last pre-update values.
2. Refresh NESTOR sampling for trial/accepted states, so accepted-state sensitivity to coil changes is measured against the updated boundary rather than a stale pre-trial boundary.
3. Promote the optional full-solve direct-coil sensitivity xfail to a passing gate once accepted-state sensitivity is bounded.
4. Only then add the first coil-only single-stage QS optimization example.

Need from user:

Nothing now.

### 2026-05-24 Finite-pressure free-boundary correction

Steps taken:

1. Audited the README beta-scan summary and confirmed the first documentation slice showed nonzero pressure profiles only indirectly.
2. Added explicit pressure and energy diagnostics to `examples/free_boundary_essos_coils_beta_scan.py`:
   - input `pressure_scale`,
   - `wp`,
   - `wb`,
   - `beta_proxy = W_p / W_B`,
   - `beta_proxy_percent = 100 W_p / W_B`.
3. Updated the README/docs plot renderer to show `PRES_SCALE` and `100 W_p/W_B` directly.
4. Updated the README and docs wording to state that zero pressure is only a reference point and finite-pressure points are the meaningful free-boundary check.
5. Changed direct-coil free-boundary provider tests to use nonzero pressure instead of validating only a vacuum case.
6. Attempted the first coil-only current/geometry optimization smoke and found that accepted equilibria were not sensitive to direct-coil parameter changes under the current free-boundary cadence.
7. Audited run diagnostics and found the short README scan ends with `ivac=-1`, `nestor_model=none`, and `vacuum_stub=True`; forced turn-on enters the dense NESTOR-like path but finite-pressure residuals are not yet bounded.

Results obtained:

1. The finite-pressure scan still shows exact recorded scalar parity between generated-mgrid and direct-coil provider plumbing.
2. The finite-pressure test subset passed: `4 passed in 23.79 s`.
3. The local docs build passed.
4. A trial near actual `1%` beta proxy was too aggressive for this low-resolution smoke and produced large residuals.
5. The current direct-coil branch must not claim coil-only single-stage optimization yet: active vacuum coupling needs to respond robustly to direct-coil parameter changes first.

Best next steps:

1. Fix active finite-pressure NESTOR/free-boundary coupling so direct-coil current/geometry changes alter the accepted equilibrium.
2. Add a regression test that fails if a direct-coil parameter perturbation leaves the active free-boundary solve unchanged.
3. Only then add the first coil-only optimization example; require nonzero pressure and non-stub active vacuum coupling.
4. Investigate the high-pressure residual blow-up before making any 1% beta performance or physics claim.

Need from user:

Nothing now.

### 2026-05-24 README/docs visualization slice

Steps taken:

1. Ran the ESSOS Landreman-Paul QA four-point beta scan with generated-mgrid and direct-coil providers:
   `PYTHONPATH=/Users/rogeriojorge/local/ESSOS_mgrid_pr:$PYTHONPATH python examples/free_boundary_essos_coils_beta_scan.py --outdir results/free_boundary_essos_coils_beta_scan_readme`.
2. Added `tools/diagnostics/render_freeb_single_stage_readme.py` to render reviewer-facing figures from the JSON summary.
3. Generated:
   - `docs/_static/figures/freeb_single_stage_architecture.png`
   - `docs/_static/figures/freeb_single_stage_beta_scan.png`
   - `docs/_static/figures/freeb_single_stage_provider_parity.png`
   - `docs/_static/figures/freeb_single_stage_beta_scan_summary.csv`
4. Added `docs/free_boundary_coil_optimization.rst` and linked it from `docs/index.rst`.
5. Added a README section for the direct-coil single-stage free-boundary lane.

Results obtained:

1. The mgrid and direct-coil `vmec_jax` providers produced identical recorded scalar diagnostics in the low-resolution beta scan.
2. The first mgrid point includes cold-start overhead; subsequent direct/mgrid timings are about 1.26 s per case for this smoke setting.
3. The documentation now separates implemented provider-level differentiability from the phase-2 production free-boundary/NESTOR adjoint.

Best next steps:

1. Run the docs build and the direct-coil fast tests after this documentation slice.
2. Commit and push this README/docs visualization update.
3. Start WP11, the coil-only optimization example, because the architecture and forward beta-scan evidence are now visible.

Need from user:

Nothing now.

### 2026-05-24

Steps taken:

1. Created fresh clone at `/Users/rogeriojorge/local/vmec_jax_freeb`.
2. Created branch `feature/freeb-essos-coil-single-stage` from current `main`.
3. Inspected current `vmec_jax` free-boundary, optimization, docs, and tests.
4. Inspected local ESSOS coil, field, objective, perturbation, and example files.
5. Completed literature/documentation pass covering JAX implicit differentiation, pytrees, checkpointing, chunked mapping, SIMSOPT free-boundary/mgrid/coils, single-stage stellarator optimization, DESC JAX equilibrium optimization, spectral PDE adjoints, Lineax, JAXopt, and CVaR-style robust risk.
6. Created this plan and branch log.

Results obtained:

1. Existing mgrid path entry point identified: `_sample_external_boundary_arrays`.
2. ESSOS Fourier and symmetry conventions identified.
3. Provider abstraction and work packages defined.
4. Full initial branch roadmap written.

Best next steps:

1. Implement provider base API.
2. Implement pure JAX direct-coil provider and tests.
3. Add optional ESSOS parity adapter.

Need from user:

Nothing now.

### 2026-05-24 Free-boundary provider bridge

Steps taken:

1. Added `sample_free_boundary_external_field(...)` in `free_boundary.py`.
2. The helper samples any external-field provider at boundary arrays and projects the result into the existing `ExternalBoundarySample` / `VacuumBoundaryFields` data model.
3. Added optional axis-field addition so direct-coil fields and axis-current fields remain separable in diagnostics.
4. Exported the helper through `vmec_jax.__init__`.
5. Added tests for direct-coil provider projection and axis-field separation.

Results obtained:

1. `pytest -q tests/test_free_boundary_coil_provider_forward.py` passed: 2 passed in 2.17 s.
2. `pytest -q tests/test_external_fields_coils_jax.py tests/test_external_fields_essos_adapter.py tests/test_external_fields_mgrid_jax.py tests/test_free_boundary_vacuum_adjoint.py tests/test_free_boundary_coil_provider_forward.py` passed: 24 passed in 15.04 s.

Best next steps:

1. Commit the provider bridge.
2. Refactor the state/static free-boundary sampler so it can call the provider bridge after constructing boundary geometry.
3. Add a low-resolution direct-coil free-boundary forward example once the state/static hook is available.

Need from user:

Nothing now.

### 2026-05-24 ESSOS mgrid export and LP-QA beta scan

Steps taken:

1. Created a clean ESSOS PR clone at `/Users/rogeriojorge/local/ESSOS_mgrid_pr` on branch `feature/mgrid-from-coils`, leaving the dirty `/Users/rogeriojorge/local/ESSOS` checkout untouched.
2. Added ESSOS `essos.mgrid.MGrid` and `coils_to_mgrid(...)`, mirroring SIMSOPT's cylindrical grid layout and VMEC NetCDF variable names.
3. Added `Coils.to_mgrid(...)` in ESSOS and tests for read/write roundtrip, ESSOS Landreman-Paul QA coil export, and SIMSOPT mgrid parity.
4. Extended `vmec_jax` free-boundary runtime plumbing so Python callers can supply a non-mgrid external-field provider while the legacy mgrid/CLI path remains unchanged.
5. Added `examples/free_boundary_essos_coils_beta_scan.py`, which loads ESSOS Landreman-Paul QA coils, writes an mgrid, runs a four-point nominal beta scan through the mgrid backend, and runs the same scan through the direct differentiable coil provider.

Results obtained:

1. ESSOS test command `pytest -q tests/test_mgrid.py` passed: 4 passed in 2.72 s, including SIMSOPT parity.
2. `vmec_jax` compile command passed for the new example and modified solver/free-boundary modules.
3. `vmec_jax` provider test command `pytest -q tests/test_external_fields_coils_jax.py tests/test_external_fields_essos_adapter.py tests/test_external_fields_mgrid_jax.py tests/test_free_boundary_vacuum_adjoint.py tests/test_free_boundary_coil_provider_forward.py` passed: 24 passed in 12.40 s.
4. Smoke example command wrote `/tmp/vmec_jax_freeb_beta_smoke/summary.json` and four wout files for `beta=0` and `beta=1` with both `mgrid` and `direct` backends.

Best next steps:

1. Push the ESSOS mgrid branch and open a PR.
2. Push the `vmec_jax` feature branch.
3. Add VMEC2000 comparison diagnostics for the generated mgrid/direct-coil cases.
4. Add the first coil-only QS optimization example.

Need from user:

Nothing now.

### 2026-05-24 Optional three-way free-boundary parity gate

Steps taken:

1. Added `tests/test_free_boundary_essos_coil_parity.py`.
2. The default/ESSOS-enabled test builds an ESSOS Landreman-Paul QA mgrid, runs `vmec_jax` free-boundary through the mgrid backend, runs the same case through the direct differentiable coil backend, writes both wouts, and verifies matching `rmnc`, `zmns`, `lmns`, `iotas`, `iotaf`, aspect, and magnetic energy.
3. Added an optional `VMEC2000_INTEGRATION=1` test that runs local `xvmec2000` on the generated mgrid and compares against the two `vmec_jax` paths.
4. Marked the VMEC2000 generated-mgrid comparison as `xfail` for now because the local VMEC2000 executable reads the generated mgrid and produces traces, but the current `vmec_jax` free-boundary trace is not yet bounded against VMEC2000 for this generated-coil case.
5. Checked ESSOS PR CI and fixed unrelated current-JAX breakages in the ESSOS PR branch: `jax.jax.tree_util` removal, `jnp.clip(a_min=...)` removal, older `jaxopt` `jax.tree_map` usage, and exact float equality in a test.

Results obtained:

1. `PYTHONPATH=/Users/rogeriojorge/local/ESSOS_mgrid_pr:$PYTHONPATH pytest -q tests/test_free_boundary_essos_coil_parity.py::test_essos_direct_coil_free_boundary_matches_generated_mgrid_backend tests/test_free_boundary_essos_coil_parity.py::test_vmec2000_generated_mgrid_free_boundary_matches_vmec_jax_and_direct_coils` passed with one skipped VMEC2000 gate when `VMEC2000_INTEGRATION` is unset.
2. `VMEC2000_INTEGRATION=1 ... pytest -q tests/test_free_boundary_essos_coil_parity.py::test_vmec2000_generated_mgrid_free_boundary_matches_vmec_jax_and_direct_coils -rx` reports the expected xfail.
3. `PYTHONPATH=/Users/rogeriojorge/local/ESSOS_mgrid_pr:$PYTHONPATH pytest -q tests/test_external_fields_coils_jax.py tests/test_external_fields_essos_adapter.py tests/test_external_fields_mgrid_jax.py tests/test_free_boundary_vacuum_adjoint.py tests/test_free_boundary_coil_provider_forward.py tests/test_free_boundary_essos_coil_parity.py::test_essos_direct_coil_free_boundary_matches_generated_mgrid_backend` passed: 25 passed in 26.19 s.
4. In ESSOS, `pytest -q` passed: 96 passed in 29.09 s.

Best next steps:

1. Fix the generated-mgrid VMEC2000 parity gap by comparing the VMEC2000 NESTOR sampling/projection path against `vmec_jax` on the same boundary after the same first accepted free-boundary update.
2. Promote the optional xfail to a passing VMEC2000 integration gate once traces and wout output are bounded.
3. Add the first coil-only QS optimization example.

Need from user:

Nothing now.

### 2026-05-24 Vacuum adjoint scaffold

Steps taken:

1. Added `vmec_jax/free_boundary_adjoint.py`.
2. Implemented `dense_vacuum_solve_jax(A, b, symmetric=False)` with `jax.lax.custom_linear_solve`.
3. Added `dense_vacuum_residual(A, x, b)` for diagnostics/tests.
4. Added dense toy tests covering primal solve parity, VJP wrt RHS, finite-difference gradient wrt RHS parameter, finite-difference gradient wrt matrix parameter, and symmetric transpose-solve behavior.

Results obtained:

1. `pytest -q tests/test_free_boundary_vacuum_adjoint.py` passed: 5 passed in 1.29 s.
2. `pytest -q tests/test_external_fields_coils_jax.py tests/test_external_fields_essos_adapter.py tests/test_external_fields_mgrid_jax.py tests/test_free_boundary_vacuum_adjoint.py` passed: 22 passed in 13.98 s.

Best next steps:

1. Commit the vacuum adjoint scaffold.
2. Begin the free-boundary provider hook by adding an internal provider-sampling function that returns the existing `ExternalBoundarySample` shape.
3. Keep the legacy mgrid call path unchanged until provider tests prove equivalent boundary samples.

Need from user:

Nothing now.

### 2026-05-24 Provider slice 2

Steps taken:

1. Added `vmec_jax.external_fields.mgrid_jax`.
2. Implemented `MGridFieldParams` as a pytree with differentiable field arrays and `extcur`.
3. Implemented `interpolate_mgrid_bfield_jax` and `sample_mgrid_field_cylindrical`.
4. Added mgrid dispatch support through `sample_external_field_cylindrical("mgrid", ...)`.
5. Added synthetic affine-field tests comparing:
   - exact affine values,
   - legacy NumPy `interpolate_mgrid_bfield`,
   - JAX mgrid dispatch,
   - gradients with respect to `extcur`,
   - gradients with respect to field values,
   - coordinate derivatives away from grid-cell boundaries.

Results obtained:

1. `pytest -q tests/test_external_fields_mgrid_jax.py` passed: 4 passed in 3.22 s.
2. `pytest -q tests/test_external_fields_coils_jax.py tests/test_external_fields_essos_adapter.py tests/test_external_fields_mgrid_jax.py` passed: 17 passed in 13.16 s.

Best next steps:

1. Commit provider slice 2.
2. Add the dense vacuum adjoint scaffold and tests.
3. Start the free-boundary provider hook after the adjoint scaffold is in place.

Need from user:

Nothing now.

### 2026-05-24 Provider slice 1

Steps taken:

1. Committed this roadmap as `2b50319 docs: add free-boundary coil optimization plan`.
2. Added `vmec_jax.external_fields` package.
3. Added the provider dispatch API in `external_fields/base.py`.
4. Added pure JAX Fourier coil evaluation and Biot-Savart sampling in `external_fields/coils_jax.py`.
5. Matched ESSOS Fourier convention and Biot-Savart scaling.
6. Added symmetry expansion, chunked point evaluation, coil length, curvature, current norm, soft coil-plasma distance, soft coil-coil distance, and smooth length/curvature penalties.
7. Added optional ESSOS adapter in `external_fields/essos_adapter.py`.
8. Added tests for geometry, analytic on-axis Biot-Savart, provider dispatch, chunking, current gradients, Fourier coefficient gradients, coordinate derivatives, symmetry ordering, engineering metrics, and ESSOS parity.

Results obtained:

1. `pytest -q tests/test_external_fields_coils_jax.py tests/test_external_fields_essos_adapter.py` passed: 13 passed in 10.54 s.
2. ESSOS was importable locally, so the optional ESSOS Biot-Savart parity test ran instead of skipping.

Best next steps:

1. Commit provider slice 1.
2. Implement JAX mgrid interpolation and synthetic gradient tests.
3. Start provider hook design in `free_boundary.py` while preserving the existing mgrid call path.

Need from user:

Nothing now.
