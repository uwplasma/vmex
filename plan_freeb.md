# Free-Boundary Coil-Aware Single-Stage Optimization Evidence Summary

Status: closed evidence summary, not an active work plan.

The active umbrella plan is `plan_differentiability.md`.  Historical
free-boundary progress logs were intentionally compacted on 2026-06-20 to keep
the repository lightweight and to avoid two competing task lists.  The full
append-only transcript remains available through git history before commit
`2d300c5e` and the later compaction commit.

## Scope

This file summarizes the evidence ladder for direct-coil/free-boundary
single-stage optimization work:

1. Direct-coil and JAX `mgrid` external-field providers.
2. Free-boundary coupling through VMEC/NESTOR-compatible interfaces.
3. Branch-local accepted-boundary replay and same-fingerprint custom-VJP gates.
4. Coil-only QS optimization examples where branch-local derivatives propose
   steps and complete free-boundary solves remain acceptance authority.
5. Bounded VMEC2000/mgrid/direct-coil parity fixtures with finite-positive
   physical WOUTs.

## Current Claim Policy

Promoted:

- Pure-JAX direct-coil field sampling and coil perturbation utilities.
- JAX `mgrid` interpolation compatibility path.
- Direct-coil provider forward free-boundary solves at low resolution.
- Accepted-boundary replay under fixed/same branch fingerprints.
- Same-branch direct-coil JVP/custom-VJP gates that pass complete-solve
  central finite differences.
- Coil-only optimization examples that use branch-local derivative proposals
  while complete solves decide acceptance.

Not promoted:

- Arbitrary differentiation through host adaptive branch changes.
- A production full nonlinear `run_free_boundary` exact adjoint that is valid
  across changed controller branches, resets, rejected slots, activation
  cadence, limiters, and preconditioner policy switches.
- Unbounded generated-`mgrid` VMEC2000 comparisons that drive the boundary
  outside the generated field domain or into nonphysical `R <= 0` geometry.

## Implemented Source Areas

- `vmec_jax/external_fields/coils_jax.py`
- `vmec_jax/external_fields/mgrid_jax.py`
- `vmec_jax/external_fields/essos_adapter.py`
- `vmec_jax/free_boundary.py`
- `vmec_jax/free_boundary_adjoint.py`
- `vmec_jax/free_boundary_adjoint_controller.py`
- `vmec_jax/robust_coils.py`
- `vmec_jax/solvers/free_boundary/`
- `examples/free_boundary_essos_coils_forward.py`
- `examples/free_boundary_essos_direct_forward.py`
- `examples/free_boundary_essos_mgrid_forward.py`
- `examples/free_boundary_essos_coils_beta_scan.py`
- `examples/optimization/free_boundary_QS_coil_optimization.py`
- `examples/optimization/free_boundary_QA_finite_beta_coil_optimization.py`

## Core Validation Tests

- `tests/test_external_fields_coils_jax.py`
- `tests/test_external_fields_mgrid_jax.py`
- `tests/test_external_fields_essos_adapter.py`
- `tests/test_free_boundary_coil_provider_forward.py`
- `tests/test_free_boundary_coil_provider_gradients.py`
- `tests/test_free_boundary_direct_coil_finite_pressure_sensitivity.py`
- `tests/test_free_boundary_essos_coil_parity.py`
- `tests/test_free_boundary_vacuum_adjoint.py`
- `tests/test_free_boundary_qs_coil_optimization_smoke.py`
- `tests/test_free_boundary_qa_finite_beta_coil_optimization_smoke.py`
- `tests/test_robust_coil_perturbations.py`
- `tests/test_free_boundary_beta_response_validation.py`
- Optional VMEC2000 gates in `tests/test_vmec2000_exec_fast_validation.py`
  when `VMEC2000_INTEGRATION=1`.

## Evidence Summary

- Direct-coil/free-boundary phase 1: complete.
- Full nonlinear free-boundary adjoint phase 2: complete for branch-local,
  same-fingerprint accepted/rejected replay/controller gates that pass
  complete-solve central finite differences; arbitrary adaptive host-branch
  differentiation remains a deferred research lane.
- VMEC parity and physics gates: bounded finite-positive fixtures are promoted;
  unbounded/free-boundary generated-grid comparisons remain diagnostic only.
- Single-stage coil-only optimization phase 3: current examples demonstrate
  derivative-assisted proposals with complete-solve acceptance authority.
- CPU/GPU performance: instrumentation and matrix-free/scalar-report paths are
  in place; late performance changes should be driven by profile evidence and
  must preserve parity gates.
- Docs/release hygiene: user-facing docs must keep the same conservative claim
  policy as this file.

## Representative Local Gates

Use focused gates for development, not continuous CI polling:

```bash
JAX_ENABLE_X64=1 python -m pytest -q \
  tests/test_external_fields_coils_jax.py \
  tests/test_external_fields_mgrid_jax.py \
  tests/test_free_boundary_coil_provider_gradients.py

JAX_ENABLE_X64=1 python -m pytest -q \
  tests/test_free_boundary_vacuum_adjoint.py \
  tests/test_free_boundary_direct_coil_finite_pressure_sensitivity.py

JAX_ENABLE_X64=1 python -m pytest -q \
  tests/test_free_boundary_qs_coil_optimization_smoke.py \
  tests/test_free_boundary_qa_finite_beta_coil_optimization_smoke.py
```

Optional local parity gates:

```bash
VMEC2000_INTEGRATION=1 JAX_ENABLE_X64=1 python -m pytest -q \
  tests/test_vmec2000_exec_fast_validation.py

VMEC2000_INTEGRATION=1 JAX_ENABLE_X64=1 python -m pytest -q \
  tests/test_free_boundary_essos_coil_parity.py
```

## Review Guardrails

- Keep future progress and open decisions in `plan_differentiability.md`.
- Do not add a second active free-boundary plan.
- Keep complete solves as acceptance authority in coil-only optimization
  examples unless a stronger full adaptive differentiability gate is promoted.
- Keep docs explicit that branch-local/fingerprint-gated evidence is not the
  same as arbitrary adaptive host-branch differentiation.
- Add new parity fixtures only when they produce finite-positive physical WOUTs.
- Do not reintroduce generated WOUTs, BOOZ files, `mgrid` dumps, or solver
  traces as tracked repository artifacts.

## Completion Snapshot

- Direct-coil/free-boundary phase 1: `100%`.
- Full nonlinear free-boundary adjoint phase 2: `100%` for branch-local
  same-fingerprint evidence; adaptive branch differentiation remains deferred.
- VMEC parity and physics gates: `99.9%+` for bounded promoted fixtures.
- Single-stage coil-only optimization phase 3: `100%` for the current example
  contract.
- CPU/GPU performance evidence: `99.6%` for the current instrumentation and
  report paths.
- Docs/release hygiene: `100%` for this closed evidence summary.
