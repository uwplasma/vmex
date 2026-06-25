# Historical Discrete-Adjoint Recovery Plan Summary

Status: historical reference, not an active plan.

The active plan for current differentiability/refactor work is
`plan_differentiability.md`.  This file used to contain the April 2026
discrete-adjoint recovery plan for the QH fixed-boundary benchmark inspired by
Skene & Burns, "Fast automated adjoints for spectral PDE solvers"
(`arXiv:2506.14792`).  It was compacted on 2026-06-20 so the repository keeps
one active plan.  The full historical text remains available in git history.

## Preserved Historical Context

The old plan established these useful design conclusions:

- Differentiate the accepted solver trajectory, not an unrelated continuous
  surrogate.
- Use solver-owned tape/replay structure and branch fingerprints.
- Validate each promoted derivative with local Taylor/JVP/VJP checks and
  central finite differences.
- Keep exact/replay runtime visible at each layer.
- Avoid claiming differentiation through adaptive branch changes unless the
  branch is fingerprint-gated and validated.

Those conclusions are now incorporated into:

- `vmec_jax/discrete_adjoint.py`
- `vmec_jax/solvers/fixed_boundary/adjoint/`
- `vmec_jax/optimizers/fixed_boundary/exact_replay.py`
- `vmec_jax/solvers/free_boundary/adjoint/`
- `docs/discrete_adjoint.rst`
- `docs/performance.rst`
- `plan_differentiability.md`

## Current Replacement

Use `plan_differentiability.md` for current differentiability claims, promotion
gates, open lanes, and review-readiness.  Use `docs/discrete_adjoint.rst` for
user-facing explanation of the implemented fixed-boundary discrete-adjoint and
replay paths.

## Review Guardrails

- Do not add new entries here.
- Keep adaptive host-branch differentiation unclaimed unless a true
  fingerprint-gated full adaptive AD-vs-central-FD gate is implemented.
- If historical details are needed, inspect git history and summarize only the
  current consequence in `plan_differentiability.md`.
