# Historical VMEC-JAX Roadmap Summary

Status: historical reference, not an active plan.

The active plan for current development is `vmec_jax_plan/plan_differentiability.md`.  This
file used to contain the release-roadmap transcript through early June 2026; it
was compacted on 2026-06-20 so the repository has one active plan and no stale
parallel task lists.  The full historical text remains available in git history.

## Preserved Historical Context

The old roadmap covered:

- VMEC2000/VMEC++ parity as the primary correctness gate.
- Required CI runtime reduction while preserving a 95% combined coverage gate.
- DMerc and Glasser `D_R` AD-vs-central-FD derivative gates.
- Fixed-boundary and free-boundary CLI/API parity.
- QA/QH/QP/QI fixed-boundary optimization examples and seed-robust QI work.
- GPU/CPU performance instrumentation and exact replay cache reductions.
- VMEC profile support for polynomial and tabulated pressure/current/iota
  profiles.
- Repository-size hygiene and release-readiness gates.

## Current Replacement

Use:

- `vmec_jax_plan/plan_differentiability.md` for current open lanes, completion percentages,
  differentiability promotion gates, source structure, and review-readiness.
- `docs/testing_strategy.rst` for testing philosophy and physics gates.
- `docs/code_structure.rst` for where implementation work belongs.
- `docs/performance.rst` for performance evidence and historical timing
  snapshots.
- `docs/free_boundary_coil_optimization.rst` and `vmec_jax_plan/plan_freeb.md` for the
  closed free-boundary evidence summary.

## Review Guardrails

- Do not add new entries here.
- Do not use this file to override the active plan.
- If an old result needs audit, retrieve the pre-compaction contents from git
  history and summarize the conclusion in `vmec_jax_plan/plan_differentiability.md`.
