# Research-Grade Differentiable VMEC Plan

Status: active umbrella plan and single source of truth for PR #20.

This file is intentionally concise.  It records the current target architecture,
promotion gates, open lanes, and recent review evidence.  Detailed historical
logs remain available in git history.  `plan_freeb.md` is a closed
free-boundary evidence summary; `plan.md` and
`discrete_adjoint_2506_plan.md` are historical references only.

Last updated: 2026-06-20.

## Current Objective

Build and review a simpler, research-grade `vmec_jax` implementation that:

1. Preserves fixed-boundary and free-boundary VMEC2000 parity.
2. Provides validated differentiability for promoted seams.
3. Keeps adaptive free-boundary branch claims conservative until true
   fingerprint-gated full adaptive AD-vs-central-FD gates pass.
4. Reduces source sprawl and large monoliths without weakening physics gates.
5. Keeps user-facing optimization examples pedagogical and reproducible.
6. Keeps the repository lightweight and free of generated solver outputs.

## Design Principles

- Domain modules own implementation.  New code should live under packages such
  as `solvers/`, `drivers/`, `optimizers/`, `io/`, `external_fields/`, and
  `validation/`, not as new root modules.
- Root modules are compatibility facades or public APIs.  Existing root facades
  are tolerated only to preserve public imports and monkeypatch/debug workflows.
- Tests should be real unit, numerical, parity, AD-vs-FD, and physics gates.
  Avoid scaffold-only tests that increase coverage without validating behavior.
- Prefer net-negative refactors: fewer lines, fewer duplicate seams, clearer
  ownership, no new broad abstractions unless they remove more code than they
  add.
- Keep differentiability claims evidence-based.  Branch-local/frozen-branch
  gates can be promoted; arbitrary adaptive branch differentiation is unclaimed
  until validated.
- Keep docs and examples executable from user-facing APIs, not from private
  wrappers that hide the optimization workflow.

## Current Source Map

- CLI and public entry points:
  - `vmec_jax/cli.py`
  - `vmec_jax/driver.py`
  - `vmec_jax/drivers/`
  - `vmec_jax/solve.py` as a compatibility facade.
- Fixed-boundary solvers:
  - `vmec_jax/solvers/fixed_boundary/`
  - Main remaining monolith:
    `vmec_jax/solvers/fixed_boundary/residual/iteration.py`.
- Free-boundary solvers and direct-coil seams:
  - `vmec_jax/solvers/free_boundary/`
  - Compatibility facades:
    `vmec_jax/free_boundary_adjoint.py`,
    `vmec_jax/free_boundary_adjoint_controller.py`.
- External fields and coils:
  - `vmec_jax/external_fields/`.
- Optimization APIs and examples:
  - `vmec_jax/optimization.py`
  - `vmec_jax/optimizers/`
  - `vmec_jax/qi_optimization.py`
  - `examples/optimization/`.
- WOUT, diagnostics, and physics quantities:
  - `vmec_jax/wout.py` facade.
  - `vmec_jax/io/wout/` implementation.
  - Mercier/DR diagnostics live in WOUT/diagnostics code, not in examples.
- Performance instrumentation:
  - `vmec_jax/solvers/fixed_boundary/performance.py`
  - Solver-specific cache/timing helpers should stay in solver packages.

## Current Evidence Snapshot

Latest local branch state:

- Branch: `codex/differentiability-refactor-plan`.
- Recent pushed commits:
  - `2d300c5e Compact differentiability plan for review`.
  - `427f315b Reduce fixed-boundary root helper sprawl`.
- The working tree should be checked with `git status --short --branch` before
  each tranche; avoid relying on stale plan text for branch state.

Latest local gates run:

- `python -m ruff check` on changed residual/performance files.
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_performance_wave13_coverage.py tests/test_refactorable_seams_coverage.py tests/test_solve_residual_iter_setup_helpers.py tests/test_solve_residual_iter_policy.py tests/test_solve_diagnostics_io.py --tb=short`
  (`67 passed`).
- `python tools/diagnostics/source_health.py --top 20 --top-functions 60 --max-root-helper-prefix-files 2`.
- `python tools/diagnostics/repo_size_audit.py --top 10 --max-total-mib 50 --max-file-mib 2`.
- `git diff --check`.

Latest source-health snapshot:

- Root Python files: `67`.
- Root helper-prefix compatibility files: `2`.
- Largest production file:
  `vmec_jax/solvers/fixed_boundary/residual/iteration.py` at `3120` lines.
- Largest production function:
  `solve_fixed_boundary_residual_iter` at `2645` lines.
- Tracked repository size before this compaction: `29.45 MiB`, no tracked file
  above `2 MiB`.

## Open Lanes

- Architecture/refactor plan: `100%`.
- Solver monolith reduction: `99.9994%`.
- Residual iteration decomposition: `99.994%`.
- Root namespace cleanup: `100%`.
- Fixed-boundary VMEC parity and physics gates: `99%+`, keep existing gates.
- Direct-coil/free-boundary phase 1: `100%`.
- Full nonlinear free-boundary adjoint phase 2:
  `99.999998%` for branch-local/fingerprint-gated evidence; arbitrary adaptive
  branch differentiation remains unclaimed.
- Single-stage coil-only optimization phase 3: `99%` for examples and
  branch-local derivative proposals with complete solves as acceptance
  authority.
- CPU/GPU performance instrumentation hygiene: `99.46%`; avoid late churn
  unless a focused gate shows a real regression or low-risk improvement.
- CI/runtime/coverage hygiene: `100%` for current local gates; batch CI should
  be checked later, not watched continuously.
- Docs/release hygiene: `100%` for current PR wording and conservative claims.
- Overall PR readiness: `99.99999999999991%`.

## Remaining Implementation Steps

Only do these if the change is net-negative or clearly improves reviewability:

1. Residual finalization payload:
   - Do not add a giant explicit key list unless it removes broader coupling.
   - Accept the current namespace seam if a cleaner replacement would add more
     code than it removes.
2. Residual trace payload:
   - Remove duplicate trace assembly only when tests prove identical trace
     contents.
   - Preserve accepted-point tape and replay diagnostics exactly.
3. Scan-resume restoration:
   - Consolidate repeated restoration fields only if it shortens both the scan
     and non-scan paths.
   - Preserve resume-state diagnostics used by tests and users.
4. Oversized validation tests:
   - Split only around reusable fixtures or repeated setup.
   - Do not weaken AD-vs-FD, VMEC2000 parity, free-boundary, or physics gates.
5. Documentation:
   - Keep `docs/code_structure.rst` synchronized with this source map.
   - Keep README lightweight and push detailed optimization/parity analysis to
     docs.

## Differentiability Promotion Gates

A differentiability feature is promoted only when the current tree contains:

1. A concrete scalar or vector objective.
2. Exact AD/JVP/VJP output.
3. Central finite-difference comparison over the same branch.
4. A branch fingerprint when the host controller is involved.
5. A tolerance justified by numerical conditioning and VMEC parity.
6. Focused tests that run in CI without excessive runtime.

Current claim policy:

- Fixed-boundary residual, implicit, and replay seams are promoted where tests
  pass.
- Direct-coil provider and branch-local free-boundary replay/controller seams
  are promoted where fingerprint-gated tests pass.
- Arbitrary adaptive full-loop branch differentiation is not promoted.

## Physics and Parity Gates

Keep these gates as the scientific backbone of the project:

- VMEC2000 fixed-boundary parity for representative axisymmetric,
  non-axisymmetric, finite-beta, multigrid, symmetric, and asymmetric cases.
- Direct-coil/mgrid/free-boundary parity only with bounded, finite-positive
  geometry fixtures.
- DMerc/DR AD-vs-FD checks for differentiable profile/stability metrics.
- Boozer/QS/QI diagnostics only when backed by reproducible examples and
  provenance.
- Optimization examples must save input/final inputs, WOUTs, history, and
  user-selectable plots without hiding the objective construction.

## Performance Gates

Performance work should be evidence-driven:

- Track cold solve, warm solve, exact callback, accepted-point replay, and
  projected/JVP paths separately.
- Do not trade VMEC2000 parity for speed.
- Promote matrix-free or scalar-adjoint paths only when they win past a clear
  size threshold and keep AD-vs-FD gates.
- GPU work should avoid forcing CPU backends; users with GPU-enabled JAX should
  be able to select GPU naturally.

## Repository Hygiene Gates

- No generated WOUTs, BOOZ files, mgrid dumps, optimization output directories,
  or solver traces in git.
- Tracked size target: below `50 MiB`.
- Individual tracked-file target: below `2 MiB` unless explicitly justified.
- Figures in docs must be compressed and current.
- Use documented downloaders or release artifacts for large validation assets.

## Review-Ready Definition

The PR is review-ready when all of the following are true:

1. Working tree is clean.
2. Source-health gate passes with root helper-prefix limit `2`.
3. Repo-size gate passes.
4. Focused residual, free-boundary, performance, docs, and example tests pass.
5. `git diff --check` passes.
6. README/docs do not overclaim adaptive full-loop differentiability.
7. No new root implementation module was added.
8. Any remaining oversized file is either a real physics gate or an explicitly
   documented compatibility facade.

## Recent Log

### 2026-06-20 Free-Boundary Evidence Log Compaction

Steps taken:

1. Re-audited `plan_freeb.md` after the active plan compaction.
2. Confirmed it was explicitly closed and should not receive new progress
   entries, but still contained `25,581` lines of historical append-only logs.
3. Replaced it with a concise free-boundary evidence summary, current claim
   policy, implemented source areas, validation tests, representative gates,
   and review guardrails.

Results obtained:

- The repository now has one active plan: this file.
- `plan_freeb.md` is now an evidence pointer instead of a parallel historical
  work log.
- The full historical free-boundary transcript remains recoverable from git
  history.

Tests and commands:

- `sed`/`rg` audits over `plan_freeb.md`, `plan_differentiability.md`, and
  `docs/code_structure.rst`.
- `wc -l plan_differentiability.md plan_freeb.md plan.md discrete_adjoint_2506_plan.md`.

Best next steps:

1. Re-run repo-size and source-health gates after this compaction.
2. Keep all new free-boundary status updates in this active plan only.

### 2026-06-20 Plan Compaction and Final Audit

Steps taken:

1. Re-audited the active branch, latest commit, source-health, repo-size, and
   plan ownership.
2. Found that `plan_differentiability.md` had grown to `34,868` lines and
   `1.6 MiB`, making it the largest tracked file.
3. Replaced the historical append-only log with this concise current-state plan
   while keeping older details available through git history.

Results obtained:

- The project again has a short single active plan that is usable for review.
- The plan now states current source ownership, open lanes, promotion gates,
  repository hygiene gates, and finite next steps in one place.
- Older detailed logs were intentionally not moved to another tracked file,
  avoiding archive sprawl.
- The active plan is now `255` lines and `12 KiB`, down from `34,868` lines
  and `1.6 MiB`.
- The tracked repository size is now `27.84 MiB`, down from `29.45 MiB`, with
  no tracked file above `2 MiB`.

Tests and commands:

- `git status --short --branch`
- `python tools/diagnostics/source_health.py --top 20 --top-functions 60 --max-root-helper-prefix-files 2`
- `python tools/diagnostics/repo_size_audit.py --top 10 --max-total-mib 50 --max-file-mib 2`
- `wc -l plan_differentiability.md plan_freeb.md plan.md discrete_adjoint_2506_plan.md`
- `rg` audits over the active plan for old open-lane and deferred-work markers.

Best next steps:

1. If code changes continue, target only the finite residual seams listed above.
2. Prepare the PR for review rather than adding more broad refactor waves.

User decisions needed:

No immediate decision.  Before merge, decide whether the historical plan logs in
git history are sufficient for auditability; current tracked files no longer
carry the full append-only transcript.
