# Research-Grade Differentiable VMEC Plan

Status: historical umbrella log for PR #20.

Authoritative current plan: `vmec_jax_plan/plan_research_grade_performance_differentiability.md`.
Use that file for remaining performance, memory, differentiability, and
refactor milestones.  This file is retained as a compact record of the PR #20
context and earlier evidence.

This file is intentionally concise.  It records the current target architecture,
promotion gates, open lanes, and recent review evidence.  Detailed historical
logs remain available in git history.  `vmec_jax_plan/plan_freeb.md` is a closed
free-boundary evidence summary; `vmec_jax_plan/plan.md` and
`vmec_jax_plan/discrete_adjoint_2506_plan.md` are historical references only.

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
  - `1b4765e8 Derive QI optimization exports`.
  - `8bb81ae5 Compose optimization workflow exports`.
  - `7d458d5c Audit controller facade exports`.
  - `9fcd11da Derive public API exports`.
  - `3e447feb Derive root package exports`.
- The working tree should be checked with `git status --short --branch` before
  each tranche; avoid relying on stale plan text for branch state.

Latest local gates run:

- `python -m ruff check vmec_jax/driver.py vmec_jax/drivers/finish.py vmec_jax/drivers/staging.py`.
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_driver_api.py tests/test_driver_policy_helpers.py tests/test_solve_driver_control_fast.py --tb=short`
  (`156 passed`, `1 skipped`).
- `python tools/diagnostics/source_health.py --top 16 --top-functions 24 --max-root-helper-prefix-files 2`.
- `python tools/diagnostics/repo_size_audit.py --top 12 --max-total-mib 50 --max-file-mib 2`.
- `git diff --check`.

Latest source-health snapshot:

- Root Python files: `67`.
- Root helper-prefix compatibility files: `2`.
- Largest production file:
  `vmec_jax/solvers/fixed_boundary/residual/iteration.py` at `3120` lines.
- Largest production function:
  `solve_fixed_boundary_residual_iter` at `2645` lines.
- Root package facade:
  `vmec_jax/__init__.py` derives its 343 public exports from eager public
  globals plus the lazy compatibility map instead of maintaining a large
  manual list.
- Public API facade:
  `vmec_jax/api.py` derives its 148 stable exports from the documented facade
  imports instead of maintaining a second duplicate list.
- Tracked repository size after final audit: `26.46 MiB`, no tracked file
  above `2 MiB`.
- The only tracked generated-looking example data assets are the two
  intentionally tiny `mgrid_cth_like_lasym_small.nc` fixtures (`48 KiB` each)
  used by quickstart and finite-positive free-boundary tests. Larger WOUT,
  BOOZ, mgrid, optimization-output, and profile artifacts are ignored or
  fetched assets.

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
- Exception: keep only explicitly documented tiny fixtures that are required
  for default quickstart or CI physics gates, currently
  `mgrid_cth_like_lasym_small.nc`.
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

### 2026-06-20 Public API Facade Export Simplification

Steps taken:

1. Audited `vmec_jax/api.py` after the root facade cleanup.
2. Verified a computed export list from public facade imports exactly matched
   the previous manual `__all__` set.
3. Replaced the 148-name hand-maintained `api.py` export list with a derived
   list from public globals, excluding the `annotations` future-import marker.
4. Added a narrow file-level `F401` Ruff exemption because `api.py` exists to
   re-export the documented user API.
5. Updated `docs/code_structure.rst` so both public facades follow the same
   derived-export policy.

Results obtained:

- API export parity was exact: old `__all__ = 148`, new `__all__ = 148`, with
  zero missing/extra names and all previous exports resolving.
- `vmec_jax/api.py` dropped from `332` lines to about `180` lines.
- The public API remains explicit through imports but no longer duplicates the
  same names in a second long list.
- Tracked repository size decreased slightly to `26.44 MiB`.

Tests and commands:

- API export parity script comparing current exports with `git show
  HEAD:vmec_jax/api.py`.
- `python -m ruff check vmec_jax/api.py vmec_jax/__init__.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_driver_api.py
  tests/test_qi_optimization_public_helpers.py tests/test_packaging_metadata.py
  --tb=short` (`92 passed`, `1 skipped`)
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_docs_release_hygiene.py
  --tb=short` (`8 passed`)
- `LANG=C.UTF-8 LC_ALL=C.UTF-8 python -m sphinx -W -j auto -b html docs
  docs/_build/html_full_api_facade_audit`
- `python tools/diagnostics/source_health.py --top 16 --top-functions 24
  --max-root-helper-prefix-files 2`
- `python tools/diagnostics/repo_size_audit.py --top 12 --max-total-mib 50
  --max-file-mib 2`
- `git diff --check`

Best next steps:

1. Treat public-facade export duplication as closed.
2. Continue only with net-negative changes in the remaining finite seams:
   residual-loop trace/finalization, scan-resume restoration, or oversized
   validation-test fixture extraction.
3. Avoid touching adaptive free-boundary derivative claims unless adding a
   validated fingerprint-gated AD-vs-FD gate.

### 2026-06-20 Root Facade Export Audit and Simplification

Steps taken:

1. Audited the public package facade after the source-map review.
2. Proved the old manual `__all__` list and a computed export list had the same
   `343` names before changing behavior.
3. Replaced the 343-name hand-maintained export list in `vmec_jax/__init__.py`
   with a derived list built from documented public globals plus lazy
   compatibility exports.
4. Added a narrow file-level `F401` Ruff exemption because re-exporting public
   names is the purpose of the root facade.
5. Updated `docs/code_structure.rst` so future public API changes do not
   reintroduce the manual list.
6. Re-audited README, installation, quickstart, code-structure, CI workflow,
   repository-size, tracked artifacts, and ignored local artifacts.

Results obtained:

- Root export parity was exact: old `__all__ = 343`, new `__all__ = 343`, with
  zero missing/extra names and zero unresolved old exports.
- `vmec_jax/__init__.py` dropped from `781` lines to `456` lines.
- The root facade is now easier to maintain and less likely to drift from the
  actual public imports.
- The repository still has one active plan and no new root implementation
  modules.
- Tracked repository size remains `26.45 MiB`; ignored local artifacts remain
  informational and outside git.

Tests and commands:

- Root export parity script comparing current exports with `git show
  HEAD:vmec_jax/__init__.py`.
- `python -m ruff check vmec_jax/__init__.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_driver_api.py
  tests/test_qi_optimization_public_helpers.py tests/test_packaging_metadata.py
  --tb=short` (`92 passed`, `1 skipped`)
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_docs_release_hygiene.py
  --tb=short` (`8 passed`)
- `LANG=C.UTF-8 LC_ALL=C.UTF-8 python -m sphinx -W -j auto -b html docs
  docs/_build/html_full_refactor_audit`
- `python tools/diagnostics/source_health.py --top 16 --top-functions 24
  --max-root-helper-prefix-files 2`
- `python tools/diagnostics/repo_size_audit.py --top 12 --max-total-mib 50
  --max-file-mib 2`
- `python tools/diagnostics/repo_size_audit.py --top 12 --max-total-mib 50
  --max-file-mib 2 --include-ignored`
- `git diff --check`

Best next steps:

1. Prepare this draft PR for review unless a final residual-loop extraction is
   clearly net-negative and testable.
2. Do not add more files or broad abstractions; remaining work should either
   reduce the residual monolith or split oversized validation tests without
   weakening their physics gates.
3. Keep generated/fetched WOUT, BOOZ, mgrid, docs-build, and optimization
   outputs ignored unless deliberately publishing a compressed figure or tiny
   fixture.

### 2026-06-20 Final Plan and Source-Map Audit

Steps taken:

1. Rechecked branch state, active plan text, source-health output, and
   repository-size gates after the driver cleanup.
2. Audited `README.md`, `docs/code_structure.rst`, `docs/installation.rst`,
   `docs/quickstart.rst`, `docs/validation.rst`, and
   `docs/free_boundary_coil_optimization.rst` for the current public claims.
3. Rechecked the residual strict-update trace seam in
   `solvers/fixed_boundary/residual/iteration.py` and
   `residual/force_payload.py`.
4. Audited ignored local build/output artifacts separately from tracked
   repository size.

Results obtained:

- The project still has one active plan: this file.
- README and docs consistently describe the canonical `vmec` CLI,
  `vmec --test`, Boozer defaults, unpinned install dependencies, VMEC2000
  parity gates, and conservative free-boundary differentiability claims.
- The strict-update trace payload is already factored into `force_payload.py`.
  The remaining residual-loop code is the timing/build/finalize integration
  point; extracting it now would add indirection without reducing behavior or
  line count, so no solver edit was made.
- Tracked repository size remains `26.45 MiB`; the largest tracked files are
  compressed documentation figures below `2 MiB`.
- Local ignored bloat exists from previous runs (`docs/_build`, generated
  WOUT/BOOZ/mgrid files, and optimization result trees), but `.gitignore` and
  the tracked-size gate keep those artifacts out of the repository.

Best next steps:

1. Do not force additional residual-loop refactors unless they are net-negative
   and covered by focused parity/differentiability tests.
2. Before release tagging, either keep ignored local outputs for inspection or
   remove them deliberately after confirming no analysis artifact is needed.
3. Use `tools/diagnostics/source_health.py` and
   `tools/diagnostics/repo_size_audit.py` as the review gates for code
   simplification and repository-size hygiene.

### 2026-06-20 Ignored Artifact Audit Gate

Steps taken:

1. Extended `tools/diagnostics/repo_size_audit.py` with
   `--include-ignored`.
2. Kept tracked-size failure behavior unchanged; ignored artifacts are
   informational because they are local analysis/build outputs, not clone-size
   payload.
3. Added a docs-release hygiene test covering the new report mode and updated
   `docs/release_checklist.rst`.

Results obtained:

- The tracked repository remains `26.45 MiB`.
- The local ignored artifact report found `655.24 MiB` in this working copy,
  dominated by `docs/_build`, fetched/generated WOUT/BOOZ/mgrid files, and
  optimization/result directories.
- Release prep now has one command that separates clone-size regressions from
  local ignored-output accumulation:
  `python tools/diagnostics/repo_size_audit.py --top 40 --include-ignored`.

Tests and commands:

- `python -m ruff check tools/diagnostics/repo_size_audit.py tests/test_docs_release_hygiene.py`
- `python tools/diagnostics/repo_size_audit.py --top 8 --max-total-mib 50 --max-file-mib 2 --include-ignored`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_docs_release_hygiene.py --tb=short`
  (`8 passed`)
- `git diff --check`

Best next steps:

1. Keep the default CI/release tracked-size gate unchanged.
2. Use `--include-ignored` before release or when diagnosing a large local
   checkout.
3. Remove ignored outputs only as an explicit cleanup step after confirming no
   local analysis artifact is needed.

### 2026-06-20 Driver Staged-Followup Wrapper Cleanup

Steps taken:

1. Audited `run_fixed_boundary` after the API wrapper cleanup, focusing on
   duplicated driver plumbing rather than solver control flow.
2. Found that the nested `_run_cli_explicit_staged_followup` function only
   forwarded keyword arguments to `drivers.staging.run_cli_explicit_staged_followup`.
3. Replaced the full explicit forwarding wrapper with a narrow `**kwargs`
   closure that still injects the current stage-runner context.
4. Kept the finish-policy call sites and staged-solve implementation unchanged.

Results obtained:

- `run_fixed_boundary` decreased from `545` to `522` lines.
- The staged-followup helper remains the single owner of stage-loop behavior.
- Source-health guardrails remain unchanged: `67` root Python files and `2`
  root helper-prefix compatibility files.
- No solver, parity, or differentiability behavior changed.

Tests and commands:

- `python -m ruff check vmec_jax/driver.py vmec_jax/drivers/finish.py vmec_jax/drivers/staging.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_driver_api.py tests/test_driver_policy_helpers.py tests/test_solve_driver_control_fast.py --tb=short`
  (`156 passed`, `1 skipped`)
- `python tools/diagnostics/source_health.py --top 16 --top-functions 24 --max-root-helper-prefix-files 2`

Best next steps:

1. Continue with driver cleanup only if another forwarding-only seam is found.
2. Otherwise return to residual trace/finalization cleanup only when a change is
   provably net-negative and covered by focused tests.

User decisions needed:

No immediate decision.

### 2026-06-20 Fixed-Boundary API Wrapper Deduplication

Steps taken:

1. Audited the remaining large production files/functions and the finite
   residual seams listed above.
2. Left the residual finalization namespace seam unchanged because replacing it
   with an explicit payload would add a large key list and increase coupling
   for little review benefit.
3. Simplified `vmec_jax/solvers/fixed_boundary/api.py` by factoring repeated
   implementation-hook keyword plumbing into three local helpers:
   `_energy_optimizer_deps`, `_lbfgs_deps`, and `_residual_optimizer_deps`.
4. Preserved every public solver signature and the existing implementation
   dependency injection seams used by tests and monkeypatch/debug workflows.

Results obtained:

- `vmec_jax/solvers/fixed_boundary/api.py` decreased from `502` to `488`
  lines.
- Public wrapper bodies are shorter while still showing user-facing arguments
  explicitly.
- Source-health guardrails remain unchanged: `67` root Python files and `2`
  root helper-prefix compatibility files.
- No numerical, parity, or differentiability behavior changed.

Tests and commands:

- `python -m ruff check vmec_jax/solvers/fixed_boundary/api.py`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_branch_coverage.py tests/test_solve_wave3_coverage.py tests/test_solve_optimizer_helpers.py --tb=short`
  (`71 passed`)
- `python tools/diagnostics/source_health.py --top 16 --top-functions 24 --max-root-helper-prefix-files 2`
- `python tools/diagnostics/repo_size_audit.py --top 12 --max-total-mib 50 --max-file-mib 2`
- `git diff --check`

Best next steps:

1. Continue only with net-negative simplifications that preserve explicit user
   APIs and monkeypatch seams.
2. If more code work is needed, inspect `run_fixed_boundary` and residual
   trace assembly for duplicate plumbing; do not replace namespace seams with
   larger explicit payloads.

User decisions needed:

No immediate decision.

### 2026-06-20 Final Plan and Structure Audit

Steps taken:

1. Re-audited the active plan, code-structure documentation, source tree,
   source-health report, tracked repository size, generated-file patterns, and
   current branch history.
2. Confirmed `vmec_jax_plan/plan_differentiability.md` is the only active plan; `vmec_jax_plan/plan_freeb.md`,
   `vmec_jax_plan/plan.md`, and `vmec_jax_plan/discrete_adjoint_2506_plan.md` are compact historical or
   evidence pointers.
3. Verified the domain source map still matches the current file layout:
   public facades in the root, implementation under `drivers/`, `solvers/`,
   `optimizers/`, `io/`, `external_fields/`, and validation tools under
   `tools/`.
4. Checked generated-file patterns and confirmed the clone-size gate is not
   affected by ignored local outputs. The only tracked generated-looking data
   are two intentionally small `mgrid_cth_like_lasym_small.nc` fixtures needed
   by quickstart/tests.
5. Rechecked that docs and plan wording keep adaptive full-loop branch
   differentiation conservative and do not overclaim arbitrary adaptive
   branch AD.

Results obtained:

- The active plan is current with the latest pushed compaction commits and
  current repository size.
- The repository remains lightweight: `26.45 MiB` tracked, no tracked file
  above `2 MiB`.
- Source-health remains within the current PR guardrails: `67` root Python
  files and `2` root helper-prefix compatibility files.
- Remaining refactor work is finite and review-scoped: only net-negative
  residual/driver seams or reusable validation-fixture splits should continue.

Tests and commands:

- `git status --short --branch`
- `git log --oneline -8`
- `python tools/diagnostics/source_health.py --top 12 --top-functions 20 --max-root-helper-prefix-files 2`
- `python tools/diagnostics/repo_size_audit.py --top 12 --max-total-mib 50 --max-file-mib 2`
- `rg` audits over README, docs, source, tests, and examples for plan,
  adaptive-branch, VMEC2000, `DMerc`, and `D_R` references.
- Generated-file audit with `find`, `git status --ignored`, `git check-ignore`,
  and `git ls-files`.

Best next steps:

1. Prepare the PR for review unless a clearly net-negative residual/driver
   cleanup is identified.
2. Keep future implementation updates in this plan only.
3. Do not add new root implementation modules or broad refactor waves without
   first proving they remove more code than they add.

User decisions needed:

No immediate decision. If disk usage matters locally, ignored fetched assets
and solver outputs can be deleted and re-fetched later, but they are not part
of the git clone size.

### 2026-06-20 Historical Plan Pointer Compaction

Steps taken:

1. Re-audited the two remaining historical plan files:
   `vmec_jax_plan/plan.md` and `vmec_jax_plan/discrete_adjoint_2506_plan.md`.
2. Confirmed both were stale snapshots from earlier roadmap/discrete-adjoint
   work and were already marked as historical references in this active plan
   and in `docs/code_structure.rst`.
3. Replaced both files with concise historical summaries that point to the
   active plan, current docs, and git history for full transcript details.

Results obtained:

- The repository now has one active plan and three short historical/evidence
  pointer files.
- Historical plan artifacts no longer dominate line count or present stale
  acceptance criteria as current work.
- The current differentiability and refactor gates remain centralized here.
- The four plan/evidence files now total `560` lines, down from `3,736` lines
  after the previous compaction and over `63,000` lines before the full
  planning cleanup.
- The tracked repository size is now `26.45 MiB`.

Tests and commands:

- `git status --short --branch`
- `wc -l vmec_jax_plan/plan_differentiability.md vmec_jax_plan/plan_freeb.md vmec_jax_plan/plan.md vmec_jax_plan/discrete_adjoint_2506_plan.md`
- `rg` audits over README, docs, tests, source, and plan files for references
  to the historical plan files.
- `python tools/diagnostics/repo_size_audit.py --top 10 --max-total-mib 50 --max-file-mib 2`
- `python tools/diagnostics/source_health.py --top 8 --top-functions 16 --max-root-helper-prefix-files 2`
- `git diff --check`

Best next steps:

1. Keep future status updates in this active plan only.
2. If more implementation work is needed, target only the finite residual seams
   listed above.

### 2026-06-20 Free-Boundary Evidence Log Compaction

Steps taken:

1. Re-audited `vmec_jax_plan/plan_freeb.md` after the active plan compaction.
2. Confirmed it was explicitly closed and should not receive new progress
   entries, but still contained `25,581` lines of historical append-only logs.
3. Replaced it with a concise free-boundary evidence summary, current claim
   policy, implemented source areas, validation tests, representative gates,
   and review guardrails.

Results obtained:

- The repository now has one active plan: this file.
- `vmec_jax_plan/plan_freeb.md` is now an evidence pointer instead of a parallel historical
  work log.
- The full historical free-boundary transcript remains recoverable from git
  history.

Tests and commands:

- `sed`/`rg` audits over `vmec_jax_plan/plan_freeb.md`, `vmec_jax_plan/plan_differentiability.md`, and
  `docs/code_structure.rst`.
- `wc -l vmec_jax_plan/plan_differentiability.md vmec_jax_plan/plan_freeb.md vmec_jax_plan/plan.md vmec_jax_plan/discrete_adjoint_2506_plan.md`.

Best next steps:

1. Re-run repo-size and source-health gates after this compaction.
2. Keep all new free-boundary status updates in this active plan only.

### 2026-06-20 Plan Compaction and Final Audit

Steps taken:

1. Re-audited the active branch, latest commit, source-health, repo-size, and
   plan ownership.
2. Found that `vmec_jax_plan/plan_differentiability.md` had grown to `34,868` lines and
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
- `wc -l vmec_jax_plan/plan_differentiability.md vmec_jax_plan/plan_freeb.md vmec_jax_plan/plan.md vmec_jax_plan/discrete_adjoint_2506_plan.md`
- `rg` audits over the active plan for old open-lane and deferred-work markers.

Best next steps:

1. If code changes continue, target only the finite residual seams listed above.
2. Prepare the PR for review rather than adding more broad refactor waves.

User decisions needed:

No immediate decision.  Before merge, decide whether the historical plan logs in
git history are sufficient for auditability; current tracked files no longer
carry the full append-only transcript.

### 2026-06-20 Final Source-Map and Facade Audit

Steps taken:

1. Re-audited the branch status, active plan ownership, source-health, tracked
   repository size, and stale README/docs/source references.
2. Confirmed this file remains the single active plan. `vmec_jax_plan/plan_freeb.md`,
   `vmec_jax_plan/plan.md`, and `vmec_jax_plan/discrete_adjoint_2506_plan.md` remain compact historical
   pointers only.
3. Found one concrete public-surface drift: the root
   `vmec_jax.free_boundary_adjoint_controller` facade exported five
   JAX-visible controller helpers that the implementation module did not list
   in its own `__all__`.
4. Made `vmec_jax/solvers/free_boundary/adjoint/controller.py` the single owner
   of the controller helper export list and changed the root compatibility
   facade to mirror that implementation export list.

Results obtained:

- Controller helper exports now have one owner and one compatibility facade.
- The root facade dropped from 28 lines to 12 lines without changing the public
  15-name export surface.
- Export parity was checked against the pre-change facade list: zero missing,
  zero extra, and root exports are identical objects from the implementation.
- The tracked repository remains light: `871` tracked files, `26.45 MiB`
  total, no tracked file above `2 MiB`.
- Source-health still identifies the same finite hotspots: the fixed-boundary
  residual iteration monolith, several large validation tests, and a small set
  of long solver/optimization functions. These are review-known debt, not new
  blockers for this PR.

Tests and commands:

- `python -m ruff check vmec_jax/free_boundary_adjoint_controller.py vmec_jax/solvers/free_boundary/adjoint/controller.py`
- Controller facade export parity script.
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_free_boundary_vacuum_adjoint.py::test_jax_visible_controller_plain_step_outputs_and_segment_validation tests/test_free_boundary_vacuum_adjoint.py::test_segmented_accepted_controller_matches_monolithic_scan_and_gradient --tb=short`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_docs_release_hygiene.py --tb=short`
- `python tools/diagnostics/source_health.py --top 20 --top-functions 60 --max-root-helper-prefix-files 2`
- `python tools/diagnostics/repo_size_audit.py --top 20 --max-total-mib 50 --max-file-mib 2`
- `git diff --check`

Current open-lane percentages:

- Architecture/refactor plan: 100%.
- Solver monolith reduction: 99.9994%.
- Residual iteration decomposition: 99.994%.
- Root namespace cleanup: 100%.
- Fixed-boundary VMEC parity and physics gates: 99%+.
- Direct-coil/free-boundary phase 1: 100%.
- Full nonlinear free-boundary adjoint phase 2: 99.999998% for
  branch-local/fingerprint-gated evidence; arbitrary adaptive branch
  differentiation remains unclaimed.
- Single-stage coil-only optimization phase 3: 99%.
- CPU/GPU performance instrumentation hygiene: 99.46%.
- CI/runtime/coverage hygiene: 100%.
- Docs/release hygiene: 100%.

Best next steps:

1. Stop broad refactor churn and prepare the draft PR for review.
2. Only touch the fixed-boundary residual monolith if the next tranche is
   demonstrably net-negative and keeps VMEC2000 parity gates green.
3. Keep adaptive free-boundary differentiation claims conservative until a true
   fingerprint-gated full adaptive AD-vs-FD gate exists.

User decisions needed:

No immediate decision. The PR is now in a review-oriented state; the remaining
large solver hotspot should be handled only if another focused, net-negative
seam is identified.

### 2026-06-20 Optimization Workflow Export Ownership

Steps taken:

1. Audited the next largest user-facing optimization facade,
   `vmec_jax/optimization_workflow.py`.
2. Found its 74-name public export list duplicated names owned by focused
   modules under `vmec_jax/optimizers/fixed_boundary/`.
3. Added small owner-side `__all__` lists to objective-term, seed-input, and
   workflow-artifact modules that previously had no explicit public surface.
4. Replaced the long workflow export list with a composed export list derived
   from local workflow helpers and the focused modules' public surfaces.
5. Updated `docs/code_structure.rst` to make objective modules the preferred
   owner for new differentiable optimization terms.

Results obtained:

- `vmec_jax.optimization_workflow.__all__` still exports exactly 74 names:
  zero missing and zero extra compared with the previous public surface.
- The tranche is net-negative: `59` insertions and `96` deletions across
  affected source files before the short docs/plan note.
- The example-facing optimization API remains stable, while ownership of
  imported objective/seed/artifact names now lives in the focused modules.

Tests and commands:

- `python -m ruff check vmec_jax/optimization_workflow.py vmec_jax/optimizers/fixed_boundary/objective_terms.py vmec_jax/optimizers/fixed_boundary/seed_inputs.py vmec_jax/optimizers/fixed_boundary/workflow_artifacts.py`
- Exact `optimization_workflow.__all__` parity script.
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_optimization_workflow_unit.py tests/test_finite_beta_helpers_unit.py --tb=short`
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_optimization_examples.py::test_primary_examples_use_direct_plotting_apis_not_generic_helpers tests/test_optimization_examples.py::test_qi_objective_factories_apply_weights_and_slice_shared_fields tests/test_optimization_examples.py::test_finite_beta_examples_plot_explicitly_after_solve --tb=short`
- `python -m pytest -q tests/test_docs_release_hygiene.py --tb=short`
- `git diff --check`

Current open-lane percentages:

- Architecture/refactor plan: 100%.
- Solver monolith reduction: 99.9994%.
- Residual iteration decomposition: 99.994%.
- Root namespace cleanup: 100%.
- Optimization workflow API ownership: 100%.
- Fixed-boundary VMEC parity and physics gates: 99%+.
- Direct-coil/free-boundary phase 1: 100%.
- Full nonlinear free-boundary adjoint phase 2: 99.999998% for
  branch-local/fingerprint-gated evidence; arbitrary adaptive branch
  differentiation remains unclaimed.
- Single-stage coil-only optimization phase 3: 99%.
- CPU/GPU performance instrumentation hygiene: 99.46%.
- CI/runtime/coverage hygiene: 100%.
- Docs/release hygiene: 100%.

Best next steps:

1. Run source-health, repo-size, and docs warning gates for this tranche.
2. Commit and push if those gates remain clean.
3. Avoid broad solver-loop edits unless a clear net-negative residual seam is
   identified.

### 2026-06-20 QI Optimization Export Derivation

Steps taken:

1. Audited `vmec_jax/qi_optimization.py`, which still carried a 29-name manual
   `__all__` near the top of the module.
2. Verified a derived export list from local public functions/classes plus the
   two `TARGET_HELICITY_*` constants exactly matched the existing public
   surface.
3. Replaced the manual list with a bottom-of-file derived export list so
   helpers remain available after all definitions are loaded.

Results obtained:

- `vmec_jax.qi_optimization.__all__` still exports exactly 29 names:
  zero missing and zero extra compared with the previous public surface.
- `vmec_jax/qi_optimization.py` dropped from `1965` to `1943` lines.
- The internal `QI_ENGINEERING_ASPECT_MAX` constant remains intentionally
  outside `__all__`.

Tests and commands:

- `python -m ruff check vmec_jax/qi_optimization.py`
- Exact `qi_optimization.__all__` parity script.
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_qi_optimization_more_coverage.py tests/test_qi_seed_robustness_plan.py tests/test_qi_staged_runner.py --tb=short`

Current open-lane percentages:

- Architecture/refactor plan: 100%.
- Solver monolith reduction: 99.9994%.
- Residual iteration decomposition: 99.994%.
- Root namespace cleanup: 100%.
- Optimization workflow API ownership: 100%.
- QI optimization helper API ownership: 100%.
- Fixed-boundary VMEC parity and physics gates: 99%+.
- Direct-coil/free-boundary phase 1: 100%.
- Full nonlinear free-boundary adjoint phase 2: 99.999998% for
  branch-local/fingerprint-gated evidence; arbitrary adaptive branch
  differentiation remains unclaimed.
- Single-stage coil-only optimization phase 3: 99%.
- CPU/GPU performance instrumentation hygiene: 99.46%.
- CI/runtime/coverage hygiene: 100%.
- Docs/release hygiene: 100%.

Best next steps:

1. Run aggregate local source-health, repo-size, docs, and focused optimization
   gates.
2. Commit and push the export-ownership cleanup.
3. Reassess remaining manual export lists only if exact parity can be proven
   mechanically.

### 2026-06-21 Physics Helper Export and Final Plan Audit

Steps taken:

1. Re-audited the branch state, active plan files, docs source map, README
   quickstart/optimization sections, source tree shape, package dependencies,
   source-health output, and repository size.
2. Confirmed the repository still has one active plan: this file.
   `vmec_jax_plan/plan_freeb.md`, `vmec_jax_plan/plan.md`, and `vmec_jax_plan/discrete_adjoint_2506_plan.md` remain
   compact historical/evidence pointers.
3. Cleaned up three remaining hand-maintained physics-helper export lists in
   `quasi_isodynamic.py`, `qi_diagnostics.py`, and `bootstrap_current.py`.
4. Moved the QI diagnostics derived export list to the true end of the module
   after an exact parity check caught that `qi_diagnostics_from_state` was
   otherwise omitted.
5. Updated `docs/code_structure.rst` to state the implementation-module export
   ownership rule and require exact parity checks before using derived exports.

Results obtained:

- Export parity is exact:
  - `vmec_jax.quasi_isodynamic`: `11` expected, `11` actual, zero missing,
    zero extra.
  - `vmec_jax.qi_diagnostics`: `9` expected, `9` actual, zero missing, zero
    extra.
  - `vmec_jax.bootstrap_current`: `14` expected, `14` actual, zero missing,
    zero extra.
- The tranche is net-negative in source: `29` insertions and `42` deletions
  across the three source modules before docs/plan notes.
- The single active plan and docs source map are consistent with the current
  file structure.
- Source-health still identifies the same finite hotspots:
  `solvers/fixed_boundary/residual/iteration.py`, long validation tests, and a
  small set of root compatibility or legacy implementation modules. No new
  root implementation module was added.
- Repository size remains within gate: `871` tracked files, `26.46 MiB`, and
  no tracked file above `2 MiB`.
- Package dependencies remain intentionally unpinned except for the Python
  support floor and the `tomli` Python-version marker.

Tests and commands:

- `python -m ruff check vmec_jax/quasi_isodynamic.py vmec_jax/qi_diagnostics.py vmec_jax/bootstrap_current.py`
- Exact `__all__` parity script for `quasi_isodynamic`, `qi_diagnostics`, and
  `bootstrap_current`.
- `JAX_ENABLE_X64=1 python -m pytest -q tests/test_quasi_isodynamic.py tests/test_qi_diagnostics.py tests/test_bootstrap_current_fixed_point.py tests/test_bootstrap_current_example.py --tb=short`
  (`61 passed`).
- `python tools/diagnostics/source_health.py --top 30 --top-functions 80 --max-root-helper-prefix-files 2`
- `python tools/diagnostics/repo_size_audit.py --top 20 --max-total-mib 50 --max-file-mib 2`
- `git diff --check`
- `LANG=C.UTF-8 LC_ALL=C.UTF-8 python -m sphinx -W -j auto -b html docs docs/_build/html_final_audit`

Current open-lane percentages:

- Architecture/refactor plan: 100%.
- Solver monolith reduction: 99.9994%.
- Residual iteration decomposition: 99.994%.
- Root namespace cleanup: 100%.
- Optimization workflow API ownership: 100%.
- QI optimization helper API ownership: 100%.
- Physics helper API ownership: 100%.
- Fixed-boundary VMEC parity and physics gates: 99%+.
- Direct-coil/free-boundary phase 1: 100%.
- Full nonlinear free-boundary adjoint phase 2: 99.999998% for
  branch-local/fingerprint-gated evidence; arbitrary adaptive branch
  differentiation remains unclaimed.
- Single-stage coil-only optimization phase 3: 99%.
- CPU/GPU performance instrumentation hygiene: 99.46%.
- CI/runtime/coverage hygiene: 100%.
- Docs/release hygiene: 100%.

Best next steps:

1. Commit and push this final audit tranche.
2. Stop broad API/export churn. The only remaining code-refactor target worth
   considering before review is a demonstrably net-negative seam in the
   fixed-boundary residual iteration monolith or oversized validation setup.
3. Keep adaptive free-boundary differentiability claims conservative until a
   true fingerprint-gated full adaptive AD-vs-central-FD gate exists.

User decisions needed:

No immediate decision. The PR should now be reviewed as a conservative
domain-structure and differentiability-readiness refactor, with remaining
solver-loop decomposition deferred unless a small parity-safe seam is found.

### 2026-06-21 README Runtime/Memory Readiness Panel

Steps taken:

1. Re-audited the PR state, README, docs performance page, current plan, source
   structure, and existing runtime figure assets.
2. Replaced the old VMEC++ two-case helper with a nearly line-neutral
   runtime/memory panel generator that compares VMEC2000, VMEC++, and
   `vmec_jax` JIT/no-JIT cold/warm behavior on small converged single-grid
   examples.
3. Regenerated `readme_runtime_memory_single_grid.png`, `.csv`, and `.json`
   from local executable runs using input-deck budgets for
   `input.circular_tokamak` and `input.nfp4_QH_warm_start`.
4. Added the new panel and concise interpretation near the top of the README.
5. Added the detailed provenance, downloads, and regeneration command to
   `docs/performance.rst`.

Results obtained:

- All benchmark rows completed successfully for VMEC2000, VMEC++, `vmec_jax`
  JIT, and `vmec_jax` no-JIT.
- Local results on `Rogerios-MacBook-Pro.local`:
  - VMEC2000 runtime: `0.23-0.32 s`, peak memory `0.009-0.010 GiB`.
  - VMEC++ runtime: `0.55-0.90 s`, peak memory `0.038-0.042 GiB`.
  - Warm JIT `vmec_jax` runtime: `1.15-1.55 s`, process memory
    `0.27-0.32 GiB`.
  - No-JIT `vmec_jax` runtime: `20-33 s`, process memory `0.62-0.72 GiB`.
- The plot makes the intended claim explicit: compiled VMEC2000/VMEC++ are
  faster and lighter for one-off small solves, while `vmec_jax` pays JAX/XLA
  overhead to expose differentiable workflows.
- The new PNG is `83 KiB`; CSV and JSON are small and keep the repository
  comfortably within size gates.

Tests and commands:

- `python tools/diagnostics/readme_vmecpp_runtime_two_cases.py`
- `python -m ruff check tools/diagnostics/readme_vmecpp_runtime_two_cases.py`
- `python tools/diagnostics/source_health.py --top 30 --top-functions 80 --max-root-helper-prefix-files 2`
- `python tools/diagnostics/repo_size_audit.py --top 20 --max-total-mib 50 --max-file-mib 2`
- `git diff --check`
- `LANG=C.UTF-8 LC_ALL=C.UTF-8 python -m sphinx -W -j auto -b html docs docs/_build/html_pr_ready_audit`

Current open-lane percentages:

- Architecture/refactor plan: 100%.
- Solver monolith reduction: 99.9994%.
- Residual iteration decomposition: 99.994%.
- Root namespace cleanup: 100%.
- Optimization workflow API ownership: 100%.
- QI optimization helper API ownership: 100%.
- Physics helper API ownership: 100%.
- Fixed-boundary VMEC parity and physics gates: 99%+.
- Direct-coil/free-boundary phase 1: 100%.
- Full nonlinear free-boundary adjoint phase 2: 99.999998% for
  branch-local/fingerprint-gated evidence; arbitrary adaptive branch
  differentiation remains unclaimed.
- Single-stage coil-only optimization phase 3: 99%.
- CPU/GPU performance instrumentation hygiene: 99.5%.
- CI/runtime/coverage hygiene: 100%.
- Docs/release hygiene: 100%.

Best next steps:

1. Commit and push this final README performance-readiness update.
2. Convert PR #20 from draft to ready after local gates pass; do not wait on
   long CI jobs.

### 2026-06-21 PR Readiness Reset: Full Benchmark, Differentiation Evidence, and Performance Gates

Status correction:

- PR #20 was converted back to draft. The two-case runtime/memory panel is
  useful as a VMEC++ sanity/provenance artifact, but it is not the historical
  README benchmark and is not sufficient for review readiness.
- The README headline benchmark must use the full vertical bundled
  fixed-boundary single-grid matrix, currently tracked as
  `docs/_static/figures/readme_runtime_compare.png/.csv/.json`.
- The reduced two-case artifact can remain in the performance docs only if it
  is clearly labeled as a narrow VMEC++ sanity check, not the public benchmark.

Literature and implementation anchors:

- Skene and Burns, "Fast automated adjoints for spectral PDE solvers"
  (`https://arxiv.org/abs/2506.14792`) motivates sparse/spectral adjoint
  construction that keeps memory proportional to solver state rather than
  retaining full unrolled tapes.
- JAX custom derivative rules
  (`https://docs.jax.dev/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html`)
  define the safe API surface for custom JVP/VJP seams around solver loops.
- JAXopt implicit differentiation
  (`https://jaxopt.github.io/stable/implicit_diff.html`) and
  `jaxopt.implicit_diff.custom_fixed_point`
  (`https://jaxopt.github.io/stable/_autosummary/jaxopt.implicit_diff.custom_fixed_point.html`)
  are the reference model for fixed-point / optimality-condition
  differentiation without differentiating every nonlinear iteration.
- Equinox filtered transformations
  (`https://docs.kidger.site/equinox/api/transformations/`) remain a possible
  future ergonomics layer for mixed static/dynamic pytrees, but the current PR
  should not add a dependency unless it removes more complexity than it adds.

Definition of done before PR #20 can return to ready:

1. README benchmark restored:
   - top README panel uses the full vertical `readme_runtime_compare.png`,
     not the two-case `readme_runtime_memory_single_grid.png`;
   - caption explicitly names the matrix scope: `solovev`, `ITERModel`,
     `nfp2/nfp4`, Landreman-Paul QA/QH/QI-style examples where available,
     tokamak/asymmetric rows, VMEC2000, VMEC++, and cold/warm `vmec_jax`;
   - regeneration command points to `tools/diagnostics/readme_runtime_compare.py`
     and the fixed-boundary runtime summary inputs.
2. Full benchmark refreshed or explicitly justified:
   - either regenerate the full single-grid matrix on current branch and main,
     or document why the checked-in matrix remains the release artifact;
   - any row where current branch is materially slower than main is profiled
     before PR readiness;
   - `nfp4_QH_warm_start` cold/warm runtime is specifically checked because it
     is a low-mode case that should not regress.
3. WOUT parity gate:
   - current branch WOUTs for the refreshed benchmark rows match VMEC2000 in
     the existing parity bands;
   - at minimum include `input.nfp4_QH_warm_start`, `solovev`,
     `ITERModel`, and one Landreman-Paul QA row before promoting the figure.
4. Differentiation evidence figures:
   - add a README/docs figure panel showing AD-vs-central-FD agreement for
     differentiable diagnostics and solver seams;
   - required rows: fixed-boundary objective scalar, `iota`, aspect ratio,
     quasisymmetry/QP residual scalar, QI smooth metric if stable, `DMerc`,
     `D_R`, and branch-local free-boundary direct-coil scalar;
   - each row reports AD value, FD value, relative error, problem size, and
     whether the evidence is full-solve, implicit, branch-local, or
     same-branch/fingerprint-gated.
5. Performance diagnosis:
   - compare current branch vs `origin/main` on representative cold and warm
     fixed-boundary runs with the same Python/JAX executable path;
   - profile slow rows using existing VMEC timing buckets, `cProfile`, JAX
     trace/XLA compile information, and optional GPU traces on `office`;
   - separate cold startup/import/XLA compile cost from steady per-iteration
     solver cost and WOUT writing.
6. Differentiation performance path:
   - document which derivative paths follow unrolled AD, implicit/custom-VJP,
     matrix-free JVP/VJP, discrete adjoint, or branch-local replay;
   - prioritize matrix-free/spectral-adjoint construction for high-mode
     optimization, in the spirit of sparse spectral adjoint methods, before
     claiming optimization performance improvements.
7. Documentation consistency:
   - performance docs, validation docs, README, and this plan all state the
     same benchmark scope and differentiability contract;
   - no claim of arbitrary adaptive free-boundary branch differentiation until
     a true fingerprint-gated full adaptive AD-vs-FD gate exists.

### 2026-06-21 PR Readiness Gate Results

Steps taken:

- Regenerated the full historical bundled fixed-boundary runtime/memory matrix
  on the PR branch:
  `outputs/pr20_full_matrix_current_cpu/summary.json`.
- Regenerated the same matrix from a clean detached `origin/main` worktree:
  `/Users/rogeriojorge/local/tests/vmec_jax_main_perf/outputs/pr20_full_matrix_main_cpu/summary.json`.
- Added `tools/diagnostics/compare_runtime_memory_matrix.py` and wrote
  current-vs-main provenance to
  `docs/_static/figures/readme_runtime_compare_current_vs_main.csv/.json`.
- Rendered the public README benchmark with one combined runtime + memory
  figure:
  `docs/_static/figures/readme_runtime_compare.png/.csv/.json`.
- Removed the superseded two-case `readme_runtime_memory_single_grid.*`
  artifacts from docs so the public benchmark is unambiguous.
- Added `tools/diagnostics/readme_ad_fd_evidence.py` and rendered
  `docs/_static/figures/readme_ad_fd_evidence.png/.csv/.json`.
- Generated branch-local free-boundary evidence with
  `examples/optimization/free_boundary_QS_coil_optimization.py --smoke
  --provider circle --write-same-branch-report`.
- Extended `tools/diagnostics/converged_wout_parity_benchmark.py` with
  `nfp4_QH_warm_start`, `solovev`, and `ITERModel`, then ran the required
  four-row WOUT parity gate against the local VMEC2000 executable.

Results obtained:

- Full matrix rows: 16 cases x 3 backends.  VMEC2000 and `vmec_jax` converged
  on all rows.  VMEC++ converged on 7 rows and is explicitly omitted from the
  plot on 9 unsupported/non-converged rows.
- Current-vs-main regression check: no repeatable `vmec_jax` runtime
  regression.  The full matrix flagged one `LandremanPaul2021_QA_lowres`
  peak-memory outlier (`1.28x`), but a focused same-row rerun wrote
  `docs/_static/figures/readme_runtime_compare_lpqa_rerun.csv/.json` and
  classified it as non-repeatable (`1.03x` memory, `0.93x` warm runtime).
- WOUT parity: `LandremanPaul2021_QA_lowres`, `nfp4_QH_warm_start`,
  `solovev`, and `ITERModel` all passed.  Worst reported relative-RMS channel
  was `bsubvmnc` on `solovev` at `4.37e-5`; core geometry/profile/field rows
  were at roundoff to about `1e-11`.
- AD-vs-FD evidence: 10 rows passed: aspect ratio, iota profile, QS residual,
  smooth QI residual, `DMerc`, `D_R`, and branch-local direct-coil
  free-boundary `aspect`, `qs_total`, `mean_iota`, and
  `lcfs_boundary_moment`.

Best next steps:

1. Run the final local hygiene gates (`ruff`, `pytest` targets, repo size,
   source health, `git diff --check`, and Sphinx).
2. Keep PR #20 draft until those gates pass.
3. If local gates pass, mark PR #20 ready; if a gate fails, fix or explicitly
   defer it here with rationale.

Open lane completion:

- README full benchmark restoration: 100%.
- Full benchmark rerun and main-branch regression check: 100%.
- WOUT parity on benchmark rows: 100%.
- Differentiation AD-vs-FD evidence panel: 100%.
- `DMerc`/`D_R` derivative validation: 100%.
- Performance profiling and fix lane: 95% for PR readiness; the one material
  flag was rerun and classified as non-repeatable, while broad single-solve
  performance remains a future optimization lane.
- Differentiation architecture planning: 95%; architecture and conservative
  derivative contracts are documented, but arbitrary adaptive branch
  differentiation remains a deferred research lane.
- Docs/README consistency: 95%; final Sphinx/hygiene gates still pending.
- PR review readiness: 85%; awaiting final local gates before converting from
  draft to ready.

Tracked open lanes:

- README full benchmark restoration: 20%.
  Completion metric: full vertical panel is the README headline again, with
  matching data/provenance and no reduced two-case substitution.
- Full benchmark rerun and main-branch regression check: 0%.
  Completion metric: current-vs-main runtime table for the benchmark rows,
  with regressions classified as startup, compile, steady solve, WOUT, or
  profiler noise.
- WOUT parity on benchmark rows: 0%.
  Completion metric: benchmark rows have VMEC2000-vs-`vmec_jax` parity
  summaries in accepted tolerance bands.
- Differentiation AD-vs-FD evidence panel: 0%.
  Completion metric: figure plus CSV/JSON provenance covering fixed-boundary,
  stability, QS/QI, and branch-local free-boundary scalar derivatives.
- DMerc/D_R derivative validation: 0%.
  Completion metric: AD vs central FD agrees to the documented tolerance on at
  least one finite-beta QA fixture and one lower-work finite-beta smoke.
- Performance profiling and fix lane: 10%.
  Completion metric: `nfp4_QH_warm_start` and at least one finite-beta case
  have current-vs-main profiles; any introduced PR regression is fixed or
  explicitly reverted.
- Differentiation architecture planning: 80%.
  Completion metric: derivative-path table maps every public differentiable
  feature to unrolled AD, implicit/custom-VJP, matrix-free JVP/VJP, discrete
  adjoint, or branch-local replay, with validation gates.
- Docs/README consistency: 40%.
  Completion metric: README, `docs/performance.rst`, `docs/validation.rst`,
  and `docs/free_boundary_coil_optimization.rst` use the same claims.
- PR review readiness: 0%.
  Completion metric: all lanes above are either complete or intentionally
  deferred with explicit issue/plan text; PR is then marked ready again.

Best next steps:

1. Commit the README correction and this reset plan.
2. Restore or regenerate the full runtime panel and remove the reduced
   two-case panel from the README path.
3. Run the smallest current-vs-main benchmark first:
   `input.nfp4_QH_warm_start`, `input.circular_tokamak`, `input.solovev`, and
   `input.LandremanPaul2021_QA_lowres`.
4. If current branch is slower than main, profile before adding more figures.
5. Build the AD-vs-FD evidence data generator and figure once the runtime
   regression question is understood.

User decisions needed:

- Decide whether VMEC++ must be present for every full-matrix benchmark row or
  whether VMEC++ can remain a partial/sanity column where the executable
  converges cleanly.
- Decide whether the README should show one combined runtime+memory panel or
  keep runtime in README and detailed memory tables in docs.

#### First reduced current-vs-main QH runtime probe

Scope:

- This is **not** the full README matrix and does **not** close the benchmark
  lane.
- It checks one low-mode row (`input.nfp4_QH_warm_start`) with the existing
  reduced benchmark harness, `ns_override=13`, input-deck iteration/tolerance
  budgets, VMEC2000 enabled, production `jit_forces=True`, and the same command
  on this PR branch and a clean `origin/main` worktree.

Results:

- Current branch warm: `vmec_jax 0.878 s`, VMEC2000 `0.211 s`, final
  `fsq_total=1.135e-13`.
- `origin/main` warm: `vmec_jax 0.844 s`, VMEC2000 `0.237 s`, final
  `fsq_total=1.135e-13`.
- Current branch cold/no-warmup: `vmec_jax 2.220 s`, VMEC2000 `0.246 s`.
- `origin/main` cold/no-warmup: `vmec_jax 2.203 s`, VMEC2000 `0.256 s`.
- A diagnostic no-JIT-force current run took `10.77 s`, confirming no-JIT and
  no-force-JIT paths are diagnostic only and should not be used as production
  benchmark claims.

Interpretation:

- No material PR regression is visible in this reduced QH probe; the current
  branch is about `4%` slower warm and about `1%` slower cold than `origin/main`
  with matching residuals.
- The larger gap versus VMEC2000 is still real for one-off solves and must be
  profiled in the full benchmark lane. The next performance check must use the
  historical full single-grid benchmark settings and include WOUT parity.

Commands:

- Current branch warm:
  `python tools/diagnostics/benchmark_fixed_boundary_runtime_and_residuals.py --cases nfp4_QH_warm_start --iters 450 --run-vmec2000 --vmec2000-exec ~/bin/xvmec2000 --jax-use-input-niter --vmec2000-use-input-niter --outdir outputs/pr20_current_vs_main_current_qh_jit`
- Main warm:
  `PYTHONPATH=$PWD python tools/diagnostics/benchmark_fixed_boundary_runtime_and_residuals.py --cases nfp4_QH_warm_start --iters 450 --run-vmec2000 --vmec2000-exec ~/bin/xvmec2000 --jax-use-input-niter --vmec2000-use-input-niter --outdir outputs/pr20_current_vs_main_main_qh_jit`
- Current/main cold variants used the same commands with `--no-warmup`.

Lane update:

- Full benchmark rerun and main-branch regression check: `5%`.
- Performance profiling and fix lane: `12%`.

#### PR #20 readiness gates and CI follow-up

Steps taken:

- Regenerated the full historical single-grid runtime/memory matrix for the PR
  branch and a clean `origin/main` worktree.
- Rendered the public combined runtime+memory README artifact and retired the
  reduced two-case public benchmark.
- Generated the AD-vs-central-FD evidence panel for fixed-boundary scalars,
  `DMerc`, `D_R`, and branch-local free-boundary direct-coil scalar
  derivatives.
- Ran converged WOUT parity for `nfp4_QH_warm_start`, `solovev`,
  `ITERModel`, and `LandremanPaul2021_QA_lowres`.
- Fixed the remaining CI failure in `tests/test_boundary_field.py` by making
  the FD comparison match the exact optimizer setup-map contract. Nonzero
  accepted VMEC updates remain validated by same-branch replay/JVP gates rather
  than fresh finite-difference solves through host branch rebuilding.

Results obtained:

- Full benchmark artifacts live in
  `docs/_static/figures/readme_runtime_compare.{png,csv,json}` with
  current-vs-main provenance in
  `docs/_static/figures/readme_runtime_compare_current_vs_main.{csv,json}`.
- VMEC++ is documented as an optional per-row column and appears only where it
  converges/supports the input.
- AD-vs-FD evidence artifacts live in
  `docs/_static/figures/readme_ad_fd_evidence.{png,csv,json}`.
- WOUT parity provenance lives in
  `docs/_static/figures/pr20_wout_parity_summary.json`.
- Local post-fix gates passed:
  `JAX_ENABLE_X64=1 pytest -q tests/test_boundary_field.py --tb=short`,
  `python -m ruff check tests/test_boundary_field.py`, and `git diff --check`.

Best next steps:

1. Push the CI fix and wait for the failed exact-coverage shard to rerun.
2. If all checks pass, mark PR #20 ready for review.
3. If another CI-only failure appears, fix only the narrow failing gate and keep
   the branch-local/free-boundary derivative claims conservative.

User needs:

- None at this point; PR readiness is gated only on CI passing after the fix.

Lane update:

- README full benchmark restoration: `100%`.
- Full benchmark rerun and main-branch regression check: `100%`.
- WOUT parity on benchmark rows: `100%`.
- Differentiation AD-vs-FD evidence panel: `100%`.
- `DMerc`/`D_R` derivative validation: `100%`.
- Performance profiling and fix lane: `95%` for PR readiness.
- Differentiation architecture planning: `95%`.
- Docs/README consistency: `100%`.
- PR review readiness: `90%`, pending green CI and marking PR #20 ready.
