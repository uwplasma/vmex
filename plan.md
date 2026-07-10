# vmec_jax Overhaul Plan — from research prototype to a fast, differentiable, research-grade VMEC

**Audience:** an autonomous agent (Cowork) with full local access, executing this plan end-to-end.
**Owner / git identity:** all commits and pushes authored as `rogeriojorge` (GitHub user `rogeriojorge`;
`gh` is already authenticated as `rogeriojorge` with `repo`+`workflow` scopes on this machine).
**Status:** this file is the single active plan. It replaces every file in `vmec_jax_plan/`
(`plan.md`, `plan_freeb.md`, `plan_differentiability.md`, `discrete_adjoint_2506_plan.md`, and the
1.5 MB / 32,088-line `plan_research_grade_performance_differentiability.md`). Those files are Phase-0
inputs (read once, extract anything still relevant into scratch `NOTES.md`) and are then deleted from
the working tree **and from git history** in Phase 1. Keep this plan.md under 200 KB; it is a working
document, updated with a short status line per phase as work proceeds, and moved to `docs/dev/` (or
deleted) at the v0.1.0 release.

**Mirror status (2026-07-09): FINAL IMPLEMENTATION PLAN.** Straight-axis finite-beta free boundary
is a supported target under the anisotropic `fixed_flux_cut` model in Phase 5; implementation proceeds
through M0–M10 without reopening the archived solver architecture.

---

## 0. Mission statement

Turn `vmec_jax` into the reference JAX implementation of the VMEC ideal-MHD equilibrium solver:

1. **End-to-end differentiable** library API (fixed and free boundary), fast on CPU and GPU, using
   implicit differentiation of the converged equilibrium — not unrolled iteration tapes and not the
   current "fingerprint-gated branch-local" machinery.
2. **A non-differentiable CLI fast path** that may use Python-side control flow, host callbacks,
   early exits, and donated buffers to beat the differentiable path in wall time.
3. **VMEC2000 parity**: iteration prints, `wout_*.nc` contents, threed1-style summaries, and
   converged physics quantities match VMEC2000 within per-quantity validation tolerances.
4. **Performance parity or better** than VMEC2000 single-thread CPU on the benchmark suite,
   including multigrid (`NS_ARRAY` ladders), which is currently slower than VMEC2000 — a named bug.
5. **A small, readable codebase**: 30–40 Python files in `vmec_jax/`, ~25–30k library lines
   (revised 2026-07-09 from the original ≤15k after the fixed-boundary core alone measured ~10k
   well-documented lines; still a >4x reduction from **229 files / ~123k lines**), physically
   meaningful names, docstrings everywhere, ≥95% coverage without repo bloat (tests currently
   ~140k lines with coverage-padding files; target ≤ ~10k test lines).
6. **A ~10 MB repository** after a `git filter-repo` history rewrite (currently 57.4 MiB packed);
   large assets move to GitHub Releases; no Claude in the contributors panel.
7. **User-friendly docs** with full derivations (energy functional → forces → spectral condensation
   → preconditioner → time stepping → free boundary → adjoint), every equation linked to the
   implementing source.
8. **simsopt-style optimization examples** for QA / QH / QP / QI that start from a circular torus
   and converge to precise configurations in a single, short, readable script each.
9. **Free-boundary showcase**: β = 0→5% scans driven by ESSOS coils (stellarator + tokamak), run
   both through generated mgrid files and through direct Biot–Savart evaluation (no mgrid),
   demonstrating agreement — and that the direct path is the interpolation-free reference.
10. **Feature superset vs VMEC++** where VMEC++ has gaps: `lasym` (non-stellarator-symmetric),
    free boundary for tokamaks (`ntor=0`) and stellarators, fixed-boundary fallback on missing
    mgrid, spline/pedestal profile types, and a 2D preconditioner option — while borrowing VMEC++'s
    hot restart, JSON input, zero-crash policy, and validation methodology.
11. **Production mirror equilibria**: fixed- and free-boundary straight-axis mirrors at finite beta,
    axisymmetric and nonaxisymmetric, with open axial field lines, isotropic and consistent
    anisotropic pressure closures, external coils, implicit derivatives, and mirror-native output;
    plus closed toroidal stellarator–mirror hybrids using the ordinary VMEC backend.

Every decision below optimizes for: *simpler to use, fewer files, faster, more manageable*.

---

## 1. Ground truth — current state (audited 2026-07-08)

Facts established by direct audit; the executor should trust these and not re-derive them.

### 1.1 Repository

- 57.38 MiB packed git history; 89 MB working tree; 955 tracked files; single `main` plus 4 remote
  branches. Top history bloat: ~30 historical revisions of a ~1.7 MB `plan_differentiability.md`
  (now a 39-line stub — the blobs live only in history), multi-MB PNGs
  (`readme_best_optimization_qh.png` 2.0 MB, `minimal_seed_showcase_state_panel.png` 1.7 MB), and
  the 1.5 MB `vmec_jax_plan/plan_research_grade_performance_differentiability.md` at tip.
- **Claude in contributors:** authorship is clean (`git shortlog -sne --all` shows only Rogerio
  Jorge ×2 emails + Matthew Feickert ×1). Claude appears **only via 70 commit-message trailers**
  `Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>`. Fix = history rewrite of commit
  *messages* (strip the trailer), not a mailmap.
- **Branches:** `origin/codex/differentiability-refactor-plan`,
  `origin/feature/freeb-essos-coil-single-stage`, `origin/phase2/freeb-adjoint-validation` are all
  **0 commits ahead** of main → delete after the rewrite. The pre-rewrite mirror head is preserved
  locally as `archive/mirror-geometry-pre-rewrite` at `e4a7f05d`; its remote history was deleted
  because the rewritten `main` made a merge or rebase unusable. It contains a *native state block
  preconditioner*, *native spline matrix-free loop*, and *square hybrid solver method lanes*.
  Reuse is behavioral and selective: port compact equations, tests, and plotting ideas only after
  validating them against the new core. Never merge or broadly cherry-pick the archived history.

### 1.2 Library (`vmec_jax/`, 229 files, ~123k lines, 49 root modules)

- **Core physics exists and is comprehensive**: Fourier transforms (`kernels/tomnsp.py` 1513,
  `fourier.py`), geometry/metrics (`kernels/bcovar.py` 1434, `kernels/jacobian.py`), forces
  (`kernels/forces.py` 2010, `kernels/residue.py`), 1D preconditioner (`preconditioner_1d_jax.py`
  2247), fixed-boundary loop (`solvers/fixed_boundary/residual/` — iteration.py 2957, update.py
  2350, runtime.py 1957, policy.py 1708 — plus a parallel `scan/` tree), multigrid (`multigrid.py`
  331, a port of VMEC2000 `interp.f`), JAX NESTOR (`solvers/free_boundary/jax_nestor_operator.py`
  1671), mgrid IO + JAX interpolation, direct-coil Biot–Savart (`external_fields/coils_jax.py`,
  `essos_adapter.py`, routed via `MGRID_FILE='DIRECT_COILS'`), wout writer (`io/wout_files/` tree
  incl. jxbforce, mercier, bsubs, nyquist), CLI with `--plot/--booz/--doctor/--test`, and
  booz_xform_jax as a hard dependency.
- **The bloat** is: (a) duplicated NumPy/JAX kernel pairs (`kernels/numpy_forces.py` 1102 vs
  `kernels/forces.py`; `preconditioner_1d.py` vs `preconditioner_1d_jax.py`), (b) facade/compat
  shims (`solve.py`, `_compat.py`, `_solve_runtime.py`), (c) the entire "branch-local adjoint"
  apparatus (`discrete_adjoint.py` 2002, `implicit.py` 1865, `solvers/*/adjoint/` ≈ 30 files with
  fingerprints, replay tapes, gate reports), (d) overlapping driver layers (`driver.py`,
  `drivers/{staging,policy,lifecycle,...}.py`, `solvers/fixed_boundary/{residual,scan}/`), and
  (e) an oversized optimization/QI workflow layer (`optimization.py` 1993,
  `optimization_workflow.py` 1881, `quasi_isodynamic/` ≈ 6k lines, `solvers/free_boundary/
  coil_optimization.py` 2646).
- ~15 files violate the project's own 1500-line rule. mypy is configured with most error codes
  disabled and per-module `ignore_errors` on the core physics; ruff ignores E402/F821/F841/E501.
  The refactor must make these crutches unnecessary.

### 1.3 Tests, examples, docs

- `tests/`: 296 files, ~138k lines, codecov project gate 95%. Includes obvious coverage-padding
  ("`test_solve_wave3..10_coverage`" files) and 3–4k-line monsters
  (`test_direct_coil_finite_pressure_sensitivity.py` 4478). Only one tiny committed fixture
  (1.1 KB); large assets already come via `tools/fetch_assets.py` — keep that pattern.
- `examples/`: QA/QH/QP scripts are ~220–250 lines (close to acceptable); **QI is the mess**:
  `QI_optimization.py` (499 lines, ~49 helper/stage references) + per-NFP variants (256 lines each,
  3-phase circular→QP-basin→QI pipelines) + helper modules `qi_optimization_cases.py` (1066),
  `qi_staged_runner.py` (805). Free-boundary `free_boundary_QS_coil_optimization.py` is 1284 lines.
  `examples/data/` and `examples/data/single_grid/` duplicate ~40 input decks.
- `docs/`: Sphinx+furo, ~35 rst pages; theory pages exist (`theory.rst`, `equations.rst`,
  `algorithms.rst`, `discrete_adjoint.rst`) but are interleaved with internal plan/lane pages
  (`aggressive_performance_plan.rst`, `accelerated_merge_readiness.rst`,
  `optimization_sweep_results.rst`, `piecewise_omnigenous_plan.rst`) that must go.

### 1.4 Performance (README 2026-07-06 snapshot)

37 normalized single-grid rows (`NS_ARRAY=151`, `FTOL=1e-14`): **warm** vmec_jax beats VMEC2000 on
33/37; **cold** on only 14/37 (Python/JAX/XLA setup dominates small cases). VMEC++ converges cleanly
on only 17/37 rows. Multigrid ladders are slower than VMEC2000 multigrid (primary suspects in §5).

### 1.5 Differentiability

AD-vs-central-FD agreement is demonstrated for fixed-boundary geometry/profile scalars, QS/QI
residuals, `DMerc`, `D_R`, and *branch-local* direct-coil free-boundary scalars. Arbitrary
differentiation through adaptive solver-control branches is explicitly unclaimed. The whole
"promoted lane / fingerprint" claim policy disappears in this overhaul: a feature is either
supported and tested, or it does not exist in the public API. Implicit differentiation (Phase 4)
makes the adaptive-branch problem moot: only the converged fixed point defines the derivative.

### 1.6 External pieces

- **ESSOS** is cloned at `/Users/rogerio/local/ESSOS`. The mgrid PR is
  **uwplasma/ESSOS #33 "Add VMEC mgrid export from ESSOS coils"**, branch `feature/mgrid-from-coils`
  (open, +325/−21 over 7 files): adds `essos.mgrid.MGrid` (SIMSOPT-compatible cylindrical grid
  layout), `coils_to_mgrid(...)`, and `Coils.to_mgrid(...)`, validated by round-trip and SIMSOPT
  parity tests. Use this branch; if it merges, use main; push fixes to the PR (as rogeriojorge) if
  needed and pin the commit hash in example docstrings.
- **booz_xform_jax** exists at `uwplasma/booz_xform_jax` (pure JAX, MIT, pip-installable from
  GitHub; also a local clone at `/Users/rogerio/local/booz_xform_jax`) and is already a runtime
  dependency wired to `vmec --booz`.
- **jaxopt is deprecated.** Modern implicit-diff stack: **Optax** (first-order optimizers) +
  **Optimistix** (root/fixed-point solves with implicit differentiation) + **Lineax** (linear
  solves). We hand-roll a thin `custom_vjp` (§6) so none of these becomes a hard dependency;
  `lineax` is an acceptable optional backend for the adjoint linear solve.

---

## 2. Environment, locations, ground rules

### 2.1 Local paths

| Code | Path | Notes |
|---|---|---|
| vmec_jax | `/Users/rogerio/local/vmec_jax` | the repo being rewritten |
| VMEC2000 (STELLOPT) | `/Users/rogerio/local/STELLOPT/VMEC2000` | executable `Release/xvmec2000`; INDATA module in `LIBSTELL/Sources/Modules/vmec_input.f` |
| VMEC++ | `/Users/rogerio/local/vmecpp` | C++/Python reference; key docs: root `AGENTS.md`, `vmec/vmec/AGENTS.md`, `docs/fourier_basis_implementation.md` |
| ESSOS | `/Users/rogerio/local/ESSOS` | check out `feature/mgrid-from-coils` (PR #33) |
| booz_xform_jax | `/Users/rogerio/local/booz_xform_jax` | default runtime dependency |

### 2.2 Python environments

Create separate venvs (never the system Python):

```bash
python -m venv ~/.venvs/vmecjax && ~/.venvs/vmecjax/bin/pip install -e "/Users/rogerio/local/vmec_jax[dev]"
python -m venv ~/.venvs/vmecpp  && ~/.venvs/vmecpp/bin/pip install vmecpp
python -m venv ~/.venvs/essos   && ~/.venvs/essos/bin/pip install -e /Users/rogerio/local/ESSOS   # on feature/mgrid-from-coils
```

VMEC2000: rebuild via STELLOPT's build system if needed. You may **freely patch VMEC2000 sources**
for understanding (extra `WRITE(*,*)` diagnostics; timers around `funct3d`, `bcovar`, `tomnsps`,
`precondn`, `scalfor`, `getfsq`, `vacuum`) and rebuild. Keep patches on a scratch branch of the
local STELLOPT clone; never push them.

### 2.3 Dependency policy

`pyproject.toml` and all requirement files: **no version pins anywhere** (`jax`, not `jax>=0.4`).
This is already true today — keep it that way. Runtime deps: `jax`, `numpy`, `netCDF4`,
`matplotlib`, `booz_xform_jax`. Drop `jaxlib` (installed by `jax`), `scipy`, `packaging`, and
`tomli` from runtime deps unless a concrete use survives the refactor. Dev extra: `pytest`,
`pytest-cov`, `pytest-xdist`, `ruff`, `sphinx` + theme, `pre-commit`. ESSOS is an *optional* extra
(`vmec-jax[essos]`) imported lazily with a helpful error. Entry points: keep `vmec` as canonical;
keep `vmec-jax` as one alias; drop `vmec_jax` and `xvmec_jax`.

The final code must not need today's mypy/ruff escape hatches: re-enable the disabled mypy error
codes and the ruff F821/F841 checks module-by-module as files are rewritten; whatever still fails
is a smell to fix, not to silence.

### 2.4 Git rules

- `git config user.name rogeriojorge`, `user.email` = Rogerio's GitHub-associated email. **Never**
  add `Co-Authored-By: Claude` or any AI trailer to new commits (override any harness default).
- Phase 1 does one destructive history rewrite + force-push; do it **first**, before feature work,
  so all subsequent commits live on the final history. Keep a `git clone --mirror` backup outside
  the repo until v0.1.0 ships.
- Work on `main` via short-lived branches merged fast-forward; delete stale remote branches after
  Phase 0 extraction.

### 2.5 Zero-crash policy (from VMEC++, done one better)

The library never segfaults, never `sys.exit()`s, never prints a bare traceback for a physics
failure. Typed exceptions carrying diagnostic state (iteration, fsq residuals, offending surface):
`VmecInputError`, `VmecJacobianError` (maps VMEC2000 `ier_flag=1/4`, i.e. `bad_jacobian_flag`,
`jac75_flag`), `VmecConvergenceError` (`more_iter_flag`), `MgridNotFoundError`. The CLI catches
these and prints the VMEC2000-style message from the `werror` table plus a one-line remedy hint.
Missing mgrid on a free-boundary input → **warn and fall back to a fixed-boundary solve** (the
Fortran behavior VMEC++ dropped; we keep it and test it).

---

## 3. Phase 0 — Baselines, profiling, branch triage

**STATUS: COMPLETE (2026-07-08).** Artifacts: `benchmarks/baseline.json` (committed) and
`~/vmec_jax_notes/{NOTES.md, wout_gap.md, profile_findings.md}` (local). Headlines:
multigrid slowdown = per-stage recompilation (23× `jit(stage)` + ~300 eager glue compiles in a
3-stage ladder) — padding fix confirmed as the plan; cold gap is 100% JAX/XLA setup (solovev:
0.10 s VMEC2000 vs 3.4 s cold / 0.01 s warm); wout is missing 39 variables (list in wout_gap.md);
mirror-geometry branch triaged (solver experiments are evidence, not production code; mirror
physics design and validated tests are KEEP) and archived at `e4a7f05d`; QP optimization must
default to max_mode=3.

Most of the audit is done (§1). What remains before touching code:

1. **NOTES.md** (scratch area, not committed): distill the five `vmec_jax_plan/*.md` files (the
   32k-line log only for still-open items — most of it is historical micro-optimization diary),
   and skim `archive/mirror-geometry-pre-rewrite`: record what its native block
   preconditioner, spline matrix-free loop, and square-hybrid solver actually do and whether any
   idea survives into §5.4/§7.5. The archive stays local until the mirror migration is complete.
2. **Baseline benchmark script** `benchmarks/run_baseline.py` (committed, small; results JSON
   committed as `benchmarks/baseline.json`): fixed suite — `solovev`, `DSHAPE`, `HELIOTRON`,
   `cth_like_fixed_bdy` (+ lasym variant), `cth_like_free_bdy`, a DIII-D-like tokamak free-boundary
   case, `nfp4_QH_warm_start`, `w7x`, precise QA/QH (Landreman–Paul) — recording wall time
   (cold+warm), peak RSS, iterations to each `ftol` stage, for: VMEC2000 (xvmec2000,
   single-thread), VMEC++ (where it converges), vmec_jax CLI (CPU; GPU if available), each
   single-grid and multigrid. This script regenerates the README plot in Phase 10.
3. **Profile vmec_jax now** (`jax.profiler` + `py-spy`): per-case split of trace/compile vs run;
   count XLA compilations across a multigrid ladder (each `ns` stage recompiling is the #1 suspect
   for "multigrid slower than VMEC2000"); host↔device syncs per iteration in the hot loop (target
   0); transform vs preconditioner vs residual cost shares.
4. **Profile VMEC2000** (timer patches per §2.2, or gprof) on the same cases: per-part budget of
   funct3d / bcovar / transforms / precondn / tridslv / vacuum, and its multigrid stage timings —
   so we know exactly what budget each part of vmec_jax must beat.
5. **wout gap list**: `ncdump -h` a VMEC2000 wout vs a vmec_jax wout for the same case; diff
   against the authoritative variable list in Appendix A; record missing/mismatched variables.

**Exit criteria:** NOTES.md (branch verdicts, open TODOs worth keeping), `benchmarks/baseline.json`,
a ranked list of multigrid slowdown causes with profile evidence, and the wout gap checklist.

---

## 4. Phase 1 — Repository consolidation and history rewrite

**STATUS: COMPLETE (2026-07-08).** History rewritten with git-filter-repo (57.4 → 11.8 MiB
packed; 0 Claude trailers; contributors = rogeriojorge + matthewfeickert), force-pushed; 4 stale
branches deleted; vmec_jax_plan/, validation/, tools/diagnostics, examples/data/single_grid
removed; figures pruned/compressed (4.8 → 1.8 MB); interim CI (fast tests + smoke + build + size
check) green locally; pre-commit 200 KB guard added. Mirror backup at ~/vmec_jax_backup.git
(keep until v0.1.0).

Goal: one branch, ≤10 MB fresh clone, no Claude in contributors, drastically fewer files.

1. **Working-tree consolidation first** (so the rewrite also shrinks the tip):
   - Delete `vmec_jax_plan/` entirely (this plan.md replaces it), `validation/`, obsolete docs
     pages (plan/lane rst files, §11), `examples/data/single_grid/` (fold unique decks into
     `examples/data/`), and the archived optimization showcase PNGs.
   - Recompress every kept image (`oxipng -o4 --strip all` / `pngquant`); target <150 KB each; keep
     only figures referenced by README/docs.
   - `tools/` shrinks to `fetch_assets.py`, `make_release_assets.py`, `compress_figures.py`.
   - Large fixtures (reference wouts, mgrids, golden stdout captures, benchmark provenance) → a
     versioned GitHub Release bundle (`vmec-jax-test-assets-vX.tar.gz`, sha256-checked, cached
     under `~/.cache/vmec_jax/`). Keep in-repo only text input decks and one tiny mgrid (<1 MB) so
     `pytest -m quick` works offline.
2. **History rewrite** with `git filter-repo` on a fresh clone (destructive; mirror-backup first):
   - `--strip-blobs-bigger-than 300K`, plus explicit `--path <old plan files, old figures, old
     fixtures> --invert-paths`.
   - **Strip the Claude trailers** with a message callback:
     `git filter-repo --message-callback 'return re.sub(rb"\n?Co-Authored-By: Claude[^\n]*", b"", message)'`.
     Verify with `git log --all --format=%b | grep -ci co-authored-by: claude` → 0. (A mailmap pass
     can also normalize `rogerio.jorge@ist.utl.pt` vs `@wisc.edu` if desired — optional.)
   - Verify: `git count-objects -vH` ≤ ~10 MB packed; `git shortlog -sne` clean.
   - Force-push `main` + tags; delete the 4 stale remote branches; confirm the GitHub contributors
     page no longer lists Claude (may take a cache cycle).
3. **Hygiene going forward:** `.gitignore` covers `wout_*.nc`, `boozmn_*.nc`, `mgrid_*.nc` (except
   the tiny fixture), `figures/`, `results/`; pre-commit `check-added-large-files` (200 KB); a CI
   job fails if packed size exceeds 15 MB.

**Exit criteria:** fresh clone ≤ 10 MB; single `main`; contributors clean; CI green on the pruned
tree (tests may be temporarily reduced — full restructure lands in Phase 9).

---

## 5. Phase 2 — Core library refactor (architecture, naming, fixed-boundary parity)

**STATUS (2026-07-09): core landed, integration/perf hardening next.** `vmec_jax/core/` has 22
modules (~11.6k lines), each A/B-proven vs the legacy kernels (420+ tests) — including the solve
loop (solovev 215/215 iterations vs VMEC2000, cth 434, machine-precision wout parity), the
complete wout writer (all 39 missing variables; found legacy lasym output bugs: buco/jcur*/ctor
x1/2, jdotb x1/16, fast-bcovar bsubsmns corruption), mgrid IO/field (ESSOS PR#33 cross-verified),
and multigrid interpolation. Golden fixtures are a GitHub release (golden-v1) with sha256 fetch;
core suite is in CI with coverage reporting.
**(1) DONE + (2) DONE (2026-07-09, per-stage variant):** `SolverRuntime` is now a registered
pytree passed to module-level jitted lanes (`_while_lane`/`_block_lane`): RunSetup arrays +
`rcon0/zcon0` as data, `Resolution` + scalar config as meta, numpy trig/mode/gather tables
derived from meta via lru-cached `_static_tables`. Proven (tests/core_new/
test_multigrid_ladder.py, JAX_LOG_COMPILES subprocess counts): second solve at same Resolution
with different boundary = 0 compiles, 0.02 s on solovev. `solve(initial_state=...)` hot restart
added (boundary delta spread with the profil3d radial profile; bare edge-row swap measured
fsqr~0.5, spread ~4e-6; cth 1% RBC(0,1) restart 298 vs 434 iters — <25% is stepper-rate-limited,
needs Phase-4 Newton/2D precond). `multigrid.solve_multigrid` (runvmec.f ladder: skip-decreasing,
per-stage banners/ftol/niter, interp.f handoff) matches xvmec2000 ladders to machine precision
(cth 5/9/15 rel 9e-15; nfp4_QH 9/17/35 identical printed wb; NOTE ladder-vs-single-grid wb
scatter 1.36e-8 on nfp4_QH is inherent to VMEC — reproduced by xvmec2000 itself — m=1 freeze).
Compile behavior: one block-lane compile per distinct stage structure per session (3 for a
3-stage ladder, ~3 s each — cold ladder 9.3 s vs cold direct 5.1 s on cth; warm ladder 0.3 s,
0 compiles; direct-after-ladder reuses the final-stage executable, 0 compiles). **Follow-up
(padding, not attempted — >2 h):** ONE executable for all stages = pad radial arrays to
max(ns_array) as pytree *data* (s grids/hs/profiles already data, so per-stage values reuse one
executable) + a static `ns_active` mask threaded through the radial reductions (energies/force
norms in fields.py `energies_and_force_norms`, getfsq sums in residuals.py, precondn/lamcal
integrals and the tridiagonal jmax in preconditioner.py, jacobian half-mesh differences at the
padded rows) with masked rows pinned to identity updates in the loop body; validate vs the
per-stage ladder to 1e-15 per stage, then flip solve_multigrid to a single lane. (3)-(4)
unchanged below.
(3) parity breadth: 3D/lasym/finite-beta/ncurr=1/high-mode across all nine golden fixtures
(known gap: legacy lasym solver drifts ~5% on asym harmonics — validate the new core against
golden directly); (4) switch one public vertical slice (CLI fixed-boundary path) to the core,
then delete the corresponding legacy modules and migrate tests — repeat until the legacy tree is
gone. Implicit diff (Phase 3) starts once the residual API is frozen by (1)-(2).

### 5.1 Target layout (~30 files; one concern per file, none over ~1000 lines)

```
vmec_jax/
  __init__.py       # public API: run, Equilibrium, VmecInput, wout io, plotting entries
  input.py          # INDATA parser + VMEC++-compatible JSON input + VmecInput pytree; convert CLI
  profiles.py       # power_series, gauss_trunc, two_power, pedestal, cubic/akima splines, line_segment
  fourier.py        # (m,n) bookkeeping, parity tables, mscale/nscale, angle grids, m=1 constraint maps
  transforms.py     # totzsps/totzspa + tomnsps/tomnspa equivalents: batched DFT matmuls + FFT path
  geometry.py       # R,Z,λ real-space fields, jacobian tau/sqrt(g), metrics guu,guv,gvv   (jacobian.f)
  fields.py         # B^u,B^v, |B|, covariant B, pressure, energies wb/wp, tcon           (bcovar.f)
  forces.py         # MHD force kernels + spectral-condensation constraint force          (forces.f, alias.f)
  residuals.py      # fsqr/fsqz/fsql via getfsq, m=1 constraint, fedge                    (residue.f90)
  preconditioner.py # precondn/lamcal 1D radial precond, vectorized tridiagonal solve, 2D option (scalfor.f, tridslv, precon2d ideas)
  step.py           # damped 2nd-order Richardson step, dtau damping (ndamp=10), irst back-off (evolve.f, restart.f)
  solver.py         # single-grid loop: lax.while_loop core + host-blocked CLI variant    (eqsolve.f)
  multigrid.py      # NS_ARRAY ladder, coarse→fine interpolation, hot restart             (runvmec.f, interp.f)
  vacuum.py         # NESTOR: Green's function, analyt/scalpot, potvac solve              (NESTOR_vacuum/)
  freeboundary.py   # free-boundary iteration, ivac/nvacskip cadence, MagneticField protocol
  mgrid.py          # mgrid netCDF read/write, interpolated MagneticField
  coils.py          # ESSOS bridge: coils -> direct Biot-Savart field, write_mgrid from coils
  implicit.py       # custom_vjp implicit differentiation of the equilibrium (Phase 4)
  wout.py           # wout writer/reader — full Appendix-A variable set incl. jxbforce, mercier, bss
  printing.py       # VMEC2000-format iteration lines, stage banners, threed1 summary     (printout.f)
  plotting.py       # vmec --plot for wout and boozmn files
  boozer.py         # thin wrapper over booz_xform_jax (--booz)
  optimize.py       # objectives: QS ratio residual, QI (Goodman-style), aspect, iota, mirror; least-squares driver
  errors.py         # typed exceptions + werror message table
  cli.py            # `vmec` entry point
```

Names follow physics with a **VMEC-canonical glossary**: community-expected names stay (`ns, mpol,
ntor, nfp, lasym, iotaf, presf, rmnc, zmns, lmns, bmnc, ...`); internal Fortran temporaries get
descriptive names (`force_R_cos` not `armn`, `dpressure_ds` not `pres1`, `sqrt_g` not `gsqrt` — with
the glossary mapping both ways). Ship `docs/glossary.rst`: VMEC2000 name ↔ vmec_jax name ↔ meaning ↔
defining equation ↔ source location. Every module header docstring names its VMEC2000 counterpart
file(s) and the equations it implements.

Deletions (absorbed or dropped): `kernels/numpy_forces.py` (single JAX implementation, used by both
lanes), `preconditioner_1d.py` (keep only the JAX one), `solve.py`/`_compat.py` facades,
`discrete_adjoint.py` + both `adjoint/` trees + `optimizers/fixed_boundary/exact_replay*` (replaced
by `implicit.py`), `drivers/` + `solvers/fixed_boundary/{residual,scan}/` (merged into
`solver.py`/`step.py`), `quasi_isodynamic/` (distilled into `optimize.py`), `robust_coils.py`,
`optimization_workflow.py`, `finite_beta.py`/`bootstrap_current.py`/`redl_bootstrap.py` (move to
`optimize.py`-adjacent helpers only if an example/test uses them; otherwise drop — record in NOTES).

### 5.2 State and purity

- `EquilibriumState`: frozen pytree dataclass — spectral coefficients (`rmnc, zmns, lmns` +
  `rmns, zmnc, lmnc` when `lasym`), velocity `xcdot`, `time_step`, damping history `otau[10]`,
  iteration counters, residual history, `irst`-equivalent restart flag. All solver functions are
  pure `state -> state`.
- One set of spectral kernels shared by everything (solver, wout, plotting, objectives) — today's
  lane duplication is the main divergence source; eliminate it.
- Static configuration (resolutions, flags) in a hashable `VmecConfig`; **mode/radial arrays padded
  to the maximum multigrid resolution** so `ns` stages share one compiled executable (§7.1).

### 5.3 Two execution lanes, one physics

- `solver.solve(...)`: `lax.while_loop` over a jitted iteration, fully traceable — the
  differentiable API's forward solver.
- `solver.solve_cli(...)`: Python `while` around the same jitted *N-iteration block* kernel
  (e.g. `nstep=10`-aligned blocks via `lax.scan`), residuals checked on host between blocks —
  enabling exact-`ftol` early exit, live VMEC2000-format prints, buffer donation
  (`jax.jit(..., donate_argnums=...)`), and zero AD bookkeeping. Both lanes call identical physics
  kernels; a regression test asserts per-block state agreement to machine precision.

### 5.4 Algorithmic parity targets (the VMEC2000 details that matter — verified from source)

These constants/behaviors must be ported exactly; they are why VMEC2000 converges in few iterations:

- **Richardson step** (`evolve.f`): `dtau = min(|log(fsq1/fsq)|, 0.15)`, averaged over the last
  `ndamp=10` steps; `b1 = 1−dtau/2·Δt·⟨otau⟩`… concretely:
  `otav = mean(otau)`, `dtau = delt*otav/2`, `xcdot = (1−dtau)/(1+dtau)·xcdot + delt·gc`,
  `xc += delt·xcdot`.
- **Back-off** (`restart.f`): on `irst=2` (Jacobian sign change) restore saved state, zero
  velocity, `delt *= 0.90`, count `ijacob`; on `irst=3` (residual grew >1e4× best) restore,
  `delt /= 1.03`. Escalation in `eqsolve`: try `guess_axis` on first bad Jacobian; reset delt at
  `ijacob=25,50`; give up at 75 (`jac75_flag`).
- **Preconditioner cadence**: `precondn`+`lamcal`+force norms+`tcon` recomputed every
  `ns4=25` iterations, not every step.
- **tcon**: `tcon(js) = min(|ard/arnorm|,|azd/aznorm|)·tcon0-scaled·(32·hs)²` per surface,
  `tcon(ns)=½·tcon(ns−1)`; constraint force spectrally filtered to `m ∈ [1, mpol−2]` (`alias.f`)
  with `faccon(m)` weights.
- **m=1 constraint** (`residue.f90`): internally rotate `(gcr,gcz)_{m=1}` to `((gcr+gcz)/√2, 0)`;
  released when `fsqz<1e-6` etc. Boundary input applies `rbss=½(rbs+zbc)`-style conversion
  (`lconm1`, `readin.f`).
- **Radial start indices**: R,Z evolved from `jmin2`, λ from `jlam` (m-dependent; `vmec_params.f`).
- **1D preconditioner matrices**: `precondn` builds `axm/axd/bxm/bxd/cx` from
  `ptau = r12²·bsq·wint/gsqrt`-type integrals; `scalfor` forms tridiagonal
  `dx = axd + bxd·m² + cx·(n·nfp)²` with `edge_pedestal=0.05` and the ZC(0,0)(ns) `fac=0.25`
  stabilization; `tridslv` = Thomas algorithm vectorized over all (m,n) columns. λ uses the
  diagonal `faclam` from `lamcal` (`1/(blam·(n·nfp)² + clam·m² ± 2mn·nfp)`-shaped, √s-damped for
  m>16).
- **Free-boundary cadence** (`funct3d.f`): vacuum activates when `fsqr+fsqz ≤ 1e-3`; full NESTOR
  solve when `mod(iter2−iter1, nvacskip)==0`, incremental otherwise; adaptive
  `nvacskip = max(nvskip0, 1/max(0.1, 1e11·(fsqr+fsqz)))`; edge force `rbsq` from
  `bsqvac + presf(ns)` enters `forces` at js=ns; `rcon0,zcon0` ramp ×0.9/step in free-boundary.
- **Stopping**: converged when `fsqr, fsqz, fsql ≤ ftolv` simultaneously (physical, not
  preconditioned, residuals).

**Exit criteria for Phase 2:** all fixed-boundary benchmark cases (sym + lasym) converge with wout
parity vs VMEC2000 per Appendix-A tolerances; file/LoC budget met; ruff+mypy clean without today's
blanket ignores; every public function documented.

---

## 6. Phase 3 (interleave with Phase 2) — Differentiability done right

Adopt implicit differentiation of the equilibrium fixed point (DESC precedent; Skene & Burns
arXiv:2506.14792 for reuse-the-forward-machinery adjoints; jaxopt paper for the IFT formulation):

- Equilibrium = root of the preconditioned force residual `F(x, p) = 0`
  (`x` = spectral state, `p` = boundary coefficients / profile params / phiedge / coil currents &
  geometry / extcur). Wrap the solve in **`jax.custom_vjp`** (implemented in `implicit.py`):
  - forward: run the fast CLI-style solver (non-traced host loop is fine — it's opaque to AD),
    return converged `x*`;
  - backward: solve the adjoint linear system `(∂F/∂x)ᵀ λ = ḡ` matrix-free — `∂F/∂x`-vector
    products via `jax.vjp(residual_fn, x*)` — with **the 1D preconditioner as the preconditioner**
    for GMRES/BiCGStab (`jax.scipy.sparse.linalg.gmres`/`bicgstab`, or lineax); then return
    `−λᵀ ∂F/∂p` via one more VJP. Cost target: a handful of residual evaluations per gradient,
    O(1) memory in iteration count.
- This **replaces** `discrete_adjoint.py`, the replay tapes, fingerprints, and branch-local gates
  entirely. Multigrid/adaptive control lives inside the opaque forward solve; only the final fixed
  point defines the derivative (coarse stages are an initializer — stop-gradient by construction).
- Free boundary: identical scheme; NESTOR is inside `F` (traceable JAX code already exists), so
  coil parameters differentiate with no special handling. This deletes the "same-branch
  fingerprint-gated" hedging from the README.
- Provide `diff_mode="implicit"` (default) and `"unrolled"` (debug-only, small cases).
- **Permanent gradient tests** (rtol ≤ 1e-6 vs central FD): boundary coefficients → aspect, iota,
  QS residual, volume, `DMerc`; pressure profile / `pres_scale` → beta, wout scalars; coil currents
  and coil Fourier dofs (ESSOS) → free-boundary boundary shape and QS residual. Plus an adjoint
  linear-solve convergence test (preconditioned GMRES residual < 1e-10 in ≤ ~50 iterations).

**Exit criteria:** gradient tests pass; an L-BFGS boundary optimization with implicit gradients
matches/beats the FD-driven result at a fraction of cost; backward memory ≤ 2× forward.

---

## 7. Phase 4 — Performance (fast everywhere; multigrid faster than VMEC2000)

Ranked workstreams — confirm ranking against Phase-0 profiles before executing:

1. **Kill recompilation and host syncs.** One compiled solver for the whole `NS_ARRAY` ladder:
   pad radial arrays to `max(ns_array)` and mask; mode arrays sized once. No `.item()`/`float()`
   in the hot loop; prints via `jax.debug.callback` (jit lane) or between blocks (CLI lane).
   Measure: exactly one XLA compile per (mpol,ntor,lasym,lfreeb) tuple per session.
2. **Hot restart, VMEC++-style but stronger.** Public API `run(input, restart_from=output)`.
   VMEC++ restores only `rmnc/zmns/lmns` at a single matching `ns` and immediately activates the
   vacuum contribution; we do the same *plus* allow resolution changes by reusing our multigrid
   interpolation (radial interp in √s of scaled coefficients, odd-m axis extrapolation
   `2x₁−x₂`, spectrum pad/truncate — VMEC++ `_continuation.py` mechanics, already half-present in
   `multigrid.py`). Reset `delt` conservatively; carry λ. Hot restart powers the β-scan example and
   every optimization loop.
3. **Cold-start cost.** The 23/37 cold-slower rows are XLA setup. Mitigations: JAX persistent
   compilation cache enabled by default in the CLI (`JAX_COMPILATION_CACHE_DIR` under
   `~/.cache/vmec_jax/xla`), smaller/fused graphs (fewer distinct jitted entry points), lazy
   imports so `vmec --help` stays <100 ms.
4. **Transforms.** Keep batched-DFT matmuls with basis-baked weights (GPU-optimal, AD-friendly;
   VMEC++'s FFTX codelets only buy 10–20%). Add a `jnp.fft.rfft` path selected at trace time when
   `ntheta·nzeta` is large enough to win on CPU (benchmark the crossover once, hard-code the rule).
   Fuse totzsp→geometry→forces→tomnsp; verify with XLA cost analysis that intermediates stay
   fused.
5. **Preconditioner.** Exact 1D port (§5.4) with the Thomas solve vectorized over (m,n) — this is
   *the* convergence-rate feature; recompute on the ns4=25 cadence. Then an optional
   **`precond="2d"`**: VMEC2000's precon2d builds the Hessian by finite-difference "jogs" and
   block-tridiagonal LU (BCYCLIC); in JAX we get exact Hessian-vector products for free via
   `jax.jvp(residual_fn, ...)`, so implement 2D as matrix-free GMRES on the Newton step
   preconditioned by the 1D operator, activated below a `prec2d_threshold` on the finest grid
   (mirroring `ictrl_prec2d`/GMRES lanes in `evolve.f`/`gmres_mod.f`). Check NOTES for anything
   worth stealing from the `codex/mirror-geometry` native block preconditioner before deleting it.
6. **CPU threading / vectorization.** VMEC++ beats Fortran via OpenMP over radial partitions; our
   analog is XLA CPU multi-threading over the big batched matmuls — ensure kernels are large
   enough to parallelize, document `XLA_FLAGS=--xla_cpu_multi_thread_eigen=...`/thread pinning in
   docs/performance.
7. **Memory.** Donate state buffers in the CLI lane; float64 mandatory
   (`jax.config.update("jax_enable_x64", True)` at solver import); audit temporaries; peak-RSS in
   the benchmark output.

**Exit criteria:** CLI ≥ VMEC2000 speed on ≥80% of suite rows (cold, CPU, single-grid);
multigrid strictly faster than our single-grid *and* faster than VMEC2000 multigrid on the suite
median; GPU runs validated; README plot regenerated from `run_baseline.py`.

---

### 7.8 GPU profiling workstream (added 2026-07-09; hardware available)

`ssh office` (pop-os, 2x RTX A4000 16GB, repo at ~/vmec_jax) is available for GPU work. Reported
symptom: vmec_jax is sometimes SLOWER on GPU than CPU — cause unknown. Plan:

1. **Environment**: venv on office with CUDA jax (`pip install -U "jax[cuda12]"`), editable
   vmec_jax at current main, golden fixtures via the conftest downloader.
2. **Benchmark matrix** (extend `benchmarks/run_baseline.py` with a `--device {cpu,gpu}` axis and
   an office-runner mode): all baseline decks x {cpu, gpu} x {legacy solver, core solver
   cli/jit lanes} x {single-grid, multigrid} x {cold, warm}, recording wall, device memory,
   compile vs run time (jax.profiler), and per-iteration step time across problem sizes
   (ns=11 -> 151, low and high mpol/ntor) — locate the GPU crossover point.
3. **Hypotheses to test** for GPU-slower-than-CPU: (a) small kernels + dispatch overhead at low
   resolution (GPU should win only at high ns*mnmax); (b) host<->device syncs per iteration in
   the legacy driver; (c) the tridiagonal Thomas solve serializes over ns on GPU (lax.scan) —
   consider cyclic reduction or a batched parallel solve, or pin the tridiagonal solve to CPU;
   (d) float64 throughput on A4000 (GA104 fp64 = 1/32 fp32) — measure; experiment with fp32
   preconditioner + fp64 physics; (e) recompiles from per-solve closures (identity-cache landed;
   structural runtime caching pending).
4. **Deliverables**: `benchmarks/gpu_baseline.json`, a docs/performance section explaining the
   crossover + tuning guidance, and implementation changes ranked by measured impact feeding
   Phase 4.

## 8. Phase 5 — Free boundary, ESSOS, mirrors

1. **mgrid path** (VMEC2000-compatible, tokamaks *and* stellarators — VMEC++ can do neither
   `ntor=0` free-boundary nor lasym; we support both): validate on `cth_like_free_bdy`
   (sym + lasym) and a DIII-D-like tokamak mgrid case against VMEC2000.
2. **Direct-coil path**: `CoilField` (ESSOS Biot–Savart, differentiable in coil dofs) evaluated on
   the NESTOR grid each vacuum update; no interpolation. `mgrid.write_mgrid(field, ...)` generates
   VMEC2000-compatible mgrids from any field (use/align with ESSOS PR #33's
   `essos.mgrid.MGrid`/`coils_to_mgrid` so the two codes interchange files).
3. **Fallback**: `lfreeb=T` + missing mgrid → `MgridNotFoundError`-grade warning + fixed-boundary
   solve (§2.5).
4. **Single-stage optimization with ESSOS** (both directions the plan's development goals name):
   fixed-boundary single-stage (coil objectives + VMEC QS objectives on one gradient tape, boundary
   from coils via a quadratic-flux surface or direct constraint) and free-boundary single-stage
   (coils → direct field → free-boundary equilibrium → QS/aspect targets; gradients via §6).
   One example each, marked advanced.
5. **Mirror physics (production scope finalized 2026-07-09).** Open mirrors are not toroidal VMEC
   with a long major radius. They use a mirror-native inverse-coordinate backend, while sharing
   numerical and software components with the toroidal core. The closed stellarator–mirror hybrid
   remains on the ordinary VMEC backend.

   **STATUS (2026-07-09): M0 contracts and M1 foundation landed.** The clean backend now has
   mirror schema/config/end-cut contracts, increasing-order CGL differentiation/quadrature and
   interpolation, FFT theta derivatives, regular-axis 2D/3D geometry, the divergence-free
   contravariant field, and differentiable analytic one/two-coil benchmarks. Scientific tests cover
   polynomial exactness, integration by parts, spectral interpolation, analytic cylinder/flared/3D
   metrics, flux conservation, `div(B)`, direct Biot–Savart parity, and shape gradients. M2 is next.

   **M2 STATUS (2026-07-09): physical reference solve in progress.** Mass-conserving isotropic
   energy now uses VMEC-style radial half cells; an independent `curl(B)/mu0 x B - grad(p)` tensor
   residual caught and prevented a nonvariational full-mesh discretization. The host reference lane
   combines L-BFGS with an exact-JAX residual-Newton polish and accepts a result only by the
   nondimensional variational force (the mirror analogue of VMEC `fsq`). The separately differenced
   continuum tensor force is always reported as a spatial-verification residual and must converge
   under refinement. A perturbed cylinder reduced variational force from `5.39e-2` to below
   `1e-12` and its continuum tensor residual to `6.09e-14`;
   forced short runs raise a typed convergence error. The analytic two-coil flux-tube fixture now
   verifies throat/center field and paraxial residual scaling. A closed-form flared-tube MMS shows
   axial spectral force convergence (`1.39e-5`, `1.67e-8`, `1.77e-11` at `nxi=9,13,17`). The
   scalable lane uses exact JAX Hessian products with damped Newton-GMRES and a tensor-product
   radial/CGL stiffness inverse; a 585-unknown solve reaches `1.69e-15` variational force and
   `5.09e-13` continuum residual. M2 continuation and broader resolution studies remain open.

   **M3 STATUS (2026-07-09): closure layer landed; equilibrium coupling next.** Isotropic,
   ANIMEC bi-Maxwellian (Suzuki et al. Eqs. 4–6), and bilinear tabulated `p_parallel(s,B)` closures
   are differentiable PyTrees. `p_perp` is always derived from
   `p_parallel - B*partial_B(p_parallel)`; independent inconsistent moments cannot enter the API.
   Firehose `sigma` and mirror ellipticity are computed and tested, including the isotropic limit,
   trapped/passing continuity, tabulated interpolation, and coefficient gradients. The remaining
   M3 half-mesh ANIMEC energy now reproduces the isotropic passing-particle cylinder and matches
   central-difference shape derivatives. A coordinate-invariant tensor divergence computes
   `J x B - div(P)` with metric connection terms; it recovers a constant-pressure cylinder at
   `1.46e-13` normalized residual and separately verifies parallel force balance in shaped 3D
   states. The lateral diagnostic now independently reports plasma/vacuum tangency and the ANIMEC
   normal-stress jump. A perturbed finite-beta bi-Maxwellian cylinder now solves to `1.50e-15`
   variational force, `1.11e-13` tensor force, and `1.03e-11` parallel residual with valid
   ellipticity. M3 functional consistency is now closed for axisymmetry. Force norms exclude
   constrained side/end-cut nodes, while pointwise values remain available. For a nonuniform
   bi-Maxwellian profile, axial refinement at `ns=9` reduces tensor/parallel residuals from
   `2.52e-3/3.26e-3` (`nxi=9`) to `1.37e-3/6.50e-5` (`nxi=25`); at `nxi=25`, radial refinement
   reduces tensor residual from `1.77e-3` (`ns=7`) to `1.01e-3` (`ns=17`). The independently
   projected continuum tensor force is >`0.999` correlated with the discrete ANIMEC gradient and
   their weak-form discrepancy falls below 3% by `(ns,nxi)=(13,25)`. Thus variational force is the
   true discrete `fsq` convergence gate and tensor force is the documented spatial-refinement gate,
   as in M2. M4 may proceed; broader combined refinement remains part of M10 promotion evidence.

   **5.1 Supported physical model**

   - Coordinates are `(s, theta, xi)`, with `s in [0,1]`, periodic `theta`, and nonperiodic
     `xi in [-1,1]`. Use the VMEC radial mesh, Fourier in `theta`, and Chebyshev–Gauss–Lobatto in
     `xi`; retain an axial-basis protocol so multi-domain Chebyshev or B-splines can replace a
     poorly conditioned global polynomial without changing physics kernels.
   - The field representation is divergence-free by construction:
     `J B^s = 0`, `J B^theta = I'(s) - d_xi(lambda)`, and
     `J B^xi = Psi'(s) + d_theta(lambda)`. The lambda surface mean is the removed gauge mode.
   - The pressure tensor is
     `P = p_perp I + (p_parallel - p_perp) b b`. A `PressureClosure` supplies a generating energy
     density, `p_parallel(s,B)`, `p_perp(s,B)`, and analytic/JAX derivatives. It must enforce the
     parallel-force consistency relation
     `p_perp = p_parallel - B (partial p_parallel / partial B)_s`; independent arbitrary pressure
     arrays are invalid inputs. Ship an isotropic flux-function closure, an ANIMEC-compatible
     bi-Maxwellian closure, and a consistency-checked tabulated `(s,B)` closure for kinetic-code
     moments. The isotropic limit must reduce exactly to the scalar-pressure kernels.
   - The equilibrium residual is the physical tensor-force balance
     `R = J x B - div(P)`, projected onto admissible geometry/lambda variations. The variational
     energy is used when the closure provides a valid generating functional; the direct tensor
     residual is always computed independently as the convergence and validation diagnostic.
   - Firehose/mirror ellipticity indicators are mandatory outputs:
     `sigma = 1/mu0 + (p_perp-p_parallel)/B^2 > 0` and
     `partial_B(sigma B) > 0`. Failing either condition is a typed invalid-equilibrium result, not
     solver convergence. These checks are necessary model-validity gates, not a full stability
     claim.

   **5.2 Boundary model**

   - The side `s=1` is the lateral plasma-vacuum interface and is fixed or varied depending on the
     solve. Both plasma and vacuum fields are tangent there. Free boundary enforces the anisotropic
     normal-stress jump
     `p_perp + B_plasma^2/(2 mu0) = B_vacuum^2/(2 mu0)` and reports plasma, vacuum, and total
     normalized `B.n`. The first production profiles vanish smoothly at `s=1`; finite edge pressure
     is added only with an explicit surface-current model and diagnostic.
   - `xi=+/-1` are fixed computational cuts crossed by open field lines, not free plasma-vacuum
     interfaces or periodic caps. The production `fixed_flux_cut` policy prescribes the end
     geometry, normal magnetic flux, and lambda gauge from the coil/vacuum reference state; equal
     reflected data give the default up-down-symmetric mirror. Variations vanish at the cuts, while
     field lines and flux pass through. This is a static equilibrium model between specified end
     planes; end loss, sheath, sources, ambipolar potential, and transport are documented non-goals.
   - The vacuum annulus uses `B_v = B_coil + grad(nu)` with `laplacian(nu)=0`, Fourier×Chebyshev
     discretization, Neumann data on the moving side interface, and prescribed external-field data
     at the end cuts and outer computational boundary. Demonstrate convergence as that outer
     boundary is expanded before claiming an unbounded-vacuum result. NESTOR algorithms and
     cadence are reused where topology permits, but its toroidal Green function is not reused
     blindly for an open surface.

   **5.3 Architecture and reuse**

   Add only `vmec_jax/mirror/{model,basis,geometry,forces,vacuum,solver,output}.py` plus a small
   `__init__.py`. Reuse `core/coils.py`, transforms, radial interpolation, state stepping,
   tridiagonal/Krylov utilities, implicit-diff helpers, typed errors, and plotting styles. Extend
   `core/plotting.py` to dispatch mirror output rather than creating a plotting package. Root
   examples contain parameters and public calls only; geometry, normals, coil construction,
   summaries, tracing, and plots belong in library code. The archived head at `e4a7f05d` is an
   evidence source only: port equations or focused tests, never its solver stacks or broad commits.

   Use one physical residual and two execution lanes, as for toroidal VMEC: a JAX-traceable solver
   and a faster host-controlled CLI solver. Adapt the exact 1D radial preconditioner first; tensor
   it with an axial Chebyshev Helmholtz/line solve, then add matrix-free Newton–GMRES with that
   separable operator as preconditioner. The archived normal-equation `JᵀJ` block-CG method is not a
   production preconditioner because it squares the condition number.

   **5.4 Finite implementation sequence**

   1. **M0 — specification and migration.** Freeze signs, units, nondimensional residual norms,
      input schema, `mout` schema, end-cut contract, and analytic fixtures. Extract only the
      two-coil formulas, CGL tests, MMS cases, and plotting requirements from the archive.
   2. **M1 — basis and geometry.** Implement CGL nodes/differentiation/quadrature, transforms,
      axis regularity, axisymmetric and 3D embeddings, metrics, and divergence-free field. Test
      polynomial exactness, integration by parts, positive Jacobian, flux conservation, and
      spectral convergence before adding a nonlinear solve.
   3. **M2 — fixed-boundary isotropic axisymmetry.** Implement energy, tensor residual, lambda
      gauge, VMEC-like stepping, separable preconditioner, continuation, and diagnostics. Validate
      cylinder, flared tube, two circular coils (`B_z` on axis and low-radius `B_r,B_z`), and MMS.
   4. **M3 — anisotropic fixed boundary.** Implement isotropic, bi-Maxwellian, and tabulated
      closures; port the ANIMEC pressure/force identities from `fbal.f`, `bcovar.f`, `forces.f`, and
      `funct3d.f` rather than translating preprocessor structure. Verify closure derivatives,
      isotropic-limit identity, energy-gradient/tensor-force agreement, and ellipticity gates.
   5. **M4 — fixed-boundary 3D mirror.** Add nonaxisymmetric/helical boundaries and finite axial
      current. Demonstrate visible pitch, nonzero lambda response, and convergence under radial,
      poloidal, and axial refinement using the same solver and residual.
      **STATUS (2026-07-09): joint geometry/lambda solve landed.** The shared host solver now packs
      fixed-cut, axis-regular, surface-gauge-free lambda variables and uses the exact packed
      objective gradient for convergence. An `mpol=1` helical-boundary case with finite current
      converges below `1e-12`, has nonzero lambda and field-line pitch, and preserves end/gauge
      constraints. A manual `mpol=1,2,3` study converges every case below `6e-16`; the `mpol=2`
      and `mpol=3` energies agree to `~2e-13` relative, while lambda amplitude and maximum pitch
      change by about 2% and 3%. A mode-aware scalable lambda preconditioner and formal `ns/nxi`
      studies remain.
   6. **M5 — open-vacuum solver.** Implement the annular scalar-potential solve and couple direct
      coils/mgrid fields. Validate Laplace MMS, reciprocity, gauge removal, coil-only fields,
      side-interface `B.n`, end flux, outer-boundary convergence, and axisymmetric comparisons
      against direct circular-loop fields.
      **STATUS (2026-07-09): variational annulus and direct-coil tangency landed.** A moving-side,
      Fourier-CGL radial/axial map evaluates `B_external + grad(nu)` and solves the quadratic vacuum
      energy. The production boundary functional enforces zero correction flux on the outer/end
      cuts while natural variation enforces plasma-side tangency; fixed-potential MMS remains
      available. Exact annulus volume, linear harmonic/Laplace MMS, uniform-field cancellation,
      and differentiable direct `CoilSet` sampling are tested. For two circular end coils, normalized
      plasma `B.n` decreases `1.79e-2, 4.71e-3, 2.10e-3, 8.31e-4` over
      `(nrho,nxi)=(5,9),(7,13),(9,17),(13,25)`. Reciprocity, outer-radius convergence, mgrid parity,
      nonaxisymmetric coils, and plasma-interface coupling remain.
   7. **M6 — axisymmetric finite-beta free boundary.** Vary the lateral interface and interior
      state jointly, with beta continuation `0, 0.01, 0.03, 0.10` and hot restarts. Validate
      isotropic and anisotropic cases against an independently generated Pleiades/WHAM-style
      reference, paraxial pressure balance, outward flux-surface expansion, and the expected
      central diamagnetic trend `B0/Bvac approximately sqrt(1-beta)` in its validity regime.
   8. **M7 — nonaxisymmetric finite-beta free boundary.** Add helical coils/boundaries, then require
      3D force, interface, field-line, and resolution gates. This lane is supported only after M6;
      no axisymmetric boundary replicated in theta counts as a 3D validation.
   9. **M8 — toroidal stellarator–mirror hybrid.** Model the closed square/rounded-square torus with
      straight mirror sides and stellarator corners using ordinary VMEC Fourier equilibrium.
      Piecewise splines are low-dimensional axis/boundary design controls projected to Fourier.
      Validate mode convergence and `wout` parity with VMEC2000 before considering a native spline
      equilibrium state. Then run the 16-coil free-boundary beta scan using solved boundaries.
   10. **M9 — implicit differentiation and optimization.** Wrap the converged mirror residual in a
       `custom_vjp`; solve JVP/VJP systems matrix-free with the primal preconditioner. Validate
       boundary, pressure, current, and coil derivatives against central differences. Do not
       differentiate through iteration histories or restore fingerprint/replay machinery.
   11. **M10 — performance, outputs, and promotion.** Benchmark CPU/GPU cold/warm time, memory,
       scaling, and CLI versus JAX lanes; add mirror-native `mout` output, restart, `--plot`, docs,
       and short root examples. Remove obsolete archived implementations only after parity data are
       recorded. Mark the feature supported only when every gate below passes.

   **5.5 Convergence, validation, and presentation gates**

   - A requested `ftol=1e-12` means every active nondimensional physical force component is
     `<=1e-12`; optimizer status alone is never convergence. Report raw/component-normalized force,
     energy change, Jacobian minimum, step norm, linear residual, `B.n`, normal-stress jump, and
     closure consistency versus iteration. Stalling is a failed solve with a typed reason.
   - Reference cases pass `ns`, `mpol`, `nxi` (and `ntor` for 3D) studies. Observables converge at
     the expected order or spectrally until roundoff; a tighter nonlinear tolerance may not conceal
     discretization error. Research runs use `max_iter >= 1000` when needed, but CI uses small
     deterministic cases with the same equations.
   - Axisymmetric finite-beta free boundary must match independent reference curves for boundary,
     on-axis `B`, flux, and pressure to documented combined tolerances; M6 also requires monotonic
     continuation from beta zero, positive ellipticity indicators, and independence from reasonable
     initial boundaries. M7 requires the corresponding 3D manufactured/reference evidence.
   - Every beta is an actual equilibrium solve. Saved output drives field-line tracing and plots;
     no prescribed stand-in boundary is plotted as a result. Required figures are horizontal in
     mirror `z`, show coils, solved surfaces, field vectors and cap-to-cap field lines, `|B|`, cross
     sections, pressure moments, on-axis analytic/reference comparisons, residual histories,
     spectra, beta trends, current/twist, and mirror/well diagnostics. Render tests inspect artists
     and nonblank pixels; documentation commits only compressed showcase images.
   - Mirror outputs are `mout_*.nc`, never fake toroidal `wout`. They store geometry, fields,
     `p_parallel`, `p_perp`, closure metadata, force/interface histories, stability indicators,
     end-cut data, coils, and provenance. Toroidal hybrids continue to use `wout` and Boozer.
   - Supported means API/CLI documentation, restart compatibility, typed failures, >=95% focused
     coverage, CPU and office-GPU benchmarks, examples, and no known case labelled converged with a
     nonzero solver flag. Free-boundary axisymmetric and 3D mirrors are separate capability flags.

---

## 9. Phase 6 — Outputs: prints, wout completeness, JSON, Boozer

1. **Prints**: replicate VMEC2000 layout byte-for-column. The authoritative formats (from
   `printout.f`, `initialize_radial.f`, `runvmec.f`) are in Appendix B — implement `printing.py`
   directly from them (screen lane: `iter, fsqr, fsqz, fsql, RAX(v=0)[, ZAX], DELT, WMHD[, DEL-BSQ]`;
   threed1 lane adds preconditioned `fsqr1,fsqz1,fsql1`, `BETA`, `<M>`, `FEDGE`). Golden stdout
   captures of VMEC2000 per benchmark case go in the release asset bundle; tests diff structure
   exactly and values within tolerance. Also: `BEGIN FORCE ITERATIONS` banner, per-stage
   `NS = … NO. FOURIER MODES = … FTOLV = … NITER = …` banners, `VACUUM PRESSURE TURNED ON AT n
   ITERATIONS`, final timing + `EXECUTION TERMINATED NORMALLY`-style `werror` messages, and the
   threed1 summary file.
2. **wout completeness**: implement the full Appendix-A variable set (unit conventions included:
   `presf/pres/mass/jcuru/jcurv/ctor` divided by μ0 on write; `phipf/chipf` × 2π·signgs; `qfact =
   1/iotaf`; `lmns` half-mesh; `bsubsmns` full-mesh). Add VMEC++'s useful extras where free
   (`fsqt`, `wdot` already exist in VMEC2000; consider `lmns_full` as an extension attribute).
   Parity test: per-variable `CompareWOut`-style relative+absolute tolerances (global default +
   looser `currumnc/currvmnc`), following vmecpp-validation methodology. wout must load in simsopt
   and booz_xform unchanged.
3. **JSON input**: accept VMEC++'s exact schema alongside INDATA (keys in Appendix C; sparse
   `{"n":…,"m":…,"value":…}` boundary coefficients; dense axis arrays). `vmec convert input.foo
   --to json|indata` round-trips. We additionally support the profile types VMEC++ lacks
   (splines, pedestal, two_power, gauss_trunc…) — same key names as INDATA.
4. **Boozer**: `booz_xform_jax` stays a default dependency; `vmec --booz [--plot]` writes
   `boozmn_*.nc` + |B| contours + spectrum plots; one integration test.

---

## 10. Phase 7 — Examples (the public face; simsopt-simple)

Rules: each example is **one file**, < ~120 lines, top docstring stating goal/physics/expected
runtime/achieved result, public API only, smoke-tested in CI at reduced resolution (`--ci` flag).

```
examples/
  data/                          # text input decks + one tiny mgrid
  fixed_boundary_run.py          # run + plot + wout + --booz equivalent, ~30 lines
  free_boundary_mgrid.py         # cth_like via mgrid
  free_boundary_direct_coils.py  # same physics via ESSOS CoilField; compares to the mgrid run
  free_boundary_beta_scan.py     # THE README example (below)
  QA_optimization.py             # circular torus -> precise QA   (nfp=2)
  QH_optimization.py             # circular torus -> precise QH   (nfp=4)
  QP_optimization.py             # circular torus -> precise QP   (nfp=2 or 3)
  QI_optimization.py             # circular torus -> precise QI   (nfp=1)
  single_stage_fixed_boundary.py # advanced, ESSOS
  single_stage_free_boundary.py  # advanced, ESSOS
```

**Optimization examples** mirror simsopt's `QH_fixed_resolution.py` (66 lines: build equilibrium →
`QuasisymmetryRatioResidual(surfaces, helicity_m, helicity_n)` + aspect target → one least-squares
call). Ours: `vmec_jax.optimize.QuasisymmetryResidual(m, n)` with (QA: m=1,n=0; QH: m=1,n=−nfp;
QP: m=0,n=1) + aspect (+ iota/mirror targets), staged `max_mode` 1→2→3 continuation inside one
visible loop, gradient-based least squares using Phase-6 implicit gradients (highlight: no finite
differences, no MPI). **QI** uses a Goodman-style penalty (B-contour alignment, mirror-ratio /
target-B shaping, elongation + iota + aspect practical targets) implemented and documented inside
`optimize.py` — the example stays one file; continuation detail (grow ntor ~2× faster than mpol) is
a documented option, not example-level machinery. Today's QP-basin-then-QI trick, if still needed,
lives inside `optimize.py` with a docstring, honestly stated. Achieved objective values go in each
docstring and are loosely asserted by the CI smoke test.

**`free_boundary_beta_scan.py`** (featured in README): Landreman–Paul precise-QA coils from ESSOS
+ a simple ESSOS tokamak coil set; for β = 0,1,2,3,4,5%: free-boundary solve (hot-restarting each β
from the previous), once via generated mgrid and once via direct Biot–Savart; plot boundary
cross-sections evolving with β for both machines, overlay mgrid vs direct, report the difference
(direct = interpolation-free reference). Output: one compressed panel figure for the README.

---

## 11. Phase 8 — Tests and coverage (≥95%, no bloat)

- `tests/` mirrors the module layout: one file per module + `test_parity/` (wout/print goldens
  from the asset bundle) + `test_gradients/` (FD checks) + `test_examples.py` (smoke).
- Markers: `quick` (offline, <2 min, every push), `parity` (asset bundle), `slow` (nightly).
  Coverage gate ≥95% on `vmec_jax/` from quick+parity (`--cov-fail-under=95`).
- Property tests: transform round-trip (tomnsp∘totzsp = identity on band-limited data), residual
  invariance under nfp rotation, lasym-off ≡ symmetric path, CLI-lane ≡ jit-lane per block,
  JSON↔INDATA round-trip.
- Delete the wave/coverage-padding files and the 3–4k-line lane tests. Budget: ≤ ~10k lines total,
  including the mirror scientific-validation suite.
- Keep the `VMEC2000_INTEGRATION=1` opt-in gate that runs xvmec2000 side-by-side locally/nightly.

---

## 12. Phase 9 — Documentation overhaul

Sphinx with `furo` (already) or `sphinx-book-theme`, MathJax, `sphinx-copybutton`, `myst-parser`.
Landing page: what/why, 3-command quickstart, gallery. Delete the internal plan/lane pages
(`aggressive_performance_plan`, `accelerated_merge_readiness`, `optimization_sweep_results`,
`piecewise_omnigenous_plan`, `free_boundary_plan`, `discrete_adjoint` in its current form).

Structure:
1. **Getting started** — install, `vmec --test`, first run, plotting, Boozer.
2. **Tutorials** — one page per example, rendered figures, expected output.
3. **Theory & numerics** (the differentiator; every equation links to the implementing function):
   ideal-MHD energy functional and the Hirshman–Whitson steepest-descent moment method; flux
   coordinates and λ; Fourier representations, parities, lasym; force residuals and the m=1
   constraint; spectral condensation (`alias`/tcon); half/full radial meshes and jmin/jlam
   conventions; the 1D preconditioner derivation + tridiagonal solve + the 2D matrix-free
   extension; Richardson time stepping, damping, and irst back-off; multigrid + hot restart;
   NESTOR (Merkel Green's-function method) and the free-boundary cadence (ivac/nvacskip); the
   implicit-differentiation adjoint with derivation and cost analysis (cite Skene & Burns 2026,
   jaxopt, DESC); CLI lane vs differentiable lane.
4. **Reference** — API autodoc; input reference (every INDATA variable + JSON schema); wout
   variable reference (Appendix A rendered); glossary; CLI reference.
5. **Performance & validation** — benchmark methodology, plots, parity tables, GPU notes,
   profiling guide (including how VMEC2000 was instrumented).
6. **Developer guide** — architecture map (module ↔ VMEC2000 subroutine), adding an objective,
   release checklist.

---

## 13. Phase 10 — Benchmarks, README, release

1. Re-run `benchmarks/run_baseline.py`; regenerate the README benchmark figure (compressed):
   vmec_jax CPU/GPU cold+warm vs VMEC2000 vs VMEC++, single-grid and multigrid.
2. Rewrite README: short pitch; install; quickstart; β-scan figure; one optimization figure;
   feature table vs VMEC2000/VMEC++ (differentiable ✓, lasym ✓, free-boundary tokamak+stellarator
   ✓, JSON ✓, hot restart ✓, zero-crash ✓, mgrid fallback ✓, GPU ✓, Boozer built-in ✓, spline
   profiles ✓, 2D preconditioner ✓); CLI reference; docs link. Remove all "research lane" language.
3. Upload the release asset bundle; tag `v0.1.0`; publish to PyPI; update the conda-forge
   feedstock; verify `pip install vmec-jax && vmec --test` in a clean venv.

---

## 14. Acceptance checklist (definition of done)

- [ ] Fresh clone ≤ 10 MB; single branch; zero `Co-Authored-By: Claude` trailers in history; Claude
      absent from the GitHub contributors panel; all new commits authored by rogeriojorge.
- [ ] `vmec_jax/` remains within the §0.5 budget of 30–40 files / ~25–30k lines after the
      mirror backend lands; no new mirror file exceeds ~800 lines; docstrings and source/equation
      cross-references are complete; ruff and mypy pass without blanket ignores.
- [ ] Fixed + free boundary (mgrid and direct-coil; tokamak and stellarator; sym and lasym)
      converge with wout + print parity vs VMEC2000 per Appendix-A tolerances; missing-mgrid
      fixed-boundary fallback works and is tested.
- [ ] Fixed-boundary axisymmetric mirror meets the component-wise `1e-12` force contract and its
      analytic field, fixed-flux end-cut, anisotropic-closure, and resolution tests;
      nonaxisymmetric mirror is supported only after its physical-residual and resolution gates.
- [ ] Straight-axis finite-beta free-boundary mirrors are supported in axisymmetric and 3D modes:
      solved lateral interfaces satisfy total `B·n` and anisotropic normal-stress balance, every
      beta scan point is a converged equilibrium, ellipticity gates pass, and axisymmetric results
      agree with independent Pleiades/WHAM-style reference data.
- [ ] Toroidal stellarator–mirror hybrid converges in the Fourier representation with VMEC2000
      parity; its spline parameterization demonstrably reduces design variables without changing
      the equilibrium equations. Free-boundary beta scans use solved surfaces and total `B·n`.
- [ ] CLI ≥ VMEC2000 speed on ≥80% of suite rows (cold CPU); multigrid faster than VMEC2000
      multigrid on the suite median and faster than our own single-grid; GPU benchmarked;
      hot restart works and is used by examples.
- [ ] Implicit-diff gradients validated vs central FD (boundary, profiles, coil dofs, extcur);
      backward memory ≤2× forward; no fingerprint/replay machinery remains.
- [ ] QA/QH/QP/QI examples: single-file, <~120 lines, from circular torus to precise
      configurations with achieved values in docstrings; β-scan free-boundary example with ESSOS
      coils (mgrid + direct, agreeing) featured in README.
- [ ] VMEC++-schema JSON inputs accepted and round-trip converted; `--booz` works out of the box;
      typed zero-crash exceptions throughout.
- [ ] Coverage ≥95% with tests ≤ ~10k lines; goldens in release assets; CI green including example
      smoke tests and a repo-size check.
- [ ] Docs rebuilt per §12 with equations linked to source; README benchmark plot regenerated;
      v0.1.0 on PyPI + conda-forge.

---

## 15. Risks and mitigations

- **History rewrite is destructive** → fresh clone + `git clone --mirror` backup kept until
  v0.1.0; verify trailer count and pack size before force-pushing.
- **Multigrid slowdown may not be recompilation** → Phase-0 profiles decide; alternates to check:
  preconditioner recompute cadence, interpolation quality on restart, `delt` reset policy across
  stages (compare `irst`/`delt` handling line-by-line with `runvmec.f`/`restart.f` and VMEC++
  `_iteration.py`).
- **Deleting 100k+ lines can drop silent capabilities** → NOTES.md records every deleted module's
  purpose; parity + gradient + example tests are the safety net; the mirror-geometry branch and
  pre-rewrite mirror keep everything recoverable.
- **Adjoint linear solve may converge slowly near marginal equilibria** → 1D preconditioner as
  GMRES preconditioner; fall back to more inner iterations; document conditioning diagnostics.
- **QI from a circular torus is genuinely hard** → staged continuation + Goodman-style residual
  live in `optimize.py`, documented; if NFP-specific seeding is truly unavoidable, the example
  docstring says so honestly.
- **ESSOS PR #33 churn** → pin the commit hash in example docstrings; push fixes to the PR as
  rogeriojorge if needed.
- **float32 GPUs** → require x64 at solver import; document the performance implication.
- **Parity tolerance fights** → per-quantity rel+abs tolerances (CompareWOut methodology) with a
  looser current-density tolerance; never invent ad-hoc tolerances per test.
- **Open-end ambiguity** → support one explicit `fixed_flux_cut` model first: fixed geometry,
  prescribed normal flux, and no end-plane variations. State clearly that this is equilibrium in a
  truncated open tube, not a sheath, end-loss, source, or transport model.
- **Anisotropic closure inconsistency** → accept only closures generated by `p_parallel(s,B)` (or
  a thermodynamically consistent energy) and derive/check `p_perp`; reject independent tables and
  fail on firehose/mirror ellipticity violations.
- **Open-vacuum truncation error** → solve on expanding outer domains and require convergence;
  never reuse toroidal NESTOR kernels without open-surface MMS and flux tests.
- **High-beta bifurcation or solver stall** → beta continuation from vacuum, hot restart,
  separable preconditioning, and explicit ellipticity/conditioning diagnostics; do not return a
  best iterate as a converged equilibrium.

---

## 16. Key references (cite in docs)

- Hirshman & Whitson, Phys. Fluids 26, 3553 (1983) — steepest-descent moment method.
- Hirshman, van Rij & Merkel, Comput. Phys. Commun. 43, 143 (1986) — NESTOR.
- Merkel, J. Comput. Phys. 66, 83 (1986) — vacuum Green's-function method.
- Schilling et al., *The Numerics of VMEC++*, arXiv:2502.04374 — hot restart, JSON, zero-crash,
  validation methodology.
- Skene & Burns, *Fast automated adjoints for spectral PDE solvers*, arXiv:2506.14792 — adjoints
  reusing forward spectral machinery; template for `implicit.py`.
- Blondel et al., *Efficient and Modular Implicit Differentiation*, NeurIPS 2022 (jaxopt) — the
  IFT/custom_vjp formulation (note: jaxopt itself is deprecated; Optax/Optimistix/Lineax are the
  living successors).
- Dudt & Kolemen (2020); Conlin et al. (2023) — DESC: JAX equilibrium solver + implicit-derivative
  optimization precedent.
- Goodman et al., *Constructing precisely quasi-isodynamic magnetic fields*, JPP (2023),
  arXiv:2211.09829 — QI objective; ntor-faster-than-mpol continuation.
- simsopt `examples/2_Intermediate/QH_fixed_resolution.py` — style target for optimization examples.
- STELLOPT VMEC wiki (princetonuniversity.github.io/STELLOPT/VMEC) — INDATA semantics.
- Cooper et al., *Three-dimensional anisotropic pressure free boundary equilibria*, CPC 180,
  1524–1533 (2009), DOI 10.1016/j.cpc.2009.04.006 — ANIMEC energy, pressure closure, normal-stress
  interface condition, and anisotropic free-boundary reference.
- STELLOPT `_ANIMEC` sources `fbal.f`, `bcovar.f`, `forces.f`, `funct3d.f`, and `jxbforce.f` —
  implementation anchors for pressure moments, effective current, edge force, and diagnostics.
- Endrizzi et al., *Physics basis for the Wisconsin HTS Axisymmetric Mirror (WHAM)*, JPP 89 (2023),
  DOI 10.1017/S0022377823000806 — finite-beta anisotropic mirror validation context.
- Frank et al., *Integrated modelling of equilibrium and transport in axisymmetric magnetic mirror
  fusion devices*, JPP 91 E110 (2025), DOI 10.1017/S002237782510055X — Pleiades anisotropic force
  balance, diamagnetic expansion, paraxial check, and ellipticity criteria.
- Frank et al., *Nonlinear anisotropic equilibrium reconstruction in axisymmetric magnetic
  mirrors*, arXiv:2509.17288 — current WHAM high-beta reconstruction benchmark context.
- Pleiades (`github.com/eepeterson/pleiades`) — independent axisymmetric circular-coil, flux, and
  scalar-pressure regression reference; its Green-function algorithm is validation, not the 3D
  mirror backend.
- Trefethen, *Spectral Methods in MATLAB*; Boyd, *Chebyshev and Fourier Spectral Methods* — CGL
  differentiation, quadrature, filtering, and convergence references.

---

## Appendix A — wout variable checklist (from VMEC2000 `wrout.f`; implement all)

**Scalars:** `version_`, `input_extension`, `mgrid_file`, `pcurr_type`, `pmass_type`, `piota_type`,
`wb`, `wp`, `gamma`, `rmax_surf`, `rmin_surf`, `zmax_surf`, `nfp`, `ns`, `mpol`, `ntor`, `mnmax`,
`mnmax_nyq`, `iter2→niter`, `itfsq`, `lasym`, `lrecon`, `lfreeb`, `lrfp`, `ier_flag`, `aspect`,
`betatotal`, `betapol`, `betator`, `betaxis`, `b0`, `rbtor0`, `rbtor`, `signgs`, `IonLarmor`,
`volavgB`, `ctor` (/μ0), `Aminor_p`, `Rmajor_p`, `volume_p`, `ftolv`, `fsql`, `fsqr`, `fsqz`,
`nextcur`, `extcur(:)`, `mgrid_mode`; if lfreeb: `mnpd`, `nobser`, `nobd`, `nbsets`, `nbfld(:)`,
`curlabel(:)`.
**Mode arrays:** `xm`, `xn`, `xm_nyq`, `xn_nyq`.
**Axis:** `raxis_cc`, `zaxis_cs` (+ `raxis_cs`, `zaxis_cc` if lasym).
**Profile inputs:** `am`, `ac`, `ai`, `am_aux_s/f`, `ac_aux_s/f`, `ai_aux_s/f`.
**Radial 1D:** `iotaf`, `q_factor` (=1/iotaf), `presf` (/μ0), `phi`, `phipf` (2π·signgs·),
`chi`, `chipf` (2π·signgs·), `jcuru` (/μ0), `jcurv` (/μ0), `iotas`, `mass` (/μ0), `pres` (/μ0),
`beta_vol`, `buco`, `bvco`, `vp`, `specw`, `phips`, `over_r`, `jdotb`, `bdotb`, `bdotgradv`,
`DMerc`, `DShear`, `DWell`, `DCurr`, `DGeod`, `equif`.
**History:** `fsqt(:)`, `wdot(:)`.
**Free-boundary potential (lfreeb):** `potvac` sin (+cos if lasym), `xmpot`, `xnpot` — note VMEC++
skips these; we implement them.
**2D (mode×radius):** `rmnc`, `zmns`, `lmns` (half mesh), `gmnc` (half), `bmnc` (half),
`bsubumnc` (half), `bsubvmnc` (half), `bsubsmns` (full), `currumnc`, `currvmnc`, `bsupumnc`,
`bsupvmnc`; if lfreeb the `*_sur` surface arrays; if lasym all `*mns/*mnc` partners
(`rmns, zmnc, lmnc, gmns, bmns, bsubumns, bsubvmns, bsubsmnc, currumns, currvmns, bsupumns,
bsupvmns` + `*_sur`).
**Tolerances:** CompareWOut-style combined rel+abs per variable; global default (start 1e-10 for
geometry Fourier coefficients, 1e-8 for derived profiles) with a documented looser bound for
`currumnc/currvmnc`; calibrate against actual VMEC2000-vs-VMEC2000 run-to-run scatter in Phase 0.

## Appendix B — VMEC2000 print formats (from `printout.f` / `initialize_radial.f` / `runvmec.f`)

```
Headers:
  iter_line  = "  ITER    FSQR      FSQZ      FSQL   "
  fsq_line   = "   fsqr      fsqz      fsql      DELT    "
  raxis_line = "RAX(v=0) "        zaxis_line = "  ZAX(v=0)      "
  threed1 hdr suffixes: "WMHD      BETA      <M>   DEL-BSQ   FEDGE" (freeb)
                        "WMHD      BETA      <M>"                  (fixed)
Data lines (Fortran FORMATs):
  screen fixed sym : (i5,1p,3e10.2, e11.3, e10.2, e12.4)             ! ITER FSQR FSQZ FSQL RAX DELT WMHD
  screen freeb sym : (i5,1p,3e10.2, e11.3, e10.2, e12.4, e11.3)      ! + DEL-BSQ
  screen lasym     : adds ZAX(v=0) column (2e11.3)
  threed1 sym      : (i6,1x,1p,7e10.2, e11.3, e12.4, e11.3, 0p,f7.3, 1p,2e9.2)
Banners:
  ' FSQR, FSQZ = Normalized Physical Force Residuals' /
  ' fsqr, fsqz = Preconditioned Force Residuals' / 23('-') /
  ' BEGIN FORCE ITERATIONS' / 23('-')
  '  NS = ',i4,' NO. FOURIER MODES = ',i4,' FTOLV = ',1p,e10.3,' NITER = ',i6   (per stage)
  '  VACUUM PRESSURE TURNED ON AT ',i4,' ITERATIONS'
```

## Appendix C — VMEC++ JSON input keys (adopt verbatim; extend for our extra features)

`lasym, nfp, mpol, ntor, ntheta, nzeta, ns_array, ftol_array, niter_array, delt, tcon0, aphi,
phiedge, nstep, pmass_type, am, am_aux_s, am_aux_f, pres_scale, adiabatic_index(=gamma), spres_ped,
ncurr, pcurr_type, ac, ac_aux_s, ac_aux_f, curtor, piota_type, ai, ai_aux_s, ai_aux_f, bloat,
raxis_c, zaxis_s, raxis_s, zaxis_c, rbc, zbs, rbs, zbc, lfreeb, mgrid_file, extcur, nvacskip,
free_boundary_method, lforbal` — boundary coefficients as sparse `{"n": int, "m": int,
"value": float}` lists; axis arrays dense length `ntor+1`. Our extensions (documented, ignored by
VMEC++): `precon_type`, `prec2d_threshold`, spline profile types, mirror-geometry keys (Phase 5.5).

## Appendix D — VMEC2000 algorithm constants (parity-critical; from source)

| Item | Value / rule | Source |
|---|---|---|
| Richardson damping window | `ndamp = 10` | vmec_params.f |
| dtau cap | `bprec·0.15` (bprec=6 with 2D precond on) | evolve.f |
| Velocity update | `xcdot ← (1−dtau)/(1+dtau)·xcdot + delt·gc; xc += delt·xcdot` | evolve.f |
| Precond recompute cadence | every `ns4 = 25` iterations | bcovar.f |
| Jacobian reset (`irst=2`) | restore state, zero velocity, `delt ×= 0.90` | restart.f |
| Residual-growth back-off (`irst=3`) | growth >1e4× best after >10 steps; `delt /= 1.03` | evolve.f/restart.f |
| Escalation | guess_axis on 1st bad jac; delt reset at ijacob=25,50; abort at 75 | eqsolve.f |
| Constraint scaling | `tcon(js)=min(|ard/arnorm|,|azd/aznorm|)·tcon0-scaled·(32hs)²; tcon(ns)=½tcon(ns−1)` | bcovar.f |
| Constraint spectrum | m ∈ [1, mpol−2], weights `faccon(m)` | alias.f |
| m=1 constraint | rotate (gcr,gcz)_{m=1} → ((gcr+gcz)/√2, 0); input `rbss=½(rbs+zbc)` when lconm1 | residue.f90, readin.f |
| Edge pedestal / ZC00 stabilization | 0.05 / fac=0.25 | scalfor.f |
| λ precond | `faclam ∝ 1/(blam(n·nfp)²+clam·m²±2mn·nfp)`, √s damping m>16 | lamcal.f90 |
| Vacuum turn-on | `fsqr+fsqz ≤ 1e-3` | funct3d.f |
| Vacuum cadence | full solve when `mod(iter2−iter1,nvacskip)=0`; adaptive `nvacskip=max(nvskip0, 1/max(0.1,1e11(fsqr+fsqz)))` | funct3d.f |
| Free-bdy constraint ramp | `rcon0,zcon0 ×= 0.9` per iteration | funct3d.f |
| Convergence | `fsqr,fsqz,fsql ≤ ftolv` simultaneously | evolve.f |
| 2D precond activation | finest grid, `fsqr+fsqz+fsql < prec2d_threshold`; GMRES/CG/TFQMR lanes via precon_type | evolve.f, precon2d.f, gmres_mod.f |
