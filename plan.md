# VMEC-JAX Master Plan and New-Agent Handoff (Living Document)

Last updated: 2026-03-09
Primary owner: `vmec_jax` contributors
Canonical repo: `<repo-root>`

---

## 0) How to use this file

This file has two roles:
1. **Direct handoff prompt** for a new coding agent.
2. **Execution plan** with checkboxes that must be updated continuously.

Update rules:
- Mark tasks as `[x]` only after code + validation are complete.
- Keep an activity log at the end of each work session.
- Never leave parity or performance claims without command outputs and artifacts.
- Keep this file in sync with:
  - `<repo-root>/README.md`
  - `<repo-root>/docs/validation.rst`
  - `<repo-root>/docs/performance.rst`
  - `<repo-root>/docs/free_boundary_plan.rst`

---

## 1) Copy/paste prompt for a new agent

You are taking over the `vmec_jax` project. Read this whole file first.

### Mission
Deliver a **fully end-to-end differentiable**, **high-performance**, **memory-efficient**, and **well-tested** ideal MHD equilibrium solver in JAX that reproduces VMEC2000 behavior (algorithms, diagnostics, outputs, and practical workflows), for:
- fixed boundary and free boundary,
- axisymmetric and non-axisymmetric,
- `lasym=False` and `lasym=True`,
- single-grid and multigrid staging.

### Non-negotiable target behavior
- Default user path should be **easy** (`vmec_jax input.name`) and robust.
- Default solver selection (scan vs non-scan) should maximize both:
  - parity with VMEC2000,
  - runtime efficiency.
- Fixed boundary and free boundary should both be validated against VMEC2000.
- `wout_*.nc` parity should be tracked quantitatively, with known exceptions documented (near-axis and near-zero-denominator diagnostics).
- Code remains differentiable for optimization/ML workflows.

### Project context
This code solves ideal MHD equilibrium for toroidal plasmas and is intended to mimic VMEC2000 first, then go beyond it by:
- seamless autodiff support,
- easier installation and Python/JAX integration,
- optimization-ready interfaces,
- modern diagnostics and CI.

### What to preserve
- VMEC2000-compatible numerics where parity is required.
- Current diagnostics infrastructure and manifest-based parity sweeps.
- Existing fixed-boundary parity behavior.
- Continuous tests/docs/CI consistency.

### What to avoid
- Hardcoded case-specific hacks.
- Environment-variable-only behavior for correctness (env vars can remain diagnostics/tuning knobs, not correctness requirements).
- Regression in differentiability or major memory/runtime regressions.

### Where everything lives
- Workspace root: `<workspace-root>`
- Main repo: `<repo-root>`
- VMEC2000 source/executable (source of truth):
  - source: `<vmec2000-source>`
  - executable: `<vmec2000-exec>`
- Other local VMEC trees (non-canonical for parity unless explicitly tested):
  - `<alt-vmec-tree-1>`
  - `<alt-vmec-tree-2>`
- Examples/data:
  - `<repo-root>/examples/data`
- Diagnostics tools:
  - `<repo-root>/tools/diagnostics`
- Tests:
  - `<repo-root>/tests`
- Docs:
  - `<repo-root>/docs`

### Immediate operating workflow
1. Run parity on target cases (fixed/free, lasym true/false).
2. Localize first mismatch with dumps.
3. Patch numerics to match VMEC2000 formulas/order.
4. Re-run parity + tests + docs build.
5. Commit small, push frequently.
6. Update this file and docs status.

### Definition of done for each block
- Comparator passes with agreed tolerances.
- Tests pass (`pytest -q`).
- CI-equivalent checks pass.
- Relevant docs updated.
- No performance regression on baseline cases.

---

## 2) Project goals and product vision

### 2.1 Short-term goal (parity-first)
Match VMEC2000 for fixed and free boundary across practical case matrix, including per-iteration diagnostics and final `wout` fields.

### 2.2 Medium-term goal (production-grade optimizer backend)
Provide stable differentiable APIs for inverse design/optimization loops with robust reproducibility and automated parity regression gates.

### 2.3 Long-term goal (beyond VMEC2000)
Keep VMEC parity mode, while introducing better robustness, richer outputs, easier deployment, and optimization-native workflows (autodiff and implicit differentiation at scale).

---

## 3) Codebase map and key files

### 3.1 Core solver paths
- Driver and orchestration:
  - `<repo-root>/vmec_jax/driver.py`
- Nonlinear solver control and scan/non-scan paths:
  - `<repo-root>/vmec_jax/solve.py`
- Free-boundary coupling and vacuum/scalpot channels:
  - `<repo-root>/vmec_jax/free_boundary.py`
- Geometry/forces/jacobian/residual:
  - `<repo-root>/vmec_jax/vmec_bcovar.py`
  - `<repo-root>/vmec_jax/vmec_forces.py`
  - `<repo-root>/vmec_jax/vmec_jacobian.py`
  - `<repo-root>/vmec_jax/vmec_residue.py`
- Fourier/tables/transforms:
  - `<repo-root>/vmec_jax/vmec_tomnsp.py`
  - `<repo-root>/vmec_jax/fourier.py`
  - `<repo-root>/vmec_jax/vmec_realspace.py`
- Preconditioners:
  - `<repo-root>/vmec_jax/preconditioner_1d.py`
  - `<repo-root>/vmec_jax/preconditioner_1d_jax.py`
- Output handling (`wout` and derived channels):
  - `<repo-root>/vmec_jax/wout.py`

### 3.2 APIs for optimization/autodiff workflows
- Optimization-facing tools:
  - `<repo-root>/vmec_jax/optimization.py`
- Programmatic output adapter style object for optimization pipelines:
  - `<repo-root>/vmec_jax/booz_input.py`
  - (contains JAX-array export channels such as `rmnc/zmns/lmns`, Nyquist fields, `xm/xn`, `xm_nyq/xn_nyq`, `iota`, etc.)

### 3.3 Diagnostics and parity infrastructure
- Fixed-boundary comparator:
  - `<repo-root>/tools/diagnostics/vmec2000_exec_stage_trace_compare.py`
- Free-boundary comparator:
  - `<repo-root>/tools/diagnostics/vmec2000_exec_freeb_scalpot_compare.py`
- Manifest and sweep runner:
  - `<repo-root>/tools/diagnostics/parity_manifest.toml`
  - `<repo-root>/tools/diagnostics/parity_sweep_manifest.py`
- Example runtime/memory sweep:
  - `<repo-root>/tools/diagnostics/example_runtime_memory_matrix.py`

### 3.4 Core docs
- Main docs index:
  - `<repo-root>/docs/index.rst`
- Algorithms and numerics:
  - `<repo-root>/docs/algorithms.rst`
- Validation:
  - `<repo-root>/docs/validation.rst`
- Performance:
  - `<repo-root>/docs/performance.rst`
- Free-boundary implementation plan:
  - `<repo-root>/docs/free_boundary_plan.rst`

---

## 4) What has been implemented so far (major milestones)

### 4.1 Fixed boundary
- Implemented parity work for axisymmetric and non-axisymmetric paths.
- Added extensive dump/compare tooling for early-iteration and stage-level mismatch localization.
- Brought scan and non-scan loop behavior closer to VMEC2000 time-control semantics.
- Added LASYM support across core transforms/constraints/preconditioner/solver channels.
- Added scan minimal-history and host-sync reduction work for performance.

### 4.2 Free boundary
- Added typed free-boundary config parsing and runtime state (`LFREEB`, `MGRID_FILE`, `EXTCUR`, `NVACSKIP`).
- Added mgrid loading/validation/interpolation and boundary sampling.
- Implemented VMEC-like dense vacuum operator path and source/channel caching.
- Added VMEC-style `ivac/ivacskip/nvacskip` control behavior in solver.
- Added broad dump-to-dump parity diagnostics (`gsource_full`, `source_sym`, `bvec_nonsing_fouri`, `amatrix`, `potvac`, plus coupling channels).
- Improved axisymmetric free-boundary parity via `nv=1` handling in axis-current and greenf normalization.
- Tightened DIII-D per-iteration thresholds; documented remaining turn-on-window drift.
- Added new bexn decomposition diagnostics (`bexn_term_r/phi/z`, `snr/snv/snz`) for turn-on drift localization.

### 4.3 Process infrastructure
- Manifest-driven parity sweeps across topology/symmetry/boundary combinations.
- CI smoke dry-run gate for manifest (`.github/workflows/ci.yml`).
- Regular updates to docs and parity thresholds with measured values.

### 4.4 Representative recent commits (for orientation)
- `486e40d` freeb: add bexn decomposition diagnostics for turn-on drift
- `7911551` freeb: add per-iteration DIII-D parity thresholds
- `9773ed6` freeb: tighten DIII-D parity thresholds and document turn-on window
- `506f596` freeb: add axis-channel diagnostics and optional axis override hook
- `7fb11fd` add free-boundary evolve trace diagnostics for turn-on parity
- `fab804f` align nv=1 greenf source assembly with VMEC
- `9311d20` align axis-current nv=1 path and tighten parity thresholds

---

## 5) Current status matrix

### 5.1 Fixed-boundary status
- `lasym=False`: broadly strong parity on benchmarked cases.
- `lasym=True`: implemented and validated on available examples, continue regression expansion.
- Scan/non-scan auto-selection: active; must keep correctness as first priority and avoid hardcoded case IDs.

### 5.2 Free-boundary status
- Non-axisymmetric free-boundary channels: strong parity on the preserved
  CTH-like `lasym=False` fixture now tracked via
  `examples/data/input.cth_like_free_bdy`.
- Axisymmetric `lasym=True` DIII-D cases:
  - turn-on-window preconditioner cache reuse now matches VMEC2000 order/cadence,
  - iter 72 `scalfor` matrices match VMEC2000 to machine precision after `jmax=15 -> 16`
    cache reassembly,
  - direct iter-80 comparator on `input.DIII-D` is now near machine precision
    (`source_sym ~2.1e-12`, `bvec_nonsing_fouri ~2.1e-12`,
    `amatrix ~1.4e-13`, `potvac ~1.8e-12`).
- Axisymmetric `lasym=False` free-boundary parity:
  - manifest case `examples/data/input.DIII-D_lasym_false` is tight at iter 80
    (`source_sym ~8.4e-3`, `bvec_nonsing_fouri ~8.4e-3`,
    `amatrix ~1.7e-3`, `potvac ~9.4e-3`),
  - iter 100+ returns to near machine precision,
  - targeted manifest rerun passes at iter 80/100/120.
- Current free-boundary matrix gaps are split between:
  - remaining non-axisymmetric `lasym=True` late reuse-step field drift on
    `input.cth_like_free_bdy_lasym_small`:
    current parity thresholds pass at iter 80/100 and the manifest runtime
    thresholds now pass again, but iter 100 still shows reused field/coupling
    deltas (`source_sym ~2.6e-8`, `bvec_nonsing_fouri ~2.4e-8`,
    `amatrix ~1.3e-11`, `potvac ~1.0e-1`, `bsqvac ~3.1e-1`,
    `freeb_coupling_pgcon ~3.1e-1`),
  - coarse but valid post-turn-on parity on `input.stellcopt`
    (`source_sym ~2.7e-1`, `bvec_nonsing_fouri ~2.8e-1`,
    `amatrix ~1.2e-1`, `potvac ~3.6e-1` at iter 80),
  - remaining preserved-mgrid dependency for the local CTH-like `lasym=False`
    smoke fixture.
- Remaining work: tighten the non-axisymmetric `lasym=True` reuse-step
  field/coupling drift and its runtime cost, tighten post-turn-on
  `input.stellcopt`, and replace preserved local free-boundary fixtures with
  distributable inputs where practical.

### 5.3 Practical parity policy
- Compare with masks where numerically justified:
  - near-axis exclusions for some quantities (first 6 radial points as needed),
  - near-zero denominator aware relative metrics,
  - explicit caveat for channels expected to be near zero (e.g., `jdotb` in vacuum/no-current cases).

---

## 6) Runbook (daily commands)

### 6.1 Tests
```bash
cd <repo-root>
pytest -q
```

### 6.2 Docs build (CI-equivalent locale settings)
```bash
cd <repo-root>
LC_ALL=C LANG=C SPHINX_FAST=1 python -m sphinx -W -j auto -b html docs docs/_build/html
```

### 6.3 Fixed-boundary parity compare
```bash
cd <repo-root>
python tools/diagnostics/vmec2000_exec_stage_trace_compare.py \
  --input <repo-root>/examples/data/input.LandremanPaul2021_QA_lowres \
  --use-input-niter --max-iter 10 --dump-level full \
  --vmec2000 <vmec2000-exec>
```

### 6.4 Free-boundary parity compare (single iteration)
```bash
cd <repo-root>
python tools/diagnostics/vmec2000_exec_freeb_scalpot_compare.py \
  --input <workspace-root>/STELLOPT/BENCHMARKS/VMEC_TEST/input.DIII-D \
  --iter 80 --max-iter 80 \
  --vmec-exec <vmec2000-exec> \
  --workdir /tmp/freeb_diiid_iter80
```

### 6.5 Manifest sweeps
```bash
cd <repo-root>
python tools/diagnostics/parity_sweep_manifest.py \
  --tier smoke --vmec-exec <vmec2000-exec>
```

---

## 7) Scan vs non-scan strategy

Target policy:
- Keep both paths.
- Default path selection should be automatic and robust.
- Prefer scan when it is both stable and faster.
- Fall back to non-scan when scan drift or convergence instability is detected from solver signals (not hardcoded case IDs or fixed `ns` thresholds).

Current requirement to enforce:
- No manual env tuning should be required for correctness.
- Env vars are diagnostic/tuning tools, not mandatory for parity.

---

## 8) CI/CD, tests, validation, documentation strategy

### 8.1 CI summary
Workflow file:
- `<repo-root>/.github/workflows/ci.yml`

Current jobs:
- Parity manifest smoke dry-run.
- Python compile check.
- Unit/regression tests (`pytest -q`).
- Build wheel/sdist.
- Sphinx docs with warnings as errors.

### 8.2 Testing layers
- Unit tests: transforms, parser, preconditioner, constraints, helper math.
- Regression tests: resume behavior, wout compatibility, parity-sensitive channels.
- Physics sanity tests: invariants and expected profile behavior.
- Integration diagnostics: VMEC2000 comparator scripts and manifest sweeps.

### 8.3 Validation artifacts
Keep machine-readable summaries under
`<repo-root>/outputs/parity_sweeps/...`
and attach key metrics in docs.

### 8.4 Documentation update checklist
- `README.md`: short product-facing summary, quickstart, key figures only.
- `docs/*.rst`: detailed numerics, parity details, caveats, performance.
- `tools/diagnostics/README.md`: diagnostics usage and artifact interpretation.
- `plan.md` (this file): execution status + roadmap.

---

## 9) External ecosystem and competitor landscape (online review)

As of 2026-03-05, relevant equilibrium/optimization tool ecosystem includes:

### 9.1 Directly comparable equilibrium tools
- **VMEC2000 / PARVMEC** (reference code in STELLOPT).
- **VMEC++** (from-scratch C++/Python reimplementation, optimization-pipeline oriented).
- **DESC** (JAX-enabled stellarator equilibrium/optimization stack with autodiff).
- **SPEC** (MRxMHD/stepped-pressure with islands and chaos).
- **PIES** (3D equilibria with islands/stochastic regions).
- **SIESTA** (iterative equilibrium solver supporting island/stochastic structures).
- **HINT** family (equilibria without requiring nested surfaces; islands/chaos use cases).

### 9.2 Adjacent tools with partial overlap
- SIMSOPT (optimization framework integrating equilibrium and coil objectives).
- M3D-C1 (extended MHD, broader than equilibrium-only but relevant in workflows).

### 9.3 Market pull / need for this project
The fusion ecosystem indicates sustained need for fast, robust equilibrium tools that are optimization/ML-ready:
- Increased private fusion funding and company activity (FIA annual reports).
- Explicit policy/roadmap emphasis on integrated simulation and design loops (IAEA outlook, ARPA-E ecosystem analyses).
- Ongoing pressure for developer-friendly, reproducible, and scalable code paths in design pipelines.

### 9.4 Why `vmec_jax` can win
- VMEC-compatible numerics and outputs for trust/adoption.
- Differentiable-first architecture for gradient-based optimization and ML coupling.
- Python-native usability and easier deployment than traditional Fortran stacks.
- Ability to run CPU/GPU through JAX, with shared codepath.

### 9.5 Sources (online)
- VMEC (STELLOPT): <https://princetonuniversity.github.io/STELLOPT/VMEC.html>
- VMEC++: <https://github.com/proximafusion/vmecpp>
- The Numerics of VMEC++: <https://arxiv.org/abs/2502.04374>
- DESC docs (derivatives): <https://desc-docs.readthedocs.io/en/latest/dev_guide/notebooks/derivatives.html>
- DESC paper series (JPP/Cambridge):
  - Part I: <https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/desc-stellarator-code-suite-part-1-quick-and-accurate-equilibria-computations/69611B218B412BC279BDF2A080135718>
  - Part II: <https://www.cambridge.org/core/services/aop-cambridge-core/content/view/5766F6B713EC93D438A35705F2C1E861/S0022377823000399a.pdf/desc_stellarator_code_suite_part_2_perturbation_and_continuation_methods.pdf>
- SPEC docs: <https://princetonuniversity.github.io/SPEC/index.html>
- SPEC in STELLOPT: <https://princetonuniversity.github.io/STELLOPT/SPEC.html>
- PIES in STELLOPT: <https://princetonuniversity.github.io/STELLOPT/PIES.html>
- SIESTA free-boundary extension reference: <https://e-archivo.uc3m.es/entities/publication/4a6d63e3-0ba8-4dba-bda0-0372be4f935e>
- HINT usage example reference: <https://openresearch-repository.anu.edu.au/server/api/core/bitstreams/c8e55d06-e502-46fa-8d32-f7f30dc0a472/content>
- IAEA World Fusion Outlook 2024: <https://www.iaea.org/publications/15777/iaea-world-fusion-outlook-2024>
- FIA 2024 report launch: <https://www.fusionindustryassociation.org/fia-launches-2024-global-fusion-industry-report/>
- FIA annual report PDF: <https://www.fusionindustryassociation.org/wp-content/uploads/2024/07/2024-annual-global-fusion-industry-report.pdf>

---

## 10) Master task plan (update continuously)

Legend:
- `[x]` done
- `[ ]` not started
- `[-]` in progress

### 10.1 Project foundations
- [x] Build fixed-boundary solver core with VMEC-compatible workflows.
- [x] Add scan and non-scan implementations.
- [x] Add broad diagnostics and comparator infrastructure.
- [x] Add manifest-based parity sweep framework.
- [x] Add CI smoke parity runner.

### 10.2 Fixed-boundary parity and performance
- [x] Axisymmetric `lasym=False` parity baseline.
- [x] Axisymmetric `lasym=True` parity baseline.
- [x] Non-axisymmetric `lasym=False` parity baseline.
- [x] Non-axisymmetric `lasym=True` parity baseline.
- [-] Remove remaining `wout` channel gaps where VMEC2000 output exists and denominators are well-conditioned.
- [-] Continue scan/non-scan auto-selection robustness without hardcoded case hacks.

### 10.3 Free-boundary parity
- [x] Parse free-boundary inputs and mgrid integration.
- [x] Implement VMEC-like dense vacuum coupling path and channel caching.
- [x] Non-axisymmetric free-boundary parity on CTH-like reference.
- [x] Axisymmetric `lasym=True` DIII-D parity is tight post turn-on.
- [x] Reduce DIII-D turn-on-window drift around iter ~72-80 (`gsource/source_sym/bvec/potvac`).
- [x] Add bexn decomposition diagnostics to localize turn-on drift.
- [x] Add an automated axisymmetric `lasym=False` free-boundary case to the manifest.
- [-] Keep a stable non-axisymmetric `lasym=False` free-boundary smoke fixture in the current checkout without depending on preserved local mgrid artifacts.
- [x] Diagnose `input.stellcopt` missing VMEC scalpot dumps before treating it as a numerical parity regression.
- [-] Tighten the remaining non-axisymmetric `lasym=True` reuse-step field/coupling drift after matching VMEC restart cadence.
- [-] Tighten post-turn-on `input.stellcopt` parity now that iter-80 comparison is valid.
- [ ] Extend free-boundary parity matrix to additional non-axisymmetric `lasym=True` real-world cases beyond local synthetic case.

### 10.4 Differentiability and optimization UX
- [x] Expose optimization-oriented output adapter channels as JAX arrays.
- [x] Keep implicit differentiation examples and tests available.
- [-] Add end-to-end optimization tutorial(s) with realistic constraints (target iota, volume, fixed major radius).
- [ ] Add benchmark report comparing implicit vs explicit gradient workflows on same objective.

### 10.5 Performance and memory
- [x] Introduce many CPU-path improvements (tomnsps batching/cache, reduced host sync where safe).
- [x] Add profiling harnesses and trace infrastructure.
- [-] Continue eliminating avoidable host/device sync in parity-critical loops.
- [-] Re-profile scan and non-scan after each parity patch, preserving default robust behavior.
- [-] Keep a current cold-start runtime/memory matrix against VMEC2000 for the 8-way boundary/symmetry/LASYM coverage set.
- [ ] Optimize wout generation hot spots further (`forces_bcovar_s`, synthesis sections) while preserving parity.

### 10.6 Documentation and maintainability
- [x] Keep free-boundary plan updated with parity findings.
- [x] Keep manifest thresholds quantitative and per-case.
- [x] Restore CI-equivalent docs build after RST heading/indentation cleanup.
- [-] Keep README concise; move deep detail to docs.
- [ ] Add a dedicated docs page for scan vs non-scan equations/workflow with VMEC2000 mapping.
- [ ] Add docs page for free-boundary turn-on-window diagnostics and interpretation.

### 10.7 Release readiness gate for fixed+free boundary
- [ ] All core manifest cases passing at target tolerances with no manual env-tuning required for correctness.
- [ ] CI green on tests + docs + smoke parity.
- [ ] Documentation complete for user onboarding and developer parity workflow.
- [ ] Known limitations explicitly listed (near-axis caveats, near-zero denominators).

---

## 11) Time horizons

### 11.1 Short term (1-2 weeks)
- Finish free-boundary turn-on-window drift tightening for DIII-D-style axisymmetric LASYM.
- Expand free-boundary LASYM=true non-axisymmetric coverage in manifest.
- Keep per-iteration thresholds strict where parity is already excellent.

### 11.2 Medium term (1-2 months)
- Stabilize default automatic scan/non-scan selection by solver signals only.
- Finish optimization/autodiff examples and tests with robust APIs.
- Produce full parity and performance report artifacts for representative case matrix.

### 11.3 Long term (quarter+)
- Move from VMEC2000 parity mode to dual-mode architecture:
  - strict parity mode,
  - enhanced differentiable optimization mode (documented tradeoffs).
- Add more advanced ML/adjoint workflows and multi-case optimization examples.
- Evaluate GPU-focused kernel strategy for large optimization batches.

---

## 12) Current immediate next steps (concrete execution)

1. **Free-boundary LASYM non-axisymmetric expansion**
   - Add at least one additional finite-pressure non-axisymmetric `lasym=True`
     free-boundary case to manifest.
   - Set realistic thresholds and add to smoke/full tier as appropriate.
   - Replace the preserved-local `lasym=False` CTH-like mgrid dependency with a
     distributable fixture or documented stable source.
   - Tighten `input.stellcopt` post-turn-on parity now that the manifest compares
     iter 80 instead of pre-turn-on iterations.

2. **Default behavior hardening**
   - Ensure `vmec_jax input.name` is robust without manual env settings.
   - Keep adaptive scan/non-scan fallback based on solver signals.

4. **Validation closure**
   - Run:
     - `pytest -q`
     - docs build
     - targeted parity sweeps for changed cases
   - Update docs and this file with measured deltas and runtimes.

---

## 13) Activity log (append-only)

### 2026-03-05
- Ran the full Python test suite:
  - `pytest -q` -> `120 passed, 12 skipped, 61 warnings` in `42.23s`.
- Committed and pushed preconditioner dump instrumentation:
  - `e203795 solve: dump preconditioner matrices for parity tracing`.
- Tightened DIII-D free-boundary thresholds and added per-iteration thresholds in manifest.
- Added axis/turn-on documentation updates in free-boundary plan.
- Added new free-boundary bexn decomposition diagnostics:
  - JAX dump now includes `snr/snv/snz` and `bexn_term_r/phi/z` + `bexn_recon`.
  - Comparator now reports these channels directly.
- Confirmed DIII-D and DIII-D_reset manifest runs pass under tightened thresholds.
- Localized DIII-D turn-on drift further:
  - iter 72 raw `gc` matches VMEC2000 to machine precision,
  - first stable mismatch is in preconditioned `gc`, not force assembly or top-level free-boundary control flow.
- Added JAX preconditioner matrix dump support:
  - `VMEC_JAX_DUMP_PRECOND_MATS=1` writes `precond_mats_ns*_iter*.npz` with `ar/br/dr/az/bz/dz`, `jmax`, and cache-use flag.
- Fixed JAX lambda dump shape for axisymmetric `lasym=True`:
  - `VMEC_JAX_DUMP_LAM=1` now writes VMEC-style `ntmax=2` channels for `ntor=0, lasym=True`, enabling direct comparison to `lam_ns*_iter*.dat`.
- Re-ran the full parity manifest (`outputs/parity_sweeps/20260305_171806/summary.json`):
  - all 6 fixed-boundary cases passed,
  - `input.DIII-D` and `input.DIII-D_reset` passed at current tightened thresholds,
  - `freeb_nonaxis_lasym_false_cth_like` is currently skipped because the manifest points to a missing local VMEC++ fixture,
  - `input.stellcopt` currently fails because VMEC emits no scalpot dump in the comparator workdir,
  - `input.cth_like_free_bdy_lasym_small` is numerically excellent at iter 80 but still fails global status via iter-100 `potvac` and runtime thresholds.
- Re-ran the non-axisymmetric `lasym=False` free-boundary CTH-like case from the preserved local input fixture:
  - iter 53/54/60 all remain tight with `source_sym ~5.3e-7`,
    `bvec_nonsing_fouri ~5.5e-7`, `amatrix ~1.1e-13`,
    `potvac <= 3.6e-4`.
- Added a temporary axisymmetric `lasym=False` DIII-D symmetric benchmark input and compared it to VMEC2000:
  - iter 80 remains in the same turn-on window envelope as `lasym=True`
    (`source_sym ~8.4e-3`, `bvec_nonsing_fouri ~8.4e-3`,
    `amatrix ~1.7e-3`, `potvac ~9.4e-3`),
  - iter 100 and 120 return to near machine precision (`~1e-12` or better in compared free-boundary channels).
- Collected a direct cold-start runtime/memory matrix (`outputs/runtime_memory_matrix_20260305/summary.json`) across 8 coverage cases:
  - fixed-boundary default runs are currently about `26x`-`50x` slower than VMEC2000 and use about `6x`-`12x` more RSS,
  - free-boundary default runs are currently about `23x`-`98x` slower and use about `12x`-`16x` more RSS,
  - worst observed case in this matrix is the local non-axisymmetric `lasym=True` free-boundary solve (`~62s`, `~1.74 GiB` RSS vs VMEC2000 `~0.63s`, `~110 MiB` RSS).
- Fixed the current RST heading/indentation issues and revalidated the CI-equivalent docs build:
  - `LC_ALL=C LANG=C SPHINX_FAST=1 python -m sphinx -W -j auto -b html docs docs/_build/html` now passes.
- Added repo-local benchmark/parity inputs:
  - `examples/data/input.cth_like_free_bdy`,
  - `examples/data/input.DIII-D_lasym_false`.
- Updated the parity manifest:
  - repointed `freeb_nonaxis_lasym_false_cth_like` to `examples/data/input.cth_like_free_bdy`,
  - added `freeb_axisym_lasym_false_diiid_sym`,
  - moved `freeb_nonaxis_lasym_false_stellcopt` to post-turn-on iter 80.
- Re-ran the corrected free-boundary subset (`outputs/parity_sweeps/20260305_183853/summary.json`):
  - `freeb_nonaxis_lasym_false_cth_like` now passes in-manifest at iter 53/54/60 with
    `source_sym ~5.3e-7`, `bvec_nonsing_fouri ~5.5e-7`,
    `amatrix ~1.1e-13`, `potvac <= 3.6e-4`,
  - `freeb_axisym_lasym_false_diiid_sym` now passes in-manifest at iter 80/100/120 with
    the same turn-on envelope as the earlier manual spot-check and near machine
    precision by iter 100+,
  - `freeb_nonaxis_lasym_false_stellcopt` now runs as a valid post-turn-on comparison
    and passes current coarse thresholds at iter 80
    (`source_sym ~2.72e-1`, `bvec_nonsing_fouri ~2.80e-1`,
    `amatrix ~1.20e-1`, `potvac ~3.56e-1`).
- Added direct half-mesh metric dump support for JAX `bcovar` parity tracing:
  - `VMEC_JAX_DUMP_GMETRIC=1` writes `gmetric_iter*.dat` in VMEC-compatible
    `(js, lt, lz, pguu, pguv, pgvv)` format.
- Narrowed the VMEC-specific axisymmetric metric convention to the diagnostics path:
  - the live `vmec_bcovar` field metric remains post-`R^2` for `bsubv`, `wb`, scalar residuals,
  - `VMEC_JAX_DUMP_GMETRIC` now reconstructs the VMEC dump convention by removing the
    cylindrical `R^2` term from `pgvv` and zeroing the axis slot in the emitted file only.
- Re-ran the DIII-D iter-72 metric dump comparison after the dump-alignment fix:
  - `pguv` and `pgvv` now match VMEC2000 exactly,
  - the remaining first-order mismatch in this block is `pguu`
    (`max_abs ~1.30e-1`, `max_rel ~3.85e-1` in
    `/private/tmp/freeb_diiid_iter72_gmetric_after_fix/.../gmetric_iter72.dat`),
  - next localization target is therefore the axisymmetric `pguu` half-mesh
    assembly/order rather than the vac/sourceterm channels.
- Revalidated the edited docs/tests around this patch:
  - `pytest -q tests/test_dump_helpers.py tests/test_vmec_bcovar_smoke.py` -> `5 passed`,
  - targeted CI regression set
    (`test_force_norms_dynamic_parity`, `test_residue_getfsq_parity`,
    `test_resume_state`, `test_wout_parity_reference`) -> `11 passed`,
  - `LC_ALL=C LANG=C SPHINX_FAST=1 python -m sphinx -W -j auto -b html docs docs/_build/html` passes.
- Re-ran the two axisymmetric DIII-D manifest cases after the `bcovar` fix:
  - the current run is blocked at comparator level because no `jax_dumps` were
    emitted (`missing vmec_jax dump: .../jax_dumps/scalpot_jax_iter*.npz`),
  - this is presently a harness/runtime issue, not a measured metric-threshold failure.
- Fixed free-boundary `MGRID_FILE` resolution for driver/comparator workflows:
  - `load_config(path)` now resolves relative `MGRID_FILE` entries against the
    input file directory instead of the process working directory,
  - added a regression test covering `run_fixed_boundary(...)` from outside the
    input directory with `MGRID_FILE='mgrid_rel.nc'`.
- Re-ran the DIII-D free-boundary comparator from the repo root after the path fix:
  - JAX dumps are emitted again (`.../jax_dumps/scalpot_jax_iter80.npz` and
    `freeb_coupling_iter80.npz`),
  - the run now returns real parity numbers instead of a missing-dump failure,
    with iter-80 metrics back in the expected turn-on envelope
    (`source_sym ~8.29e-3`, `bvec_nonsing_fouri ~8.31e-3`,
    `amatrix ~1.51e-3`, `potvac ~9.45e-3`).
 - Closed the remaining DIII-D turn-on numerical gap in the preconditioner path:
   - `preconditioner_1d_jax.py` now caches full parity coefficients and reassembles
     `scalfor` matrices for a new `jmax` without forcing a fresh `bcovar` refresh,
     matching VMEC2000 stale-cache behavior at free-boundary turn-on.
   - direct iter-72 matrix comparison now matches VMEC2000 to machine precision
     with `jmax=16` and `used_cache=True`
     (`ar/dr/br/az/dz/bz rel ~1e-14`).
 - direct `input.DIII-D` iter-80 free-boundary comparator now returns near
     machine-precision parity across the prior turn-on blocker channels
     (`source_sym ~2.06e-12`, `bvec_nonsing_fouri ~2.07e-12`,
     `amatrix ~1.44e-13`, `potvac ~1.83e-12`).
   - validation on this patch:
     `pytest -q tests/test_dump_helpers.py tests/test_tcon_precondn_diag.py`
     -> `10 passed`,
     `pytest -q` -> `128 passed, 12 skipped`,
     `python tools/diagnostics/parity_sweep_manifest.py --tier smoke ...`
     -> `failed_cases=0`
     with summary at
     `outputs/parity_sweeps/20260305_211007/summary.json`.
- Tightened the non-axisymmetric `lasym=True` CTH-like free-boundary gap:
  - the 3D turn-on residual carry is now restricted to the non-axisymmetric
    path, which keeps `input.DIII-D` at machine precision while allowing the
    local `input.cth_like_free_bdy_lasym_small` case to enter VMEC-style
    reuse cadence.
  - same-iteration restart paths now invalidate cached free-boundary control
    tuples when `iter1` changes, so JAX recomputes `ivacskip` from the updated
    restart anchor just like VMEC2000.
  - direct solver history on `input.cth_like_free_bdy_lasym_small` now matches
    the VMEC control trace around the late window:
    `(94,94,3,0)`, `(95,95,3,0)`, `(96,96,3,0)`, `(97,97,3,0)`,
    `(98,97,3,1)`, `(99,99,3,0)`, `(100,99,3,1)`.
  - direct iter-99 comparator is back to near machine precision
    (`source_sym ~2.6e-8`, `bvec_nonsing_fouri ~2.4e-8`,
    `amatrix ~1.3e-11`, `potvac ~1.1e-7`, `bsqvac ~1.3e-7`).
  - direct iter-100 comparator no longer has the old order-one reuse failure;
    cached source/matrix channels are near machine precision
    (`source_sym ~2.6e-8`, `bvec_nonsing_fouri ~2.4e-8`,
    `amatrix ~1.3e-11`), with the remaining drift confined to the reused
    field/coupling channels (`potvac ~7.1e-3`, `bsqvac ~1.25e-2`,
    `freeb_coupling_pgcon ~1.25e-2`).
### 2026-03-06
- Split free-boundary turn-on restart behavior by topology/symmetry instead of
  using one global `iter1` policy:
  - all free-boundary paths still get the same-iteration soft restart at
    turn-on,
  - only the non-axisymmetric `lasym=True` path now preserves the pre-turn-on
    `iter1` anchor, matching the late VMEC reuse cadence without regressing
    DIII-D or the non-axisymmetric `lasym=False` smoke case.
- Added a unit test for the new turn-on `iter1` reset policy in
  `tests/test_free_boundary_wp0.py`.
- Revalidated the targeted free-boundary comparators after the control-flow
  change:
  - `input.DIII-D` iter 80 remains at near machine precision
    (`source_sym ~2.06e-12`, `bvec_nonsing_fouri ~2.07e-12`,
    `amatrix ~1.44e-13`, `potvac ~1.83e-12`),
  - `input.cth_like_free_bdy` iter 60 remains tight
    (`source_sym ~5.6e-7`, `bvec_nonsing_fouri ~5.8e-7`,
    `amatrix ~1.1e-13`, `potvac ~8.4e-4`),
  - `input.cth_like_free_bdy_lasym_small` iter 60 is now near machine
    precision in the source/matrix channels with much smaller field drift
    (`potvac ~4.0e-5`, `bsqvac ~2.0e-4`),
  - `input.cth_like_free_bdy_lasym_small` iter 100 now passes the current
    parity thresholds, with the remaining miss confined to reused field and
    coupling channels plus runtime thresholds.
- Re-ran manifest parity after the turn-on-control split:
  - smoke tier passes with `failed_cases=0`
    (`outputs/parity_sweeps/20260306_074540/summary.json`),
  - full-tier `freeb_nonaxis_lasym_true_cth_like_local` now fails only by
    runtime thresholds, not by parity thresholds
    (`outputs/parity_sweeps/20260306_073934/summary.json`).
- Re-enabled jitted force kernels on the free-boundary non-scan path after
  fixing the jitted wrapper to accept `freeb_bsqvac_half`.
- Revalidated the key free-boundary parity paths with the free-boundary JIT fix:
  - direct `input.DIII-D` iter 80 remains at near machine precision
    (`source_sym ~2.06e-12`, `bvec_nonsing_fouri ~2.08e-12`,
    `amatrix ~1.26e-13`, `potvac ~1.89e-12`),
  - the full-tier `freeb_nonaxis_lasym_true_cth_like_local` manifest case now
    passes with `failed_cases=0`
    (`outputs/parity_sweeps/20260306_075253/summary.json`).
- Measured a large default-path runtime drop on the heavy local free-boundary
  `lasym=True` example:
  - direct `run_fixed_boundary("examples/data/input.cth_like_free_bdy_lasym_small")`
    fell from about `71.5s` to about `37.8s` on the same local machine and
    iteration count.
- Added a checked-in bundled example runtime/memory sweep tool:
  - `tools/diagnostics/example_runtime_memory_matrix.py` benchmarks the default
    user path against VMEC2000 and records wall time plus `/usr/bin/time -l`
    peak-memory metrics.
- Fixed a default fixed-boundary fast-path regression exposed by the new sweep:
  - `_vmec_scale_m1_factors_from_mats(...)` now works on traced JAX arrays, so
    `run_fixed_boundary("examples/data/input.circular_tokamak", verbose=False)`
    no longer crashes in the scan path with `TracerArrayConversionError`.
- Collected the current bundled example runtime/memory matrix:
  - fixed-boundary summary:
    `outputs/example_runtime_memory_matrix_20260306_080658/summary.json`,
  - bundled free-boundary summary:
    `outputs/example_runtime_memory_matrix_20260306_083756/summary.json`.
- Current performance outliers from the bundled example sweep:
  - fixed-boundary:
    `input.up_down_asymmetric_tokamak` (~`57.8s`, ~`5.05 GiB`),
    `retired staged fixed-boundary stress case` (~`158.1s`, ~`6.97 GiB`),
    `input.LandremanSenguptaPlunk_section5p3_low_res` (~`45.8s`, ~`4.02 GiB`),
  - free-boundary:
    `input.DIII-D_lasym_false` (~`402.0s`, ~`7.98 GiB`),
    `input.cth_like_free_bdy` (~`40.4s`, ~`1.76 GiB`),
    `input.cth_like_free_bdy_lasym_small` (~`36.9s`, ~`1.54 GiB`).
- Identified an important next performance candidate for fixed-boundary
  `lasym=True`:
  - forcing `VMEC_JAX_LASYM_USE_SCAN=1` drops
    `input.up_down_asymmetric_tokamak` from about `57.8s` to about `7.5s`,
    but the current scan parity guard disables that fast path, so the default
    remains conservative until the LASYM scan parity evidence is tightened.
- Enabled automatic LASYM fast-path probing in performance mode without
  hardcoded case IDs:
  - LASYM fixed-boundary starts now infer the axis in performance mode,
  - the warmed dynamic selector compares full scan/non-scan residual traces
    before taking the fast path,
  - `input.up_down_asymmetric_tokamak` now defaults to scan at about `6.7s`
    instead of the earlier ~`57.8s`,
  - `input.basic_non_stellsym_pressure` now also defaults to scan,
  - `input.LandremanSenguptaPlunk_section5p3_low_res` still falls back
    conservatively because the warmed probe detects real drift.
- Re-ran the bundled-example benchmark matrix for the README refresh:
  - reference CPU host + VMEC2000 summary:
    `outputs/example_runtime_memory_matrix_cpu_20260306/summary.json`,
  - GPU all-example summary from the `reference GPU host` benchmark clone:
    `outputs/example_runtime_memory_matrix_gpu_20260306_summary.json`,
  - corrected GPU free-boundary rerun summary:
    `outputs/example_runtime_memory_matrix_gpu_freeb_20260306_rerun_summary.json`.
- Updated the README runtime/memory figure and appended a full bundled-example
  benchmark table:
  - figure:
    `docs/_static/figures/readme_runtime_compare.png`,
  - generated markdown table artifact:
    `outputs/readme_runtime_table_20260306.md`.
- Current benchmark snapshot from the refreshed matrix:
  - reference CPU host `lasym=True` fixed-boundary:
    `input.up_down_asymmetric_tokamak` ~`6.72s` / ~`0.89 GiB`,
    `input.basic_non_stellsym_pressure` ~`29.73s` / ~`3.22 GiB`,
    `input.LandremanSenguptaPlunk_section5p3_low_res` ~`46.77s` / ~`4.07 GiB`,
  - reference CPU host free-boundary:
    `input.DIII-D_lasym_false` ~`428.24s` / ~`7.36 GiB`,
    `input.cth_like_free_bdy` ~`41.83s` / ~`1.64 GiB`,
    `input.cth_like_free_bdy_lasym_small` ~`37.59s` / ~`1.47 GiB`,
  - GPU host (`reference GPU host`, dual RTX A4000):
    `input.DIII-D_lasym_false` ~`1602.31s` / ~`6.23 GiB`,
    `retired staged fixed-boundary stress case` ~`710.51s` / ~`6.16 GiB`,
    `input.basic_non_stellsym_pressure` ~`223.36s` / ~`3.90 GiB`.
- Completed a public-repo audit pass for portability and user-facing docs:
  - removed tracked absolute workstation paths and host-specific instructions,
  - bundled `mgrid_d3d_ef.nc` and `mgrid_cth_like.nc` so the shipped
    free-boundary examples are self-contained,
  - updated README/docs/package metadata to describe full fixed/free,
    axisymmetric/non-axisymmetric, and `lasym=False/True` support,
  - regenerated and visually QA’d the README runtime figure and the docs figure set,
  - validated the audit pass with `pytest -q` (`135 passed, 12 skipped`) and
    `SPHINX_FAST=1 python -m sphinx -W -j auto -b html docs docs/_build/html`,
  - checked the GitHub repo state with `gh`: no open Dependabot alerts,
    no code-scanning analysis uploaded, and secret scanning currently disabled.
- Reduced the default GPU overhead on LASYM fixed-boundary stages by making the
  dynamic scan selector accelerator-aware:
  - CPU backends keep the timed scan/non-scan probe,
  - accelerator backends now use a short parity-only probe by default,
  - measured cold-start GPU runtimes improved to about `16.9s`
    (`input.up_down_asymmetric_tokamak`), `71.4s`
    (`input.basic_non_stellsym_pressure`), and `27.5s`
    (`input.LandremanSenguptaPlunk_section5p3_low_res`) on the reference GPU host.
- Reduced the quiet fixed-boundary GPU scan overhead further by using
  backend-aware chunk targets and capping chunk lengths to the remaining work:
  - axisymmetric quiet accelerator scans now default to larger chunks,
  - 3D quiet accelerator scans now default to moderately larger chunks,
  - short parity probes no longer burn through hundreds of masked no-op steps.
- Updated cold-start GPU benchmark points after the chunking change:
  - `input.circular_tokamak` about `13.8s` / `1.97 GiB`,
  - `input.LandremanPaul2021_QA_lowres` about `33.9s` / `2.66 GiB`,
  - `input.up_down_asymmetric_tokamak` about `16.5s` / `1.60 GiB`,
  - `input.basic_non_stellsym_pressure` about `141.1s` / `3.68 GiB`,
  - `input.LandremanSenguptaPlunk_section5p3_low_res` about `77.1s` / `2.13 GiB`
    on the reference GPU host.
- Re-profiled the parity free-boundary GPU path to localize the remaining
  steady-state cost:
  - on `input.cth_like_free_bdy` with `performance_mode=False`, the
    scalar-history deferral patch (`70fc418`) was effectively neutral
    versus `285e9f5`:
    `~127.8s` cold / `~111.3s` warm versus
    `~123.5s` cold / `~112.6s` warm on the reference GPU host,
  - a `VMEC_JAX_TIMING=1` probe on `70fc418` showed the real bottleneck is
    `compute_forces` (`~0.278s/iter`) rather than preconditioning or the
    momentum update on this non-axisymmetric free-boundary case.
- Reduced large-`ns` axisymmetric free-boundary GPU cost by passing only the
  `bsqvac` edge slice into the force kernels:
  - `vmec_forces_rz_from_wout()` now accepts either a full half-mesh
    `freeb_bsqvac_half` array or just the edge slice,
  - the solver no longer rebuilds a mostly-zero `(ns, ntheta, nzeta)` array
    on every free-boundary iteration,
  - on the heavy `input.DIII-D_lasym_false` parity probe
    (`max_iter=10`, `multigrid=False`, reference GPU host),
    `compute_forces` dropped from `~5.79s` total (`~0.579s/iter`) on
    `70fc418` to `~2.58s` total (`~0.258s/iter`) on `f35ce44`,
    with `preconditioner` also dropping from `~0.675s` to `~0.324s`
    and `update` from `~0.914s` to `~0.535s`,
  - on the smaller `input.cth_like_free_bdy` case (`ns=15`) the same change
    is effectively neutral (`~111.3s` warm on `70fc418` vs `~111.4s` warm on
    `f35ce44`), which is consistent with the optimization targeting large
    radial grids.
- Updated user-facing performance docs to explain the current CPU-vs-GPU gap:
  - README now gives a concise explanation that the slow GPU rows are mostly
    parity-path solves with host-controlled VMEC2000 logic wrapped around many
    short float64 kernels,
  - `docs/performance.rst` now has a dedicated section explaining the current
    architectural reasons in detail, with measured free-boundary examples and
    the distinction between the fast scan path and the conservative parity path.
- Created branch `codex/nonparity-performance` for an accelerated, non-parity
  solver track aimed at final-equilibrium quality rather than
  iteration-by-iteration VMEC2000 agreement.
- Added `docs/aggressive_performance_plan.rst` and linked it from the docs
  index. The new plan defines:
  - acceptance criteria for accelerated mode (`fsq_total`, `wout`
    similarity, differentiability, and Jacobian-health gates),
  - architecture changes for order-of-magnitude runtime and memory reductions
    on CPU and GPU,
  - specific workstreams for device-resident control flow, continuation,
    preconditioner redesign, free-boundary acceleration, and gradient scaling,
  - benchmark and CI gates plus an expanded example matrix.
- Began Phase 0/1/2 implementation on `codex/nonparity-performance`:
  - added explicit `solver_mode` support to the CLI and Python API with
    `default`, `parity`, and `accelerated` policies,
  - added `tools/diagnostics/benchmark_accelerated_mode.py` to compare
    baseline vs accelerated mode on runtime, memory, `fsq_total`,
    convergence, and bundled-reference `wout` quality,
  - the first accelerated-mode implementation uses the masked VMEC-control
    scan path for fixed-boundary cases while skipping parity-oriented scan
    probes and scan-corrector overhead,
  - free-boundary accelerated mode currently reuses the robust baseline path.
- Initial CPU benchmark findings from `benchmark_accelerated_mode.py`:
  - `input.up_down_asymmetric_tokamak`: about `4.1x` warm speedup and `~0.68x`
    peak memory ratio vs the current default path, with the same final
    `fsq_total`,
  - `input.circular_tokamak`: converges cleanly with good final `wout`
    similarity (`~9.8e-6` max relRMS), but is not faster yet,
  - `input.LandremanPaul2021_QA_lowres`: approximately neutral,
  - the current `lasym=True` fixed-boundary quality gap on
    `input.up_down_asymmetric_tokamak` is shared by both the default and
    accelerated scan paths, so it is not introduced by the new mode,
  - free-boundary accelerated mode is currently a bookkeeping alias, not yet a
    faster controller.
- Continued Phase 2 accelerated-mode controller work:
  - re-probed the fully non-VMEC raw scan path directly on representative
    fixed-boundary cases (`input.circular_tokamak`,
    `input.LandremanPaul2021_QA_lowres`,
    `input.up_down_asymmetric_tokamak`),
  - current result: the raw scan path is substantially faster but not yet
    acceptable as a default accelerated controller because it can diverge
    badly in `fsq_total` and final `wout` quality,
  - kept the masked VMEC-control scan as the accelerated fixed-boundary
    controller for now and moved the next optimization step to bookkeeping
    reduction instead of changing nonlinear semantics.
- Reduced accelerated-mode result bookkeeping:
  - `solve_fixed_boundary_residual_iter()` now accepts explicit
    `light_history` and `resume_state_mode` controls instead of relying only on
    env vars,
  - accelerated runs now default to `light_history=True` and a
    `resume_state_mode="minimal"` payload, which drops the large parity-era
    momentum and preconditioner caches from the returned diagnostics while
    preserving the scalar continuation state needed for staged/continued runs,
  - added regression coverage so accelerated scan results expose
    `light_history=True`, `resume_state_mode="minimal"`, and remain restartable
    from the compact resume payload.
- Added an accelerated convergence target on total residual:
  - `solve_fixed_boundary_residual_iter()` now accepts an explicit
    `fsq_total_target`,
  - accelerated fixed-boundary stages use
    a scalar target derived directly from the same input tolerance budget,
    `fsq_total_target = 3 * ftol` for the three VMEC residual channels,
    while parity mode continues to use the original VMEC-style per-component
    `ftol` convergence,
  - this keeps the non-parity contract aligned with the accelerated plan:
    use `ftol` as the main user truth without introducing an absolute
    solver-side tolerance floor.
- Updated accelerated CPU smoke results after the total-residual target patch:
  - `input.LandremanPaul2021_QA_lowres`: approximately neutral with the
    current ftol-derived target,
  - `input.circular_tokamak`: approximately runtime-neutral with
    `~1.2e-5` reference-`wout` relRMS,
  - `input.up_down_asymmetric_tokamak`: about `4.11x` warm speedup with
    `~0.64x` memory ratio vs the current default path,
  - `input.cth_like_free_bdy` smoke (`max_iter=20`) is effectively unchanged,
    confirming the new target does not perturb the current free-boundary
    accelerated alias on the existing smoke case.
- Removed fixed literal convergence thresholds from the experimental solver
  stack:
  - `solve_lambda_gd()`, `solve_fixed_boundary_gd()`, and
    `solve_fixed_boundary_lbfgs()` now derive their default `grad_tol`
    from the initial gradient scale and machine precision instead of using a
    hardcoded absolute cutoff,
  - `solve_fixed_boundary_lbfgs_vmec_residual()` now derives its
    startup/converged `m=1` residual release threshold from the input
    `FTOL` and uses a scale-aware L-BFGS curvature acceptance test rather than
    a fixed secant threshold,
  - `solve_fixed_boundary_gn_vmec_residual()` now defaults to adaptive
    damping and adaptive CG tolerances derived from local curvature,
    residual progress, and `FTOL`, with no default fixed `damping`,
    `cg_tol`, or `zero_m1_fsqz_thresh`,
  - implicit differentiation wrappers no longer inject a fixed
    `grad_tol=1e-10`; when callers omit `grad_tol`, the wrappers use the
    solver-resolved adaptive tolerance.
- Revalidated after the threshold cleanup:
  - `pytest -q` passed (`152 passed, 12 skipped`),
  - fast Sphinx build passed,
  - accelerated fixed-boundary smoke stayed at the same order of performance:
    `input.circular_tokamak` remained approximately neutral and
    `input.up_down_asymmetric_tokamak` remained about `3.15x` faster than the
    current default path in
    `outputs/accelerated_mode_threshold_cleanup_smoke/summary.json`.
- Switched accelerated fixed-boundary default staging to single-grid when the
  caller does not explicitly request multigrid:
  - reverted the abandoned raw-scan adaptive-`dt` experiment,
  - `run_fixed_boundary(..., solver_mode="accelerated")` now defaults to the
    final grid for fixed-boundary cases, while parity mode and explicit
    `multigrid=True` keep the staged VMEC-style path,
  - added regression coverage so accelerated fixed-boundary defaults expose
    `accelerated_single_grid_default=True` and parity runs still report the
    staged `NS_ARRAY`,
  - serial CPU measurements are recorded in
    `outputs/accelerated_fixed_boundary_singlegrid_serial_20260307/summary.json`:
    - `input.LandremanSenguptaPlunk_section5p3_low_res`:
      `0.241s` single-grid vs `0.284s` explicit multigrid,
    - `input.LandremanPaul2021_QA_lowres`:
      `6.15s` single-grid vs `16.50s` explicit multigrid,
    - `retired staged fixed-boundary stress case`:
      `1.37s` single-grid with final `fsq_total ~1.1e-4`.
- Added a dedicated merge-readiness page for the accelerated branch:
  - `docs/accelerated_merge_readiness.rst` now separates
    "mergeable as experimental" from "ready to become default",
  - the review checklist is explicit about what must still remain behind the
    opt-in `solver_mode="accelerated"` API,
  - recommended PR scope is now clear:
    merge the accelerated fixed-boundary feature set as experimental,
    while keeping the parity/default controller as the ordinary path.
- Fixed an accelerated single-grid stage-budget mismatch during local review:
  - when staged fixed-boundary inputs are collapsed to a single accelerated
    final-grid solve, the driver now carries forward the total `NITER_ARRAY`
    budget and final-stage `FTOL_ARRAY` value instead of falling back to
    `NITER`,
  - added regression coverage for the `LandremanPaul2021_QA_lowres`
    single-grid budget selection,
  - refreshed the serial reassessment artifact in
    `outputs/accelerated_fixed_boundary_reassessment_20260309/summary.json`:
    - `input.LandremanSenguptaPlunk_section5p3_low_res`:
      `45.48s` default vs `0.198s` accelerated single-grid,
    - `input.LandremanPaul2021_QA_lowres`:
      `8.18s` default vs `7.31s` accelerated single-grid,
    - `retired staged fixed-boundary stress case`:
      `1.25s` accelerated single-grid with final `fsq_total ~1.1e-4`.
- Added a CLI-only accelerated fixed-boundary fallback for staged inputs that
  have `NS_ARRAY` but no `NITER_ARRAY`:
  - the executable now derives a reduced total warm-start budget from the
    coarsest-to-finest `ns` ratio,
  - that budget is distributed across the stages by the squared number of new
    radial degrees of freedom introduced at each stage,
  - after the warm-start stages, the CLI applies a short parity polish on the
    final grid when the accelerated pass is still not strictly converged,
  - this policy is intentionally scoped to the non-differentiable CLI path;
    the Python API keeps the existing accelerated single-grid behavior.
- Local serial reassessment for the new CLI policy is recorded in
  `outputs/accelerated_cli_fixed_boundary_reassessment_20260309/summary.json`:
  - `input.LandremanSenguptaPlunk_section5p3_low_res`:
    unchanged in practice at ~`0.151s`, `fsq_total ~3.0e-14`,
  - `input.LandremanPaul2021_QA_lowres`:
    unchanged in practice at ~`7.12s`, `fsq_total ~3.0e-13`,
  - `retired staged fixed-boundary stress case`:
    `1.26s` plain accelerated API path vs `16.42s` CLI budgeted-multigrid +
    parity-polish path, with final `fsq_total` reduced from
    ~`1.12e-4` to ~`6.81e-6`.
- Revalidated after the CLI accelerated-controller change:
  - `pytest -q` passed (`157 passed, 12 skipped`),
  - fast Sphinx build passed,
  - direct CLI smoke on `retired staged fixed-boundary stress case` completed and wrote
    `retired staged-stress CLI output` successfully.
- Tightened the CLI-only fixed-boundary finisher on the experimental branch:
  - the finisher now preserves the original staged-policy diagnostics when it
    returns a best resumed run,
  - resumed finish attempts now call `solve_fixed_boundary_residual_iter()`
    directly instead of recursively re-entering `run_fixed_boundary()`,
  - minimal accelerated `resume_state` is sanitized before handing it to the
    strict controller so cache-valid flags do not advertise missing norm/cache
    payloads,
  - accelerated CLI finish attempts now tag explicit metadata:
    `cli_fixed_boundary_initial_policy`,
    `cli_fixed_boundary_finish_budgets`,
    `cli_fixed_boundary_finish_modes`,
    `cli_fixed_boundary_finish_converged`, and
    `cli_fixed_boundary_full_parity_fallback`.
- Current fixed-boundary CLI accelerated status on the branch:
  - `retired single-grid fixed-boundary stress case` is no longer a qualitative blocker:
    a resumed strict finisher with parity budgets `[1000, 2000]`
    reaches `fsq_total ~1.24e-14` and marks
    `converged_by_total_fsq=True` in about `89.9s`,
  - `input.up_down_asymmetric_tokamak` was already known to close under a
    resumed strict finisher,
  - `retired staged fixed-boundary stress case` remains the unresolved fixed-boundary
    holdout for the experimental branch:
    - mixed staged policies help but do not yet close the target,
    - accelerated coarse stages plus a parity fine stage reduce the final-grid
      residual from order `1e-2` to order `1e-5`, but not to `FTOL`,
    - a parity fine stage with `max_iter=5000` from the improved stage-49
      state reached `fsq_total ~4.47e-6` in about `193s` and still did not
      converge,
    - a full end-to-end CLI accelerated `retired staged stress case` run still exceeded acceptable
      multi-minute runtime and was stopped without convergence.
- Updated the CLI default policy for shipping:
  - simple fixed-boundary inputs now default to `solver_mode="accelerated"`
    on the CLI,
  - the remaining hard staged class with `NS_ARRAY` but no `NITER_ARRAY`
    defaults to `solver_mode="parity"` on the CLI until the staged
    accelerated finisher closes that gap robustly,
  - this policy is input-structure based, not case-name based.
- Follow-up staged-hybrid fixed-boundary work on `codex/nonparity-performance`:
  - CLI fixed-boundary finish attempts are no longer limited to accelerated
    mode; strict parity CLI runs now also get the same state-only finish
    controller when they miss the target on the first pass,
  - those finish attempts now restart from the equilibrium state only, not from
    cached nonlinear-controller history, because the state-only continuation was
    materially more robust on the hard staged cases,
  - the accelerated staged warm-start path now keeps accelerated coarse stages
    but runs the final stage in strict parity with the full user `NITER`
    budget, instead of starving the fine stage with the reduced warm-start
    budget,
  - the new artifact
    `outputs/accelerated_cli_fixed_boundary_hybrid_20260309/summary.json`
    records the latest measured status:
    - `input.LandremanSenguptaPlunk_section5p3_low_res` closes at
      `fsq_total ~3.0e-14`,
    - `input.LandremanPaul2021_QA_lowres` closes at
      `fsq_total ~3.0e-13`,
    - `retired single-grid fixed-boundary stress case` closes at `fsq_total ~1.24e-14`,
    - `retired staged fixed-boundary stress case` improves to a best measured
      `fsq_total ~1.61e-6` under the staged hybrid controller, but still does
      not reach `FTOL`.
- Bundled-example update:
  - `retired staged fixed-boundary input` now includes
    `NITER_ARRAY = 1000 1000 5000`,
  - this makes the example’s staged continuation explicit instead of relying on
    the legacy “`NITER` applies to every stage” interpretation,
  - the generic CLI policy for staged fixed-boundary inputs without
    `NITER_ARRAY` remains in place, but that policy is now exercised by tests
    using a synthetic staged fixed-boundary input rather than the bundled
    `retired staged stress case` file.
- Fixed-boundary CLI controller update:
  - accelerated CLI runs now keep the fast single-grid attempt as the first
    move,
  - if a staged input carries explicit `NS_ARRAY` and `NITER_ARRAY` and the
    first fast solve misses the target, the executable now replays that staged
    schedule automatically before starting strict parity finish blocks,
  - added `examples/fixed_boundary_driver_tracks.py` so users can compare the
    parity track and the optimized CLI-style track from Python on their own
    CPU/GPU machine.
- 2026-03-10 fixed-boundary reassessment:
  - new serial bundled artifact:
    `outputs/accelerated_cli_fixed_boundary_bundle_20260310/summary.json`,
  - 11 of 15 bundled fixed-boundary cases are faster under
    `solver_mode="accelerated"` with `cli_fixed_boundary_mode=True`,
  - strongest wins:
    `LandremanSenguptaPlunk_section5p3_low_res` (`249.49x`),
    `basic_non_stellsym_pressure` (`12.47x`),
    `ITERModel` (`1.78x`),
  - current slow outliers:
    `retired single-grid stress case` (`0.0036x`),
    `up_down_asymmetric_tokamak` (`0.0225x`),
    `LandremanPaul2021_QA_lowres1` (`0.93x`),
    `solovev` (`0.94x`),
  - `retired staged stress case` remains the hard outlier:
    a same-branch cold `solver_mode="default"` run took `41.67s` and stopped at
    `fsq_total ~ 6.90e-2`, while the optimized CLI-style run exceeded 15
    minutes without finishing the cold reassessment solve.
- 2026-03-10 targeted single-grid CLI finisher improvement:
  - CLI fixed-boundary single-grid finish blocks now try accelerated
    continuation before strict parity fallback,
  - added regression coverage for the new finish ordering in
    `tests/test_driver_api.py`,
  - targeted outlier artifact:
    `outputs/accelerated_cli_fixed_boundary_outliers_20260310/summary.json`,
  - `retired single-grid fixed-boundary stress case` improved from about `84.64s` to about `8.82s` while
    still converging to `fsq_total ~1.24e-14`,
  - `input.up_down_asymmetric_tokamak` improved from about `43.74s` to about
    `0.55s`, now about `2.11x` faster than the branch baseline while still
    converging to `fsq_total ~3.00e-14`.
- 2026-03-10 README benchmark refresh:
  - regenerated the top README benchmark figure as a warmed fixed-boundary
    optimized-CLI speedup plot against VMEC2000,
  - refreshed `tools/diagnostics/readme_runtime_compare.py` so it can render a
    CPU-only speedup panel when no GPU summary is available,
  - updated bundled warmed CPU matrix:
    `outputs/fixed_runtime_accel_cpu_bundle_20260310_swap/summary.json`,
  - all 16 bundled fixed-boundary cases are now faster than VMEC2000 once JAX
    is warmed and all 16 converge,
  - same-host CPU/GPU bundled summaries were collected on a reference
    GPU-capable workstation for the updated 16-case fixed-boundary matrix,
  - both backends converge on all 16 bundled fixed-boundary cases; the GPU is
    already faster on the heavier 3D QA/QH reactor-scale cases and
    `cth_like_fixed_bdy`, while the CPU still wins on the smaller axisymmetric
    cases,
  - bundled fixed-boundary examples/benchmarks now use the QA/QH reactor-scale
    replacements instead of the retired internal stress cases.
  - refreshed the top README comparison panels to use an optimized
    axisymmetric case plus a bundled 3D showcase,
  - regenerated the README `fsq_total` trace on the optimized controller for
    the README showcase cases,
  - replaced the top README speedup chart with a CPU-only runtime chart showing
    VMEC2000 and `vmec_jax` CPU together,
  - same-host CPU/GPU benchmarking on the updated 16-case fixed-boundary
    bundle now shows GPU wins on 5 heavier 3D cases and CPU wins on the
    remaining 11 smaller or more launch-latency-dominated cases.
- 2026-03-10 fixed-boundary final-`wout` audit:
  - regenerated `wout_LandremanPaul2021_QA_lowres_reference.nc` from the
    canonical VMEC2000 executable because the previously bundled file used a
    mismatched radial resolution (`ns=75` vs input `ns=50`),
  - `tools/diagnostics/benchmark_accelerated_mode.py` now aligns coefficient
    fields on a common radial grid before computing relRMS, so bundled audits no
    longer silently return `nan` just because `ns` differs,
  - full bundled fixed-boundary audit is recorded in
    `outputs/fixed_wout_audit_20260310_r3/summary.json`,
  - low `fsq_total` is not sufficient for final-output quality on the current
    branch:
    `LandremanPaul2021_QA_lowres` remains at max relRMS `~3.37e-01`,
    `LandremanPaul2021_QA_reactorScale_lowres` at `~3.58e+00`,
    `LandremanPaul2021_QH_reactorScale_lowres` at `~4.96e+00`,
    `up_down_asymmetric_tokamak` at `~1.70e-01`,
  - validated showcase-quality cases are:
    `ITERModel` (`~6.01e-06`),
    `shaped_tokamak_pressure` (`~1.55e-07`),
    `circular_tokamak` (`~1.03e-05`),
  - the README showcase was updated accordingly so it no longer presents the QA
    low-resolution case as a parity-quality final state.
- 2026-03-10 accelerated scan-core pass:
  - accelerated quiet scan runs now default to the leaner `scan_core` path
    unless explicitly overridden with `VMEC_JAX_SCAN_CORE`,
  - this removes part of the fallback/probe bookkeeping from the hot path
    without changing the fixed-boundary convergence results on the bundled
    audit,
  - attempted scan-carry buffer donation was rejected by JAX because the carry
    currently aliases internal buffers; that avenue needs a carry-structure
    refactor before it can be used safely.
- 2026-03-11 non-axisymmetric fixed-boundary audit and first repair:
  - localized the dominant non-axisymmetric state-vs-VMEC2000 field mismatch to
    stale driver-returned `chipf/iota` on `NCURR=1` fixed-boundary runs:
    geometry stayed near machine precision while `|B|` drifted because
    `run_fixed_boundary()` returned input-built flux profiles instead of the
    solved current-driven profiles,
  - `vmec_jax/driver.py` now recomputes post-solve `chipf/iota` for
    `NCURR=1` returns using the same force-balance reconstruction already used
    by `wout`,
  - added regression coverage in `tests/test_driver_api.py` ensuring
    non-axisymmetric current-driven runs return nonzero `chipf/iota` and match
    the `wout` reconstruction,
  - QA/QH/basic-non-stellsym `|B|` drift dropped from the earlier
    `~2.6%-8.0%` range down to about `0.5%-1.8%` on the patched return path,
  - a second control-flow issue was then identified on the accelerated CLI
    branch: staged 3D inputs with explicit `NS_ARRAY/NITER_ARRAY` returned
    immediately when the first single-grid pass met the scalar `fsq_total`
    target, so the intended staged follow-up never ran,
  - the CLI accelerated finisher now forces staged follow-up for staged 3D
    fixed-boundary inputs even if the first single-grid pass already satisfies
    `fsq_total`,
  - staged 3D explicit follow-up now uses a continuation policy of
    `parity -> accelerated -> parity` across three or more stages, keeping the
    coarsest and finest stages conservative while accelerating only interior
    continuation stages,
  - measured QA-lowres final-quality improvement on the accelerated CLI path:
    `rmnc` `~4.98e-04 -> 5.83e-05`,
    `zmns` `~2.71e-03 -> 2.83e-04`,
    `lmns` `~4.19e-02 -> 4.75e-03`,
  - measured QA-reactor-scale improvement on the accelerated CLI path:
    initial staged-followup fix brought the case to
    `rmnc ~4.16e-04`, `zmns ~2.02e-03`, `lmns ~3.10e-02`,
  - measured QH-reactor-scale improvement on the accelerated CLI path:
    initial staged-followup fix brought the case to
    `rmnc ~2.05e-04`, `zmns ~6.37e-04`, `lmns ~2.32e-02`,
  - final staged-3D policy was then tightened further so the first stage is
    parity-controlled on all staged non-axisymmetric fixed-boundary cases:
    `parity -> parity` for two-stage inputs and
    `parity -> accelerated -> parity` for three-or-more-stage inputs,
  - that closes the remaining reactor-scale fixed-boundary gaps on the
    accelerated CLI path:
    `LandremanPaul2021_QA_reactorScale_lowres`
    now reaches `rmnc ~2.49e-05`, `zmns ~1.61e-04`, `lmns ~2.86e-03`;
    `LandremanPaul2021_QH_reactorScale_lowres`
    now reaches `rmnc ~6.12e-05`, `zmns ~2.60e-04`, `lmns ~9.97e-03`,
  - the README showcase is switched back from the temporary ITER fallback to
    the repaired QA-lowres accelerated comparison panels.
  - full regression suite after these fixes:
    `166 passed, 12 skipped`.
- 2026-03-11 full fixed-boundary bundled reassessment:
  - full local validation remains green:
    `pytest -q` -> `166 passed, 12 skipped`,
    `SPHINX_FAST=1 ... sphinx -W ...` -> passed,
  - full warmed fixed-boundary bundled matrix is recorded in
    `outputs/accelerated_cli_fixed_boundary_full_20260311/summary.json`,
  - both the current default path and the optimized CLI-style candidate
    converged on all 16 bundled fixed-boundary cases,
  - however the optimized controller is only faster on 8 of 16 cases, roughly
    neutral on 2, and slower on 6,
  - the largest regressions are still the non-axisymmetric QA/QH-style cases
    and `basic_non_stellsym_pressure`,
  - conclusion from the full bundle is now explicit:
    branch remains useful and reviewable as experimental, but it is not ready
    to become the default fixed-boundary controller on `main`.
- 2026-03-11 targeted work on the 6 regressing bundled fixed-boundary cases:
  - localized the regression pattern to staged, current-driven 3D CLI
    fixed-boundary inputs:
    the optimized controller was paying for a single-grid accelerated warm
    start and/or an over-eager explicit staged replay,
  - tightened the policy in `vmec_jax/driver.py` so staged current-driven 3D
    inputs now split by stage count:
    two-stage inputs go straight to the explicit staged path,
    three-or-more-stage inputs try the fast single-grid path first and only
    replay the explicit stage schedule if that first solve misses the target,
  - added driver regressions in `tests/test_driver_api.py` covering both:
    deferred staged replay on three-stage current-driven 3D inputs and direct
    multigrid on two-stage current-driven 3D inputs,
  - targeted subset benchmark is recorded in
    `outputs/accelerated_cli_fixed_boundary_regressions_20260311_r3/summary.json`,
  - subset outcome after the patch:
    `LandremanPaul2021_QA_lowres` improved from `0.14x` to `1.02x`,
    `LandremanPaul2021_QA_reactorScale_lowres` improved from `0.09x` to `1.08x`,
    `LandremanPaul2021_QH_reactorScale_lowres` improved from `0.12x` to `1.00x`,
    `basic_non_stellsym_pressure` improved from `0.19x` to `12.11x`,
    `solovev` moved from `0.97x` to `0.99x`,
    `LandremanPaul2021_QA_lowres1` is now effectively neutral at `0.97x`,
  - full regression suite after this controller tuning:
    `168 passed, 12 skipped`.
- 2026-03-11 full fixed-boundary bundled reassessment after staged-3D tuning:
  - refreshed full warmed CPU matrix is recorded in
    `outputs/accelerated_cli_fixed_boundary_full_20260311_r2/summary.json`,
  - the optimized CLI-style fixed-boundary controller now converges on all 16
    bundled fixed-boundary cases and has no bundled CPU regressions left,
  - count by warmed runtime ratio versus the current branch baseline:
    13 faster, 3 roughly neutral, 0 slower,
  - representative improvements include:
    `LandremanPaul2021_QA_lowres` `1.11x`,
    `LandremanPaul2021_QA_reactorScale_lowres` `1.04x`,
    `basic_non_stellsym_pressure` `12.37x`,
    `LandremanSenguptaPlunk_section5p3_low_res` `240.59x`,
    `up_down_asymmetric_tokamak` `2.28x`,
  - current branch assessment becomes narrower:
    fixed-boundary CLI on CPU is now plausible as the default path on `main`
    if reviewers accept the non-parity scope; GPU and broader library/default
    questions remain separate.
- 2026-03-11 3D final-`wout` quality audit and export fix:
  - audited the remaining non-axisymmetric fixed-boundary `wout` mismatch
    fields using the refreshed bundled benchmark artifacts,
  - found that the dominant QA/QH error was no longer the solved geometry
    itself but symmetry-forbidden `rmns` / `zmnc` channels being exported in
    `wout` for `lasym=False` runs,
  - `vmec_jax/wout.py` now zeros `rmns` and `zmnc` when `lasym=False`,
    matching VMEC2000's symmetric `wout` convention,
  - added a reference-based regression in
    `tests/test_wout_parity_reference.py` on
    `LandremanPaul2021_QA_lowres` to lock that behavior in,
  - refreshed 3D bundled audit in
    `outputs/fixed_wout_3d_audit_20260311_r1/summary.json`,
  - quality improvements from that fix:
    `LandremanPaul2021_QA_lowres` `~3.37e-01 -> 4.19e-02`,
    `LandremanPaul2021_QA_reactorScale_lowres` `~3.56e+00 -> 3.14e-02`,
    `LandremanPaul2021_QH_reactorScale_lowres` `~4.61e+00 -> 2.22e-02`,
  - remaining notable 3D bundled quality gap is now concentrated in
    `basic_non_stellsym_pressure` (`~4.29e-01`) and residual lambda-channel
    drift on the QA/QH cases,
  - also tested a final-grid parity-polish controller for staged 3D runs, but
    rejected it because it increased runtime substantially without improving
    the benchmarked final-`wout` metrics,
  - full regression suite after the `wout` fix:
    `169 passed, 12 skipped`.
- 2026-03-12 staged 3D continuation audit for remaining non-axisymmetric gaps:
  - validated that the remaining bundled 3D quality gap was no longer a broad
    geometry mismatch but a staged lambda-continuation issue, concentrated in
    `basic_non_stellsym_pressure` and residual `lmns` drift on the QA/QH cases,
  - updated `vmec_jax/driver.py` so 3-stage `lasym=True`, current-driven,
    non-axisymmetric accelerated multigrid runs keep the entry and final stages
    on the conservative controller and accelerate only the interior stage,
  - added driver coverage in `tests/test_driver_api.py` for that policy,
  - targeted audit in
    `outputs/nonaxis_quality_subset_20260312_r2/summary.json` now shows:
    `LandremanPaul2021_QA_lowres` `~4.19e-02`,
    `LandremanPaul2021_QA_reactorScale_lowres` `~3.14e-02`,
    `LandremanPaul2021_QH_reactorScale_lowres` `~2.22e-02`,
    `basic_non_stellsym_pressure` `~3.46e-02`,
  - the same targeted audit kept all four cases converged while preserving
    runtime gains versus baseline, with `basic_non_stellsym_pressure`
    improving from `~23.69s` baseline to `~19.36s` optimized.
- 2026-03-12 current-driven 3D controller split for final non-axisymmetric gaps:
  - audited the remaining lambda-dominated mismatch on the non-axisymmetric
    fixed-boundary current-driven bundle and rejected two intermediate ideas:
    a reduced staged follow-up for deferred 3-stage runs, and a blanket
    non-scan policy for all current-driven 3D accelerated runs,
  - kept the useful part of that audit: `lasym=False` current-driven 3D CLI
    runs now go straight to staged multigrid on the conservative non-scan
    residual path,
  - targeted audit in
    `outputs/nonaxis_quality_subset_20260312_r5/summary.json` now shows:
    `LandremanPaul2021_QA_lowres` `~4.20e-03`,
    `LandremanPaul2021_QA_reactorScale_lowres` `~6.42e-04`,
    `LandremanPaul2021_QH_reactorScale_lowres` `~6.00e-05`,
  - follow-on audit on `basic_non_stellsym_pressure` showed that the remaining
    `~3.46e-02` mismatch was largely baseline-level lambda error rather than a
    unique accelerated-path geometry failure,
  - updated `vmec_jax/driver.py` so `lasym=True` current-driven 3D staged runs
    stay fully on the conservative controller; that removes the branch-specific
    quality regression and brings
    `outputs/basic_non_stellsym_pressure_audit_20260312_r2/summary.json` back
    to baseline-level quality (`~2.98e-02`) at essentially neutral warmed
    runtime (`~22.24s -> 22.31s`),
  - accelerated fixed-boundary return semantics now treat per-channel final
    `FTOL` as the acceptance truth, while keeping `fsq_total = fsqr + fsqz +
    fsql` as an internal early-stop only for accelerated blocks,
  - direct validation on the new head confirms strict requested-`FTOL`
    convergence on representative accelerated CLI fixed-boundary cases:
    `LandremanPaul2021_QA_lowres` (`~7.40e-14`, `~3.21e-14`, `~9.99e-14`
    against requested `1e-13`),
    `LandremanPaul2021_QA_reactorScale_lowres`
    (`~1.47e-15`, `~6.93e-16`, `~8.14e-16` against requested `1e-15`),
    and `basic_non_stellsym_pressure`
    (`~9.81e-11`, `~5.46e-11`, `~1.31e-12` against requested `1e-10`),
  - profiling then showed the next dominant CPU cost was controller-side JAX
    overhead rather than the fixed-boundary force kernels themselves: the
    `ptau` sign-change helper was moved to a host NumPy implementation for the
    non-scan controller, and accelerated `lasym=False` CPU CLI runs now also
    use host-side signed Fourier update assembly,
  - targeted reassessment in
    `outputs/host_updates_benchmark_20260312/summary.json` then improved
    `LandremanPaul2021_QA_lowres` from a branch-local
    `~34.83s -> 38.64s` baseline-vs-optimized result to
    `~34.83s -> 31.17s`, while keeping `~4.20e-03` max relRMS against the
    bundled VMEC2000 reference; `basic_non_stellsym_pressure` held its
    baseline-level `~2.98e-02` quality and remained slightly faster
    (`~9.12s -> 8.95s`),
  - reran the full warmed bundled fixed-boundary `lasym=False` CPU matrix in
    `outputs/fixed_lasym_false_matrix_20260312/summary.json`: all 13 cases
    converged on both paths and the optimized controller was faster on all 13,
    including the reactor-scale QA/QH cases,
  - profiled representative `lasym=False` free-boundary
    `input.cth_like_free_bdy`; the next safe overhead reduction was in
    `_sample_external_boundary_arrays`, where batching the boundary
    real-space syntheses cut the total cProfile wall time from about `60.41s`
    to about `58.21s` while keeping the direct NESTOR regression tests green,
  - full regression suite on the final split-controller head:
    `170 passed, 12 skipped`.
