# VMEC-JAX Master Plan and New-Agent Handoff (Living Document)

Last updated: 2026-03-05
Primary owner: `vmec_jax` contributors
Canonical repo: `/Users/rogeriojorge/local/test/vmec_jax`

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
  - `/Users/rogeriojorge/local/test/vmec_jax/README.md`
  - `/Users/rogeriojorge/local/test/vmec_jax/docs/validation.rst`
  - `/Users/rogeriojorge/local/test/vmec_jax/docs/performance.rst`
  - `/Users/rogeriojorge/local/test/vmec_jax/docs/free_boundary_plan.rst`

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
- Workspace root: `/Users/rogeriojorge/local/test`
- Main repo: `/Users/rogeriojorge/local/test/vmec_jax`
- VMEC2000 source/executable (source of truth):
  - source: `/Users/rogeriojorge/local/test/STELLOPT/VMEC2000/Sources`
  - executable: `/Users/rogeriojorge/local/test/STELLOPT/VMEC2000/Release/xvmec2000`
- Other local VMEC trees (non-canonical for parity unless explicitly tested):
  - `/Users/rogeriojorge/local/STELLOPT/VMEC2000`
  - `/Users/rogeriojorge/local/test/vmec2000`
- Examples/data:
  - `/Users/rogeriojorge/local/test/vmec_jax/examples/data`
- Diagnostics tools:
  - `/Users/rogeriojorge/local/test/vmec_jax/tools/diagnostics`
- Tests:
  - `/Users/rogeriojorge/local/test/vmec_jax/tests`
- Docs:
  - `/Users/rogeriojorge/local/test/vmec_jax/docs`

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
  - `/Users/rogeriojorge/local/test/vmec_jax/vmec_jax/driver.py`
- Nonlinear solver control and scan/non-scan paths:
  - `/Users/rogeriojorge/local/test/vmec_jax/vmec_jax/solve.py`
- Free-boundary coupling and vacuum/scalpot channels:
  - `/Users/rogeriojorge/local/test/vmec_jax/vmec_jax/free_boundary.py`
- Geometry/forces/jacobian/residual:
  - `/Users/rogeriojorge/local/test/vmec_jax/vmec_jax/vmec_bcovar.py`
  - `/Users/rogeriojorge/local/test/vmec_jax/vmec_jax/vmec_forces.py`
  - `/Users/rogeriojorge/local/test/vmec_jax/vmec_jax/vmec_jacobian.py`
  - `/Users/rogeriojorge/local/test/vmec_jax/vmec_jax/vmec_residue.py`
- Fourier/tables/transforms:
  - `/Users/rogeriojorge/local/test/vmec_jax/vmec_jax/vmec_tomnsp.py`
  - `/Users/rogeriojorge/local/test/vmec_jax/vmec_jax/fourier.py`
  - `/Users/rogeriojorge/local/test/vmec_jax/vmec_jax/vmec_realspace.py`
- Preconditioners:
  - `/Users/rogeriojorge/local/test/vmec_jax/vmec_jax/preconditioner_1d.py`
  - `/Users/rogeriojorge/local/test/vmec_jax/vmec_jax/preconditioner_1d_jax.py`
- Output handling (`wout` and derived channels):
  - `/Users/rogeriojorge/local/test/vmec_jax/vmec_jax/wout.py`

### 3.2 APIs for optimization/autodiff workflows
- Optimization-facing tools:
  - `/Users/rogeriojorge/local/test/vmec_jax/vmec_jax/optimization.py`
- Programmatic output adapter style object for optimization pipelines:
  - `/Users/rogeriojorge/local/test/vmec_jax/vmec_jax/booz_input.py`
  - (contains JAX-array export channels such as `rmnc/zmns/lmns`, Nyquist fields, `xm/xn`, `xm_nyq/xn_nyq`, `iota`, etc.)

### 3.3 Diagnostics and parity infrastructure
- Fixed-boundary comparator:
  - `/Users/rogeriojorge/local/test/vmec_jax/tools/diagnostics/vmec2000_exec_stage_trace_compare.py`
- Free-boundary comparator:
  - `/Users/rogeriojorge/local/test/vmec_jax/tools/diagnostics/vmec2000_exec_freeb_scalpot_compare.py`
- Manifest and sweep runner:
  - `/Users/rogeriojorge/local/test/vmec_jax/tools/diagnostics/parity_manifest.toml`
  - `/Users/rogeriojorge/local/test/vmec_jax/tools/diagnostics/parity_sweep_manifest.py`

### 3.4 Core docs
- Main docs index:
  - `/Users/rogeriojorge/local/test/vmec_jax/docs/index.rst`
- Algorithms and numerics:
  - `/Users/rogeriojorge/local/test/vmec_jax/docs/algorithms.rst`
- Validation:
  - `/Users/rogeriojorge/local/test/vmec_jax/docs/validation.rst`
- Performance:
  - `/Users/rogeriojorge/local/test/vmec_jax/docs/performance.rst`
- Free-boundary implementation plan:
  - `/Users/rogeriojorge/local/test/vmec_jax/docs/free_boundary_plan.rst`

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
cd /Users/rogeriojorge/local/test/vmec_jax
pytest -q
```

### 6.2 Docs build (CI-equivalent locale settings)
```bash
cd /Users/rogeriojorge/local/test/vmec_jax
LC_ALL=C LANG=C SPHINX_FAST=1 python -m sphinx -W -j auto -b html docs docs/_build/html
```

### 6.3 Fixed-boundary parity compare
```bash
cd /Users/rogeriojorge/local/test/vmec_jax
python tools/diagnostics/vmec2000_exec_stage_trace_compare.py \
  --input /Users/rogeriojorge/local/test/vmec_jax/examples/data/input.LandremanPaul2021_QA_lowres \
  --use-input-niter --max-iter 10 --dump-level full \
  --vmec2000 /Users/rogeriojorge/local/test/STELLOPT/VMEC2000/Release/xvmec2000
```

### 6.4 Free-boundary parity compare (single iteration)
```bash
cd /Users/rogeriojorge/local/test/vmec_jax
python tools/diagnostics/vmec2000_exec_freeb_scalpot_compare.py \
  --input /Users/rogeriojorge/local/test/STELLOPT/BENCHMARKS/VMEC_TEST/input.DIII-D \
  --iter 80 --max-iter 80 \
  --vmec-exec /Users/rogeriojorge/local/test/STELLOPT/VMEC2000/Release/xvmec2000 \
  --workdir /tmp/freeb_diiid_iter80
```

### 6.5 Manifest sweeps
```bash
cd /Users/rogeriojorge/local/test/vmec_jax
python tools/diagnostics/parity_sweep_manifest.py \
  --tier smoke --vmec-exec /Users/rogeriojorge/local/test/STELLOPT/VMEC2000/Release/xvmec2000
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
- `/Users/rogeriojorge/local/test/vmec_jax/.github/workflows/ci.yml`

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
`/Users/rogeriojorge/local/test/vmec_jax/outputs/parity_sweeps/...`
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
