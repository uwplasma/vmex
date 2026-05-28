# VMEC-JAX Research-Grade Roadmap

Last updated: 2026-05-28
Primary branch: `main`
Baseline release: `v0.0.13`
Latest known green `main` CI: `152360f`
Current candidate: main plus free-boundary direct-coil PR #18 refresh

This is the living execution plan for making `vmec_jax` accurate, fast,
differentiable, documented, and usable by external researchers. Update it when
acceptance criteria or evidence changes.

## Working Rules

- Keep VMEC2000/VMEC++ parity as a correctness gate, not a demo.
- Keep differentiable objectives smooth enough for JAX AD; keep legacy
  non-smooth metrics as diagnostics and ranking gates.
- Keep examples SIMSOPT-like: users assemble objectives, weights, targets, and
  optimizer choices in plain scripts.
- Keep CI under 10 minutes for required jobs; move expensive validation to
  nightly/manual jobs with committed summaries.
- Commit only source, tests, docs, and lightweight curated fixtures; keep
  generated sweeps and large artifacts out of git unless explicitly curated.

## Current State

- Fixed-boundary and free-boundary CLI/API paths exist and have parity fixtures.
- Optimization examples cover QA, QH, QP, and QI with continuation/ESS policies.
- Smooth differentiable QI, mirror ratio, elongation, LgradB, aspect, iota, beta,
  volavgB, magnetic-well, DMerc, and JXBFORCE profile objectives exist in the
  workflow layer.
- GPU execution works. Small/medium optimization cases are now much closer to
  CPU after the backend-adaptive replay-bucket fix and the post-`v0.0.8`
  dense-replay chunking profile. The profiler no longer hides a pre-profile
  exact solve, malformed replay-bucket environment values fall back to
  backend-adaptive defaults, production fixed-boundary auto policy uses the
  non-scan VMEC-control loop on CPU/GPU, and larger-mode accepted-point tape
  replay remains the next GPU optimization target. The accepted-point
  optimizer also now lazily JITs projected initial-state construction, cutting
  repeated new-point initialization cost in the CPU exact-callback smoke.
- Continuation correctness is now protected by both synthetic control-flow tests
  and a real projected-boundary stage test. Repeated stage schedules such as
  ``[1, 1, 2]`` carry the optimized VMEC input forward and keep projected
  high-mode coefficients zero.
- Exact-history correctness is now protected against relaxed trial-solve drift:
  final ``input.final`` and ``wout_final.nc`` use verified exact accepted
  states, and persisted WOUT artifacts no longer fall back to relaxed trial
  solves on cache misses.
- The duplicate finite-beta stage-one output path now has the same
  selected-best-exact-state save contract as the main QS workflow, so
  ``input.final`` and ``wout_final.nc`` cannot drift there either.
- Required CI coverage is now gated at 95%. The v0.0.12 release-candidate
  fast suite passed locally with `2136 passed, 20 skipped, 107 deselected` in
  8:03 and 95.06% coverage, then passed on GitHub Actions for Python
  3.10/3.11/3.12 with the same 95% py3.11 gate. After the post-release
  workflow/API/performance patches through `13ccef8`, the clean local
  CI-equivalent gate passed again with `2147 passed, 20 skipped, 107 deselected`
  in 8:08 and 95.06% coverage. After the QI public-helper, WOUT physics-gate,
  profiling-budget, and optimization-output test additions through `1037744`,
  the local required gate passed with `2153 passed, 20 skipped, 107 deselected`
  in 8:05 and 95.10% coverage; full Sphinx also passed warning-clean. The
  matching GitHub Actions run passed docs, build, parity dry-run, physics smoke,
  and Python 3.10/3.11/3.12 fast tests. On 2026-05-27 after the QI workflow
  checkpointing, QI resolution-override, and minimal-seed helicity-perturbation
  patches, the clean local required gate passed with `2328 passed, 20 skipped,
  109 deselected, 1 xfailed` and 95.00% coverage in about 6:25; full Sphinx
  also passed with warnings as errors. GitHub Actions is green through
  `152360f`. The optional
  converged VMEC2000 parity gate remains opt-in with
  `VMEC2000_INTEGRATION=1`. `solve.py` still dominates the missing-line
  surface, so future coverage should come from physics-gated refactor seams
  rather than scaffolding.
- VMEC2000 converged-wout parity now has a fast bundled matrix gate across
  fixed/free, axisymmetric/non-axisymmetric, LASYM, and single/multigrid
  representatives. The executable-backed end-state gate remains opt-in:
  `VMEC2000_INTEGRATION=1` runs the bounded circular comparison, while
  `VMEC2000_NIGHTLY=1` adds the slower matrix representatives.
- Full non-VMEC2000 physics coverage with refreshed released assets reaches
  72.35% locally (`74 passed, 4 skipped`, 27:21). This is still short of the
  80% target and is too slow for per-commit required CI without splitting the
  slow full cases.
- QI diagnostics now have a low-resolution bundled solved-state gate on
  `input.QI_stel_seed_3127`/`wout_QI_stel_seed_3127.nc`, covering the
  `qi_diagnostics_from_state` record path without running a solve or
  optimization sweep.
- QI audit and prefine mirror cleanup now use all sampled Boozer surfaces by
  default (`--prefine-mirror-surface-index all`), preventing a single-surface
  mirror gate from promoting a candidate whose other surfaces violate the
  mirror target.
- README/docs QI coverage now has a dedicated figure and CSV rendered
  from existing `QI_optimization.py` outputs. Reviewed NFP=1/NFP=2/NFP=3 rows
  are separated from deferred NFP=4 stress evidence: `input.nfp2_QI` reaches legacy QI
  `3.09e-4`, mirror `0.225`, elongation `6.43`, aspect `9.999`, iota `-0.5043`
  in 9.8 CPU minutes; the curated `input.QI_stel_seed_3127` reference-family
  baseline reaches legacy QI `1.16e-3`, mirror `0.316`, elongation `3.91`,
  aspect `3.465`, iota `-1.0366` in 1.4 CPU minutes. NFP=4 remains deferred
  rather than completed robustness. The Boozer `|B|` panels
  use line contours only.
- QI continuation-stage checkpointing is now written at workflow level and by
  standalone QI subprocesses: completed stages emit per-stage
  `qi_stage_checkpoint.json` files and a root `stage_checkpoint.json` before
  later high-mode stages or final Boozer audits can time out. Low-resolution
  remote GPU smoke validation confirmed the root and per-stage files are
  present after a later-stage timeout.
- The common-minimal showcase now uses a deterministic `1e-3`
  target-helicity perturbation on active hint modes. A focused remote QA NFP=3
  amplitude study showed `1e-3` reached the iota/aspect gates with a lower final
  objective than `1e-4`; raw VMEC seed decks remain unchanged.
- `solve.py`, `wout.py`, `free_boundary.py`, `driver.py`, and optimization
  modules are too large and need staged refactoring after parity gates are locked.
- The first staged solver/wout refactor has centralized LCFS R/Z residual edge
  masking, extracted residual-implicit packing/zero-m1 helper seams, and
  removed duplicated bss scalxc undo logic behind direct unit tests.

## Milestone 1: QI Truth And Robustness

- [x] Port the smooth differentiable QI residual into `vmec_jax`.
- [x] Add QI mirror-ratio, elongation, and LgradB objective components.
- [x] Move the legacy Goodman-style branch/shuffle QI diagnostic into source as
      an explicit non-differentiable validation utility.
- [ ] Add regression data comparing smooth QI, legacy QI, and
      `omnigenity_optimization` on the same Boozer spectra.
- [x] Make QI sweeps report both optimizer objective and legacy diagnostic so
      visually bad QI fields cannot pass unnoticed.
- [ ] Prove seed robustness with a curated multi-seed matrix: start QI from QI,
      QP, QH, QA, and a simple non-omnigenous seed; document which policies
      reliably converge. Some NFP=1/2/3 probes pass the current narrow gates,
      but full seed robustness remains open. NFP=4 is intentionally deferred:
      bounded May 2026 audits found no passing NFP=4 QI path among the
      available QH warm start, local QH-to-QI cleanup, and archived same-NFP
      references.
- [x] Diagnose remaining QI noisiness by one-DOF scans of Boozer/QI metrics and
      choose default resolutions/weights that preserve ranking while remaining
      differentiable.
- [x] Add and exercise bounded prefine-manifest summaries that rank candidate
      seeds by final objective/improvement, flag objective-history regressions,
      and recommend the next reviewed probe. A 2026-05-12 top-seed smoke
      probe from the optional NFP=2 QP/QI seed reduced the mode-1 QI objective
      by 5.0% with no summary regression flags.
- [x] Fix prefine acceptance so a completed, monotone, finite probe with an
      already-low final QI objective can be promoted even if a tiny
      two-evaluation smoke does not move. Re-summarizing the 2026-05-12
      seven-seed smoke now promotes `qi_omnigenity_nfp1` as the accepted
      stable-low-objective candidate (`objective_final=2.70e-2`).
- [x] Expose bounded QI-prefine ESS controls in the audit CLI and manifest
      commands (`--prefine-use-ess/--no-prefine-use-ess`,
      `--prefine-ess-alpha`). A 2026-05-12 mode-2 continuation ablation on
      `qi_omnigenity_nfp1` failed with the same NaN error with ESS enabled and
      disabled, so the next QI bug is in the mode-2 continuation/objective
      derivative path rather than ESS scaling alone.
- [x] Fix mode-continuation stage seeding so each next stage starts from the
      previous stage's optimized VMEC input instead of rebuilding from the
      original deck plus lifted parameter increments. This prevents projected
      high-order seed modes from reappearing at later stages. The failing
      `qi_omnigenity_nfp1` smoke (`1,1,2`, mode 2) now completes with ESS and
      without ESS; the mode-2 initial exact Jacobian is finite with max entry
      about `3.9e8` instead of the prior `1e120` derivative blow-up.
- [x] Guard final optimization outputs against relaxed trial-solve acceptance
      drift: the optimizer now tracks the best exact accepted-point residual,
      selects it for final outputs when SciPy's last trial-accepted point
      replays worse, and filters worse exact replay rows out of the accepted
      objective history. The bounded `qi_omnigenity_nfp1` mode-3 smoke
      (`1,1,2,2,3`) reached `3.05e-4` final QI objective in about 63 s with
      monotone exact objective history and one rejected trial-exact replay row.
- [x] Add regression coverage for repeated and higher continuation stages using
      both synthetic workflow tests and a real boundary-projection stage test.
      These tests verify that later stages rebuild from the previous optimized
      input, restart with zero increments, and do not resurrect high-order seed
      modes that were intentionally projected out.
- [x] Fix the public QI repeated same-mode continuation policy so
      `CONTINUATION_NFEV = 0` no longer collapses a repeated max-mode cleanup
      schedule into a single direct stage. This protects the intended QI
      cleanup workflow where repeated same-mode solves are used without
      lower-mode continuation stages.
- [x] Extend the bounded QI-prefine manifest/probe lane to include optional
      mirror-ratio and elongation penalties, so seed audits can reject or repair
      QI-looking seeds that have poor engineering geometry. A 2026-05-12 capped
      QP/NFP2 mode-1 probe with QI, mirror, and elongation terms reduced the
      field objective from `3.31` to `0.815` in two function evaluations and
      completed with monotone accepted history.
- [x] Run a tier-3 constrained seed probe comparing promoted QP/NFP2 and
      rejected QA starts. Both histories were monotone, but neither is accepted
      as robust QI evidence: the QP seed reduced the constrained scalar
      objective while still failing mirror/elongation audits, and the QA seed
      passed engineering constraints while remaining poor by smooth/legacy QI.
      The next QI step is objective/diagnostic decomposition, not larger caps.
- [x] Add QI-prefine objective diagnostic decomposition to the manifest summary:
      smooth QI, legacy QI, mirror penalty, elongation penalty, aspect, iota,
      deltas, and flags for scalar-objective improvement that worsens smooth or
      legacy QI. A capped end-to-end QP/NFP2 probe now writes these diagnostics
      from final `input`/`wout` artifacts, preventing scalar-only promotion.
- [x] Make the public QI example and diagnostics distinguish the QI+iota gate
      from the full engineering gate. `MirrorRatio` can now target all sampled
      QI surfaces with smooth extrema/softplus smoothing, and diagnostics report
      per-surface mirror ratios so high-mirror candidates cannot be promoted as
      final QI designs.
- [x] Add a required-tier low-resolution solved-state QI diagnostic gate using
      the bundled `input.QI_stel_seed_3127` seed and solved wout. This protects
      smooth/raw/legacy QI totals, mirror ratio, elongation, surface metadata,
      and Boozer resolution fields through `qi_diagnostics_from_state` without
      launching a solve.
- [ ] Find a QI-preserving mirror cleanup schedule. The current near-axis seed
      result passes smooth/legacy/iota gates but has mirror ratio about `0.97`;
      direct mirror penalties reduce mirror but currently degrade the legacy QI
      metric by one to two orders of magnitude.

Acceptance:

- Best QI has poloidally closed Boozer `|B|` contours, low legacy QI diagnostic,
  mirror ratio at target, acceptable elongation, `abs(iota) >= 0.41`, and aspect
  near the chosen target.
- Seed-robust QI is not accepted from scalar objectives alone; every accepted
  seed lane needs the numerical gates above plus visual Boozer `|B|` contour
  review or an equivalent committed diagnostic artifact.

## Milestone 2: Physics Objectives And Diagnostics

- [x] Promote differentiable `DMerc` from diagnostic/parity into an objective
      with objective-routing and JAX AD tests.
- [x] Add differentiable `J dot B`/JXBFORCE profile objective accessors with
      documented normalization and VMEC/wout parity tests.
- [x] Add differentiable state-derived toroidal current-profile objective
      accessors with documented normalization.
- [x] Add differentiable vector `J` and vector `B` objective accessors with
      documented normalization.
- [x] Add differentiable finite-beta stage-one metric/objective coverage for
      pressure/current-profile-adjacent terms, beta, volavgB, iota, aspect,
      Mercier, Redl bootstrap-current mismatch, and quasisymmetry/omnigenity
      residual wiring.  The finite-beta examples intentionally keep direct
      `FixedBoundaryExactOptimizer` calls visible because their stage-local
      pressure/current closures do not fit the public objective-tuple wrapper
      cleanly yet.
- [ ] Validate the finite-beta stage-one examples against converged
      reference/SIMSOPT finite-beta equilibria, including publication-quality
      Redl Boozer-geometry comparisons.
- [ ] Add physics gates for force residuals, profiles, fluxes, magnetic-field
      reconstruction, Mercier, and Boozer transforms.

Acceptance:

- Each physics objective has a source-level function, an example term, an AD test,
  a finite-difference test, and a documented normalization.

## Milestone 3: Differentiation Architecture

- [ ] Lock AD-vs-finite-difference derivative gates for QA/QH/QP/QI max_mode=1.
- [x] Add a required-tier accepted-point residual derivative gate for the exact
      optimizer path. The circular-tokamak max_mode=1 two-parameter boundary
      test compares the discrete-adjoint Jacobian against central finite
      differences of the accepted residual and keeps the existing dense
      Jacobian/state-tangent/scalar-cotangent checks tied together.
- [ ] Revisit the residual-root implicit layer: reduced state packing, boundary
      control embedding, lambda gauge/branch conditions, and custom VJP/JVP.
- [x] Extract and test the first pure residual-root helper seams: named
      residual packing with structural projectors, zero-m1 host flag selection,
      and shared R/Z edge-force masking used by GN, residual-iteration,
      first-step diagnostics, and residual implicit differentiation paths.
- [ ] Reduce accepted-point replay/Jacobian count in optimization without changing
      accepted equilibria.
- [ ] Add matrix-free/scalar-adjoint optimizer pathways for larger mode numbers.

Acceptance:

- Exact derivatives agree with finite differences on small cases, and optimization
  trajectories are stable without finite-difference fallback.

## Milestone 4: CPU/GPU Performance

- [ ] Profile cold and warm fixed-boundary solves by stage: input, build_static,
      VMEC iterations, residual assembly, Boozer, objective, AD replay, and output.
- [ ] Remove unnecessary accepted-point replays and cache invalidations.
- [ ] Keep CPU fast for small/medium runs; avoid regressions from GPU-specific work.
- [ ] Build a GPU-native tape/replay path that avoids excessive host transfer and
      recompilation.
- [ ] Benchmark LASYM true/false, max_mode 1/2/3, QA/QH/QP/QI, CPU/GPU.
- [x] Add real JSON comparison tooling for exact-optimizer CPU/GPU reports.
      A 2026-05-12 QH max_mode=2 Jacobian profile on commit `f0225ff` measured
      local CPU at 11.29 s for two new Jacobian points and RTX A4000 GPU at
      54.83 s. GPU was 4.86x slower overall, with the ratio tracking
      `jacobian_tape_replay` almost exactly; the next GPU optimization target
      is replay/tangent batching, not residual convergence.
- [x] Make dynamic replay bucketing backend-adaptive. CPU keeps the smaller
      bucket (`32`) because profiling showed it remains faster there; GPU uses
      bucket `128` by default. On the same QH max_mode=2 accepted-point
      Jacobian profile, RTX A4000 runtime dropped to about 15.8 s for two
      perturbed Jacobian points, only 1.40x the 11.29 s CPU baseline, with
      replay time only 1.11x CPU.
- [x] Re-profile the remaining exact-Jacobian cost after backend-adaptive
      bucketing. A 2026-05-12 local CPU smoke spent about 96% of callback time
      in `jacobian_tape_replay`, `jacobian_initial_tangents`, and
      `jacobian_residual_tangents`; an `office` cold fixed-boundary smoke still
      favored CPU (`13.80 s` versus `68.86 s` GPU). The next concrete
      performance target is tangent fusion/reuse, not fixed-boundary force
      kernels.
- [x] Reuse cached affine initial-state tangents in the scalar-gradient path.
      `objective_and_gradient_fun` now projects the reverse tape cotangent
      through the same cached tangent map as dense exact Jacobians instead of
      rebuilding an initial-state VJP each callback. A local QA `max_mode=1`
      CPU smoke improved the same-branch warm gradient callback from about
      `0.279 s` to `0.188 s`; dense least-squares Jacobian cost is unchanged.
- [x] Add detailed accepted-point preconditioner timing and re-run CPU/GPU
      exact-optimizer profiles. On `office`, QH mode-1 cold GPU tape callback
      was `9.29 s` versus `16.97 s` on the same host CPU stack, but QH mode-2
      two-point GPU tape callbacks were still `16.48 s` versus `10.34 s` on the
      local CPU. The remaining GPU bottleneck is tape-build preconditioner
      dispatch plus dense residual-tangent projection, not VMEC convergence.
- [x] Prototype and keep a guarded preconditioner-output scaling fusion for the
      accepted-point tape path. It is algebra-tested and reduced `office` GPU QH
      mode-2 two-point dense Jacobian wall time from `16.48 s` to `15.64 s`
      while reducing preconditioner time from `2.45 s` to `1.67 s`; QH mode-1
      cold callback was neutral/slightly worse (`9.29 s` to `9.47 s`), so the
      fusion is kept on non-CPU backends only. A local CPU QH mode-2 profile
      after this guard measured `10.146 s` cold and `1.546 s` hot for two dense
      Jacobian points, preserving the pre-fusion CPU path while keeping the GPU
      win. The remaining target is larger-mode tape-build/replay structure.
- [x] Add focused regression coverage for the GPU-only preconditioner-output
      fusion gate without requiring a GPU. The hot-path tests now verify that
      CPU backends do not call the fused scaler, a simulated GPU backend does
      call it with lambda-update scaling enabled, and the one-iteration update
      matches the unfused CPU path on a tiny fixed-boundary case.

Acceptance:

- Published performance tables identify when GPU is expected to win, when CPU is
  preferred, and why.

## Milestone 5: VMEC Parity And Numerical Gates

- [ ] Expand VMEC2000/VMEC++ parity gates to fixed/free, axisymmetric/non-axis,
      stellarator-symmetric/LASYM, single-grid/multigrid, and finite-beta cases.
- [ ] Compare converged equilibria rather than fragile short finite-step states
      unless the test is explicitly a solver-trace regression.
- [ ] Add efficient wout-field parity gates for geometry, profiles, B, J, iota,
      aspect, Mercier, and force residuals.
- [x] Add a required-tier converged-wout matrix gate over bundled VMEC2000
      outputs. The gate covers representative fixed/free,
      axisymmetric/non-axisymmetric, `lasym=False`/`lasym=True`, and
      single-grid/multigrid fixtures without launching VMEC2000.
- [x] Add an executable-backed converged-wout benchmark regeneration script.
      `tools/diagnostics/converged_wout_parity_benchmark.py` discovers
      `$VMEC2000_EXEC`, `~/bin/xvmec2000`, PATH, and adjacent STELLOPT builds,
      can opt into recursive local executable inventory, de-duplicates symlinks,
      and records VMEC2000/vmec_jax residual, runtime, scalar, and field
      relative-RMS metrics.
- [x] Add a required-tier bundled VMEC2000 `chipf` parity gate for QA/QH/QI,
      circular, and finite-beta shaped-tokamak wout fixtures. This locks the
      half-mesh `chipf` convention and the `chips_from_wout_chipf` round-trip
      without running VMEC2000 in required CI.
- [x] Add a required-tier bundled VMEC2000 stored-field parity gate for QA/QH/QI
      wout fixtures. The test reconstructs Cartesian `|B|` from stored
      contravariant `bsup*` fields and checks it against the same wout's
      `bmnc/bmns` Fourier representation, covering NFP angle handling, Nyquist
      evaluation, and vector-basis conversion without external executables.
- [x] Add required-tier bundled geometry/aspect, Mercier decomposition,
      JXBFORCE endpoint, and VMEC-to-Boozer-input spectral parity gates. These
      run without external executables and cover aspect reconstruction,
      `DMerc = Dshear + Dcurr + Dwell + Dgeod`, profile extrapolation, mode
      table compatibility, half-mesh `lmns`/`iota`, and QH Nyquist field spectra.
- [x] Add required-tier bundled wout profile/current gates for converged
      fixtures. These verify input flux/profile propagation (`phipf`, `phi`,
      finite-beta `pres/presf`), VMEC `iotas -> iotaf` radial smoothing, and
      surface-averaged Ampere finite differences from `buco/bvco` to
      `jcuru/jcurv` across axisymmetric, 3D current-driven, 3D finite-beta, and
      `lasym=True` wouts without launching a solver.
- [x] Preserve stellarator-asymmetric geometry through the in-memory
      VMEC-to-Boozer adapter. `booz_xform_inputs_from_state` now exports
      `rmns`, `zmnc`, and `lmnc` alongside asymmetric magnetic sine channels,
      and the required tests verify both channel presence and propagation into
      `booz_xform_jax` on a bundled `lasym=True` reference.
- [x] Expand optional reference-code gates beyond a single QH smoke: SIMSOPT
      formula and state-derived QS residual checks now cover bundled QA and QH
      fixtures, and VMEC2000 executable end-state validation now covers three
      bounded converged fixed-boundary cases including a 3D QH input.

## Progress Snapshot

Updated 2026-05-27 after the bundled profile/current wout parity gates, QI
selection hardening, exact-Jacobian host-materialization cleanup,
continuation/exact-history hardening, LASYM-Boozer parity, release-checklist
push, 85% and 90% coverage-gate pushes, optional SIMSOPT/VMEC2000 gate expansion, the
converged-wout parity matrix/benchmark pass, QI diagnostic/objective branch
hardening, the exact-output/API/release hygiene push, the custom QI seed audit
documentation/regression gate, the v0.0.10 release, scan-runner cache reuse
across boundary trials, detailed scan-timing diagnostics, reference-state wout
roundtrip diagnostics, the green `main` CI run for `e90d1a2`, LASYM bsubv
wout parity tightening, QI staged-history provenance cleanup, the v0.0.11
release, the QI optimization driver split, and the May 22 helper/refactor
coverage wave, the v0.0.13 release, the QI workflow checkpointing push, QI
resolution-override coverage, and the May 27 minimal-seed helicity-perturbation
update:

Free-boundary branch addendum, 2026-05-28: PR #18
(`feature/freeb-essos-coil-single-stage`) has merged latest `origin/main`,
validated interrupted stage-level beta-scan checkpoints, added
accepted-boundary direct-coil replay AD-vs-FD coverage, and fixed the direct
coil benchmark matrix GPU detector so mixed-platform launches such as
`JAX_PLATFORMS=cpu,cuda` still record concrete CUDA rows.  The latest `office`
quick CPU/CUDA direct-coil matrix shows the tiny `--jit-forces` direct solve is
still CPU-favorable (`0.0525 s` CPU warm versus `0.2346 s` CUDA warm), but
CUDA force assembly is already comparable/slightly faster.  Remaining GPU work
for the free-boundary branch is setup/precompute reuse and scalar/control plus
preconditioner dispatch amortization, not Biot-Savart sampling or the dense
NESTOR solve.  The branch now has measurement-only setup sub-buckets in solver
diagnostics and benchmark summaries so the next cache patch can target the
dominant setup phase rather than the broad `setup_total_s` container.  The
first CUDA probe with that split measured warm setup at `40.4 ms`, dominated by
boundary/profile construction and update constants.  Accelerator host-forward
setup now reuses the existing NumPy row-enforcement path for the initial state
when not tracing, with an explicit `VMEC_JAX_HOST_SETUP_ENFORCE` override; the
first CUDA check improved the tiny direct-coil warm solve from `0.180 s` to
`0.169 s`.  The host flux-profile setup path now also handles concrete
default-`APHI` iota profiles.  A follow-up `office` matrix at head `8eb2a342`
reported CPU warm `0.0521 s` and CUDA warm `0.2318 s` for the tiny
`--jit-forces` row, with force assembly still near parity; the next GPU lane is
therefore setup/control/preconditioner staging or caching, not further
Biot-Savart kernel work.  A subsequent host-profile setup policy
(`VMEC_JAX_HOST_PROFILE_SETUP=auto`) improved the same tiny CUDA row to
`0.1625 s` warm versus `0.0552 s` CPU, with CUDA setup/profile down to
`5.6 ms`; remaining GPU work is now residual scalar materialization,
accepted-control `fsq1`, and preconditioner dispatch.  Follow-up tests of
`VMEC_JAX_HOST_UPDATE_ON_ACCELERATOR=1`, bad-Jacobian probe bypass, and
timing-light rows did not justify a default policy change; the next
performance step is structural control-loop staging/fusion.

- Continuation correctness: 100%. Source fix is implemented and covered by
  synthetic repeated-stage tests, a real boundary-projection stage test, and
  direct/zero-continuation budget tests that lock the no-continuation policy to
  a single target-mode stage.
- Exact accepted-point history/output correctness: 99%. Best-exact selection is
  implemented and tested; accepted histories remain monotone in exact residuals,
  final saved inputs/wouts use the selected best exact state, and non-finite
  exact residuals can no longer be recorded as the best accepted point. SciPy
  failures after a finite exact point now return that best exact point with an
  explicit abnormal-termination record instead of losing the run. The finite-beta
  stage-one save helper now has a synthetic execution test proving it also writes
  the selected best exact final parameters and state.
- Differentiation architecture: 89%. Dense exact Jacobians, scalar reverse
  gradients, state tangents, accepted-residual AD-vs-finite-difference checks,
  solve-free QS residual JVP/VJP checks, QS objective routing JVP/VJP checks,
  QI shared-field objective JVP/VJP checks, optional Lineax solver seams, and
  fixed-boundary implicit validation/zero-unconverged VJP behavior are covered
  on small required-tier cases. Additional lambda-gradient-descent branch gates
  now cover descent, line-search failure, gauge enforcement, and solver-control
  validation, and QI aligned-profile/mirror smooth branches have direct AD
  checks. Full QA/QH/QP/QI max_mode=1 objective derivative gates and
  matrix-free/scalar-adjoint production paths remain open.
- QI validation and seed-robustness: staged, not complete. The tier-2 and
  tier-3 probes are bounded and monotone,
  constrained terms run end-to-end, and manifests now expose QI/engineering
  diagnostic deltas from final artifacts, including scalar-improved but
  QI-worsened cases. A new bundled near-axis seed, `input.QI_stel_seed_3127`,
  is part of the default audit and can be driven by the public QI script to
  smooth/legacy QI below `1e-3` while satisfying `abs(mean_iota) >= 0.41` and
  aspect ratio near 5. QI render selection now marks raw-fallback legacy values
  as non-promotable, so panels cannot silently select stale raw-QI evidence.
  The public QI script now runs the intended repeated same-mode continuation
  cleanup by default, and seed diagnostics now explicitly test error/fail-fast
  semantics for missing Boozer/scalar metrics. Audit and prefine mirror cleanup
  now evaluate all sampled Boozer surfaces by default and preserve that contract
  in manifests/run commands. The docs now show the exact `--case
  label:family:input:wout` workflow for arbitrary user VMEC decks, and the audit
  CLI has a fast regression proving the custom-case path works with the bundled
  `input.QI_stel_seed_3127` fixture. Final-result promotion now has a reusable
  `qi_promotion_score` that prevents mirror-clean but non-QI rows or raw-fallback
  legacy diagnostics from winning README/docs best-row selection.
  `QuasiIsodynamicResidualCeiling` now gives examples and users a differentiable
  soft-wall guard for mirror/elongation cleanup that preserves an accepted QI
  basin. Completed-stage QI checkpoint files are written before later-stage
  timeouts, and the common-minimal showcase uses a `1e-3` target-helicity hint
  after the QA NFP=3 remote amplitude study improved the final objective while
  preserving iota/aspect gates. The remaining open cleanup is running and tuning
  the guarded mirror schedule across unrelated seeds and completing stronger
  multi-seed promotion evidence.
- CPU/GPU performance: 97%. Backend-adaptive replay bucketing, scalar-gradient
  tangent reuse, detailed timing, and GPU-only preconditioner-output fusion are
  in place. Hot-path algebra and CPU/GPU fusion gating are now covered by
  focused CPU-only regressions. The exact-Jacobian residual tangent helper now
  returns the transposed Jacobian on device and blocks before host
  materialization, removing a hidden GPU host transpose/transfer cost. The
  CPU/GPU profiling wrapper now exposes optimizer method, trust-region solver,
  dynamic replay mode, and child stdout/stderr logs for production GPU matrix
  diagnosis. Symmetric GPU exact-Jacobian replay now defaults to 8-column
  chunks for 24+ DOF cases, matching the best bounded `office` profile: QH
  mode-2 exact Jacobian dropped from about `42.0 s` to about `18.0 s`, with
  tape replay dropping from about `22.2 s` to about `5.3 s`. Production
  fixed-boundary auto policy now uses the VMEC-control non-scan loop on CPU and
  GPU because May 2026 `office` profiles showed converged GPU non-scan solves
  faster than scan across QH, QA, QI, and LASYM examples. A fused
  residual-projected replay experiment was profiled and rejected because it was
  neutral on non-LASYM QH and slower on chunked LASYM CPU callbacks. The
  matrix-free linear-operator path now reuses the transpose of the setup
  `jax.linearize` object instead of tracing a second initial-state VJP, and the
  profile comparator exposes the new `linear_operator_initial_transpose`
  bucket. The exact optimizer now also JITs the projected initial-state map
  lazily for new accepted/trial points, reducing a QA max_mode=1 two-point CPU
  exact callback from `7.99 s` to `6.94 s` and peak RSS from about `1593 MiB`
  to `1456 MiB`. The strict-update accelerator helper now also skips unused
  update-RMS reductions when exact-callback histories do not need them, and
  dynamic replay payload staging recognizes accelerator backend names beyond
  the literal `gpu`. Projected accepted-point replay buckets now also feed the
  profiler budget checks and comparison summaries, so GPU/default replay work is
  no longer under-attributed as legacy tape-only replay. The next performance
  blockers are the unattributed compiled work inside accepted VMEC iteration
  loops, scan-trial timing/cache-key evidence, larger-mode accepted-point replay
  cost, and dense residual-tangent projection.
- VMEC parity and physics gates: 99%. Required-tier bundled gates now cover
  `chipf`, stored `B`, input flux/profile propagation, finite-beta
  `pres/presf`, VMEC `iotas -> iotaf` smoothing, surface-averaged current
  finite differences, aspect/geometry, Mercier/JXBFORCE profiles, and
  VMEC-to-Boozer input spectra, including asymmetric Boozer geometry-channel
  propagation for `lasym=True` plus exact LASYM lambda-channel parity into
  Boozer input objects. A new required-tier converged-wout matrix checks
  bundled VMEC2000 outputs across fixed/free, axisymmetric/non-axisymmetric,
  LASYM, and single/multigrid representatives. Optional executable-backed
  converged-wout parity now runs a bounded circular end-state comparison by
  default and keeps slower non-axisymmetric, LASYM, multigrid, and
  free-boundary representatives behind `VMEC2000_NIGHTLY=1`; the regeneration
  script records the same metrics across discovered VMEC2000 executables.
  Optional SIMSOPT QS parity covers QA and QH formula/state diagnostics. The
  parity manifest now has a fast required contract that keeps the
  self-contained optional free-boundary `LASYM=true` case bounded and ready for
  executable-backed validation; the
  bundled synthetic mgrid currents are sign/magnitude matched to the plasma
  current so stock VMEC2000 reaches the vacuum solve instead of aborting with an
  `I_TOR` mismatch, with an optional executable smoke guarding that behavior.
  Full fixed/free/LASYM/finite-beta converged-equilibrium parity is still open.
  The near-zero `bsubvmns` sine covariant-channel reference-state gap is now
  covered by a focused `up_down_asymmetric_tokamak` regression using VMEC's
  IEQUI/asymmetric `bsubv` source for that output channel only. The remaining
  LASYM blocker is the solved-state lambda convergence gap on the `m=1,3,4`
  channels. `freeb_scalpot` remains an instrumented-VMEC2000 diagnostic because
  a stock executable does not emit the required dumps.
- Refactor/API/examples: 99%. Examples are SIMSOPT-like and clearer, finite-beta
  examples expose structured stage/final summaries while preserving direct
  optimizer visibility and have focused adapter coverage. Objective tuple
  routing is now isolated behind a small assembly helper, reducing the next
  refactor surface for additional objective types; new pure helper gates cover
  driver/runtime, implicit, wout, solve, free-boundary, and VMEC-kernel seams,
  including additional branch coverage for solve cadence/preconditioning,
  implicit optional solvers, wout beta/aspect helpers, and driver serialization
  utilities. `vmec_jax.api` now has a tested public import contract for the
  optimization objects, QI promotion helpers, QI cleanup guard, and plotting
  helpers used by examples. QA/QH/QP examples now share a public
  `save_optimization_result()` helper for final input/WOUT/summary output while
  keeping result inspection and plotting visible in the scripts, and QI examples
  use public `jsonable()`/`diagnostic_float()` helpers instead of private shim
  functions. `QI_optimization.py` is no longer a two-thousand
  line monolith: it now keeps the visible user workflow in 547 lines, while
  the bundled case catalog and staged seed-robust promotion/checkpoint
  mechanics live in `qi_optimization_cases.py` and
  `qi_optimization_support.py`. Large
  solver/wout/free-boundary splits remain deferred behind parity gates.
- Docs/release hygiene: latest released baseline is `v0.0.13`, with PyPI
  publication verified. Post-release `main` has local warning-clean Sphinx and a
  clean 95% CI-equivalent coverage pass through the May 27 candidate
  (`2328 passed, 20 skipped, 109 deselected, 1 xfailed`, 95.00%). GitHub
  Actions is green through `152360f`, carrying the minimal-seed
  helicity-perturbation/docs update.
  Performance/discrete-adjoint/docs reflect the current replay and finite-beta
  policies, diagnostics docs cover detailed preconditioner timing, and a
  command-level release checklist now ties local gates, tools/validation
  lint/compile checks, GitHub Actions, artifact hygiene, and optional
  research-grade checks together. Read the Docs is configured to fail on Sphinx
  warnings, release docs use the 95% coverage gate, and package discovery is
  locked to the `vmec_jax` namespace. Released reference assets are ignored so
  local full-tier refreshes cannot accidentally bloat commits.
  The documented custom QI seed audit command was validated end-to-end on
  `input.QI_stel_seed_3127`; final seed-robust QI and GPU-production artifacts
  remain open.

Release-critical lanes requested in this push (continuation, exact
accepted-point output, VMEC parity/physics gates, and docs/release hygiene) are
near closure, and the 95% required coverage ratchet is now locked locally.
Seed-robust QI mirror cleanup, scan-trial performance evidence, and larger-mode
GPU replay remain future work.

Acceptance:

- CI and nightly tests catch real physics/numerics regressions without relying on
  large committed output files.

## Milestone 6: Refactoring And API Hygiene

- [ ] Continue narrow seam extractions that already have parity evidence:
      force/residual helpers, Mercier/Redl algebra, wout schema helpers, and
      optimization tuple/routing policy.
- [ ] Defer broad `solve.py` decomposition into orchestration, residual kernels,
      convergence, fallback policy, tracing, and output modules until solver
      parity and coverage have enough margin to catch regressions.
- [ ] Defer broad `wout.py` and `free_boundary.py` splits until each proposed
      seam has a targeted test and, for free-boundary coupling, a parity or
      executable-backed validation command.
- [ ] Keep compatibility imports in `vmec_jax.__init__`, but move implementation
      to smaller documented modules.
- [ ] Add docstrings and comments where they clarify physics, numerical method, or
      AD behavior.

Acceptance:

- Core modules are small enough to review, tests still pass, and public imports are
  backward compatible.

## Milestone 7: Tests And Coverage

- [x] Raise required CI coverage to the first 80% target with fast tests that
      exercise solver, driver, optimization, wout, free-boundary,
      quasisymmetry, parity, and constraint branches.
- [x] Raise required CI coverage to 85% with meaningful plotting, solver,
      optimization, validation-plan, and optional-reference-gate coverage.
- [x] Raise required CI coverage to 90% after the May 22 local fast suite
      reached 93.18% in 7m20s.
- [x] Raise required CI coverage to 95% after additional source refactors,
      parity/physics gates, and branch tests. The latest clean local
      CI-equivalent run reached 95.10% with `2153 passed, 20 skipped,
      107 deselected` in 8:05.
- [x] Add targeted tests before coverage-only tests: physics gates first, branch
      logic second, I/O/schema third.
- [x] Keep required CI under 10 minutes by using small fixtures and nightly heavy
      matrices.
- [x] Document the testing strategy and explain which tests are smoke, unit,
      parity, physics, and performance gates.

Acceptance:

- Coverage is high because important code paths are tested, not because smoke
  tests call APIs superficially.

## Milestone 8: Docs, Examples, And Release

- [ ] Keep README short: install, CPU/GPU selection, one free-boundary example,
      best QA/QH/QP/QI plots, and reproduction commands.
- [ ] Move detailed sweeps, all-policy panels, validation plots, and equations into
      docs.
- [ ] Make every optimization example self-contained and SIMSOPT-like: define
      parameters, construct the VMEC object, build objective tuples, solve, inspect
      `result`, save/plot selected outputs.
- [ ] Keep Read the Docs full build working and fast enough to diagnose failures.
- [ ] Release only after tests/docs pass or after explicitly documenting any
      temporary release risk.

## Immediate Implementation Queue

1. [x] Finish source-level legacy QI diagnostic tests and wire comparison
   scripts to the shared utility.
2. [x] Make QI sweep summaries distinguish the smooth differentiable objective
   (`qi_raw_total`) from the true legacy branch/shuffle diagnostic
   (`qi_legacy_total`).
3. [x] Run a small QI one-DOF noise/ranking audit with the source-level
   diagnostic.
4. [x] Add the LASYM=True derivative branch to the AD-safe state-level `DMerc`
   diagnostic and keep the user-facing `DMerc` objective on that shared path.
5. [x] Add the first differentiable Redl/bootstrap-current mismatch objective
   on the state-level finite-beta path.
6. [x] Compare the new Redl residual against SIMSOPT's RedlGeomVmec path on a
   committed finite-pressure wout fixture.
7. [ ] Add subprocess-isolated RedlGeomBoozer comparison where booz_xform is
   available and stable.
8. [x] Start the first refactor with a low-risk extraction from the largest
   modules after the new tests are green.
9. [x] Add a bounded no-optimization QI seed audit using
   `qi_diagnostics_from_state` on available solved `input`/`wout` pairs. It
   compares smooth QI, legacy QI, mirror ratio, elongation, iota, and aspect
   ratio before launching a full optimization sweep.
10. [x] Prepare the first deferred seed-robust QI sweep design artifact:
    explicit ranked seeds, hard-capped QI-prefine probe settings, run commands,
    and output locations via the prefine manifest. Full QI/QP/QH/QA/simple
    optimization rows and Boozer contour curation remain open.

## Reassessment After 2026-05-12 Push

Realistic next targets for this development cycle:

1. Keep the required CI coverage gate at 95% and preserve sub-ten-minute py3.11
   coverage runtime by pruning duplicate helper tests, splitting slow
   validation into optional tiers, or refactoring the largest modules so fewer
   synthetic branch tests are required. Current required coverage is 95.10%;
   the next staged ratchet should come from solver/physics refactor seams rather
   than superficial call coverage.
2. Raise the solved-state QI diagnostic from the new low-resolution bundled
   gate to reviewed higher-resolution evidence: smooth QI, legacy QI, mirror
   ratio, elongation, iota, aspect ratio, and Boozer `|B|` contour quality.
3. Continue GPU optimization only where profiling points to a concrete source
   change.  The guarded JVP-only basepoint-carry prototype restored QH mode-2
   GPU replay parity in a short office profile, but it remains opt-in until
   larger mode and trajectory matrices confirm the RSS/runtime tradeoff.  Current
   evidence says the expensive cold exact callback is split between tape build,
   force/residual work, and replay/residual linearization; the next isolated lane
   is `checkpoint_tape_state_jvp_columns` and related dynamic-basepoint replay.
4. Keep refactors small and test-backed.  Good next seams are pure helper
   extractions or schema/routing cleanup already covered by fast tests.  Broad
   rewrites of `solve.py`, `wout.py`, `free_boundary.py`, and replay
   architecture stay deferred until they have a concrete validation command and
   rollback plan.
5. Keep optional VMEC2000/SIMSOPT validation expanding, but keep required CI
   under 10 minutes by default.

Defer beyond the current cycle:

1. Full seed-robust QI from QA/QH/simple non-omnigenous starting points.  This
   needs multiple long sweeps plus visual Boozer audits, not just unit tests.
2. Large module refactors of `solve.py`, `wout.py`, and `free_boundary.py`.
   These should follow higher parity/coverage gates to avoid destabilizing the
   solver.
3. A fully GPU-native replay architecture.  The next step is narrower
   profiling-driven replay reduction; replacing the differentiation architecture
   is a larger design project.
4. Turning multi-seed QI sweeps into required CI.  Keep them manual/nightly
   until the matrix is reliable, bounded, and summarized as lightweight
   artifacts.

## Activity Log

- 2026-05-10: Replaced stale QH-only plan with current research-grade roadmap.
  Started QI validation lane by promoting the legacy branch/shuffle diagnostic
  from an example script into source and tests.
- 2026-05-10: Updated QI sweep diagnostics so `qi_legacy_total` is the actual
  non-differentiable branch/shuffle score, while `qi_raw_total` remains the
  smooth differentiable objective used by the optimizer.
- 2026-05-10: Added `examples/optimization/scan_qi_boozer_mode.py` to scan a
  selected Boozer coefficient and compare smooth-vs-legacy QI metric roughness.
  First bundled QI seed audit selected Boozer mode `(m=0, n=2)` and found both
  metrics minimized at scale `1.25`, with roughness about `1e-2`.
- 2026-05-10: Added a JAX AD gate for existing finite-beta workflow objectives
  (`VolavgB`, `BetaTotal`, magnetic well). `DMerc` remains a larger source port
  because the current parity implementation lives in the NumPy/wout Mercier
  kernel.
- 2026-05-10: Started the `DMerc` port by adding
  `mercier_terms_from_profile_integrals`, a JAX-differentiable implementation
  of the VMEC Mercier algebra once the surface integrals are known. The
  remaining port is the geometric surface-average assembly.
- 2026-05-10: Added `mercier_surface_integrals_from_realspace`, a JAX
  implementation of the VMEC Mercier `tpp/tbb/tjb/tjj` reductions from
  real-space channels. The remaining `DMerc` objective work is state wiring for
  `gpp` and `bdotk`.
- 2026-05-10: Added `mercier_bdotk_from_covariant_derivatives`, a JAX
  implementation of the VMEC jxbforce `itheta/izeta/bdotk` block from filtered
  covariant-field derivatives. The remaining hard part is deriving those
  filtered derivative channels from the state path without falling back to wout.
- 2026-05-10: Added `mercier_gpp_from_realspace_geometry`, a JAX
  implementation of the VMEC Mercier contravariant `gpp` geometry channel from
  even/odd real-space geometry. The remaining `DMerc` wiring is now focused on
  filtered covariant-field derivatives.
- 2026-05-10: Added `mercier_bsubs_derivatives_lasym_false`, a JAX
  implementation of the stellarator-symmetric jxbforce spectral derivative
  reconstruction for `bsubsu`/`bsubsv`. The next DMerc step is state-based
  `bsubs` wiring plus the LASYM=True branch.
- 2026-05-10: Added `mercier_bsubs_half_mesh_from_geometry` and
  `mercier_bsubs_full_mesh_from_half_mesh`, covering the VMEC `bss.f` radial
  covariant field assembly and jxbforce full-mesh averaging. Remaining DMerc
  wiring needs state-synthesized `rv12/zv12` geometry and the LASYM=True branch.
- 2026-05-10: Added `mercier_zeta_half_mesh_from_realspace_geometry`, covering
  the VMEC half-mesh `rv12/zv12` toroidal derivative geometry from even/odd
  channels. Remaining DMerc wiring needs state synthesis of those channels and
  the LASYM=True derivative branch.
- 2026-05-10: Added `mercier_realspace_geometry_channels_from_state`, a
  source-level JAX synthesis helper for the VMEC even/odd R/Z geometry channels
  used by Mercier. Remaining work is composing the state-level `DMerc` residual
  and adding LASYM=True derivative reconstruction.
- 2026-05-10: Added `mercier_terms_from_state`, a differentiable state-level
  Mercier diagnostic for stellarator-symmetric equilibria. It composes the
  JAX geometry, `gpp`, `bsubs`, derivative, `bdotk`, surface-integral, and
  algebra kernels. This was the base path later wrapped by `vj.DMerc`.
- 2026-05-10: Added `vj.DMerc`, a smooth lower-bound objective wrapper around
  `mercier_terms_from_state` for stellarator-symmetric finite-beta
  optimization examples. The same wrapper now also covers LASYM=True after the
  derivative branch below.
- 2026-05-10: Added the JAX LASYM=True `bsubs` derivative reconstruction and
  LASYM cos/sin phase geometry branch for state-level `DMerc`, with transform
  parity and AD tests. Remaining finite-beta source objective work is Redl
  bootstrap-current mismatch plus higher-level parity gates.
- 2026-05-10: Added VMEC/wout parity coverage for state-level `DMerc` on the
  bundled finite-beta QI input. This also fixed the bss half-mesh geometry path
  so `rs12/zs12` include VMEC's odd-channel correction before forming `B_s`.
- 2026-05-10: Added differentiable VMEC/JXBFORCE profile accessors
  (`jdotb`, `bdotb`, `bdotgradv`) to the state-level Mercier path, exposed
  them as `vj.JDotB`, `vj.BDotB`, and `vj.BDotGradV` objectives, and added
  unit plus full-gated VMEC/wout parity tests.
- 2026-05-10: Added `vj.ToroidalCurrent` and `vj.ToroidalCurrentGradient`
  objective objects for the state-derived VMEC/Mercier `torcur` and `ip`
  profiles, with example comments, documentation, AD checks, and finite-
  difference derivative coverage.
- 2026-05-10: Added `vj.BVector` for Cartesian magnetic-field targeting and
  `vj.JVector` for VMEC-coordinate current-density targeting
  `(itheta/sqrtg, izeta/sqrtg)`, with docs, example comments, and derivative
  tests.  This closes the first finite-beta vector-diagnostics API lane.
- 2026-05-10: Added a differentiable `vj.RedlBootstrapMismatch` objective.
  The Redl algebra follows SIMSOPT/Redl et al.; vmec_jax evaluates the needed
  geometry from state-level VMEC channels using fixed trapped-fraction
  quadrature. Added tests, docs, and example wiring. The next lane is parity
  against SIMSOPT RedlGeomVmec/RedlGeomBoozer on finite-beta equilibria.
- 2026-05-10: Added an optional SIMSOPT parity test for the Redl algebra
  itself, comparing `jdotB`, `L31`, and `L32` against
  `simsopt.mhd.bootstrap.j_dot_B_Redl` when SIMSOPT is installed.
- 2026-05-10: Added an optional SIMSOPT `RedlGeomVmec` parity test on the
  committed `wout_shaped_tokamak_pressure.nc` fixture. With shared SIMSOPT
  Redl geometry the residuals agree tightly; the public differentiable
  vmec_jax state-geometry path is held to a 2% envelope. The Boozer geometry
  lane remains subprocess-isolated because the local SIMSOPT Boozer path exits
  the Python process in this environment.
- 2026-05-10: Started the source-organization refactor by splitting pure Redl
  bootstrap profile, trapped-fraction, current-fit, and mismatch algebra into
  `vmec_jax.redl_bootstrap`. `finite_beta` keeps the state-dependent geometry
  and Mercier/JXBFORCE plumbing while preserving existing public imports.
- 2026-05-10: Continued the source split by extracting pure Mercier algebra,
  Mercier surface-integral reductions, and JXBFORCE profile reductions into
  `vmec_jax.mercier`. State-level VMEC composition remains in `finite_beta`,
  and the existing public `finite_beta`/top-level imports are preserved.
- 2026-05-10: Ran the CI-equivalent fast coverage gate locally:
  `389 passed, 21 skipped, 85 deselected`, total coverage `60.77%`, runtime
  about `6:07`. Raised the CI floor from 58% to 60%; the next coverage step
  should be targeted tests for `driver`, `optimization_workflow`, `plotting`,
  and selected solver branch logic before moving to 65%.
- 2026-05-10: Added focused optimization-workflow API tests for QI tuple
  semantics: nonzero QI targets are rejected, mixed `QuasiIsodynamicOptions`
  objects are rejected, and QI field objectives fail clearly outside the shared
  QI solve path.
- 2026-05-10: Added edge-case tests for the split finite-beta kernels: Mercier,
  Mercier surface-integral, and JXBFORCE helpers return zero profiles on
  too-short radial meshes, and trapped-fraction geometry rejects invalid array
  shapes explicitly.
- 2026-05-10: Added cheap driver-policy unit tests for solver-mode aliases,
  scalar/list parsing, final-FTOL selection, integer budget allocation, residual
  convergence checks, and history-based final residual extraction. This targets
  CI-relevant control-flow branches without adding additional VMEC solves.
- 2026-05-10: Added `least_squares_solve` dispatch tests for ordinary QS and QI
  problems. These verify SIMSOPT-like objective tuple metadata and shared
  `QuasiIsodynamicOptions` are routed to the correct low-level optimization
  path without executing expensive optimization runs.
- 2026-05-10: Added synthetic rendering coverage for the public optimization
  plotting helpers: 3-D LCFS comparison, LCFS `|B|` contour panels, and
  objective-history plots. This protects README/docs plot wrappers without
  adding real VMEC output fixtures.
- 2026-05-10: Re-ran the CI-equivalent fast coverage gate after the driver,
  workflow-dispatch, and plotting tests: `400 passed, 21 skipped, 85 deselected`,
  total coverage `61.44%`, runtime about `6:57`. Raised the CI floor from 60%
  to 61%; the next material coverage step is 65% and should still come from
  targeted tests/refactors, not synthetic coverage padding.
- 2026-05-10: Added focused optimization-workflow tests for mode-continuation
  policy helpers and QI objective factories. These cover QI residual weighting,
  Boozer surface slicing for mirror-ratio penalties, elongation penalties, and
  LgradB state-objective plumbing without executing VMEC or Boozer solves.
- 2026-05-10: Added plotting API compatibility tests for the legacy
  `plot_qh_optimization` wrapper and angle-label formatting. The examples use
  the generic plotting helpers directly, while the wrapper remains protected for
  downstream scripts that still call it.
- 2026-05-10: Added driver-policy tests for JAX backend fallback, dynamic-scan
  environment parsing, and diagnostic-history convergence fallbacks. These
  target CPU/GPU runtime-policy control flow and convergence reporting without
  adding new heavy solve cases.
- 2026-05-10: Re-ran the CI-equivalent fast coverage gate after the QI,
  plotting-wrapper, and driver-policy test slices: `405 passed, 21 skipped,
  85 deselected`, total coverage `61.65%`, runtime about `7:01`. Kept the CI
  floor at 61%; there is not enough platform margin yet to raise it to 62%.
- 2026-05-10: Added workflow tests for `FixedBoundaryVMEC.from_input`, the
  `mpol/ntor >= max(min_vmec_mode, max_mode+2)` resolution policy, QS objective
  wrappers, and lower-bound/QI helper edge paths. These protect the
  SIMSOPT-like example API and the high-mode optimization resolution policy.
- 2026-05-10: Added synthetic rendering tests for bundled overview and
  `|B|`/contravariant/covariant parity plot writers. The tests monkeypatch
  VMEC/JAX kernels and use toy `wout` data, so docs/validation plotting entry
  points are covered without adding solver runtime.
- 2026-05-10: Added synthetic coverage for the `plot_wout()` CLI diagnostic
  renderer. This protects the four-output `vmec_jax --plot wout_*.nc` workflow
  used in docs and user-facing examples.
- 2026-05-10: Re-ran the CI-equivalent fast coverage gate after workflow and
  plotting coverage slices: `411 passed, 21 skipped, 85 deselected`, total
  coverage `62.74%`, runtime about `6:31`. Raised the CI floor from 61% to 62%;
  63% is still too close to the local value for platform-safe CI gating.
- 2026-05-10: Added CLI entrypoint coverage for `python -m vmec_jax`, including
  the XLA/PjRt warning-suppression environment defaults and the module-level
  dispatch to `vmec_jax.cli.main`.
- 2026-05-10: Expanded `_compat` coverage around NumPy-mode dispatch, no-op JIT
  fallback behavior, compilation-cache environment policy, and best-effort JAX
  cache configuration updates. This protects the CPU/GPU cache and warning
  suppression defaults that affect first-run user experience.
- 2026-05-10: Re-ran the CI-equivalent fast coverage gate after the module
  entrypoint and `_compat` tests: `409 passed, 21 skipped, 85 deselected`, total
  coverage `62.91%`, runtime about `6:28`. Kept the CI floor at 62%; 63% still
  needs more margin before it is safe to enforce in CI.
- 2026-05-10: Added synthetic optimization-helper coverage for max-mode
  boundary extension, LASYM boundary coefficient families, fixed-boundary
  context precomputation, no-op boundary truncation, and Gauss-Newton edge
  exits. Verified with `python -m pytest tests/test_optimization_helpers.py -q`
  (`48 passed, 1 skipped`) and `ruff check tests/test_optimization_helpers.py`.
- 2026-05-10: Added wout helper coverage for JAX VMEC weights, beta/aspect edge
  cases, bsubv equilibrium correction guards, and ctor branch selection. Verified
  with `python -m pytest tests/test_wout_helpers.py -q` (`6 passed`) and
  `ruff check tests/test_wout_helpers.py`.
- 2026-05-10: Re-ran the CI-equivalent fast coverage gate after the optimization
  and wout helper slices: `420 passed, 21 skipped, 85 deselected`, total coverage
  `63.15%`, runtime about `6:32`. Kept the CI floor at 62%; the current 63%
  margin is still too narrow for a platform-safe threshold bump.
- 2026-05-10: Added CLI dispatch coverage for CPU default policy selection,
  explicit parity/fast modes, VMEC++ restart flags, profiling hooks, explicit
  output paths, missing-input errors, invalid `--jit-forces`, and read failures.
  Verified with `python -m pytest tests/test_cli_helpers.py -q` (`11 passed`)
  and `ruff check tests/test_cli_helpers.py`.
- 2026-05-10: Added residual diagnostic API guard coverage for missing-JAX,
  singular `gamma=1`, and pressure-shape validation paths. Verified with
  `python -m pytest tests/test_fast_physics_kernels.py -q` (`11 passed`) and
  `ruff check tests/test_fast_physics_kernels.py`.
- 2026-05-10: Added visualization helper coverage for VTK scalar/vector array
  serialization, structured-grid and polyline validation errors, 3-D point data,
  automatic wout path resolution, and the public surface/fieldline/volume export
  workflow using synthetic kernels. Verified with
  `python -m pytest tests/test_visualization_vtk.py -q` (`6 passed`) and
  `ruff check tests/test_visualization_vtk.py`.
- 2026-05-10: Re-ran the CI-equivalent fast coverage gate after CLI, residuals,
  and visualization slices: `429 passed, 21 skipped, 85 deselected`, total
  coverage `63.53%`, runtime about `6:25`. Kept the CI floor at 62%; the next
  safe bump remains 65% after more meaningful coverage/refactor work.
- 2026-05-10: Added VMEC2000 helper coverage for executable discovery, missing
  executable errors, fake subprocess execution, namelist patching inside
  `run_xvmec2000`, threed1 discovery, trace parsing, and runtime capture.
  Verified with `python -m pytest tests/test_fast_physics_kernels.py -q`
  (`12 passed`) and `ruff check tests/test_fast_physics_kernels.py`.
- 2026-05-10: Added profile edge-case coverage for LRFP iota inversion, empty
  cubic-spline current auxiliary arrays, non-monotone AUX trimming, signed
  `SPRES_PED` normalization, and unsupported profile-type errors. Verified with
  `python -m pytest tests/test_fast_physics_kernels.py -q` (`13 passed`) and
  `ruff check tests/test_fast_physics_kernels.py`.
- 2026-05-10: Re-ran the CI-equivalent fast coverage gate after VMEC2000 helper
  and profile edge-case slices: `431 passed, 21 skipped, 85 deselected`, total
  coverage `63.67%`, runtime about `6:21`. Kept the CI floor at 62%; 64% still
  has too little cross-platform margin, and the next safe threshold bump remains
  65% after more meaningful coverage/refactor work.
- 2026-05-10: Added initial-guess helper coverage for axis alias parsing,
  VMEC theta-flip logic, frozen zero-axis initialization, axis override
  extraction, and NumPy/JAX axis-recompute parity. This exposed and fixed a
  real JAX-path mismatch: when the axis scan finds no positive min-Jacobian
  improvement, `_recompute_axis_from_state_vmec_jax` now keeps VMEC's midpoint
  fallback instead of moving to the least-bad grid point. Verified with
  `python -m pytest tests/test_init_guess.py -q` (`8 passed`) and
  `ruff check vmec_jax/init_guess.py tests/test_init_guess.py`.
- 2026-05-10: Re-ran the CI-equivalent fast coverage gate after the
  initial-guess parity fix: `438 passed, 21 skipped, 85 deselected`, total
  coverage `64.08%`, runtime about `6:27`. Kept the CI floor at 62%; the next
  threshold bump should wait for a wider margin from additional physics-kernel
  or solver-control coverage.
- 2026-05-10: Added runtime-neutral helper coverage for state pack/unpack,
  `VMECState` and `Coords` PyTree roundtrips, VMEC mode ordering, Nyquist mode
  padding, and VMEC default grid sizing including axisymmetric `NZETA=1`
  collapse. Verified with
  `python -m pytest tests/test_state_coords_helpers.py tests/test_modes_helpers.py -q`
  (`6 passed`) and `ruff check tests/test_state_coords_helpers.py tests/test_modes_helpers.py`.
- 2026-05-10: Re-ran the CI-equivalent fast coverage gate after state/coords
  and mode helper tests: `444 passed, 21 skipped, 85 deselected`, total coverage
  `64.13%`, runtime about `6:26`. Kept the CI floor at 62%; 64% remains too
  tight for a threshold bump.
- 2026-05-10: Added cheap helper coverage for radial derivative edge cases
  (`ns=1`, `ns=2`), periodic angle-step validation, and Nyquist mode/basis cache
  reuse. Verified with
  `python -m pytest tests/test_grid_radial_nyquist_helpers.py -q` (`3 passed`)
  and `ruff check tests/test_grid_radial_nyquist_helpers.py`.
- 2026-05-10: Re-ran the CI-equivalent fast coverage gate after the
  radial/grid/Nyquist helper slice: `447 passed, 21 skipped, 85 deselected`,
  total coverage `64.15%`, runtime about `6:28`. Kept the CI floor at 62%;
  next meaningful target is enough coverage margin to raise the gate directly to
  65% without CI flakiness.
- 2026-05-10: Added VMEC transform-helper coverage for wrout Nyquist cosine/sine
  coefficients, jxbforce cosine/sine coefficients, Nyquist synthesis, JAX/NumPy
  `chipf` edge cases, and pure driver budget/residual helper edge cases. These
  tests compare vectorized paths against explicit loop formulas using synthetic
  VMEC trig tables, so they are physics-relevant without running a solve.
  Verified with
  `python -m pytest tests/test_wout_helpers.py tests/test_driver_api.py -q`
  (`57 passed, 1 skipped`) and
  `ruff check tests/test_driver_api.py tests/test_wout_helpers.py`.
- 2026-05-10: Re-ran the CI-equivalent fast coverage gate after the transform
  and driver-helper slice: `450 passed, 21 skipped, 85 deselected`, total
  coverage `64.66%`, runtime about `6:29`. The GitHub py3.11 job on commit
  `7ec28db` ran a slightly different optional-test mix (`440 passed, 31 skipped,
  85 deselected`) and reported `62.99%`, so the enforced CI floor stays at 62%
  until solver/physics-kernel coverage creates a wider CI-side margin. Keep the
  next threshold target at 65% only after that margin exists.
- 2026-05-11: Added fast numerical-helper coverage for Fourier projection
  roundtrips, non-stacked helical-basis fallback evaluation, VMEC-style
  angle/radial integral conventions, two-power and cubic-spline profile
  variants, LRFP iota inversion, pedestal clipping, and unsupported-profile
  guards. Verified with
  `python -m pytest tests/test_numerics_helper_edges.py -q` (`6 passed`),
  `ruff check tests/test_numerics_helper_edges.py`, and a combined helper slice
  (`63 passed, 1 skipped`). Re-ran the CI-equivalent fast coverage gate:
  `456 passed, 21 skipped, 85 deselected`, total coverage `64.84%`, runtime
  about `6:29`. This is progress but still not enough margin to raise the
  enforced CI floor above 62%.
- 2026-05-11: Added fast VMEC half-mesh and residual-diagnostic coverage for
  pshalf staggering, even/odd metric decompositions, one-surface half-mesh
  behavior, and the normalized `force_residuals_from_state` path using a
  controlled differentiable synthetic geometry. Verified the targeted slice with
  `python -m pytest tests/test_fast_physics_kernels.py tests/test_bcovar_lambda_axis_closure.py -q`
  (`19 passed`) and `ruff check tests/test_fast_physics_kernels.py tests/test_bcovar_lambda_axis_closure.py`.
  Re-ran the CI-equivalent fast coverage gate:
  `459 passed, 21 skipped, 85 deselected`, total coverage `65.00%`, runtime
  about `7:11`. Keep the enforced CI floor at 62% for now: local coverage only
  barely rounds to 65%, and previous GitHub jobs have reported lower coverage
  for the same tree due to optional-test/platform differences.
- 2026-05-11: Added fast VMEC force-balance and force-helper coverage for
  `lforbal` radial weights, VMEC current/equif/plascur formulas, lforbal
  m=1,n=0 correction application, synthetic preconditioner-derived factors,
  VMEC force radial stencils, parity coefficient selection, and R/Z residual
  scalar normalization. Verified with
  `python -m pytest tests/test_vmec_forces_fast_helpers.py tests/test_lforbal_fast_helpers.py -q`
  (`9 passed`) and `ruff check tests/test_vmec_forces_fast_helpers.py tests/test_lforbal_fast_helpers.py`.
  Re-ran the CI-equivalent fast coverage gate:
  `468 passed, 21 skipped, 85 deselected`, total coverage `65.61%`, runtime
  about `7:06`. Raised the enforced py3.11 CI coverage floor conservatively
  from 62% to 63%; do not raise to 65% until GitHub's measured coverage has
  enough cross-platform margin.
- 2026-05-11: Added fast VMEC constraint-helper coverage for `faccon` indexing,
  heuristic `tcon` scaling, short-mesh preconditioner guards, fallback
  preconditioner integration weights, and `alias_gcon` shape guards. Also
  enabled the lasym `alias_gcon` loop-parity test; it now matches the explicit
  VMEC-style reference loops to floating-point noise. Verified with
  `python -m pytest tests/test_vmec_constraints_fast_helpers.py tests/test_vmec_alias_gcon.py -q`
  (`7 passed`) and `ruff check tests/test_vmec_constraints_fast_helpers.py tests/test_vmec_alias_gcon.py`.
  Re-ran the CI-equivalent fast coverage gate:
  `474 passed, 20 skipped, 85 deselected`, total coverage `65.65%`, runtime
  about `6:19`. Keep the enforced CI floor at 63% until GitHub py3.11 has
  enough margin for a safe 64-65% bump.
- 2026-05-11: Added optional local validation gates against VMEC2000 and
  SIMSOPT without making required CI depend on either package. The VMEC2000
  gate runs two short executable stage-trace comparisons (`input.circular_tokamak`
  and `input.basic_non_stellsym_pressure`) behind `VMEC2000_INTEGRATION=1`;
  the SIMSOPT gate compares the VMEC-only QH quasisymmetry residual formula on
  `wout_nfp4_QH_warm_start.nc` behind `RUN_SIMSOPT_VALIDATION=1`. Added
  deterministic `vmec_residue` helper coverage for VMEC m=1 constraints,
  constrained Z-force zeroing, `scalxc`, getfsq radial edge policy, and dynamic
  scalar residual normalization. Verified locally with the optional validation
  commands, full docs build, and the CI-equivalent fast gate:
  `476 passed, 21 skipped, 86 deselected`, total coverage `65.80%`, runtime
  about `7:08`; GitHub Actions run `25660215594` passed.
- 2026-05-11: Hardened the `RUN_FULL=1` scalar-residual parity test so it
  prefers downloaded `_reference.nc` assets but falls back to the bundled
  VMEC2000 `wout_circular_tokamak.nc` when the larger asset bundle has not been
  fetched. Verified with
  `RUN_FULL=1 python -m pytest tests/test_residue_getfsq_parity.py -q`
  (`1 passed`).
- 2026-05-11: Added a synthetic-only `vmec_realspace` invariant slice covering
  LASYM full-grid synth/analyze round-trip, helical mixed-partial commutation
  with physical zeta derivatives, and circular axisymmetric geometry identities.
  This raises targeted `vmec_realspace.py` statement coverage from about 47% to
  about 80% without adding large fixtures or slow VMEC solves. Verified with
  `python -m pytest tests/test_vmec_realspace.py tests/test_vmec_realspace_invariants.py -q`
  and `python -m ruff check tests/test_vmec_realspace_invariants.py`; the
  CI-equivalent fast gate now reports `479 passed, 21 skipped, 86 deselected`,
  total coverage `65.83%`, runtime about `6:24`.
- 2026-05-11: Performance lane next targets after profiling review: avoid
  replay-only `update_rms` work unless update limiting or diagnostics need it,
  reuse residual-derived accepted-point objective totals for generic/QI
  workflows, and make scalar-adjoint trial acceptance cheaper while exact-checking
  accepted states. These are intentionally deferred from the docs/validation
  patch because they touch discrete-adjoint replay semantics.
- 2026-05-11: Implemented the first narrow replay-cost reduction: replay-only
  VJP/JVP maps pass `need_update_rms=False`, while the default accepted-solve
  path still reports update RMS diagnostics and any solve with
  `limit_update_rms=True` still computes RMS for clipping. Verified with
  `python -m pytest tests/test_discrete_adjoint_qh.py tests/test_discrete_adjoint_chunking.py tests/test_optimization_helpers.py -q`
  (`76 passed, 19 skipped`). A small CPU exact-Jacobian diagnostic on QH
  `max_mode=1`, 8 DOFs, 20 inner iterations completed in `7.707 s`
  (`jacobian_tape_replay=2.924 s`, `exact_tape_build=0.319 s`).
- 2026-05-11: Reassessed the QI validation and refactor roadmap from the
  current docs/tests. Fast QI tests now cover metric semantics and synthetic
  ranking, but not multi-seed optimizer robustness. The next validation step is
  a small solved-state `qi_diagnostics_from_state` gate; the full QI/QP/QH/QA
  and simple non-omnigenous seed matrix remains deferred manual/nightly work.
  Refactoring stays limited to narrow tested seams until solver/free-boundary
  parity gates have more margin.
- 2026-05-11: Added a bounded no-optimization QI seed audit CLI at
  `examples/optimization/audit_qi_seed_suitability.py`. It ranks solved
  `input`/`wout` seed pairs by QI diagnostics and engineering constraints,
  uses `OMNIGENITY_OPTIMIZATION_ROOT` for optional external reference seeds,
  and writes JSON/CSV summaries for selecting QI/QP/QH/QA/simple starts before
  expensive seed-robust optimization sweeps.
- 2026-05-11: Started the large-module refactor with narrow tested seams:
  `vmec_jax._solve_runtime` owns scan runtime helpers previously embedded in
  `solve.py`, and `vmec_jax.wout_schema` owns `WoutData` plus low-level wout
  schema helpers while `vmec_jax.wout` keeps compatibility re-exports.
- 2026-05-11: Extended the QI seed audit with `--prefine-probes plan|run`.
  The default remains audit-only. The plan mode writes a hard-capped manifest
  for tiny QI-only prefine probes, including selected seeds, commands, expected
  artifacts, and objective settings before any expensive seed-robust run.
- 2026-05-11: Continued the refactor with additional narrow seams:
  `vmec_jax.wout_io` owns low-level netCDF mode/string/variable helpers, and
  `_solve_runtime` now also owns dump iteration parsing plus simple dump gate
  policy. Public compatibility through `vmec_jax.wout` and `vmec_jax.solve`
  was preserved.
- 2026-05-12: Added continuation regression tests for repeated and higher mode
  schedules, including synthetic workflow control-flow and a real
  boundary-projection stage gate. These protect the fix that carries optimized
  VMEC inputs forward between continuation stages and keeps projected high
  modes zero.
- 2026-05-12: Added bundled wout `chipf` parity tests across circular,
  finite-beta shaped tokamak, QH, QA, and QI fixtures. The tests verify the
  VMEC2000 half-mesh convention and the `chips_from_wout_chipf` round-trip in
  required-tier CI without running an external executable.
- 2026-05-12: Re-profiled performance lanes after backend-adaptive replay
  bucketing. `office` cold fixed-boundary smoke still favors CPU (`13.80 s`
  CPU versus `68.86 s` GPU), while local exact-Jacobian callback time is
  dominated by replay and tangent construction/projection. The next performance
  patch should target accepted-point tangent fusion/reuse.
- 2026-05-12: Added the first tangent-reuse performance patch:
  scalar-gradient callbacks now reuse the cached affine initial-state tangent
  map and project reverse tape cotangents through it. This reduced one local
  same-branch warm QA gradient callback from about `0.279 s` to `0.188 s`
  while preserving dense-Jacobian gradient parity.
- 2026-05-12: Ran a tier-2 bounded QI seed-robust probe. The promoted QP/NFP2
  seed improved the prefine objective from `2.4465e-2` to `1.4150e-2`
  with monotone history, but final-state audit still failed mirror and
  elongation. The QA low-resolution comparator did not improve and remains a
  rejected QI seed candidate for this policy.
- 2026-05-12: Added a required-tier bundled stored-field parity gate:
  reconstructed Cartesian `|B|` from VMEC2000 `bsup*` fields agrees with
  bundled `bmnc/bmns` fields for QA, QH, and QI fixtures within a small
  envelope.
- 2026-05-12: Validated the hot-path helper coverage for the GPU-only
  preconditioner-output fusion. Added a CPU-only gate regression that rejects
  fused scaling on a CPU backend, simulates a GPU backend to select it with
  lambda-update scaling enabled, and verifies one-iteration `circular_tokamak`
  update parity with the unfused path. Verified with
  `pytest -q tests/test_solve_hotpaths.py tests/test_tcon_precondn_diag.py`
  (`15 passed`).
- 2026-05-12: Added the first required-tier accepted-point AD-vs-finite-
  difference derivative gate for the exact optimizer. The small circular
  tokamak max_mode=1 boundary test compares the dense discrete-adjoint Jacobian
  against central finite differences of `residual_fun`; the focused derivative
  trio passed in `73.74 s`.
- 2026-05-12: Added the new near-axis QI seed
  `examples/data/input.QI_stel_seed_3127` plus its solved wout to the default
  seed audit. A direct mode-2 QI-only prefine from this seed reached low QI
  diagnostics (smooth about `8.1e-4`, legacy about `4.4e-4`) before SciPy hit a
  non-finite trust-region step; the exact optimizer now returns the best finite
  exact point with `success=False` and records the optimizer exception. A
  simultaneous constrained mode-2 run produced a more engineering-reasonable
  state (aspect about `5.67`, elongation passing) but only legacy QI about
  `1.5e-2` and a slightly high mirror ratio, so QI-preserving constrained
  cleanup remains the next physics task.
- 2026-05-14: Consolidated `QI_optimization.py` into the single recommended
  multi-seed entry point. The default `RUN_CASE = "nfp2_qi"` keeps the current
  best mirror-aware NFP=2 lane, while `RUN_CASE = "qi_stel_seed_3127"` and
  `RUN_CASE = "nfp4_qh_warm_to_qi"` provide bounded seed-robustness probes
  from the bundled unrelated stellarator seed and the NFP=4 QH warm start.
  The script now takes NFP from the selected VMEC input, prints the active
  policy before the solve, keeps objective tuples visible, writes a
  Boozer-coordinate `|B|` line-contour plot, and promotes candidates only
  through independent smooth-QI, legacy-QI, iota, mirror, and elongation gates.
  Local CPU probes after the change gave: `qi_stel_seed_3127` passed the
  QI+iota gate in `90.6 s` (smooth QI `1.09e-3`, legacy QI `3.16e-4`,
  `|iota|=0.897`, aspect `4.995`) but failed mirror (`0.958`); the
  `nfp4_qh_warm_to_qi` case ran end-to-end in `181 s` and improved the total
  objective by `86%`, but failed the QI gate (smooth/legacy QI
  `5.18e-2`/`3.79e-2`). The remaining seed-robust QI physics task is therefore
  not script plumbing; it is a better QH/QA/simple-seed homotopy or
  Boozer-target steering policy that reaches the accepted QI basin.
- 2026-05-12: Confirmed that the near-axis QI candidate fails the engineering
  mirror gate on every sampled surface (`0.94-0.97`, target `0.21`). Added
  all-surface/smoothed `MirrorRatio` support, per-surface mirror diagnostics,
  and explicit `QI+iota` versus full-engineering gate reporting in the public
  QI examples. Probe runs showed hard/smooth mirror penalties can reduce mirror
  (`~0.39-0.69`) but currently destroy precise QI (`legacy QI ~5e-3-3e-2`),
  so the open lane is a staged or alternative QI-preserving mirror objective.
- 2026-05-12: Added and tested a differentiable `BoozerBTarget` steering term
  for QI basin homotopy experiments. It can match a reference Boozer `|B|`
  spectrum but is not a final QI diagnostic. The first NFP3 target probe from
  `input.QI_stel_seed_3127` reduced mirror to about `0.405` but degraded
  smooth/legacy QI to about `5.9e-2`/`4.5e-2`, so it was not promoted.
- 2026-05-12: Re-tuned the standalone QI example toward the current best
  mirror-aware NFP2 lane. A branch-heavy smooth-QI residual
  (`branch_width_weight=5`) from `input.nfp2_QI` reached smooth/legacy QI
  about `2.0e-3`/`2.7e-4`, aspect `5.00`, `mean_iota=-0.50`, elongation
  about `7.2`, and all-surface Boozer mirror about `0.30`. Dense stronger
  mirror cleanup was stopped for high memory; a lighter matrix-free cleanup
  preserved QI but only improved mirror from `0.304` to `0.300`.
- 2026-05-13: Added an opt-in weighted branch-shuffle QI residual component.
  It uses monotone linear branch crossings plus differentiable alpha weights
  from squash error, so it is closer to the legacy Goodman branch diagnostic
  than the smooth occupancy crossing estimator. Offline ranking improved for
  stored reference/current candidates, but a bounded NFP2 optimization with
  `weighted_shuffle_profile_weight=1` reached legacy QI `6.4e-4` and mirror
  `0.306`, worse than the previous branch-heavy mirror-aware candidate. Keep
  the term available for homotopy/ranking experiments, not enabled by default.
- 2026-05-13: Added the legacy squash/stretch endpoint correction to the
  opt-in weighted branch-shuffle path. Offline component ranking now favors the
  reference NFP2 QI candidate over the recent high-mirror candidates, although
  the total branch5 ranking still does not justify enabling the term by
  default. A short post-commit NFP2 ESS probe with
  `weighted_shuffle_profile_weight=1` and `max_nfev=6` reached smooth/legacy QI
  `4.9e-3`/`4.0e-3`, mirror `0.324`, aspect `5.03`, and `mean_iota=-0.44` in
  `105 s`, so it remains diagnostic/homotopy-only. Keep using the branch-heavy
  mirror-aware path as the promoted QI example while retaining this term for
  controlled homotopy/ranking studies.
- 2026-05-13: Added required-tier bundled wout profile/current parity gates.
  The new fast tests verify `phipf`/`phi`, finite-beta `pres/presf`, VMEC
  `iotas -> iotaf` smoothing, and surface-averaged Ampere finite differences
  from `buco/bvco` to `jcuru/jcurv` on converged bundled fixtures covering
  axisymmetric finite-beta, 3D current-driven, 3D finite-beta, and `lasym=True`
  cases. Verified focused command:
  `JAX_ENABLE_X64=1 python -m pytest tests/test_wout_profiles_currents_bundled_parity.py -q`
  (`9 passed`).
- 2026-05-13: Added a lightweight bundled solved-state QI diagnostic gate in
  `tests/test_qi_diagnostics.py` using
  `input.QI_stel_seed_3127`/`wout_QI_stel_seed_3127.nc`. The test runs
  `qi_diagnostics_from_state` on a tiny Boozer/QI grid and checks finite
  smooth/raw/legacy QI totals, mirror ratio, elongation excess, surface
  metadata, and Boozer resolution fields without launching a solve. Verified
  focused command: `JAX_ENABLE_X64=1 pytest -q tests/test_qi_diagnostics.py`
  (`4 passed`).
- 2026-05-13: Spawned parallel coverage/differentiation workers and integrated
  fast, solve-free gates for QS residual JVP/VJP, QS/QI objective routing
  JVP/VJP, driver/runtime policy helpers, implicit packing/linear-solve helpers,
  synthetic wout read/write/equif/iota helpers, solver/free-boundary helper
  branches, VMEC-kernel helper branches, additional solve branch coverage,
  and implicit/wout/driver branch coverage. Focused validation:
  `JAX_ENABLE_X64=1 pytest -q tests/test_quasisymmetry.py tests/test_optimization_examples.py tests/test_qi_diagnostics.py tests/test_driver_policy_helpers.py tests/test_solve_runtime.py tests/test_wout_additional_helpers.py tests/test_implicit_helpers.py tests/test_solve_additional_helpers.py tests/test_free_boundary_additional_helpers.py tests/test_vmec_kernel_additional_helpers.py tests/test_solve_branch_coverage.py tests/test_implicit_wout_driver_branch_coverage.py`
  (`137 passed, 1 skipped`). Required coverage now measures 69.17% on the full
  required tier (`730 passed, 21 skipped, 85 deselected`, 8:54).
- 2026-05-13: Refreshed released full-test reference assets locally with
  `python tools/fetch_assets.py --force` and ran the full non-VMEC2000 physics
  coverage tier:
  `RUN_FULL=1 JAX_ENABLE_X64=1 pytest -q -m "full and not vmec2000" --cov=vmec_jax --cov-append --cov-report=xml --cov-report=term-missing:skip-covered`.
  Result: `74 passed, 4 skipped`, 27:21, combined line coverage 72.35%. This is
  useful nightly evidence but too slow and still too low for the requested 80%
  per-commit gate without larger `solve.py`/`wout.py`/`implicit.py` refactors.
- 2026-05-13: Completed a larger parallel coverage push across solve, driver,
  optimization, wout, free-boundary, tomnsp, preconditioner, quasisymmetry,
  parity, and constraint lanes. Added fast helper/branch tests plus a few
  low-risk helper seams and fixed two real free-boundary defects: missing
  unit-current defaulting for raw-free mgrid data and an undefined
  `nu_fourp` local in VMEC nonsingular terms. Validation:
  `ruff check vmec_jax/driver.py vmec_jax/free_boundary.py vmec_jax/optimization.py vmec_jax/solve.py vmec_jax/vmec_tomnsp.py tests`
  passed; changed-lane pytest passed with `477 passed, 1 skipped, 2 deselected`
  in `2:51`; fast docs passed; required coverage passed with
  `896 passed, 21 skipped, 85 deselected`, `81.10%` coverage, and `9:46` using
  the compact terminal report. Raised the py3.11 CI coverage floor to 80%.
  The next testing lane is preserving the 80% gate while adding more
  physics/parity coverage rather than synthetic branch coverage.
- 2026-05-14: Validated the new guarded QI cleanup path from the standalone
  `QI_optimization.py` example on CPU. It completed in `502 s` with monotone
  accepted objective history, aspect `5.0085`, `mean_iota=-0.4974`, legacy QI
  `1.98e-4`, smooth QI `2.04e-3`, max elongation `7.93`, and all-surface mirror
  `0.299`. This is a useful QI-preserving cleanup run but it is not promoted
  over the documented best because it narrowly misses the smooth-QI gate and
  still misses the mirror target. The next QI lane remains a mirror-preserving
  objective/homotopy that lowers mirror below `0.21` without degrading QI.
- 2026-05-14: Ran two bounded post-cleanup QI mirror probes from that final
  input. A moderate mirror-weight probe (`MIRROR_WEIGHT=50`) reduced mirror to
  `0.271` in `97 s` while keeping legacy QI `3.6e-4`, but smooth QI increased
  to `2.27e-3`. A high mirror-weight/QI-ceiling probe (`MIRROR_WEIGHT=1000`)
  reduced mirror further to `0.233` in `245 s`, but smooth QI stayed high
  (`2.51e-3`) and aspect drifted to `5.30`. Conclusion: the current basin has a
  real QI/mirror tradeoff; simply increasing mirror weights is not enough. The
  next robust-QI step should be a staged objective design or alternate seed
  homotopy, not changing the promoted example to these probe settings.
- 2026-05-14: Tried a gradual mirror-threshold homotopy
  (`0.28 -> 0.25 -> 0.23 -> 0.21`) followed by a QI re-tightening pass. The
  staged path reduced mirror from about `0.303` to `0.241`, but smooth QI rose
  from `2.02e-3` to `2.64e-3`; the re-tightening pass pulled smooth QI down to
  `2.21e-3` but mirror relaxed back to `0.266`. Auditing the alternate bundled
  `input.QI_stel_seed_3127` under the current Goodman-style metric gave
  smooth/legacy QI about `5.0e-2`/`4.7e-2`, so it is not currently a shortcut
  seed. Do not spend more time on weight-only mirror schedules from this basin;
  the remaining route is a different QI/mirror objective parameterization,
  seed homotopy, or importing the stronger legacy omnigenity strategy more
  directly.
- 2026-05-17: Post-`v0.0.8` performance pass on `office` (two RTX A4000 GPUs).
  A bounded QH mode-2 exact-Jacobian callback showed GPU solving/tape-build was
  competitive, but full-column GPU tape replay dominated runtime (`~22.2 s`
  replay, `~42.0 s` callback). Testing replay policies showed `whole_scan` was
  worse, while 8-column replay chunking was best among tested chunks. Promoted
  that heuristic as the GPU default for 24+ DOF dense exact Jacobians; the same
  default GPU callback then ran in `~18.0 s` with replay `~5.3 s`, faster than
  the bounded CPU callback measured in the same session.
- 2026-05-17: Re-profiled production fixed-boundary policy on `office`. Raw
  force kernels were no longer the bottleneck; the scan loop was slower than
  the VMEC-control non-scan loop once the solve was required to converge. For
  `input.nfp4_QH_warm_start` with `max_iter=500`, the auto-selected GPU
  non-scan path converged to `~1.11e-13` in `~16.0 s` versus `~20.6 s` with
  explicit GPU scan. A four-case GPU scan/non-scan sweep also favored non-scan:
  QH `33.10 -> 17.85 s`, QA `149.27 -> 69.32 s`, LASYM tokamak
  `164.57 -> 140.69 s`, and QI `108.67 -> 50.78 s`. The public API/CLI auto
  policy and fixed-boundary profiler default now match this non-scan production
  policy; explicit fast/scan modes remain available for experiments.
- 2026-05-17: Tightened the CLI fixed-boundary finish policy for explicit
  low-budget runs. If a caller supplies `max_iter`, all accelerated/parity
  finish attempts combined are capped at `2 * max_iter` and report
  non-convergence rather than silently spending repeated extra full-budget
  blocks. Input-deck-driven production runs without an explicit override keep
  the robust finish behavior. Accelerated finish retries also inherit
  `use_scan=False` from the new production auto policy, so GPU/CPU auto paths
  do not re-enter the slower scan loop during finish attempts. On `office`, the
  explicit `max_iter=100` QH GPU diagnostic now stops after two finish blocks
  (`[100, 100]`) and reports non-convergence at the cap; the same low-budget
  diagnostic previously spent five finish blocks and took about `56.2 s`, while
  the capped source-tree run took about `42.9 s`.
- 2026-05-17: CPU/GPU performance-closure follow-up verified the current
  policies locally on CPU. A `profile_fixed_boundary.py` QH warm-start run with
  explicit `--iters 50` used the public non-scan policy, reported finish
  budgets `[50, 50]` with `cli_fixed_boundary_finish_budget_cap=100`, and
  stopped as non-converged at `1.876 s` instead of spending unbounded finish
  attempts. Raw 50-iteration diagnostics were `0.649 s` for non-scan default
  and `0.414 s` for explicit scan, but both were unconverged, so the production
  non-scan default remains based on converged CPU/GPU measurements. A local QH
  `max_mode=1` exact Jacobian callback took `16.115 s`; measured components
  were `7.428 s` exact solve/tape (`4.130 s` tape build), `3.644 s` replay,
  `2.736 s` initial tangents, and `2.295 s` residual tangents. No broad replay
  rewrite was attempted; the safe code change was to make malformed
  `VMEC_JAX_LASYM_REPLAY_COLUMN_CHUNK` values fall back to the measured
  backend/input auto heuristic instead of aborting exact callbacks.
- 2026-05-17: Added the converged-wout parity matrix and benchmark
  regeneration lane. Required CI now has
  `tests/test_converged_wout_matrix_parity.py`, a no-executable bundled-wout
  gate covering fixed/free, axisymmetric/non-axisymmetric, LASYM, and
  single/multigrid representatives. The opt-in executable end-state gate in
  `tests/test_vmec2000_converged_parity.py` keeps the bounded circular
  comparison under `VMEC2000_INTEGRATION=1` and moves slower representative
  cases behind `VMEC2000_NIGHTLY=1`. The new
  `tools/diagnostics/converged_wout_parity_benchmark.py` discovers
  `~/bin/xvmec2000` plus STELLOPT/PATH executables by default, can opt into
  recursive local executable inventory, de-duplicates symlinks, and writes
  VMEC2000/vmec_jax residual, runtime, scalar, and field rel-RMS summaries for
  benchmark regeneration.
- 2026-05-14: Added an opt-in dense branch-shuffle output grid
  (`shuffle_profile_nphi_out`) to the differentiable QI residual and propagated
  it through diagnostics and `QuasiIsodynamicOptions`. This brings vmec_jax
  closer to the legacy `omnigenity_optimization` `arr_out=True` QI objective
  without changing defaults. Focused tests passed (`60 passed`), and a short
  production smoke run with `shuffle_profile_nphi_out=301` completed in `41 s`
  for `max_nfev=3`. This is infrastructure for the next QI objective-design
  pass, not yet a promoted optimized stellarator.
- 2026-05-17: Prepared the `v0.0.9` patch release update. Scope after
  `v0.0.8`: Node 24 CI action-runtime refresh, fixed-boundary profiler option
  reporting, GPU exact-Jacobian replay chunking, GPU replay/profile
  documentation, CPU/GPU non-scan fixed-boundary production policy, explicit
  finish-budget caps, QI README/docs coverage for the NFP2 and seed-3127 cases,
  expanded bounded VMEC2000 parity gates, and release/docs hygiene.
- 2026-05-19: Released `v0.0.10`, refreshed docs/release notes for the PyPI
  artifact, added converged-wout mode-hotspot and reference-state roundtrip
  diagnostics, exposed scan-trial timing subphases, and changed the VMEC2000
  scan-runner cache to reuse compiled scan loops across perturbed fixed-boundary
  trial points by carrying boundary edge rows dynamically.
- 2026-05-20: Reduced matrix-free exact-optimizer setup cost by deriving the
  initial-state transpose from the existing `jax.linearize` object instead of
  tracing a second initial-state VJP. A cold QA `max_mode=1` CPU linear-operator
  smoke with `inner_max_iter=trial_max_iter=4` completed in `24.0 s`, reported
  `linear_operator_initial_transpose = 0.78 s`, and selected
  `exact_tape_build_solve_call` as the next patch target.
- 2026-05-20: Added lazy JIT compilation for the optimizer's projected
  parameter-to-initial-state map. The QA `max_mode=1` two-point CPU exact
  callback profile improved from `7.99 s` to `6.94 s` with peak RSS dropping
  from about `1593 MiB` to `1456 MiB`; the next patch target moved to
  accepted-solver iteration-loop unattributed compiled work.
