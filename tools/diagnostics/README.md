# Diagnostics (Developer-Only)

This folder contains parity breakdown scripts, investigation notebooks-as-scripts,
and research utilities used during the VMEC2000 parity push.

These scripts are intentionally not part of the stable, user-facing examples:

- they may rely on optional external installs (VMEC2000, simsopt),
- they may be slow or produce large reports/figures,
- their CLI/API may change without notice.

For user-facing entrypoints, start from:

- `examples/showcase_axisym_input_to_wout.py`
- `vmec_jax.api`

Common parity/validation scripts (moved from `examples/validation/`):

- `pipeline_parity_summary.py`
- `getfsq_parity_cases.py`
- `end_to_end_solve_parity_summary.py`
- `benchmark_fixed_boundary_runtime_and_residuals.py`
- `axisym_stage_parity.py`
- `axisym_first_step_diagnostics.py`
- `parity_sweep_manifest.py` + `parity_manifest.toml` (fixed/free boundary matrix)

Performance profiling:

- `fixed_boundary_performance_decomposition.py --input examples/data/input.nfp4_QH_warm_start`
  writes a cold/warm fixed-boundary profiling report plus a source-level map
  from VMEC2000/VMEC++ algorithm buckets to vmec_jax modules and profiler keys.
  This is the main M1/M2 performance-roadmap entrypoint for separating
  import/backend setup, JAX trace/compile, steady solve, VMEC2000 baseline, and
  process peak RSS. Add `--cprofile` to write Python call-stack profiles for
  the timed cold/warm vmec_jax runs.
- `profile_exact_optimizer.py --callback jacobian --perturb-scale ...`
  measures accepted-point optimizer callback phases for QA/QH/QP fixed-boundary
  quasisymmetry objectives, optimizer/global cache growth, RSS growth, and JSON
  budget status for CPU/GPU production profiling.
  Initial aspect/QS metrics are skipped by default to avoid an unmeasured exact
  solve before the profile; add `--initial-metrics` for that sanity check. Use
  `--vmec-timing-detail` for targeted preconditioner subphase timing (`apply`
  vs mode scaling) when `exact_tape_build` is the bottleneck. Add
  `--sync-replay-timing` only for targeted diagnostics when replay/tangent
  dispatch must be separated from device-ready time.
- `profile_qi_boozer_gpu.py --solver-device gpu --repeat 2`
  isolates the QI/Boozer residual path from the outer optimizer.  Use it before
  launching a full QI sweep on GPU; it reports VMEC solve time, first Boozer/QI
  evaluation time, warm repeated evaluation time, the `--jit-booz` setting,
  active/default JAX backend, active GPU status, and contamination warnings
  such as CPU/GPU platform mixing in one process.
- `gpu_cpu_performance_matrix.py --mode qi-boozer --backend cpu --backend gpu`
  launches the QI/Boozer profiler in separate child processes and writes one
  report per backend plus a compact matrix JSON.  Add `--jit-booz` to compare
  the jitted Boozer path used by the public QI optimization helpers; use
  `--repeat 3` or higher to separate first-call compilation/staging from warm
  residual cost.
- `compare_profile_reports.py cpu.json gpu.json --label cpu --label gpu`
  compares two or more profiler JSON reports without rerunning VMEC.  It emits
  text or JSON ratios for total runtime, compile/replay/cache time when present,
  QI VMEC solve / first-call / warm-call timings when present, accepted-point
  tape build, initial tangent/VJP projection, residual tangent projection,
  contamination warning count, callback count, observed RSS peak, solve count,
  accepted-point replay count, and cache growth.

QI landscape diagnostics:

- `qi_landscape_scan.py --input results/qi_opt/ess/input.final --dofs rc11,zs11`
  scans one or two boundary coefficient increments around an existing QI input
  state, evaluates smooth QI residual, mirror ratio, LCFS elongation, aspect,
  and mean iota, then writes JSON/CSV plus a line or contour-line plot.  The 2D
  view uses `matplotlib.contour` lines rather than filled contours so adjacent
  metric ridges remain visually comparable.  By default executed scans use
  trial solves for speed; add `--exact-solve` before comparing scalar values to
  promotion gates.
- `qi_basin_survey.py --input examples/data/input.QI_stel_seed_3127`
  writes a deterministic large-step basin-survey plan for far-seed QI runs.
  Add `--execute --save-candidate-inputs` to run bounded VMEC/QI diagnostics,
  rank candidates by QI/legacy/mirror/elongation/iota/aspect gates, and emit
  top `input.candidate` files for later differentiable local refinement.  Add
  `--exact-solve` for reviewed acceptance/promotion runs.
- `qi_basin_promote.py --candidates results/diagnostics/qi_basin_survey/top_candidates.json`
  consumes those top candidate inputs and applies bounded local refinement
  policies: direct mode-3, repeated continuation, QI-then-augmented-Lagrangian
  cleanup, and soft-wall cleanup.  By default it only writes a plan; add
  `--execute` after reviewing the candidate matrix.
- `qi_filter_search.py --input results/diagnostics/qi_basin_survey/top_candidate/input.candidate`
  performs a hard-gated feasibility search when scalar penalties jump between
  incompatible QI/iota basins.  It accepts only trials that preserve previous
  gates while improving the current failed gate, in the order QI, iota, then
  mirror/elongation.  The history file is checkpointed after each trial; use
  `--max-trials-per-iteration` and `--verbose` for interactive office/GPU
  debugging.

Free-boundary manifest notes:

- each free-boundary case can define quantitative pass/fail limits via
  `[cases.metric_thresholds_rel_scaled]` (for keys such as `source_sym`,
  `bvec_nonsing_fouri`, `amatrix`, `potvac`);
- optional per-iteration limits are supported via
  `[cases.metric_thresholds_rel_scaled_by_iter."<iter>"]`;
- optional performance gates are supported per case:
  - `max_runtime_s`
  - `max_total_runtime_s`
  - `[cases.runtime_thresholds_s_by_iter."<iter>"]`;
- `parity_sweep_manifest.py` now fails a case when command `rc=0` but metric
  or runtime thresholds are exceeded;
- a local self-contained non-axisymmetric `LASYM=T` free-boundary case is
  included:
  - `examples/data/input.cth_like_free_bdy_lasym_small`
  - `examples/data/mgrid_cth_like_lasym_small.nc`

Late-iteration free-boundary diagnostics:

- VMEC2000 `scalpot` dumps now include cached source channels
  (`source_sym_cached`, `gsource_cached`, `bvecNS_cached`) plus
  `source_cache_iter` metadata.
- The free-boundary comparator uses these cached channels when `fouri` dump
  files are absent (common at `ivacskip > 0`), so `source_sym`/`bvecNS`
  alignment remains observable beyond vacuum turn-on iterations.

Fixed-boundary implicit-AD debugging:

- `VMEC_JAX_IMPLICIT_KEEP_ALL_ACTIVE=1` disables the reduced active-column drop
  in the lasym=`False` implicit solve and uses the full active state instead.
- `VMEC_JAX_IMPLICIT_DISABLE_REDUCED_ACTIVE=1` bypasses the reduced
  stellarator-symmetric adjoint path entirely and falls back to the full-state
  adjoint.
- These flags are diagnostic tools for derivative investigations, not parity or
  performance defaults.
