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
