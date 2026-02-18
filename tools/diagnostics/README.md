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
