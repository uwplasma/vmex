# Tools

This directory contains developer-facing tools, not end-user examples.

- `fetch_assets.py`: downloads optional large validation/reference assets
  (reference netCDF files and wout fixtures) from GitHub release bundles.

- `profile_hotpaths.py`: cold-vs-warm wall-time + peak-RSS profile of the
  production hot paths (fixed-boundary solve and the differentiable
  `value_and_grad` adjoint). Backend-agnostic — the same script produces the
  CPU and GPU numbers with `--device cpu` / `--device gpu`.

Hardware parity across forward solves and boundary gradients is audited by
`benchmarks/device_parity.py`; use `--quick` for its reduced-grid smoke mode.

- `diagnose_input.py`: runs the first VMEC force evaluation without entering
  the nonlinear iteration. Its default shareable report contains only runtime
  information, pass/fail checks, and a diagnostic code; it omits the input
  path and all input-derived values. `--details` exposes parsed controls and
  numerical values for local use with non-confidential decks only:
  `python tools/diagnose_input.py path/to/input.case`.

Tools may write to ignored `outputs/` or a user-selected scratch directory.
They should not write tracked artifacts unless the command is explicitly a
documentation or release-artifact promotion step.
