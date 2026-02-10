# Validation scripts

Scripts in this folder compare `vmec_jax` kernels and solver outputs against
bundled VMEC2000 reference `wout_*.nc` files.

Recommended (fast):
- `pipeline_parity_summary.py`: solver-free pipeline snapshot on reference `wout` states.
- `getfsq_parity_cases.py`: scalar residual parity (`fsqr/fsqz/fsql`) on reference `wout` states.
- `end_to_end_solve_parity_summary.py`: short end-to-end solve snapshot and a few key comparisons (`--fast` keeps it under ~1 minute).
- `benchmark_fixed_boundary_runtime_and_residuals.py`: runtime + residual traces for a fixed iteration budget (communication-oriented; use `--disable-jit --no-warmup` for quick runs).

Additional:
- `axisym_stage_parity.py`: axisymmetric stage-by-stage checks to localize the first mismatch.
- `axisym_first_step_diagnostics.py`: first-iteration diagnostics on axisymmetric initial guesses.
- `bsub_parity_figures.py`, `bmag_parity_figures.py`, `bsup_parity_figures.py`: figure generation vs reference `wout`.
Per-iteration VMEC2000 trace parity (needs the executable) lives in `tools/diagnostics/vmec2000_exec_stage_trace_compare.py`.

Most scripts write artifacts under `examples/outputs/`.
