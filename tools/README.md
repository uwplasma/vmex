# Tools

This directory contains developer-facing tools, not end-user examples.

- `fetch_assets.py`: downloads optional large validation/reference assets
  (reference netCDF files and wout fixtures) from GitHub release bundles.

- `profile_hotpaths.py`: cold-vs-warm wall-time + peak-RSS profile of the
  production hot paths (fixed-boundary solve and the differentiable
  `value_and_grad` adjoint). Backend-agnostic — the same script produces the
  CPU numbers and can be run on a GPU box (`JAX_PLATFORMS=cuda`).

Tools may write to ignored `outputs/` or a user-selected scratch directory.
They should not write tracked artifacts unless the command is explicitly a
documentation or release-artifact promotion step.
