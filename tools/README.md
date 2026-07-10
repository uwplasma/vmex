# Tools

This directory contains developer-facing tools, not end-user examples.

- `fetch_assets.py`: downloads optional large validation/reference assets
  (reference netCDF files and wout fixtures) from GitHub release bundles.

Tools may write to ignored `outputs/` or a user-selected scratch directory.
They should not write tracked artifacts unless the command is explicitly a
documentation or release-artifact promotion step.
