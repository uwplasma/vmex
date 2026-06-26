# VMEC data input and output

This package is for persisted VMEC data formats and file-backed diagnostics.

- `wout_files/`: WOUT netCDF schema, readers/writers, profile channels, Mercier and
  JXB outputs, parity helpers, and minimal WOUT construction.

Keep driver console output and high-level workflow printing out of this package;
those helpers live in `vmec_jax/drivers/interface.py`.
