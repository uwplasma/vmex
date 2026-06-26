# WOUT helpers

This folder implements WOUT netCDF handling and derived WOUT-compatible
diagnostics.

- `netcdf.py` and `schema.py`: file format reading/writing and shape checks.
- `minimal.py` and `state.py`: construction from VMEC state objects.
- `flux.py`, `nyquist.py`, `bsubs.py`, `jxbforce.py`, `mercier.py`: derived
  VMEC channels.
- `diagnostics.py`, `debug.py`, `parity.py`: validation and comparison helpers.

Prefer adding new WOUT fields here instead of expanding the root `wout.py`
facade.  The root facade should remain a stable import surface.
