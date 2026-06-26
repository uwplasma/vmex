# Driver helpers

`vmec_jax.driver` is the public facade.  This folder contains internal helper
modules used by that facade:

- `interface.py`: bundled example lookup, lightweight load/save wrappers, and
  VMEC-style console banners.
- `policy.py`: user option normalization and run-policy decisions.
- `runtime.py`: backend, device, timing, and runtime setup.
- `solve.py`: calls into fixed/free-boundary solver APIs.
- `output.py`, `results.py`, and `lifecycle.py`: result assembly and finalization.
- `dynamic_scan.py`, `staging.py`, `flux.py`, and `debug.py`: specialized
  workflow helpers.

Do not add generic data-format code here.  Persisted VMEC formats belong under
`vmec_jax/io/`; numerical solver implementation belongs under `vmec_jax/solvers/`.
