# VMEC Kernels

This package contains low-level VMEC numerical kernels:

- `tomnsp.py`: VMEC `fixaray` tables and Fourier force transforms.
- `realspace.py`: VMEC-compatible real-space synthesis and analysis.
- `parity.py`: even/odd-m and signed-mode conversion helpers.
- `jacobian.py`: half-mesh Jacobian construction.
- `bcovar.py`: half-mesh metric and covariant/contravariant field kernels.
- `constraints.py`: alias/gcon constraint-force helpers.
- `forces.py`: R/Z/lambda force and residual kernels.
- `residue.py`: VMEC scalar residual norms and force normalization.
- `lforbal.py`: VMEC `lforbal` force-balance correction.
- `numpy_forces.py`: CPU hot-path NumPy patching for force assembly.

These files are intentionally not in the package root: they are internal
equilibrium-kernel implementation details used by the solver, WOUT writer,
parity diagnostics, and differentiability tests. Public user APIs should import
from `vmec_jax`, `vmec_jax.api`, or high-level objective/optimizer packages.

