# Solver implementations

This package contains numerical equilibrium solvers and low-level solver
machinery.  Public users should normally import from `vmec_jax.driver`,
`vmec_jax.solve`, or `vmec_jax.free_boundary`, not from this package directly.

- `fixed_boundary/`: fixed-boundary VMEC residual, scan, optimization, adjoint,
  preconditioning, and performance helpers.
- `free_boundary/`: free-boundary field coupling, direct-coil/mgrid providers,
  NESTOR-style operators, validation, and branch-local adjoint helpers.

Keep solver state transitions close to the solver domain.  File formats belong
under `vmec_jax/io/`, and user-facing workflow orchestration belongs under
`vmec_jax/drivers/`.
