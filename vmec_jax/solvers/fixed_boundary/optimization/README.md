# Fixed-Boundary Solver Updates

This package contains the numerical update strategies used inside fixed-boundary
VMEC iterations, such as gradient descent, Gauss-Newton, L-BFGS, and lambda
updates.

These are equilibrium-solver internals, not user-facing stellarator
optimization scripts. User-facing optimization workflows live under
`vmec_jax.optimizers` and `examples/optimization`.
