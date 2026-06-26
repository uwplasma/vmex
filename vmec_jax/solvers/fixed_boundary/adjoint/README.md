# Fixed-Boundary Adjoint Helpers

This package contains fixed-boundary derivative and adjoint helper code.

- `implicit_linear_algebra.py`: active/full-coordinate implicit-adjoint
  linear maps, packing helpers, and dense/chunked validation paths.
- `residual_linear_algebra.py`: residual-adjoint and residual-tangent solve
  routing for the implicit fixed-boundary path.
- `strict_updates.py`, `replay_policy.py`, and `replay_payload.py`: accepted
  update/replay helpers used by exact fixed-boundary validation.

Use this layer when a scalar objective needs validated sensitivities through a
VMEC solve. Keep branch- or optimization-policy decisions outside this package;
the files here should focus on differentiating a well-defined solve path.
