# Fixed-boundary solver

Fixed-boundary implementation is split by responsibility:

- `residual/`: VMEC residual iteration, force payloads, scan adapters, startup
  policy, and iteration metrics.
- `scan/`: VMEC2000-style adaptive scan controller and staged convergence.
- `optimization/`: residual and energy minimization algorithms.
- `preconditioning/`: radial and metric preconditioner operators.
- `adjoint/`: fixed-boundary replay and discrete-adjoint helpers.
- `diagnostics/`: axis reset and first-step diagnostics.

`api.py` is the internal solver facade imported by higher-level public modules.
New implementation code should normally go into one of the folders above.
