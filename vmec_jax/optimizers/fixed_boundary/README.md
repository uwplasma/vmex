# Fixed-Boundary Optimizers

This package contains reusable optimization machinery for boundary-shape
optimization with fixed-boundary VMEC solves.

Main entry points:

- `parameterization.py`: Fourier boundary degrees of freedom and ESS scaling.
- `objective_terms.py`: composable least-squares objective term helpers.
- `scipy_least_squares.py`: SciPy least-squares bridge used by examples.
- `scalar_trust.py`, `scalar_lbfgs.py`, and `gauss_newton.py`: alternative
  differentiable local optimization strategies.
- `exact_replay.py`, `replay_policy.py`, and `state_cache.py`: accepted-point
  replay/cache policies used to avoid unnecessary exact solves.
- `qs_residuals.py`: quasisymmetry residual adapters that keep example scripts
  concise. Quasi-isodynamic optimization terms live in
  `vmec_jax/quasi_isodynamic/optimization_terms.py`.

Example scripts should assemble objectives explicitly and call these helpers;
problem-specific parameters should stay in the script, not hidden inside a
large wrapper.
