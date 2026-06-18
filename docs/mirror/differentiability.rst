Mirror Differentiability
========================

The mirror differentiability lane is separate from the fast CLI/example lane.
CLI examples may use NumPy, SciPy, and Matplotlib to keep runtime and memory
small. Differentiable research APIs should keep the residual, linearization,
and derivative rules in JAX and should not differentiate through long
host-side optimizer loops.

Current API
-----------

The current promoted building blocks are axisymmetric and reduced-coordinate
only:

- ``axisym_reduced_residual_jax`` returns the reduced fixed-boundary residual
  ``F(x, p)`` as the JAX gradient of the reduced mirror energy.
- ``axisym_reduced_residual_jacobian_jax`` returns ``dF/dx`` using either the
  energy Hessian, ``jax.jacfwd``, or ``jax.jacrev``.
- ``axisym_reduced_residual_matvec_jax`` applies the reduced energy Hessian to
  a vector without forming the dense matrix.
- ``axisym_reduced_residual_linear_solve_jax`` solves tiny-grid dense systems
  or ridge-stabilized matrix-free ``jax.scipy.sparse.linalg.cg`` systems with
  the same forward/transpose call shape.
- ``axisym_reduced_implicit_state_sensitivity_jax`` applies the forward
  implicit equation ``F_x dx/dp = -F_p``.
- ``axisym_reduced_implicit_adjoint_jax`` solves the adjoint equation
  ``F_x.T adjoint = dL/dx``.
- ``axisym_reduced_implicit_source_state_jax`` returns a cached converged
  reduced state with a custom reverse-mode derivative with respect to a linear
  reduced source. The primal call does not run an optimizer; the VJP uses the
  same implicit adjoint solve as ``axisym_reduced_implicit_adjoint_jax``.
- ``axisym_reduced_residual_pressure_jacobian_jax`` returns the residual
  derivative with respect to pressure polynomial coefficients.
- ``axisym_reduced_implicit_pressure_sensitivity_jax`` applies the forward
  implicit equation for those pressure coefficients.
- ``axisym_reduced_implicit_pressure_state_jax`` returns a cached converged
  reduced state with a custom reverse-mode derivative with respect to pressure
  coefficients.

These functions are intended as method gates for implicit differentiation:

.. code-block:: text

   dF/dx * dx/dp = -dF/dp
   (dF/dx)^T * adjoint = dL/dx

Validation Status
-----------------

The validation example ``examples/mirror_implicit_sensitivity.py`` manufactures
an exact tiny-grid reduced root using a linear reduced source and a small state
ridge. It then compares the implicit sensitivity against a finite difference of
an independently solved perturbed source problem.

This validates the residual, Jacobian, dense linear-solve machinery, first
matrix-free Hessian-vector path, and explicit forward/adjoint implicit wrappers.
The test suite also validates the custom reverse-mode source wrapper against
the explicit adjoint and a separately solved perturbed root, then applies the
forward wrapper to a tiny converged fixed-boundary cylinder with a local state
ridge about the solved state. A second gate validates pressure-coefficient
Jacobians, forward sensitivities, and custom VJPs against a separately solved
perturbed-pressure root. These are the first differentiable solved-state
contracts, but only for reduced axisymmetric source and pressure-coefficient
perturbations.

The benchmark example ``examples/mirror_implicit_solve_benchmark.py`` compares
dense and matrix-free wrapper calls on a small ``ns``/``nxi`` ladder and writes
JSON, CSV, and optional runtime/memory/error plots under ``results/``.

Next Steps
----------

1. Keep dense solves as the correctness reference on tiny grids.
2. Benchmark the matrix-free CG path on larger reduced grids and compare it
   with a lineax-backed operator if that dependency becomes part of the public
   solver stack.
3. Extend the benchmark ladder after the physical residual conditioning is
   improved enough for larger low-ridge systems.
4. Extend the custom implicit wrapper from pressure coefficients to boundary
   coefficients and current/flux profile coefficients.
5. Promote the differentiable API only after those additional
   physical-parameter derivatives agree with finite differences and the
   existing fixed-boundary solver diagnostics.
