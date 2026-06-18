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
- ``axisym_reduced_residual_linear_solve_jax`` solves dense tiny-grid systems
  ``(dF/dx) dx = rhs`` or ``(dF/dx)^T adjoint = rhs`` for validation.

These functions are intended as method gates for implicit differentiation:

.. code-block:: text

   dF/dx * dx/dp = -dF/dp
   (dF/dx)^T * adjoint = dL/dx

Validation Status
-----------------

The validation example ``examples/mirror_implicit_sensitivity.py`` manufactures
an exact tiny-grid reduced root using a linear reduced source and a small state
ridge. It then compares the dense implicit sensitivity against a finite
difference of an independently solved perturbed source problem.

This validates the residual, Jacobian, and dense linear-solve machinery. It is
not yet a production differentiable equilibrium solve.

Next Steps
----------

1. Keep dense solves as the correctness reference on tiny grids.
2. Add a matrix-free or lineax-backed linear solve with the same forward and
   transpose semantics.
3. Validate the scalable solve against the dense reference.
4. Wrap a small converged solved state with a custom implicit derivative rule.
5. Promote the differentiable API only after it agrees with finite differences
   and the existing fixed-boundary solver diagnostics.
