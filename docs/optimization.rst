Optimization and differentiability
==================================

.. note::

   This page is a short orientation. Full optimization tutorials (QA/QH/QP/QI
   from a circular torus, free-boundary beta scans, single-stage runs) arrive
   with the rewritten ``examples/`` gallery — see :doc:`tutorials`.

Objectives (:mod:`vmec_jax.core.optimize`)
------------------------------------------

Simsopt-style building blocks, all pure functions of a converged core state:

- :class:`~vmec_jax.core.optimize.QuasisymmetryRatioResidual` — the
  Landreman-Paul two-term quasisymmetry ratio residual (QA: ``m=1, n=0``;
  QH: ``m=1, n=-1``; QP: ``m=0, n=1`` in ``nfp`` units);
- scalar targets: :func:`~vmec_jax.core.optimize.aspect_ratio`,
  :func:`~vmec_jax.core.optimize.mean_iota`,
  :func:`~vmec_jax.core.optimize.edge_iota`,
  :func:`~vmec_jax.core.optimize.mirror_ratio`,
  :func:`~vmec_jax.core.optimize.volume`,
  :func:`~vmec_jax.core.optimize.magnetic_well`,
  :func:`~vmec_jax.core.optimize.d_merc`,
  :func:`~vmec_jax.core.optimize.l_grad_b`;
- a Goodman-style quasi-isodynamic residual
  (:func:`~vmec_jax.core.optimize.quasi_isodynamic_residual`);
- :func:`~vmec_jax.core.optimize.least_squares` — a thin
  ``scipy.optimize.least_squares`` driver over boundary Fourier degrees of
  freedom (:func:`~vmec_jax.core.optimize.pack_boundary` /
  :func:`~vmec_jax.core.optimize.unpack_boundary`), taking simsopt-style
  ``(callable, target, weight)`` terms.

Gradients (:mod:`vmec_jax.core.implicit`)
-----------------------------------------

Derivatives through the equilibrium use **implicit differentiation** of the
converged fixed point: the solve is wrapped in ``jax.custom_vjp``; the
forward pass runs the fast (opaque) solver, and the backward pass solves the
adjoint linear system matrix-free with preconditioned GMRES. This replaced
the earlier discrete-adjoint / replay-tape machinery entirely — coarse
multigrid stages are just an initializer and are stop-gradient by
construction. See the *Implicit differentiation* section of
:doc:`algorithms` for the formulation and cost analysis.

Gradient accuracy is validated in CI against central finite differences for
boundary coefficients, profile parameters, and coil degrees of freedom.

.. code-block:: python

   import jax
   from vmec_jax.core import implicit
   from vmec_jax.core.input import VmecInput

   inp = VmecInput.from_file("input.solovev")
   p0 = implicit.params_from_input(inp)

   sol = implicit.run(inp, p0)                        # ImplicitSolution pytree
   grad = jax.grad(lambda p: implicit.run(inp, p).wb)(p0)   # adjoint gradient

Free-boundary decks differentiate the same way: the NESTOR vacuum solve is
inside the residual, so coil parameters (ESSOS coil sets,
:mod:`vmec_jax.core.coils`) get gradients with no special handling.
