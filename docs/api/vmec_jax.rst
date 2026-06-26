vmec_jax package
================

``vmec_jax`` re-exports a broad convenience surface for interactive use, but
the stable user-facing entrypoints are documented in :doc:`public_api`.

For lower-level work, import the specific submodule you need rather than
depending on the full package re-export surface. The submodule reference below
is the authoritative API documentation for those internal layers.

Submodules
----------

.. toctree::
   :hidden:

   generated/vmec_jax.solvers.free_boundary.validation

.. autosummary::
   :toctree: generated
   :recursive:

   vmec_jax.boundary
   vmec_jax.bootstrap_current
   vmec_jax.config
   vmec_jax.coords
   vmec_jax.diagnostics
   vmec_jax.energy
   vmec_jax.external_fields
   vmec_jax.field
   vmec_jax.fieldlines
   vmec_jax.finite_beta
   vmec_jax.fourier
   vmec_jax.solvers.free_boundary.validation
   vmec_jax.solvers.free_boundary.derivatives
   vmec_jax.solvers.free_boundary.adjoint.branch_local_derivatives
   vmec_jax.solvers.free_boundary.adjoint.controller
   vmec_jax.geom
   vmec_jax.grids
   vmec_jax.implicit
   vmec_jax.init_guess
   vmec_jax.integrals
   vmec_jax.kernels
   vmec_jax.kernels.bcovar
   vmec_jax.kernels.constraints
   vmec_jax.kernels.forces
   vmec_jax.kernels.jacobian
   vmec_jax.kernels.lforbal
   vmec_jax.kernels.numpy_forces
   vmec_jax.kernels.parity
   vmec_jax.kernels.realspace
   vmec_jax.kernels.residue
   vmec_jax.kernels.tomnsp
   vmec_jax.mercier
   vmec_jax.modes
   vmec_jax.namelist
   vmec_jax.optimization
   vmec_jax.quasi_isodynamic.optimization
   vmec_jax.optimization_workflow
   vmec_jax.profiles
   vmec_jax.quasi_isodynamic.diagnostics
   vmec_jax.quasi_isodynamic
   vmec_jax.quasisymmetry
   vmec_jax.radial
   vmec_jax.redl_bootstrap
   vmec_jax.residuals
   vmec_jax.robust_coils
   vmec_jax.solve
   vmec_jax.state
   vmec_jax.static
   vmec_jax.visualization
   vmec_jax.wout
