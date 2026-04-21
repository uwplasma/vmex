vmec_jax package
================

``vmec_jax`` re-exports a broad convenience surface for interactive use, but
the stable user-facing entrypoints are documented in :doc:`public_api`.

For lower-level work, import the specific submodule you need rather than
depending on the full package re-export surface. The submodule reference below
is the authoritative API documentation for those internal layers.

Submodules
----------

.. autosummary::
   :toctree: generated
   :recursive:

   api
   boundary
   config
   coords
   diagnostics
   energy
   field
   fieldlines
   fourier
   geom
   grids
   implicit
   init_guess
   integrals
   modes
   namelist
   optimization
   profiles
   radial
   residuals
   solve
   state
   static
   visualization
   vmec_constraints
   vmec_bcovar
   vmec_forces
   vmec_jacobian
   vmec_parity
   vmec_residue
   vmec_tomnsp
   wout
