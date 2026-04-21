Public API
==========

``vmec_jax.api`` is the recommended import surface for most users. It keeps the
common solve / I/O / plotting entrypoints stable while the lower-level modules
continue to evolve.

Typical usage::

   import vmec_jax.api as vj

   fixed = vj.run_fixed_boundary("examples/data/input.circular_tokamak")
   freeb = vj.run_free_boundary("examples/data/input.cth_like_free_bdy_lasym_small")

The module exports the following user-facing helpers:

.. automodule:: vmec_jax.api
   :members:
   :undoc-members:
   :show-inheritance:
