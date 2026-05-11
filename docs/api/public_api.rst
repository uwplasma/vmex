Public API
==========

``vmec_jax.api`` is the recommended import surface for most users. It keeps the
common solve / I/O / plotting entrypoints stable while the lower-level modules
continue to evolve.

Typical usage::

   import vmec_jax.api as vj

   fixed = vj.run_fixed_boundary("examples/data/input.circular_tokamak")
   freeb = vj.run_free_boundary("examples/data/input.cth_like_free_bdy_lasym_small")

The optimization examples intentionally use the broader top-level import::

   import vmec_jax as vj

That exposes the explicit workflow objects used in the scripts, including
``FixedBoundaryVMEC``, objective wrappers, ``LeastSquaresProblem``, and
``least_squares_solve``.  Use this path when assembling custom objective
tuples or reproducing the optimization sweeps.

The module exports the following user-facing helpers:

.. automodule:: vmec_jax.api
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
