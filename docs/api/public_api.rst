Public API
==========

``vmec_jax.api`` is the recommended import surface for most users. It keeps the
common solve, I/O, plotting, diagnostic, and example-level optimization
entrypoints stable while lower-level numerical kernels continue to evolve.

Typical usage::

   import vmec_jax.api as vj

   fixed = vj.run_fixed_boundary("examples/data/input.circular_tokamak")
   freeb = vj.run_free_boundary("examples/data/input.cth_like_free_bdy_lasym_small")

The primary optimization examples intentionally use the top-level import, which
mirrors this public surface plus the broader scientific namespace::

   import vmec_jax as vj

Both ``vmec_jax`` and ``vmec_jax.api`` expose the documented workflow objects
used in the scripts, including ``example_paths``/``load_example`` and
``ExampleData`` for bundled cases, ``FixedBoundaryVMEC``, objective wrappers,
``LeastSquaresProblem``, ``least_squares_solve``, QI diagnostics,
``BoozerBTarget``/``boozer_b_target_from_wout`` homotopy helpers, and plotting
helpers. Lower-level solver kernels, force assembly routines, and replay
internals remain submodule-level APIs.

The module exports the following user-facing helpers:

.. automodule:: vmec_jax.api
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
