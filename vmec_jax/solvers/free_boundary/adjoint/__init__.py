"""Branch-local free-boundary adjoint support helpers.

The root module :mod:`vmec_jax.free_boundary_adjoint` is a compatibility shim.
The public validation/report implementation lives in
:mod:`vmec_jax.solvers.free_boundary.adjoint.facade`, while this package also
holds the lower-level trace, replay, runtime, objective, and pytree utilities
used by that facade.
"""
