"""Compatibility facade for free-boundary adjoint controller primitives."""

from __future__ import annotations

from vmec_jax.solvers.free_boundary.adjoint import controller as _controller

__all__ = list(_controller.__all__)

for _name in __all__:
    globals()[_name] = getattr(_controller, _name)

del _controller, _name
