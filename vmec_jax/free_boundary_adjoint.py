"""Compatibility shim for free-boundary adjoint utilities.

The implementation lives in :mod:`vmec_jax.solvers.free_boundary.adjoint.facade`
so the root package stays small while existing imports keep working.
"""

from __future__ import annotations

from vmec_jax.solvers.free_boundary.adjoint import facade as _facade

for _name in dir(_facade):
    if _name.startswith("__") and _name != "__all__":
        continue
    globals()[_name] = getattr(_facade, _name)

__all__ = _facade.__all__
_facade.install_compat_facade_module(__name__)

del _facade, _name
