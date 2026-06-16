"""Compatibility facade for fixed-boundary solver implementations.

The implementation lives in
``vmec_jax.solvers.fixed_boundary.residual.iteration`` so the public
``vmec_jax.solve`` module stays small while existing imports and internal
monkeypatch seams continue to work.
"""

from __future__ import annotations

import sys
import types

from .solvers.fixed_boundary.residual import iteration as _iteration


def _export_iteration_symbols() -> None:
    for name, value in vars(_iteration).items():
        if name.startswith("__") and name.endswith("__"):
            continue
        globals()[name] = value


_export_iteration_symbols()


class _SolveFacadeModule(types.ModuleType):
    """Forward assignments to the implementation module.

    A number of internal tests and downstream debugging workflows monkeypatch
    private ``vmec_jax.solve`` symbols.  The exported solver functions execute
    in the implementation module's global namespace, so assignments on this
    facade must be mirrored there to preserve legacy behavior.
    """

    def __setattr__(self, name, value):
        if not (name.startswith("__") and name.endswith("__")) and hasattr(_iteration, name):
            setattr(_iteration, name, value)
        super().__setattr__(name, value)


sys.modules[__name__].__class__ = _SolveFacadeModule

__all__ = tuple(name for name in vars(_iteration) if not (name.startswith("__") and name.endswith("__")))
