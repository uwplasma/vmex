"""Public API for straight-axis magnetic-mirror equilibria.

The mirror backend uses a nonperiodic axial coordinate and is separate from
toroidal VMEC.  Its public surface is intentionally small: inputs and state,
fixed/free-boundary solves, continuation, implicit derivatives, MOUT I/O, and
plotting.  Numerical kernels remain available from their owning submodules.
"""

from importlib import import_module as _import_module


# Public names are lazy so importing ``vmec_jax.mirror`` does not initialize
# every exterior-vacuum and solver dependency.
_LAZY_ATTRS: dict[str, tuple[str, str | None]] = {
    # Model and configuration.
    "EndCondition": (".model", "EndCondition"),
    "MirrorBoundary": (".model", "MirrorBoundary"),
    "MirrorConfig": (".model", "MirrorConfig"),
    "MirrorResolution": (".model", "MirrorResolution"),
    "MirrorState": (".model", "MirrorState"),
    # Coefficient-native fixed- and free-boundary solves.
    "SplineMirrorBoundary": (".splines", "SplineMirrorBoundary"),
    "SplineMirrorDiscretization": (".splines", "SplineMirrorDiscretization"),
    "SplineMirrorState": (".splines", "SplineMirrorState"),
    "solve_fixed_boundary_cli": (".splines", "solve_fixed_boundary_cli"),
    "solve_free_boundary_cli": (".free_boundary", "solve_free_boundary_cli"),
    # Continuation.
    "solve_beta_scan_cli": (".free_boundary", "solve_beta_scan_cli"),
    # Implicit differentiation.
    "spline_fixed_boundary_adjoint": (
        ".implicit",
        "spline_fixed_boundary_adjoint",
    ),
    "spline_fixed_boundary_tangent": (
        ".implicit",
        "spline_fixed_boundary_tangent",
    ),
    "free_boundary_adjoint": (
        ".implicit",
        "free_boundary_adjoint",
    ),
    # MOUT and plots.
    "mout_from_result": (".output", "mout_from_result"),
    "read_mout": (".output", "read_mout"),
    "write_mout": (".output", "write_mout"),
    "plot_mout": (".output", "plot_mout"),
}

__all__ = sorted(_LAZY_ATTRS)


def __getattr__(name: str):
    entry = _LAZY_ATTRS.get(name)
    if entry is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = entry
    module = _import_module(module_name, __name__)
    value = module if attribute is None else getattr(module, attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_ATTRS))
