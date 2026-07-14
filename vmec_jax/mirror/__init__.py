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
    "PressureClosure": (".model", "PressureClosure"),
    "IsotropicPressureClosure": (".model", "IsotropicPressureClosure"),
    "BiMaxwellianPressureClosure": (".model", "BiMaxwellianPressureClosure"),
    "TabulatedPressureClosure": (".model", "TabulatedPressureClosure"),
    # Fixed and free-boundary solves.
    "MirrorSolveResult": (".solver", "MirrorSolveResult"),
    "solve_fixed_boundary_cli": (".solver", "solve_fixed_boundary_cli"),
    "FreeBoundaryMirrorResult": (".free_boundary", "FreeBoundaryMirrorResult"),
    "solve_free_boundary_cli": (".free_boundary", "solve_free_boundary_cli"),
    "build_vacuum_grid": (".vacuum", "build_vacuum_grid"),
    # Continuation.
    "solve_beta_scan_cli": (".continuation", "solve_beta_scan_cli"),
    # Implicit differentiation.
    "fixed_boundary_adjoint": (".implicit", "fixed_boundary_adjoint"),
    "solve_fixed_boundary_implicit": (
        ".implicit",
        "solve_fixed_boundary_implicit",
    ),
    "spline_fixed_boundary_adjoint": (
        ".implicit",
        "spline_fixed_boundary_adjoint",
    ),
    "free_boundary_adjoint": (
        ".free_boundary_implicit",
        "free_boundary_adjoint",
    ),
    # MOUT and plots.
    "MoutData": (".output", "MoutData"),
    "mout_from_result": (".output", "mout_from_result"),
    "read_mout": (".output", "read_mout"),
    "write_mout": (".output", "write_mout"),
    "plot_mout": (".plotting", "plot_mout"),
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
