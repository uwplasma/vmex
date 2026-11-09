"""Public API for open and closed magnetic-mirror equilibria.

Open mirrors use a nonperiodic straight axis; stellarator-mirror hybrids use a
periodic B-spline axis. Its public surface is intentionally small: inputs and
state, solves, continuation, implicit derivatives, MOUT I/O, and plotting.
Numerical kernels remain available from their owning submodules.
"""

from vmex import _lazy_exports

from .._compat import ensure_configured as _ensure_configured

_ensure_configured()


# Public names are lazy so importing ``vmex.mirror`` does not initialize
# every exterior-vacuum and solver dependency.
_LAZY_ATTRS: dict[str, tuple[str, str | None]] = {
    # Model and configuration.
    "MirrorBoundary": (".model", "MirrorBoundary"),
    "MirrorConfig": (".model", "MirrorConfig"),
    "MirrorInput": (".model", "MirrorInput"),
    "MirrorResolution": (".model", "MirrorResolution"),
    "MirrorState": (".model", "MirrorState"),
    # Coefficient-native fixed- and free-boundary solves.
    "SplineMirrorBoundary": (".splines", "SplineMirrorBoundary"),
    "SplineMirrorDiscretization": (".splines", "SplineMirrorDiscretization"),
    "SplineMirrorState": (".splines", "SplineMirrorState"),
    "build_stellarator_mirror_hybrid": (
        ".splines",
        "build_stellarator_mirror_hybrid",
    ),
    "build_qi_mirror_hybrid": (".splines", "build_qi_mirror_hybrid"),
    "sample_closed_polyline": (".splines", "sample_closed_polyline"),
    "splice_straight_legs": (".splines", "splice_straight_legs"),
    "QIMirrorSplice": (".splines", "QIMirrorSplice"),
    "trace_closed_field_line": (".splines", "trace_closed_field_line"),
    "gk_closed_fieldline_geometry": (
        ".turbulence",
        "gk_closed_fieldline_geometry",
    ),
    "solve_fixed_boundary": (".splines", "solve_fixed_boundary"),
    "solve_fixed_boundary_from_radius": (
        ".splines",
        "solve_fixed_boundary_from_radius",
    ),
    "solve_free_boundary": (".free_boundary", "solve_free_boundary"),
    # One-call drivers for a MirrorInput.
    "MirrorSolution": (".free_boundary", "MirrorSolution"),
    "solve_mirror": (".free_boundary", "solve_mirror"),
    "solve_mirror_beta_scan": (".free_boundary", "solve_mirror_beta_scan"),
    "CircularCoils": (".analytic", "CircularCoils"),
    # Continuation.
    "solve_beta_scan": (".free_boundary", "solve_beta_scan"),
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
    "plot_stellarator_mirror_hybrid": (
        ".output",
        "plot_stellarator_mirror_hybrid",
    ),
    # Free-boundary restart I/O and beta-scan summary.
    "FreeBoundaryRestart": (".output", "FreeBoundaryRestart"),
    "save_free_boundary_restart": (".output", "save_free_boundary_restart"),
    "load_free_boundary_restart": (".output", "load_free_boundary_restart"),
    "summarize_axisymmetric_beta_scan": (
        ".output",
        "summarize_axisymmetric_beta_scan",
    ),
    # Reported mirror ratios and mirror lengths (one definition, section 31.4-R3).
    "MirrorRatioDiagnostics": (".metrics", "MirrorRatioDiagnostics"),
    "MirrorWell": (".metrics", "MirrorWell"),
    "mirror_ratio_diagnostics": (".metrics", "mirror_ratio_diagnostics"),
}

__all__ = sorted(_LAZY_ATTRS)


__getattr__, __dir__ = _lazy_exports(__name__, _LAZY_ATTRS, globals())
