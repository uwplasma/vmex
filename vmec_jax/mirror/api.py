"""Small public surface for experimental mirror-geometry primitives."""

from .core.basis import ChebyshevLobattoBasis, ThetaFourierBasis
from .core.boundary import MirrorBoundary
from .core.config import MirrorConfig, MirrorResolution
from .core.grids import MirrorGrid, make_mirror_grid
from .core.profiles import IPrimeProfile, PressureProfile, PsiPrimeProfile
from .core.state import MirrorStateAxisym
from .solvers.fixed_boundary.api import MirrorFixedBoundaryResult, MirrorSolveOptions, run_mirror_fixed_boundary


__all__ = [
    "ChebyshevLobattoBasis",
    "IPrimeProfile",
    "MirrorBoundary",
    "MirrorConfig",
    "MirrorFixedBoundaryResult",
    "MirrorGrid",
    "MirrorResolution",
    "MirrorSolveOptions",
    "MirrorStateAxisym",
    "PressureProfile",
    "PsiPrimeProfile",
    "ThetaFourierBasis",
    "make_mirror_grid",
    "run_mirror_fixed_boundary",
]
