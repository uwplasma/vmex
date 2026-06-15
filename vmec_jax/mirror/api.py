"""Small public surface for experimental mirror-geometry primitives."""

from .core.basis import ChebyshevLobattoBasis, ThetaFourierBasis
from .core.boundary import MirrorBoundary
from .core.config import MirrorConfig, MirrorResolution
from .core.grids import MirrorGrid, make_mirror_grid
from .core.profiles import IPrimeProfile, PressureProfile, PsiPrimeProfile
from .core.state import MirrorStateAxisym
from .io.mout import is_mirror_output, load_mirror_output, read_mirror_output, write_mirror_output
from .io.schema import MirrorOutput
from .plotting.export import mirror_axisym_slice_to_csv, mirror_output_to_npz, plot_mirror_output
from .solvers.fixed_boundary.api import MirrorFixedBoundaryResult, MirrorSolveOptions, run_mirror_fixed_boundary


__all__ = [
    "ChebyshevLobattoBasis",
    "IPrimeProfile",
    "MirrorBoundary",
    "MirrorConfig",
    "MirrorFixedBoundaryResult",
    "MirrorGrid",
    "MirrorOutput",
    "MirrorResolution",
    "MirrorSolveOptions",
    "MirrorStateAxisym",
    "PressureProfile",
    "PsiPrimeProfile",
    "ThetaFourierBasis",
    "is_mirror_output",
    "load_mirror_output",
    "make_mirror_grid",
    "mirror_axisym_slice_to_csv",
    "mirror_output_to_npz",
    "plot_mirror_output",
    "read_mirror_output",
    "run_mirror_fixed_boundary",
    "write_mirror_output",
]
