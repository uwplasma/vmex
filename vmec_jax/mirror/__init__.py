"""Open-field-line magnetic-mirror equilibrium support.

The mirror backend uses ``(s, theta, xi)`` coordinates: a VMEC-like radial
mesh, a periodic poloidal angle, and a nonperiodic axial coordinate.  It is a
separate topology from toroidal VMEC, but shares JAX kernels, coil fields,
solver controls, and diagnostics with :mod:`vmec_jax.core`.

Only implemented, tested contracts are exported here.  Solver, vacuum, and
output APIs will be added as their corresponding M2-M6 plan gates land.
"""

from .basis import ChebyshevBasis, MirrorGrid, ThetaBasis, build_mirror_grid
from .geometry import (
    ContravariantField,
    MirrorGeometry,
    contravariant_field,
    divergence_b,
    evaluate_geometry,
    magnetic_field_squared,
)
from .forces import (
    AnisotropicMirrorEnergy,
    AnisotropicForceResidual,
    IsotropicForceResidual,
    InterfaceResidual,
    MirrorEnergy,
    VariationalResidual,
    anisotropic_fixed_boundary_energy_gradient,
    anisotropic_fixed_boundary_variational_residual,
    anisotropic_force_residual,
    anisotropic_mirror_energy,
    fixed_boundary_energy_gradient,
    fixed_boundary_variational_residual,
    isotropic_force_residual,
    interface_residual,
    mass_profile_from_pressure,
    mirror_energy,
)
from .model import (
    MIRROR_INPUT_SCHEMA,
    MIRROR_OUTPUT_SCHEMA,
    AnisotropyIndicators,
    BiMaxwellianPressureClosure,
    EndCondition,
    MirrorBoundary,
    MirrorConfig,
    MirrorResolution,
    MirrorState,
    IsotropicPressureClosure,
    PressureClosure,
    PressureMoments,
    TabulatedPressureClosure,
    anisotropy_indicators,
    project_fixed_boundary_state,
)
from .solver import (
    MirrorConvergenceError,
    MirrorSolveResult,
    SeparableMirrorPreconditioner,
    solve_anisotropic_fixed_boundary_cli,
    solve_fixed_boundary_cli,
)

__all__ = [
    "MIRROR_INPUT_SCHEMA",
    "MIRROR_OUTPUT_SCHEMA",
    "ChebyshevBasis",
    "ContravariantField",
    "EndCondition",
    "AnisotropyIndicators",
    "AnisotropicMirrorEnergy",
    "AnisotropicForceResidual",
    "BiMaxwellianPressureClosure",
    "IsotropicForceResidual",
    "InterfaceResidual",
    "IsotropicPressureClosure",
    "MirrorBoundary",
    "MirrorConfig",
    "MirrorConvergenceError",
    "MirrorGeometry",
    "MirrorGrid",
    "MirrorEnergy",
    "MirrorResolution",
    "MirrorSolveResult",
    "MirrorState",
    "PressureClosure",
    "PressureMoments",
    "ThetaBasis",
    "TabulatedPressureClosure",
    "VariationalResidual",
    "SeparableMirrorPreconditioner",
    "build_mirror_grid",
    "anisotropy_indicators",
    "anisotropic_fixed_boundary_energy_gradient",
    "anisotropic_fixed_boundary_variational_residual",
    "anisotropic_force_residual",
    "anisotropic_mirror_energy",
    "contravariant_field",
    "divergence_b",
    "evaluate_geometry",
    "fixed_boundary_energy_gradient",
    "fixed_boundary_variational_residual",
    "isotropic_force_residual",
    "interface_residual",
    "magnetic_field_squared",
    "mass_profile_from_pressure",
    "mirror_energy",
    "project_fixed_boundary_state",
    "solve_anisotropic_fixed_boundary_cli",
    "solve_fixed_boundary_cli",
]
