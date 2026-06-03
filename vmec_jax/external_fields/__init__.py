"""External magnetic-field providers for free-boundary VMEC calculations.

The mgrid backend preserves VMEC2000 compatibility.  Direct coil providers are
JAX-native and intended for differentiable, coil-aware optimization.
"""

from .base import ExternalFieldProviderConfig, ExternalFieldSample, sample_external_field_cylindrical
from .coils_jax import (
    CoilFieldParams,
    apply_stellarator_symmetry_to_currents,
    apply_stellarator_symmetry_to_curves,
    biot_savart_xyz,
    build_coil_field_geometry,
    coil_coil_distance_soft,
    coil_current_norm,
    coil_curvatures,
    coil_lengths,
    coil_plasma_distance_soft,
    compute_gamma_dash,
    compute_gamma_dashdash,
    curvature_penalty,
    fourier_curves_to_gamma,
    length_penalty,
    sample_coil_field_cylindrical,
    sample_coil_field_cylindrical_from_geometry,
    sample_coil_field_cylindrical_from_geometry_jit,
    sample_coil_field_xyz_from_geometry,
)
from .essos_adapter import from_essos_coils
from .mgrid_jax import MGridFieldParams, interpolate_mgrid_bfield_jax, sample_mgrid_field_cylindrical

__all__ = [
    "CoilFieldParams",
    "ExternalFieldProviderConfig",
    "ExternalFieldSample",
    "MGridFieldParams",
    "apply_stellarator_symmetry_to_currents",
    "apply_stellarator_symmetry_to_curves",
    "biot_savart_xyz",
    "build_coil_field_geometry",
    "coil_coil_distance_soft",
    "coil_current_norm",
    "coil_curvatures",
    "coil_lengths",
    "coil_plasma_distance_soft",
    "compute_gamma_dash",
    "compute_gamma_dashdash",
    "curvature_penalty",
    "fourier_curves_to_gamma",
    "from_essos_coils",
    "interpolate_mgrid_bfield_jax",
    "length_penalty",
    "sample_coil_field_cylindrical",
    "sample_coil_field_cylindrical_from_geometry",
    "sample_coil_field_cylindrical_from_geometry_jit",
    "sample_coil_field_xyz_from_geometry",
    "sample_external_field_cylindrical",
    "sample_mgrid_field_cylindrical",
]
