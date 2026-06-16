"""Validation helpers for mirror geometry."""

from .manufactured import make_mms_case
from .coils import (
    AxisymmetricFieldRZ,
    circular_loop_field_rz,
    circular_loop_on_axis_bz,
    mirror_boundary_from_on_axis_bz,
    mirror_boundary_from_two_coil_flux_tube,
    on_axis_mirror_ratio,
    two_coil_field_rz,
    two_coil_on_axis_bz,
    two_coil_on_axis_mirror_ratio,
)
from .wham import (
    WhamCoilFixture,
    WhamLoopTable,
    build_wham_loop_table,
    load_wham_fixture,
    mirror_boundary_from_vacuum_flux_tube,
    wham_on_axis_mirror_ratio,
    wham_reference_field,
    wham_vacuum_field_rz,
)

__all__ = [
    "AxisymmetricFieldRZ",
    "WhamCoilFixture",
    "WhamLoopTable",
    "build_wham_loop_table",
    "circular_loop_field_rz",
    "circular_loop_on_axis_bz",
    "load_wham_fixture",
    "make_mms_case",
    "mirror_boundary_from_on_axis_bz",
    "mirror_boundary_from_two_coil_flux_tube",
    "mirror_boundary_from_vacuum_flux_tube",
    "on_axis_mirror_ratio",
    "two_coil_field_rz",
    "two_coil_on_axis_bz",
    "two_coil_on_axis_mirror_ratio",
    "wham_on_axis_mirror_ratio",
    "wham_reference_field",
    "wham_vacuum_field_rz",
]
