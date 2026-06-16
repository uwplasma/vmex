"""WHAM-inspired coil fixture and vacuum-field helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .coils import AxisymmetricFieldRZ, circular_loop_field_rz, mirror_boundary_from_on_axis_bz

if TYPE_CHECKING:
    from ..core.boundary import MirrorBoundary

MU0 = 4.0e-7 * np.pi


@dataclass(frozen=True)
class WhamCoilFixture:
    """Metadata for the WHAM-inspired two-coil circular-loop pack."""

    source: str
    reference: str
    coil_centers_z_m: tuple[float, ...]
    nz: int
    nr: int
    dz_hf_m: float
    r_in_hf_m: float
    r_out_hf_m: float
    i_coil_a: float
    reference_points_r_m: np.ndarray
    reference_points_z_m: np.ndarray
    reference_br_t: np.ndarray
    reference_bz_t: np.ndarray

    @property
    def num_loops(self) -> int:
        return int(len(self.coil_centers_z_m) * self.nz * self.nr)


@dataclass(frozen=True)
class WhamLoopTable:
    """Expanded circular-loop pack arrays."""

    radius_m: np.ndarray
    z_m: np.ndarray
    current_a: np.ndarray


def default_wham_fixture_path() -> Path:
    """Return the default WHAM fixture path."""
    packaged = Path(__file__).resolve().with_name("data") / "wham_coils.json"
    if packaged.exists():
        return packaged
    return Path(__file__).resolve().parents[3] / "validation" / "mirror" / "wham_coils.json"


def load_wham_fixture(path: str | Path | None = None) -> WhamCoilFixture:
    """Load the WHAM-inspired coil fixture metadata."""
    source = default_wham_fixture_path() if path is None else Path(path)
    data = json.loads(source.read_text())
    reference = data.get("reference_field", {})
    return WhamCoilFixture(
        source=str(data["source"]),
        reference=str(data["reference"]),
        coil_centers_z_m=tuple(float(x) for x in data["coil_centers_z_m"]),
        nz=int(data["nz"]),
        nr=int(data["nr"]),
        dz_hf_m=float(data["dz_HF_m"]),
        r_in_hf_m=float(data["r_in_HF_m"]),
        r_out_hf_m=float(data["r_out_HF_m"]),
        i_coil_a=float(data["I_coil_A"]),
        reference_points_r_m=np.asarray(reference.get("r_m", []), dtype=float),
        reference_points_z_m=np.asarray(reference.get("z_m", []), dtype=float),
        reference_br_t=np.asarray(reference.get("B_r_T", []), dtype=float),
        reference_bz_t=np.asarray(reference.get("B_z_T", []), dtype=float),
    )


def build_wham_loop_table(fixture: WhamCoilFixture | None = None) -> WhamLoopTable:
    """Expand fixture metadata into circular-loop radii, axial centers, and currents."""
    fixture = fixture or load_wham_fixture()
    radii = np.linspace(fixture.r_in_hf_m, fixture.r_out_hf_m, fixture.nr)
    offsets = np.linspace(-0.5 * fixture.dz_hf_m, 0.5 * fixture.dz_hf_m, fixture.nz)
    radius_blocks = []
    z_blocks = []
    for center in fixture.coil_centers_z_m:
        zz, rr = np.meshgrid(center + offsets, radii, indexing="ij")
        radius_blocks.append(rr.ravel())
        z_blocks.append(zz.ravel())
    radius = np.concatenate(radius_blocks)
    z = np.concatenate(z_blocks)
    current = np.full_like(radius, fixture.i_coil_a, dtype=float)
    return WhamLoopTable(radius_m=radius, z_m=z, current_a=current)


def wham_vacuum_field_rz(radius_m, z_m, fixture: WhamCoilFixture | None = None) -> AxisymmetricFieldRZ:
    """Evaluate the WHAM-inspired vacuum field from the circular-loop fixture."""
    table = build_wham_loop_table(fixture)
    r = np.asarray(radius_m, dtype=float)
    z = np.asarray(z_m, dtype=float)
    r, z = np.broadcast_arrays(r, z)
    br = np.zeros_like(r, dtype=float)
    bz = np.zeros_like(r, dtype=float)
    for loop_radius, loop_z, current in zip(table.radius_m, table.z_m, table.current_a):
        field = circular_loop_field_rz(r, z - loop_z, loop_radius_m=loop_radius, current_a=current)
        br += field.br
        bz += field.bz
    return AxisymmetricFieldRZ(br=br, bz=bz, bmag=np.sqrt(br**2 + bz**2))


def wham_reference_field(fixture: WhamCoilFixture | None = None) -> AxisymmetricFieldRZ:
    """Return the stored WHAM reference field table."""
    fixture = fixture or load_wham_fixture()
    bmag = np.sqrt(fixture.reference_br_t**2 + fixture.reference_bz_t**2)
    return AxisymmetricFieldRZ(br=fixture.reference_br_t, bz=fixture.reference_bz_t, bmag=bmag)


def wham_on_axis_mirror_ratio(fixture: WhamCoilFixture | None = None, *, num_points: int = 101) -> float:
    """Return the on-axis vacuum mirror ratio over the coil-center span."""
    fixture = fixture or load_wham_fixture()
    z = np.linspace(min(fixture.coil_centers_z_m), max(fixture.coil_centers_z_m), int(num_points))
    field = wham_vacuum_field_rz(np.zeros_like(z), z, fixture)
    return float(np.max(field.bmag) / np.min(field.bmag))


def mirror_boundary_from_vacuum_flux_tube(
    psi_value: float,
    z_grid,
    fixture: WhamCoilFixture | None = None,
    *,
    radius_floor: float = 1.0e-4,
) -> MirrorBoundary:
    """Build an axisymmetric fixed boundary from the near-axis vacuum flux tube."""
    z = np.asarray(z_grid, dtype=float)
    field = wham_vacuum_field_rz(np.zeros_like(z), z, fixture)
    return mirror_boundary_from_on_axis_bz(psi_value, z, field.bz, radius_floor=radius_floor)
