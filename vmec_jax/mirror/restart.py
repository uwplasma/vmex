"""Compact restart files for free-boundary mirror continuation."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from .model import MirrorBoundary, MirrorState

RESTART_SCHEMA = "vmec_jax.mirror.free_boundary_restart/1"


@dataclass(frozen=True)
class FreeBoundaryRestart:
    """Minimal state needed to hot-start another free-boundary solve."""

    boundary: MirrorBoundary
    plasma_state: MirrorState
    vacuum_potential: Any
    mass_scale: float

    @classmethod
    def from_result(cls, result: Any) -> "FreeBoundaryRestart":
        """Extract restart data from a free-boundary solve result."""

        return cls(
            boundary=result.boundary,
            plasma_state=result.plasma_state,
            vacuum_potential=result.vacuum_potential,
            mass_scale=float(result.mass_scale),
        )


def _finite_array(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite numeric values")
    return array


def save_free_boundary_restart(
    path: str | Path, restart: FreeBoundaryRestart | Any
) -> Path:
    """Atomically write a compressed, data-only free-boundary restart."""

    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")
    if not path.parent.is_dir():
        raise FileNotFoundError(f"restart directory does not exist: {path.parent}")
    if not isinstance(restart, FreeBoundaryRestart):
        restart = FreeBoundaryRestart.from_result(restart)

    arrays = {
        "boundary_radius": _finite_array(
            restart.boundary.radius_scale, name="boundary_radius"
        ),
        "radius_scale": _finite_array(
            restart.plasma_state.radius_scale, name="radius_scale"
        ),
        "lambda_stream": _finite_array(
            restart.plasma_state.lambda_stream, name="lambda_stream"
        ),
        "vacuum_potential": _finite_array(
            restart.vacuum_potential, name="vacuum_potential"
        ),
        "mass_scale": _finite_array(restart.mass_scale, name="mass_scale"),
    }
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".npz", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, schema=RESTART_SCHEMA, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return path


def load_free_boundary_restart(
    path: str | Path, plasma_grid: Any, vacuum_grid: Any
) -> FreeBoundaryRestart:
    """Load restart data after strict schema, shape, and finiteness checks."""

    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        schema = str(np.asarray(data["schema"]).item())
        if schema != RESTART_SCHEMA:
            raise ValueError(f"unsupported mirror restart schema: {schema!r}")
        boundary = _finite_array(data["boundary_radius"], name="boundary_radius")
        radius = _finite_array(data["radius_scale"], name="radius_scale")
        lam = _finite_array(data["lambda_stream"], name="lambda_stream")
        potential = _finite_array(data["vacuum_potential"], name="vacuum_potential")
        mass_scale = float(_finite_array(data["mass_scale"], name="mass_scale"))

    expected_boundary = (plasma_grid.ntheta, plasma_grid.nxi)
    if boundary.shape != expected_boundary:
        raise ValueError(f"restart boundary shape {boundary.shape} != {expected_boundary}")
    if radius.shape != plasma_grid.shape or lam.shape != plasma_grid.shape:
        raise ValueError("restart plasma state does not match the requested grid")
    if potential.shape != vacuum_grid.shape:
        raise ValueError(
            f"restart vacuum shape {potential.shape} != {vacuum_grid.shape}"
        )
    if mass_scale <= 0.0:
        raise ValueError("restart mass_scale must be positive")
    return FreeBoundaryRestart(
        boundary=MirrorBoundary(boundary),
        plasma_state=MirrorState(radius, lam),
        vacuum_potential=potential,
        mass_scale=mass_scale,
    )
