"""Mirror-native outputs and compact restart files.

``mout`` is deliberately separate from VMEC's toroidal ``wout`` schema. It
stores physical-grid arrays so a solved open-ended equilibrium can be plotted
or inspected without reconstructing the solver objects.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, fields
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from .forces import staggered_field_strength
from .geometry import contravariant_field, evaluate_geometry, magnetic_field_xyz
from .model import MIRROR_OUTPUT_SCHEMA, MirrorBoundary, MirrorState

RESTART_SCHEMA = "vmec_jax.mirror.free_boundary_restart/1"


@dataclass(frozen=True)
class MoutData:
    """Data-only representation of one straight-axis mirror equilibrium."""

    s: Any
    theta: Any
    xi: Any
    z: Any
    boundary_radius: Any
    radius_scale: Any
    lambda_stream: Any
    mod_b: Any
    b_xyz: Any
    p_perpendicular: Any
    p_parallel: Any
    history: Any
    coil_xyz: Any
    ftol: float
    iterations: int
    converged: bool
    mass_scale: float
    variational_max: float
    normal_stress_rms: float
    b_normal_rms: float
    staggered_weak_max: float = np.nan
    pointwise_force_rms: float = np.nan
    normalized_divergence_rms: float = np.nan
    closure: str = "unknown"
    message: str = ""
    schema: str = MIRROR_OUTPUT_SCHEMA


def mout_from_result(
    result: Any,
    grid: Any,
    config: Any,
    *,
    axial_flux_derivative: Any,
    current_derivative: Any = 0.0,
    boundary: Any | None = None,
    perpendicular_pressure: Any | None = None,
    parallel_pressure: Any | None = None,
    coil_xyz: Any | None = None,
    closure: str = "unknown",
) -> MoutData:
    """Collect a fixed- or free-boundary result and plotting fields."""

    state = getattr(result, "plasma_state", None)
    free_boundary = state is not None
    if state is None:
        state = getattr(result, "state", None)
    if state is None:
        raise ValueError("mirror result has no solved state")
    solved_boundary = getattr(result, "boundary", boundary)
    if solved_boundary is None:
        raise ValueError("fixed-boundary mirror output requires boundary=")
    geometry = evaluate_geometry(state, grid)
    field = contravariant_field(
        state,
        geometry,
        grid,
        axial_flux_derivative=axial_flux_derivative,
        current_derivative=current_derivative,
    )
    b_xyz = np.asarray(magnetic_field_xyz(field, geometry))
    mod_b = np.asarray(
        staggered_field_strength(
            state,
            grid,
            axial_flux_derivative=axial_flux_derivative,
            current_derivative=current_derivative,
        )
    )
    shape = tuple(state.radius_scale.shape)
    if perpendicular_pressure is None:
        perpendicular_pressure = getattr(result, "perpendicular_pressure", None)
    if perpendicular_pressure is None and hasattr(result.energy, "pressure"):
        perpendicular_pressure = np.broadcast_to(np.asarray(result.energy.pressure)[:, None, None], shape)
    perpendicular = np.full(shape, np.nan) if perpendicular_pressure is None else np.asarray(perpendicular_pressure)
    if parallel_pressure is None and closure == "isotropic":
        parallel_pressure = perpendicular
    parallel = np.full(shape, np.nan) if parallel_pressure is None else np.asarray(parallel_pressure)
    if perpendicular.shape != shape:
        raise ValueError("perpendicular_pressure must match the solved state")
    if parallel.shape != perpendicular.shape:
        raise ValueError("parallel_pressure must match perpendicular_pressure")
    coils = np.empty((0, 0, 3)) if coil_xyz is None else np.asarray(coil_xyz)
    if coils.ndim != 3 or coils.shape[-1] != 3:
        raise ValueError("coil_xyz must have shape (ncoil, npoint, 3)")
    interface = getattr(result, "interface", None)
    force = getattr(result, "plasma_force", getattr(result, "force", None))
    weak_force = getattr(
        result,
        "plasma_staggered_weak_force",
        getattr(result, "staggered_weak_force", None),
    )
    variational = getattr(result, "variational", None)
    variational_max = getattr(result, "variational_max", None)
    if variational_max is None:
        variational_max = variational.maximum
    history = np.asarray(result.history)
    if not free_boundary and history.ndim == 2 and history.shape[1] >= 5:
        history = history[:, [0, 4]]
    return MoutData(
        s=np.asarray(grid.s),
        theta=np.asarray(grid.theta),
        xi=np.asarray(grid.xi),
        z=np.asarray(grid.z),
        boundary_radius=np.asarray(solved_boundary.radius_scale),
        radius_scale=np.asarray(state.radius_scale),
        lambda_stream=np.asarray(state.lambda_stream),
        mod_b=mod_b,
        b_xyz=b_xyz,
        p_perpendicular=perpendicular,
        p_parallel=parallel,
        history=history,
        coil_xyz=coils,
        ftol=float(config.ftol),
        iterations=int(result.iterations),
        converged=bool(result.converged),
        mass_scale=float(getattr(result, "mass_scale", 1.0)),
        variational_max=float(variational_max),
        normal_stress_rms=(float(interface.normal_stress_rms) if interface is not None else np.nan),
        b_normal_rms=(float(interface.vacuum_b_normal_rms) if interface is not None else np.nan),
        staggered_weak_max=(float(weak_force.maximum) if weak_force is not None else np.nan),
        pointwise_force_rms=(float(force.normalized_rms) if force is not None else np.nan),
        normalized_divergence_rms=float(getattr(result, "normalized_divergence_rms", np.nan)),
        closure=str(closure),
        message=str(result.message),
    )


def _finite_shape(data: MoutData) -> tuple[int, int, int]:
    shape = np.asarray(data.radius_scale).shape
    if len(shape) != 3:
        raise ValueError("radius_scale must have shape (ns, ntheta, nxi)")
    ns, ntheta, nxi = shape
    expected = {
        "boundary_radius": (ntheta, nxi),
        "lambda_stream": shape,
        "mod_b": shape,
        "b_xyz": (*shape, 3),
        "p_perpendicular": shape,
        "p_parallel": shape,
        "s": (ns,),
        "theta": (ntheta,),
        "xi": (nxi,),
        "z": (nxi,),
    }
    for name, wanted in expected.items():
        if np.asarray(getattr(data, name)).shape != wanted:
            raise ValueError(f"{name} must have shape {wanted}")
    return ns, ntheta, nxi


def write_mout(path: str | Path, data: MoutData, *, overwrite: bool = True) -> Path:
    """Write a compact mirror-native NetCDF file."""

    import netCDF4

    path = Path(path)
    if path.suffix.lower() != ".nc":
        path = path.with_suffix(".nc")
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ns, ntheta, nxi = _finite_shape(data)
    history = np.asarray(data.history)
    if history.ndim != 2:
        raise ValueError("history must be a two-dimensional table")
    coils = np.asarray(data.coil_xyz)
    with netCDF4.Dataset(path, "w", format="NETCDF4") as dataset:
        dataset.setncattr("schema", data.schema)
        for name in (
            "ftol",
            "iterations",
            "converged",
            "mass_scale",
            "variational_max",
            "normal_stress_rms",
            "b_normal_rms",
            "staggered_weak_max",
            "pointwise_force_rms",
            "normalized_divergence_rms",
            "closure",
            "message",
        ):
            value = getattr(data, name)
            dataset.setncattr(name, int(value) if isinstance(value, (bool, np.bool_)) else value)
        for name, size in (
            ("s", ns),
            ("theta", ntheta),
            ("xi", nxi),
            ("xyz", 3),
            ("history_row", history.shape[0]),
            ("history_column", history.shape[1]),
            ("coil", coils.shape[0]),
            ("coil_point", coils.shape[1]),
        ):
            dataset.createDimension(name, size)
        variables = {
            "s": (("s",), data.s),
            "theta": (("theta",), data.theta),
            "xi": (("xi",), data.xi),
            "z": (("xi",), data.z),
            "boundary_radius": (("theta", "xi"), data.boundary_radius),
            "radius_scale": (("s", "theta", "xi"), data.radius_scale),
            "lambda_stream": (("s", "theta", "xi"), data.lambda_stream),
            "mod_b": (("s", "theta", "xi"), data.mod_b),
            "b_xyz": (("s", "theta", "xi", "xyz"), data.b_xyz),
            "p_perpendicular": (("s", "theta", "xi"), data.p_perpendicular),
            "p_parallel": (("s", "theta", "xi"), data.p_parallel),
            "history": (("history_row", "history_column"), history),
            "coil_xyz": (("coil", "coil_point", "xyz"), coils),
        }
        for name, (dimensions, values) in variables.items():
            variable = dataset.createVariable(name, "f8", dimensions, zlib=True, complevel=4)
            variable[:] = np.asarray(values)
    return path


def read_mout(path: str | Path) -> MoutData:
    """Read a :class:`MoutData` file and validate its schema."""

    import netCDF4

    with netCDF4.Dataset(Path(path)) as dataset:
        schema = str(dataset.getncattr("schema"))
        if schema != MIRROR_OUTPUT_SCHEMA:
            raise ValueError(f"unsupported mirror output schema: {schema}")
        arrays = {
            name: np.asarray(dataset[name][:])
            for name in (
                "s",
                "theta",
                "xi",
                "z",
                "boundary_radius",
                "radius_scale",
                "lambda_stream",
                "mod_b",
                "b_xyz",
                "p_perpendicular",
                "p_parallel",
                "history",
                "coil_xyz",
            )
        }
        attributes = {}
        for field in fields(MoutData):
            if field.name in arrays or field.name == "schema":
                continue
            if field.name in dataset.ncattrs():
                attributes[field.name] = dataset.getncattr(field.name)
            elif field.default is not MISSING:
                attributes[field.name] = field.default
            else:
                raise ValueError(f"mout file is missing required attribute: {field.name}")
    attributes["converged"] = bool(attributes["converged"])
    attributes["iterations"] = int(attributes["iterations"])
    data = MoutData(**arrays, **attributes, schema=schema)
    _finite_shape(data)
    return data


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


def _finite_restart_array(value: Any, *, name: str) -> np.ndarray:
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
        "boundary_radius": _finite_restart_array(
            restart.boundary.radius_scale, name="boundary_radius"
        ),
        "radius_scale": _finite_restart_array(
            restart.plasma_state.radius_scale, name="radius_scale"
        ),
        "lambda_stream": _finite_restart_array(
            restart.plasma_state.lambda_stream, name="lambda_stream"
        ),
        "vacuum_potential": _finite_restart_array(
            restart.vacuum_potential, name="vacuum_potential"
        ),
        "mass_scale": _finite_restart_array(restart.mass_scale, name="mass_scale"),
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
        boundary = _finite_restart_array(
            data["boundary_radius"], name="boundary_radius"
        )
        radius = _finite_restart_array(data["radius_scale"], name="radius_scale")
        lam = _finite_restart_array(data["lambda_stream"], name="lambda_stream")
        potential = _finite_restart_array(
            data["vacuum_potential"], name="vacuum_potential"
        )
        mass_scale = float(
            _finite_restart_array(data["mass_scale"], name="mass_scale")
        )

    expected_boundary = (plasma_grid.ntheta, plasma_grid.nxi)
    if boundary.shape != expected_boundary:
        raise ValueError(
            f"restart boundary shape {boundary.shape} != {expected_boundary}"
        )
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


__all__ = [
    "FreeBoundaryRestart",
    "MoutData",
    "load_free_boundary_restart",
    "mout_from_result",
    "read_mout",
    "save_free_boundary_restart",
    "write_mout",
]
