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
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np

from .forces import MU0, staggered_field_strength
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


def boundary_fourier_amplitudes(boundary: MirrorBoundary) -> Any:
    """Return real-signal theta-mode amplitudes along the mirror boundary.

    The result has shape ``(ntheta // 2 + 1, nxi)``. Mode zero is the theta
    mean; positive modes use peak-amplitude normalization. The Nyquist mode on
    an even grid is not doubled.
    """

    radius = jnp.asarray(boundary.radius_scale)
    if radius.ndim != 2:
        raise ValueError("boundary radius must have shape (ntheta, nxi)")
    ntheta = radius.shape[0]
    coefficients = jnp.fft.rfft(radius, axis=0) / float(ntheta)
    scale = jnp.full(coefficients.shape[0], 2.0, dtype=radius.dtype).at[0].set(1.0)
    if ntheta % 2 == 0 and ntheta > 1:
        scale = scale.at[-1].set(1.0)
    return jnp.abs(coefficients) * scale[:, None]


def boundary_fourier_norms(
    boundary: MirrorBoundary,
    grid: "MirrorGrid",
    *,
    central_fraction: float | None = None,
) -> tuple[Any, Any]:
    """Return weighted axial L2 and maximum amplitude of each theta mode.

    Axial norms remain meaningful when a mode vanishes at a symmetry plane,
    where a relative error based on one collocation point is ill-conditioned.
    ``central_fraction`` evaluates the Fourier-CGL interpolant on the fixed
    window ``|xi| <= central_fraction``. This avoids comparing different
    near-cap CGL nodes as axial resolution changes.
    """

    radius = jnp.asarray(boundary.radius_scale)
    if radius.shape[1] != grid.nxi:
        raise ValueError("boundary axial size does not match mirror grid")
    if central_fraction is not None:
        if not 0.0 < central_fraction < 1.0:
            raise ValueError("central_fraction must lie strictly between zero and one")
        nodes, weights = np.polynomial.legendre.leggauss(grid.nxi)
        quadrature_nodes = central_fraction * nodes
        quadrature_radius = grid.axial_basis.interpolate(
            radius, quadrature_nodes, axis=1
        )
        amplitudes = boundary_fourier_amplitudes(
            boundary.__class__(quadrature_radius)
        )
        extrema_nodes = np.linspace(-central_fraction, central_fraction, 257)
        extrema_radius = grid.axial_basis.interpolate(
            radius, extrema_nodes, axis=1
        )
        maximum = jnp.max(
            boundary_fourier_amplitudes(boundary.__class__(extrema_radius)), axis=1
        )
        weights = jnp.asarray(weights)
    else:
        amplitudes = boundary_fourier_amplitudes(boundary)
        weights = jnp.asarray(grid.axial_basis.weights)
        maximum = jnp.max(amplitudes, axis=1)
    l2 = jnp.sqrt(
        jnp.sum(amplitudes**2 * weights[None, :], axis=1) / jnp.sum(weights)
    )
    return l2, maximum


@dataclass(frozen=True)
class AxisymmetricBetaDiagnostics:
    """Scalar checks for one axisymmetric free-boundary beta point."""

    requested_beta: Any
    achieved_reference_beta: Any
    volume_averaged_beta: Any
    local_axis_beta: Any
    center_radius: Any
    center_axis_field: Any
    center_vacuum_side_field: Any
    diamagnetic_field_ratio: Any
    paraxial_field_ratio: Any
    paraxial_relative_error: Any


@dataclass(frozen=True)
class NonaxisymmetricBetaDiagnostics:
    """Global and modal checks for one theta-dependent beta point."""

    requested_beta: Any
    achieved_reference_beta: Any
    volume_averaged_beta: Any
    center_mean_radius: Any
    center_mean_field: Any
    center_boundary_modes: Any
    boundary_mode_l2: Any
    boundary_mode_max: Any
    boundary_mode_core_l2: Any
    boundary_mode_core_max: Any
    plasma_volume: Any
    plasma_energy: Any


def _integration_measure(result: "FreeBoundaryMirrorResult", grid: "MirrorGrid"):
    geometry = result.plasma_energy.geometry
    weights = (
        jnp.asarray(grid.radial_weights)[:, None, None]
        * jnp.asarray(grid.theta_basis.weights)[None, :, None]
        * jnp.asarray(grid.axial_basis.weights)[None, None, :]
    )
    return weights * geometry.sqrt_g


def _volume_average(values, result: "FreeBoundaryMirrorResult", grid: "MirrorGrid"):
    measure = _integration_measure(result, grid)
    return jnp.sum(jnp.asarray(values) * measure) / jnp.sum(measure)


def summarize_axisymmetric_beta_scan(
    results: tuple["FreeBoundaryMirrorResult", ...],
    requested_betas: Any,
    grid: "MirrorGrid",
    *,
    reference_field: float,
) -> tuple[AxisymmetricBetaDiagnostics, ...]:
    """Summarize solved beta points against the beta-zero equilibrium.

    ``achieved_reference_beta`` uses the supplied vacuum reference field,
    while ``local_axis_beta`` uses the finite-beta plasma field. The paraxial
    comparison is ``B/B_vac = sqrt(1-beta)`` and is meaningful for a long,
    approximately cylindrical mirror away from beta one.
    """

    betas = jnp.asarray(requested_betas)
    if betas.ndim != 1 or betas.size != len(results):
        raise ValueError("requested_betas must have one value per result")
    if not results:
        raise ValueError("beta diagnostics require at least one result")
    if grid.ntheta != 1:
        raise ValueError("axisymmetric beta diagnostics require ntheta=1")
    center = int(np.argmin(np.abs(np.asarray(grid.z))))
    baseline_field = jnp.sqrt(results[0].plasma_b_squared[0, 0, center])
    reference_field_squared = float(reference_field) ** 2
    summaries = []
    for requested_beta, result in zip(betas, results, strict=True):
        pressure = result.perpendicular_pressure
        axis_field = jnp.sqrt(result.plasma_b_squared[0, 0, center])
        if hasattr(result.vacuum_field, "lateral_field_xyz"):
            vacuum_xyz = result.vacuum_field.lateral_field_xyz[center]
        else:
            vacuum_xyz = result.vacuum_field.total_xyz[0, 0, center]
        vacuum_side_field = jnp.linalg.norm(vacuum_xyz)
        achieved_beta = (
            2.0 * MU0 * pressure[0, 0, center] / reference_field_squared
        )
        local_beta = 2.0 * MU0 * pressure[0, 0, center] / axis_field**2
        average_pressure = _volume_average(pressure, result, grid)
        average_b_squared = _volume_average(result.plasma_b_squared, result, grid)
        average_beta = 2.0 * MU0 * average_pressure / average_b_squared
        diamagnetic_ratio = axis_field / baseline_field
        paraxial_ratio = jnp.sqrt(jnp.maximum(1.0 - achieved_beta, 0.0))
        summaries.append(
            AxisymmetricBetaDiagnostics(
                requested_beta=requested_beta,
                achieved_reference_beta=achieved_beta,
                volume_averaged_beta=average_beta,
                local_axis_beta=local_beta,
                center_radius=result.boundary.radius_scale[0, center],
                center_axis_field=axis_field,
                center_vacuum_side_field=vacuum_side_field,
                diamagnetic_field_ratio=diamagnetic_ratio,
                paraxial_field_ratio=paraxial_ratio,
                paraxial_relative_error=(diamagnetic_ratio - paraxial_ratio)
                / jnp.maximum(paraxial_ratio, jnp.finfo(axis_field.dtype).tiny),
            )
        )
    return tuple(summaries)


def summarize_nonaxisymmetric_beta_scan(
    results: tuple["FreeBoundaryMirrorResult", ...],
    requested_betas: Any,
    grid: "MirrorGrid",
    *,
    reference_field: float,
) -> tuple[NonaxisymmetricBetaDiagnostics, ...]:
    """Summarize solved 3D beta points with global and Fourier observables."""

    betas = jnp.asarray(requested_betas)
    if betas.ndim != 1 or betas.size != len(results):
        raise ValueError("requested_betas must have one value per result")
    if not results:
        raise ValueError("beta diagnostics require at least one result")
    if grid.ntheta <= 1:
        raise ValueError("nonaxisymmetric beta diagnostics require ntheta > 1")
    center = int(np.argmin(np.abs(np.asarray(grid.z))))
    reference_field_squared = float(reference_field) ** 2
    summaries = []
    for requested_beta, result in zip(betas, results, strict=True):
        pressure = result.perpendicular_pressure
        center_field = jnp.sqrt(result.plasma_b_squared[0, :, center])
        boundary_modes = boundary_fourier_amplitudes(result.boundary)
        mode_l2, mode_max = boundary_fourier_norms(result.boundary, grid)
        core_l2, core_max = boundary_fourier_norms(
            result.boundary, grid, central_fraction=0.75
        )
        measure = _integration_measure(result, grid)
        summaries.append(
            NonaxisymmetricBetaDiagnostics(
                requested_beta=requested_beta,
                achieved_reference_beta=(
                    2.0
                    * MU0
                    * jnp.mean(pressure[0, :, center])
                    / reference_field_squared
                ),
                volume_averaged_beta=(
                    2.0
                    * MU0
                    * _volume_average(pressure, result, grid)
                    / _volume_average(result.plasma_b_squared, result, grid)
                ),
                center_mean_radius=jnp.mean(
                    result.boundary.radius_scale[:, center]
                ),
                center_mean_field=jnp.mean(center_field),
                center_boundary_modes=boundary_modes[:, center],
                boundary_mode_l2=mode_l2,
                boundary_mode_max=mode_max,
                boundary_mode_core_l2=core_l2,
                boundary_mode_core_max=core_max,
                plasma_volume=jnp.sum(measure),
                plasma_energy=result.plasma_energy.total,
            )
        )
    return tuple(summaries)


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


_PLOT_DPI = 110


def _matplotlib():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _as_mout(mout):
    if hasattr(mout, "boundary_radius") and hasattr(mout, "b_xyz"):
        return mout, "mout"
    path = Path(mout)
    stem = path.stem
    return read_mout(path), stem[5:] if stem.startswith("mout_") else stem


def _theta_samples(data, values, theta_dense):
    """Periodically resample a ``(ntheta, nxi)`` mirror table."""

    values = np.asarray(values, dtype=float)
    theta = np.asarray(data.theta, dtype=float)
    if theta.size == 1:
        return np.broadcast_to(values[0], (len(theta_dense), values.shape[1]))
    order = np.argsort(np.mod(theta, 2.0 * np.pi))
    theta = np.mod(theta[order], 2.0 * np.pi)
    table = values[order]
    spacing = 2.0 * np.pi / theta.size
    if np.allclose(np.diff(np.r_[theta, theta[0] + 2.0 * np.pi]), spacing):
        modes = np.fft.fftfreq(theta.size, d=1.0 / theta.size)
        coefficients = np.fft.fft(table, axis=0) / theta.size
        phase = np.exp(
            1j
            * (np.asarray(theta_dense)[:, None] - theta[0])
            * modes[None, :]
        )
        return np.real(phase @ coefficients)
    theta_extended = np.concatenate([theta, [theta[0] + 2.0 * np.pi]])
    table_extended = np.concatenate([table, table[:1]], axis=0)
    return np.stack(
        [
            np.interp(
                np.mod(theta_dense, 2.0 * np.pi),
                theta_extended,
                table_extended[:, iz],
            )
            for iz in range(values.shape[1])
        ],
        axis=1,
    )


def _field_line(data, radial_index: int, theta0: float, z_order):
    """Trace one cap-to-cap line using the saved Cartesian field samples."""

    z = np.asarray(data.z, dtype=float)[z_order]
    theta_nodes = np.asarray(data.theta, dtype=float)
    b_xyz = np.take(
        np.asarray(data.b_xyz, dtype=float)[radial_index], z_order, axis=1
    )
    radius = np.sqrt(float(np.asarray(data.s)[radial_index])) * np.take(
        np.asarray(data.radius_scale)[radial_index], z_order, axis=1
    )
    angles = np.empty(z.size)
    angles[0] = theta0
    periodic_theta = np.r_[np.mod(theta_nodes, 2.0 * np.pi), 2.0 * np.pi]
    for iz in range(z.size - 1):
        angle = angles[iz]
        if theta_nodes.size == 1:
            vector = b_xyz[0, iz]
            local_radius = radius[0, iz]
        else:
            vector = np.asarray(
                [
                    np.interp(
                        np.mod(angle, 2.0 * np.pi),
                        periodic_theta,
                        np.r_[b_xyz[:, iz, component], b_xyz[0, iz, component]],
                    )
                    for component in range(3)
                ]
            )
            local_radius = np.interp(
                np.mod(angle, 2.0 * np.pi),
                periodic_theta,
                np.r_[radius[:, iz], radius[0, iz]],
            )
        b_theta = -np.sin(angle) * vector[0] + np.cos(angle) * vector[1]
        denominator = local_radius * vector[2]
        pitch = 0.0 if abs(denominator) < 1.0e-14 else b_theta / denominator
        angles[iz + 1] = angle + (z[iz + 1] - z[iz]) * pitch
    radius_table = np.take(
        np.asarray(data.radius_scale)[radial_index], z_order, axis=1
    )
    radius_samples = _theta_samples(data, radius_table, angles)
    radius_line = radius_samples[np.arange(z.size), np.arange(z.size)] * np.sqrt(
        float(np.asarray(data.s)[radial_index])
    )
    return z, radius_line * np.cos(angles), radius_line * np.sin(angles)


def plot_mout(
    mout: MoutData | str | Path,
    outdir: str | Path,
    *,
    name: str | None = None,
) -> dict[str, Path]:
    """Render summary, cross-section, ``|B|``, and horizontal 3D plots."""

    plt = _matplotlib()
    data, default_name = _as_mout(mout)
    label = name or default_name
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    z_order = np.argsort(np.asarray(data.z))
    z = np.asarray(data.z)[z_order]
    s = np.asarray(data.s)
    center = int(np.argmin(np.abs(z)))
    boundary = np.take(np.asarray(data.boundary_radius), z_order, axis=1)
    mod_b = np.take(np.asarray(data.mod_b), z_order, axis=2)
    pressure = np.take(np.asarray(data.p_perpendicular), z_order, axis=2)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), constrained_layout=True)
    axes[0, 0].plot(z, np.mean(boundary, axis=0), color="#0072B2", lw=2)
    axes[0, 0].fill_between(
        z,
        np.min(boundary, axis=0),
        np.max(boundary, axis=0),
        color="#0072B2",
        alpha=0.2,
    )
    axes[0, 0].set(
        title="Solved LCFS", xlabel="Axial position z [m]", ylabel="Radius [m]"
    )
    axes[0, 1].plot(
        z, np.mean(mod_b[0], axis=0), label="axis", color="#009E73", lw=2
    )
    axes[0, 1].plot(
        z, np.mean(mod_b[-1], axis=0), label="LCFS", color="#D55E00", lw=2
    )
    axes[0, 1].set(
        title="Magnetic-field strength",
        xlabel="Axial position z [m]",
        ylabel="|B| [T]",
    )
    axes[0, 1].legend()
    axes[1, 0].plot(
        np.sqrt(s),
        np.mean(pressure[:, :, center], axis=1) / 1.0e3,
        "o-",
        color="#CC79A7",
    )
    axes[1, 0].set(
        title="Midplane pressure",
        xlabel="Normalized radius sqrt(s)",
        ylabel="p_perp [kPa]",
    )
    history = np.asarray(data.history)
    if history.size:
        axes[1, 1].semilogy(
            history[:, 0], np.maximum(history[:, -1], 1.0e-18), color="#0072B2"
        )
    axes[1, 1].axhline(
        float(data.ftol), color="0.25", ls="--", lw=1, label="ftol"
    )
    axes[1, 1].set(
        title=f"Convergence ({int(data.iterations)} iterations)",
        xlabel="Residual evaluation",
        ylabel="Maximum normalized residual",
    )
    axes[1, 1].legend()
    for axis in axes.flat:
        axis.grid(alpha=0.22)
    paths["summary"] = outdir / f"{label}_summary.png"
    fig.savefig(paths["summary"], dpi=_PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

    theta_dense = np.linspace(0.0, 2.0 * np.pi, 129)
    indices = np.unique(np.round(np.linspace(0, len(z) - 1, 6)).astype(int))
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.8), constrained_layout=True)
    radial_indices = np.unique(np.round(np.linspace(0, len(s) - 1, 7)).astype(int))
    for axis, iz in zip(axes.flat, indices, strict=False):
        for radial_index in radial_indices:
            table = np.take(
                np.asarray(data.radius_scale)[radial_index], z_order, axis=1
            )
            radius = (
                np.sqrt(s[radial_index])
                * _theta_samples(data, table, theta_dense)[:, iz]
            )
            axis.plot(
                radius * np.cos(theta_dense), radius * np.sin(theta_dense), lw=0.9
            )
        axis.set(
            title=f"z = {z[iz]:.3g} m", xlabel="x [m]", ylabel="y [m]", aspect="equal"
        )
    for axis in axes.flat[len(indices) :]:
        axis.set_visible(False)
    paths["cross_sections"] = outdir / f"{label}_cross_sections.png"
    fig.savefig(paths["cross_sections"], dpi=_PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

    boundary_b = _theta_samples(data, mod_b[-1], theta_dense)
    fig, axis = plt.subplots(figsize=(10.5, 4.2), constrained_layout=True)
    contour = axis.contour(
        z, theta_dense, boundary_b, 18, cmap="viridis", linewidths=0.9
    )
    axis.clabel(contour, inline=True, fontsize=7, fmt="%.3g")
    fig.colorbar(contour, ax=axis, label="LCFS |B| [T]")
    axis.set(
        title="Boundary magnetic-field strength",
        xlabel="Axial position z [m]",
        ylabel="Poloidal angle theta",
    )
    paths["modB"] = outdir / f"{label}_modB.png"
    fig.savefig(paths["modB"], dpi=_PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

    radius_dense = _theta_samples(data, boundary, theta_dense)
    zz, tt = np.meshgrid(z, theta_dense)
    fig = plt.figure(figsize=(11.5, 6.2), constrained_layout=True)
    axis = fig.add_subplot(111, projection="3d")
    norm = plt.Normalize(float(np.min(boundary_b)), float(np.max(boundary_b)))
    surface = axis.plot_surface(
        zz,
        radius_dense * np.cos(tt),
        radius_dense * np.sin(tt),
        facecolors=plt.cm.viridis(norm(boundary_b)),
        linewidth=0,
        alpha=0.8,
    )
    surface.set_rasterized(True)
    for coil in np.asarray(data.coil_xyz):
        closed = np.vstack([coil, coil[0]])
        axis.plot(closed[:, 2], closed[:, 0], closed[:, 1], color="#C44E52", lw=2)
    for theta0 in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False):
        line_z, line_x, line_y = _field_line(data, len(s) - 1, theta0, z_order)
        axis.plot(line_z, 1.01 * line_x, 1.01 * line_y, color="black", lw=3.2)
        axis.plot(
            line_z, 1.01 * line_x, 1.01 * line_y, color="#00BFC4", lw=1.5
        )
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap="viridis"),
        ax=axis,
        shrink=0.72,
        pad=0.05,
        label="LCFS |B| [T]",
    )
    axis.set(
        title="Solved mirror equilibrium",
        xlabel="z [m]",
        ylabel="x [m]",
        zlabel="y [m]",
    )
    axis.set_box_aspect((2.2, 1.0, 1.0))
    axis.view_init(elev=22, azim=-57)
    paths["3d"] = outdir / f"{label}_3d.png"
    fig.savefig(paths["3d"], dpi=_PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return paths


__all__ = [
    "AxisymmetricBetaDiagnostics",
    "FreeBoundaryRestart",
    "MoutData",
    "NonaxisymmetricBetaDiagnostics",
    "boundary_fourier_amplitudes",
    "boundary_fourier_norms",
    "load_free_boundary_restart",
    "mout_from_result",
    "plot_mout",
    "read_mout",
    "save_free_boundary_restart",
    "summarize_axisymmetric_beta_scan",
    "summarize_nonaxisymmetric_beta_scan",
    "write_mout",
]


if TYPE_CHECKING:
    from .basis import MirrorGrid
    from .free_boundary import FreeBoundaryMirrorResult
