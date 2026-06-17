"""Read and write mirror-native ``mout_*.nc`` files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..kernels.energy import total_energy_3d, total_energy_axisym
from ..kernels.fields import evaluate_axisym_field, evaluate_field_3d
from ..kernels.forces import axisym_projected_energy_residual, projected_energy_residual_3d
from ..kernels.geometry import evaluate_axisym_geometry, evaluate_geometry_3d
from ..kernels.residuals import field_diagnostics
from .schema import (
    MOUT_COORDINATE_DIMS,
    MOUT_FIELD_DIMS,
    MOUT_GEOMETRY_DIMS,
    MOUT_GLOBAL_ATTRIBUTES,
    MOUT_HISTORY_DIMS,
    MOUT_PROFILE_DIMS,
    MirrorOutput,
    MirrorOutputDiagnostics,
    MirrorOutputField,
    MirrorOutputGeometry,
    MirrorOutputHistory,
    MirrorOutputProfiles,
)

_GEOMETRY_ATTRS = {"X": "x", "Y": "y", "Z": "z"}
_FIELD_ATTRS = {
    "B_sup_s": "b_sup_s",
    "B_sup_theta": "b_sup_theta",
    "B_sup_xi": "b_sup_xi",
    "B_cov_s": "b_cov_s",
    "B_cov_theta": "b_cov_theta",
    "B_cov_xi": "b_cov_xi",
    "B_x": "b_x",
    "B_y": "b_y",
    "B_z": "b_z",
    "Bmag": "bmag",
    "lambda": "lam",
}
_PROFILE_ATTRS = {"Psi_prime": "psi_prime", "I_prime": "i_prime"}
_HISTORY_ATTRS = {"min_Bmag": "min_bmag", "max_Bmag": "max_bmag"}
_INT_HISTORY_NAMES = {
    "solve_history_stage_index",
    "solve_history_iteration",
    "solve_history_active_force_dof",
    "solve_history_accepted",
}


def _as_axisym_3d(values, *, ntheta: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 2:
        return np.broadcast_to(arr[:, None, :], (arr.shape[0], int(ntheta), arr.shape[1])).copy()
    if arr.ndim == 3:
        return arr
    raise ValueError(f"expected 2-D or 3-D mirror array, got shape {arr.shape}")


def _write_float_variable(ds: Any, name: str, dims: tuple[str, ...], data) -> None:
    var = ds.createVariable(name, "f8", dims)
    var[:] = np.asarray(data, dtype=np.float64)


def _write_int_variable(ds: Any, name: str, dims: tuple[str, ...], data) -> None:
    var = ds.createVariable(name, "i4", dims)
    var[:] = np.asarray(data, dtype=np.int32)


def _read_float(variables: Any, name: str) -> np.ndarray:
    return np.asarray(variables[name][:], dtype=float)


def _read_optional_float(variables: Any, name: str, fallback: str) -> np.ndarray:
    return _read_float(variables, name if name in variables else fallback)


def _read_int(variables: Any, name: str) -> np.ndarray:
    return np.asarray(variables[name][:], dtype=int)


def _read_optional_int(variables: Any, name: str, fallback_value: int, fallback_length: int) -> np.ndarray:
    if name in variables:
        return _read_int(variables, name)
    return np.full(int(fallback_length), int(fallback_value), dtype=int)


def _history_array(trace, name: str, *, dtype=float) -> np.ndarray:
    return np.asarray([getattr(row, name) for row in trace], dtype=dtype)


def _attrs_with_result_metadata(result) -> dict[str, str]:
    attrs = dict(MOUT_GLOBAL_ATTRIBUTES)
    attrs["solver_optimizer"] = str(result.options.optimizer)
    attrs["solver_reduced_coordinate_scaling"] = str(result.options.reduced_coordinate_scaling)
    attrs["solver_residual_linear_maxiter"] = str(int(result.options.residual_linear_maxiter))
    attrs["solver_residual_linear_maxiter_policy"] = str(result.options.residual_linear_maxiter_policy)
    attrs["solver_residual_linear_adaptive_factor"] = str(float(result.options.residual_linear_adaptive_factor))
    attrs["solver_residual_linear_solver"] = str(result.options.residual_linear_solver)
    attrs["solver_residual_preconditioner"] = str(result.options.residual_preconditioner)
    attrs["solver_residual_radial_alpha"] = str(float(result.options.residual_radial_alpha))
    attrs["solver_residual_lambda_alpha"] = str(float(result.options.residual_lambda_alpha))
    attrs["solver_residual_xi_alpha"] = str(float(result.options.residual_xi_alpha))
    attrs["solver_maxiter"] = str(int(result.options.maxiter))
    if result.optimizer_summaries:
        summary = result.optimizer_summaries[-1]
        if summary.residual_linear_maxiter_effective_max is not None:
            attrs["solver_residual_linear_maxiter_effective_max"] = str(
                int(summary.residual_linear_maxiter_effective_max)
            )
        if summary.residual_linear_maxiter_effective_last is not None:
            attrs["solver_residual_linear_maxiter_effective_last"] = str(
                int(summary.residual_linear_maxiter_effective_last)
            )
    attrs["pressure_continuation"] = ",".join(str(float(stage)) for stage in result.options.pressure_continuation)
    return attrs


def _geometry_output(result, geometry) -> MirrorOutputGeometry:
    grid = result.grid
    r = _as_axisym_3d(geometry.r, ntheta=grid.ntheta)
    theta = np.asarray(grid.theta, dtype=float)
    z = np.broadcast_to(np.asarray(grid.z, dtype=float)[None, None, :], r.shape).copy()
    boundary_r = result.boundary.radius_on_grid_3d(grid)
    return MirrorOutputGeometry(
        r=r,
        x=r * np.cos(theta)[None, :, None],
        y=r * np.sin(theta)[None, :, None],
        z=z,
        sqrtg=_as_axisym_3d(geometry.sqrtg, ntheta=grid.ntheta),
        g_ss=_as_axisym_3d(geometry.g_ss, ntheta=grid.ntheta),
        g_stheta=_as_axisym_3d(geometry.g_stheta, ntheta=grid.ntheta),
        g_sxi=_as_axisym_3d(geometry.g_sxi, ntheta=grid.ntheta),
        g_thetatheta=_as_axisym_3d(geometry.g_thetatheta, ntheta=grid.ntheta),
        g_thetaxi=_as_axisym_3d(geometry.g_thetaxi, ntheta=grid.ntheta),
        g_xixi=_as_axisym_3d(geometry.g_xixi, ntheta=grid.ntheta),
        boundary_r=boundary_r,
    )


def _field_output(result, field) -> MirrorOutputField:
    ntheta = result.grid.ntheta
    return MirrorOutputField(
        b_sup_s=_as_axisym_3d(field.b_sup_s, ntheta=ntheta),
        b_sup_theta=_as_axisym_3d(field.b_sup_theta, ntheta=ntheta),
        b_sup_xi=_as_axisym_3d(field.b_sup_xi, ntheta=ntheta),
        b_cov_s=_as_axisym_3d(field.b_cov_s, ntheta=ntheta),
        b_cov_theta=_as_axisym_3d(field.b_cov_theta, ntheta=ntheta),
        b_cov_xi=_as_axisym_3d(field.b_cov_xi, ntheta=ntheta),
        b_x=np.asarray(field.b_x, dtype=float),
        b_y=np.asarray(field.b_y, dtype=float),
        b_z=_as_axisym_3d(field.b_z, ntheta=ntheta),
        bmag=_as_axisym_3d(field.bmag, ntheta=ntheta),
        lam=_as_axisym_3d(result.state.lam, ntheta=ntheta),
    )


def _profile_output(result, field) -> MirrorOutputProfiles:
    grid = result.grid
    psi_prime = result.psi_prime.evaluate(grid.s_full, dtype=float)
    i_prime = result.i_prime.evaluate(grid.s_full, dtype=float)
    pressure = result.pressure.evaluate(grid.s_full, dtype=float)
    dpressure_ds = result.pressure.derivative(grid.s_full, dtype=float)
    b2 = np.asarray(field.b2, dtype=float)
    if b2.ndim == 2:
        b2_s_average = np.einsum("k,ik->i", grid.w_xi, b2) / np.sum(grid.w_xi)
    else:
        weights = grid.w_theta[:, None] * grid.w_xi[None, :]
        b2_s_average = np.einsum("jk,ijk->i", weights, b2) / np.sum(weights)
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = np.divide(
            2.0 * result.options.mu0 * pressure,
            b2_s_average,
            out=np.zeros_like(pressure),
            where=b2_s_average > 0.0,
        )
    return MirrorOutputProfiles(
        psi_prime=psi_prime,
        i_prime=i_prime,
        pressure=pressure,
        dpressure_ds=dpressure_ds,
        beta=beta,
        gamma=float(result.pressure.gamma),
    )


def _diagnostics_output(energy, residual, geometry, field_diag) -> MirrorOutputDiagnostics:
    return MirrorOutputDiagnostics(
        energy_b=energy.magnetic,
        energy_p=energy.pressure,
        energy_total=energy.total,
        residual_norm=residual.norm,
        force_norm=residual.norm,
        fsq=residual.fsq,
        normalized_force=residual.normalized_force,
        active_force_dof=residual.active_dof,
        min_sqrtg=float(np.min(geometry.sqrtg)),
        max_sqrtg=float(np.max(geometry.sqrtg)),
        min_bmag=field_diag.min_bmag,
        max_bmag=field_diag.max_bmag,
        mirror_ratio=field_diag.mirror_ratio,
    )


def _history_output(trace) -> MirrorOutputHistory:
    trace = tuple(trace)
    return MirrorOutputHistory(
        stage_index=_history_array(trace, "stage_index", dtype=int),
        iteration=_history_array(trace, "iteration", dtype=int),
        pressure_scale=_history_array(trace, "pressure_scale"),
        energy_total=_history_array(trace, "energy_total"),
        residual_norm=_history_array(trace, "residual_norm"),
        fsq=_history_array(trace, "fsq"),
        normalized_force=_history_array(trace, "normalized_force"),
        active_force_dof=_history_array(trace, "active_force_dof", dtype=int),
        min_sqrtg=_history_array(trace, "min_sqrtg"),
        max_sqrtg=_history_array(trace, "max_sqrtg"),
        min_bmag=_history_array(trace, "min_bmag"),
        max_bmag=_history_array(trace, "max_bmag"),
        mirror_ratio=_history_array(trace, "mirror_ratio"),
        step_size=_history_array(trace, "step_size"),
        accepted=_history_array(trace, "accepted", dtype=int),
    )


def _write_dimensions(ds: Any, output: MirrorOutput) -> None:
    ds.createDimension("ns", output.ns)
    ds.createDimension("ntheta", output.ntheta)
    ds.createDimension("nxi", output.nxi)
    ds.createDimension("history_steps", output.history.iteration.size)


def _write_output_groups(ds: Any, output: MirrorOutput) -> None:
    for name, dims in MOUT_COORDINATE_DIMS.items():
        _write_float_variable(ds, name, dims, getattr(output, name))
    for name, dims in MOUT_GEOMETRY_DIMS.items():
        _write_float_variable(ds, name, dims, getattr(output.geometry, _GEOMETRY_ATTRS.get(name, name)))
    for name, dims in MOUT_FIELD_DIMS.items():
        _write_float_variable(ds, name, dims, getattr(output.field, _FIELD_ATTRS[name]))
    for name, dims in MOUT_PROFILE_DIMS.items():
        _write_float_variable(ds, name, dims, getattr(output.profiles, _PROFILE_ATTRS.get(name, name)))
    _write_float_variable(ds, "gamma", (), output.profiles.gamma)


def _write_diagnostic_scalars(ds: Any, output: MirrorOutput) -> None:
    scalar_values = {
        "energy_B": output.diagnostics.energy_b,
        "energy_p": output.diagnostics.energy_p,
        "energy_total": output.diagnostics.energy_total,
        "residual_norm": output.diagnostics.residual_norm,
        "force_norm": output.diagnostics.force_norm,
        "fsq": output.diagnostics.fsq,
        "normalized_force": output.diagnostics.normalized_force,
        "min_sqrtg": output.diagnostics.min_sqrtg,
        "max_sqrtg": output.diagnostics.max_sqrtg,
        "min_Bmag": output.diagnostics.min_bmag,
        "max_Bmag": output.diagnostics.max_bmag,
        "mirror_ratio": output.diagnostics.mirror_ratio,
    }
    for name, value in scalar_values.items():
        _write_float_variable(ds, name, (), value)
    _write_int_variable(ds, "active_force_dof", (), output.diagnostics.active_force_dof)


def _write_history(ds: Any, output: MirrorOutput) -> None:
    for name, dims in MOUT_HISTORY_DIMS.items():
        attr = name.removeprefix("solve_history_")
        attr = _HISTORY_ATTRS.get(attr, attr)
        writer = _write_int_variable if name in _INT_HISTORY_NAMES else _write_float_variable
        writer(ds, name, dims, getattr(output.history, attr))


def _read_scalar(variables: Any, name: str) -> float:
    return float(np.asarray(variables[name][:]).reshape(()))


def _read_optional_scalar(variables: Any, name: str, fallback: str) -> float:
    return _read_scalar(variables, name if name in variables else fallback)


def _read_optional_int_scalar(variables: Any, name: str, fallback_value: int) -> int:
    if name in variables:
        return int(np.asarray(variables[name][:]).reshape(()))
    return int(fallback_value)


def _read_geometry(variables: Any) -> MirrorOutputGeometry:
    return MirrorOutputGeometry(
        r=_read_float(variables, "r"),
        x=_read_float(variables, "X"),
        y=_read_float(variables, "Y"),
        z=_read_float(variables, "Z"),
        sqrtg=_read_float(variables, "sqrtg"),
        g_ss=_read_float(variables, "g_ss"),
        g_stheta=_read_float(variables, "g_stheta"),
        g_sxi=_read_float(variables, "g_sxi"),
        g_thetatheta=_read_float(variables, "g_thetatheta"),
        g_thetaxi=_read_float(variables, "g_thetaxi"),
        g_xixi=_read_float(variables, "g_xixi"),
        boundary_r=_read_float(variables, "boundary_r"),
    )


def _read_field(variables: Any) -> MirrorOutputField:
    return MirrorOutputField(
        b_sup_s=_read_float(variables, "B_sup_s"),
        b_sup_theta=_read_float(variables, "B_sup_theta"),
        b_sup_xi=_read_float(variables, "B_sup_xi"),
        b_cov_s=_read_float(variables, "B_cov_s"),
        b_cov_theta=_read_float(variables, "B_cov_theta"),
        b_cov_xi=_read_float(variables, "B_cov_xi"),
        b_x=_read_float(variables, "B_x"),
        b_y=_read_float(variables, "B_y"),
        b_z=_read_float(variables, "B_z"),
        bmag=_read_float(variables, "Bmag"),
        lam=_read_float(variables, "lambda"),
    )


def _read_profiles(variables: Any) -> MirrorOutputProfiles:
    return MirrorOutputProfiles(
        psi_prime=_read_float(variables, "Psi_prime"),
        i_prime=_read_float(variables, "I_prime"),
        pressure=_read_float(variables, "pressure"),
        dpressure_ds=_read_float(variables, "dpressure_ds"),
        beta=_read_float(variables, "beta"),
        gamma=_read_scalar(variables, "gamma"),
    )


def _read_diagnostics(variables: Any) -> MirrorOutputDiagnostics:
    return MirrorOutputDiagnostics(
        energy_b=_read_scalar(variables, "energy_B"),
        energy_p=_read_scalar(variables, "energy_p"),
        energy_total=_read_scalar(variables, "energy_total"),
        residual_norm=_read_scalar(variables, "residual_norm"),
        force_norm=_read_scalar(variables, "force_norm"),
        fsq=_read_optional_scalar(variables, "fsq", "residual_norm"),
        normalized_force=_read_optional_scalar(variables, "normalized_force", "force_norm"),
        active_force_dof=_read_optional_int_scalar(variables, "active_force_dof", 0),
        min_sqrtg=_read_scalar(variables, "min_sqrtg"),
        max_sqrtg=_read_scalar(variables, "max_sqrtg"),
        min_bmag=_read_scalar(variables, "min_Bmag"),
        max_bmag=_read_scalar(variables, "max_Bmag"),
        mirror_ratio=_read_scalar(variables, "mirror_ratio"),
    )


def _read_history(variables: Any) -> MirrorOutputHistory:
    return MirrorOutputHistory(
        stage_index=_read_int(variables, "solve_history_stage_index"),
        iteration=_read_int(variables, "solve_history_iteration"),
        pressure_scale=_read_float(variables, "solve_history_pressure_scale"),
        energy_total=_read_float(variables, "solve_history_energy_total"),
        residual_norm=_read_float(variables, "solve_history_residual_norm"),
        fsq=_read_optional_float(variables, "solve_history_fsq", "solve_history_residual_norm"),
        normalized_force=_read_optional_float(
            variables,
            "solve_history_normalized_force",
            "solve_history_residual_norm",
        ),
        active_force_dof=_read_optional_int(
            variables,
            "solve_history_active_force_dof",
            fallback_value=0,
            fallback_length=np.asarray(variables["solve_history_iteration"][:]).size,
        ),
        min_sqrtg=_read_float(variables, "solve_history_min_sqrtg"),
        max_sqrtg=_read_float(variables, "solve_history_max_sqrtg"),
        min_bmag=_read_float(variables, "solve_history_min_Bmag"),
        max_bmag=_read_float(variables, "solve_history_max_Bmag"),
        mirror_ratio=_read_float(variables, "solve_history_mirror_ratio"),
        step_size=_read_float(variables, "solve_history_step_size"),
        accepted=_read_int(variables, "solve_history_accepted").astype(bool),
    )


def mirror_output_from_result(result) -> MirrorOutput:
    """Build an in-memory mirror output payload from a fixed-boundary result."""
    grid = result.grid
    if np.asarray(result.state.a).ndim == 3:
        geometry = evaluate_geometry_3d(result.state, grid)
        field = evaluate_field_3d(
            result.state,
            grid,
            geometry,
            psi_prime=result.psi_prime,
            i_prime=result.i_prime,
        )
        energy = total_energy_3d(field, result.pressure, geometry, grid, mu0=result.options.mu0)
        residual = projected_energy_residual_3d(
            result.state,
            grid,
            psi_prime=result.psi_prime,
            i_prime=result.i_prime,
            pressure=result.pressure,
            mu0=result.options.mu0,
        )
    else:
        geometry = evaluate_axisym_geometry(result.state, grid)
        field = evaluate_axisym_field(
            result.state,
            grid,
            geometry,
            psi_prime=result.psi_prime,
            i_prime=result.i_prime,
        )
        energy = total_energy_axisym(field, result.pressure, geometry, grid, mu0=result.options.mu0)
        residual = axisym_projected_energy_residual(
            result.state,
            grid,
            psi_prime=result.psi_prime,
            i_prime=result.i_prime,
            pressure=result.pressure,
            mu0=result.options.mu0,
        )
    return MirrorOutput(
        path=None,
        attributes=_attrs_with_result_metadata(result),
        s=np.asarray(grid.s_full, dtype=float),
        theta=np.asarray(grid.theta, dtype=float),
        xi=np.asarray(grid.xi, dtype=float),
        z=np.asarray(grid.z, dtype=float),
        w_s=np.asarray(grid.w_s, dtype=float),
        w_theta=np.asarray(grid.w_theta, dtype=float),
        w_xi=np.asarray(grid.w_xi, dtype=float),
        geometry=_geometry_output(result, geometry),
        field=_field_output(result, field),
        profiles=_profile_output(result, field),
        diagnostics=_diagnostics_output(energy, residual, geometry, field_diagnostics(field, grid)),
        history=_history_output(result.trace),
    )


def write_mirror_output(path: str | Path, output_or_result, *, overwrite: bool = False) -> Path:
    """Write a mirror-native NetCDF output file and return its path."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists (pass overwrite=True to overwrite)")
    output = (
        output_or_result if isinstance(output_or_result, MirrorOutput) else mirror_output_from_result(output_or_result)
    )

    try:
        import netCDF4  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("netCDF4 is required to write mirror output files") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(path, mode="w", format="NETCDF3_CLASSIC") as ds:
        try:
            ds.set_fill_off()
        except Exception:
            pass
        _write_dimensions(ds, output)
        for key, value in output.attributes.items():
            ds.setncattr(key, str(value))
        _write_output_groups(ds, output)
        _write_diagnostic_scalars(ds, output)
        _write_history(ds, output)
    return path


def read_mirror_output(path: str | Path) -> MirrorOutput:
    """Read a mirror-native NetCDF output file."""
    path = Path(path)
    try:
        import netCDF4  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("netCDF4 is required to read mirror output files") from exc

    with netCDF4.Dataset(path) as ds:
        attrs = {name: str(getattr(ds, name)) for name in ds.ncattrs()}
        variables = ds.variables
        return MirrorOutput(
            path=path,
            attributes=attrs,
            s=_read_float(variables, "s"),
            theta=_read_float(variables, "theta"),
            xi=_read_float(variables, "xi"),
            z=_read_float(variables, "z"),
            w_s=_read_float(variables, "w_s"),
            w_theta=_read_float(variables, "w_theta"),
            w_xi=_read_float(variables, "w_xi"),
            geometry=_read_geometry(variables),
            field=_read_field(variables),
            profiles=_read_profiles(variables),
            diagnostics=_read_diagnostics(variables),
            history=_read_history(variables),
        )


def load_mirror_output(path: str | Path) -> MirrorOutput:
    """Alias for :func:`read_mirror_output`."""
    return read_mirror_output(path)


def is_mirror_output(path: str | Path) -> bool:
    """Return ``True`` when *path* is a mirror-native NetCDF output."""
    path = Path(path)
    if path.suffix.lower() != ".nc":
        return False
    if path.name.lower().startswith("mout_"):
        return True
    try:
        import netCDF4  # type: ignore

        with netCDF4.Dataset(path) as ds:
            return str(getattr(ds, "geometry_type", "")).strip().lower() == "mirror"
    except Exception:
        return False
