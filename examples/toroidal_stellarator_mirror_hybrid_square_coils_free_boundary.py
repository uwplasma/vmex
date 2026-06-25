"""Free-boundary square-coil stellarator-mirror hybrid beta scan.

Edit the parameters in the first block below, then run this file directly:

    python examples/toroidal_stellarator_mirror_hybrid_square_coils_free_boundary.py

The script builds a square array of circular/elliptical coils, writes one
free-boundary VMEC input per beta value, runs ``vmec_jax.run_free_boundary``
with the direct-coil external-field provider, writes WOUT files, and plots the
solved LCFS and solved-equilibrium field-line traces.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vmec_jax._compat import enable_x64
from vmec_jax.driver import run_free_boundary, write_wout_from_fixed_boundary_run
from vmec_jax.external_fields import (
    build_coil_field_geometry,
    ellipse_coil_field_params,
    sample_coil_field_xyz_from_geometry,
)
from vmec_jax.field import b2_from_bsup, b_cartesian_from_bsup, bsup_from_geom, lamscale_from_phips
from vmec_jax.fieldlines import FieldLine, trace_fieldline_on_surface
from vmec_jax.free_boundary_validation import virtual_casing_finite_beta_boundary_diagnostics
from vmec_jax.geom import eval_geom
from vmec_jax.namelist import InData, write_indata
from vmec_jax.plotting import fix_matplotlib_3d, prepare_matplotlib_3d
from vmec_jax.profiles import pressure_profile_to_vmec_am, standard_finite_beta_profiles
from vmec_jax.toroidal_hybrid import (
    recommended_square_axis_nzeta,
    square_axis_stellarator_mirror_hybrid_indata,
    square_axis_stellarator_mirror_hybrid_projection_error,
)
from vmec_jax.wout import equilibrium_aspect_ratio_from_state, equilibrium_iota_profiles_from_state


# ---------------------------------------------------------------------------
# Input parameters.  Keep this block as the user-facing control surface.
# ---------------------------------------------------------------------------

OUTDIR = ROOT / "results" / "toroidal_stellarator_mirror_hybrid_square_coils"
BETAS_PERCENT = (0.0, 1.0, 3.0, 10.0)

N_COILS_PER_SIDE = 4
COIL_SQUARE_SIDE_LENGTH = 3.0
COIL_MAJOR_RADIUS = 0.50
COIL_MINOR_RADIUS = 0.50
COIL_CURRENT = 8.0e5
COIL_SEGMENTS = 96
COIL_CHUNK_SIZE: int | None = 512

PLASMA_AXIS_HALF_WIDTH = 0.5 * COIL_SQUARE_SIDE_LENGTH
PLASMA_AXIS_KIND = "spline"
PLASMA_AXIS_SQUARE_POWER = 3.0
PLASMA_AXIS_SPLINE_CORNER_RADIUS_FACTOR = 1.14
PLASMA_MINOR_RADIUS = 0.03
SIDE_ELONGATION = 0.08
SIDE_MINOR_MODULATION = 0.08
CORNER_ELLIPTICITY = 0.04
CORNER_AMPLITUDE = 0.004
CORNER_ROTATION = 0.30
CORNER_HELICITY = 1

NFP = 1
MPOL = 6
NTOR = 23
NS_ARRAY = (9, 13, 17)
NS = NS_ARRAY[-1]
NZETA = max(64, recommended_square_axis_nzeta(NTOR))
NITER_ARRAY = (4000, 8000, 12000)
FTOL_ARRAY = (1.0e-8, 1.0e-10, 1.0e-12)
USE_MULTIGRID_SCHEDULE = True
ENFORCE_RECOMMENDED_NZETA = True
MAX_BOUNDARY_PROJECTION_ERROR: float | None = 5.0e-5
NVACSKIP = 1
MAX_ITER = NITER_ARRAY[-1]
FTOL = 1.0e-12
PHIEDGE = -0.04 * PLASMA_MINOR_RADIUS**2 / 0.03**2
TOROIDAL_CURRENT = 3.0e3
DELT: float | None = 0.02
FREE_BOUNDARY_ACTIVATE_FSQ: float | None = 1.0e-3
SOLVER_MODE = "parity"
RETURN_BEST_SCORED_STATE = True
LIMIT_UPDATE_RMS = False
BACKTRACKING = False
USE_DIRECT_FALLBACK = False
BETA_CONTINUATION_RESTART = True
CHECKPOINT_EACH_BETA = True
JIT_FORCES: bool | str = "auto"

FIELD_LINE_COUNT = 3
FIELD_LINE_STEPS = 900
FIELD_LINE_TURNS = 1.25


SCHEMA = "toroidal_stellarator_mirror_hybrid_square_coils_free_boundary_solve"
SCHEMA_VERSION = "0.2"


@dataclass(frozen=True)
class ExampleConfig:
    outdir: Path = OUTDIR
    betas_percent: tuple[float, ...] = BETAS_PERCENT
    n_coils_per_side: int = N_COILS_PER_SIDE
    coil_square_side_length: float = COIL_SQUARE_SIDE_LENGTH
    coil_major_radius: float = COIL_MAJOR_RADIUS
    coil_minor_radius: float = COIL_MINOR_RADIUS
    coil_current: float = COIL_CURRENT
    coil_segments: int = COIL_SEGMENTS
    coil_chunk_size: int | None = COIL_CHUNK_SIZE
    plasma_axis_half_width: float = PLASMA_AXIS_HALF_WIDTH
    plasma_axis_kind: str = PLASMA_AXIS_KIND
    plasma_axis_square_power: float = PLASMA_AXIS_SQUARE_POWER
    plasma_axis_spline_corner_radius_factor: float = PLASMA_AXIS_SPLINE_CORNER_RADIUS_FACTOR
    plasma_minor_radius: float = PLASMA_MINOR_RADIUS
    side_elongation: float = SIDE_ELONGATION
    side_minor_modulation: float = SIDE_MINOR_MODULATION
    corner_ellipticity: float = CORNER_ELLIPTICITY
    corner_amplitude: float = CORNER_AMPLITUDE
    corner_rotation: float = CORNER_ROTATION
    corner_helicity: int = CORNER_HELICITY
    nfp: int = NFP
    mpol: int = MPOL
    ntor: int = NTOR
    ns: int = NS
    ns_array: tuple[int, ...] = NS_ARRAY
    nzeta: int = NZETA
    max_iter: int = MAX_ITER
    ftol: float = FTOL
    niter_array: tuple[int, ...] = NITER_ARRAY
    ftol_array: tuple[float, ...] = FTOL_ARRAY
    use_multigrid_schedule: bool = USE_MULTIGRID_SCHEDULE
    enforce_recommended_nzeta: bool = ENFORCE_RECOMMENDED_NZETA
    max_boundary_projection_error: float | None = MAX_BOUNDARY_PROJECTION_ERROR
    nvacskip: int = NVACSKIP
    phiedge: float = PHIEDGE
    toroidal_current: float = TOROIDAL_CURRENT
    delt: float | None = DELT
    free_boundary_activate_fsq: float | None = FREE_BOUNDARY_ACTIVATE_FSQ
    solver_mode: str | None = SOLVER_MODE
    return_best_scored_state: bool = RETURN_BEST_SCORED_STATE
    limit_update_rms: bool = LIMIT_UPDATE_RMS
    backtracking: bool = BACKTRACKING
    use_direct_fallback: bool = USE_DIRECT_FALLBACK
    beta_continuation_restart: bool = BETA_CONTINUATION_RESTART
    checkpoint_each_beta: bool = CHECKPOINT_EACH_BETA
    jit_forces: bool | str = JIT_FORCES
    field_line_count: int = FIELD_LINE_COUNT
    field_line_steps: int = FIELD_LINE_STEPS
    field_line_turns: float = FIELD_LINE_TURNS
    write_plots: bool = True


@dataclass(frozen=True)
class SquareCoilSet:
    params: Any
    centers: np.ndarray
    normals: np.ndarray
    major_axes: np.ndarray
    currents: np.ndarray
    side_index: np.ndarray
    side_coordinate: np.ndarray


@dataclass(frozen=True)
class SolvedBetaCase:
    beta_percent: float
    input_path: Path
    wout_path: Path
    run: Any
    wall_s: float
    theta: np.ndarray
    zeta: np.ndarray
    R: np.ndarray
    Z: np.ndarray
    Bmag: np.ndarray
    Bmag_near_axis: np.ndarray
    Bxyz: np.ndarray
    bsupu: np.ndarray
    bsupv: np.ndarray
    field_lines: tuple[FieldLine, ...]
    row: dict[str, Any]


def _json_sanitize(value: Any) -> Any:
    """Convert arrays and non-finite numbers into strict JSON values."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_sanitize(value.tolist())
    if isinstance(value, np.generic):
        return _json_sanitize(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_sanitize(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_sanitize(val) for val in value]
    return value


def _finite_history(values: Any) -> np.ndarray:
    try:
        arr = np.asarray(values, dtype=float).reshape(-1)
    except Exception:
        return np.zeros((0,), dtype=float)
    return arr[np.isfinite(arr)]


def _history_stats(values: Any) -> dict[str, float | int | None]:
    arr = _finite_history(values)
    if not arr.size:
        return {"count": 0, "first": None, "final": None, "min": None, "max": None}
    return {
        "count": int(arr.size),
        "first": float(arr[0]),
        "final": float(arr[-1]),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _finite_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except Exception:
        return None
    return result if np.isfinite(result) else None


def _classify_stall(row: dict[str, Any]) -> str:
    if bool(row.get("converged")):
        return "converged"
    if row.get("n_iter") is None:
        return "no_iteration_result"
    bad_resets = int(row.get("bad_resets") or 0)
    ijacob = int(row.get("ijacob") or 0)
    if bad_resets > 0 or ijacob != 0:
        return "bad_jacobian_or_restart_limited"
    bmin = row.get("free_boundary_bnormal_history_min")
    bfinal = row.get("free_boundary_bnormal_history_final")
    if bmin not in (None, 0.0) and bfinal is not None and float(bfinal) > 2.0 * float(bmin):
        return "free_boundary_bnormal_cycling"
    return "max_iter_residual_floor"


def build_square_coils(config: ExampleConfig) -> SquareCoilSet:
    """Create ``4*N`` elliptical coils on a square centered at the origin."""

    n = int(config.n_coils_per_side)
    if n < 1:
        raise ValueError("n_coils_per_side must be positive")
    half = 0.5 * float(config.coil_square_side_length)
    q_values = np.linspace(-half, half, n + 2, dtype=float)[1:-1]
    vertical = np.asarray([0.0, 0.0, 1.0])
    side_specs = (
        (lambda q: np.asarray([q, half, 0.0]), np.asarray([1.0, 0.0, 0.0])),
        (lambda q: np.asarray([half, -q, 0.0]), np.asarray([0.0, -1.0, 0.0])),
        (lambda q: np.asarray([-q, -half, 0.0]), np.asarray([-1.0, 0.0, 0.0])),
        (lambda q: np.asarray([-half, q, 0.0]), np.asarray([0.0, 1.0, 0.0])),
    )
    centers: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    major_axes: list[np.ndarray] = []
    side_index: list[int] = []
    side_coordinate: list[float] = []
    for idx, (center_for, side_tangent) in enumerate(side_specs):
        tangent = side_tangent / np.linalg.norm(side_tangent)
        for q in q_values:
            centers.append(np.asarray(center_for(float(q)), dtype=float))
            normals.append(tangent)
            major_axes.append(vertical)
            side_index.append(idx)
            side_coordinate.append(float(q))
    centers_arr = np.asarray(centers, dtype=float)
    normals_arr = np.asarray(normals, dtype=float)
    major_axes_arr = np.asarray(major_axes, dtype=float)
    currents = np.full((centers_arr.shape[0],), float(config.coil_current), dtype=float)
    chunk_size = None if config.coil_chunk_size is None else int(config.coil_chunk_size)
    if chunk_size is not None and chunk_size <= 0:
        chunk_size = None
    params = ellipse_coil_field_params(
        centers=centers_arr,
        normals=normals_arr,
        major_axes=major_axes_arr,
        major_radius=float(config.coil_major_radius),
        minor_radius=float(config.coil_minor_radius),
        currents=currents,
        n_segments=int(config.coil_segments),
        nfp=1,
        stellsym=False,
        regularization_epsilon=1.0e-6 * min(config.coil_major_radius, config.coil_minor_radius),
        chunk_size=chunk_size,
    )
    return SquareCoilSet(
        params=params,
        centers=centers_arr,
        normals=normals_arr,
        major_axes=major_axes_arr,
        currents=currents,
        side_index=np.asarray(side_index, dtype=int),
        side_coordinate=np.asarray(side_coordinate, dtype=float),
    )


def _case_label(beta_percent: float) -> str:
    return f"beta_{float(beta_percent):06.3f}".replace(".", "p").replace("-", "m")


def _pressure_terms(beta_percent: float) -> tuple[list[float], float]:
    if float(beta_percent) == 0.0:
        return [0.0], 0.0
    profiles = standard_finite_beta_profiles(float(beta_percent))
    am, pres_scale = pressure_profile_to_vmec_am(profiles.pressure_pa, pres_scale=1.0)
    return [float(x) for x in am], float(pres_scale)


def _stage_values(config: ExampleConfig) -> tuple[list[int] | int, list[int] | int, list[float] | float]:
    if not bool(config.use_multigrid_schedule):
        return int(config.ns), int(config.max_iter), float(config.ftol)
    ns_values = [int(value) for value in config.ns_array]
    niter_values = [int(value) for value in config.niter_array]
    ftol_values = [float(value) for value in config.ftol_array]
    if not (len(ns_values) == len(niter_values) == len(ftol_values)):
        raise ValueError("ns_array, niter_array, and ftol_array must have matching lengths")
    if not ns_values:
        raise ValueError("multigrid schedule must contain at least one stage")
    if any(value < 3 for value in ns_values):
        raise ValueError("all NS_ARRAY values must be at least 3")
    if any(value <= 0 for value in niter_values):
        raise ValueError("all NITER_ARRAY values must be positive")
    if any((not np.isfinite(value)) or value <= 0.0 for value in ftol_values):
        raise ValueError("all FTOL_ARRAY values must be finite and positive")
    return ns_values, niter_values, ftol_values


def _run_budget(config: ExampleConfig, *, restart_state: Any | None) -> int:
    if bool(config.use_multigrid_schedule) and restart_state is None:
        return int(sum(int(value) for value in config.niter_array))
    return int(config.max_iter)


def _validate_example_config(config: ExampleConfig) -> None:
    if int(config.mpol) < 3:
        raise ValueError("mpol must be at least 3 so the square-hybrid corner shaping fits")
    if int(config.ntor) < 4:
        raise ValueError("ntor must be at least 4 so the square-like axis fits")
    if int(config.nzeta) < 8:
        raise ValueError("nzeta must be at least 8")
    if config.solver_mode is not None:
        solver_mode = str(config.solver_mode).strip().lower()
        if solver_mode not in {"default", "parity", "accelerated"}:
            raise ValueError("solver_mode must be one of: default, parity, accelerated, or None")
    if int(config.nvacskip) < 1:
        raise ValueError("nvacskip must be at least 1")
    if bool(config.enforce_recommended_nzeta):
        recommended = recommended_square_axis_nzeta(int(config.ntor))
        if int(config.nzeta) < recommended:
            raise ValueError(
                f"NZETA={int(config.nzeta)} is underresolved for NTOR={int(config.ntor)}; "
                f"use at least {recommended} or set enforce_recommended_nzeta=False for a diagnostic-only run"
            )
    if config.max_boundary_projection_error is not None:
        limit = float(config.max_boundary_projection_error)
        if not np.isfinite(limit) or limit <= 0.0:
            raise ValueError("max_boundary_projection_error must be positive, finite, or None")
        projection = _boundary_projection_payload(config)
        observed = float(projection["max_abs_component_error"])
        if observed > limit:
            raise ValueError(
                "square-hybrid boundary projection error is too large for a production solve: "
                f"max_abs_component_error={observed:.3e} exceeds {limit:.3e} "
                f"for MPOL={int(config.mpol)}, NTOR={int(config.ntor)}, NZETA={int(config.nzeta)}. "
                "Increase MPOL/NTOR/NZETA, keep plasma_axis_kind='spline', or set "
                "max_boundary_projection_error=None for a diagnostic-only run."
            )


def make_free_boundary_indata(config: ExampleConfig, *, beta_percent: float) -> InData:
    """Return the free-boundary input deck for one beta case."""

    ns_values, niter_values, ftol_values = _stage_values(config)
    indata = square_axis_stellarator_mirror_hybrid_indata(
        nfp=int(config.nfp),
        mpol=int(config.mpol),
        ntor=int(config.ntor),
        ntheta_fit=max(64, 4 * int(config.mpol)),
        nzeta_fit=max(128, 8 * int(config.ntor)),
        ns_array=ns_values,
        niter_array=niter_values,
        ftol_array=ftol_values,
        phiedge=float(config.phiedge),
        axis_half_width=float(config.plasma_axis_half_width),
        axis_kind=str(config.plasma_axis_kind),
        axis_square_power=float(config.plasma_axis_square_power),
        axis_spline_corner_radius_factor=float(config.plasma_axis_spline_corner_radius_factor),
        minor_radius=float(config.plasma_minor_radius),
        side_elongation=float(config.side_elongation),
        side_minor_modulation=float(config.side_minor_modulation),
        corner_ellipticity=float(config.corner_ellipticity),
        corner_amplitude=float(config.corner_amplitude),
        corner_rotation=float(config.corner_rotation),
        corner_helicity=int(config.corner_helicity),
    )
    am, pres_scale = _pressure_terms(float(beta_percent))
    indata.scalars.update(
        {
            "LFREEB": True,
            "MGRID_FILE": "DIRECT_COILS",
            "EXTCUR": [1.0],
            "NS_ARRAY": ns_values,
            "NITER_ARRAY": niter_values,
            "FTOL_ARRAY": ftol_values,
            "NITER": int(config.max_iter),
            "FTOL": float(config.ftol),
            "NZETA": int(config.nzeta),
            "NTHETA": 0,
            "NVACSKIP": max(1, int(config.nvacskip)),
            "PMASS_TYPE": "power_series",
            "AM": am,
            "PRES_SCALE": pres_scale,
            "NCURR": 1,
            "CURTOR": float(config.toroidal_current),
            "PCURR_TYPE": "power_series_i",
            "AC": [float(config.toroidal_current)],
            "PIOTA_TYPE": "power_series",
            "AI": [0.0],
        }
    )
    if config.delt is not None:
        indata.scalars["DELT"] = float(config.delt)
    return indata


def _boundary_projection_payload(config: ExampleConfig) -> dict[str, Any]:
    """Return the Fourier truncation error for the configured input boundary."""

    ns_values, niter_values, ftol_values = _stage_values(config)
    return square_axis_stellarator_mirror_hybrid_projection_error(
        nfp=int(config.nfp),
        mpol=int(config.mpol),
        ntor=int(config.ntor),
        ntheta_fit=max(64, 4 * int(config.mpol)),
        nzeta_fit=max(128, 8 * int(config.ntor)),
        ns_array=ns_values,
        niter_array=niter_values,
        ftol_array=ftol_values,
        phiedge=float(config.phiedge),
        axis_half_width=float(config.plasma_axis_half_width),
        axis_kind=str(config.plasma_axis_kind),
        axis_square_power=float(config.plasma_axis_square_power),
        axis_spline_corner_radius_factor=float(config.plasma_axis_spline_corner_radius_factor),
        minor_radius=float(config.plasma_minor_radius),
        side_elongation=float(config.side_elongation),
        side_minor_modulation=float(config.side_minor_modulation),
        corner_ellipticity=float(config.corner_ellipticity),
        corner_amplitude=float(config.corner_amplitude),
        corner_rotation=float(config.corner_rotation),
        corner_helicity=int(config.corner_helicity),
    )


def _solved_surface_and_field(run: Any, config: ExampleConfig) -> tuple[np.ndarray, ...]:
    """Sample the solved boundary and contravariant field on the VMEC grid."""

    geom = eval_geom(run.state, run.static)
    lamscale = lamscale_from_phips(run.flux.phips, run.static.s)
    bsupu, bsupv = bsup_from_geom(
        geom,
        phipf=run.flux.phipf,
        chipf=run.flux.chipf,
        nfp=int(config.nfp),
        signgs=int(run.signgs),
        lamscale=lamscale,
        flux_is_internal=True,
    )
    bmag = np.sqrt(np.asarray(b2_from_bsup(geom, bsupu, bsupv), dtype=float))
    bxyz = np.asarray(
        b_cartesian_from_bsup(geom, bsupu, bsupv, zeta=run.static.grid.zeta, nfp=int(config.nfp)),
        dtype=float,
    )
    near_axis_idx = 1 if int(bmag.shape[0]) > 1 else 0
    theta = np.asarray(run.static.grid.theta, dtype=float)
    zeta = np.asarray(run.static.grid.zeta, dtype=float)
    return (
        theta,
        zeta,
        np.asarray(geom.R[-1], dtype=float),
        np.asarray(geom.Z[-1], dtype=float),
        bmag[-1],
        bmag[near_axis_idx],
        np.moveaxis(bxyz[-1], -1, 0),
        np.asarray(bsupu[-1], dtype=float),
        np.asarray(bsupv[-1], dtype=float),
    )


def _trace_solved_field_lines(
    *,
    R: np.ndarray,
    Z: np.ndarray,
    bsupu: np.ndarray,
    bsupv: np.ndarray,
    Bmag: np.ndarray,
    config: ExampleConfig,
) -> tuple[FieldLine, ...]:
    lines = []
    dphi = 2.0 * np.pi * float(config.field_line_turns) / max(2, int(config.field_line_steps) - 1)
    for idx in range(int(config.field_line_count)):
        theta0 = 2.0 * np.pi * (idx + 0.5) / max(1, int(config.field_line_count))
        lines.append(
            trace_fieldline_on_surface(
                R=R,
                Z=Z,
                bsupu=bsupu,
                bsupv=bsupv,
                Bmag=Bmag,
                nfp=int(config.nfp),
                theta0=theta0,
                phi0=0.0,
                n_steps=int(config.field_line_steps),
                dphi=dphi,
            )
        )
    return tuple(lines)


def _surface_xyz_from_rz(R: np.ndarray, Z: np.ndarray, zeta: np.ndarray, *, nfp: int) -> np.ndarray:
    phi = np.asarray(zeta, dtype=float) / float(max(1, int(nfp)))
    return np.stack((R * np.cos(phi)[None, :], R * np.sin(phi)[None, :], Z), axis=0)


def _edge_pressure_from_run(run: Any) -> float:
    scalars = getattr(getattr(run, "indata", None), "scalars", {})
    am = np.asarray(scalars.get("AM", [0.0]), dtype=float).reshape(-1)
    pres_scale = float(scalars.get("PRES_SCALE", 1.0) or 0.0)
    if not am.size:
        return 0.0
    # VMEC power-series pressure is evaluated at s=1 on the LCFS.
    return float(pres_scale * np.sum(am))


def _virtual_casing_row_metrics(
    *,
    run: Any,
    coils: SquareCoilSet,
    config: ExampleConfig,
    R: np.ndarray,
    Z: np.ndarray,
    zeta: np.ndarray,
    Bxyz: np.ndarray,
) -> dict[str, Any]:
    surface_xyz = _surface_xyz_from_rz(R, Z, zeta, nfp=int(config.nfp))
    points_xyz = np.moveaxis(surface_xyz, 0, -1)
    try:
        __import__("virtual_casing_jax.functional")
    except ImportError:
        return {
            "virtual_casing_status": "skipped_missing_virtual_casing_jax",
            "virtual_casing_external_bnormal_residual_rms": None,
            "virtual_casing_external_bnormal_residual_max": None,
            "virtual_casing_pressure_balance_rms": None,
            "virtual_casing_pressure_balance_max": None,
        }
    try:
        coil_xyz = np.asarray(
            sample_coil_field_xyz_from_geometry(
                build_coil_field_geometry(coils.params),
                points_xyz,
                regularization_epsilon=float(getattr(coils.params, "regularization_epsilon", 0.0)),
                chunk_size=getattr(coils.params, "chunk_size", None),
            ),
            dtype=float,
        )
        target_external_b = np.moveaxis(coil_xyz, -1, 0)
        diagnostics = virtual_casing_finite_beta_boundary_diagnostics(
            surface_xyz,
            Bxyz,
            target_external_b=target_external_b,
            pressure=_edge_pressure_from_run(run),
            nfp=int(config.nfp),
            digits=6,
            quad_nt=max(2 * int(surface_xyz.shape[1]), int(surface_xyz.shape[1])),
            quad_np=max(2 * int(surface_xyz.shape[2]), int(surface_xyz.shape[2])),
        )
    except Exception as exc:
        return {
            "virtual_casing_status": f"failed:{type(exc).__name__}",
            "virtual_casing_external_bnormal_residual_rms": None,
            "virtual_casing_external_bnormal_residual_max": None,
            "virtual_casing_pressure_balance_rms": None,
            "virtual_casing_pressure_balance_max": None,
        }
    return {
        "virtual_casing_status": "computed",
        "virtual_casing_external_bnormal_residual_rms": diagnostics.external_bnormal_residual_rms,
        "virtual_casing_external_bnormal_residual_max": diagnostics.external_bnormal_residual_max,
        "virtual_casing_pressure_balance_rms": diagnostics.pressure_balance_rms,
        "virtual_casing_pressure_balance_max": diagnostics.pressure_balance_max,
    }


def _run_one_beta(
    config: ExampleConfig,
    coils: SquareCoilSet,
    *,
    beta_percent: float,
    restart_state: Any | None = None,
) -> SolvedBetaCase:
    label = _case_label(beta_percent)
    case_dir = Path(config.outdir) / label
    case_dir.mkdir(parents=True, exist_ok=True)
    input_path = case_dir / f"input.square_coil_hybrid_{label}"
    wout_path = case_dir / f"wout_square_coil_hybrid_{label}.nc"
    indata = make_free_boundary_indata(config, beta_percent=float(beta_percent))
    write_indata(input_path, indata)

    t0 = time.perf_counter()
    run_budget = _run_budget(config, restart_state=restart_state)
    use_multigrid = bool(config.use_multigrid_schedule and restart_state is None)
    previous_return_best = os.environ.get("VMEC_JAX_RETURN_BEST_SCORED_STATE")
    os.environ["VMEC_JAX_RETURN_BEST_SCORED_STATE"] = "1" if bool(config.return_best_scored_state) else "0"
    try:
        run = run_free_boundary(
            input_path,
            max_iter=int(run_budget),
            multigrid=use_multigrid,
            multigrid_use_input_niter=True,
            verbose=True,
            jit_forces=config.jit_forces,
            solver_mode=config.solver_mode,
            free_boundary_activate_fsq=(
                None if config.free_boundary_activate_fsq is None else float(config.free_boundary_activate_fsq)
            ),
            external_field_provider_kind="direct_coils",
            external_field_provider_params=coils.params,
            limit_update_rms=bool(config.limit_update_rms),
            backtracking=bool(config.backtracking),
            use_direct_fallback=bool(config.use_direct_fallback),
            restart_state=restart_state,
            use_initial_guess=False,
        )
    finally:
        if previous_return_best is None:
            os.environ.pop("VMEC_JAX_RETURN_BEST_SCORED_STATE", None)
        else:
            os.environ["VMEC_JAX_RETURN_BEST_SCORED_STATE"] = previous_return_best
    wall_s = time.perf_counter() - t0
    write_wout_from_fixed_boundary_run(wout_path, run, include_fsq=True)
    theta, zeta, R, Z, Bmag, Bmag_near_axis, Bxyz, bsupu, bsupv = _solved_surface_and_field(run, config)
    field_lines = _trace_solved_field_lines(R=R, Z=Z, bsupu=bsupu, bsupv=bsupv, Bmag=Bmag, config=config)

    diag = run.result.diagnostics if run.result is not None else {}
    freeb = diag.get("free_boundary", {}) if isinstance(diag, dict) else {}
    nestor = freeb.get("last_nestor_diagnostics", {}) if isinstance(freeb, dict) else {}
    w_stats = _history_stats([] if run.result is None else getattr(run.result, "w_history", []))
    bnormal_stats = _history_stats(diag.get("freeb_nestor_bnormal_rms_history", []) if isinstance(diag, dict) else [])
    try:
        aspect = float(equilibrium_aspect_ratio_from_state(state=run.state, static=run.static))
    except Exception:
        aspect = None
    try:
        _chips, iotas, _iotaf = equilibrium_iota_profiles_from_state(
            state=run.state,
            static=run.static,
            indata=run.indata,
            signgs=int(run.signgs),
        )
        mean_iota = float(np.nanmean(np.asarray(iotas, dtype=float)))
    except Exception:
        mean_iota = None
    row = {
        "beta_percent": float(beta_percent),
        "input": str(input_path),
        "wout": str(wout_path),
        "wall_s": float(wall_s),
        "n_iter": None if run.result is None else int(getattr(run.result, "n_iter", -1)),
        "run_budget": int(run_budget),
        "used_multigrid_schedule": bool(use_multigrid),
        "solver_mode": diag.get("solver_mode") if isinstance(diag, dict) else config.solver_mode,
        "requested_solver_mode": config.solver_mode,
        "use_scan": diag.get("use_scan") if isinstance(diag, dict) else None,
        "nvacskip": int(config.nvacskip),
        "return_best_scored_state_requested": bool(config.return_best_scored_state),
        "converged": None
        if run.result is None
        else bool(diag.get("converged", getattr(run.result, "converged", False))),
        "converged_strict": None if run.result is None else diag.get("converged_strict"),
        "requested_ftol": diag.get("requested_ftol") if isinstance(diag, dict) else None,
        "final_fsq": diag.get("final_fsq") if isinstance(diag, dict) else None,
        "final_fsqr": diag.get("final_fsqr") if isinstance(diag, dict) else None,
        "final_fsqz": diag.get("final_fsqz") if isinstance(diag, dict) else None,
        "final_fsql": diag.get("final_fsql") if isinstance(diag, dict) else None,
        "return_best_scored_state": diag.get("return_best_scored_state") if isinstance(diag, dict) else None,
        "returned_best_scored_state": diag.get("returned_best_scored_state") if isinstance(diag, dict) else None,
        "best_scored_iter": diag.get("best_scored_iter") if isinstance(diag, dict) else None,
        "best_scored_fsq": _finite_float_or_none(diag.get("best_scored_fsq") if isinstance(diag, dict) else None),
        "best_scored_fsqr": _finite_float_or_none(diag.get("best_scored_fsqr") if isinstance(diag, dict) else None),
        "best_scored_fsqz": _finite_float_or_none(diag.get("best_scored_fsqz") if isinstance(diag, dict) else None),
        "best_scored_fsql": _finite_float_or_none(diag.get("best_scored_fsql") if isinstance(diag, dict) else None),
        "best_scored_full_boundary_count": (
            diag.get("best_scored_full_boundary_count") if isinstance(diag, dict) else None
        ),
        "best_scored_fresh_boundary_count": (
            diag.get("best_scored_fresh_boundary_count") if isinstance(diag, dict) else None
        ),
        "free_boundary_convergence_blocked_count": (
            diag.get("free_boundary_convergence_blocked_count") if isinstance(diag, dict) else None
        ),
        "free_boundary_fresh_convergence_gate": (
            diag.get("free_boundary_fresh_convergence_gate") if isinstance(diag, dict) else None
        ),
        "free_boundary_fresh_convergence_recheck_count": (
            diag.get("free_boundary_fresh_convergence_recheck_count") if isinstance(diag, dict) else None
        ),
        "free_boundary_fresh_convergence_reject_count": (
            diag.get("free_boundary_fresh_convergence_reject_count") if isinstance(diag, dict) else None
        ),
        "free_boundary_fresh_convergence_failed_count": (
            diag.get("free_boundary_fresh_convergence_failed_count") if isinstance(diag, dict) else None
        ),
        "final_iter2_for_recompute": diag.get("final_iter2_for_recompute") if isinstance(diag, dict) else None,
        "free_boundary_current_residual_norms": (
            diag.get("free_boundary_current_residual_norms") if isinstance(diag, dict) else None
        ),
        "free_boundary_bnormal_rms": nestor.get("bnormal_rms") if isinstance(nestor, dict) else None,
        "free_boundary_bsqvac_rms": nestor.get("bsqvac_rms") if isinstance(nestor, dict) else None,
        "free_boundary_gsource_rms": nestor.get("gsource_rms") if isinstance(nestor, dict) else None,
        "history_w_count": w_stats["count"],
        "history_w_first": w_stats["first"],
        "history_w_final": w_stats["final"],
        "history_w_min": w_stats["min"],
        "history_w_max": w_stats["max"],
        "free_boundary_bnormal_history_count": bnormal_stats["count"],
        "free_boundary_bnormal_history_first": bnormal_stats["first"],
        "free_boundary_bnormal_history_final": bnormal_stats["final"],
        "free_boundary_bnormal_history_min": bnormal_stats["min"],
        "free_boundary_bnormal_history_max": bnormal_stats["max"],
        "bad_resets": diag.get("bad_resets") if isinstance(diag, dict) else None,
        "ijacob": diag.get("ijacob") if isinstance(diag, dict) else None,
        "final_residual_recomputed_on_accepted_state": (
            diag.get("final_residual_recomputed_on_accepted_state") if isinstance(diag, dict) else None
        ),
        "bmag_min": float(np.nanmin(Bmag)),
        "bmag_mean": float(np.nanmean(Bmag)),
        "bmag_max": float(np.nanmax(Bmag)),
        "bmag_mirror_ratio": float(np.nanmax(Bmag) / np.nanmin(Bmag)),
        "near_axis_bmag_min": float(np.nanmin(Bmag_near_axis)),
        "near_axis_bmag_mean": float(np.nanmean(Bmag_near_axis)),
        "near_axis_bmag_max": float(np.nanmax(Bmag_near_axis)),
        "near_axis_mirror_ratio": float(np.nanmax(Bmag_near_axis) / np.nanmin(Bmag_near_axis)),
        "aspect": aspect,
        "mean_iota": mean_iota,
    }
    row.update(_virtual_casing_row_metrics(run=run, coils=coils, config=config, R=R, Z=Z, zeta=zeta, Bxyz=Bxyz))
    components = [row["final_fsqr"], row["final_fsqz"], row["final_fsql"]]
    finite_components = [float(value) for value in components if value is not None and np.isfinite(float(value))]
    row["final_fsq_component_sum"] = float(sum(finite_components)) if finite_components else None
    if row["final_fsq_component_sum"] is not None and row["history_w_min"] not in (None, 0.0):
        row["final_fsq_component_sum_over_history_w_min"] = float(row["final_fsq_component_sum"]) / float(
            row["history_w_min"]
        )
    else:
        row["final_fsq_component_sum_over_history_w_min"] = None
    row["stall_classification"] = _classify_stall(row)
    return SolvedBetaCase(
        beta_percent=float(beta_percent),
        input_path=input_path,
        wout_path=wout_path,
        run=run,
        wall_s=wall_s,
        theta=theta,
        zeta=zeta,
        R=R,
        Z=Z,
        Bmag=Bmag,
        Bmag_near_axis=Bmag_near_axis,
        Bxyz=Bxyz,
        bsupu=bsupu,
        bsupv=bsupv,
        field_lines=field_lines,
        row=row,
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    keys = [
        "beta_percent",
        "input",
        "wout",
        "wall_s",
        "n_iter",
        "run_budget",
        "used_multigrid_schedule",
        "solver_mode",
        "requested_solver_mode",
        "use_scan",
        "nvacskip",
        "return_best_scored_state_requested",
        "converged",
        "converged_strict",
        "requested_ftol",
        "final_fsq",
        "final_fsqr",
        "final_fsqz",
        "final_fsql",
        "final_fsq_component_sum",
        "return_best_scored_state",
        "returned_best_scored_state",
        "best_scored_iter",
        "best_scored_fsq",
        "best_scored_fsqr",
        "best_scored_fsqz",
        "best_scored_fsql",
        "best_scored_full_boundary_count",
        "best_scored_fresh_boundary_count",
        "free_boundary_convergence_blocked_count",
        "free_boundary_fresh_convergence_gate",
        "free_boundary_fresh_convergence_recheck_count",
        "free_boundary_fresh_convergence_reject_count",
        "free_boundary_fresh_convergence_failed_count",
        "final_iter2_for_recompute",
        "free_boundary_current_residual_norms",
        "free_boundary_bnormal_rms",
        "free_boundary_bsqvac_rms",
        "free_boundary_gsource_rms",
        "virtual_casing_status",
        "virtual_casing_external_bnormal_residual_rms",
        "virtual_casing_external_bnormal_residual_max",
        "virtual_casing_pressure_balance_rms",
        "virtual_casing_pressure_balance_max",
        "history_w_count",
        "history_w_first",
        "history_w_final",
        "history_w_min",
        "history_w_max",
        "free_boundary_bnormal_history_count",
        "free_boundary_bnormal_history_first",
        "free_boundary_bnormal_history_final",
        "free_boundary_bnormal_history_min",
        "free_boundary_bnormal_history_max",
        "bad_resets",
        "ijacob",
        "final_residual_recomputed_on_accepted_state",
        "bmag_min",
        "bmag_mean",
        "bmag_max",
        "bmag_mirror_ratio",
        "near_axis_bmag_min",
        "near_axis_bmag_mean",
        "near_axis_bmag_max",
        "near_axis_mirror_ratio",
        "aspect",
        "mean_iota",
        "final_fsq_component_sum_over_history_w_min",
        "stall_classification",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})
    return path


def _import_matplotlib():
    prepare_matplotlib_3d()
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    return plt, Normalize, ScalarMappable


def _xyz_from_rz(R: np.ndarray, Z: np.ndarray, zeta: np.ndarray, *, nfp: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phi = np.asarray(zeta, dtype=float) / float(max(1, int(nfp)))
    return R * np.cos(phi)[None, :], R * np.sin(phi)[None, :], Z


def _closed(values: np.ndarray) -> np.ndarray:
    return np.r_[values, values[:1]]


def _write_coils_json(path: Path, coils: SquareCoilSet, config: ExampleConfig) -> Path:
    payload = {
        "format": "vmec_jax_square_axis_ellipse_coils",
        "coil_count": int(coils.centers.shape[0]),
        "n_coils_per_side": int(config.n_coils_per_side),
        "centers": coils.centers,
        "normals": coils.normals,
        "major_axes": coils.major_axes,
        "currents": coils.currents,
        "side_index": coils.side_index,
        "side_coordinate": coils.side_coordinate,
        "base_curve_dofs": np.asarray(coils.params.base_curve_dofs),
        "n_segments": int(coils.params.n_segments),
    }
    path.write_text(json.dumps(_json_sanitize(payload), indent=2, allow_nan=False) + "\n")
    return path


def _write_geometry_plot(outdir: Path, coils: SquareCoilSet, cases: list[SolvedBetaCase], config: ExampleConfig) -> Path:
    plt, Normalize, ScalarMappable = _import_matplotlib()
    outdir.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=min(case.beta_percent for case in cases), vmax=max(case.beta_percent for case in cases))
    gamma = np.asarray(build_coil_field_geometry(coils.params)[0])
    fig = plt.figure(figsize=(7.4, 6.2))
    ax = fig.add_subplot(111, projection="3d")
    for coil in gamma:
        closed = np.vstack([coil, coil[:1]])
        ax.plot(closed[:, 0], closed[:, 1], closed[:, 2], color="tab:orange", linewidth=0.85, alpha=0.85)
    for case in cases:
        X, Y, Z = _xyz_from_rz(case.R, case.Z, case.zeta, nfp=config.nfp)
        color = cmap(norm(case.beta_percent))
        ax.plot_surface(X, Y, Z, color=color, linewidth=0, alpha=0.18, shade=False)
        for line in case.field_lines:
            ax.plot(line.x, line.y, line.z, color="black", linewidth=1.35, alpha=0.82)
    ax.set_title("solved free-boundary LCFS, coils, and field lines")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    fix_matplotlib_3d(ax)
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.72, pad=0.08, label="beta [%]")
    path = outdir / "square_coil_hybrid_solved_geometry_3d.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_top_view_plot(outdir: Path, coils: SquareCoilSet, cases: list[SolvedBetaCase], config: ExampleConfig) -> Path:
    plt, Normalize, ScalarMappable = _import_matplotlib()
    outdir.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=min(case.beta_percent for case in cases), vmax=max(case.beta_percent for case in cases))
    gamma = np.asarray(build_coil_field_geometry(coils.params)[0])
    fig, ax = plt.subplots(figsize=(6.6, 6.1), constrained_layout=True)
    for coil in gamma:
        closed = np.vstack([coil, coil[:1]])
        ax.plot(closed[:, 0], closed[:, 1], color="tab:orange", linewidth=0.8, alpha=0.65)
    ax.plot(coils.centers[:, 0], coils.centers[:, 1], "o", color="tab:orange", markersize=4.0, label="coil centers")
    for case in cases:
        X, Y, _Z = _xyz_from_rz(case.R, case.Z, case.zeta, nfp=config.nfp)
        ax.plot(
            _closed(X[0]),
            _closed(Y[0]),
            color=cmap(norm(case.beta_percent)),
            linewidth=1.4,
            label=f"beta={case.beta_percent:g}%",
        )
    ax.set_aspect("equal", "box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("solved LCFS top view")
    ax.legend(fontsize="x-small", ncols=2)
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.78, pad=0.02, label="beta [%]")
    path = outdir / "square_coil_hybrid_solved_top_view.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_cross_sections_plot(outdir: Path, cases: list[SolvedBetaCase]) -> Path:
    plt, _Normalize, _ScalarMappable = _import_matplotlib()
    outdir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8), constrained_layout=True)
    targets = (0.0, 0.25 * np.pi)
    titles = ("side section", "corner section")
    for ax, target, title in zip(axes, targets, titles):
        for case in cases:
            idx = int(np.argmin(np.abs(np.mod(case.zeta, 2.0 * np.pi) - target)))
            ax.plot(
                _closed(case.R[:, idx] - np.mean(case.R[:, idx])),
                _closed(case.Z[:, idx]),
                label=f"{case.beta_percent:g}%",
            )
        ax.set_aspect("equal", "box")
        ax.set_xlabel("R - <R>")
        ax.set_ylabel("Z")
        ax.set_title(title)
    axes[0].legend(fontsize="small")
    path = outdir / "square_coil_hybrid_solved_cross_sections.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_bmag_plot(outdir: Path, cases: list[SolvedBetaCase]) -> Path:
    plt, _Normalize, _ScalarMappable = _import_matplotlib()
    outdir.mkdir(parents=True, exist_ok=True)
    selected = (cases[0], cases[-1])
    vmin = float(min(np.nanmin(case.Bmag) for case in selected))
    vmax = float(max(np.nanmax(case.Bmag) for case in selected))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True, sharey=True)
    mesh = None
    for ax, case in zip(axes, selected):
        mesh = ax.pcolormesh(case.zeta, case.theta, case.Bmag, shading="auto", vmin=vmin, vmax=vmax)
        ax.set_title(f"solved beta={case.beta_percent:g}%")
        ax.set_xlabel("zeta")
    axes[0].set_ylabel("theta")
    fig.colorbar(mesh, ax=axes, label="|B|")
    path = outdir / "square_coil_hybrid_solved_boundary_bmag.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_beta_response_plot(outdir: Path, cases: list[SolvedBetaCase]) -> Path:
    plt, _Normalize, _ScalarMappable = _import_matplotlib()
    outdir.mkdir(parents=True, exist_ok=True)
    beta = np.asarray([case.beta_percent for case in cases], dtype=float)
    near_min = np.asarray([case.row.get("near_axis_bmag_min") for case in cases], dtype=float)
    near_mean = np.asarray([case.row.get("near_axis_bmag_mean") for case in cases], dtype=float)
    near_max = np.asarray([case.row.get("near_axis_bmag_max") for case in cases], dtype=float)
    near_ratio = np.asarray([case.row.get("near_axis_mirror_ratio") for case in cases], dtype=float)
    boundary_ratio = np.asarray([case.row.get("bmag_mirror_ratio") for case in cases], dtype=float)
    beta_frac = np.clip(beta / 100.0, 0.0, 0.95)
    expected_ratio = near_ratio[0] / np.sqrt(1.0 - beta_frac) if near_ratio.size else beta_frac

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), constrained_layout=True)
    axes[0].plot(beta, near_min, "o-", label="near-axis min")
    axes[0].plot(beta, near_mean, "o-", label="near-axis mean")
    axes[0].plot(beta, near_max, "o-", label="near-axis max")
    axes[0].set_xlabel("beta [%]")
    axes[0].set_ylabel("|B|")
    axes[0].set_title("near-axis field response")
    axes[0].legend(fontsize="small")

    axes[1].plot(beta, near_ratio, "o-", label="near-axis")
    axes[1].plot(beta, boundary_ratio, "s-", label="LCFS")
    axes[1].plot(beta, expected_ratio, "--", color="0.35", label=r"$R_m(0)/\sqrt{1-\beta}$")
    axes[1].set_xlabel("beta [%]")
    axes[1].set_ylabel("mirror ratio")
    axes[1].set_title("mirror-ratio trend")
    axes[1].legend(fontsize="small")
    path = outdir / "square_coil_hybrid_solved_beta_response.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_convergence_plot(outdir: Path, cases: list[SolvedBetaCase]) -> Path:
    plt, _Normalize, _ScalarMappable = _import_matplotlib()
    outdir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 7.4), constrained_layout=True)

    def _history(diag: dict[str, Any], *names: str) -> np.ndarray:
        for name in names:
            value = diag.get(name)
            if value is None:
                continue
            arr = np.asarray(value, dtype=float).reshape(-1)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                return arr
        return np.zeros((0,), dtype=float)

    for case in cases:
        diag = case.run.result.diagnostics if case.run.result is not None else {}
        diag = diag if isinstance(diag, dict) else {}
        total = np.zeros((0,), dtype=float)
        if case.run.result is not None:
            total = np.asarray(getattr(case.run.result, "w_history", []), dtype=float).reshape(-1)
            total = total[np.isfinite(total)]
        if not total.size:
            total = _history(diag, "fsq1_history", "w_history")
        if not total.size:
            parts = [_history(diag, key) for key in ("fsqr1_history", "fsqz1_history", "fsql1_history")]
            min_size = min((part.size for part in parts if part.size), default=0)
            if min_size:
                total = sum(part[:min_size] for part in parts if part.size)
        if not total.size:
            final = case.row.get("final_fsq") or case.row.get("final_fsq_component_sum")
            if final is not None and np.isfinite(float(final)):
                total = np.asarray([float(final)])
        if total.size:
            x = np.arange(total.size)
            marker = None
            if total.size == 1:
                x = np.asarray([case.row.get("n_iter") or 0], dtype=float)
                marker = "o"
            (line,) = axes[0].semilogy(x, total, marker=marker, linewidth=1.2, label=f"{case.beta_percent:g}%")
            final_fsq = case.row.get("final_fsq_component_sum") or case.row.get("final_fsq")
            final_iter = case.row.get("final_iter2_for_recompute") or case.row.get("n_iter") or x[-1]
            if final_fsq is not None and np.isfinite(float(final_fsq)):
                axes[0].semilogy(
                    [float(final_iter)],
                    [float(final_fsq)],
                    marker="x",
                    linestyle="None",
                    color=line.get_color(),
                    markersize=6,
                    label=f"{case.beta_percent:g}% final",
                )

        bnormal = _history(diag, "freeb_nestor_bnormal_rms_history")
        if not bnormal.size:
            final_bnormal = case.row.get("free_boundary_bnormal_rms")
            if final_bnormal is not None and np.isfinite(float(final_bnormal)):
                bnormal = np.asarray([float(final_bnormal)])
        if bnormal.size:
            x = np.arange(bnormal.size)
            marker = None
            if bnormal.size == 1:
                x = np.asarray([case.row.get("n_iter") or 0], dtype=float)
                marker = "o"
            (line,) = axes[1].semilogy(x, bnormal, marker=marker, linewidth=1.2, label=f"{case.beta_percent:g}%")
            final_bnormal = case.row.get("free_boundary_bnormal_rms")
            final_iter = case.row.get("final_iter2_for_recompute") or case.row.get("n_iter") or x[-1]
            if final_bnormal is not None and np.isfinite(float(final_bnormal)):
                axes[1].semilogy(
                    [float(final_iter)],
                    [float(final_bnormal)],
                    marker="x",
                    linestyle="None",
                    color=line.get_color(),
                    markersize=6,
                    label=f"{case.beta_percent:g}% final",
                )

    beta = np.asarray([case.beta_percent for case in cases], dtype=float)
    axes[2].plot(beta, [case.row.get("mean_iota") for case in cases], "o-", label="mean iota")
    axes[0].set_xlabel("iteration")
    axes[1].set_xlabel("iteration")
    axes[2].set_xlabel("beta [%]")
    axes[0].set_ylabel("fsq")
    axes[1].set_ylabel("B.n RMS")
    axes[2].set_ylabel("mean iota")
    axes[0].set_title("free-boundary solve diagnostics")
    if axes[0].lines:
        axes[0].legend(fontsize="small", ncols=2)
    if axes[1].lines:
        axes[1].legend(fontsize="small", ncols=2)
    path = outdir / "square_coil_hybrid_solved_convergence_iota.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_plots(outdir: Path, coils: SquareCoilSet, cases: list[SolvedBetaCase], config: ExampleConfig) -> dict[str, str]:
    figure_dir = outdir / "figures"
    return {
        "geometry_3d": str(_write_geometry_plot(figure_dir, coils, cases, config)),
        "top_view": str(_write_top_view_plot(figure_dir, coils, cases, config)),
        "cross_sections": str(_write_cross_sections_plot(figure_dir, cases)),
        "boundary_bmag": str(_write_bmag_plot(figure_dir, cases)),
        "beta_response": str(_write_beta_response_plot(figure_dir, cases)),
        "convergence_iota": str(_write_convergence_plot(figure_dir, cases)),
    }


def _metrics_payload(
    *,
    config: ExampleConfig,
    coils: SquareCoilSet,
    coils_json: Path,
    summary_csv: Path,
    rows: list[dict[str, Any]],
    figures: dict[str, str],
    complete: bool,
) -> dict[str, Any]:
    all_converged = bool(complete) and all(bool(row.get("converged")) for row in rows)
    completed = [float(row.get("beta_percent")) for row in rows]
    remaining = [float(beta) for beta in config.betas_percent if float(beta) not in completed]
    return {
        "metrics_schema": SCHEMA,
        "metrics_schema_version": SCHEMA_VERSION,
        "workflow_status": "actual_vmec_jax_free_boundary_beta_scan",
        "free_boundary_solve_status": (
            "converged" if all_converged else "not_converged_or_max_iter" if complete else "partial_running"
        ),
        "hybrid_fixture_kind": "square_axis_toroidal_stellarator_mirror_hybrid",
        "actual_free_boundary_solve": True,
        "production_free_boundary_claim": False,
        "betas_percent": [float(beta) for beta in config.betas_percent],
        "completed_betas_percent": completed,
        "remaining_betas_percent": remaining,
        "coil_count": int(coils.centers.shape[0]),
        "n_coils_per_side": int(config.n_coils_per_side),
        "plasma_axis_half_width": float(config.plasma_axis_half_width),
        "plasma_axis_kind": str(config.plasma_axis_kind),
        "plasma_axis_spline_corner_radius_factor": float(config.plasma_axis_spline_corner_radius_factor),
        "coil_square_side_length": float(config.coil_square_side_length),
        "toroidal_current": float(config.toroidal_current),
        "boundary_projection": _boundary_projection_payload(config),
        "delt": None if config.delt is None else float(config.delt),
        "ns": int(config.ns),
        "ns_array": [int(value) for value in config.ns_array],
        "mpol": int(config.mpol),
        "ntor": int(config.ntor),
        "recommended_nzeta": int(recommended_square_axis_nzeta(int(config.ntor))),
        "nzeta_underrecommended": bool(int(config.nzeta) < recommended_square_axis_nzeta(int(config.ntor))),
        "max_iter": int(config.max_iter),
        "ftol": float(config.ftol),
        "niter_array": [int(value) for value in config.niter_array],
        "ftol_array": [float(value) for value in config.ftol_array],
        "use_multigrid_schedule": bool(config.use_multigrid_schedule),
        "enforce_recommended_nzeta": bool(config.enforce_recommended_nzeta),
        "max_boundary_projection_error": (
            None if config.max_boundary_projection_error is None else float(config.max_boundary_projection_error)
        ),
        "nvacskip": int(config.nvacskip),
        "return_best_scored_state": bool(config.return_best_scored_state),
        "free_boundary_activate_fsq": (
            None if config.free_boundary_activate_fsq is None else float(config.free_boundary_activate_fsq)
        ),
        "solver_mode": None if config.solver_mode is None else str(config.solver_mode),
        "limit_update_rms": bool(config.limit_update_rms),
        "backtracking": bool(config.backtracking),
        "use_direct_fallback": bool(config.use_direct_fallback),
        "beta_continuation_restart": bool(config.beta_continuation_restart),
        "checkpoint_each_beta": bool(config.checkpoint_each_beta),
        "coils_json": str(coils_json),
        "summary_csv": str(summary_csv),
        "rows": rows,
        "figures": figures,
    }


def run_example(config: ExampleConfig = ExampleConfig()) -> Path:
    enable_x64(True)
    _validate_example_config(config)
    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    coils = build_square_coils(config)
    coils_json = _write_coils_json(outdir / "square_mirror_hybrid_coils.json", coils, config)
    cases = []
    summary_csv = outdir / "square_coil_hybrid_free_boundary_solve_summary.csv"
    metrics_path = outdir / "square_coil_hybrid_free_boundary_solve_metrics.json"
    restart_state = None
    for beta in config.betas_percent:
        case = _run_one_beta(
            config,
            coils,
            beta_percent=float(beta),
            restart_state=restart_state if bool(config.beta_continuation_restart) else None,
        )
        cases.append(case)
        if bool(config.beta_continuation_restart):
            restart_state = case.run.state
        if bool(config.checkpoint_each_beta):
            rows = [solved.row for solved in cases]
            _write_csv(summary_csv, rows)
            metrics = _metrics_payload(
                config=config,
                coils=coils,
                coils_json=coils_json,
                summary_csv=summary_csv,
                rows=rows,
                figures={},
                complete=False,
            )
            metrics_path.write_text(json.dumps(_json_sanitize(metrics), indent=2, allow_nan=False) + "\n")
    rows = [case.row for case in cases]
    _write_csv(summary_csv, rows)
    figures = _write_plots(outdir, coils, cases, config) if bool(config.write_plots) else {}
    metrics = _metrics_payload(
        config=config,
        coils=coils,
        coils_json=coils_json,
        summary_csv=summary_csv,
        rows=rows,
        figures=figures,
        complete=True,
    )
    metrics_path.write_text(json.dumps(_json_sanitize(metrics), indent=2, allow_nan=False) + "\n")
    return metrics_path


if __name__ == "__main__":  # pragma: no cover
    print(run_example())
