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
from dataclasses import dataclass, replace
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
)
from vmec_jax.fieldlines import FieldLine, trace_fieldline_on_surface
from vmec_jax.solvers.free_boundary.validation import (
    free_boundary_promotion_status,
    sample_solved_boundary_field,
    virtual_casing_diagnostics_from_run,
)
from vmec_jax.namelist import InData, write_indata
from vmec_jax.plotting import fix_matplotlib_3d, prepare_matplotlib_3d
from vmec_jax.profiles import pressure_profile_to_vmec_am, standard_finite_beta_profiles
from vmec_jax.toroidal_hybrid import (
    SquareAxisSplineControls,
    recommend_square_axis_stellarator_mirror_hybrid_resolution,
    recommended_square_axis_ntheta,
    recommended_square_axis_nzeta,
    square_axis_free_boundary_edge_control_projection_payload,
    square_axis_resolution_deck_status,
    square_axis_spline_control_fourier_map_status,
    square_axis_spline_symmetric_control_basis,
    square_axis_strict_schedule_status,
    square_axis_strict_convergence_assessment,
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
PLASMA_AXIS_KIND = "control_spline"
PLASMA_AXIS_SQUARE_POWER = 3.0
PLASMA_AXIS_SPLINE_CORNER_RADIUS_FACTOR = 1.14
PLASMA_AXIS_SPLINE_CONTROL_COUNT = 16
PLASMA_AXIS_SPLINE_CONTROLS: SquareAxisSplineControls | None = None
PLASMA_AXIS_CONTROL_SYMMETRY = "square"
PLASMA_AXIS_REDUCED_RADII: tuple[float, ...] | None = None
PLASMA_MINOR_RADIUS = 0.03
SIDE_ELONGATION = 0.08
SIDE_MINOR_MODULATION = 0.08
SIDE_POWER = 1.0
CORNER_POWER = 1.0
CORNER_ELLIPTICITY = 0.04
CORNER_AMPLITUDE = 0.004
CORNER_ROTATION = 0.30
CORNER_HELICITY = 1

NFP = 1
MPOL = 5
NTOR = 28
NS_ARRAY = (9, 13, 17)
NS = NS_ARRAY[-1]
NTHETA: int | None = None
NZETA: int | None = None
NITER_ARRAY = (4000, 8000, 24000)
FTOL_ARRAY = (1.0e-8, 1.0e-10, 1.0e-12)
USE_MULTIGRID_SCHEDULE = True
ENFORCE_RECOMMENDED_NZETA = True
AUTO_BUMP_NZETA_TO_RECOMMENDED = True
AUTO_BUMP_MODE_DECK_TO_RECOMMENDED = True
MAX_BOUNDARY_PROJECTION_ERROR: float | None = 5.0e-12
NSTEP = 1
NVACSKIP = 1
MAX_ITER = NITER_ARRAY[-1]
FTOL = 1.0e-12
PHIEDGE = -0.04 * PLASMA_MINOR_RADIUS**2 / 0.03**2
TOROIDAL_CURRENT = 3.0e3
DELT: float | None = 0.02
FREE_BOUNDARY_ACTIVATE_FSQ: float | None = 1.0e-3
FREE_BOUNDARY_EDGE_CONTROL_PROJECTION = "square"
FREE_BOUNDARY_EDGE_CONTROL_RCOND = 1.0e-12
FREE_BOUNDARY_EDGE_CONTROL_RIDGE = 0.0
FREE_BOUNDARY_EDGE_CONTROL_TRUST_RADIUS: float | None = None
FREE_BOUNDARY_EDGE_CONTROL_UPDATE_MODE = "native_coordinate"
SOLVER_MODE = "parity"
RETURN_BEST_SCORED_STATE = True
FREE_BOUNDARY_DRIFT_RESTART = True
FREE_BOUNDARY_DRIFT_RESTART_FACTOR = 2.0
FREE_BOUNDARY_DRIFT_RESTART_STEP_FACTOR = 0.5
FREE_BOUNDARY_DRIFT_RESTART_MIN_ITER_SINCE_BEST = 20
FREE_BOUNDARY_DRIFT_RESTART_STREAK = 5
FREE_BOUNDARY_DRIFT_RESTART_MAX_RESTARTS = 4
LIMIT_UPDATE_RMS = False
BACKTRACKING = False
USE_DIRECT_FALLBACK = False
BETA_CONTINUATION_RESTART = True
CHECKPOINT_EACH_BETA = True
JIT_FORCES: bool | str = "auto"
PREFLIGHT_ONLY = False

FIELD_LINE_COUNT = 3
FIELD_LINE_STEPS = 900
FIELD_LINE_TURNS = 1.25
VIRTUAL_CASING_QUAD_FACTOR = 2
VIRTUAL_CASING_CHUNK_SIZE: int | str | None = "auto"
VIRTUAL_CASING_TARGET_CHUNK_SIZE: int | str | None = "auto"


SCHEMA = "toroidal_stellarator_mirror_hybrid_square_coils_free_boundary_solve"
SCHEMA_VERSION = "0.5"
STRICT_COMPONENT_FTOL_TARGET = 1.0e-12


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
    plasma_axis_spline_control_count: int = PLASMA_AXIS_SPLINE_CONTROL_COUNT
    plasma_axis_spline_controls: SquareAxisSplineControls | None = PLASMA_AXIS_SPLINE_CONTROLS
    plasma_axis_control_symmetry: str = PLASMA_AXIS_CONTROL_SYMMETRY
    plasma_axis_reduced_radii: tuple[float, ...] | None = PLASMA_AXIS_REDUCED_RADII
    plasma_minor_radius: float = PLASMA_MINOR_RADIUS
    side_elongation: float = SIDE_ELONGATION
    side_minor_modulation: float = SIDE_MINOR_MODULATION
    side_power: float = SIDE_POWER
    corner_power: float = CORNER_POWER
    corner_ellipticity: float = CORNER_ELLIPTICITY
    corner_amplitude: float = CORNER_AMPLITUDE
    corner_rotation: float = CORNER_ROTATION
    corner_helicity: int = CORNER_HELICITY
    nfp: int = NFP
    mpol: int = MPOL
    ntor: int = NTOR
    ns: int = NS
    ns_array: tuple[int, ...] = NS_ARRAY
    ntheta: int | None = NTHETA
    nzeta: int | None = NZETA
    max_iter: int = MAX_ITER
    ftol: float = FTOL
    niter_array: tuple[int, ...] = NITER_ARRAY
    ftol_array: tuple[float, ...] = FTOL_ARRAY
    use_multigrid_schedule: bool = USE_MULTIGRID_SCHEDULE
    enforce_recommended_nzeta: bool = ENFORCE_RECOMMENDED_NZETA
    auto_bump_nzeta_to_recommended: bool = AUTO_BUMP_NZETA_TO_RECOMMENDED
    auto_bump_mode_deck_to_recommended: bool = AUTO_BUMP_MODE_DECK_TO_RECOMMENDED
    max_boundary_projection_error: float | None = MAX_BOUNDARY_PROJECTION_ERROR
    nstep: int = NSTEP
    nvacskip: int = NVACSKIP
    phiedge: float = PHIEDGE
    toroidal_current: float = TOROIDAL_CURRENT
    delt: float | None = DELT
    free_boundary_activate_fsq: float | None = FREE_BOUNDARY_ACTIVATE_FSQ
    free_boundary_edge_control_projection: str = FREE_BOUNDARY_EDGE_CONTROL_PROJECTION
    free_boundary_edge_control_rcond: float = FREE_BOUNDARY_EDGE_CONTROL_RCOND
    free_boundary_edge_control_ridge: float = FREE_BOUNDARY_EDGE_CONTROL_RIDGE
    free_boundary_edge_control_trust_radius: float | None = FREE_BOUNDARY_EDGE_CONTROL_TRUST_RADIUS
    free_boundary_edge_control_update_mode: str = FREE_BOUNDARY_EDGE_CONTROL_UPDATE_MODE
    solver_mode: str | None = SOLVER_MODE
    return_best_scored_state: bool = RETURN_BEST_SCORED_STATE
    free_boundary_drift_restart: bool = FREE_BOUNDARY_DRIFT_RESTART
    free_boundary_drift_restart_factor: float = FREE_BOUNDARY_DRIFT_RESTART_FACTOR
    free_boundary_drift_restart_step_factor: float = FREE_BOUNDARY_DRIFT_RESTART_STEP_FACTOR
    free_boundary_drift_restart_min_iter_since_best: int = FREE_BOUNDARY_DRIFT_RESTART_MIN_ITER_SINCE_BEST
    free_boundary_drift_restart_streak: int = FREE_BOUNDARY_DRIFT_RESTART_STREAK
    free_boundary_drift_restart_max_restarts: int = FREE_BOUNDARY_DRIFT_RESTART_MAX_RESTARTS
    limit_update_rms: bool = LIMIT_UPDATE_RMS
    backtracking: bool = BACKTRACKING
    use_direct_fallback: bool = USE_DIRECT_FALLBACK
    beta_continuation_restart: bool = BETA_CONTINUATION_RESTART
    checkpoint_each_beta: bool = CHECKPOINT_EACH_BETA
    jit_forces: bool | str = JIT_FORCES
    preflight_only: bool = PREFLIGHT_ONLY
    field_line_count: int = FIELD_LINE_COUNT
    field_line_steps: int = FIELD_LINE_STEPS
    field_line_turns: float = FIELD_LINE_TURNS
    virtual_casing_quad_factor: int = VIRTUAL_CASING_QUAD_FACTOR
    virtual_casing_chunk_size: int | str | None = VIRTUAL_CASING_CHUNK_SIZE
    virtual_casing_target_chunk_size: int | str | None = VIRTUAL_CASING_TARGET_CHUNK_SIZE
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
class EffectiveSquareAxisResolution:
    """Requested, recommended, and effective grid sizes for a square-axis deck."""

    requested_ntheta: int | None
    effective_ntheta: int
    recommended_ntheta: int
    ntheta_auto_defaulted: bool
    ntheta_auto_bumped_to_recommended: bool
    requested_nzeta: int | None
    effective_nzeta: int
    recommended_nzeta: int
    nzeta_auto_defaulted: bool
    nzeta_auto_bumped_to_recommended: bool
    enforce_recommended_nzeta: bool
    auto_bump_nzeta_to_recommended: bool

    @property
    def ntheta_underrecommended(self) -> bool:
        """Whether the effective VMEC theta grid is below the square-axis recommendation."""

        return bool(self.effective_ntheta < self.recommended_ntheta)

    @property
    def nzeta_underrecommended(self) -> bool:
        """Whether the effective VMEC zeta grid is below the square-axis recommendation."""

        return bool(self.effective_nzeta < self.recommended_nzeta)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly combined resolution summary."""

        return {
            "requested_ntheta": self.requested_ntheta,
            "effective_ntheta": int(self.effective_ntheta),
            "recommended_ntheta": int(self.recommended_ntheta),
            "ntheta_auto_defaulted": bool(self.ntheta_auto_defaulted),
            "ntheta_auto_bumped_to_recommended": bool(self.ntheta_auto_bumped_to_recommended),
            "ntheta_underrecommended": self.ntheta_underrecommended,
            "requested_nzeta": self.requested_nzeta,
            "effective_nzeta": int(self.effective_nzeta),
            "recommended_nzeta": int(self.recommended_nzeta),
            "nzeta_auto_defaulted": bool(self.nzeta_auto_defaulted),
            "nzeta_auto_bumped_to_recommended": bool(self.nzeta_auto_bumped_to_recommended),
            "nzeta_underrecommended": self.nzeta_underrecommended,
            "enforce_recommended_nzeta": bool(self.enforce_recommended_nzeta),
            "auto_bump_nzeta_to_recommended": bool(self.auto_bump_nzeta_to_recommended),
        }

    def ntheta_payload(self) -> dict[str, Any]:
        """Return the legacy per-grid theta payload used by existing outputs."""

        return {
            "requested_ntheta": self.requested_ntheta,
            "effective_ntheta": int(self.effective_ntheta),
            "recommended_ntheta": int(self.recommended_ntheta),
            "auto_defaulted": bool(self.ntheta_auto_defaulted),
            "auto_bumped_to_recommended": bool(self.ntheta_auto_bumped_to_recommended),
        }

    def nzeta_payload(self) -> dict[str, Any]:
        """Return the legacy per-grid zeta payload used by existing outputs."""

        return {
            "requested_nzeta": self.requested_nzeta,
            "effective_nzeta": int(self.effective_nzeta),
            "recommended_nzeta": int(self.recommended_nzeta),
            "auto_defaulted": bool(self.nzeta_auto_defaulted),
            "auto_bumped_to_recommended": bool(self.nzeta_auto_bumped_to_recommended),
            "enforce_recommended_nzeta": bool(self.enforce_recommended_nzeta),
            "auto_bump_nzeta_to_recommended": bool(self.auto_bump_nzeta_to_recommended),
        }


@dataclass(frozen=True)
class EffectiveSquareAxisModeDeck:
    """Requested and effective Fourier modes for the square-axis target."""

    requested_mpol: int
    requested_ntor: int
    effective_mpol: int
    effective_ntor: int
    target_max_component_error: float | None
    requested_projection_max_abs_component_error: float
    effective_projection_max_abs_component_error: float
    requested_projection_meets_target: bool
    effective_projection_meets_target: bool
    auto_bump_mode_deck_to_recommended: bool
    mode_deck_auto_bumped_to_recommended: bool
    recommendation_status: str | None
    recommended_mpol: int
    recommended_ntor: int
    recommended_nzeta: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly mode-deck provenance block."""

        return {
            "requested_mpol": int(self.requested_mpol),
            "requested_ntor": int(self.requested_ntor),
            "effective_mpol": int(self.effective_mpol),
            "effective_ntor": int(self.effective_ntor),
            "target_max_component_error": self.target_max_component_error,
            "requested_projection_max_abs_component_error": float(
                self.requested_projection_max_abs_component_error
            ),
            "effective_projection_max_abs_component_error": float(
                self.effective_projection_max_abs_component_error
            ),
            "requested_projection_meets_target": bool(self.requested_projection_meets_target),
            "effective_projection_meets_target": bool(self.effective_projection_meets_target),
            "auto_bump_mode_deck_to_recommended": bool(self.auto_bump_mode_deck_to_recommended),
            "mode_deck_auto_bumped_to_recommended": bool(self.mode_deck_auto_bumped_to_recommended),
            "recommendation_status": self.recommendation_status,
            "recommended_mpol": int(self.recommended_mpol),
            "recommended_ntor": int(self.recommended_ntor),
            "recommended_nzeta": int(self.recommended_nzeta),
        }


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


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _compact_json_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict) and not value:
        return None
    if isinstance(value, (list, tuple)) and not value:
        return None
    return json.dumps(_json_sanitize(value), sort_keys=True, separators=(",", ":"))


def _bool_env(value: Any) -> str:
    return "1" if bool(value) else "0"


def _solver_env_overrides(config: ExampleConfig) -> dict[str, str]:
    """Return solver environment flags used by this strict free-boundary example."""

    return {
        "VMEC_JAX_RETURN_BEST_SCORED_STATE": _bool_env(config.return_best_scored_state),
        "VMEC_JAX_FREEB_DRIFT_RESTART": _bool_env(config.free_boundary_drift_restart),
        "VMEC_JAX_FREEB_DRIFT_RESTART_FACTOR": f"{float(config.free_boundary_drift_restart_factor):.17g}",
        "VMEC_JAX_FREEB_DRIFT_RESTART_STEP_FACTOR": (
            f"{float(config.free_boundary_drift_restart_step_factor):.17g}"
        ),
        "VMEC_JAX_FREEB_DRIFT_RESTART_MIN_ITER_SINCE_BEST": str(
            max(0, int(config.free_boundary_drift_restart_min_iter_since_best))
        ),
        "VMEC_JAX_FREEB_DRIFT_RESTART_STREAK": str(max(1, int(config.free_boundary_drift_restart_streak))),
        "VMEC_JAX_FREEB_DRIFT_RESTART_MAX_RESTARTS": str(
            max(0, int(config.free_boundary_drift_restart_max_restarts))
        ),
    }


def _edge_control_row_metrics(edge_projection: dict[str, Any]) -> dict[str, Any]:
    """Return flat CSV-safe reduced edge-control diagnostics."""

    state_residual = _dict_or_empty(edge_projection.get("state_residual"))
    state_coordinates = _dict_or_empty(edge_projection.get("state_coordinates"))
    reduced_unknown = _dict_or_empty(edge_projection.get("reduced_unknown_vector"))
    update_direction = _dict_or_empty(edge_projection.get("update_direction"))
    reduced_update = _dict_or_empty(edge_projection.get("reduced_update_direction"))
    force_direction = _dict_or_empty(edge_projection.get("force_direction")) or update_direction
    reduced_force = _dict_or_empty(edge_projection.get("reduced_force_direction")) or reduced_update
    native_last_step = _dict_or_empty(edge_projection.get("native_last_step"))
    return {
        "free_boundary_edge_control_projection_native_force_l2": _finite_float_or_none(
            native_last_step.get("control_force_l2")
        ),
        "free_boundary_edge_control_projection_native_velocity_l2": _finite_float_or_none(
            native_last_step.get("control_velocity_l2")
        ),
        "free_boundary_edge_control_projection_native_update_l2": _finite_float_or_none(
            native_last_step.get("control_update_l2")
        ),
        "free_boundary_edge_control_projection_native_trust_scale": _finite_float_or_none(
            native_last_step.get("trust_scale")
        ),
        "free_boundary_edge_control_projection_state_residual_status": state_residual.get("status"),
        "free_boundary_edge_control_projection_state_residual_linf": _finite_float_or_none(
            state_residual.get("residual_linf")
        ),
        "free_boundary_edge_control_projection_state_residual_rms": _finite_float_or_none(
            state_residual.get("residual_rms")
        ),
        "free_boundary_edge_control_projection_state_residual_rel": _finite_float_or_none(
            state_residual.get("residual_rel")
        ),
        "free_boundary_edge_control_projection_state_coordinate_linf": _finite_float_or_none(
            state_coordinates.get("coordinate_linf")
        ),
        "free_boundary_edge_control_projection_state_coordinate_l2": _finite_float_or_none(
            state_coordinates.get("coordinate_l2")
        ),
        "free_boundary_edge_control_projection_state_coordinate_by_label": _compact_json_or_none(
            state_coordinates.get("coordinate_by_label")
        ),
        "free_boundary_edge_control_projection_state_reconstruction_residual_linf": _finite_float_or_none(
            state_coordinates.get("reconstruction_residual_linf")
        ),
        "free_boundary_edge_control_projection_state_reconstruction_residual_rms": _finite_float_or_none(
            state_coordinates.get("reconstruction_residual_rms")
        ),
        "free_boundary_edge_control_projection_state_reconstruction_residual_rel": _finite_float_or_none(
            state_coordinates.get("reconstruction_residual_rel")
        ),
        "free_boundary_edge_control_projection_reduced_unknown_status": reduced_unknown.get("status"),
        "free_boundary_edge_control_projection_reduced_unknown_size": reduced_unknown.get(
            "reduced_unknown_size"
        ),
        "free_boundary_edge_control_projection_full_edge_size": reduced_unknown.get("full_edge_size"),
        "free_boundary_edge_control_projection_unknown_reduction_fraction": _finite_float_or_none(
            reduced_unknown.get("reduction_fraction")
        ),
        "free_boundary_edge_control_projection_unknown_decoded_residual_linf": _finite_float_or_none(
            reduced_unknown.get("decoded_residual_linf")
        ),
        "free_boundary_edge_control_projection_unknown_decoded_residual_rel": _finite_float_or_none(
            reduced_unknown.get("decoded_residual_rel")
        ),
        "free_boundary_edge_control_projection_update_direction_linf": _finite_float_or_none(
            update_direction.get("residual_linf")
        ),
        "free_boundary_edge_control_projection_update_direction_rms": _finite_float_or_none(
            update_direction.get("residual_rms")
        ),
        "free_boundary_edge_control_projection_update_direction_rel": _finite_float_or_none(
            update_direction.get("residual_rel")
        ),
        "free_boundary_edge_control_projection_force_direction_linf": _finite_float_or_none(
            force_direction.get("residual_linf")
        ),
        "free_boundary_edge_control_projection_force_direction_rms": _finite_float_or_none(
            force_direction.get("residual_rms")
        ),
        "free_boundary_edge_control_projection_force_direction_rel": _finite_float_or_none(
            force_direction.get("residual_rel")
        ),
        "free_boundary_edge_control_projection_force_direction_captured_fraction": _finite_float_or_none(
            force_direction.get("captured_fraction")
        ),
        "free_boundary_edge_control_projection_reduced_update_status": reduced_update.get("status"),
        "free_boundary_edge_control_projection_reduced_update_size": reduced_update.get(
            "reduced_update_size"
        ),
        "free_boundary_edge_control_projection_full_update_size": reduced_update.get("full_update_size"),
        "free_boundary_edge_control_projection_reduced_update_linf": _finite_float_or_none(
            reduced_update.get("update_linf")
        ),
        "free_boundary_edge_control_projection_reduced_update_by_label": _compact_json_or_none(
            reduced_update.get("update_by_label")
        ),
        "free_boundary_edge_control_projection_reduced_update_decoded_residual_linf": _finite_float_or_none(
            reduced_update.get("decoded_residual_linf")
        ),
        "free_boundary_edge_control_projection_reduced_update_decoded_residual_rel": _finite_float_or_none(
            reduced_update.get("decoded_residual_rel")
        ),
        "free_boundary_edge_control_projection_reduced_update_captured_fraction": _finite_float_or_none(
            reduced_update.get("captured_fraction")
        ),
        "free_boundary_edge_control_projection_reduced_force_status": reduced_force.get("status"),
        "free_boundary_edge_control_projection_reduced_force_size": reduced_force.get(
            "reduced_update_size"
        ),
        "free_boundary_edge_control_projection_reduced_force_linf": _finite_float_or_none(
            reduced_force.get("update_linf")
        ),
        "free_boundary_edge_control_projection_reduced_force_decoded_residual_linf": _finite_float_or_none(
            reduced_force.get("decoded_residual_linf")
        ),
        "free_boundary_edge_control_projection_reduced_force_decoded_residual_rel": _finite_float_or_none(
            reduced_force.get("decoded_residual_rel")
        ),
        "free_boundary_edge_control_projection_reduced_force_captured_fraction": _finite_float_or_none(
            reduced_force.get("captured_fraction")
        ),
    }


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


def _square_axis_sample_kwargs(config: ExampleConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "axis_half_width": float(config.plasma_axis_half_width),
        "axis_kind": str(config.plasma_axis_kind),
        "axis_square_power": float(config.plasma_axis_square_power),
        "axis_spline_corner_radius_factor": float(config.plasma_axis_spline_corner_radius_factor),
        "minor_radius": float(config.plasma_minor_radius),
        "side_elongation": float(config.side_elongation),
        "side_minor_modulation": float(config.side_minor_modulation),
        "side_power": float(config.side_power),
        "corner_power": float(config.corner_power),
        "corner_ellipticity": float(config.corner_ellipticity),
        "corner_amplitude": float(config.corner_amplitude),
        "corner_rotation": float(config.corner_rotation),
        "corner_helicity": int(config.corner_helicity),
    }
    controls = _resolved_axis_spline_controls(config)
    if controls is not None:
        kwargs["axis_spline_controls"] = controls
    return kwargs


def _resolved_axis_spline_controls(config: ExampleConfig) -> SquareAxisSplineControls | None:
    """Return explicit spline controls, optionally from reduced radii."""

    reduced = config.plasma_axis_reduced_radii
    explicit = config.plasma_axis_spline_controls
    axis_kind = str(config.plasma_axis_kind).strip().lower()
    if reduced is not None and axis_kind != "control_spline":
        raise ValueError("plasma_axis_reduced_radii requires plasma_axis_kind='control_spline'")
    if reduced is not None and explicit is not None:
        raise ValueError("set either plasma_axis_spline_controls or plasma_axis_reduced_radii, not both")
    if explicit is not None:
        controls = explicit.validate()
    elif reduced is not None or axis_kind == "control_spline":
        controls = SquareAxisSplineControls.rounded_square(
            axis_half_width=float(config.plasma_axis_half_width),
            corner_radius_factor=float(config.plasma_axis_spline_corner_radius_factor),
            control_count=int(config.plasma_axis_spline_control_count),
        )
    else:
        return None
    if reduced is None:
        return controls.validate()
    basis = square_axis_spline_symmetric_control_basis(
        controls,
        symmetry=str(config.plasma_axis_control_symmetry),
    )
    return basis.controls_from_reduced(np.asarray(reduced, dtype=float))


def _spline_controls_payload(controls: SquareAxisSplineControls | None) -> dict[str, Any] | None:
    if controls is None:
        return None
    validated = controls.validate()
    return {
        "zeta": [float(value) for value in validated.zeta],
        "radius": [float(value) for value in validated.radius],
    }


def _boundary_fit_grid_for_modes(*, mpol: int, ntor: int) -> dict[str, int]:
    return {
        "ntheta_fit": max(64, 4 * int(mpol)),
        "nzeta_fit": max(128, 8 * int(ntor)),
    }


def _boundary_fit_grid(config: ExampleConfig) -> dict[str, int]:
    return _boundary_fit_grid_for_modes(mpol=int(config.mpol), ntor=int(config.ntor))


def _resolved_nzeta(config: ExampleConfig) -> int:
    """Return the VMEC toroidal grid count for the current mode deck."""

    recommended = int(recommended_square_axis_nzeta(int(config.ntor)))
    default_nzeta = int(max(64, recommended))
    if config.nzeta is None:
        return default_nzeta
    requested = int(config.nzeta)
    if requested <= 0:
        return default_nzeta
    if bool(config.enforce_recommended_nzeta) and bool(config.auto_bump_nzeta_to_recommended):
        return int(max(requested, recommended))
    return requested


def _resolved_ntheta(config: ExampleConfig) -> int:
    """Return the VMEC poloidal grid count for the current mode deck."""

    recommended = int(recommended_square_axis_ntheta(int(config.mpol)))
    if config.ntheta is None:
        return recommended
    requested = int(config.ntheta)
    if requested <= 0:
        return recommended
    if config.max_boundary_projection_error is not None:
        return int(max(requested, recommended))
    return requested


def _effective_square_axis_resolution(config: ExampleConfig) -> EffectiveSquareAxisResolution:
    """Return the resolved VMEC grids and the reason for each automatic choice."""

    requested_ntheta = None if config.ntheta is None else int(config.ntheta)
    requested_nzeta = None if config.nzeta is None else int(config.nzeta)
    recommended_ntheta = int(recommended_square_axis_ntheta(int(config.mpol)))
    recommended_nzeta = int(recommended_square_axis_nzeta(int(config.ntor)))
    effective_ntheta = int(_resolved_ntheta(config))
    effective_nzeta = int(_resolved_nzeta(config))
    return EffectiveSquareAxisResolution(
        requested_ntheta=requested_ntheta,
        effective_ntheta=effective_ntheta,
        recommended_ntheta=recommended_ntheta,
        ntheta_auto_defaulted=bool(requested_ntheta is None or requested_ntheta <= 0),
        ntheta_auto_bumped_to_recommended=bool(
            requested_ntheta is not None and requested_ntheta > 0 and effective_ntheta > requested_ntheta
        ),
        requested_nzeta=requested_nzeta,
        effective_nzeta=effective_nzeta,
        recommended_nzeta=recommended_nzeta,
        nzeta_auto_defaulted=bool(requested_nzeta is None or requested_nzeta <= 0),
        nzeta_auto_bumped_to_recommended=bool(
            requested_nzeta is not None and requested_nzeta > 0 and effective_nzeta > requested_nzeta
        ),
        enforce_recommended_nzeta=bool(config.enforce_recommended_nzeta),
        auto_bump_nzeta_to_recommended=bool(config.auto_bump_nzeta_to_recommended),
    )


def _nzeta_resolution_payload(config: ExampleConfig) -> dict[str, Any]:
    """Summarize requested and effective toroidal-grid resolution."""

    return _effective_square_axis_resolution(config).nzeta_payload()


def _ntheta_resolution_payload(config: ExampleConfig) -> dict[str, Any]:
    """Summarize requested and effective poloidal-grid resolution."""

    return _effective_square_axis_resolution(config).ntheta_payload()


_MODE_DECK_CACHE: dict[tuple[Any, ...], EffectiveSquareAxisModeDeck] = {}


def _mode_deck_cache_key(config: ExampleConfig) -> tuple[Any, ...]:
    """Return a compact cache key for square-axis projection recommendations."""

    controls_payload = _spline_controls_payload(config.plasma_axis_spline_controls)
    return (
        int(config.nfp),
        int(config.mpol),
        int(config.ntor),
        None if config.max_boundary_projection_error is None else float(config.max_boundary_projection_error),
        bool(config.auto_bump_mode_deck_to_recommended),
        tuple(int(value) for value in config.ns_array),
        tuple(int(value) for value in config.niter_array),
        tuple(float(value) for value in config.ftol_array),
        float(config.phiedge),
        float(config.plasma_axis_half_width),
        str(config.plasma_axis_kind),
        float(config.plasma_axis_square_power),
        float(config.plasma_axis_spline_corner_radius_factor),
        int(config.plasma_axis_spline_control_count),
        None if config.plasma_axis_reduced_radii is None else tuple(float(value) for value in config.plasma_axis_reduced_radii),
        None
        if controls_payload is None
        else (
            tuple(float(value) for value in controls_payload["zeta"]),
            tuple(float(value) for value in controls_payload["radius"]),
        ),
        float(config.plasma_minor_radius),
        float(config.side_elongation),
        float(config.side_minor_modulation),
        float(config.side_power),
        float(config.corner_power),
        float(config.corner_ellipticity),
        float(config.corner_amplitude),
        float(config.corner_rotation),
        int(config.corner_helicity),
    )


def _projection_payload_for_modes(config: ExampleConfig, *, mpol: int, ntor: int) -> dict[str, Any]:
    """Return the square-axis projection error for one explicit mode deck."""

    ns_values, niter_values, ftol_values = _stage_values(config)
    return square_axis_stellarator_mirror_hybrid_projection_error(
        nfp=int(config.nfp),
        mpol=int(mpol),
        ntor=int(ntor),
        **_boundary_fit_grid_for_modes(mpol=int(mpol), ntor=int(ntor)),
        ns_array=ns_values,
        niter_array=niter_values,
        ftol_array=ftol_values,
        phiedge=float(config.phiedge),
        **_square_axis_sample_kwargs(config),
    )


def _effective_square_axis_mode_deck(config: ExampleConfig) -> EffectiveSquareAxisModeDeck:
    """Return the mode deck that should be used for a strict production run."""

    cache_key = _mode_deck_cache_key(config)
    cached = _MODE_DECK_CACHE.get(cache_key)
    if cached is not None:
        return cached

    requested_mpol = int(config.mpol)
    requested_ntor = int(config.ntor)
    target = None if config.max_boundary_projection_error is None else float(config.max_boundary_projection_error)
    requested_projection = _projection_payload_for_modes(
        config,
        mpol=requested_mpol,
        ntor=requested_ntor,
    )
    requested_error = float(requested_projection["max_abs_component_error"])
    requested_meets = bool(target is None or requested_error <= target)
    effective_mpol = requested_mpol
    effective_ntor = requested_ntor
    effective_projection = requested_projection
    recommendation_status: str | None = None
    recommended_mpol = requested_mpol
    recommended_ntor = requested_ntor
    recommended_nzeta = int(recommended_square_axis_nzeta(requested_ntor))

    if target is not None and not requested_meets:
        recommendation = recommend_square_axis_stellarator_mirror_hybrid_resolution(
            target_max_component_error=target,
            mpol=requested_mpol,
            ntor=requested_ntor,
            max_mpol=max(8, requested_mpol + 2),
            max_ntor=max(32, requested_ntor + 8),
            nfp=int(config.nfp),
            ns_array=[int(value) for value in config.ns_array],
            niter_array=[int(value) for value in config.niter_array],
            ftol_array=[float(value) for value in config.ftol_array],
            phiedge=float(config.phiedge),
            **_square_axis_sample_kwargs(config),
        )
        recommendation_status = str(recommendation.get("status"))
        suggested = dict(recommendation["recommended"])
        recommended_mpol = int(suggested["mpol"])
        recommended_ntor = int(suggested["ntor"])
        recommended_nzeta = int(suggested["recommended_nzeta"])
        if bool(config.auto_bump_mode_deck_to_recommended) and recommendation_status == "met":
            effective_mpol = recommended_mpol
            effective_ntor = recommended_ntor
            effective_projection = _projection_payload_for_modes(
                config,
                mpol=effective_mpol,
                ntor=effective_ntor,
            )

    effective_error = float(effective_projection["max_abs_component_error"])
    effective_meets = bool(target is None or effective_error <= target)
    deck = EffectiveSquareAxisModeDeck(
        requested_mpol=requested_mpol,
        requested_ntor=requested_ntor,
        effective_mpol=effective_mpol,
        effective_ntor=effective_ntor,
        target_max_component_error=target,
        requested_projection_max_abs_component_error=requested_error,
        effective_projection_max_abs_component_error=effective_error,
        requested_projection_meets_target=requested_meets,
        effective_projection_meets_target=effective_meets,
        auto_bump_mode_deck_to_recommended=bool(config.auto_bump_mode_deck_to_recommended),
        mode_deck_auto_bumped_to_recommended=bool(
            effective_mpol != requested_mpol or effective_ntor != requested_ntor
        ),
        recommendation_status=recommendation_status,
        recommended_mpol=recommended_mpol,
        recommended_ntor=recommended_ntor,
        recommended_nzeta=recommended_nzeta,
    )
    if len(_MODE_DECK_CACHE) > 64:
        _MODE_DECK_CACHE.clear()
    _MODE_DECK_CACHE[cache_key] = deck
    return deck


def _effective_solve_config(config: ExampleConfig) -> ExampleConfig:
    """Return a config with production mode-deck auto-promotion applied."""

    deck = _effective_square_axis_mode_deck(config)
    if not bool(deck.mode_deck_auto_bumped_to_recommended):
        return config
    return replace(config, mpol=int(deck.effective_mpol), ntor=int(deck.effective_ntor))


def _run_budget(config: ExampleConfig, *, restart_state: Any | None) -> int:
    if bool(config.use_multigrid_schedule) and restart_state is None:
        return int(sum(int(value) for value in config.niter_array))
    return int(config.max_iter)


def _validate_example_config(config: ExampleConfig) -> None:
    if int(config.mpol) < 3:
        raise ValueError("mpol must be at least 3 so the square-hybrid corner shaping fits")
    if int(config.ntor) < 4:
        raise ValueError("ntor must be at least 4 so the square-like axis fits")
    if int(config.nstep) < 1:
        raise ValueError("nstep must be at least 1")
    if config.solver_mode is not None:
        solver_mode = str(config.solver_mode).strip().lower()
        if solver_mode not in {"default", "parity", "accelerated"}:
            raise ValueError("solver_mode must be one of: default, parity, accelerated, or None")
    if not np.isfinite(float(config.free_boundary_drift_restart_factor)) or float(
        config.free_boundary_drift_restart_factor
    ) < 1.0:
        raise ValueError("free_boundary_drift_restart_factor must be finite and at least 1")
    if not (0.0 < float(config.free_boundary_drift_restart_step_factor) <= 1.0):
        raise ValueError("free_boundary_drift_restart_step_factor must be in (0, 1]")
    if int(config.free_boundary_drift_restart_min_iter_since_best) < 0:
        raise ValueError("free_boundary_drift_restart_min_iter_since_best must be nonnegative")
    if int(config.free_boundary_drift_restart_streak) < 1:
        raise ValueError("free_boundary_drift_restart_streak must be positive")
    if int(config.free_boundary_drift_restart_max_restarts) < 0:
        raise ValueError("free_boundary_drift_restart_max_restarts must be nonnegative")
    if int(config.nvacskip) < 1:
        raise ValueError("nvacskip must be at least 1")
    if config.plasma_axis_reduced_radii is not None and str(config.plasma_axis_kind).strip().lower() != "control_spline":
        raise ValueError("plasma_axis_reduced_radii requires plasma_axis_kind='control_spline'")
    mode_deck = _effective_square_axis_mode_deck(config)
    solve_config = (
        config
        if not bool(mode_deck.mode_deck_auto_bumped_to_recommended)
        else replace(config, mpol=int(mode_deck.effective_mpol), ntor=int(mode_deck.effective_ntor))
    )
    ntheta = _resolved_ntheta(solve_config)
    nzeta = _resolved_nzeta(solve_config)
    if ntheta < 8:
        raise ValueError("ntheta must be at least 8")
    if nzeta < 8:
        raise ValueError("nzeta must be at least 8")
    if config.max_boundary_projection_error is not None:
        limit = float(config.max_boundary_projection_error)
        if not np.isfinite(limit) or limit <= 0.0:
            raise ValueError("max_boundary_projection_error must be positive, finite, or None")
        schedule = _strict_schedule_payload(config)
        if not bool(schedule["requested_final_ftol_meets_target"]):
            raise ValueError(
                "square-hybrid production solves require a final component-wise FTOL of 1e-12 or tighter: "
                f"requested_final_ftol={schedule['requested_final_ftol']!r}. "
                "Use FTOL_ARRAY ending at 1e-12, or set max_boundary_projection_error=None for a "
                "diagnostic-only run."
            )
        if not bool(mode_deck.effective_projection_meets_target):
            raise ValueError(
                "square-hybrid boundary projection error is too large for a production solve: "
                f"max_abs_component_error={mode_deck.effective_projection_max_abs_component_error:.3e} "
                f"exceeds {limit:.3e} for MPOL={int(solve_config.mpol)}, "
                f"NTOR={int(solve_config.ntor)}, NZETA={nzeta}. "
                "Suggested finite Fourier closure for the current spline-smoothed target: "
                f"MPOL={mode_deck.recommended_mpol}, NTOR={mode_deck.recommended_ntor}, "
                f"NZETA>={mode_deck.recommended_nzeta}. Increase MPOL/NTOR/NZETA, keep "
                "auto_bump_mode_deck_to_recommended=True when a feasible recommendation exists, "
                "keep plasma_axis_kind='control_spline' or 'spline', or set "
                "max_boundary_projection_error=None for a diagnostic-only run."
            )
    if bool(solve_config.enforce_recommended_nzeta):
        recommended = recommended_square_axis_nzeta(int(solve_config.ntor))
        if nzeta < recommended:
            bump_note = (
                f" (requested NTOR={int(mode_deck.requested_ntor)}; mode deck auto-bumped to "
                f"NTOR={int(mode_deck.effective_ntor)})"
                if bool(mode_deck.mode_deck_auto_bumped_to_recommended)
                else ""
            )
            raise ValueError(
                f"NZETA={nzeta} is underresolved for effective NTOR={int(solve_config.ntor)}{bump_note}; "
                f"use at least {recommended}, keep auto_bump_nzeta_to_recommended=True, or set "
                "enforce_recommended_nzeta=False for a diagnostic-only run"
            )


def make_free_boundary_indata(config: ExampleConfig, *, beta_percent: float) -> InData:
    """Return the free-boundary input deck for one beta case."""

    solve_config = _effective_solve_config(config)
    ns_values, niter_values, ftol_values = _stage_values(solve_config)
    resolution = _effective_square_axis_resolution(solve_config)
    indata = square_axis_stellarator_mirror_hybrid_indata(
        nfp=int(solve_config.nfp),
        mpol=int(solve_config.mpol),
        ntor=int(solve_config.ntor),
        **_boundary_fit_grid(solve_config),
        ns_array=ns_values,
        niter_array=niter_values,
        ftol_array=ftol_values,
        phiedge=float(solve_config.phiedge),
        **_square_axis_sample_kwargs(solve_config),
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
            "NZETA": resolution.effective_nzeta,
            "NTHETA": resolution.effective_ntheta,
            "NSTEP": int(config.nstep),
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

    mode_deck = _effective_square_axis_mode_deck(config)
    projection = _projection_payload_for_modes(
        config,
        mpol=int(mode_deck.effective_mpol),
        ntor=int(mode_deck.effective_ntor),
    )
    projection.update(
        {
            "requested_mpol": int(mode_deck.requested_mpol),
            "requested_ntor": int(mode_deck.requested_ntor),
            "mpol": int(mode_deck.effective_mpol),
            "ntor": int(mode_deck.effective_ntor),
            "mode_deck": mode_deck.to_dict(),
        }
    )
    return projection


def _resolution_deck_payload(config: ExampleConfig) -> dict[str, Any]:
    """Return the cheap representation/grid gate for edited mode settings."""

    solve_config = _effective_solve_config(config)
    resolution = _effective_square_axis_resolution(solve_config)
    return square_axis_resolution_deck_status(
        projection=_boundary_projection_payload(config),
        mpol=int(solve_config.mpol),
        ntor=int(solve_config.ntor),
        ns=int(solve_config.ns),
        ntheta=resolution.effective_ntheta,
        nzeta=resolution.effective_nzeta,
        target_max_component_error=solve_config.max_boundary_projection_error,
    )


def _strict_schedule_payload(config: ExampleConfig) -> dict[str, Any]:
    """Summarize whether the requested schedule is strict enough for claims."""

    ns_values, niter_values, ftol_values = _stage_values(config)
    status = square_axis_strict_schedule_status(
        ns_array=ns_values,
        niter_array=niter_values,
        ftol_array=ftol_values,
        target_ftol=STRICT_COMPONENT_FTOL_TARGET,
    )
    return {
        **status,
        "claim_requires_converged_strict": True,
        "claim_requires_fresh_final_residual": True,
        "claim_requires_virtual_casing_for_finite_beta": bool(any(float(beta) != 0.0 for beta in config.betas_percent)),
    }


def _control_fourier_map_payload(config: ExampleConfig) -> dict[str, Any]:
    """Return spline-control to Fourier-map diagnostics for this deck."""

    solve_config = _effective_solve_config(config)
    controls = _resolved_axis_spline_controls(solve_config)
    if controls is None:
        return {
            "status": "not_applicable_for_axis_kind",
            "axis_kind": str(solve_config.plasma_axis_kind),
        }
    sample_kwargs = _square_axis_sample_kwargs(solve_config)
    sample_kwargs.pop("axis_kind", None)
    sample_kwargs.pop("axis_spline_controls", None)
    payload: dict[str, Any] = {
        "status": "available",
        "axis_kind": str(solve_config.plasma_axis_kind),
    }
    for symmetry in ("square", "stellarator"):
        try:
            status = square_axis_spline_control_fourier_map_status(
                controls=controls,
                symmetry=symmetry,
                nfp=int(solve_config.nfp),
                mpol=int(solve_config.mpol),
                ntor=int(solve_config.ntor),
                **_boundary_fit_grid(solve_config),
                **sample_kwargs,
            )
        except Exception as exc:
            payload[symmetry] = {
                "status": f"failed:{type(exc).__name__}",
                "error": repr(exc),
            }
            continue
        payload[symmetry] = {
            "status": status.get("status"),
            "labels": status.get("labels"),
            "control_count": status.get("control_count"),
            "mode_count": status.get("mode_count"),
            "condition_number": status.get("condition_number"),
            "jacobian_shape": status.get("jacobian_shape"),
        }
    return payload


def _edge_control_projection_payload(config: ExampleConfig) -> dict[str, Any] | None:
    """Return the reduced edge-control payload passed to the solver."""

    solve_config = _effective_solve_config(config)
    symmetry = str(solve_config.free_boundary_edge_control_projection).strip().lower()
    if symmetry in {"", "none", "off", "false"}:
        return None
    if symmetry not in {"square", "stellarator"}:
        raise ValueError("free_boundary_edge_control_projection must be 'square', 'stellarator', or 'none'")
    controls = _resolved_axis_spline_controls(solve_config)
    if controls is None:
        raise ValueError("free_boundary_edge_control_projection requires plasma_axis_kind='control_spline'")
    rcond = float(solve_config.free_boundary_edge_control_rcond)
    if not np.isfinite(rcond) or rcond <= 0.0:
        raise ValueError("free_boundary_edge_control_rcond must be positive and finite")
    ridge = float(solve_config.free_boundary_edge_control_ridge)
    if not np.isfinite(ridge) or ridge < 0.0:
        raise ValueError("free_boundary_edge_control_ridge must be finite and nonnegative")
    trust_radius = solve_config.free_boundary_edge_control_trust_radius
    if trust_radius is not None and (not np.isfinite(float(trust_radius)) or float(trust_radius) <= 0.0):
        raise ValueError("free_boundary_edge_control_trust_radius must be positive and finite when supplied")
    update_mode = str(solve_config.free_boundary_edge_control_update_mode).strip().lower()
    if update_mode not in {"projected_delta", "coordinate", "native_coordinate"}:
        raise ValueError(
            "free_boundary_edge_control_update_mode must be 'projected_delta', "
            "'coordinate', or 'native_coordinate'"
        )
    sample_kwargs = {
        key: value
        for key, value in _square_axis_sample_kwargs(solve_config).items()
        if key not in {"axis_kind", "axis_spline_controls"}
    }
    payload = square_axis_free_boundary_edge_control_projection_payload(
        controls=controls,
        symmetry=symmetry,
        rcond=rcond,
        ridge=ridge,
        trust_radius=None if trust_radius is None else float(trust_radius),
        source="toroidal_stellarator_mirror_hybrid_square_coils_free_boundary",
        nfp=int(solve_config.nfp),
        mpol=int(solve_config.mpol),
        ntor=int(solve_config.ntor),
        **_boundary_fit_grid(solve_config),
        **sample_kwargs,
    )
    payload["update_mode"] = update_mode
    return payload


def _edge_control_projection_summary(config: ExampleConfig) -> dict[str, Any]:
    """Return JSON-safe metadata for the requested edge-control projection."""

    requested = str(config.free_boundary_edge_control_projection).strip().lower()
    payload = _edge_control_projection_payload(config)
    if payload is None:
        return {
            "requested": requested,
            "enabled": False,
            "status": "disabled",
        }
    jacobian = np.asarray(payload.get("control_jacobian"), dtype=float)
    return {
        "requested": requested,
        "enabled": bool(payload.get("enabled", False)),
        "status": "enabled" if bool(payload.get("enabled", False)) else "disabled",
        "basis_symmetry": payload.get("basis_symmetry"),
        "labels": list(payload.get("labels", [])),
        "control_count": int(jacobian.shape[1]) if jacobian.ndim == 2 else None,
        "mode_count": int(payload.get("mode_count", 0)),
        "jacobian_shape": [int(value) for value in jacobian.shape],
        "rcond": float(payload.get("rcond", config.free_boundary_edge_control_rcond)),
        "ridge": float(payload.get("ridge", config.free_boundary_edge_control_ridge)),
        "trust_radius": payload.get("trust_radius", config.free_boundary_edge_control_trust_radius),
        "update_mode": str(payload.get("update_mode", config.free_boundary_edge_control_update_mode)),
        "rank": payload.get("rank"),
        "rank_deficient": payload.get("rank_deficient"),
        "condition_number": payload.get("condition_number"),
        "gram_condition_number": payload.get("gram_condition_number"),
        "max_offdiag_column_correlation": payload.get("max_offdiag_column_correlation"),
        "native_reduced_solver_ready": payload.get("native_reduced_solver_ready"),
    }


def _spline_bridge_payload(config: ExampleConfig, *, resolution_deck: dict[str, Any]) -> dict[str, Any]:
    """State what the spline controls can and cannot do in the current solver."""

    controls = _resolved_axis_spline_controls(config)
    uses_controls = controls is not None
    edge_control = _edge_control_projection_summary(config)
    edge_enabled = bool(edge_control.get("enabled"))
    edge_update_mode = str(config.free_boundary_edge_control_update_mode).strip().lower()
    native_edge_controls = bool(uses_controls and edge_enabled and edge_update_mode == "native_coordinate")
    return {
        "real_space_axis_basis": "periodic_spline_controls" if uses_controls else "sampled_fourier_target",
        "nonlinear_solver_boundary_basis": (
            "reduced_spline_edge_controls_with_vmec_fourier_decode"
            if native_edge_controls
            else "vmec_fourier_coefficients"
        ),
        "solver_native_spline_controls": native_edge_controls,
        "solver_native_spline_scope": "lcfs_edge_only" if native_edge_controls else None,
        "solver_edge_control_projection_enabled": edge_enabled,
        "solver_edge_control_update_mode": edge_update_mode,
        "solver_edge_control_projection": edge_control,
        "requires_fourier_projection": True,
        "can_reduce_input_shape_dofs": bool(uses_controls),
        "can_project_free_boundary_edge_updates": edge_enabled,
        "can_reduce_free_boundary_edge_dofs": edge_enabled,
        "can_reduce_nonlinear_solver_dofs": native_edge_controls,
        "requires_native_spline_state_for_reduced_nonlinear_dofs": bool(uses_controls and not native_edge_controls),
        "recommended_next_action": (
            "profile_native_spline_edge_control_strict_convergence"
            if native_edge_controls and resolution_deck.get("status") == "production_ready"
            else "profile_projected_edge_control_strict_convergence"
            if edge_enabled and resolution_deck.get("status") == "production_ready"
            else "repair_projection_or_zeta_deck_before_solver_profiling"
        ),
        "interpretation": (
            "The control-spline path smooths and reduces the input target before projection. "
            "With native-coordinate edge updates, the LCFS edge is advanced in reduced spline "
            "coordinates and decoded to VMEC Fourier coefficients for force evaluation; the "
            "interior VMEC state remains on the existing Fourier/radial basis."
        ),
    }


def _preflight_payload(config: ExampleConfig) -> dict[str, Any]:
    """Return cheap checks that should pass before a long square-coil solve."""

    mode_deck = _effective_square_axis_mode_deck(config)
    solve_config = _effective_solve_config(config)
    resolution = _effective_square_axis_resolution(solve_config)
    projection = _boundary_projection_payload(config)
    resolution_deck = square_axis_resolution_deck_status(
        projection=projection,
        mpol=int(solve_config.mpol),
        ntor=int(solve_config.ntor),
        ns=int(solve_config.ns),
        ntheta=resolution.effective_ntheta,
        nzeta=resolution.effective_nzeta,
        target_max_component_error=solve_config.max_boundary_projection_error,
    )
    schedule = _strict_schedule_payload(config)
    production_ready = bool(
        resolution_deck.get("status") == "production_ready"
        and schedule.get("requested_final_ftol_meets_target")
    )
    edge_control = _edge_control_projection_summary(config)
    spline_bridge = _spline_bridge_payload(config, resolution_deck=resolution_deck)
    convergence_assessment = square_axis_strict_convergence_assessment(
        resolution_deck=resolution_deck,
        strict_schedule=schedule,
        edge_control_projection_enabled=bool(edge_control.get("enabled", False)),
        edge_control_update_mode=str(solve_config.free_boundary_edge_control_update_mode),
        solver_native_spline_controls=bool(spline_bridge.get("solver_native_spline_controls", False)),
        target_ftol=STRICT_COMPONENT_FTOL_TARGET,
    )
    return {
        "schema": "square_coil_hybrid_preflight",
        "schema_version": 1,
        "status": "production_ready" if production_ready else "diagnostic_only",
        "production_ready_for_strict_profile": production_ready,
        "configuration": {
            "requested_mpol": int(mode_deck.requested_mpol),
            "requested_ntor": int(mode_deck.requested_ntor),
            "mpol": int(mode_deck.effective_mpol),
            "ntor": int(mode_deck.effective_ntor),
            "mode_deck_auto_bumped_to_recommended": bool(mode_deck.mode_deck_auto_bumped_to_recommended),
            "ntheta": resolution.effective_ntheta,
            "requested_ntheta": resolution.requested_ntheta,
            "recommended_ntheta": resolution.recommended_ntheta,
            "nzeta": resolution.effective_nzeta,
            "requested_nzeta": resolution.requested_nzeta,
            "recommended_nzeta": resolution.recommended_nzeta,
            "axis_kind": str(solve_config.plasma_axis_kind),
            "axis_spline_control_count": int(solve_config.plasma_axis_spline_control_count),
            "side_power": float(solve_config.side_power),
            "corner_power": float(solve_config.corner_power),
            "max_boundary_projection_error": (
                None
                if solve_config.max_boundary_projection_error is None
                else float(solve_config.max_boundary_projection_error)
            ),
            "return_best_scored_state": bool(solve_config.return_best_scored_state),
            "free_boundary_drift_restart": bool(solve_config.free_boundary_drift_restart),
            "free_boundary_drift_restart_factor": float(solve_config.free_boundary_drift_restart_factor),
            "free_boundary_drift_restart_step_factor": float(
                solve_config.free_boundary_drift_restart_step_factor
            ),
            "free_boundary_drift_restart_min_iter_since_best": int(
                solve_config.free_boundary_drift_restart_min_iter_since_best
            ),
            "free_boundary_drift_restart_streak": int(solve_config.free_boundary_drift_restart_streak),
            "free_boundary_drift_restart_max_restarts": int(
                solve_config.free_boundary_drift_restart_max_restarts
            ),
        },
        "effective_mode_deck": mode_deck.to_dict(),
        "strict_schedule": schedule,
        "strict_convergence_assessment": convergence_assessment,
        "effective_resolution": resolution.to_dict(),
        "ntheta_resolution": resolution.ntheta_payload(),
        "nzeta_resolution": resolution.nzeta_payload(),
        "boundary_projection": projection,
        "resolution_deck": resolution_deck,
        "control_fourier_map": _control_fourier_map_payload(config),
        "edge_control_projection": edge_control,
        "spline_bridge": spline_bridge,
    }


def _write_preflight_report(path: Path, config: ExampleConfig) -> tuple[Path, dict[str, Any]]:
    """Write pre-solve resolution and strict-target diagnostics."""

    payload = _preflight_payload(config)
    path.write_text(json.dumps(_json_sanitize(payload), indent=2, allow_nan=False) + "\n")
    return path, payload


def _solved_surface_and_field(run: Any, config: ExampleConfig) -> tuple[np.ndarray, ...]:
    """Sample the solved boundary and contravariant field on the VMEC grid."""

    sample = sample_solved_boundary_field(run, nfp=int(config.nfp))
    return (
        sample.theta,
        sample.zeta,
        sample.R,
        sample.Z,
        sample.Bmag,
        sample.Bmag_near_axis,
        sample.Bxyz,
        sample.bsupu,
        sample.bsupv,
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


def _virtual_casing_row_metrics(
    *,
    run: Any,
    coils: SquareCoilSet,
    config: ExampleConfig,
) -> dict[str, Any]:
    base = {
        "virtual_casing_quad_factor": int(config.virtual_casing_quad_factor),
        "virtual_casing_chunk_size": config.virtual_casing_chunk_size,
        "virtual_casing_target_chunk_size": config.virtual_casing_target_chunk_size,
        "virtual_casing_grid_adequacy_status": "not_computed",
        "virtual_casing_surface_ntheta": None,
        "virtual_casing_surface_nzeta": None,
        "virtual_casing_quad_ntheta": None,
        "virtual_casing_quad_nzeta": None,
        "virtual_casing_quad_factor_theta": None,
        "virtual_casing_quad_factor_zeta": None,
    }
    try:
        __import__("virtual_casing_jax.functional")
    except ImportError:
        return {
            **base,
            "virtual_casing_status": "skipped_missing_virtual_casing_jax",
            "virtual_casing_external_bnormal_residual_rms": None,
            "virtual_casing_external_bnormal_residual_max": None,
            "virtual_casing_pressure_balance_rms": None,
            "virtual_casing_pressure_balance_max": None,
        }
    try:
        diagnostics = virtual_casing_diagnostics_from_run(
            run,
            coil_params=coils.params,
            nfp=int(config.nfp),
            digits=6,
            quad_factor=int(config.virtual_casing_quad_factor),
            chunk_size=config.virtual_casing_chunk_size,
            target_chunk_size=config.virtual_casing_target_chunk_size,
        )
    except Exception as exc:
        return {
            **base,
            "virtual_casing_status": f"failed:{type(exc).__name__}",
            "virtual_casing_external_bnormal_residual_rms": None,
            "virtual_casing_external_bnormal_residual_max": None,
            "virtual_casing_pressure_balance_rms": None,
            "virtual_casing_pressure_balance_max": None,
        }
    return {
        **base,
        "virtual_casing_status": "computed",
        "virtual_casing_grid_adequacy_status": diagnostics.grid_adequacy_status,
        "virtual_casing_surface_ntheta": diagnostics.surface_ntheta,
        "virtual_casing_surface_nzeta": diagnostics.surface_nphi,
        "virtual_casing_quad_ntheta": diagnostics.quad_ntheta,
        "virtual_casing_quad_nzeta": diagnostics.quad_nphi,
        "virtual_casing_quad_factor_theta": diagnostics.quad_factor_theta,
        "virtual_casing_quad_factor_zeta": diagnostics.quad_factor_phi,
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
    mode_deck = _effective_square_axis_mode_deck(config)
    solve_config = _effective_solve_config(config)
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
    edge_control_projection = _edge_control_projection_payload(solve_config)
    env_overrides = _solver_env_overrides(solve_config)
    previous_env = {name: os.environ.get(name) for name in env_overrides}
    os.environ.update(env_overrides)
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
            use_direct_fallback=bool(config.use_direct_fallback),
            restart_state=restart_state,
            free_boundary_edge_control_projection=edge_control_projection,
            use_initial_guess=False,
        )
    finally:
        for name, previous in previous_env.items():
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous
    wall_s = time.perf_counter() - t0
    write_wout_from_fixed_boundary_run(wout_path, run, include_fsq=True)
    theta, zeta, R, Z, Bmag, Bmag_near_axis, Bxyz, bsupu, bsupv = _solved_surface_and_field(run, solve_config)
    field_lines = _trace_solved_field_lines(R=R, Z=Z, bsupu=bsupu, bsupv=bsupv, Bmag=Bmag, config=solve_config)

    diag = run.result.diagnostics if run.result is not None else {}
    freeb = diag.get("free_boundary", {}) if isinstance(diag, dict) else {}
    edge_projection_diag = freeb.get("edge_control_projection", {}) if isinstance(freeb, dict) else {}
    if not isinstance(edge_projection_diag, dict):
        edge_projection_diag = {}
    edge_state_diag = edge_projection_diag.get("state_residual")
    if not isinstance(edge_state_diag, dict):
        edge_state_diag = {}
    edge_update_diag = edge_projection_diag.get("update_direction")
    if not isinstance(edge_update_diag, dict):
        edge_update_diag = {}
    edge_force_diag = edge_projection_diag.get("force_direction")
    if not isinstance(edge_force_diag, dict):
        edge_force_diag = edge_update_diag
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
    resolution = _effective_square_axis_resolution(solve_config)
    row = {
        "beta_percent": float(beta_percent),
        "input": str(input_path),
        "wout": str(wout_path),
        "wall_s": float(wall_s),
        "requested_mpol": int(mode_deck.requested_mpol),
        "requested_ntor": int(mode_deck.requested_ntor),
        "mpol": int(mode_deck.effective_mpol),
        "ntor": int(mode_deck.effective_ntor),
        "mode_deck_auto_bumped_to_recommended": bool(mode_deck.mode_deck_auto_bumped_to_recommended),
        "ntheta": resolution.effective_ntheta,
        "requested_ntheta": resolution.requested_ntheta,
        "recommended_ntheta": resolution.recommended_ntheta,
        "ntheta_auto_bumped_to_recommended": resolution.ntheta_auto_bumped_to_recommended,
        "nzeta": resolution.effective_nzeta,
        "requested_nzeta": resolution.requested_nzeta,
        "recommended_nzeta": resolution.recommended_nzeta,
        "nzeta_auto_bumped_to_recommended": resolution.nzeta_auto_bumped_to_recommended,
        "n_iter": None if run.result is None else int(getattr(run.result, "n_iter", -1)),
        "run_budget": int(run_budget),
        "used_multigrid_schedule": bool(use_multigrid),
        "solver_mode": diag.get("solver_mode") if isinstance(diag, dict) else config.solver_mode,
        "requested_solver_mode": config.solver_mode,
        "use_scan": diag.get("use_scan") if isinstance(diag, dict) else None,
        "nvacskip": int(config.nvacskip),
        "return_best_scored_state_requested": bool(config.return_best_scored_state),
        "free_boundary_drift_restart_requested": bool(config.free_boundary_drift_restart),
        "free_boundary_drift_restart_factor": float(config.free_boundary_drift_restart_factor),
        "free_boundary_drift_restart_step_factor": float(config.free_boundary_drift_restart_step_factor),
        "free_boundary_drift_restart_min_iter_since_best": int(
            config.free_boundary_drift_restart_min_iter_since_best
        ),
        "free_boundary_drift_restart_streak": int(config.free_boundary_drift_restart_streak),
        "free_boundary_drift_restart_max_restarts": int(config.free_boundary_drift_restart_max_restarts),
        "free_boundary_edge_control_projection_requested": str(config.free_boundary_edge_control_projection),
        "free_boundary_edge_control_update_mode": str(config.free_boundary_edge_control_update_mode),
        "free_boundary_edge_control_ridge": float(config.free_boundary_edge_control_ridge),
        "free_boundary_edge_control_trust_radius": (
            None
            if config.free_boundary_edge_control_trust_radius is None
            else float(config.free_boundary_edge_control_trust_radius)
        ),
        "free_boundary_edge_control_projection_enabled": edge_projection_diag.get("enabled"),
        "free_boundary_edge_control_projection_apply_count": edge_projection_diag.get("apply_count"),
        "free_boundary_edge_control_projection_delta_projection_count": edge_projection_diag.get(
            "delta_projection_count"
        ),
        "free_boundary_edge_control_projection_coordinate_update_count": edge_projection_diag.get(
            "coordinate_update_count"
        ),
        "free_boundary_edge_control_projection_native_coordinate_update_count": edge_projection_diag.get(
            "native_coordinate_update_count"
        ),
        "free_boundary_edge_control_projection_native_velocity_reset_count": edge_projection_diag.get(
            "native_velocity_reset_count"
        ),
        "free_boundary_edge_control_projection_control_count": edge_projection_diag.get("control_count"),
        "free_boundary_edge_control_projection_mode_count": edge_projection_diag.get("mode_count"),
        "free_boundary_edge_control_projection_reason": edge_projection_diag.get("reason"),
        "free_boundary_edge_control_projection_state_captured_fraction": edge_state_diag.get(
            "captured_fraction"
        ),
        "free_boundary_edge_control_projection_update_direction_captured_fraction": edge_update_diag.get(
            "captured_fraction"
        ),
        "free_boundary_edge_control_projection_force_direction_captured_fraction": edge_force_diag.get(
            "captured_fraction"
        ),
        **_edge_control_row_metrics(edge_projection_diag),
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
        "best_scored_component_max": _finite_float_or_none(
            diag.get("best_scored_component_max") if isinstance(diag, dict) else None
        ),
        "best_scored_full_boundary_count": (
            diag.get("best_scored_full_boundary_count") if isinstance(diag, dict) else None
        ),
        "best_scored_fresh_boundary_count": (
            diag.get("best_scored_fresh_boundary_count") if isinstance(diag, dict) else None
        ),
        "best_scored_drift_restart_count": (
            diag.get("best_scored_drift_restart_count") if isinstance(diag, dict) else None
        ),
        "best_scored_drift_streak": diag.get("best_scored_drift_streak") if isinstance(diag, dict) else None,
        "best_scored_drift_last_restart_iter": (
            diag.get("best_scored_drift_last_restart_iter") if isinstance(diag, dict) else None
        ),
        "best_scored_drift_last_ratio": (
            diag.get("best_scored_drift_last_ratio") if isinstance(diag, dict) else None
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
    row.update(_virtual_casing_row_metrics(run=run, coils=coils, config=config))
    promotion = free_boundary_promotion_status(
        beta_percent=float(beta_percent),
        strict_components_met=row.get("converged_strict"),
        final_residual_recomputed=row.get("final_residual_recomputed_on_accepted_state"),
        virtual_casing_status=row.get("virtual_casing_status"),
        virtual_casing_grid_adequacy_status=row.get("virtual_casing_grid_adequacy_status"),
        direct_coil_backend=True,
    )
    row.update(
        {
            "boundary_condition_mode": promotion["boundary_condition_mode"],
            "coil_bnormal_role": promotion["coil_bnormal_role"],
            "production_candidate": promotion["production_candidate"],
            "promotion_blockers": ",".join(str(item) for item in promotion["promotion_blockers"]),
            "virtual_casing_required": promotion["virtual_casing_required"],
            "virtual_casing_grid_adequacy_status": promotion["virtual_casing_grid_adequacy_status"],
            "virtual_casing_available": promotion["virtual_casing_available"],
        }
    )
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
        "requested_mpol",
        "requested_ntor",
        "mpol",
        "ntor",
        "mode_deck_auto_bumped_to_recommended",
        "ntheta",
        "requested_ntheta",
        "recommended_ntheta",
        "ntheta_auto_bumped_to_recommended",
        "nzeta",
        "requested_nzeta",
        "recommended_nzeta",
        "nzeta_auto_bumped_to_recommended",
        "n_iter",
        "run_budget",
        "used_multigrid_schedule",
        "solver_mode",
        "requested_solver_mode",
        "use_scan",
        "nvacskip",
        "return_best_scored_state_requested",
        "free_boundary_drift_restart_requested",
        "free_boundary_drift_restart_factor",
        "free_boundary_drift_restart_step_factor",
        "free_boundary_drift_restart_min_iter_since_best",
        "free_boundary_drift_restart_streak",
        "free_boundary_drift_restart_max_restarts",
        "free_boundary_edge_control_projection_requested",
        "free_boundary_edge_control_update_mode",
        "free_boundary_edge_control_ridge",
        "free_boundary_edge_control_trust_radius",
        "free_boundary_edge_control_projection_enabled",
        "free_boundary_edge_control_projection_apply_count",
        "free_boundary_edge_control_projection_delta_projection_count",
        "free_boundary_edge_control_projection_coordinate_update_count",
        "free_boundary_edge_control_projection_native_coordinate_update_count",
        "free_boundary_edge_control_projection_native_velocity_reset_count",
        "free_boundary_edge_control_projection_native_force_l2",
        "free_boundary_edge_control_projection_native_velocity_l2",
        "free_boundary_edge_control_projection_native_update_l2",
        "free_boundary_edge_control_projection_native_trust_scale",
        "free_boundary_edge_control_projection_control_count",
        "free_boundary_edge_control_projection_mode_count",
        "free_boundary_edge_control_projection_reason",
        "free_boundary_edge_control_projection_state_captured_fraction",
        "free_boundary_edge_control_projection_update_direction_captured_fraction",
        "free_boundary_edge_control_projection_force_direction_captured_fraction",
        "free_boundary_edge_control_projection_state_residual_status",
        "free_boundary_edge_control_projection_state_residual_linf",
        "free_boundary_edge_control_projection_state_residual_rms",
        "free_boundary_edge_control_projection_state_residual_rel",
        "free_boundary_edge_control_projection_state_coordinate_linf",
        "free_boundary_edge_control_projection_state_coordinate_l2",
        "free_boundary_edge_control_projection_state_coordinate_by_label",
        "free_boundary_edge_control_projection_state_reconstruction_residual_linf",
        "free_boundary_edge_control_projection_state_reconstruction_residual_rms",
        "free_boundary_edge_control_projection_state_reconstruction_residual_rel",
        "free_boundary_edge_control_projection_reduced_unknown_status",
        "free_boundary_edge_control_projection_reduced_unknown_size",
        "free_boundary_edge_control_projection_full_edge_size",
        "free_boundary_edge_control_projection_unknown_reduction_fraction",
        "free_boundary_edge_control_projection_unknown_decoded_residual_linf",
        "free_boundary_edge_control_projection_unknown_decoded_residual_rel",
        "free_boundary_edge_control_projection_update_direction_linf",
        "free_boundary_edge_control_projection_update_direction_rms",
        "free_boundary_edge_control_projection_update_direction_rel",
        "free_boundary_edge_control_projection_force_direction_linf",
        "free_boundary_edge_control_projection_force_direction_rms",
        "free_boundary_edge_control_projection_force_direction_rel",
        "free_boundary_edge_control_projection_reduced_update_status",
        "free_boundary_edge_control_projection_reduced_update_size",
        "free_boundary_edge_control_projection_full_update_size",
        "free_boundary_edge_control_projection_reduced_update_linf",
        "free_boundary_edge_control_projection_reduced_update_by_label",
        "free_boundary_edge_control_projection_reduced_update_decoded_residual_linf",
        "free_boundary_edge_control_projection_reduced_update_decoded_residual_rel",
        "free_boundary_edge_control_projection_reduced_update_captured_fraction",
        "free_boundary_edge_control_projection_reduced_force_status",
        "free_boundary_edge_control_projection_reduced_force_size",
        "free_boundary_edge_control_projection_reduced_force_linf",
        "free_boundary_edge_control_projection_reduced_force_decoded_residual_linf",
        "free_boundary_edge_control_projection_reduced_force_decoded_residual_rel",
        "free_boundary_edge_control_projection_reduced_force_captured_fraction",
        "converged",
        "converged_strict",
        "boundary_condition_mode",
        "coil_bnormal_role",
        "production_candidate",
        "promotion_blockers",
        "virtual_casing_required",
        "virtual_casing_available",
        "virtual_casing_grid_adequacy_status",
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
        "best_scored_component_max",
        "best_scored_full_boundary_count",
        "best_scored_fresh_boundary_count",
        "best_scored_drift_restart_count",
        "best_scored_drift_streak",
        "best_scored_drift_last_restart_iter",
        "best_scored_drift_last_ratio",
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
        "virtual_casing_quad_factor",
        "virtual_casing_chunk_size",
        "virtual_casing_target_chunk_size",
        "virtual_casing_surface_ntheta",
        "virtual_casing_surface_nzeta",
        "virtual_casing_quad_ntheta",
        "virtual_casing_quad_nzeta",
        "virtual_casing_quad_factor_theta",
        "virtual_casing_quad_factor_zeta",
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
    preflight_json: Path,
    preflight: dict[str, Any],
    summary_csv: Path,
    rows: list[dict[str, Any]],
    figures: dict[str, str],
    complete: bool,
) -> dict[str, Any]:
    all_converged = bool(complete) and all(bool(row.get("converged")) for row in rows)
    preflight_only = bool(config.preflight_only)
    completed = [float(row.get("beta_percent")) for row in rows]
    remaining = [float(beta) for beta in config.betas_percent if float(beta) not in completed]
    mode_deck = _effective_square_axis_mode_deck(config)
    solve_config = _effective_solve_config(config)
    resolved_controls = _resolved_axis_spline_controls(solve_config)
    resolution = _effective_square_axis_resolution(solve_config)
    reduced_radii = (
        None
        if config.plasma_axis_reduced_radii is None
        else [float(value) for value in config.plasma_axis_reduced_radii]
    )
    return {
        "metrics_schema": SCHEMA,
        "metrics_schema_version": SCHEMA_VERSION,
        "workflow_status": (
            "preflight_only" if preflight_only else "actual_vmec_jax_free_boundary_beta_scan"
        ),
        "free_boundary_solve_status": (
            "not_run_preflight_only"
            if preflight_only
            else "converged"
            if all_converged
            else "not_converged_or_max_iter"
            if complete
            else "partial_running"
        ),
        "hybrid_fixture_kind": "square_axis_toroidal_stellarator_mirror_hybrid",
        "actual_free_boundary_solve": not preflight_only,
        "preflight_only": preflight_only,
        "production_free_boundary_claim": False,
        "betas_percent": [float(beta) for beta in config.betas_percent],
        "completed_betas_percent": completed,
        "remaining_betas_percent": remaining,
        "coil_count": int(coils.centers.shape[0]),
        "n_coils_per_side": int(config.n_coils_per_side),
        "plasma_axis_half_width": float(solve_config.plasma_axis_half_width),
        "plasma_axis_kind": str(solve_config.plasma_axis_kind),
        "plasma_axis_spline_corner_radius_factor": float(solve_config.plasma_axis_spline_corner_radius_factor),
        "plasma_axis_spline_control_count": int(solve_config.plasma_axis_spline_control_count),
        "plasma_axis_control_symmetry": str(solve_config.plasma_axis_control_symmetry),
        "plasma_axis_reduced_radii": reduced_radii,
        "plasma_axis_spline_controls": _spline_controls_payload(resolved_controls),
        "side_power": float(solve_config.side_power),
        "corner_power": float(solve_config.corner_power),
        "coil_square_side_length": float(config.coil_square_side_length),
        "toroidal_current": float(config.toroidal_current),
        "boundary_projection": _boundary_projection_payload(config),
        "resolution_deck": _resolution_deck_payload(config),
        "effective_mode_deck": mode_deck.to_dict(),
        "preflight_json": str(preflight_json),
        "preflight": preflight,
        "delt": None if config.delt is None else float(config.delt),
        "ns": int(solve_config.ns),
        "ns_array": [int(value) for value in solve_config.ns_array],
        "requested_mpol": int(mode_deck.requested_mpol),
        "requested_ntor": int(mode_deck.requested_ntor),
        "mpol": int(mode_deck.effective_mpol),
        "ntor": int(mode_deck.effective_ntor),
        "auto_bump_mode_deck_to_recommended": bool(config.auto_bump_mode_deck_to_recommended),
        "mode_deck_auto_bumped_to_recommended": bool(mode_deck.mode_deck_auto_bumped_to_recommended),
        "recommended_ntheta": resolution.recommended_ntheta,
        "ntheta": resolution.effective_ntheta,
        "requested_ntheta": resolution.requested_ntheta,
        "effective_resolution": resolution.to_dict(),
        "ntheta_resolution": resolution.ntheta_payload(),
        "ntheta_underrecommended": resolution.ntheta_underrecommended,
        "recommended_nzeta": resolution.recommended_nzeta,
        "nzeta": resolution.effective_nzeta,
        "requested_nzeta": resolution.requested_nzeta,
        "nzeta_resolution": resolution.nzeta_payload(),
        "nzeta_underrecommended": resolution.nzeta_underrecommended,
        "max_iter": int(solve_config.max_iter),
        "ftol": float(solve_config.ftol),
        "niter_array": [int(value) for value in solve_config.niter_array],
        "ftol_array": [float(value) for value in solve_config.ftol_array],
        "use_multigrid_schedule": bool(solve_config.use_multigrid_schedule),
        "enforce_recommended_nzeta": bool(solve_config.enforce_recommended_nzeta),
        "auto_bump_nzeta_to_recommended": bool(solve_config.auto_bump_nzeta_to_recommended),
        "max_boundary_projection_error": (
            None
            if solve_config.max_boundary_projection_error is None
            else float(solve_config.max_boundary_projection_error)
        ),
        "nvacskip": int(solve_config.nvacskip),
        "return_best_scored_state": bool(solve_config.return_best_scored_state),
        "free_boundary_drift_restart": bool(solve_config.free_boundary_drift_restart),
        "free_boundary_drift_restart_factor": float(solve_config.free_boundary_drift_restart_factor),
        "free_boundary_drift_restart_step_factor": float(
            solve_config.free_boundary_drift_restart_step_factor
        ),
        "free_boundary_drift_restart_min_iter_since_best": int(
            solve_config.free_boundary_drift_restart_min_iter_since_best
        ),
        "free_boundary_drift_restart_streak": int(solve_config.free_boundary_drift_restart_streak),
        "free_boundary_drift_restart_max_restarts": int(solve_config.free_boundary_drift_restart_max_restarts),
        "free_boundary_activate_fsq": (
            None
            if solve_config.free_boundary_activate_fsq is None
            else float(solve_config.free_boundary_activate_fsq)
        ),
        "free_boundary_edge_control_projection": str(solve_config.free_boundary_edge_control_projection),
        "free_boundary_edge_control_rcond": float(solve_config.free_boundary_edge_control_rcond),
        "free_boundary_edge_control_ridge": float(solve_config.free_boundary_edge_control_ridge),
        "free_boundary_edge_control_trust_radius": (
            None
            if solve_config.free_boundary_edge_control_trust_radius is None
            else float(solve_config.free_boundary_edge_control_trust_radius)
        ),
        "free_boundary_edge_control_update_mode": str(solve_config.free_boundary_edge_control_update_mode),
        "solver_mode": None if solve_config.solver_mode is None else str(solve_config.solver_mode),
        "limit_update_rms": bool(solve_config.limit_update_rms),
        "backtracking": bool(solve_config.backtracking),
        "use_direct_fallback": bool(solve_config.use_direct_fallback),
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
    preflight_json, preflight = _write_preflight_report(outdir / "square_coil_hybrid_preflight.json", config)
    coils = build_square_coils(config)
    coils_json = _write_coils_json(outdir / "square_mirror_hybrid_coils.json", coils, config)
    cases = []
    summary_csv = outdir / "square_coil_hybrid_free_boundary_solve_summary.csv"
    metrics_path = outdir / "square_coil_hybrid_free_boundary_solve_metrics.json"
    if bool(config.preflight_only):
        _write_csv(summary_csv, [])
        metrics = _metrics_payload(
            config=config,
            coils=coils,
            coils_json=coils_json,
            preflight_json=preflight_json,
            preflight=preflight,
            summary_csv=summary_csv,
            rows=[],
            figures={},
            complete=True,
        )
        metrics_path.write_text(json.dumps(_json_sanitize(metrics), indent=2, allow_nan=False) + "\n")
        return metrics_path

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
                preflight_json=preflight_json,
                preflight=preflight,
                summary_csv=summary_csv,
                rows=rows,
                figures={},
                complete=False,
            )
            metrics_path.write_text(json.dumps(_json_sanitize(metrics), indent=2, allow_nan=False) + "\n")
    rows = [case.row for case in cases]
    _write_csv(summary_csv, rows)
    figures = _write_plots(outdir, coils, cases, config) if bool(config.write_plots) and cases else {}
    metrics = _metrics_payload(
        config=config,
        coils=coils,
        coils_json=coils_json,
        preflight_json=preflight_json,
        preflight=preflight,
        summary_csv=summary_csv,
        rows=rows,
        figures=figures,
        complete=True,
    )
    metrics_path.write_text(json.dumps(_json_sanitize(metrics), indent=2, allow_nan=False) + "\n")
    return metrics_path


if __name__ == "__main__":  # pragma: no cover
    print(run_example())
