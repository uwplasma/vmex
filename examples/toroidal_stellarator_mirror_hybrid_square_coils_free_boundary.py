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
from vmec_jax.external_fields import build_coil_field_geometry, ellipse_coil_field_params
from vmec_jax.field import b2_from_bsup, bsup_from_geom, lamscale_from_phips
from vmec_jax.fieldlines import FieldLine, trace_fieldline_on_surface
from vmec_jax.geom import eval_geom
from vmec_jax.namelist import InData, write_indata
from vmec_jax.plotting import fix_matplotlib_3d, prepare_matplotlib_3d
from vmec_jax.profiles import pressure_profile_to_vmec_am, standard_finite_beta_profiles
from vmec_jax.toroidal_hybrid import square_axis_stellarator_mirror_hybrid_indata
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

PLASMA_AXIS_HALF_WIDTH = 0.5 * COIL_SQUARE_SIDE_LENGTH
PLASMA_AXIS_SQUARE_POWER = 3.0
PLASMA_MINOR_RADIUS = 0.03
SIDE_ELONGATION = 0.08
SIDE_MINOR_MODULATION = 0.08
CORNER_ELLIPTICITY = 0.04
CORNER_AMPLITUDE = 0.004
CORNER_ROTATION = 0.30
CORNER_HELICITY = 1

NFP = 1
MPOL = 5
NTOR = 12
NS = 9
NZETA = 32
MAX_ITER = 1000
FTOL = 1.0e-8
PHIEDGE = 0.04
TOROIDAL_CURRENT = 3.0e3
DELT: float | None = None
FREE_BOUNDARY_ACTIVATE_FSQ: float | None = 1.0e-2
LIMIT_UPDATE_RMS = False
BETA_CONTINUATION_RESTART = True
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
    plasma_axis_half_width: float = PLASMA_AXIS_HALF_WIDTH
    plasma_axis_square_power: float = PLASMA_AXIS_SQUARE_POWER
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
    nzeta: int = NZETA
    max_iter: int = MAX_ITER
    ftol: float = FTOL
    phiedge: float = PHIEDGE
    toroidal_current: float = TOROIDAL_CURRENT
    delt: float | None = DELT
    free_boundary_activate_fsq: float | None = FREE_BOUNDARY_ACTIVATE_FSQ
    limit_update_rms: bool = LIMIT_UPDATE_RMS
    beta_continuation_restart: bool = BETA_CONTINUATION_RESTART
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
    bsupu: np.ndarray
    bsupv: np.ndarray
    field_lines: tuple[FieldLine, ...]
    row: dict[str, Any]


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


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
        chunk_size=512,
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


def make_free_boundary_indata(config: ExampleConfig, *, beta_percent: float) -> InData:
    """Return the free-boundary input deck for one beta case."""

    indata = square_axis_stellarator_mirror_hybrid_indata(
        nfp=int(config.nfp),
        mpol=int(config.mpol),
        ntor=int(config.ntor),
        ntheta_fit=max(64, 4 * int(config.mpol)),
        nzeta_fit=max(128, 8 * int(config.ntor)),
        ns_array=int(config.ns),
        niter_array=int(config.max_iter),
        ftol_array=float(config.ftol),
        phiedge=float(config.phiedge),
        axis_half_width=float(config.plasma_axis_half_width),
        axis_square_power=float(config.plasma_axis_square_power),
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
            "NS_ARRAY": [int(config.ns)],
            "NITER_ARRAY": [int(config.max_iter)],
            "FTOL_ARRAY": [float(config.ftol)],
            "NITER": int(config.max_iter),
            "FTOL": float(config.ftol),
            "NZETA": int(config.nzeta),
            "NTHETA": 0,
            "NVACSKIP": max(1, int(config.nzeta)),
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
    theta = np.asarray(run.static.grid.theta, dtype=float)
    zeta = np.asarray(run.static.grid.zeta, dtype=float)
    return (
        theta,
        zeta,
        np.asarray(geom.R[-1], dtype=float),
        np.asarray(geom.Z[-1], dtype=float),
        bmag[-1],
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
    run = run_free_boundary(
        input_path,
        max_iter=int(config.max_iter),
        multigrid=False,
        verbose=False,
        jit_forces=config.jit_forces,
        free_boundary_activate_fsq=(
            None if config.free_boundary_activate_fsq is None else float(config.free_boundary_activate_fsq)
        ),
        external_field_provider_kind="direct_coils",
        external_field_provider_params=coils.params,
        limit_update_rms=bool(config.limit_update_rms),
        restart_state=restart_state,
        use_initial_guess=False,
    )
    wall_s = time.perf_counter() - t0
    write_wout_from_fixed_boundary_run(wout_path, run, include_fsq=True)
    theta, zeta, R, Z, Bmag, bsupu, bsupv = _solved_surface_and_field(run, config)
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
        "converged": None
        if run.result is None
        else bool(diag.get("converged", getattr(run.result, "converged", False))),
        "converged_strict": None if run.result is None else diag.get("converged_strict"),
        "requested_ftol": diag.get("requested_ftol") if isinstance(diag, dict) else None,
        "final_fsq": diag.get("final_fsq") if isinstance(diag, dict) else None,
        "final_fsqr": diag.get("final_fsqr") if isinstance(diag, dict) else None,
        "final_fsqz": diag.get("final_fsqz") if isinstance(diag, dict) else None,
        "final_fsql": diag.get("final_fsql") if isinstance(diag, dict) else None,
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
        "aspect": aspect,
        "mean_iota": mean_iota,
    }
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
        "converged",
        "converged_strict",
        "requested_ftol",
        "final_fsq",
        "final_fsqr",
        "final_fsqz",
        "final_fsql",
        "final_fsq_component_sum",
        "free_boundary_bnormal_rms",
        "free_boundary_bsqvac_rms",
        "free_boundary_gsource_rms",
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
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")
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
            axes[0].semilogy(x, total, marker=marker, linewidth=1.2, label=f"{case.beta_percent:g}%")

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
            axes[1].semilogy(x, bnormal, marker=marker, linewidth=1.2, label=f"{case.beta_percent:g}%")

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
        "convergence_iota": str(_write_convergence_plot(figure_dir, cases)),
    }


def run_example(config: ExampleConfig = ExampleConfig()) -> Path:
    enable_x64(True)
    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    coils = build_square_coils(config)
    coils_json = _write_coils_json(outdir / "square_mirror_hybrid_coils.json", coils, config)
    cases = []
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
    rows = [case.row for case in cases]
    summary_csv = _write_csv(outdir / "square_coil_hybrid_free_boundary_solve_summary.csv", rows)
    figures = _write_plots(outdir, coils, cases, config) if bool(config.write_plots) else {}
    all_converged = all(bool(row.get("converged")) for row in rows)
    metrics = {
        "metrics_schema": SCHEMA,
        "metrics_schema_version": SCHEMA_VERSION,
        "workflow_status": "actual_vmec_jax_free_boundary_beta_scan",
        "free_boundary_solve_status": "converged" if all_converged else "not_converged_or_max_iter",
        "hybrid_fixture_kind": "square_axis_toroidal_stellarator_mirror_hybrid",
        "actual_free_boundary_solve": True,
        "production_free_boundary_claim": False,
        "betas_percent": [float(beta) for beta in config.betas_percent],
        "coil_count": int(coils.centers.shape[0]),
        "n_coils_per_side": int(config.n_coils_per_side),
        "plasma_axis_half_width": float(config.plasma_axis_half_width),
        "coil_square_side_length": float(config.coil_square_side_length),
        "toroidal_current": float(config.toroidal_current),
        "delt": None if config.delt is None else float(config.delt),
        "ns": int(config.ns),
        "mpol": int(config.mpol),
        "ntor": int(config.ntor),
        "max_iter": int(config.max_iter),
        "ftol": float(config.ftol),
        "free_boundary_activate_fsq": (
            None if config.free_boundary_activate_fsq is None else float(config.free_boundary_activate_fsq)
        ),
        "limit_update_rms": bool(config.limit_update_rms),
        "beta_continuation_restart": bool(config.beta_continuation_restart),
        "coils_json": str(coils_json),
        "summary_csv": str(summary_csv),
        "rows": rows,
        "figures": figures,
    }
    metrics_path = outdir / "square_coil_hybrid_free_boundary_solve_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, default=_json_default) + "\n")
    return metrics_path


if __name__ == "__main__":  # pragma: no cover
    print(run_example())
