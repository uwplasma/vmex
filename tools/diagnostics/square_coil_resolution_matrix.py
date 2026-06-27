#!/usr/bin/env python
"""Classify square-coil resolution decks before launching long solves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.toroidal_stellarator_mirror_hybrid_square_coils_free_boundary import (
    ExampleConfig,
    _boundary_fit_grid,
    _effective_solve_config,
    _effective_square_axis_mode_deck,
    _effective_square_axis_resolution,
    _square_axis_sample_kwargs,
)
from vmec_jax.toroidal_hybrid import (
    recommended_square_axis_ntheta,
    recommended_square_axis_nzeta,
    square_axis_resolution_deck_status,
    square_axis_strict_convergence_assessment,
    square_axis_strict_schedule_status,
    square_axis_spline_control_fourier_map_status,
    square_axis_stellarator_mirror_hybrid_projection_error,
)


DEFAULT_DECKS = "5:20:48,5:28:48,5:28:64,6:32:72,7:28:auto,8:32:auto"
DEFAULT_VMEC2000_EXEC = "/home/rjorge/miniforge3/envs/qh-gpu/bin/xvmec"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--decks",
        default=DEFAULT_DECKS,
        help=(
            "Comma-separated MPOL:NTOR[:NZETA[:MGRID_NPHI]] rows. "
            "Use 'auto' for NZETA or MGRID_NPHI."
        ),
    )
    p.add_argument("--format", choices=("markdown", "tsv", "json"), default="markdown")
    p.add_argument("--target-error", default=f"{ExampleConfig().max_boundary_projection_error:.0e}")
    p.add_argument(
        "--ntheta",
        default="auto",
        help="VMEC NTHETA value for every row, or 'auto' for the square-axis recommendation from MPOL.",
    )
    p.add_argument("--ns-array", default="9,13,17")
    p.add_argument("--niter-array", default="4000,8000,24000")
    p.add_argument("--ftol-array", default="1e-8,1e-10,1e-12")
    p.add_argument("--axis-kind", default=ExampleConfig().plasma_axis_kind)
    p.add_argument("--axis-corner-factor", type=float, default=ExampleConfig().plasma_axis_spline_corner_radius_factor)
    p.add_argument("--side-power", type=float, default=ExampleConfig().side_power)
    p.add_argument("--corner-power", type=float, default=ExampleConfig().corner_power)
    p.add_argument("--mgrid-nr", type=int, default=88)
    p.add_argument("--mgrid-nz", type=int, default=64)
    p.add_argument("--mgrid-padding-fraction", type=float, default=1.2)
    p.add_argument("--mgrid-min-padding", type=float, default=0.5)
    p.add_argument("--delt", type=float, default=ExampleConfig().delt)
    p.add_argument("--coil-segments", type=int, default=64)
    p.add_argument("--coil-chunk-size", type=int, default=512)
    p.add_argument("--python", default="python3")
    p.add_argument("--profile-script", default="tools/diagnostics/profile_square_coil_free_boundary.py")
    p.add_argument("--outdir-root", type=Path, default=Path("results"))
    p.add_argument("--vmec2000-exec", default=DEFAULT_VMEC2000_EXEC)
    p.add_argument("--vmec2000-timeout", type=int, default=21600)
    p.add_argument("--print-preflight-commands", action="store_true")
    p.add_argument("--print-scale-commands", action="store_true")
    p.add_argument("--print-vmec2000-commands", action="store_true")
    p.add_argument(
        "--print-jax-commands",
        action="store_true",
        help="Also emit strict direct-GPU vmec_jax commands using reduced edge controls.",
    )
    p.add_argument(
        "--edge-control-projection",
        choices=("none", "square", "stellarator", "full"),
        default="stellarator",
        help="Reduced edge-control basis used by --print-jax-commands and strict readiness diagnostics.",
    )
    p.add_argument(
        "--edge-control-update-mode",
        choices=("projected_delta", "coordinate", "native_coordinate"),
        default="native_coordinate",
        help="Reduced edge-control update mode used by --print-jax-commands and strict readiness diagnostics.",
    )
    p.add_argument("--edge-control-rcond", type=float, default=1.0e-12)
    p.add_argument("--edge-control-ridge", type=float, default=0.0)
    p.add_argument("--edge-control-trust-radius", type=float, default=None)
    p.add_argument(
        "--edge-control-native-force-metric",
        choices=("pullback", "least_squares"),
        default="least_squares",
        help="Native-coordinate reduced edge-force metric used by emitted JAX commands.",
    )
    p.add_argument(
        "--strict-backtracking-accept-ratio",
        type=float,
        default=1.0,
        help="Fresh-merit acceptance ratio used by emitted strict JAX commands.",
    )
    p.add_argument(
        "--include-control-map",
        action="store_true",
        help="Also report reduced spline-control to Fourier-map conditioning.",
    )
    return p


def _parse_optional_float(raw: str) -> float | None:
    key = str(raw).strip().lower()
    if key in {"", "none", "null", "false", "no", "0"}:
        return None
    return float(key)


def _parse_int_list(raw: str) -> tuple[int, ...]:
    values = tuple(int(tok.strip()) for tok in str(raw).replace(";", ",").split(",") if tok.strip())
    if not values:
        raise ValueError("expected at least one integer value")
    return values


def _parse_float_list(raw: str) -> tuple[float, ...]:
    values = tuple(float(tok.strip()) for tok in str(raw).replace(";", ",").split(",") if tok.strip())
    if not values:
        raise ValueError("expected at least one float value")
    return values


def _parse_deck_token(raw: str) -> dict[str, int | None]:
    parts = [part.strip().lower() for part in str(raw).split(":")]
    if len(parts) not in {2, 3, 4}:
        raise ValueError("deck rows must be MPOL:NTOR[:NZETA[:MGRID_NPHI]]")
    mpol = int(parts[0])
    ntor = int(parts[1])

    def _optional_int(part: str | None) -> int | None:
        if part is None or part in {"", "auto", "none", "null"}:
            return None
        return int(part)

    nzeta = _optional_int(parts[2] if len(parts) >= 3 else None)
    mgrid_nphi = _optional_int(parts[3] if len(parts) >= 4 else None)
    return {"mpol": mpol, "ntor": ntor, "nzeta": nzeta, "mgrid_nphi": mgrid_nphi}


def _parse_decks(raw: str) -> list[dict[str, int | None]]:
    decks = [_parse_deck_token(tok) for tok in str(raw).replace(";", ",").split(",") if tok.strip()]
    if not decks:
        raise ValueError("expected at least one deck")
    return decks


def _parse_optional_int(raw: str | int | None) -> int | None:
    if raw is None:
        return None
    key = str(raw).strip().lower()
    if key in {"", "auto", "none", "null", "false", "no", "0"}:
        return None
    value = int(key)
    if value <= 0:
        raise ValueError("expected a positive integer or 'auto'")
    return value


def _case_label(row: dict[str, Any], args: argparse.Namespace) -> str:
    return (
        f"mpol{int(row['mpol'])}_ntor{int(row['ntor'])}_nzeta{int(row['nzeta'])}"
        f"_mgrid{int(args.mgrid_nr)}x{int(args.mgrid_nz)}x{int(row['mgrid_nphi'])}"
    )


def _profile_command(
    row: dict[str, Any],
    args: argparse.Namespace,
    *,
    resolution_only: bool,
    vmec2000: bool,
    scale_only: bool = False,
    jax_direct: bool = False,
) -> list[str]:
    ns_array = tuple(int(value) for value in row["ns_array"])
    niter_array = tuple(int(value) for value in row["niter_array"])
    ftol_array = tuple(float(value) for value in row["ftol_array"])
    command = [
        str(args.python),
        str(args.profile_script),
        "--outdir",
        str(Path(args.outdir_root) / f"square_coil_resolution_{_case_label(row, args)}"),
        "--beta-percent",
        "0",
        "--mpol",
        str(int(row["mpol"])),
        "--ntor",
        str(int(row["ntor"])),
        "--ns",
        str(int(ns_array[-1])),
        "--ntheta",
        str(int(row["ntheta"])),
        "--nzeta",
        str(int(row["nzeta"])),
        "--ns-array",
        ",".join(str(value) for value in ns_array),
        "--niter-array",
        ",".join(str(value) for value in niter_array),
        "--ftol-array",
        ",".join(f"{value:.0e}" for value in ftol_array),
        "--max-iter",
        str(int(niter_array[-1])),
        "--ftol",
        f"{float(ftol_array[-1]):.0e}",
        "--phiedge",
        f"{ExampleConfig().phiedge:.16g}",
        "--delt",
        f"{float(args.delt):.16g}",
        "--activate-fsq",
        f"{ExampleConfig().free_boundary_activate_fsq:.0e}",
        "--nvacskip",
        str(int(ExampleConfig().nvacskip)),
        "--nstep",
        "1",
        "--axis-kind",
        str(args.axis_kind),
        "--axis-corner-factor",
        f"{float(args.axis_corner_factor):.16g}",
        "--side-power",
        f"{float(args.side_power):.16g}",
        "--corner-power",
        f"{float(args.corner_power):.16g}",
        "--n-coils-per-side",
        str(int(ExampleConfig().n_coils_per_side)),
        "--coil-segments",
        str(int(args.coil_segments)),
        "--coil-chunk-size",
        str(int(args.coil_chunk_size)),
        "--mgrid-nr",
        str(int(args.mgrid_nr)),
        "--mgrid-nz",
        str(int(args.mgrid_nz)),
        "--mgrid-nphi",
        str(int(row["mgrid_nphi"])),
        "--mgrid-padding-fraction",
        f"{float(args.mgrid_padding_fraction):.16g}",
        "--mgrid-min-padding",
        f"{float(args.mgrid_min_padding):.16g}",
        "--max-boundary-projection-error",
        "none" if row["projection_target_max_component_error"] is None else f"{float(row['projection_target_max_component_error']):.0e}",
        "--solver-mode",
        "parity",
    ]
    if resolution_only:
        command.append("--resolution-diagnostics-only")
    if scale_only:
        command.append("--scale-diagnostics-only")
    if vmec2000:
        command.extend(
            [
                "--skip-direct",
                "--skip-mgrid",
                "--skip-provider-parity",
                "--run-vmec2000",
                "--vmec2000-exec",
                str(args.vmec2000_exec),
                "--vmec2000-timeout",
                str(int(args.vmec2000_timeout)),
            ]
        )
    if jax_direct:
        edge_projection = str(args.edge_control_projection)
        command.extend(
            [
                "--skip-mgrid",
                "--skip-provider-parity",
                "--freeb-anderson-pressure",
                "--strict-backtracking",
                "--strict-trial-heartbeat",
                "--strict-backtracking-accept-ratio",
                f"{float(args.strict_backtracking_accept_ratio):.16g}",
                "--jax-hot-restart-count",
                "2",
                "--jax-hot-restart-iters",
                str(int(niter_array[-1])),
                "--jax-hot-restart-policy",
                "freeb",
                "--jit-forces",
                "--jit-direct-sampler",
                "--verbose-solver",
                "--return-best-scored-state",
            ]
        )
        if edge_projection != "none":
            command.extend(
                [
                    "--freeb-edge-control-projection",
                    edge_projection,
                    "--freeb-edge-control-rcond",
                    f"{float(args.edge_control_rcond):.16g}",
                    "--freeb-edge-control-ridge",
                    f"{float(args.edge_control_ridge):.16g}",
                    "--freeb-edge-control-update-mode",
                    str(args.edge_control_update_mode),
                    "--freeb-edge-control-native-force-metric",
                    str(args.edge_control_native_force_metric),
                ]
            )
            if args.edge_control_trust_radius is not None:
                command.extend(
                    [
                        "--freeb-edge-control-trust-radius",
                        f"{float(args.edge_control_trust_radius):.16g}",
                    ]
                )
    return command


def _shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _control_map_rows(config: ExampleConfig) -> dict[str, Any]:
    """Return compact reduced-control conditioning diagnostics for one deck."""

    sample_kwargs = _square_axis_sample_kwargs(config)
    axis_kind = str(sample_kwargs.pop("axis_kind", config.plasma_axis_kind)).strip().lower()
    controls = sample_kwargs.pop("axis_spline_controls", None)
    if controls is None or axis_kind not in {"control_spline", "spline_controls", "periodic_spline"}:
        return {
            "control_map_status": "not_applicable_for_axis_kind",
            "control_map_axis_kind": axis_kind,
        }
    out: dict[str, Any] = {
        "control_map_status": "available",
        "control_map_axis_kind": axis_kind,
    }
    for symmetry in ("square", "stellarator", "full"):
        try:
            status = square_axis_spline_control_fourier_map_status(
                controls=controls,
                symmetry=symmetry,
                nfp=int(config.nfp),
                mpol=int(config.mpol),
                ntor=int(config.ntor),
                **_boundary_fit_grid(config),
                **sample_kwargs,
            )
        except Exception as exc:
            out[f"control_map_{symmetry}_status"] = f"failed:{type(exc).__name__}"
            out[f"control_map_{symmetry}_condition"] = None
            out[f"control_map_{symmetry}_count"] = None
            continue
        out[f"control_map_{symmetry}_status"] = status.get("status")
        out[f"control_map_{symmetry}_condition"] = status.get("condition_number")
        out[f"control_map_{symmetry}_count"] = status.get("control_count")
    return out


def build_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Return one cheap resolution status row per requested deck."""

    ns_array = _parse_int_list(args.ns_array)
    niter_array = _parse_int_list(args.niter_array)
    ftol_array = _parse_float_list(args.ftol_array)
    if not (len(ns_array) == len(niter_array) == len(ftol_array)):
        raise ValueError("ns-array, niter-array, and ftol-array must have matching lengths")
    target_error = _parse_optional_float(args.target_error)
    requested_ntheta = _parse_optional_int(args.ntheta)
    rows: list[dict[str, Any]] = []
    for deck in _parse_decks(args.decks):
        requested_mpol = int(deck["mpol"])
        requested_ntor = int(deck["ntor"])
        input_ntheta = int(requested_ntheta or recommended_square_axis_ntheta(requested_mpol))
        raw_requested_nzeta = None if deck["nzeta"] is None else int(deck["nzeta"])
        input_nzeta = int(raw_requested_nzeta or max(64, recommended_square_axis_nzeta(requested_ntor)))
        raw_requested_mgrid_nphi = None if deck["mgrid_nphi"] is None else int(deck["mgrid_nphi"])
        config = ExampleConfig(
            mpol=requested_mpol,
            ntor=requested_ntor,
            ns=int(ns_array[-1]),
            ns_array=ns_array,
            niter_array=niter_array,
            ftol_array=ftol_array,
            max_iter=int(niter_array[-1]),
            ftol=float(ftol_array[-1]),
            ntheta=input_ntheta,
            nzeta=input_nzeta,
            plasma_axis_kind=str(args.axis_kind),
            plasma_axis_spline_corner_radius_factor=float(args.axis_corner_factor),
            side_power=float(args.side_power),
            corner_power=float(args.corner_power),
            max_boundary_projection_error=target_error,
            enforce_recommended_nzeta=bool(target_error is not None),
            auto_bump_nzeta_to_recommended=bool(target_error is not None),
            auto_bump_mode_deck_to_recommended=bool(target_error is not None),
            write_plots=False,
        )
        requested_projection = square_axis_stellarator_mirror_hybrid_projection_error(
            nfp=int(config.nfp),
            mpol=requested_mpol,
            ntor=requested_ntor,
            **_boundary_fit_grid(config),
            ns_array=list(ns_array),
            niter_array=list(niter_array),
            ftol_array=list(ftol_array),
            phiedge=float(config.phiedge),
            **_square_axis_sample_kwargs(config),
        )
        requested_mgrid_nphi = int(raw_requested_mgrid_nphi or input_nzeta)
        requested_status = square_axis_resolution_deck_status(
            projection=requested_projection,
            mpol=requested_mpol,
            ntor=requested_ntor,
            ns=int(ns_array[-1]),
            ntheta=input_ntheta,
            nzeta=input_nzeta,
            mgrid_nphi=requested_mgrid_nphi,
            target_max_component_error=target_error,
        )
        mode_deck = _effective_square_axis_mode_deck(config)
        effective_config = _effective_solve_config(config)
        effective_resolution = _effective_square_axis_resolution(effective_config)
        effective_mgrid_nphi = int(raw_requested_mgrid_nphi or effective_resolution.effective_nzeta)
        effective_projection = square_axis_stellarator_mirror_hybrid_projection_error(
            nfp=int(effective_config.nfp),
            mpol=int(effective_config.mpol),
            ntor=int(effective_config.ntor),
            **_boundary_fit_grid(effective_config),
            ns_array=list(ns_array),
            niter_array=list(niter_array),
            ftol_array=list(ftol_array),
            phiedge=float(effective_config.phiedge),
            **_square_axis_sample_kwargs(effective_config),
        )
        status = square_axis_resolution_deck_status(
            projection=effective_projection,
            mpol=int(effective_config.mpol),
            ntor=int(effective_config.ntor),
            ns=int(ns_array[-1]),
            ntheta=int(effective_resolution.effective_ntheta),
            nzeta=int(effective_resolution.effective_nzeta),
            mgrid_nphi=effective_mgrid_nphi,
            target_max_component_error=target_error,
        )
        strict_schedule = square_axis_strict_schedule_status(
            ns_array=ns_array,
            niter_array=niter_array,
            ftol_array=ftol_array,
            target_ftol=1.0e-12,
        )
        edge_projection = str(args.edge_control_projection).strip().lower()
        edge_control_enabled = edge_projection not in {"", "none", "off", "false"}
        strict_assessment = square_axis_strict_convergence_assessment(
            resolution_deck=status,
            strict_schedule=strict_schedule,
            edge_control_projection_enabled=bool(edge_control_enabled),
            edge_control_update_mode=str(args.edge_control_update_mode),
            solver_native_spline_controls=False,
            target_ftol=1.0e-12,
        )
        row = {
            **status,
            "requested_mpol": requested_mpol,
            "requested_ntor": requested_ntor,
            "requested_ntheta": None if requested_ntheta is None else int(requested_ntheta),
            "input_ntheta": input_ntheta,
            "requested_nzeta": raw_requested_nzeta,
            "input_nzeta": input_nzeta,
            "requested_mgrid_nphi": raw_requested_mgrid_nphi,
            "input_mgrid_nphi": requested_mgrid_nphi,
            "requested_status": requested_status.get("status"),
            "requested_reasons": requested_status.get("reasons"),
            "requested_projection_max_abs_component_error": float(
                requested_projection["max_abs_component_error"]
            ),
            "mpol": int(effective_config.mpol),
            "ntor": int(effective_config.ntor),
            "ntheta": int(effective_resolution.effective_ntheta),
            "nzeta": int(effective_resolution.effective_nzeta),
            "mgrid_nphi": effective_mgrid_nphi,
            "mode_deck_auto_bumped_to_recommended": bool(
                mode_deck.mode_deck_auto_bumped_to_recommended
            ),
            "nzeta_auto_bumped_to_recommended": bool(effective_resolution.nzeta_auto_bumped_to_recommended),
            "ns_array": ns_array,
            "niter_array": niter_array,
            "ftol_array": ftol_array,
            "axis_kind": str(args.axis_kind),
            "side_power": float(args.side_power),
            "corner_power": float(args.corner_power),
            "projection_max_abs_error": float(effective_projection["max_abs_error"]),
            "projection_max_abs_error_rel": float(effective_projection["max_abs_error_rel"]),
            "projection_max_abs_component_error_rel": float(
                effective_projection["max_abs_component_error_rel"]
            ),
            "strict_schedule_status": strict_schedule.get("status"),
            "strict_schedule_reasons": strict_schedule.get("reasons"),
            "requested_final_ftol": strict_schedule.get("requested_final_ftol"),
            "requested_final_ftol_meets_target": strict_schedule.get("requested_final_ftol_meets_target"),
            "total_iteration_budget": strict_schedule.get("total_iteration_budget"),
            "edge_control_projection": edge_projection,
            "edge_control_rcond": float(args.edge_control_rcond),
            "edge_control_ridge": float(args.edge_control_ridge),
            "edge_control_trust_radius": (
                None if args.edge_control_trust_radius is None else float(args.edge_control_trust_radius)
            ),
            "edge_control_update_mode": str(args.edge_control_update_mode),
            "edge_control_native_force_metric": str(args.edge_control_native_force_metric),
            "strict_full_fourier_status": strict_assessment.get("full_fourier_strict_profile_status"),
            "strict_reduced_control_status": strict_assessment.get("reduced_control_profile_status"),
            "strict_solver_native_spline_status": strict_assessment.get("solver_native_spline_status"),
            "strict_solver_native_spline_edge_controls": strict_assessment.get(
                "solver_native_spline_edge_controls"
            ),
            "strict_solver_native_spline_scope": strict_assessment.get("solver_native_spline_scope"),
            "strict_full_native_spline_state_required": strict_assessment.get(
                "full_native_spline_state_required_for_less_fourier_pressure"
            ),
            "strict_vmec2000_reference_role": strict_assessment.get("vmec2000_reference_role"),
            "strict_vmec2000_expected_to_fix_fourier_bottleneck": strict_assessment.get(
                "vmec2000_expected_to_fix_fourier_bottleneck"
            ),
            "strict_recommended_primary_solver_lane": strict_assessment.get("recommended_primary_solver_lane"),
            "strict_fast_cli_reference_lane": strict_assessment.get("fast_cli_reference_lane"),
            "strict_differentiable_solver_lane": strict_assessment.get("differentiable_solver_lane"),
            "strict_native_spline_recommendation": strict_assessment.get("native_spline_recommendation"),
            "strict_derivative_method_priority": strict_assessment.get("derivative_method_priority"),
            "strict_assessment_blockers": strict_assessment.get("blockers"),
            "strict_assessment_next_steps": strict_assessment.get("recommended_next_steps"),
        }
        if bool(args.include_control_map):
            row.update(_control_map_rows(effective_config))
        if bool(args.print_preflight_commands):
            row["preflight_command"] = _shell_join(
                _profile_command(row, args, resolution_only=True, vmec2000=False)
            )
        if bool(args.print_scale_commands):
            row["scale_command"] = _shell_join(
                _profile_command(row, args, resolution_only=False, scale_only=True, vmec2000=False)
            )
        if bool(args.print_vmec2000_commands):
            row["vmec2000_command"] = _shell_join(_profile_command(row, args, resolution_only=False, vmec2000=True))
        if bool(args.print_jax_commands):
            row["jax_command"] = _shell_join(
                _profile_command(row, args, resolution_only=False, vmec2000=False, jax_direct=True)
            )
        rows.append(row)
    return rows


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return str(value)


def _print_table(rows: list[dict[str, Any]], *, markdown: bool) -> None:
    keys = [
        "requested_mpol",
        "requested_ntor",
        "requested_nzeta",
        "mpol",
        "ntor",
        "ntheta",
        "recommended_ntheta",
        "nzeta",
        "recommended_nzeta",
        "mode_deck_auto_bumped_to_recommended",
        "nzeta_auto_bumped_to_recommended",
        "mgrid_nphi",
        "status",
        "reasons",
        "requested_status",
        "requested_reasons",
        "mode_count",
        "requested_projection_max_abs_component_error",
        "projection_max_abs_component_error",
        "projection_target_max_component_error",
        "strict_schedule_status",
        "requested_final_ftol",
        "requested_final_ftol_meets_target",
        "total_iteration_budget",
        "edge_control_projection",
        "edge_control_rcond",
        "edge_control_ridge",
        "edge_control_trust_radius",
        "edge_control_update_mode",
        "edge_control_native_force_metric",
        "strict_full_fourier_status",
        "strict_reduced_control_status",
        "strict_solver_native_spline_status",
        "strict_solver_native_spline_edge_controls",
        "strict_solver_native_spline_scope",
        "strict_full_native_spline_state_required",
        "strict_vmec2000_reference_role",
        "strict_vmec2000_expected_to_fix_fourier_bottleneck",
        "strict_recommended_primary_solver_lane",
        "strict_fast_cli_reference_lane",
        "strict_differentiable_solver_lane",
        "strict_native_spline_recommendation",
        "strict_derivative_method_priority",
        "strict_assessment_blockers",
    ]
    if any("preflight_command" in row for row in rows):
        keys.append("preflight_command")
    if any("scale_command" in row for row in rows):
        keys.append("scale_command")
    if any("vmec2000_command" in row for row in rows):
        keys.append("vmec2000_command")
    if any("jax_command" in row for row in rows):
        keys.append("jax_command")
    if any("control_map_status" in row for row in rows):
        keys.extend(
            [
                "control_map_status",
                "control_map_square_status",
                "control_map_square_count",
                "control_map_square_condition",
                "control_map_stellarator_status",
                "control_map_stellarator_count",
                "control_map_stellarator_condition",
            ]
        )
    sep = " | " if markdown else "\t"
    if markdown:
        print("| " + sep.join(keys) + " |")
        print("| " + sep.join("---" for _ in keys) + " |")
        for row in rows:
            print("| " + sep.join(_format_value(row.get(key)) for key in keys) + " |")
        return
    print(sep.join(keys))
    for row in rows:
        print(sep.join(_format_value(row.get(key)) for key in keys))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    rows = build_rows(args)
    if args.format == "json":
        print(json.dumps(rows, indent=2, sort_keys=True, allow_nan=False))
    else:
        _print_table(rows, markdown=args.format == "markdown")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
