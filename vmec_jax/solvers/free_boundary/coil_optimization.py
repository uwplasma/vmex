"""Helpers for single-stage direct-coil free-boundary optimization.

These helpers intentionally do not run VMEC and do not decide whether a coil
step is accepted.  They only turn a validated same-branch derivative report
into bounded optimizer-coordinate trial points.  A normal complete free-boundary
solve must still evaluate every proposal before it is trusted.
"""

from __future__ import annotations

from copy import deepcopy
import time
from typing import Any, Mapping, Sequence

import numpy as np

from vmec_jax.external_fields import CoilFieldParams, build_coil_field_geometry
from vmec_jax._compat import jax, jnp
from vmec_jax.finite_beta import finite_beta_scalars_from_state
from vmec_jax.quasi_isodynamic import boozer_output_from_state
from vmec_jax.quasisymmetry import (
    quasisymmetry_boozer_mode_residual_from_boozer_output,
    quasisymmetry_ratio_residual_from_state,
)
from vmec_jax.wout import equilibrium_aspect_ratio_from_state, equilibrium_iota_profiles_from_state
from vmec_jax.solvers.free_boundary.adjoint.trace_metadata import (
    direct_coil_accepted_trace_controller_slot_fingerprint,
)

__all__ = [
    "DEFAULT_DERIVATIVE_PROPOSAL_VECTOR_KEYS",
    "DEFAULT_SAME_BRANCH_VECTOR_KEYS", "STATE_ONLY_SAME_BRANCH_KEYS", "SUPPORTED_SAME_BRANCH_VECTOR_KEYS",
    "SINGLE_STAGE_LIMITATIONS",
    "direct_coil_optimization_workflow_metadata",
    "direct_coil_qs_summary_configs",
    "nestor_profile_policy_from_results",
    "same_branch_nestor_profile_from_vector_replay",
    "parse_float_list",
    "parse_same_branch_vector_keys",
    "parse_profile_matrix_free_solvers",
    "same_branch_current_only_coil_geometry_cache",
    "same_branch_derivative_gate_evidence",
    "same_branch_derivative_proposal_from_report",
    "same_branch_derivative_proposals_from_report",
    "same_branch_complete_fd_report_metadata",
    "same_branch_rejected_slot_gate_from_vector_replay",
    "same_branch_replay_plan_cache",
    "same_branch_replay_mode_count_guard",
    "same_branch_replay_options_from_args",
    "same_branch_report_direction_policy",
    "same_branch_report_vector_keys_from_args",
    "same_branch_report_runtime_configs",
    "same_branch_report_mode_count",
    "run_same_branch_scalar_report_section",
    "run_same_branch_vector_report_section",
    "same_branch_scalar_result_summary",
    "same_branch_scalar_function_registry",
    "same_branch_vector_result_summary",
]


DEFAULT_SAME_BRANCH_VECTOR_KEYS = ("aspect", "qs_total", "mean_iota", "lcfs_boundary_moment")
DEFAULT_DERIVATIVE_PROPOSAL_VECTOR_KEYS = ("aspect", "qs_total", "mean_iota")
SUPPORTED_SAME_BRANCH_VECTOR_KEYS = DEFAULT_SAME_BRANCH_VECTOR_KEYS + (
    "state_norm",
    "boozer_qs_total",
    "accepted_bnormal_rms",
    "betatotal",
)
STATE_ONLY_SAME_BRANCH_KEYS = tuple(
    key for key in SUPPORTED_SAME_BRANCH_VECTOR_KEYS if key != "accepted_bnormal_rms"
)
SINGLE_STAGE_LIMITATIONS = (
    "The QS term is a VMEC-state quasisymmetry-ratio residual, not a Boozer-space exact-adjoint objective.",
    "Production full-loop direct-coil free-boundary adjoints are not promoted yet.",
    "ESSOS and VMEC2000 generated-mgrid comparisons remain optional external-asset diagnostics.")


def direct_coil_optimization_workflow_metadata(repo_root: Any) -> dict[str, Any]:
    """Return the pedagogic workflow contract recorded in summary artifacts."""

    return {
        "flow": "single_stage_direct_coil_no_mgrid",
        "field_backend": "direct_coils",
        "workflow_steps": [
            "load or synthesize direct coils",
            "select coil-current and coil-Fourier optimization variables",
            "write VMEC input with MGRID_FILE='DIRECT_COILS'",
            "run complete free-boundary solves with direct JAX Biot-Savart sampling",
            "score VMEC residual, VMEC-state QS residual, aspect, and mean-iota terms",
        ],
        "optimized_dofs": "coil currents and selected coil Fourier coefficients only",
        "plasma_boundary_optimized": False,
        "python_provider_required": True,
        "uses_mgrid_file": False,
        "mgrid_compatibility_example": str(repo_root / "examples" / "free_boundary_essos_mgrid_forward.py"),
        "vmec_input_replay": (
            "MGRID_FILE='DIRECT_COILS' is a vmec_jax Python-provider tag. "
            "Run this optimization script, or call run_free_boundary with CoilFieldParams, "
            "so the solver receives the direct-coil provider."
        ),
    }


def direct_coil_qs_summary_configs(
    args: Any,
    *,
    input_path: Any,
    workflow: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return objective, VMEC, and optimizer summary configs for direct-coil QS examples."""

    objective_model = {
        "description": "Deterministic direct-coil free-boundary objective with VMEC residual, QS, aspect, and iota terms.",
        "qs_note": (
            "The QS term is evaluated from the accepted VMEC state. Full coil-to-Boozer/QS exact "
            "gradients through adaptive free-boundary branch selection remain a separate promotion gate."
        ),
        "helicity_m": int(args.helicity_m),
        "helicity_n": int(args.helicity_n),
        "qs_surfaces": parse_float_list(str(args.qs_surfaces)),
        "qs_ntheta": int(args.qs_ntheta),
        "qs_nphi": int(args.qs_nphi),
        "target_aspect": float(args.target_aspect),
        "target_iota": float(args.target_iota),
        "residual_weight": float(args.residual_weight),
        "qs_weight": float(args.qs_weight),
        "aspect_weight": float(args.aspect_weight),
        "iota_weight": float(args.iota_weight),
        "failure_objective": float(args.failure_objective),
    }
    vmec_config = {
        "input_template": args.input,
        "generated_input": input_path,
        "external_field_provider_kind": "direct_coils",
        "mgrid_file": "DIRECT_COILS",
        "uses_generated_mgrid": False,
        "python_provider_required": True,
        "uses_mgrid_file": False,
        "vmec_input_replay": workflow["vmec_input_replay"],
        "mgrid_compatibility_example": workflow["mgrid_compatibility_example"],
        "vmec_max_iter": int(args.vmec_max_iter),
        "ftol": float(args.ftol),
        "ns": int(args.ns),
        "mpol": int(args.mpol),
        "ntor": int(args.ntor),
        "nzeta": int(args.nzeta),
        "beta_percent": float(args.beta),
        "pressure_profile": str(args.pressure_profile),
        "pressure_scale": float(args.pressure_scale),
        "phiedge": float(args.phiedge),
        "activate_fsq": float(args.activate_fsq),
        "jit_forces": bool(args.jit_forces),
    }
    optimizer_config = {"method": "Powell", "max_iter": int(args.max_iter), "max_evals": int(args.max_evals),
                        "xtol": float(args.xtol), "ftol": float(args.optimizer_ftol)}
    return objective_model, vmec_config, optimizer_config


def parse_profile_matrix_free_solvers(value: str | Sequence[str] | None) -> tuple[str, ...]:
    """Parse matrix-free solver names for the same-branch NESTOR profile."""

    if value is None:
        return ("gmres", "bicgstab")
    if isinstance(value, str):
        raw = value.replace(",", " ").split()
    else:
        raw = [str(item) for item in value]
    solvers = tuple(item.strip().lower() for item in raw if item.strip())
    unsupported = tuple(item for item in solvers if item not in {"gmres", "bicgstab"})
    if unsupported:
        raise ValueError(f"unsupported matrix-free NESTOR solver(s): {unsupported}")
    return solvers or ("gmres", "bicgstab")


def parse_same_branch_vector_keys(value: str | Sequence[str] | None) -> tuple[str, ...]:
    """Parse branch-local vector report scalar keys from a small CLI option."""

    if value is None:
        keys = DEFAULT_SAME_BRANCH_VECTOR_KEYS
    elif isinstance(value, str):
        keys = tuple(part.strip() for part in value.replace(",", " ").split() if part.strip())
    else:
        keys = tuple(str(part).strip() for part in value if str(part).strip())
    keys = tuple("accepted_bnormal_rms" if key == "bnormal_rms" else key for key in keys)
    if not keys:
        raise ValueError("expected at least one same-branch vector scalar key")
    unsupported = tuple(key for key in keys if key not in SUPPORTED_SAME_BRANCH_VECTOR_KEYS)
    if unsupported:
        supported = ", ".join(SUPPORTED_SAME_BRANCH_VECTOR_KEYS)
        raise ValueError(f"Unsupported same-branch vector scalar key(s) {unsupported}; supported keys: {supported}")
    return keys


def same_branch_report_vector_keys_from_args(args: Any) -> tuple[str, ...]:
    """Return vector-report keys with a cheaper default for proposal-only runs.

    Ordinary validation reports keep the promoted multi-scalar default.  A
    derivative proposal only consumes the objective terms currently supported by
    ``same_branch_derivative_proposals_from_report``: aspect, QS, and mean-iota.
    When the user does not explicitly request vector keys, use that narrower
    set to avoid compiling/replaying an unused LCFS moment scalar.  Any explicit
    ``--same-branch-report-vector-keys`` value is honored unchanged.
    """

    requested = getattr(args, "same_branch_report_vector_keys", None)
    if requested is None and bool(getattr(args, "same_branch_derivative_proposal", False)):
        return DEFAULT_DERIVATIVE_PROPOSAL_VECTOR_KEYS
    return parse_same_branch_vector_keys(requested)


def parse_float_list(text: str) -> list[float]:
    """Parse comma/space-separated floats from a small CLI option."""
    values = [float(part) for part in str(text).replace(",", " ").split() if part]
    if not values:
        raise ValueError("expected at least one floating-point value")
    return values


def same_branch_report_direction_policy(
    args: Any,
    variables: list[tuple[str, tuple[int, ...]]],
) -> tuple[str, str, str]:
    """Return requested/effective same-branch report direction policy."""

    requested = str(getattr(args, "same_branch_report_direction", "auto")).strip().lower()
    if requested not in {"auto", "all", "current-only"}:
        raise ValueError("--same-branch-report-direction must be one of auto, all, current-only")
    has_current = any(kind == "current" for kind, _index in variables)
    if requested == "auto":
        if bool(getattr(args, "same_branch_derivative_proposal", False)) and has_current:
            return requested, "current-only", "auto selected current-only for derivative-proposal evidence"
        return requested, "all", "auto selected mixed direction for ordinary same-branch validation"
    if requested == "current-only" and not has_current:
        raise ValueError("--same-branch-report-direction=current-only requires at least one selected current variable")
    return requested, requested, "explicit user selection"


def same_branch_report_runtime_configs(
    args: Any,
    variables: list[tuple[str, tuple[int, ...]]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return summary configs for same-branch reports and derivative proposals."""

    requested, effective, reason = same_branch_report_direction_policy(args, variables)
    proposal_steps_text = str(getattr(args, "same_branch_proposal_steps", "")).strip()
    proposal_steps = (
        parse_float_list(proposal_steps_text)
        if proposal_steps_text
        else [float(getattr(args, "same_branch_proposal_step"))]
    )
    report_config = {
        "enabled": bool(getattr(args, "write_same_branch_report", False)),
        "mode": str(getattr(args, "same_branch_report_mode", "vector")),
        "ad_mode": str(getattr(args, "same_branch_report_ad_mode", "direct")),
        "vector_keys": list(same_branch_report_vector_keys_from_args(args)),
        "default_derivative_detail": "direct vector JVP for several physical scalars"
        if str(getattr(args, "same_branch_report_mode", "vector")) == "vector"
        and str(getattr(args, "same_branch_report_ad_mode", "direct")) == "direct"
        else "user-selected report mode",
        "contract": (
            "production-forward values plus fixed accepted-branch replay derivatives; "
            "does not differentiate adaptive host branch selection"
        ),
        "eps": float(getattr(args, "same_branch_report_eps")),
        "max_iter": int(getattr(args, "same_branch_report_max_iter") or getattr(args, "vmec_max_iter")),
        "anchor": str(getattr(args, "same_branch_report_anchor", "best")),
        "direction_policy": {"requested": requested, "effective": effective, "reason": reason},
        "diagnostic_disable_analytic": bool(getattr(args, "same_branch_report_disable_analytic", False)),
        "diagnostic_freeze_vacuum_field": bool(getattr(args, "same_branch_report_freeze_vacuum_field", False)),
        "diagnostic_freeze_bsqvac": bool(getattr(args, "same_branch_report_freeze_bsqvac", False)),
        "nestor_solve_mode": str(getattr(args, "same_branch_report_nestor_solve_mode", "dense")),
        "nestor_operator_solver": str(getattr(args, "same_branch_report_nestor_operator_solver", "gmres")),
        "nestor_operator_tol": float(getattr(args, "same_branch_report_nestor_operator_tol", 1.0e-11)),
        "nestor_operator_atol": float(getattr(args, "same_branch_report_nestor_operator_atol", 1.0e-13)),
        "nestor_operator_maxiter": getattr(args, "same_branch_report_nestor_operator_maxiter", None),
        "nestor_operator_restart": getattr(args, "same_branch_report_nestor_operator_restart", None),
        "replay_max_mode_count": int(getattr(args, "same_branch_report_replay_max_mode_count", 220)),
        "profile_nestor": str(getattr(args, "same_branch_report_profile_nestor", "none")),
        "profile_matrix_free_solvers": list(
            parse_profile_matrix_free_solvers(getattr(args, "same_branch_report_profile_matrix_free_solvers", None))
        ),
        "profile_min_mode_count": int(getattr(args, "same_branch_report_profile_min_mode_count", 96)),
        "profile_min_speedup": float(getattr(args, "same_branch_report_profile_min_speedup", 1.15)),
        "profile_max_mode_count": int(getattr(args, "same_branch_report_profile_max_mode_count", 220)),
        "rejected_slot_gate": bool(getattr(args, "same_branch_report_rejected_slot_gate", False)),
    }
    proposal_config = {
        "enabled": bool(getattr(args, "same_branch_derivative_proposal", False)),
        "requires_same_branch_report": True,
        "requires_report_mode": "vector",
        "requires_report_ad_mode": "direct for JVP-only proposal; custom_vjp is report-only",
        "scope": "one fixed-accepted-branch directional proposal followed by a normal complete-solve objective evaluation",
        "step_size": float(getattr(args, "same_branch_proposal_step")),
        "step_sizes": proposal_steps,
        "max_trials": int(getattr(args, "same_branch_proposal_max_trials")),
        "max_base_abs_delta": float(getattr(args, "same_branch_proposal_max_base_delta")),
        "differentiates_adaptive_controller": False,
    }
    return report_config, proposal_config


def same_branch_report_mode_count(report: dict[str, Any]) -> int:
    """Return the VMEC Fourier mode count for report-size policy decisions."""

    try:
        static = report["base"]["init"].static
        return int(np.asarray(static.modes.m).size)
    except Exception:
        return 0


def same_branch_complete_fd_report_metadata(
    *,
    input_path: Any,
    report_anchor: str,
    eps: float,
    direction_policy: tuple[str, str, str],
    direction_x: Any,
    direction_variables: list[dict[str, Any]],
    report: dict[str, Any],
) -> dict[str, Any]:
    """Return compact same-branch complete-solve FD report metadata."""

    requested, effective, reason = direction_policy
    compatibility = report["branch_compatibility"]
    plus = compatibility["plus"]
    minus = compatibility["minus"]
    branch_compatibility = {
        "same_branch": bool(compatibility["same_branch"]),
        "plus_changed_fields": list(plus["changed_fields"]),
        "minus_changed_fields": list(minus["changed_fields"]),
        "plus_max_abs_scalar_delta": float(plus["max_abs_scalar_delta"]),
        "minus_max_abs_scalar_delta": float(minus["max_abs_scalar_delta"]),
        "plus_max_rel_scalar_delta": float(plus["max_rel_scalar_delta"]),
        "minus_max_rel_scalar_delta": float(minus["max_rel_scalar_delta"]),
    }
    return {
        "phase": "phase-2-same-branch-complete-solve-fd",
        "scope": "coil-only proxy-objective validation; not arbitrary adaptive-branch differentiation",
        "input": str(input_path),
        "report_anchor": str(report_anchor),
        "eps": float(eps),
        "direction_policy": {"requested": requested, "effective": effective, "reason": reason},
        "direction_x": np.asarray(direction_x, dtype=float).tolist(),
        "direction_variables": direction_variables,
        "branch_compatibility": branch_compatibility,
        "values": report["values"],
        "objective_values": report["objective_values"],
        "primary_objective": report["primary_objective"],
    }


def same_branch_replay_options_from_args(args: Any) -> dict[str, Any]:
    """Return branch-local replay options shared by scalar/vector report paths."""

    return {"use_stacked_step_controls": True, "use_accepted_only_fast_path": True,
            "jit_preconditioner_apply": not bool(getattr(args, "same_branch_report_disable_jit_preconditioner", False)),
            "include_analytic": not bool(getattr(args, "same_branch_report_disable_analytic", False)),
            "include_mode_diagnostics": False,
            "nestor_solve_mode": str(getattr(args, "same_branch_report_nestor_solve_mode", "dense")),
            "nestor_operator_solver": str(getattr(args, "same_branch_report_nestor_operator_solver", "gmres")),
            "nestor_operator_tol": float(getattr(args, "same_branch_report_nestor_operator_tol", 1.0e-11)),
            "nestor_operator_atol": float(getattr(args, "same_branch_report_nestor_operator_atol", 1.0e-13)),
            "nestor_operator_maxiter": getattr(args, "same_branch_report_nestor_operator_maxiter", None),
            "nestor_operator_restart": getattr(args, "same_branch_report_nestor_operator_restart", None),
            "enable_current_only_jvp_cache": bool(
                getattr(args, "same_branch_report_enable_current_jvp_cache", False)
            ),
            "freeze_vacuum_field": bool(getattr(args, "same_branch_report_freeze_vacuum_field", False)),
            "freeze_freeb_bsqvac": bool(getattr(args, "same_branch_report_freeze_bsqvac", False))}


def same_branch_replay_mode_count_guard(mode_count: int, replay_max_mode_count: int) -> tuple[bool, str, dict[str, Any]]:
    """Return the replay mode-count guard state and JSON metadata."""

    triggered = int(replay_max_mode_count) > 0 and int(mode_count) > int(replay_max_mode_count)
    reason = (
        f"mode_count {int(mode_count)} exceeds replay cap {int(replay_max_mode_count)}; "
        "set --same-branch-report-replay-max-mode-count 0 to disable this guard"
    )
    return bool(triggered), reason, {"enabled": int(replay_max_mode_count) > 0, "triggered": bool(triggered),
                                     "mode_count": int(mode_count), "max_mode_count": int(replay_max_mode_count),
                                     "reason": reason if triggered else "not triggered"}


def same_branch_scalar_function_registry(
    *,
    args: Any,
    qs_surfaces: Sequence[float],
    qs_angle_cache_for_static: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return production and replay scalar functions for same-branch reports."""

    from vmec_jax.free_boundary_adjoint import free_boundary_boundary_geometry_jax
    from vmec_jax.state import pack_state

    def lcfs_boundary_moment(state: Any, static: Any) -> Any:
        geometry = free_boundary_boundary_geometry_jax(state, static)
        r = jnp.asarray(geometry["R"])
        z = jnp.asarray(geometry["Z"])
        return jnp.mean((r - 1.0) * (r - 1.0) + z * z)

    def mean_iota_from_state(state: Any, static: Any, indata: Any, signgs: int) -> Any:
        _chips, iotas, _iotaf = equilibrium_iota_profiles_from_state(
            state=state,
            static=static,
            indata=indata,
            signgs=int(signgs),
        )
        iota_arr = jnp.asarray(iotas)
        return jnp.mean(iota_arr[1:] if iota_arr.size > 1 else iota_arr)

    def accepted_bnormal_rms_from_payload(payload: dict[str, Any]) -> float:
        values = [
            float(np.sqrt(np.mean(np.square(np.asarray(trace["freeb_nestor_trace"]["bnormal"], dtype=float)))))
            for trace in payload["traces"]
            if trace.get("freeb_bsqvac_half") is not None
            and isinstance(trace.get("freeb_nestor_trace"), dict)
            and trace["freeb_nestor_trace"].get("bnormal") is not None
        ]
        if not values:
            return 0.0
        return float(np.mean(values))

    def accepted_bnormal_rms_from_replay(replay: dict[str, Any]) -> Any:
        bnormal = jnp.asarray(replay["history"]["bnormal_rms"])
        accepted = jnp.asarray(replay["history"]["accepted"], dtype=bnormal.dtype)
        active = jnp.asarray(replay["controls"]["has_active_freeb_replay"], dtype=bnormal.dtype)
        weights = accepted * active
        denom = jnp.maximum(jnp.sum(weights), jnp.asarray(1.0, dtype=bnormal.dtype))
        return jnp.sum(weights * bnormal) / denom

    def qs_total_from_state(state: Any, static: Any, indata: Any, signgs: int) -> Any:
        qs = quasisymmetry_ratio_residual_from_state(
            state=state,
            static=static,
            indata=indata,
            signgs=int(signgs),
            surfaces=qs_surfaces,
            helicity_m=int(args.helicity_m),
            helicity_n=int(args.helicity_n),
            ntheta=int(args.qs_ntheta),
            nphi=int(args.qs_nphi),
            angle_cache=qs_angle_cache_for_static(static),
        )
        return qs["total"]

    def boozer_qs_total_from_state(state: Any, static: Any, indata: Any, signgs: int) -> Any:
        field = boozer_output_from_state(
            state=state,
            static=static,
            indata=indata,
            signgs=int(signgs),
            surfaces=qs_surfaces,
            mboz=int(getattr(args, "same_branch_boozer_mboz", 8)),
            nboz=int(getattr(args, "same_branch_boozer_nboz", 8)),
            jit_booz=False,
        )
        qs = quasisymmetry_boozer_mode_residual_from_boozer_output(
            field["booz"],
            helicity_m=int(args.helicity_m),
            helicity_n=int(args.helicity_n),
            nfp=int(field["nfp"]),
            normalize=bool(getattr(args, "same_branch_boozer_normalize", True)),
        )
        return qs["total"]

    scalar_value_fns = {
        "state_norm": lambda payload: float(np.linalg.norm(np.asarray(pack_state(payload["result"].state), dtype=float))),
        "aspect": lambda payload: float(
            np.asarray(
                equilibrium_aspect_ratio_from_state(
                    state=payload["result"].state,
                    static=payload["init"].static,
                )
            )
        ),
        "mean_iota": lambda payload: float(
            np.asarray(
                mean_iota_from_state(
                    payload["result"].state,
                    payload["init"].static,
                    payload["init"].indata,
                    payload["init"].signgs,
                )
            )
        ),
        "qs_total": lambda payload: float(
            np.asarray(
                qs_total_from_state(
                    payload["result"].state,
                    payload["init"].static,
                    payload["init"].indata,
                    payload["init"].signgs,
                )
            )
        ),
        "boozer_qs_total": lambda payload: float(
            np.asarray(
                boozer_qs_total_from_state(
                    payload["result"].state,
                    payload["init"].static,
                    payload["init"].indata,
                    payload["init"].signgs,
                )
            )
        ),
        "lcfs_boundary_moment": lambda payload: float(
            np.asarray(lcfs_boundary_moment(payload["result"].state, payload["init"].static))
        ),
        "accepted_bnormal_rms": accepted_bnormal_rms_from_payload,
        "betatotal": lambda payload: float(
            np.asarray(
                finite_beta_scalars_from_state(
                    state=payload["result"].state,
                    static=payload["init"].static,
                    indata=payload["init"].indata,
                    signgs=payload["init"].signgs,
                )["betatotal"]
            )
        ),
    }
    scalar_replay_fns = {
        "state_norm": lambda replay, _payload: jnp.linalg.norm(pack_state(replay["state"])),
        "aspect": lambda replay, payload: equilibrium_aspect_ratio_from_state(
            state=replay["state"],
            static=payload["init"].static,
        ),
        "mean_iota": lambda replay, payload: mean_iota_from_state(
            replay["state"],
            payload["init"].static,
            payload["init"].indata,
            payload["init"].signgs,
        ),
        "qs_total": lambda replay, payload: qs_total_from_state(
            replay["state"],
            payload["init"].static,
            payload["init"].indata,
            payload["init"].signgs,
        ),
        "boozer_qs_total": lambda replay, payload: boozer_qs_total_from_state(
            replay["state"],
            payload["init"].static,
            payload["init"].indata,
            payload["init"].signgs,
        ),
        "lcfs_boundary_moment": lambda replay, payload: lcfs_boundary_moment(
            replay["state"],
            payload["init"].static,
        ),
        "accepted_bnormal_rms": lambda replay, _payload: accepted_bnormal_rms_from_replay(replay),
        "betatotal": lambda replay, payload: finite_beta_scalars_from_state(
            state=replay["state"],
            static=payload["init"].static,
            indata=payload["init"].indata,
            signgs=payload["init"].signgs,
        )["betatotal"],
    }
    return scalar_value_fns, scalar_replay_fns


def same_branch_replay_plan_cache(
    report: dict[str, Any],
    replay_kwargs: dict[str, Any],
    *,
    timing_key: str,
    scope: str,
) -> tuple[dict[str, Any] | None, dict[str, Any], float | None]:
    """Build an accepted-trace replay plan for repeated same-branch reports."""

    from vmec_jax.free_boundary_adjoint import direct_coil_accepted_trace_controller_replay_plan

    try:
        t0 = time.perf_counter()
        replay_plan = direct_coil_accepted_trace_controller_replay_plan(
            tuple(report["base"]["traces"]),
            static=report["base"]["init"].static,
            use_preconditioner_policy_segments=bool(
                replay_kwargs.get("use_preconditioner_policy_segments", False)
            ),
            use_segment_preconditioner_controls=bool(
                replay_kwargs.get("use_segment_preconditioner_controls", False)
            ),
            use_stacked_step_controls=bool(replay_kwargs.get("use_stacked_step_controls", False)),
            use_accepted_only_fast_path=bool(replay_kwargs.get("use_accepted_only_fast_path", True)),
        )
        return replay_plan, {"available": True, "timing_key": timing_key, "scope": scope}, float(
            time.perf_counter() - t0
        )
    except Exception as exc:  # pragma: no cover - synthetic tests may omit stackable trace controls.
        return None, {"available": False, "reason": f"{type(exc).__name__}: {exc}", "scope": scope}, None


def same_branch_current_only_coil_geometry_cache(
    params: CoilFieldParams,
    direction_params: CoilFieldParams,
) -> tuple[tuple[Any, Any] | None, dict[str, Any], float | None]:
    """Cache fixed coil geometry when same-branch reports vary currents only."""

    try:
        direction_dofs = np.asarray(direction_params.base_curve_dofs, dtype=float)
        if np.any(direction_dofs):
            return None, {"available": False, "reason": "direction includes coil-shape dofs"}, None
        t0 = time.perf_counter()
        gamma, gamma_dash, _currents = build_coil_field_geometry(params)
        return (
            (gamma, gamma_dash),
            {
                "available": True,
                "scope": "current-only branch-local vector/profile replays",
                "timing_key": "branch_local_current_only_coil_geometry_build_wall_s",
            },
            float(time.perf_counter() - t0),
        )
    except Exception as exc:  # pragma: no cover - defensive; report artifacts should not abort examples.
        return None, {"available": False, "reason": f"{type(exc).__name__}: {exc}"}, None


def _vector_jacobian_directional(jacobian: Any, direction: Any, n_outputs: int) -> np.ndarray:
    """Contract a row-stacked pytree Jacobian with one pytree direction."""

    leaves = jax.tree_util.tree_leaves(
        jax.tree_util.tree_map(
            lambda jac_leaf, direction_leaf: jnp.sum(
                jnp.reshape(jnp.asarray(jac_leaf), (int(n_outputs), -1))
                * jnp.reshape(jnp.asarray(direction_leaf), (1, -1)),
                axis=1,
            ),
            jacobian,
            direction,
        )
    )
    if not leaves:
        return np.zeros(int(n_outputs), dtype=float)
    total = leaves[0]
    for leaf in leaves[1:]:
        total = total + leaf
    return np.asarray(total, dtype=float)


def _controller_slot_summary_from_result(result: dict[str, Any]) -> dict[str, Any]:
    """Return compact accepted/rejected slot metadata from a replay result."""

    from vmec_jax.free_boundary_adjoint import direct_coil_accepted_trace_controller_slot_summary

    summary = result.get("controller_slot_summary")
    if isinstance(summary, dict) and summary:
        return summary
    metadata = result.get("replay_branch_metadata", {})
    if isinstance(metadata, dict) and metadata:
        return direct_coil_accepted_trace_controller_slot_summary(metadata)
    return {}


def _pytree_directional_vdot(gradient: Any, direction: Any) -> float:
    """Contract one pytree gradient with one pytree direction."""

    leaves = jax.tree_util.tree_leaves(
        jax.tree_util.tree_map(
            lambda grad_leaf, direction_leaf: jnp.sum(jnp.asarray(grad_leaf) * jnp.asarray(direction_leaf)),
            gradient,
            direction,
        )
    )
    if not leaves:
        return 0.0
    total = leaves[0]
    for leaf in leaves[1:]:
        total = total + leaf
    return float(np.asarray(total, dtype=float))


def _branch_replay_common_summary(result: dict[str, Any], *, state_only_replay: bool) -> dict[str, Any]:
    summary = {
        "available": True,
        "scope": "fixed accepted branch only; does not differentiate adaptive host branch selection",
        "uses_production_forward": bool(result["uses_production_forward"]),
        "differentiates_adaptive_controller": bool(result["differentiates_adaptive_controller"]),
        "differentiates_run_free_boundary": bool(result["differentiates_run_free_boundary"]),
        "differentiates_fixed_accepted_branch": bool(result["differentiates_fixed_accepted_branch"]),
        "replay_ad_mode": str(result["replay_ad_mode"]),
        "production_values_source": str(result.get("production_values_source", "unknown")),
        "replay_payload_source": str(result.get("replay_payload_source", "unknown")),
        "includes_payload": bool(result.get("includes_payload", True)),
        "includes_replay_graph_metadata": bool(result.get("includes_replay_graph_metadata", True)),
        "state_only_replay": bool(state_only_replay),
        "replay_option_flags": result["replay_option_flags"],
        "replay_graph_metadata": result.get("replay_graph_metadata", {}),
        "replay_branch_metadata": result.get("replay_branch_metadata", {}),
        "controller_slot_summary": _controller_slot_summary_from_result(result),
        "timings": {str(key): float(value) for key, value in result.get("timings", {}).items()},
    }
    signature = result.get("directional_jvp_signature")
    if isinstance(signature, dict):
        summary["directional_jvp_signature"] = signature
    elif isinstance(summary["replay_option_flags"], dict) and isinstance(
        summary["replay_option_flags"].get("directional_jvp_signature"), dict
    ):
        summary["directional_jvp_signature"] = summary["replay_option_flags"]["directional_jvp_signature"]
    cache_info = result.get("directional_jvp_cache_info")
    if isinstance(cache_info, dict):
        summary["directional_jvp_cache_info"] = cache_info
    elif isinstance(summary["replay_option_flags"], dict) and isinstance(
        summary["replay_option_flags"].get("directional_jvp_cache_info"), dict
    ):
        summary["directional_jvp_cache_info"] = summary["replay_option_flags"]["directional_jvp_cache_info"]
    return summary


def same_branch_scalar_result_summary(
    scalar: dict[str, Any],
    scalar_key: str,
    *,
    report: dict[str, Any],
    direction_params: Any,
    state_only_replay: bool,
) -> dict[str, Any]:
    """Return JSON-ready evidence for one fixed-branch scalar gradient replay."""

    exact_directional = _pytree_directional_vdot(scalar["grad"], direction_params)
    complete_fd_directional = float(report["objective_values"][scalar_key]["central_fd_directional"])
    return {
        **_branch_replay_common_summary(scalar, state_only_replay=state_only_replay),
        "mode": "scalar",
        "scalar_key": str(scalar["scalar_key"]),
        "value": float(scalar["value"]),
        "replay_value": float(np.asarray(scalar["replay_value"], dtype=float)),
        "base_abs_delta": float(scalar["base_abs_delta"]),
        "base_rel_delta": float(
            scalar.get(
                "base_rel_delta",
                float(scalar["base_abs_delta"])
                / max(
                    1.0,
                    abs(float(scalar["value"])),
                    abs(float(np.asarray(scalar["replay_value"], dtype=float))),
                ),
            )
        ),
        "exact_directional": float(exact_directional),
        "complete_fd_directional": complete_fd_directional,
        "abs_error": float(abs(exact_directional - complete_fd_directional)),
    }


def same_branch_vector_result_summary(
    vector: dict[str, Any],
    scalar_keys: tuple[str, ...],
    *,
    report: dict[str, Any],
    direction_params: Any,
    state_only_keys: Sequence[str],
) -> dict[str, Any]:
    """Return JSON-ready evidence for one fixed-branch vector/JVP replay."""

    if vector.get("directional_derivatives") is None:
        directionals = _vector_jacobian_directional(vector["jacobian"], direction_params, len(scalar_keys))
    else:
        directionals = [
            float(np.asarray(vector["directional_derivatives"][key], dtype=float))
            for key in scalar_keys
        ]
    replay_flags = vector.get("replay_option_flags", {})
    return {
        **_branch_replay_common_summary(
            vector,
            state_only_replay=all(key in state_only_keys for key in scalar_keys),
        ),
        "derivative_mode": str(vector.get("derivative_mode", "full_jacobian_vjp")),
        "scalar_keys": list(scalar_keys),
        "directional_jvp_fast_path": str(replay_flags.get("directional_jvp_fast_path", "none")),
        "directional_uses_fixed_coil_geometry": bool(replay_flags.get("directional_uses_fixed_coil_geometry", False)),
        "max_base_abs_delta": float(vector["max_base_abs_delta"]),
        "max_base_rel_delta": float(vector.get("max_base_rel_delta", 0.0)),
        "scalars": {
            key: {
                "value": float(vector["values"][key]),
                "replay_value": float(np.asarray(vector["replay_value_map"][key], dtype=float)),
                "base_abs_delta": float(vector["base_abs_delta"][key]),
                "base_rel_delta": float(
                    vector.get("base_rel_delta", {}).get(
                        key,
                        float(vector["base_abs_delta"][key])
                        / max(
                            1.0,
                            abs(float(vector["values"][key])),
                            abs(float(np.asarray(vector["replay_value_map"][key], dtype=float))),
                        ),
                    )
                ),
                "exact_directional": float(directionals[index]),
                "complete_fd_directional": float(report["objective_values"][key]["central_fd_directional"]),
                "abs_error": float(abs(directionals[index] - report["objective_values"][key]["central_fd_directional"])),
            }
            for index, key in enumerate(scalar_keys)
        },
    }


def run_same_branch_scalar_report_section(
    *,
    enabled: bool,
    scalar_key: str,
    scalar_uses_state_only_replay: bool,
    base_params: Any,
    report: dict[str, Any],
    report_base_values: dict[str, float],
    replay_payload: dict[str, Any] | None,
    replay_kwargs: dict[str, Any],
    ad_mode: str,
    scalar_value_fns: dict[str, Any],
    scalar_replay_fns: dict[str, Any],
    direction_params: Any,
    compact_report: dict[str, Any],
    timings: dict[str, float],
    initial_summary: dict[str, Any],
) -> dict[str, Any]:
    """Run the optional scalar same-branch replay report."""

    if not enabled:
        return initial_summary

    from vmec_jax.free_boundary_adjoint import direct_coil_run_free_boundary_branch_local_scalar_value_and_grad_jax

    scalar_replay_plan, scalar_plan_cache, scalar_plan_wall_s = same_branch_replay_plan_cache(
        report,
        replay_kwargs,
        timing_key="branch_local_scalar_replay_plan_build_wall_s",
        scope="scalar replay with unchanged accepted traces and controller policy",
    )
    compact_report["branch_local_scalar_replay_plan_cache"] = scalar_plan_cache
    if scalar_plan_wall_s is not None:
        timings["branch_local_scalar_replay_plan_build_wall_s"] = scalar_plan_wall_s
    t0 = time.perf_counter()
    scalar = direct_coil_run_free_boundary_branch_local_scalar_value_and_grad_jax(
        params=base_params,
        complete_payload=report["base"],
        scalar_key=scalar_key,
        production_values={scalar_key: report_base_values[scalar_key]},
        replay_payload=replay_payload,
        replay_plan=scalar_replay_plan,
        scalar_fn=lambda payload: {scalar_key: scalar_value_fns[scalar_key](payload)},
        replay_scalar_fn=lambda replay, payload: scalar_replay_fns[scalar_key](replay, payload),
        replay_kwargs={**replay_kwargs, "state_only_replay": scalar_uses_state_only_replay},
        replay_ad_mode=ad_mode,
        include_trace_replay_diagnostics=False,
        include_payload=False,
        include_replay_graph_metadata=False,
    )
    timings["branch_local_scalar_wall_s"] = float(time.perf_counter() - t0)
    scalar_timings = {str(key): float(value) for key, value in scalar.get("timings", {}).items()}
    for key, value in scalar_timings.items():
        timings[f"branch_local_scalar_{key}"] = value
    branch_local_scalar = same_branch_scalar_result_summary(
        scalar,
        scalar_key,
        report=report,
        direction_params=direction_params,
        state_only_replay=scalar_uses_state_only_replay,
    )
    branch_local_scalar["mode"] = "scalar"
    return branch_local_scalar


def run_same_branch_vector_report_section(
    *,
    enabled: bool,
    vector_keys: tuple[str, ...],
    vector_uses_state_only_replay: bool,
    base_params: Any,
    direction_params: Any,
    report: dict[str, Any],
    replay_kwargs: dict[str, Any],
    run_branch_local_vector: Any,
    compact_report: dict[str, Any],
    timings: dict[str, float],
    json_safe_payload_fn: Any,
    initial_vector_summary: dict[str, Any],
    initial_gate_summary: dict[str, Any],
    cache_probe: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
    """Run the optional vector/JVP same-branch replay report."""

    if not enabled:
        return initial_vector_summary, initial_gate_summary, None, None

    from vmec_jax.free_boundary_adjoint import (
        direct_coil_branch_local_scalars_report_from_complete_fd,
        direct_coil_same_branch_physical_scalar_gate_report,
    )

    current_only_coil_geometry, current_only_geometry_cache, current_only_geometry_wall_s = (
        same_branch_current_only_coil_geometry_cache(base_params, direction_params)
    )
    run_branch_local_vector.current_only_coil_geometry = current_only_coil_geometry
    compact_report["current_only_coil_geometry_cache"] = current_only_geometry_cache
    if current_only_geometry_wall_s is not None:
        timings["branch_local_current_only_coil_geometry_build_wall_s"] = current_only_geometry_wall_s
    main_vector_replay_plan, vector_plan_cache, vector_plan_wall_s = same_branch_replay_plan_cache(
        report,
        replay_kwargs,
        timing_key="branch_local_vector_replay_plan_build_wall_s",
        scope="base vector/profile replays with unchanged accepted traces and controller policy",
    )
    compact_report["branch_local_vector_replay_plan_cache"] = vector_plan_cache
    if vector_plan_wall_s is not None:
        timings["branch_local_vector_replay_plan_build_wall_s"] = vector_plan_wall_s
    t0 = time.perf_counter()
    vector = run_branch_local_vector(
        vector_keys,
        {**replay_kwargs, "state_only_replay": vector_uses_state_only_replay},
        replay_plan_for_call=main_vector_replay_plan,
    )
    timings["branch_local_vector_wall_s"] = float(time.perf_counter() - t0)
    vector_timings = {str(key): float(value) for key, value in vector.get("timings", {}).items()}
    for key, value in vector_timings.items():
        timings[f"branch_local_vector_{key}"] = value
    branch_local_vector = same_branch_vector_result_summary(
        vector,
        vector_keys,
        report=report,
        direction_params=direction_params,
        state_only_keys=STATE_ONLY_SAME_BRANCH_KEYS,
    )

    production_rtol = {
        key: 2.0e-2 if key == "qs_total" else 1.0e-2 if key == "accepted_bnormal_rms" else 5.0e-3
        for key in vector_keys
    }
    try:
        scalars_report = direct_coil_branch_local_scalars_report_from_complete_fd(
            report,
            vector,
            scalar_keys=vector_keys,
            rtol=production_rtol,
            atol={key: 5.0e-8 for key in vector_keys},
            base_value_atol={key: 2.0e-3 for key in vector_keys},
        )
        physical_gate = direct_coil_same_branch_physical_scalar_gate_report(
            report,
            scalars_report,
            scalar_keys=vector_keys,
        )
        branch_local_vector_gate = {
            "available": True,
            "passed": bool(physical_gate.get("passed", False)),
            "scope": "same-branch production-forward vector/JVP physical-scalar gate",
            "differentiates_adaptive_controller": False,
            "differentiates_run_free_boundary": False,
            "differentiates_fixed_accepted_branch": bool(
                scalars_report.get("differentiates_fixed_accepted_branch", False)
            ),
            "scalar_report": json_safe_payload_fn(scalars_report),
            "physical_scalar_gate": json_safe_payload_fn(physical_gate),
        }
    except Exception as exc:  # pragma: no cover - report artifacts should not abort the example.
        branch_local_vector_gate = {
            "available": False,
            "passed": False,
            "scope": "same-branch production-forward vector/JVP physical-scalar gate",
            "reason": f"{type(exc).__name__}: {exc}",
        }
    if bool(cache_probe):
        t0 = time.perf_counter()
        probe = run_branch_local_vector(
            vector_keys,
            {**replay_kwargs, "state_only_replay": vector_uses_state_only_replay},
            replay_plan_for_call=main_vector_replay_plan,
        )
        probe_wall_s = float(time.perf_counter() - t0)
        probe_timings = {str(key): float(value) for key, value in probe.get("timings", {}).items()}
        for key, value in probe_timings.items():
            timings[f"branch_local_vector_cache_probe_{key}"] = value
        timings["branch_local_vector_cache_probe_wall_s"] = probe_wall_s
        cache_info = probe.get("directional_jvp_cache_info", {})
        signature = probe.get("directional_jvp_signature", {})
        compact_report["branch_local_vector_current_jvp_cache_probe"] = {
            "available": True,
            "scope": "repeat same-payload current-only branch-local vector/JVP cache probe",
            "wall_s": probe_wall_s,
            "cache_hit": bool(cache_info.get("hit", False)) if isinstance(cache_info, dict) else False,
            "directional_jvp_cache_info": json_safe_payload_fn(cache_info),
            "cache_key_digest": str(signature.get("cache_key_digest", ""))
            if isinstance(signature, dict)
            else "",
            "timings": probe_timings,
        }
    return branch_local_vector, branch_local_vector_gate, branch_local_vector, main_vector_replay_plan


def nestor_profile_policy_from_results(
    results: list[dict[str, Any]],
    *,
    mode_count: int,
    min_mode_count: int,
    min_speedup: float,
) -> dict[str, Any]:
    """Decide whether matrix-free NESTOR should be promoted for this report."""

    dense = [item for item in results if item.get("nestor_solve_mode") == "dense" and item.get("available")]
    matrix_free = [
        item
        for item in results
        if item.get("nestor_solve_mode") == "matrix_free" and item.get("available")
    ]
    if not dense:
        return {
            "promote_matrix_free": False,
            "reason": "dense baseline timing is unavailable",
            "mode_count": int(mode_count),
        }
    if not matrix_free:
        return {
            "promote_matrix_free": False,
            "reason": "matrix-free timing is unavailable",
            "mode_count": int(mode_count),
        }
    dense_best_entry = min(dense, key=lambda item: float(item["wall_s"]))
    dense_best = float(dense_best_entry["wall_s"])
    mf_best_entry = min(matrix_free, key=lambda item: float(item["wall_s"]))
    mf_best = float(mf_best_entry["wall_s"])
    speedup = dense_best / mf_best if mf_best > 0.0 else np.inf
    if int(mode_count) < int(min_mode_count):
        reason = f"mode_count {int(mode_count)} below threshold {int(min_mode_count)}"
        promote = False
    elif speedup < float(min_speedup):
        reason = f"matrix-free speedup {speedup:.3g} below threshold {float(min_speedup):.3g}"
        promote = False
    else:
        reason = "matrix-free is faster beyond the configured mode-count and speedup thresholds"
        promote = True
    return {
        "promote_matrix_free": bool(promote),
        "reason": reason,
        "mode_count": int(mode_count),
        "min_mode_count": int(min_mode_count),
        "min_speedup": float(min_speedup),
        "dense_best_wall_s": dense_best,
        "matrix_free_best_wall_s": mf_best,
        "matrix_free_best_solver": str(mf_best_entry.get("nestor_operator_solver", "unknown")),
        "speedup_dense_over_matrix_free": float(speedup),
        "recommended_report_options": {
            "same_branch_report_nestor_solve_mode": "matrix_free" if promote else "dense",
            "same_branch_report_nestor_operator_solver": str(
                mf_best_entry.get("nestor_operator_solver", "gmres")
            )
            if promote
            else str(dense_best_entry.get("nestor_operator_solver", "gmres")),
            "reason": "use promoted matrix-free replay settings" if promote else "keep dense replay settings",
        },
    }


def same_branch_nestor_profile_from_vector_replay(
    *,
    args: Any,
    same_branch: bool,
    mode: str,
    report: dict[str, Any],
    mode_count: int,
    replay_mode_count_guard_triggered: bool,
    replay_mode_count_guard_reason: str,
    replay_max_mode_count: int,
    missing_vector_keys: tuple[str, ...],
    vector_keys: tuple[str, ...],
    replay_kwargs: dict[str, Any],
    vector_uses_state_only_replay: bool,
    main_vector_summary: dict[str, Any] | None,
    main_vector_replay_plan: dict[str, Any] | None,
    timings: dict[str, float],
    run_branch_local_vector: Any,
    summarize_vector_result: Any,
) -> dict[str, Any]:
    """Return optional dense-vs-matrix-free NESTOR replay profile evidence."""

    request = str(getattr(args, "same_branch_report_profile_nestor", "none")).strip().lower()
    profile: dict[str, Any] = {"enabled": False, "request": request, "reason": "not requested"}
    if request == "none":
        return profile
    profile = {"enabled": True, "request": request, "mode_count": int(mode_count), "results": [],
               "scope": "same complete-solve payload replay/JVP timings; no additional full FD solves"}
    profile_max_mode_count = int(getattr(args, "same_branch_report_profile_max_mode_count", 220))
    if request != "dense-vs-matrix-free":
        profile["reason"] = "--same-branch-report-profile-nestor must be none or dense-vs-matrix-free"
        return profile
    if not (same_branch and mode == "vector" and "base" in report and not missing_vector_keys):
        profile["reason"] = "requires same-branch vector report with all requested scalar keys"
        return profile
    if replay_mode_count_guard_triggered:
        profile.update({"reason": replay_mode_count_guard_reason, "skipped_due_to_replay_mode_count_cap": True,
                        "replay_max_mode_count": replay_max_mode_count,
                        "policy": {"promote_matrix_free": False, "reason": "profile skipped by replay mode-count cap",
                                   "mode_count": int(mode_count), "replay_max_mode_count": replay_max_mode_count}})
        return profile
    if profile_max_mode_count > 0 and int(mode_count) > profile_max_mode_count:
        profile.update({"reason": (f"mode_count {int(mode_count)} exceeds profile cap {profile_max_mode_count}; "
                                   "set --same-branch-report-profile-max-mode-count 0 to disable this guard"),
                        "skipped_due_to_mode_count_cap": True,
                        "profile_max_mode_count": profile_max_mode_count,
                        "policy": {"promote_matrix_free": False, "reason": "profile skipped by mode-count cap",
                                   "mode_count": int(mode_count), "profile_max_mode_count": profile_max_mode_count}})
        return profile

    def _result_from_summary(solve_mode: str, operator_solver: str, wall_s: float, timing_source: str,
                             summary: dict[str, Any]) -> dict[str, Any]:
        return {"available": True, "nestor_solve_mode": solve_mode, "nestor_operator_solver": operator_solver,
                "wall_s": float(wall_s), "timing_source": timing_source, "timings": summary["timings"],
                "max_base_abs_delta": float(summary["max_base_abs_delta"]),
                "max_abs_error": max(float(item["abs_error"]) for item in summary["scalars"].values()),
                "replay_option_flags": summary["replay_option_flags"]}

    profile_results: list[dict[str, Any]] = []
    profile_cases = [("dense", str(getattr(args, "same_branch_report_nestor_operator_solver", "gmres")))] + [
        ("matrix_free", solver)
        for solver in parse_profile_matrix_free_solvers(getattr(args, "same_branch_report_profile_matrix_free_solvers", None))
    ]
    for solve_mode, operator_solver in profile_cases:
        case_kwargs = {**replay_kwargs, "state_only_replay": vector_uses_state_only_replay,
                       "nestor_solve_mode": solve_mode, "nestor_operator_solver": operator_solver}
        if (main_vector_summary is not None and solve_mode == str(replay_kwargs["nestor_solve_mode"])
                and operator_solver == str(replay_kwargs["nestor_operator_solver"])):
            profile_results.append(_result_from_summary(
                solve_mode, operator_solver, float(timings.get("branch_local_vector_wall_s", 0.0)),
                "main_branch_local_vector_report", main_vector_summary))
            continue
        t0 = time.perf_counter()
        try:
            vector = run_branch_local_vector(vector_keys, case_kwargs, replay_plan_for_call=main_vector_replay_plan)
            summary = summarize_vector_result(vector, vector_keys)
            profile_results.append(_result_from_summary(
                solve_mode, operator_solver, float(time.perf_counter() - t0),
                "independent_profile_replay", summary))
        except Exception as exc:  # pragma: no cover - profile diagnostics should not abort promoted reports.
            profile_results.append({"available": False, "nestor_solve_mode": solve_mode,
                                    "nestor_operator_solver": operator_solver,
                                    "wall_s": float(time.perf_counter() - t0),
                                    "error": f"{type(exc).__name__}: {exc}"})
    profile["results"] = profile_results
    profile["policy"] = nestor_profile_policy_from_results(profile_results, mode_count=int(mode_count),
                                                           min_mode_count=int(getattr(args, "same_branch_report_profile_min_mode_count", 96)),
                                                           min_speedup=float(getattr(args, "same_branch_report_profile_min_speedup", 1.15)))
    return profile


def same_branch_rejected_slot_gate_from_vector_replay(
    *,
    requested: bool,
    same_branch: bool,
    replay_mode_count_guard_triggered: bool,
    replay_mode_count_guard_reason: str,
    mode: str,
    report: dict[str, Any],
    missing_vector_keys: tuple[str, ...],
    vector_keys: tuple[str, ...],
    replay_kwargs: dict[str, Any],
    run_branch_local_vector: Any,
    summarize_vector_result: Any,
    main_vector_replay_plan: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], float | None]:
    """Return the fixed accepted/rejected controller-slot gate artifact.

    This is a branch-local replay gate: it checks whether a fixed rejected
    controller slot can be replayed under the same fingerprint.  It does not
    claim derivatives through arbitrary host-side adaptive branch selection.
    """

    gate: dict[str, Any] = {
        "available": False,
        "requested": bool(requested),
        "passed": False,
        "reason": "not requested",
        "differentiates_adaptive_controller": False,
        "differentiates_run_free_boundary": False,
        "same_stacked_step_policy_branch": False,
    }
    if not requested:
        return gate, None
    if replay_mode_count_guard_triggered:
        gate["reason"] = replay_mode_count_guard_reason
        return gate, None
    if not (same_branch and mode == "vector" and "base" in report and not missing_vector_keys):
        gate["reason"] = "requires same-branch vector report with all requested scalar keys"
        return gate, None
    base_traces = tuple(report["base"].get("traces", ()))
    if not base_traces:
        gate["reason"] = "base complete-solve payload has no traces"
        return gate, None

    slot_vector_keys = tuple(key for key in ("aspect", "mean_iota", "lcfs_boundary_moment") if key in vector_keys)[:1]
    if not slot_vector_keys:
        slot_vector_keys = tuple(vector_keys[:1])
    slot_uses_state_only_replay = all(key in STATE_ONLY_SAME_BRANCH_KEYS for key in slot_vector_keys)
    rejected_trace = deepcopy(base_traces[-1])
    rejected_trace["step_status"] = "rejected"
    padded_traces = base_traces + (rejected_trace,)
    t0 = time.perf_counter()
    rejected_replay_plan = None
    try:
        from vmec_jax.free_boundary_adjoint import direct_coil_accepted_trace_controller_replay_plan

        inherited_contexts = (
            {}
            if main_vector_replay_plan is None
            else dict(main_vector_replay_plan.get("boundary_replay_contexts_by_shape", {}))
        )
        rejected_replay_plan = direct_coil_accepted_trace_controller_replay_plan(
            padded_traces,
            static=report["base"]["init"].static,
            use_preconditioner_policy_segments=bool(
                replay_kwargs.get("use_preconditioner_policy_segments", False)
            ),
            use_segment_preconditioner_controls=bool(
                replay_kwargs.get("use_segment_preconditioner_controls", False)
            ),
            use_stacked_step_controls=bool(replay_kwargs.get("use_stacked_step_controls", False)),
            use_accepted_only_fast_path=False,
            boundary_replay_contexts_by_shape=inherited_contexts,
        )
    except Exception:
        # The replay call below can still build its own plan.  Keep this gate
        # diagnostic non-fatal so examples do not fail solely due to plan
        # precomputation metadata.
        rejected_replay_plan = None
    rejected_vector = run_branch_local_vector(
        slot_vector_keys,
        {
            **replay_kwargs,
            "state_only_replay": slot_uses_state_only_replay,
            "traces": padded_traces,
            "use_accepted_only_fast_path": False,
        },
        include_replay_graph_metadata=False,
        replay_plan_for_call=rejected_replay_plan,
    )
    wall_s = float(time.perf_counter() - t0)
    rejected_summary = summarize_vector_result(rejected_vector, slot_vector_keys)
    rejected_metadata = rejected_summary.get("replay_branch_metadata", {})
    rejected_controller_slot_summary = rejected_summary.get("controller_slot_summary", {})
    controller_slot_fingerprint = (
        direct_coil_accepted_trace_controller_slot_fingerprint(rejected_metadata)
        if isinstance(rejected_metadata, dict)
        else {}
    )
    status_masks = rejected_metadata.get("status_masks", {}) if isinstance(rejected_metadata, dict) else {}
    status_acceptance_source = (
        rejected_metadata.get("status_acceptance_source")
        if isinstance(rejected_metadata, dict)
        else None
    )
    if status_acceptance_source is None and isinstance(status_masks, dict):
        status_acceptance_source = status_masks.get("status_acceptance_source")
    accepted_mask = np.asarray(rejected_metadata.get("accepted_mask", []), dtype=bool)
    rejected_mask = np.asarray(rejected_metadata.get("rejected_mask", []), dtype=bool)
    status_accept_mask = (
        np.asarray(status_masks.get("accept_mask", []), dtype=bool)
        if isinstance(status_masks, dict)
        else np.asarray([], dtype=bool)
    )
    if status_accept_mask.size == 0 and accepted_mask.size:
        status_accept_mask = accepted_mask
    status_derived_rejected_slot = bool(
        status_acceptance_source == "trace_step_status"
        and status_accept_mask.size
        and np.any(np.logical_not(status_accept_mask))
    )
    passed = bool(
        same_branch
        and rejected_summary["replay_option_flags"].get("use_stacked_step_controls", False)
        and not rejected_summary["replay_option_flags"].get("use_accepted_only_fast_path", True)
        and np.any(rejected_mask)
        and np.isfinite(float(rejected_summary["max_base_abs_delta"]))
        and float(rejected_summary["max_base_abs_delta"]) <= 2.0e-3
        and not bool(rejected_summary.get("differentiates_adaptive_controller", True))
        and not bool(rejected_summary.get("differentiates_run_free_boundary", True))
        and bool(rejected_summary.get("differentiates_fixed_accepted_branch", False))
    )
    return {
        "available": True,
        "requested": True,
        "passed": passed,
        "scope": (
            "fixed accepted/rejected controller-slot replay; "
            "does not differentiate adaptive host branch selection"
        ),
        "differentiates_adaptive_controller": False,
        "differentiates_run_free_boundary": False,
        "same_branch": same_branch,
        "same_stacked_step_policy_branch": bool(
            rejected_summary["replay_option_flags"].get("use_stacked_step_controls", False)
        ),
        "scalar_keys": list(slot_vector_keys),
        "full_report_scalar_keys": list(vector_keys),
        "fixed_rejected_controller_slot_present": bool(np.any(rejected_mask)),
        "fixed_rejected_controller_slots": int(np.count_nonzero(rejected_mask)),
        "status_derived_rejected_controller_slot_present": status_derived_rejected_slot,
        "status_acceptance_source": status_acceptance_source,
        "controller_slot_fingerprint": controller_slot_fingerprint,
        "directional_jvp_fast_path": str(rejected_summary.get("directional_jvp_fast_path", "none")),
        "directional_uses_fixed_coil_geometry": bool(
            rejected_summary.get("directional_uses_fixed_coil_geometry", False)
        ),
        "controller_slot_summary": rejected_controller_slot_summary,
        "replay_option_flags": rejected_summary["replay_option_flags"],
        "reused_boundary_replay_contexts": bool(
            rejected_replay_plan is not None
            and main_vector_replay_plan is not None
            and rejected_replay_plan.get("boundary_replay_contexts_by_shape")
            is not main_vector_replay_plan.get("boundary_replay_contexts_by_shape")
            and set(rejected_replay_plan.get("boundary_replay_contexts_by_shape", {}))
            >= set(main_vector_replay_plan.get("boundary_replay_contexts_by_shape", {}))
        ),
        "replay_branch_metadata": rejected_metadata,
        "max_base_abs_delta": float(rejected_summary["max_base_abs_delta"]),
        "scalars": rejected_summary["scalars"],
        "wall_s": wall_s,
    }, wall_s


def same_branch_derivative_proposal_from_report(
    report: dict[str, Any],
    objective_model: dict[str, Any],
    best: dict[str, Any] | None,
    *,
    step_size: float,
    max_base_abs_delta: float = 2.0e-3,
) -> dict[str, Any]:
    """Return one conservative derivative-assisted proposal from a report."""

    proposals = same_branch_derivative_proposals_from_report(
        report,
        objective_model,
        best,
        step_sizes=(float(step_size),),
        max_base_abs_delta=float(max_base_abs_delta),
        max_trials=1,
    )
    if proposals and proposals[0].get("available", False):
        return proposals[0]
    if proposals:
        return proposals[0]
    return {"available": False, "reason": "no same-branch derivative proposal was generated"}


def same_branch_derivative_gate_evidence(report: dict[str, Any]) -> dict[str, Any]:
    """Return compact gate evidence attached to derivative-assisted proposals."""

    vector = report.get("branch_local_vector_jacobian", {})
    replay_flags = vector.get("replay_option_flags", {}) if isinstance(vector, dict) else {}
    directional_signature = vector.get("directional_jvp_signature", {}) if isinstance(vector, dict) else {}
    if not isinstance(directional_signature, dict) and isinstance(replay_flags, dict):
        directional_signature = replay_flags.get("directional_jvp_signature", {})
    if not isinstance(directional_signature, dict):
        directional_signature = {}
    current_only_cache = report.get("current_only_coil_geometry_cache", {})
    current_jvp_cache_probe = report.get("branch_local_vector_current_jvp_cache_probe", {})
    vector_cache_info = vector.get("directional_jvp_cache_info", {}) if isinstance(vector, dict) else {}
    if not isinstance(vector_cache_info, dict) and isinstance(replay_flags, dict):
        vector_cache_info = replay_flags.get("directional_jvp_cache_info", {})
    if not isinstance(vector_cache_info, dict):
        vector_cache_info = {}
    probe_cache_info = (
        current_jvp_cache_probe.get("directional_jvp_cache_info", {})
        if isinstance(current_jvp_cache_probe, dict)
        else {}
    )
    if not isinstance(probe_cache_info, dict):
        probe_cache_info = {}
    vector_gate = report.get("branch_local_vector_gate", {})
    physical_gate = vector_gate.get("physical_scalar_gate", {}) if isinstance(vector_gate, dict) else {}
    scalar_evidence = vector.get("scalars", {}) if isinstance(vector, dict) else {}
    scalar_base_rel_delta = {
        str(key): float(value.get("base_rel_delta", np.nan))
        for key, value in scalar_evidence.items()
        if isinstance(value, dict) and "base_rel_delta" in value
    }
    rejected_slot_gate = report.get("accepted_rejected_controller_slot_gate", {})
    rejected_slot_requested = isinstance(rejected_slot_gate, dict) and bool(rejected_slot_gate.get("requested", False))
    return {
        "directional_jvp_fast_path": str(
            vector.get("directional_jvp_fast_path", replay_flags.get("directional_jvp_fast_path", "none"))
            if isinstance(vector, dict)
            else "none"
        ),
        "directional_uses_fixed_coil_geometry": bool(
            vector.get(
                "directional_uses_fixed_coil_geometry",
                replay_flags.get("directional_uses_fixed_coil_geometry", False),
            )
            if isinstance(vector, dict)
            else False
        ),
        "current_only_coil_geometry_cache_available": bool(
            isinstance(current_only_cache, dict) and current_only_cache.get("available", False)
        ),
        "current_only_coil_geometry_cache_reason": str(
            current_only_cache.get("reason", "") if isinstance(current_only_cache, dict) else ""
        ),
        "current_only_coil_geometry_source": str(
            replay_flags.get("current_only_coil_geometry_source", "")
            if isinstance(replay_flags, dict)
            else ""
        ),
        "directional_jvp_signature": dict(directional_signature),
        "directional_jvp_cache_candidate": bool(directional_signature.get("jit_cache_candidate", False)),
        "directional_jvp_cache_enabled": bool(vector_cache_info.get("enabled", False)),
        "directional_jvp_cache_hit": bool(vector_cache_info.get("hit", False)),
        "directional_jvp_cache_closure_bound": bool(vector_cache_info.get("closure_bound", False)),
        "directional_jvp_cache_info": dict(vector_cache_info),
        "current_jvp_cache_probe_available": bool(
            isinstance(current_jvp_cache_probe, dict) and current_jvp_cache_probe.get("available", False)
        ),
        "current_jvp_cache_probe_hit": bool(
            isinstance(current_jvp_cache_probe, dict) and current_jvp_cache_probe.get("cache_hit", False)
        ),
        "current_jvp_cache_probe_wall_s": float(
            current_jvp_cache_probe.get("wall_s", 0.0) if isinstance(current_jvp_cache_probe, dict) else 0.0
        ),
        "current_jvp_cache_probe_info": dict(probe_cache_info),
        "branch_local_vector_gate_available": bool(
            isinstance(vector_gate, dict) and vector_gate.get("available", False)
        ),
        "branch_local_vector_gate_passed": bool(
            isinstance(vector_gate, dict) and vector_gate.get("passed", False)
        ),
        "branch_local_vector_max_base_abs_delta": float(
            vector.get("max_base_abs_delta", np.nan) if isinstance(vector, dict) else np.nan
        ),
        "branch_local_vector_max_base_rel_delta": float(
            vector.get("max_base_rel_delta", np.nan) if isinstance(vector, dict) else np.nan
        ),
        "branch_local_scalar_base_rel_delta": scalar_base_rel_delta,
        "physical_scalar_gate_passed": bool(
            isinstance(physical_gate, dict) and physical_gate.get("passed", False)
        ),
        "accepted_rejected_controller_slot_gate_requested": bool(rejected_slot_requested),
        "accepted_rejected_controller_slot_gate_available": bool(
            isinstance(rejected_slot_gate, dict) and rejected_slot_gate.get("available", False)
        ),
        "accepted_rejected_controller_slot_gate_passed": bool(
            isinstance(rejected_slot_gate, dict) and rejected_slot_gate.get("passed", False)
        ),
        "accepted_rejected_controller_slot_scope": str(
            rejected_slot_gate.get("scope", "") if isinstance(rejected_slot_gate, dict) else ""
        ),
        "same_stacked_step_policy_branch": bool(
            isinstance(rejected_slot_gate, dict) and rejected_slot_gate.get("same_stacked_step_policy_branch", False)
        ),
        "fixed_rejected_controller_slots": int(
            rejected_slot_gate.get("fixed_rejected_controller_slots", 0)
            if isinstance(rejected_slot_gate, dict)
            else 0
        ),
        "controller_slot_summary": (
            dict(rejected_slot_gate.get("controller_slot_summary", {}))
            if isinstance(rejected_slot_gate, dict)
            and isinstance(rejected_slot_gate.get("controller_slot_summary", {}), dict)
            else {}
        ),
    }


def _unavailable_derivative_proposal(reason: str) -> list[dict[str, Any]]:
    """Return the standard unavailable-proposal payload."""

    return [{"available": False, "reason": str(reason)}]


def _unavailable_derivative_proposal_from_report(reason: str, report: dict[str, Any]) -> list[dict[str, Any]]:
    """Return an unavailable proposal while preserving report provenance."""

    return [
        {
            "available": False,
            "reason": str(reason),
            "gate_evidence": same_branch_derivative_gate_evidence(report),
        }
    ]


def _same_branch_derivative_vector_evidence(
    report: dict[str, Any],
    *,
    max_base_abs_delta: float,
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate report-level branch-local vector evidence for proposals."""

    vector = report.get("branch_local_vector_jacobian", {})
    if not bool(vector.get("available", False)):
        return None, str(vector.get("reason", "branch-local vector report unavailable"))
    same_branch = bool(report.get("branch_compatibility", {}).get("same_branch", vector.get("same_branch", False)))
    if not same_branch:
        return None, "complete-solve finite-difference branch fingerprint is not unchanged"
    if not bool(vector.get("uses_production_forward", False)):
        return None, "branch-local vector report did not use production-forward scalar values"
    if bool(vector.get("differentiates_adaptive_controller", True)):
        return None, "branch-local vector report claims adaptive-controller differentiation"
    if bool(vector.get("differentiates_run_free_boundary", True)):
        return None, "branch-local vector report claims run_free_boundary differentiation"
    if not bool(vector.get("differentiates_fixed_accepted_branch", False)):
        return None, "branch-local vector report does not differentiate a fixed accepted branch"
    replay_ad_mode = str(vector.get("replay_ad_mode", "")).strip().lower()
    if replay_ad_mode != "direct":
        return None, "branch-local proposal requires direct JVP replay_ad_mode"
    derivative_mode = str(vector.get("derivative_mode", "")).strip().lower()
    if derivative_mode != "directional_jvp":
        return None, "branch-local proposal requires directional_jvp derivative_mode"
    vector_gate = report.get("branch_local_vector_gate")
    if isinstance(vector_gate, dict) and bool(vector_gate.get("available", False)):
        if not bool(vector_gate.get("passed", False)):
            return None, "branch-local vector gate did not pass"
        physical_gate = vector_gate.get("physical_scalar_gate", {})
        if isinstance(physical_gate, dict) and not bool(physical_gate.get("passed", False)):
            return None, "branch-local physical-scalar gate did not pass"

    report_base_delta = float(vector.get("max_base_abs_delta", np.inf))
    if not np.isfinite(report_base_delta):
        return None, "branch-local vector report has non-finite replay base delta"
    if report_base_delta > float(max_base_abs_delta):
        return (
            None,
            (
                f"branch-local replay base delta {report_base_delta:.3e} exceeds proposal cap "
                f"{float(max_base_abs_delta):.3e}"
            ),
        )

    rejected_slot_gate = report.get("accepted_rejected_controller_slot_gate")
    if isinstance(rejected_slot_gate, dict) and bool(rejected_slot_gate.get("requested", False)):
        if not bool(rejected_slot_gate.get("available", False)):
            return None, str(
                rejected_slot_gate.get(
                    "reason",
                    "requested accepted/rejected controller-slot gate is unavailable",
                )
            )
        if not bool(rejected_slot_gate.get("passed", False)):
            return None, "accepted/rejected controller-slot gate did not pass"

    return {
        "vector": vector,
        "replay_ad_mode": replay_ad_mode,
        "derivative_mode": derivative_mode,
        "report_base_delta": report_base_delta,
        "report_base_rel_delta": float(vector.get("max_base_rel_delta", np.nan)),
    }, None


def _validated_branch_local_scalar(
    scalars: Mapping[str, Any],
    key: str,
    weight: float,
    *,
    max_base_abs_delta: float,
    omitted_terms: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """Return validated scalar evidence for one weighted objective term."""

    if float(weight) == 0.0:
        return None
    scalar = scalars.get(key)
    if scalar is None:
        omitted_terms[key] = {
            "weight": float(weight),
            "reason": "not included in branch-local vector/JVP report",
        }
        return None
    value = float(scalar.get("value", np.nan))
    deriv = float(scalar.get("exact_directional", np.nan))
    base_delta = float(scalar.get("base_abs_delta", 0.0))
    base_rel_delta = float(scalar.get("base_rel_delta", np.nan))
    if not (np.isfinite(value) and np.isfinite(deriv) and np.isfinite(base_delta)):
        raise ValueError(f"non-finite branch-local scalar evidence for {key}")
    if base_delta > float(max_base_abs_delta):
        raise ValueError(
            f"branch-local scalar {key} base delta {base_delta:.3e} exceeds proposal cap "
            f"{float(max_base_abs_delta):.3e}"
        )
    return {
        "value": value,
        "exact_directional": deriv,
        "base_abs_delta": base_delta,
        "base_rel_delta": base_rel_delta,
    }


def _same_branch_proposal_directional_terms(
    vector: Mapping[str, Any],
    objective_model: Mapping[str, Any],
    *,
    max_base_abs_delta: float,
) -> tuple[dict[str, Any] | None, str | None]:
    """Assemble weighted objective-direction evidence from branch-local scalars."""

    scalars = vector.get("scalars", {})
    contributions: dict[str, dict[str, float]] = {}
    omitted_terms: dict[str, dict[str, Any]] = {}
    directional = 0.0

    if float(objective_model.get("residual_weight", 0.0)) != 0.0:
        omitted_terms["residual_proxy"] = {
            "weight": float(objective_model.get("residual_weight", 0.0)),
            "reason": (
                "not included in branch-local vector/JVP report; the complete "
                "free-boundary solve remains acceptance authority"
            ),
        }

    try:
        qs_scalar = _validated_branch_local_scalar(
            scalars,
            "qs_total",
            float(objective_model.get("qs_weight", 0.0)),
            max_base_abs_delta=max_base_abs_delta,
            omitted_terms=omitted_terms,
        )
        aspect_scalar = _validated_branch_local_scalar(
            scalars,
            "aspect",
            float(objective_model.get("aspect_weight", 0.0)),
            max_base_abs_delta=max_base_abs_delta,
            omitted_terms=omitted_terms,
        )
        iota_scalar = _validated_branch_local_scalar(
            scalars,
            "mean_iota",
            float(objective_model.get("iota_weight", 0.0)),
            max_base_abs_delta=max_base_abs_delta,
            omitted_terms=omitted_terms,
        )
    except ValueError as exc:
        return None, str(exc)

    if qs_scalar is not None:
        deriv = float(qs_scalar["exact_directional"])
        contribution = float(objective_model.get("qs_weight", 0.0)) * deriv
        contributions["qs_total"] = {
            "exact_directional": deriv,
            "base_abs_delta": float(qs_scalar["base_abs_delta"]),
            "base_rel_delta": float(qs_scalar["base_rel_delta"]),
            "contribution": contribution,
        }
        directional += contribution

    if aspect_scalar is not None:
        value = float(aspect_scalar["value"])
        deriv = float(aspect_scalar["exact_directional"])
        target = float(objective_model.get("target_aspect", value))
        contribution = 2.0 * float(objective_model.get("aspect_weight", 0.0)) * (value - target) * deriv
        contributions["aspect"] = {
            "value": value,
            "target": target,
            "exact_directional": deriv,
            "base_abs_delta": float(aspect_scalar["base_abs_delta"]),
            "base_rel_delta": float(aspect_scalar["base_rel_delta"]),
            "contribution": contribution,
        }
        directional += contribution

    if iota_scalar is not None:
        value = float(iota_scalar["value"])
        deriv = float(iota_scalar["exact_directional"])
        target = float(objective_model.get("target_iota", value))
        contribution = 2.0 * float(objective_model.get("iota_weight", 0.0)) * (value - target) * deriv
        contributions["mean_iota"] = {
            "value": value,
            "target": target,
            "exact_directional": deriv,
            "base_abs_delta": float(iota_scalar["base_abs_delta"]),
            "base_rel_delta": float(iota_scalar["base_rel_delta"]),
            "contribution": contribution,
        }
        directional += contribution

    if not contributions:
        return None, "no report scalars map to the objective terms"
    if not np.isfinite(directional):
        return None, "non-finite directional derivative"
    if directional == 0.0:
        return None, "zero directional derivative"
    return {
        "directional": float(directional),
        "contributions": contributions,
        "omitted_terms": omitted_terms,
    }, None


def same_branch_derivative_proposals_from_report(
    report: dict[str, Any],
    objective_model: dict[str, Any],
    best: dict[str, Any] | None,
    *,
    step_sizes: Sequence[float],
    max_base_abs_delta: float = 2.0e-3,
    max_trials: int | None = None,
) -> list[dict[str, Any]]:
    """Return bounded derivative-assisted proposals from one same-branch report.

    Each proposal uses the same validated fixed-accepted-branch directional JVP
    and differs only by optimizer-coordinate step length.  Every returned
    ``trial_x`` is still a suggestion; the production complete solve remains
    the sole acceptance authority.
    """

    if best is None or "x" not in best:
        return _unavailable_derivative_proposal("no best point is available")
    raw_step_sizes = [float(step) for step in step_sizes]
    step_sizes = [step for step in raw_step_sizes if np.isfinite(step) and step > 0.0]
    if not step_sizes:
        return _unavailable_derivative_proposal("no positive finite proposal step sizes were requested")
    if max_trials is not None and int(max_trials) > 0:
        step_sizes = step_sizes[: int(max_trials)]
    evidence, reason = _same_branch_derivative_vector_evidence(
        report,
        max_base_abs_delta=float(max_base_abs_delta),
    )
    if evidence is None:
        return _unavailable_derivative_proposal_from_report(str(reason), report)
    direction_terms, reason = _same_branch_proposal_directional_terms(
        evidence["vector"],
        objective_model,
        max_base_abs_delta=float(max_base_abs_delta),
    )
    if direction_terms is None:
        return _unavailable_derivative_proposal_from_report(str(reason), report)

    direction_x = np.asarray(report.get("direction_x", []), dtype=float)
    x_best = np.asarray(best["x"], dtype=float)
    if direction_x.shape != x_best.shape:
        return _unavailable_derivative_proposal(
            f"direction_x shape {direction_x.shape} does not match best x shape {x_best.shape}"
        )

    gate_evidence = same_branch_derivative_gate_evidence(report)
    directional = float(direction_terms["directional"])
    contributions = direction_terms["contributions"]
    omitted_terms = direction_terms["omitted_terms"]
    proposals = []
    for trial_index, step_size in enumerate(step_sizes):
        alpha = -float(step_size) * float(np.sign(directional))
        trial_x = x_best + alpha * direction_x
        proposals.append(
            {
                "available": True,
                "scope": "fixed accepted-branch directional proposal; complete solve decides acceptance",
                "same_branch": True,
                "uses_production_forward": True,
                "replay_ad_mode": evidence["replay_ad_mode"],
                "derivative_mode": evidence["derivative_mode"],
                "differentiates_adaptive_controller": False,
                "differentiates_run_free_boundary": False,
                "differentiates_fixed_accepted_branch": True,
                "complete_solve_acceptance_authority": True,
                "max_base_abs_delta": evidence["report_base_delta"],
                "max_base_rel_delta": evidence["report_base_rel_delta"],
                "max_base_abs_delta_allowed": float(max_base_abs_delta),
                "directional_derivative": float(directional),
                "contributions": contributions,
                "gate_evidence": gate_evidence,
                "objective_terms_used": sorted(contributions),
                "objective_terms_omitted": omitted_terms,
                "alpha": float(alpha),
                "step_size": float(step_size),
                "trial_index": int(trial_index),
                "n_requested_trials": int(len(step_sizes)),
                "direction_x": direction_x.tolist(),
                "base_x": x_best.tolist(),
                "trial_x": trial_x.tolist(),
            }
        )
    return proposals
