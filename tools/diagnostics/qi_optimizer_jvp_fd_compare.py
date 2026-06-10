#!/usr/bin/env python
"""Compare QI exact-optimizer JVPs against central finite differences.

This is a diagnostic for QI cleanup stalls: if direct fixed-boundary
perturbations change the QI metric but the matrix-free optimizer does not move,
run this tool at the same input deck and coefficient direction.

Example
-------
Compare the VMEC input coefficient ``RBC(n=0,m=1)`` (optimizer kind ``rc``):

.. code-block:: bash

   PYTHONPATH=. JAX_PLATFORMS=cuda python tools/diagnostics/qi_optimizer_jvp_fd_compare.py \
     --input examples/data/input.nfp2_QI \
     --kind rc --m 1 --n 0 --max-mode 3 --solver-device gpu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import vmec_jax as vj
from vmec_jax._compat import enable_x64
from vmec_jax.optimization_workflow import build_quasi_isodynamic_objective_stage


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="VMEC input deck to diagnose.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional JSON report path.")
    parser.add_argument("--kind", choices=("rc", "rs", "zc", "zs"), default="rc")
    parser.add_argument("--m", type=int, default=1, help="Optimizer poloidal mode number.")
    parser.add_argument("--n", type=int, default=0, help="Optimizer toroidal mode number.")
    parser.add_argument("--max-mode", type=int, default=3)
    parser.add_argument("--min-vmec-mode", type=int, default=6)
    parser.add_argument("--epsilon", type=float, default=1.0e-5)
    parser.add_argument("--inner-max-iter", type=int, default=450)
    parser.add_argument("--inner-ftol", type=float, default=1.0e-9)
    parser.add_argument("--trial-max-iter", type=int, default=450)
    parser.add_argument("--trial-ftol", type=float, default=1.0e-9)
    parser.add_argument("--solver-device", choices=("cpu", "gpu", "none", "default"), default="default")
    parser.add_argument("--exact-path", choices=("auto", "tape", "scan"), default="auto")
    parser.add_argument("--mboz", type=int, default=7)
    parser.add_argument("--nboz", type=int, default=7)
    parser.add_argument("--nphi", type=int, default=61)
    parser.add_argument("--nalpha", type=int, default=13)
    parser.add_argument("--n-bounce", type=int, default=17)
    parser.add_argument("--surfaces", type=str, default="0.1,0.28,0.46,0.64,0.82,1.0")
    parser.add_argument(
        "--dense-jacobian",
        action="store_true",
        help="Also materialize the dense exact Jacobian and compare the selected column.",
    )
    parser.add_argument(
        "--state-fd",
        action="store_true",
        help=(
            "Also compare the accepted packed-state JVP against central finite "
            "differences of complete accepted VMEC solves."
        ),
    )
    parser.add_argument(
        "--dynamic-axis-tangent",
        action="store_true",
        help=(
            "Also replay one tangent whose initial-state JVP differentiates "
            "through VMEC's inferred axis guess instead of freezing it."
        ),
    )
    parser.add_argument(
        "--state-fd-iters",
        type=str,
        default="",
        help=(
            "Optional comma/space-separated accepted-solve iteration counts "
            "for state-JVP-vs-state-FD localization, e.g. '1,2,5,10'."
        ),
    )
    return parser.parse_args()


def _surfaces(text: str) -> np.ndarray:
    values = [float(item) for item in str(text).replace(",", " ").split()]
    if not values:
        raise ValueError("--surfaces must contain at least one value.")
    return np.asarray(values, dtype=float)


def _int_list(text: str) -> list[int]:
    values = [int(item) for item in str(text).replace(",", " ").split() if item.strip()]
    return [value for value in values if value > 0]


def _spec_index(specs, *, kind: str, m: int, n: int) -> int:
    matches = [
        i
        for i, spec in enumerate(specs)
        if spec.kind == kind and int(spec.m) == int(m) and int(spec.n) == int(n)
    ]
    if not matches:
        available = ", ".join(f"{spec.kind}(m={spec.m},n={spec.n})" for spec in specs[:40])
        raise ValueError(
            f"No active parameter found for {kind}(m={m}, n={n}). "
            f"First active parameters: {available}"
        )
    return int(matches[0])


def _packed_block_report(layout, lhs, rhs) -> dict[str, dict[str, float]]:
    """Return per-state-block mismatch diagnostics for two packed vectors."""

    from vmec_jax.state import unpack_state

    lhs_state = unpack_state(lhs, layout)
    rhs_state = unpack_state(rhs, layout)
    report: dict[str, dict[str, float]] = {}
    for name in ("Rcos", "Rsin", "Zcos", "Zsin", "Lcos", "Lsin"):
        lhs_block = np.asarray(getattr(lhs_state, name), dtype=float)
        rhs_block = np.asarray(getattr(rhs_state, name), dtype=float)
        diff = lhs_block - rhs_block
        lhs_norm = float(np.linalg.norm(lhs_block))
        rhs_norm = float(np.linalg.norm(rhs_block))
        diff_norm = float(np.linalg.norm(diff))
        dot = float(np.vdot(lhs_block.reshape(-1), rhs_block.reshape(-1)))
        report[name] = {
            "lhs_norm": lhs_norm,
            "rhs_norm": rhs_norm,
            "diff_norm": diff_norm,
            "relative_diff_norm": diff_norm / max(lhs_norm, rhs_norm, np.finfo(float).eps),
            "max_abs_diff": float(np.max(np.abs(diff))) if diff.size else 0.0,
            "cosine_similarity": dot / max(lhs_norm * rhs_norm, np.finfo(float).eps),
        }
    return report


def _tape_payload_report(tape) -> dict[str, object]:
    """Summarize an exact replay tape without serializing large arrays."""

    step_traces = tuple(getattr(tape, "step_traces", ()) or ())
    first_keys = sorted(step_traces[0].keys()) if step_traces else []
    explicit_cache_controls = any(
        ("constraint_cache_update" in trace) or ("precond_cache_update" in trace)
        for trace in step_traces
    )
    stacked_step_traces = getattr(tape, "stacked_step_traces", None)
    static_flags = getattr(tape, "step_trace_static_flags", None)
    dynamic_base_carries_stacked = getattr(tape, "dynamic_base_carries_stacked", None)
    dynamic_initial_carry = getattr(tape, "dynamic_initial_carry", None)
    diagnostics = dict(getattr(tape, "diagnostics", {}) or {})
    timing = diagnostics.get("timing", {})
    if isinstance(timing, dict):
        diagnostics["timing"] = {
            str(key): float(value)
            for key, value in timing.items()
            if str(key).startswith("tape_") or str(key).startswith("solve_")
        }
    return {
        "jvp_only": bool(getattr(tape, "jvp_only", False)),
        "step_trace_count": int(len(step_traces)),
        "first_step_trace_keys": first_keys,
        "stacked_step_trace_keys": sorted(stacked_step_traces.keys()) if isinstance(stacked_step_traces, dict) else [],
        "step_trace_static_flag_keys": sorted(static_flags.keys()) if isinstance(static_flags, dict) else [],
        "constraint_static_flag_summary": {
            str(key): {
                "shape": list(np.shape(value)),
                "norm": float(np.linalg.norm(np.asarray(value, dtype=float))),
            }
            for key, value in (static_flags.items() if isinstance(static_flags, dict) else ())
            if str(key).startswith("constraint_")
        },
        "has_explicit_cache_controls": bool(explicit_cache_controls),
        "has_stacked_step_traces": stacked_step_traces is not None,
        "has_dynamic_base_carries_stacked": dynamic_base_carries_stacked is not None,
        "has_dynamic_initial_carry": dynamic_initial_carry is not None,
        "dynamic_initial_carry_len": int(len(dynamic_initial_carry or ())),
        "compact_diagnostics": diagnostics,
    }


def _optimizer_profile_report(optimizer, *, prefix: str = "") -> dict[str, object]:
    """Return exact-tape/JVP profile entries relevant to this diagnostic."""

    profile = optimizer._profile_dump()
    if not prefix:
        return profile
    return {
        key: value
        for key, value in profile.items()
        if key.startswith(prefix)
        or "jvp" in key
        or "replay" in key
        or "tape" in key
        or "jacobian" in key
    }


def _trace_control_fd_report(base_traces, plus_traces, minus_traces, eps: float) -> list[dict[str, object]]:
    """Return per-step central-FD diagnostics for recorded controller controls."""

    scalar_keys = (
        "time_step",
        "dt_eff",
        "b1",
        "fac",
        "force_scale",
        "flip_sign",
        "fsq_prev_before",
        "constraint_cache_update",
        "precond_cache_update",
        "max_update_rms_pre",
        "max_coeff_delta_rms_pre",
        "lambda_update_scale",
        "update_rms_preclip",
        "update_rms_postclip",
        "update_rms_scale",
    )
    array_keys = (
        "inv_tau_before",
        "vRcc_before",
        "vRss_before",
        "vZsc_before",
        "vZcs_before",
        "vLsc_before",
        "vLcs_before",
        "constraint_tcon",
        "constraint_precond_active",
        "constraint_tcon_active",
        "constraint_rcon0",
        "constraint_zcon0",
    )
    tuple_array_keys = ("constraint_precond_diag",)
    nsteps = min(len(base_traces), len(plus_traces), len(minus_traces))
    report: list[dict[str, object]] = []
    for step in range(nsteps):
        base = base_traces[step]
        plus = plus_traces[step]
        minus = minus_traces[step]
        item: dict[str, object] = {
            "step": int(step + 1),
            "branch": str(base.get("branch", "")),
            "step_status": str(base.get("step_status", "")),
            "restart_path": str(base.get("restart_path", "")),
        }
        scalar_report: dict[str, dict[str, float]] = {}
        for key in scalar_keys:
            if key not in base or key not in plus or key not in minus:
                continue
            if base.get(key) is None or plus.get(key) is None or minus.get(key) is None:
                continue
            base_value = float(base[key])
            fd_value = (float(plus[key]) - float(minus[key])) / (2.0 * float(eps))
            scalar_report[key] = {
                "base": base_value,
                "fd": fd_value,
                "plus": float(plus[key]),
                "minus": float(minus[key]),
                "abs_delta": abs(float(plus[key]) - float(minus[key])),
            }
        if scalar_report:
            item["scalars"] = scalar_report
        array_report: dict[str, dict[str, float]] = {}
        for key in array_keys:
            if key not in base or key not in plus or key not in minus:
                continue
            if base.get(key) is None or plus.get(key) is None or minus.get(key) is None:
                continue
            base_arr = np.asarray(base[key], dtype=float)
            fd_arr = (np.asarray(plus[key], dtype=float) - np.asarray(minus[key], dtype=float)) / (
                2.0 * float(eps)
            )
            array_report[key] = {
                "base_norm": float(np.linalg.norm(base_arr)),
                "fd_norm": float(np.linalg.norm(fd_arr)),
                "fd_max_abs": float(np.max(np.abs(fd_arr))) if fd_arr.size else 0.0,
            }
        if array_report:
            item["arrays"] = array_report
        tuple_report: dict[str, list[dict[str, float]]] = {}
        for key in tuple_array_keys:
            if key not in base or key not in plus or key not in minus:
                continue
            if base.get(key) is None or plus.get(key) is None or minus.get(key) is None:
                continue
            pieces: list[dict[str, float]] = []
            for base_piece, plus_piece, minus_piece in zip(base[key], plus[key], minus[key], strict=False):
                base_arr = np.asarray(base_piece, dtype=float)
                fd_arr = (np.asarray(plus_piece, dtype=float) - np.asarray(minus_piece, dtype=float)) / (
                    2.0 * float(eps)
                )
                pieces.append(
                    {
                        "base_norm": float(np.linalg.norm(base_arr)),
                        "fd_norm": float(np.linalg.norm(fd_arr)),
                        "fd_max_abs": float(np.max(np.abs(fd_arr))) if fd_arr.size else 0.0,
                    }
                )
            tuple_report[key] = pieces
        if tuple_report:
            item["tuple_arrays"] = tuple_report
        report.append(item)
    return report


def _dynamic_replay_base_report(base_traces, static) -> list[dict[str, object]]:
    """Compare differentiable dynamic replay primals against recorded host trace."""

    from vmec_jax.discrete_adjoint import (
        _dynamic_replay_initial_carry,
        _dynamic_fsq1_from_force_channels,
        _packed_dynamic_replay_step_from_carry,
        _static_flags_from_replay_step_traces,
        _trace_preconditioner_use_lax_tridi,
        _trace_preconditioner_use_precomputed_tridi,
        preconditioned_force_channels_from_raw_forces,
        raw_force_residual_from_state,
        state_dependent_preconditioner_from_forces,
    )
    from vmec_jax.state import pack_state, unpack_state

    if not base_traces:
        return []
    static_flags = _static_flags_from_replay_step_traces(tuple(base_traces))
    carry = _dynamic_replay_initial_carry(base_traces[0])
    report: list[dict[str, object]] = []
    velocity_slots = {
        "vRcc_after": 3,
        "vRss_after": 4,
        "vRsc_after": 5,
        "vRcs_after": 6,
        "vZsc_after": 7,
        "vZcs_after": 8,
        "vZcc_after": 9,
        "vZss_after": 10,
        "vLsc_after": 11,
        "vLcs_after": 12,
        "vLcc_after": 13,
        "vLss_after": 14,
    }
    for step, trace in enumerate(base_traces, start=1):
        layout = trace["state_pre"].layout
        state_pre = unpack_state(carry[0], layout)
        wout_like = trace["wout_like"] if "wout_like" in trace else static_flags["wout_like"]
        trig = trace["trig"] if "trig" in trace else static_flags["trig"]
        w_mode_mn = trace["w_mode_mn"] if "w_mode_mn" in trace else static_flags["w_mode_mn"]
        lambda_update_scale = (
            trace["lambda_update_scale"] if "lambda_update_scale" in trace else static_flags["lambda_update_scale"]
        )
        residual_out = raw_force_residual_from_state(
            state_pre,
            static,
            wout_like=wout_like,
            trig=trig,
            apply_lforbal=static_flags["apply_lforbal"],
            include_edge_residual=static_flags["include_edge_residual"],
            apply_m1_constraints=static_flags["apply_m1_constraints"],
            zero_m1=trace["zero_m1"],
            constraint_tcon0=trace.get("constraint_tcon0"),
            constraint_tcon=trace.get("constraint_tcon"),
            constraint_precond_diag=trace.get("constraint_precond_diag"),
            constraint_precond_active=trace.get("constraint_precond_active"),
            constraint_tcon_active=trace.get("constraint_tcon_active"),
            constraint_rcon0=trace.get("constraint_rcon0"),
            constraint_zcon0=trace.get("constraint_zcon0"),
            freeb_bsqvac_half=trace.get("freeb_bsqvac_half", None),
            freeb_pres_scale=trace.get("freeb_pres_scale", None) if "freeb_pres_scale" in trace else static_flags.get("freeb_pres_scale", None),
        )
        tridi_policy = _trace_preconditioner_use_precomputed_tridi(trace, static_flags)
        lax_tridi_policy = _trace_preconditioner_use_lax_tridi(trace, static_flags)
        preconditioner_out = state_dependent_preconditioner_from_forces(
            k=residual_out["k"],
            static=static,
            trig=trig,
            dtype=np.asarray(carry[0]).dtype,
            jmax_override=int(static_flags["precond_jmax"]),
            w_mode_mn=w_mode_mn,
            use_precomputed=tridi_policy,
            use_lax_tridi=lax_tridi_policy,
        )
        force_out = preconditioned_force_channels_from_raw_forces(
            frzl=residual_out["frzl"],
            mats=preconditioner_out["mats"],
            jmax=preconditioner_out["jmax"],
            cfg=static.cfg,
            lam_prec=preconditioner_out["lam_prec"],
            w_mode_mn=preconditioner_out["w_mode_mn"],
            lambda_update_scale=lambda_update_scale,
            use_precomputed=tridi_policy,
            use_lax_tridi=lax_tridi_policy,
        )
        fsq1 = _dynamic_fsq1_from_force_channels(
            state_pre=state_pre,
            static=static,
            vmec2000_control=bool(static_flags["vmec2000_control"]),
            frzl_pre=force_out["frzl_pre"],
        )
        carry = _packed_dynamic_replay_step_from_carry(
            carry,
            trace,
            static=static,
            static_flags=static_flags,
            preconditioner_jmax_override=int(static_flags["precond_jmax"]),
            preconditioner_use_precomputed_tridi=static_flags.get(
                "preconditioner_use_precomputed_tridi",
                None,
            ),
            preconditioner_use_lax_tridi=static_flags.get(
                "preconditioner_use_lax_tridi",
                None,
            ),
        )
        item: dict[str, object] = {
            "step": int(step),
            "step_status": str(trace.get("step_status", "")),
            "restart_path": str(trace.get("restart_path", "")),
            "fsq1_replay": float(np.asarray(fsq1)),
        }
        force_report: dict[str, dict[str, float]] = {}
        block_sources = {
            "raw": (
                residual_out["frzl"],
                {
                    "frcc": "frzl_frcc",
                    "frss": "frzl_frss",
                    "frsc": "frzl_frsc",
                    "frcs": "frzl_frcs",
                    "fzsc": "frzl_fzsc",
                    "fzcs": "frzl_fzcs",
                    "fzcc": "frzl_fzcc",
                    "fzss": "frzl_fzss",
                    "flsc": "frzl_flsc",
                    "flcs": "frzl_flcs",
                    "flcc": "frzl_flcc",
                    "flss": "frzl_flss",
                },
            ),
            "rz": (
                force_out["frzl_rz"],
                {
                    "frcc": "frzl_rz_frcc",
                    "frss": "frzl_rz_frss",
                    "frsc": "frzl_rz_frsc",
                    "frcs": "frzl_rz_frcs",
                    "fzsc": "frzl_rz_fzsc",
                    "fzcs": "frzl_rz_fzcs",
                    "fzcc": "frzl_rz_fzcc",
                    "fzss": "frzl_rz_fzss",
                    "flsc": "frzl_rz_flsc",
                    "flcs": "frzl_rz_flcs",
                    "flcc": "frzl_rz_flcc",
                    "flss": "frzl_rz_flss",
                },
            ),
        }
        block_report: dict[str, dict[str, dict[str, float]]] = {}
        for source_name, (source_blocks, trace_keys) in block_sources.items():
            source_report: dict[str, dict[str, float]] = {}
            for block_name, trace_key in trace_keys.items():
                if trace_key not in trace or trace.get(trace_key) is None:
                    continue
                replay_block = np.asarray(getattr(source_blocks, block_name), dtype=float)
                recorded_block = np.asarray(trace[trace_key], dtype=float)
                diff = replay_block - recorded_block
                diff_norm = float(np.linalg.norm(diff))
                if diff_norm <= 0.0:
                    continue
                replay_norm = float(np.linalg.norm(replay_block))
                recorded_norm = float(np.linalg.norm(recorded_block))
                source_report[block_name] = {
                    "replay_norm": replay_norm,
                    "recorded_norm": recorded_norm,
                    "diff_norm": diff_norm,
                    "relative_diff_norm": diff_norm / max(replay_norm, recorded_norm, np.finfo(float).eps),
                    "max_abs_diff": float(np.max(np.abs(diff))) if diff.size else 0.0,
                }
            if source_report:
                block_report[source_name] = source_report
        if block_report:
            item["force_blocks"] = block_report
        for key in (
            "frcc_u",
            "frss_u",
            "frsc_u",
            "frcs_u",
            "fzsc_u",
            "fzcs_u",
            "fzcc_u",
            "fzss_u",
            "flsc_u",
            "flcs_u",
            "flcc_u",
            "flss_u",
        ):
            if key not in trace or trace.get(key) is None or key not in force_out:
                continue
            replay_force = np.asarray(force_out[key], dtype=float)
            recorded_force = np.asarray(trace[key], dtype=float)
            diff = replay_force - recorded_force
            diff_norm = float(np.linalg.norm(diff))
            if diff_norm <= 0.0:
                continue
            replay_norm = float(np.linalg.norm(replay_force))
            recorded_norm = float(np.linalg.norm(recorded_force))
            force_report[key] = {
                "replay_norm": replay_norm,
                "recorded_norm": recorded_norm,
                "diff_norm": diff_norm,
                "relative_diff_norm": diff_norm / max(replay_norm, recorded_norm, np.finfo(float).eps),
                "max_abs_diff": float(np.max(np.abs(diff))) if diff.size else 0.0,
            }
        if force_report:
            item["forces"] = force_report
        if "state_post" in trace:
            replay_state = np.asarray(carry[0], dtype=float)
            recorded_state = np.asarray(pack_state(trace["state_post"]), dtype=float)
            state_diff = replay_state - recorded_state
            item["state_post"] = {
                "replay_norm": float(np.linalg.norm(replay_state)),
                "recorded_norm": float(np.linalg.norm(recorded_state)),
                "diff_norm": float(np.linalg.norm(state_diff)),
                "relative_diff_norm": float(np.linalg.norm(state_diff))
                / max(
                    float(np.linalg.norm(replay_state)),
                    float(np.linalg.norm(recorded_state)),
                    np.finfo(float).eps,
                ),
                "max_abs_diff": float(np.max(np.abs(state_diff))) if state_diff.size else 0.0,
            }
        velocities: dict[str, dict[str, float]] = {}
        for key, slot in velocity_slots.items():
            if key not in trace or trace.get(key) is None:
                continue
            replay_value = np.asarray(carry[slot], dtype=float)
            recorded_value = np.asarray(trace[key], dtype=float)
            diff = replay_value - recorded_value
            diff_norm = float(np.linalg.norm(diff))
            if diff_norm <= 0.0:
                continue
            replay_norm = float(np.linalg.norm(replay_value))
            recorded_norm = float(np.linalg.norm(recorded_value))
            velocities[key] = {
                "replay_norm": replay_norm,
                "recorded_norm": recorded_norm,
                "diff_norm": diff_norm,
                "relative_diff_norm": diff_norm / max(replay_norm, recorded_norm, np.finfo(float).eps),
                "max_abs_diff": float(np.max(np.abs(diff))) if diff.size else 0.0,
            }
        if velocities:
            item["velocities"] = velocities
        report.append(item)
    return report


def _dynamic_stacked_replay_base_report(tape, base_traces, static) -> list[dict[str, object]]:
    """Compare compact stacked dynamic replay primals against recorded traces."""

    from vmec_jax.discrete_adjoint import (
        _packed_dynamic_replay_step_from_carry,
        _trace_preconditioner_use_lax_tridi,
        _trace_preconditioner_use_precomputed_tridi,
    )
    from vmec_jax.state import pack_state

    stacked = getattr(tape, "stacked_step_traces", None)
    static_flags = getattr(tape, "step_trace_static_flags", None)
    carry = getattr(tape, "dynamic_initial_carry", None)
    if stacked is None or static_flags is None or carry is None:
        return []
    n_steps = min(len(base_traces), int(next(iter(stacked.values())).shape[0]))
    report: list[dict[str, object]] = []
    for idx in range(n_steps):
        trace = {key: value[idx] for key, value in stacked.items()}
        active = bool(np.asarray(trace.get("active", True)))
        if active:
            carry = _packed_dynamic_replay_step_from_carry(
                carry,
                trace,
                static=static,
                static_flags=static_flags,
                preconditioner_jmax_override=int(static_flags["precond_jmax"]),
                preconditioner_use_precomputed_tridi=_trace_preconditioner_use_precomputed_tridi(
                    trace,
                    static_flags,
                ),
                preconditioner_use_lax_tridi=_trace_preconditioner_use_lax_tridi(
                    trace,
                    static_flags,
                ),
            )
        recorded_state = np.asarray(pack_state(base_traces[idx]["state_post"]), dtype=float)
        replay_state = np.asarray(carry[0], dtype=float)
        diff = replay_state - recorded_state
        replay_norm = float(np.linalg.norm(replay_state))
        recorded_norm = float(np.linalg.norm(recorded_state))
        diff_norm = float(np.linalg.norm(diff))
        report.append(
            {
                "step": int(idx + 1),
                "active": active,
                "state_post": {
                    "replay_norm": replay_norm,
                    "recorded_norm": recorded_norm,
                    "diff_norm": diff_norm,
                    "relative_diff_norm": diff_norm / max(replay_norm, recorded_norm, np.finfo(float).eps),
                    "max_abs_diff": float(np.max(np.abs(diff))) if diff.size else 0.0,
                },
            }
        )
    return report


def _dynamic_replay_jvp_trace_report(
    base_traces,
    plus_traces,
    minus_traces,
    static,
    initial_tangent,
    eps: float,
) -> list[dict[str, object]]:
    """Compare stepwise replay carry JVPs against central FD trace arrays."""

    from vmec_jax._compat import jax, jnp
    from vmec_jax.discrete_adjoint import (
        _dynamic_replay_initial_carry,
        _packed_dynamic_replay_step_from_carry,
        _static_flags_from_replay_step_traces,
    )
    from vmec_jax.state import pack_state

    if not base_traces:
        return []
    static_flags = _static_flags_from_replay_step_traces(tuple(base_traces))
    carry = _dynamic_replay_initial_carry(base_traces[0])

    def _zero_tree(value):
        return jax.tree_util.tree_map(lambda x: jnp.zeros_like(jnp.asarray(x)), value)

    carry_tangent = (jnp.asarray(initial_tangent, dtype=jnp.asarray(carry[0]).dtype),) + tuple(
        _zero_tree(value) for value in carry[1:]
    )
    velocity_slots = {
        "vRcc_after": 3,
        "vRss_after": 4,
        "vRsc_after": 5,
        "vRcs_after": 6,
        "vZsc_after": 7,
        "vZcs_after": 8,
        "vZcc_after": 9,
        "vZss_after": 10,
        "vLsc_after": 11,
        "vLcs_after": 12,
        "vLcc_after": 13,
        "vLss_after": 14,
    }
    report: list[dict[str, object]] = []
    nsteps = min(len(base_traces), len(plus_traces), len(minus_traces))
    for step in range(nsteps):
        trace = base_traces[step]

        def _step(carry_arg):
            return _packed_dynamic_replay_step_from_carry(
                carry_arg,
                trace,
                static=static,
                static_flags=static_flags,
                preconditioner_jmax_override=int(static_flags["precond_jmax"]),
                preconditioner_use_precomputed_tridi=static_flags.get(
                    "preconditioner_use_precomputed_tridi",
                    None,
                ),
                preconditioner_use_lax_tridi=static_flags.get(
                    "preconditioner_use_lax_tridi",
                    None,
                ),
            )

        carry, carry_tangent = jax.jvp(_step, (carry,), (carry_tangent,))
        item: dict[str, object] = {
            "step": int(step + 1),
            "step_status": str(trace.get("step_status", "")),
            "restart_path": str(trace.get("restart_path", "")),
        }
        plus = plus_traces[step]
        minus = minus_traces[step]
        if trace.get("state_post") is not None and plus.get("state_post") is not None and minus.get("state_post") is not None:
            fd_state = (
                np.asarray(pack_state(plus["state_post"]), dtype=float)
                - np.asarray(pack_state(minus["state_post"]), dtype=float)
            ) / (2.0 * float(eps))
            jvp_state = np.asarray(carry_tangent[0], dtype=float)
            diff = jvp_state - fd_state
            jvp_norm = float(np.linalg.norm(jvp_state))
            fd_norm = float(np.linalg.norm(fd_state))
            diff_norm = float(np.linalg.norm(diff))
            item["state_post_tangent"] = {
                "jvp_norm": jvp_norm,
                "fd_norm": fd_norm,
                "diff_norm": diff_norm,
                "relative_diff_norm": diff_norm / max(jvp_norm, fd_norm, np.finfo(float).eps),
                "max_abs_diff": float(np.max(np.abs(diff))) if diff.size else 0.0,
                "cosine_similarity": float(np.vdot(jvp_state, fd_state))
                / max(jvp_norm * fd_norm, np.finfo(float).eps),
            }
        velocities: dict[str, dict[str, float]] = {}
        for key, slot in velocity_slots.items():
            if trace.get(key) is None or plus.get(key) is None or minus.get(key) is None:
                continue
            fd_value = (np.asarray(plus[key], dtype=float) - np.asarray(minus[key], dtype=float)) / (
                2.0 * float(eps)
            )
            jvp_value = np.asarray(carry_tangent[slot], dtype=float)
            diff = jvp_value - fd_value
            jvp_norm = float(np.linalg.norm(jvp_value))
            fd_norm = float(np.linalg.norm(fd_value))
            diff_norm = float(np.linalg.norm(diff))
            velocities[key] = {
                "jvp_norm": jvp_norm,
                "fd_norm": fd_norm,
                "diff_norm": diff_norm,
                "relative_diff_norm": diff_norm / max(jvp_norm, fd_norm, np.finfo(float).eps),
                "max_abs_diff": float(np.max(np.abs(diff))) if diff.size else 0.0,
                "cosine_similarity": float(np.vdot(jvp_value, fd_value))
                / max(jvp_norm * fd_norm, np.finfo(float).eps),
            }
        if velocities:
            item["velocity_tangents"] = velocities
        report.append(item)
    return report


def main() -> None:
    args = _parse_args()
    enable_x64(True)

    solver_device = None if args.solver_device in {"none", "default"} else str(args.solver_device)
    exact_path = None if args.exact_path == "auto" else str(args.exact_path)
    input_file = Path(args.input).expanduser()
    surfaces = _surfaces(args.surfaces)

    vmec = vj.FixedBoundaryVMEC.from_input(
        input_file,
        max_mode=int(args.max_mode),
        min_vmec_mode=int(args.min_vmec_mode),
        project_input_boundary_to_max_mode=True,
    )
    qi_options = vj.QuasiIsodynamicOptions(
        surfaces=surfaces,
        mboz=int(args.mboz),
        nboz=int(args.nboz),
        nphi=int(args.nphi),
        nalpha=int(args.nalpha),
        n_bounce=int(args.n_bounce),
        include_bounce_endpoints=True,
        softness=2.0e-2,
        width_weight=1.0,
        branch_width_weight=0.5,
        branch_width_softness=2.0e-2,
        profile_weight=0.1,
        shuffle_profile_weight=1.0,
        shuffle_profile_softness=2.0e-2,
        weighted_shuffle_profile_weight=0.0,
        weighted_shuffle_profile_softness=2.0e-2,
        phimin=0.0,
        jit_booz=True,
    )
    problem = vj.LeastSquaresProblem.from_tuples(
        [(vj.QuasiIsodynamicResidual(qi_options).J, 0.0, 1.0)]
    )
    def _build_stage(inner_max_iter: int):
        return build_quasi_isodynamic_objective_stage(
            vmec.cfg,
            vmec.indata,
            stage_mode=int(args.max_mode),
            scalar_objectives=problem.objective_terms,
            qi_objectives=problem.qi_objective_terms,
            surfaces=qi_options.surfaces,
            mboz=qi_options.mboz,
            nboz=qi_options.nboz,
            nphi=qi_options.nphi,
            nalpha=qi_options.nalpha,
            n_bounce=qi_options.n_bounce,
            include_bounce_endpoints=qi_options.include_bounce_endpoints,
            softness=qi_options.softness,
            width_weight=qi_options.width_weight,
            branch_width_weight=qi_options.branch_width_weight,
            branch_width_softness=qi_options.branch_width_softness,
            profile_weight=qi_options.profile_weight,
            shuffle_profile_weight=qi_options.shuffle_profile_weight,
            shuffle_profile_softness=qi_options.shuffle_profile_softness,
            shuffle_profile_nphi_out=qi_options.shuffle_profile_nphi_out,
            weighted_shuffle_profile_weight=qi_options.weighted_shuffle_profile_weight,
            weighted_shuffle_profile_softness=qi_options.weighted_shuffle_profile_softness,
            aligned_profile_weight=qi_options.aligned_profile_weight,
            aligned_profile_softness=qi_options.aligned_profile_softness,
            aligned_profile_trap_level=qi_options.aligned_profile_trap_level,
            aligned_profile_trap_softness=qi_options.aligned_profile_trap_softness,
            phimin=qi_options.phimin,
            jit_booz=qi_options.jit_booz,
            project_input_boundary_to_max_mode=vmec.project_input_boundary_to_max_mode,
            include=vmec.include,
            fix=vmec.fix,
            inner_max_iter=int(inner_max_iter),
            inner_ftol=float(args.inner_ftol),
            trial_max_iter=int(args.trial_max_iter),
            trial_ftol=float(args.trial_ftol),
            solver_device=solver_device,
            exact_path=exact_path,
        )

    stage = _build_stage(int(args.inner_max_iter))

    params0 = np.zeros(len(stage.specs), dtype=float)
    direction = np.zeros_like(params0)
    idx = _spec_index(stage.specs, kind=str(args.kind), m=int(args.m), n=int(args.n))
    direction[idx] = 1.0
    eps = float(args.epsilon)

    r0 = np.asarray(stage.optimizer.residual_fun(params0), dtype=float)
    linear_operator = stage.optimizer.residual_linear_operator(params0)
    jvp = np.asarray(linear_operator.matvec(direction), dtype=float)
    r_plus = np.asarray(stage.optimizer.residual_fun(params0 + eps * direction), dtype=float)
    r_minus = np.asarray(stage.optimizer.residual_fun(params0 - eps * direction), dtype=float)
    fd = (r_plus - r_minus) / (2.0 * eps)

    diff = jvp - fd
    fd_norm = float(np.linalg.norm(fd))
    jvp_norm = float(np.linalg.norm(jvp))
    diff_norm = float(np.linalg.norm(diff))
    denom = max(fd_norm, jvp_norm, np.finfo(float).eps)
    dot = float(np.vdot(jvp, fd))
    cosine = dot / max(fd_norm * jvp_norm, np.finfo(float).eps)
    report = {
        "input": str(input_file),
        "max_mode": int(args.max_mode),
        "parameter": {
            "index": idx,
            "name": stage.specs[idx].name,
            "kind": stage.specs[idx].kind,
            "m": int(stage.specs[idx].m),
            "n": int(stage.specs[idx].n),
        },
        "epsilon": eps,
        "residual_size": int(r0.size),
        "residual_norm": float(np.linalg.norm(r0)),
        "residual_plus_norm": float(np.linalg.norm(r_plus)),
        "residual_minus_norm": float(np.linalg.norm(r_minus)),
        "residual_plus_minus_delta_norm": float(np.linalg.norm(r_plus - r_minus)),
        "jvp_norm": jvp_norm,
        "fd_norm": fd_norm,
        "diff_norm": diff_norm,
        "relative_diff_norm": diff_norm / denom,
        "max_abs_diff": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "cosine_similarity": cosine,
        "solver_device": solver_device or "default",
        "exact_path": exact_path or "auto",
    }
    if bool(args.dense_jacobian):
        jac = np.asarray(stage.optimizer.jacobian_fun(params0), dtype=float)
        dense_col = jac[:, idx]
        dense_diff = dense_col - fd
        dense_norm = float(np.linalg.norm(dense_col))
        dense_diff_norm = float(np.linalg.norm(dense_diff))
        dense_dot = float(np.vdot(dense_col, fd))
        report["dense_jacobian"] = {
            "shape": [int(jac.shape[0]), int(jac.shape[1])],
            "column_norm": dense_norm,
            "diff_norm": dense_diff_norm,
            "relative_diff_norm": dense_diff_norm / max(dense_norm, fd_norm, np.finfo(float).eps),
            "max_abs_diff": float(np.max(np.abs(dense_diff))) if dense_diff.size else 0.0,
            "cosine_similarity": dense_dot / max(dense_norm * fd_norm, np.finfo(float).eps),
            "matrix_free_column_diff_norm": float(np.linalg.norm(dense_col - jvp)),
        }
    if bool(args.state_fd) or bool(args.dynamic_axis_tangent):
        from vmec_jax._compat import jax, jnp
        from vmec_jax.discrete_adjoint import checkpoint_tape_state_jvp, replay_scan_cache_diagnostics
        from vmec_jax.init_guess import initial_guess_from_boundary
        from vmec_jax.state import pack_state, unpack_state

        replay_scan_cache_diagnostics(reset=True)
        state0, state_tangents = stage.optimizer.state_tangent_columns_fun(params0)
        state_tangent_replay_report = replay_scan_cache_diagnostics(reset=True)
        _state_exact, state_payload = stage.optimizer._solve_exact_with_tape_for_jvp(params0)
        packed0 = jnp.asarray(pack_state(state0), dtype=jnp.float64)
        state_jvp = jnp.asarray(state_tangents[idx], dtype=jnp.float64)

        def _residuals_from_packed(packed):
            return jnp.asarray(
                stage.optimizer._residuals_fn(unpack_state(packed, stage.optimizer._layout)),
                dtype=jnp.float64,
            ).reshape(-1)

        residuals_from_state, residual_state_linear = jax.linearize(
            _residuals_from_packed,
            packed0,
        )
        jvp_from_state_jvp = np.asarray(residual_state_linear(state_jvp), dtype=float)
        state_jvp_np = np.asarray(state_jvp, dtype=float)
        residuals_from_state_np = np.asarray(residuals_from_state, dtype=float)
        state_report = {
            "packed_state_size": int(state_jvp_np.size),
            "base_residual_norm_from_state_linearize": float(np.linalg.norm(residuals_from_state_np)),
            "state_jvp_norm": float(np.linalg.norm(state_jvp_np)),
            "residual_jvp_from_state_jvp_norm": float(np.linalg.norm(jvp_from_state_jvp)),
            "residual_state_jvp_vs_matrix_free_norm": float(np.linalg.norm(jvp_from_state_jvp - jvp)),
            "state_tangent_replay_diagnostics": state_tangent_replay_report,
            "state_tangent_tape": _tape_payload_report(state_payload["tape"]),
            "optimizer_profile_after_state_tangent": _optimizer_profile_report(stage.optimizer, prefix="state_tangent"),
        }
        if bool(args.state_fd):
            state_plus = stage.optimizer._solve_forward(params0 + eps * direction, trial=False)
            state_minus = stage.optimizer._solve_forward(params0 - eps * direction, trial=False)
            packed_plus = jnp.asarray(pack_state(state_plus), dtype=jnp.float64)
            packed_minus = jnp.asarray(pack_state(state_minus), dtype=jnp.float64)
            state_fd = (packed_plus - packed_minus) / (2.0 * eps)
            jvp_from_state_fd = np.asarray(residual_state_linear(state_fd), dtype=float)
            state_fd_np = np.asarray(state_fd, dtype=float)
            state_diff = state_jvp_np - state_fd_np
            state_fd_norm = float(np.linalg.norm(state_fd_np))
            state_jvp_norm = float(np.linalg.norm(state_jvp_np))
            state_dot = float(np.vdot(state_jvp_np, state_fd_np))
            state_cosine = state_dot / max(state_fd_norm * state_jvp_norm, np.finfo(float).eps)
            state_report.update(
                {
                    "state_fd_norm": state_fd_norm,
                    "state_diff_norm": float(np.linalg.norm(state_diff)),
                    "state_relative_diff_norm": float(np.linalg.norm(state_diff))
                    / max(state_jvp_norm, state_fd_norm, np.finfo(float).eps),
                    "state_max_abs_diff": float(np.max(np.abs(state_diff))) if state_diff.size else 0.0,
                    "state_cosine_similarity": state_cosine,
                    "residual_jvp_from_state_fd_norm": float(np.linalg.norm(jvp_from_state_fd)),
                    "residual_state_fd_vs_complete_fd_norm": float(np.linalg.norm(jvp_from_state_fd - fd)),
                    "residual_state_fd_vs_complete_fd_relative_norm": float(np.linalg.norm(jvp_from_state_fd - fd))
                    / max(float(np.linalg.norm(jvp_from_state_fd)), fd_norm, np.finfo(float).eps),
                    "state_block_report": _packed_block_report(
                        stage.optimizer._layout,
                        state_jvp_np,
                        state_fd_np,
                    ),
                }
            )
        if bool(args.dynamic_axis_tangent):
            _, payload = stage.optimizer._solve_exact_with_tape(params0, return_payload=True)
            params_j = jnp.asarray(params0, dtype=jnp.float64)
            direction_j = jnp.asarray(direction, dtype=jnp.float64)

            def _initial_state_packed_dynamic_axis(params_arg):
                boundary_now = stage.optimizer._boundary_from_params(params_arg)
                state_initial = initial_guess_from_boundary(
                    stage.optimizer._static,
                    boundary_now,
                    stage.optimizer._indata,
                    vmec_project=True,
                    axis_override=None,
                )
                return jnp.asarray(pack_state(state_initial), dtype=jnp.float64)

            _, initial_dynamic_axis_jvp = jax.jvp(
                _initial_state_packed_dynamic_axis,
                (params_j,),
                (direction_j,),
            )
            dynamic_axis_report = {
                "initial_state_jvp_norm": float(np.linalg.norm(np.asarray(initial_dynamic_axis_jvp, dtype=float))),
            }
            if bool(args.state_fd):
                initial_plus = stage.optimizer._initial_state_from_params(
                    params0 + eps * direction,
                    profile_name="diagnostic_initial_fd",
                )
                initial_minus = stage.optimizer._initial_state_from_params(
                    params0 - eps * direction,
                    profile_name="diagnostic_initial_fd",
                )
                initial_fd = (
                    jnp.asarray(pack_state(initial_plus), dtype=jnp.float64)
                    - jnp.asarray(pack_state(initial_minus), dtype=jnp.float64)
                ) / (2.0 * eps)
                initial_dynamic_axis_np = np.asarray(initial_dynamic_axis_jvp, dtype=float)
                initial_fd_np = np.asarray(initial_fd, dtype=float)
                initial_diff = initial_dynamic_axis_np - initial_fd_np
                initial_fd_norm = float(np.linalg.norm(initial_fd_np))
                initial_jvp_norm = float(np.linalg.norm(initial_dynamic_axis_np))
                dynamic_axis_report.update(
                    {
                        "initial_state_fd_norm": initial_fd_norm,
                        "initial_state_diff_norm": float(np.linalg.norm(initial_diff)),
                        "initial_state_relative_diff_norm": float(np.linalg.norm(initial_diff))
                        / max(initial_jvp_norm, initial_fd_norm, np.finfo(float).eps),
                        "initial_state_cosine_similarity": float(
                            np.vdot(initial_dynamic_axis_np, initial_fd_np)
                        )
                        / max(initial_jvp_norm * initial_fd_norm, np.finfo(float).eps),
                        "initial_state_max_abs_diff": (
                            float(np.max(np.abs(initial_diff))) if initial_diff.size else 0.0
                        ),
                    }
                )
            final_dynamic_axis_jvp = checkpoint_tape_state_jvp(
                tape=payload["tape"],
                static=stage.optimizer._static,
                initial_tangent=initial_dynamic_axis_jvp,
                rebuild_preconditioner=True,
            )
            residual_dynamic_axis_jvp = np.asarray(
                residual_state_linear(final_dynamic_axis_jvp),
                dtype=float,
            )
            dynamic_diff = residual_dynamic_axis_jvp - fd
            dynamic_axis_report.update(
                {
                    "final_state_jvp_norm": float(np.linalg.norm(np.asarray(final_dynamic_axis_jvp, dtype=float))),
                    "residual_jvp_norm": float(np.linalg.norm(residual_dynamic_axis_jvp)),
                    "residual_vs_complete_fd_norm": float(np.linalg.norm(dynamic_diff)),
                    "residual_vs_complete_fd_relative_norm": float(np.linalg.norm(dynamic_diff))
                    / max(float(np.linalg.norm(residual_dynamic_axis_jvp)), fd_norm, np.finfo(float).eps),
                    "residual_vs_complete_fd_cosine_similarity": float(np.vdot(residual_dynamic_axis_jvp, fd))
                    / max(float(np.linalg.norm(residual_dynamic_axis_jvp)) * fd_norm, np.finfo(float).eps),
                }
            )
            state_report["dynamic_axis_tangent"] = dynamic_axis_report
        report["state_fd"] = state_report
    state_fd_iters = _int_list(args.state_fd_iters)
    if state_fd_iters:
        from vmec_jax._compat import jnp
        from vmec_jax.discrete_adjoint import replay_scan_cache_diagnostics, residual_branch_fingerprint
        from vmec_jax.solve import solve_fixed_boundary_residual_iter
        from vmec_jax.state import pack_state

        by_iter = []

        def _complete_solve_result(iter_stage, iter_params, *, iter_count: int):
            state0 = iter_stage.optimizer._initial_state_from_params(
                iter_params,
                profile_name="diagnostic_iter_initial",
            )
            solve_kwargs = dict(iter_stage.optimizer._exact_solver_kwargs)
            solve_kwargs["light_history"] = False
            solve_kwargs["adjoint_trace"] = True
            return solve_fixed_boundary_residual_iter(
                state0,
                iter_stage.optimizer._static,
                max_iter=int(iter_count),
                ftol=iter_stage.optimizer._inner_ftol,
                **solve_kwargs,
            )

        for iter_count in state_fd_iters:
            iter_stage = stage if int(iter_count) == int(args.inner_max_iter) else _build_stage(int(iter_count))
            iter_params0 = np.zeros(len(iter_stage.specs), dtype=float)
            iter_direction = np.zeros_like(iter_params0)
            iter_idx = _spec_index(iter_stage.specs, kind=str(args.kind), m=int(args.m), n=int(args.n))
            iter_direction[iter_idx] = 1.0
            replay_scan_cache_diagnostics(reset=True)
            iter_state, iter_tangents = iter_stage.optimizer.state_tangent_columns_fun(iter_params0)
            iter_state_tangent_replay_report = replay_scan_cache_diagnostics(reset=True)
            iter_state_jvp = np.asarray(iter_tangents[iter_idx], dtype=float)
            _iter_exact_state, iter_payload = iter_stage.optimizer._solve_exact_with_tape_for_jvp(iter_params0)
            iter_initial_tangents = iter_stage.optimizer._initial_tangent_columns(
                jnp.asarray(iter_params0, dtype=jnp.float64),
                iter_payload["axis_override"],
                profile_prefix="diagnostic_iter",
            )
            iter_initial_jvp = np.asarray(iter_initial_tangents[iter_idx], dtype=float)
            iter_state_pre_helper = np.asarray(
                iter_stage.optimizer._solver_initial_state_packed_from_params(
                    jnp.asarray(iter_params0, dtype=jnp.float64),
                    iter_payload["axis_override"],
                ),
                dtype=float,
            )
            iter_state_pre_helper_plus = np.asarray(
                iter_stage.optimizer._solver_initial_state_packed_from_params(
                    jnp.asarray(iter_params0 + eps * iter_direction, dtype=jnp.float64),
                    iter_payload["axis_override"],
                ),
                dtype=float,
            )
            iter_state_pre_helper_minus = np.asarray(
                iter_stage.optimizer._solver_initial_state_packed_from_params(
                    jnp.asarray(iter_params0 - eps * iter_direction, dtype=jnp.float64),
                    iter_payload["axis_override"],
                ),
                dtype=float,
            )
            iter_state_pre_helper_fd = (iter_state_pre_helper_plus - iter_state_pre_helper_minus) / (2.0 * eps)
            iter_base_result = _complete_solve_result(iter_stage, iter_params0, iter_count=int(iter_count))
            iter_plus_result = _complete_solve_result(
                iter_stage,
                iter_params0 + eps * iter_direction,
                iter_count=int(iter_count),
            )
            iter_minus_result = _complete_solve_result(
                iter_stage,
                iter_params0 - eps * iter_direction,
                iter_count=int(iter_count),
            )
            base_trace = iter_base_result.diagnostics["adjoint_step_trace"][0]
            plus_trace = iter_plus_result.diagnostics["adjoint_step_trace"][0]
            minus_trace = iter_minus_result.diagnostics["adjoint_step_trace"][0]
            iter_state_pre_fd = np.asarray(
                (
                    jnp.asarray(pack_state(plus_trace["state_pre"]), dtype=jnp.float64)
                    - jnp.asarray(pack_state(minus_trace["state_pre"]), dtype=jnp.float64)
                )
                / (2.0 * eps),
                dtype=float,
            )
            iter_fd = np.asarray(
                (
                    jnp.asarray(pack_state(iter_plus_result.state), dtype=jnp.float64)
                    - jnp.asarray(pack_state(iter_minus_result.state), dtype=jnp.float64)
                )
                / (2.0 * eps),
                dtype=float,
            )
            base_fingerprint = residual_branch_fingerprint(iter_base_result)
            plus_fingerprint = residual_branch_fingerprint(iter_plus_result)
            minus_fingerprint = residual_branch_fingerprint(iter_minus_result)
            base_traces = iter_base_result.diagnostics["adjoint_step_trace"]
            plus_traces = iter_plus_result.diagnostics["adjoint_step_trace"]
            minus_traces = iter_minus_result.diagnostics["adjoint_step_trace"]
            iter_diff = iter_state_jvp - iter_fd
            iter_state_jvp_norm = float(np.linalg.norm(iter_state_jvp))
            iter_state_fd_norm = float(np.linalg.norm(iter_fd))
            iter_diff_norm = float(np.linalg.norm(iter_diff))
            iter_dot = float(np.vdot(iter_state_jvp, iter_fd))
            iter_state_pre_diff = iter_initial_jvp - iter_state_pre_fd
            iter_state_pre_jvp_norm = float(np.linalg.norm(iter_initial_jvp))
            iter_state_pre_fd_norm = float(np.linalg.norm(iter_state_pre_fd))
            iter_state_pre_diff_norm = float(np.linalg.norm(iter_state_pre_diff))
            iter_state_pre_dot = float(np.vdot(iter_initial_jvp, iter_state_pre_fd))
            iter_state_pre_base = np.asarray(pack_state(base_trace["state_pre"]), dtype=float)
            iter_state_pre_base_diff = iter_state_pre_helper - iter_state_pre_base
            iter_state_pre_helper_fd_diff = iter_state_pre_helper_fd - iter_state_pre_fd
            iter_state_pre_helper_fd_norm = float(np.linalg.norm(iter_state_pre_helper_fd))
            iter_state_pre_helper_fd_diff_norm = float(np.linalg.norm(iter_state_pre_helper_fd_diff))
            iter_state_pre_helper_fd_dot = float(np.vdot(iter_state_pre_helper_fd, iter_state_pre_fd))
            by_iter.append(
                {
                    "inner_max_iter": int(iter_count),
                    "parameter_index": int(iter_idx),
                    "packed_state_size": int(iter_state_jvp.size),
                    "state_pre_jvp_norm": iter_state_pre_jvp_norm,
                    "state_pre_fd_norm": iter_state_pre_fd_norm,
                    "state_pre_diff_norm": iter_state_pre_diff_norm,
                    "state_pre_relative_diff_norm": iter_state_pre_diff_norm
                    / max(iter_state_pre_jvp_norm, iter_state_pre_fd_norm, np.finfo(float).eps),
                    "state_pre_max_abs_diff": (
                        float(np.max(np.abs(iter_state_pre_diff))) if iter_state_pre_diff.size else 0.0
                    ),
                    "state_pre_cosine_similarity": iter_state_pre_dot
                    / max(iter_state_pre_jvp_norm * iter_state_pre_fd_norm, np.finfo(float).eps),
                    "state_pre_helper_base_diff_norm": float(np.linalg.norm(iter_state_pre_base_diff)),
                    "state_pre_helper_base_max_abs_diff": (
                        float(np.max(np.abs(iter_state_pre_base_diff))) if iter_state_pre_base_diff.size else 0.0
                    ),
                    "state_pre_helper_fd_norm": iter_state_pre_helper_fd_norm,
                    "state_pre_helper_fd_diff_norm": iter_state_pre_helper_fd_diff_norm,
                    "state_pre_helper_fd_relative_diff_norm": iter_state_pre_helper_fd_diff_norm
                    / max(iter_state_pre_helper_fd_norm, iter_state_pre_fd_norm, np.finfo(float).eps),
                    "state_pre_helper_fd_cosine_similarity": iter_state_pre_helper_fd_dot
                    / max(iter_state_pre_helper_fd_norm * iter_state_pre_fd_norm, np.finfo(float).eps),
                    "state_pre_helper_fd_block_report": _packed_block_report(
                        base_trace["state_pre"].layout,
                        iter_state_pre_helper_fd,
                        iter_state_pre_fd,
                    ),
                    "state_pre_block_report": _packed_block_report(
                        base_trace["state_pre"].layout,
                        iter_initial_jvp,
                        iter_state_pre_fd,
                    ),
                    "state_jvp_norm": iter_state_jvp_norm,
                    "state_fd_norm": iter_state_fd_norm,
                    "state_diff_norm": iter_diff_norm,
                    "state_relative_diff_norm": iter_diff_norm
                    / max(iter_state_jvp_norm, iter_state_fd_norm, np.finfo(float).eps),
                    "state_max_abs_diff": float(np.max(np.abs(iter_diff))) if iter_diff.size else 0.0,
                    "state_cosine_similarity": iter_dot
                    / max(iter_state_jvp_norm * iter_state_fd_norm, np.finfo(float).eps),
                    "state_block_report": _packed_block_report(
                        iter_stage.optimizer._layout,
                        iter_state_jvp,
                        iter_fd,
                    ),
                    "state_tangent_replay_diagnostics": iter_state_tangent_replay_report,
                    "state_tangent_tape": _tape_payload_report(iter_payload["tape"]),
                    "optimizer_profile_after_state_tangent": _optimizer_profile_report(
                        iter_stage.optimizer,
                        prefix="state_tangent",
                    ),
                    "trace_control_fd": _trace_control_fd_report(
                        base_traces,
                        plus_traces,
                        minus_traces,
                        eps,
                    ),
                    "dynamic_replay_base": _dynamic_replay_base_report(
                        base_traces,
                        iter_stage.optimizer._static,
                    ),
                    "dynamic_stacked_replay_base": _dynamic_stacked_replay_base_report(
                        iter_payload["tape"],
                        base_traces,
                        iter_stage.optimizer._static,
                    ),
                    "dynamic_replay_jvp_trace": _dynamic_replay_jvp_trace_report(
                        base_traces,
                        plus_traces,
                        minus_traces,
                        iter_stage.optimizer._static,
                        iter_initial_jvp,
                        eps,
                    ),
                    "fingerprint_plus_matches_base": bool(plus_fingerprint == base_fingerprint),
                    "fingerprint_minus_matches_base": bool(minus_fingerprint == base_fingerprint),
                    "fingerprint_plus_matches_minus": bool(plus_fingerprint == minus_fingerprint),
                    "base_fingerprint": base_fingerprint,
                    "plus_fingerprint": plus_fingerprint,
                    "minus_fingerprint": minus_fingerprint,
                    "base_residual_norm": float(
                        np.linalg.norm(
                            np.asarray(iter_stage.optimizer._residuals_eval_fn(iter_base_result.state), dtype=float)
                        )
                    ),
                }
            )
        report["state_fd_by_iter"] = by_iter
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output_json is not None:
        output_json = Path(args.output_json).expanduser()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(text + "\n")


if __name__ == "__main__":
    main()
