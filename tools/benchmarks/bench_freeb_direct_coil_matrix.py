#!/usr/bin/env python3
"""Run the lightweight direct-coil free-boundary benchmark matrix.

The matrix is intentionally bounded: by default it runs only CPU rows with
small quick settings and records a compact summary JSON plus the child JSON
paths.  GPU rows are opt-in with ``--include-gpu`` and are skipped when no JAX
GPU device is visible.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "results" / "bench_freeb_direct_coil_matrix" / "summary.json"
BADJAC_PROBE0_ENV = {"VMEC_JAX_BADJAC_INITIAL_STATE_PROBE_ITERS": "0"}
POLICY_ABLATION_ENVS = {
    "no_residual_metrics": {"VMEC_JAX_HOST_RESIDUAL_METRICS": "0"},
    "no_fsq1_norms": {"VMEC_JAX_HOST_FSQ1_NORMS": "0"},
    "no_profile_setup": {"VMEC_JAX_HOST_PROFILE_SETUP": "0"},
    "host_policies_off": {
        "VMEC_JAX_HOST_RESIDUAL_METRICS": "0",
        "VMEC_JAX_HOST_FSQ1_NORMS": "0",
        "VMEC_JAX_HOST_PROFILE_SETUP": "0",
    },
}
ChildSpec = tuple[str, Path, list[str], dict[str, str]]


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Summary JSON path, or an output directory.")
    p.add_argument("--quick", action=argparse.BooleanOptionalAction, default=True, help="Use small CPU-safe defaults.")
    p.add_argument("--include-gpu", action="store_true", help="Also run GPU rows when a JAX GPU device is available.")
    p.add_argument(
        "--include-badjac-probe0",
        action="store_true",
        help=(
            "Add direct-solve rows with VMEC_JAX_BADJAC_INITIAL_STATE_PROBE_ITERS=0 so the "
            "ptau-only bad-Jacobian control-path timing is captured in the summary JSON."
        ),
    )
    p.add_argument(
        "--include-timing-light",
        action="store_true",
        help=(
            "Add direct-solve rows with VMEC_JAX_TIMING disabled. These rows measure production-like "
            "wall time without synchronization from detailed timing probes."
        ),
    )
    p.add_argument(
        "--include-policy-ablation",
        action="store_true",
        help=(
            "Add direct-solve JIT-forces rows that disable host residual metrics, fsq1 norms, "
            "profile setup, and all three together. This is benchmark-only evidence for "
            "CPU/GPU control-path overhead and does not change solver defaults."
        ),
    )
    p.add_argument("--backend-note", default="", help="Optional note copied into the summary JSON.")
    p.add_argument("--timeout-s", type=float, default=240.0, help="Per-child benchmark timeout.")
    return p


def _summary_path(path: Path) -> Path:
    path = path.expanduser()
    if path.suffix.lower() == ".json":
        return path.resolve()
    return (path / "summary.json").resolve()


def _jsonify(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return value


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonify(data), indent=2, sort_keys=True, allow_nan=False) + "\n")


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _gpu_available() -> tuple[bool, dict[str, Any]]:
    try:
        from vmec_jax._compat import has_jax, jax

        if not has_jax() or jax is None:
            return False, {"has_jax": False, "devices": [], "reason": "jax_unavailable"}
        devices = list(jax.devices())
        # `jax.devices()` follows the current default platform. On hosts where
        # users set `JAX_PLATFORMS=cpu,cuda`, it can report only CPU devices
        # even though CUDA is importable. Probe concrete accelerator platforms
        # explicitly so `--include-gpu` does not falsely skip GPU rows.
        for platform in ("cuda", "rocm", "gpu"):
            try:
                for device in jax.devices(platform):
                    if all(str(device) != str(existing) for existing in devices):
                        devices.append(device)
            except Exception:
                continue
        info = {
            "has_jax": True,
            "default_backend": str(jax.default_backend()),
            "devices": [str(device) for device in devices],
            "platforms": sorted({str(getattr(device, "platform", "unknown")) for device in devices}),
        }
        return any(str(getattr(device, "platform", "")).lower() in {"gpu", "cuda", "rocm"} for device in devices), info
    except Exception as exc:
        return False, {"has_jax": None, "devices": [], "error": repr(exc)}


def _gpu_platform_name(gpu_probe: dict[str, Any]) -> str:
    """Return the concrete JAX platform name for GPU child processes."""

    platforms = {str(item).lower() for item in gpu_probe.get("platforms", [])}
    devices = [str(item).lower() for item in gpu_probe.get("devices", [])]
    if "cuda" in platforms:
        return "cuda"
    if any("cuda" in item for item in devices):
        return "cuda"
    if "rocm" in platforms:
        return "rocm"
    if any("rocm" in item for item in devices):
        return "rocm"
    return "gpu"


def _case_counts(payload: dict[str, Any] | None) -> dict[str, int]:
    counts = {"completed": 0, "skipped": 0, "failed": 0}
    if not payload:
        return counts
    for case in payload.get("cases", []):
        status = str(case.get("status", "failed"))
        counts[status if status in counts else "failed"] += 1
    return counts


def _compact_nestor_snapshot(case: dict[str, Any]) -> dict[str, Any] | None:
    freeb = case.get("free_boundary")
    last_diag = freeb.get("last_nestor_diagnostics") if isinstance(freeb, dict) else None
    if not (
        isinstance(case.get("active_nestor_timing_improvement"), dict)
        or isinstance(case.get("trial_nestor_timing_improvement"), dict)
        or isinstance(last_diag, dict)
    ):
        return None

    final_diagnostics: dict[str, Any] = {}
    if isinstance(last_diag, dict):
        phase_time_s = {
            label: last_diag[key]
            for label, key in (
                ("cache_build", "cache_build_time_s"),
                ("source", "source_time_s"),
                ("bvec", "bvec_time_s"),
                ("matrix", "matrix_time_s"),
                ("linear_solve", "linear_solve_time_s"),
                ("vacuum_channels", "vacuum_channels_time_s"),
            )
            if key in last_diag
        }
        sample_phase_time_s = {
            label: last_diag[key]
            for label, key in (
                ("setup", "sample_setup_time_s"),
                ("boundary_geometry", "sample_boundary_geometry_time_s"),
                ("external_field", "sample_external_field_time_s"),
                ("axis_field", "sample_axis_field_time_s"),
                ("projection", "sample_projection_time_s"),
                ("total", "sample_total_time_s"),
            )
            if key in last_diag
        }
        provider = {
            label: last_diag[key]
            for label, key in (
                ("jit_sampler", "provider_jit_sampler"),
                ("chunk_size", "provider_chunk_size"),
                ("coil_count", "provider_coil_count"),
                ("segments_per_coil", "provider_segments_per_coil"),
                ("geometry_cached", "provider_coil_geometry_cached"),
            )
            if key in last_diag
        }
        lu_built = {
            label: last_diag[key]
            for label, key in (
                ("physical_matrix", "physical_matrix_lu_built"),
                ("mode_matrix", "mode_matrix_lu_built"),
            )
            if key in last_diag
        }
        final_diagnostics = {
            key: last_diag[key]
            for key in ("sample_points", "sample_time_s", "solve_time_s")
            if key in last_diag
        }
        if phase_time_s:
            final_diagnostics["phase_time_s"] = phase_time_s
        if sample_phase_time_s:
            final_diagnostics["sample_phase_time_s"] = sample_phase_time_s
        if provider:
            final_diagnostics["provider"] = provider
        if lu_built:
            final_diagnostics["lu_built"] = lu_built

    out: dict[str, Any] = {
        "active": {
            "cold": case.get("cold_solver_timing", {}).get("active_nestor_timing_summary"),
            "warm": case.get("warm_solver_timing", {}).get("active_nestor_timing_summary"),
            "improvement": case.get("active_nestor_timing_improvement"),
        },
        "trial": {
            "cold": case.get("cold_solver_timing", {}).get("trial_nestor_timing_summary"),
            "warm": case.get("warm_solver_timing", {}).get("trial_nestor_timing_summary"),
            "improvement": case.get("trial_nestor_timing_improvement"),
        },
    }
    if isinstance(freeb, dict):
        out["model"] = freeb.get("nestor_model")
        out["provider_kind"] = freeb.get("last_provider_kind")
        out["final_recompute"] = {
            "attempted": freeb.get("final_nestor_recompute_attempted"),
            "failed": freeb.get("final_nestor_recompute_failed"),
            "sample_time_s": freeb.get("final_nestor_sample_time_s"),
            "solve_time_s": freeb.get("final_nestor_solve_time_s"),
        }
    if final_diagnostics:
        out["final_diagnostics"] = final_diagnostics
    return out


_SOLVER_TIMING_KEYS = (
    "solve_total_s",
    "iterations",
    "iteration_loop_s",
    "iteration_loop_unattributed_s",
    "setup_total_s",
    "setup_static_grid_rebuild_s",
    "setup_freeb_policy_s",
    "setup_boundary_profiles_s",
    "setup_cache_key_hash_s",
    "setup_ptau_constants_s",
    "setup_index_constants_s",
    "setup_update_constants_s",
    "setup_axis_reset_s",
    "compute_forces_s",
    "compute_forces_first_s",
    "compute_forces_rest_s",
    "compute_forces_calls",
    "preconditioner_s",
    "iteration_control_s",
    "iteration_control_fsq1_s",
    "iteration_control_fsq1_precond_norm_s",
    "iteration_control_fsq1_scalar_build_s",
    "iteration_control_fsq1_payload_get_s",
    "iteration_control_fsq1_direct_get_s",
    "iteration_control_fsq1_unattributed_s",
    "iteration_control_badjac_s",
    "iteration_control_badjac_ptau_get_s",
    "iteration_control_badjac_state_jacobian_s",
    "iteration_control_badjac_unattributed_s",
    "iteration_control_vmec_time_s",
    "iteration_control_restart_s",
    "iteration_control_evolve_s",
    "iteration_control_unattributed_s",
    "precond_apply_s",
    "precond_mode_scale_s",
    "precond_refresh_s",
    "precond_refresh_seed_s",
    "precond_refresh_calls",
    "precond_reassemble_calls",
    "precond_cache_hit_count",
    "precond_refresh_seed_reuse_count",
    "update_s",
    "update_state_s",
    "update_state_ready_s",
    "update_trace_build_s",
    "update_trace_finalize_s",
    "iteration_prepare_s",
    "iteration_residual_metrics_s",
    "iteration_post_update_s",
    "finalize_s",
    "finalize_nestor_recompute_s",
    "finalize_residual_recompute_s",
    "finalize_residual_device_get_s",
    "finalize_diag_build_s",
    "finalize_unattributed_s",
    "compute_forces_per_iter_s",
    "preconditioner_per_iter_s",
    "update_per_iter_s",
    "update_state_per_iter_s",
    "update_state_ready_per_iter_s",
    "update_trace_build_per_iter_s",
    "update_trace_finalize_per_iter_s",
)


def _compact_solver_timing_snapshot(case: dict[str, Any]) -> dict[str, Any] | None:
    """Return solve-loop timing buckets that explain warm direct-solve cost."""

    out: dict[str, Any] = {}
    for label, source_key in (("cold", "cold_solver_timing"), ("warm", "warm_solver_timing")):
        source = case.get(source_key)
        timing = source.get("timing") if isinstance(source, dict) else None
        if not isinstance(timing, dict):
            continue
        compact = {key: timing[key] for key in _SOLVER_TIMING_KEYS if key in timing}
        if compact:
            out[label] = compact
    return out or None


_PHASE_ENTRY_KEYS = ("name", "key", "seconds", "per_iter_s", "fraction_of_solve")


def _compact_phase_entries(phases: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not isinstance(phases, list):
        return rows
    for phase in phases:
        if not isinstance(phase, dict):
            continue
        compact = {key: phase.get(key) for key in _PHASE_ENTRY_KEYS if key in phase}
        if compact.get("name") is not None and compact.get("seconds") is not None:
            rows.append(compact)
    return rows


def _compact_phase_timing_snapshot(case: dict[str, Any]) -> dict[str, Any] | None:
    """Promote the child solve's normalized warm phase timing into the matrix."""

    comparison = case.get("phase_timing_comparison")
    warm = comparison.get("warm") if isinstance(comparison, dict) else None
    if not isinstance(warm, dict) or not warm.get("timing_available"):
        return None

    all_phases = _compact_phase_entries(warm.get("all_named_phases"))
    top_phases = _compact_phase_entries(warm.get("top_named_phases")) or all_phases[:5]
    out = {
        "warm": {
            "solve_total_s": warm.get("solve_total_s"),
            "iterations": warm.get("iterations"),
            "named_phase_total_s": warm.get("named_phase_total_s"),
            "named_phase_fraction_of_solve": warm.get("named_phase_fraction_of_solve"),
            "top_named_phases": top_phases[:5],
        }
    }
    if all_phases:
        out["warm"]["all_named_phases"] = all_phases
    return out


def _timing_snapshot(payload: dict[str, Any] | None, *, include_nestor: bool = False) -> list[dict[str, Any]]:
    if not payload:
        return []
    rows: list[dict[str, Any]] = []
    for case in payload.get("cases", []):
        row = {
            "label": case.get("label"),
            "status": case.get("status"),
            "cold_or_compile_s": case.get("cold_or_compile_s"),
        }
        warm = case.get("warm")
        if isinstance(warm, dict):
            row["warm_min_s"] = warm.get("min_s")
            row["warm_mean_s"] = warm.get("mean_s")
        if case.get("reason"):
            row["reason"] = case.get("reason")
        if bool(include_nestor):
            nestor = _compact_nestor_snapshot(case)
            if nestor:
                row["nestor"] = nestor
            solver = _compact_solver_timing_snapshot(case)
            if solver:
                row["solver"] = solver
            phase_timing = _compact_phase_timing_snapshot(case)
            if phase_timing:
                row["phase_timing"] = phase_timing
        rows.append(row)
    return rows


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    return number if number == number and number not in (float("inf"), float("-inf")) else None


def _ratio(numerator: Any, denominator: Any) -> float | None:
    num = _finite_float(numerator)
    den = _finite_float(denominator)
    if num is None or den is None or den == 0.0:
        return None
    return float(num / den)


def _nested_value(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _case_metrics(case: dict[str, Any]) -> dict[str, Any]:
    nestor = case.get("nestor") if isinstance(case.get("nestor"), dict) else {}
    solver = case.get("solver") if isinstance(case.get("solver"), dict) else {}
    return {
        "cold_or_compile_s": _finite_float(case.get("cold_or_compile_s")),
        "warm_min_s": _finite_float(case.get("warm_min_s")),
        "warm_mean_s": _finite_float(case.get("warm_mean_s")),
        "warm_solver_total_s": _finite_float(_nested_value(solver, ("warm", "solve_total_s"))),
        "warm_iteration_loop_s": _finite_float(_nested_value(solver, ("warm", "iteration_loop_s"))),
        "warm_iteration_loop_unattributed_s": _finite_float(
            _nested_value(solver, ("warm", "iteration_loop_unattributed_s"))
        ),
        "warm_setup_total_s": _finite_float(_nested_value(solver, ("warm", "setup_total_s"))),
        "warm_setup_static_grid_rebuild_s": _finite_float(
            _nested_value(solver, ("warm", "setup_static_grid_rebuild_s"))
        ),
        "warm_setup_freeb_policy_s": _finite_float(_nested_value(solver, ("warm", "setup_freeb_policy_s"))),
        "warm_setup_boundary_profiles_s": _finite_float(
            _nested_value(solver, ("warm", "setup_boundary_profiles_s"))
        ),
        "warm_setup_cache_key_hash_s": _finite_float(_nested_value(solver, ("warm", "setup_cache_key_hash_s"))),
        "warm_setup_ptau_constants_s": _finite_float(_nested_value(solver, ("warm", "setup_ptau_constants_s"))),
        "warm_setup_index_constants_s": _finite_float(_nested_value(solver, ("warm", "setup_index_constants_s"))),
        "warm_setup_update_constants_s": _finite_float(
            _nested_value(solver, ("warm", "setup_update_constants_s"))
        ),
        "warm_iteration_control_s": _finite_float(_nested_value(solver, ("warm", "iteration_control_s"))),
        "warm_iteration_control_fsq1_s": _finite_float(_nested_value(solver, ("warm", "iteration_control_fsq1_s"))),
        "warm_iteration_control_fsq1_precond_norm_s": _finite_float(
            _nested_value(solver, ("warm", "iteration_control_fsq1_precond_norm_s"))
        ),
        "warm_iteration_control_fsq1_scalar_build_s": _finite_float(
            _nested_value(solver, ("warm", "iteration_control_fsq1_scalar_build_s"))
        ),
        "warm_iteration_control_fsq1_payload_get_s": _finite_float(
            _nested_value(solver, ("warm", "iteration_control_fsq1_payload_get_s"))
        ),
        "warm_iteration_control_fsq1_direct_get_s": _finite_float(
            _nested_value(solver, ("warm", "iteration_control_fsq1_direct_get_s"))
        ),
        "warm_iteration_control_fsq1_unattributed_s": _finite_float(
            _nested_value(solver, ("warm", "iteration_control_fsq1_unattributed_s"))
        ),
        "warm_iteration_control_badjac_s": _finite_float(
            _nested_value(solver, ("warm", "iteration_control_badjac_s"))
        ),
        "warm_iteration_control_badjac_ptau_get_s": _finite_float(
            _nested_value(solver, ("warm", "iteration_control_badjac_ptau_get_s"))
        ),
        "warm_iteration_control_badjac_state_jacobian_s": _finite_float(
            _nested_value(solver, ("warm", "iteration_control_badjac_state_jacobian_s"))
        ),
        "warm_iteration_control_badjac_unattributed_s": _finite_float(
            _nested_value(solver, ("warm", "iteration_control_badjac_unattributed_s"))
        ),
        "warm_iteration_control_vmec_time_s": _finite_float(
            _nested_value(solver, ("warm", "iteration_control_vmec_time_s"))
        ),
        "warm_iteration_control_restart_s": _finite_float(
            _nested_value(solver, ("warm", "iteration_control_restart_s"))
        ),
        "warm_iteration_control_evolve_s": _finite_float(
            _nested_value(solver, ("warm", "iteration_control_evolve_s"))
        ),
        "warm_iteration_control_unattributed_s": _finite_float(
            _nested_value(solver, ("warm", "iteration_control_unattributed_s"))
        ),
        "warm_compute_forces_s": _finite_float(_nested_value(solver, ("warm", "compute_forces_s"))),
        "warm_preconditioner_s": _finite_float(_nested_value(solver, ("warm", "preconditioner_s"))),
        "warm_precond_refresh_s": _finite_float(_nested_value(solver, ("warm", "precond_refresh_s"))),
        "warm_precond_refresh_seed_s": _finite_float(_nested_value(solver, ("warm", "precond_refresh_seed_s"))),
        "warm_precond_refresh_calls": _finite_float(_nested_value(solver, ("warm", "precond_refresh_calls"))),
        "warm_precond_reassemble_calls": _finite_float(_nested_value(solver, ("warm", "precond_reassemble_calls"))),
        "warm_precond_cache_hit_count": _finite_float(_nested_value(solver, ("warm", "precond_cache_hit_count"))),
        "warm_precond_refresh_seed_reuse_count": _finite_float(
            _nested_value(solver, ("warm", "precond_refresh_seed_reuse_count"))
        ),
        "warm_precond_apply_s": _finite_float(_nested_value(solver, ("warm", "precond_apply_s"))),
        "warm_update_s": _finite_float(_nested_value(solver, ("warm", "update_s"))),
        "warm_update_state_s": _finite_float(_nested_value(solver, ("warm", "update_state_s"))),
        "warm_update_state_ready_s": _finite_float(_nested_value(solver, ("warm", "update_state_ready_s"))),
        "warm_iteration_residual_metrics_s": _finite_float(
            _nested_value(solver, ("warm", "iteration_residual_metrics_s"))
        ),
        "warm_finalize_s": _finite_float(_nested_value(solver, ("warm", "finalize_s"))),
        "warm_finalize_nestor_recompute_s": _finite_float(
            _nested_value(solver, ("warm", "finalize_nestor_recompute_s"))
        ),
        "warm_finalize_residual_recompute_s": _finite_float(
            _nested_value(solver, ("warm", "finalize_residual_recompute_s"))
        ),
        "warm_finalize_residual_device_get_s": _finite_float(
            _nested_value(solver, ("warm", "finalize_residual_device_get_s"))
        ),
        "warm_finalize_diag_build_s": _finite_float(_nested_value(solver, ("warm", "finalize_diag_build_s"))),
        "warm_finalize_unattributed_s": _finite_float(_nested_value(solver, ("warm", "finalize_unattributed_s"))),
        "warm_compute_forces_per_iter_s": _finite_float(
            _nested_value(solver, ("warm", "compute_forces_per_iter_s"))
        ),
        "warm_preconditioner_per_iter_s": _finite_float(
            _nested_value(solver, ("warm", "preconditioner_per_iter_s"))
        ),
        "warm_update_per_iter_s": _finite_float(_nested_value(solver, ("warm", "update_per_iter_s"))),
        "active_nestor_warm_sample_s": _finite_float(
            _nested_value(nestor, ("active", "warm", "sample_time_s", "total_s"))
        ),
        "active_nestor_warm_solve_s": _finite_float(
            _nested_value(nestor, ("active", "warm", "solve_time_s", "total_s"))
        ),
        "trial_nestor_warm_sample_s": _finite_float(
            _nested_value(nestor, ("trial", "warm", "sample_time_s", "total_s"))
        ),
        "final_recompute_sample_s": _finite_float(_nested_value(nestor, ("final_recompute", "sample_time_s"))),
        "final_recompute_solve_s": _finite_float(_nested_value(nestor, ("final_recompute", "solve_time_s"))),
        "final_external_field_sample_s": _finite_float(
            _nested_value(nestor, ("final_diagnostics", "sample_phase_time_s", "external_field"))
        ),
        "sample_points": _nested_value(nestor, ("final_diagnostics", "sample_points")),
        "provider": _nested_value(nestor, ("final_diagnostics", "provider")),
    }


def _cpu_gpu_comparison(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build a compact cross-backend comparison from already-recorded child JSON."""

    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        if row.get("status") != "completed":
            continue
        backend = str(row.get("backend", ""))
        label = str(row.get("label", ""))
        for case in row.get("timings", []):
            if not isinstance(case, dict) or case.get("status") != "completed":
                continue
            case_label = str(case.get("label", ""))
            by_key[(backend, label, case_label)] = case

    comparisons: list[dict[str, Any]] = []
    labels = sorted({(label, case_label) for _, label, case_label in by_key})
    for label, case_label in labels:
        cpu = by_key.get(("cpu", label, case_label))
        gpu = by_key.get(("gpu", label, case_label))
        if cpu is None or gpu is None:
            continue
        cpu_metrics = _case_metrics(cpu)
        gpu_metrics = _case_metrics(gpu)
        comparisons.append(
            {
                "label": label,
                "case": case_label,
                "cpu": cpu_metrics,
                "gpu": gpu_metrics,
                "ratios_gpu_over_cpu": {
                    "cold_or_compile": _ratio(
                        gpu_metrics.get("cold_or_compile_s"),
                        cpu_metrics.get("cold_or_compile_s"),
                    ),
                    "warm_min": _ratio(gpu_metrics.get("warm_min_s"), cpu_metrics.get("warm_min_s")),
                    "warm_mean": _ratio(gpu_metrics.get("warm_mean_s"), cpu_metrics.get("warm_mean_s")),
                    "warm_solver_total": _ratio(
                        gpu_metrics.get("warm_solver_total_s"),
                        cpu_metrics.get("warm_solver_total_s"),
                    ),
                    "warm_iteration_loop": _ratio(
                        gpu_metrics.get("warm_iteration_loop_s"),
                        cpu_metrics.get("warm_iteration_loop_s"),
                    ),
                    "warm_iteration_loop_unattributed": _ratio(
                        gpu_metrics.get("warm_iteration_loop_unattributed_s"),
                        cpu_metrics.get("warm_iteration_loop_unattributed_s"),
                    ),
                    "warm_setup_total": _ratio(
                        gpu_metrics.get("warm_setup_total_s"),
                        cpu_metrics.get("warm_setup_total_s"),
                    ),
                    "warm_setup_static_grid_rebuild": _ratio(
                        gpu_metrics.get("warm_setup_static_grid_rebuild_s"),
                        cpu_metrics.get("warm_setup_static_grid_rebuild_s"),
                    ),
                    "warm_setup_freeb_policy": _ratio(
                        gpu_metrics.get("warm_setup_freeb_policy_s"),
                        cpu_metrics.get("warm_setup_freeb_policy_s"),
                    ),
                    "warm_setup_boundary_profiles": _ratio(
                        gpu_metrics.get("warm_setup_boundary_profiles_s"),
                        cpu_metrics.get("warm_setup_boundary_profiles_s"),
                    ),
                    "warm_setup_cache_key_hash": _ratio(
                        gpu_metrics.get("warm_setup_cache_key_hash_s"),
                        cpu_metrics.get("warm_setup_cache_key_hash_s"),
                    ),
                    "warm_setup_ptau_constants": _ratio(
                        gpu_metrics.get("warm_setup_ptau_constants_s"),
                        cpu_metrics.get("warm_setup_ptau_constants_s"),
                    ),
                    "warm_setup_index_constants": _ratio(
                        gpu_metrics.get("warm_setup_index_constants_s"),
                        cpu_metrics.get("warm_setup_index_constants_s"),
                    ),
                    "warm_setup_update_constants": _ratio(
                        gpu_metrics.get("warm_setup_update_constants_s"),
                        cpu_metrics.get("warm_setup_update_constants_s"),
                    ),
                    "warm_iteration_control": _ratio(
                        gpu_metrics.get("warm_iteration_control_s"),
                        cpu_metrics.get("warm_iteration_control_s"),
                    ),
                    "warm_iteration_control_fsq1": _ratio(
                        gpu_metrics.get("warm_iteration_control_fsq1_s"),
                        cpu_metrics.get("warm_iteration_control_fsq1_s"),
                    ),
                    "warm_iteration_control_fsq1_precond_norm": _ratio(
                        gpu_metrics.get("warm_iteration_control_fsq1_precond_norm_s"),
                        cpu_metrics.get("warm_iteration_control_fsq1_precond_norm_s"),
                    ),
                    "warm_iteration_control_fsq1_scalar_build": _ratio(
                        gpu_metrics.get("warm_iteration_control_fsq1_scalar_build_s"),
                        cpu_metrics.get("warm_iteration_control_fsq1_scalar_build_s"),
                    ),
                    "warm_iteration_control_fsq1_payload_get": _ratio(
                        gpu_metrics.get("warm_iteration_control_fsq1_payload_get_s"),
                        cpu_metrics.get("warm_iteration_control_fsq1_payload_get_s"),
                    ),
                    "warm_iteration_control_fsq1_direct_get": _ratio(
                        gpu_metrics.get("warm_iteration_control_fsq1_direct_get_s"),
                        cpu_metrics.get("warm_iteration_control_fsq1_direct_get_s"),
                    ),
                    "warm_iteration_control_fsq1_unattributed": _ratio(
                        gpu_metrics.get("warm_iteration_control_fsq1_unattributed_s"),
                        cpu_metrics.get("warm_iteration_control_fsq1_unattributed_s"),
                    ),
                    "warm_iteration_control_badjac": _ratio(
                        gpu_metrics.get("warm_iteration_control_badjac_s"),
                        cpu_metrics.get("warm_iteration_control_badjac_s"),
                    ),
                    "warm_iteration_control_badjac_ptau_get": _ratio(
                        gpu_metrics.get("warm_iteration_control_badjac_ptau_get_s"),
                        cpu_metrics.get("warm_iteration_control_badjac_ptau_get_s"),
                    ),
                    "warm_iteration_control_badjac_state_jacobian": _ratio(
                        gpu_metrics.get("warm_iteration_control_badjac_state_jacobian_s"),
                        cpu_metrics.get("warm_iteration_control_badjac_state_jacobian_s"),
                    ),
                    "warm_iteration_control_badjac_unattributed": _ratio(
                        gpu_metrics.get("warm_iteration_control_badjac_unattributed_s"),
                        cpu_metrics.get("warm_iteration_control_badjac_unattributed_s"),
                    ),
                    "warm_iteration_control_vmec_time": _ratio(
                        gpu_metrics.get("warm_iteration_control_vmec_time_s"),
                        cpu_metrics.get("warm_iteration_control_vmec_time_s"),
                    ),
                    "warm_iteration_control_restart": _ratio(
                        gpu_metrics.get("warm_iteration_control_restart_s"),
                        cpu_metrics.get("warm_iteration_control_restart_s"),
                    ),
                    "warm_iteration_control_evolve": _ratio(
                        gpu_metrics.get("warm_iteration_control_evolve_s"),
                        cpu_metrics.get("warm_iteration_control_evolve_s"),
                    ),
                    "warm_compute_forces": _ratio(
                        gpu_metrics.get("warm_compute_forces_s"),
                        cpu_metrics.get("warm_compute_forces_s"),
                    ),
                    "warm_preconditioner": _ratio(
                        gpu_metrics.get("warm_preconditioner_s"),
                        cpu_metrics.get("warm_preconditioner_s"),
                    ),
                    "warm_precond_refresh": _ratio(
                        gpu_metrics.get("warm_precond_refresh_s"),
                        cpu_metrics.get("warm_precond_refresh_s"),
                    ),
                    "warm_precond_refresh_seed": _ratio(
                        gpu_metrics.get("warm_precond_refresh_seed_s"),
                        cpu_metrics.get("warm_precond_refresh_seed_s"),
                    ),
                    "warm_precond_apply": _ratio(
                        gpu_metrics.get("warm_precond_apply_s"),
                        cpu_metrics.get("warm_precond_apply_s"),
                    ),
                    "warm_update": _ratio(gpu_metrics.get("warm_update_s"), cpu_metrics.get("warm_update_s")),
                    "warm_update_state": _ratio(
                        gpu_metrics.get("warm_update_state_s"),
                        cpu_metrics.get("warm_update_state_s"),
                    ),
                    "warm_update_state_ready": _ratio(
                        gpu_metrics.get("warm_update_state_ready_s"),
                        cpu_metrics.get("warm_update_state_ready_s"),
                    ),
                    "warm_iteration_residual_metrics": _ratio(
                        gpu_metrics.get("warm_iteration_residual_metrics_s"),
                        cpu_metrics.get("warm_iteration_residual_metrics_s"),
                    ),
                    "warm_finalize": _ratio(
                        gpu_metrics.get("warm_finalize_s"),
                        cpu_metrics.get("warm_finalize_s"),
                    ),
                    "warm_finalize_nestor_recompute": _ratio(
                        gpu_metrics.get("warm_finalize_nestor_recompute_s"),
                        cpu_metrics.get("warm_finalize_nestor_recompute_s"),
                    ),
                    "warm_finalize_residual_recompute": _ratio(
                        gpu_metrics.get("warm_finalize_residual_recompute_s"),
                        cpu_metrics.get("warm_finalize_residual_recompute_s"),
                    ),
                    "warm_finalize_residual_device_get": _ratio(
                        gpu_metrics.get("warm_finalize_residual_device_get_s"),
                        cpu_metrics.get("warm_finalize_residual_device_get_s"),
                    ),
                    "warm_finalize_diag_build": _ratio(
                        gpu_metrics.get("warm_finalize_diag_build_s"),
                        cpu_metrics.get("warm_finalize_diag_build_s"),
                    ),
                    "warm_finalize_unattributed": _ratio(
                        gpu_metrics.get("warm_finalize_unattributed_s"),
                        cpu_metrics.get("warm_finalize_unattributed_s"),
                    ),
                    "active_nestor_warm_sample": _ratio(
                        gpu_metrics.get("active_nestor_warm_sample_s"),
                        cpu_metrics.get("active_nestor_warm_sample_s"),
                    ),
                    "active_nestor_warm_solve": _ratio(
                        gpu_metrics.get("active_nestor_warm_solve_s"),
                        cpu_metrics.get("active_nestor_warm_solve_s"),
                    ),
                    "final_recompute_sample": _ratio(
                        gpu_metrics.get("final_recompute_sample_s"),
                        cpu_metrics.get("final_recompute_sample_s"),
                    ),
                    "final_recompute_solve": _ratio(
                        gpu_metrics.get("final_recompute_solve_s"),
                        cpu_metrics.get("final_recompute_solve_s"),
                    ),
                    "final_external_field_sample": _ratio(
                        gpu_metrics.get("final_external_field_sample_s"),
                        cpu_metrics.get("final_external_field_sample_s"),
                    ),
                },
            }
        )
    return comparisons


_BOTTLENECK_RATIO_TO_METRIC = {
    "warm_min": "warm_min_s",
    "warm_mean": "warm_mean_s",
    "warm_solver_total": "warm_solver_total_s",
    "warm_iteration_loop": "warm_iteration_loop_s",
    "warm_iteration_loop_unattributed": "warm_iteration_loop_unattributed_s",
    "warm_setup_total": "warm_setup_total_s",
    "warm_setup_static_grid_rebuild": "warm_setup_static_grid_rebuild_s",
    "warm_setup_freeb_policy": "warm_setup_freeb_policy_s",
    "warm_setup_boundary_profiles": "warm_setup_boundary_profiles_s",
    "warm_setup_cache_key_hash": "warm_setup_cache_key_hash_s",
    "warm_setup_ptau_constants": "warm_setup_ptau_constants_s",
    "warm_setup_index_constants": "warm_setup_index_constants_s",
    "warm_setup_update_constants": "warm_setup_update_constants_s",
    "warm_iteration_control": "warm_iteration_control_s",
    "warm_iteration_control_fsq1": "warm_iteration_control_fsq1_s",
    "warm_iteration_control_fsq1_precond_norm": "warm_iteration_control_fsq1_precond_norm_s",
    "warm_iteration_control_fsq1_scalar_build": "warm_iteration_control_fsq1_scalar_build_s",
    "warm_iteration_control_fsq1_payload_get": "warm_iteration_control_fsq1_payload_get_s",
    "warm_iteration_control_fsq1_direct_get": "warm_iteration_control_fsq1_direct_get_s",
    "warm_iteration_control_fsq1_unattributed": "warm_iteration_control_fsq1_unattributed_s",
    "warm_iteration_control_badjac": "warm_iteration_control_badjac_s",
    "warm_iteration_control_badjac_ptau_get": "warm_iteration_control_badjac_ptau_get_s",
    "warm_iteration_control_badjac_state_jacobian": "warm_iteration_control_badjac_state_jacobian_s",
    "warm_iteration_control_badjac_unattributed": "warm_iteration_control_badjac_unattributed_s",
    "warm_iteration_control_vmec_time": "warm_iteration_control_vmec_time_s",
    "warm_iteration_control_restart": "warm_iteration_control_restart_s",
    "warm_iteration_control_evolve": "warm_iteration_control_evolve_s",
    "warm_compute_forces": "warm_compute_forces_s",
    "warm_preconditioner": "warm_preconditioner_s",
    "warm_precond_refresh": "warm_precond_refresh_s",
    "warm_precond_refresh_seed": "warm_precond_refresh_seed_s",
    "warm_precond_apply": "warm_precond_apply_s",
    "warm_update": "warm_update_s",
    "warm_update_state": "warm_update_state_s",
    "warm_update_state_ready": "warm_update_state_ready_s",
    "warm_iteration_residual_metrics": "warm_iteration_residual_metrics_s",
    "warm_finalize": "warm_finalize_s",
    "warm_finalize_nestor_recompute": "warm_finalize_nestor_recompute_s",
    "warm_finalize_residual_recompute": "warm_finalize_residual_recompute_s",
    "warm_finalize_residual_device_get": "warm_finalize_residual_device_get_s",
    "warm_finalize_diag_build": "warm_finalize_diag_build_s",
    "warm_finalize_unattributed": "warm_finalize_unattributed_s",
    "active_nestor_warm_sample": "active_nestor_warm_sample_s",
    "active_nestor_warm_solve": "active_nestor_warm_solve_s",
    "final_recompute_sample": "final_recompute_sample_s",
    "final_recompute_solve": "final_recompute_solve_s",
    "final_external_field_sample": "final_external_field_sample_s",
}


def _gpu_bottleneck_summary(comparisons: list[dict[str, Any]], *, top_n: int = 8) -> list[dict[str, Any]]:
    """Rank warm GPU-over-CPU bottlenecks from completed comparison rows."""

    rows: list[dict[str, Any]] = []
    for comparison in comparisons:
        ratios = comparison.get("ratios_gpu_over_cpu", {})
        cpu_metrics = comparison.get("cpu", {})
        gpu_metrics = comparison.get("gpu", {})
        if not isinstance(ratios, dict) or not isinstance(cpu_metrics, dict) or not isinstance(gpu_metrics, dict):
            continue
        for ratio_key, metric_key in _BOTTLENECK_RATIO_TO_METRIC.items():
            ratio = _finite_float(ratios.get(ratio_key))
            cpu_s = _finite_float(cpu_metrics.get(metric_key))
            gpu_s = _finite_float(gpu_metrics.get(metric_key))
            if ratio is None or cpu_s is None or gpu_s is None or ratio <= 1.0:
                continue
            rows.append(
                {
                    "label": comparison.get("label"),
                    "case": comparison.get("case"),
                    "phase": ratio_key,
                    "ratio_gpu_over_cpu": ratio,
                    "cpu_s": cpu_s,
                    "gpu_s": gpu_s,
                    "gpu_minus_cpu_s": float(gpu_s - cpu_s),
                }
            )
    rows.sort(key=lambda item: (item["gpu_minus_cpu_s"], item["gpu_s"], item["ratio_gpu_over_cpu"]), reverse=True)
    return rows[: max(int(top_n), 0)]


def _warm_phase_bottleneck_summary(rows: list[dict[str, Any]], *, top_n: int = 8) -> list[dict[str, Any]]:
    """Rank absolute warm phase costs for CPU-only or mixed matrix runs."""

    summary: list[dict[str, Any]] = []
    for row in rows:
        if row.get("status") != "completed":
            continue
        backend = row.get("backend")
        label = row.get("label")
        timings = row.get("timings", [])
        if not isinstance(timings, list):
            continue
        for case in timings:
            if not isinstance(case, dict) or case.get("status") != "completed":
                continue
            phase_timing = case.get("phase_timing")
            warm = phase_timing.get("warm") if isinstance(phase_timing, dict) else None
            if not isinstance(warm, dict):
                continue
            solve_total_s = _finite_float(warm.get("solve_total_s"))
            iterations = warm.get("iterations")
            phases = warm.get("all_named_phases") or warm.get("top_named_phases") or []
            if not isinstance(phases, list):
                continue
            for phase in phases:
                if not isinstance(phase, dict):
                    continue
                seconds = _finite_float(phase.get("seconds"))
                if seconds is None or seconds <= 0.0:
                    continue
                fraction = _finite_float(phase.get("fraction_of_solve"))
                if fraction is None and solve_total_s is not None and solve_total_s > 0.0:
                    fraction = float(seconds / solve_total_s)
                summary.append(
                    {
                        "backend": backend,
                        "label": label,
                        "case": case.get("label"),
                        "phase": phase.get("name"),
                        "key": phase.get("key"),
                        "seconds": seconds,
                        "per_iter_s": _finite_float(phase.get("per_iter_s")),
                        "fraction_of_solve": fraction,
                        "solve_total_s": solve_total_s,
                        "iterations": iterations,
                    }
                )
    summary.sort(key=lambda item: (item["seconds"], item.get("fraction_of_solve") or 0.0), reverse=True)
    return summary[: max(int(top_n), 0)]


def _format_optional_seconds(value: Any) -> str:
    seconds = _finite_float(value)
    return "n/a" if seconds is None else f"{seconds:.4g}s"


def _format_optional_pct(value: Any) -> str:
    fraction = _finite_float(value)
    return "n/a" if fraction is None else f"{100.0 * fraction:.1f}%"


def _with_badjac_probe0_rows(specs: list[ChildSpec]) -> list[ChildSpec]:
    rows: list[ChildSpec] = []
    for label, out, args, env_overrides in specs:
        rows.append((label, out, args, env_overrides))
        if label not in {"direct_solve", "direct_solve_jit_forces"}:
            continue
        probe_label = f"{label}_badjac_probe0"
        probe_out = out.with_name(f"{out.stem}_badjac_probe0{out.suffix}")
        rows.append((probe_label, probe_out, list(args), {**env_overrides, **BADJAC_PROBE0_ENV}))
    return rows


def _with_timing_light_rows(specs: list[ChildSpec]) -> list[ChildSpec]:
    rows: list[ChildSpec] = []
    for label, out, args, env_overrides in specs:
        rows.append((label, out, args, env_overrides))
        if label != "direct_solve_jit_forces":
            continue
        light_label = f"{label}_timing_light"
        light_out = out.with_name(f"{out.stem}_timing_light{out.suffix}")
        rows.append(
            (
                light_label,
                light_out,
                list(args),
                {
                    **env_overrides,
                    "VMEC_JAX_TIMING": "0",
                    "VMEC_JAX_TIMING_DETAIL": "0",
                },
            )
        )
    return rows


def _with_policy_ablation_rows(specs: list[ChildSpec]) -> list[ChildSpec]:
    rows: list[ChildSpec] = []
    for label, out, args, env_overrides in specs:
        rows.append((label, out, args, env_overrides))
        if label != "direct_solve_jit_forces":
            continue
        for ablation_label, ablation_env in POLICY_ABLATION_ENVS.items():
            ablation_out = out.with_name(f"{out.stem}_{ablation_label}{out.suffix}")
            rows.append(
                (
                    f"{label}_{ablation_label}",
                    ablation_out,
                    list(args),
                    {**env_overrides, **ablation_env},
                )
            )
    return rows


def _child_specs(
    *,
    quick: bool,
    outdir: Path,
    backend: str,
    include_badjac_probe0: bool = False,
    include_timing_light: bool = False,
    include_policy_ablation: bool = False,
) -> list[ChildSpec]:
    suffix = f"_{backend}.json"
    if quick:
        specs: list[ChildSpec] = [
            (
                "provider",
                outdir / f"bench_external_field_providers{suffix}",
                ["--points", "8", "--segments", "8", "--warm-repeats", "1", "--skip-essos"],
                {},
            ),
            (
                "direct_solve",
                outdir / f"bench_freeb_direct_coil_solve{suffix}",
                ["--max-iter", "2", "--warm-repeats", "1"],
                {},
            ),
            (
                "direct_solve_jit_forces",
                outdir / f"bench_freeb_direct_coil_solve_jit_forces{suffix}",
                ["--max-iter", "2", "--warm-repeats", "1", "--jit-forces"],
                {},
            ),
            (
                "gradient",
                outdir / f"bench_freeb_coil_gradient{suffix}",
                ["--points", "8", "--segments", "8", "--matrix-size", "8", "--warm-repeats", "1"],
                {},
            ),
        ]
        if include_timing_light:
            specs = _with_timing_light_rows(specs)
        if include_policy_ablation:
            specs = _with_policy_ablation_rows(specs)
        return _with_badjac_probe0_rows(specs) if include_badjac_probe0 else specs
    specs = [
        (
            "provider",
            outdir / f"bench_external_field_providers{suffix}",
            ["--points", "48", "--segments", "48", "--warm-repeats", "5", "--skip-essos"],
            {},
        ),
        (
            "direct_solve",
            outdir / f"bench_freeb_direct_coil_solve{suffix}",
            ["--max-iter", "2", "--warm-repeats", "1"],
            {},
        ),
        (
            "direct_solve_jit_forces",
            outdir / f"bench_freeb_direct_coil_solve_jit_forces{suffix}",
            ["--max-iter", "2", "--warm-repeats", "1", "--jit-forces"],
            {},
        ),
        (
            "gradient",
            outdir / f"bench_freeb_coil_gradient{suffix}",
            ["--points", "24", "--segments", "48", "--matrix-size", "24", "--warm-repeats", "5"],
            {},
        ),
    ]
    if include_timing_light:
        specs = _with_timing_light_rows(specs)
    if include_policy_ablation:
        specs = _with_policy_ablation_rows(specs)
    return _with_badjac_probe0_rows(specs) if include_badjac_probe0 else specs


def _script_for(label: str) -> Path:
    if label.startswith("direct_solve"):
        return REPO_ROOT / "tools" / "benchmarks" / "bench_freeb_direct_coil_solve.py"
    return {
        "provider": REPO_ROOT / "tools" / "benchmarks" / "bench_external_field_providers.py",
        "gradient": REPO_ROOT / "tools" / "benchmarks" / "bench_freeb_coil_gradient.py",
    }[label]


def _run_child(
    label: str,
    out: Path,
    args: list[str],
    *,
    backend: str,
    timeout_s: float,
    jax_platform: str | None = None,
    env_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}" if env.get("PYTHONPATH") else str(REPO_ROOT)
    if label.startswith("direct_solve"):
        env.setdefault("VMEC_JAX_TIMING", "1")
        env.setdefault("VMEC_JAX_TIMING_DETAIL", "1")
    if backend == "cpu":
        env["JAX_PLATFORMS"] = "cpu"
    elif backend == "gpu":
        env["JAX_PLATFORMS"] = str(jax_platform or "gpu")
    overrides = dict(env_overrides or {})
    env.update(overrides)

    cmd = [sys.executable, str(_script_for(label)), "--out", str(out), *args]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=float(timeout_s),
            check=False,
        )
        elapsed_s = float(time.perf_counter() - t0)
        payload = _load_json(out)
        return {
            "label": label,
            "backend": backend,
            "status": "completed" if proc.returncode == 0 else "failed",
            "returncode": int(proc.returncode),
            "elapsed_s": elapsed_s,
            "output_json": out,
            "command": cmd,
            "jax_platform": env.get("JAX_PLATFORMS"),
            "env_overrides": overrides,
            "badjac_initial_state_probe_iters": env.get("VMEC_JAX_BADJAC_INITIAL_STATE_PROBE_ITERS"),
            "stdout_tail": proc.stdout.strip().splitlines()[-8:],
            "stderr_tail": proc.stderr.strip().splitlines()[-8:],
            "child_status": None if payload is None else payload.get("status"),
            "child_backend": None if payload is None else payload.get("backend"),
            "case_counts": _case_counts(payload),
            "timings": _timing_snapshot(payload, include_nestor=label.startswith("direct_solve")),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "label": label,
            "backend": backend,
            "status": "failed",
            "reason": "timeout",
            "elapsed_s": float(time.perf_counter() - t0),
            "timeout_s": float(timeout_s),
            "output_json": out,
            "command": cmd,
            "jax_platform": env.get("JAX_PLATFORMS"),
            "env_overrides": overrides,
            "badjac_initial_state_probe_iters": env.get("VMEC_JAX_BADJAC_INITIAL_STATE_PROBE_ITERS"),
            "stdout_tail": (exc.stdout or "").strip().splitlines()[-8:] if isinstance(exc.stdout, str) else [],
            "stderr_tail": (exc.stderr or "").strip().splitlines()[-8:] if isinstance(exc.stderr, str) else [],
        }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = _summary_path(args.out)
    outdir = summary.parent
    outdir.mkdir(parents=True, exist_ok=True)

    gpu_ok, gpu_probe = _gpu_available()
    rows: list[dict[str, Any]] = []
    for label, out, child_args, child_env in _child_specs(
        quick=bool(args.quick),
        outdir=outdir,
        backend="cpu",
        include_badjac_probe0=bool(args.include_badjac_probe0),
        include_timing_light=bool(args.include_timing_light),
        include_policy_ablation=bool(args.include_policy_ablation),
    ):
        rows.append(
            _run_child(
                label,
                out,
                child_args,
                backend="cpu",
                timeout_s=float(args.timeout_s),
                env_overrides=child_env,
            )
        )

    if args.include_gpu:
        if gpu_ok:
            gpu_platform = _gpu_platform_name(gpu_probe)
            for label, out, child_args, child_env in _child_specs(
                quick=bool(args.quick),
                outdir=outdir,
                backend="gpu",
                include_badjac_probe0=bool(args.include_badjac_probe0),
                include_timing_light=bool(args.include_timing_light),
                include_policy_ablation=bool(args.include_policy_ablation),
            ):
                rows.append(
                    _run_child(
                        label,
                        out,
                        child_args,
                        backend="gpu",
                        timeout_s=float(args.timeout_s),
                        jax_platform=gpu_platform,
                        env_overrides=child_env,
                    )
                )
        else:
            rows.append(
                {
                    "label": "gpu_matrix",
                    "backend": "gpu",
                    "status": "skipped",
                    "reason": "jax_gpu_unavailable",
                    "gpu_probe": gpu_probe,
                }
            )

    status = "completed" if all(row["status"] in {"completed", "skipped"} for row in rows) else "failed"
    comparisons = _cpu_gpu_comparison(rows)
    gpu_bottlenecks = _gpu_bottleneck_summary(comparisons)
    warm_phase_bottlenecks = _warm_phase_bottleneck_summary(rows)
    payload = {
        "status": status,
        "script": str(Path(__file__).resolve()),
        "quick": bool(args.quick),
        "include_gpu": bool(args.include_gpu),
        "include_badjac_probe0": bool(args.include_badjac_probe0),
        "include_timing_light": bool(args.include_timing_light),
        "include_policy_ablation": bool(args.include_policy_ablation),
        "backend_note": str(args.backend_note),
        "gpu_probe": gpu_probe,
        "output_dir": outdir,
        "rows": rows,
        "cpu_gpu_comparison": comparisons,
        "gpu_bottleneck_summary": gpu_bottlenecks,
        "warm_phase_bottleneck_summary": warm_phase_bottlenecks,
    }
    _write_json(summary, payload)

    print(f"[bench-freeb-direct-coil-matrix] wrote {summary}")
    for row in rows:
        detail = row.get("reason") or row.get("child_status") or row.get("returncode")
        print(f"[bench-freeb-direct-coil-matrix] {row['backend']} {row['label']}: {row['status']} ({detail})")
    for item in comparisons:
        ratios = item.get("ratios_gpu_over_cpu", {})
        warm_ratio = ratios.get("warm_min")
        cold_ratio = ratios.get("cold_or_compile")
        if warm_ratio is None and cold_ratio is None:
            continue
        warm_text = "n/a" if warm_ratio is None else f"{warm_ratio:.2f}x"
        cold_text = "n/a" if cold_ratio is None else f"{cold_ratio:.2f}x"
        print(
            "[bench-freeb-direct-coil-matrix] "
            f"gpu/cpu {item['label']} {item['case']}: cold={cold_text} warm_min={warm_text}"
        )
    for item in gpu_bottlenecks[:5]:
        print(
            "[bench-freeb-direct-coil-matrix] "
            f"gpu bottleneck {item['label']} {item['case']} {item['phase']}: "
            f"{item['ratio_gpu_over_cpu']:.2f}x, +{item['gpu_minus_cpu_s']:.4g}s"
        )
    for item in warm_phase_bottlenecks[:5]:
        print(
            "[bench-freeb-direct-coil-matrix] "
            f"warm phase {item['backend']} {item['label']} {item['case']} {item['phase']}: "
            f"{_format_optional_seconds(item.get('seconds'))}, "
            f"{_format_optional_pct(item.get('fraction_of_solve'))} solve, "
            f"per_iter={_format_optional_seconds(item.get('per_iter_s'))}"
        )
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
