#!/usr/bin/env python3
"""Profile vmec_jax exact fixed-boundary optimization callbacks.

This is intentionally a diagnostics tool, not a tutorial example.  It mirrors
the QA/QH fixed-resolution examples but keeps the run short and prints a timing
breakdown from :class:`vmec_jax.FixedBoundaryExactOptimizer`.
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

import numpy as np

_PROCESS_START = time.perf_counter()

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ACCEPTED_REPLAY_PROFILE_NAMES = (
    "jacobian_tape_replay",
    "jacobian_projected_replay_total",
    "jacobian_fused_projected_replay_total",
    "jacobian_chunked_projected_replay_projection_total",
    "gradient_tape_replay",
    "state_tangent_tape_replay",
    "b_cartesian_tangent_tape_replay",
    "linear_operator_tape_vjp",
)
TAPE_BUILD_PROFILE_NAMES = ("exact_tape_build",)
RESIDUAL_TANGENT_PROFILE_NAMES = (
    "jacobian_residual_tangents",
    "jacobian_projected_replay_residual_tangents",
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--problem", choices=("qa", "qh", "qp"), default="qa")
    p.add_argument("--max-mode", type=int, default=1)
    p.add_argument("--max-nfev", type=int, default=3)
    p.add_argument("--inner-max-iter", type=int, default=0)
    p.add_argument("--inner-ftol", type=float, default=0.0)
    p.add_argument("--trial-max-iter", type=int, default=300)
    p.add_argument("--trial-ftol", type=float, default=1e-10)
    p.add_argument("--solver-device", choices=("auto", "cpu", "gpu", "default"), default="auto")
    p.add_argument(
        "--exact-jit-forces",
        choices=("auto", "on", "off"),
        default="auto",
        help=(
            "Override jit_forces for accepted-point exact/tape solves. "
            "This is diagnostics-only; auto preserves optimizer defaults."
        ),
    )
    p.add_argument("--mpol", type=int, default=5)
    p.add_argument("--ntor", type=int, default=5)
    p.add_argument(
        "--stellarator-asymmetric",
        action="store_true",
        help="Set LASYM=T, include RBS/ZBC boundary parameters, and seed zero asymmetric modes.",
    )
    p.add_argument(
        "--asymmetric-seed",
        type=float,
        default=1.0e-7,
        help="Seed value applied to zero RBS/ZBC optimization parameters when --stellarator-asymmetric is set.",
    )
    p.add_argument("--ess", action="store_true")
    p.add_argument("--alpha", type=float, default=0.8)
    p.add_argument(
        "--method",
        choices=(
            "auto",
            "auto_scalar",
            "auto_adjoint",
            "scipy",
            "scipy_matrix_free",
            "gauss_newton",
            "lbfgs_adjoint",
            "scalar_trust",
        ),
        default="scipy",
    )
    p.add_argument("--scipy-tr-solver", choices=("lsmr", "exact", "none"), default="lsmr")
    p.add_argument("--lsmr-maxiter", type=int, default=0)
    p.add_argument(
        "--lbfgs-step-bound",
        type=float,
        default=0.01,
        help="Scaled-space half-width for method=lbfgs_adjoint.",
    )
    p.add_argument(
        "--scalar-step-bound",
        type=float,
        default=0.01,
        help="Initial/max scaled-space trust radius for method=scalar_trust.",
    )
    p.add_argument(
        "--scalar-cost-only-trials",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="For method=scalar_trust, filter trial points through forward residual solves before exact gradients.",
    )
    p.add_argument("--trace-outdir", type=str, default="")
    p.add_argument(
        "--device-memory-profile-out",
        type=str,
        default="",
        help=(
            "Optional path for jax.profiler.save_device_memory_profile(). "
            "Use with pprof/XProf to inspect live device buffers after the run."
        ),
    )
    p.add_argument("--json-out", type=str, default="")
    p.add_argument(
        "--callback",
        choices=("trial", "exact", "accepted", "jacobian", "gradient", "linear", "run"),
        default="run",
        help=(
            "Profile one callback family and exit. 'run' executes the short "
            "optimizer run controlled by --max-nfev. 'accepted' is an alias "
            "for the exact accepted-point residual/tape callback."
        ),
    )
    p.add_argument("--repeats", type=int, default=1, help="Callback repetitions for --callback modes.")
    p.add_argument(
        "--perturb-scale",
        type=float,
        default=0.0,
        help=(
            "When profiling callback modes, use deterministic distinct parameter "
            "vectors with this RMS perturbation scale. This measures realistic "
            "new accepted-point tape/replay cost instead of same-point cache hits."
        ),
    )
    p.add_argument(
        "--perturb-seed",
        type=int,
        default=1234,
        help="Random seed for --perturb-scale callback points.",
    )
    p.add_argument(
        "--clear-between-repeats",
        action="store_true",
        help=(
            "Clear point-specific exact optimizer caches between callback repetitions. "
            "Shape/branch initial-tangent caches are preserved."
        ),
    )
    p.add_argument(
        "--gradient-only",
        action="store_true",
        help="Profile one exact reverse-adjoint scalar-gradient callback instead of running the optimizer.",
    )
    p.add_argument(
        "--check-gradient",
        action="store_true",
        help="Also build the dense exact Jacobian and compare the reverse gradient to J.T @ r.",
    )
    p.add_argument(
        "--check-linear-operator",
        action="store_true",
        help="Build the dense exact Jacobian and compare matrix-free Jv/J.Tv products at the initial point.",
    )
    p.add_argument(
        "--linear-operator-repeats",
        type=int,
        default=1,
        help="Number of same-point matvec/rmatvec products to time in --check-linear-operator mode.",
    )
    p.add_argument(
        "--trial-use-scan",
        action="store_true",
        help="Legacy alias for --trial-scan=on.",
    )
    p.add_argument(
        "--trial-scan",
        choices=("auto", "on", "off"),
        default="auto",
        help=(
            "Trial residual solve policy: auto uses vmec_jax defaults, on forces "
            "the lax.scan trial path, and off forces the Python-loop trial path. "
            "Exact adjoint solves are unchanged."
        ),
    )
    p.add_argument(
        "--initial-metrics",
        action="store_true",
        help=(
            "Compute and print initial aspect/QS before profiling. Off by "
            "default because it performs an exact solve that is immediately "
            "cleared and otherwise warms/distorts cold callback timings."
        ),
    )
    p.add_argument(
        "--trace-callbacks",
        action="store_true",
        help="Include SciPy residual/Jacobian callback source timings in the JSON history.",
    )
    p.add_argument(
        "--run-repeats",
        type=int,
        default=1,
        help=(
            "Repeat the short optimizer run in the same Python process. Between "
            "repeats, point/tape caches are cleared while compiled JAX/XLA "
            "executables remain warm. This separates in-process warm runtime "
            "from first-run compilation overhead."
        ),
    )
    p.add_argument(
        "--vmec-timing",
        action="store_true",
        help="Enable VMEC_JAX_TIMING so exact tape profiles include solver phase timings.",
    )
    p.add_argument(
        "--vmec-timing-detail",
        action="store_true",
        help=(
            "Enable detailed VMEC_JAX_TIMING_DETAIL preconditioner subphase timings. "
            "This adds extra synchronization and is for diagnostics only."
        ),
    )
    p.add_argument(
        "--sync-replay-timing",
        action="store_true",
        help=(
            "Enable VMEC_JAX_OPT_SYNC_REPLAY_TIMING so exact callback replay/tangent "
            "timers split dispatch from device-ready time. Diagnostics only."
        ),
    )
    p.set_defaults(jvp_only_exact_tape=None)
    p.add_argument(
        "--jvp-only-exact-tape",
        dest="jvp_only_exact_tape",
        action="store_true",
        help=(
            "Enable VMEC_JAX_OPT_JVP_ONLY_EXACT_TAPE for forward-tangent exact "
            "callbacks. Diagnostics only; use to compare the lean tape path."
        ),
    )
    p.add_argument(
        "--no-jvp-only-exact-tape",
        dest="jvp_only_exact_tape",
        action="store_false",
        help="Explicitly disable VMEC_JAX_OPT_JVP_ONLY_EXACT_TAPE in the child profile.",
    )
    p.set_defaults(jvp_only_basepoint_carries=None)
    p.add_argument(
        "--jvp-only-basepoint-carries",
        dest="jvp_only_basepoint_carries",
        action="store_true",
        help=(
            "Enable VMEC_JAX_JVP_ONLY_EXACT_TAPE_BASEPOINT_CARRIES=1. "
            "Use with --jvp-only-exact-tape for GPU replay diagnostics."
        ),
    )
    p.add_argument(
        "--no-jvp-only-basepoint-carries",
        dest="jvp_only_basepoint_carries",
        action="store_false",
        help="Explicitly disable VMEC_JAX_JVP_ONLY_EXACT_TAPE_BASEPOINT_CARRIES.",
    )
    p.add_argument(
        "--budget-total-wall-s",
        type=float,
        default=0.0,
        help="Fail/warn when total measured callback wall time exceeds this many seconds.",
    )
    p.add_argument(
        "--budget-repeat-wall-s",
        type=float,
        default=0.0,
        help="Fail/warn when any single callback repeat exceeds this many seconds.",
    )
    p.add_argument(
        "--budget-rss-growth-mb",
        type=float,
        default=0.0,
        help="Fail/warn when process RSS grows by more than this many MiB during callback profiling.",
    )
    p.add_argument(
        "--budget-cache-entries",
        type=int,
        default=None,
        help="Fail/warn when final observed in-process cache entries exceed this total.",
    )
    p.add_argument(
        "--budget-cache-entry-growth",
        type=int,
        default=None,
        help="Fail/warn when observed in-process cache entries grow by more than this amount.",
    )
    p.add_argument(
        "--budget-tape-build-wall-s",
        type=float,
        default=0.0,
        help="Fail/warn when cumulative accepted-point exact tape build time exceeds this many seconds.",
    )
    p.add_argument(
        "--budget-replay-wall-s",
        type=float,
        default=0.0,
        help="Fail/warn when cumulative accepted-point tape replay time exceeds this many seconds.",
    )
    p.add_argument(
        "--budget-residual-tangent-wall-s",
        type=float,
        default=0.0,
        help="Fail/warn when dense residual-tangent projection time exceeds this many seconds.",
    )
    p.add_argument(
        "--budget-accepted-replays",
        type=int,
        default=None,
        help="Fail/warn when accepted-point replay profile counts exceed this total.",
    )
    p.add_argument(
        "--budget-action",
        choices=("fail", "warn"),
        default="fail",
        help="Whether exceeded callback/cache budgets should fail the process or only warn.",
    )
    return p


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return _build_parser().parse_args(argv)


def _normalize_callback_args(args: argparse.Namespace) -> argparse.Namespace:
    """Apply legacy callback aliases after parsing."""
    if args.gradient_only and not args.check_gradient:
        args.callback = "gradient"
    if getattr(args, "trial_use_scan", False):
        args.trial_scan = "on"
    return args


def _iota_mean_fn(vj, state, *, static, indata, signgs):
    chips, iotas, iotaf = vj.equilibrium_iota_profiles_from_state(
        state=state, static=static, indata=indata, signgs=signgs
    )
    del chips, iotaf
    iotas = np.asarray(iotas, dtype=float)
    return 0.0 if iotas.size <= 1 else float(np.mean(iotas[1:]))


def _print_profile(profile: dict[str, dict]) -> None:
    rows = sorted(
        (
            name,
            int(rec.get("count", 0)),
            float(rec.get("wall_time_s", 0.0)),
            float(rec.get("mean_wall_time_s", 0.0)),
        )
        for name, rec in profile.items()
    )
    rows.sort(key=lambda row: row[2], reverse=True)
    print("\nCallback timing profile:")
    print(f"{'name':48s} {'count':>7s} {'total_s':>12s} {'mean_s':>12s}")
    for name, count, total, mean in rows:
        print(f"{name:48s} {count:7d} {total:12.3f} {mean:12.3f}")


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _jax_runtime_info(jax_module: Any, *, phase_timing: dict[str, float] | None = None) -> dict[str, object]:
    try:
        backend = str(jax_module.default_backend())
    except Exception:
        backend = "unknown"
    try:
        t_devices = time.perf_counter()
        devices = [str(device) for device in jax_module.devices()]
        devices_wall_s = float(time.perf_counter() - t_devices)
        if phase_timing is not None and "jax_devices_s" not in phase_timing:
            phase_timing["jax_devices_s"] = devices_wall_s
    except Exception:
        devices = []
        devices_wall_s = None
    info: dict[str, object] = {
        "jax_version": getattr(jax_module, "__version__", None),
        "default_backend": backend,
        "devices": devices,
        "xla_python_client_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        "vmec_jax_opt_jvp_only_exact_tape": os.environ.get("VMEC_JAX_OPT_JVP_ONLY_EXACT_TAPE"),
        "vmec_jax_jvp_only_exact_tape_basepoint_carries": os.environ.get(
            "VMEC_JAX_JVP_ONLY_EXACT_TAPE_BASEPOINT_CARRIES"
        ),
        "vmec_jax_opt_trial_scan": os.environ.get("VMEC_JAX_OPT_TRIAL_SCAN"),
    }
    if devices_wall_s is not None:
        info["jax_devices_s"] = devices_wall_s
    return info


def _runtime_info(*, phase_timing: dict[str, float] | None = None) -> dict[str, object]:
    try:
        import jax

        return _jax_runtime_info(jax, phase_timing=phase_timing)
    except Exception as exc:  # pragma: no cover - diagnostics only
        return {"error": repr(exc)}


def _phase_timing_payload(
    phase_timing: dict[str, float] | None,
    *,
    wall_time_s: float | None = None,
) -> dict[str, Any]:
    out = dict(phase_timing or {})
    if wall_time_s is not None:
        out["run_wall_s"] = float(wall_time_s)
    return _json_safe(out)


def _profile_timing_alias(profile: dict[str, dict[str, float | int]]) -> dict[str, float | int]:
    timing: dict[str, float | int] = {}
    for name, rec in sorted(profile.items()):
        if not isinstance(rec, dict) or "wall_time_s" not in rec:
            continue
        try:
            value = float(rec.get("wall_time_s", 0.0))
        except Exception:
            continue
        timing[str(name)] = int(value) if str(name).endswith("_count") else value
    return timing


def _attach_common_timing_aliases(
    payload: dict[str, Any],
    *,
    wall_time_s: float | None,
    phase_timing: dict[str, float] | None,
    profile: dict[str, dict[str, float | int]] | None = None,
    timing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if wall_time_s is None:
        for key in ("total_wall_time_s", "wall_time_s", "runtime_s"):
            try:
                value = payload.get(key)
                if value is not None:
                    wall_time_s = float(value)
                    break
            except Exception:
                continue
    if wall_time_s is not None:
        payload.setdefault("total_wall_time_s", float(wall_time_s))
        payload["wall_time_s"] = float(wall_time_s)
    payload["phase_timing"] = _phase_timing_payload(phase_timing, wall_time_s=wall_time_s)
    if timing is None:
        profile_source = profile if isinstance(profile, dict) else payload.get("profile")
        timing = _profile_timing_alias(profile_source if isinstance(profile_source, dict) else {})
    payload["timing"] = _json_safe(timing)
    return payload


def _history_payload_with_aliases(
    history: dict[str, Any],
    *,
    phase_timing: dict[str, float] | None = None,
) -> dict[str, Any]:
    payload = dict(history)
    profile = payload.get("profile")
    return _attach_common_timing_aliases(
        payload,
        wall_time_s=None,
        phase_timing=phase_timing,
        profile=profile if isinstance(profile, dict) else None,
    )


def _attach_replay_scan_cache_diagnostics(
    payload: dict[str, Any],
    diagnostics: dict[str, int | float],
) -> dict[str, Any]:
    """Attach replay-scan cache counters to optimizer run payloads."""

    if diagnostics:
        payload["replay_scan_cache_diagnostics"] = dict(diagnostics)
    return payload


def _attach_optimizer_run_metadata(
    payload: dict[str, Any],
    *,
    args: argparse.Namespace,
    specs_count: int,
    solver_device_resolved: str,
) -> dict[str, Any]:
    """Attach stable run metadata to optimizer-history JSON payloads."""

    payload.setdefault("problem", str(args.problem))
    payload.setdefault("max_mode", int(args.max_mode))
    payload.setdefault("dofs", int(specs_count))
    payload.setdefault("method", str(args.method))
    payload.setdefault("solver_device_requested", str(args.solver_device))
    payload.setdefault("solver_device_resolved", str(solver_device_resolved))
    return payload


def _sum_timing_aliases(payloads: list[dict[str, Any]]) -> dict[str, float | int]:
    out: dict[str, float | int] = {}
    for payload in payloads:
        timing = payload.get("timing")
        if not isinstance(timing, dict):
            continue
        for key, value in timing.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, int):
                out[str(key)] = int(out.get(str(key), 0)) + int(value)
            elif isinstance(value, float):
                out[str(key)] = float(out.get(str(key), 0.0)) + float(value)
    return out


def _env_flag_enabled(name: str) -> bool:
    flag = os.getenv(name, "").strip().lower()
    return flag in ("1", "true", "yes", "on")


def _requested_jvp_only_exact_tape(args: argparse.Namespace) -> bool:
    value = getattr(args, "jvp_only_exact_tape", None)
    if value is not None:
        return bool(value)
    return _env_flag_enabled("VMEC_JAX_OPT_JVP_ONLY_EXACT_TAPE")


def _requested_jvp_only_basepoint_carries(args: argparse.Namespace) -> bool:
    value = getattr(args, "jvp_only_basepoint_carries", None)
    if value is not None:
        return bool(value)
    return _env_flag_enabled("VMEC_JAX_JVP_ONLY_EXACT_TAPE_BASEPOINT_CARRIES")


def _profile_has_jvp_only_exact_tape(profile: dict[str, dict[str, float | int]]) -> bool:
    for name in ("exact_solve_with_tape_jvp_only_total", "exact_tape_build_jvp_only"):
        rec = profile.get(name)
        if not isinstance(rec, dict):
            continue
        if int(rec.get("count", 0) or 0) > 0 or float(rec.get("wall_time_s", 0.0) or 0.0) > 0.0:
            return True
    return False


def _runtime_default_backend(runtime: dict[str, object] | None) -> str:
    if not isinstance(runtime, dict):
        return ""
    return str(runtime.get("default_backend") or "").strip().lower()


def _effective_jvp_only_exact_tape(
    args: argparse.Namespace,
    profile: dict[str, dict[str, float | int]],
) -> bool:
    return _profile_has_jvp_only_exact_tape(profile) or _requested_jvp_only_exact_tape(args)


def _effective_jvp_only_basepoint_carries(
    args: argparse.Namespace,
    profile: dict[str, dict[str, float | int]],
    runtime: dict[str, object] | None,
) -> bool:
    value = getattr(args, "jvp_only_basepoint_carries", None)
    if value is not None:
        return bool(value)
    env_value = os.getenv("VMEC_JAX_JVP_ONLY_EXACT_TAPE_BASEPOINT_CARRIES")
    if env_value is not None and env_value.strip() != "":
        return _env_flag_enabled("VMEC_JAX_JVP_ONLY_EXACT_TAPE_BASEPOINT_CARRIES")
    backend = _runtime_default_backend(runtime)
    return _profile_has_jvp_only_exact_tape(profile) and backend in ("gpu", "cuda", "rocm")


def _cache_len(value: Any) -> int:
    try:
        return int(len(value))
    except Exception:
        return 0


def _cache_info_currsize(fn: Any) -> int:
    try:
        info = fn.cache_info()
        return int(getattr(info, "currsize", 0))
    except Exception:
        return 0


def _module_cache_lengths(module_name: str, names: tuple[str, ...]) -> dict[str, int]:
    module = sys.modules.get(module_name)
    if module is None:
        return {}
    return {name: _cache_len(getattr(module, name, None)) for name in names}


def _sum_cache_entries(tree: Any) -> int:
    if isinstance(tree, dict):
        return int(sum(_sum_cache_entries(value) for key, value in tree.items() if key != "total_entries"))
    if isinstance(tree, bool):
        return 0
    if isinstance(tree, int):
        return int(tree)
    return 0


def _cache_snapshot(opt: Any | None = None, *, include_global: bool = True) -> dict[str, Any]:
    """Return small cache cardinalities without retaining cache contents."""
    out: dict[str, Any] = {}
    if opt is not None:
        out["optimizer"] = {
            "exact_cache": _cache_len(getattr(opt, "_exact_cache", None)),
            "exact_state_cache": _cache_len(getattr(opt, "_exact_state_cache", None)),
            "exact_residual_cache": _cache_len(getattr(opt, "_exact_residual_cache", None)),
            "exact_jacobian_cache": _cache_len(getattr(opt, "_exact_jacobian_cache", None)),
            "trial_residual_cache": _cache_len(getattr(opt, "_trial_residual_cache", None)),
            "initial_state_cache": _cache_len(getattr(opt, "_initial_state_cache", None)),
            "exact_state_key_by_id": _cache_len(getattr(opt, "_exact_state_key_by_id", None)),
            "initial_tangent_cache": _cache_len(getattr(opt, "_initial_tangent_cache", None)),
            "discrete_jacobian_helper_cache": _cache_len(
                getattr(opt, "_discrete_jacobian_helper_cache", None)
            ),
            "scan_exact_helper_cache": _cache_len(getattr(opt, "_scan_exact_helper_cache", None)),
        }
    if include_global:
        out["solve"] = _module_cache_lengths(
            "vmec_jax.solve",
            (
                "_SCAN_RUNNER_CACHE",
                "_COMPUTE_FORCES_CACHE",
                "_STRICT_UPDATE_STEP_JIT_CACHE",
            ),
        )
        out["discrete_adjoint"] = _module_cache_lengths(
            "vmec_jax.discrete_adjoint",
            (
                "_CHECKPOINT_TAPE_SCAN_CACHE",
                "_CHECKPOINT_TAPE_DYNAMIC_SCAN_CACHE",
                "_CHECKPOINT_TAPE_DYNAMIC_BASEPOINT_SCAN_CACHE",
                "_CHECKPOINT_TAPE_DYNAMIC_BASEPOINT_VJP_SCAN_CACHE",
            ),
        )
        precond = _module_cache_lengths(
            "vmec_jax.preconditioner_1d_jax",
            ("_LAMBDA_PRECOND_JIT_CACHE",),
        )
        precond_module = sys.modules.get("vmec_jax.preconditioner_1d_jax")
        if precond_module is not None:
            precond["_make_rz_preconditioner_apply_jit"] = _cache_info_currsize(
                getattr(precond_module, "_make_rz_preconditioner_apply_jit", None)
            )
        out["preconditioner_1d_jax"] = precond
        out["vmec_numpy_forces"] = _module_cache_lengths(
            "vmec_jax.vmec_numpy_forces",
            ("_NP_STACK_CACHE",),
        )
    out["total_entries"] = _sum_cache_entries(out)
    return out


def _flatten_ints(tree: Any, *, prefix: str = "") -> dict[str, int]:
    if isinstance(tree, dict):
        out: dict[str, int] = {}
        for key, value in tree.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten_ints(value, prefix=child_prefix))
        return out
    if isinstance(tree, bool):
        return {}
    if isinstance(tree, int):
        return {prefix: int(tree)}
    return {}


def _cache_growth(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_flat = _flatten_ints(before)
    after_flat = _flatten_ints(after)
    keys = sorted(set(before_flat) | set(after_flat))
    entries = {
        key: {
            "before": int(before_flat.get(key, 0)),
            "after": int(after_flat.get(key, 0)),
            "delta": int(after_flat.get(key, 0) - before_flat.get(key, 0)),
        }
        for key in keys
        if key != "total_entries"
    }
    return {
        "total_entries_before": int(before.get("total_entries", 0)),
        "total_entries_after": int(after.get("total_entries", 0)),
        "total_entries_delta": int(after.get("total_entries", 0) - before.get("total_entries", 0)),
        "entries": entries,
    }


def _profile_delta(
    before: dict[str, dict[str, float | int]],
    after: dict[str, dict[str, float | int]],
) -> dict[str, dict[str, float | int]]:
    out: dict[str, dict[str, float | int]] = {}
    for name in sorted(set(before) | set(after)):
        rec_before = before.get(name, {})
        rec_after = after.get(name, {})
        count = int(rec_after.get("count", 0)) - int(rec_before.get("count", 0))
        wall = float(rec_after.get("wall_time_s", 0.0)) - float(rec_before.get("wall_time_s", 0.0))
        if count or abs(wall) > 0.0:
            out[name] = {
                "count": count,
                "wall_time_s": wall,
                "mean_wall_time_s": wall / count if count else 0.0,
            }
    return out


def _profile_wall_time(profile: dict[str, dict[str, float | int]], names: tuple[str, ...]) -> float:
    return float(sum(float(profile.get(name, {}).get("wall_time_s", 0.0)) for name in names))


def _profile_count(profile: dict[str, dict[str, float | int]], names: tuple[str, ...]) -> int:
    return int(sum(int(profile.get(name, {}).get("count", 0)) for name in names))


def _replay_scan_cache_snapshot(*, reset: bool = False) -> dict[str, int | float]:
    try:
        from vmec_jax.discrete_adjoint import replay_scan_cache_diagnostics

        return dict(replay_scan_cache_diagnostics(reset=reset))
    except Exception:
        return {}


def _replay_scan_cache_delta(
    before: dict[str, int | float],
    after: dict[str, int | float],
) -> dict[str, int | float]:
    out: dict[str, int | float] = {}
    for key in sorted(set(before) | set(after)):
        b = before.get(key, 0)
        a = after.get(key, 0)
        if str(key).endswith("_count"):
            out[key] = int(a) - int(b)
        else:
            out[key] = float(a) - float(b)
    return out


def _sum_replay_scan_cache_diagnostics(samples: list[dict[str, Any]]) -> dict[str, int | float]:
    out: dict[str, int | float] = {}
    for sample in samples:
        diagnostics = sample.get("replay_scan_cache_diagnostics")
        if not isinstance(diagnostics, dict):
            continue
        for key, value in diagnostics.items():
            if str(key).endswith("_count"):
                out[key] = int(out.get(key, 0)) + int(value)
            else:
                out[key] = float(out.get(key, 0.0)) + float(value)
    return out


def _current_rss_bytes() -> int | None:
    try:
        import psutil  # type: ignore

        return int(psutil.Process().memory_info().rss)
    except Exception:
        pass
    proc_statm = Path("/proc/self/statm")
    if proc_statm.exists():
        try:
            pages = int(proc_statm.read_text(encoding="utf-8").split()[1])
            return pages * int(os.sysconf("SC_PAGE_SIZE"))
        except Exception:
            pass
    try:
        rss_kb = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return int(rss_kb) * 1024
    except Exception:
        pass
    try:
        import resource

        maxrss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform == "darwin":
            return maxrss
        return maxrss * 1024
    except Exception:
        return None


def _budget_limits(args: argparse.Namespace) -> dict[str, float | int | None]:
    cache_entries = args.budget_cache_entries
    cache_entry_growth = args.budget_cache_entry_growth
    accepted_replays = args.budget_accepted_replays
    return {
        "total_wall_s": float(args.budget_total_wall_s) if float(args.budget_total_wall_s) > 0.0 else None,
        "repeat_wall_s": float(args.budget_repeat_wall_s) if float(args.budget_repeat_wall_s) > 0.0 else None,
        "rss_growth_mb": float(args.budget_rss_growth_mb) if float(args.budget_rss_growth_mb) > 0.0 else None,
        "cache_entries": cache_entries if cache_entries is not None and int(cache_entries) >= 0 else None,
        "cache_entry_growth": (
            cache_entry_growth if cache_entry_growth is not None and int(cache_entry_growth) >= 0 else None
        ),
        "tape_build_wall_s": (
            float(args.budget_tape_build_wall_s) if float(args.budget_tape_build_wall_s) > 0.0 else None
        ),
        "replay_wall_s": float(args.budget_replay_wall_s) if float(args.budget_replay_wall_s) > 0.0 else None,
        "residual_tangent_wall_s": (
            float(args.budget_residual_tangent_wall_s)
            if float(args.budget_residual_tangent_wall_s) > 0.0
            else None
        ),
        "accepted_replays": accepted_replays if accepted_replays is not None and int(accepted_replays) >= 0 else None,
    }


def _evaluate_budgets(
    *,
    args: argparse.Namespace,
    samples: list[dict[str, Any]],
    profile: dict[str, dict[str, float | int]],
    total_wall_s: float,
    cache_growth: dict[str, Any],
    rss_before_bytes: int | None,
    rss_after_bytes: int | None,
) -> dict[str, Any]:
    limits = _budget_limits(args)
    max_repeat_wall_s = max((float(sample.get("wall_time_s", 0.0)) for sample in samples), default=0.0)
    rss_growth_mb = None
    if rss_before_bytes is not None and rss_after_bytes is not None:
        rss_growth_mb = (int(rss_after_bytes) - int(rss_before_bytes)) / (1024.0 * 1024.0)
    measurements = {
        "total_wall_s": float(total_wall_s),
        "max_repeat_wall_s": float(max_repeat_wall_s),
        "rss_growth_mb": rss_growth_mb,
        "cache_entries": int(cache_growth.get("total_entries_after", 0)),
        "cache_entry_growth": int(cache_growth.get("total_entries_delta", 0)),
        "tape_build_wall_s": _profile_wall_time(profile, TAPE_BUILD_PROFILE_NAMES),
        "replay_wall_s": _profile_wall_time(profile, ACCEPTED_REPLAY_PROFILE_NAMES),
        "residual_tangent_wall_s": _profile_wall_time(profile, RESIDUAL_TANGENT_PROFILE_NAMES),
        "accepted_replays": _profile_count(profile, ACCEPTED_REPLAY_PROFILE_NAMES),
    }
    exceeded: list[dict[str, float | int | str]] = []

    def _check(name: str, value: float | int | None, limit: float | int | None) -> None:
        if value is None or limit is None:
            return
        if float(value) > float(limit):
            exceeded.append({"name": name, "value": value, "limit": limit})

    _check("total_wall_s", measurements["total_wall_s"], limits["total_wall_s"])
    _check("repeat_wall_s", measurements["max_repeat_wall_s"], limits["repeat_wall_s"])
    _check("rss_growth_mb", measurements["rss_growth_mb"], limits["rss_growth_mb"])
    _check("cache_entries", measurements["cache_entries"], limits["cache_entries"])
    _check("cache_entry_growth", measurements["cache_entry_growth"], limits["cache_entry_growth"])
    _check("tape_build_wall_s", measurements["tape_build_wall_s"], limits["tape_build_wall_s"])
    _check("replay_wall_s", measurements["replay_wall_s"], limits["replay_wall_s"])
    _check(
        "residual_tangent_wall_s",
        measurements["residual_tangent_wall_s"],
        limits["residual_tangent_wall_s"],
    )
    _check("accepted_replays", measurements["accepted_replays"], limits["accepted_replays"])
    return {
        "ok": not exceeded,
        "action": str(args.budget_action),
        "limits": limits,
        "measurements": measurements,
        "exceeded": exceeded,
    }


def _build_callback_payload(
    *,
    args: argparse.Namespace,
    specs_count: int,
    solver_device_resolved: str,
    samples: list[dict[str, Any]],
    profile: dict[str, dict[str, float | int]],
    cache_before: dict[str, Any],
    cache_after: dict[str, Any],
    rss_before_bytes: int | None,
    rss_after_bytes: int | None,
    total_wall_s: float,
    phase_timing: dict[str, float] | None = None,
    runtime: dict[str, object] | None = None,
    trace_outdir: str | None = None,
    device_memory_profile_out: str | None = None,
) -> dict[str, Any]:
    growth = _cache_growth(cache_before, cache_after)
    budget_status = _evaluate_budgets(
        args=args,
        samples=samples,
        profile=profile,
        total_wall_s=total_wall_s,
        cache_growth=growth,
        rss_before_bytes=rss_before_bytes,
        rss_after_bytes=rss_after_bytes,
    )
    payload = {
        "schema_version": 2,
        "report_kind": "exact_optimizer_callback_profile",
        "problem": args.problem,
        "max_mode": int(args.max_mode),
        "dofs": int(specs_count),
        "callback": "exact" if args.callback == "accepted" else args.callback,
        "perturb_scale": float(args.perturb_scale),
        "perturb_seed": int(args.perturb_seed),
        "clear_between_repeats": bool(args.clear_between_repeats),
        "initial_metrics": bool(getattr(args, "initial_metrics", False)),
        "sync_replay_timing": bool(getattr(args, "sync_replay_timing", False)),
        "trial_scan": str(getattr(args, "trial_scan", "auto")),
        "jvp_only_exact_tape_requested": _requested_jvp_only_exact_tape(args),
        "jvp_only_basepoint_carries_requested": _requested_jvp_only_basepoint_carries(args),
        "jvp_only_exact_tape": _effective_jvp_only_exact_tape(args, profile),
        "jvp_only_basepoint_carries": _effective_jvp_only_basepoint_carries(args, profile, runtime),
        "solver_device_requested": args.solver_device,
        "solver_device_resolved": solver_device_resolved,
        "runtime": _runtime_info() if runtime is None else runtime,
        "trace_outdir": trace_outdir,
        "device_memory_profile_out": device_memory_profile_out,
        "total_wall_time_s": float(total_wall_s),
        "rss_before_bytes": rss_before_bytes,
        "rss_after_bytes": rss_after_bytes,
        "samples": samples,
        "profile": profile,
        "replay_scan_cache_diagnostics": _sum_replay_scan_cache_diagnostics(samples),
        "cache": {
            "before": cache_before,
            "after": cache_after,
            "growth": growth,
        },
        "budget_status": budget_status,
    }
    return _attach_common_timing_aliases(
        payload,
        wall_time_s=float(total_wall_s),
        phase_timing=phase_timing,
        profile=profile,
    )


def _start_profiler_trace(jax_module: Any, trace_outdir: str | None) -> str | None:
    """Start a JAX profiler trace and return the resolved output directory."""
    if not trace_outdir:
        return None
    trace_out = Path(trace_outdir).expanduser().resolve()
    trace_out.mkdir(parents=True, exist_ok=True)
    jax_module.profiler.start_trace(str(trace_out))
    return str(trace_out)


def _stop_profiler_trace(jax_module: Any, trace_outdir: str | None) -> None:
    if not trace_outdir:
        return
    jax_module.profiler.stop_trace()
    print(f"Trace written to {trace_outdir}")


def _save_device_memory_profile(jax_module: Any, output_path: str | None) -> str | None:
    """Save a JAX device-memory profile and return the resolved path."""
    if not output_path:
        return None
    mem_out = Path(output_path).expanduser().resolve()
    mem_out.parent.mkdir(parents=True, exist_ok=True)
    jax_module.profiler.save_device_memory_profile(str(mem_out))
    print(f"Device memory profile written to {mem_out}")
    return str(mem_out)


def _clear_optimizer_point_caches(opt) -> None:
    """Clear solved-state/tape caches without dropping structural helpers."""
    opt._exact_cache.clear()
    opt._exact_state_cache.clear()
    if hasattr(opt, "_exact_state_key_by_id"):
        opt._exact_state_key_by_id.clear()
    if hasattr(opt, "_exact_residual_cache"):
        opt._exact_residual_cache.clear()
    if hasattr(opt, "_exact_jacobian_cache"):
        opt._exact_jacobian_cache.clear()
    opt._trial_residual_cache.clear()
    if hasattr(opt, "_initial_state_cache"):
        opt._initial_state_cache.clear()
    # Initial tangent columns are keyed by structural branch, not by accepted
    # point. Keep them warm while forcing cold exact accepted-point tapes.
    opt._last_jacobian_residual = None


def _install_profile_timing_supplements(opt) -> None:
    """Record new solver timing buckets while preserving older optimizer builds."""

    original = opt._profile_solver_timing
    supplemental_keys = (
        ("scan_setup_s", "scan_setup"),
        ("scan_run_setup_s", "scan_run_setup"),
        ("scan_runner_cache_lookup_s", "scan_runner_cache_lookup"),
        ("scan_runner_cache_build_s", "scan_runner_cache_build"),
        ("compute_forces_first_s", "compute_forces_first"),
        ("compute_forces_rest_s", "compute_forces_rest"),
        ("scan_runner_cache_hit_device_run_s", "scan_runner_cache_hit_device_run"),
        ("scan_runner_cache_hit_dispatch_s", "scan_runner_cache_hit_dispatch"),
        ("scan_runner_cache_hit_ready_s", "scan_runner_cache_hit_ready"),
        ("scan_runner_cache_miss_device_run_s", "scan_runner_cache_miss_device_run"),
        ("scan_runner_cache_miss_dispatch_s", "scan_runner_cache_miss_dispatch"),
        ("scan_runner_cache_miss_ready_s", "scan_runner_cache_miss_ready"),
        ("scan_runner_cache_bypass_device_run_s", "scan_runner_cache_bypass_device_run"),
        ("scan_runner_cache_bypass_dispatch_s", "scan_runner_cache_bypass_dispatch"),
        ("scan_runner_cache_bypass_ready_s", "scan_runner_cache_bypass_ready"),
        ("iteration_control_fsq1_precond_norm_s", "iteration_control_fsq1_precond_norm"),
        ("iteration_control_fsq1_scalar_build_s", "iteration_control_fsq1_scalar_build"),
        ("iteration_control_fsq1_payload_get_s", "iteration_control_fsq1_payload_get"),
        ("iteration_control_fsq1_direct_get_s", "iteration_control_fsq1_direct_get"),
        ("iteration_control_fsq1_unattributed_s", "iteration_control_fsq1_unattributed"),
        ("iteration_control_badjac_ptau_get_s", "iteration_control_badjac_ptau_get"),
        ("iteration_control_badjac_state_jacobian_s", "iteration_control_badjac_state_jacobian"),
        ("iteration_control_badjac_unattributed_s", "iteration_control_badjac_unattributed"),
    )
    supplemental_counter_keys = (
        ("scan_runner_cache_hit_count", "scan_runner_cache_hit_count"),
        ("scan_runner_cache_miss_count", "scan_runner_cache_miss_count"),
        ("scan_runner_cache_bypass_count", "scan_runner_cache_bypass_count"),
    )

    def _profile_solver_timing_with_supplements(
        diagnostics,
        *,
        profile_prefix: str,
        phase_wall_s: float,
        unattributed_name: str | None,
    ) -> float:
        before_counts = {
            f"{profile_prefix}_{suffix}": int(
                getattr(opt, "_profile", {}).get(f"{profile_prefix}_{suffix}", {}).get("count", 0)
            )
            for _key, suffix in supplemental_keys + supplemental_counter_keys
        }
        solver_total = original(
            diagnostics,
            profile_prefix=profile_prefix,
            phase_wall_s=phase_wall_s,
            unattributed_name=unattributed_name,
        )
        if not isinstance(diagnostics, dict):
            return solver_total
        timing = diagnostics.get("timing")
        if not isinstance(timing, dict):
            return solver_total
        for key, suffix in supplemental_keys:
            if key not in timing:
                continue
            profile_name = f"{profile_prefix}_{suffix}"
            after_count = int(getattr(opt, "_profile", {}).get(profile_name, {}).get("count", 0))
            if after_count != before_counts[profile_name]:
                continue
            try:
                value = float(timing.get(key, 0.0))
            except Exception:
                continue
            opt._profile_add(profile_name, value)
        for key, suffix in supplemental_counter_keys:
            if key not in timing:
                continue
            profile_name = f"{profile_prefix}_{suffix}"
            after_count = int(getattr(opt, "_profile", {}).get(profile_name, {}).get("count", 0))
            if after_count != before_counts[profile_name]:
                continue
            try:
                value = int(timing.get(key, 0))
            except Exception:
                continue
            add_counter = getattr(opt, "_profile_add_counter", None)
            if callable(add_counter):
                add_counter(profile_name, value)
            else:
                opt._profile_add(profile_name, float(value))
        for key, raw_value in sorted(timing.items()):
            if not (str(key).startswith("scan_runner_cache_miss_category_") and str(key).endswith("_count")):
                continue
            profile_name = f"{profile_prefix}_{key}"
            if profile_name in before_counts:
                after_count = int(getattr(opt, "_profile", {}).get(profile_name, {}).get("count", 0))
                if after_count != before_counts[profile_name]:
                    continue
            try:
                value = int(raw_value)
            except Exception:
                continue
            add_counter = getattr(opt, "_profile_add_counter", None)
            if callable(add_counter):
                add_counter(profile_name, value)
            else:
                opt._profile_add(profile_name, float(value))
        return solver_total

    opt._profile_solver_timing = _profile_solver_timing_with_supplements


def main() -> int:
    args = _normalize_callback_args(_parse_args())
    phase_timing: dict[str, float] = {
        "process_to_main_s": float(time.perf_counter() - _PROCESS_START),
    }
    if args.vmec_timing or args.vmec_timing_detail:
        os.environ["VMEC_JAX_TIMING"] = "1"
    if args.vmec_timing_detail:
        os.environ["VMEC_JAX_TIMING_DETAIL"] = "1"
    if args.sync_replay_timing:
        os.environ["VMEC_JAX_OPT_SYNC_REPLAY_TIMING"] = "1"
    if args.jvp_only_exact_tape is not None:
        os.environ["VMEC_JAX_OPT_JVP_ONLY_EXACT_TAPE"] = (
            "1" if bool(args.jvp_only_exact_tape) else "0"
        )
    if args.jvp_only_basepoint_carries is not None:
        os.environ["VMEC_JAX_JVP_ONLY_EXACT_TAPE_BASEPOINT_CARRIES"] = (
            "1" if bool(args.jvp_only_basepoint_carries) else "0"
        )
    if args.trial_scan == "on":
        os.environ["VMEC_JAX_OPT_TRIAL_SCAN"] = "1"
    elif args.trial_scan == "off":
        os.environ["VMEC_JAX_OPT_TRIAL_SCAN"] = "0"

    t_import = time.perf_counter()
    import vmec_jax as vj
    from vmec_jax._compat import enable_x64, jax
    from vmec_jax.config import config_from_indata
    from vmec_jax.optimization import rebuild_indata_with_resolution
    phase_timing["vmec_jax_import_s"] = float(time.perf_counter() - t_import)
    if jax is not None:
        t_devices = time.perf_counter()
        try:
            _ = jax.devices()
        finally:
            phase_timing["jax_devices_pre_run_s"] = float(time.perf_counter() - t_devices)

    enable_x64(True)

    root = Path(__file__).resolve().parents[2]
    problem_defaults = {
        "qa": {
            "input": "input.nfp2_QA",
            "helicity_m": 1,
            "helicity_n": 0,
            "target_aspect": 6.0,
            "target_iota": 0.41,
            "min_abs_iota": None,
            "iota_weight": 1.0,
        },
        "qh": {
            "input": "input.nfp4_QH_warm_start",
            "helicity_m": 1,
            "helicity_n": -1,
            "target_aspect": 7.0,
            "target_iota": None,
            "min_abs_iota": None,
            "iota_weight": 1.0,
        },
        "qp": {
            "input": "input.nfp2_QI",
            "helicity_m": 0,
            "helicity_n": -1,
            "target_aspect": 5.0,
            "target_iota": None,
            "min_abs_iota": 0.41,
            # The QP example uses tuple weight=40000, i.e. residual scale=200.
            "iota_weight": 200.0,
        },
    }
    problem_cfg = problem_defaults[str(args.problem)]
    input_file = root / "examples" / "data" / str(problem_cfg["input"])
    helicity_m = int(problem_cfg["helicity_m"])
    helicity_n = int(problem_cfg["helicity_n"])
    target_aspect = float(problem_cfg["target_aspect"])
    target_iota = problem_cfg["target_iota"]
    min_abs_iota = problem_cfg["min_abs_iota"]

    cfg, indata = vj.load_config(str(input_file))
    indata = rebuild_indata_with_resolution(indata, mpol=args.mpol, ntor=args.ntor)
    if args.stellarator_asymmetric:
        scalars = dict(indata.scalars)
        scalars["LASYM"] = True
        indexed = {key: dict(value) for key, value in indata.indexed.items()}
        from vmec_jax.namelist import InData

        indata = InData(scalars=scalars, indexed=indexed, source_path=indata.source_path)
    cfg = config_from_indata(indata)
    static = vj.build_static(cfg)
    boundary = vj.boundary_from_indata(indata, static.modes, apply_m1_constraint=False)
    indata, static, boundary = vj.extend_boundary_for_max_mode(
        indata, static, boundary, int(args.max_mode)
    )
    boundary_input = vj.boundary_input_from_indata(indata, static.modes)
    specs = vj.boundary_param_specs(
        boundary_input,
        static.modes,
        max_mode=int(args.max_mode),
        min_coeff=0.0,
        include=("rc", "zs", "rs", "zc") if args.stellarator_asymmetric else ("rc", "zs"),
        fix=("rc00",),
    )
    residuals_fn = vj.make_qs_residuals_fn(
        static,
        indata,
        helicity_m=helicity_m,
        helicity_n=helicity_n,
        target_aspect=target_aspect,
        target_iota=target_iota,
        min_abs_iota=min_abs_iota,
        surfaces=np.arange(0.0, 1.01, 0.1),
        aspect_weight=1.0,
        iota_weight=float(problem_cfg["iota_weight"]),
        qs_weight=1.0,
    )
    opt = vj.FixedBoundaryExactOptimizer(
        static,
        indata,
        boundary,
        specs,
        residuals_fn,
        boundary_input=boundary_input,
        inner_max_iter=args.inner_max_iter,
        inner_ftol=args.inner_ftol,
        trial_max_iter=args.trial_max_iter,
        trial_ftol=args.trial_ftol,
        solver_device=args.solver_device,
    )
    if args.exact_jit_forces == "on":
        opt._exact_solver_kwargs["jit_forces"] = True
    elif args.exact_jit_forces == "off":
        opt._exact_solver_kwargs["jit_forces"] = False
    _install_profile_timing_supplements(opt)
    if args.trial_scan == "on":
        opt._trial_solver_kwargs["use_scan"] = True
    elif args.trial_scan == "off":
        opt._trial_solver_kwargs["use_scan"] = False
    params0 = np.zeros(len(specs))
    if args.stellarator_asymmetric and float(args.asymmetric_seed) != 0.0:
        for index, spec in enumerate(specs):
            if spec.kind in ("rs", "zc"):
                params0[index] = float(args.asymmetric_seed)
    x_scale = vj.create_x_scale(specs, alpha=float(args.alpha)) if args.ess else np.ones(len(specs))

    print(
        f"Problem={args.problem} max_mode={args.max_mode} dofs={len(specs)} "
        f"lasym={args.stellarator_asymmetric} "
        f"inner=({args.inner_max_iter}, {args.inner_ftol:g}) "
        f"trial=({args.trial_max_iter}, {args.trial_ftol:g})"
    )
    runtime_info = _runtime_info(phase_timing=phase_timing)
    print(f"Requested solver_device={args.solver_device} resolved={opt._solver_device_name or 'default'}")
    print(f"Runtime={json.dumps(runtime_info, sort_keys=True)}")
    if args.initial_metrics:
        print(f"Initial aspect={opt.aspect_ratio(params0):.6f} qs={opt.quasisymmetry_objective(params0):.6e}")
    opt.clear_caches()
    opt._profile = {}

    if args.callback != "run":
        repeats = max(1, int(args.repeats))
        perturb_scale = float(args.perturb_scale)
        rng = np.random.default_rng(int(args.perturb_seed))
        samples: list[dict[str, Any]] = []
        cache_before = _cache_snapshot(opt)
        rss_before = _current_rss_bytes()
        total_t0 = time.perf_counter()
        trace_outdir = _start_profiler_trace(jax, args.trace_outdir)
        try:
            for repeat in range(repeats):
                if repeat > 0 and args.clear_between_repeats:
                    _clear_optimizer_point_caches(opt)
                if perturb_scale > 0.0:
                    params = params0 + perturb_scale * rng.standard_normal(params0.shape)
                else:
                    params = params0
                profile_before = opt._profile_dump()
                repeat_cache_before = _cache_snapshot(opt)
                replay_scan_before = _replay_scan_cache_snapshot(reset=True)
                repeat_rss_before = _current_rss_bytes()
                t0 = time.perf_counter()
                if args.callback == "trial":
                    value = opt.forward_residual_fun(params)
                    metric = float(np.linalg.norm(value))
                    shape = list(np.asarray(value).shape)
                    extra: dict[str, Any] = {}
                elif args.callback in ("exact", "accepted"):
                    value = opt.residual_fun(params)
                    metric = float(np.linalg.norm(value))
                    shape = list(np.asarray(value).shape)
                    extra = {}
                elif args.callback == "jacobian":
                    value = opt.jacobian_fun(params)
                    metric = float(np.linalg.norm(value))
                    shape = list(np.asarray(value).shape)
                    extra = {}
                elif args.callback == "gradient":
                    cost, grad = opt.objective_and_gradient_fun(params)
                    metric = float(np.linalg.norm(grad))
                    shape = [int(np.asarray(grad).size)]
                    extra = {"cost": float(cost)}
                elif args.callback == "linear":
                    op = opt.residual_linear_operator(params)
                    direction = np.ones(len(specs), dtype=float)
                    cotangent = np.ones(op.shape[0], dtype=float)
                    jv = op.matvec(direction)
                    jtw = op.rmatvec(cotangent)
                    metric = float(np.linalg.norm(jv) + np.linalg.norm(jtw))
                    shape = [int(op.shape[0]), int(op.shape[1])]
                    extra = {}
                else:  # pragma: no cover - guarded by argparse
                    raise ValueError(args.callback)
                wall_time_s = time.perf_counter() - t0
                profile_after = opt._profile_dump()
                replay_scan_after = _replay_scan_cache_snapshot(reset=True)
                repeat_cache_after = _cache_snapshot(opt)
                repeat_rss_after = _current_rss_bytes()
                repeat_growth = _cache_growth(repeat_cache_before, repeat_cache_after)
                sample = {
                    "repeat": repeat,
                    "wall_time_s": wall_time_s,
                    "metric_norm": metric,
                    "param_step_norm": float(np.linalg.norm(params - params0)),
                    "shape": shape,
                    "profile_delta": _profile_delta(profile_before, profile_after),
                    "replay_scan_cache_diagnostics": _replay_scan_cache_delta(
                        replay_scan_before,
                        replay_scan_after,
                    ),
                    "cache_before": repeat_cache_before,
                    "cache_after": repeat_cache_after,
                    "cache_growth": repeat_growth,
                    "rss_before_bytes": repeat_rss_before,
                    "rss_after_bytes": repeat_rss_after,
                }
                sample.update(extra)
                samples.append(sample)
        finally:
            _stop_profiler_trace(jax, trace_outdir)
        profile = opt._profile_dump()
        total_wall_s = time.perf_counter() - total_t0
        cache_after = _cache_snapshot(opt)
        rss_after = _current_rss_bytes()
        device_memory_profile_out = _save_device_memory_profile(
            jax,
            args.device_memory_profile_out,
        )
        callback_payload = _build_callback_payload(
            args=args,
            specs_count=len(specs),
            solver_device_resolved=opt._solver_device_name or "default",
            samples=samples,
            profile=profile,
            cache_before=cache_before,
            cache_after=cache_after,
            rss_before_bytes=rss_before,
            rss_after_bytes=rss_after,
            total_wall_s=total_wall_s,
            phase_timing=phase_timing,
            runtime=runtime_info,
            trace_outdir=trace_outdir,
            device_memory_profile_out=device_memory_profile_out,
        )
        effective_callback = callback_payload["callback"]
        print(f"\nCallback={effective_callback} repeats={repeats}")
        for sample in samples:
            print(
                f"  repeat={sample['repeat']} wall={float(sample['wall_time_s']):.3f}s "
                f"norm={float(sample['metric_norm']):.6e} "
                f"||dx||={float(sample['param_step_norm']):.3e} "
                f"shape={sample['shape']} "
                f"cache_delta={sample['cache_growth']['total_entries_delta']}"
            )
        _print_profile(profile)
        cache_growth = callback_payload["cache"]["growth"]
        budget_status = callback_payload["budget_status"]
        print(
            "\nCache growth: "
            f"entries {cache_growth['total_entries_before']} -> "
            f"{cache_growth['total_entries_after']} "
            f"(delta {cache_growth['total_entries_delta']})"
        )
        if budget_status["exceeded"]:
            print("\nBudget exceeded:")
            for item in budget_status["exceeded"]:
                print(f"  {item['name']}: value={item['value']} limit={item['limit']}")
        elif any(value is not None for value in budget_status["limits"].values()):
            print("\nBudgets OK")
        if args.json_out:
            out = Path(args.json_out).expanduser().resolve()
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(_json_safe(callback_payload), indent=2), encoding="utf-8")
            print(f"Wrote {out}")
        if budget_status["exceeded"] and budget_status["action"] == "fail":
            return 2
        return 0

    if args.check_linear_operator:
        rng = np.random.default_rng(1234)
        jac = opt.jacobian_fun(params0)
        op = opt.residual_linear_operator(params0)
        v = rng.standard_normal(jac.shape[1])
        w = rng.standard_normal(jac.shape[0])
        jv_ref = np.asarray(jac, dtype=float) @ v
        jv_op = op.matvec(v)
        vmat = rng.standard_normal((jac.shape[1], min(3, jac.shape[1])))
        jvmat_ref = np.asarray(jac, dtype=float) @ vmat
        jvmat_op = op.matmat(vmat)
        jtw_ref = np.asarray(jac, dtype=float).T @ w
        jtw_op = op.rmatvec(w)
        for _ in range(max(1, int(args.linear_operator_repeats)) - 1):
            _ = op.matvec(v)
            _ = op.rmatvec(w)
        jv_err = float(np.linalg.norm(jv_op - jv_ref) / max(np.linalg.norm(jv_ref), 1.0))
        jvmat_err = float(np.linalg.norm(jvmat_op - jvmat_ref) / max(np.linalg.norm(jvmat_ref), 1.0))
        jtw_err = float(np.linalg.norm(jtw_op - jtw_ref) / max(np.linalg.norm(jtw_ref), 1.0))
        print(f"LinearOperator check: rel ||Jv - dense Jv||={jv_err:.6e}")
        print(f"LinearOperator check: rel ||JX - dense JX||={jvmat_err:.6e}")
        print(f"LinearOperator check: rel ||J.Tw - dense J.Tw||={jtw_err:.6e}")
        _print_profile(opt._profile_dump())
        return 0

    if args.gradient_only:
        gradient_t0 = time.perf_counter()
        cost, grad = opt.objective_and_gradient_fun(params0)
        print(
            f"\nReverse-gradient callback: cost={cost:.6e} "
            f"||grad||={float(np.linalg.norm(grad)):.6e}"
        )
        if args.check_gradient:
            res = opt.residual_fun(params0)
            jac = opt.jacobian_fun(params0)
            dense_grad = np.asarray(jac, dtype=float).T @ np.asarray(res, dtype=float)
            abs_err = float(np.linalg.norm(grad - dense_grad))
            rel_err = abs_err / max(float(np.linalg.norm(dense_grad)), 1.0)
            print(
                f"Gradient check: ||g_adj - J.T r||={abs_err:.6e} "
                f"relative={rel_err:.6e}"
            )
        profile = opt._profile_dump()
        gradient_wall_s = float(time.perf_counter() - gradient_t0)
        _print_profile(profile)
        if args.json_out:
            out = Path(args.json_out).expanduser().resolve()
            out.parent.mkdir(parents=True, exist_ok=True)
            payload = _attach_common_timing_aliases(
                {
                    "problem": args.problem,
                    "max_mode": int(args.max_mode),
                    "dofs": len(specs),
                    "cost": float(cost),
                    "gradient_norm": float(np.linalg.norm(grad)),
                    "profile": profile,
                    "runtime": runtime_info,
                },
                wall_time_s=gradient_wall_s,
                phase_timing=phase_timing,
                profile=profile,
            )
            out.write_text(
                json.dumps(_json_safe(payload), indent=2),
                encoding="utf-8",
            )
            print(f"Wrote {out}")
        return 0

    trace_out = Path(args.trace_outdir).expanduser().resolve() if args.trace_outdir else None
    if trace_out is not None:
        import jax

        trace_out.mkdir(parents=True, exist_ok=True)
        jax.profiler.start_trace(str(trace_out))

    run_repeats = max(1, int(args.run_repeats))
    histories: list[dict[str, object]] = []
    replay_scan_diagnostics_by_repeat: list[dict[str, Any]] = []
    result = None
    try:
        tr_solver = None if args.scipy_tr_solver == "none" else args.scipy_tr_solver
        for repeat in range(run_repeats):
            if repeat > 0:
                _clear_optimizer_point_caches(opt)
                opt._profile = {}
            if run_repeats > 1:
                print(f"\n=== optimizer run repeat {repeat + 1}/{run_repeats} ===")
            result = opt.run(
                params0,
                method=args.method,
                max_nfev=args.max_nfev,
                ftol=1e-3,
                gtol=1e-3,
                xtol=1e-3,
                x_scale=x_scale,
                verbose=1,
                iota_fn=(
                    None
                    if args.problem != "qa"
                    else lambda state: _iota_mean_fn(vj, state, static=static, indata=indata, signgs=opt._signgs)
                ),
                target_iota=target_iota,
                target_aspect=target_aspect,
                scipy_tr_solver=tr_solver,
                scipy_lsmr_maxiter=None if args.lsmr_maxiter <= 0 else int(args.lsmr_maxiter),
                lbfgs_step_bound=float(args.lbfgs_step_bound),
                scalar_step_bound=float(args.scalar_step_bound),
                scalar_cost_only_trials=args.scalar_cost_only_trials,
                trace_callbacks=args.trace_callbacks,
            )
            replay_scan_diag = _replay_scan_cache_snapshot(reset=True)
            replay_scan_diagnostics_by_repeat.append({"replay_scan_cache_diagnostics": replay_scan_diag})
            hist_repeat = _history_payload_with_aliases(dict(result["_history_dump"]))
            _attach_replay_scan_cache_diagnostics(hist_repeat, replay_scan_diag)
            _attach_optimizer_run_metadata(
                hist_repeat,
                args=args,
                specs_count=len(specs),
                solver_device_resolved=opt._solver_device_name or "default",
            )
            hist_repeat["repeat"] = repeat
            hist_repeat["runtime"] = runtime_info
            histories.append(hist_repeat)
    finally:
        if trace_out is not None:
            import jax

            jax.profiler.stop_trace()
            print(f"Trace written to {trace_out}")

    if result is None:  # pragma: no cover - defensive
        raise RuntimeError("optimizer run did not produce a result")
    hist = _history_payload_with_aliases(dict(result["_history_dump"]), phase_timing=phase_timing)
    _attach_replay_scan_cache_diagnostics(
        hist,
        _sum_replay_scan_cache_diagnostics(replay_scan_diagnostics_by_repeat),
    )
    _attach_optimizer_run_metadata(
        hist,
        args=args,
        specs_count=len(specs),
        solver_device_resolved=opt._solver_device_name or "default",
    )
    hist["runtime"] = runtime_info
    print(
        f"\nFinal objective={hist['objective_final']:.6e} qs={hist['qs_final']:.6e} "
        f"aspect={hist['aspect_final']:.6f} wall={hist['total_wall_time_s']:.3f}s "
        f"nfev={hist['nfev']} njev={hist['njev']}"
    )
    if "iota_final" in hist:
        print(f"Final iota={hist['iota_final']:.6f} target={hist.get('target_iota'):.6f}")
    _print_profile(hist.get("profile", {}))

    if args.json_out:
        out = Path(args.json_out).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        if run_repeats == 1:
            payload = hist
        else:
            repeat_wall_s = float(
                sum(float(run.get("wall_time_s", 0.0)) for run in histories if isinstance(run, dict))
            )
            payload = _attach_common_timing_aliases(
                {
                    "problem": args.problem,
                    "max_mode": int(args.max_mode),
                    "dofs": len(specs),
                    "method": args.method,
                    "solver_device_requested": args.solver_device,
                    "solver_device_resolved": opt._solver_device_name or "default",
                    "runtime": runtime_info,
                    "run_repeats": run_repeats,
                    "runs": histories,
                },
                wall_time_s=repeat_wall_s,
                phase_timing=phase_timing,
                timing=_sum_timing_aliases(histories),
            )
        out.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
        print(f"Wrote {out}")

    if args.device_memory_profile_out:
        import jax

        mem_out = Path(args.device_memory_profile_out).expanduser().resolve()
        mem_out.parent.mkdir(parents=True, exist_ok=True)
        jax.profiler.save_device_memory_profile(str(mem_out))
        print(f"Device memory profile written to {mem_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
