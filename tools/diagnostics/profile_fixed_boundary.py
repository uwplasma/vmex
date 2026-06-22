#!/usr/bin/env python3
"""Profile vmec_jax fixed-boundary iterations with JAX profiler.

Example:
  python tools/diagnostics/profile_fixed_boundary.py --input /path/to/input.qa_signgs1 --iters 3 --outdir tmp/jax_profile
"""

from __future__ import annotations

import argparse
import cProfile
from contextlib import contextmanager
import json
import os
from pathlib import Path
import pstats
import resource
import sys
from typing import Any
import time

import numpy as np

_PROCESS_START = time.perf_counter()
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Match vmec_jax's import-time defaults before this diagnostics tool imports
# JAX directly.  Otherwise persistent-cache hits can emit repeated harmless
# PjRt/XLA compatibility warnings before vmec_jax has a chance to configure the
# logging environment.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="VMEC input file")
    p.add_argument("--iters", type=int, default=3, help="Number of iterations to run")
    p.add_argument("--outdir", type=str, default="tmp/jax_profile", help="Profiler trace output directory")
    p.add_argument("--no-warmup", action="store_true", help="Disable warmup run")
    p.add_argument("--simple-profile", action="store_true", help="Use a timing-only profiler (no TensorBoard trace)")
    p.add_argument("--json-out", type=str, default=None, help="Write a compact JSON timing/solver diagnostic summary.")
    p.add_argument("--cprofile-out", type=str, default=None, help="Write cProfile stats for the timed profiled run.")
    p.add_argument(
        "--cprofile-text-out",
        type=str,
        default=None,
        help="Write a text cProfile summary for the timed profiled run.",
    )
    p.add_argument(
        "--vmec-timing",
        action="store_true",
        help="Enable VMEC_JAX_TIMING so non-scan fixed-boundary runs include solver phase timings in JSON.",
    )
    p.add_argument(
        "--vmec-timing-detail",
        action="store_true",
        help=(
            "Enable detailed VMEC_JAX_TIMING_DETAIL preconditioner subphase timings. "
            "This adds extra synchronization and is for diagnostics only."
        ),
    )
    p.add_argument("--jit-forces", action="store_true", help="Enable jit_forces (default)")
    p.add_argument("--no-jit-forces", action="store_true", help="Disable jit_forces")
    p.add_argument("--use-input-niter", action="store_true", help="Use NITER from input for staging")
    p.set_defaults(use_scan=None)
    p.add_argument(
        "--use-scan",
        dest="use_scan",
        action="store_true",
        help="Force the lax.scan iteration path.",
    )
    p.add_argument(
        "--no-use-scan",
        dest="use_scan",
        action="store_false",
        help="Force the Python/VMEC-control iteration path. By default the profiler follows production policy.",
    )
    p.add_argument(
        "--solver-mode",
        choices=("auto", "default", "parity", "accelerated"),
        default="auto",
        help="Solver policy passed to run_fixed_boundary (default: auto).",
    )
    p.add_argument(
        "--solver-device",
        choices=("auto", "default", "cpu", "gpu"),
        default="auto",
        help="JAX solver device override passed to run_fixed_boundary (default: auto).",
    )
    p.add_argument(
        "--finish-policy",
        choices=("auto", "none", "bounded", "converge"),
        default=None,
        help=(
            "Public fixed-boundary post-solve finish policy. If omitted, the legacy "
            "--auto-cli-policy/--no-auto-cli-policy switch selects auto/none."
        ),
    )
    p.set_defaults(auto_cli_policy=True)
    p.add_argument(
        "--auto-cli-policy",
        dest="auto_cli_policy",
        action="store_true",
        help="Allow run_fixed_boundary to apply its public CLI-style accelerated finish policy (default).",
    )
    p.add_argument(
        "--no-auto-cli-policy",
        dest="auto_cli_policy",
        action="store_false",
        help="Benchmark the raw requested solver path without the public CLI-style finish policy.",
    )
    p.set_defaults(dynamic_scan=False)
    p.add_argument(
        "--dynamic-scan",
        dest="dynamic_scan",
        action="store_true",
        help="Allow the dynamic scan/non-scan selector probes during profiling.",
    )
    p.add_argument(
        "--no-dynamic-scan",
        dest="dynamic_scan",
        action="store_false",
        help="Disable dynamic scan selector probes during profiling (default).",
    )
    p.set_defaults(multigrid=None)
    p.add_argument("--multigrid", dest="multigrid", action="store_true", help="Force multigrid staging.")
    p.add_argument("--no-multigrid", dest="multigrid", action="store_false", help="Force a direct single-grid solve.")
    p.add_argument(
        "--require-scan",
        action="store_true",
        help="Exit with an error if diagnostics show that the requested run did not use the scan path.",
    )
    p.add_argument(
        "--require-no-scan",
        action="store_true",
        help="Exit with an error if diagnostics show that the requested run used the scan path.",
    )
    p.add_argument("--dump-hlo", action="store_true", help="Dump tomnsps_rzl HLO to the output directory")
    return p.parse_args()


@contextmanager
def _temporary_env(updates: dict[str, str | None]):
    previous = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


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


def _peak_rss_mib() -> float | None:
    """Return process peak RSS in MiB using platform-specific ru_maxrss units."""
    try:
        rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return None
    if sys.platform == "darwin":
        return rss / (1024.0 * 1024.0)
    return rss / 1024.0


def _effective_jit_forces(args: argparse.Namespace) -> bool:
    """Return the production default unless the profiler flag overrides it."""
    if bool(getattr(args, "jit_forces", False)) and bool(getattr(args, "no_jit_forces", False)):
        raise ValueError("Specify at most one of --jit-forces or --no-jit-forces.")
    if bool(getattr(args, "jit_forces", False)):
        return True
    if bool(getattr(args, "no_jit_forces", False)):
        return False
    return True


def _compact_diagnostics(diag: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "solver_mode",
        "accelerated_mode",
        "cli_fixed_boundary_mode",
        "fixed_boundary_finish_policy",
        "cli_fixed_boundary_finish_enabled",
        "cli_fixed_boundary_initial_policy",
        "cli_accelerated_fixed_policy",
        "cli_staged_followup_policy",
        "cli_fixed_boundary_full_parity_fallback",
        "cli_fixed_boundary_partial_parity_fallback",
        "cli_fixed_boundary_staged_followup_used",
        "cli_fixed_boundary_finish_budgets",
        "cli_fixed_boundary_finish_modes",
        "cli_fixed_boundary_finish_fsq",
        "cli_fixed_boundary_finish_converged",
        "cli_fixed_boundary_finish_budget_cap",
        "cli_fixed_boundary_finish_budget_exhausted",
        "multigrid_user_provided",
        "multigrid_ns_stages",
        "multigrid_niter_stages",
        "multigrid_ftol_stages",
        "multigrid_stage_modes",
        "multigrid_stage_wall_s",
        "multigrid_stage_solve_total_s",
        "multigrid_final_stage_niter_exhausted",
        "accelerated_single_grid_default",
        "cli_accelerated_stage_wall_s",
        "cli_accelerated_stage_solve_total_s",
        "cli_staged_followup_stage_wall_s",
        "cli_staged_followup_stage_solve_total_s",
        "cli_fixed_boundary_staged_followup_wall_s",
        "cli_fixed_boundary_staged_followup_solve_total_s",
        "accelerated_scan",
        "accelerated_stage_chunked",
        "accelerated_stage_chunk_count",
        "accelerated_stage_chunk_iters",
        "accelerated_stage_early_switch",
        "accelerated_stage_switch_reason",
        "accelerated_stage_probe_chunk_iters",
        "accelerated_stage_effective_mode",
        "use_scan",
        "vmec2000_scan",
        "scan_use_precomputed",
        "scan_use_lax_tridi",
        "abort_scan",
        "host_update_assembly",
        "host_fsq1_norms_on_accelerator",
        "host_residual_metrics_on_accelerator",
        "jit_strict_update_enabled",
        "jit_strict_update_work",
        "jit_strict_update_cpu_work_limit",
        "numpy_preconditioner_apply",
        "numpy_preconditioner_apply_mode_count",
        "numpy_preconditioner_apply_max_iter_cutoff",
        "numpy_preconditioner_apply_min_mode_count",
        "numpy_force_fast_path",
        "numpy_force_fast_path_active",
        "numpy_force_fast_path_max_iter",
        "requested_ftol",
        "fsq_total_target",
        "final_fsqr",
        "final_fsqz",
        "final_fsql",
        "converged",
        "converged_strict",
        "converged_by_total_fsq",
        "setup_axis_reset_applied",
        "setup_axis_reset_done",
        "setup_axis_force_probe_available",
        "setup_axis_force_probe_reused",
        "solver_device",
        "solver_device_auto_reroute",
        "solver_device_requested_backend",
        "timing",
    )
    return {key: _json_safe(diag[key]) for key in keys if key in diag}


def _summarize_run(*, args: argparse.Namespace, run: Any, wall_time: float | None, jax_module: Any) -> dict[str, Any]:
    res = getattr(run, "result", None)
    diag = dict(getattr(res, "diagnostics", {}) or {})
    effective_jit_forces = getattr(args, "effective_jit_forces", None)
    if effective_jit_forces is None:
        effective_jit_forces = _effective_jit_forces(args)
    effective_finish_policy = getattr(args, "effective_finish_policy", None)
    if effective_finish_policy is None:
        finish_policy = getattr(args, "finish_policy", None)
        effective_finish_policy = finish_policy if finish_policy is not None else ("auto" if bool(args.auto_cli_policy) else "none")
    w_hist = np.asarray(getattr(res, "w_history", np.zeros((0,), dtype=float)), dtype=float)
    fsqr_hist = np.asarray(getattr(res, "fsqr2_history", np.zeros((0,), dtype=float)), dtype=float)
    fsqz_hist = np.asarray(getattr(res, "fsqz2_history", np.zeros((0,), dtype=float)), dtype=float)
    fsql_hist = np.asarray(getattr(res, "fsql2_history", np.zeros((0,), dtype=float)), dtype=float)
    try:
        t_devices = time.perf_counter()
        devices = [str(d) for d in jax_module.devices()]
        devices_wall_s = time.perf_counter() - t_devices
    except Exception:
        devices = []
        devices_wall_s = None
    try:
        backend = str(jax_module.default_backend())
    except Exception:
        backend = "unknown"
    timing_summary = _json_safe(diag.get("timing")) if isinstance(diag.get("timing"), dict) else None
    phase_timing = dict(getattr(args, "phase_timing", {}) or {})
    if devices_wall_s is not None and "jax_devices_s" not in phase_timing:
        phase_timing["jax_devices_s"] = float(devices_wall_s)
    if wall_time is not None:
        phase_timing["run_wall_s"] = float(wall_time)
    summary = {
        "input": str(Path(args.input).expanduser()),
        "requested_iters": int(args.iters),
        "wall_time_sec": None if wall_time is None else float(wall_time),
        "wall_time_s": None if wall_time is None else float(wall_time),
        "timing": timing_summary,
        "phase_timing": _json_safe(phase_timing),
        "jax_version": getattr(jax_module, "__version__", "unknown"),
        "jax_default_backend": backend,
        "jax_devices": devices,
        "process_peak_rss_mib": _peak_rss_mib(),
        "args": {
            "solver_mode": str(args.solver_mode),
            "solver_device": str(args.solver_device),
            "multigrid": args.multigrid,
            "use_input_niter": bool(args.use_input_niter),
            "use_scan": None if args.use_scan is None else bool(args.use_scan),
            "jit_forces": bool(effective_jit_forces),
            "no_jit_forces": bool(args.no_jit_forces),
            "auto_cli_policy": bool(args.auto_cli_policy),
            "finish_policy": str(effective_finish_policy),
            "dynamic_scan": bool(args.dynamic_scan),
        },
        "result": {
            "n_iter": None if res is None else int(getattr(res, "n_iter", -1)),
            "history_len": int(w_hist.size),
            "final_w": None if w_hist.size == 0 else float(w_hist[-1]),
            "final_fsqr": None if fsqr_hist.size == 0 else float(fsqr_hist[-1]),
            "final_fsqz": None if fsqz_hist.size == 0 else float(fsqz_hist[-1]),
            "final_fsql": None if fsql_hist.size == 0 else float(fsql_hist[-1]),
        },
        "diagnostics": _compact_diagnostics(diag),
    }
    return summary


def _scan_requirement_error(args: argparse.Namespace, summary: dict[str, Any]) -> str | None:
    if bool(getattr(args, "require_scan", False)) and bool(getattr(args, "require_no_scan", False)):
        return "Specify at most one of --require-scan or --require-no-scan."
    diagnostics = summary.get("diagnostics", {}) if isinstance(summary, dict) else {}
    actual_scan = bool(
        diagnostics.get("use_scan")
        or diagnostics.get("vmec2000_scan")
        or diagnostics.get("accelerated_scan")
    )
    if bool(getattr(args, "require_scan", False)) and not actual_scan:
        return "Required scan path, but run diagnostics did not report scan execution."
    if bool(getattr(args, "require_no_scan", False)) and actual_scan:
        return "Required non-scan path, but run diagnostics reported scan execution."
    return None


def _print_run_summary(summary: dict[str, Any]) -> None:
    result = summary["result"]
    diag = summary["diagnostics"]
    print(
        "[profile_fixed_boundary] "
        f"backend={summary['jax_default_backend']} "
        f"history_len={result['history_len']} "
        f"n_iter={result['n_iter']} "
        f"final_w={result['final_w']} "
        f"converged={diag.get('converged')}",
        flush=True,
    )
    policy_bits = []
    for key in (
        "solver_mode",
        "cli_fixed_boundary_mode",
        "fixed_boundary_finish_policy",
        "cli_fixed_boundary_finish_enabled",
        "cli_fixed_boundary_initial_policy",
        "multigrid_ns_stages",
        "multigrid_niter_stages",
        "multigrid_stage_modes",
        "multigrid_stage_wall_s",
        "multigrid_stage_solve_total_s",
        "cli_accelerated_stage_wall_s",
        "cli_accelerated_stage_solve_total_s",
        "cli_staged_followup_stage_wall_s",
        "cli_staged_followup_stage_solve_total_s",
        "cli_fixed_boundary_staged_followup_wall_s",
        "cli_fixed_boundary_staged_followup_solve_total_s",
        "solver_device",
        "use_scan",
        "vmec2000_scan",
        "scan_use_precomputed",
        "scan_use_lax_tridi",
        "cli_fixed_boundary_finish_budgets",
        "cli_fixed_boundary_full_parity_fallback",
    ):
        if key in diag:
            policy_bits.append(f"{key}={diag[key]}")
    if policy_bits:
        print("[profile_fixed_boundary] " + " ".join(policy_bits), flush=True)
    timing = diag.get("timing")
    if isinstance(timing, dict):
        timing_bits = []
        for key in (
            "iterations",
            "solve_total_s",
            "setup_total_s",
            "setup_static_grid_rebuild_s",
            "setup_boundary_profiles_s",
            "setup_profile_data_s",
            "setup_trig_tables_s",
            "setup_boundary_profiles_unattributed_s",
            "setup_axis_reset_s",
            "setup_cache_key_hash_s",
            "setup_ptau_constants_s",
            "setup_index_constants_s",
            "iteration_loop_s",
            "compute_forces_s",
            "compute_forces_main_reuse_count",
            "force_eval_all_s",
            "force_eval_extra_s",
            "compute_forces_main_s",
            "compute_forces_auto_flip_s",
            "compute_forces_trial_s",
            "compute_forces_backtracking_s",
            "preconditioner_s",
            "precond_refresh_s",
            "precond_refresh_seed_s",
            "precond_refresh_seed_lambda_s",
            "precond_refresh_seed_rz_matrices_s",
            "precond_apply_s",
            "precond_apply_scale_m1_rhs_s",
            "precond_apply_rz_s",
            "precond_apply_fused_payload_s",
            "precond_apply_output_blocks_s",
            "precond_apply_sync_s",
            "precond_mode_scale_s",
            "update_s",
            "update_state_s",
            "scan_total_s",
            "scan_setup_s",
            "scan_run_setup_s",
            "scan_preflight_s",
            "scan_device_run_s",
            "scan_device_dispatch_s",
            "scan_device_ready_s",
            "scan_host_materialize_s",
            "scan_postprocess_s",
            "scan_unattributed_s",
            "scan_runner_cache_hit_count",
            "scan_runner_cache_miss_count",
            "scan_runner_cache_bypass_count",
            "scan_cold_cache_miss_s",
            "scan_cold_cache_miss_ready_s",
            "scan_cache_build_wrapper_s",
        ):
            if key in timing:
                value = timing[key]
                if isinstance(value, (int, float)):
                    if key == "iterations" or key.endswith("_count"):
                        timing_bits.append(f"{key}={int(value)}")
                    else:
                        timing_bits.append(f"{key}={float(value):.6g}")
        for key, value in sorted(timing.items()):
            if not (str(key).startswith("scan_runner_cache_miss_category_") and str(key).endswith("_count")):
                continue
            if isinstance(value, (int, float)):
                timing_bits.append(f"{key}={int(value)}")
        if timing_bits:
            print("[profile_fixed_boundary] timing " + " ".join(timing_bits), flush=True)


def _run_with_optional_cprofile(args: argparse.Namespace, func):
    if not args.cprofile_out:
        return func()
    profiler = cProfile.Profile()
    try:
        profiler.enable()
        return func()
    finally:
        profiler.disable()
        cprofile_path = Path(args.cprofile_out).expanduser().resolve()
        cprofile_path.parent.mkdir(parents=True, exist_ok=True)
        profiler.dump_stats(str(cprofile_path))
        phase_timing = getattr(args, "phase_timing", None)
        if isinstance(phase_timing, dict):
            phase_timing["cprofile_out"] = str(cprofile_path)
        text_out = args.cprofile_text_out
        if text_out:
            text_path = Path(text_out).expanduser().resolve()
            text_path.parent.mkdir(parents=True, exist_ok=True)
            with text_path.open("w", encoding="utf-8") as stream:
                stats = pstats.Stats(profiler, stream=stream).sort_stats("cumtime")
                stats.print_stats(80)
            if isinstance(phase_timing, dict):
                phase_timing["cprofile_text_out"] = str(text_path)


def _dump_tomnsps_hlo(input_path: str, outdir: Path) -> None:
    try:
        import jax
        from jax import numpy as jnp
    except Exception as exc:  # pragma: no cover
        print(f"[profile_fixed_boundary] HLO dump skipped (JAX missing): {exc}")
        return
    try:
        from vmec_jax.config import load_config
        from vmec_jax.vmec_tomnsp import vmec_trig_tables, tomnsps_rzl
    except Exception as exc:  # pragma: no cover
        print(f"[profile_fixed_boundary] HLO dump skipped (vmec_jax import failed): {exc}")
        return

    cfg, _ = load_config(input_path)
    trig = vmec_trig_tables(
        ntheta=int(cfg.ntheta),
        nzeta=int(cfg.nzeta),
        nfp=int(cfg.nfp),
        mmax=int(cfg.mpol) - 1,
        nmax=int(cfg.ntor),
        lasym=bool(cfg.lasym),
        dtype=jnp.float64,
        cache=True,
    )
    ns = int(cfg.ns)
    ntheta3 = int(trig.ntheta3)
    nzeta = int(np.asarray(trig.cosnv).shape[0])
    shape = (ns, ntheta3, nzeta)
    aval = jax.ShapeDtypeStruct(shape, jnp.float64)

    def _tomnsps_core(
        armn_even,
        armn_odd,
        brmn_even,
        brmn_odd,
        crmn_even,
        crmn_odd,
        azmn_even,
        azmn_odd,
        bzmn_even,
        bzmn_odd,
        czmn_even,
        czmn_odd,
    ):
        return tomnsps_rzl(
            armn_even=armn_even,
            armn_odd=armn_odd,
            brmn_even=brmn_even,
            brmn_odd=brmn_odd,
            crmn_even=crmn_even,
            crmn_odd=crmn_odd,
            azmn_even=azmn_even,
            azmn_odd=azmn_odd,
            bzmn_even=bzmn_even,
            bzmn_odd=bzmn_odd,
            czmn_even=czmn_even,
            czmn_odd=czmn_odd,
            mpol=int(cfg.mpol),
            ntor=int(cfg.ntor),
            nfp=int(cfg.nfp),
            lasym=bool(cfg.lasym),
            trig=trig,
            include_edge=False,
            masks=None,
        )

    try:
        hlo_ir = (
            jax.jit(_tomnsps_core)
            .lower(aval, aval, aval, aval, aval, aval, aval, aval, aval, aval, aval, aval)
            .compiler_ir(dialect="hlo")
        )
        if hasattr(hlo_ir, "as_text"):
            hlo = hlo_ir.as_text()
        elif hasattr(hlo_ir, "as_hlo_text"):
            hlo = hlo_ir.as_hlo_text()
        else:
            hlo = str(hlo_ir)
    except Exception as exc:  # pragma: no cover
        print(f"[profile_fixed_boundary] HLO dump failed: {exc}")
        return

    out_path = outdir / "tomnsps_rzl.hlo.txt"
    out_path.write_text(hlo, encoding="utf-8")
    print(f"[profile_fixed_boundary] tomnsps_rzl HLO written to {out_path}")


def main() -> int:
    args = _parse_args()
    phase_timing = {
        "process_to_main_s": float(time.perf_counter() - _PROCESS_START),
    }
    try:
        # Import vmec_jax first so its _compat module can set JAX/XLA import-time
        # defaults such as GPU demand allocation and the persistent cache dir.
        t_import = time.perf_counter()
        from vmec_jax import driver as vj
        from vmec_jax._compat import jax
        phase_timing["vmec_jax_import_s"] = float(time.perf_counter() - t_import)

        if jax is None:
            raise ImportError("vmec_jax imported without JAX support")
        t_devices = time.perf_counter()
        try:
            _ = jax.devices()
        finally:
            phase_timing["jax_devices_pre_run_s"] = float(time.perf_counter() - t_devices)
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"JAX is required for profiling: {exc}") from exc

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if args.jit_forces and args.no_jit_forces:
        raise SystemExit("Specify at most one of --jit-forces or --no-jit-forces.")
    jit_forces = _effective_jit_forces(args)
    args.effective_jit_forces = jit_forces
    args.phase_timing = phase_timing
    solver_device = None if str(args.solver_device) == "auto" else str(args.solver_device)
    solver_mode = None if str(args.solver_mode) == "auto" else str(args.solver_mode)
    finish_policy = args.finish_policy if args.finish_policy is not None else ("auto" if args.auto_cli_policy else "none")
    args.effective_finish_policy = str(finish_policy)

    def _run_profile_once():
        run_kwargs = dict(
            solver="vmec2000_iter",
            solver_mode=solver_mode,
            multigrid_use_input_niter=bool(args.use_input_niter),
            multigrid=args.multigrid,
            verbose=False,
            jit_forces=bool(jit_forces),
            use_scan=args.use_scan,
            solver_device=solver_device,
            finish_policy=str(finish_policy),
        )
        if not bool(args.use_input_niter):
            run_kwargs["max_iter"] = int(args.iters)
        return vj.run_fixed_boundary(args.input, **run_kwargs)

    env_updates = {
        # The dynamic scan selector can execute multiple warm/probe solves around
        # the requested run.  Keep profiler defaults single-path unless explicitly
        # requested with --dynamic-scan.
        "VMEC_JAX_DYNAMIC_SCAN": "1" if bool(args.dynamic_scan) else "0",
    }
    if bool(args.vmec_timing) or bool(args.vmec_timing_detail):
        env_updates["VMEC_JAX_TIMING"] = "1"
    if bool(args.vmec_timing_detail):
        env_updates["VMEC_JAX_TIMING_DETAIL"] = "1"

    if bool(args.dump_hlo):
        _dump_tomnsps_hlo(str(args.input), outdir)

    with _temporary_env(env_updates):
        if not args.no_warmup:
            t_warmup = time.perf_counter()
            warm = _run_profile_once()
            try:
                res = warm.result
                if res is not None and hasattr(res, "fsqr2_history"):
                    _ = float(np.asarray(res.fsqr2_history)[-1])
            except Exception:
                pass
            phase_timing["warmup_wall_s"] = float(time.perf_counter() - t_warmup)
        else:
            phase_timing["warmup_wall_s"] = 0.0

        use_simple = bool(args.simple_profile)
        if not use_simple:
            try:
                jax.profiler.start_trace(str(outdir))
            except Exception as exc:
                print(f"[profile_fixed_boundary] profiler start failed: {exc}")
                print("[profile_fixed_boundary] falling back to timing-only profile (no trace).")
                use_simple = True

        wall_time = None
        if use_simple:
            t0 = time.perf_counter()
            run = _run_with_optional_cprofile(args, _run_profile_once)
            res = run.result
            if res is not None and hasattr(res, "fsqr2_history"):
                _ = float(np.asarray(res.fsqr2_history)[-1])
            t1 = time.perf_counter()
            wall_time = float(t1 - t0)
            print(f"[profile_fixed_boundary] total wall time: {wall_time:.3f}s")
        else:
            t0 = time.perf_counter()
            try:
                run = _run_with_optional_cprofile(args, _run_profile_once)
                res = run.result
                if res is not None and hasattr(res, "fsqr2_history"):
                    _ = float(np.asarray(res.fsqr2_history)[-1])
            finally:
                jax.profiler.stop_trace()
            wall_time = float(time.perf_counter() - t0)
            print(f"[profile_fixed_boundary] trace saved to {outdir}")

    summary = _summarize_run(args=args, run=run, wall_time=wall_time, jax_module=jax)
    scan_error = _scan_requirement_error(args, summary)
    if scan_error is not None:
        raise SystemExit(scan_error)
    _print_run_summary(summary)
    if args.json_out:
        json_path = Path(args.json_out).expanduser().resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True), encoding="utf-8")
        print(f"[profile_fixed_boundary] summary written to {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
