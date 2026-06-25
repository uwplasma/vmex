#!/usr/bin/env python3
"""Run a fixed-boundary performance decomposition and write a roadmap report.

This diagnostic is the first implementation artifact for
``plan_research_grade_performance_differentiability.md`` milestones M1 and M2.
It runs cold and warm ``vmec_jax`` profiler passes in separate subprocesses,
optionally runs VMEC2000, and writes both machine-readable JSON and a compact
Markdown report that maps VMEC2000/VMEC++ algorithmic buckets onto vmec_jax
modules and profiler keys.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


ALGORITHM_MAP: list[dict[str, Any]] = [
    {
        "bucket": "VMEC controller / Richardson iteration",
        "vmec2000_sources": [
            "VMEC2000/Sources/General/eqsolve.f",
            "VMEC2000/Sources/General/vmec_main.f",
        ],
        "vmecpp_sources": [
            "src/vmecpp/cpp/vmecpp/common/flow_control/flow_control.h",
            "src/vmecpp/cpp/vmecpp/common/util/util.h",
        ],
        "vmec_jax_sources": [
            "vmec_jax/driver.py",
            "vmec_jax/drivers/staging.py",
            "vmec_jax/solvers/fixed_boundary/residual/iteration.py",
            "vmec_jax/solvers/fixed_boundary/scan/controller.py",
        ],
        "profiler_keys": [
            "run_wall_s",
            "solve_total_s",
            "scan_total_s",
            "scan_device_run_s",
            "scan_host_materialize_s",
        ],
        "refactor_target": "Split controller policy from residual-step kernels and keep CLI fast path non-differentiable when useful.",
    },
    {
        "bucket": "Inverse Fourier transform to real-space mesh",
        "vmec2000_sources": [
            "VMEC2000/Sources/General/totzsp_mod.f",
            "VMEC2000/Sources/General/realspace.f",
        ],
        "vmecpp_sources": [
            "src/vmecpp/cpp/vmecpp/free_boundary/surface_geometry/surface_geometry.cc",
            "src/vmecpp/cpp/vmecpp/free_boundary/laplace_solver/laplace_solver.cc",
        ],
        "vmec_jax_sources": [
            "vmec_jax/vmec_tomnsp.py",
            "vmec_jax/vmec_utils.py",
        ],
        "profiler_keys": [
            "compute_forces_s",
            "force_eval_all_s",
            "scan_device_run_s",
        ],
        "refactor_target": "Precompute basis tables by shape and keep transform arrays compact and donation-friendly.",
    },
    {
        "bucket": "Covariant fields / metric assembly",
        "vmec2000_sources": [
            "VMEC2000/Sources/General/bcovar.f",
            "VMEC2000/Sources/General/jacobian.f",
        ],
        "vmecpp_sources": [
            "src/vmecpp/cpp/vmecpp/vmec/thread_local_storage/thread_local_storage.h",
        ],
        "vmec_jax_sources": [
            "vmec_jax/vmec_bcovar.py",
            "vmec_jax/vmec_forces.py",
        ],
        "profiler_keys": [
            "compute_forces_main_s",
            "force_eval_all_s",
            "force_eval_extra_s",
        ],
        "refactor_target": "Separate reusable geometry/field kernels from force residual assembly for both primal and derivative paths.",
    },
    {
        "bucket": "Forces, symmetrization, and residuals",
        "vmec2000_sources": [
            "VMEC2000/Sources/General/forces.f",
            "VMEC2000/Sources/General/residue.f90",
            "VMEC2000/Sources/General/symforce.f",
        ],
        "vmecpp_sources": [
            "src/vmecpp/cpp/vmecpp/vmec/thread_local_storage/thread_local_storage.h",
        ],
        "vmec_jax_sources": [
            "vmec_jax/vmec_forces.py",
            "vmec_jax/solvers/fixed_boundary/residual/iteration.py",
        ],
        "profiler_keys": [
            "compute_forces_s",
            "preconditioner_s",
            "update_s",
        ],
        "refactor_target": "Promote matrix-free linearized residual operators for implicit/custom-VJP derivatives.",
    },
    {
        "bucket": "Preconditioner and state update",
        "vmec2000_sources": [
            "VMEC2000/Sources/General/precondn.f",
            "VMEC2000/Sources/General/precon2d.f",
            "VMEC2000/Sources/General/blocktridiagonalsolver.f90",
        ],
        "vmecpp_sources": [
            "src/vmecpp/cpp/vmecpp/common/flow_control/flow_control.h",
        ],
        "vmec_jax_sources": [
            "vmec_jax/preconditioner_1d_jax.py",
            "vmec_jax/solvers/fixed_boundary/preconditioning",
        ],
        "profiler_keys": [
            "preconditioner_s",
            "precond_refresh_s",
            "precond_apply_s",
            "update_state_s",
        ],
        "refactor_target": "Keep preconditioner refresh cadence explicit and differentiability-aware; avoid rebuilding on every callback.",
    },
    {
        "bucket": "WOUT, diagnostics, and stability channels",
        "vmec2000_sources": [
            "VMEC2000/Sources/General/fileout.f",
            "VMEC2000/Sources/General/wrout.f",
        ],
        "vmecpp_sources": [
            "src/vmecpp/cpp/util/netcdf_io/netcdf_io.cc",
            "src/vmecpp/cpp/util/hdf5_io/hdf5_io.cc",
        ],
        "vmec_jax_sources": [
            "vmec_jax/io/wout",
            "vmec_jax/io/wout/mercier.py",
            "vmec_jax/wout.py",
        ],
        "profiler_keys": [
            "run_wall_s",
        ],
        "refactor_target": "Write only requested outputs on the fast CLI path; keep rich diagnostics opt-in or cached.",
    },
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="examples/data/input.nfp4_QH_warm_start",
        help="VMEC input deck to profile.",
    )
    parser.add_argument("--iters", type=int, default=6, help="Iteration budget for profiling smoke runs.")
    parser.add_argument("--outdir", default="outputs/performance_decomposition", help="Output directory.")
    parser.add_argument("--python", default=sys.executable, help="Python executable for child profiler runs.")
    parser.add_argument("--solver-mode", default="auto", choices=("auto", "default", "parity", "accelerated"))
    parser.add_argument("--solver-device", default="auto", choices=("auto", "default", "cpu", "gpu"))
    parser.add_argument("--use-input-niter", action="store_true", help="Use NITER from the input deck.")
    parser.add_argument(
        "--finish-policy",
        choices=("auto", "none", "bounded", "converge"),
        default=None,
        help=(
            "Public vmec_jax finish policy. If omitted, --auto-cli-policy/--no-auto-cli-policy "
            "selects auto/none for backward-compatible diagnostics."
        ),
    )
    parser.set_defaults(auto_cli_policy=True)
    parser.add_argument(
        "--auto-cli-policy",
        dest="auto_cli_policy",
        action="store_true",
        help="Allow run_fixed_boundary to apply the public CLI-style finish policy (default).",
    )
    parser.add_argument(
        "--no-auto-cli-policy",
        dest="auto_cli_policy",
        action="store_false",
        help="Benchmark the raw requested solver path without the public CLI-style finish policy.",
    )
    parser.add_argument("--skip-runs", action="store_true", help="Only write the static algorithm map.")
    parser.add_argument("--cprofile", action="store_true", help="Collect cProfile stats for timed vmec_jax child runs.")
    parser.add_argument("--skip-vmec2000", action="store_true", help="Do not run VMEC2000.")
    parser.add_argument("--vmec2000-exec", default=None, help="Optional xvmec2000 executable.")
    parser.add_argument("--skip-vmecpp", action="store_true", help="Do not run VMEC++.")
    parser.add_argument("--vmecpp-exec", default=None, help="Optional VMEC++ CLI executable.")
    parser.add_argument("--timeout-s", type=float, default=120.0, help="External executable timeout.")
    return parser.parse_args()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _run_child_profile(
    *,
    args: argparse.Namespace,
    input_path: Path,
    outdir: Path,
    label: str,
    no_warmup: bool,
) -> dict[str, Any]:
    json_out = outdir / f"vmec_jax_{label}.json"
    trace_out = outdir / f"trace_{label}"
    cmd = [
        str(args.python),
        "tools/diagnostics/profile_fixed_boundary.py",
        "--input",
        str(input_path),
        "--iters",
        str(int(args.iters)),
        "--outdir",
        str(trace_out),
        "--simple-profile",
        "--json-out",
        str(json_out),
        "--vmec-timing",
        "--solver-mode",
        str(args.solver_mode),
        "--solver-device",
        str(args.solver_device),
        "--no-dynamic-scan",
    ]
    if no_warmup:
        cmd.append("--no-warmup")
    if args.use_input_niter:
        cmd.append("--use-input-niter")
    finish_policy = args.finish_policy if args.finish_policy is not None else ("auto" if args.auto_cli_policy else "none")
    cmd.extend(["--finish-policy", str(finish_policy)])
    if args.finish_policy is None and not bool(args.auto_cli_policy):
        cmd.append("--no-auto-cli-policy")
    if args.cprofile:
        cmd.extend(
            [
                "--cprofile-out",
                str(outdir / f"vmec_jax_{label}.prof"),
                "--cprofile-text-out",
                str(outdir / f"vmec_jax_{label}_cprofile.txt"),
            ]
        )

    started = time.perf_counter()
    completed = subprocess.run(
        cmd,
        cwd=_REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    elapsed = time.perf_counter() - started
    profile = None
    if json_out.exists():
        profile = json.loads(json_out.read_text(encoding="utf-8"))
    return {
        "label": label,
        "kind": "vmec_jax",
        "returncode": int(completed.returncode),
        "elapsed_wall_s": float(elapsed),
        "command": cmd,
        "stdout_tail": completed.stdout[-4000:],
        "json_path": str(json_out),
        "profile": profile,
    }


def _run_vmec2000(
    *,
    args: argparse.Namespace,
    input_path: Path,
    outdir: Path,
) -> dict[str, Any]:
    from vmec_jax.vmec2000_exec import find_vmec2000_exec, flatten_threed1, run_xvmec2000, threed1_fsq_total

    exec_path = Path(args.vmec2000_exec).expanduser() if args.vmec2000_exec else find_vmec2000_exec()
    if exec_path is None:
        return {"kind": "vmec2000", "status": "unavailable", "reason": "xvmec2000 executable not found"}
    workdir = outdir / "vmec2000_work"
    updates = {
        "NSTEP": "1",
        "NITER": str(int(args.iters)),
        "NITER_ARRAY": str(int(args.iters)),
        "FTOL": "1e-30",
        "FTOL_ARRAY": "1e-30",
    }
    try:
        result = run_xvmec2000(
            input_path=input_path,
            exec_path=exec_path,
            workdir=workdir,
            timeout_s=float(args.timeout_s),
            indata_updates=updates,
            keep_workdir=True,
        )
    except subprocess.TimeoutExpired:
        return {
            "kind": "vmec2000",
            "status": "timeout",
            "exec_path": str(exec_path),
            "timeout_s": float(args.timeout_s),
        }
    rows = flatten_threed1(result.stages)
    fsq_total = threed1_fsq_total(rows)
    return {
        "kind": "vmec2000",
        "status": "success",
        "exec_path": str(exec_path),
        "runtime_s": float(result.runtime_s),
        "row_count": int(fsq_total.size),
        "final_fsq_total": None if fsq_total.size == 0 else float(fsq_total[-1]),
        "workdir": str(workdir),
    }


def _run_vmecpp(
    *,
    args: argparse.Namespace,
    input_path: Path,
    outdir: Path,
) -> dict[str, Any]:
    exec_path = Path(args.vmecpp_exec).expanduser() if args.vmecpp_exec else None
    if exec_path is None:
        discovered = shutil.which("vmecpp")
        if discovered is None:
            return {"kind": "vmecpp", "status": "unavailable", "reason": "vmecpp executable not found"}
        exec_path = Path(discovered)
    workdir = outdir / "vmecpp_work"
    workdir.mkdir(parents=True, exist_ok=True)
    cmd = [str(exec_path), "-q", str(input_path)]
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=workdir,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=float(args.timeout_s),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "kind": "vmecpp",
            "status": "timeout",
            "exec_path": str(exec_path),
            "timeout_s": float(args.timeout_s),
            "stdout_tail": (exc.stdout if isinstance(exc.stdout, str) else "")[-4000:],
        }
    runtime_s = time.perf_counter() - started
    wouts = sorted(workdir.glob("wout_*.nc"))
    status = "success" if proc.returncode == 0 and wouts else "failed"
    return {
        "kind": "vmecpp",
        "status": status,
        "returncode": int(proc.returncode),
        "exec_path": str(exec_path),
        "runtime_s": float(runtime_s),
        "workdir": str(workdir),
        "wout_count": len(wouts),
        "stdout_tail": proc.stdout[-4000:],
    }


def _profile_value(run: dict[str, Any], key: str) -> Any:
    profile = run.get("profile")
    if not isinstance(profile, dict):
        return None
    phase = profile.get("phase_timing")
    if isinstance(phase, dict) and key in phase:
        return phase[key]
    diag = profile.get("diagnostics")
    if isinstance(diag, dict):
        timing = diag.get("timing")
        if isinstance(timing, dict) and key in timing:
            return timing[key]
    if key in profile:
        return profile[key]
    return None


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _vmec_jax_phase_breakdown(run: dict[str, Any]) -> dict[str, float | None]:
    """Return comparable host/solver timing buckets from a profiler run."""
    profile = run.get("profile")
    if not isinstance(profile, dict):
        return {}
    phase = profile.get("phase_timing")
    if not isinstance(phase, dict):
        phase = {}
    diagnostics = profile.get("diagnostics")
    timing = diagnostics.get("timing") if isinstance(diagnostics, dict) else None
    if not isinstance(timing, dict):
        timing = {}

    child_elapsed_s = _float_or_none(run.get("elapsed_wall_s"))
    run_wall_s = _float_or_none(phase.get("run_wall_s"))
    solve_total_s = _float_or_none(timing.get("solve_total_s"))
    setup_total_s = _float_or_none(timing.get("setup_total_s"))
    iteration_loop_s = _float_or_none(timing.get("iteration_loop_s"))
    compute_forces_s = _float_or_none(timing.get("compute_forces_s"))
    preconditioner_s = _float_or_none(timing.get("preconditioner_s"))
    precond_refresh_s = _float_or_none(timing.get("precond_refresh_s"))
    precond_refresh_seed_s = _float_or_none(timing.get("precond_refresh_seed_s"))
    precond_refresh_seed_lambda_s = _float_or_none(timing.get("precond_refresh_seed_lambda_s"))
    precond_refresh_seed_rz_matrices_s = _float_or_none(timing.get("precond_refresh_seed_rz_matrices_s"))
    precond_apply_s = _float_or_none(timing.get("precond_apply_s"))
    update_s = _float_or_none(timing.get("update_s"))

    run_minus_solve_s = None
    if run_wall_s is not None and solve_total_s is not None:
        run_minus_solve_s = max(0.0, run_wall_s - solve_total_s)
    child_minus_run_s = None
    if child_elapsed_s is not None and run_wall_s is not None:
        child_minus_run_s = max(0.0, child_elapsed_s - run_wall_s)
    warmup_wall_s = _float_or_none(phase.get("warmup_wall_s"))
    child_minus_run_and_warmup_s = None
    if child_elapsed_s is not None and run_wall_s is not None and warmup_wall_s is not None:
        child_minus_run_and_warmup_s = max(0.0, child_elapsed_s - run_wall_s - warmup_wall_s)

    return {
        "child_elapsed_s": child_elapsed_s,
        "process_to_main_s": _float_or_none(phase.get("process_to_main_s")),
        "vmec_jax_import_s": _float_or_none(phase.get("vmec_jax_import_s")),
        "jax_devices_pre_run_s": _float_or_none(phase.get("jax_devices_pre_run_s")),
        "warmup_wall_s": warmup_wall_s,
        "profiled_run_wall_s": run_wall_s,
        "solver_setup_total_s": setup_total_s,
        "solver_setup_static_grid_rebuild_s": _float_or_none(timing.get("setup_static_grid_rebuild_s")),
        "solver_setup_boundary_profiles_s": _float_or_none(timing.get("setup_boundary_profiles_s")),
        "solver_setup_axis_reset_s": _float_or_none(timing.get("setup_axis_reset_s")),
        "solver_setup_cache_key_hash_s": _float_or_none(timing.get("setup_cache_key_hash_s")),
        "solver_setup_ptau_constants_s": _float_or_none(timing.get("setup_ptau_constants_s")),
        "solver_setup_index_constants_s": _float_or_none(timing.get("setup_index_constants_s")),
        "solver_iteration_loop_s": iteration_loop_s,
        "solver_solve_total_s": solve_total_s,
        "solver_compute_forces_s": compute_forces_s,
        "solver_preconditioner_s": preconditioner_s,
        "solver_precond_refresh_s": precond_refresh_s,
        "solver_precond_refresh_seed_s": precond_refresh_seed_s,
        "solver_precond_refresh_seed_lambda_s": precond_refresh_seed_lambda_s,
        "solver_precond_refresh_seed_rz_matrices_s": precond_refresh_seed_rz_matrices_s,
        "solver_precond_apply_s": precond_apply_s,
        "solver_update_s": update_s,
        "profiled_run_minus_solver_s": run_minus_solve_s,
        "child_elapsed_minus_profiled_run_s": child_minus_run_s,
        "child_elapsed_minus_profiled_run_and_warmup_s": child_minus_run_and_warmup_s,
        "process_peak_rss_mib": _float_or_none(profile.get("process_peak_rss_mib")),
    }


def _build_analysis(runs: dict[str, Any]) -> dict[str, Any]:
    analysis: dict[str, Any] = {
        "vmec_jax_phase_breakdown": {},
        "runtime_ratios": {},
    }
    for label in ("vmec_jax_cold", "vmec_jax_warm"):
        run = runs.get(label)
        if isinstance(run, dict):
            analysis["vmec_jax_phase_breakdown"][label] = _vmec_jax_phase_breakdown(run)

    cold = analysis["vmec_jax_phase_breakdown"].get("vmec_jax_cold", {})
    warm = analysis["vmec_jax_phase_breakdown"].get("vmec_jax_warm", {})
    cold_run = _float_or_none(cold.get("profiled_run_wall_s"))
    warm_run = _float_or_none(warm.get("profiled_run_wall_s"))
    if cold_run is not None and warm_run not in (None, 0.0):
        analysis["runtime_ratios"]["vmec_jax_cold_to_warm_profiled_run"] = cold_run / warm_run
    vmec2000 = runs.get("vmec2000", {})
    vmec2000_runtime = _float_or_none(vmec2000.get("runtime_s")) if isinstance(vmec2000, dict) else None
    if vmec2000_runtime not in (None, 0.0):
        if cold_run is not None:
            analysis["runtime_ratios"]["vmec_jax_cold_to_vmec2000"] = cold_run / vmec2000_runtime
        if warm_run is not None:
            analysis["runtime_ratios"]["vmec_jax_warm_to_vmec2000"] = warm_run / vmec2000_runtime
    vmecpp = runs.get("vmecpp", {})
    vmecpp_runtime = _float_or_none(vmecpp.get("runtime_s")) if isinstance(vmecpp, dict) else None
    if vmecpp_runtime not in (None, 0.0):
        if cold_run is not None:
            analysis["runtime_ratios"]["vmec_jax_cold_to_vmecpp"] = cold_run / vmecpp_runtime
        if warm_run is not None:
            analysis["runtime_ratios"]["vmec_jax_warm_to_vmecpp"] = warm_run / vmecpp_runtime
    return analysis


def _format_float(value: Any, digits: int = 4) -> str:
    number = _float_or_none(value)
    if number is None:
        return ""
    return f"{number:.{digits}g}"


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    runs = report.get("runs", {})
    analysis = report.get("analysis", {})
    lines = [
        "# Fixed-Boundary Performance Decomposition",
        "",
        f"Input: `{report['input']}`",
        f"Iterations: `{report['iters']}`",
        "",
        "## Run Summary",
        "",
        "| Run | Status | Wall s | Peak RSS MiB | Final FSQ |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label in ("vmec_jax_cold", "vmec_jax_warm"):
        run = runs.get(label, {})
        profile = run.get("profile") if isinstance(run, dict) else None
        result = profile.get("result", {}) if isinstance(profile, dict) else {}
        lines.append(
            "| "
            + f"{label} | {run.get('returncode')} | "
            + f"{_profile_value(run, 'run_wall_s') or run.get('elapsed_wall_s')} | "
            + f"{profile.get('process_peak_rss_mib') if isinstance(profile, dict) else ''} | "
            + f"{result.get('final_fsqr')} / {result.get('final_fsqz')} / {result.get('final_fsql')} |"
        )
    vmec2000 = runs.get("vmec2000", {})
    lines.append(
        "| vmec2000 | "
        + f"{vmec2000.get('status')} | {vmec2000.get('runtime_s', '')} |  | "
        + f"{vmec2000.get('final_fsq_total', '')} |"
    )
    vmecpp = runs.get("vmecpp", {})
    lines.append(
        "| vmecpp | "
        + f"{vmecpp.get('status')} | {vmecpp.get('runtime_s', '')} |  | "
        + f"wouts={vmecpp.get('wout_count', '')} |"
    )
    phase_breakdown = analysis.get("vmec_jax_phase_breakdown", {}) if isinstance(analysis, dict) else {}
    if phase_breakdown:
        lines += [
            "",
            "## vmec_jax Phase Decomposition",
            "",
            "| Phase | Cold | Warm |",
            "| --- | ---: | ---: |",
        ]
        phase_labels = [
            ("child_elapsed_s", "Child process elapsed"),
            ("process_to_main_s", "Process start to profiler main"),
            ("vmec_jax_import_s", "vmec_jax import"),
            ("jax_devices_pre_run_s", "JAX device discovery before run"),
            ("warmup_wall_s", "Warmup run wall"),
            ("profiled_run_wall_s", "Profiled run wall"),
            ("solver_solve_total_s", "Solver total"),
            ("solver_setup_total_s", "Solver setup"),
            ("solver_setup_static_grid_rebuild_s", "Setup: static-grid rebuild"),
            ("solver_setup_boundary_profiles_s", "Setup: boundary/profiles/trig"),
            ("solver_setup_axis_reset_s", "Setup: axis reset"),
            ("solver_setup_cache_key_hash_s", "Setup: cache-key hashing"),
            ("solver_setup_ptau_constants_s", "Setup: p/tau constants"),
            ("solver_setup_index_constants_s", "Setup: index/mode constants"),
            ("solver_iteration_loop_s", "Solver iteration loop"),
            ("solver_compute_forces_s", "Force evaluation"),
            ("solver_preconditioner_s", "Preconditioner"),
            ("solver_precond_refresh_s", "Preconditioner refresh"),
            ("solver_precond_refresh_seed_s", "Preconditioner seed"),
            ("solver_precond_refresh_seed_lambda_s", "Preconditioner seed: lambda"),
            ("solver_precond_refresh_seed_rz_matrices_s", "Preconditioner seed: R/Z matrices"),
            ("solver_precond_apply_s", "Preconditioner apply"),
            ("solver_update_s", "State update"),
            ("profiled_run_minus_solver_s", "Profiled run minus solver"),
            ("child_elapsed_minus_profiled_run_s", "Child elapsed minus profiled run"),
            ("child_elapsed_minus_profiled_run_and_warmup_s", "Child elapsed minus profiled run and warmup"),
            ("process_peak_rss_mib", "Process peak RSS MiB"),
        ]
        cold = phase_breakdown.get("vmec_jax_cold", {})
        warm = phase_breakdown.get("vmec_jax_warm", {})
        for key, label in phase_labels:
            lines.append(
                "| "
                + f"{label} | {_format_float(cold.get(key))} | {_format_float(warm.get(key))} |"
            )
    ratios = analysis.get("runtime_ratios", {}) if isinstance(analysis, dict) else {}
    if ratios:
        lines += [
            "",
            "## Runtime Ratios",
            "",
            "| Ratio | Value |",
            "| --- | ---: |",
        ]
        for key, value in sorted(ratios.items()):
            lines.append(f"| `{key}` | {_format_float(value)} |")
    lines += [
        "",
        "## Algorithm Map",
        "",
        "| Bucket | VMEC2000 sources | VMEC++ sources | vmec_jax sources | Profiler keys | Refactor target |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["algorithm_map"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["bucket"],
                    "<br>".join(f"`{x}`" for x in row["vmec2000_sources"]),
                    "<br>".join(f"`{x}`" for x in row["vmecpp_sources"]),
                    "<br>".join(f"`{x}`" for x in row["vmec_jax_sources"]),
                    "<br>".join(f"`{x}`" for x in row["profiler_keys"]),
                    row["refactor_target"],
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## Immediate Interpretation",
        "",
        "- Cold vmec_jax includes Python import, JAX backend discovery, JAX tracing, and XLA compilation.",
        "- Warm vmec_jax reports warmup separately and uses the post-warmup timed run as the optimization-loop baseline.",
        "- VMEC2000 remains the executable latency baseline; parity is still required before any performance claim.",
        "- This report is a diagnostic artifact, not a release benchmark.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = _parse_args()
    input_path = Path(args.input).expanduser()
    if not input_path.is_absolute():
        input_path = (_REPO_ROOT / input_path).resolve()
    outdir = Path(args.outdir).expanduser()
    if not outdir.is_absolute():
        outdir = (_REPO_ROOT / outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    runs: dict[str, Any] = {}
    if not args.skip_runs:
        runs["vmec_jax_cold"] = _run_child_profile(
            args=args,
            input_path=input_path,
            outdir=outdir,
            label="cold",
            no_warmup=True,
        )
        runs["vmec_jax_warm"] = _run_child_profile(
            args=args,
            input_path=input_path,
            outdir=outdir,
            label="warm",
            no_warmup=False,
        )
        if not args.skip_vmec2000:
            runs["vmec2000"] = _run_vmec2000(args=args, input_path=input_path, outdir=outdir)
        if not args.skip_vmecpp:
            runs["vmecpp"] = _run_vmecpp(args=args, input_path=input_path, outdir=outdir)

    report = {
        "input": str(input_path),
        "iters": int(args.iters),
        "algorithm_map": ALGORITHM_MAP,
        "runs": _json_safe(runs),
    }
    report["analysis"] = _json_safe(_build_analysis(runs))
    json_out = outdir / "performance_decomposition.json"
    md_out = outdir / "performance_decomposition.md"
    json_out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(report, md_out)
    print(f"Wrote {json_out}")
    print(f"Wrote {md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
