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
    parser.add_argument("--skip-runs", action="store_true", help="Only write the static algorithm map.")
    parser.add_argument("--skip-vmec2000", action="store_true", help="Do not run VMEC2000.")
    parser.add_argument("--vmec2000-exec", default=None, help="Optional xvmec2000 executable.")
    parser.add_argument("--timeout-s", type=float, default=120.0, help="VMEC2000 timeout.")
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


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    runs = report.get("runs", {})
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
        "- Warm vmec_jax excludes the same-process warmup and is the relevant optimization-loop baseline.",
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

    report = {
        "input": str(input_path),
        "iters": int(args.iters),
        "algorithm_map": ALGORITHM_MAP,
        "runs": _json_safe(runs),
    }
    json_out = outdir / "performance_decomposition.json"
    md_out = outdir / "performance_decomposition.md"
    json_out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(report, md_out)
    print(f"Wrote {json_out}")
    print(f"Wrote {md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
