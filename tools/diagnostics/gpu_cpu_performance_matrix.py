#!/usr/bin/env python3
"""Run vmec_jax CPU/GPU profiling jobs through one reproducible wrapper.

The default backend is ``auto``: the child process inherits the caller's JAX
backend environment.  Pass ``--backend cpu`` or ``--backend gpu`` only when the
comparison intentionally needs an explicit device process.
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
PROFILE_FIXED_BOUNDARY = REPO_ROOT / "tools" / "diagnostics" / "profile_fixed_boundary.py"
PROFILE_EXACT_OPTIMIZER = REPO_ROOT / "tools" / "diagnostics" / "profile_exact_optimizer.py"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Launch fixed-boundary or exact-callback profilers across selected "
            "JAX backends and write a compact comparison JSON."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("fixed-boundary", "exact-callback"),
        default="fixed-boundary",
        help="Profiler family to run.",
    )
    parser.add_argument(
        "--backend",
        action="append",
        choices=("auto", "cpu", "gpu"),
        default=None,
        help=(
            "Backend process to launch. Repeat for a matrix. The default is "
            "'auto', which preserves the caller's JAX backend selection."
        ),
    )
    parser.add_argument("--outdir", type=Path, default=Path("outputs/performance_profiles"))
    parser.add_argument("--json-out", type=Path, default=None, help="Matrix summary JSON path.")
    parser.add_argument("--python", default=sys.executable, help="Python executable for child profilers.")
    parser.add_argument("--dry-run", action="store_true", help="Print and record child commands without running them.")
    parser.add_argument("--keep-going", action="store_true", help="Continue launching later backends after a failure.")
    parser.add_argument(
        "--x64",
        choices=("enable", "disable", "inherit"),
        default="enable",
        help="How to set JAX_ENABLE_X64 in child profiler processes.",
    )
    parser.add_argument(
        "--replay-column-chunk",
        default=None,
        help="Optional VMEC_JAX_REPLAY_COLUMN_CHUNK override for tape replay profiling.",
    )
    parser.add_argument(
        "--dynamic-replay-bucket",
        default=None,
        help="Optional VMEC_JAX_DYNAMIC_REPLAY_BUCKET override for tape replay profiling.",
    )
    parser.add_argument(
        "--dynamic-replay-mode",
        choices=("basepoint", "whole_scan", "scan"),
        default=None,
        help="Optional VMEC_JAX_DYNAMIC_REPLAY_MODE override for exact tape replay profiling.",
    )
    parser.add_argument("--trace", action="store_true", help="Collect TensorBoard/XProf traces where supported.")
    parser.add_argument(
        "--device-memory-profile",
        action="store_true",
        help="Save JAX device memory profiles for exact-callback runs.",
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Append one raw argument to the child profiler. Repeat for values.",
    )

    fixed = parser.add_argument_group("fixed-boundary mode")
    fixed.add_argument("--input", type=Path, default=Path("examples/data/input.nfp4_QH_warm_start"))
    fixed.add_argument("--iters", type=int, default=20)
    fixed.add_argument(
        "--solver-mode",
        choices=("auto", "default", "parity", "accelerated"),
        default="accelerated",
    )
    fixed.add_argument("--no-warmup", action="store_true", help="Skip the fixed-boundary warmup run.")
    fixed.set_defaults(use_scan=True)
    fixed.add_argument("--use-scan", dest="use_scan", action="store_true", help="Use the scan iteration path.")
    fixed.add_argument("--no-use-scan", dest="use_scan", action="store_false", help="Do not pass --use-scan.")
    fixed.set_defaults(raw_solver_policy=True)
    fixed.add_argument(
        "--raw-solver-policy",
        dest="raw_solver_policy",
        action="store_true",
        help="Disable public CLI finish policy for raw throughput timing.",
    )
    fixed.add_argument(
        "--public-cli-policy",
        dest="raw_solver_policy",
        action="store_false",
        help="Measure the public run_fixed_boundary CLI/API policy.",
    )
    fixed.set_defaults(single_grid=True)
    fixed.add_argument("--single-grid", dest="single_grid", action="store_true", help="Pass --no-multigrid.")
    fixed.add_argument("--allow-multigrid", dest="single_grid", action="store_false")

    exact = parser.add_argument_group("exact-callback mode")
    exact.add_argument("--problem", choices=("qa", "qh"), default="qh")
    exact.add_argument("--max-mode", type=int, default=2)
    exact.add_argument(
        "--callback",
        choices=("trial", "exact", "accepted", "jacobian", "gradient", "linear", "run"),
        default="jacobian",
    )
    exact.add_argument("--repeats", type=int, default=2)
    exact.add_argument("--perturb-scale", type=float, default=1.0e-4)
    exact.add_argument("--inner-max-iter", type=int, default=80)
    exact.add_argument("--trial-max-iter", type=int, default=40)
    exact.add_argument("--inner-ftol", type=float, default=0.0)
    exact.add_argument("--trial-ftol", type=float, default=1.0e-10)
    exact.add_argument(
        "--method",
        choices=("scipy", "scipy_matrix_free", "gauss_newton", "lbfgs_adjoint", "scalar_trust"),
        default="scipy",
        help="Optimizer method for exact-callback --callback run profiling.",
    )
    exact.add_argument(
        "--scipy-tr-solver",
        choices=("lsmr", "exact", "none"),
        default="lsmr",
        help="SciPy trust-region linear solver for method=scipy.",
    )
    exact.add_argument("--lsmr-maxiter", type=int, default=0, help="Optional scipy LSMR iteration cap.")
    exact.add_argument(
        "--trial-use-scan",
        action="store_true",
        help="Forward --trial-use-scan to profile_exact_optimizer for relaxed trial solves.",
    )
    exact.add_argument("--vmec-timing", action="store_true", help="Enable VMEC_JAX_TIMING in exact profiler.")
    return parser


def _repo_path(path: Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _prepend_pythonpath(env: dict[str, str]) -> None:
    current = env.get("PYTHONPATH", "")
    parts = [str(REPO_ROOT)]
    if current:
        parts.append(current)
    env["PYTHONPATH"] = os.pathsep.join(parts)


def child_env(
    *,
    backend: str,
    args: argparse.Namespace,
    base_env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Return a child environment while preserving backend selection by default."""
    env = dict(os.environ if base_env is None else base_env)
    _prepend_pythonpath(env)

    if backend == "cpu":
        env["JAX_PLATFORMS"] = "cpu"
        env.pop("JAX_PLATFORM_NAME", None)
    elif backend == "gpu":
        env["JAX_PLATFORM_NAME"] = "gpu"
        env.pop("JAX_PLATFORMS", None)
    elif backend != "auto":  # pragma: no cover - guarded by argparse
        raise ValueError(f"unknown backend {backend!r}")

    if args.x64 == "enable":
        env["JAX_ENABLE_X64"] = "1"
    elif args.x64 == "disable":
        env["JAX_ENABLE_X64"] = "0"

    if args.replay_column_chunk is not None:
        env["VMEC_JAX_REPLAY_COLUMN_CHUNK"] = str(args.replay_column_chunk)
    if args.dynamic_replay_bucket is not None:
        env["VMEC_JAX_DYNAMIC_REPLAY_BUCKET"] = str(args.dynamic_replay_bucket)
    if args.dynamic_replay_mode is not None:
        env["VMEC_JAX_DYNAMIC_REPLAY_MODE"] = str(args.dynamic_replay_mode)
    return env


def env_summary(env: dict[str, str]) -> dict[str, str | None]:
    keys = (
        "JAX_PLATFORM_NAME",
        "JAX_PLATFORMS",
        "JAX_ENABLE_X64",
        "VMEC_JAX_REPLAY_COLUMN_CHUNK",
        "VMEC_JAX_DYNAMIC_REPLAY_BUCKET",
        "VMEC_JAX_DYNAMIC_REPLAY_MODE",
    )
    return {key: env.get(key) for key in keys if env.get(key) is not None}


def solver_device_for_backend(backend: str) -> str:
    return backend if backend in {"cpu", "gpu"} else "auto"


def report_stem(args: argparse.Namespace, backend: str) -> str:
    if args.mode == "fixed-boundary":
        return f"fixed_boundary_{backend}_iters{int(args.iters)}"
    return f"exact_{args.problem}_m{int(args.max_mode)}_{args.callback}_{args.method}_{backend}"


def build_child_command(
    *,
    args: argparse.Namespace,
    backend: str,
    report_path: Path,
    trace_dir: Path,
    memory_profile_path: Path | None = None,
) -> list[str]:
    solver_device = solver_device_for_backend(backend)
    if args.mode == "fixed-boundary":
        command = [
            str(args.python),
            str(PROFILE_FIXED_BOUNDARY),
            "--input",
            str(_repo_path(args.input)),
            "--iters",
            str(int(args.iters)),
            "--outdir",
            str(trace_dir),
            "--solver-mode",
            str(args.solver_mode),
            "--solver-device",
            solver_device,
            "--json-out",
            str(report_path),
        ]
        if not args.trace:
            command.append("--simple-profile")
        if args.no_warmup:
            command.append("--no-warmup")
        if args.use_scan:
            command.append("--use-scan")
        if args.raw_solver_policy:
            command.append("--no-auto-cli-policy")
        if args.single_grid:
            command.append("--no-multigrid")
        command.extend(str(item) for item in args.extra_arg)
        return command

    command = [
        str(args.python),
        str(PROFILE_EXACT_OPTIMIZER),
        "--problem",
        str(args.problem),
        "--max-mode",
        str(int(args.max_mode)),
        "--callback",
        str(args.callback),
        "--repeats",
        str(int(args.repeats)),
        "--perturb-scale",
        f"{float(args.perturb_scale):.17g}",
        "--inner-max-iter",
        str(int(args.inner_max_iter)),
        "--trial-max-iter",
        str(int(args.trial_max_iter)),
        "--inner-ftol",
        f"{float(args.inner_ftol):.17g}",
        "--trial-ftol",
        f"{float(args.trial_ftol):.17g}",
        "--method",
        str(args.method),
        "--scipy-tr-solver",
        str(args.scipy_tr_solver),
        "--solver-device",
        solver_device,
        "--json-out",
        str(report_path),
    ]
    if int(args.lsmr_maxiter) > 0:
        command.extend(["--lsmr-maxiter", str(int(args.lsmr_maxiter))])
    if args.trial_use_scan:
        command.append("--trial-use-scan")
    if args.vmec_timing:
        command.append("--vmec-timing")
    if args.trace:
        command.extend(["--trace-outdir", str(trace_dir)])
    if memory_profile_path is not None:
        command.extend(["--device-memory-profile-out", str(memory_profile_path)])
    command.extend(str(item) for item in args.extra_arg)
    return command


def _load_profile_summary(report_path: Path, *, label: str) -> dict[str, Any] | None:
    if not report_path.exists():
        return None
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    try:
        from compare_profile_reports import summarize_payload

        return summarize_payload(payload, path=report_path, label=label, top_profile=6)
    except Exception:
        return {"label": label, "path": str(report_path), "payload_keys": sorted(payload)}


def run_backend(args: argparse.Namespace, backend: str, outdir: Path) -> dict[str, Any]:
    stem = report_stem(args, backend)
    report_path = outdir / f"{stem}.json"
    trace_dir = outdir / f"{stem}_trace"
    stdout_path = outdir / f"{stem}.stdout.log"
    stderr_path = outdir / f"{stem}.stderr.log"
    memory_profile_path = outdir / f"{stem}.device_memory.prof" if args.device_memory_profile else None
    env = child_env(backend=backend, args=args)
    command = build_child_command(
        args=args,
        backend=backend,
        report_path=report_path,
        trace_dir=trace_dir,
        memory_profile_path=memory_profile_path,
    )
    entry: dict[str, Any] = {
        "backend": backend,
        "command": command,
        "env": env_summary(env),
        "report_path": str(report_path),
        "trace_dir": str(trace_dir) if args.trace else None,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "device_memory_profile_path": None if memory_profile_path is None else str(memory_profile_path),
        "dry_run": bool(args.dry_run),
    }
    if args.dry_run:
        entry["exit_code"] = None
        return entry

    report_path.parent.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(command, cwd=str(REPO_ROOT), env=env, stdout=stdout, stderr=stderr, check=False)
    entry["wall_time_s"] = float(time.perf_counter() - t0)
    entry["exit_code"] = int(completed.returncode)
    entry["summary"] = _load_profile_summary(report_path, label=backend)
    return entry


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _metric(summary: dict[str, Any] | None, key: str) -> Any:
    if not isinstance(summary, dict):
        return None
    metrics = summary.get("metrics")
    if not isinstance(metrics, dict):
        return None
    return metrics.get(key)


def print_report(payload: dict[str, Any]) -> None:
    rows = []
    for run in payload["runs"]:
        summary = run.get("summary")
        rows.append(
            (
                run["backend"],
                "dry-run" if run.get("dry_run") else str(run.get("exit_code")),
                run.get("wall_time_s"),
                _metric(summary, "total_runtime_s"),
                _metric(summary, "replay_time_s"),
                run["report_path"],
            )
        )
    headers = ("backend", "exit", "wrapper_s", "profile_s", "replay_s", "report")
    widths = [
        max(len(headers[col]), *(len(_format_cell(row[col])) for row in rows)) if rows else len(headers[col])
        for col in range(len(headers))
    ]
    print("  ".join(headers[col].ljust(widths[col]) for col in range(len(headers))))
    print("  ".join("-" * widths[col] for col in range(len(headers))))
    for row in rows:
        print("  ".join(_format_cell(row[col]).ljust(widths[col]) for col in range(len(headers))))


def _format_cell(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    backends = args.backend or ["auto"]
    outdir = args.outdir.expanduser()
    if not outdir.is_absolute():
        outdir = REPO_ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    for backend in backends:
        entry = run_backend(args, backend, outdir)
        runs.append(entry)
        if entry.get("exit_code") not in (None, 0) and not args.keep_going:
            break

    payload = {
        "schema_version": 1,
        "report_kind": "gpu_cpu_performance_matrix",
        "mode": args.mode,
        "backends": backends,
        "repo_root": str(REPO_ROOT),
        "runs": runs,
    }
    json_out = args.json_out or outdir / "gpu_cpu_performance_matrix.json"
    write_json(json_out, payload)
    print_report(payload)
    print(f"matrix JSON written to {json_out if Path(json_out).is_absolute() else REPO_ROOT / json_out}")
    if any(run.get("exit_code") not in (None, 0) for run in runs):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
