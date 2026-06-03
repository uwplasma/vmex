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
PROFILE_QI_BOOZER = REPO_ROOT / "tools" / "diagnostics" / "profile_qi_boozer_gpu.py"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Launch fixed-boundary or exact-callback profilers across selected "
            "JAX backends and write a compact comparison JSON."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("fixed-boundary", "exact-callback", "qi-boozer"),
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
        "--vmec-timing",
        action="store_true",
        help="Enable VMEC_JAX_TIMING in child profilers that support solver phase timings.",
    )
    parser.add_argument(
        "--vmec-timing-detail",
        action="store_true",
        help=(
            "Enable detailed VMEC_JAX_TIMING_DETAIL preconditioner subphase timings in child profilers. "
            "This adds extra synchronization and is for diagnostics only."
        ),
    )
    parser.add_argument(
        "--sync-replay-timing",
        action="store_true",
        help=(
            "Forward exact-callback replay/tangent synchronization timing to child profilers. "
            "This splits dispatch and device-ready buckets and is for diagnostics only."
        ),
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Append one raw argument to the child profiler. Repeat for values.",
    )

    fixed = parser.add_argument_group("fixed-boundary mode")
    fixed.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input file. Defaults to QH warm-start for fixed-boundary and nfp2_QI for qi-boozer.",
    )
    fixed.add_argument("--iters", type=int, default=20)
    fixed.add_argument(
        "--use-input-niter",
        action="store_true",
        help="Use NITER/NITER_ARRAY from the input deck instead of the --iters budget.",
    )
    fixed.add_argument(
        "--solver-mode",
        choices=("auto", "default", "parity", "accelerated"),
        default="accelerated",
    )
    fixed.add_argument("--no-warmup", action="store_true", help="Skip the fixed-boundary warmup run.")
    fixed.set_defaults(use_scan=None)
    fixed.add_argument("--use-scan", dest="use_scan", action="store_true", help="Force the scan iteration path.")
    fixed.add_argument(
        "--no-use-scan",
        dest="use_scan",
        action="store_false",
        help="Force the non-scan fixed-boundary path.",
    )
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
    exact.add_argument("--problem", choices=("qa", "qh", "qp"), default="qh")
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
        help="Legacy alias for --trial-scan=on.",
    )
    exact.add_argument(
        "--trial-scan",
        choices=("auto", "on", "off"),
        default="auto",
        help="Forward trial residual solve policy to profile_exact_optimizer.",
    )
    exact.set_defaults(jvp_only_exact_tape=None)
    exact.add_argument(
        "--jvp-only-exact-tape",
        dest="jvp_only_exact_tape",
        action="store_true",
        help=(
            "Forward VMEC_JAX_OPT_JVP_ONLY_EXACT_TAPE=1 to exact-callback "
            "profilers for lean-tape before/after comparisons."
        ),
    )
    exact.add_argument(
        "--no-jvp-only-exact-tape",
        dest="jvp_only_exact_tape",
        action="store_false",
        help="Forward VMEC_JAX_OPT_JVP_ONLY_EXACT_TAPE=0 to exact-callback profilers.",
    )
    exact.set_defaults(jvp_only_basepoint_carries=None)
    exact.add_argument(
        "--jvp-only-basepoint-carries",
        dest="jvp_only_basepoint_carries",
        action="store_true",
        help=(
            "Forward VMEC_JAX_JVP_ONLY_EXACT_TAPE_BASEPOINT_CARRIES=1 to "
            "exact-callback profilers. Use with --jvp-only-exact-tape for GPU "
            "replay diagnostics."
        ),
    )
    exact.add_argument(
        "--no-jvp-only-basepoint-carries",
        dest="jvp_only_basepoint_carries",
        action="store_false",
        help="Forward VMEC_JAX_JVP_ONLY_EXACT_TAPE_BASEPOINT_CARRIES=0 to exact-callback profilers.",
    )
    qi = parser.add_argument_group("qi-boozer mode")
    qi.add_argument("--repeat", type=int, default=2, help="QI residual evaluations after the VMEC solve.")
    qi.add_argument("--mpol", type=int, default=6)
    qi.add_argument("--ntor", type=int, default=6)
    qi.add_argument("--mboz", type=int, default=10)
    qi.add_argument("--nboz", type=int, default=10)
    qi.add_argument("--nphi", type=int, default=61)
    qi.add_argument("--nalpha", type=int, default=13)
    qi.add_argument("--n-bounce", type=int, default=21)
    qi.add_argument("--surfaces", default="0.1,0.25,0.5,0.75,1.0")
    qi.add_argument("--jit-booz", action="store_true", help="Use the jitted Boozer path in QI profiling.")
    return parser


def _repo_path(path: Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _input_for_mode(args: argparse.Namespace) -> Path:
    if args.input is not None:
        return Path(args.input)
    if args.mode == "qi-boozer":
        return Path("examples/data/input.nfp2_QI")
    return Path("examples/data/input.nfp4_QH_warm_start")


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
    if getattr(args, "sync_replay_timing", False):
        env["VMEC_JAX_OPT_SYNC_REPLAY_TIMING"] = "1"
    trial_scan = "on" if getattr(args, "trial_use_scan", False) else str(getattr(args, "trial_scan", "auto"))
    if trial_scan == "on":
        env["VMEC_JAX_OPT_TRIAL_SCAN"] = "1"
    elif trial_scan == "off":
        env["VMEC_JAX_OPT_TRIAL_SCAN"] = "0"
    if getattr(args, "jvp_only_exact_tape", None) is not None:
        env["VMEC_JAX_OPT_JVP_ONLY_EXACT_TAPE"] = (
            "1" if bool(args.jvp_only_exact_tape) else "0"
        )
    if getattr(args, "jvp_only_basepoint_carries", None) is not None:
        env["VMEC_JAX_JVP_ONLY_EXACT_TAPE_BASEPOINT_CARRIES"] = (
            "1" if bool(args.jvp_only_basepoint_carries) else "0"
        )
    return env


def env_summary(env: dict[str, str]) -> dict[str, str | None]:
    keys = (
        "JAX_PLATFORM_NAME",
        "JAX_PLATFORMS",
        "JAX_ENABLE_X64",
        "VMEC_JAX_REPLAY_COLUMN_CHUNK",
        "VMEC_JAX_DYNAMIC_REPLAY_BUCKET",
        "VMEC_JAX_DYNAMIC_REPLAY_MODE",
        "VMEC_JAX_OPT_SYNC_REPLAY_TIMING",
        "VMEC_JAX_OPT_TRIAL_SCAN",
        "VMEC_JAX_OPT_JVP_ONLY_EXACT_TAPE",
        "VMEC_JAX_JVP_ONLY_EXACT_TAPE_BASEPOINT_CARRIES",
        "VMEC_JAX_TRIDI_PRECOMPUTE",
        "VMEC_JAX_TRIDI_SOLVE",
    )
    return {key: env.get(key) for key in keys if env.get(key) is not None}


def solver_device_for_backend(backend: str) -> str:
    return backend if backend in {"cpu", "gpu"} else "auto"


def report_stem(args: argparse.Namespace, backend: str) -> str:
    if args.mode == "fixed-boundary":
        if bool(getattr(args, "use_input_niter", False)):
            return f"fixed_boundary_{backend}_input_niter"
        return f"fixed_boundary_{backend}_iters{int(args.iters)}"
    if args.mode == "qi-boozer":
        jit = "jit" if bool(args.jit_booz) else "nojit"
        return f"qi_boozer_{backend}_{jit}_repeat{int(args.repeat)}"
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
            str(_repo_path(_input_for_mode(args))),
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
        if args.use_input_niter:
            command.append("--use-input-niter")
        if args.use_scan is True:
            command.append("--use-scan")
        elif args.use_scan is False:
            command.append("--no-use-scan")
        if args.raw_solver_policy:
            command.append("--no-auto-cli-policy")
        if args.single_grid:
            command.append("--no-multigrid")
        if args.vmec_timing:
            command.append("--vmec-timing")
        if args.vmec_timing_detail:
            command.append("--vmec-timing-detail")
        command.extend(str(item) for item in args.extra_arg)
        return command

    if args.mode == "qi-boozer":
        qi_solver_device = "none" if solver_device == "auto" else solver_device
        command = [
            str(args.python),
            str(PROFILE_QI_BOOZER),
            "--input",
            str(_repo_path(_input_for_mode(args))),
            "--output",
            str(report_path),
            "--solver-device",
            qi_solver_device,
            "--mpol",
            str(int(args.mpol)),
            "--ntor",
            str(int(args.ntor)),
            "--mboz",
            str(int(args.mboz)),
            "--nboz",
            str(int(args.nboz)),
            "--nphi",
            str(int(args.nphi)),
            "--nalpha",
            str(int(args.nalpha)),
            "--n-bounce",
            str(int(args.n_bounce)),
            "--surfaces",
            str(args.surfaces),
            "--repeat",
            str(int(args.repeat)),
        ]
        if args.jit_booz:
            command.append("--jit-booz")
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
    trial_scan = "on" if args.trial_use_scan else str(args.trial_scan)
    if trial_scan != "auto":
        command.extend(["--trial-scan", trial_scan])
    if args.jvp_only_exact_tape is True:
        command.append("--jvp-only-exact-tape")
    elif args.jvp_only_exact_tape is False:
        command.append("--no-jvp-only-exact-tape")
    if args.jvp_only_basepoint_carries is True:
        command.append("--jvp-only-basepoint-carries")
    elif args.jvp_only_basepoint_carries is False:
        command.append("--no-jvp-only-basepoint-carries")
    if args.vmec_timing:
        command.append("--vmec-timing")
    if args.vmec_timing_detail:
        command.append("--vmec-timing-detail")
    if args.sync_replay_timing:
        command.append("--sync-replay-timing")
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
        diagnostics_dir = str(Path(__file__).resolve().parent)
        if diagnostics_dir not in sys.path:
            sys.path.insert(0, diagnostics_dir)
        from compare_profile_reports import summarize_payload

        summary = summarize_payload(payload, path=report_path, label=label, top_profile=6)
        return _augment_profile_summary(summary, payload=payload)
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
    _attach_matrix_summary_sections(entry)
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


def _summary_section(summary: dict[str, Any] | None, key: str) -> dict[str, Any]:
    if not isinstance(summary, dict):
        return {}
    value = summary.get(key)
    return value if isinstance(value, dict) else {}


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _metadata_value(summary: dict[str, Any] | None, key: str) -> Any:
    return _summary_section(summary, "metadata").get(key)


def _run_env_value(run: dict[str, Any] | None, key: str) -> Any:
    if not isinstance(run, dict):
        return None
    env = run.get("env")
    if not isinstance(env, dict):
        return None
    return env.get(key)


def _metadata_or_env(
    run: dict[str, Any] | None,
    summary: dict[str, Any] | None,
    *,
    metadata_key: str,
    env_key: str,
) -> Any:
    return _first_present(_metadata_value(summary, metadata_key), _run_env_value(run, env_key))


def _contains_data(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_contains_data(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_data(child) for child in value)
    return value is not None


def _scan_status_metric(summary: dict[str, Any] | None, status: str, key: str) -> Any:
    trial_scan = _summary_section(summary, "trial_scan_summary")
    cache_status = trial_scan.get("cache_status")
    if not isinstance(cache_status, dict):
        return None
    status_metrics = cache_status.get(status)
    if not isinstance(status_metrics, dict):
        return None
    return status_metrics.get(key)


def _matrix_scan_cache_summary(summary: dict[str, Any] | None) -> dict[str, Any]:
    trial_scan = _summary_section(summary, "trial_scan_summary")
    return {
        "trial": {
            "total_s": _first_present(
                _metric(summary, "trial_solver_scan_total_s"),
                _metric(summary, "vmec_scan_total_s"),
                trial_scan.get("total_s"),
            ),
            "setup_s": _first_present(_metric(summary, "trial_solver_scan_setup_s"), trial_scan.get("setup_s")),
            "run_setup_s": _first_present(
                _metric(summary, "trial_solver_scan_run_setup_s"),
                trial_scan.get("run_setup_s"),
            ),
            "preflight_s": _first_present(
                _metric(summary, "trial_solver_scan_preflight_s"),
                trial_scan.get("preflight_s"),
            ),
            "device_run_s": _first_present(
                _first_present(
                    _metric(summary, "trial_solver_scan_device_run_s"),
                    _metric(summary, "vmec_scan_device_run_s"),
                ),
                trial_scan.get("device_run_s"),
            ),
            "device_dispatch_s": _first_present(
                _metric(summary, "trial_solver_scan_device_dispatch_s"),
                _metric(summary, "vmec_scan_device_dispatch_s"),
                trial_scan.get("device_dispatch_s"),
            ),
            "device_ready_s": _first_present(
                _metric(summary, "trial_solver_scan_device_ready_s"),
                _metric(summary, "vmec_scan_device_ready_s"),
                trial_scan.get("device_ready_s"),
            ),
            "host_materialize_s": _first_present(
                _metric(summary, "trial_solver_scan_host_materialize_s"),
                _metric(summary, "vmec_scan_host_materialize_s"),
                trial_scan.get("host_materialize_s"),
            ),
            "postprocess_s": _first_present(
                _metric(summary, "trial_solver_scan_postprocess_s"),
                trial_scan.get("postprocess_s"),
            ),
            "unattributed_s": _first_present(
                _metric(summary, "trial_solver_scan_unattributed_s"),
                trial_scan.get("unattributed_s"),
            ),
            "cache": {
                "lookup_s": _first_present(
                    _metric(summary, "trial_solver_scan_runner_cache_lookup_s"),
                    trial_scan.get("cache_lookup_s"),
                ),
                "build_s": _first_present(
                    _metric(summary, "trial_solver_scan_runner_cache_build_s"),
                    trial_scan.get("cache_build_s"),
                ),
                "hit_count": _first_present(
                    _metric(summary, "trial_solver_scan_runner_cache_hit_count"),
                    _scan_status_metric(summary, "hit", "count"),
                ),
                "miss_count": _first_present(
                    _metric(summary, "trial_solver_scan_runner_cache_miss_count"),
                    _scan_status_metric(summary, "miss", "count"),
                ),
                "bypass_count": _first_present(
                    _metric(summary, "trial_solver_scan_runner_cache_bypass_count"),
                    _scan_status_metric(summary, "bypass", "count"),
                ),
                "miss_fraction": trial_scan.get("cache_miss_fraction"),
                "hit_device_run_s": _first_present(
                    _metric(summary, "trial_solver_scan_runner_cache_hit_device_run_s"),
                    _scan_status_metric(summary, "hit", "device_run_s"),
                ),
                "hit_dispatch_s": _first_present(
                    _metric(summary, "trial_solver_scan_runner_cache_hit_dispatch_s"),
                    _scan_status_metric(summary, "hit", "dispatch_s"),
                ),
                "hit_ready_s": _first_present(
                    _metric(summary, "trial_solver_scan_runner_cache_hit_ready_s"),
                    _scan_status_metric(summary, "hit", "ready_s"),
                ),
                "miss_device_run_s": _first_present(
                    _metric(summary, "trial_solver_scan_runner_cache_miss_device_run_s"),
                    _scan_status_metric(summary, "miss", "device_run_s"),
                ),
                "miss_dispatch_s": _first_present(
                    _metric(summary, "trial_solver_scan_runner_cache_miss_dispatch_s"),
                    _scan_status_metric(summary, "miss", "dispatch_s"),
                ),
                "miss_ready_s": _first_present(
                    _metric(summary, "trial_solver_scan_runner_cache_miss_ready_s"),
                    _scan_status_metric(summary, "miss", "ready_s"),
                ),
                "bypass_device_run_s": _first_present(
                    _metric(summary, "trial_solver_scan_runner_cache_bypass_device_run_s"),
                    _scan_status_metric(summary, "bypass", "device_run_s"),
                ),
                "bypass_dispatch_s": _first_present(
                    _metric(summary, "trial_solver_scan_runner_cache_bypass_dispatch_s"),
                    _scan_status_metric(summary, "bypass", "dispatch_s"),
                ),
                "bypass_ready_s": _first_present(
                    _metric(summary, "trial_solver_scan_runner_cache_bypass_ready_s"),
                    _scan_status_metric(summary, "bypass", "ready_s"),
                ),
            },
        },
        "replay": {
            "hit_count": _metric(summary, "replay_scan_cache_hit_count"),
            "miss_count": _metric(summary, "replay_scan_cache_miss_count"),
            "lookup_s": _metric(summary, "replay_scan_cache_lookup_s"),
            "build_s": _metric(summary, "replay_scan_cache_build_s"),
        },
    }


def _matrix_projected_replay_jvp_summary(
    run: dict[str, Any] | None,
    summary: dict[str, Any] | None,
) -> dict[str, Any]:
    projected = _summary_section(summary, "projected_replay_summary")
    return {
        "jvp": {
            "exact_tape": _metadata_or_env(
                run,
                summary,
                metadata_key="jvp_only_exact_tape",
                env_key="VMEC_JAX_OPT_JVP_ONLY_EXACT_TAPE",
            ),
            "basepoint_carries": _metadata_or_env(
                run,
                summary,
                metadata_key="jvp_only_basepoint_carries",
                env_key="VMEC_JAX_JVP_ONLY_EXACT_TAPE_BASEPOINT_CARRIES",
            ),
            "exact_solve_with_tape_s": _metric(summary, "exact_solve_with_tape_jvp_only_s"),
            "tape_build_s": _metric(summary, "exact_tape_build_jvp_only_s"),
            "initial_tangents_s": _metric(summary, "initial_tangents_s"),
            "initial_tangents_linearize_s": _metric(summary, "initial_tangents_linearize_s"),
            "initial_tangents_vmap_dispatch_s": _metric(summary, "initial_tangents_vmap_dispatch_s"),
            "initial_tangents_vmap_ready_s": _metric(summary, "initial_tangents_vmap_ready_s"),
            "residual_tangents_s": _metric(summary, "residual_tangents_s"),
        },
        "projected_replay": {
            "total_s": _first_present(
                _metric(summary, "projected_replay_total_s"),
                projected.get("total_s"),
            ),
            "dispatch_s": _first_present(
                _metric(summary, "projected_replay_dispatch_s"),
                projected.get("dispatch_s"),
            ),
            "residual_tangents_s": _first_present(
                _metric(summary, "projected_residual_tangents_s"),
                projected.get("residual_tangents_s"),
            ),
            "count": projected.get("count"),
            "share_of_total": projected.get("share_of_total"),
            "residual_tangent_share_of_projected": projected.get("residual_tangent_share_of_projected"),
        },
    }


def _augment_profile_summary(summary: dict[str, Any], *, payload: dict[str, Any]) -> dict[str, Any]:
    metadata = summary.setdefault("metadata", {})
    if isinstance(metadata, dict):
        runtime = payload.get("runtime")
        if not isinstance(runtime, dict):
            runtime = {}
        metadata.setdefault("trial_scan", payload.get("trial_scan") or runtime.get("vmec_jax_opt_trial_scan"))
        metadata.setdefault("sync_replay_timing", payload.get("sync_replay_timing"))
        metadata.setdefault(
            "jvp_only_basepoint_carries",
            payload.get("jvp_only_basepoint_carries")
            if "jvp_only_basepoint_carries" in payload
            else runtime.get("vmec_jax_jvp_only_exact_tape_basepoint_carries"),
        )
    if _contains_data(_matrix_scan_cache_summary(summary)):
        summary["matrix_scan_cache_summary"] = _matrix_scan_cache_summary(summary)
    projected_jvp = _matrix_projected_replay_jvp_summary(None, summary)
    if _contains_data(projected_jvp):
        summary["matrix_projected_replay_jvp_summary"] = projected_jvp
    return summary


def _attach_matrix_summary_sections(run: dict[str, Any]) -> None:
    summary = run.get("summary")
    if not isinstance(summary, dict):
        return
    scan_cache = _matrix_scan_cache_summary(summary)
    if _contains_data(scan_cache):
        summary["matrix_scan_cache_summary"] = scan_cache
    projected_jvp = _matrix_projected_replay_jvp_summary(run, summary)
    if _contains_data(projected_jvp):
        summary["matrix_projected_replay_jvp_summary"] = projected_jvp


def _print_table(headers: tuple[str, ...], rows: list[tuple[Any, ...]]) -> None:
    widths = [
        max(len(headers[col]), *(len(_format_cell(row[col])) for row in rows)) if rows else len(headers[col])
        for col in range(len(headers))
    ]
    print("  ".join(headers[col].ljust(widths[col]) for col in range(len(headers))))
    print("  ".join("-" * widths[col] for col in range(len(headers))))
    for row in rows:
        print("  ".join(_format_cell(row[col]).ljust(widths[col]) for col in range(len(headers))))


def _print_optional_table(title: str, headers: tuple[str, ...], rows: list[tuple[Any, ...]]) -> None:
    if not rows or not any(any(value is not None for value in row[1:]) for row in rows):
        return
    print()
    print(title)
    _print_table(headers, rows)


def print_report(payload: dict[str, Any]) -> None:
    rows = []
    trial_scan_timing_rows = []
    scan_cache_rows = []
    projected_jvp_rows = []
    for run in payload["runs"]:
        summary = run.get("summary")
        _attach_matrix_summary_sections(run)
        scan_cache = _matrix_scan_cache_summary(summary)
        trial_scan = scan_cache["trial"]
        trial_cache = trial_scan["cache"]
        replay_cache = scan_cache["replay"]
        projected_jvp = _matrix_projected_replay_jvp_summary(run, summary)
        jvp = projected_jvp["jvp"]
        projected = projected_jvp["projected_replay"]
        rows.append(
            (
                run["backend"],
                "dry-run" if run.get("dry_run") else str(run.get("exit_code")),
                run.get("wall_time_s"),
                _metric(summary, "total_runtime_s"),
                _metric(summary, "vmec_solve_s"),
                _metric(summary, "vmec_compute_forces_s"),
                _metric(summary, "vmec_preconditioner_s"),
                _metric(summary, "vmec_update_s"),
                _metric(summary, "qi_first_call_s"),
                _metric(summary, "qi_warm_min_s"),
                _metric(summary, "exact_solve_s"),
                _metric(summary, "exact_solve_with_tape_jvp_only_s"),
                _metric(summary, "exact_tape_build_s"),
                _metric(summary, "exact_tape_build_jvp_only_s"),
                _metric(summary, "exact_tape_build_unattributed_s"),
                _metric(summary, "replay_time_s"),
                _metric(summary, "accepted_replay_dispatch_s"),
                _metric(summary, "accepted_replay_ready_s"),
                _metric(summary, "initial_tangents_s"),
                _metric(summary, "residual_tangents_s"),
                _first_present(
                    _metric(summary, "trial_solver_scan_device_run_s"),
                    _metric(summary, "vmec_scan_device_run_s"),
                ),
                _metric(summary, "callback_count"),
                _metric(summary, "accepted_point_replay_count"),
                _metric(summary, "contamination_warning_count"),
                run["report_path"],
            )
        )
        trial_scan_timing_rows.append(
            (
                run["backend"],
                trial_scan["total_s"],
                trial_scan["setup_s"],
                trial_scan["run_setup_s"],
                trial_scan["preflight_s"],
                trial_scan["device_run_s"],
                trial_scan["device_dispatch_s"],
                trial_scan["device_ready_s"],
                trial_scan["host_materialize_s"],
                trial_scan["postprocess_s"],
                trial_scan["unattributed_s"],
            )
        )
        scan_cache_rows.append(
            (
                run["backend"],
                trial_cache["hit_count"],
                trial_cache["miss_count"],
                trial_cache["bypass_count"],
                trial_cache["miss_fraction"],
                trial_cache["lookup_s"],
                trial_cache["build_s"],
                trial_cache["hit_ready_s"],
                trial_cache["miss_ready_s"],
                trial_cache["bypass_ready_s"],
                replay_cache["hit_count"],
                replay_cache["miss_count"],
                replay_cache["lookup_s"],
                replay_cache["build_s"],
            )
        )
        projected_jvp_rows.append(
            (
                run["backend"],
                jvp["exact_tape"],
                jvp["basepoint_carries"],
                jvp["exact_solve_with_tape_s"],
                jvp["tape_build_s"],
                jvp["initial_tangents_s"],
                jvp["initial_tangents_linearize_s"],
                jvp["initial_tangents_vmap_dispatch_s"],
                jvp["initial_tangents_vmap_ready_s"],
                jvp["residual_tangents_s"],
                projected["total_s"],
                projected["dispatch_s"],
                projected["residual_tangents_s"],
                projected["count"],
                projected["share_of_total"],
            )
        )
    headers = (
        "backend",
        "exit",
        "wrapper_s",
        "profile_s",
        "vmec_s",
        "forces_s",
        "precond_s",
        "update_s",
        "qi_first_s",
        "qi_warm_s",
        "exact_s",
        "exact_jvp_s",
        "tape_build_s",
        "tape_jvp_s",
        "tape_unattr_s",
        "replay_s",
        "replay_dispatch_s",
        "replay_ready_s",
        "init_tangent_s",
        "resid_tangent_s",
        "scan_device_s",
        "callbacks",
        "replays",
        "warnings",
        "report",
    )
    _print_table(headers, rows)
    _print_optional_table(
        "Scan timing:",
        (
            "backend",
            "scan_s",
            "setup_s",
            "run_setup_s",
            "preflight_s",
            "device_s",
            "dispatch_s",
            "ready_s",
            "host_s",
            "post_s",
            "unattr_s",
        ),
        trial_scan_timing_rows,
    )
    _print_optional_table(
        "Scan cache details:",
        (
            "backend",
            "trial_hits",
            "trial_misses",
            "trial_bypasses",
            "trial_miss_frac",
            "trial_lookup_s",
            "trial_build_s",
            "trial_hit_ready_s",
            "trial_miss_ready_s",
            "trial_bypass_ready_s",
            "replay_hits",
            "replay_misses",
            "replay_lookup_s",
            "replay_build_s",
        ),
        scan_cache_rows,
    )
    _print_optional_table(
        "Projected replay / JVP details:",
        (
            "backend",
            "jvp_tape",
            "base_carries",
            "exact_jvp_s",
            "tape_jvp_s",
            "init_tangent_s",
            "init_linearize_s",
            "init_vmap_dispatch_s",
            "init_vmap_ready_s",
            "resid_tangent_s",
            "proj_replay_s",
            "proj_dispatch_s",
            "proj_resid_tangent_s",
            "proj_count",
            "proj_share",
        ),
        projected_jvp_rows,
    )


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
