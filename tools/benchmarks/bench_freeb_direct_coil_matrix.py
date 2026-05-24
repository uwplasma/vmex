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


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Summary JSON path, or an output directory.")
    p.add_argument("--quick", action=argparse.BooleanOptionalAction, default=True, help="Use small CPU-safe defaults.")
    p.add_argument("--include-gpu", action="store_true", help="Also run GPU rows when a JAX GPU device is available.")
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
        devices = jax.devices()
        info = {
            "has_jax": True,
            "default_backend": str(jax.default_backend()),
            "devices": [str(device) for device in devices],
            "platforms": sorted({str(getattr(device, "platform", "unknown")) for device in devices}),
        }
        return any(str(getattr(device, "platform", "")).lower() in {"gpu", "cuda", "rocm"} for device in devices), info
    except Exception as exc:
        return False, {"has_jax": None, "devices": [], "error": repr(exc)}


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
        rows.append(row)
    return rows


def _child_specs(*, quick: bool, outdir: Path, backend: str) -> list[tuple[str, Path, list[str]]]:
    suffix = f"_{backend}.json"
    if quick:
        return [
            (
                "provider",
                outdir / f"bench_external_field_providers{suffix}",
                ["--points", "8", "--segments", "8", "--warm-repeats", "1", "--skip-essos"],
            ),
            (
                "direct_solve",
                outdir / f"bench_freeb_direct_coil_solve{suffix}",
                ["--max-iter", "2", "--warm-repeats", "1"],
            ),
            (
                "gradient",
                outdir / f"bench_freeb_coil_gradient{suffix}",
                ["--points", "8", "--segments", "8", "--matrix-size", "8", "--warm-repeats", "1"],
            ),
        ]
    return [
        (
            "provider",
            outdir / f"bench_external_field_providers{suffix}",
            ["--points", "48", "--segments", "48", "--warm-repeats", "5", "--skip-essos"],
        ),
        (
            "direct_solve",
            outdir / f"bench_freeb_direct_coil_solve{suffix}",
            ["--max-iter", "2", "--warm-repeats", "1"],
        ),
        (
            "gradient",
            outdir / f"bench_freeb_coil_gradient{suffix}",
            ["--points", "24", "--segments", "48", "--matrix-size", "24", "--warm-repeats", "5"],
        ),
    ]


def _script_for(label: str) -> Path:
    return {
        "provider": REPO_ROOT / "tools" / "benchmarks" / "bench_external_field_providers.py",
        "direct_solve": REPO_ROOT / "tools" / "benchmarks" / "bench_freeb_direct_coil_solve.py",
        "gradient": REPO_ROOT / "tools" / "benchmarks" / "bench_freeb_coil_gradient.py",
    }[label]


def _run_child(label: str, out: Path, args: list[str], *, backend: str, timeout_s: float) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}" if env.get("PYTHONPATH") else str(REPO_ROOT)
    if backend == "cpu":
        env["JAX_PLATFORMS"] = "cpu"
    elif backend == "gpu":
        env["JAX_PLATFORMS"] = "gpu"

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
            "stdout_tail": proc.stdout.strip().splitlines()[-8:],
            "stderr_tail": proc.stderr.strip().splitlines()[-8:],
            "child_status": None if payload is None else payload.get("status"),
            "child_backend": None if payload is None else payload.get("backend"),
            "case_counts": _case_counts(payload),
            "timings": _timing_snapshot(payload, include_nestor=(label == "direct_solve")),
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
    for label, out, child_args in _child_specs(quick=bool(args.quick), outdir=outdir, backend="cpu"):
        rows.append(_run_child(label, out, child_args, backend="cpu", timeout_s=float(args.timeout_s)))

    if args.include_gpu:
        if gpu_ok:
            for label, out, child_args in _child_specs(quick=bool(args.quick), outdir=outdir, backend="gpu"):
                rows.append(_run_child(label, out, child_args, backend="gpu", timeout_s=float(args.timeout_s)))
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
    payload = {
        "status": status,
        "script": str(Path(__file__).resolve()),
        "quick": bool(args.quick),
        "include_gpu": bool(args.include_gpu),
        "backend_note": str(args.backend_note),
        "gpu_probe": gpu_probe,
        "output_dir": outdir,
        "rows": rows,
    }
    _write_json(summary, payload)

    print(f"[bench-freeb-direct-coil-matrix] wrote {summary}")
    for row in rows:
        detail = row.get("reason") or row.get("child_status") or row.get("returncode")
        print(f"[bench-freeb-direct-coil-matrix] {row['backend']} {row['label']}: {row['status']} ({detail})")
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
