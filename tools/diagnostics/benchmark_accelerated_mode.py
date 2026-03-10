"""Benchmark baseline vs accelerated solver modes on bundled examples.

This tool compares two vmec_jax solver policies on the same cases and reports:

- cold and warm runtime,
- peak process memory,
- convergence and final fsq_total,
- final wout similarity against bundled reference wouts when available.

The intended baseline is the current default path, and the intended candidate is
the experimental ``solver_mode="accelerated"`` path.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vmec_jax.config import load_config


_RE_TIME_VALUE_DARWIN = re.compile(
    r"^\s*([0-9]+)\s+(peak memory footprint|maximum resident set size)\s*$",
    re.MULTILINE,
)
_RE_TIME_VALUE_LINUX = re.compile(
    r"^\s*Maximum resident set size \(kbytes\):\s*([0-9]+)\s*$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class CaseSpec:
    id: str
    input_path: Path
    source: str
    lfreeb: bool
    lasym: bool
    axisymmetric: bool
    ns: int
    mpol: int
    ntor: int
    nfp: int


def _child_env(*, jax_platforms: str | None) -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("VMEC_JAX_SCAN_PRINT", "0")
    env.setdefault("PYTHONUNBUFFERED", "1")
    if jax_platforms:
        env["JAX_PLATFORMS"] = str(jax_platforms)
    return env


def _format_seconds(value: float | None) -> str:
    if value is None or not (value == value):
        return "-"
    return f"{value:.2f}s"


def _format_ratio(value: float | None) -> str:
    if value is None or not (value == value):
        return "-"
    return f"{value:.2f}x"


def _parse_time_metrics(stderr: str) -> dict[str, int | None]:
    out: dict[str, int | None] = {
        "peak_footprint_bytes": None,
        "max_rss_bytes": None,
    }
    for value_s, label in _RE_TIME_VALUE_DARWIN.findall(stderr):
        value = int(value_s)
        if label == "peak memory footprint":
            out["peak_footprint_bytes"] = value
        elif label == "maximum resident set size":
            out["max_rss_bytes"] = value
    if out["max_rss_bytes"] is None:
        match = _RE_TIME_VALUE_LINUX.search(stderr)
        if match is not None:
            value = int(match.group(1)) * 1024
            out["max_rss_bytes"] = value
            out["peak_footprint_bytes"] = value
    return out


def _run_timed_subprocess(*, cmd: list[str], cwd: Path, timeout_s: float, env: dict[str, str]) -> dict[str, Any]:
    wrapped_cmd = cmd
    time_bin = Path("/usr/bin/time")
    if time_bin.exists():
        if platform.system().lower() == "darwin":
            wrapped_cmd = [str(time_bin), "-l", *cmd]
        else:
            wrapped_cmd = [str(time_bin), "-v", *cmd]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            wrapped_cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=float(timeout_s),
            check=False,
            env=env,
        )
        dt = time.perf_counter() - t0
        metrics = _parse_time_metrics(proc.stderr)
        return {
            "returncode": int(proc.returncode),
            "time_real_s": float(dt),
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "timed_out": False,
            **metrics,
        }
    except subprocess.TimeoutExpired as exc:
        dt = time.perf_counter() - t0
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        metrics = _parse_time_metrics(stderr)
        return {
            "returncode": 124,
            "time_real_s": float(dt),
            "stdout": stdout,
            "stderr": stderr,
            "timed_out": True,
            **metrics,
        }


def _json_tail(text: str) -> dict[str, Any] | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def _discover_cases() -> list[CaseSpec]:
    cases: list[CaseSpec] = []
    for input_path in sorted((REPO_ROOT / "examples" / "data").glob("input.*")):
        cfg, _ = load_config(input_path)
        cases.append(
            CaseSpec(
                id=input_path.name.removeprefix("input."),
                input_path=input_path.resolve(),
                source="bundled",
                lfreeb=bool(cfg.lfreeb),
                lasym=bool(cfg.lasym),
                axisymmetric=int(cfg.ntor) == 0,
                ns=int(cfg.ns),
                mpol=int(cfg.mpol),
                ntor=int(cfg.ntor),
                nfp=int(cfg.nfp),
            )
        )
    return cases


def _select_cases(cases: list[CaseSpec], *, ids: set[str] | None, kind: str) -> list[CaseSpec]:
    out = []
    for case in cases:
        if ids is not None and case.id not in ids:
            continue
        if kind == "fixed" and case.lfreeb:
            continue
        if kind == "freeb" and not case.lfreeb:
            continue
        out.append(case)
    return out


def _case_to_json(case: CaseSpec) -> dict[str, Any]:
    rec = asdict(case)
    rec["input_path"] = str(case.input_path)
    return rec


def _run_solver_mode_case(
    *,
    case: CaseSpec,
    solver_mode: str,
    cli_fixed_boundary_mode: bool,
    max_iter: int | None,
    warm_runs: int,
    timeout_s: float,
    env: dict[str, str],
) -> dict[str, Any]:
    code = r"""
import json
import sys
import time
from pathlib import Path

import jax
import numpy as np

from vmec_jax.api import load_wout, residual_scalars_from_state, run_fixed_boundary, wout_from_fixed_boundary_run


def _rel_rms(a, b, eps=1.0e-16):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return float("nan")
    num = float(np.sqrt(np.mean((a - b) ** 2)))
    denom = float(np.sqrt(np.mean(b ** 2)))
    return num / max(float(eps), denom)


def _reference_wout_path(input_path: Path) -> Path | None:
    name = input_path.name
    case = name.removeprefix("input.")
    data_dir = input_path.parent
    for cand in (
        data_dir / f"wout_{case}_reference.nc",
        data_dir / f"wout_{case}.nc",
    ):
        if cand.exists():
            return cand
    return None


def _quality_metrics(run, input_path: Path) -> dict:
    ref_path = _reference_wout_path(input_path)
    out = {
        "ref_wout_path": None if ref_path is None else str(ref_path),
        "ref_wout_available": bool(ref_path is not None),
        "quality_max_rel_rms": None,
        "quality_fields": {},
    }
    if ref_path is None:
        return out
    try:
        wnew = wout_from_fixed_boundary_run(run, include_fsq=True, fast_bcovar=True)
        wref = load_wout(ref_path)
    except Exception:
        return out
    fields = ("rmnc", "rmns", "zmnc", "zmns", "lmnc", "lmns", "iotaf", "presf", "bmnc", "bmns")
    metrics = {}
    worst = None
    for field in fields:
        if not hasattr(wnew, field) or not hasattr(wref, field):
            continue
        rel = _rel_rms(getattr(wnew, field), getattr(wref, field))
        metrics[field] = None if not np.isfinite(rel) else float(rel)
        if np.isfinite(rel):
            worst = rel if worst is None else max(worst, rel)
    out["quality_fields"] = metrics
    out["quality_max_rel_rms"] = None if worst is None else float(worst)
    return out


input_path = Path(sys.argv[1])
solver_mode = str(sys.argv[2])
cli_fixed_boundary_mode = bool(int(sys.argv[3]))
max_iter = None if sys.argv[4] == "none" else int(sys.argv[4])
warm_runs = int(sys.argv[5])


def _run_once():
    kwargs = dict(
        verbose=False,
        solver_mode=solver_mode,
        cli_fixed_boundary_mode=bool(cli_fixed_boundary_mode),
    )
    if max_iter is not None:
        kwargs["max_iter"] = int(max_iter)
    t0 = time.perf_counter()
    run = run_fixed_boundary(input_path, **kwargs)
    jax.block_until_ready(run.state.Rcos)
    dt = time.perf_counter() - t0
    return run, dt


run_cold, cold_dt = _run_once()
warm_times = []
run_last = run_cold
for _ in range(max(0, warm_runs)):
    run_last, dt = _run_once()
    warm_times.append(float(dt))

res = getattr(run_last, "result", None)
diag = {} if res is None else dict(getattr(res, "diagnostics", {}) or {})
if res is not None and getattr(res, "w_history", None) is not None and len(res.w_history) > 0:
    fsq_total = float(np.asarray(res.w_history, dtype=float)[-1])
else:
    fsqr, fsqz, fsql = residual_scalars_from_state(
        state=run_last.state,
        static=run_last.static,
        indata=run_last.indata,
        signgs=int(run_last.signgs),
        use_vmec_synthesis=True,
    )
    fsq_total = float(fsqr + fsqz + fsql)

payload = {
    "backend": "vmec_jax",
    "solver_mode": solver_mode,
    "cli_fixed_boundary_mode": bool(cli_fixed_boundary_mode),
    "runtime_cold_s": float(cold_dt),
    "runtime_warm_s": float(np.mean(warm_times)) if warm_times else None,
    "compile_overhead_s": float(max(0.0, cold_dt - np.mean(warm_times))) if warm_times else None,
    "ok": bool(res is not None),
    "n_iter": -1 if res is None else int(getattr(res, "n_iter", -1)),
    "converged": bool(diag.get("converged", False)),
    "use_scan": bool(diag.get("use_scan", False)),
    "accelerated_scan": bool(diag.get("accelerated_scan", False)),
    "cli_fixed_boundary_initial_policy": diag.get("cli_fixed_boundary_initial_policy"),
    "cli_fixed_boundary_staged_followup_used": bool(diag.get("cli_fixed_boundary_staged_followup_used", False)),
    "cli_fixed_boundary_full_parity_fallback": bool(diag.get("cli_fixed_boundary_full_parity_fallback", False)),
    "free_boundary": bool(diag.get("free_boundary", False)),
    "fsq_total": float(fsq_total),
    "platform": str(jax.default_backend()),
    "device_kind": str(jax.devices()[0].device_kind) if jax.devices() else "unknown",
}
payload.update(_quality_metrics(run_last, input_path))
print(json.dumps(payload))
"""
    out = _run_timed_subprocess(
        cmd=[
            sys.executable,
            "-c",
            code,
            str(case.input_path),
            str(solver_mode),
            "1" if bool(cli_fixed_boundary_mode) else "0",
            "none" if max_iter is None else str(int(max_iter)),
            str(int(warm_runs)),
        ],
        cwd=REPO_ROOT,
        timeout_s=float(timeout_s),
        env=env,
    )
    payload = _json_tail(out["stdout"])
    rec: dict[str, Any] = {
        "backend": "vmec_jax",
        "case_id": case.id,
        "solver_mode": str(solver_mode),
        "cli_fixed_boundary_mode": bool(cli_fixed_boundary_mode),
        "returncode": int(out["returncode"]),
        "time_real_s": float(out["time_real_s"]),
        "max_rss_bytes": out["max_rss_bytes"],
        "peak_footprint_bytes": out["peak_footprint_bytes"],
        "timed_out": bool(out.get("timed_out", False)),
        "stdout_tail": "\n".join(out["stdout"].splitlines()[-10:]),
        "stderr_tail": "\n".join(out["stderr"].splitlines()[-12:]),
        "child": payload,
    }
    if payload is None:
        rec["ok"] = False
        return rec
    rec.update(payload)
    rec["ok"] = bool(payload.get("ok", False)) and (int(out["returncode"]) == 0)
    return rec


def _speedup(candidate: dict[str, Any] | None, baseline: dict[str, Any] | None, key: str) -> float | None:
    if candidate is None or baseline is None:
        return None
    c = candidate.get(key)
    b = baseline.get(key)
    if not isinstance(c, (int, float)) or not isinstance(b, (int, float)) or c <= 0.0 or b <= 0.0:
        return None
    return float(b) / float(c)


def _mem_ratio(candidate: dict[str, Any] | None, baseline: dict[str, Any] | None) -> float | None:
    if candidate is None or baseline is None:
        return None
    c = candidate.get("peak_footprint_bytes")
    b = baseline.get("peak_footprint_bytes")
    if not isinstance(c, int) or not isinstance(b, int) or c <= 0 or b <= 0:
        return None
    return float(c) / float(b)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ids", type=str, default="", help="Comma-separated case ids to run.")
    p.add_argument("--kind", choices=("fixed", "freeb", "all"), default="fixed", help="Case filter.")
    p.add_argument("--baseline-mode", type=str, default="default", help="Baseline solver mode.")
    p.add_argument("--candidate-mode", type=str, default="accelerated", help="Candidate solver mode.")
    p.add_argument(
        "--baseline-cli-fixed-boundary-mode",
        action="store_true",
        help="Enable cli_fixed_boundary_mode for the baseline vmec_jax run.",
    )
    p.add_argument(
        "--candidate-cli-fixed-boundary-mode",
        action="store_true",
        help="Enable cli_fixed_boundary_mode for the candidate vmec_jax run.",
    )
    p.add_argument("--max-iter", type=int, default=None, help="Optional max_iter override.")
    p.add_argument("--warm-runs", type=int, default=1, help="Number of warmed reruns in the same child process.")
    p.add_argument("--timeout-s", type=float, default=1800.0, help="Timeout per case/mode subprocess.")
    p.add_argument(
        "--jax-platforms",
        type=str,
        default="",
        help="Value to set for JAX_PLATFORMS in child processes (for example 'cpu' or 'cuda,cpu').",
    )
    p.add_argument(
        "--quality-rtol",
        type=float,
        default=1.0e-2,
        help="Target maximum relRMS for reference wout fields.",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=REPO_ROOT / "outputs" / f"accelerated_mode_benchmark_{time.strftime('%Y%m%d_%H%M%S')}",
        help="Directory for summary artifacts.",
    )
    args = p.parse_args()

    ids = {tok.strip() for tok in args.ids.split(",") if tok.strip()} or None
    cases = _discover_cases()
    cases = _select_cases(cases, ids=ids, kind=str(args.kind))
    if not cases:
        raise SystemExit("No cases selected.")

    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    env = _child_env(jax_platforms=args.jax_platforms.strip() or None)

    results: list[dict[str, Any]] = []
    rows = [
        "case                                   base_warm  cand_warm  warm_up  mem_ratio  fsq_total     quality   converged",
        "---------------------------------------------------------------------------------------------------------------",
    ]
    comparisons: list[dict[str, Any]] = []

    for idx, case in enumerate(cases, start=1):
        print(f"[{idx}/{len(cases)}] {case.id}", flush=True)
        baseline = _run_solver_mode_case(
            case=case,
            solver_mode=str(args.baseline_mode),
            cli_fixed_boundary_mode=bool(args.baseline_cli_fixed_boundary_mode),
            max_iter=args.max_iter,
            warm_runs=int(args.warm_runs),
            timeout_s=float(args.timeout_s),
            env=env,
        )
        candidate = _run_solver_mode_case(
            case=case,
            solver_mode=str(args.candidate_mode),
            cli_fixed_boundary_mode=bool(args.candidate_cli_fixed_boundary_mode),
            max_iter=args.max_iter,
            warm_runs=int(args.warm_runs),
            timeout_s=float(args.timeout_s),
            env=env,
        )
        results.extend((baseline, candidate))
        quality = candidate.get("quality_max_rel_rms")
        quality_pass = (
            True
            if quality is None
            else bool(isinstance(quality, (int, float)) and quality <= float(args.quality_rtol))
        )
        converged_pass = bool(candidate.get("converged", False))
        warm_speedup = _speedup(candidate, baseline, "runtime_warm_s")
        cold_speedup = _speedup(candidate, baseline, "runtime_cold_s")
        mem_ratio = _mem_ratio(candidate, baseline)
        comp = {
            "case_id": case.id,
            "baseline_mode": str(args.baseline_mode),
            "candidate_mode": str(args.candidate_mode),
            "baseline_cli_fixed_boundary_mode": bool(args.baseline_cli_fixed_boundary_mode),
            "candidate_cli_fixed_boundary_mode": bool(args.candidate_cli_fixed_boundary_mode),
            "warm_speedup": warm_speedup,
            "cold_speedup": cold_speedup,
            "memory_ratio": mem_ratio,
            "candidate_fsq_total": candidate.get("fsq_total"),
            "candidate_quality_max_rel_rms": quality,
            "candidate_quality_pass": quality_pass,
            "candidate_converged": converged_pass,
            "candidate_initial_policy": candidate.get("cli_fixed_boundary_initial_policy"),
            "candidate_staged_followup_used": bool(candidate.get("cli_fixed_boundary_staged_followup_used", False)),
            "candidate_full_parity_fallback": bool(candidate.get("cli_fixed_boundary_full_parity_fallback", False)),
        }
        comparisons.append(comp)
        rows.append(
            f"{case.id:38s}  "
            f"{_format_seconds(baseline.get('runtime_warm_s')):>9s}  "
            f"{_format_seconds(candidate.get('runtime_warm_s')):>9s}  "
            f"{_format_ratio(warm_speedup):>7s}  "
            f"{_format_ratio(mem_ratio):>9s}  "
            f"{candidate.get('fsq_total', float('nan')):>11.3e}  "
            f"{(quality if isinstance(quality, (int, float)) else float('nan')):>8.2e}  "
            f"{str(converged_pass):>9s}"
        )

    summary = {
        "cases": [_case_to_json(case) for case in cases],
        "results": results,
        "comparisons": comparisons,
        "baseline_mode": str(args.baseline_mode),
        "candidate_mode": str(args.candidate_mode),
        "baseline_cli_fixed_boundary_mode": bool(args.baseline_cli_fixed_boundary_mode),
        "candidate_cli_fixed_boundary_mode": bool(args.candidate_cli_fixed_boundary_mode),
        "quality_rtol": float(args.quality_rtol),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "jax_platforms": str(args.jax_platforms),
    }
    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    table_path = outdir / "summary.txt"
    table_path.write_text("\n".join(rows) + "\n")
    print(f"summary={summary_path}")
    print("\n".join(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
