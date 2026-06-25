"""Benchmark runtime and memory across bundled vmec_jax examples.

This tool measures the default user path (`run_fixed_boundary(input, verbose=False)`)
against a VMEC2000 executable when available. It records wall time,
backend-reported runtime, and memory signals from `/usr/bin/time -l/-v`.
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
from vmec_jax.vmec2000_exec import find_vmec2000_exec


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
    env.setdefault("VMEC_JAX_BENCH_WARM_RUNS", "0")
    # Minimal scan mode keeps only fsq/w_history to avoid OOM on large NS runs.
    env.setdefault("VMEC_JAX_SCAN_MINIMAL", "1")
    if jax_platforms:
        env["JAX_PLATFORMS"] = str(jax_platforms)
    return env


def _format_seconds(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}s"


def _format_ratio(value: float | None) -> str:
    if value is None:
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


def _run_timed_subprocess(
    *,
    cmd: list[str],
    cwd: Path,
    timeout_s: float,
    env: dict[str, str],
) -> dict[str, Any]:
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
    text = text.strip()
    if not text:
        return None
    lines = [line for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        line = line.strip()
        if not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def _run_vmec_jax_case(
    *,
    case: CaseSpec,
    timeout_s: float,
    env: dict[str, str],
    runner_label: str,
    solver_mode: str | None,
    solver_device: str | None,
    cli_fixed_boundary_mode: bool,
) -> dict[str, Any]:
    case_workdir = REPO_ROOT / "outputs" / "_bench_stage" / f"work_vmec_jax_{case.id}"
    case_workdir.mkdir(parents=True, exist_ok=True)
    staged = _stage_input_with_mgrid(input_path=case.input_path, dst_dir=case_workdir)
    code = r"""
import json
import sys
import time
from pathlib import Path
import jax
from vmec_jax.api import run_fixed_boundary

input_path = Path(sys.argv[1])
solver_mode_arg = str(sys.argv[2])
solver_mode = None if solver_mode_arg in ("", "None", "none", "null") else solver_mode_arg
solver_device_arg = str(sys.argv[3])
solver_device = None if solver_device_arg in ("", "None", "none", "null") else solver_device_arg
cli_fixed_boundary_mode = bool(int(sys.argv[4]))
warm_runs = int(sys.argv[5])
t0 = time.perf_counter()
run = run_fixed_boundary(
    input_path,
    verbose=False,
    solver_mode=solver_mode,
    solver_device=solver_device,
    cli_fixed_boundary_mode=bool(cli_fixed_boundary_mode),
)
dt = time.perf_counter() - t0
cold_res = getattr(run, "result", None)
cold_diag = {} if cold_res is None else dict(getattr(cold_res, "diagnostics", {}) or {})
warm_times = []
for _ in range(max(0, warm_runs)):
    t1 = time.perf_counter()
    run = run_fixed_boundary(
        input_path,
        verbose=False,
        solver_mode=solver_mode,
        solver_device=solver_device,
        cli_fixed_boundary_mode=bool(cli_fixed_boundary_mode),
    )
    warm_times.append(time.perf_counter() - t1)
res = getattr(run, "result", None)
diag = {} if res is None else dict(getattr(res, "diagnostics", {}) or {})
free_diag = diag.get("free_boundary", False)
if isinstance(free_diag, dict):
    free_boundary = bool(free_diag.get("enabled", False))
else:
    free_boundary = bool(free_diag)
policy_keys = (
    "host_update_assembly",
    "numpy_force_fast_path",
    "numpy_force_fast_path_active",
    "numpy_force_fast_path_max_iter",
    "numpy_preconditioner_apply",
    "jit_strict_update_enabled",
    "jit_strict_update_work",
    "jit_strict_update_cpu_work_limit",
    "use_scan",
)
execution_policy_cold = {key: cold_diag.get(key) for key in policy_keys if key in cold_diag}
execution_policy = {key: diag.get(key) for key in policy_keys if key in diag}
payload = {
    "backend": "vmec_jax",
    "runtime_s": float(dt),
    "runtime_cold_s": float(dt),
    "runtime_warm_s": (None if not warm_times else float(sum(warm_times) / len(warm_times))),
    "ok": bool(res is not None),
    "n_iter": -1 if res is None else int(getattr(res, "n_iter", -1)),
    "converged": bool(diag.get("converged", False)),
    "use_scan": bool(diag.get("use_scan", False)),
    "free_boundary": free_boundary,
    "solver_mode": diag.get("solver_mode", solver_mode),
    "solver_device": diag.get("solver_device", solver_device),
    "cli_fixed_boundary_mode": bool(cli_fixed_boundary_mode),
    "platform": str(jax.default_backend()),
    "device_kind": str(jax.devices()[0].device_kind) if jax.devices() else "unknown",
    "execution_policy_cold": execution_policy_cold,
    "execution_policy": execution_policy,
}
print(json.dumps(payload))
"""
    out = _run_timed_subprocess(
        cmd=[
            sys.executable,
            "-c",
            code,
            str(staged),
            "" if solver_mode is None else str(solver_mode),
            "" if solver_device is None else str(solver_device),
            "1" if bool(cli_fixed_boundary_mode) else "0",
            str(int(env.get("VMEC_JAX_BENCH_WARM_RUNS", "0"))),
        ],
        cwd=case_workdir,
        timeout_s=float(timeout_s),
        env=env,
    )
    payload = _json_tail(out["stdout"])
    rec = {
        "backend": "vmec_jax",
        "runner_label": runner_label,
        "case_id": case.id,
        "returncode": int(out["returncode"]),
        "time_real_s": float(out["time_real_s"]),
        "max_rss_bytes": out["max_rss_bytes"],
        "peak_footprint_bytes": out["peak_footprint_bytes"],
        "timed_out": bool(out.get("timed_out", False)),
        "stdout_tail": "\n".join(out["stdout"].splitlines()[-10:]),
        "stderr_tail": "\n".join(out["stderr"].splitlines()[-12:]),
        "child": payload,
    }
    if payload is not None:
        rec["runtime_s"] = float(payload.get("runtime_s", out["time_real_s"]))
        rec["runtime_cold_s"] = float(payload.get("runtime_cold_s", rec["runtime_s"]))
        if payload.get("runtime_warm_s", None) is not None:
            rec["runtime_warm_s"] = float(payload["runtime_warm_s"])
        rec["ok"] = bool(payload.get("ok", False)) and (int(out["returncode"]) == 0)
        rec["n_iter"] = int(payload.get("n_iter", -1))
        rec["converged"] = bool(payload.get("converged", False))
        rec["use_scan"] = bool(payload.get("use_scan", False))
        payload_solver_mode = payload.get("solver_mode", solver_mode)
        rec["solver_mode"] = None if payload_solver_mode is None else str(payload_solver_mode)
        payload_solver_device = payload.get("solver_device", solver_device)
        rec["solver_device"] = None if payload_solver_device is None else str(payload_solver_device)
        rec["cli_fixed_boundary_mode"] = bool(payload.get("cli_fixed_boundary_mode", cli_fixed_boundary_mode))
        rec["platform"] = str(payload.get("platform", "unknown"))
        rec["device_kind"] = str(payload.get("device_kind", "unknown"))
        rec["execution_policy_cold"] = dict(payload.get("execution_policy_cold", {}) or {})
        rec["execution_policy"] = dict(payload.get("execution_policy", {}) or {})
    else:
        rec["runtime_s"] = float(out["time_real_s"])
        rec["ok"] = False
    return rec


def _run_vmec2000_case(*, case: CaseSpec, exec_path: Path, timeout_s: float, env: dict[str, str]) -> dict[str, Any]:
    case_workdir = REPO_ROOT / "outputs" / "_bench_stage" / f"work_vmec2000_{case.id}"
    case_workdir.mkdir(parents=True, exist_ok=True)
    staged = _stage_input_with_mgrid(input_path=case.input_path, dst_dir=case_workdir)
    code = r"""
import json
import sys
from pathlib import Path
from vmec_jax.vmec2000_exec import run_xvmec2000

input_path = Path(sys.argv[1])
exec_path = Path(sys.argv[2])
res = run_xvmec2000(input_path=input_path, exec_path=exec_path, timeout_s=float(sys.argv[3]), keep_workdir=False)
payload = {
    "backend": "vmec2000",
    "runtime_s": float(res.runtime_s),
    "ok": bool("EXECUTION TERMINATED NORMALLY" in res.stdout),
    "n_stage": int(len(res.stages)),
    "has_threed1": bool(res.threed1_path is not None),
}
print(json.dumps(payload))
"""
    out = _run_timed_subprocess(
        cmd=[sys.executable, "-c", code, str(staged), str(exec_path), str(float(timeout_s))],
        cwd=case_workdir,
        timeout_s=float(timeout_s) + 30.0,
        env=env,
    )
    payload = _json_tail(out["stdout"])
    rec = {
        "backend": "vmec2000",
        "case_id": case.id,
        "returncode": int(out["returncode"]),
        "time_real_s": float(out["time_real_s"]),
        "max_rss_bytes": out["max_rss_bytes"],
        "peak_footprint_bytes": out["peak_footprint_bytes"],
        "timed_out": bool(out.get("timed_out", False)),
        "stdout_tail": "\n".join(out["stdout"].splitlines()[-10:]),
        "stderr_tail": "\n".join(out["stderr"].splitlines()[-12:]),
        "child": payload,
    }
    if payload is not None:
        rec["runtime_s"] = float(payload.get("runtime_s", out["time_real_s"]))
        rec["ok"] = bool(payload.get("ok", False)) and (int(out["returncode"]) == 0)
        rec["n_stage"] = int(payload.get("n_stage", -1))
    else:
        rec["runtime_s"] = float(out["time_real_s"])
        rec["ok"] = False
    return rec


def _stage_input_with_mgrid(*, input_path: Path, dst_dir: Path) -> Path:
    """Copy input + referenced mgrid file (if any) into dst_dir.

    VMEC++ resolves relative mgrid paths relative to cwd, so staging avoids
    cwd-dependent failures and avoids polluting the source tree with outputs.
    """

    text = input_path.read_text()
    dst_input = dst_dir / input_path.name
    dst_input.write_text(text)

    if "mgrid_file" not in text.lower():
        return dst_input

    def _resolve_support_file(name: str) -> Path | None:
        direct = (input_path.parent / name).resolve()
        if direct.exists():
            return direct
        search_roots = [
            REPO_ROOT / "examples" / "data",
            REPO_ROOT / "examples_single_grid" / "data",
            REPO_ROOT.parent / "STELLOPT" / "BENCHMARKS" / "VMEC_TEST",
            REPO_ROOT.parent / "external",
            REPO_ROOT / "outputs",
        ]
        for root in search_roots:
            if not root.exists():
                continue
            exact = (root / name).resolve()
            if exact.exists():
                return exact
            matches = list(root.rglob(name))
            if matches:
                return matches[0].resolve()
        return None

    for line in text.splitlines():
        if "mgrid_file" not in line.lower():
            continue
        if "=" not in line:
            continue
        rhs = line.split("=", 1)[1]
        rhs = rhs.split("!", 1)[0].strip().rstrip(",")
        rhs = rhs.strip().strip('"').strip("'")
        if not rhs or rhs.strip().upper() == "NONE":
            continue
        src = _resolve_support_file(rhs)
        if src is None:
            raise FileNotFoundError(
                f"Could not locate support file {rhs!r} referenced by {input_path}"
            )
        (dst_dir / src.name).write_bytes(src.read_bytes())
        break
    return dst_input


def _run_vmecpp_case(*, case: CaseSpec, timeout_s: float, env: dict[str, str], outdir: Path) -> dict[str, Any]:
    """Run VMEC++ CLI (`vmecpp`) and measure runtime/memory via /usr/bin/time."""

    case_workdir = outdir / f"work_vmecpp_{case.id}"
    case_workdir.mkdir(parents=True, exist_ok=True)
    staged = _stage_input_with_mgrid(input_path=case.input_path, dst_dir=case_workdir)

    out = _run_timed_subprocess(
        cmd=["vmecpp", "-q", str(staged)],
        cwd=case_workdir,
        timeout_s=float(timeout_s),
        env=env,
    )
    rec = {
        "backend": "vmecpp",
        "case_id": case.id,
        "returncode": int(out["returncode"]),
        "time_real_s": float(out["time_real_s"]),
        "max_rss_bytes": out["max_rss_bytes"],
        "peak_footprint_bytes": out["peak_footprint_bytes"],
        "timed_out": bool(out.get("timed_out", False)),
        "stdout_tail": "\n".join(out["stdout"].splitlines()[-10:]),
        "stderr_tail": "\n".join(out["stderr"].splitlines()[-12:]),
    }
    # VMEC++ doesn't currently emit a reliable "total runtime" scalar in a stable
    # machine-readable form, so we use the wall time.
    rec["runtime_s"] = float(out["time_real_s"])
    rec["ok"] = int(out["returncode"]) == 0
    # Avoid accumulating large netCDF outputs in the benchmark workdir.
    for p in case_workdir.glob("wout_*.nc"):
        try:
            p.unlink()
        except OSError:
            pass
    return rec


def _discover_cases(
    *,
    inputs_dir: Path,
    inputs_glob: str,
    include_external_diiid: bool,
) -> list[CaseSpec]:
    cases: list[CaseSpec] = []
    for input_path in sorted(Path(inputs_dir).expanduser().resolve().glob(str(inputs_glob))):
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
    if include_external_diiid:
        external_env = os.environ.get("VMEC_JAX_EXTERNAL_DIIID_INPUTS", "").strip()
        if external_env:
            extras = [Path(tok).expanduser() for tok in external_env.split(os.pathsep) if tok.strip()]
        else:
            extras = [
                REPO_ROOT.parent / "STELLOPT" / "BENCHMARKS" / "VMEC_TEST" / "input.DIII-D",
                REPO_ROOT.parent / "STELLOPT" / "BENCHMARKS" / "VMEC_TEST" / "input.DIII-D_reset",
            ]
        for extra in extras:
            if not extra.exists():
                continue
            cfg, _ = load_config(extra)
            cases.append(
                CaseSpec(
                    id=extra.name.removeprefix("input."),
                    input_path=extra.resolve(),
                    source="external",
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


def _select_cases(cases: list[CaseSpec], *, ids: set[str] | None, kind: str | None) -> list[CaseSpec]:
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


def _case_row(case: CaseSpec, results_by_case: dict[str, dict[str, dict[str, Any]]]) -> str:
    vmec_jax = results_by_case.get(case.id, {}).get("vmec_jax")
    vmec2000 = results_by_case.get(case.id, {}).get("vmec2000")
    vmecpp = results_by_case.get(case.id, {}).get("vmecpp")
    rt_jax = None
    if vmec_jax is not None:
        rt_jax = float(
            vmec_jax.get(
                "runtime_warm_s",
                vmec_jax.get("runtime_s", vmec_jax.get("time_real_s", float("nan"))),
            )
        )
    rt_vmec = None if vmec2000 is None else float(vmec2000.get("runtime_s", vmec2000.get("time_real_s", float("nan"))))
    rt_pp = None if vmecpp is None else float(vmecpp.get("runtime_s", vmecpp.get("time_real_s", float("nan"))))
    ratio = None
    if rt_jax is not None and rt_vmec is not None and rt_vmec > 0.0:
        ratio = rt_jax / rt_vmec
    mem_ratio = None
    if vmec_jax is not None and vmec2000 is not None:
        mem_jax = vmec_jax.get("peak_footprint_bytes")
        mem_vmec = vmec2000.get("peak_footprint_bytes")
        if isinstance(mem_jax, int) and isinstance(mem_vmec, int) and mem_vmec > 0:
            mem_ratio = float(mem_jax) / float(mem_vmec)
    kind = "freeb" if case.lfreeb else "fixed"
    sym = "axisym" if case.axisymmetric else "nonaxis"
    lasym = "lasym" if case.lasym else "sym"
    return (
        f"{case.id:38s}  {kind:5s}  {sym:7s}  {lasym:5s}  "
        f"{_format_seconds(rt_jax):>8s}  {_format_seconds(rt_pp):>8s}  {_format_seconds(rt_vmec):>8s}  "
        f"{_format_ratio(ratio):>7s}  {_format_ratio(mem_ratio):>7s}"
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ids", type=str, default="", help="Comma-separated case ids to run.")
    p.add_argument("--kind", choices=("fixed", "freeb"), default=None, help="Filter to fixed or free-boundary cases.")
    p.add_argument(
        "--inputs-dir",
        type=Path,
        default=REPO_ROOT / "examples" / "data",
        help="Directory containing input.* files to benchmark.",
    )
    p.add_argument(
        "--inputs-glob",
        type=str,
        default="input.*",
        help="Glob (relative to --inputs-dir) selecting input files.",
    )
    p.add_argument(
        "--backend",
        choices=("all", "both", "vmec_jax", "vmec2000", "vmecpp"),
        default="both",
        help="Backends to benchmark. both=vmec_jax+vmec2000, all=vmec_jax+vmecpp+vmec2000.",
    )
    p.add_argument(
        "--include-external-diiid",
        action="store_true",
        help="Include external DIII-D benchmark inputs from VMEC_JAX_EXTERNAL_DIIID_INPUTS or a sibling STELLOPT checkout.",
    )
    p.add_argument(
        "--vmec-exec",
        type=Path,
        default=find_vmec2000_exec(root=REPO_ROOT.parent),
        help="Path to xvmec2000.",
    )
    p.add_argument("--timeout-s", type=float, default=1800.0, help="Timeout per vmec_jax case.")
    p.add_argument("--vmec-timeout-s", type=float, default=1800.0, help="Timeout per VMEC2000 case.")
    p.add_argument(
        "--jax-platforms",
        type=str,
        default="",
        help="Value to set for JAX_PLATFORMS in vmec_jax child processes (for example 'cpu' or 'cuda,cpu').",
    )
    p.add_argument(
        "--runner-label",
        type=str,
        default="default",
        help="Free-form label stored in vmec_jax records (for example 'cpu' or 'gpu').",
    )
    p.add_argument(
        "--solver-mode",
        type=str,
        default=None,
        help="Solver mode passed to run_fixed_boundary. Omit to benchmark the public default policy.",
    )
    p.add_argument(
        "--solver-device",
        type=str,
        default=None,
        help="Solver device passed to run_fixed_boundary: auto|default|cpu|gpu.",
    )
    p.add_argument(
        "--cli-fixed-boundary-mode",
        action="store_true",
        help="Enable cli_fixed_boundary_mode for vmec_jax runs.",
    )
    p.add_argument(
        "--warm-runs",
        type=int,
        default=0,
        help="Additional warmed vmec_jax runs to average inside the child process.",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=REPO_ROOT / "outputs" / f"example_runtime_memory_matrix_{time.strftime('%Y%m%d_%H%M%S')}",
        help="Directory for summary artifacts.",
    )
    args = p.parse_args()

    ids = {tok.strip() for tok in args.ids.split(",") if tok.strip()} or None
    cases = _discover_cases(
        inputs_dir=Path(args.inputs_dir),
        inputs_glob=str(args.inputs_glob),
        include_external_diiid=bool(args.include_external_diiid),
    )
    cases = _select_cases(cases, ids=ids, kind=args.kind)
    if not cases:
        raise SystemExit("No cases selected.")

    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    results_by_case: dict[str, dict[str, dict[str, Any]]] = {}

    run_vmec_jax = args.backend in ("all", "both", "vmec_jax")
    run_vmecpp = args.backend in ("all", "vmecpp")
    run_vmec2000 = args.backend in ("all", "both", "vmec2000")
    child_env = _child_env(jax_platforms=args.jax_platforms.strip() or None)
    child_env["VMEC_JAX_BENCH_WARM_RUNS"] = str(max(0, int(args.warm_runs)))
    vmec_exec = None if args.vmec_exec is None else Path(args.vmec_exec).expanduser().resolve()
    if run_vmec2000 and (vmec_exec is None or not vmec_exec.exists()):
        raise SystemExit("VMEC2000 executable not found. Use --vmec-exec.")

    print(f"selected_cases={len(cases)}")
    print(f"outdir={outdir}")
    if vmec_exec is not None:
        print(f"vmec_exec={vmec_exec}")

    for idx, case in enumerate(cases, start=1):
        kind = "freeb" if case.lfreeb else "fixed"
        sym = "axisym" if case.axisymmetric else "nonaxis"
        print(f"[{idx}/{len(cases)}] {case.id} ({kind}, {sym}, lasym={case.lasym})", flush=True)
        results_by_case.setdefault(case.id, {})
        if run_vmec_jax:
            rec = _run_vmec_jax_case(
                case=case,
                timeout_s=float(args.timeout_s),
                env=child_env,
                runner_label=str(args.runner_label),
                solver_mode=args.solver_mode,
                solver_device=args.solver_device,
                cli_fixed_boundary_mode=bool(args.cli_fixed_boundary_mode),
            )
            results.append(rec)
            results_by_case[case.id]["vmec_jax"] = rec
            print(
                f"  vmec_jax: rc={rec['returncode']} runtime={_format_seconds(float(rec.get('runtime_s', rec['time_real_s'])))} "
                f"peak={rec.get('peak_footprint_bytes')} platform={rec.get('platform', '-')}",
                flush=True,
            )
        if run_vmecpp:
            rec = _run_vmecpp_case(
                case=case,
                timeout_s=float(args.timeout_s),
                env=child_env,
                outdir=outdir,
            )
            results.append(rec)
            results_by_case[case.id]["vmecpp"] = rec
            print(
                f"  vmecpp:   rc={rec['returncode']} runtime={_format_seconds(float(rec.get('runtime_s', rec['time_real_s'])))} "
                f"peak={rec.get('peak_footprint_bytes')}",
                flush=True,
            )
        if run_vmec2000:
            rec = _run_vmec2000_case(
                case=case,
                exec_path=vmec_exec,
                timeout_s=float(args.vmec_timeout_s),
                env=child_env,
            )
            results.append(rec)
            results_by_case[case.id]["vmec2000"] = rec
            print(
                f"  vmec2000: rc={rec['returncode']} runtime={_format_seconds(float(rec.get('runtime_s', rec['time_real_s'])))} "
                f"peak={rec.get('peak_footprint_bytes')}",
                flush=True,
            )

    summary = {
        "cases": [_case_to_json(case) for case in cases],
        "results": results,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runner_label": str(args.runner_label),
        "solver_mode": args.solver_mode,
        "solver_device": args.solver_device,
        "cli_fixed_boundary_mode": bool(args.cli_fixed_boundary_mode),
        "warm_runs": int(args.warm_runs),
        "jax_platforms": str(args.jax_platforms),
    }
    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    rows = [
        "case                                   kind   topo     sym    vmec_jax    vmecpp  vmec2000    speed      mem",
        "----------------------------------------------------------------------------------------------------------",
    ]
    rows.extend(_case_row(case, results_by_case) for case in cases)
    table = "\n".join(rows)
    table_path = outdir / "summary.txt"
    table_path.write_text(table + "\n")

    print(f"summary={summary_path}")
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
