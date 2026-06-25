"""Render the README single-grid runtime/memory comparison.

This communication benchmark compares two small, converged single-grid examples:

- ``input.circular_tokamak``
- ``input.nfp4_QH_warm_start``

It measures VMEC2000, VMEC++, and vmec_jax.  For vmec_jax it records cold and
warm timings in the same Python process for both JIT-enabled and no-JIT runs.
The warm time is the mean of the solves after the first solve.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vmec_jax.vmec2000_exec import _patch_indata, find_vmec2000_exec


REPO_ROOT = Path(__file__).resolve().parents[2]
DARWIN_MEM_RE = re.compile(r"^\s*([0-9]+)\s+(peak memory footprint|maximum resident set size)\s*$", re.MULTILINE)
LINUX_MEM_RE = re.compile(r"^\s*Maximum resident set size \(kbytes\):\s*([0-9]+)\s*$", re.MULTILINE)


def _time_prefix() -> list[str]:
    time_bin = Path("/usr/bin/time")
    if not time_bin.exists():
        return []
    return [str(time_bin), "-l"] if platform.system().lower() == "darwin" else [str(time_bin), "-v"]


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _parse_mem(stderr: str) -> dict[str, int | None]:
    out: dict[str, int | None] = {"peak_footprint_bytes": None, "max_rss_bytes": None}
    for value_s, label in DARWIN_MEM_RE.findall(stderr):
        key = "peak_footprint_bytes" if label == "peak memory footprint" else "max_rss_bytes"
        out[key] = int(value_s)
    if out["max_rss_bytes"] is None:
        match = LINUX_MEM_RE.search(stderr)
        if match:
            out["max_rss_bytes"] = int(match.group(1)) * 1024
            out["peak_footprint_bytes"] = out["max_rss_bytes"]
    return out


def _best_mem_bytes(record: dict[str, Any]) -> int | None:
    for key in ("peak_footprint_bytes", "max_rss_bytes"):
        value = record.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return None


def _run(cmd: list[str], *, cwd: Path, timeout_s: float, env: dict[str, str] | None = None) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [*_time_prefix(), *cmd],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=float(timeout_s),
            env=env,
            check=False,
        )
        elapsed = time.perf_counter() - t0
        return {
            "returncode": proc.returncode,
            "time_real_s": elapsed,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "timed_out": False,
            **_parse_mem(proc.stderr),
        }
    except subprocess.TimeoutExpired as exc:
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return {
            "returncode": 124,
            "time_real_s": time.perf_counter() - t0,
            "stdout": exc.stdout if isinstance(exc.stdout, str) else "",
            "stderr": stderr,
            "timed_out": True,
            **_parse_mem(stderr),
        }


def _json_tail(stdout: str) -> dict[str, Any]:
    for line in reversed([line.strip() for line in stdout.splitlines() if line.strip()]):
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    return {}


def _write_input(input_path: Path, workdir: Path, updates: dict[str, str]) -> Path:
    workdir.mkdir(parents=True, exist_ok=True)
    dst = workdir / input_path.name
    text = input_path.read_text()
    dst.write_text(text if not updates else _patch_indata(text, updates=updates))
    return dst


def _updates(args: argparse.Namespace) -> dict[str, str]:
    return _runtime_updates(ns=args.ns, niter=args.niter, ftol=args.ftol, nstep=args.nstep)


def _runtime_updates(
    *,
    ns: int | None,
    niter: int | None,
    ftol: float | None,
    nstep: int | None,
) -> dict[str, str]:
    out: dict[str, str] = {}
    if ns is not None:
        out["NS_ARRAY"] = str(ns)
    if niter is not None:
        out["NITER_ARRAY"] = str(niter)
    if ftol is not None:
        out["FTOL_ARRAY"] = f"{float(ftol):.3e}"
    if nstep is not None:
        out["NSTEP"] = str(nstep)
    return out


def _external_record(
    *,
    case: str,
    backend: str,
    cmd: list[str],
    input_path: Path,
    workdir: Path,
    updates: dict[str, str],
    timeout_s: float,
) -> dict[str, Any]:
    local_input = _write_input(input_path, workdir, updates)
    actual_cmd = [part if part != "{input}" else local_input.name for part in cmd]
    run = _run(actual_cmd, cwd=workdir, timeout_s=timeout_s)
    return {
        "case": case,
        "backend": backend,
        "policy": "",
        "runtime_cold_s": run["time_real_s"],
        "runtime_warm_s": None,
        "ok": run["returncode"] == 0,
        "error": None if run["returncode"] == 0 else run["stderr"][-2000:],
        **run,
    }


def _vmec_jax_record(
    *,
    case: str,
    input_path: Path,
    workdir: Path,
    updates: dict[str, str],
    timeout_s: float,
    policy: str,
    warm_repeats: int,
) -> dict[str, Any]:
    local_input = _write_input(input_path, workdir, updates)
    code = r"""
import json
import sys
import time
from pathlib import Path
import jax
from vmec_jax.api import run_fixed_boundary

path = Path(sys.argv[1])
policy = sys.argv[2]
warm_repeats = int(sys.argv[3])
jit_forces = policy == "jit"
times = []
iters = []
converged = []
for _ in range(1 + max(0, warm_repeats)):
    t0 = time.perf_counter()
    run = run_fixed_boundary(
        path,
        solver="vmec2000_iter",
        cli_fixed_boundary_mode=True,
        verbose=False,
        jit_forces=jit_forces,
    )
    times.append(time.perf_counter() - t0)
    result = getattr(run, "result", None)
    diagnostics = {} if result is None else dict(getattr(result, "diagnostics", {}) or {})
    iters.append(None if result is None else int(getattr(result, "n_iter", -1)))
    converged.append(bool(diagnostics.get("converged", False)))
print(json.dumps({
    "runtime_cold_s": float(times[0]),
    "runtime_warm_s": None if len(times) == 1 else float(sum(times[1:]) / (len(times) - 1)),
    "runtime_all_s": [float(x) for x in times],
    "n_iter": iters,
    "converged": converged,
    "jax_disable_jit": bool(jax.config.jax_disable_jit),
    "jax_backend": str(jax.default_backend()),
    "device_kind": str(jax.devices()[0].device_kind) if jax.devices() else "unknown",
}))
"""
    env = dict(os.environ)
    env.update({"PYTHONUNBUFFERED": "1", "VMEC_JAX_SCAN_PRINT": "0", "VMEC_JAX_SCAN_MINIMAL": "1"})
    if policy == "nojit":
        env["JAX_DISABLE_JIT"] = "1"
        env["VMEC_JAX_VMEC2000_FORCE_NOJIT"] = "1"
    else:
        env.pop("JAX_DISABLE_JIT", None)
        env["VMEC_JAX_VMEC2000_FORCE_JIT"] = "1"
    run = _run([sys.executable, "-c", code, str(local_input), policy, str(warm_repeats)], cwd=workdir, timeout_s=timeout_s, env=env)
    payload = _json_tail(run["stdout"])
    return {
        "case": case,
        "backend": "vmec_jax",
        "policy": policy,
        "ok": run["returncode"] == 0 and bool(payload),
        "error": None if run["returncode"] == 0 else run["stderr"][-2000:],
        **payload,
        **run,
    }


def _write_csv(records: list[dict[str, Any]], outpath: Path) -> None:
    fields = ["case", "backend", "policy", "runtime_cold_s", "runtime_warm_s", "peak_memory_gib", "ok"]
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with outpath.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for rec in records:
            mem = _best_mem_bytes(rec)
            writer.writerow(
                {
                    "case": rec["case"],
                    "backend": rec["backend"],
                    "policy": rec.get("policy", ""),
                    "runtime_cold_s": rec.get("runtime_cold_s"),
                    "runtime_warm_s": rec.get("runtime_warm_s"),
                    "peak_memory_gib": None if mem is None else mem / (1024**3),
                    "ok": bool(rec.get("ok", False)),
                }
            )


def _plot(records: list[dict[str, Any]], outpath: Path, *, updates: dict[str, str]) -> None:
    import numpy as np

    cases = sorted({rec["case"] for rec in records})
    by_key = {(rec["case"], rec["backend"], rec.get("policy", "")): rec for rec in records}
    runtime_series = [
        ("VMEC2000", "#1f77b4", ("vmec2000", ""), "runtime_cold_s"),
        ("VMEC++", "#2ca02c", ("vmecpp", ""), "runtime_cold_s"),
        ("vmec_jax JIT cold", "#ff7f0e", ("vmec_jax", "jit"), "runtime_cold_s"),
        ("vmec_jax JIT warm", "#f2a65a", ("vmec_jax", "jit"), "runtime_warm_s"),
        ("vmec_jax no-JIT cold", "#d62728", ("vmec_jax", "nojit"), "runtime_cold_s"),
        ("vmec_jax no-JIT warm", "#ff9896", ("vmec_jax", "nojit"), "runtime_warm_s"),
    ]
    memory_series = [
        ("VMEC2000", "#1f77b4", ("vmec2000", "")),
        ("VMEC++", "#2ca02c", ("vmecpp", "")),
        ("vmec_jax JIT process", "#ff7f0e", ("vmec_jax", "jit")),
        ("vmec_jax no-JIT process", "#d62728", ("vmec_jax", "nojit")),
    ]

    plt = _pyplot()
    fig, axes = plt.subplots(2, 1, figsize=(12.8, 8.0), sharex=True)
    x = np.arange(len(cases), dtype=float)
    for axis, series, width, ylabel in (
        (axes[0], runtime_series, 0.12, "runtime (s, log)"),
        (axes[1], memory_series, 0.17, "peak process memory (GiB)"),
    ):
        offsets = np.linspace(-width * (len(series) - 1) / 2, width * (len(series) - 1) / 2, len(series))
        for item, off in zip(series, offsets):
            label, color, key = item[:3]
            metric = item[3] if len(item) > 3 else None
            values = []
            for case in cases:
                rec = by_key.get((case, *key), {})
                raw = (rec.get(metric) if metric else _best_mem_bytes(rec)) if rec.get("ok") else None
                values.append(None if raw is None else (raw if metric else raw / (1024**3)))
            axis.bar(x + off, values, width=width, color=color, label=label)
        if ylabel.startswith("runtime"):
            axis.set_yscale("log")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", which="both", alpha=0.2)
        axis.legend(frameon=False, ncol=3 if ylabel.startswith("runtime") else 2, fontsize=8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(cases)
    settings = "input-deck budgets" if not updates else ", ".join(f"{k}={v}" for k, v in updates.items())
    fig.suptitle(f"Single-grid fixed-boundary runtime and memory ({settings})", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-dir", type=Path, default=REPO_ROOT / "examples" / "data")
    parser.add_argument("--outdir", type=Path, default=REPO_ROOT / "docs" / "_static" / "figures")
    parser.add_argument("--workdir", type=Path, default=REPO_ROOT / "outputs" / "readme_runtime_memory_single_grid_work")
    parser.add_argument("--reuse-workdir", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    parser.add_argument("--vmec2000-exec", type=Path, default=None)
    parser.add_argument("--warm-repeats", type=int, default=1)
    parser.add_argument("--ns", type=int, default=None)
    parser.add_argument("--niter", type=int, default=None)
    parser.add_argument("--ftol", type=float, default=None)
    parser.add_argument("--nstep", type=int, default=None)
    parser.add_argument("--cases", nargs="+", default=["circular_tokamak", "nfp4_QH_warm_start"])
    args = parser.parse_args()

    workdir = args.workdir.expanduser().resolve()
    results_path = workdir / "results.json"
    if bool(args.reuse_workdir) and results_path.exists():
        payload = json.loads(results_path.read_text())
        records = payload["records"]
    else:
        updates = _updates(args)
        vmec2000 = args.vmec2000_exec.expanduser().resolve() if args.vmec2000_exec else find_vmec2000_exec(root=REPO_ROOT.parent)
        vmecpp = shutil.which("vmecpp")
        if vmec2000 is None:
            raise SystemExit("Missing VMEC2000 executable; pass --vmec2000-exec.")
        if vmecpp is None:
            raise SystemExit("Missing vmecpp executable on PATH.")
        records: list[dict[str, Any]] = []
        for case in args.cases:
            input_path = args.inputs_dir.expanduser().resolve() / f"input.{case}"
            if not input_path.exists():
                raise FileNotFoundError(input_path)
            case_work = workdir / case
            shutil.rmtree(case_work, ignore_errors=True)
            print(f"[readme-runtime] {case}: VMEC2000", flush=True)
            records.append(_external_record(case=case, backend="vmec2000", cmd=[str(vmec2000), "{input}"], input_path=input_path, workdir=case_work / "vmec2000", updates=updates, timeout_s=float(args.timeout_s)))
            print(f"[readme-runtime] {case}: VMEC++", flush=True)
            records.append(_external_record(case=case, backend="vmecpp", cmd=[str(vmecpp), "--legacy", "{input}"], input_path=input_path, workdir=case_work / "vmecpp", updates=updates, timeout_s=float(args.timeout_s)))
            for policy in ("jit", "nojit"):
                print(f"[readme-runtime] {case}: vmec_jax {policy}", flush=True)
                records.append(_vmec_jax_record(case=case, input_path=input_path, workdir=case_work / f"vmec_jax_{policy}", updates=updates, timeout_s=float(args.timeout_s), policy=policy, warm_repeats=int(args.warm_repeats)))
        payload = {
            "metadata": {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "host": platform.node(),
                "platform": platform.platform(),
                "python": platform.python_version(),
                "updates": updates,
                "warm_repeats": int(args.warm_repeats),
                "vmec2000_exec": str(vmec2000),
                "vmecpp_exec": str(vmecpp),
            },
            "records": records,
        }
        workdir.mkdir(parents=True, exist_ok=True)
        results_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    figure = args.outdir.expanduser().resolve() / "readme_runtime_memory_single_grid.png"
    csv_path = args.outdir.expanduser().resolve() / "readme_runtime_memory_single_grid.csv"
    json_path = args.outdir.expanduser().resolve() / "readme_runtime_memory_single_grid.json"
    _plot(records, figure, updates=payload["metadata"].get("updates", {}))
    _write_csv(records, csv_path)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {figure}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
