#!/usr/bin/env python3
"""Run parity/benchmark sweeps from a manifest.

The manifest describes fixed-boundary and free-boundary cases from vmec_jax,
STELLOPT, SIMSOPT, and VMEC++ trees. This runner executes the appropriate
comparator script per case and writes a machine-readable summary.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised on Python 3.10 in CI
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "tools" / "diagnostics" / "parity_manifest.toml"
DEFAULT_VMEC_EXEC = REPO_ROOT.parent / "STELLOPT" / "VMEC2000" / "Release" / "xvmec2000"


def _resolve_path(path_like: str, base_dir: Path) -> Path:
    p = Path(path_like).expanduser()
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def _parse_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with path.open("rb") as f:
        data = tomllib.load(f)
    meta = {
        "version": int(data.get("version", 1)),
        "name": str(data.get("name", "parity_manifest")),
        "vmec2000_default": str(data.get("vmec2000_default", "")),
        "notes": str(data.get("notes", "")),
    }
    raw_cases = data.get("cases", [])
    if not isinstance(raw_cases, list):
        raise ValueError("manifest `cases` must be an array of tables")
    cases: list[dict[str, Any]] = []
    for c in raw_cases:
        if not isinstance(c, dict):
            continue
        cc = dict(c)
        cc["id"] = str(cc.get("id", "")).strip()
        if not cc["id"]:
            continue
        cc["tier"] = str(cc.get("tier", "full")).strip().lower()
        cc["enabled"] = bool(cc.get("enabled", True))
        cc["compare"] = str(cc.get("compare", "stage_trace")).strip().lower()
        cc["input"] = str(cc.get("input", "")).strip()
        if "env" in cc and not isinstance(cc["env"], dict):
            raise ValueError(f"case {cc['id']}: env must be key/value table")
        cases.append(cc)
    return meta, cases


def _evaluate_freeb_thresholds(case: dict[str, Any], runs: list[dict[str, Any]]) -> tuple[bool, dict[str, Any]]:
    """Evaluate free-boundary metric thresholds against run JSON payloads.

    Manifest schema (per case):
      [cases.metric_thresholds_rel_scaled]
      source_sym = 1e-2
      bvec_nonsing_fouri = 1e-2
      amatrix = 2e-1
      potvac = 1.0

      [cases.metric_thresholds_rel_scaled_by_iter."53"]
      source_sym = 1e-2
      bvec_nonsing_fouri = 1e-2
    """

    thr = case.get("metric_thresholds_rel_scaled", {})
    thr_by_iter = case.get("metric_thresholds_rel_scaled_by_iter", {})
    if not isinstance(thr, dict):
        thr = {}
    if not isinstance(thr_by_iter, dict):
        thr_by_iter = {}
    if (not thr) and (not thr_by_iter):
        return True, {}

    observations: dict[str, list[tuple[int | None, float]]] = {}
    for rec in runs:
        payload = rec.get("metrics_full", {})
        if not isinstance(payload, dict):
            continue
        iter_idx = rec.get("iter")
        for key, metric in payload.items():
            if not isinstance(metric, dict) or ("rel_scaled" not in metric):
                continue
            observations.setdefault(str(key), []).append((iter_idx, float(metric["rel_scaled"])))

    report: dict[str, Any] = {}
    all_ok = True
    if thr:
        global_report: dict[str, Any] = {}
        for key, lim in thr.items():
            metric_key = str(key)
            limit = float(lim)
            obs = observations.get(metric_key, [])
            if obs:
                observed_max = float(max(abs(v) for _, v in obs))
                observed_iters = [it for it, _ in obs]
                ok = bool(observed_max <= limit)
            else:
                observed_max = float("nan")
                observed_iters = []
                ok = False
            global_report[metric_key] = {
                "limit_rel_scaled": limit,
                "observed_max_rel_scaled": observed_max,
                "observed_iters": observed_iters,
                "pass": ok,
            }
            if not ok:
                all_ok = False
        report["global"] = global_report

    if thr_by_iter:
        by_iter_report: dict[str, Any] = {}
        for iter_key, metric_limits in thr_by_iter.items():
            if not isinstance(metric_limits, dict):
                continue
            iter_key_str = str(iter_key)
            try:
                iter_target = int(iter_key_str)
            except ValueError:
                all_ok = False
                by_iter_report[iter_key_str] = {"error": "iter key must be integer-like", "pass": False}
                continue
            iter_report: dict[str, Any] = {}
            iter_ok = True
            for metric_key, lim in metric_limits.items():
                metric_name = str(metric_key)
                limit = float(lim)
                obs = observations.get(metric_name, [])
                vals = [abs(v) for it, v in obs if it == iter_target]
                if vals:
                    observed_max = float(max(vals))
                    ok = bool(observed_max <= limit)
                else:
                    observed_max = float("nan")
                    ok = False
                iter_report[metric_name] = {
                    "limit_rel_scaled": limit,
                    "observed_max_rel_scaled": observed_max,
                    "pass": ok,
                }
                if not ok:
                    iter_ok = False
            iter_report["pass"] = iter_ok
            by_iter_report[iter_key_str] = iter_report
            if not iter_ok:
                all_ok = False
        report["by_iter"] = by_iter_report

    return all_ok, report


def _evaluate_runtime_thresholds(case: dict[str, Any], runs: list[dict[str, Any]]) -> tuple[bool, dict[str, Any]]:
    """Evaluate per-run and total runtime thresholds.

    Manifest schema (per case):
      max_runtime_s = 30.0
      max_total_runtime_s = 80.0

      [cases.runtime_thresholds_s_by_iter."53"]
      max_runtime_s = 20.0
    """

    has_any = any(k in case for k in ("max_runtime_s", "max_total_runtime_s", "runtime_thresholds_s_by_iter"))
    if not has_any:
        return True, {}

    report: dict[str, Any] = {}
    all_ok = True
    runtime_vals = [float(r.get("runtime_s", 0.0)) for r in runs if "runtime_s" in r]
    total_runtime = float(sum(runtime_vals))
    report["observed_total_runtime_s"] = total_runtime

    if "max_runtime_s" in case:
        lim = float(case["max_runtime_s"])
        observed_max = float(max(runtime_vals)) if runtime_vals else 0.0
        ok = bool(observed_max <= lim)
        report["max_runtime_s"] = {
            "limit_s": lim,
            "observed_max_s": observed_max,
            "pass": ok,
        }
        if not ok:
            all_ok = False

    if "max_total_runtime_s" in case:
        lim = float(case["max_total_runtime_s"])
        ok = bool(total_runtime <= lim)
        report["max_total_runtime_s"] = {
            "limit_s": lim,
            "observed_s": total_runtime,
            "pass": ok,
        }
        if not ok:
            all_ok = False

    by_iter = case.get("runtime_thresholds_s_by_iter", {})
    if isinstance(by_iter, dict) and by_iter:
        by_iter_report: dict[str, Any] = {}
        for iter_key, rules in by_iter.items():
            iter_key_str = str(iter_key)
            try:
                iter_target = int(iter_key_str)
            except ValueError:
                all_ok = False
                by_iter_report[iter_key_str] = {"error": "iter key must be integer-like", "pass": False}
                continue
            if not isinstance(rules, dict):
                all_ok = False
                by_iter_report[iter_key_str] = {"error": "iter runtime rule must be a table", "pass": False}
                continue
            rec = next((r for r in runs if int(r.get("iter", -1)) == iter_target), None)
            iter_runtime = float(rec.get("runtime_s", float("nan"))) if rec is not None else float("nan")
            iter_ok = True
            iter_report: dict[str, Any] = {"observed_runtime_s": iter_runtime}
            if "max_runtime_s" in rules:
                lim = float(rules["max_runtime_s"])
                ok = bool(math.isfinite(iter_runtime) and iter_runtime <= lim)
                iter_report["max_runtime_s"] = {"limit_s": lim, "observed_s": iter_runtime, "pass": ok}
                if not ok:
                    iter_ok = False
            iter_report["pass"] = iter_ok
            by_iter_report[iter_key_str] = iter_report
            if not iter_ok:
                all_ok = False
        report["by_iter"] = by_iter_report

    report["pass"] = all_ok
    return all_ok, report


def _build_stage_trace_cmd(case: dict[str, Any], *, vmec_exec: Path, workdir: Path) -> list[str]:
    script = REPO_ROOT / "tools" / "diagnostics" / "vmec2000_exec_stage_trace_compare.py"
    cmd = [
        sys.executable,
        str(script),
        "--input",
        str(case["input"]),
        "--vmec2000",
        str(vmec_exec),
        "--max-iter",
        str(int(case.get("max_iter", 10))),
        "--rtol",
        str(float(case.get("rtol", 1e-3))),
        "--atol",
        str(float(case.get("atol", 1e-10))),
        "--dump-level",
        str(case.get("dump_level", "lite")),
        "--vmec-timeout",
        str(float(case.get("vmec_timeout", 120.0))),
        "--workdir",
        str(workdir),
    ]
    if bool(case.get("use_input_niter", False)):
        cmd.append("--use-input-niter")
    if "single_ns" in case and case["single_ns"] not in (None, ""):
        cmd.extend(["--single-ns", str(int(case["single_ns"]))])
    if "vmec_nstep" in case and case["vmec_nstep"] not in (None, ""):
        cmd.extend(["--vmec-nstep", str(int(case["vmec_nstep"]))])
    if "ns_array" in case and case["ns_array"] not in (None, ""):
        v = case["ns_array"]
        if isinstance(v, list):
            v = ",".join(str(x) for x in v)
        cmd.extend(["--ns-array", str(v)])
    if "niter_array" in case and case["niter_array"] not in (None, ""):
        v = case["niter_array"]
        if isinstance(v, list):
            v = ",".join(str(x) for x in v)
        cmd.extend(["--niter-array", str(v)])
    if "ftol_array" in case and case["ftol_array"] not in (None, ""):
        v = case["ftol_array"]
        if isinstance(v, list):
            v = ",".join(str(x) for x in v)
        cmd.extend(["--ftol-array", str(v)])
    return cmd


def _build_freeb_scalpot_cmd(
    case: dict[str, Any],
    *,
    vmec_exec: Path,
    iter_idx: int,
    max_iter: int,
    workdir: Path,
    json_path: Path,
) -> list[str]:
    script = REPO_ROOT / "tools" / "diagnostics" / "vmec2000_exec_freeb_scalpot_compare.py"
    return [
        sys.executable,
        str(script),
        "--input",
        str(case["input"]),
        "--vmec-exec",
        str(vmec_exec),
        "--iter",
        str(int(iter_idx)),
        "--max-iter",
        str(int(max_iter)),
        "--workdir",
        str(workdir),
        "--json",
        str(json_path),
    ]


def _run_cmd(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    stdout_path: Path,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True, check=False)
    dt = time.perf_counter() - t0
    stdout_path.write_text((proc.stdout or "") + "\n" + (proc.stderr or ""), encoding="utf-8")
    return {
        "returncode": int(proc.returncode),
        "runtime_s": float(dt),
        "stdout_path": str(stdout_path),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--vmec-exec", type=Path, default=DEFAULT_VMEC_EXEC)
    p.add_argument("--tier", type=str, default="smoke", help="Case tier filter: smoke|full|planning|all")
    p.add_argument("--ids", type=str, default="", help="Comma-separated case ids (overrides tier filter).")
    p.add_argument("--max-cases", type=int, default=0, help="Optional hard limit after filtering (0=all).")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--output-root", type=Path, default=REPO_ROOT / "outputs" / "parity_sweeps")
    args = p.parse_args()

    manifest_path = args.manifest.resolve()
    if not manifest_path.exists():
        raise SystemExit(f"missing manifest: {manifest_path}")
    meta, cases = _parse_manifest(manifest_path)

    vmec_exec = args.vmec_exec.resolve()
    if (not args.dry_run) and (not vmec_exec.exists()):
        raise SystemExit(f"missing VMEC2000 executable: {vmec_exec}")

    selected_ids = [x.strip() for x in str(args.ids).split(",") if x.strip()]
    if selected_ids:
        picked = [c for c in cases if c["id"] in selected_ids]
    else:
        tier = str(args.tier).strip().lower()
        if tier == "all":
            picked = list(cases)
        else:
            picked = [c for c in cases if str(c.get("tier", "")).lower() == tier]
    picked = [c for c in picked if bool(c.get("enabled", True))]
    if args.max_cases and int(args.max_cases) > 0:
        picked = picked[: int(args.max_cases)]

    if not picked:
        raise SystemExit("no cases selected")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = args.output_root.resolve() / stamp
    out_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "manifest": str(manifest_path),
        "manifest_name": meta["name"],
        "manifest_version": meta["version"],
        "vmec_exec": str(vmec_exec),
        "timestamp": stamp,
        "tier": str(args.tier),
        "ids": selected_ids,
        "cases": [],
    }

    print(f"manifest={manifest_path}")
    print(f"vmec_exec={vmec_exec}")
    print(f"selected_cases={len(picked)}")

    n_fail = 0
    for i, case in enumerate(picked, start=1):
        case_id = str(case["id"])
        compare = str(case.get("compare", "stage_trace")).lower()
        case_dir = out_root / case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        input_path = _resolve_path(str(case["input"]), REPO_ROOT)
        if not input_path.exists():
            print(f"[{i}/{len(picked)}] SKIP {case_id} missing_input={input_path}")
            n_fail += 1
            summary["cases"].append(
                {"id": case_id, "status": "missing_input", "input": str(input_path), "compare": compare}
            )
            continue

        base_env = os.environ.copy()
        for k, v in dict(case.get("env", {})).items():
            base_env[str(k)] = str(v)

        print(f"[{i}/{len(picked)}] RUN  {case_id} ({compare})")
        case_rec: dict[str, Any] = {
            "id": case_id,
            "compare": compare,
            "input": str(input_path),
            "tier": str(case.get("tier", "")),
            "goal": str(case.get("goal", "")),
            "runs": [],
        }

        if compare == "stage_trace":
            run_dir = case_dir / "stage_trace"
            run_dir.mkdir(parents=True, exist_ok=True)
            case_workdir = run_dir / "workdir"
            cmd_case = dict(case)
            cmd_case["input"] = str(input_path)
            cmd = _build_stage_trace_cmd(cmd_case, vmec_exec=vmec_exec, workdir=case_workdir)
            case_rec["cmd"] = cmd
            if args.dry_run:
                print("  DRY-RUN:", " ".join(cmd))
                rc = 0
                rec = {"returncode": 0, "runtime_s": 0.0, "stdout_path": ""}
            else:
                rec = _run_cmd(cmd, cwd=REPO_ROOT, env=base_env, stdout_path=run_dir / "stdout.log")
                rc = int(rec["returncode"])
                print(f"  rc={rc} runtime={rec['runtime_s']:.2f}s")
            case_rec["runs"].append(rec)
            runtime_ok, runtime_report = _evaluate_runtime_thresholds(case, case_rec["runs"])
            if runtime_report:
                case_rec["runtime_thresholds_s"] = runtime_report
            case_ok = bool((rc == 0) and runtime_ok)
            case_rec["status"] = "pass" if case_ok else "fail"
            if not case_ok:
                n_fail += 1

        elif compare == "freeb_scalpot":
            iter_list = case.get("iter_list", [53])
            if not isinstance(iter_list, list):
                iter_list = [iter_list]
            max_iter = int(case.get("max_iter", max(int(x) for x in iter_list)))
            all_ok = True
            for iter_idx in [int(x) for x in iter_list]:
                run_dir = case_dir / f"iter_{iter_idx}"
                run_dir.mkdir(parents=True, exist_ok=True)
                case_workdir = run_dir / "workdir"
                cmd_case = dict(case)
                cmd_case["input"] = str(input_path)
                json_path = run_dir / f"summary_iter{iter_idx}.json"
                cmd = _build_freeb_scalpot_cmd(
                    cmd_case,
                    vmec_exec=vmec_exec,
                    iter_idx=iter_idx,
                    max_iter=max_iter,
                    workdir=case_workdir,
                    json_path=json_path,
                )
                rec: dict[str, Any] = {"iter": int(iter_idx), "cmd": cmd}
                if args.dry_run:
                    print("  DRY-RUN:", " ".join(cmd))
                    rec.update({"returncode": 0, "runtime_s": 0.0, "stdout_path": "", "json_path": str(json_path)})
                    rc = 0
                else:
                    run_rec = _run_cmd(cmd, cwd=REPO_ROOT, env=base_env, stdout_path=run_dir / "stdout.log")
                    rec.update(run_rec)
                    rec["json_path"] = str(json_path)
                    rc = int(run_rec["returncode"])
                    if json_path.exists():
                        try:
                            payload = json.loads(json_path.read_text(encoding="utf-8"))
                        except Exception:
                            payload = {}
                        # Keep a compact metric excerpt in the sweep summary.
                        metric_excerpt: dict[str, Any] = {}
                        for key in ("source_sym", "gsource_full", "bvec_nonsing_fouri", "potvac", "amatrix"):
                            if key in payload:
                                metric_excerpt[key] = payload[key]
                        rec["metrics"] = metric_excerpt
                        rec["metrics_full"] = payload
                    print(f"  iter={iter_idx} rc={rc} runtime={rec['runtime_s']:.2f}s")
                if rc != 0:
                    all_ok = False
                case_rec["runs"].append(rec)
            if args.dry_run:
                # Dry-runs validate command construction only; no scalpot JSON
                # exists yet, so metric-threshold evaluation would otherwise
                # fail every free-boundary case for missing observations.
                thresholds_ok, thresholds_report = True, {}
            else:
                thresholds_ok, thresholds_report = _evaluate_freeb_thresholds(case, case_rec["runs"])
            runtime_ok, runtime_report = _evaluate_runtime_thresholds(case, case_rec["runs"])
            if thresholds_report:
                case_rec["metric_thresholds_rel_scaled"] = thresholds_report
            if runtime_report:
                case_rec["runtime_thresholds_s"] = runtime_report
            case_ok = bool(all_ok and thresholds_ok and runtime_ok)
            case_rec["status"] = "pass" if case_ok else "fail"
            if not case_ok:
                n_fail += 1
        else:
            print(f"  SKIP unsupported compare mode: {compare}")
            case_rec["status"] = "unsupported_compare"
            n_fail += 1

        summary["cases"].append(case_rec)

    summary["failed_cases"] = int(n_fail)
    summary["selected_case_count"] = len(picked)
    out_json = out_root / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"summary={out_json}")
    print(f"failed_cases={n_fail}")
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
