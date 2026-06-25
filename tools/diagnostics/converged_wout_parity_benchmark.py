#!/usr/bin/env python3
"""Regenerate converged-wout VMEC2000 parity benchmark summaries.

This runner executes representative converged cases with one or more VMEC2000
executables, optionally runs vmec_jax on the same patched input, and writes a
compact JSON summary of residuals, runtimes, and field relative-RMS errors.
Slow non-circular cases are opt-in via ``--nightly``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vmec_jax.vmec2000_exec import _patch_indata, find_vmec2000_exec, run_xvmec2000
from vmec_jax.wout import read_wout, state_from_wout, wout_minimal_from_fixed_boundary


@dataclass(frozen=True)
class BenchmarkCase:
    case: str
    input_relpath: str
    updates: dict[str, str]
    lfreeb: bool
    axisymmetric: bool
    lasym: bool
    multigrid: bool
    mgrid_relpath: str | None = None
    nightly: bool = False
    timeout_s: float = 120.0


BENCHMARK_CASES = (
    BenchmarkCase(
        case="circular_tokamak",
        input_relpath="examples/data/input.circular_tokamak",
        updates={
            "NITER": "300",
            "NS_ARRAY": "13",
            "NITER_ARRAY": "300",
            "FTOL_ARRAY": "1e-10",
            "NSTEP": "50",
        },
        lfreeb=False,
        axisymmetric=True,
        lasym=False,
        multigrid=False,
    ),
    BenchmarkCase(
        case="LandremanPaul2021_QA_lowres",
        input_relpath="examples/data/input.LandremanPaul2021_QA_lowres",
        updates={
            "NITER": "1000",
            "NS_ARRAY": "16, 31, 50",
            "NITER_ARRAY": "600, 1000, 1000",
            "FTOL_ARRAY": "1e-10, 1e-10, 1e-10",
            "NSTEP": "50",
        },
        lfreeb=False,
        axisymmetric=False,
        lasym=False,
        multigrid=True,
        nightly=True,
        timeout_s=240.0,
    ),
    BenchmarkCase(
        case="nfp4_QH_warm_start",
        input_relpath="examples/data/input.nfp4_QH_warm_start",
        updates={
            "NITER": "1500",
            "NS_ARRAY": "35",
            "NITER_ARRAY": "1500",
            "FTOL_ARRAY": "1e-13",
            "NSTEP": "200",
        },
        lfreeb=False,
        axisymmetric=False,
        lasym=False,
        multigrid=False,
        nightly=True,
        timeout_s=180.0,
    ),
    BenchmarkCase(
        case="solovev",
        input_relpath="examples/data/input.solovev",
        updates={
            "NITER": "500",
            "NS_ARRAY": "11",
            "NITER_ARRAY": "500",
            "FTOL_ARRAY": "1e-14",
            "NSTEP": "250",
        },
        lfreeb=False,
        axisymmetric=True,
        lasym=False,
        multigrid=False,
        nightly=True,
        timeout_s=120.0,
    ),
    BenchmarkCase(
        case="ITERModel",
        input_relpath="examples/data/input.ITERModel",
        updates={
            "NITER": "1000",
            "NS_ARRAY": "13",
            "NITER_ARRAY": "1000",
            "FTOL_ARRAY": "1e-14",
            "NSTEP": "200",
        },
        lfreeb=False,
        axisymmetric=True,
        lasym=False,
        multigrid=False,
        nightly=True,
        timeout_s=180.0,
    ),
    BenchmarkCase(
        case="up_down_asymmetric_tokamak",
        input_relpath="examples_single_grid/data/input.up_down_asymmetric_tokamak",
        updates={
            "NITER": "800",
            "NS_ARRAY": "17",
            "NITER_ARRAY": "800",
            "FTOL_ARRAY": "1e-10",
            "NSTEP": "50",
        },
        lfreeb=False,
        axisymmetric=True,
        lasym=True,
        multigrid=False,
        nightly=True,
        timeout_s=180.0,
    ),
    BenchmarkCase(
        case="basic_non_stellsym_pressure",
        input_relpath="examples_single_grid/data/input.basic_non_stellsym_pressure",
        updates={
            "NITER": "1200",
            "NS_ARRAY": "25",
            "NITER_ARRAY": "1200",
            "FTOL_ARRAY": "1e-10",
            "NSTEP": "50",
        },
        lfreeb=False,
        axisymmetric=False,
        lasym=True,
        multigrid=False,
        nightly=True,
        timeout_s=240.0,
    ),
    BenchmarkCase(
        case="cth_like_free_bdy",
        input_relpath="examples_single_grid/data/input.cth_like_free_bdy",
        updates={
            "NITER": "5000",
            "NITER_ARRAY": "5000",
            "FTOL_ARRAY": "1e-10",
        },
        lfreeb=True,
        axisymmetric=False,
        lasym=False,
        multigrid=False,
        mgrid_relpath="examples_single_grid/data/mgrid_cth_like.nc",
        nightly=True,
        timeout_s=600.0,
    ),
)


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    for p in paths:
        try:
            key = p.expanduser().resolve()
        except FileNotFoundError:
            key = p.expanduser()
        if key in seen or not key.exists():
            continue
        seen.add(key)
        out.append(key)
    return out


def _local_vmec2000_execs(root: Path) -> list[Path]:
    skip_dirs = {".git", ".venv", "__pycache__", "node_modules", "site-packages", "dist", "build"}
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs and not d.startswith(".")]
        if "xvmec2000" not in filenames:
            continue
        p = Path(dirpath) / "xvmec2000"
        if os.access(p, os.X_OK):
            found.append(p)
    return found


def discover_vmec2000_execs(*, scan_local: bool = False) -> list[Path]:
    paths: list[Path] = []
    if os.environ.get("VMEC2000_EXEC"):
        paths.append(Path(os.environ["VMEC2000_EXEC"]).expanduser())
    paths.append(Path.home() / "bin" / "xvmec2000")
    found = find_vmec2000_exec()
    if found is not None:
        paths.append(found)
    which = shutil.which("xvmec2000")
    if which:
        paths.append(Path(which))
    paths.extend(
        [
            REPO_ROOT.parent / "STELLOPT" / "VMEC2000" / "Release" / "xvmec2000",
            REPO_ROOT / "vmec2000" / "build" / "xvmec2000",
            REPO_ROOT / "vmec2000" / "build" / "Release" / "xvmec2000",
        ]
    )
    if scan_local:
        paths.extend(_local_vmec2000_execs(REPO_ROOT.parent))
    return _dedupe_paths(paths)


def _write_patched_input(case: BenchmarkCase, dst: Path) -> Path:
    src = REPO_ROOT / case.input_relpath
    if not src.exists():
        raise FileNotFoundError(src)
    updates = dict(case.updates)
    if case.mgrid_relpath is not None:
        mgrid = REPO_ROOT / case.mgrid_relpath
        if not mgrid.exists():
            raise FileNotFoundError(mgrid)
        updates["MGRID_FILE"] = f"'{mgrid}'"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(_patch_indata(src.read_text(), updates=updates), encoding="utf-8")
    return dst


def _rel_rms(got: Any, ref: Any, *, radial_skip: int = 0) -> float:
    got_arr = np.asarray(got, dtype=float)
    ref_arr = np.asarray(ref, dtype=float)
    if got_arr.shape != ref_arr.shape:
        return float("nan")
    if got_arr.ndim >= 1 and radial_skip:
        got_arr = got_arr[radial_skip:, ...]
        ref_arr = ref_arr[radial_skip:, ...]
    if got_arr.size == 0 or (not np.isfinite(got_arr).all()) or (not np.isfinite(ref_arr).all()):
        return float("nan")
    diff_rms = float(np.sqrt(np.mean((got_arr - ref_arr) ** 2)))
    ref_rms = float(np.sqrt(np.mean(ref_arr**2)))
    return diff_rms / ref_rms if ref_rms > 0.0 else diff_rms


def _wout_metrics(wout) -> dict[str, Any]:
    residuals = np.asarray([wout.fsqr, wout.fsqz, wout.fsql], dtype=float)
    return {
        "ns": int(wout.ns),
        "mpol": int(wout.mpol),
        "ntor": int(wout.ntor),
        "nfp": int(wout.nfp),
        "lasym": bool(wout.lasym),
        "fsq_rss": float(np.linalg.norm(residuals)),
        "wb": float(wout.wb),
        "wp": float(wout.wp),
        "volume_p": float(wout.volume_p),
        "aspect": float(wout.aspect),
    }


def _comparison_fields(*, lasym: bool) -> dict[str, int]:
    fields = {
        "rmnc": 0,
        "zmns": 0,
        "lmns": 0,
        "phipf": 1,
        "chipf": 1,
        "iotas": 1,
        "iotaf": 1,
        "pres": 1,
        "presf": 1,
        "gmnc": 1,
        "bmnc": 1,
        "bsupumnc": 1,
        "bsupvmnc": 1,
        "bsubumnc": 1,
        "bsubvmnc": 1,
    }
    if lasym:
        fields.update(
            {
                "rmns": 0,
                "zmnc": 0,
                "gmns": 1,
                "bmns": 1,
                "bsupumns": 1,
                "bsupvmns": 1,
                "bsubumns": 1,
                "bsubvmns": 1,
            }
        )
    return fields


def _compare_wouts(got, ref, *, lasym: bool) -> dict[str, float]:
    return {
        name: _rel_rms(getattr(got, name), getattr(ref, name), radial_skip=skip)
        for name, skip in _comparison_fields(lasym=lasym).items()
    }


def _mode_numbers_for_field(wout, field_shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray] | None:
    if len(field_shape) < 2:
        return None
    n_modes = int(field_shape[1])
    xm = np.asarray(getattr(wout, "xm", []), dtype=int)
    xn = np.asarray(getattr(wout, "xn", []), dtype=int)
    if n_modes == xm.size:
        return xm, xn
    xm_nyq = np.asarray(getattr(wout, "xm_nyq", []), dtype=int)
    xn_nyq = np.asarray(getattr(wout, "xn_nyq", []), dtype=int)
    if n_modes == xm_nyq.size:
        return xm_nyq, xn_nyq
    return None


def _field_mode_hotspots(got, ref, name: str, *, radial_skip: int = 0, top_n: int = 5) -> list[dict[str, Any]]:
    got_arr = np.asarray(getattr(got, name), dtype=float)
    ref_arr = np.asarray(getattr(ref, name), dtype=float)
    if got_arr.shape != ref_arr.shape:
        return []
    modes = _mode_numbers_for_field(got, got_arr.shape)
    if modes is None:
        return []
    xm, xn = modes
    if radial_skip and got_arr.ndim >= 1:
        got_arr = got_arr[radial_skip:, ...]
        ref_arr = ref_arr[radial_skip:, ...]
    if got_arr.size == 0 or (not np.isfinite(got_arr).all()) or (not np.isfinite(ref_arr).all()):
        return []

    rows: list[dict[str, Any]] = []
    for idx, (m, n) in enumerate(zip(xm, xn, strict=True)):
        got_col = got_arr[:, idx]
        ref_col = ref_arr[:, idx]
        diff = got_col - ref_col
        diff_rms = float(np.sqrt(np.mean(diff * diff)))
        ref_rms = float(np.sqrt(np.mean(ref_col * ref_col)))
        rows.append(
            {
                "index": int(idx),
                "m": int(m),
                "n": int(n),
                "rel_rms": diff_rms / ref_rms if ref_rms > 0.0 else diff_rms,
                "diff_rms": diff_rms,
                "ref_rms": ref_rms,
                "max_abs_diff": float(np.max(np.abs(diff))),
            }
        )
    rows.sort(key=lambda row: (float(row["rel_rms"]), float(row["diff_rms"])), reverse=True)
    return rows[: int(top_n)]


def _compare_wout_mode_hotspots(got, ref, *, lasym: bool, top_n: int = 5) -> dict[str, list[dict[str, Any]]]:
    return {
        name: _field_mode_hotspots(got, ref, name, radial_skip=skip, top_n=top_n)
        for name, skip in _comparison_fields(lasym=lasym).items()
    }


def _reference_state_roundtrip_rel_rms(ref, input_path: Path, *, lasym: bool) -> dict[str, float]:
    """Rebuild a wout from the reference state to isolate output-path drift."""
    from vmec_jax.config import config_from_indata
    from vmec_jax.namelist import read_indata
    from vmec_jax.static import build_static

    indata = read_indata(input_path)
    static = build_static(config_from_indata(indata))
    state = state_from_wout(ref)
    rebuilt = wout_minimal_from_fixed_boundary(
        path="reference_state_roundtrip",
        state=state,
        static=static,
        indata=indata,
        signgs=int(getattr(ref, "signgs", 1)),
        fsqr=float(getattr(ref, "fsqr", 0.0)),
        fsqz=float(getattr(ref, "fsqz", 0.0)),
        fsql=float(getattr(ref, "fsql", 0.0)),
        fsqt=np.asarray(getattr(ref, "fsqt", np.zeros((1,), dtype=float)), dtype=float),
        converged=True,
    )
    return _compare_wouts(rebuilt, ref, lasym=lasym)


def _run_vmec_jax(case: BenchmarkCase, input_path: Path, out_path: Path):
    from vmec_jax.driver import run_fixed_boundary, run_free_boundary, write_wout_from_fixed_boundary_run

    run_fn = run_free_boundary if case.lfreeb else run_fixed_boundary
    run = run_fn(
        str(input_path),
        solver="vmec2000_iter",
        solver_mode="parity",
        multigrid_use_input_niter=True,
        verbose=False,
        jit_forces=False,
    )
    return write_wout_from_fixed_boundary_run(str(out_path), run)


def _selected_cases(case_ids: set[str], *, include_nightly: bool) -> list[BenchmarkCase]:
    out = []
    for case in BENCHMARK_CASES:
        if case_ids and case.case not in case_ids:
            continue
        if case.nightly and not include_nightly:
            continue
        out.append(case)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", default=[], help="Case id to run; may be repeated. Default: non-nightly.")
    parser.add_argument("--nightly", action="store_true", help="Include slow converged-wout cases.")
    parser.add_argument("--vmec-exec", action="append", default=[], help="VMEC2000 executable path; may be repeated.")
    parser.add_argument("--all-discovered-execs", action="store_true", help="Run every discovered xvmec2000 executable.")
    parser.add_argument(
        "--scan-local-execs",
        action="store_true",
        help="Also scan the parent checkout tree recursively for executable files named xvmec2000.",
    )
    parser.add_argument("--skip-vmec-jax", action="store_true", help="Only run VMEC2000 and record VMEC2000 wout metrics.")
    parser.add_argument("--dry-run", action="store_true", help="Print selected executables/cases without running.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "converged_wout_parity")
    args = parser.parse_args()

    requested_execs = [Path(p).expanduser() for p in args.vmec_exec]
    discovered = discover_vmec2000_execs(scan_local=bool(args.scan_local_execs))
    execs = _dedupe_paths(requested_execs) if requested_execs else discovered
    if execs and not args.all_discovered_execs:
        execs = execs[:1]
    if not execs:
        raise SystemExit("No VMEC2000 executable found. Pass --vmec-exec or set VMEC2000_EXEC.")

    cases = _selected_cases(set(args.case), include_nightly=bool(args.nightly))
    if not cases:
        raise SystemExit("No cases selected.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "executables": [str(p) for p in execs],
        "cases": [],
        "skip_vmec_jax": bool(args.skip_vmec_jax),
    }

    print("executables:")
    for exe in execs:
        print(f"  {exe}")
    print("cases:")
    for case in cases:
        print(f"  {case.case}")
    if args.dry_run:
        return 0

    failed = 0
    for exe in execs:
        exe_label = exe.parent.name if exe.name == "xvmec2000" else exe.stem
        for case in cases:
            case_dir = args.output_dir / exe_label / case.case
            input_path = _write_patched_input(case, case_dir / "input" / Path(case.input_relpath).name)
            rec: dict[str, Any] = {
                "case": case.case,
                "vmec_exec": str(exe),
                "input": str(input_path),
                "lfreeb": case.lfreeb,
                "axisymmetric": case.axisymmetric,
                "lasym": case.lasym,
                "multigrid": case.multigrid,
            }
            try:
                vmec = run_xvmec2000(
                    input_path,
                    exec_path=exe,
                    workdir=case_dir / "vmec2000",
                    timeout_s=case.timeout_s,
                    keep_workdir=True,
                )
                wout_vmec = vmec.workdir / f"wout_{case.case}.nc"
                rec["vmec2000_runtime_s"] = vmec.runtime_s
                rec["vmec2000_stdout_tail"] = (vmec.stdout + "\n" + vmec.stderr).splitlines()[-20:]
                if not wout_vmec.exists():
                    raise FileNotFoundError(wout_vmec)
                wref = read_wout(wout_vmec)
                rec["vmec2000"] = _wout_metrics(wref)
                if not case.lfreeb:
                    rec["reference_state_roundtrip_rel_rms"] = _reference_state_roundtrip_rel_rms(
                        wref,
                        input_path,
                        lasym=case.lasym,
                    )

                if not args.skip_vmec_jax:
                    wout_jax_path = case_dir / f"wout_{case.case}_vmec_jax.nc"
                    _run_vmec_jax(case, input_path, wout_jax_path)
                    wjax = read_wout(wout_jax_path)
                    rec["vmec_jax"] = _wout_metrics(wjax)
                    rec["rel_rms"] = _compare_wouts(wjax, wref, lasym=case.lasym)
                    rec["mode_hotspots"] = _compare_wout_mode_hotspots(wjax, wref, lasym=case.lasym)
                rec["status"] = "pass"
            except Exception as exc:
                rec["status"] = "fail"
                rec["error"] = repr(exc)
                failed += 1
            summary["cases"].append(rec)
            print(f"{rec['status']:>4s} {exe} {case.case}")

    summary["failed_cases"] = failed
    out_json = args.output_dir / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"summary={out_json}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
