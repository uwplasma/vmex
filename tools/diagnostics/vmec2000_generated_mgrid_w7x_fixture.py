#!/usr/bin/env python
"""Generate and run a bounded VMEC2000 free-boundary W7-X mgrid fixture.

This diagnostic is intentionally optional and local.  It uses SIMSOPT's W7-X
configuration data to generate a VMEC-compatible ``mgrid`` at runtime, patches
the SIMSOPT W7-X fixed-boundary input into a low-resolution free-boundary deck,
then runs raw ``xvmec2000``.  The promotion gate is conservative:

* VMEC2000 must reach active vacuum coupling,
* VMEC2000 must write a parseable WOUT,
* residuals must be finite and below the requested bound,
* geometry scalars must be finite and strictly positive.

Generated ``mgrid`` and WOUT files stay in the chosen work directory and are
not committed to the repository.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
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

from vmec_jax.namelist import read_indata, write_indata
from vmec_jax.vmec2000_exec import Vmec2000ExecResult, find_vmec2000_exec, run_xvmec2000
from vmec_jax.wout import read_wout


DEFAULT_OUT = REPO_ROOT / "results" / "vmec2000_w7x_generated_mgrid_fixture.json"
DEFAULT_WORKDIR = REPO_ROOT / "results" / "vmec2000_w7x_generated_mgrid_fixture"
DEFAULT_W7X_INPUT_NAME = "input.W7-X_standard_configuration"
DEFAULT_MGRID_NAME = "mgrid.w7x.nc"
DEFAULT_CASE = "w7x_generated_mgrid_freeb"
DEFAULT_FINAL_FSQ_LIMIT = 1.0e-8
DEFAULT_REFERENCE = {
    "volume_p": 28.6017247168422,
    "rmnc_axis_00": 5.561878306096512,
}


def _jsonify(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonify(value.tolist())
    if isinstance(value, np.generic):
        return _jsonify(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (bool, int, str)) or value is None:
        return value
    return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonify(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _find_simsopt_w7x_input() -> Path:
    """Find the SIMSOPT W7-X VMEC input deck used by the documented example."""
    candidates: list[Path] = []
    if os.getenv("SIMSOPT_W7X_INPUT"):
        candidates.append(Path(os.environ["SIMSOPT_W7X_INPUT"]).expanduser())
    try:
        import simsopt

        simsopt_root = Path(simsopt.__file__).resolve().parents[2]
        candidates.append(simsopt_root / "tests" / "test_files" / DEFAULT_W7X_INPUT_NAME)
    except Exception:
        pass
    candidates.extend(
        [
            REPO_ROOT.parent / "simsopt" / "tests" / "test_files" / DEFAULT_W7X_INPUT_NAME,
            REPO_ROOT.parent / "simsopt_rjorg" / "tests" / "test_files" / DEFAULT_W7X_INPUT_NAME,
            Path("/Users/rogeriojorge/local/simsopt/tests/test_files") / DEFAULT_W7X_INPUT_NAME,
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    searched = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Could not find {DEFAULT_W7X_INPUT_NAME}. Set SIMSOPT_W7X_INPUT or install SIMSOPT with tests. "
        f"Searched:\n  {searched}"
    )


def _generate_w7x_mgrid(path: Path, *, nr: int, nz: int, nphi: int) -> dict[str, Any]:
    """Generate a W7-X mgrid from SIMSOPT coil data."""
    from simsopt.configs import get_data

    _base_curves, _base_currents, _magnetic_axis, nfp, bs = get_data("w7x")
    path.parent.mkdir(parents=True, exist_ok=True)
    bs.to_mgrid(
        str(path),
        nr=int(nr),
        nz=int(nz),
        nphi=int(nphi),
        rmin=4.5,
        rmax=6.3,
        zmin=-1.0,
        zmax=1.0,
        nfp=int(nfp),
    )
    return {
        "path": path,
        "size_bytes": path.stat().st_size,
        "nr": int(nr),
        "nz": int(nz),
        "nphi": int(nphi),
        "nfp": int(nfp),
        "rmin": 4.5,
        "rmax": 6.3,
        "zmin": -1.0,
        "zmax": 1.0,
    }


def write_w7x_generated_mgrid_input(
    path: Path,
    *,
    source_input: Path,
    mgrid_file: str,
    nphi: int = 24,
    mpol: int = 6,
    ntor: int = 6,
    ns_array: tuple[int, ...] = (13, 25),
    ftol_array: tuple[float, ...] = (1.0e-7, 1.0e-10),
    niter_array: tuple[int, ...] = (15000, 15000),
) -> Path:
    """Patch the SIMSOPT W7-X fixed-boundary deck into a VMEC2000 free-boundary deck."""
    if not (len(ns_array) == len(ftol_array) == len(niter_array)):
        raise ValueError("ns_array, ftol_array, and niter_array must have matching lengths")

    indata = deepcopy(read_indata(source_input))
    indata.scalars.update(
        {
            "LFREEB": True,
            "MGRID_FILE": str(mgrid_file),
            "EXTCUR": [1.0],
            "NZETA": int(nphi),
            "NTHETA": 0,
            "MPOL": int(mpol),
            "NTOR": int(ntor),
            "NS_ARRAY": [int(value) for value in ns_array],
            "FTOL_ARRAY": [float(value) for value in ftol_array],
            "NITER_ARRAY": [int(value) for value in niter_array],
            "NITER": int(max(niter_array)),
            "FTOL": float(ftol_array[-1]),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    write_indata(path, indata)
    return path


def _wout_path_for_input(run: Vmec2000ExecResult) -> Path:
    name = run.input_path.name
    case = name[len("input.") :] if name.startswith("input.") else name
    return run.workdir / f"wout_{case}.nc"


def _active_vacuum_evidence(run: Vmec2000ExecResult) -> dict[str, Any]:
    stdout_active = "VACUUM PRESSURE TURNED ON" in str(run.stdout)
    delbsq_values = [
        float(row.delbsq)
        for stage in run.stages
        for row in stage.rows
        if row.delbsq is not None and np.isfinite(float(row.delbsq))
    ]
    return {
        "active": bool(stdout_active or delbsq_values),
        "stdout_vacuum_pressure_turned_on": bool(stdout_active),
        "delbsq_row_count": len(delbsq_values),
        "last_delbsq": float(delbsq_values[-1]) if delbsq_values else None,
    }


def _quality_from_wout(wout: Any, *, final_fsq_limit: float) -> dict[str, Any]:
    residuals = {
        "fsqr": float(wout.fsqr),
        "fsqz": float(wout.fsqz),
        "fsql": float(wout.fsql),
    }
    fsq_total = float(sum(residuals.values()))
    geometry = {
        "aspect": float(wout.aspect),
        "volume_p": float(wout.volume_p),
        "Rmajor_p": float(wout.Rmajor_p),
        "Aminor_p": float(wout.Aminor_p),
    }
    bad_geometry = [
        name
        for name, value in geometry.items()
        if (not np.isfinite(float(value))) or float(value) <= 0.0
    ]
    residual_ok = bool(np.isfinite(fsq_total) and fsq_total <= float(final_fsq_limit))
    return {
        "promotable": bool(residual_ok and not bad_geometry),
        "residual_ok": residual_ok,
        "final_fsq_limit": float(final_fsq_limit),
        "residuals": residuals,
        "fsq_total": fsq_total,
        "geometry_positive": not bad_geometry,
        "bad_geometry": bad_geometry,
        "geometry": geometry,
        "reference_gaps": {
            "volume_p_rel": abs(float(wout.volume_p) - DEFAULT_REFERENCE["volume_p"])
            / abs(DEFAULT_REFERENCE["volume_p"]),
            "rmnc_axis_00_rel": abs(float(wout.rmnc[0, 0]) - DEFAULT_REFERENCE["rmnc_axis_00"])
            / abs(DEFAULT_REFERENCE["rmnc_axis_00"]),
        },
        "layout": {
            "ns": int(wout.ns),
            "mpol": int(wout.mpol),
            "ntor": int(wout.ntor),
            "nfp": int(wout.nfp),
            "ier_flag": int(wout.ier_flag),
        },
    }


def run_w7x_generated_mgrid_vmec2000_fixture(
    *,
    workdir: Path,
    vmec2000_exec: Path | None = None,
    timeout_s: float = 240.0,
    nr: int = 64,
    nz: int = 65,
    nphi: int = 24,
    final_fsq_limit: float = DEFAULT_FINAL_FSQ_LIMIT,
) -> dict[str, Any]:
    """Generate the W7-X mgrid fixture, run VMEC2000, and return a report."""
    workdir = Path(workdir).expanduser().resolve()
    if workdir.exists():
        shutil.rmtree(workdir)
    input_dir = workdir / "inputs"
    run_dir = workdir / "vmec2000"
    input_dir.mkdir(parents=True)
    run_dir.mkdir(parents=True)

    source_input = _find_simsopt_w7x_input()
    mgrid_path = input_dir / DEFAULT_MGRID_NAME
    mgrid_summary = _generate_w7x_mgrid(mgrid_path, nr=nr, nz=nz, nphi=nphi)
    input_path = write_w7x_generated_mgrid_input(
        input_dir / f"input.{DEFAULT_CASE}",
        source_input=source_input,
        mgrid_file=mgrid_path.name,
        nphi=nphi,
    )

    exec_path = vmec2000_exec or find_vmec2000_exec()
    if exec_path is None:
        raise FileNotFoundError("VMEC2000 executable not found. Set VMEC2000_EXEC or pass --vmec2000-exec.")
    run = run_xvmec2000(
        input_path,
        exec_path=exec_path,
        workdir=run_dir,
        timeout_s=float(timeout_s),
        keep_workdir=True,
    )
    wout_path = _wout_path_for_input(run)
    active = _active_vacuum_evidence(run)
    report: dict[str, Any] = {
        "created_at": _now_utc(),
        "scope": "optional VMEC2000 W7-X generated-mgrid free-boundary promotion fixture",
        "source": {
            "simsopt_w7x_input": source_input,
            "simsopt_config": "simsopt.configs.get_data('w7x')",
        },
        "workdir": workdir,
        "input": input_path,
        "mgrid": mgrid_summary,
        "vmec2000": {
            "exec": exec_path,
            "returncode": int(run.returncode),
            "runtime_s": float(run.runtime_s),
            "stdout_tail": str(run.stdout).splitlines()[-80:],
            "stderr_tail": str(run.stderr).splitlines()[-40:],
            "threed1": run.threed1_path,
            "stage_count": len(run.stages),
            "active_vacuum": active,
            "wout_path": wout_path,
            "wout_exists": wout_path.exists(),
            "wout_size_bytes": wout_path.stat().st_size if wout_path.exists() else 0,
        },
    }
    if wout_path.exists():
        wout = read_wout(wout_path)
        quality = _quality_from_wout(wout, final_fsq_limit=final_fsq_limit)
        report["vmec2000"]["wout_quality"] = quality
        report["promoted"] = bool(run.returncode == 0 and active["active"] and quality["promotable"])
    else:
        report["promoted"] = False
        report["vmec2000"]["wout_quality"] = {"promotable": False, "reason": "missing_wout"}
    return report


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--vmec2000-exec", type=Path, default=None)
    p.add_argument("--timeout", type=float, default=240.0)
    p.add_argument("--nr", type=int, default=64)
    p.add_argument("--nz", type=int, default=65)
    p.add_argument("--nphi", type=int, default=24)
    p.add_argument("--final-fsq-limit", type=float, default=DEFAULT_FINAL_FSQ_LIMIT)
    p.add_argument("--strict", action="store_true", help="Exit nonzero unless the fixture is promoted.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_w7x_generated_mgrid_vmec2000_fixture(
        workdir=args.workdir,
        vmec2000_exec=args.vmec2000_exec,
        timeout_s=float(args.timeout),
        nr=int(args.nr),
        nz=int(args.nz),
        nphi=int(args.nphi),
        final_fsq_limit=float(args.final_fsq_limit),
    )
    _write_json(Path(args.out), report)
    print(json.dumps(_jsonify(report), indent=2, sort_keys=True, allow_nan=False))
    if args.strict and not bool(report.get("promoted", False)):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
