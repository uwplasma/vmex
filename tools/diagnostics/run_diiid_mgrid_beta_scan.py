#!/usr/bin/env python
"""Run a DIII-D mgrid free-boundary pressure scan and write WOUT summaries.

This diagnostic is intentionally outside default CI. It is the reproducible
generation path for reviewer beta panels that compare vacuum and finite-beta
axisymmetric free-boundary equilibria using the VMEC2000-compatible mgrid
backend.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np

import vmec_jax as vj
from vmec_jax.namelist import read_indata, write_indata
from vmec_jax.wout import read_wout


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "examples" / "data" / "input.DIII-D_lasym_false"
DEFAULT_OUTDIR = REPO_ROOT / "results" / "freeb_diiid_mgrid_beta_scan"
DEFAULT_PRESSURE_SCALES = (0.0, 0.25, 0.50, 0.72, 1.0, 1.35, 1.8)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Base DIII-D VMEC input.")
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="Output directory.")
    p.add_argument(
        "--pressure-scales",
        type=float,
        nargs="+",
        default=list(DEFAULT_PRESSURE_SCALES),
        help="Multipliers applied to the base AM pressure polynomial.",
    )
    p.add_argument("--ns-array", default="16,51,101", help="Comma-separated NS_ARRAY override.")
    p.add_argument("--niter-array", default="1000,4000,20000", help="Comma-separated NITER_ARRAY override.")
    p.add_argument("--ftol-array", default="1e-8,1e-11,1e-12", help="Comma-separated FTOL_ARRAY override.")
    p.add_argument("--max-iter", type=int, default=None, help="Optional max_iter override for every solve.")
    p.add_argument("--verbose", action="store_true", help="Print VMEC iteration tables.")
    return p


def _parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in str(text).split(",") if item.strip()]


def _parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in str(text).split(",") if item.strip()]


def _label(scale: float) -> str:
    if abs(scale) < 1.0e-15:
        return "b0"
    return f"b{int(round(100.0 * float(scale))):03d}"


def _copy_mgrid_if_needed(base_input: Path, outdir: Path, indata: Any) -> None:
    mgrid_name = str(indata.scalars.get("MGRID_FILE", "")).strip("'\"")
    if not mgrid_name:
        return
    src = (base_input.parent / mgrid_name).resolve()
    dst = outdir / mgrid_name
    if src.exists() and not dst.exists():
        shutil.copy2(src, dst)


def _scale_pressure_am(am: Any, scale: float) -> list[float]:
    values = am if isinstance(am, list) else [am]
    return [float(value) * float(scale) for value in values]


def _fsq_total(wout: Any) -> float:
    return float(wout.fsqr) + float(wout.fsqz) + float(wout.fsql)


def _mean_iota(wout: Any) -> float:
    iota = np.asarray(getattr(wout, "iotaf", getattr(wout, "iotas", [])), dtype=float)
    iota = iota[np.isfinite(iota)]
    if iota.size > 1:
        iota = iota[1:]
    return float(np.mean(iota)) if iota.size else float("nan")


def _beta_percent(wout: Any) -> float:
    return 100.0 * float(getattr(wout, "betatotal", getattr(wout, "beta_total", np.nan)))


def _run_case(base_input: Path, outdir: Path, pressure_scale: float, args: argparse.Namespace) -> dict[str, Any]:
    base = read_indata(base_input)
    indata = read_indata(base_input)
    indata.scalars["AM"] = _scale_pressure_am(base.scalars.get("AM", 0.0), pressure_scale)
    ns_array = _parse_int_list(args.ns_array)
    niter_array = _parse_int_list(args.niter_array)
    ftol_array = _parse_float_list(args.ftol_array)
    indata.scalars["NS_ARRAY"] = ns_array
    indata.scalars["NITER_ARRAY"] = niter_array
    indata.scalars["FTOL_ARRAY"] = ftol_array
    # VMEC2000 reads NS from NS_ARRAY for staged inputs and rejects an explicit
    # standalone NS in this DIII-D deck.  Keep the generated input executable-
    # compatible so the same file can be used for vmec_jax and VMEC2000 parity.
    indata.scalars.pop("NS", None)
    indata.scalars["NITER"] = niter_array[-1]
    indata.scalars["FTOL"] = ftol_array[-1]

    label = _label(pressure_scale)
    input_path = outdir / f"input.diiid_{label}_mg101"
    write_indata(input_path, indata)
    _copy_mgrid_if_needed(base_input, outdir, indata)

    kwargs: dict[str, Any] = {
        "verbose": bool(args.verbose),
        "solver": "vmec2000_iter",
        "solver_mode": "parity",
        "multigrid_use_input_niter": True,
    }
    if args.max_iter is not None:
        kwargs["max_iter"] = int(args.max_iter)
    run = vj.run_free_boundary(input_path, **kwargs)
    wout_path = outdir / f"wout_diiid_{label}_mg101.nc"
    vj.write_wout_from_fixed_boundary_run(wout_path, run, include_fsq=True)
    wout = read_wout(wout_path)
    return {
        "pressure_scale": float(pressure_scale),
        "input": str(input_path),
        "wout": str(wout_path),
        "actual_beta_percent": _beta_percent(wout),
        "aspect": float(wout.aspect),
        "mean_iota": _mean_iota(wout),
        "fsq_total": _fsq_total(wout),
        "ns": int(wout.ns),
        "mpol": int(wout.mpol),
        "ntor": int(wout.ntor),
        "nfp": int(wout.nfp),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    base_input = args.input.expanduser().resolve()
    if not base_input.exists():
        raise SystemExit(f"input does not exist: {base_input}")
    outdir = args.outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    rows = [_run_case(base_input, outdir, scale, args) for scale in args.pressure_scales]
    summary = {
        "script": str(Path(__file__).resolve()),
        "input": str(base_input),
        "outdir": str(outdir),
        "rows": rows,
    }
    path = outdir / "summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(path)
    for row in rows:
        print(
            f"{row['pressure_scale']:g}: beta={row['actual_beta_percent']:.3f}% "
            f"fsq={row['fsq_total']:.3e} wout={row['wout']}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
