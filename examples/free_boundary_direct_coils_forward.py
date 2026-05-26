#!/usr/bin/env python
"""Forward free-boundary solve from synthetic direct coils.

This example is intentionally independent of ESSOS assets and mgrid files.  It
builds a small circular coil as ``vmec_jax.external_fields.CoilFieldParams``,
writes a tiny free-boundary input deck with ``MGRID_FILE='DIRECT_COILS'``, and
runs a short direct Biot-Savart free-boundary solve.

Quick smoke without running VMEC:

    python examples/free_boundary_direct_coils_forward.py --dry-run

Short forward run:

    python examples/free_boundary_direct_coils_forward.py --max-iter 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vmec_jax._compat import enable_x64, jnp
from vmec_jax.driver import run_free_boundary, write_wout_from_fixed_boundary_run
from vmec_jax.external_fields import (
    CoilFieldParams,
    coil_current_norm,
    coil_lengths,
    sample_coil_field_cylindrical,
)
from vmec_jax.wout import equilibrium_aspect_ratio_from_state, equilibrium_iota_profiles_from_state


DEFAULT_OUTDIR = REPO_ROOT / "results" / "free_boundary_direct_coils_forward"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def make_circular_coil_params(
    *,
    current: float = 3.0e7,
    radius: float = 1.8,
    n_segments: int = 96,
    regularization_epsilon: float = 0.0,
    chunk_size: int | None = None,
) -> CoilFieldParams:
    """Return a single circular direct-coil provider in the midplane."""

    dofs = np.zeros((1, 3, 3), dtype=float)
    dofs[0, 0, 2] = float(radius)  # x = radius * cos(2*pi*t)
    dofs[0, 1, 1] = float(radius)  # y = radius * sin(2*pi*t)
    return CoilFieldParams(
        base_curve_dofs=jnp.asarray(dofs),
        base_currents=jnp.asarray([float(current)], dtype=float),
        n_segments=int(n_segments),
        nfp=1,
        stellsym=False,
        regularization_epsilon=float(regularization_epsilon),
        chunk_size=chunk_size,
    )


def write_tiny_direct_free_boundary_input(
    path: Path,
    *,
    ns: int = 7,
    max_iter: int = 4,
    ftol: float = 1.0e-8,
    mpol: int = 4,
    ntor: int = 0,
    nzeta: int = 2,
    ntheta: int = 8,
    pressure_scale: float = 1.0e4,
) -> Path:
    """Write a low-resolution axisymmetric free-boundary input deck."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""
&INDATA
  LFREEB = T
  MGRID_FILE = 'DIRECT_COILS'
  EXTCUR = 1.0
  LASYM = F
  NFP = 1
  MPOL = {int(mpol)}
  NTOR = {int(ntor)}
  NS = {int(ns)}
  NZETA = {int(nzeta)}
  NTHETA = {int(ntheta)}
  NS_ARRAY = {int(ns)}
  FTOL_ARRAY = {float(ftol):.16E}
  NITER_ARRAY = {int(max_iter)}
  NITER = {int(max_iter)}
  FTOL = {float(ftol):.16E}
  NSTEP = 20
  NVACSKIP = 1
  GAMMA = 0.0
  PHIEDGE = 1.0
  CURTOR = 0.0
  SPRES_PED = 1.0
  NCURR = 0
  PRES_SCALE = {float(pressure_scale):.16E}
  AM = 1.0 -1.0
  AI = 0.4 0.0
  AC = 0.0
  RAXIS = 1.0
  ZAXIS = 0.0
  RBC(0,0) = 1.0  ZBS(0,0) = 0.0
  RBC(0,1) = 0.25 ZBS(0,1) = 0.25
  RBC(0,2) = 0.03 ZBS(0,2) = 0.00
/
""".lstrip()
    )
    return path


def _coil_summary(params: CoilFieldParams) -> dict[str, Any]:
    br, bphi, bz = sample_coil_field_cylindrical(params, 1.0, 0.0, 0.0)
    return {
        "n_base_coils": int(np.asarray(params.base_currents).size),
        "n_segments": int(params.n_segments),
        "current_norm": float(np.asarray(coil_current_norm(params))),
        "length_mean": float(np.mean(np.asarray(coil_lengths(params), dtype=float))),
        "sample_R1_Z0_phi0": {
            "br": float(np.asarray(br)),
            "bphi": float(np.asarray(bphi)),
            "bz": float(np.asarray(bz)),
        },
    }


def _summarize_run(
    run: Any,
    params: CoilFieldParams,
    *,
    input_path: Path,
    wout_path: Path | None,
    wall_s: float,
    dry_run: bool,
) -> dict[str, Any]:
    diag = getattr(run.result, "diagnostics", {}) if run is not None and getattr(run, "result", None) is not None else {}
    freeb = diag.get("free_boundary", {}) if isinstance(diag, dict) else {}
    nestor = freeb.get("last_nestor_diagnostics", {}) if isinstance(freeb, dict) else {}
    summary: dict[str, Any] = {
        "backend": "direct_coils",
        "dry_run": bool(dry_run),
        "input": input_path,
        "wout": wout_path,
        "wall_s": float(wall_s),
        "coil": _coil_summary(params),
        "n_iter": None if run is None or getattr(run, "result", None) is None else int(getattr(run.result, "n_iter", -1)),
        "fsqr": diag.get("final_fsqr") if isinstance(diag, dict) else None,
        "fsqz": diag.get("final_fsqz") if isinstance(diag, dict) else None,
        "fsql": diag.get("final_fsql") if isinstance(diag, dict) else None,
        "free_boundary_vacuum_stub": freeb.get("vacuum_stub") if isinstance(freeb, dict) else None,
        "free_boundary_nestor_model": freeb.get("nestor_model") if isinstance(freeb, dict) else None,
        "free_boundary_bnormal_rms": nestor.get("bnormal_rms") if isinstance(nestor, dict) else None,
        "free_boundary_bsqvac_rms": nestor.get("bsqvac_rms") if isinstance(nestor, dict) else None,
    }
    if run is not None:
        try:
            summary["aspect"] = float(equilibrium_aspect_ratio_from_state(state=run.state, static=run.static))
        except Exception:
            summary["aspect"] = None
        try:
            _chips, iotas, _iotaf = equilibrium_iota_profiles_from_state(
                state=run.state,
                static=run.static,
                indata=run.indata,
                signgs=int(run.signgs),
            )
            summary["mean_iota"] = float(np.nanmean(np.asarray(iotas, dtype=float)))
        except Exception:
            summary["mean_iota"] = None
    else:
        summary["aspect"] = None
        summary["mean_iota"] = None
    return summary


def run_forward(args: argparse.Namespace) -> dict[str, Any]:
    enable_x64(True)
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    params = make_circular_coil_params(
        current=float(args.coil_current),
        radius=float(args.coil_radius),
        n_segments=int(args.n_segments),
        regularization_epsilon=float(args.regularization_epsilon),
        chunk_size=args.chunk_size,
    )
    input_path = write_tiny_direct_free_boundary_input(
        outdir / "input.direct_coils",
        ns=int(args.ns),
        max_iter=int(args.max_iter),
        ftol=float(args.ftol),
        mpol=int(args.mpol),
        ntor=int(args.ntor),
        nzeta=int(args.nzeta),
        ntheta=int(args.ntheta),
        pressure_scale=float(args.pressure_scale),
    )

    t0 = time.perf_counter()
    run = None
    wout_path: Path | None = None
    if not bool(args.dry_run):
        run = run_free_boundary(
            input_path,
            max_iter=int(args.max_iter),
            multigrid=False,
            verbose=bool(args.verbose),
            jit_forces=bool(args.jit_forces),
            external_field_provider_kind="direct_coils",
            external_field_provider_params=params,
            free_boundary_activate_fsq=float(args.activate_fsq),
        )
        wout_path = outdir / "wout_direct_coils.nc"
        write_wout_from_fixed_boundary_run(wout_path, run, include_fsq=True)
    wall_s = time.perf_counter() - t0

    summary = _summarize_run(
        run,
        params,
        input_path=input_path,
        wout_path=wout_path,
        wall_s=wall_s,
        dry_run=bool(args.dry_run),
    )
    summary["jit_forces"] = bool(args.jit_forces)
    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=_json_default) + "\n")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--dry-run", action="store_true", help="Write input and summary without running VMEC.")
    parser.add_argument("--max-iter", type=int, default=4)
    parser.add_argument("--ftol", type=float, default=1.0e-8)
    parser.add_argument("--ns", type=int, default=7)
    parser.add_argument("--mpol", type=int, default=4)
    parser.add_argument("--ntor", type=int, default=0)
    parser.add_argument("--nzeta", type=int, default=2)
    parser.add_argument("--ntheta", type=int, default=8)
    parser.add_argument("--pressure-scale", type=float, default=1.0e4)
    parser.add_argument("--coil-current", type=float, default=3.0e7)
    parser.add_argument("--coil-radius", type=float, default=1.8)
    parser.add_argument("--n-segments", type=int, default=96)
    parser.add_argument("--regularization-epsilon", type=float, default=0.0)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--activate-fsq", type=float, default=1.0e99)
    parser.add_argument(
        "--jit-forces",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use JIT force kernels; --no-jit-forces is a parity/debug escape hatch.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    summary = run_forward(build_parser().parse_args(argv))
    print(f"Wrote input: {summary['input']}")
    print(f"Wrote summary: {Path(summary['input']).parent / 'summary.json'}")
    if summary["wout"] is not None:
        print(f"Wrote wout: {summary['wout']}")
    print(
        "Final: "
        f"dry_run={summary['dry_run']} fsqr={summary['fsqr']} fsqz={summary['fsqz']} "
        f"fsql={summary['fsql']} aspect={summary['aspect']} mean_iota={summary['mean_iota']} "
        f"coil_length_mean={summary['coil']['length_mean']:.6g}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
