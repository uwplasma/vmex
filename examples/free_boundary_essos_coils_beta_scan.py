#!/usr/bin/env python
"""Free-boundary beta scan from ESSOS Landreman-Paul QA coils.

This example demonstrates both free-boundary external-field backends:

1. ESSOS coils -> mgrid file -> vmec_jax free-boundary solve.
2. ESSOS coils -> vmec_jax direct JAX Biot-Savart provider -> free-boundary solve.

The direct-coil provider path avoids writing an mgrid file.  Its field sampling
is differentiable with respect to coil Fourier coefficients and currents.  The
full production NESTOR/free-boundary adjoint is still phase-2 work, so this
example should be treated as a forward research lane plus provider-gradient
foundation rather than a publication claim for full-solve exact adjoints.

Run from the repository root:

    python examples/free_boundary_essos_coils_beta_scan.py

Use smaller settings for a quick smoke run:

    python examples/free_boundary_essos_coils_beta_scan.py --betas 0 1 --max-iter 2 --mgrid-nr 8 --mgrid-nz 8 --mgrid-nphi 4
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

from vmec_jax.driver import run_free_boundary, write_wout_from_fixed_boundary_run
from vmec_jax.external_fields import from_essos_coils
from vmec_jax.namelist import read_indata, write_indata
from vmec_jax.wout import equilibrium_aspect_ratio_from_state, equilibrium_iota_profiles_from_state, read_wout


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "examples" / "data" / "input.LandremanPaul2021_QA_reactorScale_lowres"
DEFAULT_RESULTS = REPO_ROOT / "results" / "free_boundary_essos_coils_beta_scan"
DEFAULT_NOMINAL_BETA_PERCENT = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)

# This is a nominal pressure scale for the example scan.  The actual VMEC beta
# should be read from the resulting wout files.
PRESSURE_SCALE_FOR_ONE_PERCENT_BETA = 34.46233666638


def _candidate_essos_input_dirs() -> list[Path]:
    candidates = []
    user_env = None
    import os

    if os.getenv("ESSOS_INPUT_DIR"):
        user_env = Path(os.environ["ESSOS_INPUT_DIR"]).expanduser()
    if user_env is not None:
        candidates.append(user_env)
    candidates.extend(
        [
            REPO_ROOT.parent / "ESSOS_mgrid_pr" / "examples" / "input_files",
            REPO_ROOT.parent / "ESSOS" / "examples" / "input_files",
            Path("/Users/rogeriojorge/local/ESSOS_mgrid_pr/examples/input_files"),
            Path("/Users/rogeriojorge/local/ESSOS/examples/input_files"),
            Path.cwd() / "examples" / "input_files",
        ]
    )
    return candidates


def find_essos_landreman_paul_qa_coils() -> Path:
    """Find the ESSOS Landreman-Paul QA coil JSON in local example assets."""

    name = "ESSOS_biot_savart_LandremanPaulQA.json"
    for directory in _candidate_essos_input_dirs():
        path = directory / name
        if path.exists():
            return path
    searched = "\n  ".join(str(p) for p in _candidate_essos_input_dirs())
    raise FileNotFoundError(
        f"Could not find {name}. Set ESSOS_INPUT_DIR to the ESSOS examples/input_files directory. Searched:\n  {searched}"
    )


def make_free_boundary_indata(
    base_indata,
    *,
    beta_percent: float,
    mgrid_file: str,
    niter: int,
    ftol: float,
    ns: int,
    mpol: int,
    ntor: int,
    nzeta: int,
) -> Any:
    """Create a small free-boundary input deck for one nominal beta."""

    indata = deepcopy(base_indata)
    indata.scalars["LFREEB"] = True
    indata.scalars["MGRID_FILE"] = str(mgrid_file)
    indata.scalars["EXTCUR"] = [1.0]
    indata.scalars["NS_ARRAY"] = [int(ns)]
    indata.scalars["NITER_ARRAY"] = [int(niter)]
    indata.scalars["FTOL_ARRAY"] = [float(ftol)]
    indata.scalars["NITER"] = int(niter)
    indata.scalars["FTOL"] = float(ftol)
    indata.scalars["MPOL"] = int(mpol)
    indata.scalars["NTOR"] = int(ntor)
    indata.scalars["NZETA"] = int(nzeta)
    indata.scalars["NTHETA"] = 0
    indata.scalars["NVACSKIP"] = max(1, int(nzeta))
    indata.scalars["PMASS_TYPE"] = "power_series"
    # p(s) = PRES_SCALE * (1 - s), so beta increases monotonically with the
    # requested nominal beta percentage.
    indata.scalars["AM"] = [1.0, -1.0]
    indata.scalars["PRES_SCALE"] = float(PRESSURE_SCALE_FOR_ONE_PERCENT_BETA) * float(beta_percent)
    return indata


def summarize_run(run, wout_path: Path, *, backend: str, beta_percent: float, wall_s: float) -> dict[str, Any]:
    """Collect lightweight scalar diagnostics for the JSON summary."""

    diag = getattr(run.result, "diagnostics", {}) if run.result is not None else {}
    summary: dict[str, Any] = {
        "backend": backend,
        "nominal_beta_percent": float(beta_percent),
        "wall_s": float(wall_s),
        "wout": str(wout_path),
        "n_iter": None if run.result is None else int(getattr(run.result, "n_iter", -1)),
        "fsqr": None,
        "fsqz": None,
        "fsql": None,
        "aspect": None,
        "mean_iota": None,
        "pressure_scale": float(PRESSURE_SCALE_FOR_ONE_PERCENT_BETA) * float(beta_percent),
        "max_pressure": None,
        "wp": None,
        "wb": None,
        "beta_proxy": None,
        "beta_proxy_percent": None,
    }
    for key in ("final_fsqr", "final_fsqz", "final_fsql"):
        val = diag.get(key)
        if val is not None:
            summary[key.replace("final_", "")] = float(val)
    try:
        summary["aspect"] = float(equilibrium_aspect_ratio_from_state(state=run.state, static=run.static))
    except Exception:
        pass
    try:
        _chips, iotas, _iotaf = equilibrium_iota_profiles_from_state(
            state=run.state,
            static=run.static,
            indata=run.indata,
            signgs=int(run.signgs),
        )
        summary["mean_iota"] = float(np.nanmean(np.asarray(iotas, dtype=float)))
    except Exception:
        pass
    try:
        wout = read_wout(wout_path)
        summary["max_pressure"] = float(np.nanmax(np.asarray(wout.presf, dtype=float)))
        summary["wp"] = float(wout.wp)
        summary["wb"] = float(wout.wb)
        if float(wout.wb) != 0.0:
            summary["beta_proxy"] = float(wout.wp) / float(wout.wb)
            summary["beta_proxy_percent"] = 100.0 * float(wout.wp) / float(wout.wb)
    except Exception:
        pass
    return summary


def run_one_case(
    *,
    backend: str,
    input_path: Path,
    output_dir: Path,
    beta_percent: float,
    max_iter: int,
    direct_coil_params=None,
) -> dict[str, Any]:
    """Run one mgrid or direct-coil free-boundary case."""

    t0 = time.perf_counter()
    if backend == "mgrid":
        run = run_free_boundary(
            input_path,
            max_iter=int(max_iter),
            multigrid=False,
            verbose=False,
            jit_forces=False,
        )
    elif backend == "direct":
        run = run_free_boundary(
            input_path,
            max_iter=int(max_iter),
            multigrid=False,
            verbose=False,
            jit_forces=False,
            external_field_provider_kind="direct_coils",
            external_field_provider_params=direct_coil_params,
        )
    else:
        raise ValueError(f"unknown backend {backend!r}")
    wall_s = time.perf_counter() - t0
    wout_path = output_dir / f"wout_{backend}_beta_{float(beta_percent):.3f}.nc"
    write_wout_from_fixed_boundary_run(wout_path, run, include_fsq=True)
    return summarize_run(run, wout_path, backend=backend, beta_percent=beta_percent, wall_s=wall_s)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--coils-json", type=Path, default=None)
    parser.add_argument("--betas", type=float, nargs="*", default=list(DEFAULT_NOMINAL_BETA_PERCENT))
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--ftol", type=float, default=1.0e-8)
    parser.add_argument("--ns", type=int, default=12)
    parser.add_argument("--mpol", type=int, default=4)
    parser.add_argument("--ntor", type=int, default=4)
    parser.add_argument("--mgrid-nr", type=int, default=16)
    parser.add_argument("--mgrid-nz", type=int, default=16)
    parser.add_argument("--mgrid-nphi", type=int, default=8)
    parser.add_argument("--mgrid-rmin", type=float, default=5.0)
    parser.add_argument("--mgrid-rmax", type=float, default=15.0)
    parser.add_argument("--mgrid-zmin", type=float, default=-5.0)
    parser.add_argument("--mgrid-zmax", type=float, default=5.0)
    parser.add_argument("--skip-mgrid-runs", action="store_true")
    parser.add_argument("--skip-direct-runs", action="store_true")
    args = parser.parse_args(argv)

    try:
        from essos.coils import Coils_from_json
    except Exception as exc:
        raise ImportError(
            "This example requires ESSOS with Coils_from_json. Install the ESSOS mgrid branch or set PYTHONPATH to it."
        ) from exc

    coils_json = args.coils_json or find_essos_landreman_paul_qa_coils()
    coils = Coils_from_json(str(coils_json))
    if not hasattr(coils, "to_mgrid"):
        raise AttributeError(
            "ESSOS Coils.to_mgrid is not available. Use the ESSOS PR branch that adds mgrid generation from coils."
        )
    direct_params = from_essos_coils(coils, chunk_size=256)

    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    mgrid_file = outdir / "mgrid_landreman_paul_qa_from_essos.nc"
    print(f"Writing mgrid from ESSOS coils: {mgrid_file}")
    coils.to_mgrid(
        mgrid_file,
        nr=args.mgrid_nr,
        nphi=args.mgrid_nphi,
        nz=args.mgrid_nz,
        rmin=args.mgrid_rmin,
        rmax=args.mgrid_rmax,
        zmin=args.mgrid_zmin,
        zmax=args.mgrid_zmax,
        nfp=int(coils.nfp),
    )

    base_indata = read_indata(args.input)
    summaries = []
    for beta_percent in args.betas:
        beta_tag = f"{float(beta_percent):.3f}".replace(".", "p")
        if not args.skip_mgrid_runs:
            mgrid_indata = make_free_boundary_indata(
                base_indata,
                beta_percent=beta_percent,
                mgrid_file=mgrid_file.name,
                niter=args.max_iter,
                ftol=args.ftol,
                ns=args.ns,
                mpol=args.mpol,
                ntor=args.ntor,
                nzeta=args.mgrid_nphi,
            )
            input_mgrid = outdir / f"input.lpqa_mgrid_beta_{beta_tag}"
            write_indata(input_mgrid, mgrid_indata)
            print(f"Running mgrid beta={beta_percent:.3f}%: {input_mgrid}")
            summaries.append(
                run_one_case(
                    backend="mgrid",
                    input_path=input_mgrid,
                    output_dir=outdir,
                    beta_percent=beta_percent,
                    max_iter=args.max_iter,
                )
            )

        if not args.skip_direct_runs:
            direct_indata = make_free_boundary_indata(
                base_indata,
                beta_percent=beta_percent,
                mgrid_file="DIRECT_COILS",
                niter=args.max_iter,
                ftol=args.ftol,
                ns=args.ns,
                mpol=args.mpol,
                ntor=args.ntor,
                nzeta=args.mgrid_nphi,
            )
            input_direct = outdir / f"input.lpqa_direct_beta_{beta_tag}"
            write_indata(input_direct, direct_indata)
            print(f"Running direct-coil beta={beta_percent:.3f}%: {input_direct}")
            summaries.append(
                run_one_case(
                    backend="direct",
                    input_path=input_direct,
                    output_dir=outdir,
                    beta_percent=beta_percent,
                    max_iter=args.max_iter,
                    direct_coil_params=direct_params,
                )
            )

    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps({"coils_json": str(coils_json), "mgrid": str(mgrid_file), "runs": summaries}, indent=2))
    print(f"Wrote summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
