#!/usr/bin/env python
"""Forward free-boundary solve from ESSOS coils without writing an mgrid.

This is the minimal direct-coil example for the free-boundary research branch:

1. load the ESSOS Landreman-Paul QA coils,
2. convert them to ``vmec_jax`` ``CoilFieldParams``,
3. solve one low-resolution free-boundary equilibrium using direct Biot-Savart
   sampling, and
4. write ``wout_direct_coils.nc`` plus ``summary.json``.

The production full-solve adjoint is still phase-2 work.  This example is a
forward provider/coupling validation lane and does not use plasma-boundary
coefficients as optimization variables.

Run from the repository root:

    export ESSOS_ROOT=/path/to/ESSOS_mgrid_pr
    export ESSOS_INPUT_DIR=$ESSOS_ROOT/examples/input_files
    PYTHONPATH=.:$ESSOS_ROOT:$PYTHONPATH python examples/free_boundary_essos_coils_forward.py --beta 1.0 --max-iter 20
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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vmec_jax.driver import run_free_boundary, write_wout_from_fixed_boundary_run
from vmec_jax.external_fields import coil_current_norm, coil_lengths, from_essos_coils
from vmec_jax.namelist import read_indata, write_indata
from vmec_jax.wout import equilibrium_aspect_ratio_from_state, equilibrium_iota_profiles_from_state

from examples.free_boundary_essos_coils_beta_scan import (
    DEFAULT_INPUT,
    PRESSURE_SCALE_FOR_ONE_PERCENT_BETA,
    find_essos_landreman_paul_qa_coils,
    make_free_boundary_indata,
)


DEFAULT_OUTDIR = REPO_ROOT / "results" / "free_boundary_essos_coils_forward"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _summarize_run(run: Any, params: Any, wout_path: Path, wall_s: float, beta_percent: float) -> dict[str, Any]:
    diag = getattr(run.result, "diagnostics", {}) if run.result is not None else {}
    freeb = diag.get("free_boundary", {}) if isinstance(diag, dict) else {}
    nestor = freeb.get("last_nestor_diagnostics", {}) if isinstance(freeb, dict) else {}
    summary: dict[str, Any] = {
        "backend": "direct_coils",
        "surface_dofs_optimized": False,
        "nominal_beta_percent": float(beta_percent),
        "pressure_scale": float(PRESSURE_SCALE_FOR_ONE_PERCENT_BETA) * float(beta_percent),
        "wall_s": float(wall_s),
        "wout": wout_path,
        "n_iter": None if run.result is None else int(getattr(run.result, "n_iter", -1)),
        "fsqr": diag.get("final_fsqr"),
        "fsqz": diag.get("final_fsqz"),
        "fsql": diag.get("final_fsql"),
        "free_boundary_vacuum_stub": freeb.get("vacuum_stub") if isinstance(freeb, dict) else None,
        "free_boundary_nestor_model": freeb.get("nestor_model") if isinstance(freeb, dict) else None,
        "free_boundary_bnormal_rms": nestor.get("bnormal_rms") if isinstance(nestor, dict) else None,
        "free_boundary_bsqvac_rms": nestor.get("bsqvac_rms") if isinstance(nestor, dict) else None,
        "coil_current_norm": float(np.asarray(coil_current_norm(params))),
        "coil_length_mean": float(np.mean(np.asarray(coil_lengths(params), dtype=float))),
    }
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
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--coils-json", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--beta", type=float, default=1.0, help="Nominal beta percentage for the pressure scale.")
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--ftol", type=float, default=1.0e-8)
    parser.add_argument("--ns", type=int, default=12)
    parser.add_argument("--mpol", type=int, default=4)
    parser.add_argument("--ntor", type=int, default=4)
    parser.add_argument("--nzeta", type=int, default=8)
    parser.add_argument("--activate-fsq", type=float, default=1.0e99)
    parser.add_argument("--coil-current-scale", type=float, default=1.0)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument(
        "--jit-forces",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use JIT force kernels; --no-jit-forces is a parity/debug escape hatch.",
    )
    args = parser.parse_args(argv)

    try:
        from essos.coils import Coils_from_json
    except Exception as exc:
        raise ImportError("Install ESSOS or put an ESSOS checkout on PYTHONPATH to run this example.") from exc

    coils_json = args.coils_json or find_essos_landreman_paul_qa_coils()
    coils = Coils_from_json(str(coils_json))
    if float(args.coil_current_scale) != 1.0:
        coils.currents_scale = float(coils.currents_scale) * float(args.coil_current_scale)
    params = from_essos_coils(coils, chunk_size=int(args.chunk_size))

    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    indata = make_free_boundary_indata(
        deepcopy(read_indata(args.input)),
        beta_percent=float(args.beta),
        mgrid_file="DIRECT_COILS",
        niter=int(args.max_iter),
        ftol=float(args.ftol),
        ns=int(args.ns),
        mpol=int(args.mpol),
        ntor=int(args.ntor),
        nzeta=int(args.nzeta),
    )
    input_path = outdir / "input.direct_coils"
    write_indata(input_path, indata)

    t0 = time.perf_counter()
    run = run_free_boundary(
        input_path,
        max_iter=int(args.max_iter),
        multigrid=False,
        verbose=False,
        jit_forces=bool(args.jit_forces),
        external_field_provider_kind="direct_coils",
        external_field_provider_params=params,
        free_boundary_activate_fsq=float(args.activate_fsq),
    )
    wall_s = time.perf_counter() - t0

    wout_path = outdir / "wout_direct_coils.nc"
    write_wout_from_fixed_boundary_run(wout_path, run, include_fsq=True)
    summary = _summarize_run(run, params, wout_path, wall_s, float(args.beta))
    summary["input"] = input_path
    summary["coils_json"] = coils_json
    summary["coil_current_scale"] = float(args.coil_current_scale)
    summary["jit_forces"] = bool(args.jit_forces)
    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=_json_default) + "\n")

    print(f"Wrote input: {input_path}")
    print(f"Wrote wout: {wout_path}")
    print(f"Wrote summary: {summary_path}")
    print(
        "Final: "
        f"fsqr={summary['fsqr']} fsqz={summary['fsqz']} fsql={summary['fsql']} "
        f"aspect={summary['aspect']} mean_iota={summary['mean_iota']} "
        f"coil_length_mean={summary['coil_length_mean']:.6g}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
