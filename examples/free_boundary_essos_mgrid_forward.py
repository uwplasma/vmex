#!/usr/bin/env python
"""Pedagogic free-boundary solve using ESSOS coils through an mgrid file.

Workflow shown explicitly:

1. Load Landreman-Paul QA coils from ESSOS.
2. Generate a VMEC-compatible ``mgrid`` file from those coils.
3. Write a low-resolution finite-pressure free-boundary VMEC input.
4. Run ``vmec_jax`` with the legacy/compatibility ``mgrid`` backend.
5. Write ``wout_mgrid.nc`` and ``summary.json``.

Run from the repository root:

    export ESSOS_ROOT=/Users/rogeriojorge/local/ESSOS_mgrid_pr
    export ESSOS_INPUT_DIR=$ESSOS_ROOT/examples/input_files
    PYTHONPATH=.:$ESSOS_ROOT:$PYTHONPATH python examples/free_boundary_essos_mgrid_forward.py --max-iter 10

Use ``--dry-run`` to write the input/mgrid/summary without running VMEC.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.free_boundary_essos_example_common import (
    DEFAULT_INPUT,
    DEFAULT_PRESSURE_SCALE,
    json_default,
    load_essos_coils,
    make_lpqa_free_boundary_indata,
    mgrid_bounds_from_indata,
    run_one_free_boundary_solve,
    summarize_free_boundary_run,
)
from vmec_jax.namelist import read_indata, write_indata


DEFAULT_OUTDIR = REPO_ROOT / "results" / "free_boundary_essos_mgrid_forward"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Base VMEC input deck.")
    parser.add_argument("--coils-json", type=Path, default=None, help="ESSOS coil JSON. Defaults to LP-QA example coils.")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--dry-run", action="store_true", help="Generate input/mgrid/summary but skip the VMEC solve.")
    parser.add_argument("--ns", type=int, default=7)
    parser.add_argument("--max-iter", type=int, default=2)
    parser.add_argument("--ftol", type=float, default=1.0e-8)
    parser.add_argument("--mpol", type=int, default=3)
    parser.add_argument("--ntor", type=int, default=2)
    parser.add_argument("--nzeta", type=int, default=8)
    parser.add_argument("--pressure-scale", type=float, default=DEFAULT_PRESSURE_SCALE)
    parser.add_argument("--phiedge-scale", type=float, default=1.0)
    parser.add_argument("--extcur-scale", type=float, default=1.0)
    parser.add_argument("--mgrid-nr", type=int, default=20)
    parser.add_argument("--mgrid-nz", type=int, default=20)
    parser.add_argument("--mgrid-nphi", type=int, default=8)
    parser.add_argument("--mgrid-padding-fraction", type=float, default=0.30)
    parser.add_argument("--mgrid-min-padding", type=float, default=0.50)
    parser.add_argument("--activate-fsq", type=float, default=1.0e99)
    parser.add_argument(
        "--jit-forces",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use JIT force kernels; --no-jit-forces is useful for parity debugging.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    coils = load_essos_coils(args.coils_json)
    base_indata = read_indata(args.input)

    mgrid_path = outdir / "mgrid_lpqa_from_essos.nc"
    input_path = outdir / "input.lpqa_mgrid"
    wout_path = outdir / "wout_mgrid.nc"
    summary_path = outdir / "summary.json"

    indata = make_lpqa_free_boundary_indata(
        base_indata,
        mgrid_file=mgrid_path.name,
        ns=int(args.ns),
        max_iter=int(args.max_iter),
        ftol=float(args.ftol),
        mpol=int(args.mpol),
        ntor=int(args.ntor),
        nzeta=int(args.nzeta),
        pressure_scale=float(args.pressure_scale),
        phiedge_scale=float(args.phiedge_scale),
        extcur_scale=float(args.extcur_scale),
    )
    bounds = mgrid_bounds_from_indata(
        indata,
        padding_fraction=float(args.mgrid_padding_fraction),
        min_padding=float(args.mgrid_min_padding),
    )

    print(f"Writing mgrid from ESSOS coils: {mgrid_path}")
    coils.to_mgrid(
        mgrid_path,
        nr=int(args.mgrid_nr),
        nz=int(args.mgrid_nz),
        nphi=int(args.mgrid_nphi),
        rmin=float(bounds["rmin"]),
        rmax=float(bounds["rmax"]),
        zmin=float(bounds["zmin"]),
        zmax=float(bounds["zmax"]),
        nfp=int(coils.nfp),
    )
    write_indata(input_path, indata)

    run = None
    wall_s = 0.0
    if not bool(args.dry_run):
        run, wall_s = run_one_free_boundary_solve(
            input_path=input_path,
            wout_path=wout_path,
            max_iter=int(args.max_iter),
            jit_forces=bool(args.jit_forces),
            activate_fsq=float(args.activate_fsq),
        )
    else:
        wout_path = None
        wall_s = 0.0

    summary = summarize_free_boundary_run(
        backend="mgrid",
        input_path=input_path,
        wout_path=wout_path,
        wall_s=wall_s,
        run=run,
        mgrid_path=mgrid_path,
        dry_run=bool(args.dry_run),
    )
    summary["mgrid_bounds"] = bounds
    summary["coils_json"] = str(args.coils_json) if args.coils_json is not None else "ESSOS Landreman-Paul QA default"
    summary_path.write_text(json.dumps(summary, indent=2, default=json_default) + "\n")

    print(f"Wrote input: {input_path}")
    print(f"Wrote summary: {summary_path}")
    if wout_path is not None:
        print(f"Wrote wout: {wout_path}")
    print(
        "Final: "
        f"dry_run={summary['dry_run']} fsqr={summary['fsqr']} fsqz={summary['fsqz']} "
        f"fsql={summary['fsql']} aspect={summary['aspect']} mean_iota={summary['mean_iota']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
