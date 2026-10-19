#!/usr/bin/env python3
"""Release regression matrix: a fixed set of decks, one compact JSON summary.

Each deck in ``DECKS`` runs as written (its own ``NS_ARRAY`` ladder) through
:func:`vmex.solve_file`, the same route as ``vmex <input>``, twice in one
process: the first solve includes compilation (``cold_s``), the second does
not (``warm_s``). The physics scalars come from the written ``wout``.

    python benchmarks/regression_matrix.py                      # write benchmarks/regression_matrix.json
    python benchmarks/regression_matrix.py --against OLD.json   # also diff against a previous record

``--against`` exits non-zero when a physics scalar drifts by more than
``RTOL`` (relative, with an ``ATOL`` floor) or when a deck stops converging.
Iteration counts and timings are reported, not gated: they depend on the
machine and on solver tuning, which is what a release is allowed to change.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import jax  # noqa: E402

import vmex  # noqa: E402
from vmex.core.wout import read_wout  # noqa: E402

DATA = REPO / "examples" / "data"
DECKS = {  # deck -> files copied next to it
    "solovev": [],  # analytic Solov'ev tokamak
    "DSHAPE": [],  # finite-beta D-shaped tokamak (Hirshman and Whitson 1983)
    "circular_tokamak": [],
    "LandremanPaul2021_QA_lowres": [],  # vacuum QA, nfp 2
    "LandremanPaul2021_QH_reactorScale_lowres": [],  # vacuum QH, nfp 4
    "nfp4_QH_finite_beta": [],  # finite-beta QH
    "basic_non_stellsym_pressure": [],  # lasym, finite beta
    "cth_like_free_bdy_lasym_small": ["mgrid_cth_like_lasym_small.nc"],  # free boundary
}
PHYSICS = ("wb", "wp", "betatotal", "raxis", "volume", "iota_0.1", "iota_0.5", "iota_1.0")
RTOL, ATOL = 1e-6, 1e-12


def summarize(name: str, files: list[str], work: Path) -> dict:
    """Solve one deck cold and warm and return its summary row."""
    for f in [f"input.{name}", *files]:
        (work / f).write_bytes((DATA / f).read_bytes())
    deck = work / f"input.{name}"
    t0 = time.perf_counter()
    vmex.solve_file(deck, outdir=work, raise_on_max_iterations=False)
    t1 = time.perf_counter()
    r = vmex.solve_file(deck, outdir=work, raise_on_max_iterations=False)
    t2 = time.perf_counter()
    w = read_wout(work / f"wout_{name}.nc")
    s = np.linspace(0.0, 1.0, w.ns)
    iota = np.abs(np.asarray(w.iotaf))
    row = dict(ns=int(w.ns), mpol=int(w.mpol), ntor=int(w.ntor), lasym=bool(w.lasym), lfreeb=bool(w.lfreeb),
               converged=bool(r.converged), iterations=int(r.iterations),
               cold_s=round(t1 - t0, 2), warm_s=round(t2 - t1, 3),
               fsqr=float(r.fsqr), fsqz=float(r.fsqz), fsql=float(r.fsql),
               wb=float(w.wb), wp=float(w.wp), betatotal=float(w.betatotal),
               raxis=float(np.ravel(w.raxis_cc)[0]), volume=float(w.volume_p))
    for rho in (0.1, 0.5, 1.0):
        row[f"iota_{rho}"] = float(np.interp(rho * rho, s, iota))
    return row


def compare(new: dict, old: dict) -> list[str]:
    """Physics drifts and lost convergence between two records."""
    bad = []
    for name, row in new["decks"].items():
        ref = old["decks"].get(name)
        if ref is None:
            continue
        if ref["converged"] and not row["converged"]:
            bad.append(f"{name}: no longer converges")
        for key in PHYSICS:
            a, b = row[key], ref[key]
            if abs(a - b) > ATOL + RTOL * abs(b):
                bad.append(f"{name}.{key}: {b:.10g} -> {a:.10g}")
    return bad


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=REPO / "benchmarks" / "regression_matrix.json")
    parser.add_argument("--against", type=Path, help="previous record to diff against")
    parser.add_argument("--decks", help="comma-separated subset of the matrix")
    args = parser.parse_args()
    names = args.decks.split(",") if args.decks else list(DECKS)
    rows = {}
    with tempfile.TemporaryDirectory() as td:
        for name in names:
            work = Path(td) / name
            work.mkdir()
            rows[name] = summarize(name, DECKS[name], work)
            print(f"{name:42s} ns={rows[name]['ns']:4d} it={rows[name]['iterations']:6d} "
                  f"cold={rows[name]['cold_s']:7.2f}s warm={rows[name]['warm_s']:7.3f}s "
                  f"beta={rows[name]['betatotal']:.4e} iota(0.5)={rows[name]['iota_0.5']:.6f}", flush=True)
    record = dict(
        _provenance=dict(measurement_commit=subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                                                   text=True).stdout.strip(),
                         measurement_date=date.today().isoformat(),
                         vmex=vmex.__version__, jax=jax.__version__, device=jax.devices()[0].platform,
                         machine=platform.machine(), system=platform.system(),
                         command="python benchmarks/regression_matrix.py"),
        tolerance=dict(rtol=RTOL, atol=ATOL, gated=list(PHYSICS)),
        decks=rows,
    )
    args.out.write_text(json.dumps(record, indent=1) + "\n")
    if args.against:
        bad = compare(record, json.loads(args.against.read_text()))
        print("\n".join(bad) or f"no physics drift beyond rtol={RTOL}")
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
